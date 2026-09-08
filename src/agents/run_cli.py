"""Run one document through the agent chain from the command line.

Usage::

    uv run python src/agents/run_cli.py --file passport.jpg
    uv run python src/agents/run_cli.py --file passport.jpg --confidence-threshold 1.0
    uv run python src/agents/run_cli.py --delete-agents

Mirrors ``src/workflows/start.py``, but there is no worker to register with: the
agents live on the API and this process drives the conversation directly.
"""
# ruff: noqa: E402

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(override=True)

from agents.files import mistral_client, upload_document
from agents.orchestrator import PersonalDocumentRun
from agents.registry import delete_registry, ensure_registry
from shared.document_media import get_document_content_type
from shared.extraction_fields import PERSONAL_DOCUMENT_CATEGORIES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", help="Path to a PDF, JPEG, PNG or WebP document.")
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.9,
        help="Below this, the category needs manual review (default: 0.9).",
    )
    parser.add_argument(
        "--category",
        choices=PERSONAL_DOCUMENT_CATEGORIES,
        help="Answer a review prompt non-interactively.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Log every handoff, tool call and API round trip.",
    )
    parser.add_argument(
        "--sync-agents",
        action="store_true",
        help="Create or refresh the pdp-* agents and exit.",
    )
    parser.add_argument(
        "--delete-agents",
        action="store_true",
        help="Delete every pdp-* agent and exit.",
    )
    return parser.parse_args()


def ask_for_category(classification: dict) -> str:
    confidence = classification.get("confidence", 0.0)
    print(
        f"\nLow confidence ({confidence * 100:.0f}%): "
        f"{classification.get('category')} — {classification.get('explanation', '')}"
    )
    options = ", ".join(PERSONAL_DOCUMENT_CATEGORIES)
    while True:
        answer = input(f"Choose a category [{options}]: ").strip()
        if answer in PERSONAL_DOCUMENT_CATEGORIES:
            return answer
        print(f"'{answer}' is not one of: {options}")


async def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(name)s %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.delete_agents:
        async with mistral_client() as client:
            removed = await delete_registry(client)
        print(f"Deleted {len(removed)} agent(s): {', '.join(removed) or '(none)'}")
        return 0

    if args.sync_agents:
        async with mistral_client() as client:
            registry = await ensure_registry(client)
        print(json.dumps(registry.__dict__, indent=2))
        return 0

    if not args.file:
        print("FAILED: --file is required (or use --sync-agents / --delete-agents).")
        return 1

    path = Path(args.file)
    if not path.is_file():
        print(f"FAILED: no such file: {path}")
        return 1

    content_type = get_document_content_type(path.name)
    async with mistral_client() as client:
        file_id = await upload_document(
            client, path.read_bytes(), path.name, content_type
        )
    print(f"Uploaded {path.name} as {file_id} ({content_type})")

    run = PersonalDocumentRun(
        file_id=file_id,
        filename=path.name,
        content_type=content_type,
        confidence_threshold=args.confidence_threshold,
    )
    await run.start()
    print(f"Conversation: {run.conversation_id}")

    if run.pending == "category":
        category = args.category or ask_for_category(run.classification or {})
        await run.resume_with_category(category)

    print(json.dumps(run.result(), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
