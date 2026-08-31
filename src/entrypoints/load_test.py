"""
Load test script: Launch 10 workflows in parallel to demonstrate load balancing.
Each workflow will be distributed across available workers.
"""
import asyncio
import os
import sys
import uuid
from pathlib import Path
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv(override=True)

from mistralai.client import Mistral
from mistralai.workflows.client import get_mistral_client

API_KEY = os.environ["MISTRAL_API_KEY"]
BASE_URL = os.environ.get("SERVER_URL", "https://api.mistral.ai")
WORKFLOWS_CLIENT = None

# Number of workflows to launch (from CLI arg, default to 10)
NUM_WORKFLOWS = int(sys.argv[1]) if len(sys.argv) > 1 else 10


class PdfOcrInput(BaseModel):
    file_id: str
    filename: str
    confidence_threshold: float = 0.9
    manual_review_timeout_seconds: float | None = 5.0


def get_workflows_sdk_client():
    global WORKFLOWS_CLIENT
    if WORKFLOWS_CLIENT is None:
        WORKFLOWS_CLIENT = get_mistral_client(
            server_url=BASE_URL,
            api_key=API_KEY,
        )
    return WORKFLOWS_CLIENT


async def upload_pdf(pdf_bytes: bytes, filename: str) -> str:
    """Upload PDF and return file ID."""
    async with Mistral(api_key=API_KEY) as client:
        resp = await client.files.upload_async(
            file={"file_name": filename, "content": pdf_bytes, "content_type": "application/pdf"},
            purpose="ocr",
        )
    return resp.id


async def trigger_workflow(file_id: str, filename: str) -> str:
    """Trigger workflow and return execution ID."""
    execution_id = f"load-test-{uuid.uuid4().hex[:12]}"
    client = get_workflows_sdk_client()
    resp = await client.workflows.execute_workflow_async(
        workflow_identifier="pdf_ocr_workflow",
        input=PdfOcrInput(
            file_id=file_id,
            filename=filename,
            confidence_threshold=0.9,
            manual_review_timeout_seconds=5.0,
        ).model_dump(mode="json"),
        execution_id=execution_id,
    )
    return resp.execution_id


async def launch_workflow(pdf_path: Path, index: int) -> tuple[int, str]:
    """Launch a single workflow and return (index, execution_id)."""
    pdf_bytes = pdf_path.read_bytes()
    file_id = await upload_pdf(pdf_bytes, pdf_path.name)
    execution_id = await trigger_workflow(file_id, pdf_path.name)
    return index, execution_id


async def main():
    project_root = Path(__file__).resolve().parent.parent.parent
    input_dir = project_root / "input_doc"
    pdf_files = list(input_dir.glob("*.pdf"))

    if not pdf_files:
        print("❌ No PDF files found in input_doc/")
        sys.exit(1)

    print(f"\n🚀 Launching {NUM_WORKFLOWS} workflows (load balancing on available workers)")
    print(f"📁 Using PDFs from: {input_dir}\n")

    # Launch all workflows in parallel
    tasks = [
        launch_workflow(pdf_files[i % len(pdf_files)], i + 1)
        for i in range(NUM_WORKFLOWS)
    ]

    execution_ids = []
    for index, execution_id in await asyncio.gather(*tasks):
        execution_ids.append(execution_id)
        print(f"  ✓ Workflow {index:2d} → {execution_id}")

    print(f"\n✅ All {NUM_WORKFLOWS} workflows launched!")
    print(f"\nExecution IDs (for monitoring):")
    for i, eid in enumerate(execution_ids, 1):
        print(f"  {i:2d}. {eid}")

    # Small delay to allow async clients to close gracefully
    await asyncio.sleep(0.1)


if __name__ == "__main__":
    asyncio.run(main())
