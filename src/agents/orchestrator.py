"""Drive one personal document through the agent chain.

This is the Agents API counterpart of ``workflows.personal_doc_workflow``. Instead of
a durable workflow with a signal, a run is a single stored conversation that can be
left parked when classification confidence is low and resumed later from its
``conversation_id``.

Three phases, each hop a real server-side handoff:

  1. supervisor → preprocessor (images) or → classifier (PDFs). ``function.call``
     entries for the OpenCV tools are pumped locally.
  2. images only: the *enhanced* document is appended and handed to the classifier,
     so classification and extraction read the improved image.
  3. after a deterministic confidence gate (and an optional human category), the
     category's extractor agent is addressed directly, in its own conversation.

``steps`` deliberately mirrors the workflow's shape so the Streamlit renderer is
shared verbatim.
"""

import json
import logging
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Optional

from mistralai.client import Mistral
from mistralai.client.errors import SDKError
from pydantic import ValidationError

from agents.files import call_with_retry, get_signed_url, mistral_client
from agents.registry import AgentRegistry, ensure_registry, extractor_name
from agents.tools import APPLY_TOOL_NAME, TOOL_DISPATCH
from shared.document_media import build_mistral_document_chunk
from shared.extraction_fields import PERSONAL_DOCUMENT_SPECIFIC_FIELDS
from shared.personal_documents import (
    PersonalDocumentClassification,
    classification_prompt,
    enrich_with_mrz_fallback,
    extraction_prompt,
    find_unusable_fields,
    get_personal_extraction_output_model,
    strip_unusable_fields,
    validate_preprocessing_decision,
)
from shared.preprocessing import enhanced_image_filename

logger = logging.getLogger(__name__)

#: Matches the ``max_turns`` cap of the workflow's preprocessing agent.
MAX_TOOL_ITERATIONS = 14

#: Every turn re-sends the whole conversation, so an unbounded chain of image
#: transformations burns tokens fast (and each extra operation degrades the page).
#: Past this many, the tool refuses and tells the agent to finalise.
MAX_PREPROCESSING_OPERATIONS = 4

#: Total tool calls allowed per run. The operation budget alone was not enough: an
#: agent that had spent it kept re-inspecting the image instead of answering, and
#: burned the whole iteration budget. This bounds inspections too.
MAX_TOOL_CALLS = 10

#: One retry for a malformed extraction. The usual cause is a reply truncated by a
#: repetition loop, which a short corrective nudge reliably clears.
EXTRACTION_ATTEMPTS = 2

#: Kept free of MRZ talk: an MRZ-themed nudge sent to a GTC extractor made it null
#: every field. The MRZ half is appended only for categories that have one.
EXTRACTION_RETRY_NUDGE = (
    "Your previous reply could not be used. Reply again with the complete JSON "
    "object and nothing else, filling in every field you can read from the document. "
    "Never repeat a character sequence to pad a value."
)

EXTRACTION_RETRY_MRZ_NUDGE = (
    " A document number is the short identifier printed on the document, never the "
    "machine-readable zone. MRZ characters belong only in specific.mrz.raw_lines."
)


def extraction_retry_nudge(category: str) -> str:
    """The corrective nudge, mentioning the MRZ only where the category has one."""
    fields = PERSONAL_DOCUMENT_SPECIFIC_FIELDS.get(category, [])
    if any(key == "mrz" for key, _ in fields):
        return EXTRACTION_RETRY_NUDGE + EXTRACTION_RETRY_MRZ_NUDGE
    return EXTRACTION_RETRY_NUDGE


STEP_KEYS = ("preprocess", "ocr", "classify", "extract")


class AgentRunError(RuntimeError):
    """Raised when the agent chain does not produce a usable result."""


def _pending_steps() -> dict[str, dict[str, Any]]:
    return {key: {"status": "pending", "result": None} for key in STEP_KEYS}


def entry_type(entry: Any) -> str:
    return str(getattr(entry, "type", "") or "")


def content_to_text(content: Any) -> str:
    """Flatten message content, which is either a string or a list of chunks."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            str(getattr(chunk, "text", ""))
            for chunk in content
            if getattr(chunk, "type", None) == "text"
        )
    return str(content)


def message_text(outputs: list[Any], agent_id: Optional[str] = None) -> str:
    """The agent's *last* message.

    Only the last one: a tool-using agent emits chatter between calls, and joining
    every message would feed that chatter to the JSON parser.
    """
    for entry in reversed(outputs):
        if entry_type(entry) != "message.output":
            continue
        if agent_id is not None and getattr(entry, "agent_id", None) != agent_id:
            continue
        return content_to_text(entry.content).strip()
    return ""


def function_calls(outputs: list[Any]) -> list[Any]:
    return [entry for entry in outputs if entry_type(entry) == "function.call"]


def user_entry(chunks: list[dict[str, Any]]) -> dict[str, Any]:
    """Wrap content chunks in a message entry — ``inputs`` takes entries, not chunks."""
    return {
        "type": "message.input",
        "object": "entry",
        "role": "user",
        "content": chunks,
    }


def json_payload(text: str) -> dict[str, Any]:
    """Parse the first JSON object in an agent reply.

    Tolerates a ```json fence, and trailing content: agents under a strict
    response_format sometimes emit the same object twice in one message, which
    ``json.loads`` rejects as "Extra data".
    """
    candidate = text.strip()
    if candidate.startswith("```"):
        candidate = candidate.split("```")[1]
        candidate = candidate.removeprefix("json").strip()
    start = candidate.find("{")
    if start == -1:
        raise AgentRunError(f"Agent reply contained no JSON object: {text[:200]}")
    try:
        decoded, _ = json.JSONDecoder().raw_decode(candidate[start:])
    except json.JSONDecodeError as exc:
        raise AgentRunError(f"Agent reply was not valid JSON: {text[:200]}") from exc
    if not isinstance(decoded, dict):
        raise AgentRunError("Agent reply JSON must be an object.")
    return decoded


@dataclass
class PersonalDocumentRun:
    """One document's journey through the agent chain.

    Held in ``st.session_state`` between Streamlit reruns, so it stays a plain data
    object: every method opens its own client and the only durable handle is
    ``conversation_id``.
    """

    file_id: str
    filename: str
    content_type: str = "application/pdf"
    confidence_threshold: float = 0.9

    conversation_id: Optional[str] = None
    extraction_conversation_id: Optional[str] = None
    steps: dict[str, dict[str, Any]] = field(default_factory=_pending_steps)
    classification: Optional[dict[str, Any]] = None
    extraction: Optional[dict[str, Any]] = None
    preprocessing: Optional[dict[str, Any]] = None
    pending: Optional[str] = None
    error: Optional[str] = None

    # Resolved during the run.
    _registry: Optional[AgentRegistry] = None
    _processed_file_id: Optional[str] = None
    _processed_filename: Optional[str] = None
    _processed_content_type: Optional[str] = None
    _phase_one_outputs: list[Any] = field(default_factory=list)
    _applied_operations: list[str] = field(default_factory=list)
    _preview_sent: bool = False
    _answered_tool_calls: set[str] = field(default_factory=set)
    _tool_calls_made: int = 0
    _seen_entry_ids: set[str] = field(default_factory=set)

    @property
    def is_image(self) -> bool:
        return self.content_type.startswith("image/")

    @property
    def done(self) -> bool:
        return self.extraction is not None or self.error is not None

    def result(self) -> dict[str, Any]:
        """The same payload shape the workflow returns as ``structuredContent``."""
        return {
            "filename": self.filename,
            "conversation_id": self.conversation_id,
            "extraction_conversation_id": self.extraction_conversation_id,
            "preprocessing": self.preprocessing,
            "classification": self.classification,
            "personal_document_info": self.extraction,
        }

    # ── Phase driving ─────────────────────────────────────────────────────────

    async def start(self) -> "PersonalDocumentRun":
        """Run phases 1–3, stopping early if a human has to pick the category."""
        async with mistral_client() as client:
            self._registry = await ensure_registry(client)
            self._processed_file_id = self.file_id
            self._processed_filename = self.filename
            self._processed_content_type = self.content_type

            await self._phase_one(client)
            classification = await self._phase_two(client)

            self.classification = classification
            if classification["confidence"] < self.confidence_threshold:
                self.steps["classify"] = {
                    "status": "waiting_human",
                    "result": classification,
                }
                self.pending = "category"
                return self

            self.steps["classify"] = {"status": "done", "result": classification}
            await self._extract(client, classification["category"])
        return self

    async def resume_with_category(self, category: str) -> "PersonalDocumentRun":
        """Apply a reviewer's category and finish the parked run."""
        if self.pending != "category":
            raise AgentRunError("This run is not waiting for a category.")
        if self.conversation_id is None:
            raise AgentRunError("This run has no conversation to resume.")

        classification = dict(self.classification or {})
        classification["category"] = category
        classification["confidence"] = 1.0
        classification["explanation"] = f"Manually selected category: {category}"
        self.classification = classification
        self.pending = None
        self.steps["classify"] = {"status": "done", "result": classification}

        async with mistral_client() as client:
            if self._registry is None:
                self._registry = await ensure_registry(client)
            await self._extract(client, category)
        return self

    # ── Phase 1: supervisor → preprocessor (images) or → classifier (PDFs) ────

    async def _phase_one(self, client: Mistral) -> None:
        assert self._registry is not None
        self.steps["preprocess"]["status"] = "running"

        chunks: list[dict[str, Any]] = [{"type": "text", "text": self._opening_brief()}]
        if not self.is_image:
            # A PDF goes straight to classification, so it needs the document now.
            chunks.append(await self._document_chunk(client))
            self.steps["ocr"] = {
                "status": "done",
                "result": "Document prepared for Document QnA (OCR handled by Mistral Document AI).",
            }

        response = await call_with_retry(
            lambda: client.beta.conversations.start_async(
                agent_id=self._registry.supervisor,
                handoff_execution="server",
                store=True,
                inputs=[user_entry(chunks)],
            ),
            "conversations.start",
        )
        self.conversation_id = response.conversation_id
        logger.info(
            "phase 1 started conversation %s (%s)",
            self.conversation_id,
            "image" if self.is_image else "pdf",
        )
        outputs = await self._pump(client, response)

        if not self.is_image:
            self.preprocessing = {
                "status": "skipped",
                "file_id": self.file_id,
                "filename": self.filename,
                "content_type": self.content_type,
                "operations": [],
                "reason": "Preprocessing is currently supported for images only.",
            }
            self.steps["preprocess"] = {"status": "done", "result": self.preprocessing}
            self._phase_one_outputs = outputs
            return

        try:
            self.preprocessing = self._read_preprocessing_decision(outputs)
            self._processed_file_id = self.preprocessing["file_id"]
            self._processed_filename = self.preprocessing["filename"]
            self._processed_content_type = (
                self.preprocessing["content_type"] or self.content_type
            )
        except Exception as exc:  # noqa: BLE001 - degrade exactly like the workflow
            logger.warning("Preprocessing failed, using the original file: %s", exc)
            self.preprocessing = {
                "status": "skipped",
                "file_id": self.file_id,
                "filename": self.filename,
                "content_type": self.content_type,
                "operations": [],
                "error": str(exc),
            }
        self.steps["preprocess"] = {"status": "done", "result": self.preprocessing}
        self._phase_one_outputs = outputs

    def _opening_brief(self) -> str:
        kind = "an image" if self.is_image else "a PDF"
        brief = (
            f"A new personal document has arrived.\n"
            f"- filename: {self.filename}\n"
            f"- file_id: {self.file_id}\n"
            f"- content_type: {self.content_type}\n"
            f"This document is {kind}."
        )
        if self.is_image:
            return (
                f"{brief} Hand off to the preprocessing agent, which should inspect "
                f"file_id '{self.file_id}' and improve it for OCR."
            )
        return f"{brief} Skip preprocessing and hand off to the classifier.\n\n{classification_prompt(self.filename)}"

    def _read_preprocessing_decision(self, outputs: list[Any]) -> dict[str, Any]:
        assert self._registry is not None
        text = message_text(outputs, agent_id=self._registry.preprocessor)
        if not text:
            raise AgentRunError("The preprocessing agent returned no decision.")
        decision = validate_preprocessing_decision(json_payload(text), self.file_id)
        return {
            "status": "done",
            "file_id": decision.final_file_id,
            "filename": enhanced_image_filename(self.filename)
            if decision.operations
            else self.filename,
            "content_type": "image/png" if decision.operations else None,
            "operations": decision.operations,
            "rationale": decision.rationale,
        }

    # ── Phase 2: classification ───────────────────────────────────────────────

    async def _phase_two(self, client: Mistral) -> dict[str, Any]:
        assert self._registry is not None
        self.steps["classify"]["status"] = "running"

        if not self.is_image:
            # The PDF chain already reached the classifier in phase 1.
            return self._read_classification(self._phase_one_outputs)

        logger.info("phase 2: handing the enhanced document to the classifier")
        self.steps["ocr"]["status"] = "running"
        chunk = await self._document_chunk(client)
        self.steps["ocr"] = {
            "status": "done",
            "result": "Document prepared for Document QnA (OCR handled by Mistral Document AI).",
        }

        response = await self._append(
            client,
            inputs=[
                user_entry(
                    [
                        {
                            "type": "text",
                            "text": (
                                "Preprocessing is complete. Hand off to the classifier "
                                "now.\n\n"
                                + classification_prompt(
                                    self._processed_filename or self.filename
                                )
                            ),
                        },
                        chunk,
                    ]
                )
            ],
        )
        outputs = await self._pump(client, response)
        return self._read_classification(outputs)

    def _read_classification(self, outputs: list[Any]) -> dict[str, Any]:
        assert self._registry is not None
        text = message_text(outputs, agent_id=self._registry.classifier)
        if not text:
            raise AgentRunError("The classifier agent returned no classification.")
        parsed = PersonalDocumentClassification.model_validate(json_payload(text))
        return parsed.model_dump(mode="json")

    # ── Phase 3: extraction ───────────────────────────────────────────────────

    async def _extract(self, client: Mistral, category: str) -> None:
        """Extract with the category's own agent, in its own conversation.

        Deliberately not a handoff. While the classifier held handoffs to the
        extractors it took one in the same response as the classification, running
        extraction before the confidence gate could stop it. Addressing the extractor
        directly keeps the gate authoritative, and extraction needs no history beyond
        the document and the category.
        """
        assert self._registry is not None
        self.steps["extract"]["status"] = "running"
        extractor_id = self._registry.extractor_for(category)
        filename = self._processed_filename or self.filename
        logger.info("phase 3: extracting with %s", extractor_name(category))

        chunk = await self._document_chunk(client)
        prompt = extraction_prompt(filename, category)
        model = get_personal_extraction_output_model(category)

        for attempt in range(1, EXTRACTION_ATTEMPTS + 1):
            if attempt == 1:
                response = await call_with_retry(
                    lambda: client.beta.conversations.start_async(
                        agent_id=extractor_id,
                        store=True,
                        inputs=[user_entry([{"type": "text", "text": prompt}, chunk])],
                    ),
                    "conversations.start (extraction)",
                )
                self.extraction_conversation_id = response.conversation_id
            else:
                response = await call_with_retry(
                    lambda: client.beta.conversations.append_async(
                        conversation_id=self.extraction_conversation_id,
                        inputs=[
                            user_entry(
                                [
                                    {
                                        "type": "text",
                                        "text": extraction_retry_nudge(category),
                                    }
                                ]
                            )
                        ],
                    ),
                    "conversations.append (extraction retry)",
                )

            text = message_text(response.outputs)
            if not text:
                raise AgentRunError(
                    f"The {extractor_name(category)} agent returned no extraction."
                )
            try:
                parsed = model.model_validate(json_payload(text)).model_dump(
                    mode="json"
                )
            except (AgentRunError, ValidationError) as exc:
                if attempt == EXTRACTION_ATTEMPTS:
                    raise
                logger.warning(
                    "extraction reply was unusable (%s); asking once more", exc
                )
                continue

            # Valid JSON is not enough. Two shapes seen live: MRZ text copied into
            # an ordinary field, and sibling fields swallowed into a string value.
            unusable = find_unusable_fields(parsed, category)
            if unusable and attempt < EXTRACTION_ATTEMPTS:
                logger.warning(
                    "unusable value in %s; asking once more", ", ".join(unusable)
                )
                continue
            break

        warnings: list[str] = []
        if unusable := find_unusable_fields(parsed, category):
            # Report nothing rather than garbage, and say so out loud.
            parsed = strip_unusable_fields(parsed, unusable)
            warnings.append(
                "Discarded an unusable value the model placed in: "
                + ", ".join(unusable)
            )
            logger.warning("discarded unusable values in %s", ", ".join(unusable))

        self.extraction = enrich_with_mrz_fallback(parsed, category)
        if warnings:
            self.extraction["warnings"] = warnings
        self.steps["extract"] = {"status": "done", "result": self.extraction}

    # ── Shared plumbing ───────────────────────────────────────────────────────

    async def _document_chunk(self, client: Mistral) -> dict[str, str]:
        """Signed-URL content chunk for whichever file the chain should read."""
        file_id = self._processed_file_id or self.file_id
        return build_mistral_document_chunk(
            await get_signed_url(client, file_id),
            self._processed_filename or self.filename,
            self._processed_content_type or self.content_type,
        )

    async def _pump(self, client: Mistral, response: Any) -> list[Any]:
        """Execute local function tools until the chain stops asking for them.

        Returns every output entry seen, so a caller can attribute messages to the
        agent that produced them.
        """
        collected: list[Any] = self._remember(response.outputs)
        for _ in range(MAX_TOOL_ITERATIONS):
            for handoff in response.outputs:
                if entry_type(handoff) == "agent.handoff":
                    logger.info(
                        "handoff %s -> %s",
                        handoff.previous_agent_name,
                        handoff.next_agent_name,
                    )
            # A call already answered must never be answered twice: the API rejects
            # the whole append with "tool_call_ids that already have a result".
            calls = [
                call
                for call in function_calls(response.outputs)
                if call.tool_call_id not in self._answered_tool_calls
            ]
            if not calls:
                return collected

            inputs: list[dict[str, Any]] = []
            previews: list[str] = []
            for call in calls:
                result, preview_url = await self._run_tool(client, call)
                self._answered_tool_calls.add(call.tool_call_id)
                inputs.append(result)
                if preview_url:
                    previews.append(preview_url)
            # Let the agent *see* what it produced, the way the workflow's agent does.
            if previews:
                inputs.append(
                    user_entry(
                        [{"type": "image_url", "image_url": url} for url in previews]
                    )
                )

            response = await self._append(client, inputs=inputs)
            collected.extend(self._remember(response.outputs))

        # Out of iterations. Returning what we have lets the caller degrade the way a
        # failed preprocessing decision already does — falling back to the original
        # file — rather than losing the whole run to a chatty agent.
        logger.warning(
            "the agent chain exceeded %d tool iterations; continuing without it",
            MAX_TOOL_ITERATIONS,
        )
        return collected

    def _remember(self, outputs: list[Any]) -> list[Any]:
        """Record which entries this run has already seen, for history recovery."""
        for entry in outputs:
            if entry_id := getattr(entry, "id", None):
                self._seen_entry_ids.add(str(entry_id))
        return list(outputs)

    async def _append(self, client: Mistral, inputs: list[dict[str, Any]]) -> Any:
        """Append entries to this run's conversation, retrying rate limits.

        A retried append is not free: if the original attempt actually succeeded and
        only its response was lost (a read timeout), re-sending the same function
        results is rejected. That rejection means the work *did* land, so the
        recovery is to read back what we have not seen rather than to fail the run.
        """
        logger.info("appending %d entry/entries", len(inputs))
        try:
            return await call_with_retry(
                lambda: client.beta.conversations.append_async(
                    conversation_id=self.conversation_id,
                    handoff_execution="server",
                    inputs=inputs,
                ),
                "conversations.append",
            )
        except SDKError as exc:
            if "already have a result" not in str(exc):
                raise
            logger.warning(
                "append had already been applied; recovering from conversation history"
            )
            return await self._recover_from_history(client)

    async def _recover_from_history(self, client: Mistral) -> Any:
        """Return the entries appended since this run last saw the conversation."""
        history = await call_with_retry(
            lambda: client.beta.conversations.get_history_async(
                conversation_id=self.conversation_id
            ),
            "conversations.get_history",
        )
        unseen = [
            entry
            for entry in (history.entries or [])
            if str(getattr(entry, "id", "")) not in self._seen_entry_ids
        ]
        logger.info("recovered %d unseen entry/entries", len(unseen))
        return SimpleNamespace(conversation_id=self.conversation_id, outputs=unseen)

    @staticmethod
    def _requested_operation(call: Any) -> Optional[str]:
        """The operation an apply call is asking for, or None if unreadable."""
        arguments = call.arguments
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                return None
        if isinstance(arguments, dict):
            operation = arguments.get("operation")
            return str(operation) if operation is not None else None
        return None

    async def _run_tool(
        self, client: Mistral, call: Any
    ) -> tuple[dict[str, Any], Optional[str]]:
        """Run one tool call. Returns its function.result entry and any preview URL."""
        payload: dict[str, Any]
        self._tool_calls_made += 1
        if call.name not in TOOL_DISPATCH:
            payload = {"error": f"Unknown tool '{call.name}'."}
        elif self._tool_calls_made > MAX_TOOL_CALLS:
            payload = {
                "error": (
                    f"Tool budget spent ({MAX_TOOL_CALLS} calls). Reply with the "
                    "final JSON decision now; do not call any more tools."
                )
            }
        elif (
            call.name == APPLY_TOOL_NAME
            and len(self._applied_operations) >= MAX_PREPROCESSING_OPERATIONS
        ):
            payload = {
                "error": (
                    f"Operation budget spent ({MAX_PREPROCESSING_OPERATIONS} applied: "
                    f"{', '.join(self._applied_operations)}). Reply with the final "
                    "JSON decision now; do not call any more tools."
                )
            }
        elif call.name == APPLY_TOOL_NAME and (
            self._requested_operation(call) in self._applied_operations
        ):
            # The prompt forbids repeats, and `validate_preprocessing_decision`
            # rejects a decision that lists one twice — which would throw away every
            # operation applied so far. Enforce it here instead of hoping.
            repeated = self._requested_operation(call)
            payload = {
                "error": (
                    f"'{repeated}' has already been applied and may not be used "
                    f"twice. Applied so far: {', '.join(self._applied_operations)}. "
                    "Choose a different operation or reply with the final JSON."
                )
            }
        else:
            arguments = call.arguments
            if isinstance(arguments, str):
                arguments = json.loads(arguments)
            try:
                payload = await TOOL_DISPATCH[call.name](client, **dict(arguments))
            except Exception as exc:  # noqa: BLE001 - the agent must see the failure
                logger.warning("Tool %s failed: %s", call.name, exc)
                payload = {"error": str(exc)}
            else:
                if operation := payload.get("operation"):
                    self._applied_operations.append(str(operation))

        # The signed preview URL goes as an image chunk rather than as JSON text, and
        # only once: re-sending a full-page preview each turn blows the token budget.
        preview_url = payload.pop("preview_url", None)
        if self._preview_sent:
            preview_url = None
        elif preview_url:
            self._preview_sent = True
        entry = {
            "type": "function.result",
            "object": "entry",
            "tool_call_id": call.tool_call_id,
            "result": json.dumps(payload, ensure_ascii=False),
        }
        return entry, preview_url


async def process_document(
    file_id: str,
    filename: str,
    content_type: str = "application/pdf",
    confidence_threshold: float = 0.9,
) -> PersonalDocumentRun:
    """Convenience entry point: start a run and return it (possibly parked)."""
    run = PersonalDocumentRun(
        file_id=file_id,
        filename=filename,
        content_type=content_type,
        confidence_threshold=confidence_threshold,
    )
    return await run.start()
