"""OCR-backed Batch API processing for personal documents."""

import asyncio
import json
import os
from datetime import timedelta
from typing import Any

import mistralai.workflows as workflows
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict

from shared.document_media import build_mistral_document_chunk
from shared.extraction_fields import (
    PERSONAL_COMMON_FIELDS,
    PERSONAL_DOCUMENT_CATEGORIES,
    PERSONAL_DOCUMENT_SPECIFIC_FIELDS,
)
from workflows.personal_doc_workflow import (
    PersonalDocumentCategory,
    PersonalDocumentClassification,
    _fields_text,
    enrich_with_mrz_fallback,
    get_personal_extraction_output_model,
)

load_dotenv(override=True)

MAX_BATCH_DOCUMENTS = 100
POLL_INTERVAL_SECONDS = 5


def _batch_client():
    """Create the API client inside an activity, outside the workflow sandbox."""
    from mistralai.client import Mistral

    return Mistral(
        api_key=os.environ["MISTRAL_API_KEY"],
        server_url=os.environ.get("SERVER_URL", "https://api.mistral.ai"),
    )


class BatchDocumentInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    document_id: str
    file_id: str
    filename: str
    content_type: str


class BatchManualCategorySignal(BaseModel):
    model_config = ConfigDict(extra="forbid")

    document_id: str
    category: PersonalDocumentCategory


def batch_jsonl(entries: list[dict[str, Any]]) -> bytes:
    """Encode Batch API request rows without writing temporary files."""
    return b"\n".join(
        json.dumps(entry, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        for entry in entries
    )


def _mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return model_dump(mode="json")
    return {}


def parse_batch_result_rows(raw_jsonl: str) -> dict[str, dict[str, Any]]:
    """Map Mistral Batch JSONL output/error rows back to custom document IDs."""
    parsed: dict[str, dict[str, Any]] = {}
    for line in raw_jsonl.splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        custom_id = str(row.get("custom_id", ""))
        if not custom_id:
            continue
        error = row.get("error")
        response = row.get("response")
        if error:
            parsed[custom_id] = {"ok": False, "error": str(error)}
        elif isinstance(response, dict):
            parsed[custom_id] = {"ok": True, "body": response.get("body", response)}
        else:
            parsed[custom_id] = {"ok": False, "error": "Batch response is missing."}
    return parsed


def ocr_markdown_from_body(body: dict[str, Any]) -> str:
    pages = body.get("pages") or []
    return "\n\n".join(
        page.get("markdown", "") for page in pages if isinstance(page, dict)
    ).strip()


def chat_json_from_body(body: dict[str, Any]) -> dict[str, Any]:
    choices = body.get("choices") or []
    if not choices or not isinstance(choices[0], dict):
        raise ValueError("Batch chat response has no choices.")
    message = choices[0].get("message") or {}
    content = message.get("content")
    if not isinstance(content, str):
        raise ValueError("Batch chat response has no JSON content.")
    decoded = json.loads(content)
    if not isinstance(decoded, dict):
        raise ValueError("Batch chat response JSON must be an object.")
    return decoded


def _json_schema_response_format(name: str, model: type[BaseModel]) -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {"name": name, "schema": model.model_json_schema()},
    }


def build_ocr_batch_entries(documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "custom_id": document["document_id"],
            "body": {
                "document": build_mistral_document_chunk(
                    document["signed_url"],
                    document["filename"],
                    document["content_type"],
                )
            },
        }
        for document in documents
    ]


def build_classification_batch_entries(
    documents: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    schema = _json_schema_response_format(
        "personal_document_classification", PersonalDocumentClassification
    )
    categories = "\n".join(f"- {category}" for category in PERSONAL_DOCUMENT_CATEGORIES)
    return [
        {
            "custom_id": document["document_id"],
            "body": {
                "temperature": 0.0,
                "response_format": schema,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are an expert in classifying personal identity and compliance documents. "
                            "Classify from the OCR text, not from the filename. Return only valid JSON."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"Classify '{document['filename']}' into exactly one category:\n{categories}\n\n"
                            "Return confidence between 0 and 1 and a short explanation.\n\n"
                            f"OCR markdown:\n{document['ocr_markdown']}"
                        ),
                    },
                ],
            },
        }
        for document in documents
    ]


def build_extraction_batch_entries(
    documents: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    common_fields_text = _fields_text(PERSONAL_COMMON_FIELDS)
    for document in documents:
        category = document["classification"]["category"]
        extraction_model = get_personal_extraction_output_model(category)
        specific_fields_text = _fields_text(
            PERSONAL_DOCUMENT_SPECIFIC_FIELDS.get(category, [])
        )
        entries.append(
            {
                "custom_id": document["document_id"],
                "body": {
                    "temperature": 0.0,
                    "response_format": _json_schema_response_format(
                        f"personal_extraction_{category}", extraction_model
                    ),
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                "You extract identity and compliance information from personal documents. "
                                "Return only valid JSON. Treat visibly printed fields as the source of truth; "
                                "never infer or correct values. For an MRZ, transcribe raw lines only."
                            ),
                        },
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": (
                                        f"Extract fields from '{document['filename']}' for '{category}'.\n\n"
                                        f"Common fields:\n{common_fields_text}\n\n"
                                        "Category-specific fields:\n"
                                        f"{specific_fields_text or '- (none)'}\n\n"
                                        "OCR markdown (use as an aid; verify against the source document):\n"
                                        f"{document['ocr_markdown']}"
                                    ),
                                },
                                build_mistral_document_chunk(
                                    document["signed_url"],
                                    document["filename"],
                                    document["content_type"],
                                ),
                            ],
                        },
                    ],
                },
            }
        )
    return entries


@workflows.activity(
    start_to_close_timeout=timedelta(minutes=10), retry_policy_max_attempts=2
)
async def get_batch_signed_urls(documents: list[dict[str, Any]]) -> dict[str, str]:
    async with _batch_client() as client:
        signed_urls = await asyncio.gather(
            *[
                client.files.get_signed_url_async(file_id=document["file_id"])
                for document in documents
            ]
        )
    return {
        document["document_id"]: signed_url.url
        for document, signed_url in zip(documents, signed_urls, strict=True)
    }


@workflows.activity(
    start_to_close_timeout=timedelta(minutes=10), retry_policy_max_attempts=2
)
async def submit_batch_job(
    entries: list[dict[str, Any]], endpoint: str, job_type: str
) -> dict[str, Any]:
    models = {
        "ocr": os.environ.get("MISTRAL_OCR_MODEL", "mistral-ocr-latest"),
        "classification": os.environ.get(
            "MISTRAL_CLASSIFIER_MODEL", "mistral-medium-latest"
        ),
        "extraction": os.environ.get(
            "MISTRAL_EXTRACTOR_MODEL", "mistral-medium-latest"
        ),
    }
    async with _batch_client() as client:
        upload = await client.files.upload_async(
            file={
                "file_name": f"personal-doc-{job_type}.jsonl",
                "content": batch_jsonl(entries),
            },
            purpose="batch",
        )
        job = await client.batch.jobs.create_async(
            input_files=[upload.id],
            endpoint=endpoint,
            model=models[job_type],
            metadata={"job_type": job_type},
        )
    payload = _mapping(job)
    return {"id": payload.get("id", job.id), "status": payload.get("status", "QUEUED")}


@workflows.activity(
    start_to_close_timeout=timedelta(hours=25), retry_policy_max_attempts=1
)
async def collect_batch_job(job_id: str) -> dict[str, Any]:
    async with _batch_client() as client:
        while True:
            job = await client.batch.jobs.get_async(job_id=job_id)
            payload = _mapping(job)
            status = str(payload.get("status", getattr(job, "status", "FAILED")))
            if status not in {"QUEUED", "RUNNING", "CANCELLATION_REQUESTED"}:
                break
            await asyncio.sleep(POLL_INTERVAL_SECONDS)

        outputs = ""
        errors = ""
        output_file = payload.get("output_file", getattr(job, "output_file", None))
        error_file = payload.get("error_file", getattr(job, "error_file", None))
        if output_file:
            outputs = (await client.files.download_async(file_id=output_file)).text
        if error_file:
            errors = (await client.files.download_async(file_id=error_file)).text
    return {"status": status, "outputs": outputs, "errors": errors}


@workflows.workflow.define(
    name="batch_personal_document_workflow", execution_timeout=timedelta(days=2)
)
class BatchPersonalDocumentWorkflow(workflows.InteractiveWorkflow):
    def __init__(self):
        self.documents: dict[str, dict[str, Any]] = {}
        self.jobs: list[dict[str, Any]] = []

    @workflows.workflow.query(name="get_batch_status")
    def get_batch_status(self) -> dict[str, Any]:
        visible_fields = (
            "document_id",
            "filename",
            "content_type",
            "status",
            "classification",
            "extraction",
            "error",
        )
        documents = [
            {field: document.get(field) for field in visible_fields}
            for document in self.documents.values()
        ]
        counts: dict[str, int] = {}
        for document in documents:
            status = document["status"]
            counts[status] = counts.get(status, 0) + 1
        return {"counts": counts, "jobs": self.jobs, "documents": documents}

    @workflows.workflow.signal(name="manual_batch_category")
    async def manual_batch_category(self, payload: BatchManualCategorySignal) -> None:
        document = self.documents.get(payload.document_id)
        if document and document["status"] == "awaiting_review":
            document["classification"]["category"] = payload.category.value
            document["classification"]["confidence"] = 1.0
            document["classification"]["explanation"] = (
                f"Manually selected category: {payload.category.value}"
            )
            document["status"] = "queued_for_extraction"

    async def _run_job(
        self, entries: list[dict[str, Any]], endpoint: str, job_type: str
    ) -> dict[str, dict[str, Any]]:
        job = await submit_batch_job(entries, endpoint, job_type)
        self.jobs.append({"id": job["id"], "type": job_type, "status": job["status"]})
        collected = await collect_batch_job(job["id"])
        self.jobs[-1]["status"] = collected["status"]
        rows = parse_batch_result_rows(collected["outputs"])
        rows.update(parse_batch_result_rows(collected["errors"]))
        return rows

    async def _extract(self, documents: list[dict[str, Any]]) -> None:
        if not documents:
            return
        for document in documents:
            document["status"] = "extracting"
        signed_urls = await get_batch_signed_urls(documents)
        for document in documents:
            document["signed_url"] = signed_urls[document["document_id"]]
        rows = await self._run_job(
            build_extraction_batch_entries(documents),
            "/v1/chat/completions",
            "extraction",
        )
        for document in documents:
            row = rows.get(document["document_id"])
            if not row or not row.get("ok"):
                document["status"] = "failed"
                document["error"] = (row or {}).get(
                    "error", "Extraction request failed."
                )
                continue
            try:
                extracted = chat_json_from_body(row["body"])
                document["extraction"] = enrich_with_mrz_fallback(
                    extracted, document["classification"]["category"]
                )
                document["status"] = "completed"
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                document["status"] = "failed"
                document["error"] = f"Invalid extraction response: {exc}"

    @workflows.workflow.entrypoint
    async def run(
        self, documents: list[BatchDocumentInput], confidence_threshold: float = 0.9
    ) -> dict[str, Any]:
        if not documents or len(documents) > MAX_BATCH_DOCUMENTS:
            raise ValueError(
                f"A batch must contain between 1 and {MAX_BATCH_DOCUMENTS} documents."
            )
        if len({document.document_id for document in documents}) != len(documents):
            raise ValueError("Each batch document requires a unique document_id.")

        self.documents = {
            document.document_id: {
                **document.model_dump(mode="json"),
                "status": "ocr_pending",
                "classification": None,
                "extraction": None,
                "error": None,
            }
            for document in documents
        }
        all_documents = list(self.documents.values())

        signed_urls = await get_batch_signed_urls(all_documents)
        for document in all_documents:
            document["signed_url"] = signed_urls[document["document_id"]]
            document["status"] = "ocr_running"
        ocr_rows = await self._run_job(
            build_ocr_batch_entries(all_documents),
            "/v1/ocr",
            "ocr",
        )
        ocr_successes: list[dict[str, Any]] = []
        for document in all_documents:
            row = ocr_rows.get(document["document_id"])
            if not row or not row.get("ok"):
                document["status"] = "failed"
                document["error"] = (row or {}).get("error", "OCR request failed.")
                continue
            document["ocr_markdown"] = ocr_markdown_from_body(row["body"])
            if not document["ocr_markdown"]:
                document["status"] = "failed"
                document["error"] = "OCR returned no markdown."
                continue
            document["status"] = "classification_running"
            ocr_successes.append(document)

        if ocr_successes:
            classification_rows = await self._run_job(
                build_classification_batch_entries(ocr_successes),
                "/v1/chat/completions",
                "classification",
            )
            confident: list[dict[str, Any]] = []
            for document in ocr_successes:
                row = classification_rows.get(document["document_id"])
                if not row or not row.get("ok"):
                    document["status"] = "failed"
                    document["error"] = (row or {}).get(
                        "error", "Classification request failed."
                    )
                    continue
                try:
                    classification = PersonalDocumentClassification.model_validate(
                        chat_json_from_body(row["body"])
                    ).model_dump(mode="json")
                except (TypeError, ValueError, json.JSONDecodeError) as exc:
                    document["status"] = "failed"
                    document["error"] = f"Invalid classification response: {exc}"
                    continue
                document["classification"] = classification
                if classification["confidence"] < confidence_threshold:
                    document["status"] = "awaiting_review"
                else:
                    document["status"] = "queued_for_extraction"
                    confident.append(document)
            await self._extract(confident)

        while any(
            document["status"] == "awaiting_review"
            for document in self.documents.values()
        ):
            await workflows.workflow.wait_condition(
                lambda: any(
                    document["status"] == "queued_for_extraction"
                    for document in self.documents.values()
                )
            )
            reviewed = [
                document
                for document in self.documents.values()
                if document["status"] == "queued_for_extraction"
            ]
            await self._extract(reviewed)

        status = self.get_batch_status()
        return {"counts": status["counts"], "documents": status["documents"]}
