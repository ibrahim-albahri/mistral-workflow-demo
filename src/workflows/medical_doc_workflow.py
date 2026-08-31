import asyncio
import logging
import os
from functools import lru_cache
from datetime import timedelta
from typing import Optional

import mistralai.workflows as workflows
import mistralai.workflows.plugins.mistralai as workflows_mistralai
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, create_model

from shared.extraction_fields import COMMON_FIELDS, DOCUMENT_CATEGORIES, SPECIFIC_FIELDS

load_dotenv(override=True)

for name in ("mistralai_workflows", "httpx", "httpcore"):
    logging.getLogger(name).setLevel(logging.WARNING)


class ManualCategorySignal(BaseModel):
    category: str


class DocumentClassification(BaseModel):
    category: str = Field(description=f"One of: {', '.join(DOCUMENT_CATEGORIES)}")
    confidence: float = Field(ge=0.0, le=1.0)
    explanation: str


@lru_cache(maxsize=None)
def get_extraction_output_model(category: str) -> type[BaseModel]:
    common_model = create_model(
        "CommonExtractionFields",
        __config__=ConfigDict(extra="forbid"),
        **{key: (Optional[str], None) for key, _ in COMMON_FIELDS},
    )
    specific_model = create_model(
        f"SpecificExtractionFields_{category}",
        __config__=ConfigDict(extra="forbid"),
        **{key: (Optional[str], None) for key, _ in SPECIFIC_FIELDS.get(category, [])},
    )
    return create_model(
        f"ExtractionOutput_{category}",
        __config__=ConfigDict(extra="forbid"),
        common=(common_model, ...),
        specific=(specific_model, ...),
    )


# ── Activities ────────────────────────────────────────────────────────────────

# Resolve the uploaded file ID into a temporary signed URL for Document QnA.
@workflows.activity(start_to_close_timeout=timedelta(minutes=5), retry_policy_max_attempts=2)
async def get_document_signed_url(file_id: str) -> str:
    client = workflows_mistralai.get_mistral_client()
    signed_url = await client.files.get_signed_url_async(file_id=file_id)
    return signed_url.url

# Classify the document category directly from the document URL using structured output.
@workflows.activity(start_to_close_timeout=timedelta(minutes=2), retry_policy_max_attempts=2)
async def classify_document(document_url: str, filename: str) -> dict:
    client = workflows_mistralai.get_mistral_client()
    model = os.environ.get("MISTRAL_CLASSIFIER_MODEL", "mistral-medium-latest")
    response = await client.chat.parse_async(
        response_format=DocumentClassification,
        model=model,
        temperature=0.0,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an expert in medical document classification. "
                    "Return only valid JSON that matches the schema."
                ),
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            f"Classify the medical document '{filename}' into exactly one category from:\n"
                            + "\n".join(f"- {c}" for c in DOCUMENT_CATEGORIES)
                            + "\n\n"
                            "Return confidence between 0 and 1 and a short explanation."
                        ),
                    },
                    {
                        "type": "document_url",
                        "document_url": document_url,
                        "document_name": filename,
                    },
                ],
            },
        ],
    )
    parsed = response.choices[0].message.parsed if response.choices and response.choices[0].message else None
    if parsed is None:
        raise RuntimeError("Classification response could not be parsed.")
    return parsed.model_dump(mode="json")

# Extract common and category-specific fields from the document with schema-constrained JSON.
@workflows.activity(start_to_close_timeout=timedelta(minutes=2), retry_policy_max_attempts=2)
async def extract_patient_info(document_url: str, filename: str, category: str) -> dict:
    client = workflows_mistralai.get_mistral_client()
    model = os.environ.get("MISTRAL_EXTRACTOR_MODEL", "mistral-medium-latest")
    extraction_model = get_extraction_output_model(category)
    common_fields_text = "\n".join(f"- {key}" for key, _ in COMMON_FIELDS)
    specific_fields_text = "\n".join(f"- {key}" for key, _ in SPECIFIC_FIELDS.get(category, []))
    response = await client.chat.parse_async(
        response_format=extraction_model,
        model=model,
        temperature=0.0,
        messages=[
            {
                "role": "system",
                "content": (
                    "You extract administrative and medical information from a medical document. "
                    "Return only valid JSON that matches the schema. "
                    "If a field is missing, set it to null."
                ),
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            f"Extract fields from '{filename}' for category '{category}'.\n\n"
                            "Populate these common fields:\n"
                            f"{common_fields_text}\n\n"
                            "Populate these category-specific fields:\n"
                            f"{specific_fields_text if specific_fields_text else '- (none)'}\n\n"
                            "Return null for missing values."
                        ),
                    },
                    {
                        "type": "document_url",
                        "document_url": document_url,
                        "document_name": filename,
                    },
                ],
            },
        ],
    )
    parsed = response.choices[0].message.parsed if response.choices and response.choices[0].message else None
    if parsed is None:
        raise RuntimeError("Extraction response could not be parsed.")
    return parsed.model_dump(mode="json")


# ── Workflow ──────────────────────────────────────────────────────────────────

@workflows.workflow.define(name="pdf_ocr_workflow")
class PdfOcrWorkflow(workflows.InteractiveWorkflow):
    def __init__(self):
        # UI-facing step state exposed through the `get_steps` query.
        self.steps = {
            "ocr": {"status": "pending", "result": None},
            "classify": {"status": "pending", "result": None},
            "extract": {"status": "pending", "result": None},
        }
        # Set by the `manual_category` signal when a user overrides classification.
        self._manual_category = None

    @workflows.workflow.query(name="get_steps")
    def get_steps(self) -> dict:
        # Queries are synchronous/read-only: frontend polls this for progress.
        return self.steps

    @workflows.workflow.signal(name="manual_category")
    async def manual_category_signal(self, payload: ManualCategorySignal) -> None:
        # Signals are async/write: this mutates workflow state while run() is active.
        self._manual_category = payload.category

    @workflows.workflow.entrypoint
    async def run(
        self,
        file_id: str,
        filename: str,
        confidence_threshold: float = 0.9,
        manual_review_timeout_seconds: Optional[float] = None,
    ) -> workflows_mistralai.ChatAssistantWorkflowOutput:
        # TodoList items let clients display coarse workflow progress in addition to step data.
        ocr_item = workflows_mistralai.TodoListItem(
            title="Prepare document for Document QnA",
            description="Generate a signed URL so Mistral Document AI can read the document.",
        )
        classify_item = workflows_mistralai.TodoListItem(
            title="Classify document type",
            description="Predict the medical document category with confidence.",
        )
        extract_item = workflows_mistralai.TodoListItem(
            title="Extract structured fields",
            description="Extract patient and document-specific fields.",
        )

        async with workflows_mistralai.TodoList(items=[ocr_item, classify_item, extract_item]):
            self.steps["ocr"]["status"] = "running"
            async with ocr_item:
                signed_document_url = await get_document_signed_url(file_id)
            self.steps["ocr"] = {
                "status": "done",
                "result": "Document prepared for Document QnA (OCR handled by Mistral Document AI).",
            }

            self.steps["classify"]["status"] = "running"
            async with classify_item:
                classification = await classify_document(signed_document_url, filename)

                if classification.get("confidence", 0.0) < confidence_threshold:
                    # Low confidence enters a human-in-the-loop branch.
                    self.steps["classify"] = {"status": "waiting_human", "result": classification}
                    try:
                        # wait_condition pauses deterministically until signal or timeout.
                        await workflows.workflow.wait_condition(
                            lambda: self._manual_category is not None,
                            timeout=manual_review_timeout_seconds,
                            timeout_summary="manual_category_review",
                        )
                    except asyncio.TimeoutError:
                        # Non-interactive callers can set a timeout to avoid waiting forever.
                        classification["explanation"] = (
                            f"{classification.get('explanation', '').strip()} "
                            "Manual review timed out; using model-predicted category."
                        ).strip()
                    else:
                        # Manual override takes precedence when a signal arrives in time.
                        classification["category"] = self._manual_category
                        classification["confidence"] = 1.0
                        classification["explanation"] = f"Manually selected category: {self._manual_category}"

            self.steps["classify"] = {"status": "done", "result": classification}

            self.steps["extract"]["status"] = "running"
            async with extract_item:
                patient_info = await extract_patient_info(signed_document_url, filename, classification["category"])
            self.steps["extract"] = {"status": "done", "result": patient_info}

        return workflows_mistralai.ChatAssistantWorkflowOutput(
            # Return both human-readable text and structured payload for the UI.
            content=[workflows_mistralai.TextOutput(text=f"Processing complete for {filename}.")],
            structuredContent={
                "filename": filename,
                "ocr_text": "Document processed with Document QnA (no standalone OCR text payload).",
                "classification": classification,
                "patient_info": patient_info,
            },
        )
        
async def main() -> None:
    print("Worker ready — waiting for tasks...\n")
    await workflows.run_worker([PdfOcrWorkflow])


if __name__ == "__main__":
    asyncio.run(main())
