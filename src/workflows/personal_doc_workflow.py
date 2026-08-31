import asyncio
import logging
import os
from datetime import timedelta
from functools import lru_cache
from typing import Optional

import mistralai.workflows as workflows
import mistralai.workflows.plugins.mistralai as workflows_mistralai
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, create_model

from shared.extraction_fields import (
    PERSONAL_COMMON_FIELDS,
    PERSONAL_DOCUMENT_CATEGORIES,
    PERSONAL_DOCUMENT_SPECIFIC_FIELDS,
)

load_dotenv(override=True)

for name in ("mistralai_workflows", "httpx", "httpcore"):
    logging.getLogger(name).setLevel(logging.WARNING)


class ManualCategorySignal(BaseModel):
    category: str


class PersonalDocumentClassification(BaseModel):
    category: str = Field(description=f"One of: {', '.join(PERSONAL_DOCUMENT_CATEGORIES)}")
    confidence: float = Field(ge=0.0, le=1.0)
    explanation: str


@lru_cache(maxsize=None)
def get_personal_extraction_output_model(category: str) -> type[BaseModel]:
    common_model = create_model(
        "PersonalCommonExtractionFields",
        __config__=ConfigDict(extra="forbid"),
        **{key: (Optional[str], None) for key, _ in PERSONAL_COMMON_FIELDS},
    )
    specific_model = create_model(
        f"PersonalSpecificExtractionFields_{category}",
        __config__=ConfigDict(extra="forbid"),
        **{key: (Optional[str], None) for key, _ in PERSONAL_DOCUMENT_SPECIFIC_FIELDS.get(category, [])},
    )
    return create_model(
        f"PersonalExtractionOutput_{category}",
        __config__=ConfigDict(extra="forbid"),
        common=(common_model, ...),
        specific=(specific_model, ...),
    )


@workflows.activity(start_to_close_timeout=timedelta(minutes=5), retry_policy_max_attempts=2)
async def get_personal_document_signed_url(file_id: str) -> str:
    client = workflows_mistralai.get_mistral_client()
    signed_url = await client.files.get_signed_url_async(file_id=file_id)
    return signed_url.url


@workflows.activity(start_to_close_timeout=timedelta(minutes=2), retry_policy_max_attempts=2)
async def classify_personal_document(document_url: str, filename: str) -> dict:
    client = workflows_mistralai.get_mistral_client()
    model = os.environ.get("MISTRAL_CLASSIFIER_MODEL", "mistral-medium-latest")
    response = await client.chat.parse_async(
        response_format=PersonalDocumentClassification,
        model=model,
        temperature=0.0,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an expert in classifying personal identity and compliance documents. "
                    "Return only valid JSON that matches the schema."
                ),
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            f"Classify the personal document '{filename}' into exactly one category from:\n"
                            + "\n".join(f"- {c}" for c in PERSONAL_DOCUMENT_CATEGORIES)
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
        raise RuntimeError("Personal document classification response could not be parsed.")
    return parsed.model_dump(mode="json")


@workflows.activity(start_to_close_timeout=timedelta(minutes=2), retry_policy_max_attempts=2)
async def extract_personal_document_info(document_url: str, filename: str, category: str) -> dict:
    client = workflows_mistralai.get_mistral_client()
    model = os.environ.get("MISTRAL_EXTRACTOR_MODEL", "mistral-medium-latest")
    extraction_model = get_personal_extraction_output_model(category)
    common_fields_text = "\n".join(f"- {key}" for key, _ in PERSONAL_COMMON_FIELDS)
    specific_fields_text = "\n".join(f"- {key}" for key, _ in PERSONAL_DOCUMENT_SPECIFIC_FIELDS.get(category, []))
    response = await client.chat.parse_async(
        response_format=extraction_model,
        model=model,
        temperature=0.0,
        messages=[
            {
                "role": "system",
                "content": (
                    "You extract identity and compliance information from personal documents. "
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
        raise RuntimeError("Personal document extraction response could not be parsed.")
    return parsed.model_dump(mode="json")


@workflows.workflow.define(name="personal_document_workflow")
class PersonalDocumentWorkflow(workflows.InteractiveWorkflow):
    def __init__(self):
        self.steps = {
            "ocr": {"status": "pending", "result": None},
            "classify": {"status": "pending", "result": None},
            "extract": {"status": "pending", "result": None},
        }
        self._manual_category = None

    @workflows.workflow.query(name="get_steps")
    def get_steps(self) -> dict:
        return self.steps

    @workflows.workflow.signal(name="manual_category")
    async def manual_category_signal(self, payload: ManualCategorySignal) -> None:
        self._manual_category = payload.category

    @workflows.workflow.entrypoint
    async def run(
        self,
        file_id: str,
        filename: str,
        confidence_threshold: float = 0.9,
        manual_review_timeout_seconds: Optional[float] = None,
    ) -> workflows_mistralai.ChatAssistantWorkflowOutput:
        ocr_item = workflows_mistralai.TodoListItem(
            title="Prepare personal document for Document QnA",
            description="Generate a signed URL so Mistral Document AI can read the document.",
        )
        classify_item = workflows_mistralai.TodoListItem(
            title="Classify personal document type",
            description="Predict the personal document category with confidence.",
        )
        extract_item = workflows_mistralai.TodoListItem(
            title="Extract structured fields",
            description="Extract document-specific identity and compliance fields.",
        )

        async with workflows_mistralai.TodoList(items=[ocr_item, classify_item, extract_item]):
            self.steps["ocr"]["status"] = "running"
            async with ocr_item:
                signed_document_url = await get_personal_document_signed_url(file_id)
            self.steps["ocr"] = {
                "status": "done",
                "result": "Document prepared for Document QnA (OCR handled by Mistral Document AI).",
            }

            self.steps["classify"]["status"] = "running"
            async with classify_item:
                classification = await classify_personal_document(signed_document_url, filename)

                if classification.get("confidence", 0.0) < confidence_threshold:
                    self.steps["classify"] = {"status": "waiting_human", "result": classification}
                    try:
                        await workflows.workflow.wait_condition(
                            lambda: self._manual_category is not None,
                            timeout=manual_review_timeout_seconds,
                            timeout_summary="manual_category_review",
                        )
                    except asyncio.TimeoutError:
                        classification["explanation"] = (
                            f"{classification.get('explanation', '').strip()} "
                            "Manual review timed out; using model-predicted category."
                        ).strip()
                    else:
                        classification["category"] = self._manual_category
                        classification["confidence"] = 1.0
                        classification["explanation"] = f"Manually selected category: {self._manual_category}"

            self.steps["classify"] = {"status": "done", "result": classification}

            self.steps["extract"]["status"] = "running"
            async with extract_item:
                extracted_info = await extract_personal_document_info(
                    signed_document_url,
                    filename,
                    classification["category"],
                )
            self.steps["extract"] = {"status": "done", "result": extracted_info}

        return workflows_mistralai.ChatAssistantWorkflowOutput(
            content=[workflows_mistralai.TextOutput(text=f"Processing complete for {filename}.")],
            structuredContent={
                "filename": filename,
                "ocr_text": "Document processed with Document QnA (no standalone OCR text payload).",
                "classification": classification,
                "personal_document_info": extracted_info,
            },
        )


async def main() -> None:
    print("Worker ready — waiting for personal-document tasks...\n")
    await workflows.run_worker([PersonalDocumentWorkflow])


if __name__ == "__main__":
    asyncio.run(main())
