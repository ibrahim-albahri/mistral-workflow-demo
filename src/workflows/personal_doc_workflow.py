import asyncio
import json
import logging
import os
from datetime import timedelta
from enum import Enum
from functools import lru_cache
from typing import Any, Optional

import mistralai.workflows as workflows
import mistralai.workflows.plugins.mistralai as workflows_mistralai
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, create_model

from shared.document_media import build_mistral_document_chunk
from shared.extraction_fields import (
    PERSONAL_COMMON_FIELDS,
    PERSONAL_DOCUMENT_CATEGORIES,
    PERSONAL_DOCUMENT_SPECIFIC_FIELDS,
    PERSONAL_FIELD_TYPES,
)
from shared.mrz import parse_mrz
from shared.preprocessing import (
    PREPROCESSING_OPERATIONS,
    apply_preprocessing_operation_bytes,
    enhanced_image_filename,
    inspect_image_bytes,
    preview_image_bytes,
)

load_dotenv(override=True)

for name in ("mistralai_workflows", "httpx", "httpcore"):
    logging.getLogger(name).setLevel(logging.WARNING)


class PersonalDocumentCategory(str, Enum):
    ID = "id"
    PASSPORT = "passport"
    PROOF_OF_ADDRESS = "proof_of_address"
    GTC = "gtc"
    OTHER = "other"


class ManualCategorySignal(BaseModel):
    category: PersonalDocumentCategory


class PersonalDocumentClassification(BaseModel):
    category: PersonalDocumentCategory = Field(
        description=f"One of: {', '.join(PERSONAL_DOCUMENT_CATEGORIES)}"
    )
    confidence: float = Field(ge=0.0, le=1.0)
    explanation: str


class PreprocessingDecision(BaseModel):
    """Final, machine-validated output requested from the preprocessing agent."""

    final_file_id: str
    operations: list[str] = Field(default_factory=list)
    rationale: str = Field(min_length=1)

    def validate_operations(self) -> None:
        unknown = set(self.operations) - set(PREPROCESSING_OPERATIONS)
        if unknown:
            raise ValueError(f"Unknown preprocessing operations: {sorted(unknown)}")
        if len(self.operations) != len(set(self.operations)):
            raise ValueError("A preprocessing operation may only be used once.")


def validate_preprocessing_decision(
    payload: str | dict[str, Any], original_file_id: str
) -> PreprocessingDecision:
    """Parse an agent response and protect the workflow from invalid plans."""
    raw = json.loads(payload) if isinstance(payload, str) else payload
    decision = PreprocessingDecision.model_validate(raw)
    decision.validate_operations()
    if not decision.final_file_id:
        raise ValueError("The preprocessing decision has no final file ID.")
    if not decision.operations and decision.final_file_id != original_file_id:
        raise ValueError("A no-op decision must retain the original file ID.")
    return decision


class MrzExtraction(BaseModel):
    """The model only transcribes MRZ lines; parsing is performed locally."""

    raw_lines: list[str] | None = Field(
        default=None,
        description="MRZ lines transcribed exactly, one line per list item. Do not parse or correct them.",
    )


def _personal_field_definition(key: str, description: str) -> tuple[object, Field]:
    field_type = MrzExtraction if key == "mrz" else PERSONAL_FIELD_TYPES.get(key, str)
    return Optional[field_type], Field(default=None, description=description)


@lru_cache(maxsize=None)
def get_personal_extraction_output_model(category: str) -> type[BaseModel]:
    common_model = create_model(
        "PersonalCommonExtractionFields",
        __config__=ConfigDict(extra="forbid"),
        **{
            key: _personal_field_definition(key, description)
            for key, description in PERSONAL_COMMON_FIELDS
        },
    )
    specific_model = create_model(
        f"PersonalSpecificExtractionFields_{category}",
        __config__=ConfigDict(extra="forbid"),
        **{
            key: _personal_field_definition(key, description)
            for key, description in PERSONAL_DOCUMENT_SPECIFIC_FIELDS.get(category, [])
        },
    )
    return create_model(
        f"PersonalExtractionOutput_{category}",
        __config__=ConfigDict(extra="forbid"),
        common=(common_model, ...),
        specific=(specific_model, ...),
    )


def _fields_text(fields: list[tuple[str, str]]) -> str:
    return "\n".join(f"- {key}: {description}" for key, description in fields)


def enrich_with_mrz_fallback(extracted_info: dict, category: str) -> dict:
    """Use a checksum-valid MRZ only to fill fields missing from visual extraction."""
    common = dict(extracted_info.get("common") or {})
    specific = dict(extracted_info.get("specific") or {})
    output = {**extracted_info, "common": common, "specific": specific}
    raw_mrz = specific.get("mrz")
    if isinstance(raw_mrz, dict):
        raw_mrz = raw_mrz.get("raw_lines")
    parsed_mrz = parse_mrz(raw_mrz)

    if raw_mrz is not None:
        specific["mrz"] = parsed_mrz
    if not parsed_mrz["checksum_valid"]:
        return output

    parsed = parsed_mrz["parsed"]
    targets: list[tuple[dict, str, str]] = [
        (common, "full_name", "full_name"),
        (common, "date_of_birth", "date_of_birth"),
        (common, "document_number", "document_number"),
        (common, "expiry_date", "expiry_date"),
        (common, "nationality", "nationality"),
    ]
    if category == PersonalDocumentCategory.ID.value:
        targets.append((specific, "sex", "sex"))
    elif category == PersonalDocumentCategory.PASSPORT.value:
        targets.extend(
            [
                (specific, "passport_number", "document_number"),
                (specific, "country_of_issue", "country_of_issue"),
            ]
        )

    disagreements: list[str] = parsed_mrz["disagreements"]
    for destination, field_name, mrz_name in targets:
        mrz_value = parsed.get(mrz_name)
        visual_value = destination.get(field_name)
        if mrz_value is None:
            continue
        if visual_value is None:
            destination[field_name] = mrz_value
        elif str(visual_value).strip() != str(mrz_value).strip():
            disagreements.append(f"Visible {field_name} differs from MRZ value.")
    return output


@workflows.activity(
    start_to_close_timeout=timedelta(minutes=5), retry_policy_max_attempts=2
)
async def get_personal_document_signed_url(file_id: str) -> str:
    client = workflows_mistralai.get_mistral_client()
    signed_url = await client.files.get_signed_url_async(file_id=file_id)
    return signed_url.url


async def _download_image_file(file_id: str) -> bytes:
    """Download a Mistral file inside an activity, never in workflow code."""
    import httpx

    client = workflows_mistralai.get_mistral_client()
    signed_url = await client.files.get_signed_url_async(file_id=file_id)
    async with httpx.AsyncClient(timeout=60) as http:
        response = await http.get(signed_url.url)
        response.raise_for_status()
        return response.content


async def _upload_processed_image(image_bytes: bytes, filename: str) -> str:
    from mistralai.client import Mistral

    async with Mistral(
        api_key=os.environ["MISTRAL_API_KEY"],
        server_url=os.environ.get("SERVER_URL", "https://api.mistral.ai"),
    ) as client:
        response = await client.files.upload_async(
            file={
                "file_name": filename,
                "content": image_bytes,
                "content_type": "image/png",
            },
            purpose="ocr",
        )
    return response.id


@workflows.activity(
    start_to_close_timeout=timedelta(minutes=2), retry_policy_max_attempts=2
)
async def inspect_preprocessing_image(file_id: str, filename: str) -> dict[str, Any]:
    """Return metrics and a bounded preview file reference for an image artifact."""
    image_bytes = await _download_image_file(file_id)
    preview_id = await _upload_processed_image(
        preview_image_bytes(image_bytes), f"{enhanced_image_filename(filename)}.preview.png"
    )
    client = workflows_mistralai.get_mistral_client()
    preview_url = await client.files.get_signed_url_async(file_id=preview_id)
    return {
        "file_id": file_id,
        "filename": filename,
        "metrics": inspect_image_bytes(image_bytes),
        "preview_file_id": preview_id,
        "preview_url": preview_url.url,
    }


@workflows.activity(
    start_to_close_timeout=timedelta(minutes=3), retry_policy_max_attempts=2
)
async def apply_preprocessing_tool(
    source_file_id: str, source_filename: str, operation: str
) -> dict[str, Any]:
    """Apply one agent-selected operation and persist a replacement artifact."""
    if operation not in PREPROCESSING_OPERATIONS:
        raise ValueError(f"Unsupported preprocessing operation: {operation}")
    source_bytes = await _download_image_file(source_file_id)
    processed_bytes = apply_preprocessing_operation_bytes(source_bytes, operation)
    filename = enhanced_image_filename(source_filename)
    file_id = await _upload_processed_image(processed_bytes, filename)
    preview_id = await _upload_processed_image(
        preview_image_bytes(processed_bytes), f"{filename}.preview.png"
    )
    client = workflows_mistralai.get_mistral_client()
    preview_url = await client.files.get_signed_url_async(file_id=preview_id)
    return {
        "file_id": file_id,
        "filename": filename,
        "content_type": "image/png",
        "operation": operation,
        "metrics": inspect_image_bytes(processed_bytes),
        "preview_file_id": preview_id,
        "preview_url": preview_url.url,
    }


def _agent_text(outputs: Any) -> str:
    """Extract the final textual response across SDK output model versions."""
    return "\n".join(
        str(getattr(output, "text", ""))
        for output in outputs
        if getattr(output, "text", None)
    ).strip()


async def run_preprocessing_agent(file_id: str, filename: str) -> dict[str, Any]:
    """Run the durable agent and return the selected persistent artifact."""
    inspection = await inspect_preprocessing_image(file_id, filename)
    agent = workflows_mistralai.Agent(
        model=os.environ.get("MISTRAL_PREPROCESSING_AGENT_MODEL", "mistral-medium-latest"),
        name="document-preprocessing-agent",
        description="Selects image preprocessing operations for document OCR.",
        instructions=(
            "You optimize a document image for OCR. Inspect the supplied metrics and "
            "preview URL, then call apply_preprocessing_tool only when justified. "
            "Pass the latest file_id and filename to every tool. Never use an operation "
            "more than once. Inspect each tool result before another operation. Return "
            "ONLY JSON: {final_file_id, operations, rationale}. Use the original file_id "
            "and an empty operations list when no transformation is needed."
        ),
        tools=[apply_preprocessing_tool],
    )
    outputs = await workflows_mistralai.Runner.run(
        agent=agent,
        inputs=[
            {
                "type": "text",
                "text": json.dumps(
                    {key: value for key, value in inspection.items() if key != "preview_url"}
                ),
            },
            {"type": "image_url", "image_url": inspection["preview_url"]},
        ],
        session=workflows_mistralai.RemoteSession(),
        max_turns=14,
    )
    decision = validate_preprocessing_decision(_agent_text(outputs), file_id)
    return {
        "status": "done",
        "file_id": decision.final_file_id,
        "filename": enhanced_image_filename(filename)
        if decision.operations
        else filename,
        "content_type": "image/png" if decision.operations else None,
        "operations": decision.operations,
        "rationale": decision.rationale,
        "metrics": inspection["metrics"],
    }


@workflows.activity(
    start_to_close_timeout=timedelta(minutes=2), retry_policy_max_attempts=2
)
async def classify_personal_document(
    document_content: dict[str, str], filename: str
) -> dict:
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
                    "Classify from the document contents, not from its filename. "
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
                    document_content,
                ],
            },
        ],
    )
    parsed = (
        response.choices[0].message.parsed
        if response.choices and response.choices[0].message
        else None
    )
    if parsed is None:
        raise RuntimeError(
            "Personal document classification response could not be parsed."
        )
    return parsed.model_dump(mode="json")


@workflows.activity(
    start_to_close_timeout=timedelta(minutes=2), retry_policy_max_attempts=2
)
async def extract_personal_document_info(
    document_content: dict[str, str],
    filename: str,
    category: str,
) -> dict:
    client = workflows_mistralai.get_mistral_client()
    model = os.environ.get("MISTRAL_EXTRACTOR_MODEL", "mistral-medium-latest")
    extraction_model = get_personal_extraction_output_model(category)
    common_fields_text = _fields_text(PERSONAL_COMMON_FIELDS)
    specific_fields_text = _fields_text(
        PERSONAL_DOCUMENT_SPECIFIC_FIELDS.get(category, [])
    )
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
                    "Treat visibly printed fields as the source of truth: transcribe them exactly, "
                    "never infer or correct a value, and set unsupported fields to null. "
                    "Use the document contents rather than the filename. Dates must keep the format "
                    "shown on the document. A document number is the identity/travel-document identifier, "
                    "not an account, customer, or registration number. For an MRZ, transcribe raw lines "
                    "only; do not derive values from it."
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
                            "For proof of address, extract the account holder's address, not the provider's. "
                            "For GTC, return each key clause as a separate list item. Return null for missing values."
                        ),
                    },
                    document_content,
                ],
            },
        ],
    )
    parsed = (
        response.choices[0].message.parsed
        if response.choices and response.choices[0].message
        else None
    )
    if parsed is None:
        raise RuntimeError("Personal document extraction response could not be parsed.")
    extracted_info = enrich_with_mrz_fallback(parsed.model_dump(mode="json"), category)
    print(
        "Personal document extraction result "
        f"for {filename}: {json.dumps(extracted_info, ensure_ascii=False, sort_keys=True)}"
    )
    return extracted_info


@workflows.workflow.define(name="personal_document_workflow")
class PersonalDocumentWorkflow(workflows.InteractiveWorkflow):
    def __init__(self):
        self.steps = {
            "preprocess": {"status": "pending", "result": None},
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
        self._manual_category = payload.category.value

    @workflows.workflow.entrypoint
    async def run(
        self,
        file_id: str,
        filename: str,
        confidence_threshold: float = 0.9,
        manual_review_timeout_seconds: Optional[float] = None,
        content_type: str = "application/pdf",
    ) -> workflows_mistralai.ChatAssistantWorkflowOutput:
        preprocess_item = workflows_mistralai.TodoListItem(
            title="Adaptively preprocess image",
            description="Use an agent to select only the image improvements needed for OCR.",
        )
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

        async with workflows_mistralai.TodoList(
            items=[preprocess_item, ocr_item, classify_item, extract_item]
        ):
            processed_file_id = file_id
            processed_filename = filename
            processed_content_type = content_type
            preprocessing_result: dict[str, Any]
            if content_type.startswith("image/"):
                self.steps["preprocess"]["status"] = "running"
                async with preprocess_item:
                    try:
                        preprocessing_result = await run_preprocessing_agent(file_id, filename)
                    except Exception as exc:
                        preprocessing_result = {
                            "status": "skipped",
                            "file_id": file_id,
                            "filename": filename,
                            "content_type": content_type,
                            "operations": [],
                            "error": str(exc),
                        }
                    else:
                        processed_file_id = preprocessing_result["file_id"]
                        processed_filename = preprocessing_result["filename"]
                        processed_content_type = preprocessing_result["content_type"] or content_type
                self.steps["preprocess"] = {
                    "status": "done",
                    "result": preprocessing_result,
                }
            else:
                preprocessing_result = {
                    "status": "skipped",
                    "file_id": file_id,
                    "filename": filename,
                    "content_type": content_type,
                    "operations": [],
                    "reason": "Preprocessing is currently supported for images only.",
                }
                self.steps["preprocess"] = {"status": "done", "result": preprocessing_result}

            self.steps["ocr"]["status"] = "running"
            async with ocr_item:
                signed_document_url = await get_personal_document_signed_url(processed_file_id)
                document_content = build_mistral_document_chunk(
                    signed_document_url, processed_filename, processed_content_type
                )
            self.steps["ocr"] = {
                "status": "done",
                "result": "Document prepared for Document QnA (OCR handled by Mistral Document AI).",
            }

            self.steps["classify"]["status"] = "running"
            async with classify_item:
                classification = await classify_personal_document(
                    document_content, filename
                )

                if classification.get("confidence", 0.0) < confidence_threshold:
                    self.steps["classify"] = {
                        "status": "waiting_human",
                        "result": classification,
                    }
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
                        classification["explanation"] = (
                            f"Manually selected category: {self._manual_category}"
                        )

            self.steps["classify"] = {"status": "done", "result": classification}

            self.steps["extract"]["status"] = "running"
            async with extract_item:
                extracted_info = await extract_personal_document_info(
                    document_content,
                    filename,
                    classification["category"],
                )
            self.steps["extract"] = {"status": "done", "result": extracted_info}

        return workflows_mistralai.ChatAssistantWorkflowOutput(
            content=[
                workflows_mistralai.TextOutput(
                    text=f"Processing complete for {filename}."
                )
            ],
            structuredContent={
                "filename": filename,
                "preprocessing": preprocessing_result,
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
