"""Schemas, prompts, and enrichment shared by the workflow and agent implementations.

Nothing here imports ``mistralai.workflows``: both ``workflows.personal_doc_workflow``
and the Agents API implementation in ``agents/`` build on this module so the two stay
behaviourally identical.
"""

import json
import re
from collections.abc import Iterable
from enum import Enum
from functools import lru_cache
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, create_model

from shared.extraction_fields import (
    PERSONAL_COMMON_FIELDS,
    PERSONAL_DOCUMENT_CATEGORIES,
    PERSONAL_DOCUMENT_SPECIFIC_FIELDS,
    PERSONAL_FIELD_TYPES,
)
from shared.mrz import parse_mrz
from shared.preprocessing import PREPROCESSING_OPERATIONS


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
    """Parse an agent response and protect the caller from invalid plans."""
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


# ── Prompts ───────────────────────────────────────────────────────────────────
# Shared verbatim so the workflow and the agents produce comparable results.

CLASSIFIER_SYSTEM_PROMPT = (
    "You are an expert in classifying personal identity and compliance documents. "
    "Classify from the document contents, not from its filename. "
    "Return only valid JSON that matches the schema."
)

#: The part that applies to every category.
EXTRACTOR_BASE_PROMPT = (
    "You extract identity and compliance information from personal documents. "
    "Return only valid JSON that matches the schema. Fill in every field the "
    "document shows — leave a field null only when the document genuinely does not "
    "contain it. "
    "Treat visibly printed fields as the source of truth: transcribe them exactly, "
    "never infer or correct a value, and set unsupported fields to null. "
    "Use the document contents rather than the filename. Dates must keep the format "
    "shown on the document. Never repeat a sequence to pad a value, and never put "
    "one field's value inside another field."
)

#: Only meaningful for the two categories whose schema has an ``mrz`` field. Sending
#: it to, say, the GTC extractor buried the actual task and it returned nulls.
EXTRACTOR_MRZ_PROMPT = (
    " A document number is the identity/travel-document identifier, not an account, "
    "customer, or registration number, and it is short — the identifier printed on "
    "the document, never more than about 20 characters. For an MRZ, transcribe raw "
    "lines only; do not derive values from it. MRZ text belongs solely in the mrz "
    "field's raw_lines, exactly two lines for a passport and three for an ID card, "
    "each transcribed once. Never copy MRZ characters into any other field."
)


def extractor_system_prompt(category: str) -> str:
    """Instructions for one category's extractor, without irrelevant guidance."""
    fields = PERSONAL_DOCUMENT_SPECIFIC_FIELDS.get(category, [])
    if any(key == "mrz" for key, _ in fields):
        return EXTRACTOR_BASE_PROMPT + EXTRACTOR_MRZ_PROMPT
    return EXTRACTOR_BASE_PROMPT


#: The workflow uses one model for every category, so it needs the whole thing.
EXTRACTOR_SYSTEM_PROMPT = EXTRACTOR_BASE_PROMPT + EXTRACTOR_MRZ_PROMPT

#: Thresholds the rule-based pipeline in ``shared.preprocessing`` uses. Handing the
#: same numbers to the agent keeps its choices comparable to ``_run_pipeline``'s and
#: stops it from "improving" an image that is already fine.
PREPROCESSING_GUIDANCE = (
    "Apply an operation only when the metrics justify it:\n"
    "- sharpen: only if blur_variance < 80 (higher means sharper; do not sharpen a sharp image)\n"
    "- denoise: only if noise_estimate > 8 (it is slow and softens text)\n"
    "- exposure: only if brightness < 85 (too dark) or > 195 (blown out)\n"
    "- shadow_removal: only if the preview shows uneven lighting or a gradient across the page\n"
    "- orientation: only if the preview is rotated by about 90, 180 or 270 degrees\n"
    "- deskew: only if the preview is slightly tilted\n"
    "- margin_crop: only if there are wide empty borders\n"
    "- dpi_normalization: only if width is far below 2480\n"
    "Never destroy content. Machine-readable zones, stamps and signatures must "
    "survive every operation you choose.\n"
)

PREPROCESSING_SYSTEM_PROMPT = (
    "You optimize a document image for OCR. Call inspect_document_image first and "
    "study the metrics and the preview, then call apply_preprocessing_operation only "
    "when justified. Emit exactly ONE tool call per turn and wait for its result "
    "before deciding the next step; never describe a tool call in your message text. "
    "Pass the latest file_id and filename to every tool. Never use an operation more "
    "than once. Prefer the smallest set of operations that helps — doing nothing is "
    "the right answer for a clean scan.\n\n"
    f"{PREPROCESSING_GUIDANCE}\n"
    "When no further operation is justified, reply with the JSON object "
    "{final_file_id, operations, rationale}, where final_file_id is the newest "
    "file_id you produced and operations lists what you applied, in order. Use the "
    "original file_id and an empty operations list when nothing was needed."
)


def classification_prompt(filename: str) -> str:
    return (
        f"Classify the personal document '{filename}' into exactly one category from:\n"
        + "\n".join(f"- {c}" for c in PERSONAL_DOCUMENT_CATEGORIES)
        + "\n\n"
        "Return confidence between 0 and 1 and a short explanation."
    )


def extraction_prompt(filename: str, category: str) -> str:
    common_fields_text = _fields_text(PERSONAL_COMMON_FIELDS)
    specific_fields = PERSONAL_DOCUMENT_SPECIFIC_FIELDS.get(category, [])
    specific_fields_text = _fields_text(specific_fields)
    prompt = (
        f"Extract fields from '{filename}' for category '{category}'.\n\n"
        "Populate these common fields:\n"
        f"{common_fields_text}\n\n"
        "Populate these category-specific fields:\n"
        f"{specific_fields_text if specific_fields_text else '- (none)'}\n\n"
        "For proof of address, extract the account holder's address, not the provider's. "
        "For GTC, return each key clause as a separate list item. Return null for missing values."
    )
    if any(key == "mrz" for key, _ in specific_fields):
        # A checksum-valid MRZ is what `enrich_with_mrz_fallback` uses to fill the
        # fields the visual pass misses, so omitting it costs real data. The rest of
        # the guidance is about keeping MRZ text *out* of other fields, which on its
        # own reads as a reason to skip the zone altogether — hence this.
        prompt += (
            "\n\nThe machine-readable zone is required. It is the block of monospaced "
            "characters padded with '<' at the foot of the document. Transcribe each "
            "of its lines verbatim into specific.mrz.raw_lines, one string per line — "
            "characters exactly as printed, including every '<', nothing added or "
            "removed, each line written out once. Set specific.mrz.raw_lines to null "
            "only if the document genuinely has no machine-readable zone. Every other "
            "field still comes from the printed text, never from these lines."
        )
    return prompt


# ── MRZ enrichment ────────────────────────────────────────────────────────────


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


# ── Extraction sanity checks ──────────────────────────────────────────────────

#: A run of MRZ filler characters. Seeing this outside the MRZ field means the model
#: has copied machine-readable-zone text where it does not belong — usually while
#: looping on the filler, which also truncates or bloats the reply.
MRZ_FILLER = "<<"

#: No legitimate value on these documents is this long; MRZ contamination always is.
MAX_REASONABLE_FIELD_LENGTH = 120


def looks_like_mrz_text(value: str) -> bool:
    """Whether a field value is machine-readable-zone text rather than an answer.

    The length rule alone is not safe: a GTC ``acceptance_text`` is legitimately a
    long sentence, and flagging it discarded a perfectly good value. Prose contains
    spaces; MRZ text and repetition loops do not, so length only counts against a
    value that has none.
    """
    if MRZ_FILLER in value:
        return True
    return len(value) > MAX_REASONABLE_FIELD_LENGTH and " " not in value


def looks_like_embedded_json(value: str, field_names: Iterable[str]) -> bool:
    """Whether sibling fields have been swallowed into this string value.

    Seen live: an extractor answered a GTC document with document_title, issuer_name
    and key_clauses spliced *inside* acceptance_text. It parses as valid JSON and
    reads as prose, so nothing else catches it, yet three fields come back null with
    their real values trapped in a fourth.
    """
    return any(
        re.search(rf'"{re.escape(name)}"\s*:', value) for name in field_names
    )


def extraction_field_names(category: str) -> list[str]:
    """Every field name the extraction schema defines for a category."""
    return [key for key, _ in PERSONAL_COMMON_FIELDS] + [
        key for key, _ in PERSONAL_DOCUMENT_SPECIFIC_FIELDS.get(category, [])
    ]


def find_unusable_fields(extracted_info: dict, category: str) -> list[str]:
    """Return the field paths whose value is not a usable answer.

    Two shapes, both observed against the live API: machine-readable-zone text
    copied into an ordinary field, and sibling fields swallowed into a string.
    """
    field_names = extraction_field_names(category)
    unusable: list[str] = []
    for section in ("common", "specific"):
        values = extracted_info.get(section) or {}
        if not isinstance(values, dict):
            continue
        for key, value in values.items():
            if key == "mrz" or not isinstance(value, str):
                continue
            if looks_like_mrz_text(value) or looks_like_embedded_json(
                value, field_names
            ):
                unusable.append(f"{section}.{key}")
    return unusable


def find_mrz_contamination(extracted_info: dict) -> list[str]:
    """Field paths where MRZ text has leaked into a non-MRZ field."""
    contaminated: list[str] = []
    for section in ("common", "specific"):
        values = extracted_info.get(section) or {}
        if not isinstance(values, dict):
            continue
        for key, value in values.items():
            if key == "mrz" or not isinstance(value, str):
                continue
            if looks_like_mrz_text(value):
                contaminated.append(f"{section}.{key}")
    return contaminated


def strip_unusable_fields(extracted_info: dict, fields: list[str]) -> dict:
    """Null the named fields rather than reporting unusable text as a value."""
    output = {
        **extracted_info,
        "common": dict(extracted_info.get("common") or {}),
        "specific": dict(extracted_info.get("specific") or {}),
    }
    for path in fields:
        section, _, key = path.partition(".")
        if section in output and key in output[section]:
            output[section][key] = None
    return output
