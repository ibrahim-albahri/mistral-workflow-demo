# ruff: noqa: E402

from pathlib import Path
import asyncio
import json
import inspect
import sys
from types import SimpleNamespace

import pytest
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from shared.document_media import (
    build_mistral_document_chunk,
    get_document_content_type,
)
from shared.extraction_display import format_extraction_value
from shared.extraction_fields import (
    PERSONAL_DOCUMENT_CATEGORIES,
    PERSONAL_DOCUMENT_LABELS,
)
from shared.mrz import parse_mrz
from shared.workflow_results import workflow_result_mapping, workflow_status_name
from workflows import personal_doc_workflow
from workflows.personal_doc_workflow import (
    PersonalDocumentClassification,
    PersonalDocumentWorkflow,
    enrich_with_mrz_fallback,
    extract_personal_document_info,
    get_personal_extraction_output_model,
)
from workflows.batch_personal_doc_workflow import (
    MAX_BATCH_DOCUMENTS,
    BatchPersonalDocumentWorkflow,
    batch_jsonl,
    build_classification_batch_entries,
    build_extraction_batch_entries,
    build_ocr_batch_entries,
    chat_json_from_body,
    ocr_markdown_from_body,
    parse_batch_result_rows,
)


def test_personal_document_categories_are_defined():
    assert PERSONAL_DOCUMENT_CATEGORIES == [
        "id",
        "passport",
        "proof_of_address",
        "gtc",
        "other",
    ]
    assert PERSONAL_DOCUMENT_LABELS["id"] == "🆔 ID"
    assert PERSONAL_DOCUMENT_LABELS["passport"] == "🛂 Passport"
    assert PERSONAL_DOCUMENT_LABELS["proof_of_address"] == "🏠 Proof of Address"
    assert PERSONAL_DOCUMENT_LABELS["gtc"] == "📜 GTC"
    assert PERSONAL_DOCUMENT_LABELS["other"] == "❓ Other"


def test_personal_document_workflow_is_registered():
    assert PersonalDocumentWorkflow.__name__ == "PersonalDocumentWorkflow"


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        ("document.PDF", "application/pdf"),
        ("scan.jpg", "image/jpeg"),
        ("scan.JPEG", "image/jpeg"),
        ("scan.PnG", "image/png"),
        ("scan.WEBP", "image/webp"),
    ],
)
def test_document_content_type_is_detected_case_insensitively(filename, expected):
    assert get_document_content_type(filename) == expected


def test_unsupported_document_format_is_rejected():
    with pytest.raises(ValueError, match="Unsupported document format"):
        get_document_content_type("passport.heic")


def test_pdf_uses_document_url_chunk():
    assert build_mistral_document_chunk(
        "https://example.test/document",
        "passport.pdf",
        "application/pdf",
    ) == {
        "type": "document_url",
        "document_url": "https://example.test/document",
        "document_name": "passport.pdf",
    }


@pytest.mark.parametrize("content_type", ["image/jpeg", "image/png", "image/webp"])
def test_image_uses_image_url_chunk(content_type):
    assert build_mistral_document_chunk(
        "https://example.test/image",
        "identity-image",
        content_type,
    ) == {
        "type": "image_url",
        "image_url": "https://example.test/image",
    }


def test_personal_document_workflow_defaults_to_pdf_content_type():
    parameter = inspect.signature(PersonalDocumentWorkflow.run).parameters[
        "content_type"
    ]
    assert parameter.default == "application/pdf"


def test_personal_extraction_schema_preserves_descriptions_and_types():
    model = get_personal_extraction_output_model("passport")
    common_model = model.model_fields["common"].annotation
    specific_model = model.model_fields["specific"].annotation

    assert (
        common_model.model_fields["document_number"].description
        == "Document identifier or registration number"
    )
    assert specific_model.model_fields["holder_signature"].annotation == bool | None
    assert (
        specific_model.model_fields["mrz"].description
        == "Machine-readable zone lines, transcribed exactly when visible"
    )


def test_personal_document_categories_are_constrained():
    with pytest.raises(ValueError):
        PersonalDocumentClassification(
            category="driver_license", confidence=1, explanation="test"
        )


def test_parses_valid_td3_passport_mrz():
    result = parse_mrz(
        [
            "P<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<",
            "L898902C36UTO7408122F1204159ZE184226B<<<<<10",
        ]
    )

    assert result["format"] == "TD3"
    assert result["checksum_valid"] is True
    assert result["parsed"]["document_number"] == "L898902C3"
    assert result["parsed"]["full_name"] == "ERIKSSON ANNA MARIA"


def test_parses_valid_td1_identity_mrz_and_normalises_compact_input():
    result = parse_mrz(
        "I<UTOD231458907<<<<<<<<<<<<<<<7408122F1204159UTO<<<<<<<<<<<6ERIKSSON<<ANNA<MARIA<<<<<<<<<<"
    )

    assert result["format"] == "TD1"
    assert result["checksum_valid"] is True
    assert result["parsed"]["date_of_birth"] == "740812"


def test_mrz_checksum_failure_is_reported_without_raising():
    result = parse_mrz(
        [
            "P<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<",
            "L898902C35UTO7408122F1204159ZE184226B<<<<<10",
        ]
    )

    assert result["checksum_valid"] is False
    assert any("Document number" in error for error in result["validation_errors"])


def test_mrz_fills_only_missing_visual_fields_and_records_disagreements():
    extracted = {
        "common": {
            "full_name": "Visible Name",
            "date_of_birth": None,
            "document_number": None,
            "expiry_date": None,
            "nationality": None,
        },
        "specific": {
            "passport_number": "VISIBLE123",
            "country_of_issue": None,
            "mrz": {
                "raw_lines": [
                    "P<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<",
                    "L898902C36UTO7408122F1204159ZE184226B<<<<<10",
                ]
            },
        },
    }

    result = enrich_with_mrz_fallback(extracted, "passport")

    assert result["common"]["full_name"] == "Visible Name"
    assert result["common"]["date_of_birth"] == "740812"
    assert result["specific"]["passport_number"] == "VISIBLE123"
    assert result["specific"]["country_of_issue"] == "UTO"
    assert result["specific"]["mrz"]["disagreements"]


def test_extraction_prompt_includes_visual_first_rules(monkeypatch, capsys):
    captured = {}

    class FakeClient:
        def __init__(self):
            self.chat = self

        async def parse_async(self, **kwargs):
            captured.update(kwargs)
            parsed = kwargs["response_format"](common={}, specific={})
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(parsed=parsed))]
            )

    monkeypatch.setattr(
        personal_doc_workflow.workflows_mistralai,
        "get_mistral_client",
        lambda: FakeClient(),
    )

    result = asyncio.run(
        extract_personal_document_info(
            {"type": "document_url", "document_url": "https://example.test"},
            "scan.pdf",
            "passport",
        )
    )

    system_text = captured["messages"][0]["content"]
    user_text = captured["messages"][1]["content"][0]["text"]
    assert result["common"]["full_name"] is None
    assert "visibly printed fields as the source of truth" in system_text
    assert "holder_signature: Whether the holder signature is present" in user_text
    assert "Personal document extraction result for scan.pdf" in capsys.readouterr().out


def test_typed_values_have_table_safe_display_values():
    assert format_extraction_value(True) == "Yes"
    assert format_extraction_value(["A", "B"]) == "A, B"
    assert '"format": "TD3"' in format_extraction_value({"format": "TD3"})


def test_workflow_results_support_json_and_sdk_model_shapes():
    class FakeResult(BaseModel):
        structured_content: dict

    payload = {"structuredContent": {"personal_document_info": {"common": {}}}}
    assert workflow_result_mapping(payload) == payload
    assert workflow_result_mapping('{"structuredContent": {}}') == {
        "structuredContent": {}
    }
    assert workflow_result_mapping(FakeResult(structured_content={"ok": True})) == {
        "structured_content": {"ok": True}
    }
    assert workflow_result_mapping("not json") == {}


def test_workflow_status_name_supports_enum_like_values():
    class Status:
        value = "COMPLETED"

    assert workflow_status_name(Status()) == "COMPLETED"
    assert workflow_status_name("WorkflowStatus.FAILED") == "FAILED"


def test_batch_workflow_is_registered_and_has_expected_limit():
    assert BatchPersonalDocumentWorkflow.__name__ == "BatchPersonalDocumentWorkflow"
    assert MAX_BATCH_DOCUMENTS == 100


def test_batch_jsonl_and_result_rows_preserve_custom_ids():
    payload = batch_jsonl([{"custom_id": "doc-1", "body": {"document": {}}}])
    assert json.loads(payload.decode("utf-8"))["custom_id"] == "doc-1"
    rows = parse_batch_result_rows(
        '{"custom_id":"doc-1","response":{"body":{"pages":[{"markdown":"Hello"}]}}}\n'
        '{"custom_id":"doc-2","error":{"message":"bad input"}}'
    )
    assert rows["doc-1"]["ok"] is True
    assert rows["doc-2"] == {"ok": False, "error": "{'message': 'bad input'}"}


def test_batch_requests_include_ocr_and_source_document():
    document = {
        "document_id": "doc-1",
        "filename": "passport.pdf",
        "content_type": "application/pdf",
        "signed_url": "https://example.test/passport",
        "ocr_markdown": "# Passport\nANNA ERIKSSON",
        "classification": {"category": "passport"},
    }
    assert (
        build_ocr_batch_entries([document])[0]["body"]["document"]["type"]
        == "document_url"
    )
    assert (
        "OCR markdown"
        in build_classification_batch_entries([document])[0]["body"]["messages"][1][
            "content"
        ]
    )
    extraction_content = build_extraction_batch_entries([document])[0]["body"][
        "messages"
    ][1]["content"]
    assert "OCR markdown" in extraction_content[0]["text"]
    assert extraction_content[1]["document_url"] == "https://example.test/passport"


def test_batch_ocr_and_chat_response_parsers():
    assert (
        ocr_markdown_from_body(
            {"pages": [{"markdown": "first"}, {"markdown": "second"}]}
        )
        == "first\n\nsecond"
    )
    assert chat_json_from_body(
        {"choices": [{"message": {"content": '{"category":"passport"}'}}]}
    ) == {"category": "passport"}
