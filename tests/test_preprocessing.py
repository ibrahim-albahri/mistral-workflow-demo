# ruff: noqa: E402

from dataclasses import replace
import io
from pathlib import Path
import sys

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from shared.preprocessing import (
    PREPROCESSING_OPERATIONS,
    apply_preprocessing_operation_bytes,
    conservative_ocr_config,
    enhanced_image_filename,
    inspect_image_bytes,
    preview_image_bytes,
    preprocess_image_bytes,
    upload_payload,
)


def image_bytes(mode: str = "RGBA") -> bytes:
    image = Image.new(mode, (32, 24), (30, 120, 220, 180))
    output = io.BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def test_conservative_profile_disables_content_removal_steps():
    config = conservative_ocr_config()

    assert config.enable_orientation is True
    assert config.enable_deskew is True
    assert config.enable_shadow_removal is True
    assert config.enable_perspective is False
    assert config.enable_sauvola is False
    assert config.enable_stamp_removal is False
    assert config.enable_table_grid_removal is False


def test_preprocess_image_bytes_returns_a_valid_png_for_rgba_input():
    config = replace(conservative_ocr_config(), target_width=64)

    enhanced = preprocess_image_bytes(image_bytes(), config)

    with Image.open(io.BytesIO(enhanced)) as image:
        assert image.format == "PNG"
        assert image.mode == "RGB"
        assert image.width == 64


def test_preprocess_image_bytes_rejects_non_image_input():
    with pytest.raises(ValueError, match="could not be decoded"):
        preprocess_image_bytes(b"not an image")


def test_upload_payload_uses_enhanced_png_only_when_available():
    original = b"original"
    enhanced = b"enhanced"

    assert upload_payload(original, "passport.jpg", "image/jpeg") == (
        original,
        "passport.jpg",
        "image/jpeg",
    )
    assert upload_payload(original, "passport.jpg", "image/jpeg", enhanced) == (
        enhanced,
        "passport-enhanced.png",
        "image/png",
    )
    assert upload_payload(original, "passport.pdf", "application/pdf") == (
        original,
        "passport.pdf",
        "application/pdf",
    )
    assert enhanced_image_filename("archive.scan.JPEG") == "archive.scan-enhanced.png"
    assert enhanced_image_filename("archive.scan-enhanced.png") == "archive.scan-enhanced.png"


def test_agent_preprocessing_operations_produce_valid_pngs():
    source = image_bytes("RGB")
    for operation in set(PREPROCESSING_OPERATIONS) - {"sauvola"}:
        output = apply_preprocessing_operation_bytes(source, operation)
        with Image.open(io.BytesIO(output)) as image:
            assert image.format == "PNG"


def test_image_inspection_and_preview_are_json_safe_and_bounded():
    source = image_bytes("RGB")
    metrics = inspect_image_bytes(source)
    preview = preview_image_bytes(source, max_dimension=16)

    assert metrics["width"] == 32
    assert metrics["height"] == 24
    with Image.open(io.BytesIO(preview)) as image:
        assert max(image.size) <= 16
