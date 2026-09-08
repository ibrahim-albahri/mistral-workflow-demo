"""Streamlit rendering shared by the workflow UI and the Agents UI.

Extracted from ``entrypoints/app.py`` so both entry points show identical step
panels. The only difference between them is what happens when a low-confidence
classification needs a human: the workflow sends a signal, the agent run resumes a
conversation — so that action is injected as a callback.
"""

import io
from typing import Any, Callable, Optional

import streamlit as st

from shared.extraction_display import format_extraction_value
from shared.extraction_fields import PERSONAL_DOCUMENT_LABELS

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

COMMON_FIELD_LABELS = {
    "full_name": "Full Name",
    "date_of_birth": "Date of Birth",
    "document_number": "Document Number",
    "issue_date": "Issue Date",
    "expiry_date": "Expiry Date",
    "nationality": "Nationality",
    "address": "Address",
}

STEPS_CONFIG = [
    ("preprocess", "Adaptive Image Preprocessing"),
    ("ocr", "✅ Document Preparation"),
    ("classify", "🏷️ Classification"),
    ("extract", "🧾 Personal Document Extraction"),
]


def get_document_preview(document_bytes: bytes, content_type: str):
    if content_type.startswith("image/"):
        return document_bytes
    if content_type != "application/pdf" or not fitz:
        return None
    try:
        doc = fitz.open(stream=document_bytes, filetype="pdf")
        page = doc[0]
        pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5))
        return io.BytesIO(pix.tobytes("ppm"))
    except Exception:
        return None


def render_manual_category(
    result: Optional[dict], on_submit: Callable[[str], None], key_prefix: str = ""
) -> None:
    confidence = (result or {}).get("confidence", 0.0)
    st.warning(
        f"⚠️ Insufficient confidence ({confidence * 100:.0f}%). "
        "Please choose the category manually."
    )
    selected = st.selectbox(
        "Category",
        options=list(PERSONAL_DOCUMENT_LABELS.keys()),
        format_func=lambda k: PERSONAL_DOCUMENT_LABELS[k],
        key=f"{key_prefix}manual_category_select",
    )
    if st.button("Validate", key=f"{key_prefix}manual_category_submit"):
        on_submit(selected)


def render_preprocess(result: dict) -> None:
    operations = result.get("operations") or []
    if result.get("status") == "skipped":
        st.caption(result.get("reason") or result.get("error") or "Skipped")
    elif operations:
        st.markdown("Applied: " + ", ".join(operations))
        st.caption(result.get("rationale", ""))
    else:
        st.caption(result.get("rationale", "No preprocessing was needed."))


def render_classify(result: dict) -> None:
    category = result.get("category", "gtc")
    confidence = result.get("confidence", 0.0)
    label = PERSONAL_DOCUMENT_LABELS.get(category, f"❓ {category}")
    col1, col2 = st.columns([3, 1])
    col1.markdown(f"**{label}**")
    col1.caption(result.get("explanation", ""))
    col2.metric("Confidence", f"{confidence * 100:.0f}%")
    col2.progress(confidence)


def render_extract(result: dict) -> None:
    common = result.get("common", {})
    specific = result.get("specific", {})

    st.markdown("**🧍 Common Information**")
    common_rows = [
        {"Field": COMMON_FIELD_LABELS.get(k, k), "Value": format_extraction_value(v)}
        for k, v in common.items()
        if v is not None
    ]
    if common_rows:
        st.table(common_rows)
    else:
        st.info("No common information found.")

    if not specific:
        return

    st.markdown("**📋 Specific Information**")
    specific_rows = [
        {
            "Field": k.replace("_", " ").capitalize(),
            "Value": format_extraction_value(v),
        }
        for k, v in specific.items()
        if v is not None and k != "mrz"
    ]
    if specific_rows:
        st.table(specific_rows)
    else:
        st.info("No specific information found.")

    mrz = specific.get("mrz")
    if isinstance(mrz, dict):
        st.markdown("**MRZ validation**")
        st.json(mrz, expanded=False)


def render_step(
    key: str,
    step: dict,
    on_manual_category: Optional[Callable[[str], None]] = None,
    key_prefix: str = "",
) -> None:
    status = step.get("status", "pending")
    result: Any = step.get("result")

    if status == "pending":
        st.markdown("⏳ Pending…")
    elif status == "running":
        st.markdown("⚙️ In Progress…")
    elif status == "waiting_human":
        if on_manual_category is None:
            st.warning("Awaiting manual category selection.")
        else:
            render_manual_category(result, on_manual_category, key_prefix)
    elif status == "done" and result is not None:
        if key == "preprocess":
            render_preprocess(result)
        elif key == "ocr":
            st.markdown("✅ Prepared for Document QnA")
            if isinstance(result, str) and result.strip():
                st.caption(result)
        elif key == "classify":
            render_classify(result)
        elif key == "extract":
            render_extract(result)
