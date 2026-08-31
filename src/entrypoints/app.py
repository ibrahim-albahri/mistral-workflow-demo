"""
Streamlit UI for PDF OCR + Classification + Extraction Workflow.
"""
import asyncio
import io
import os
import time
import uuid

import streamlit as st
from dotenv import load_dotenv
from pydantic import BaseModel

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

load_dotenv(override=True)

from mistralai.client import Mistral
from mistralai.workflows.client import get_mistral_client
from shared.extraction_fields import CATEGORY_LABELS

API_KEY = os.environ["MISTRAL_API_KEY"]
BASE_URL = os.environ.get("SERVER_URL", "https://api.mistral.ai")

COMMON_FIELD_LABELS = {
    "full_name": "Full Name",
    "patient_address": "Address",
    "social_security_number": "Social Security Number",
}

STEPS_CONFIG = [
    ("ocr",      "✅ Document Preparation"),
    ("classify", "🏷️ Classification"),
    ("extract",  "👤 Patient Extraction"),
]

class PdfOcrInput(BaseModel):
    file_id: str
    filename: str
    confidence_threshold: float = 0.9

# ── Workflow Activities ─────────────────────────────────────────────────────

def get_workflows_client():
    # Do not cache across run_async() calls: each call creates/closes its own
    # event loop, and reusing an async client across loops triggers
    # "Event loop is closed" errors.
    return get_mistral_client(
        server_url=BASE_URL,
        api_key=API_KEY,
    )


def run_async(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        # Drain pending async generators before closing the loop to avoid:
        # "Task was destroyed but it is pending! ... async_generator_athrow"
        loop.run_until_complete(loop.shutdown_asyncgens())
        loop.close()


async def upload_pdf(pdf_bytes: bytes, filename: str) -> str:
    async with Mistral(api_key=API_KEY) as client:
        resp = await client.files.upload_async(
            file={"file_name": filename, "content": pdf_bytes, "content_type": "application/pdf"},
            purpose="ocr",
        )
    return resp.id


async def trigger_workflow(file_id: str, filename: str, confidence_threshold: float) -> str:
    execution_id = f"pdf-ocr-{uuid.uuid4().hex[:12]}"
    async with get_workflows_client() as client:
        resp = await client.workflows.execute_workflow_async(
            workflow_identifier="pdf_ocr_workflow",
            input=PdfOcrInput(file_id=file_id, filename=filename, confidence_threshold=confidence_threshold).model_dump(mode="json"),
            execution_id=execution_id,
        )
    return resp.execution_id


async def poll_steps(execution_id: str) -> dict:
    async with get_workflows_client() as client:
        resp = await client.workflows.executions.query_workflow_execution_async(
            execution_id=execution_id,
            name="get_steps",
        )
    return resp.result or {}


async def get_execution_details(execution_id: str):
    async with get_workflows_client() as client:
        return await client.workflows.executions.get_workflow_execution_async(execution_id=execution_id)


class ManualCategorySignal(BaseModel):
    category: str


async def send_signal(execution_id: str, category: str):
    async with get_workflows_client() as client:
        await client.workflows.executions.signal_workflow_execution_async(
            execution_id=execution_id,
            name="manual_category",
            input=ManualCategorySignal(category=category).model_dump(mode="json"),
        )


def backfill_steps_from_execution_result(steps: dict, result: object) -> dict:
    if not isinstance(result, dict):
        return steps

    structured = result.get("structuredContent")
    if not isinstance(structured, dict):
        structured = result.get("structured_content")
    if not isinstance(structured, dict):
        structured = result

    ocr_text = structured.get("ocr_text")
    classification = structured.get("classification")
    patient_info = structured.get("patient_info")

    updated = dict(steps)
    if ocr_text is not None:
        updated["ocr"] = {"status": "done", "result": ocr_text}
    if isinstance(classification, dict):
        updated["classify"] = {"status": "done", "result": classification}
    if isinstance(patient_info, dict):
        updated["extract"] = {"status": "done", "result": patient_info}
    return updated


# ── PDF rendering ─────────────────────────────────────────────────────────────

def get_pdf_first_page(pdf_bytes: bytes):
    """Convert first page of PDF to image."""
    if not fitz:
        return None
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        page = doc[0]
        pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5))  # 1.5x zoom for clarity
        img_bytes = pix.tobytes("ppm")
        return io.BytesIO(img_bytes)
    except Exception:
        return None


# ── Step renderers ────────────────────────────────────────────────────────────

def render_step(key: str, step: dict):
    status = step.get("status", "pending")
    result = step.get("result")

    if status == "pending":
        st.markdown("⏳ Pending…")
    elif status == "running":
        st.spinner("⚙️ In Progress…")
        # spinner widget needs a context manager — use a visual substitute
        st.markdown("⚙️ In Progress…")
    elif status == "waiting_human":
        result = step.get("result", {})
        confidence = result.get("confidence", 0.0) if result else 0.0
        st.warning(
            f"⚠️ Insufficient confidence ({confidence * 100:.0f}%). "
            "Please choose the category manually."
        )
        selected = st.selectbox(
            "Category",
            options=list(CATEGORY_LABELS.keys()),
            format_func=lambda k: CATEGORY_LABELS[k],
            key="manual_category_select",
        )
        if st.button("Validate", key="manual_category_submit"):
            run_async(send_signal(st.session_state.execution_id, selected))
            st.session_state.signal_sent = True
            st.rerun()
    elif status == "done" and result is not None:
        if key == "ocr":
            st.markdown("✅ Prepared for Document QnA")
            if isinstance(result, str) and result.strip():
                st.caption(result)
        elif key == "classify":
            category  = result.get("category", "other")
            confidence = result.get("confidence", 0.0)
            explanation = result.get("explanation", "")
            label = CATEGORY_LABELS.get(category, f"❓ {category}")
            col1, col2 = st.columns([3, 1])
            col1.markdown(f"**{label}**")
            col1.caption(explanation)
            col2.metric("Confidence", f"{confidence * 100:.0f}%")
            col2.progress(confidence)
        elif key == "extract":
            common = result.get("common", {})
            specific = result.get("specific", {})

            st.markdown("**🧍 Patient Information**")
            common_rows = [
                {"Field": COMMON_FIELD_LABELS.get(k, k), "Value": ", ".join(v) if isinstance(v, list) else v}
                for k, v in common.items() if v is not None
            ]
            if common_rows:
                st.table(common_rows)
            else:
                st.info("No common information found.")

            if specific:
                st.markdown("**📋 Specific Information**")
                specific_rows = [
                    {"Field": k.replace("_", " ").capitalize(), "Value": ", ".join(v) if isinstance(v, list) else v}
                    for k, v in specific.items() if v is not None
                ]
                if specific_rows:
                    st.table(specific_rows)
                else:
                    st.info("No specific information found.")


# ── UI ────────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="PDF OCR & Classification", page_icon="📄", layout="wide")
st.title("📄 PDF OCR & Classification")
st.caption("Upload a PDF → OCR → Classification → Patient Extraction")

with st.sidebar:
    st.header("⚙️ Parameters")
    confidence_threshold = st.slider(
        "Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.9,
        step=0.05,
        help="Below this threshold, classification requires manual validation.",
    )
    st.caption(f"Current threshold: **{confidence_threshold * 100:.0f}%**")
    if confidence_threshold >= 1.0:
        st.info("☝️ Manual validation always required")
    elif confidence_threshold == 0.0:
        st.info("✅ Manual validation never required")

# Init session state
if "execution_id" not in st.session_state:
    st.session_state.execution_id = None
if "done" not in st.session_state:
    st.session_state.done = False
if "steps" not in st.session_state:
    st.session_state.steps = {}
if "poll_error" not in st.session_state:
    st.session_state.poll_error = None
if "signal_sent" not in st.session_state:
    st.session_state.signal_sent = False


uploaded = st.file_uploader("Choose a PDF file", type=["pdf"])

if uploaded is not None:
    st.info(f"**{uploaded.name}** — {uploaded.size / 1024:.1f} KB")

    if st.button("Start Workflow", type="primary"):
        st.session_state.execution_id = None
        st.session_state.done = False
        st.session_state.steps = {}
        st.session_state.poll_error = None
        st.session_state.signal_sent = False

        pdf_bytes = uploaded.read()
        filename = uploaded.name

        with st.status("Uploading PDF…", expanded=False) as s:
            file_id = run_async(upload_pdf(pdf_bytes, filename))
            s.update(label="Upload ✓", state="complete")

        execution_id = run_async(trigger_workflow(file_id, filename, confidence_threshold))
        st.session_state.execution_id = execution_id
        st.rerun()

if st.session_state.execution_id and not st.session_state.done:
    execution_id = st.session_state.execution_id

    try:
        steps = run_async(poll_steps(execution_id))
        st.session_state.steps = steps
        st.session_state.poll_error = None
    except Exception as exc:
        steps = st.session_state.steps
        st.session_state.poll_error = str(exc)

    col_pdf, col_steps = st.columns([1, 1.2])

    with col_pdf:
        st.markdown("### 📄 Document")
        if uploaded:
            pdf_bytes = uploaded.read()
            uploaded.seek(0)
            img = get_pdf_first_page(pdf_bytes)
            if img:
                st.image(img, width='stretch')
            else:
                st.info("PyMuPDF not available for preview")

    with col_steps:
        if st.session_state.poll_error:
            st.warning(f"Progress polling failed: {st.session_state.poll_error}")
        for key, title in STEPS_CONFIG:
            st.markdown(f"### {title}")
            step = steps.get(key, {"status": "pending", "result": None})
            render_step(key, step)

    # Check if all done
    all_done = all(
        steps.get(k, {}).get("status") == "done"
        for k, _ in STEPS_CONFIG
    )

    waiting_human = any(
        steps.get(k, {}).get("status") == "waiting_human"
        for k, _ in STEPS_CONFIG
    )

    if all_done:
        st.session_state.done = True
        st.success("✅ Completed!")
    elif waiting_human and not st.session_state.signal_sent:
        pass
    elif waiting_human and st.session_state.signal_sent:
        time.sleep(0.5)
        st.rerun()
    else:
        try:
            execution = run_async(get_execution_details(execution_id))
            wf_status = execution.status
            if wf_status == "COMPLETED":
                st.session_state.steps = backfill_steps_from_execution_result(
                    st.session_state.steps,
                    execution.result,
                )
                st.session_state.done = True
                st.success("✅ Completed!")
            elif wf_status in ("FAILED", "CANCELED", "TERMINATED"):
                st.error(f"Workflow ended with status: {wf_status}")
                st.session_state.done = True
            else:
                time.sleep(0.5)
                st.rerun()
        except Exception as exc:
            st.session_state.poll_error = str(exc)
            time.sleep(0.5)
            st.rerun()

elif st.session_state.execution_id and st.session_state.done:
    steps = st.session_state.steps
    for key, title in STEPS_CONFIG:
        st.markdown(f"### {title}")
        step = steps.get(key, {"status": "pending", "result": None})
        render_step(key, step)
    st.success("✅ Completed!")
