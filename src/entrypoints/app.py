"""
Streamlit UI for personal documents: ID / Passport / Proof of Address / GTC.
"""

import asyncio
import csv
import io
import json
import os
import time
import uuid
from typing import Any

import streamlit as st
from dotenv import load_dotenv
from pydantic import BaseModel

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

from mistralai.client import Mistral
from mistralai.workflows.client import get_mistral_client
from shared.document_media import (
    SUPPORTED_DOCUMENT_EXTENSIONS,
    get_document_content_type,
)
from shared.extraction_display import format_extraction_value
from shared.extraction_fields import PERSONAL_DOCUMENT_LABELS
from shared.workflow_results import workflow_result_mapping, workflow_status_name

load_dotenv(override=True)

API_KEY = os.environ["MISTRAL_API_KEY"]
BASE_URL = os.environ.get("SERVER_URL", "https://api.mistral.ai")

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
    ("ocr", "✅ Document Preparation"),
    ("classify", "🏷️ Classification"),
    ("extract", "🧾 Personal Document Extraction"),
]


class PersonalDocumentInput(BaseModel):
    file_id: str
    filename: str
    confidence_threshold: float = 0.9
    content_type: str = "application/pdf"


class ManualCategorySignal(BaseModel):
    category: str


class BatchDocumentInput(BaseModel):
    document_id: str
    file_id: str
    filename: str
    content_type: str


class BatchPersonalDocumentInput(BaseModel):
    documents: list[BatchDocumentInput]
    confidence_threshold: float = 0.9


class BatchManualCategorySignal(BaseModel):
    document_id: str
    category: str


def get_workflows_client():
    return get_mistral_client(
        server_url=BASE_URL,
        api_key=API_KEY,
    )


def run_async(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.run_until_complete(loop.shutdown_asyncgens())
        loop.close()


async def upload_document(
    document_bytes: bytes, filename: str, content_type: str
) -> str:
    async with Mistral(api_key=API_KEY) as client:
        resp = await client.files.upload_async(
            file={
                "file_name": filename,
                "content": document_bytes,
                "content_type": content_type,
            },
            purpose="ocr",
        )
    return resp.id


async def trigger_workflow(
    file_id: str,
    filename: str,
    confidence_threshold: float,
    content_type: str,
) -> str:
    execution_id = f"personal-doc-{uuid.uuid4().hex[:12]}"
    async with get_workflows_client() as client:
        resp = await client.workflows.execute_workflow_async(
            workflow_identifier="personal_document_workflow",
            input=PersonalDocumentInput(
                file_id=file_id,
                filename=filename,
                confidence_threshold=confidence_threshold,
                content_type=content_type,
            ).model_dump(mode="json"),
            execution_id=execution_id,
        )
    return resp.execution_id


async def trigger_batch_workflow(
    documents: list[BatchDocumentInput], confidence_threshold: float
) -> str:
    execution_id = f"personal-doc-batch-{uuid.uuid4().hex[:12]}"
    async with get_workflows_client() as client:
        resp = await client.workflows.execute_workflow_async(
            workflow_identifier="batch_personal_document_workflow",
            input=BatchPersonalDocumentInput(
                documents=documents, confidence_threshold=confidence_threshold
            ).model_dump(mode="json"),
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


async def poll_batch_status(execution_id: str) -> dict:
    async with get_workflows_client() as client:
        resp = await client.workflows.executions.query_workflow_execution_async(
            execution_id=execution_id,
            name="get_batch_status",
        )
    return resp.result or {}


async def get_execution_details(execution_id: str):
    async with get_workflows_client() as client:
        return await client.workflows.executions.get_workflow_execution_async(
            execution_id=execution_id
        )


async def send_signal(execution_id: str, category: str):
    async with get_workflows_client() as client:
        await client.workflows.executions.signal_workflow_execution_async(
            execution_id=execution_id,
            name="manual_category",
            input=ManualCategorySignal(category=category).model_dump(mode="json"),
        )


async def send_batch_signal(execution_id: str, document_id: str, category: str):
    async with get_workflows_client() as client:
        await client.workflows.executions.signal_workflow_execution_async(
            execution_id=execution_id,
            name="manual_batch_category",
            input=BatchManualCategorySignal(
                document_id=document_id, category=category
            ).model_dump(mode="json"),
        )


def batch_csv_bytes(documents: list[dict]) -> bytes:
    """Return the compact, one-row-per-document batch export."""
    output = io.StringIO(newline="")
    fields = [
        "document_id",
        "filename",
        "content_type",
        "status",
        "category",
        "confidence",
        "explanation",
        *COMMON_FIELD_LABELS,
        "specific_fields",
        "error",
    ]
    writer = csv.DictWriter(output, fieldnames=fields)
    writer.writeheader()
    for document in documents:
        classification = document.get("classification") or {}
        extraction = document.get("extraction") or {}
        common = extraction.get("common") or {}
        writer.writerow(
            {
                "document_id": document.get("document_id", ""),
                "filename": document.get("filename", ""),
                "content_type": document.get("content_type", ""),
                "status": document.get("status", ""),
                "category": classification.get("category", ""),
                "confidence": classification.get("confidence", ""),
                "explanation": classification.get("explanation", ""),
                **{key: common.get(key, "") for key in COMMON_FIELD_LABELS},
                "specific_fields": json.dumps(
                    extraction.get("specific") or {}, ensure_ascii=False
                ),
                "error": document.get("error") or "",
            }
        )
    return output.getvalue().encode("utf-8")


def backfill_steps_from_execution_result(steps: dict, result: Any) -> dict:
    result_mapping = workflow_result_mapping(result)
    if not result_mapping:
        return steps

    structured = result_mapping.get("structuredContent")
    if not isinstance(structured, dict):
        structured = result_mapping.get("structured_content")
    if not isinstance(structured, dict):
        structured = result_mapping

    ocr_text = structured.get("ocr_text")
    classification = structured.get("classification")
    personal_document_info = structured.get("personal_document_info")

    updated = dict(steps)
    if ocr_text is not None:
        updated["ocr"] = {"status": "done", "result": ocr_text}
    if isinstance(classification, dict):
        updated["classify"] = {"status": "done", "result": classification}
    if isinstance(personal_document_info, dict):
        updated["extract"] = {"status": "done", "result": personal_document_info}
    return updated


def get_document_preview(document_bytes: bytes, content_type: str):
    if content_type.startswith("image/"):
        return document_bytes
    if content_type != "application/pdf" or not fitz:
        return None
    try:
        doc = fitz.open(stream=document_bytes, filetype="pdf")
        page = doc[0]
        pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5))
        img_bytes = pix.tobytes("ppm")
        return io.BytesIO(img_bytes)
    except Exception:
        return None


def render_step(key: str, step: dict):
    status = step.get("status", "pending")
    result = step.get("result")

    if status == "pending":
        st.markdown("⏳ Pending…")
    elif status == "running":
        st.markdown("⚙️ In Progress…")
    elif status == "waiting_human":
        result = step.get("result", {})
        confidence = result.get("confidence", 0.0) if result else 0.0
        st.warning(
            f"⚠️ Insufficient confidence ({confidence * 100:.0f}%). Please choose the category manually."
        )
        selected = st.selectbox(
            "Category",
            options=list(PERSONAL_DOCUMENT_LABELS.keys()),
            format_func=lambda k: PERSONAL_DOCUMENT_LABELS[k],
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
            category = result.get("category", "gtc")
            confidence = result.get("confidence", 0.0)
            explanation = result.get("explanation", "")
            label = PERSONAL_DOCUMENT_LABELS.get(category, f"❓ {category}")
            col1, col2 = st.columns([3, 1])
            col1.markdown(f"**{label}**")
            col1.caption(explanation)
            col2.metric("Confidence", f"{confidence * 100:.0f}%")
            col2.progress(confidence)
        elif key == "extract":
            common = result.get("common", {})
            specific = result.get("specific", {})

            st.markdown("**🧍 Common Information**")
            common_rows = [
                {
                    "Field": COMMON_FIELD_LABELS.get(k, k),
                    "Value": format_extraction_value(v),
                }
                for k, v in common.items()
                if v is not None
            ]
            if common_rows:
                st.table(common_rows)
            else:
                st.info("No common information found.")

            if specific:
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


st.set_page_config(page_title="Personal Documents", page_icon="🧾", layout="wide")
st.title("🧾 Personal Documents")
st.caption(
    "Upload an ID, passport, proof of address, or GTC → OCR → Classification → Extraction"
)

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
if "batch_execution_id" not in st.session_state:
    st.session_state.batch_execution_id = None
if "batch_status" not in st.session_state:
    st.session_state.batch_status = {}
if "batch_done" not in st.session_state:
    st.session_state.batch_done = False
if "batch_error" not in st.session_state:
    st.session_state.batch_error = None

uploaded = st.file_uploader(
    "Choose a PDF or image file",
    type=list(SUPPORTED_DOCUMENT_EXTENSIONS),
)

if uploaded is not None:
    st.info(f"**{uploaded.name}** — {uploaded.size / 1024:.1f} KB")

    if st.button("Start Workflow", type="primary"):
        st.session_state.execution_id = None
        st.session_state.done = False
        st.session_state.steps = {}
        st.session_state.poll_error = None
        st.session_state.signal_sent = False

        document_bytes = uploaded.read()
        filename = uploaded.name
        content_type = get_document_content_type(filename)

        with st.status("Uploading document…", expanded=False) as s:
            file_id = run_async(upload_document(document_bytes, filename, content_type))
            s.update(label="Upload ✓", state="complete")

        execution_id = run_async(
            trigger_workflow(file_id, filename, confidence_threshold, content_type)
        )
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

    col_document, col_steps = st.columns([1, 1.2])

    with col_document:
        st.markdown("### 📄 Document")
        if uploaded:
            document_bytes = uploaded.read()
            uploaded.seek(0)
            content_type = get_document_content_type(uploaded.name)
            preview = get_document_preview(document_bytes, content_type)
            if preview:
                st.image(preview, width="stretch")
            else:
                st.info("Preview is unavailable for this document.")

    with col_steps:
        if st.session_state.poll_error:
            st.warning(f"Progress polling failed: {st.session_state.poll_error}")
        for key, title in STEPS_CONFIG:
            st.markdown(f"### {title}")
            step = steps.get(key, {"status": "pending", "result": None})
            render_step(key, step)

    all_done = all(steps.get(k, {}).get("status") == "done" for k, _ in STEPS_CONFIG)
    waiting_human = any(
        steps.get(k, {}).get("status") == "waiting_human" for k, _ in STEPS_CONFIG
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
            wf_status = workflow_status_name(execution.status)
            if wf_status == "COMPLETED":
                st.session_state.steps = backfill_steps_from_execution_result(
                    st.session_state.steps,
                    execution.result,
                )
                st.session_state.done = True
                st.success("✅ Completed!")
                st.rerun()
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


st.divider()
st.header("Batch processing")
st.caption(
    "Upload up to 100 documents. Mistral OCR runs first, followed by Batch API classification and extraction."
)
batch_uploads = st.file_uploader(
    "Choose PDF or image files for a batch",
    type=list(SUPPORTED_DOCUMENT_EXTENSIONS),
    accept_multiple_files=True,
    key="batch_file_uploader",
)

if len(batch_uploads or []) > 100:
    st.error("A batch can contain at most 100 documents.")
elif batch_uploads:
    st.caption(f"{len(batch_uploads)} document(s) selected")
    if st.button("Start Batch Workflow", type="primary"):
        st.session_state.batch_execution_id = None
        st.session_state.batch_status = {}
        st.session_state.batch_done = False
        st.session_state.batch_error = None
        documents: list[BatchDocumentInput] = []
        with st.status("Uploading batch documents…", expanded=True) as status:
            for index, uploaded_batch_file in enumerate(batch_uploads, start=1):
                filename = uploaded_batch_file.name
                content_type = get_document_content_type(filename)
                file_id = run_async(
                    upload_document(
                        uploaded_batch_file.getvalue(), filename, content_type
                    )
                )
                documents.append(
                    BatchDocumentInput(
                        document_id=uuid.uuid4().hex,
                        file_id=file_id,
                        filename=filename,
                        content_type=content_type,
                    )
                )
                status.update(label=f"Uploaded {index}/{len(batch_uploads)} documents…")
            status.update(label="Uploads complete", state="complete")
        st.session_state.batch_execution_id = run_async(
            trigger_batch_workflow(documents, confidence_threshold)
        )
        st.rerun()

if st.session_state.batch_execution_id:
    try:
        batch_status = run_async(poll_batch_status(st.session_state.batch_execution_id))
        st.session_state.batch_status = batch_status
        st.session_state.batch_error = None
    except Exception as exc:
        batch_status = st.session_state.batch_status
        st.session_state.batch_error = str(exc)

    if st.session_state.batch_error:
        st.warning(f"Batch progress polling failed: {st.session_state.batch_error}")

    documents = batch_status.get("documents", [])
    counts = batch_status.get("counts", {})
    if counts:
        st.caption(" · ".join(f"{status}: {count}" for status, count in counts.items()))
    if documents:
        table_rows = [
            {
                "Filename": document.get("filename"),
                "Status": document.get("status"),
                "Category": (document.get("classification") or {}).get("category"),
                "Confidence": (document.get("classification") or {}).get("confidence"),
                "Error": document.get("error"),
            }
            for document in documents
        ]
        st.dataframe(table_rows, width="stretch", hide_index=True)

        awaiting_review = [
            document
            for document in documents
            if document.get("status") == "awaiting_review"
        ]
        for document in awaiting_review:
            with st.expander(
                f"Review required: {document.get('filename')}", expanded=True
            ):
                classification = document.get("classification") or {}
                st.warning(
                    f"Model confidence: {classification.get('confidence', 0.0) * 100:.0f}%"
                )
                category = st.selectbox(
                    "Category",
                    options=list(PERSONAL_DOCUMENT_LABELS),
                    format_func=lambda key: PERSONAL_DOCUMENT_LABELS[key],
                    key=f"batch-category-{document['document_id']}",
                )
                if st.button(
                    "Validate category", key=f"batch-submit-{document['document_id']}"
                ):
                    run_async(
                        send_batch_signal(
                            st.session_state.batch_execution_id,
                            document["document_id"],
                            category,
                        )
                    )
                    st.rerun()

        for document in documents:
            extraction = document.get("extraction")
            if extraction:
                with st.expander(f"Extraction: {document.get('filename')}"):
                    render_step("extract", {"status": "done", "result": extraction})

        st.download_button(
            "Download batch CSV",
            data=batch_csv_bytes(documents),
            file_name="personal-document-batch.csv",
            mime="text/csv",
        )

    active_statuses = {
        "ocr_pending",
        "ocr_running",
        "classification_running",
        "queued_for_extraction",
        "extracting",
    }
    if any(document.get("status") in active_statuses for document in documents):
        time.sleep(1)
        st.rerun()
    elif documents and not any(
        document.get("status") == "awaiting_review" for document in documents
    ):
        st.session_state.batch_done = True
        st.success("Batch processing complete.")
