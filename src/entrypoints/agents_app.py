"""Streamlit UI for the Mistral Agents implementation.

Same four step panels as the workflow UI (they share ``shared.streamlit_ui``), but
there is no worker and nothing to poll: the run executes in-process and, when
classification confidence is low, is parked in ``st.session_state`` until a reviewer
picks a category — at which point its stored conversation is resumed.
"""

import asyncio
import json
import os

import streamlit as st
from dotenv import load_dotenv

from agents.files import mistral_client, upload_document
from agents.orchestrator import PersonalDocumentRun
from agents.registry import (
    CLASSIFIER_NAME,
    PREPROCESSOR_NAME,
    SUPERVISOR_NAME,
    extractor_name,
)
from shared.document_media import (
    SUPPORTED_DOCUMENT_EXTENSIONS,
    get_document_content_type,
)
from shared.extraction_fields import PERSONAL_DOCUMENT_CATEGORIES
from shared.streamlit_ui import STEPS_CONFIG, get_document_preview, render_step

load_dotenv(override=True)

os.environ["MISTRAL_API_KEY"]  # fail fast with a clear KeyError if unset


def run_async(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.run_until_complete(loop.shutdown_asyncgens())
        loop.close()


async def upload(document_bytes: bytes, filename: str, content_type: str) -> str:
    async with mistral_client() as client:
        return await upload_document(client, document_bytes, filename, content_type)


st.set_page_config(
    page_title="Personal Documents (Agents)", page_icon="🤖", layout="wide"
)
st.title("🤖 Personal Documents — Mistral Agents")
st.caption(
    "Supervisor agent → preprocessing agent → classifier agent → per-category "
    "extractor agent, wired together with server-side handoffs."
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

    st.divider()
    st.header("🧩 Agent chain")
    st.markdown(
        f"- `{SUPERVISOR_NAME}` → `{PREPROCESSOR_NAME}`, `{CLASSIFIER_NAME}`\n"
        f"- `{PREPROCESSOR_NAME}` → `{CLASSIFIER_NAME}`\n"
        f"- `{CLASSIFIER_NAME}` → "
        + ", ".join(f"`{extractor_name(c)}`" for c in PERSONAL_DOCUMENT_CATEGORIES)
    )
    st.caption(
        "Each agent carries the handoff to its own successor: a conversation stays "
        "on whichever agent it was handed to."
    )

if "run" not in st.session_state:
    st.session_state.run = None
if "run_error" not in st.session_state:
    st.session_state.run_error = None

uploaded = st.file_uploader(
    "Choose a PDF or image file", type=list(SUPPORTED_DOCUMENT_EXTENSIONS)
)

if uploaded is not None:
    st.info(f"**{uploaded.name}** — {uploaded.size / 1024:.1f} KB")
    document_bytes = uploaded.getvalue()
    content_type = get_document_content_type(uploaded.name)

    if st.button("Start Agent Run", type="primary"):
        st.session_state.run = None
        st.session_state.run_error = None
        with st.status("Uploading document…", expanded=False) as status:
            file_id = run_async(upload(document_bytes, uploaded.name, content_type))
            status.update(label="Upload ✓", state="complete")

        run = PersonalDocumentRun(
            file_id=file_id,
            filename=uploaded.name,
            content_type=content_type,
            confidence_threshold=confidence_threshold,
        )
        with st.status("Running the agent chain…", expanded=False) as status:
            try:
                run_async(run.start())
                status.update(label="Agent chain ✓", state="complete")
            except Exception as exc:  # noqa: BLE001 - surfaced in the UI below
                run.error = str(exc)
                status.update(label="Agent chain failed", state="error")
                st.session_state.run_error = str(exc)
        st.session_state.run = run
        st.rerun()

run: PersonalDocumentRun | None = st.session_state.run

if run is not None:
    col_document, col_steps = st.columns([1, 1.2])

    with col_document:
        st.markdown("### 📄 Document")
        if uploaded is not None:
            preview = get_document_preview(
                uploaded.getvalue(), get_document_content_type(uploaded.name)
            )
            if preview:
                st.image(preview, width="stretch")
            else:
                st.info("Preview is unavailable for this document.")
        if run.conversation_id:
            st.caption(f"Conversation `{run.conversation_id}`")

    with col_steps:
        if st.session_state.run_error:
            st.error(f"The agent run failed: {st.session_state.run_error}")

        def resume(category: str) -> None:
            try:
                run_async(run.resume_with_category(category))
            except Exception as exc:  # noqa: BLE001 - surfaced on the next rerun
                st.session_state.run_error = str(exc)
            st.session_state.run = run
            st.rerun()

        for key, title in STEPS_CONFIG:
            st.markdown(f"### {title}")
            render_step(
                key,
                run.steps.get(key, {"status": "pending", "result": None}),
                on_manual_category=resume,
                key_prefix="agents-",
            )

    if run.extraction is not None:
        st.success("✅ Completed!")
        with st.expander("Raw result"):
            st.code(json.dumps(run.result(), indent=2, ensure_ascii=False), "json")
    elif run.pending == "category":
        st.info("Waiting for a reviewer to confirm the category.")
