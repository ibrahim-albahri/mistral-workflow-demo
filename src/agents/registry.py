"""Create (or reuse) the persistent agents that make up the document processor.

Topology — a handoff chain. The probe in ``probe_handoff.py`` established that a
conversation stays on whichever agent it was handed off to, so every agent has to
carry the handoff to its own successor:

    pdp-supervisor ──> pdp-preprocessor ──> pdp-classifier
                   └────────────────────────>┘

Extraction is **not** a handoff. The classifier, when it held handoffs to the
extractors, took one in the same response as the classification — running extraction
before the confidence gate could stop it. So the orchestrator addresses
``pdp-extractor-<category>`` directly in its own conversation, and the gate stays
authoritative.

The API also rejects ``completion_args`` on an agent-bound conversation, so a response
schema cannot be supplied per call. That is why there is one extractor agent per
category, each with its own schema baked in at creation time.
"""

import os
from dataclasses import dataclass, field
from typing import Any

from mistralai.client import Mistral
from mistralai.extra.utils import response_format_from_pydantic_model

from agents.tools import PREPROCESSING_TOOL_SCHEMAS
from shared.extraction_fields import (
    PERSONAL_DOCUMENT_CATEGORIES,
    PERSONAL_DOCUMENT_SPECIFIC_FIELDS,
)
from shared.personal_documents import (
    CLASSIFIER_SYSTEM_PROMPT,
    PREPROCESSING_SYSTEM_PROMPT,
    PersonalDocumentClassification,
    PreprocessingDecision,
    extractor_system_prompt,
    get_personal_extraction_output_model,
)

SUPERVISOR_NAME = "pdp-supervisor"
PREPROCESSOR_NAME = "pdp-preprocessor"
CLASSIFIER_NAME = "pdp-classifier"

#: Comfortably above a full extraction (GTC key_clauses is the longest).
EXTRACTION_MAX_TOKENS = 4000


def extractor_name(category: str) -> str:
    return f"pdp-extractor-{category}"


def _model(env_var: str) -> str:
    return os.environ.get(env_var, "mistral-medium-latest")


SUPERVISOR_INSTRUCTIONS = (
    "You route personal identity and compliance documents through a processing "
    f"chain. When the user says the document is an image, hand off to "
    f"{PREPROCESSOR_NAME} so the image is optimised before it is read. When the user "
    f"says the document is a PDF, hand off to {CLASSIFIER_NAME} directly — PDFs are "
    "never preprocessed. Hand off immediately; never answer the user yourself and "
    "never ask a clarifying question."
)

PREPROCESSOR_INSTRUCTIONS = (
    PREPROCESSING_SYSTEM_PROMPT
    + " Reply with the JSON object and stop there. Do not hand off to another agent "
    "until the user explicitly asks you to."
)

CLASSIFIER_INSTRUCTIONS = (
    CLASSIFIER_SYSTEM_PROMPT + " Classify the document, reply with the JSON, and stop."
)


@dataclass(frozen=True)
class AgentSpec:
    name: str
    model: str
    description: str
    instructions: str
    tools: list[dict[str, Any]] = field(default_factory=list)
    completion_args: dict[str, Any] | None = None
    handoffs: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class AgentRegistry:
    """Ids of the agents that make up one processing chain."""

    supervisor: str
    preprocessor: str
    classifier: str
    extractors: dict[str, str]

    def extractor_for(self, category: str) -> str:
        try:
            return self.extractors[category]
        except KeyError as exc:
            raise ValueError(f"No extractor agent for category '{category}'.") from exc


def extractor_spec(category: str) -> AgentSpec:
    """One extractor per category — the schema has to be baked into the agent."""
    specific_fields = PERSONAL_DOCUMENT_SPECIFIC_FIELDS.get(category, [])
    field_names = ", ".join(key for key, _ in specific_fields) or "none"
    return AgentSpec(
        name=extractor_name(category),
        model=_model("MISTRAL_EXTRACTOR_MODEL"),
        description=f"Extracts structured fields from a '{category}' document.",
        instructions=(
            f"{extractor_system_prompt(category)} You only ever handle documents of "
            f"category '{category}'. Its category-specific fields are: {field_names}."
        ),
        completion_args={
            "temperature": 0,
            # A degraded MRZ can send the model into a repetition loop on filler
            # characters, which truncates the JSON and makes the reply unparseable.
            # Bounding the reply keeps that failure cheap and fast.
            "max_tokens": EXTRACTION_MAX_TOKENS,
            "response_format": response_format_from_pydantic_model(
                get_personal_extraction_output_model(category)
            ),
        },
    )


async def _find_agent(client: Mistral, name: str):
    agents = await client.beta.agents.list_async(name=name, page_size=100)
    for agent in agents:
        if agent.name == name:
            return agent
    return None


async def ensure_agent(client: Mistral, spec: AgentSpec) -> str:
    """Create the agent, or update the existing one to match the spec. Returns its id."""
    existing = await _find_agent(client, spec.name)
    payload: dict[str, Any] = {
        "model": spec.model,
        "name": spec.name,
        "description": spec.description,
        "instructions": spec.instructions,
    }
    # The API rejects empty `tools`/`handoffs` lists, so only send them when populated.
    if spec.tools:
        payload["tools"] = spec.tools
    if spec.handoffs:
        payload["handoffs"] = spec.handoffs
    if spec.completion_args is not None:
        payload["completion_args"] = spec.completion_args

    if existing is None:
        created = await client.beta.agents.create_async(**payload)
        return created.id

    # On update, an omitted `handoffs` leaves whatever the agent already has, so a
    # spec that drops its handoffs has to clear them explicitly with null.
    if not spec.handoffs:
        payload["handoffs"] = None
    updated = await client.beta.agents.update_async(agent_id=existing.id, **payload)
    return updated.id


async def ensure_registry(client: Mistral) -> AgentRegistry:
    """Create or refresh the whole chain, leaves first so handoffs can be wired."""
    extractors = {
        category: await ensure_agent(client, extractor_spec(category))
        for category in PERSONAL_DOCUMENT_CATEGORIES
    }

    classifier = await ensure_agent(
        client,
        AgentSpec(
            name=CLASSIFIER_NAME,
            model=_model("MISTRAL_CLASSIFIER_MODEL"),
            description="Classifies a personal document into one of the known categories.",
            instructions=CLASSIFIER_INSTRUCTIONS,
            completion_args={
                "temperature": 0,
                "response_format": response_format_from_pydantic_model(
                    PersonalDocumentClassification
                ),
            },
            # Deliberately no handoffs to the extractors. When it had them it
            # handed off to one in the same response as the classification, which
            # runs extraction before the confidence gate has had a say. Extraction
            # is addressed directly instead, so the gate is authoritative.
        ),
    )

    preprocessor = await ensure_agent(
        client,
        AgentSpec(
            name=PREPROCESSOR_NAME,
            model=_model("MISTRAL_PREPROCESSING_AGENT_MODEL"),
            description="Selects image preprocessing operations for document OCR.",
            instructions=PREPROCESSOR_INSTRUCTIONS,
            tools=PREPROCESSING_TOOL_SCHEMAS,
            # Without a baked schema the model leaks planned tool calls into its
            # message text instead of emitting them as function.call entries.
            completion_args={
                "temperature": 0,
                "response_format": response_format_from_pydantic_model(
                    PreprocessingDecision
                ),
            },
            handoffs=[classifier],
        ),
    )

    supervisor = await ensure_agent(
        client,
        AgentSpec(
            name=SUPERVISOR_NAME,
            model=_model("MISTRAL_SUPERVISOR_AGENT_MODEL"),
            description="Routes a personal document through preprocessing and classification.",
            instructions=SUPERVISOR_INSTRUCTIONS,
            handoffs=[preprocessor, classifier],
        ),
    )

    return AgentRegistry(
        supervisor=supervisor,
        preprocessor=preprocessor,
        classifier=classifier,
        extractors=extractors,
    )


async def delete_registry(client: Mistral) -> list[str]:
    """Delete every pdp-* agent. Returns the names removed."""
    removed: list[str] = []
    agents = await client.beta.agents.list_async(page_size=100)
    for agent in agents:
        if not agent.name.startswith("pdp-"):
            continue
        await client.beta.agents.delete_async(agent_id=agent.id)
        removed.append(agent.name)
    return removed
