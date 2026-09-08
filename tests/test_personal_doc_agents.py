# ruff: noqa: E402

import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from agents import orchestrator as orch
from agents.files import MAX_ATTEMPTS, _is_retryable, call_with_retry
from agents.orchestrator import (
    MAX_PREPROCESSING_OPERATIONS,
    MAX_TOOL_ITERATIONS,
    AgentRunError,
    PersonalDocumentRun,
    content_to_text,
    function_calls,
    json_payload,
    message_text,
    user_entry,
)
from agents.registry import (
    CLASSIFIER_NAME,
    PREPROCESSOR_NAME,
    SUPERVISOR_NAME,
    AgentRegistry,
    AgentSpec,
    ensure_agent,
    ensure_registry,
    extractor_name,
    extractor_spec,
)
from agents.tools import (
    AGENT_OPERATIONS,
    APPLY_TOOL_NAME,
    INSPECT_TOOL_NAME,
    PREPROCESSING_TOOL_SCHEMAS,
    UNAVAILABLE_OPERATIONS,
)
from shared.extraction_fields import PERSONAL_DOCUMENT_CATEGORIES
from shared.preprocessing import PREPROCESSING_OPERATIONS


# ── Fakes ─────────────────────────────────────────────────────────────────────


def entry(entry_type: str, **fields):
    return SimpleNamespace(type=entry_type, **fields)


def message(text: str, agent_id: str | None = None):
    return entry("message.output", content=text, agent_id=agent_id)


def call(name: str, arguments: dict, tool_call_id: str = "tc-1", agent_id=None):
    return entry(
        "function.call",
        name=name,
        arguments=json.dumps(arguments),
        tool_call_id=tool_call_id,
        agent_id=agent_id,
    )


class FakeAgents:
    """Stands in for ``client.beta.agents``."""

    def __init__(self, existing=None):
        self.existing = list(existing or [])
        self.created: list[dict] = []
        self.updated: list[dict] = []
        self._counter = 0

    async def list_async(self, *, name=None, page_size=None):
        if name is None:
            return list(self.existing)
        return [agent for agent in self.existing if agent.name == name]

    async def create_async(self, **payload):
        self._counter += 1
        self.created.append(payload)
        agent = SimpleNamespace(id=f"ag_{self._counter}", name=payload["name"])
        self.existing.append(agent)
        return agent

    async def update_async(self, *, agent_id, **payload):
        self.updated.append({"agent_id": agent_id, **payload})
        return SimpleNamespace(id=agent_id, name=payload["name"])

    async def delete_async(self, *, agent_id):
        self.existing = [a for a in self.existing if a.id != agent_id]


class FakeConversations:
    """Replays a scripted list of responses and records what was sent."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.appends: list[dict] = []
        self.starts: list[dict] = []

    def _next(self, conversation_id="conv-1"):
        outputs = self.responses.pop(0) if self.responses else []
        return SimpleNamespace(conversation_id=conversation_id, outputs=outputs)

    async def start_async(self, **kwargs):
        self.starts.append(kwargs)
        return self._next()

    async def append_async(self, **kwargs):
        self.appends.append(kwargs)
        return self._next(kwargs["conversation_id"])


class FakeClient:
    def __init__(self, conversations=None, agents=None):
        self.beta = SimpleNamespace(
            conversations=conversations or FakeConversations([]),
            agents=agents or FakeAgents(),
        )
        self.files = SimpleNamespace()


REGISTRY = AgentRegistry(
    supervisor="ag-sup",
    preprocessor="ag-pre",
    classifier="ag-cls",
    extractors={
        category: f"ag-ex-{category}" for category in PERSONAL_DOCUMENT_CATEGORIES
    },
)


def make_run(**overrides) -> PersonalDocumentRun:
    run = PersonalDocumentRun(
        file_id=overrides.pop("file_id", "file-1"),
        filename=overrides.pop("filename", "passport.jpg"),
        content_type=overrides.pop("content_type", "image/jpeg"),
        confidence_threshold=overrides.pop("confidence_threshold", 0.9),
    )
    run.conversation_id = "conv-1"
    run._registry = REGISTRY
    run._processed_file_id = run.file_id
    run._processed_filename = run.filename
    run._processed_content_type = run.content_type
    for key, value in overrides.items():
        setattr(run, key, value)
    return run


# ── Entry helpers ─────────────────────────────────────────────────────────────


def test_message_text_returns_only_the_last_message_from_that_agent():
    outputs = [
        message("first", agent_id="ag-pre"),
        message("classifier chatter", agent_id="ag-cls"),
        message("last", agent_id="ag-pre"),
    ]

    assert message_text(outputs, agent_id="ag-pre") == "last"
    assert message_text(outputs, agent_id="ag-cls") == "classifier chatter"
    assert message_text(outputs) == "last"
    assert message_text([], agent_id="ag-pre") == ""


def test_content_to_text_flattens_chunk_lists_and_ignores_non_text_chunks():
    chunks = [
        SimpleNamespace(type="text", text='{"a": 1}'),
        SimpleNamespace(type="image_url", image_url="https://example.test/x.png"),
    ]

    assert content_to_text(chunks) == '{"a": 1}'
    assert content_to_text("plain") == "plain"


def test_json_payload_accepts_fences_and_a_repeated_object():
    assert json_payload('{"category": "gtc"}') == {"category": "gtc"}
    assert json_payload('```json\n{"category": "id"}\n```') == {"category": "id"}
    # Agents under a strict response_format sometimes emit the object twice.
    assert json_payload('{"category": "id"}{"category": "id"}') == {"category": "id"}
    assert json_payload('Here you go: {"category": "gtc"}') == {"category": "gtc"}


def test_json_payload_rejects_replies_without_an_object():
    with pytest.raises(AgentRunError):
        json_payload("no json here")
    with pytest.raises(AgentRunError):
        json_payload("{not valid}")


def test_user_entry_wraps_chunks_as_a_message_entry():
    wrapped = user_entry([{"type": "text", "text": "hi"}])

    assert wrapped["type"] == "message.input"
    assert wrapped["object"] == "entry"
    assert wrapped["role"] == "user"
    assert wrapped["content"] == [{"type": "text", "text": "hi"}]


def test_function_calls_selects_only_function_call_entries():
    outputs = [message("x"), call("t", {}), entry("agent.handoff")]

    assert [e.name for e in function_calls(outputs)] == ["t"]


# ── Registry ──────────────────────────────────────────────────────────────────


def test_extractor_spec_bakes_the_category_schema_into_the_agent():
    spec = extractor_spec("passport")

    assert spec.name == "pdp-extractor-passport"
    schema = spec.completion_args["response_format"]["json_schema"]
    assert schema["name"] == "PersonalExtractionOutput_passport"
    assert spec.completion_args["temperature"] == 0
    assert "passport_number" in json.dumps(schema["schema"])
    # No handoffs: an extractor is the end of the chain.
    assert spec.handoffs == []


def test_ensure_agent_creates_when_absent_and_omits_empty_lists():
    agents = FakeAgents()
    client = FakeClient(agents=agents)
    spec = AgentSpec(name="pdp-x", model="m", description="d", instructions="i")

    agent_id = asyncio.run(ensure_agent(client, spec))

    assert agent_id == "ag_1"
    payload = agents.created[0]
    # The API rejects empty tools/handoffs lists.
    assert "tools" not in payload and "handoffs" not in payload
    assert payload["name"] == "pdp-x"


def test_ensure_agent_updates_an_existing_agent_instead_of_duplicating_it():
    agents = FakeAgents(existing=[SimpleNamespace(id="ag-old", name="pdp-x")])
    client = FakeClient(agents=agents)
    spec = AgentSpec(
        name="pdp-x",
        model="m",
        description="d",
        instructions="new instructions",
        handoffs=["ag-y"],
    )

    agent_id = asyncio.run(ensure_agent(client, spec))

    assert agent_id == "ag-old"
    assert agents.created == []
    assert agents.updated[0]["agent_id"] == "ag-old"
    assert agents.updated[0]["instructions"] == "new instructions"
    assert agents.updated[0]["handoffs"] == ["ag-y"]


def test_ensure_registry_wires_each_agent_to_its_own_successor():
    """A conversation stays on the agent it was handed to, so the chain is linear."""
    agents = FakeAgents()
    client = FakeClient(agents=agents)

    registry = asyncio.run(ensure_registry(client))

    by_name = {payload["name"]: payload for payload in agents.created}
    assert set(registry.extractors) == set(PERSONAL_DOCUMENT_CATEGORIES)

    # The classifier must NOT be able to reach an extractor: when it could, it
    # handed off in the same response as the classification, jumping the gate.
    classifier = by_name[CLASSIFIER_NAME]
    assert "handoffs" not in classifier

    preprocessor = by_name[PREPROCESSOR_NAME]
    assert preprocessor["handoffs"] == [registry.classifier]
    assert {t["function"]["name"] for t in preprocessor["tools"]} == {
        INSPECT_TOOL_NAME,
        APPLY_TOOL_NAME,
    }

    supervisor = by_name[SUPERVISOR_NAME]
    assert supervisor["handoffs"] == [registry.preprocessor, registry.classifier]
    # The supervisor must not reach an extractor directly.
    assert not set(supervisor["handoffs"]) & set(registry.extractors.values())


def test_registry_extractor_for_rejects_an_unknown_category():
    assert REGISTRY.extractor_for("passport") == "ag-ex-passport"
    with pytest.raises(ValueError):
        REGISTRY.extractor_for("nonsense")


def test_extractor_name_matches_the_registry_keys():
    for category in PERSONAL_DOCUMENT_CATEGORIES:
        assert extractor_name(category) == f"pdp-extractor-{category}"


# ── Tools ─────────────────────────────────────────────────────────────────────


def test_agent_operations_exclude_operations_without_their_dependency():
    assert "sauvola" in PREPROCESSING_OPERATIONS
    assert "sauvola" not in AGENT_OPERATIONS


def test_the_agent_is_never_offered_a_content_destroying_operation():
    """table_grid_removal inpainted the MRZ away in a real run, losing every field
    the MRZ would otherwise have filled in."""
    from agents.tools import DESTRUCTIVE_OPERATIONS

    assert "table_grid_removal" in DESTRUCTIVE_OPERATIONS
    for operation in DESTRUCTIVE_OPERATIONS:
        assert operation in PREPROCESSING_OPERATIONS, operation
        assert operation not in AGENT_OPERATIONS, operation

    assert set(AGENT_OPERATIONS) == (
        set(PREPROCESSING_OPERATIONS) - UNAVAILABLE_OPERATIONS - DESTRUCTIVE_OPERATIONS
    )


def test_the_agent_menu_matches_the_conservative_rule_based_profile():
    """The agent may pick exactly what conservative_ocr_config leaves enabled."""
    from shared.preprocessing import conservative_ocr_config

    config = conservative_ocr_config()

    assert config.enable_table_grid_removal is False
    assert config.enable_stamp_removal is False
    assert config.enable_perspective is False
    assert config.enable_sauvola is False
    for operation in ("table_grid_removal", "stamp_removal", "perspective", "sauvola"):
        assert operation not in AGENT_OPERATIONS, operation


def test_the_agent_keeps_every_non_destructive_operation():
    for operation in (
        "sharpen",
        "exposure",
        "denoise",
        "orientation",
        "deskew",
        "margin_crop",
        "shadow_removal",
        "dpi_normalization",
    ):
        assert operation in AGENT_OPERATIONS, operation


def test_tool_schema_enum_matches_the_operations_actually_offered():
    apply_schema = next(
        schema
        for schema in PREPROCESSING_TOOL_SCHEMAS
        if schema["function"]["name"] == APPLY_TOOL_NAME
    )
    enum = apply_schema["function"]["parameters"]["properties"]["operation"]["enum"]

    assert enum == list(AGENT_OPERATIONS)


# ── The tool pump ─────────────────────────────────────────────────────────────


def test_pump_executes_a_tool_and_appends_a_function_result(monkeypatch):
    async def fake_inspect(client, file_id, filename):
        return {
            "file_id": file_id,
            "filename": filename,
            "metrics": {"brightness": 120},
            "preview_url": "https://example.test/preview.png",
        }

    monkeypatch.setitem(orch.TOOL_DISPATCH, INSPECT_TOOL_NAME, fake_inspect)

    conversations = FakeConversations([[message('{"done": true}', agent_id="ag-pre")]])
    client = FakeClient(conversations=conversations)
    run = make_run()
    first = SimpleNamespace(
        conversation_id="conv-1",
        outputs=[call(INSPECT_TOOL_NAME, {"file_id": "file-1", "filename": "p.jpg"})],
    )

    outputs = asyncio.run(run._pump(client, first))

    sent = conversations.appends[0]["inputs"]
    result_entry = sent[0]
    assert result_entry["type"] == "function.result"
    assert result_entry["tool_call_id"] == "tc-1"
    payload = json.loads(result_entry["result"])
    assert payload["metrics"] == {"brightness": 120}
    # The preview goes as an image chunk, never as JSON text.
    assert "preview_url" not in payload
    assert sent[1]["content"] == [
        {"type": "image_url", "image_url": "https://example.test/preview.png"}
    ]
    assert message_text(outputs) == '{"done": true}'


def test_pump_sends_the_preview_image_only_once(monkeypatch):
    async def fake_apply(client, source_file_id, source_filename, operation):
        return {
            "file_id": "file-2",
            "filename": "p-enhanced.png",
            "operation": operation,
            "metrics": {},
            "preview_url": "https://example.test/second.png",
        }

    monkeypatch.setitem(orch.TOOL_DISPATCH, APPLY_TOOL_NAME, fake_apply)

    conversations = FakeConversations(
        [
            [
                call(
                    APPLY_TOOL_NAME,
                    {
                        "source_file_id": "f",
                        "source_filename": "p",
                        "operation": "deskew",
                    },
                    "tc-2",
                )
            ],
            [message("done", agent_id="ag-pre")],
        ]
    )
    client = FakeClient(conversations=conversations)
    run = make_run(_preview_sent=True)
    first = SimpleNamespace(
        conversation_id="conv-1",
        outputs=[
            call(
                APPLY_TOOL_NAME,
                {"source_file_id": "f", "source_filename": "p", "operation": "sharpen"},
                "tc-1",
            )
        ],
    )

    asyncio.run(run._pump(client, first))

    for append in conversations.appends:
        kinds = {item.get("type") for item in append["inputs"]}
        assert kinds == {"function.result"}, "a preview was re-sent"


def test_pump_enforces_the_preprocessing_operation_budget(monkeypatch):
    applied = []

    async def fake_apply(client, source_file_id, source_filename, operation):
        applied.append(operation)
        return {"file_id": "f", "filename": "p", "operation": operation, "metrics": {}}

    monkeypatch.setitem(orch.TOOL_DISPATCH, APPLY_TOOL_NAME, fake_apply)

    run = make_run()
    run._applied_operations = ["a"] * MAX_PREPROCESSING_OPERATIONS
    client = FakeClient()

    result, preview = asyncio.run(
        run._run_tool(
            client,
            call(
                APPLY_TOOL_NAME,
                {"source_file_id": "f", "source_filename": "p", "operation": "deskew"},
                "tc-9",
            ),
        )
    )

    assert applied == [], "the tool ran despite a spent budget"
    assert preview is None
    assert "budget" in json.loads(result["result"])["error"].lower()


def test_an_operation_is_never_applied_twice(monkeypatch):
    """A repeat would also make the final decision fail duplicate validation."""
    applied = []

    async def fake_apply(client, source_file_id, source_filename, operation):
        applied.append(operation)
        return {"file_id": "f", "filename": "p", "operation": operation, "metrics": {}}

    monkeypatch.setitem(orch.TOOL_DISPATCH, APPLY_TOOL_NAME, fake_apply)

    run = make_run()
    run._applied_operations = ["table_grid_removal"]

    result, _ = asyncio.run(
        run._run_tool(
            FakeClient(),
            call(
                APPLY_TOOL_NAME,
                {
                    "source_file_id": "f",
                    "source_filename": "p",
                    "operation": "table_grid_removal",
                },
                "tc-repeat",
            ),
        )
    )

    assert applied == [], "a repeated operation was applied"
    error = json.loads(result["result"])["error"]
    assert "already been applied" in error
    assert run._applied_operations == ["table_grid_removal"]


def test_a_different_operation_is_still_allowed_after_one_is_applied(monkeypatch):
    async def fake_apply(client, source_file_id, source_filename, operation):
        return {"file_id": "f2", "filename": "p", "operation": operation, "metrics": {}}

    monkeypatch.setitem(orch.TOOL_DISPATCH, APPLY_TOOL_NAME, fake_apply)

    run = make_run()
    run._applied_operations = ["deskew"]

    result, _ = asyncio.run(
        run._run_tool(
            FakeClient(),
            call(
                APPLY_TOOL_NAME,
                {"source_file_id": "f", "source_filename": "p", "operation": "sharpen"},
                "tc-ok",
            ),
        )
    )

    assert json.loads(result["result"])["file_id"] == "f2"
    assert run._applied_operations == ["deskew", "sharpen"]


def test_run_tool_reports_a_failure_back_to_the_agent(monkeypatch):
    async def boom(client, **kwargs):
        raise RuntimeError("scikit-image is required")

    monkeypatch.setitem(orch.TOOL_DISPATCH, APPLY_TOOL_NAME, boom)
    run = make_run()

    result, _ = asyncio.run(
        run._run_tool(
            client=FakeClient(),
            call=call(
                APPLY_TOOL_NAME,
                {"source_file_id": "f", "source_filename": "p", "operation": "sauvola"},
                "tc-3",
            ),
        )
    )

    assert "scikit-image" in json.loads(result["result"])["error"]
    assert run._applied_operations == []


def test_run_tool_reports_an_unknown_tool_rather_than_raising():
    run = make_run()

    result, _ = asyncio.run(run._run_tool(FakeClient(), call("not_a_tool", {}, "tc-4")))

    assert "Unknown tool" in json.loads(result["result"])["error"]


def test_pump_gives_up_at_the_iteration_cap_without_losing_the_run(monkeypatch):
    """A chatty agent must not fail the run; phase 1 degrades to the original file."""

    async def fake_inspect(client, file_id, filename):
        return {"file_id": file_id, "filename": filename, "metrics": {}}

    monkeypatch.setitem(orch.TOOL_DISPATCH, INSPECT_TOOL_NAME, fake_inspect)

    looping = [
        [call(INSPECT_TOOL_NAME, {"file_id": "f", "filename": "p"}, f"tc-loop-{i}")]
        for i in range(MAX_TOOL_ITERATIONS + 2)
    ]
    client = FakeClient(conversations=FakeConversations(looping))
    run = make_run()
    first = SimpleNamespace(
        conversation_id="conv-1",
        outputs=[
            call(INSPECT_TOOL_NAME, {"file_id": "f", "filename": "p"}, "tc-first")
        ],
    )

    outputs = asyncio.run(run._pump(client, first))

    # It returns rather than raising, and no decision means "skipped" upstream.
    assert outputs
    with pytest.raises(AgentRunError, match="no decision"):
        run._read_preprocessing_decision(outputs)


def test_the_total_tool_budget_stops_an_agent_that_only_re_inspects(monkeypatch):
    """The operation budget alone left an agent free to loop on inspections."""
    inspections = []

    async def fake_inspect(client, file_id, filename):
        inspections.append(file_id)
        return {"file_id": file_id, "filename": filename, "metrics": {}}

    monkeypatch.setitem(orch.TOOL_DISPATCH, INSPECT_TOOL_NAME, fake_inspect)

    run = make_run()
    run._tool_calls_made = orch.MAX_TOOL_CALLS

    result, _ = asyncio.run(
        run._run_tool(
            FakeClient(),
            call(INSPECT_TOOL_NAME, {"file_id": "f", "filename": "p"}, "tc-over"),
        )
    )

    assert inspections == [], "the tool ran despite a spent budget"
    assert "budget spent" in json.loads(result["result"])["error"]


def test_tool_calls_are_counted_across_both_tools(monkeypatch):
    async def fake_inspect(client, file_id, filename):
        return {"file_id": file_id, "filename": filename, "metrics": {}}

    monkeypatch.setitem(orch.TOOL_DISPATCH, INSPECT_TOOL_NAME, fake_inspect)
    run = make_run()

    for index in range(3):
        asyncio.run(
            run._run_tool(
                FakeClient(),
                call(INSPECT_TOOL_NAME, {"file_id": "f", "filename": "p"}, f"t{index}"),
            )
        )

    assert run._tool_calls_made == 3


# ── Duplicate-append recovery ─────────────────────────────────────────────────


def sdk_error(message: str):
    from mistralai.client.errors import SDKError

    error = SDKError.__new__(SDKError)
    error.status_code = 400
    error.message = message
    error.body = message
    error.args = (message,)
    return error


class DuplicateThenHistory:
    """Rejects the append as a duplicate, then serves the conversation history."""

    def __init__(self, history_entries):
        self.history_entries = history_entries
        self.history_calls = 0

    async def append_async(self, **kwargs):
        raise sdk_error(
            "Function results reference tool_call_ids that already have a result: ['tc-1']"
        )

    async def get_history_async(self, *, conversation_id):
        self.history_calls += 1
        return SimpleNamespace(
            conversation_id=conversation_id, entries=self.history_entries
        )


def test_a_duplicate_append_recovers_the_unseen_entries_from_history():
    """A retried append whose original succeeded must not fail the run."""
    seen = message("already seen", agent_id="ag-pre")
    seen.id = "entry-1"
    fresh = message(
        '{"category": "passport", "confidence": 0.9, "explanation": "ok"}',
        agent_id="ag-cls",
    )
    fresh.id = "entry-2"

    conversations = DuplicateThenHistory([seen, fresh])
    client = FakeClient(conversations=conversations)
    run = make_run()
    run._seen_entry_ids = {"entry-1"}

    response = asyncio.run(run._append(client, inputs=[]))

    assert conversations.history_calls == 1
    assert [e.id for e in response.outputs] == ["entry-2"]
    assert run._read_classification(response.outputs)["category"] == "passport"


def test_an_unrelated_sdk_error_is_not_swallowed_as_a_duplicate():
    class AlwaysBadRequest:
        async def append_async(self, **kwargs):
            raise sdk_error("Something else went wrong")

    from mistralai.client.errors import SDKError

    run = make_run()
    with pytest.raises(SDKError):
        asyncio.run(
            run._append(FakeClient(conversations=AlwaysBadRequest()), inputs=[])
        )


def test_pump_never_answers_the_same_tool_call_twice(monkeypatch):
    """The API rejects a whole append if any result repeats a tool_call_id."""
    runs = []

    async def fake_inspect(client, file_id, filename):
        runs.append(file_id)
        return {"file_id": file_id, "filename": filename, "metrics": {}}

    monkeypatch.setitem(orch.TOOL_DISPATCH, INSPECT_TOOL_NAME, fake_inspect)

    duplicate = call(INSPECT_TOOL_NAME, {"file_id": "f", "filename": "p"}, "tc-dup")
    # The same call comes back a second time (e.g. recovered from history).
    conversations = FakeConversations(
        [[duplicate], [message("done", agent_id="ag-pre")]]
    )
    client = FakeClient(conversations=conversations)
    run = make_run()
    first = SimpleNamespace(conversation_id="conv-1", outputs=[duplicate])

    asyncio.run(run._pump(client, first))

    assert runs == ["f"], "the tool ran twice for one tool_call_id"
    assert run._answered_tool_calls == {"tc-dup"}
    # Only the first iteration appended anything.
    assert len(conversations.appends) == 1


def test_remember_records_entry_ids_for_later_recovery():
    run = make_run()
    first = message("a", agent_id="ag-pre")
    first.id = "e1"
    second = message("b", agent_id="ag-pre")
    second.id = "e2"

    returned = run._remember([first, second])

    assert [e.id for e in returned] == ["e1", "e2"]
    assert run._seen_entry_ids == {"e1", "e2"}


# ── Preprocessing decision ────────────────────────────────────────────────────


def test_preprocessing_decision_is_read_from_the_preprocessor_message():
    run = make_run()
    outputs = [
        message(
            '{"final_file_id": "file-2", "operations": ["deskew"], "rationale": "tilted"}',
            agent_id="ag-pre",
        )
    ]

    decision = run._read_preprocessing_decision(outputs)

    assert decision["file_id"] == "file-2"
    assert decision["operations"] == ["deskew"]
    assert decision["filename"] == "passport-enhanced.png"
    assert decision["content_type"] == "image/png"


def test_a_no_op_decision_keeps_the_original_file_and_filename():
    run = make_run()
    outputs = [
        message(
            '{"final_file_id": "file-1", "operations": [], "rationale": "clean"}',
            agent_id="ag-pre",
        )
    ]

    decision = run._read_preprocessing_decision(outputs)

    assert decision["file_id"] == "file-1"
    assert decision["filename"] == "passport.jpg"
    assert decision["content_type"] is None


def test_a_decision_naming_an_unknown_operation_is_rejected():
    run = make_run()
    outputs = [
        message(
            '{"final_file_id": "file-2", "operations": ["nope"], "rationale": "x"}',
            agent_id="ag-pre",
        )
    ]

    with pytest.raises(ValueError, match="Unknown preprocessing operations"):
        run._read_preprocessing_decision(outputs)


# ── The confidence gate ───────────────────────────────────────────────────────


def test_a_low_confidence_run_parks_for_review_without_extracting(monkeypatch):
    run = make_run(confidence_threshold=0.9)
    extracted: list[str] = []

    async def fake_phase_one(client):
        run.steps["preprocess"] = {"status": "done", "result": {"operations": []}}

    async def fake_phase_two(client):
        return {"category": "gtc", "confidence": 0.4, "explanation": "unsure"}

    async def fake_extract(client, category):
        extracted.append(category)

    monkeypatch.setattr(run, "_phase_one", fake_phase_one)
    monkeypatch.setattr(run, "_phase_two", fake_phase_two)
    monkeypatch.setattr(run, "_extract", fake_extract)
    monkeypatch.setattr(orch, "mistral_client", _fake_client_cm)
    monkeypatch.setattr(orch, "ensure_registry", _fake_ensure_registry)

    asyncio.run(run.start())

    assert extracted == []
    assert run.pending == "category"
    assert run.steps["classify"]["status"] == "waiting_human"
    assert run.extraction is None


def test_a_confident_run_extracts_without_review(monkeypatch):
    run = make_run(confidence_threshold=0.9)
    extracted: list[str] = []

    async def fake_phase_one(client):
        return None

    async def fake_phase_two(client):
        return {"category": "passport", "confidence": 0.97, "explanation": "clear"}

    async def fake_extract(client, category):
        extracted.append(category)
        run.extraction = {"common": {}, "specific": {}}

    monkeypatch.setattr(run, "_phase_one", fake_phase_one)
    monkeypatch.setattr(run, "_phase_two", fake_phase_two)
    monkeypatch.setattr(run, "_extract", fake_extract)
    monkeypatch.setattr(orch, "mistral_client", _fake_client_cm)
    monkeypatch.setattr(orch, "ensure_registry", _fake_ensure_registry)

    asyncio.run(run.start())

    assert extracted == ["passport"]
    assert run.pending is None
    assert run.steps["classify"]["status"] == "done"


def test_resume_with_category_overrides_the_classification_and_extracts(monkeypatch):
    run = make_run(
        pending="category",
        classification={"category": "gtc", "confidence": 0.4, "explanation": "unsure"},
    )
    extracted: list[str] = []

    async def fake_extract(client, category):
        extracted.append(category)

    monkeypatch.setattr(run, "_extract", fake_extract)
    monkeypatch.setattr(orch, "mistral_client", _fake_client_cm)

    asyncio.run(run.resume_with_category("passport"))

    assert extracted == ["passport"]
    assert run.classification["category"] == "passport"
    assert run.classification["confidence"] == 1.0
    assert run.classification["explanation"] == "Manually selected category: passport"
    assert run.pending is None
    assert run.steps["classify"]["status"] == "done"


def test_resume_is_refused_when_the_run_is_not_waiting():
    run = make_run(pending=None)

    with pytest.raises(AgentRunError, match="not waiting"):
        asyncio.run(run.resume_with_category("passport"))


# ── Extraction ────────────────────────────────────────────────────────────────


def test_extract_addresses_the_category_agent_directly_and_uses_the_mrz(monkeypatch):
    async def fake_signed_url(client, file_id):
        return f"https://example.test/{file_id}"

    monkeypatch.setattr(orch, "get_signed_url", fake_signed_url)

    extraction = {
        "common": {
            "full_name": None,
            "date_of_birth": None,
            "document_number": None,
            "issue_date": None,
            "expiry_date": None,
            "nationality": None,
            "address": None,
        },
        "specific": {
            "passport_number": None,
            "country_of_issue": None,
            "place_of_birth": None,
            "passport_type": None,
            "holder_signature": None,
            "mrz": {
                "raw_lines": [
                    "P<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<",
                    "L898902C36UTO7408122F1204159ZE184226B<<<<<10",
                ]
            },
        },
    }
    conversations = FakeConversations(
        [[message(json.dumps(extraction), agent_id="ag-ex-passport")]]
    )
    client = FakeClient(conversations=conversations)
    run = make_run()

    asyncio.run(run._extract(client, "passport"))

    # A fresh conversation on the passport extractor, not an append + handoff.
    assert conversations.appends == []
    started = conversations.starts[0]
    assert started["agent_id"] == "ag-ex-passport"
    chunks = started["inputs"][0]["content"]
    assert "passport" in chunks[0]["text"]
    assert chunks[1]["type"] == "image_url"

    assert run.steps["extract"]["status"] == "done"
    # The checksum-valid MRZ filled the fields the visual pass left empty.
    assert run.extraction["common"]["full_name"] == "ERIKSSON ANNA MARIA"
    assert run.extraction["common"]["document_number"] == "L898902C3"
    assert run.extraction["specific"]["mrz"]["checksum_valid"] is True


def test_a_truncated_extraction_is_retried_once_in_the_same_conversation(monkeypatch):
    """A repetition loop truncates the JSON; one corrective nudge clears it."""

    async def fake_signed_url(client, file_id):
        return f"https://example.test/{file_id}"

    monkeypatch.setattr(orch, "get_signed_url", fake_signed_url)

    good = {
        "common": {
            "full_name": "ERIKSSON ANNA MARIA",
            "date_of_birth": None,
            "document_number": "L898902C3",
            "issue_date": None,
            "expiry_date": None,
            "nationality": None,
            "address": None,
        },
        "specific": {
            "passport_number": "L898902C3",
            "country_of_issue": None,
            "place_of_birth": None,
            "passport_type": None,
            "holder_signature": None,
            "mrz": None,
        },
    }
    truncated = (
        '{"common": {"full_name": "ERIKSSON ANNA MARIA", "document_number": "L8989<<<<<'
    )
    conversations = FakeConversations(
        [
            [message(truncated, agent_id="ag-ex-passport")],
            [message(json.dumps(good), agent_id="ag-ex-passport")],
        ]
    )
    client = FakeClient(conversations=conversations)
    run = make_run()

    asyncio.run(run._extract(client, "passport"))

    # The retry is an append to the extraction conversation, not a new one.
    assert len(conversations.starts) == 1
    assert len(conversations.appends) == 1
    assert (
        "could not be used"
        in conversations.appends[0]["inputs"][0]["content"][0]["text"]
    )
    assert run.extraction["common"]["full_name"] == "ERIKSSON ANNA MARIA"
    assert run.steps["extract"]["status"] == "done"


def test_extraction_gives_up_after_the_retry(monkeypatch):
    async def fake_signed_url(client, file_id):
        return f"https://example.test/{file_id}"

    monkeypatch.setattr(orch, "get_signed_url", fake_signed_url)

    conversations = FakeConversations(
        [
            [message("{broken", agent_id="ag-ex-passport")],
            [message("{still broken", agent_id="ag-ex-passport")],
        ]
    )
    run = make_run()

    with pytest.raises(AgentRunError):
        asyncio.run(run._extract(FakeClient(conversations=conversations), "passport"))
    assert len(conversations.appends) == 1


def test_extractor_output_is_bounded_to_stop_a_repetition_loop():
    from agents.registry import EXTRACTION_MAX_TOKENS

    spec = extractor_spec("passport")

    assert spec.completion_args["max_tokens"] == EXTRACTION_MAX_TOKENS


def test_extract_raises_when_the_extractor_says_nothing(monkeypatch):
    async def fake_signed_url(client, file_id):
        return f"https://example.test/{file_id}"

    monkeypatch.setattr(orch, "get_signed_url", fake_signed_url)
    client = FakeClient(conversations=FakeConversations([[]]))
    run = make_run()

    with pytest.raises(AgentRunError, match="no extraction"):
        asyncio.run(run._extract(client, "passport"))


# ── Phase 1 wiring ────────────────────────────────────────────────────────────


def test_a_pdf_skips_preprocessing_and_carries_the_document_immediately(monkeypatch):
    async def fake_signed_url(client, file_id):
        return f"https://example.test/{file_id}"

    monkeypatch.setattr(orch, "get_signed_url", fake_signed_url)

    conversations = FakeConversations(
        [
            [
                message(
                    '{"category": "gtc", "confidence": 0.95, "explanation": "terms"}',
                    agent_id="ag-cls",
                )
            ]
        ]
    )
    client = FakeClient(conversations=conversations)
    run = make_run(filename="terms.pdf", content_type="application/pdf")

    asyncio.run(run._phase_one(client))

    chunks = conversations.starts[0]["inputs"][0]["content"]
    assert chunks[0]["text"].count("PDF") >= 1
    assert "Skip preprocessing" in chunks[0]["text"]
    assert chunks[1]["type"] == "document_url"
    assert run.preprocessing["status"] == "skipped"
    assert run.steps["ocr"]["status"] == "done"
    assert conversations.starts[0]["handoff_execution"] == "server"


def test_an_image_starts_without_the_document_so_the_agent_works_from_previews(
    monkeypatch,
):
    conversations = FakeConversations(
        [
            [
                message(
                    '{"final_file_id": "file-1", "operations": [], "rationale": "clean"}',
                    agent_id="ag-pre",
                )
            ]
        ]
    )
    client = FakeClient(conversations=conversations)
    run = make_run()

    asyncio.run(run._phase_one(client))

    chunks = conversations.starts[0]["inputs"][0]["content"]
    assert len(chunks) == 1, "the raw image should not be in the opening message"
    assert (
        PREPROCESSOR_NAME.split("-")[-1] in chunks[0]["text"]
        or "preprocessing" in chunks[0]["text"]
    )
    assert run.preprocessing["operations"] == []


def test_a_failed_preprocessing_decision_degrades_to_the_original_file():
    conversations = FakeConversations([[message("not json at all", agent_id="ag-pre")]])
    client = FakeClient(conversations=conversations)
    run = make_run()

    asyncio.run(run._phase_one(client))

    assert run.preprocessing["status"] == "skipped"
    assert run.preprocessing["file_id"] == "file-1"
    assert run.preprocessing["operations"] == []
    assert "error" in run.preprocessing
    assert run.steps["preprocess"]["status"] == "done"


# ── Retry ─────────────────────────────────────────────────────────────────────


class FakeSDKError(Exception):
    def __init__(self, status_code):
        super().__init__(f"status {status_code}")
        self.status_code = status_code


def test_rate_limits_and_timeouts_are_classified_as_retryable():
    import httpx
    from mistralai.client.errors import SDKError

    rate_limited = SDKError.__new__(SDKError)
    rate_limited.status_code = 429
    server_error = SDKError.__new__(SDKError)
    server_error.status_code = 503
    bad_request = SDKError.__new__(SDKError)
    bad_request.status_code = 400

    assert _is_retryable(rate_limited)
    assert _is_retryable(server_error)
    assert not _is_retryable(bad_request)
    assert _is_retryable(httpx.ReadTimeout("slow"))
    assert not _is_retryable(ValueError("nope"))


def test_call_with_retry_returns_the_result_once_the_call_succeeds(monkeypatch):
    import agents.files as files_module

    monkeypatch.setattr(files_module.asyncio, "sleep", _no_sleep)

    attempts = {"n": 0}
    from mistralai.client.errors import SDKError

    async def flaky():
        attempts["n"] += 1
        if attempts["n"] < 3:
            error = SDKError.__new__(SDKError)
            error.status_code = 429
            raise error
        return "done"

    assert asyncio.run(call_with_retry(flaky, "test")) == "done"
    assert attempts["n"] == 3


def test_call_with_retry_gives_up_after_the_budget(monkeypatch):
    import agents.files as files_module

    monkeypatch.setattr(files_module.asyncio, "sleep", _no_sleep)

    from mistralai.client.errors import SDKError

    attempts = {"n": 0}

    async def always_limited():
        attempts["n"] += 1
        error = SDKError.__new__(SDKError)
        error.status_code = 429
        raise error

    with pytest.raises(SDKError):
        asyncio.run(call_with_retry(always_limited, "test"))
    assert attempts["n"] == MAX_ATTEMPTS


def test_call_with_retry_does_not_retry_a_client_error(monkeypatch):
    attempts = {"n": 0}

    async def bad_request():
        attempts["n"] += 1
        raise ValueError("bad input")

    with pytest.raises(ValueError):
        asyncio.run(call_with_retry(bad_request, "test"))
    assert attempts["n"] == 1


# ── Shared helpers used by the fixtures above ─────────────────────────────────


async def _no_sleep(_seconds):
    return None


class _FakeClientCM:
    async def __aenter__(self):
        return FakeClient()

    async def __aexit__(self, *args):
        return False


def _fake_client_cm():
    return _FakeClientCM()


async def _fake_ensure_registry(client):
    return REGISTRY


# ── MRZ contamination ─────────────────────────────────────────────────────────


def test_mrz_text_in_an_ordinary_field_is_detected():
    """Real failure: the model looped MRZ filler into common.document_number."""
    from shared.personal_documents import find_unusable_fields

    extracted = {
        "common": {
            "full_name": "ERIKSSON ANNA MARIA",
            "document_number": "L898902C36UT07408122F1204159ZE184226B<<<<<10",
        },
        "specific": {"mrz": {"raw_lines": ["P<UTOERIKSSON<<ANNA<MARIA<<<<<<<<"]}},
    }

    assert find_unusable_fields(extracted, "passport") == ["common.document_number"]


def test_a_clean_extraction_reports_no_contamination():
    from shared.personal_documents import find_unusable_fields

    extracted = {
        "common": {"full_name": "ERIKSSON ANNA MARIA", "document_number": "L898902C3"},
        "specific": {"mrz": {"raw_lines": ["P<UTOERIKSSON<<ANNA<MARIA<<<<<<<<"]}},
    }

    assert find_unusable_fields(extracted, "gtc") == []


def test_an_absurdly_long_value_counts_as_contamination():
    from shared.personal_documents import (
        MAX_REASONABLE_FIELD_LENGTH,
        find_unusable_fields,
    )

    extracted = {"common": {"address": "A" * (MAX_REASONABLE_FIELD_LENGTH + 1)}}

    assert find_unusable_fields(extracted, "passport") == ["common.address"]


def test_strip_mrz_contamination_nulls_only_the_named_fields():
    from shared.personal_documents import strip_unusable_fields

    extracted = {
        "common": {"full_name": "ANNA", "document_number": "junk<<<<"},
        "specific": {"passport_number": "L898902C3"},
    }

    cleaned = strip_unusable_fields(extracted, ["common.document_number"])

    assert cleaned["common"]["document_number"] is None
    assert cleaned["common"]["full_name"] == "ANNA"
    assert cleaned["specific"]["passport_number"] == "L898902C3"
    # The input is left alone.
    assert extracted["common"]["document_number"] == "junk<<<<"


def _passport_payload(document_number):
    return {
        "common": {
            "full_name": "ERIKSSON ANNA MARIA",
            "date_of_birth": None,
            "document_number": document_number,
            "issue_date": None,
            "expiry_date": None,
            "nationality": None,
            "address": None,
        },
        "specific": {
            "passport_number": "L898902C3",
            "country_of_issue": None,
            "place_of_birth": None,
            "passport_type": None,
            "holder_signature": None,
            "mrz": None,
        },
    }


def test_a_contaminated_extraction_is_retried_then_accepted(monkeypatch):
    async def fake_signed_url(client, file_id):
        return f"https://example.test/{file_id}"

    monkeypatch.setattr(orch, "get_signed_url", fake_signed_url)

    conversations = FakeConversations(
        [
            [
                message(
                    json.dumps(_passport_payload("L898902C3<<<<<10<<<<<10")),
                    agent_id="ag-ex-passport",
                )
            ],
            [
                message(
                    json.dumps(_passport_payload("L898902C3")),
                    agent_id="ag-ex-passport",
                )
            ],
        ]
    )
    run = make_run()

    asyncio.run(run._extract(FakeClient(conversations=conversations), "passport"))

    assert len(conversations.appends) == 1, "the contaminated reply was not retried"
    assert run.extraction["common"]["document_number"] == "L898902C3"
    assert "warnings" not in run.extraction


def test_contamination_surviving_the_retry_is_stripped_and_reported(monkeypatch):
    async def fake_signed_url(client, file_id):
        return f"https://example.test/{file_id}"

    monkeypatch.setattr(orch, "get_signed_url", fake_signed_url)

    dirty = json.dumps(_passport_payload("L898902C3<<<<<10<<<<<10"))
    conversations = FakeConversations(
        [
            [message(dirty, agent_id="ag-ex-passport")],
            [message(dirty, agent_id="ag-ex-passport")],
        ]
    )
    run = make_run()

    asyncio.run(run._extract(FakeClient(conversations=conversations), "passport"))

    # Reported as nothing rather than as MRZ noise, and said out loud.
    assert run.extraction["common"]["document_number"] is None
    assert run.extraction["warnings"] == [
        "Discarded an unusable value the model placed in: common.document_number"
    ]
    assert run.extraction["common"]["full_name"] == "ERIKSSON ANNA MARIA"
    assert run.steps["extract"]["status"] == "done"


def test_the_extractor_prompt_says_where_mrz_text_may_go():
    from shared.personal_documents import EXTRACTOR_SYSTEM_PROMPT

    assert "raw_lines" in EXTRACTOR_SYSTEM_PROMPT
    assert "never repeat a sequence" in EXTRACTOR_SYSTEM_PROMPT.lower()


def test_mrz_transcription_is_demanded_only_for_categories_that_have_one():
    """An MRZ-less prompt was leaving specific.mrz null, which silently disabled
    enrich_with_mrz_fallback and lost every field it would have filled."""
    from shared.extraction_fields import PERSONAL_DOCUMENT_SPECIFIC_FIELDS
    from shared.personal_documents import extraction_prompt

    for category in PERSONAL_DOCUMENT_CATEGORIES:
        has_mrz = any(
            key == "mrz" for key, _ in PERSONAL_DOCUMENT_SPECIFIC_FIELDS[category]
        )
        prompt = extraction_prompt("doc", category)
        assert ("specific.mrz.raw_lines" in prompt) is has_mrz, category
        if has_mrz:
            assert "machine-readable zone is required" in prompt, category


# ── Contamination false positives ─────────────────────────────────────────────


def test_long_prose_is_not_mistaken_for_mrz_text():
    """A GTC acceptance_text is legitimately a long sentence. Flagging it discarded
    a good value and triggered a retry that nulled every other field too."""
    from shared.personal_documents import find_unusable_fields

    acceptance = (
        "By signing the supply agreement, the customer acknowledges having received "
        "and accepted these general terms and conditions."
    )
    assert len(acceptance) > 120, "the fixture must exercise the length rule"

    extracted = {"common": {}, "specific": {"acceptance_text": acceptance}}

    assert find_unusable_fields(extracted, "gtc") == []


def test_a_long_unspaced_value_is_still_treated_as_contamination():
    from shared.personal_documents import find_unusable_fields

    extracted = {"common": {"address": "AB1234" * 40}}

    assert find_unusable_fields(extracted, "passport") == ["common.address"]


def test_mrz_filler_is_caught_however_short_the_value():
    from shared.personal_documents import looks_like_mrz_text

    assert looks_like_mrz_text("L898902C3<<<<<10")
    assert not looks_like_mrz_text("L898902C3")
    assert not looks_like_mrz_text("Rue de la Loi 16, 1000 Brussels, Belgium")


def test_the_retry_nudge_mentions_the_mrz_only_where_there_is_one():
    """An MRZ-themed nudge sent to the GTC extractor made it null every field."""
    from agents.orchestrator import extraction_retry_nudge

    assert "machine-readable zone" in extraction_retry_nudge("passport")
    assert "machine-readable zone" in extraction_retry_nudge("id")
    for category in ("gtc", "proof_of_address", "other"):
        assert "machine-readable" not in extraction_retry_nudge(category), category
        assert "JSON" in extraction_retry_nudge(category), category


# ── Guidance / schema drift ───────────────────────────────────────────────────


def test_the_guidance_never_advertises_an_operation_the_agent_cannot_pick():
    """This drifted once: the prompt still offered table_grid_removal after the
    tool schema dropped it, so the agent burned turns on rejected calls."""
    from agents.tools import DESTRUCTIVE_OPERATIONS
    from shared.personal_documents import PREPROCESSING_GUIDANCE

    for operation in DESTRUCTIVE_OPERATIONS | UNAVAILABLE_OPERATIONS:
        assert f"- {operation}:" not in PREPROCESSING_GUIDANCE, operation
    for operation in AGENT_OPERATIONS:
        assert f"- {operation}:" in PREPROCESSING_GUIDANCE, operation


# ── Sibling fields swallowed into a string ────────────────────────────────────


def test_sibling_fields_swallowed_into_a_string_are_detected():
    """Seen live on a GTC document: document_title, issuer_name and key_clauses
    ended up inside acceptance_text, which parses fine and reads like prose."""
    from shared.personal_documents import find_unusable_fields

    corrupted = (
        "By signing the supply agreement, the customer acknowledges having received "
        "and accepted these general terms and conditions.', 'document_title\": "
        '"GENERAL TERMS AND CONDITIONS", "issuer_name": "Northwind Utilities N.V."'
    )
    assert isinstance(corrupted, str)
    extracted = {"common": {}, "specific": {"acceptance_text": corrupted}}

    assert find_unusable_fields(extracted, "gtc") == ["specific.acceptance_text"]


def test_ordinary_prose_mentioning_a_field_word_is_not_flagged():
    from shared.personal_documents import find_unusable_fields

    extracted = {
        "common": {"address": "Rue de la Loi 16, 1000 Brussels"},
        "specific": {
            "acceptance_text": "The issuer name and document title appear overleaf."
        },
    }

    assert find_unusable_fields(extracted, "gtc") == []


def test_extraction_field_names_cover_common_and_category_specific():
    from shared.personal_documents import extraction_field_names

    passport = extraction_field_names("passport")
    assert "full_name" in passport and "passport_number" in passport
    assert "acceptance_text" not in passport
    assert "acceptance_text" in extraction_field_names("gtc")


def test_each_extractor_gets_only_the_guidance_its_category_needs():
    """The GTC extractor buried under identity/MRZ instructions returned nulls for
    fields that were plainly on the page."""
    from shared.personal_documents import EXTRACTOR_BASE_PROMPT, extractor_system_prompt

    for category in ("passport", "id"):
        assert "MRZ" in extractor_system_prompt(category), category
    for category in ("gtc", "proof_of_address", "other"):
        assert "MRZ" not in extractor_system_prompt(category), category
        assert EXTRACTOR_BASE_PROMPT in extractor_system_prompt(category), category


def test_the_workflow_prompt_still_covers_every_category():
    """The workflow uses one model for all categories, so it keeps the MRZ half."""
    from shared.personal_documents import EXTRACTOR_SYSTEM_PROMPT

    assert "MRZ" in EXTRACTOR_SYSTEM_PROMPT
    assert "Return only valid JSON" in EXTRACTOR_SYSTEM_PROMPT


def test_extractor_instructions_name_the_category_specific_fields():
    spec = extractor_spec("gtc")

    assert "gtc" in spec.instructions
    for field in ("document_title", "issuer_name", "key_clauses"):
        assert field in spec.instructions, field
