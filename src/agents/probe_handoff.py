"""Probe the Agents API behaviours the orchestrator depends on.

Run with: ``uv run python src/agents/probe_handoff.py``

It creates throwaway ``probe-*`` agents, exercises them, prints the raw entry types
the API returns, and deletes them again (including any left behind by an earlier
crashed run). Four assumptions are checked:

  (a) a supervisor hands off server-side and the specialist runs in the same response;
  (b) a function tool declared on a handed-off agent surfaces as a ``function.call``
      entry that can be satisfied with ``conversations.append``;
  (c) ``completion_args`` is REJECTED on an agent-bound conversation, so a response
      schema has to be baked into the agent at creation time;
  (d) after a handoff the conversation stays on the handed-off agent, so a directed
      mid-conversation handoff only works when THAT agent holds the target in its
      own ``handoffs``; the target's baked ``response_format`` then governs its reply.
"""

import asyncio
import json
import os
import sys

from dotenv import load_dotenv
from mistralai.client import Mistral
from mistralai.extra.utils import response_format_from_pydantic_model
from pydantic import BaseModel

load_dotenv(override=True)

MODEL = os.environ.get("MISTRAL_CLASSIFIER_MODEL", "mistral-medium-latest")
PROBE_PREFIX = "probe-"


class Verdict(BaseModel):
    verdict: str
    score: int


def describe(outputs) -> str:
    parts = []
    for entry in outputs:
        entry_type = getattr(entry, "type", "?")
        if entry_type == "agent.handoff":
            parts.append(
                f"agent.handoff({entry.previous_agent_name}->{entry.next_agent_name})"
            )
        elif entry_type == "function.call":
            parts.append(f"function.call({entry.name}, id={entry.tool_call_id})")
        elif entry_type == "message.output":
            parts.append(f"message.output({str(entry.content)[:70]!r})")
        else:
            parts.append(str(entry_type))
    return " | ".join(parts) or "<empty>"


def message_text(outputs) -> str:
    return "".join(
        str(entry.content)
        for entry in outputs
        if getattr(entry, "type", None) == "message.output"
    )


async def delete_probe_agents(client: Mistral) -> None:
    """Remove every probe-* agent, including leftovers from a crashed run."""
    agents = await client.beta.agents.list_async(page_size=100)
    for agent in agents:
        if not agent.name.startswith(PROBE_PREFIX):
            continue
        try:
            await client.beta.agents.delete_async(agent_id=agent.id)
            print(f"cleanup: deleted {agent.name} ({agent.id})")
        except Exception as exc:  # noqa: BLE001 - cleanup is best effort
            print(f"cleanup: could not delete {agent.id}: {exc}")


async def run_probes(client: Mistral) -> list[str]:
    failures: list[str] = []

    # (d) target: a peer whose response schema is baked in at creation time.
    formatter = await client.beta.agents.create_async(
        model=MODEL,
        name="probe-formatter",
        description="Scores a statement and answers as strict JSON.",
        instructions="Score the statement out of 10. Answer only with the JSON schema.",
        completion_args={
            "temperature": 0,
            "response_format": response_format_from_pydantic_model(Verdict),
        },
    )

    specialist = await client.beta.agents.create_async(
        model=MODEL,
        name="probe-specialist",
        description="Reports the weather for a city.",
        instructions=(
            "You report weather. Always call get_weather with the city before "
            "answering, then state the temperature in one sentence. When the user "
            "names an agent to hand off to, hand off to that agent immediately."
        ),
        handoffs=[formatter.id],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Return the current temperature for a city.",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                        "additionalProperties": False,
                    },
                },
            }
        ],
    )

    supervisor = await client.beta.agents.create_async(
        model=MODEL,
        name="probe-supervisor",
        description="Routes questions to specialists.",
        instructions=(
            "You route questions. Any weather question must be handed off to "
            "probe-specialist. Never answer yourself."
        ),
        handoffs=[specialist.id],
    )

    # (a) + (b): handoff, then a client-executed function call.
    print("\n--- (a)/(b) supervisor handoff + function tool ---")
    response = await client.beta.conversations.start_async(
        agent_id=supervisor.id,
        handoff_execution="server",
        store=True,
        inputs="What is the weather in Paris?",
    )
    conversation_id = response.conversation_id
    print(f"start -> {describe(response.outputs)}")

    if any(getattr(e, "type", None) == "agent.handoff" for e in response.outputs):
        print("  (a) OK: server-side handoff happened inside one response")
    else:
        failures.append("(a) no agent.handoff entry returned")

    calls = [e for e in response.outputs if getattr(e, "type", None) == "function.call"]
    if not calls:
        failures.append("(b) no function.call entry from the handed-off agent")
    else:
        print(f"  (b) function.call received: {calls[0].name}({calls[0].arguments})")
        follow_up = await client.beta.conversations.append_async(
            conversation_id=conversation_id,
            handoff_execution="server",
            inputs=[
                {
                    "type": "function.result",
                    "object": "entry",
                    "tool_call_id": calls[0].tool_call_id,
                    "result": json.dumps({"temperature_c": 17}),
                }
            ],
        )
        print(f"append(function.result) -> {describe(follow_up.outputs)}")
        if message_text(follow_up.outputs):
            print("  (b) OK: function result accepted, agent produced a message")
        else:
            failures.append("(b) function.result did not yield a message.output")

    # (c): completion_args must be rejected on an agent-bound conversation.
    print("\n--- (c) completion_args on an agent conversation ---")
    try:
        await client.beta.conversations.start_async(
            agent_id=formatter.id,
            store=False,
            inputs="Score this: 'Paris is warm today.'",
            completion_args={"temperature": 0},
        )
        failures.append("(c) completion_args was accepted; the design can be simpler")
    except Exception as exc:  # noqa: BLE001 - the rejection is the expected result
        detail = str(exc).replace("\n", " ")[:120]
        print(f"  (c) OK: rejected as expected -> {detail}")

    # (d): directed mid-conversation handoff to an agent with a baked schema.
    print("\n--- (d) directed handoff + baked response_format ---")
    directed = await client.beta.conversations.append_async(
        conversation_id=conversation_id,
        handoff_execution="server",
        inputs=(
            "Hand off to probe-formatter now and have it score this statement out "
            "of 10: 'Paris is warm today.'"
        ),
    )
    print(f"append(directive) -> {describe(directed.outputs)}")
    handoffs = [
        e for e in directed.outputs if getattr(e, "type", None) == "agent.handoff"
    ]
    if handoffs and handoffs[-1].next_agent_name == "probe-formatter":
        print("  (d) OK: directed handoff to probe-formatter")
    else:
        failures.append("(d) directed handoff to the named agent did not happen")
    try:
        Verdict.model_validate_json(message_text(directed.outputs))
        print("  (d) OK: reply matched the target agent's baked schema")
    except Exception as exc:  # noqa: BLE001 - schema conformance is the assertion
        failures.append(f"(d) reply did not match the baked schema: {exc}")

    return failures


async def main() -> int:
    client = Mistral(
        api_key=os.environ["MISTRAL_API_KEY"],
        server_url=os.environ.get("SERVER_URL", "https://api.mistral.ai"),
    )
    failures: list[str] = []

    async with client:
        await delete_probe_agents(client)
        try:
            failures = await run_probes(client)
        finally:
            await delete_probe_agents(client)

    print("\n--- summary ---")
    if failures:
        for failure in failures:
            print(f"FAILED {failure}")
        return 1
    print("All probed behaviours hold.")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
