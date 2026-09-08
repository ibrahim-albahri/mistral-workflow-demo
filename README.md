# Personal document processor

A [Mistral Workflows](https://docs.mistral.ai/workflows/getting-started/introduction) demo for classifying and extracting information from personal documents.

The `personal_document_workflow` supports IDs, passports, proofs of address, general terms and conditions (GTC), and uncategorized documents. It accepts PDFs plus JPEG, PNG, and WebP images.

## Setup

Install the project dependencies:

```bash
make installdeps
```

Then copy `.env.example` to `.env` and set `MISTRAL_API_KEY`. You can optionally override the chat completion models via `MISTRAL_CLASSIFIER_MODEL` and `MISTRAL_EXTRACTOR_MODEL`.

## Run the workflow worker

Register the workflow with AI Studio and poll for executions:

```bash
make start-worker
```

The worker auto-discovers workflow classes in `src/workflows/`; this project registers `personal_document_workflow`.

## Run the UI

In a separate terminal, start the Streamlit interface:

```bash
make streamlit
```

Upload a supported document and click **Start Workflow**. The UI shows document preparation, classification, and extracted fields, including MRZ validation when available.

### Batch processing

The **Batch processing** section accepts up to 100 PDFs or supported images. It runs Mistral OCR for every document, submits the OCR-backed classification and extraction stages through the Mistral Batch API, and shows status and errors for each document. Set `MISTRAL_OCR_MODEL` to override the default `mistral-ocr-latest` OCR model.

Low-confidence classifications are presented for individual category review. Documents classified with sufficient confidence continue to extraction immediately; reviewed documents are submitted in subsequent extraction jobs. A batch completes with partial results when individual OCR, classification, or extraction requests fail, and the UI provides a one-row-per-document CSV export.

You can monitor execution progress and extracted data in [AI Studio](https://console.mistral.ai/build/workflows/).

## Mistral Agents implementation

`src/agents/` processes the same documents with the [Mistral Agents API](https://docs.mistral.ai/agents/agents_introduction/)
instead of Workflows. There is no worker and no Temporal: the agents live on the
API, and a run is a single stored conversation driven by an in-process orchestrator.

### The agent chain

Handoffs are one-way — a conversation stays on whichever agent it was handed to, and
control never returns to the supervisor — so each agent carries the handoff to its
own successor:

```
pdp-supervisor ──> pdp-preprocessor ──> pdp-classifier ──> pdp-extractor-<category>
               └──────────────────────>┘
```

`pdp-supervisor` routes images through preprocessing and sends PDFs straight to
classification. `pdp-preprocessor` owns the two OpenCV function tools
(`inspect_document_image`, `apply_preprocessing_operation`), which the orchestrator
executes locally and returns as `function.result` entries. There is one extractor
agent **per category**, because the API rejects `completion_args` on an agent-bound
conversation — a response schema has to be baked into the agent at creation time, and
the extraction schema differs per category.

### Running it

```bash
make agents-sync                        # create or refresh the pdp-* agents
make agents-run file=passport.jpg       # process one document
make agents-run file=passport.jpg threshold=1.0   # force manual category review
make agents-streamlit                   # the UI
make agents-delete                      # remove every pdp-* agent
```

Set `MISTRAL_SUPERVISOR_AGENT_MODEL` to override the supervisor's model; the
classifier, extractor, and preprocessing models come from the same variables the
workflow uses.

`uv run python src/agents/probe_handoff.py` checks the four API behaviours this design
relies on (server-side handoff, client-executed function tools, the `completion_args`
rejection, and directed mid-conversation handoff) against the live API.

### Differences from the workflow version

| Workflow | Agents |
| --- | --- |
| durable `InteractiveWorkflow` + worker | in-process orchestrator, no worker |
| `@workflows.activity` | plain function exposed as a `FunctionTool` |
| `get_steps` query polled by the UI | `steps` dict on the run object |
| `manual_category` signal + `wait_condition` | the run parks; the UI resumes its stored conversation |
| activity retry policies | explicit backoff in `agents/files.py` |

Both implementations share their schemas, prompts, and MRZ enrichment via
`shared/personal_documents.py`, and their Streamlit step panels via
`shared/streamlit_ui.py`, so their results are directly comparable.

### Guard rails

Temporal gives the workflow version budgets, retries and idempotency for free. Here
each one is written and tested by hand, in `agents/orchestrator.py` unless noted:

| Guard | Why it exists |
| --- | --- |
| `MAX_PREPROCESSING_OPERATIONS`, `MAX_TOOL_CALLS` | an agent that had spent its operations kept re-inspecting the image until the iteration cap |
| no-repeat operation check | a repeated operation also fails `validate_preprocessing_decision`, discarding all preprocessing |
| `_answered_tool_calls` | a retried append whose original succeeded re-sent a `function.result`, which the API rejects |
| `_recover_from_history` | that rejection proves the work landed, so the run reads back what it missed instead of failing |
| `call_with_retry` (`agents/files.py`) | 429/5xx/timeout backoff; the SDK's 30s default timeout is also raised to 300s |
| `find_unusable_fields` (`shared/`) | extractors have copied MRZ text into ordinary fields, and swallowed sibling fields into a string — both parse as valid JSON |
| one extraction retry, then strip + `warnings` | reports nothing rather than garbage, and names what it discarded |
| `DESTRUCTIVE_OPERATIONS` (`agents/tools.py`) | `table_grid_removal` inpainted a passport's MRZ away; the agent is offered only what `conservative_ocr_config` enables |

A run re-sends its whole conversation every turn, so image chains are token-hungry
and can exhaust an account's rate limit; previews shown to the agent are capped at
512px and sent once for the same reason.

### Known limitations

- **GTC extraction under-fills.** On a text PDF the extractor reliably returns
  `acceptance_text` but leaves `document_title`, `issuer_name`, `version_date` and
  `key_clauses` null. Whether the workflow's `chat.parse` path does better on the
  same document is **not yet measured** — the comparison was blocked by rate limits.
  Run `agents-run` and the workflow against one document before drawing conclusions.
- **MRZ transcription fidelity** limits the enrichment: if the model mis-transcribes
  the zone, `parse_mrz` rejects it and `enrich_with_mrz_fallback` correctly declines
  to fill fields from it, so those fields stay null rather than becoming wrong.
- The confidence gate is `confidence < threshold`, so a model returning exactly `1.0`
  is never held for review even at `threshold=1.0`. This matches the workflow's
  existing behaviour; use a threshold above 1.0 to force a review.

## Development

```bash
uv run ruff format .
uv run ruff check --fix .
uv run pytest
```

## Clean up

When finished, stop the Streamlit app and worker with `Ctrl+C`.
