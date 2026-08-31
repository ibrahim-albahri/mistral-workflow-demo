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

You can monitor execution progress and extracted data in [AI Studio](https://console.mistral.ai/build/workflows/).

## Development

```bash
uv run ruff format .
uv run ruff check --fix .
uv run pytest
```

## Clean up

When finished, stop the Streamlit app and worker with `Ctrl+C`.
