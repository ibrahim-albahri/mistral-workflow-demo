# OCR-driven document processing for healthcare

A [Mistral Workflows](https://docs.mistral.ai/workflows/getting-started/introduction) project.

This project shows how to use Mistral Workflows to process documents in healthcare.

## Setup

Run the following command to install the required project dependencies:

```bash
make installdeps
```

Then copy `.env.example` to `.env` and set your `MISTRAL_API_KEY`.
You can optionally override the chat completion models via
`MISTRAL_CLASSIFIER_MODEL` and `MISTRAL_EXTRACTOR_MODEL`.

## Commands

### Register workflows in AI Studio

Use the following command to auto-discoves all workflow classes in `src/workflows/`, register them with AI Studio, and starts polling for executions. The task deployent name is set to your hostname:

```bash
make start-worker
```

### Execute the workflow

In a separate terminal, start the Streamlit app to view the document extraction UI:

```bash
make streamlit
```

Upload one of the provided PDFs and click "Start Workflow".

You can monitor your workflow's progress and view the extracted data in [AI Studio](https://console.mistral.ai/build/workflows/).

### Load test

The following command runs a load test that triggers the workflow 10 times using the provided input documents:

```bash
make load-test
```

You can also provide a custom amount of times to trigger the workflow:

```bash
make load-test n=5
```

## Development

```bash
# Format
uv run ruff format .

# Lint
uv run ruff check --fix .
```

## Clean up

When you're done, stop the Streamlit app and the worker using `Ctrl+C`.
