.PHONY: start-worker execute installdeps streamlit agents-run agents-sync agents-delete agents-streamlit

## Install dependencies
installdeps:
	uv sync

## Auto-discover all workflows and start the worker (with file-watch auto-reload)
start-worker:
	uv run python src/dev_worker.py

## Trigger a workflow execution
## Usage: make execute workflow=hello-world input='{"name": "World"}'
execute:
	uv run python src/workflows/start.py $(if $(workflow),--workflow $(workflow),) $(if $(input),--input '$(input)',)

## Start the Streamlit app
streamlit:
	PYTHONPATH=src uv run streamlit run src/entrypoints/app.py

## ── Mistral Agents implementation (no worker needed) ────────────────────────

## Create or refresh the pdp-* agents
## Usage: make agents-sync
agents-sync:
	PYTHONPATH=src uv run python src/agents/run_cli.py --sync-agents

## Run one document through the agent chain
## Usage: make agents-run file=passport.jpg [threshold=0.9]
agents-run:
	PYTHONPATH=src uv run python src/agents/run_cli.py --file '$(file)' $(if $(threshold),--confidence-threshold $(threshold),)

## Delete every pdp-* agent
agents-delete:
	PYTHONPATH=src uv run python src/agents/run_cli.py --delete-agents

## Start the Streamlit app for the agents implementation
agents-streamlit:
	PYTHONPATH=src uv run streamlit run src/entrypoints/agents_app.py
