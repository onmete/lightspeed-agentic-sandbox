VENV     := .venv
PYTHON   := $(VENV)/bin/python
PIP      := $(VENV)/bin/pip
PYTEST   := $(VENV)/bin/pytest

CONTAINER_RUNTIME := $(shell command -v podman 2>/dev/null || command -v docker 2>/dev/null)
IMAGE := lightspeed-agentic-sandbox:latest

.PHONY: venv install install-all install-eval test lint format eval eval-report image eval-container clean help

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

venv: ## Create virtual environment
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip

install: venv ## Install package in editable mode with dev deps
	$(PIP) install -e ".[dev]"

install-all: venv ## Install with all provider SDKs + dev + eval deps
	$(PIP) install -e ".[all,dev,eval]"

install-eval: venv ## Install with eval deps only (no provider SDKs)
	$(PIP) install -e ".[eval]"

test: ## Run unit tests
	$(PYTEST) tests/ -v

lint: ## Run ruff linter
	$(VENV)/bin/ruff check src/ tests/ evals/

format: ## Auto-format with ruff
	$(VENV)/bin/ruff format src/ tests/ evals/
	$(VENV)/bin/ruff check --fix src/ tests/ evals/

eval: ## Run evals (real API calls — skips providers without credentials)
	$(PYTEST) evals/ -v

eval-report: ## Run evals and generate JSON report
	$(PYTEST) evals/ -v --eval-report=evals/report.json

image: ## Build production container image
	$(CONTAINER_RUNTIME) build -t $(IMAGE) .

eval-container: image ## Run evals inside production container
	$(CONTAINER_RUNTIME) run --rm \
		-v $(CURDIR)/evals:/app/evals:Z \
		-v $(CURDIR)/evals/workspace/skills:/app/skills/skills:Z \
		-v $(CURDIR)/evals/workspace/tools:/tmp/agent-workspace/tools:Z \
		-e ANTHROPIC_API_KEY \
		-e CLAUDE_CODE_USE_VERTEX \
		-e ANTHROPIC_VERTEX_PROJECT_ID \
		-e CLOUD_ML_REGION \
		-e GOOGLE_APPLICATION_CREDENTIALS \
		-e GOOGLE_API_KEY \
		-e GEMINI_API_KEY \
		-e OPENAI_API_KEY \
		-e OPENAI_BASE_URL \
		-e AWS_ACCESS_KEY_ID \
		-e AWS_SECRET_ACCESS_KEY \
		-e AWS_REGION \
		$(IMAGE) \
		python -m pytest evals/ -v --eval-report=evals/report.json

clean: ## Remove build artifacts and caches
	rm -rf dist/ build/ *.egg-info .pytest_cache .mypy_cache .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
