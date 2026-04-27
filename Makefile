UV := uv

CONTAINER_RUNTIME := $(shell command -v podman 2>/dev/null || command -v docker 2>/dev/null)
IMAGE := lightspeed-agentic-sandbox:latest

.PHONY: install install-all lock test lint format mypy verify eval eval-report image clean help

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

install: ## Install package with dev dependencies via uv
	$(UV) sync --extra dev

install-all: ## Install all provider, dev, and eval dependencies via uv
	$(UV) sync --all-extras

lock: ## Refresh uv.lock from pyproject.toml
	$(UV) lock

test: ## Run unit tests
	$(UV) run pytest tests/ -v

lint: ## Run ruff linter
	$(UV) run ruff check .

format: ## Auto-format with ruff
	$(UV) run ruff format .
	$(UV) run ruff check . --fix

mypy: ## Run mypy against application package
	$(UV) run mypy src/lightspeed_agentic

verify: ## Run non-mutating formatting, lint, and type checks
	$(UV) run ruff format . --check
	$(UV) run ruff check .
	$(UV) run mypy src/lightspeed_agentic

image: ## Build production container image
	$(CONTAINER_RUNTIME) build -t $(IMAGE) .

EVAL_ARGS ?=

eval: image ## Run evals against live containers (use EVAL_ARGS to filter, e.g. EVAL_ARGS="-k claude")
	PYTEST="$(UV) run pytest" bash evals/run.sh $(EVAL_ARGS)

eval-report: image ## Run evals and generate JSON report
	PYTEST="$(UV) run pytest" bash evals/run.sh --eval-report=evals/report.json $(EVAL_ARGS)

clean: ## Remove build artifacts and caches
	rm -rf dist/ build/ *.egg-info .venv .pytest_cache .mypy_cache .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
