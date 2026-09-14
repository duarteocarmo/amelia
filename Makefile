default: help

.PHONY: help
help: # Show help for each of the Makefile recipes.
	@grep -E '^[a-zA-Z0-9 -]+:.*#'  Makefile | sort | while read -r l; do printf "\033[1;32m$$(echo $$l | cut -f 1 -d':')\033[00m:$$(echo $$l | cut -f 2- -d'#')\n"; done

.PHONY: install
install: # Install dependencies into the virtual environment
	uv sync

MODEL ?= nemotron-3-nano-4b
TASK ?= pt-exams
LIMIT ?=
BASE_URL ?= http://localhost:8000/v1

.PHONY: eval
eval: # Run a configured Modal evaluation (MODEL=... TASK=... LIMIT=...)
	uv run modal run -m amelia_evals.modal_runner --model '$(MODEL)' --task '$(TASK)' $(if $(LIMIT),--limit '$(LIMIT)')

.PHONY: eval-rig
eval-rig: # Evaluate an existing vLLM endpoint (MODEL=... TASK=... LIMIT=... BASE_URL=...)
	uv run python -m amelia_evals.rig_runner --model '$(MODEL)' --task '$(TASK)' --base-url '$(BASE_URL)' $(if $(LIMIT),--limit '$(LIMIT)')

.PHONY: view
view: # Open the Inspect log viewer (recursive, so per-task subfolders are shown)
	uv run inspect view --recursive

.PHONY: format
format: # Format Python code with Ruff
	uv run ruff format src

.PHONY: check
check: # Lint and type check Python code
	uv run ruff check src
	uv run ty check

.PHONY: test
test: # Run tests in parallel
	uv run pytest -n auto

.PHONY: clean
clean: # Clean up temporary files
	@rm -rf __pycache__ src/**/__pycache__ .pytest_cache .ruff_cache
