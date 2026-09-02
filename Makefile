default: help

.PHONY: help
help: # Show help for each of the Makefile recipes.
	@grep -E '^[a-zA-Z0-9 -]+:.*#'  Makefile | sort | while read -r l; do printf "\033[1;32m$$(echo $$l | cut -f 1 -d':')\033[00m:$$(echo $$l | cut -f 2- -d'#')\n"; done

.PHONY: install
install: # Install dependencies into the virtual environment
	uv sync

.PHONY: list
list: # List available inspect tasks
	uv run inspect list tasks pt_exams.py

MODEL ?= openai/gpt-4.1-mini
CONFIG ?= default
LIMIT ?= 10

.PHONY: eval
eval: # Run the pt_exams evaluation (MODEL=... CONFIG=... LIMIT=...)
	LLAMA_CPP_BASE_URL=http://localhost:8080/v1 \
  LLAMA_CPP_API_KEY=local \
  uv run inspect eval pt_exams.py \
    --model 'openai-api/llama-cpp/LiquidAI/LFM2.5-2.6B-GGUF:Q4_K_M' \
    --max-connections 2 \
		--limit 5


.PHONY: view
view: # Open the Inspect log viewer
	uv run inspect view

.PHONY: format
format: # Format Python code with Ruff
	uv run ruff format pt_exams.py

.PHONY: lint
lint: # Lint Python code with Ruff
	uv run ruff check pt_exams.py

.PHONY: clean
clean: # Clean up temporary files
	@rm -rf __pycache__ .pytest_cache .ruff_cache logs
