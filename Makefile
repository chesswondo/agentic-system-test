.PHONY: install lint test test-live run batch run-live batch-live

install:
	uv sync

lint:
	uv run ruff check --fix . && uv run ruff format .

test:
	uv run pytest -q

# Live LLM tests and runs need GOOGLE_API_KEY and cost money.
test-live:
	uv run pytest -q -m live

run:
	uv run picos run --scenario scenarios/default.yaml --seed 42 --llm fake

batch:
	uv run picos batch --scenario scenarios/default.yaml --n 5 --llm fake

run-live:
	uv run picos run --scenario scenarios/default.yaml --seed 42 --llm google_genai

batch-live:
	uv run picos batch --scenario scenarios/default.yaml --n 5 --llm google_genai
