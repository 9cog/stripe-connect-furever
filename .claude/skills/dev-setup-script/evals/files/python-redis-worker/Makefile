.PHONY: test lint worker

test:
	.venv/bin/pytest -q

lint:
	.venv/bin/ruff check .

worker:
	.venv/bin/rq worker thumbs
