# pricing-service

Quote engine plus the CSS for the internal quoting page.

## Setup

1. Install Python (version in `.python-version`) and Node (see `.tool-versions`).
2. `python -m venv .venv && .venv/bin/pip install -e '.[dev]'`
3. `npm run build:css`
4. `make test`
