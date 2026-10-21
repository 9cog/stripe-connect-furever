# Acme platform

Monorepo for the customer web app, its API, and the recommendations service.

| Path | What | Who works on it |
|---|---|---|
| `apps/web` | Customer web app (port 3000) | most of the team |
| `apps/api` | REST API (port 4100), Postgres | backend |
| `packages/shared` | Shared helpers | everyone, indirectly |
| `services/ml` | Python recommendations worker, Redis | ML team only |

## Getting started

### Web
1. Node 20 (`nvm use`), then `corepack enable`
2. `pnpm install` at the repo root
3. `cp apps/web/.env.local.example apps/web/.env.local`
4. `pnpm dev:web` → http://localhost:3000

The web app talks to the deployed staging API by default, so you don't need to run the API locally.

### API
1. Everything in "Web" step 1–2
2. `cp apps/api/.env.example apps/api/.env`
3. `docker compose up -d api-db`
4. `pnpm --filter @acme/api db:migrate`
5. `pnpm dev:api`

### ML service
1. Python 3.11+
2. `cd services/ml && python3 -m venv .venv && .venv/bin/pip install -r requirements.txt`
3. `cp services/ml/.env.example services/ml/.env`
4. `docker compose up -d ml-redis`
5. `make -C services/ml test`
