# Acme Orders API

Fastify + Prisma service that owns the orders table.

## Getting started

1. Install Node 20 (we use `nvm use`).
2. `corepack enable && pnpm install`
3. `cp .env.example .env`
4. `docker compose up -d db`
5. `pnpm db:migrate` then `pnpm db:seed`
6. `pnpm dev` → http://localhost:4000/healthz

## Tests

`pnpm test`
