---
name: dev-setup-script
description: Write and verify an idempotent one-shot `setup.sh` that bootstraps a repository's local dev environment (toolchain checks, dependency install, .env from example, local services via Docker, language virtualenvs, next-steps summary). Use this whenever the user asks for a setup, bootstrap, install, onboarding, or "getting started" script, wants "one command to get running", wants new contributors to be able to set up quickly, or complains that the README setup steps are manual and error-prone — even if they don't say "script". Also use it when asked to fix or extend an existing setup/bootstrap script.
---

# Dev Setup Script

Produce a `setup.sh` a new contributor can run once on a fresh machine and again any time later without breaking anything. The script exists to replace the "Getting started" section of the README with something executable, so it has to be *correct for this repo* — not a generic template — and it has to be proven to work before you hand it over.

## 1. Inventory the repo before writing anything

The script's contents are dictated entirely by what the repo already declares. Read these signals first (see `references/toolchain-signals.md` for what each one implies):

- **Version pins**: `.node-version`, `.nvmrc`, `.python-version`, `.ruby-version`, `.tool-versions`, `engines` in `package.json`, `requires-python` in `pyproject.toml`.
- **Dependency manifests + lockfiles**: `package.json` + which lockfile exists (`yarn.lock` → yarn, `pnpm-lock.yaml` → pnpm, `package-lock.json` → npm, `bun.lockb` → bun); `requirements*.txt`, `pyproject.toml`, `Pipfile`, `poetry.lock`, `uv.lock`; `Gemfile`; `go.mod`; `Cargo.toml`.
- **Environment**: `.env.example` / `.env.sample` / `.env.template` — the variables tell you which services and secrets the app needs (a `MONGO_URI`, `DATABASE_URL`, `REDIS_URL`, `STRIPE_SECRET_KEY` each imply a setup step or a next-step reminder).
- **Services**: `Dockerfile`, `docker-compose.yml`, `render.yaml`, `Procfile`, `fly.toml` — these tell you the *exact* image/version the project expects (e.g. a `FROM mongo:4.2` means start `mongo:4.2`, not `latest`).
- **Existing automation**: `Makefile`, `scripts/`, `bin/setup`, `justfile` — extend or call these rather than duplicating them.
- **README "Getting started"**: the manual steps you are automating. Every step there should map to a phase in the script or a line in the final "next steps" summary.
- **`.gitignore`**: confirms which generated artifacts (`.env`, `.venv`, `node_modules`) are safe to create and that your script's outputs won't get committed.

Then check what's actually on the machine (`command -v node yarn python3 docker mongod psql redis-server ...`). This tells you which fallback paths matter *today* and what you can realistically test.

If the repo has no manifests at all, or its stack is unclear, say so and ask rather than guessing a toolchain.

## 2. Write the script

Start from `assets/setup.sh.template` — it already has the strict-mode header, colored `step/ok/warn/fail` helpers, flag parsing with a heredoc `--help`, and the closing "next steps" block. Fill in the phases; delete the ones that don't apply. Save as `setup.sh` at the repo root (or wherever the repo's existing convention puts scripts) and `chmod +x` it.

Principles that make the script trustworthy:

- **Idempotent.** Running it twice is the normal case (people re-run after pulling). Never overwrite an existing `.env`; reuse an existing virtualenv, container, or volume; `docker start` a stopped container instead of creating a new one.
- **Honest about versions.** Compare the installed major version against the pin and *warn* on mismatch rather than fail — most projects work on adjacent versions and a hard fail on day one is worse than a warning. Fail only when the tool is entirely missing.
- **Services match the project's own declarations.** Prefer Docker with the same image tag the `Dockerfile`/compose file uses, a named container, and a named volume so data persists. Check if the service is already reachable before starting anything (someone may run it natively). Fall back to Homebrew / apt only when Docker is absent; if neither exists, print how to point the env var at an external instance instead of failing. `references/service-patterns.md` has drop-in blocks for common databases and queues.
- **Skip flags for the slow or optional parts** (`--no-<service>`, `--no-python`). Contributors who run the DB elsewhere, or CI, need to opt out.
- **`--help` is a heredoc**, not `grep '^#'` over the file — grepping comments leaks every internal `# ---` separator into the help text.
- **End with next steps** the script could not do for them: fill in secrets, the dev command, webhook forwarders, the URL to open.
- **Quiet by default.** Use `--silent`/`--quiet` on installers so the phase markers stay readable; the user can drop the flag if they need to debug.

Don't add phases the repo doesn't call for. A Go service with no env file needs a `go mod download` and a build check, not a Python venv.

## 3. Prove it works

An untested setup script is worse than a README, because people trust it. Do all of these before reporting done:

1. `bash -n setup.sh` for syntax, then `./setup.sh --help` and `./setup.sh --bogus-flag` to check the argument handling paths.
2. **Run it for real** from a clean state (`rm -f .env` first if you created one earlier). Use the skip flags to avoid anything genuinely slow (a multi-GB image pull) but exercise everything else — dependency install, `.env` creation, virtualenv.
3. **Run it again** immediately. Confirm the second run reports "already exists / reusing" rather than redoing or clobbering work. Append a marker line to `.env` between runs to prove it was preserved.
4. **Clean up** every artifact the test runs created that the user didn't ask for (`.env`, `.venv`, `node_modules`, containers you started) and confirm `git status` shows only the script. Generated files should already be gitignored — if they aren't, flag it.

Report what you ran and what you skipped (and why) so the user knows exactly how much of the script has been exercised on this machine.

## Output format

Final reply: a short summary of what the script does (phases + flags), what was verified and what wasn't, and the one-line usage. Offer to add a "Quick start: `./setup.sh`" line to the README if the README still lists the manual steps.

## Example

**Repo signals**: `.node-version` = 22.16.0, `yarn.lock`, `.env.example` with `MONGO_URI` and `STRIPE_SECRET_KEY`, `Dockerfile` = `FROM mongo:4.2`, `requirements.txt` used by `scripts/setup-accounts.py`, README says to `brew install mongodb-community`.

**Resulting phases**: Node major-version check against 22 → `yarn install --silent` → copy `.env.example` → Mongo via `docker run -d --name <repo>-mongo -v <repo>-mongo-data:/data/db mongo:4.2` with reachability check and `brew` fallback → `.venv` + `pip install -r requirements.txt` → next steps: add Stripe keys, `yarn dev`, `stripe listen --forward-to localhost:3000/api/webhooks`. Flags: `--no-mongo`, `--no-python`.
