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
- **Services match the project's own declarations.** Prefer Docker with the same image tag the `Dockerfile`/compose file uses, a named container, and a named volume so data persists. If the repo has a compose file, call `docker compose up -d <service>` rather than re-implementing it with `docker run`. Check if the service is already reachable before starting anything (someone may run it natively). Fall back to Homebrew / a user-level binary only when Docker isn't usable; if nothing works, print how to point the env var at an external instance instead of failing. `references/service-patterns.md` has drop-in blocks for common databases and queues.
- **"Docker is usable" means the daemon answers, not that the binary exists.** `command -v docker` is true on plenty of machines where the daemon is stopped (Docker Desktop not launched, a CI image with only the CLI). Gate every Docker path on `docker info >/dev/null 2>&1` (the template's `docker_ready`), and when the binary exists but the daemon doesn't answer, say so explicitly — "Docker is installed but not running; start Docker Desktop or use --no-<service>" — rather than falling through to a confusing compose error.
- **Non-interactive all the way through.** A setup script runs unattended, so every command must be one that never stops to prompt: `prisma migrate deploy` not `prisma migrate dev`, `rails db:prepare` not interactive generators, `apt`/`pip` with their quiet/yes flags. If the repo's own script (`pnpm db:migrate`, a Makefile target) wraps an interactive command, call the non-interactive equivalent directly and mention in the report which command developers should still use when they change the schema. Only seed when the database was freshly created (or behind an explicit `--seed` flag) unless you can see the seed is idempotent — re-running an insert-based seed creates duplicates.
- **Skip flags for the slow or optional parts** (`--no-<service>`, `--no-python`). Contributors who run the DB elsewhere, or CI, need to opt out.
- **`--help` is a heredoc**, not `grep '^#'` over the file — grepping comments leaks every internal `# ---` separator into the help text.
- **End with next steps** the script could not do for them: fill in secrets, the dev command, webhook forwarders, the URL to open.
- **Quiet by default.** Use `--silent`/`--quiet` on installers so the phase markers stay readable; the user can drop the flag if they need to debug.

Don't add phases the repo doesn't call for. A Go service with no env file needs a `go mod download` and a build check, not a Python venv.

## 3. Fix the small things the setup exposes

Writing and running a setup script is the first time anyone has bootstrapped the repo from scratch in a while, so it tends to surface small breakages — a missing `go.sum`, a stale lockfile, a `.env` holding secrets that isn't gitignored, a README that still lists the manual steps. These are part of the job: a setup script that "works" but leaves a new contributor with a dirty `git status` or a committed secret hasn't solved their problem. Apply fixes that are small, mechanical and clearly correct, and list each one in the report so the user can review it:

- **Missing or stale lockfile / checksum file** (`go.sum`, `pnpm-lock.yaml` not matching `package.json`, etc.): regenerate it with the project's own tool (`go mod tidy`, `pnpm install --lockfile-only`) and keep the result in the repo. Don't delete it after testing — the whole point is that it should be committed. Check that the manifest itself (`go.mod`, `package.json`) is unchanged.
- **`.env` (or another file that will hold secrets) not in `.gitignore`**: add it. Same for directories the script creates (`.venv/`, `out/`, `.redis/`).
- **README**: add a short "Quick start: `./setup.sh`" line at the top of the getting-started section. Keep the manual steps below it for people who want to understand them.
- **Directories the app expects** (an output dir named in `.env.example`): have the script create them.

Anything bigger — changing application code, restructuring the Makefile, fixing an app bug you noticed — goes in the report as a suggestion, not a change.

**Version pins are a team decision, not a small fix.** When the version files disagree (`.python-version` vs `requires-python`, `.tool-versions` / `.nvmrc` vs `engines`, either vs CI), make the *script* resolve the conflict — pick an interpreter that satisfies the strictest constraint and warn about the mismatch — but leave every pin file, `requires-python`, `engines` and CI config exactly as you found them. Pins often encode a constraint you can't see (a native extension that isn't built for the newer version yet, a deploy target), and changing them shifts every developer's toolchain at once. In the report, lay out the conflict as a small table (file → version) and recommend which way to align them, so the team can make the change deliberately.

## 4. Prove it works

An untested setup script is worse than a README, because people trust it. Do all of these before reporting done:

1. `bash -n setup.sh` for syntax, then `./setup.sh --help` and `./setup.sh --bogus-flag` to check the argument handling paths.
2. **Run it for real** from a clean state (`rm -f .env` first if you created one earlier). Use the skip flags to avoid anything genuinely slow (a multi-GB image pull) but exercise everything else — dependency install, `.env` creation, virtualenv.
3. **Run it again** immediately. Confirm the second run reports "already exists / reusing" rather than redoing or clobbering work. Append a marker line to `.env` between runs to prove it was preserved.
4. **Clean up** every artifact the test runs created that the user didn't ask for (`.env`, `.venv`, `node_modules`, `bin/`, containers, background processes you started) and confirm `git status` shows only the script and the deliberate fixes from step 3. Keep the fixes — a regenerated `go.sum` is a deliverable, not a test artifact.

**Test within the machine as you find it.** Don't start system daemons (`dockerd`, `systemctl start ...`), install system packages, or re-point registries to get a path to run — that tests a machine the user doesn't have, and changes theirs without asking. Running a service as your own user process (`redis-server --daemonize yes --dir ./.redis`) is fine if you stop it afterwards. For a path you can't run for real — Docker when the daemon is down, Homebrew on Linux — exercise the script's branching with a stub: put a small fake `docker` on `PATH` that echoes its arguments and returns the exit codes you want to test, and confirm the script calls the right commands and handles failure. Then say plainly in the report which paths ran for real, which ran against a stub, and which didn't run at all.

## Output format

Final reply, in this order:
1. What the script does (phases + flags) and the one-line usage.
2. The repo fixes you made in step 3, each with a one-line reason.
3. What was verified for real, what against a stub, and what not at all.
4. Anything you noticed but deliberately didn't change.

## Example

**Repo signals**: `.node-version` = 22.16.0, `yarn.lock`, `.env.example` with `MONGO_URI` and `STRIPE_SECRET_KEY`, `Dockerfile` = `FROM mongo:4.2`, `requirements.txt` used by `scripts/setup-accounts.py`, README says to `brew install mongodb-community`.

**Resulting phases**: Node major-version check against 22 → `yarn install --silent` → copy `.env.example` → Mongo: already reachable? else if `docker info` succeeds, `docker run -d --name <repo>-mongo -v <repo>-mongo-data:/data/db mongo:4.2` (reusing the container on re-runs); else if the binary exists but the daemon is down, say so; else `brew` fallback → `.venv` + `pip install -r requirements.txt` → next steps: add Stripe keys, `yarn dev`, `stripe listen --forward-to localhost:3000/api/webhooks`. Flags: `--no-mongo`, `--no-python`.

**Fixes alongside**: `.venv/` added to `.gitignore`; README "Getting started" gets a `./setup.sh` quick-start line.

**Verification on a machine without a running Docker daemon**: Node, yarn, `.env`, and venv phases ran for real twice; the Mongo phase ran against a stub `docker` covering "daemon down", "container exists but stopped", and "fresh run"; the Homebrew path didn't run (Linux).
