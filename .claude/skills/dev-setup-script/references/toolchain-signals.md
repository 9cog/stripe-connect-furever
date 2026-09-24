# Toolchain signals → setup phases

What each file in a repo tells you, and what the corresponding phase of `setup.sh` should do. Read the ones that match the repo; ignore the rest.

## Version pins

| File | Meaning | Phase behaviour |
|---|---|---|
| `.node-version`, `.nvmrc` | Exact Node version | Compare major of `node -v`; warn on mismatch. Mention `nvm use` / `fnm use` in the warning if either is installed. |
| `engines.node` in `package.json` | Range like `22.x` | Same as above, using the leading major. |
| `.python-version` | pyenv pin | Compare major.minor of `python3 --version`; warn. |
| `requires-python` in `pyproject.toml` | Range | Warn if installed version is outside it (a simple major.minor check is enough). |
| `.ruby-version` | rbenv/rvm pin | Compare `ruby -v`; warn. |
| `.tool-versions` | asdf, may list several tools | If `asdf` is present, `asdf install` handles everything — run it and skip per-tool checks. Otherwise check each listed tool. |
| `go.mod` `go 1.xx` line | Minimum Go | Compare `go version`; warn if lower. |
| `rust-toolchain(.toml)` | rustup pin | `rustup show` installs it on demand; just confirm `cargo` exists. |

## Dependency managers

| Signal | Use | Quiet install |
|---|---|---|
| `yarn.lock` | yarn | `yarn install --silent` |
| `pnpm-lock.yaml` | pnpm | `pnpm install --silent` |
| `bun.lockb` / `bun.lock` | bun | `bun install` |
| `package-lock.json` only | npm | `npm install --silent` (or `npm ci` when the lockfile is authoritative) |
| Several lockfiles present | Whichever the README/CI uses; warn about the extras (they drift) | |
| `requirements.txt` | pip in a venv | `python3 -m venv .venv && .venv/bin/pip install -q -r requirements.txt` |
| `pyproject.toml` + `poetry.lock` | poetry | `poetry install --quiet` |
| `pyproject.toml` + `uv.lock` | uv | `uv sync` |
| `Pipfile` | pipenv | `pipenv install --dev` |
| `Gemfile` | bundler | `bundle install --quiet` |
| `go.mod` | go modules | `go mod download` then `go build ./...` as a smoke test |
| `Cargo.toml` | cargo | `cargo fetch` (or `cargo build` if a compile check is wanted) |
| `composer.json` | composer | `composer install --quiet` |

When a repo mixes ecosystems (e.g. a Node app with a Python `scripts/` folder), give the secondary one its own `--no-<lang>` skip flag — most contributors only need the primary.

## Environment files

`.env.example`, `.env.sample`, `.env.template`, `.env.dist` — copy to `.env` only if `.env` is absent. Read the variable names to decide:

- `DATABASE_URL`, `POSTGRES_*`, `MONGO_URI`, `MYSQL_*`, `REDIS_URL`, `RABBITMQ_URL`, `ELASTICSEARCH_URL` → a service phase (see `service-patterns.md`).
- `*_SECRET`, `*_KEY`, `*_TOKEN`, `NEXTAUTH_SECRET` → a "fill these in" warning after copying, and a line in next steps. Don't try to generate secrets unless the README says how.
- `PORT`, `*_URL` pointing at localhost → the URL to print in next steps.

If there is no example file but the README lists env vars, generate `.env` from the README's list with placeholder values and say so.

## Services

| File | What it tells you |
|---|---|
| `Dockerfile` with `FROM <db>:<tag>` and nothing else | The project ships its DB as a container — use that exact image tag. |
| `docker-compose.yml` | Just `docker compose up -d <service>` — don't re-implement what compose already does. Add `--no-docker` to skip it. |
| `render.yaml`, `fly.toml`, `Procfile` | Names the services the app needs in production; the same ones are needed locally. |
| README `brew install ...` / `apt install ...` lines | The native fallback path when Docker is absent. |

## Existing automation

- `Makefile` with `setup`/`install`/`bootstrap` targets → the script should call `make <target>` for that step, not duplicate it. Or ask whether the user wants the script to *become* that target.
- `scripts/setup*`, `bin/setup`, `script/bootstrap` → extend the existing one rather than adding a second entry point.
- `justfile`, `Taskfile.yml` → same idea; call the existing recipe.

## `.gitignore`

Check that everything the script generates is ignored: `.env`, `.venv`/`venv`, `node_modules`, `vendor/`, `target/`. If something isn't, add it or warn — a setup script that leaves untracked junk in `git status` will get its outputs committed by accident.
