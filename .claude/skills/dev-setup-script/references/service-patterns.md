# Local service patterns

Drop-in bash blocks for starting the databases and queues a repo depends on. Every block follows the same ladder so the script behaves predictably:

1. **Already reachable** on its port → do nothing (the contributor runs it natively, or a previous run started it).
2. **Docker daemon answers** (`docker_ready`, i.e. `docker info` succeeds — not just `command -v docker`) → reuse a running container, `docker start` a stopped one, or `docker run` a new one with a *named container* and a *named volume* so data survives.
3. **Docker installed but the daemon is down** → say exactly that ("Docker is installed but not running — start Docker Desktop, or re-run with --no-<service>") before trying anything else, so the user isn't left guessing why the next step failed.
4. **Homebrew or a user-level binary** (`redis-server`, `pg_ctl`) → start the native service. A user-level daemon should keep its data inside the repo (`./.redis/`), be gitignored, and be stoppable with a printed command.
5. **Nothing available** → don't fail; print how to point the env var at an external instance.

Use the image tag the repo itself declares (Dockerfile, compose file, CI config). Only fall back to a sensible default tag if the repo declares nothing. Name containers and volumes after the repo (`<repo>-postgres`, `<repo>-postgres-data`) so several projects can coexist on one machine.

All blocks assume the template's helpers (`step`, `ok`, `warn`, `fail`, `info`, `has`, `port_open`, `docker_ready`, `wait_for_port`) are defined.

---

## Generic ladder (adapt per service)

```bash
start_docker_service() {  # start_docker_service <container> <image> <host_port> <container_port> <volume> <data_path> [extra docker run args...]
  local name="$1" image="$2" hport="$3" cport="$4" vol="$5" data="$6"; shift 6
  if docker ps --format '{{.Names}}' | grep -qx "$name"; then
    ok "Container '$name' already running"
  elif docker ps -a --format '{{.Names}}' | grep -qx "$name"; then
    docker start "$name" >/dev/null
    ok "Started existing container '$name'"
  else
    info "Pulling $image and starting container..."
    docker run -d --name "$name" -p "${hport}:${cport}" -v "${vol}:${data}" "$@" "$image" >/dev/null
    ok "'$name' running on port $hport"
  fi
  info "Stop it later with: docker stop $name"
}
```

---

## PostgreSQL

```bash
PG_CONTAINER="${PROJECT}-postgres"; PG_IMAGE="postgres:16"; PG_PORT=5432
PG_USER="postgres"; PG_PASSWORD="postgres"; PG_DB="${PROJECT}_dev"

if port_open "$PG_PORT"; then
  ok "PostgreSQL already reachable on localhost:$PG_PORT"
elif docker_ready; then
  start_docker_service "$PG_CONTAINER" "$PG_IMAGE" "$PG_PORT" 5432 "${PG_CONTAINER}-data" /var/lib/postgresql/data \
    -e POSTGRES_USER="$PG_USER" -e POSTGRES_PASSWORD="$PG_PASSWORD" -e POSTGRES_DB="$PG_DB"
  # Wait for readiness before any migrations run.
  for _ in $(seq 1 30); do
    docker exec "$PG_CONTAINER" pg_isready -U "$PG_USER" >/dev/null 2>&1 && break
    sleep 1
  done
elif has docker; then
  warn "Docker is installed but not running — start Docker Desktop (or the docker service), or re-run with --no-<service>."
elif has brew; then
  brew install postgresql@16 && brew services start postgresql@16
  createdb "$PG_DB" 2>/dev/null || true
  ok "PostgreSQL started via Homebrew"
else
  fail "Neither Docker nor Homebrew found — install one, or set DATABASE_URL in .env to an external PostgreSQL."
fi
```

Match `PG_USER`/`PG_PASSWORD`/`PG_DB` to whatever `DATABASE_URL` in `.env.example` expects, so the copied `.env` works with zero edits.

## MySQL / MariaDB

```bash
MYSQL_CONTAINER="${PROJECT}-mysql"; MYSQL_IMAGE="mysql:8"; MYSQL_PORT=3306

if port_open "$MYSQL_PORT"; then
  ok "MySQL already reachable on localhost:$MYSQL_PORT"
elif docker_ready; then
  start_docker_service "$MYSQL_CONTAINER" "$MYSQL_IMAGE" "$MYSQL_PORT" 3306 "${MYSQL_CONTAINER}-data" /var/lib/mysql \
    -e MYSQL_ROOT_PASSWORD=root -e MYSQL_DATABASE="${PROJECT}_dev"
  for _ in $(seq 1 30); do
    docker exec "$MYSQL_CONTAINER" mysqladmin ping -proot --silent >/dev/null 2>&1 && break
    sleep 1
  done
elif has docker; then
  warn "Docker is installed but not running — start Docker Desktop (or the docker service), or re-run with --no-<service>."
elif has brew; then
  brew install mysql && brew services start mysql
  ok "MySQL started via Homebrew"
else
  fail "Neither Docker nor Homebrew found — install one, or point DATABASE_URL at an external MySQL."
fi
```

## MongoDB

```bash
MONGO_CONTAINER="${PROJECT}-mongo"; MONGO_IMAGE="mongo:7"; MONGO_PORT=27017   # use the repo's Dockerfile tag if it has one

if port_open "$MONGO_PORT"; then
  ok "MongoDB already reachable on localhost:$MONGO_PORT"
elif docker_ready; then
  start_docker_service "$MONGO_CONTAINER" "$MONGO_IMAGE" "$MONGO_PORT" 27017 "${MONGO_CONTAINER}-data" /data/db
elif has docker; then
  warn "Docker is installed but not running — start Docker Desktop (or the docker service), or re-run with --no-<service>."
elif has brew; then
  brew tap mongodb/brew >/dev/null 2>&1 || true
  brew install mongodb-community && brew services start mongodb-community
  ok "MongoDB started via Homebrew"
else
  fail "Neither Docker nor Homebrew found — install one, or point MONGO_URI at an external MongoDB."
fi
```

## Redis

```bash
REDIS_CONTAINER="${PROJECT}-redis"; REDIS_IMAGE="redis:7"; REDIS_PORT=6379

if port_open "$REDIS_PORT"; then
  ok "Redis already reachable on localhost:$REDIS_PORT"
elif docker_ready; then
  start_docker_service "$REDIS_CONTAINER" "$REDIS_IMAGE" "$REDIS_PORT" 6379 "${REDIS_CONTAINER}-data" /data
elif has docker; then
  warn "Docker is installed but not running — start Docker Desktop (or the docker service), or re-run with --no-<service>."
elif has brew; then
  brew install redis && brew services start redis
  ok "Redis started via Homebrew"
else
  fail "Neither Docker nor Homebrew found — install one, or point REDIS_URL at an external Redis."
fi
```

## RabbitMQ

```bash
MQ_CONTAINER="${PROJECT}-rabbitmq"; MQ_IMAGE="rabbitmq:3-management"; MQ_PORT=5672

if port_open "$MQ_PORT"; then
  ok "RabbitMQ already reachable on localhost:$MQ_PORT"
elif docker_ready; then
  start_docker_service "$MQ_CONTAINER" "$MQ_IMAGE" "$MQ_PORT" 5672 "${MQ_CONTAINER}-data" /var/lib/rabbitmq -p 15672:15672
  info "Management UI: http://localhost:15672 (guest/guest)"
else
  fail "Docker not found — install it, or point RABBITMQ_URL at an external broker."
fi
```

## The repo has a `docker-compose.yml`

Don't re-implement it. One phase:

```bash
if [ "$SKIP_DOCKER" = true ]; then
  step "Skipping docker compose (--no-docker)"
elif docker_ready && docker compose version >/dev/null 2>&1; then
  step "Starting services with docker compose"
  docker compose up -d --quiet-pull
  ok "Services up (stop with: docker compose down)"
elif has docker; then
  warn "Docker is installed but not running — start it, or re-run with --no-docker and point the env vars at your own services."
else
  warn "docker compose not available — start the services in docker-compose.yml manually or point the env vars at external instances."
fi
```

## Migrations / seeds

If the README's getting-started runs migrations, run them *after* the DB readiness wait and *only* when the DB is actually reachable. Guard with the same `SKIP_*` flag.

Use the non-interactive form — a setup script must never stop at a prompt:

| Tool | In the setup script | Not |
|---|---|---|
| Prisma | `prisma migrate deploy` (+ `prisma generate`) | `prisma migrate dev`, or a `db:migrate` script wrapping it — it can prompt for a migration name or a reset |
| Rails | `bin/rails db:prepare` | `db:setup` on an existing DB |
| Django | `python manage.py migrate --noinput` | `migrate` without `--noinput` |
| Alembic | `alembic upgrade head` | |

Seeds: run them only when the database was just created (e.g. the migrate step found no prior migrations) or behind an explicit `--seed` flag, unless you've read the seed and it's idempotent (upserts). Say in the report which command developers should still use day to day (e.g. `pnpm db:migrate` when they change the schema).

## Testing services locally

If the Docker daemon is running and the image is cached (`docker images | grep <image>`), run the service phase for real. Otherwise don't start `dockerd`, change registries, or pull from mirrors to make it run — test the branching with a stub `docker` on `PATH` instead:

```bash
mkdir -p "$SCRATCH/fakebin"
cat > "$SCRATCH/fakebin/docker" <<'STUB'
#!/usr/bin/env bash
echo "docker $*" >> "${FAKE_DOCKER_LOG:-/dev/null}"
case "$1" in
  info)    exit "${FAKE_DOCKER_INFO_RC:-0}" ;;       # set to 1 to simulate "daemon down"
  ps)      printf '%s\n' ${FAKE_DOCKER_PS:-} ;;       # container names to report
  compose) [ "$2" = version ] && exit 0; exit "${FAKE_COMPOSE_RC:-0}" ;;
  *)       exit 0 ;;
esac
STUB
chmod +x "$SCRATCH/fakebin/docker"

# daemon down: expect the "installed but not running" message, not a compose error
PATH="$SCRATCH/fakebin:$PATH" FAKE_DOCKER_INFO_RC=1 ./setup.sh
# daemon up: expect the right compose/run arguments in the log
FAKE_DOCKER_LOG="$SCRATCH/docker.log" PATH="$SCRATCH/fakebin:$PATH" ./setup.sh && cat "$SCRATCH/docker.log"
```

A stub can't prove the service really starts, so readiness waits will time out against it — give the script a way to shorten the wait (an env var) or accept the timeout message as the expected outcome. Report the service phase as "tested against a stub", not "tested".
