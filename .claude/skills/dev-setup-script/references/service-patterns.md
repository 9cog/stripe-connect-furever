# Local service patterns

Drop-in bash blocks for starting the databases and queues a repo depends on. Every block follows the same ladder so the script behaves predictably:

1. **Already reachable** on its port → do nothing (the contributor runs it natively, or a previous run started it).
2. **Docker available** → reuse a running container, `docker start` a stopped one, or `docker run` a new one with a *named container* and a *named volume* so data survives.
3. **Homebrew / apt available** → install and start the native service.
4. **Nothing available** → don't fail; print how to point the env var at an external instance.

Use the image tag the repo itself declares (Dockerfile, compose file, CI config). Only fall back to a sensible default tag if the repo declares nothing. Name containers and volumes after the repo (`<repo>-postgres`, `<repo>-postgres-data`) so several projects can coexist on one machine.

All blocks assume the template's helpers (`step`, `ok`, `warn`, `fail`, `info`, `has`, `port_open`) are defined.

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
elif has docker; then
  start_docker_service "$PG_CONTAINER" "$PG_IMAGE" "$PG_PORT" 5432 "${PG_CONTAINER}-data" /var/lib/postgresql/data \
    -e POSTGRES_USER="$PG_USER" -e POSTGRES_PASSWORD="$PG_PASSWORD" -e POSTGRES_DB="$PG_DB"
  # Wait for readiness before any migrations run.
  for _ in $(seq 1 30); do
    docker exec "$PG_CONTAINER" pg_isready -U "$PG_USER" >/dev/null 2>&1 && break
    sleep 1
  done
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
elif has docker; then
  start_docker_service "$MYSQL_CONTAINER" "$MYSQL_IMAGE" "$MYSQL_PORT" 3306 "${MYSQL_CONTAINER}-data" /var/lib/mysql \
    -e MYSQL_ROOT_PASSWORD=root -e MYSQL_DATABASE="${PROJECT}_dev"
  for _ in $(seq 1 30); do
    docker exec "$MYSQL_CONTAINER" mysqladmin ping -proot --silent >/dev/null 2>&1 && break
    sleep 1
  done
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
elif has docker; then
  start_docker_service "$MONGO_CONTAINER" "$MONGO_IMAGE" "$MONGO_PORT" 27017 "${MONGO_CONTAINER}-data" /data/db
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
elif has docker; then
  start_docker_service "$REDIS_CONTAINER" "$REDIS_IMAGE" "$REDIS_PORT" 6379 "${REDIS_CONTAINER}-data" /data
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
elif has docker; then
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
elif has docker && docker compose version >/dev/null 2>&1; then
  step "Starting services with docker compose"
  docker compose up -d --quiet-pull
  ok "Services up (stop with: docker compose down)"
else
  warn "docker compose not available — start the services in docker-compose.yml manually or point the env vars at external instances."
fi
```

## Migrations / seeds

If the README's getting-started runs migrations (`npx prisma migrate dev`, `rails db:setup`, `python manage.py migrate`, `alembic upgrade head`), run them *after* the DB readiness wait and *only* when the DB phase wasn't skipped. Guard with the same `SKIP_*` flag.

## Testing services locally

When verifying the script, a full image pull can take minutes and a lot of disk — run with `--no-<service>` to exercise every other phase, and say in the final report that the service phase was not executed on this machine. If the image is already cached (`docker images | grep <image>`), run it for real.
