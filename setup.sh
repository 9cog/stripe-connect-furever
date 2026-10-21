#!/usr/bin/env bash
#
# setup.sh — one-shot dev environment bootstrap for stripe-connect-furever
#
# What it does:
#   1. Verifies Node/Yarn versions and installs JS dependencies
#   2. Creates .env from .env.example (never overwrites an existing one)
#   3. Starts a local MongoDB instance (Docker, matching the project's Dockerfile)
#   4. Optionally sets up a Python virtualenv + requirements.txt
#      (used by scripts/setup-accounts.py and the platform/ integration code)
#
# Usage:
#   ./setup.sh              # full setup
#   ./setup.sh --no-mongo   # skip starting Mongo (e.g. you run it elsewhere)
#   ./setup.sh --no-python  # skip Python venv setup
#   ./setup.sh --help

set -euo pipefail

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

MONGO_CONTAINER_NAME="furever-mongo"
MONGO_IMAGE="mongo:4.2"
MONGO_PORT="27017"
VENV_DIR=".venv"

SKIP_MONGO=false
SKIP_PYTHON=false

print_help() {
  cat <<'EOF'
setup.sh — one-shot dev environment bootstrap for stripe-connect-furever

What it does:
  1. Verifies Node/Yarn versions and installs JS dependencies
  2. Creates .env from .env.example (never overwrites an existing one)
  3. Starts a local MongoDB instance (Docker, matching the project's Dockerfile)
  4. Optionally sets up a Python virtualenv + requirements.txt
     (used by scripts/setup-accounts.py and the platform/ integration code)

Usage:
  ./setup.sh              # full setup
  ./setup.sh --no-mongo   # skip starting Mongo (e.g. you run it elsewhere)
  ./setup.sh --no-python  # skip Python venv setup
  ./setup.sh --help
EOF
}

for arg in "$@"; do
  case "$arg" in
    --no-mongo) SKIP_MONGO=true ;;
    --no-python) SKIP_PYTHON=true ;;
    --help|-h)
      print_help
      exit 0
      ;;
    *)
      echo "Unknown option: $arg (use --help)" >&2
      exit 1
      ;;
  esac
done

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
c_reset=$'\033[0m'; c_bold=$'\033[1m'; c_green=$'\033[32m'; c_yellow=$'\033[33m'; c_red=$'\033[31m'; c_blue=$'\033[34m'

step() { echo "${c_bold}${c_blue}==>${c_reset} ${c_bold}$*${c_reset}"; }
info() { echo "    $*"; }
ok()   { echo "    ${c_green}✓${c_reset} $*"; }
warn() { echo "    ${c_yellow}!${c_reset} $*"; }
fail() { echo "    ${c_red}✗${c_reset} $*" >&2; }

has() { command -v "$1" >/dev/null 2>&1; }

# ---------------------------------------------------------------------------
# 1. Node / package manager
# ---------------------------------------------------------------------------
step "Checking Node.js toolchain"

if ! has node; then
  fail "node not found. Install Node $(cat .node-version 2>/dev/null || echo '22.x') first (https://nodejs.org)."
  exit 1
fi

NODE_VERSION="$(node -v | sed 's/^v//')"
REQUIRED_NODE="$(cat .node-version 2>/dev/null || echo '22.0.0')"
NODE_MAJOR="${NODE_VERSION%%.*}"
REQUIRED_MAJOR="${REQUIRED_NODE%%.*}"

if [ "$NODE_MAJOR" != "$REQUIRED_MAJOR" ]; then
  warn "Node v$NODE_VERSION is installed but .node-version wants $REQUIRED_NODE.x — continuing anyway."
else
  ok "Node v$NODE_VERSION"
fi

if has yarn; then
  PKG_MANAGER="yarn"
elif has npm; then
  warn "yarn not found, falling back to npm"
  PKG_MANAGER="npm"
else
  fail "Neither yarn nor npm found."
  exit 1
fi

step "Installing JavaScript dependencies (${PKG_MANAGER})"
if [ "$PKG_MANAGER" = "yarn" ]; then
  yarn install --silent
else
  npm install --silent
fi
ok "Dependencies installed"

# ---------------------------------------------------------------------------
# 2. Environment file
# ---------------------------------------------------------------------------
step "Setting up .env"

if [ -f .env ]; then
  ok ".env already exists, leaving it untouched"
else
  if [ ! -f .env.example ]; then
    fail ".env.example is missing, cannot bootstrap .env"
    exit 1
  fi
  cp .env.example .env
  ok "Created .env from .env.example"
  warn "Fill in STRIPE_SECRET_KEY / STRIPE_PUBLIC_KEY / STRIPE_WEBHOOK_SECRET"
  info "Get test keys from https://dashboard.stripe.com/test/apikeys"
fi

# ---------------------------------------------------------------------------
# 3. MongoDB
# ---------------------------------------------------------------------------
if [ "$SKIP_MONGO" = true ]; then
  step "Skipping MongoDB setup (--no-mongo)"
else
  step "Setting up MongoDB"

  mongo_reachable() {
    if has docker; then
      docker run --rm --network host mongo:4.2 mongo --quiet --eval 'db.runCommand({ping:1})' \
        "mongodb://127.0.0.1:${MONGO_PORT}" >/dev/null 2>&1 && return 0
    fi
    (exec 3<>"/dev/tcp/127.0.0.1/${MONGO_PORT}") >/dev/null 2>&1 && exec 3>&- 3<&-
  }

  if mongo_reachable; then
    ok "MongoDB already reachable on localhost:${MONGO_PORT}"
  elif has docker; then
    if docker ps --format '{{.Names}}' | grep -qx "$MONGO_CONTAINER_NAME"; then
      ok "MongoDB container '${MONGO_CONTAINER_NAME}' already running"
    elif docker ps -a --format '{{.Names}}' | grep -qx "$MONGO_CONTAINER_NAME"; then
      docker start "$MONGO_CONTAINER_NAME" >/dev/null
      ok "Started existing MongoDB container '${MONGO_CONTAINER_NAME}'"
    else
      info "Pulling ${MONGO_IMAGE} and starting container (matches project Dockerfile)..."
      docker run -d \
        --name "$MONGO_CONTAINER_NAME" \
        -p "${MONGO_PORT}:27017" \
        -v furever-mongo-data:/data/db \
        "$MONGO_IMAGE" >/dev/null
      ok "MongoDB running in container '${MONGO_CONTAINER_NAME}' on port ${MONGO_PORT}"
    fi
    info "Stop it later with: docker stop ${MONGO_CONTAINER_NAME}"
  elif has brew; then
    warn "Docker not found; falling back to Homebrew MongoDB"
    brew tap mongodb/brew >/dev/null 2>&1 || true
    brew install mongodb-community@7.0
    brew services start mongodb-community@7.0
    ok "MongoDB started via Homebrew"
  else
    fail "Neither Docker nor Homebrew found — install one of them to run MongoDB."
    info "Or point MONGO_URI in .env at an external MongoDB instance."
  fi
fi

# ---------------------------------------------------------------------------
# 4. Python environment (platform/ integration + scripts/setup-accounts.py)
# ---------------------------------------------------------------------------
if [ "$SKIP_PYTHON" = true ]; then
  step "Skipping Python setup (--no-python)"
else
  step "Setting up Python environment"

  if ! has python3; then
    warn "python3 not found — skipping (needed for scripts/setup-accounts.py and platform/)"
  else
    if [ ! -d "$VENV_DIR" ]; then
      python3 -m venv "$VENV_DIR"
      ok "Created virtualenv at ${VENV_DIR}"
    else
      ok "Reusing existing virtualenv at ${VENV_DIR}"
    fi

    # shellcheck disable=SC1091
    source "${VENV_DIR}/bin/activate"
    pip install --quiet --upgrade pip
    pip install --quiet -r requirements.txt
    deactivate
    ok "Python dependencies installed (activate with: source ${VENV_DIR}/bin/activate)"
  fi
fi

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
echo
step "Setup complete"
info "Next steps:"
info "  1. Add your Stripe test keys to .env"
info "  2. Run the app:        ${PKG_MANAGER} dev"
info "  3. Forward webhooks:   stripe listen --forward-to localhost:3000/api/webhooks"
info "  4. Visit:              http://localhost:3000"
echo
