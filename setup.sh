#!/usr/bin/env bash
# One-shot local setup for macOS / Linux.
# Run from the repository root:
#   chmod +x setup.sh && ./setup.sh

set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
APP="$ROOT/hvac-cost-estimator"

echo "==> App directory: $APP"
cd "$APP"

if [[ ! -d .venv ]]; then
  echo "==> Creating Python venv..."
  python3 -m venv .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate
echo "==> Installing Python requirements..."
pip install -r requirements.txt

if [[ ! -f .env ]]; then
  echo "==> Copying .env.example -> .env"
  cp .env.example .env
else
  echo "==> .env already exists (left unchanged)"
fi

echo "==> Installing frontend npm packages..."
cd frontend
npm install

echo ""
echo "Setup complete."
echo "Start API:  cd hvac-cost-estimator/backend && source ../.venv/bin/activate && uvicorn main:app --reload"
echo "Start UI:   cd hvac-cost-estimator/frontend && npm run dev"
echo "Dashboard:  http://localhost:5173"
echo "API docs:   http://localhost:8000/docs"
