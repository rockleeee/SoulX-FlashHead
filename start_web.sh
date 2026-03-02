#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

if ! command -v python >/dev/null 2>&1; then
  echo "[ERROR] python not found (require Python 3.10+)"
  exit 1
fi

if ! command -v node >/dev/null 2>&1; then
  echo "[ERROR] node not found (require Node.js 18+)"
  exit 1
fi

echo "========================================"
echo "FlashHead Web startup script (Linux)"
echo "========================================"
echo

echo "[1/3] Starting backend on :8000 ..."
python backend/main.py &
BACKEND_PID=$!

cleanup() {
  if kill -0 "$BACKEND_PID" >/dev/null 2>&1; then
    kill "$BACKEND_PID" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT INT TERM

sleep 3

echo "[2/3] Starting frontend dev server on :3000 ..."
npm --prefix frontend run dev
