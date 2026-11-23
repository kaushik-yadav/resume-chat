#!/usr/bin/env bash
# One-time local setup on macOS/Linux. Requires: uv, Node 20+, Java 21+ (for the Firestore emulator).
set -euo pipefail
cd "$(dirname "$0")/../backend"
uv venv --python 3.11
uv sync
[ -f .env ] || { cp .env.example .env; echo "Created backend/.env - add your GEMINI_API_KEY."; }
cd ../frontend && npm install
echo "Done. Run: npm run emulators | npm run backend | npm run frontend (three terminals)."
