# One-time local setup on Windows (PowerShell). Requires: uv, Node 20+, Java 21+ (for the Firestore emulator).
$ErrorActionPreference = "Stop"
Push-Location "$PSScriptRoot\..\backend"
uv venv --python 3.11
uv sync
if (-not (Test-Path .env)) { Copy-Item .env.example .env; Write-Host "Created backend/.env - add your GEMINI_API_KEY." }
Pop-Location
npm --prefix "$PSScriptRoot\..\frontend" install
Write-Host "Done. Run: npm run emulators | npm run backend | npm run frontend (three terminals)."
