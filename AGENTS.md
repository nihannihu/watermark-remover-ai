# AGENTS.md

This repository is a Windows-oriented local AI watermark-removal app.

## Project context
- Primary goal: remove watermarks from images using local/offline model assets when available.
- Main runtime assumptions:
  - Windows setup via `setup.bat`
  - Python-based environment under `python\`
  - Local model downloads to user cache and Hugging Face snapshot storage
- Avoid broad rewrites unless the request explicitly requires them.

## Repository rules
- Preserve the existing app structure and entrypoints.
- Prefer minimal, safe fixes over large refactors.
- Keep dependency changes justified and Windows-compatible.
- Do not change model behavior casually.
- Validate with targeted tests or smoke checks when code changes affect setup, UI, or model loading.

## Working expectations
- Inspect before editing.
- Search for related references before changing shared logic.
- Keep changes focused and reversible.
- Report what was validated and what remains unverified.