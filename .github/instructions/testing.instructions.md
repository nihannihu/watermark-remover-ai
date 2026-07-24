---
applyTo: "**/*.{py,bat,md,json,yml,yaml}"
---

# Testing instructions

- Add or update tests when behavior changes.
- Prefer targeted tests for bug fixes and regression coverage.
- Validate setup changes with smoke checks where automated tests are not available.
- Never claim a test or build passed unless it was actually run.
- If a change affects model loading, UI flow, or setup, verify the affected path directly.