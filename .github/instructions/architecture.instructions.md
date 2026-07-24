---
applyTo: "**/*.{py,bat,md,json,yml,yaml}"
---

# Architecture instructions

- Understand the repository before making changes.
- Preserve existing entrypoints, application flow, and Windows setup expectations.
- Prefer additive changes and small fixes over architectural rewrites.
- When changing shared logic, search all call sites and update affected consumers.
- Keep model, UI, and setup concerns separated where possible.
- Favor maintainable, readable code over clever shortcuts.
- Document architectural assumptions when they affect setup or runtime behavior.