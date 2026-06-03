# To Fix - Closed Archive

**Status:** DONE / CLOSED on 2026-06-03.

This file used to hold the large Phase 0-4 issue log and completed work
history. It grew past a useful active-backlog size, so the remaining
pending items were migrated into:

- `docs/Data Studio/tofix1.md`

Use `tofix1.md` for active work. Do not append new pending items here.

## Migration Boundary

The sweet spot for backlog management is:

- `tofix.md` stays closed as the archive/index.
- `tofix1.md` contains only active pending work.
- Each `tofixN.md` should stay small enough to audit in one pass.
- Completed implementation details belong in commits, tests, and focused
  design docs, not in an ever-growing active backlog.

## Historical Context

The full completed-history version of this file is available in git
history before this migration. The most important closed themes were:

- backend tensor residency and ArrayFire layout contract,
- image/audio/text dataset loading and audits,
- compile-gate and pin-contract improvements,
- DataInputDialog loader refactor,
- node registration cleanup,
- Phase 3 text wiring,
- Phase 4 time-series/operator framework work,
- initial Text Fix B frontend/backend activation slices.

Current pending work starts in `tofix1.md`.
