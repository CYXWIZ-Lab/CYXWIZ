# To Fix - Closed Archive

**Status:** DONE / CLOSED on 2026-06-03.

This file used to hold the large Phase 0-4 issue log and completed work
history. It grew past a useful active-backlog size, so the remaining
pending items were migrated into:

- `docs/Data Studio/done1.md`

Use the next active `tofixN.md` file for active work. Do not append new
pending items here.

## Migration Boundary

The sweet spot for backlog management is:

- `tofix.md` stays closed as the archive/index.
- `done1.md` and `done2.md` are closed archives.
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

Closed follow-up work from this archive is now in `done1.md` and
`done2.md`. Current pending work should use the next active `tofixN.md`
file instead of reopening closed `doneN.md` archives.
