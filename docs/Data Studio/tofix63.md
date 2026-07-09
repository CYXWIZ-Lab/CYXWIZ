# tofix63 - Materialization Cache Management Follow-up

## Status

Open.

## Background

`tofix47` and `track47` completed the first persistent materialization cache
surface:

- prepared datasets are saved and reused when source data and graph inputs are
  unchanged,
- cache artifact and manifest paths are visible in the Training Dashboard,
- cache row and column counts are surfaced,
- artifact and manifest paths can be copied,
- `Open cache location` opens the cache directory for inspection.

The remaining cache work is product-facing management and deeper inspection. It
should stay separate from `tofix47` so the completed reuse path remains stable.

## Problem

The current cache surface is intentionally non-destructive. Users can inspect
where a prepared dataset lives, but they cannot yet manage cache lifecycle from
Studio.

Likely follow-up needs:

- rebuild a prepared dataset on demand,
- delete a prepared dataset or cache entry,
- inspect more manifest fields directly in the debugger,
- explain stale, corrupt, unsupported, and save-failure cases more clearly if
  real usage exposes gaps.

These actions touch user data and training behavior, so they need explicit UI
states and confirmation instead of being folded into the first cache release.

## Required Scope

- Add `Rebuild prepared dataset` only if the user intent is explicit and the UI
  makes it clear that the next training run will regenerate the cache.
- Add `Delete prepared dataset` only with confirmation and an exact cache path
  preview.
- Keep `Inspect prepared dataset` and `Open cache location` non-destructive.
- Surface additional manifest fields in the debugger only where they help
  explain a cache decision.
- Keep cache mutation events visible in training/debug history.

## Safety Rules

- Never silently delete cache artifacts.
- Never rebuild without making the cache decision visible.
- Never return stale data after a destructive action.
- If a cache mutation fails, keep training behavior predictable and report a
  structured error.
- Keep cache controls scoped to materialized prepared datasets; do not add a
  general-purpose cache manager in this ticket.

## Validation

- Rebuild action invalidates or bypasses the existing cache and produces a new
  prepared dataset.
- Delete action removes the selected cache entry and the next unchanged training
  run rebuilds it.
- Inspect/open actions remain non-destructive.
- Debug history records cache mutation decisions.
- Existing materialization cache tests still pass:
  - `test_materialization_cache.exe`
  - `test_pipeline_materializer_cache.exe`
  - `test_text_gui_training_launch.exe`
  - `test_graph_training_sequence_preflight.exe`
- Release engine build still passes.

## Non-goals

- Do not reopen `tofix47`.
- Do not add cloud/distributed cache behavior.
- Do not cache intermediate tables.
- Do not create a broad storage subsystem.
- Do not add destructive actions without confirmation.

## Acceptance Criteria

- Cache management follow-ups are implemented as explicit, inspectable Studio
  actions.
- Destructive cache actions require confirmation and name the exact target.
- Training/debug history explains each cache lifecycle decision.
- Existing prepared dataset reuse behavior remains unchanged unless the user
  explicitly chooses rebuild or delete.
