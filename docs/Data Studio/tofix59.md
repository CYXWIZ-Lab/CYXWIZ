# tofix59 - Properties Truth Cleanup-Safe Stale Key Expansion

## Purpose

Extend the `tofix38` Properties truth work by expanding the catalog of raw
parameters that can be safely cleaned from old or migrated nodes.

The current raw parameter inspector can show unknown/stale keys and can remove
only one proven-safe duplicate: DataInput's legacy `dataset` key when
`dataset_name` is already set.

This ticket should broaden cleanup only where the engine contract proves the
old key is redundant.

## Scope

- Audit node parameter aliases and legacy keys already supported by compiler,
  loader, materializer, runtime, or exporter paths.
- Identify keys that are safe to remove only when a canonical key is present
  and takes precedence.
- Add cleanup-safe metadata for those keys in the Properties truth resolver.
- Keep unknown keys visible but not automatically removable unless the resolver
  can prove they are redundant.
- Add focused resolver tests for every cleanup-safe rule.

## Guardrails

- Do not add a "delete all stale keys" action.
- Do not mark a key cleanup-safe merely because the current truth slice does
  not understand it.
- Do not break old saved graphs, templates, or imported graph formats.
- Preserve lazy alias normalization: old keys should continue to load until a
  user explicitly removes proven-safe duplicates.
- Keep the cleanup catalog small and evidence-backed.

## Candidate Areas

- Output class aliases: `classes` vs `num_classes`.
- DataInput aliases beyond `dataset` if compiler/loader precedence is explicit.
- Text/token sequence aliases only where a single effective key is proven.
- Deprecated folded text nodes only if the canonical replacement fully owns the
  value.

## Acceptance Criteria

- The raw parameter inspector distinguishes:
  - mapped canonical keys,
  - mapped aliases still needed for compatibility,
  - stale unknown keys that are not cleanup-safe,
  - stale duplicate keys that are cleanup-safe.
- Every cleanup-safe key has a short reason shown in the UI.
- Cleanup actions remain explicit per key.
- Focused tests cover each cleanup-safe rule and each legacy-only non-cleanup
  case.
- `test_properties_truth` passes.
- `cyxwiz-engine` builds.
