# tofix62 - KMeans Routing Test Timeout Follow-up

## Status

Open.

## Background

`tofix48` made the Data Studio Properties panel truth-backed and closed the
review findings around metadata truth, aliases, and runtime-facing defaults.
Focused validation passed:

- `test_properties_truth.exe`
- `test_pipeline_operator_metadata.exe`
- `cmake --build build --target cyxwiz-engine --config Debug -- /m:1`

During follow-up verification, `test_pipeline_executor_operator_routing.exe`
timed out after four minutes. The captured log reached the tiny
`KMeansCluster` routing case and stopped after:

```text
[Data Studio] Started executing node: KMeans (parallel batch)
```

The failing case uses a 3-row, 2-column CSV with:

```json
{"feature_cols":"x,y","n_clusters":"2","max_iter":"10","n_init":"1","init":" RANDOM "}
```

This points at the ArrayFire-backed KMeans execution path or backend
initialization behavior, not the Properties truth resolver changes.

## Problem

The routing test currently has a hard-to-diagnose timeout in the
`KMeansCluster` success path. Because the test process hangs instead of
returning a structured failure, it is difficult to distinguish:

- an ArrayFire backend initialization stall,
- a KMeans implementation loop that does not converge or return,
- an environment-specific GPU/runtime issue,
- a test fixture issue around the selected `init` mode,
- or missing timeout/cancellation protection around clustering operators.

Leaving this as an implicit known issue weakens the routing test suite because
future failures can be mistaken for the same hang.

## Required Investigation

- Reproduce the timeout with the smallest standalone KMeans pipeline.
- Capture the active ArrayFire backend, device, and any fallback warnings before
  KMeans starts fitting.
- Compare `init=random` and `init=kmeans++` on the same tiny dataset.
- Check whether the hang occurs before or inside `Clustering::KMeans`.
- Add enough scoped logging or test instrumentation to identify the last
  successful step without flooding normal test output.
- Confirm whether the timeout is environment-specific by running on CPU and GPU
  backends where available.

## Required Fix

The fix should be the smallest change that makes the behavior bounded and
diagnosable:

- If KMeans is hanging because of a real implementation bug, fix that path.
- If the issue is backend-specific, return a structured unsupported/fallback
  result instead of hanging.
- If the test fixture is selecting a known unstable backend path, make the test
  assert normalization and routing without relying on an unbounded GPU fit.
- If cancellation/timeout protection is missing, add bounded protection at the
  operator or test harness level without masking real algorithm failures.

## Validation

- `test_pipeline_executor_operator_routing.exe` completes without manual
  timeout.
- The KMeans routing case still proves `init:" RANDOM "` normalizes to
  `random`.
- Failure paths return specific errors rather than hanging.
- Existing focused truth tests still pass:
  - `test_properties_truth.exe`
  - `test_pipeline_operator_metadata.exe`
- Engine build still passes.

## Non-goals

- Do not reopen `tofix48`.
- Do not rewrite clustering broadly.
- Do not add new KMeans UI controls.
- Do not hide the KMeans case by deleting routing coverage.
- Do not claim GPU KMeans performance improvements without benchmark evidence.

## Acceptance Criteria

- The routing test has no unbounded KMeans hang on the local debug build.
- KMeans routing coverage remains meaningful and checks normalized runtime
  parameters.
- Any backend limitation is reported as a structured error, fallback, or skipped
  condition with an explicit reason.
- The ticket documents the root cause found during investigation.
