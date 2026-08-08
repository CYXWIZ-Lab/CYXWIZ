# Done 33 - Runtime Architecture Follow-Up After Track 22

**Created:** 2026-06-25
**Source:** Follow-up carried forward after completing `donetrack22.md` and the
current `done22.md` implementation slice.

## Status

Archived after the 2026-07-07 follow-up slice.

Track 22 is complete for the current runtime-architecture hardening pass.
This document records the completed source queue. Remaining follow-up work is
carried forward in `tofix57.md` so this archive does not continue to look like
the active queue.

Completed slice:

- `DataStudioExecutionPlan` now validates central enum, integer, and float
  runtime parameter capabilities before execution.
- `PipelineExecutor` and `DataStudioExecutionPlan` share
  `ValidatePipelineRuntimeParameterCapabilities` instead of owning duplicate
  bounds and enum validation logic.
- `test_data_studio_execution_plan` covers representative integer and float
  failures before execution.
- `test_pipeline_executor_operator_routing` preserves existing executor
  diagnostics for central validation failures.

## Current Code Truth

The completed Track 22 work established these contracts:

- `PipelineRuntimeSupport` is the central source for Data Studio runtime
  support, owner, validation, materializer, and compatibility-alias facts.
- `DataStudioExecutionPlan` is the typed plan contract for source, transform,
  sink, and training-launch handoff steps.
- `CompiledGraphPlan` remains the model-training graph plan. Do not replace it
  with a second model graph plan.
- `PipelineMaterializer` is intentionally Arrow-table-only for default
  materialization. Non-Arrow storage domains pass through with explicit
  diagnostics.
- Node Browser and Node Info consume central `support_axes`; frontend code must
  not add another support matrix.
- Static schema validation constraints live in central runtime capability
  tables and are guarded by drift and bad-schema routing tests.
- ArrayFire backend fallback uses shared reason codes, eval barriers,
  one-time logging, source-scan guards, and deterministic forced-fallback test
  coverage.

## Follow-Up Queue

### 1. Extend Typed Data Studio Execution Plan Usage

Move future executable Data Studio paths toward `DataStudioExecutionPlan`
instead of adding new direct execution branches.

Expected work:

- Add plan coverage for each new source, transform, sink, or training-launch
  shape before changing runtime execution.
- Keep compatibility alias resolution before plan construction.
- Keep the handoff to `CompiledGraphPlan` explicit when a Data Studio graph
  enters model training.
- Add drift tests when a new node owner or mode is introduced.

Acceptance:

- New runtime paths can be represented as typed plan steps.
- No node has multiple competing runtime owners.
- Unsupported or fail-closed nodes fail before execution.

### 2. Continue Legacy Alias Retirement

Do not add string-only aliases directly in `PipelineExecutor`.

For every remaining or newly discovered alias, choose exactly one outcome:

- migrate to first-class typed metadata,
- normalize to an existing canonical node before planning,
- keep as a documented hidden compatibility alias,
- remove after proving no supported graph depends on it.

Acceptance:

- Alias decisions live in central capability metadata or a nearby tested
  decision table.
- Existing saved graphs either preserve behavior or fail with a clear
  compatibility reason.
- Routing tests cover any alias that still executes.

### 3. Expand Materializer Only By Domain-Safe Adapters

`PipelineMaterializer` must not silently convert Parquet, text, image, audio,
or other non-Arrow domains into Arrow.

Future adapters must preserve domain semantics:

- Parquet adapter: preserve row-group or streaming semantics.
- Text adapter: preserve sequence/classification contracts.
- Image/audio adapters: remain explicit pass-through unless a tested domain
  adapter exists.
- Any adapter should emit user-visible diagnostics when materialization is
  skipped or narrowed.

Acceptance:

- Arrow behavior remains covered.
- Non-Arrow pass-through remains explicit.
- New adapters have domain-specific tests and do not pretend to be generic
  conversion.

### 4. Keep Frontend Support State Centralized

Further UI polish is allowed, but support truth must keep flowing from
`support_axes`.

Expected work:

- Improve blocked-node affordances in graph-building workflows.
- Add clearer compact reasons for unsupported runtime, compile, training, and
  materializer axes.
- Add UI or contract tests that recreate frontend predicates from
  `support_axes`.

Acceptance:

- UI cannot add, enable, or claim support for centrally blocked nodes.
- Node Browser and Node Info remain consistent for support status and reasons.

### 5. Maintain Schema Validation Coverage For New Nodes

Every new executable Data Studio node must include validation coverage before
it is considered runtime-supported.

Required checks:

- required parameters validate before execution,
- enum values and numeric bounds live in central runtime capability tables
  when static,
- invalid schemas fail before `DataInput`, SQL, operator execution, or sink
  side effects,
- at least one representative bad-schema routing test is registered.

Acceptance:

- Validation drift tests include the new node.
- Bad-schema routing coverage includes the new node when it has static
  validation constraints.

### 6. Apply ArrayFire Fallback Policy To New Backend Paths

Any new ArrayFire-first backend path must use the Track 22 fallback pattern.

Required behavior:

- add `eval()` barriers around large lazy expression chains,
- classify backend failures as `cuda_jit_param_overflow`,
  `arrayfire_jit_compile_failure`, or `gpu_backend_exception`,
- log one time per operation, reason, backend, and shape/context,
- suppress repeated NVRTC compiler dumps,
- keep CPU fallback deterministic when a CPU fallback exists,
- add a forced or mocked backend-failure test when the new path introduces its
  own fallback policy surface.

Acceptance:

- Source-scan guards stay green.
- Focused smoke tests prove backend success or clean CPU fallback.
- Runtime messages describe performance fallback, not graph validation failure.

## Non-Goals

Do not use this follow-up to:

- replace `CompiledGraphPlan`,
- add a second frontend support matrix,
- silently broaden `PipelineMaterializer` into a generic converter,
- add untested aliases in `PipelineExecutor`,
- claim GPU success in UI or docs without backend placement or fallback data,
- turn maintenance guardrails into broad refactors unrelated to a new runtime
  path.

## Verification Targets

Keep these green when working from this document:

- `test_pipeline_operator_metadata`
- `test_data_studio_execution_plan`
- `test_pipeline_executor_operator_routing`
- `test_text_gui_training_launch`
- `test_text_loader_csv_preflight`
- `cyxwiz-tests` focused filters touched by the change
- `cyxwiz-engine` Debug build
- `git diff --check`
