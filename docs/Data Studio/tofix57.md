# To Fix 57 - Data Studio Runtime Architecture Follow-Up After done33

Created: 2026-07-07
Source: Follow-up after completing the narrow `done33.md` schema-validation
slice in commit `0f6ce2ca Add assistant plugin docs and Data Studio follow-ups`.

## Status

Open.

`done33.md` is now the archived source ticket for the Track 22
runtime-architecture maintenance queue. This file records what was completed
from that queue and keeps the remaining work visible for the next
implementation pass.

## Completed From tofix33

The completed slice addressed part of item 5, schema validation coverage:

- `DataStudioExecutionPlan` now validates central enum, integer, and float
  runtime parameter capabilities before execution.
- `PipelineExecutor` and `DataStudioExecutionPlan` share
  `ValidatePipelineRuntimeParameterCapabilities` instead of owning duplicate
  bounds and enum validation logic.
- `test_data_studio_execution_plan` covers representative integer and float
  failures before execution.
- `test_pipeline_executor_operator_routing` still preserves existing executor
  diagnostics for central validation failures.

## Remaining Follow-Up Queue

### 1. Expand DataStudioExecutionPlan Adoption

Future executable Data Studio paths should be represented as typed
`DataStudioExecutionPlan` steps before adding execution behavior.

Required guardrails:

- keep compatibility alias resolution before plan construction,
- keep model-training handoff explicit through `CompiledGraphPlan`,
- add drift coverage when a new owner, support mode, source, sink, transform,
  or training-launch shape is introduced,
- fail unsupported or fail-closed nodes before execution.

### 2. Continue Legacy Alias Retirement

Do not add string-only aliases directly in `PipelineExecutor`.

For each remaining alias, choose one central decision:

- normalize to an existing canonical node,
- migrate to first-class typed metadata,
- keep as a documented hidden compatibility alias,
- remove only after proving no supported saved graph depends on it.

Alias decisions should remain in central capability metadata or a nearby tested
decision table.

### 3. Keep Materializer Expansion Domain-Safe

`PipelineMaterializer` is still Arrow-table-only by default. Do not broaden it
into a generic converter.

Future materializer work should add domain-specific adapters only when their
semantics are tested:

- Parquet: preserve row-group or streaming semantics,
- text: preserve sequence/classification contracts,
- image/audio: explicit pass-through unless a tested domain adapter exists,
- skipped or narrowed materialization must produce user-visible diagnostics.

### 4. Keep Frontend Support Truth Centralized

Node Browser, Node Info, and graph-building affordances must continue deriving
support state from `support_axes`.

Next useful work:

- improve blocked-node affordances in graph-building workflows,
- add compact reasons for unsupported runtime, compile, training, and
  materializer axes,
- add UI or contract tests that recreate frontend predicates from
  `support_axes`.

### 5. Maintain Validation Coverage For New Runtime Nodes

For every new executable Data Studio node:

- required parameters must validate before execution,
- static enum and numeric bounds must live in central capability tables,
- invalid schemas must fail before `DataInput`, SQL, operator execution, or
  sink side effects,
- bad-schema routing coverage must be registered when static constraints exist.

The shared capability validator added in the `tofix33` slice should be reused
instead of adding another local validator.

### 6. Apply ArrayFire Fallback Policy To New Backend Paths

Any new ArrayFire-first backend path must use the Track 22 fallback pattern:

- add `eval()` barriers around large lazy expression chains,
- classify backend failures with shared reason codes,
- log once per operation, reason, backend, and shape/context,
- suppress repeated NVRTC compiler dumps,
- keep CPU fallback deterministic when available,
- add forced or mocked backend-failure coverage for new fallback surfaces.

## Non-Goals

Do not use this ticket to:

- replace `CompiledGraphPlan`,
- add another frontend support matrix,
- turn `PipelineMaterializer` into a generic converter,
- add untested aliases in `PipelineExecutor`,
- claim GPU success without backend placement or fallback evidence,
- do broad refactors unrelated to a concrete runtime path.

## Verification Targets

Keep these green for future work from this follow-up:

- `test_data_studio_execution_plan`
- `test_pipeline_executor_operator_routing`
- `test_pipeline_operator_metadata`
- `test_text_gui_training_launch`
- focused `cyxwiz-tests` filters touched by the change
- `cyxwiz-engine` Debug build when engine/runtime paths change
- `git diff --check`
