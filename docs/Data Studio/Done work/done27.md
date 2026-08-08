# Done 27 - Data Studio Runtime and Materializer Parity

Status: completed.

Closeout evidence:

- Added focused parity target:
  `test_pipeline_operator_materializer_parity`
- Added parity coverage for numeric, categorical, text, analytics, and
  time-series operator-backed nodes.
- Validated `PipelineExecutor` output against `PipelineMaterializer` output for
  schema, row counts, and deterministic numeric output values.
- Confirmed current `TimeSeriesSplit` contract is a single Arrow table with an
  appended partition column, so it fits the same parity harness.
- Extended the parity harness to compare unsigned Arrow integer scalar types
  produced by CSV integer compaction.

Validation command:

```powershell
cmake --build build --config Debug --target test_pipeline_operator_materializer_parity -- /m:1 /v:minimal
build\bin\Debug\test_pipeline_operator_materializer_parity.exe
```

Observed result:

```text
Pipeline operator materializer parity passed
```

This document tracks the follow-up work split out from `tofix7.md`.

`tofix7.md` cleaned up the support contract: metadata, capability axes,
operator-backed routing, blocked-node truth, and workflow lanes.

This file is about the next narrower problem:

`When a Data Studio node is marked real/operator-backed, does the runtime and
materializer path execute it consistently for the same dataset modes?`

---

## Goal

Make Data Studio runtime behavior match the support contract for
operator-backed preprocessing, text, and time-series nodes.

The target is not to add another support matrix. The source of truth remains:

- `PipelineRuntimeCapabilities`
- `PipelineOperatorFactory`
- `PipelineExecutor`
- `PipelineMaterializer`
- `NodeMetadataRegistry`

The target is to remove split behavior where a node is real in one path but
limited, bypassed, or differently handled in another path.

---

## Current Engine Truth

From `tofix7.md` closeout:

- operator-backed canonical execution now runs before typed legacy dispatch
  in `PipelineExecutor::ExecuteNode`
- operator-backed canonical names are guarded against also being registered as
  legacy-dispatched or fail-closed names
- operator-backed real nodes have metadata tests for implemented status,
  non-blocked badge, support state, owner, runtime, and materializer axes

Resolved gap:

- canonical operator-backed candidates now have executor/materializer parity
  checks for Arrow-table materializer inputs
- unsupported/non-Arrow materializer storage modes remain explicit and
  centralized outside this ticket

---

## Candidate Nodes

### Time-series and numeric transforms

| Node | Current concern | Expected direction |
|---|---|---|
| `TimeSeriesWindow` | Operator-backed, but training/materializer flow can be dataset-mode sensitive. | Add parity tests for executor and materializer paths. |
| `TimeSeriesSplit` | Operator exists; split outputs need consistent runtime/materializer handling. | Verify output contract and downstream routing. |
| `TimeSeriesFeatures` | Operator exists; legacy/time-series helper path may differ. | Prefer operator path and test materialized output. |
| `LogTransform` | Operator exists. | Confirm operator output and materializer output match. |
| `Differencing` | Operator exists. | Confirm operator output and materializer output match. |

### Text transforms

| Node | Current concern | Expected direction |
|---|---|---|
| `TextTokenizer` | Operator exists, but text flow can depend on data mode. | Add executor/materializer parity tests for Arrow text input. |
| `TFIDFVectorizer` | Operator-backed route is canonical, but legacy/fail-closed history makes parity important. | Add canonical runtime tests and prevent legacy fallback. |
| `CountVectorizer` | Same as `TFIDFVectorizer`. | Add canonical runtime tests and prevent legacy fallback. |
| `SentimentAnalyzer` | Operator exists; output semantics should be checked end to end. | Add output schema/value-contract tests. |

### Analytics / preprocessing operators

| Node | Current concern | Expected direction |
|---|---|---|
| `PCANode` | Operator-backed path is canonical. | Add runtime/materializer parity coverage. |
| `StandardScaler` | Operator-backed route is canonical; legacy fake-success history exists. | Add parity tests and remove dead placeholder assumptions. |
| `MinMaxScaler` | Same as `StandardScaler`. | Add parity tests. |
| `RobustScaler` | Same as `StandardScaler`. | Add parity tests. |
| `LabelEncoder` | Operator-backed path is canonical. | Verify encoded output schema and values. |
| `OrdinalEncoder` | Operator-backed path is canonical. | Verify encoded output schema and values. |
| `TargetEncoder` | Operator-backed path is canonical. | Verify target column handling. |
| `OutlierDetector` | Operator-backed path is canonical. | Verify output flag/score contract. |

---

## Implementation Plan

Work in small batches.

1. Add parity test helpers
   - Build minimal Arrow tables in tests.
   - Execute node through `PipelineExecutor` canonical path.
   - Materialize equivalent operator path where supported.
   - Compare schema, row count, and key output columns.

2. Start with the safest numeric operators
   - `StandardScaler`
   - `MinMaxScaler`
   - `RobustScaler`
   - `LogTransform`
   - `Differencing`

3. Add time-series parity coverage
   - `TimeSeriesWindow`
   - `TimeSeriesSplit`
   - `TimeSeriesFeatures`

4. Add text parity coverage
   - `TextTokenizer`
   - `CountVectorizer`
   - `TFIDFVectorizer`
   - `SentimentAnalyzer`

5. Add analytics parity coverage
   - `PCANode`
   - clustering nodes if deterministic output contracts are stable enough
   - encoders and outlier detection

6. Remove or quarantine dead legacy placeholder assumptions
   - Only after tests prove canonical operator behavior.
   - Do not delete compatibility aliases that are still intentionally mapped.

---

## Non-goals

- Do not create a second capability registry.
- Do not mark placeholder dataset or augmentation sources real.
- Do not implement NER or sequence-output training here.
- Do not redesign the Data Studio UI.
- Do not broaden materializer support beyond what tests prove.

---

## Done Criteria

- Completed: canonical operator-backed nodes have runtime/materializer parity
  tests.
- Completed: dataset-mode limits are documented as Arrow-table materializer
  scope with explicit unsupported-source behavior elsewhere.
- Completed: canonical operator-backed parity is proven through
  `PipelineOperatorFactory` routing.
- Completed: `tofix7.md` remains closed as the support-contract cleanup, while
  this ticket closes the runtime/materializer parity follow-up work.
