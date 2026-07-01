# Track 27 - Runtime and Materializer Parity Tracker

Implementation tracker and closeout evidence for `done27.md`.

## Scope

Prove that operator-backed Data Studio nodes execute consistently through:

- `PipelineExecutor`
- `PipelineMaterializer`
- `PipelineOperatorFactory`
- `PipelineRuntimeCapabilities`
- `NodeMetadataRegistry`

This tracker should stay narrow. It is not a second support matrix and should not redefine runtime truth.

## Current Engine Truth

- Operator-backed nodes are declared by `PipelineRuntimeCapabilities`.
- Operators are constructed by `PipelineOperatorFactory`.
- `PipelineExecutor::ExecuteNode` routes operator-backed nodes before legacy typed dispatch.
- `PipelineMaterializer` applies operator-backed nodes through `PipelineOperatorFactory` for Arrow-table sources.
- Materializer support is currently Arrow-table only.
- Unsupported storage modes should report explicit skip/fail information, not silently bypass operators.

## Lean Guardrails

- Add tests before implementation changes.
- Do not add a new registry or support matrix.
- Fix only mismatches exposed by parity tests.
- Keep helpers small and local to tests unless reused by production code.
- Compare stable contracts: schema, row count, key output columns, and deterministic values.

## Test Strategy

Parity test home:

`cyxwiz-engine/tests/test_pipeline_operator_materializer_parity.cpp`

Implemented behavior:

- Build minimal Arrow tables in memory.
- Run equivalent graph through `PipelineExecutor`.
- Run the same graph through `PipelineMaterializer` or `MaterializeTable`.
- Compare output schema, row count, and key values.
- Keep unsupported materializer storage modes explicit and centralized.

## Coverage Matrix

| Node | Family | Executor path | Materializer path | Parity status | Notes |
|---|---|---:|---:|---|---|
| `StandardScaler` | numeric preprocessing | Existing operator-backed | Existing Arrow materializer route | Implemented in parity test | Compares schema, rows, and `x` values. |
| `MinMaxScaler` | numeric preprocessing | Existing operator-backed | Existing Arrow materializer route | Implemented in parity test | Compares schema, rows, and `x` values. |
| `RobustScaler` | numeric preprocessing | Existing operator-backed | Existing Arrow materializer route | Implemented in parity test | Compares schema, rows, and `x` values. |
| `LogTransform` | time-series/numeric | Existing operator-backed | Existing Arrow materializer route | Implemented in parity test | Compares schema, rows, and `y` values. |
| `Differencing` | time-series/numeric | Existing operator-backed | Existing Arrow materializer route | Implemented in parity test | Compares schema, shortened row count, and `y` values. |
| `LabelEncoder` | categorical preprocessing | Existing operator-backed | Existing Arrow materializer route | Implemented in parity test | Compares encoded `category` values. |
| `OrdinalEncoder` | categorical preprocessing | Existing operator-backed | Existing Arrow materializer route | Implemented in parity test | Compares encoded `category` and `group` values. |
| `TargetEncoder` | categorical preprocessing | Existing operator-backed | Existing Arrow materializer route | Implemented in parity test | Compares encoded `category` target-mean values. |
| `OutlierDetector` | preprocessing analytics | Existing operator-backed | Existing Arrow materializer route | Implemented in parity test | Compares `is_outlier` flag values. |
| `PCANode` | analytics | Existing operator-backed | Existing Arrow materializer route | Implemented in parity test | Compares schema, rows, `pc_0`, `pc_1`, and `y` values. |
| `SentimentAnalyzer` | text analytics | Existing operator-backed | Existing Arrow materializer route | Implemented in parity test | Compares all output sentiment/label columns on a tiny text fixture. |
| `TextTokenizer` | text preprocessing | Existing operator-backed | Existing Arrow materializer route | Implemented in parity test | Compares token columns and `y` values. |
| `CountVectorizer` | text preprocessing | Existing operator-backed | Existing Arrow materializer route | Implemented in parity test | Dense tiny fixture; compares all output columns. |
| `TFIDFVectorizer` | text preprocessing | Existing operator-backed | Existing Arrow materializer route | Implemented in parity test | Dense tiny fixture; compares all output columns. |
| `TimeSeriesWindow` | time-series | Existing operator-backed | Existing Arrow materializer route | Implemented in parity test | Compares full generated window table, including metadata time column. |
| `TimeSeriesFeatures` | time-series | Existing operator-backed | Existing Arrow materializer route | Implemented in parity test | Compares original sliced columns plus lag/rolling feature columns. |
| `TimeSeriesSplit` | time-series | Existing operator-backed | Existing Arrow materializer route | Implemented in parity test | Current implementation is a single table with appended partition column. |

## Storage Mode Checks

| Source kind | Materializer expectation | Status |
|---|---|---|
| Arrow table | Apply supported operator-backed nodes | Parity coverage implemented |
| Parquet-backed | Explicit unsupported-source skip | Existing test coverage in text launch helper |
| Legacy text dataset | Explicit unsupported-source skip | Existing test coverage in text launch helper |
| Image dataset | Explicit unsupported-source skip | Existing test coverage in text launch helper |
| Audio dataset | Explicit unsupported-source skip | Existing test coverage in text launch helper |

## Known Gaps

- `TimeSeriesSplit` currently maps cleanly to a single Arrow table by appending a partition column; if future split nodes produce multiple physical outputs, add a separate contract test for that shape.
- Text vectorizer parity must keep input fixtures tiny to avoid dense memory cost hiding correctness failures.

## Done Criteria

- Completed: parity test target is added and wired into CMake.
- Completed: numeric preprocessing parity is proven.
- Completed: text preprocessing parity is proven for tiny deterministic fixtures.
- Completed: analytics and categorical preprocessing parity is proven.
- Completed: time-series parity is proven for current single-table operator contracts.
- Completed: unsupported storage modes remain explicit and centralized.
