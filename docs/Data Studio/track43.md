# track43 - Materialization Memory Safety

## Scope

Implement the first TF-IDF slice from `tofix43.md`.

Essentials:

- estimate dense TF-IDF raw output bytes from input rows and `max_features`
- apply a conservative peak multiplier before allocation-heavy work
- compare the estimate to available system RAM
- emit materialization progress with risk/decision text before tokenization
- block clearly unsafe materialization before building token maps or Arrow builders
- keep Dashboard and Studio Debugger visibility through the existing materialization trace fields

Deferred:

- compiler-wide materialization preflight
- user-configured memory budgets
- actual memory growth sampling during materialization
- sparse TF-IDF output
- chunked or disk-backed materialization
- other operators beyond TF-IDF

## Lean Guardrail

Do not build a broad materializer framework yet. Add only a small typed memory
guard that TF-IDF can call and that tests can exercise without creating huge
tables.

## Implementation Notes

- Reuse existing `PipelineOperatorProgress::estimated_memory_bytes`.
- Put risk level and suggestions in the materialization message for current UI.
- Treat unavailable memory detection conservatively instead of silently allowing
  large allocations.

## Validation

- [x] Unit-test memory guard thresholds.
- [x] Unit-test TF-IDF emits a preflight memory event on small data.
- [x] Verify small TF-IDF materialization still works.
- [x] Run targeted TF-IDF truth executable.


Verified:

- `cmake --build D:\Dev\CyxWiz_Claude\build --config Release --target test_computation_truth_tfidf_loss`
- `D:\Dev\CyxWiz_Claude\build\bin\Release\test_computation_truth_tfidf_loss.exe`
- `cmake --build D:\Dev\CyxWiz_Claude\build --config Release --target test_pipeline_operator_metadata`
- `D:\Dev\CyxWiz_Claude\build\bin\Release\test_pipeline_operator_metadata.exe`
- `cmake --build D:\Dev\CyxWiz_Claude\build --config Release --target cyxwiz-engine`
## Slice 2 - Structured Risk Visibility

Status: complete.

Added a typed `memory_risk_level` field through:

- `PipelineOperatorProgress`
- `TrainingTraceEvent` JSON persistence/loading
- graph training launch materialization forwarding
- Training Dashboard materialization summary
- Studio Debugger materialization breakdown
- debug support bundle export

TF-IDF now sets `memory_risk_level` on its preflight event. Other materializers
can leave it empty until they adopt the memory guard.

Verified:

- `cmake --build D:\Dev\CyxWiz_Claude\build --config Release --target test_computation_truth_tfidf_loss`
- `D:\Dev\CyxWiz_Claude\build\bin\Release\test_computation_truth_tfidf_loss.exe`
- `cmake --build D:\Dev\CyxWiz_Claude\build --config Release --target test_debugger_contracts`
- `D:\Dev\CyxWiz_Claude\build\bin\Release\test_debugger_contracts.exe`
- `cmake --build D:\Dev\CyxWiz_Claude\build --config Release --target cyxwiz-engine`
## Slice 3 - Decision Status Propagation

Status: complete.

Added `PipelineOperatorProgress::status` so materializer operators can report
structured decisions independently from message text. TF-IDF preflight maps
memory decisions to:

- `running` for safe
- `warning` for warning
- `risky` for risky
- `blocked` for blocked

Graph training launch now forwards that status into `TrainingTraceEvent`, and
Training Dashboard shows non-running decision status beside memory risk.

Verified:

- `cmake --build D:\Dev\CyxWiz_Claude\build --config Release --target test_computation_truth_tfidf_loss`
- `D:\Dev\CyxWiz_Claude\build\bin\Release\test_computation_truth_tfidf_loss.exe`
- `cmake --build D:\Dev\CyxWiz_Claude\build --config Release --target cyxwiz-engine`
## Slice 4 - CountVectorizer Memory Guard

Status: complete.

Extended the existing dense materialization memory preflight from TF-IDF to
CountVectorizer. The guard now runs before the Arrow text column is copied into
`std::vector<std::string>`, reports structured `status` and `memory_risk_level`,
and blocks clearly unsafe dense `rows x max_features` output before the heavy
count matrix path starts.

Kept this slice narrow: no new materializer framework, no sparse output path,
and no extra configuration surface.

Verified:

- `cmake --build D:\Dev\CyxWiz_Claude\build --config Release --target test_operator_configure_resets`
- `D:\Dev\CyxWiz_Claude\build\bin\Release\test_operator_configure_resets.exe`
- `cmake --build D:\Dev\CyxWiz_Claude\build --config Release --target cyxwiz-engine`
## Slice 5 - TimeSeriesWindow Memory Guard

Status: complete.

Added dense materialization preflight to TimeSeriesWindow. The guard estimates
window output before reading source values into local vectors or reserving Arrow
builders, using planned input rows, `input_width`, `shift`, feature column
count, label column, and optional time metadata column.

The operator now emits structured `status` and `memory_risk_level` on the first
progress event and blocks unsafe window materialization before the dense output
path starts.

Kept this slice narrow: no compiler-wide planner, no streaming window writer,
and no new user-facing configuration knobs.

Verified:

- `cmake --build D:\Dev\CyxWiz_Claude\build --config Release --target test_operator_configure_resets`
- `D:\Dev\CyxWiz_Claude\build\bin\Release\test_operator_configure_resets.exe`
- `cmake --build D:\Dev\CyxWiz_Claude\build --config Release --target cyxwiz-engine`