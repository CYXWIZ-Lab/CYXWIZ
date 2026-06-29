# To Fix 36 - Test Suite Stabilization After To Fix 35

Created: 2026-06-28
Source: Follow-up from `done35.md` closure validation.

## Purpose

`done35.md` is implementation-complete for class imbalance, weighted losses,
FocalLoss graph support, DataLoader balancing, stratified splits, and native
tree-family table runtime nodes.

The remaining work is not part of the `done35` feature scope. It is test-suite
stability and validation cleanup discovered while trying to run broader Debug
coverage after the tree artifact packaging follow-up.

## Current Code Truth

### To Fix 35 Status

Closed from an implementation standpoint:

- Studio loss catalog and graph loss wiring are complete.
- `DataSplit.stratified` is compiled and honored for supported Arrow/text
  paths.
- DataLoader class balancing is exposed and applied for supported train
  batchers.
- CrossEntropy class weights, BCEWithLogits positive weights, common loss
  reductions, SmoothL1/Huber `beta`, and FocalLoss `alpha`/`gamma` are wired.
- `DecisionTreeClassifier`, `RandomForestClassifier`, and
  `GradientBoostingClassifier` are real table-path runtime operators.
- Native `cyxwiz_tree_model` JSON artifacts round-trip for the tree family.
- `TreeModelPredictor` loads native tree artifacts and appends predictions.
- `.cyxmodel` packaging now carries optional tree artifacts at
  `tree/model.json`, with manifest/probe/extraction support.

Validated targeted coverage:

- `test_decision_tree_operator.exe`
- `test_random_forest_operator.exe`
- `test_gradient_boosting_operator.exe`
- `test_tree_model_artifact.exe`
- `test_cyxmodel_sequence_assets.exe`
- `test_job_execution_service.exe "[p2p][service]"`
- `test_job_execution_service.exe "[p2p][connect]"`
- `test_job_execution_service.exe "[p2p][job]"`

### Full-Suite Validation Status

The broader Debug test run did not provide a clean suite result. The observed
failures are separate from the `done35` feature implementation.

Observed failures:

- `ONNXLoaderTests` timed out after protobuf descriptor registration failure.
- `P2PServiceTests` originally failed with a missing `CrossEntropyLoss`
  constructor export, then the lightweight P2P cases passed after adding the
  explicit two-argument constructor export.
- `P2PServiceTests` still hangs in the streaming/download/concurrent area.
- `SecurityTests` fails on cleanup/access-denied behavior around test state
  files.
- Several `ExportONNX` tests fail on cleanup/access-denied behavior under
  `test_onnx_export_data`.

## Follow-Up Work

### Phase 1 - P2P Test Hang Isolation

Goal: make `P2PServiceTests` deterministic and bounded.

Tasks:

- [ ] Isolate the exact hanging section under
  `test_job_execution_service.exe "[p2p][streaming]"`.
- [ ] Add client-side deadlines to blocking streaming RPC reads/writes in the
  tests.
- [ ] Ensure `JobExecutionServiceTest` teardown always stops the server and
  joins or releases background work.
- [ ] Run `"[p2p][download]"` and `"[p2p][concurrent]"` separately after the
  streaming hang is bounded.
- [ ] Keep the already-passing `"[p2p][service]"`, `"[p2p][connect]"`, and
  `"[p2p][job]"` cases passing.

Acceptance:

- Running the full `test_job_execution_service.exe` exits without hanging.
- Any failing P2P assertion reports normally through Catch2 instead of leaving
  a stale test process.

### Phase 2 - ONNX Loader Protobuf Conflict

Goal: remove the duplicate protobuf descriptor registration failure in
`ONNXLoaderTests`.

Observed signal:

- `File already exists in database: onnx/onnx-ml.proto`
- `GeneratedDatabase()->Add(encoded_file_descriptor, size)` failure

Likely cause:

- `test_onnx_loader.cpp` links ONNX Runtime and ONNX protobuf model-generation
  code into the same process.

Tasks:

- [ ] Split ONNX test model generation away from the ONNX Runtime loader test
  process, or use a checked-in/minimal fixture model instead of linking
  `ONNX::onnx` into `test_onnx_loader`.
- [ ] Keep `CYXWIZ_HAS_ONNX` loader coverage active.
- [ ] Keep `CYXWIZ_HAS_ONNX_EXPORT` serialization coverage in the export test
  target, where protobuf ownership belongs.

Acceptance:

- `ONNXLoaderTests` exits normally and does not trigger duplicate descriptor
  registration.
- Loader tests still cover loading, specs extraction, inference, unload, and
  repeated load/unload when ONNX Runtime is available.

### Phase 3 - Cleanup-Resilient Test Directories

Goal: stop Windows file-lock/cleanup failures from masking the actual test
result.

Affected areas:

- `tests/unit/test_onnx_export.cpp`
- `cyxwiz-server-node/tests/test_security.cpp`

Tasks:

- [ ] Replace shared fixed cleanup paths with unique temp directories where
  practical.
- [ ] Use `std::error_code` cleanup variants during teardown so cleanup
  failures can be reported without throwing over the real assertion result.
- [ ] Ensure files are closed before cleanup, especially ONNX output files,
  security JSON state files, audit logs, and wallet state files.
- [ ] Add narrow retry/backoff only where Windows file handles are expected to
  release asynchronously.

Acceptance:

- ONNX export tests do not fail solely because `remove` or `remove_all` gets
  access denied during teardown.
- Security tests do not fail solely because state-file cleanup is denied.
- Real behavioral failures remain visible as assertions.

### Phase 4 - Re-Run Focused Then Full Validation

Goal: produce a clean, useful validation story after the targeted fixes.

Tasks:

- [ ] Rebuild Debug targets touched by this follow-up.
- [ ] Run focused P2P, ONNX loader, ONNX export, and security test targets.
- [ ] Re-run the full Debug `ctest` suite with a bounded timeout policy.
- [ ] Document any remaining failures as separate feature or environment
  tickets.

Acceptance:

- Full Debug `ctest` either passes or leaves only documented failures with
  clear ownership and no stale processes.

## Non-Goals

- Reopening `done35` implementation work.
- Changing tree-classifier runtime behavior unless a regression is discovered.
- Changing ONNX export semantics while fixing loader-test isolation.
- Hiding failing tests without preserving equivalent focused coverage.

## Verification Targets

- `cmake --build build --config Debug --target test_job_execution_service`
- `build/bin/Debug/test_job_execution_service.exe`
- `build/bin/Debug/test_onnx_loader.exe`
- Relevant ONNX export test target or Catch2 filter
- `build/bin/Debug/test_security.exe`
- Full Debug `ctest` after focused fixes
