# 51) Runtime fail-closed behavior and training-capability gap register

## 51.1 Scope

This section documents how the engine classifies node runtime support and how compile-time and runtime contracts block execution for unsupported branches (fail-closed behavior).

The focus is:

- node runtime capability matrix (operator-backed, legacy, or fail-closed),
- compile-time capability validation for training graphs,
- hard-fail behavior in pipeline execution,
- migration/alias strategy for legacy names,
- concrete gap inventory that still blocks full parity.

## 51.2 Runtime capability model

`pipeline_runtime_capabilities` defines a strict, table-driven contract:

- `PipelineRuntimeSupportMode`:
  - `OperatorBacked`
  - `LegacyExecutor`
  - `FailClosed`
- `PipelineRuntimeFailMode`:
  - `Real`
  - `HardFail`
  - `Simulated`
  - `Passthrough`
- `PipelineTrainingBackendSupportMode`:
  - `Allowed`
  - `UnsupportedSequentialModelLayer`
  - `UnsupportedTrainingControl`

Capability sources are explicit lists:

- operator runtime nodes (`GetPipelineOperatorRuntimeCapabilities`)
- fail-closed runtime nodes (`GetPipelineFailClosedRuntimeCapabilities`)
- legacy runtime nodes (`GetPipelineLegacyRuntimeCapabilities`)
- legacy aliases (`GetPipelineLegacyAliasDecisionCapabilities`)
- training backend support tables (`GetPipelineUnsupportedSequentialModelLayerCapabilities`, `GetPipelineUnsupportedTrainingControlCapabilities`, `GetPipelineSupportedTrainingBackendCapabilities`)

`ResolvePipelineRuntimeSupport` resolves in a fixed order:

1. operator-backed map,
2. fail-closed map (with explicit reason and `HardFail`),
3. legacy runtime map,
4. fallback `Unknown`.

`with_validation_axes(...)` enriches resolved support with:

- source-node flag
- required input count
- required parameters
- allowed values and integer/float constraints.

### 51.2.1 ASCII representation

```text
legacy node name
 -> (with alias resolution)
 -> runtime support resolver
    -> operator-backed (real) | fail-closed (hard fail) | legacy (executor)
       -> validation axes (parameters, arity, required args, source flag)
```

## 51.3 Compile-side capability enforcement

The compiler enforces unsupported nodes through multiple passes:

- Backend placement classifies each layer; unsupported sequential layers are mapped to an explicit unsupported placement object (`BuildUnsupportedSequentialModelPlacement`) with reason from training support.
- `ValidateUnsupportedTrainingControlNodes` iterates all nodes and emits a compile error when backend mode is `UnsupportedTrainingControl`.
- `IsModelLayer(type)` returns true for unsupported model types that are intentionally recognized, so unsupported branches remain diagnosed rather than silently skipped.

Consequences:

- user-visible compile issues are emitted early before launch,
- unsupported constructs are preserved in diagnostics with concrete node IDs and names,
- sequencing for unsupported training paths remains explicit and explainable.

### 51.3.1 ASCII contract

```text
ML nodes
 -> GraphCompiler
    -> training support resolve
       -> model layer classifier
          -> backend placement or unsupported placement
       -> training control validator
    -> Issue(ERROR) for unsupported nodes
 -> launch blocked
```

## 51.4 Runtime execution enforcement

`PipelineExecutor` performs final execution-time enforcement in two layers:

1. **Schema/contract precheck in `ValidatePipeline(...)`**
   - rejects unknown node types,
   - rejects nodes with invalid parameter sets,
   - rejects nodes where `pipeline_executor_supported == false` with `UnsupportedNode`.
2. **Execution-time hard-fail in `ExecuteNode(...)`**
   - immediately errors with `FailUnsupportedNode(...)` for `PipelineRuntimeSupportMode::FailClosed`,
   - routes operator-backed nodes through operator execution,
   - otherwise falls into typed legacy execution.

The fail-closed contract is intentionally hard-fail:

- no silent passthrough,
- no placeholder success,
- no partial graph completion when an unsupported runtime branch is reached.

## 51.5 Unsupported-capability inventory and gap register

### 51.5.1 Fail-closed families (runtime)

`GetPipelineFailClosedRuntimeCapabilities` is large and includes:

- classic dimensionality-reduction and classical ML families (`TSNENode`, `UMAPNode`, `SVMClassifier`, `KNNClassifier`, `NaiveBayesClassifier`, `LogisticRegressionNode`, `SVMRegressor`),
- dataset/model analytics and export placeholders (`LearningCurvesNode`, `FeatureImportanceNode`, `CrossValidationNode`, `ExportSQL`, `ExportExcel`, `ImageFolderDataset`, `MNISTDataset`, `CIFAR10Dataset`, `HuggingFaceDataset`, `KaggleDataset`),
- tensor utility/math operators (add/multiply/reshape/compare/math operators),
- preprocessing and transformation nodes where runtime execution is intentionally absent (`Normalize`, `OneHotEncode`, `AudioInput`, `Spectrogram`, `MelSpectrogram`, `MFCC`),
- model/activation/loss/optimizer runtime families not yet implemented (`ReLU`, `Sigmoid`, `MSELoss`, `Adam`, ...),
- RL and advanced vision nodes not yet wired (`GymEnvironment`, `ReplayBuffer`, `PolicyNetwork`, `ValueNetwork`, `DNNDetect`, `PretrainedYOLO`, `DNNModelLoad`).

### 51.5.2 Unsupported sequential-model-layer classes

`GetPipelineUnsupportedSequentialModelLayerCapabilities` includes layers that compile but are not supported by SequentialModel backend today:

- convolution/pooling family (`Conv2D`, `MaxPool2D`, `AvgPool2D`, `GlobalMaxPool`, `GlobalAvgPool`, `ConvTranspose2D`, `Upsample`, `PixelShuffle`),
- RL model sketches (`PolicyNetwork`, `ValueNetwork`),
- normalization/attention/sequence variants not yet wired (`GroupNorm`, `InstanceNorm`, `SelfAttention`, `CrossAttention`, `LinearAttention`),
- recurrent variants with partial/legacy status (`RNN`, `Bidirectional`).

### 51.5.3 Unsupported training-control classes

`GetPipelineUnsupportedTrainingControlCapabilities` currently blocks common scheduler/regularizer nodes:

- `StepLR`, `CosineAnnealing`, `ReduceOnPlateau`, `ExponentialLR`, `WarmupScheduler`,
- `L1Regularization`, `L2Regularization`, `ElasticNet`.

### 51.5.4 Supported training-backend list (current)

`GetPipelineSupportedTrainingBackendCapabilities` marks active backend participation for:

- `Dense`, `Dropout`, `BatchNorm`, `LayerNorm`, `MultiHeadAttention`,
- recurrent/sequence stack (`LSTM`, `GRU`, `TimeDistributed`),
- transformer stack (`TransformerEncoder`, `PositionalEncoding`, `TransformerDecoder`),
- `Embedding` and `RNN`-related control for supported paths.

### 51.5.5 Design implication

Gap handling is intentionally explicit and centralized:

- unsupported nodes are first represented in capability tables,
- surfaced as compile errors or runtime errors with stable reasons,
- easy to migrate by shifting entries from fail-closed/unsupported lists to operator or legacy backends when implementation lands.

## 51.6 Legacy alias and compatibility policy

`GetPipelineLegacyAliasDecisionCapabilities` captures name migration without forcing immediate contract breaks:

- some aliases are `NormalizeToCanonical` (shared executor/parameter contract),
- some are `HiddenCompatibilityAlias` (legacy behavior kept but hidden from canonical migration surface),
- each alias entry carries a reason explaining contract drift and migration status.

This is a deliberate complexity boundary: old project files and node names keep loading while new canonical paths mature.

## 51.7 End-to-end fail-closed contract

```text
Graph save/load -> GraphCompiler
  -> unsupported model layer? ---- yes --> Issue(ERROR) and block compile
  -> unsupported control node? ---- yes --> Issue(ERROR) and block compile
  -> else continue -> runtime config

Runtime Start -> PipelineExecutor::ValidatePipeline
  -> runtime_type resolved by capability tables
  -> unknown / not supported -> runtime error
  -> fail-closed -> FailUnsupportedNode + abort
  -> operator-backed -> ExecuteOperator(...)
  -> legacy typed executor -> ExecuteTypedLegacyNode(...)
```

## 51.8 Evidence anchors

| Claim family | Source |
|---|---|
| Runtime mode tables and capability structures | `cyxwiz-engine/src/core/pipeline_runtime_capabilities.h:24-96`, `cyxwiz-engine/src/core/pipeline_runtime_capabilities.h:122-214` |
| Fail-closed resolution priority and reason propagation | `cyxwiz-engine/src/core/pipeline_runtime_capabilities.cpp:888-948`, `cyxwiz-engine/src/core/pipeline_runtime_capabilities.cpp:1005-1007` |
| Unsupported and supported training capability tables | `cyxwiz-engine/src/core/pipeline_runtime_capabilities.cpp:687-751`, `cyxwiz-engine/src/core/pipeline_runtime_capabilities.cpp:761-812` |
| Training support resolution and reason predicates | `cyxwiz-engine/src/core/pipeline_runtime_capabilities.cpp:1073-1075`, `cyxwiz-engine/src/core/pipeline_runtime_capabilities.cpp:1313-1330`, `cyxwiz-engine/src/core/pipeline_runtime_capabilities.cpp:1362-1395` |
| Graph compiler unsupported layer/control enforcement | `cyxwiz-engine/src/core/graph_compiler.cpp:1474-1478`, `cyxwiz-engine/src/core/graph_compiler.cpp:2651-2665`, `cyxwiz-engine/src/core/graph_compiler.cpp:2831`, `cyxwiz-engine/src/core/graph_compiler.cpp:4130` |
| Pipeline executor schema and fail-closed runtime enforcement | `cyxwiz-engine/src/core/pipeline_executor.cpp:3148-3155`, `cyxwiz-engine/src/core/pipeline_executor.cpp:3211-3213`, `cyxwiz-engine/src/core/pipeline_executor.cpp:3294-3300`, `cyxwiz-engine/src/core/pipeline_executor.cpp:3394-3403` |
| Alias mapping for legacy compatibility names | `cyxwiz-engine/src/core/pipeline_runtime_capabilities.cpp:294-419`, `cyxwiz-engine/src/core/pipeline_runtime_capabilities.cpp:1040-1060` |
