# To Fix 7 - CyxWiz Node Support Matrix

This document is the concrete support matrix that should sit behind
CyxWiz node visibility, compile validation, and runtime truth.

It answers a simple question:

`For a given node, what does the system actually support today?`

The matrix is based on the current frontend surface plus the active
engine paths:

- frontend visibility / documentation
- `GraphCompiler`
- `ModelBuilder`
- `PipelineExecutor`
- `PipelineOperatorFactory`

---

## Status Legend

- `Visible`: shown in UI, toolbar, menus, or documentation
- `Compiler`: recognized by `GraphCompiler` for training graphs
- `Builder`: actually built into the training model by `ModelBuilder`
- `Pipeline`: has a `PipelineExecutor` path
- `Operator`: has a real `IPipelineOperator` registration
- `Truth`:
  - `Real` = usable end to end on the current primary path
  - `Partial` = implemented on one path but not consistently surfaced
  - `Blocked` = visible or known, but compile/runtime now fails closed
  - `Placeholder` = returns success / passthrough / stub behavior
  - `Misleading` = visible or compiler-accepted beyond real support

---

## Executive Summary

The main problem is not lack of nodes. It is keeping the existing support
contract complete across every execution path.

Current engine truth:

- `PipelineRuntimeCapabilities` already acts as the central support contract
  for Data Studio / `PipelineExecutor` runtime support.
- `NodeMetadataRegistry`, Data Studio `NodeRegistry`, `PipelineExecutor`,
  `PipelineMaterializer`, and `GraphCompiler` already consume that capability
  layer.
- The lean path is to extend that existing capability layer with clearer
  training axes, not to create a second support matrix.

Current recurring patterns:

1. Nodes are `Visible` but not really buildable.
2. Several previously compiler-accepted nodes are now hard-blocked before
   `ModelBuilder`, which is truthful but still leaves UI/product labeling
   work.
3. Nodes have a real `Operator` implementation, but the active runtime still
   carries both operator-backed and legacy-executor ownership.
4. Training truth is split between compiler support, model-builder support,
   loss/optimizer config support, and unsupported training controls.
5. Some nodes only make sense in `Classic ML` workflows, some only in
   `Deep Learning`, but the current UI does not enforce that clearly.

Design rule:

- Do not build a duplicate matrix object.
- Extend `PipelineRuntimeCapabilities` and metadata support axes so UI,
  compiler, runtime, and docs read from the same source of truth.

---

## A. Training Graph Nodes

### Core deep-learning training path

| Node | Visible | Compiler | Builder | Truth | Notes | Recommended action |
|---|---|---:|---:|---|---|---|
| `Dense` | Yes | Yes | Yes | `Real` | Core training path works. | Keep visible. |
| `Dropout` | Yes | Yes | Yes | `Real` | Supported in `SequentialModel`. | Keep visible. |
| `Flatten` | Yes | Yes | Yes | `Real` | Supported in `SequentialModel`. | Keep visible. |
| `BatchNorm` | Yes | Yes | Yes | `Real` | Supported in current builder path. | Keep visible. |
| Activations (`ReLU`, `Sigmoid`, `Tanh`, etc.) | Yes | Yes | Yes | `Real` | Core activation path is present. | Keep visible. |
| `MSELoss` | Yes | Yes | Yes | `Real` | Supported by loss builder. | Keep visible. |
| `CrossEntropyLoss` | Yes | Yes | Yes | `Real` | Supported by loss builder. | Keep visible. |
| `BCELoss` | Yes | Yes | Yes | `Real` | Supported by loss builder. | Keep visible. |
| `BCEWithLogits` | Yes | Yes | Yes | `Real` | Supported by loss builder. | Keep visible. |
| `SGD` | Yes | Yes | Yes | `Real` | Compiler + builder path aligned. | Keep visible. |
| `Adam` | Yes | Yes | Yes | `Real` | Compiler + builder path aligned. | Keep visible. |
| `AdamW` | Yes | Yes | Yes | `Real` | Compiler + builder path aligned. | Keep visible. |

### Additional optimizer nodes now wired

| Node | Visible | Compiler | Builder | Truth | Notes | Recommended action |
|---|---|---:|---:|---|---|---|
| `RMSprop` | Yes | Yes | Yes | `Real` | Optimizer node is recognized by `GraphCompiler`, maps to `OptimizerType::RMSprop`, and is covered by compiler regression tests. | Keep visible. |
| `Adagrad` | Yes | Yes | Yes | `Real` | Optimizer node is recognized by `GraphCompiler`, maps to `OptimizerType::AdaGrad`, and is covered by compiler regression tests. | Keep visible. |
| `NAdam` | Yes | Yes | Yes | `Real` | Optimizer node is recognized by `GraphCompiler`, maps to `OptimizerType::NAdam`, and is covered by compiler regression tests. | Keep visible. |

### Deep-learning architecture nodes that overstate support

| Node | Visible | Compiler | Builder | Truth | Notes | Recommended action |
|---|---|---:|---:|---|---|---|
| `Conv2D` | Template/blocked | Blocked with error | No | `Blocked` | Compiler and metadata now fail closed before `ModelBuilder` can silently miss CNN support. | Implement CNN wrappers before unblocking. |
| `MaxPool2D` | Template/blocked | Blocked with error | No | `Blocked` | Compiler and metadata now fail closed. | Implement CNN wrappers before unblocking. |
| `AvgPool2D` | Template/blocked | Blocked with error | No | `Blocked` | Compiler and metadata now fail closed. | Implement CNN wrappers before unblocking. |
| `GlobalMaxPool` | Template/blocked | Blocked with error | No | `Blocked` | Visible/documented beyond build support, but no longer accepted as trainable. Metadata now marks it blocked. | Implement before unblocking. |
| `GlobalAvgPool` | Template/blocked | Blocked with error | No | `Blocked` | Visible/documented beyond build support, but no longer accepted as trainable. Metadata now marks it blocked. | Implement before unblocking. |
| `ConvTranspose2D` | Template/blocked | Blocked with error | No | `Blocked` | No longer accepted as trainable; metadata now marks it blocked. | Implement before unblocking. |
| `Upsample` | Template/blocked | Blocked with error | No | `Blocked` | No longer accepted as trainable; metadata now marks it blocked. | Implement before unblocking. |
| `PixelShuffle` | Template/blocked | Blocked with error | No | `Blocked` | No longer accepted as trainable; metadata now marks it blocked. | Implement before unblocking. |
| `MultiHeadAttention` | Yes/blocked | Blocked with error | No | `Blocked` | Pattern/template guards and the central training support table prevent unavailable attention from compiling as a trainable layer. | Add end-to-end support before unblocking. |
| `SelfAttention` / `CrossAttention` / `LinearAttention` | Yes/blocked | Blocked with error | No | `Blocked` | Visible/analyzable attention concepts now fail closed in selected training paths. | Add backend modules before unblocking. |
| `LayerNorm` | Yes/blocked | Blocked with error | No | `Blocked` | Visible/analyzable normalization concept now fails closed before `ModelBuilder`. | Add backend module before unblocking. |
| `GroupNorm` | Yes/blocked | Blocked with error | No | `Blocked` | Same issue. | Add backend module before unblocking. |
| `InstanceNorm` | Yes/blocked | Blocked with error | No | `Blocked` | Same issue. | Add backend module before unblocking. |

### Recurrent training nodes

| Node | Visible | Compiler | Builder | Truth | Notes | Recommended action |
|---|---|---:|---:|---|---|---|
| `LSTM` | Yes | Yes | Yes | `Partial` | Sequential text/time-series path exists; metadata now correctly marks it implemented. Broader sequence-output tasks such as NER are not complete. | Keep visible with task limits. |
| `GRU` | Yes | Yes | Yes | `Partial` | Sequential text/time-series path exists; metadata now correctly marks it implemented. Broader sequence-output tasks are not complete. | Keep visible with task limits. |
| `RNN` | Yes | Blocked with error | No | `Blocked` | Compiler now fails closed because the sequential builder path is not ready. | Implement or keep hidden. |
| `Bidirectional` | Yes | Blocked with error | No | `Blocked` | Compiler now fails closed; NER-style bidirectional tagging still needs first-class contracts. | Implement under `tofix14.md` / `tofix19.md`. |
| `TimeDistributed` | Yes | Yes, shape-guarded | Yes | `Partial` | Builds as `TimeDistributedDenseModule` for sequence-shaped inputs and rejects non-sequence tensors. Backend placement now reports an explicit `timedistributed_sequence_wrapper` unknown-placement reason instead of treating it as unclassified. Full NER still needs sequence-batch executor wiring and token metrics. | Keep visible with task limits. |

---

## B. Classic ML / Data Studio Model Nodes

### Nodes with the strongest current story

| Node | Visible | Pipeline | Operator | Truth | Notes | Recommended action |
|---|---|---:|---:|---|---|---|
| `LinearRegressionNode` | Yes | Yes | Yes | `Partial` | Real operator exists, but legacy executor also has its own path. | Route canonical execution through operator path. |
| `PolynomialRegressionNode` | Yes | Yes | Yes | `Partial` | Same as above. | Route through operator path. |

### Nodes that are still unsupported, but now fail closed in the legacy executor

| Node | Visible | Pipeline | Operator | Truth | Notes | Recommended action |
|---|---|---:|---:|---|---|---|
| `LogisticRegression` | Template/blocked | Fails closed | No | `Blocked` | Legacy placeholder success was replaced with explicit unsupported error; metadata no longer marks it implemented. | Hide or implement. |
| `DecisionTree` | Template/blocked | Fails closed | No | `Blocked` | Legacy placeholder success was replaced with explicit unsupported error; metadata no longer marks it implemented. | Hide or implement. |
| `RandomForest` | Template/blocked | Fails closed | No | `Blocked` | Legacy placeholder success was replaced with explicit unsupported error; metadata no longer marks it implemented. | Hide or implement. |
| `GradientBoosting` | Template/blocked | Fails closed | No | `Blocked` | Legacy placeholder success was replaced with explicit unsupported error; metadata now marks it blocked. | Hide or implement. |
| `SVM` | Template/blocked | Fails closed | No | `Blocked` | Legacy placeholder success was replaced with explicit unsupported error; metadata no longer marks it implemented. | Hide or implement. |
| `KNN` | Template/blocked | Fails closed | No | `Blocked` | Legacy placeholder success was replaced with explicit unsupported error; metadata no longer marks it implemented. | Hide or implement. |
| `NaiveBayes` | Template/blocked | Fails closed | No | `Blocked` | Legacy placeholder success was replaced with explicit unsupported error; metadata now marks it blocked. | Hide or implement. |

---

## C. Preprocessing and Feature Engineering Nodes

### Operator-backed and directionally correct

| Node | Visible | Pipeline | Operator | Truth | Notes | Recommended action |
|---|---|---:|---:|---|---|---|
| `TimeSeriesWindow` | Yes | Yes | Yes | `Partial` | Operator-backed and important for training graphs; materializer path still limited by dataset mode. | Keep visible, but standardize canonical execution. |
| `TimeSeriesSplit` | Yes | Yes | Yes | `Partial` | Real operator exists. | Keep visible; add parity tests. |
| `TimeSeriesFeatures` | Yes | Yes | Yes | `Partial` | Real operator exists. | Keep visible; route through operator path first. |
| `LogTransform` | Yes | Yes | Yes | `Partial` | Real operator exists. | Same as above. |
| `Differencing` | Yes | Yes | Yes | `Partial` | Real operator exists. | Same as above. |
| `TextTokenizer` | Yes | Yes | Yes | `Partial` | Real operator exists, but text flow still depends on data mode. | Keep visible with capability messaging. |
| `TFIDFVectorizer` | Yes | Fails closed in legacy / Yes in operator path | Yes | `Partial` | Real operator exists; legacy fake success is blocked, but canonical routing is not unified. | Route through operator path. |
| `CountVectorizer` | Yes | Fails closed in legacy / Yes in operator path | Yes | `Partial` | Same split support issue. | Route through operator path. |
| `SentimentAnalyzer` | Yes | Fails closed in legacy / Yes in operator path | Yes | `Partial` | Real operator exists; legacy fake success is blocked. | Route through operator path. |
| `PCANode` | Yes | Fails closed in legacy / Yes in operator path | Yes | `Partial` | Real operator exists; legacy fake success is blocked. | Route through operator path. |
| `KMeansCluster` | Yes | Yes | Yes | `Partial` | Operator exists. | Keep visible; prefer operator path. |
| `DBSCANCluster` | Yes | Yes | Yes | `Partial` | Operator exists. | Same. |
| `HierarchicalCluster` | Yes | Yes | Yes | `Partial` | Operator exists. | Same. |
| `GMMCluster` | Yes | Yes | Yes | `Partial` | Operator exists. | Same. |
| `FFTNode` | Yes | Yes | Yes | `Partial` | Operator exists, legacy executor also exists. | Converge to operator path. |
| `Convolution1D` | Yes | Yes | Yes | `Partial` | Operator exists. | Converge to operator path. |
| `FilterDesigner` | Yes | Yes | Yes | `Partial` | Operator exists. | Converge to operator path. |
| `StandardScaler` | Yes | Fails closed in legacy / Yes in operator path | Yes | `Partial` | Operator exists; legacy fake success is blocked. | Route through operator path and remove dead placeholder branch. |
| `MinMaxScaler` | Yes | Fails closed in legacy / Yes in operator path | Yes | `Partial` | Same split support issue. | Same. |
| `RobustScaler` | Yes | Fails closed in legacy / Yes in operator path | Yes | `Partial` | Same split support issue. | Same. |
| `LabelEncoder` | Yes | Yes | Yes | `Partial` | Operator exists. | Same. |
| `OrdinalEncoder` | Yes | Yes | Yes | `Partial` | Operator exists. | Same. |
| `TargetEncoder` | Yes | Yes | Yes | `Partial` | Operator exists. | Same. |
| `OutlierDetector` | Yes | Yes | Yes | `Partial` | Operator exists. | Same. |
| `TimeSeriesDecomposition` | Yes | Yes | Yes | `Partial` | Operator exists. | Same. |
| `ARIMAForecaster` | Yes | Yes | Yes | `Partial` | Operator exists. | Same. |
| `ExponentialSmoothing` | Yes | Yes | Yes | `Partial` | Operator exists. | Same. |

### Nodes that are still unsupported in graph execution

| Node | Visible | Pipeline | Operator | Truth | Notes | Recommended action |
|---|---|---:|---:|---|---|---|
| `t-SNE` | Template/blocked | Fails closed | No | `Blocked` | Legacy placeholder success was replaced with explicit unsupported error; metadata no longer marks it implemented. | Hide or implement. |
| `DataProfiler` | Template/UI-only | Fails closed | No | `Blocked/UI-only` | Now treated as a panel/report workflow, not a transform. | Keep as panel or implement real output. |
| `Regex` | Template/blocked | Fails closed | No | `Blocked` | Legacy placeholder success was replaced with explicit unsupported error; metadata no longer marks it implemented. | Hide or implement. |
| `JSONPath` | Template/blocked | Fails closed | No | `Blocked` | Legacy placeholder success was replaced with explicit unsupported error; metadata no longer marks it implemented. | Hide or implement. |

---

## D. Evaluation Nodes

| Node | Visible | Pipeline | Truth | Notes | Recommended action |
|---|---|---:|---|---|---|
| `ConfusionMatrix` | Template/UI-only | Fails closed | `Blocked/UI-only` | Legacy placeholder success was replaced with explicit unsupported error; metadata no longer marks it implemented. | Treat as panel/tool until real graph execution exists. |
| `ROCCurve` | Template/UI-only | Fails closed | `Blocked/UI-only` | Legacy placeholder success was replaced with explicit unsupported error; metadata no longer marks it implemented. | Same. |
| `LearningCurves` | Template/UI-only | Fails closed | `Blocked/UI-only` | Legacy placeholder success was replaced with explicit unsupported error; metadata no longer marks it implemented. | Same. |
| `FeatureImportance` | Template/UI-only | Fails closed | `Blocked/UI-only` | Legacy placeholder success was replaced with explicit unsupported error; metadata no longer marks it implemented. | Same. |
| `CrossValidation` | Template/UI-only | Fails closed | `Blocked/UI-only` | Legacy placeholder success was replaced with explicit unsupported error; metadata no longer marks it implemented. | Same. |

---

## E. Dataset Source and Augmentation Nodes

| Node | Visible | Pipeline | Truth | Notes | Recommended action |
|---|---|---:|---|---|---|
| `ImageFolderDataset` | Yes | Yes | `Placeholder` | Placeholder metadata / dataset creation path. | Hide or mark experimental. |
| `MNISTDataset` | Yes | Yes | `Placeholder` | Placeholder load path. | Hide or mark experimental. |
| `CIFAR10Dataset` | Yes | Yes | `Placeholder` | Placeholder load path. | Hide or mark experimental. |
| `HuggingFaceDataset` | Yes | Yes | `Placeholder` | Placeholder API integration path. | Hide or mark experimental. |
| `KaggleDataset` | Yes | Yes | `Placeholder` | Placeholder API integration path. | Hide or mark experimental. |
| `AugmentationPreset` | Yes | Yes | `Placeholder` | Placeholder transformation. | Hide or mark experimental. |
| `GeometricTransform` | Yes | Yes | `Placeholder` | Placeholder transformation. | Hide or mark experimental. |
| `ColorTransform` | Yes | Yes | `Placeholder` | Placeholder transformation. | Hide or mark experimental. |
| `MorphologyTransform` | Yes | Yes | `Placeholder` | Placeholder transformation. | Hide or mark experimental. |
| `AdvancedAugment` | Yes | Yes | `Placeholder` | Placeholder transformation. | Hide or mark experimental. |
| `IFFT` | Yes | Yes | `Placeholder` | Placeholder branch. | Hide or implement. |
| `WaveletTransform` | Yes | Yes | `Placeholder` | Placeholder branch. | Hide or implement. |

---

## F. Output / Save / Export Nodes

| Node | Visible | Pipeline | Truth | Notes | Recommended action |
|---|---|---:|---|---|---|
| `SaveDataset` | Yes | Yes | `Real` | Data Studio node registry includes it as an output concept. | Keep visible. |
| `ExportCSV` | Yes | Yes | `Real` | Output concept is clear and aligns with the product. | Keep visible. |
| Model export UI | Yes | N/A | `Partial` | UI exists, but model export truth is less clean than dataset export. | Clarify save vs export vs deployable artifact. |

---

## Recommended UI Policy

### Keep visible now

- `Dense`
- `Dropout`
- `Flatten`
- `BatchNorm`
- core activations
- `MSELoss`, `CrossEntropyLoss`, `BCELoss`, `BCEWithLogits`
- `SGD`, `Adam`, `AdamW`
- `DataInput`
- `SaveDataset`, `ExportCSV`
- operator-backed preprocessing and time-series nodes, provided runtime
  routing is corrected

### Hide or hard-block now

- `Conv2D`, `MaxPool2D`, `AvgPool2D`, `GlobalMaxPool`,
  `GlobalAvgPool`, `ConvTranspose2D`, `Upsample`, `PixelShuffle`
- `MultiHeadAttention`
- `LayerNorm`, `GroupNorm`, `InstanceNorm`
- classical ML nodes that are still placeholder-only
- evaluation nodes that are still placeholder-only
- placeholder dataset and augmentation sources

### Mark experimental only if explicitly labeled

- recurrent layers after end-to-end validation
- operator-backed nodes still trapped between old/new runtime paths

---

## Best First Fixes

1. Extend the existing `PipelineRuntimeCapabilities` contract so it can
   describe all required support axes used by:
   - frontend node menus
   - `GraphCompiler`
   - runtime execution entry points
   - docs/help panels

2. Add explicit training support axes instead of treating the current
   supported-training-backend table as a complete training-node matrix:
   - training compile support
   - model-builder layer support (`Model Builder` axis now separates supported
     model layers from blocked sequential-model concepts)
   - activation support (`Activation=supported` is projected from central
     training-role capabilities)
   - loss support (`Loss=supported` is projected from central training-role
     capabilities)
   - optimizer support (`Optimizer=supported` is projected from central
     training-role capabilities)
   - training-control support (`training_control` role now marks unsupported
     scheduler/regularization controls without claiming execution support)

3. Hide, label, or keep hard-blocked every node in this document marked
   `Blocked`, unless there is a real reason to keep it visible as
   experimental or UI-only.

4. Route operator-backed nodes through the operator path first, then
   remove legacy placeholder branches.

5. Split product lanes explicitly into:
   - `Classic ML`
   - `Deep Learning`
   - `Data Studio / Analytics` (`data_studio_analytics` workflow lane now
     covers operator-backed analytics, preprocessing, time-series, and text
     nodes that are not already assigned to Classic ML or Deep Learning)

6. Add automated capability tests for any node labeled `Real` (operator-backed
   real capabilities now assert implemented metadata, non-blocked badge,
   `Support State=real`, owner, runtime, and materializer axes).

---

## Bottom Line

CyxWiz already has many of the right node concepts.

What it does not yet have is one truthful support layer that tells the
user, the compiler, and the runtime the same story.

That is the main job for the next cleanup pass.
