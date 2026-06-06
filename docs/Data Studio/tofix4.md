# To Fix 4 - Frontend Node vs Backend Implementation Audit

This document audits CyxWiz node coverage from the frontend node system
down into the current backend/compiler/execution paths.

The goal is to identify nodes that are:

- exposed in the frontend as usable
- marked as implemented in metadata
- but missing a real backend path
- or only wired to placeholder / passthrough execution
- or silently ignored/dropped by compilation/model building

This is the dangerous class of issue that makes a node look available
but produces invalid graphs, silent no-ops, or misleading outputs.

---

## Executive Summary

The main issue is not just "some nodes are missing."

The real problem is that CyxWiz currently has multiple node execution
surfaces with inconsistent coverage:

1. training graph compilation via `GraphCompiler`
2. sequential model construction via `ModelBuilder`
3. Data Studio execution via `PipelineExecutor`
4. newer operator-based execution via `node_executors/*`

Some nodes are:

- recognized by `GraphCompiler` but not buildable by `ModelBuilder`
- exposed in the UI but not recognized as real model layers
- marked `Implemented` in metadata but only run placeholder passthrough
- backed by newer operators, while the active executor still calls the
  old placeholder path

That means users can build graphs that look valid in the frontend but
do not execute truthfully.

---

## Priority 0: Nodes That Should Be Hidden or Hard-Blocked Right Now

### 1. `Conv2D`, `MaxPool2D`, `AvgPool2D`, `GlobalMaxPool`, `GlobalAvgPool`

**Severity:** High

**Status:** Fixed in current branch.

Implemented:

- `GraphCompiler` now hard-blocks these nodes when they are on the selected training path.
- the error names the compiler/model-builder mismatch instead of allowing `ModelBuilder` to skip the layer.
- `test_graph_compiler_deferred_nodes` verifies each affected CNN/pooling node fails compile before model construction.

Relevant files:

- `cyxwiz-engine/src/core/graph_compiler.cpp:922`
- `cyxwiz-engine/src/core/model_builder.cpp:296`

Problem:

- `GraphCompiler::IsModelLayer()` treats these as valid model layers.
- `ModelBuilder` does not actually build them into `SequentialModel`.
- `ModelBuilder` explicitly warns that CNN layers are not yet supported
  in `SequentialModel`.

Effect:

- graph can compile as if CNN layers are valid
- model build silently skips them
- downstream shape assumptions can become wrong
- users may get runtime mismatch or a model that is not the graph they built

**Recommendation:**

- either fully implement CNN module wrappers in `ModelBuilder`
- or block these nodes from training graphs until the backend path is real

---

### 2. `ConvTranspose2D`, `Upsample`, `PixelShuffle`

**Severity:** High

**Status:** Fixed in current branch.

Implemented:

- `GraphCompiler` now hard-blocks these upsampling nodes on the selected training path.
- loaded/imported graphs are covered even when a node has no metadata registration.
- `test_graph_compiler_deferred_nodes` verifies each affected node fails compile with the backend-gap message.

Relevant files:

- `cyxwiz-engine/src/core/graph_compiler.cpp:931`
- `cyxwiz-engine/src/core/graph_compiler.cpp:1340`
- `cyxwiz-engine/src/core/model_builder.cpp:303`

Problem:

- `GraphCompiler` recognizes them as model layers and even extracts
  some parameters / shapes.
- `ModelBuilder` has no implementation case for them.
- they fall into the generic unknown-layer path and are not added to the model

Effect:

- frontend suggests these nodes are available
- compile/model-build contract is broken
- model execution does not match the graph

**Recommendation:**

- hard-block them for training until model-builder support exists

---

### 3. `MultiHeadAttention`

**Severity:** High

**Status:** Fixed in current branch.

Implemented:

- `MultiHeadAttention` metadata is now `Template`, so add-search and pattern instantiation treat it as planned/unavailable instead of implemented.
- the Nodes > Add Layer > Attention quick-add entry is disabled, preventing toolbar bypass of the metadata guard.
- `test_pattern_template_guard` verifies patterns containing `MultiHeadAttention` are rejected without creating partial graphs.

Relevant files:

- `cyxwiz-engine/src/core/node_metadata_registry.cpp:951`
- `cyxwiz-engine/src/gui/main_window.cpp:872`
- `cyxwiz-engine/src/core/model_builder.cpp`

Problem:

- the node is registered as implemented
- the toolbar explicitly adds it from the main window
- but `ModelBuilder` has no `MultiHeadAttention` case
- `GraphCompiler::IsModelLayer()` also does not include it

Effect:

- the frontend directly promotes a node that the training backend does
  not actually build
- this is one of the clearest UI/backend mismatches in the project

**Recommendation:**

- remove it from the quick-add toolbar until there is a real backend path
- or implement proper attention-module support end to end

---

### 4. `RNN` and `Bidirectional`

**Severity:** High

**Status:** Fixed in current branch.

Implemented:

- `GraphCompiler` now hard-blocks `RNN` and `Bidirectional` on the selected training path because `ModelBuilder` has no corresponding module cases.
- the guard is independent of metadata status, so imported graphs cannot bypass it.
- `test_graph_compiler_deferred_nodes` verifies both node types fail compile before model construction.

Relevant files:

- `cyxwiz-engine/src/core/graph_compiler.cpp:943`
- `cyxwiz-engine/src/core/model_builder.cpp`

Problem:

- `GraphCompiler::IsModelLayer()` includes both
- `ModelBuilder` has no implementation cases for either one

Effect:

- they are treated as real recurrent layers during compile
- but dropped during model construction

**Recommendation:**

- hard-block until `ModelBuilder` and runtime modules exist

---

## Priority 1: Training Nodes Exposed in UI but Not Fully Wired

### 5. `RMSprop`, `Adagrad`, `NAdam`

**Severity:** High

**Status:** Fixed in current branch.

Implemented:

- `GraphCompiler` now treats `RMSprop`, `Adagrad`, and `NAdam` as supported optimizer nodes.
- `TrainingConfiguration::GetOptimizerType()` maps them to backend `OptimizerType::RMSprop`,
  `OptimizerType::AdaGrad`, and `OptimizerType::NAdam`.
- `TrainingConfiguration::GetOptimizerName()` preserves their UI names instead of falling back to `Adam`.
- validation messages list the complete supported optimizer set.
- graph training launch legacy loop-parameter handling recognizes `Adagrad` and `NAdam` too.
- `test_graph_compiler_deferred_nodes` verifies compile acceptance, optimizer id selection,
  backend type mapping, name mapping, and learning-rate extraction for all six exposed optimizers.

Relevant files:

- `cyxwiz-engine/src/core/graph_compiler.cpp:88`
- `cyxwiz-engine/src/core/graph_compiler.cpp:900`
- `cyxwiz-engine/src/core/graph_compiler.cpp:1964`
- `cyxwiz-engine/src/core/graph_compiler.h:309`

Problem:

- `GraphCompiler` error messaging still says the graph must have
  `SGD`, `Adam`, or `AdamW`
- `FindOptimizerNode()` only returns those three nodes
- later validation logic knows that `RMSprop`, `Adagrad`, and `NAdam`
  are optimizer nodes
- `TrainingConfiguration::GetOptimizerType()` and
  `GetOptimizerName()` only map `SGD`, `Adam`, and `AdamW`
- the backend library itself does support more optimizers

Effect:

- frontend exposes optimizer nodes the backend can support
- engine compile path still rejects or downgrades them

This is a wiring bug, not a backend limitation.

**Recommendation:**

- extend `FindOptimizerNode()`
- extend `TrainingConfiguration::GetOptimizerType()`
- extend `GetOptimizerName()`
- audit all optimizer-specific parameter extraction

---

### 6. Learning-rate scheduler nodes are UI-only right now

Affected nodes:

- `StepLR`
- `CosineAnnealing`
- `ReduceOnPlateau`
- `ExponentialLR`
- `WarmupScheduler`

**Severity:** High

Relevant files:

- `cyxwiz-engine/src/core/node_metadata_registry.cpp:1052`
- `cyxwiz-engine/src/core/graph_compiler.h`
- `cyxwiz-engine/src/core/model_builder.cpp`

Problem:

- scheduler nodes are registered as implemented in metadata
- they are configurable in frontend node UI
- there is no real compile-to-runtime scheduler path in core training

Effect:

- users can add scheduler nodes that look first-class
- training backend ignores them

**Recommendation:**

- either implement scheduler propagation into training execution
- or mark these nodes as unavailable/experimental

---

### 7. Regularization nodes are exposed but not part of training execution

Affected nodes:

- `L1Regularization`
- `L2Regularization`
- `ElasticNet`

**Severity:** High

Relevant files:

- `cyxwiz-engine/src/gui/node_editor_nodes.cpp`
- `cyxwiz-engine/src/core/graph_compiler.cpp`
- `cyxwiz-engine/src/core/model_builder.cpp`

Problem:

- these nodes exist in frontend/editor/docs
- they are not wired through compile/model/training execution

Effect:

- they communicate capability that the engine does not actually apply

**Recommendation:**

- block them from production graphs until regularization semantics are
  implemented

---

## Priority 2: Nodes Marked Implemented but Running Placeholder Logic

These are especially risky because they do not fail fast. They often
return success while passing input through unchanged.

### 8. Analytics / ML nodes with placeholder `PipelineExecutor` behavior

Affected nodes include:

- `KMeansCluster`
- `DBSCANCluster`
- `HierarchicalCluster`
- `GMMCluster`
- `PCANode`
- `TSNENode`
- `DecisionTreeClassifier`
- `RandomForestClassifier`
- `SVMClassifier`
- `KNNClassifier`
- `LogisticRegressionNode`
- `LinearRegressionNode`
- `PolynomialRegressionNode`

**Severity:** High

Relevant files:

- `cyxwiz-engine/src/core/pipeline_executor.cpp:2802`
- `cyxwiz-engine/src/core/pipeline_executor.cpp:2838`
- `cyxwiz-engine/src/core/pipeline_executor.cpp:2864`
- `cyxwiz-engine/src/core/pipeline_executor.cpp:1884`
- `cyxwiz-engine/src/core/pipeline_executor.cpp:2916`
- `cyxwiz-engine/src/core/pipeline_executor.cpp:3124`
- `cyxwiz-engine/src/core/pipeline_executor.cpp:3150`
- `cyxwiz-engine/src/core/pipeline_executor.cpp:3176`

Problem:

- many of these functions log `Placeholder` or `Passing through data unchanged`
- they still return success and register output datasets

Effect:

- users may believe the algorithm executed
- downstream nodes receive unchanged data or fake outputs
- debugging is harder because the graph appears to work

**Recommendation:**

- do not silently pass through for ML algorithm nodes
- fail with a clear "not implemented" execution error instead
- only return success when the actual algorithm path exists

---

### 9. Evaluation nodes with placeholder success paths

Affected nodes:

- `ConfusionMatrixNode`
- `ROCCurveNode`
- `LearningCurvesNode`
- `FeatureImportanceNode`
- `CrossValidationNode`

**Severity:** High

Relevant files:

- `cyxwiz-engine/src/core/pipeline_executor.cpp:3232`
- `cyxwiz-engine/src/core/pipeline_executor.cpp:3239`
- `cyxwiz-engine/src/core/pipeline_executor.cpp:3253`
- `cyxwiz-engine/src/core/pipeline_executor.cpp:3260`
- `cyxwiz-engine/src/core/pipeline_executor.cpp:3267`

Problem:

- these functions are explicit placeholder implementations
- they still report outputs and success

Effect:

- frontend can display a completed evaluation workflow with no real evaluation

**Recommendation:**

- treat them as unimplemented, not successful placeholders

---

### 10. Text analytics nodes have real operators in one path, but placeholder logic in another

Affected nodes:

- `TFIDFVectorizer`
- `CountVectorizer`
- `SentimentAnalyzer`

**Severity:** High

Relevant files:

- `cyxwiz-engine/src/core/node_executors/pipeline_operator_factory.cpp:58`
- `cyxwiz-engine/src/core/pipeline_executor.cpp:4110`
- `cyxwiz-engine/src/core/pipeline_executor.cpp:4136`
- `cyxwiz-engine/src/core/pipeline_executor.cpp:4188`

Problem:

- `node_executors/*` contains real operator implementations
- `PipelineOperatorFactory` registers them
- but `PipelineExecutor` still uses placeholder passthrough behavior

Effect:

- duplicate execution systems disagree
- the newer real path is undermined by the older placeholder path

**Recommendation:**

- pick one execution path as canonical
- route these nodes through the real operator implementation
- remove placeholder executor branches once migrated

---

### 11. Time-series analysis nodes have the same split-brain problem

Affected nodes:

- `TimeSeriesDecomposition`
- `ARIMAForecaster`
- `ExponentialSmoothing`

**Severity:** High

Relevant files:

- `cyxwiz-engine/src/core/node_executors/pipeline_operator_factory.cpp:135`
- `cyxwiz-engine/src/core/pipeline_executor.cpp`

Problem:

- operator implementations exist and are registered in
  `PipelineOperatorFactory`
- but the main `PipelineExecutor` dispatch does not provide a matching,
  obvious active execution path for them

Effect:

- node support depends on which executor path is actually used
- coverage is inconsistent and easy to misread

**Recommendation:**

- unify executor routing
- do not keep "registered real operator" and "legacy placeholder path"
  in parallel for the same node family

---

## Priority 3: DNN Nodes That Look Real But Have No Visible Backend Execution Path

Affected nodes:

- `DNNModelLoad`
- `DNNDetect`
- `PretrainedYOLO`

**Severity:** High

Relevant files:

- `cyxwiz-engine/src/core/node_metadata_registry.cpp:1069`
- `cyxwiz-engine/src/core/node_metadata_registry.cpp:1075`
- `cyxwiz-engine/src/core/node_metadata_registry.cpp:1082`
- no corresponding training/pipeline execution path found in:
  - `graph_compiler.cpp`
  - `model_builder.cpp`
  - `pipeline_executor.cpp`

Problem:

- metadata marks them as implemented
- they are visible as real DNN nodes
- there is no clear end-to-end execution path in the current node backend

Effect:

- these nodes over-promise capability in the graph UI

**Recommendation:**

- either wire them into a real inference execution path
- or mark them as template / external / experimental

---

## Priority 4: Data / Utility Nodes With Weak or Misleading Execution Contracts

### 12. `DataProfiler`

**Severity:** Medium

Relevant files:

- `cyxwiz-engine/src/core/node_metadata_registry.cpp:903`
- `cyxwiz-engine/src/core/pipeline_executor.cpp:4291`

Problem:

- node is marked implemented
- executor currently just logs input shape and passes data through

Effect:

- this is not actual profiling behavior

**Recommendation:**

- either implement real profiling output
- or surface it as a report/panel action rather than a fake transform node

---

### 13. Utility nodes returning placeholder success

Affected nodes:

- `CalculatorNode`
- `UnitConverter`
- `RegexTester`
- `JSONPathExtractor`

**Severity:** Medium

Relevant files:

- `cyxwiz-engine/src/core/pipeline_executor.cpp:4215`
- `cyxwiz-engine/src/core/pipeline_executor.cpp:4230`
- `cyxwiz-engine/src/core/pipeline_executor.cpp:4240`

Problem:

- several utility nodes still use placeholder logic or passthrough behavior

Effect:

- same issue as analytics nodes: apparent success without real work

**Recommendation:**

- fail closed until implemented

---

## Priority 5: Template Nodes Are Correctly Blocked, But Need Import-Time Guardrails

Template nodes in metadata are currently blocked in add-search:

- `cyxwiz-engine/src/gui/node_editor_add_search.cpp`

This is good. However:

- old graphs
- imported graphs
- hand-edited project files

can still contain template node types.

**Recommendation:**

- add compile-time/project-load validation that rejects template nodes
- do not rely only on UI add-search blocking

---

## Most Important Structural Problem

The deepest issue is duplicated execution architecture:

- `GraphCompiler` / training path
- `PipelineExecutor`
- `PipelineOperatorFactory` / `node_executors`

These do not currently express one coherent source of truth for node
support.

That leads directly to:

- nodes recognized in compile but not model build
- real operators hidden behind legacy placeholders
- nodes marked implemented in metadata without an executable path

---

## Recommended Fix Order

### Phase 1: Prevent user harm

1. Hard-block nodes that compile into dropped/ignored training layers:
   - `Conv2D`
   - `MaxPool2D`
   - `AvgPool2D`
   - `GlobalMaxPool`
   - `GlobalAvgPool`
   - `ConvTranspose2D`
   - `Upsample`
   - `PixelShuffle`
   - `MultiHeadAttention`
   - `RNN`
   - `Bidirectional`
   - `PolicyNetwork`
   - `ValueNetwork`

2. Block unsupported optimizers/schedulers/regularization nodes until
   compile/runtime support is complete.

3. Change placeholder executor nodes to fail loudly instead of returning
   fake success.

### Phase 2: Unify execution

1. Choose the canonical execution path for Data Studio nodes.
2. Route nodes with real `node_executors` through that path.
3. Remove placeholder duplicates from `PipelineExecutor`.

### Phase 3: Make metadata honest

1. Audit every `NodeImplementationStatus::Implemented`.
2. Downgrade nodes to template/experimental where the backend path is
   not real.
3. Add automated coverage checks so metadata cannot drift from runtime
   support again.

---

## Best First Engineering Tasks

### Task 1: Training node safety pass

Scope:

- `graph_compiler.cpp`
- `model_builder.cpp`
- node availability gating

Deliverables:

- unsupported training nodes blocked before execution
- clear user-facing error messages

### Task 2: Optimizer coverage fix

Scope:

- `FindOptimizerNode`
- `TrainingConfiguration::GetOptimizerType`
- `TrainingConfiguration::GetOptimizerName`

Deliverables:

- `RMSprop`, `Adagrad`, `NAdam` fully recognized end to end

### Task 3: Placeholder-node audit

Scope:

- `pipeline_executor.cpp`

Deliverables:

- every placeholder node either:
  - fails explicitly, or
  - is migrated to a real operator

### Task 4: Metadata truth pass

Scope:

- `node_metadata_registry.cpp`

Deliverables:

- no node marked implemented without a real execution contract

---

## Closing Assessment

Right now the engine has a node-coverage truth problem.

The frontend node system is richer than the actual backend execution
contract. That is acceptable during development only if the UI is honest
about it.

At the moment, some nodes are:

- over-exposed
- silently ignored
- placeholder-backed
- or routed through the wrong execution layer

`tofix4.md` should be treated as a node-safety backlog before more node
surface is added to the frontend.
