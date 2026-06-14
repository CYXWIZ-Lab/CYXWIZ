# To Fix 4 - Frontend Node vs Backend Implementation Audit

This document audits CyxWiz node coverage from the frontend node system
down into the current backend/compiler/execution paths.

The goal is to identify nodes that are:

- exposed in the frontend as usable
- marked as implemented in metadata
- but missing a real backend path
- historically wired to placeholder / passthrough execution
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
- marked `Implemented` in metadata but not fully wired to a truthful
  backend path
- backed by newer operators, while legacy executor routing still needs
  convergence with that operator path

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

### 4a. RL graph nodes: `GymEnvironment`, `ReplayBuffer`, `PolicyNetwork`, `ValueNetwork`

**Severity:** High

**Status:** Fixed in current branch.

Implemented:

- `PolicyNetwork` and `ValueNetwork` are now listed in the central
  unsupported sequential-model-layer capability table.
- `GymEnvironment`, `ReplayBuffer`, `PolicyNetwork`, and `ValueNetwork`
  are now listed in the central fail-closed PipelineExecutor runtime
  capability table because Data Studio graph execution has no real RL
  runtime path for them.
- `NodeMetadataRegistry` derives their blocked/template metadata from
  the same central runtime/training backend support contracts used by
  the other blocked nodes.
- backend-placement reporting now classifies emitted unsupported
  sequential-model layers as `unsupported` instead of claiming GPU
  placement or falling back to an `unknown` backend capability.
- all unsupported sequential-model entries now count as model layers for
  graph validation, so selected `LayerNorm`/attention-style nodes report
  the backend gap directly instead of also claiming the graph has no
  model layer.
- fail-closed PipelineExecutor capabilities now distinguish graph-runtime
  gaps from training-compiler support with `blocks_metadata_status`; this
  keeps `NERSequenceBuilder` implemented for its selected training path
  while still exposing that PipelineExecutor cannot run it as a graph node.
- selected training paths containing either node still fail compile with
  the explicit reinforcement-learning training contract error.
- side/disconnected RL sketches remain non-blocking for the selected
  supervised training path.
- metadata drift coverage verifies unsupported training nodes are not
  marked implemented, while training-only nodes keep implemented metadata
  when only their PipelineExecutor graph runtime is fail-closed.

Relevant files:

- `cyxwiz-engine/src/core/pipeline_runtime_capabilities.cpp`
- `cyxwiz-engine/src/core/backend_placement_capabilities.h`
- `cyxwiz-engine/src/core/node_metadata_registry.cpp`
- `cyxwiz-engine/src/core/graph_compiler.cpp`
- `cyxwiz-engine/tests/test_pipeline_operator_metadata.cpp`
- `cyxwiz-engine/tests/test_graph_compiler_deferred_nodes.cpp`

Problem:

- the recommended hard-block list already included `PolicyNetwork` and
  `ValueNetwork`
- selected RL paths were blocked by graph-compiler sketch detection
- unsupported layer compile errors and backend-placement reporting could
  disagree, with placement showing GPU-capable or unknown even though
  `ModelBuilder/SequentialModel` cannot execute the layer
- `GymEnvironment`, `ReplayBuffer`, `PolicyNetwork`, and `ValueNetwork`
  still registered as implemented despite having no truthful
  PipelineExecutor graph runtime
- `NERSequenceBuilder` showed the opposite split: it is a real training
  contract node, but its PipelineExecutor graph-runtime gap could make
  metadata look blocked if all fail-closed capabilities shared one status
  rule

Effect:

- the UI/backend support contract could still overstate RL node support
  even though supervised Studio training cannot build/train them and
  Data Studio pipeline execution cannot run them as graph nodes
- truthful metadata now reports unsupported RL nodes as blocked without
  hiding implemented training-only paths such as selected
  `NERSequenceBuilder` compilation
- backend-placement summaries now show `unsupported` for blocked
  sequential-model layers, making diagnostics match the compile verdict
- unsupported layer diagnostics no longer include the unrelated "Graph
  must have at least one model layer" error for visible-but-unimplemented
  layer nodes

**Recommendation:**

- keep them blocked/template until there is a real RL training backend
  contract, including environment stepping, rollout/replay schema,
  policy/value losses, target-network handling, and episodic metrics

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

**Status:** Fixed in current branch.

Implemented:

- `GraphCompiler` now hard-blocks scheduler nodes because they are not wired into training execution.
- the guard covers loaded/imported graphs for `StepLR`, `CosineAnnealing`, `ReduceOnPlateau`, `ExponentialLR`, and `WarmupScheduler`.
- registered `CosineAnnealing` metadata is now `Template`, so metadata-driven UI/pattern paths no longer advertise it as implemented.
- all scheduler metadata in this blocked group now carries the explicit
  `Blocked` badge, including `CosineAnnealing`.
- tests cover both compiler rejection and template-pattern rejection.

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

**Status:** Fixed in current branch.

Implemented:

- `GraphCompiler` now hard-blocks `L1Regularization`, `L2Regularization`, and `ElasticNet`.
- the guard covers loaded/imported graphs even though these node types are not metadata-registered.
- `test_graph_compiler_deferred_nodes` verifies all three fail compile with the training-execution gap message.

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

## Priority 2: Nodes Marked Implemented but Historically Running Placeholder Logic

These were especially risky because they did not fail fast. The audited
legacy `PipelineExecutor` dispatch now fails closed for the listed
placeholder families, and metadata truth has been corrected for the
registered blocked/UI-only nodes. This section remains active for the
canonical operator-routing work.

### 8. Analytics / ML nodes with placeholder `PipelineExecutor` behavior

**Status:** Fixed in current branch for the legacy `PipelineExecutor`
path. Unsupported nodes still fail closed with explicit runtime errors
instead of returning passthrough/fake success. Exact registered
operator-backed names now route through `PipelineOperatorFactory`, and
their stale fail-closed dispatch branches have been removed.
Central fail-closed runtime reasons now describe the current hard-fail
state instead of saying disabled placeholders are "still" running.
Typed fail-closed entries that have `NodeType` values now resolve through
the central runtime support API instead of remaining string-only legacy
names.
The metadata drift guard now only permits untyped fail-closed entries for
documented legacy aliases: `PCA`, `TrainTestSplit`, and `ParquetInput`.

Routed operator-backed nodes include:

- `KMeansCluster`
- `DBSCANCluster`
- `HierarchicalCluster`
- `GMMCluster`
- `PCANode`
- `LinearRegressionNode`
- `PolynomialRegressionNode`

Additional current-branch UI wiring fix:

- Data Studio's quick-add PCA menu entry now instantiates the real
  operator-backed `PCANode` instead of the fail-closed legacy `PCA`
  alias.
- default PCA parameters now match `PCAOperator`
  (`n_components`, `center`, `scale`) instead of carrying the stale
  legacy `variance_threshold` parameter.

Still fail-closed unsupported nodes include:

- `TSNENode`
- `DecisionTreeClassifier`
- `RandomForestClassifier`
- `GradientBoostingClassifier`
- `SVMClassifier`
- `KNNClassifier`
- `NaiveBayesClassifier`
- `LogisticRegressionNode`

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

Problem before the 2026-06-07 runtime-truth pass:

- many of these functions logged `Placeholder` or `Passing through data unchanged`
- they returned success and registered output datasets

Current remaining problem:

- real operator-backed implementations are now the active path for exact
  registered names, but broader runtime ownership is still split

Effect now:

- fake-success user harm is fixed for the audited legacy dispatch path
- user-facing runtime/metadata reasons no longer imply that disabled
  passthrough placeholders are active behavior
- metadata/runtime checks can now resolve enum support for fail-closed
  nodes such as `UMAPNode`, `SVMRegressor`, `PRCurveNode`,
  `RegressionMetricsNode`, `WordEmbeddings`, `NamedEntityRecognizer`,
  image dataset loaders, and augmentation/image-preprocessing nodes
- future fail-closed additions must carry a `NodeType` mapping unless
  they are intentionally legacy-only aliases with no one-to-one metadata
  node
- source scan no longer finds active/quarantined placeholder helper
  bodies in `PipelineExecutor`; support truth now depends on completing
  broader runtime ownership convergence
- `test_pipeline_executor_operator_routing` now checks representative
  typed fail-closed families end-to-end through `PipelineExecutor` and
  verifies the runtime error uses the central fail-closed reason

**Recommendation:**

- do not silently pass through for ML algorithm nodes
- fail with a clear "not implemented" execution error instead
- only return success when the actual algorithm path exists

---

### 9. Evaluation nodes with historical placeholder success paths

**Status:** Fixed in current branch for the legacy `PipelineExecutor` path. Evaluation nodes now fail closed instead of registering fake output datasets.

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

Problem before the runtime-truth pass:

- these functions were explicit placeholder implementations
- they reported outputs and success

Current remaining problem:

- evaluation is still not a truthful graph-execution stage; the legacy
  path now fails closed instead of pretending evaluation completed

Effect now:

- fake completed evaluation output is blocked in the audited legacy path
- real graph-level evaluation still needs implementation or UI-only
  labeling

**Recommendation:**

- treat them as unimplemented, not successful placeholders

---

### 10. Text analytics nodes have real operators in one path, but placeholder logic in another

**Status:** Fixed for exact registered runtime names in current branch.
The legacy `PipelineExecutor` path no longer masks the real
operator-backed path with passthrough success, and these exact node names
now route through `PipelineOperatorFactory`. Broader runtime convergence
and dead legacy branch cleanup remain tracked in `tofix5.md`.

Affected nodes:

- `TFIDFVectorizer`
- `CountVectorizer`
- `SentimentAnalyzer`

**Severity:** High

Relevant files:

- `cyxwiz-engine/src/core/node_executors/pipeline_operator_factory.cpp`
- `cyxwiz-engine/src/core/pipeline_executor.cpp`

Problem now:

- `node_executors/*` contains real operator implementations
- `PipelineOperatorFactory` registers them
- the legacy `PipelineExecutor` now routes the exact registered node
  names through the operator-backed implementation
- source scan no longer finds stale placeholder-era helper bodies for
  these exact node names; runtime ownership convergence remains

Effect:

- duplicate execution systems still need ownership cleanup
- the newer real path is no longer masked by fake success for these
  exact runtime names

**Recommendation:**

- pick one execution path as canonical
- remove placeholder executor branches once migrated

---

### 11. Time-series analysis nodes have the same split-brain problem

**Status:** Fixed for exact registered runtime names in current branch.
The legacy `PipelineExecutor` now routes these advanced time-series node
names through `PipelineOperatorFactory`. Broader runtime convergence and
dead legacy branch cleanup remain tracked in `tofix5.md`.

Affected nodes:

- `TimeSeriesDecomposition`
- `ARIMAForecaster`
- `ExponentialSmoothing`

**Severity:** High

Relevant files:

- `cyxwiz-engine/src/core/node_executors/pipeline_operator_factory.cpp:135`
- `cyxwiz-engine/src/core/pipeline_executor.cpp`

Problem now:

- operator implementations exist and are registered in
  `PipelineOperatorFactory`
- the main `PipelineExecutor` dispatch now has a matching active
  execution path for the exact registered node names
- source scan no longer finds stale placeholder-era helper bodies for
  these exact node names; runtime ownership convergence remains

Effect:

- node support is visible through the legacy executor for these exact
  runtime names
- ownership remains split until the runtime convergence work in
  `tofix5.md` is complete

**Recommendation:**

- unify executor routing
- do not keep "registered real operator" and "legacy placeholder path"
  in parallel for the same node family

---

## Priority 3: DNN Nodes That Look Real But Have No Visible Backend Execution Path

**Status:** Fixed in current branch for runtime and metadata truth.
`PipelineExecutor` now fails closed if these DNN nodes reach the legacy
execution path, and `NodeMetadataRegistry` marks them as template/blocked
instead of implemented.

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

Problem before this pass:

- metadata marked them as implemented
- they were visible as real DNN nodes
- there was no clear end-to-end execution path in the current node backend

Current remaining problem:

- real DNN inference/training support is still not implemented; the UI
  should keep treating these as blocked/template unless a real runtime
  path lands

Effect now:

- these nodes no longer over-promise through metadata status
- implementation remains future work

**Recommendation:**

- either wire them into a real inference execution path
- or mark them as template / external / experimental

---

## Priority 4: Data / Utility Nodes With Weak or Misleading Execution Contracts

### 12. `DataProfiler`

**Status:** Fixed in current branch for the legacy `PipelineExecutor`
path and metadata truth. It now fails closed as a panel/report workflow
instead of pretending to transform data, and metadata marks it
template/UI-only rather than implemented.

**Severity:** Medium

Relevant files:

- `cyxwiz-engine/src/core/node_metadata_registry.cpp:903`
- `cyxwiz-engine/src/core/pipeline_executor.cpp:4291`

Problem before the runtime-truth pass:

- node was marked implemented
- executor logged input shape and passed data through

Current remaining problem:

- the node still needs either a real profiling output contract or a
  polished UI-only/report workflow

Effect now:

- fake transform behavior is blocked
- real profiling/report behavior is still separate work

**Recommendation:**

- either implement real profiling output
- or surface it as a report/panel action rather than a fake transform node

---

### 13. Utility nodes that historically returned placeholder success

**Status:** Fixed in current branch for the legacy `PipelineExecutor`
path and metadata truth. The affected utility nodes now return explicit
unsupported execution errors instead of placeholder success, and metadata
marks them template/blocked instead of implemented.

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

Problem before this pass:

- several utility nodes used placeholder logic or passthrough behavior

Current remaining problem:

- real graph execution for these utility nodes is still not implemented

Effect now:

- apparent success without real work is blocked in runtime and metadata

**Recommendation:**

- keep them blocked until implemented

### 14. Legacy `TextClean` stop-word option was seeded but unsupported

**Status:** Fixed in current branch.

Implemented:

- new Data Studio `TextClean` nodes no longer seed the unsupported
  `remove_stopwords=false` parameter
- imported/saved graphs that set `remove_stopwords=true` still fail
  closed with a specific runtime-parameter validation error
- the executor comment now points at the fail-closed validation path
  rather than describing a silent MVP omission

Relevant files:

- `cyxwiz-engine/src/gui/data_studio/pipeline_canvas.cpp`
- `cyxwiz-engine/src/core/pipeline_executor.cpp`

Problem:

- the legacy `TextClean` executor cannot truthfully remove stop words
  without a dictionary-backed implementation
- UI-created nodes still carried an unsupported parameter, even though
  the tooltip did not advertise it and the executor rejected `true`

Effect now:

- newly created nodes no longer carry a misleading unsupported option
- old/imported graphs remain safe because enabling the option fails
  before execution

**Recommendation:**

- only expose stop-word removal in Data Studio once there is a real
  dictionary-backed runtime path

### 15. Data Studio palette exposed internal `ArrowDataset` as a node

**Status:** Fixed in current branch.

Implemented:

- the Data Studio quick-add palette no longer offers `ArrowDataset`
- the `ArrowDataset` tooltip now labels it as an internal storage type,
  not an addable pipeline runtime node

Relevant files:

- `cyxwiz-engine/src/gui/data_studio/pipeline_canvas.cpp`

Problem:

- `ArrowDataset` is an internal data representation used by registry,
  batcher, and materializer code
- it is intentionally not a declared Data Studio runtime capability
- the palette still let users create a graph node named `ArrowDataset`,
  which had no truthful `PipelineExecutor` dispatch contract

Effect now:

- users are guided through executable input nodes instead of an internal
  storage abstraction
- old/imported `ArrowDataset` graph nodes are still not promoted as
  supported runtime nodes

**Recommendation:**

- keep storage abstractions out of the add-node palette unless they gain
  a real graph execution contract

### 16. Data Studio file tooltips advertised unsupported formats

**Status:** Fixed in current branch.

Implemented:

- `FileInput` tooltip no longer lists Excel/HDF5 as supported pipeline
  input formats
- `SaveDataset` tooltip now lists only the runtime-supported CSV and
  Parquet export formats

Relevant files:

- `cyxwiz-engine/src/gui/data_studio/pipeline_canvas.cpp`

Problem:

- the Data Studio palette/tooltips could advertise broader file support
  than `PipelineExecutor` and the runtime capability tables allow
- this repeated the same node-truth failure mode in documentation/UI
  text rather than metadata

Effect now:

- the quick-add UI describes the executable file contract instead of
  implying unsupported Excel/HDF5/Arrow export paths

**Recommendation:**

- keep tooltip format lists derived from runtime capability tables where
  practical

### 17. Legacy `TextVectorize` seeded unsupported `max_features`

**Status:** Fixed in current branch.

Implemented:

- new Data Studio `TextVectorize` nodes no longer seed the unsupported
  `max_features=1000` parameter
- imported/saved graphs that still include `max_features` continue to
  fail closed through runtime-parameter validation

Relevant files:

- `cyxwiz-engine/src/gui/data_studio/pipeline_canvas.cpp`
- `cyxwiz-engine/src/core/pipeline_executor.cpp`

Problem:

- the legacy `TextVectorize` path only supports its simple count-based
  feature contract
- `PipelineExecutor` rejects `max_features`, but the Data Studio
  quick-add path still created nodes with that parameter by default

Effect now:

- new UI-created `TextVectorize` nodes match the executable legacy
  contract
- older graphs remain safe because unsupported `max_features` still
  fails before execution

**Recommendation:**

- expose feature limits only on real vectorizer operator nodes such as
  `CountVectorizer` or `TFIDFVectorizer`, not on the legacy alias

### 18. Time-series quick-add used legacy aliases despite real operators

**Status:** Fixed in current branch.

Implemented:

- Data Studio quick-add now creates operator-backed
  `TimeSeriesWindow`, `TimeSeriesFeatures`, and `Differencing` nodes
  for the window/features/diff menu entries
- defaults now use the operator parameter contracts:
  `value_col`, `input_width`, `shift`, `lag_values`,
  `rolling_windows`, `lag`, and `order`
- legacy `TSWindow`, `TSFeatures`, and `TSDiff` tooltip handling remains
  for imported/saved graphs

Relevant files:

- `cyxwiz-engine/src/gui/data_studio/pipeline_canvas.cpp`
- `cyxwiz-engine/src/core/pipeline_runtime_capabilities.cpp`

Problem:

- the runtime already has canonical operator-backed time-series nodes
- the palette still created legacy alias nodes, keeping new graphs on
  the compatibility dispatch path instead of the converged operator path

Effect now:

- new time-series graphs use the real operator-backed runtime where the
  operator exists
- legacy aliases remain loadable and executable for existing projects

**Recommendation:**

- continue migrating quick-add aliases to operator-backed names only
  where the parameter contract is clear and covered by runtime
  capabilities

### 19. Text quick-add used legacy aliases despite real operators

**Status:** Fixed in current branch.

Implemented:

- Data Studio quick-add now creates operator-backed `TextTokenizer` for
  tokenization and `CountVectorizer` for vectorization
- defaults now use the operator parameter contracts:
  `text_col`, `tokenizer_type`, `max_length`, `min_word_freq`,
  `max_vocab_size`, `max_features`, and `norm`
- legacy `TextTokenize` and `TextVectorize` tooltip/default handling
  remains for compatibility paths and old/imported graphs

Relevant files:

- `cyxwiz-engine/src/gui/data_studio/pipeline_canvas.cpp`
- `cyxwiz-engine/src/core/pipeline_runtime_capabilities.cpp`

Problem:

- real text operators already exist and are registered through
  `PipelineOperatorFactory`
- new Data Studio graphs still started from legacy alias nodes, including
  a `TextVectorize` path with a smaller/simple feature contract

Effect now:

- new tokenization/vectorization graphs use the operator-backed runtime
  path by default
- legacy text aliases remain available to saved/imported graphs without
  being promoted by quick-add

**Recommendation:**

- prefer exact operator-backed node names in new graph creation, and
  keep legacy aliases as compatibility-only dispatch names

### 20. `TSLag` was still promoted as a new-graph quick-add

**Status:** Fixed in current branch.

Implemented:

- removed `TSLag` from the Data Studio quick-add palette
- retained `TSLag` default/tooltip handling for compatibility with
  saved/imported graphs
- new lag-style time-series work should use the operator-backed
  `TimeSeriesFeatures` node, which supports `lag_values`

Relevant files:

- `cyxwiz-engine/src/gui/data_studio/pipeline_canvas.cpp`

Problem:

- after moving time-series quick-add entries to operator-backed nodes,
  `TSLag` remained as a promoted legacy alias
- this kept new graphs on compatibility dispatch even though the
  canonical time-series feature operator already covers lag features

Effect now:

- new graph creation no longer advertises `TSLag` as a first-class
  runtime node
- old/imported `TSLag` graphs still have compatibility handling

**Recommendation:**

- keep compatibility aliases loadable but avoid adding them from new
  graph creation surfaces

### 21. `SaveDataset` was still promoted instead of canonical `DataOutput`

**Status:** Fixed in current branch.

Implemented:

- Data Studio's "Save Dataset" quick-add now creates canonical
  `DataOutput`
- `DataOutput` defaults to the metadata/runtime parameter contract:
  blank required `file_path` plus runtime-supported `file_type=csv`
- `SaveDataset` tooltip handling remains for old/imported compatibility
  graphs

Relevant files:

- `cyxwiz-engine/src/gui/data_studio/pipeline_canvas.cpp`
- `cyxwiz-engine/src/core/pipeline_runtime_capabilities.cpp`

Problem:

- `SaveDataset` is documented in runtime capabilities as a legacy
  saved-pipeline output alias
- new graphs still created that alias instead of the canonical
  browser-visible output node

Effect now:

- new graph creation uses the canonical output node name
- saved/imported `SaveDataset` graphs remain compatible

**Recommendation:**

- use typed/canonical node names from quick-add surfaces; reserve
  legacy aliases for import and saved-pipeline compatibility

### 22. Legacy-only feature engineering aliases were still promoted

**Status:** Fixed in current branch.

Implemented:

- removed `PolynomialFeatures` and `Binning` from the Data Studio
  quick-add palette
- retained their default/tooltip/runtime compatibility handling for
  saved/imported graphs

Relevant files:

- `cyxwiz-engine/src/gui/data_studio/pipeline_canvas.cpp`
- `cyxwiz-engine/src/core/pipeline_runtime_capabilities.cpp`

Problem:

- runtime capabilities document both names as legacy saved-pipeline
  aliases with no browser-visible typed metadata
- the quick-add palette still promoted them as new first-class graph
  nodes

Effect now:

- new graph creation no longer advertises legacy-only feature
  engineering aliases
- old/imported graphs remain compatible through the existing legacy
  dispatch path

**Recommendation:**

- add canonical typed/operator-backed feature-engineering nodes before
  reintroducing polynomial expansion or binning to quick-add

### 23. File input quick-add still created legacy `FileInput`

**Status:** Fixed in current branch.

Implemented:

- Data Studio's "File Input" quick-add now creates canonical
  `DataInput`
- `DataInput` defaults now match the metadata/runtime contract:
  `source_type=file`, blank required `file_path`, and `type=auto`
- `FileInput` tooltip/default handling remains for legacy compatibility
  graphs

Relevant files:

- `cyxwiz-engine/src/gui/data_studio/pipeline_canvas.cpp`
- `cyxwiz-engine/src/core/pipeline_runtime_capabilities.cpp`
- `cyxwiz-engine/src/core/node_metadata_registry.cpp`

Problem:

- `FileInput` is still executable but represents the older file-source
  path
- the canonical smart input metadata and runtime capability is
  `DataInput`
- quick-add promoted the older node name for new graphs

Effect now:

- new graph creation uses the canonical input node contract
- old/imported `FileInput` graphs remain compatible

**Recommendation:**

- continue using canonical smart I/O nodes in new UI surfaces and keep
  format-specific/legacy aliases as import compatibility only

### 24. Duplicate-removal quick-add used old runtime spelling

**Status:** Fixed in current branch.

Implemented:

- added `RemoveDuplicateRows` as the canonical executable runtime string
  for `NodeType::RemoveDuplicateRows`
- retained `RemoveDuplicates` as an old graph compatibility alias
- Data Studio quick-add now creates `RemoveDuplicateRows`
- execution diagnostics now use `RemoveDuplicateRows` for canonical
  graphs and keep `RemoveDuplicates` for legacy alias graphs
- metadata/runtime tests now verify that enum lookup prefers
  `RemoveDuplicateRows` while the old `RemoveDuplicates` runtime string
  still resolves to the same typed node

Relevant files:

- `cyxwiz-engine/src/core/pipeline_runtime_capabilities.cpp`
- `cyxwiz-engine/src/gui/data_studio/pipeline_canvas.cpp`
- `cyxwiz-engine/src/core/pipeline_executor.cpp`
- `cyxwiz-engine/tests/test_pipeline_operator_metadata.cpp`

Problem:

- metadata and enum naming use `RemoveDuplicateRows`
- Data Studio quick-add still created the older `RemoveDuplicates`
  spelling
- changing only the UI name would have broken execution because the
  runtime string resolver did not know the canonical spelling

Effect now:

- new graphs use the canonical typed node string
- old/imported `RemoveDuplicates` graphs remain executable

**Recommendation:**

- when migrating UI node strings, add explicit runtime aliases before
  changing graph creation surfaces

### 25. `TextClean` remains an intentional quick-add compatibility node

**Status:** Documented in current branch.

Current decision:

- keep `TextClean` in Data Studio quick-add for now
- there is no typed metadata/operator-backed replacement such as
  `TextCleaner`
- the legacy executor path is not a fake placeholder; it has guarded
  execution for lowercase, HTML removal, and special-character removal
- unsupported stop-word removal remains fail-closed

Relevant files:

- `cyxwiz-engine/src/gui/data_studio/pipeline_canvas.cpp`
- `cyxwiz-engine/src/core/pipeline_runtime_capabilities.cpp`
- `cyxwiz-engine/src/core/pipeline_executor.cpp`

Reason:

- removing `TextClean` would remove real, guarded Data Studio
  functionality rather than just hiding an unsafe placeholder
- unlike `TextTokenize`/`TextVectorize`, there is no exact
  operator-backed node available today

**Recommendation:**

- keep `TextClean` until a canonical typed/operator-backed cleaner lands,
  then migrate quick-add the same way as tokenizer/vectorizer nodes

### 26. Quick-add runtime contract now has regression coverage

**Status:** Fixed in current branch.

Implemented:

- metadata/runtime tests now lock the Data Studio quick-add node set to
  runtime-supported names
- `PipelineCanvas` now exposes the canonical quick-add list through
  `GetQuickAddNodes()`, so the test verifies the same contract the UI
  renders instead of duplicating a stale list
- typed quick-add nodes must also resolve to implemented metadata; the
  only exception remains compatibility-only `TextClean`
- tests explicitly reject the legacy aliases removed from quick-add:
  `ArrowDataset`, `FileInput`, `SaveDataset`, `RemoveDuplicates`,
  `TextTokenize`, `TextVectorize`, `TSWindow`, `TSFeatures`, `TSLag`,
  `TSDiff`, `PCA`, `PolynomialFeatures`, and `Binning`
- `TextClean` is captured as the only intentional compatibility-node
  exception because it still has real guarded execution and no exact
  typed replacement

Relevant files:

- `cyxwiz-engine/tests/test_pipeline_operator_metadata.cpp`
- `cyxwiz-engine/src/gui/data_studio/pipeline_canvas.cpp`
- `cyxwiz-engine/src/gui/data_studio/pipeline_canvas.h`

Problem:

- quick-add regressions are easy to introduce because the palette is UI
  code while runtime support truth lives in capability tables
- prior fixes moved many menu entries from old aliases to canonical
  runtime names, but no single test described the intended new-graph
  creation contract

Effect now:

- future quick-add additions must resolve to known supported runtime
  nodes
- intentionally retained compatibility nodes must be named explicitly

**Recommendation:**

- if `pipeline_canvas.cpp` gains a new quick-add entry, update this test
  contract and prefer canonical typed/operator-backed node names

---

## Priority 5: Template Nodes Are Correctly Blocked, But Need Import-Time Guardrails

**Status:** Fixed in current branch for compile-time training-path guardrails.

Implemented:

- `GraphCompiler` validates every node on the selected training path
  against `NodeMetadataRegistry`.
- nodes marked `Template`, `Deprecated`, or `External` now produce a
  compile error before training can start.
- the guard applies to loaded/imported/hand-edited graphs because it is
  not only a UI add-search check.
- `test_graph_compiler_deferred_nodes` verifies template/deferred nodes
  on the training path fail compile, while disconnected side-path
  template nodes do not block the selected trainable path.

Template nodes in metadata are currently blocked in add-search:

- `cyxwiz-engine/src/gui/node_editor_add_search.cpp`

This is good. Before the compile guard:

- old graphs
- imported graphs
- hand-edited project files

could still contain template node types.

**Recommendation:**

- keep compile-time validation as the source of truth for training
  execution
- add project-load warnings later if users need earlier feedback, but do
  not rely on project load alone for safety

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

### Progress note: typed Binning runtime coverage

The legacy-only `Binning` operator has been promoted behind a canonical
`BinningNode` runtime contract while preserving `Binning` as an executable
legacy alias.

Updated coverage:

- `NodeType::BinningNode` is now serializable and registered in metadata.
- The runtime capability map resolves `BinningNode` to the legacy executor
  and keeps `Binning` as a compatibility alias.
- `PipelineExecutor` routes typed `BinningNode` execution through the real
  binning implementation.
- Data Studio quick-add exposes `BinningNode` instead of the legacy
  `Binning` alias.
- Metadata tests assert that quick-add nodes do not expose the legacy alias
  and that both canonical and compatibility runtime names resolve correctly.
- Executor routing tests cover canonical `BinningNode` success and legacy
  `Binning` alias success.

### Progress note: typed Polynomial Features runtime coverage

The legacy-only `PolynomialFeatures` operator has been promoted behind a
canonical `PolynomialFeaturesNode` runtime contract while preserving
`PolynomialFeatures` as an executable legacy alias.

Updated coverage:

- `NodeType::PolynomialFeaturesNode` is now serializable and registered in
  metadata.
- The runtime capability map resolves `PolynomialFeaturesNode` to the legacy
  executor and keeps `PolynomialFeatures` as a compatibility alias.
- `PipelineExecutor` routes typed `PolynomialFeaturesNode` execution through
  the real polynomial feature implementation.
- Data Studio quick-add exposes `PolynomialFeaturesNode` instead of the legacy
  `PolynomialFeatures` alias.
- Metadata tests assert both canonical and compatibility runtime names resolve
  to typed metadata.
- Executor routing tests cover canonical `PolynomialFeaturesNode` success and
  legacy `PolynomialFeatures` alias success.

### Progress note: typed Time Series Lag runtime coverage

The legacy-only `TSLag` operator has been promoted behind a canonical
`TimeSeriesLag` runtime contract while preserving `TSLag` as an executable
legacy alias.

Updated coverage:

- `NodeType::TimeSeriesLag` is now serializable and registered in metadata.
- The runtime capability map resolves `TimeSeriesLag` to the legacy executor
  and keeps `TSLag` as a compatibility alias.
- `PipelineExecutor` routes typed `TimeSeriesLag` execution through the real
  lag-column implementation.
- Data Studio quick-add exposes `TimeSeriesLag` instead of the legacy `TSLag`
  alias.
- Metadata tests assert both canonical and compatibility runtime names resolve
  to typed metadata.
- Executor routing tests cover canonical `TimeSeriesLag` success and legacy
  `TSLag` alias success.

### Progress note: typed Time Series Window compatibility coverage

The legacy-only `TSWindow` operator has been tied to the existing canonical
`TimeSeriesWindow` metadata while preserving `TSWindow` as an executable
legacy alias.

Updated coverage:

- The runtime capability map resolves `TSWindow` to `NodeType::TimeSeriesWindow`.
- `PipelineExecutor` routes the typed `TSWindow` compatibility alias through
  the existing legacy window implementation.
- Data Studio quick-add continues to expose canonical `TimeSeriesWindow`
  instead of the legacy `TSWindow` alias.
- Metadata tests assert that `TSWindow` remains legacy-executor routed while
  resolving to typed `TimeSeriesWindow` metadata.

### Progress note: typed Time Series Features compatibility coverage

The legacy-only `TSFeatures` operator has been tied to the existing canonical
`TimeSeriesFeatures` metadata while preserving `TSFeatures` as an executable
legacy alias.

Updated coverage:

- The runtime capability map resolves `TSFeatures` to
  `NodeType::TimeSeriesFeatures`.
- `PipelineExecutor` routes the typed `TSFeatures` compatibility alias through
  the existing legacy feature implementation.
- Data Studio quick-add continues to expose canonical `TimeSeriesFeatures`
  instead of the legacy `TSFeatures` alias.
- Metadata tests assert that `TSFeatures` remains legacy-executor routed while
  resolving to typed `TimeSeriesFeatures` metadata.

### Progress note: typed Time Series Differencing compatibility coverage

The legacy-only `TSDiff` operator has been tied to the existing canonical
`Differencing` metadata while preserving `TSDiff` as an executable legacy
alias.

Updated coverage:

- The runtime capability map resolves `TSDiff` to `NodeType::Differencing`.
- `PipelineExecutor` routes the typed `TSDiff` compatibility alias through the
  existing legacy differencing implementation.
- Data Studio quick-add continues to expose canonical `Differencing` instead
  of the legacy `TSDiff` alias.
- Metadata tests assert that `TSDiff` remains legacy-executor routed while
  resolving to typed `Differencing` metadata.

### Progress note: typed Text Vectorize compatibility coverage

The legacy-only `TextVectorize` operator has been tied to the existing
canonical `CountVectorizer` metadata while preserving `TextVectorize` as an
executable legacy alias.

Updated coverage:

- The runtime capability map resolves `TextVectorize` to
  `NodeType::CountVectorizer`.
- `PipelineExecutor` routes the typed `TextVectorize` compatibility alias
  through the existing legacy vectorization implementation.
- Data Studio quick-add continues to expose canonical `CountVectorizer`
  instead of the legacy `TextVectorize` alias.
- Metadata tests assert that `TextVectorize` remains legacy-executor routed
  while resolving to typed `CountVectorizer` metadata.

### Progress note: typed Text Tokenize compatibility coverage

The legacy-only `TextTokenize` operator has been tied to the existing
canonical `TextTokenizer` metadata while preserving `TextTokenize` as an
executable legacy alias.

Updated coverage:

- The runtime capability map resolves `TextTokenize` to
  `NodeType::TextTokenizer`.
- `PipelineExecutor` routes the typed `TextTokenize` compatibility alias
  through the existing legacy tokenization implementation.
- Data Studio quick-add continues to expose canonical `TextTokenizer` instead
  of the legacy `TextTokenize` alias.
- Metadata tests assert that `TextTokenize` remains legacy-executor routed
  while resolving to typed `TextTokenizer` metadata.

### Progress note: typed Text Clean runtime coverage

The legacy-only `TextClean` operator has been promoted behind a canonical
`TextCleanNode` runtime contract while preserving `TextClean` as an executable
legacy alias.

Updated coverage:

- `NodeType::TextCleanNode` is now serializable and registered in metadata.
- The runtime capability map resolves `TextCleanNode` to the legacy executor
  and keeps `TextClean` as a compatibility alias.
- `PipelineExecutor` routes typed `TextCleanNode` execution through the real
  text-cleaning implementation.
- Data Studio quick-add exposes `TextCleanNode` instead of the legacy
  `TextClean` alias.
- Metadata tests assert both canonical and compatibility runtime names resolve
  to typed metadata.
- Executor routing tests cover canonical `TextCleanNode` success and legacy
  `TextClean` alias success.

### Progress note: typed Save Dataset compatibility coverage

The legacy-only `SaveDataset` output node has been tied to the existing
canonical `DataOutput` metadata while preserving `SaveDataset` legacy behavior.

Updated coverage:

- The runtime capability map resolves `SaveDataset` to `NodeType::DataOutput`.
- `PipelineExecutor` routes the typed `SaveDataset` compatibility alias through
  the legacy save implementation so optional `path`, `name`, and downstream
  in-memory alias behavior remain intact.
- Data Studio quick-add continues to expose canonical `DataOutput` instead of
  the legacy `SaveDataset` alias.
- Metadata tests assert that `SaveDataset` remains legacy-executor routed while
  resolving to typed `DataOutput` metadata.

### Progress note: typed Deploy to Node Editor runtime coverage

The legacy-only `DeployToNodeEditor` handoff node has been promoted behind a
canonical `DeployToNodeEditorNode` runtime contract while preserving
`DeployToNodeEditor` as an executable legacy alias.

Updated coverage:

- `NodeType::DeployToNodeEditorNode` is now serializable and registered in
  metadata.
- The runtime capability map resolves `DeployToNodeEditorNode` to the legacy
  executor and keeps `DeployToNodeEditor` as a compatibility alias.
- `PipelineExecutor` routes typed `DeployToNodeEditorNode` execution through
  the existing deployment handoff implementation.
- Metadata tests assert both canonical and compatibility runtime names resolve
  to typed metadata.

### Progress note: typed PCA compatibility coverage

The legacy `PCA` canvas alias has been tied to the existing operator-backed
`PCANode` runtime contract instead of remaining fail-closed.

Updated coverage:

- The runtime capability map resolves `PCA` to `NodeType::PCANode`.
- `PCA` now advertises operator-backed support through the existing
  `PCAOperator`.
- Data Studio quick-add continues to expose canonical `PCANode` instead of the
  legacy `PCA` alias.
- Metadata tests assert that legacy `PCA` resolves to typed `PCANode` metadata
  and no longer appears in the fail-closed exception list.

### Progress note: typed Parquet Input compatibility coverage

The legacy `ParquetInput` source alias has been tied to the existing
`DataInput` runtime contract instead of remaining fail-closed.

Updated coverage:

- The runtime capability map resolves `ParquetInput` to `NodeType::DataInput`.
- `PipelineExecutor` routes `ParquetInput` through `DataInput` and defaults
  the legacy alias to `type=parquet` when no file type is supplied.
- Metadata tests assert that legacy `ParquetInput` resolves to typed
  `DataInput` metadata and no longer appears in the fail-closed exception list.
- Executor routing tests assert that `ParquetInput` reaches the loader path
  instead of failing through the fail-closed runtime path.

### Progress note: Cell Extractor runtime implementation

`CellExtractor` has been moved from fail-closed template metadata to a real
PipelineExecutor implementation.

Updated coverage:

- `CellExtractor` metadata is now marked implemented.
- The runtime capability map resolves `CellExtractor` to the legacy executor.
- `PipelineExecutor` extracts one requested row/column value into a one-row
  dataset named `ds_cell_<node_id>`.
- Runtime validation requires `column` and validates non-negative `row`.
- Executor routing tests cover successful cell extraction.

### Progress note: Cell Updater runtime implementation

`CellUpdater` has been moved from fail-closed template metadata to a real
PipelineExecutor implementation.

Updated coverage:

- `CellUpdater` metadata is now marked implemented.
- The runtime capability map resolves `CellUpdater` to the legacy executor.
- `PipelineExecutor` updates one requested row/column value and preserves the
  remaining table rows and schema.
- Runtime validation requires `column` and `value`, and validates
  non-negative `row`.
- Executor routing tests cover successful cell update.

### Progress note: Row Appender runtime implementation

`RowAppender` has been moved from fail-closed template metadata to a real
PipelineExecutor implementation.

Updated coverage:

- `RowAppender` metadata is now marked implemented.
- The runtime capability map resolves `RowAppender` to the legacy executor
  with two required input datasets.
- `PipelineExecutor` appends rows from two schema-compatible input tables using
  `UNION ALL`.
- The first implementation fails clearly when schemas differ instead of
  coercing or fabricating missing values.
- Executor routing tests cover successful row append.

### Progress note: Column Appender runtime implementation

`ColumnAppender` has been moved from fail-closed template metadata to a real
PipelineExecutor implementation.

Updated coverage:

- `ColumnAppender` metadata is now marked implemented.
- The runtime capability map resolves `ColumnAppender` to the legacy executor
  with two required input datasets.
- `PipelineExecutor` appends columns from two row-aligned input tables.
- Duplicate right-side column names are suffixed using the `suffix` parameter.
- The first implementation fails clearly when row counts differ instead of
  padding or truncating rows.
- Executor routing tests cover successful column append.

### Progress note: Unpivot runtime implementation

`Unpivot` has been moved from fail-closed template metadata to a real
PipelineExecutor implementation.

Updated coverage:

- `Unpivot` metadata is now marked implemented.
- The runtime capability map resolves `Unpivot` to the legacy executor.
- `PipelineExecutor` melts wide tables into long tables using configured
  `id_columns`, `variable_name`, and `value_name`.
- Value cells are emitted as strings so mixed-type input columns do not create
  unsafe union coercion failures.
- Runtime validation rejects missing ID columns and output-name conflicts.
- Executor routing tests cover successful unpivot execution.

### Progress note: Export JSON runtime implementation

`ExportJSON` has been moved from fail-closed template metadata to a real
PipelineExecutor implementation.

Updated coverage:

- `ExportJSON` metadata is now marked implemented.
- The runtime capability map resolves `ExportJSON` to the legacy executor.
- `PipelineExecutor` writes Arrow table rows as a JSON array.
- Export path handling accepts both `file_path` and the Data Studio `path`
  alias, matching `ExportCSV`.
- Executor routing tests cover real file creation and emitted JSON content.

At the moment, some nodes are:

- over-exposed
- silently ignored
- placeholder-backed
- or routed through the wrong execution layer

`tofix4.md` should be treated as a node-safety backlog before more node
surface is added to the frontend.
