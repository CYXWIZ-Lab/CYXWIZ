# To Fix 73 - Backend Algorithm Truth, XGBoost Integration, and Cross-Computation Parity

## Status

Open - audit, design, and staged implementation ticket. This ticket records
the next backend-computation program; it does not claim that XGBoost or every
listed candidate algorithm is currently supported.

## Decision statement

Finish truth and parity for the loss functions and optimizers already present
in CyxWiz before expanding the catalog. Add real XGBoost as an optional,
isolated integration with its own artifact identity; do not rename the current
native Gradient Boosting implementation or make a large third-party dependency
part of the mandatory engine core.

Every algorithm exposed by the GUI, graph compiler, C++ API, Python API, or
plugin API must have one explicit support state:

```text
Absent
  -> Backend implemented
  -> Cross-computation verified
  -> Engine training/runtime wired
  -> Artifact/checkpoint lifecycle verified
  -> GUI exposed as Implemented
```

No layer may silently substitute another loss, optimizer, or model when the
requested implementation is unavailable.

## Why this ticket exists

The APS classifier discussion exposed a naming and capability gap:

- CyxWiz has a functional native `GradientBoostingClassifier` Data Studio
  operator.
- CyxWiz does not contain an XGBoost dependency, XGBoost C API adapter,
  `DMatrix` bridge, XGBoost objective configuration, or XGBoost-compatible
  model loading.
- The native operator is standard in-repo gradient boosting; it must not be
  described as XGBoost.

The same kind of drift exists across loss and optimizer surfaces. The backend
contains more implementations than graph training exposes, graph training
does not forward all compiled optimizer settings, Python bindings expose only
part of the loss catalog, and broad numerical parity exists for only a small
subset. Adding more GUI nodes before closing those gaps would increase the
number of unsupported or weakly proven states.

This ticket extends the computation-truth direction established by
`tofix39 - Engine Computation Truth, Numerical Parity, and Training Lifecycle`.
`tofix39` remains the general truth harness; this ticket owns the algorithm
inventory, staged parity matrix, and optional XGBoost boundary.

## Verified current implementation truth - 2026-07-27

### Native Gradient Boosting is implemented

The current implementation lives in the engine's Data Studio/table runtime,
not in the `cyxwiz-backend` tensor-training library.

Working capabilities:

- Arrow-table classifier training;
- automatic or explicit numeric feature selection;
- numeric and string class labels;
- binary and multiclass prediction;
- `n_estimators`, `learning_rate`, `max_depth`, `min_samples_split`, and
  `min_samples_leaf` configuration;
- native `cyxwiz_tree_model` JSON artifact saving and loading;
- inference on another table through `TreeModelPredictor`;
- task progress and materialization tracing.

Verified focused tests:

- `test_gradient_boosting_operator.exe` - passed;
- `test_tree_model_artifact.exe` - passed;
- `test_tree_model_predictor_operator.exe` - passed.

Current limitations:

- it is a CPU, in-memory, exhaustive-split implementation;
- it is classification-only;
- it emits class predictions rather than a production probability/explanation
  contract;
- it does not implement regularized XGBoost objectives;
- it has no row/column subsampling, histogram/approximate tree builder,
  learned missing-value direction, distributed execution, or GPU training;
- its native artifact is not an XGBoost artifact;
- APS-scale `60,000 x 170` performance has not been benchmarked as a production
  acceptance target.

APS may be used as an acceptance dataset, but no APS path, schema, class ratio,
or model setting belongs in the generic implementation.

### XGBoost is absent

Repository source and build configuration contain no functional XGBoost
integration. In particular, there is currently no:

- XGBoost package/dependency declaration;
- C or C++ API adapter;
- Arrow/table-to-`DMatrix` conversion boundary;
- objective/evaluation parameter mapping;
- native XGBoost JSON/UBJ artifact owner;
- XGBoost predictor node or plugin;
- deterministic cross-computation test against XGBoost;
- capability probe that can mark XGBoost available or unavailable at runtime.

The UI must therefore continue to use the name `Gradient Boosting` for the
native implementation. `XGBoost` must remain absent or explicitly unavailable
until the optional integration passes this ticket's acceptance criteria.

## Current loss-function inventory

The C++ backend currently declares 16 loss types. "Graph wired" means the
normal `GraphCompiler -> ModelBuilder -> TrainingExecutor` path creates the
requested loss rather than defaulting to another loss.

| Loss | Backend implementation | Graph wired | Python class | Computation-truth depth |
| --- | --- | --- | --- | --- |
| MSE | Yes | Yes | Yes | Broad parity still pending |
| CrossEntropy | Yes | Yes | Yes | Forward/reference and transformer coverage exists; full matrix pending |
| BCE | Yes | Yes | No | Pending |
| BCEWithLogits | Yes | Yes, including `pos_weight` | Yes | Backend checks exist; full parity matrix pending |
| NLL | Yes | Yes | No | Pending |
| L1 | Yes | Yes | No | Pending |
| SmoothL1 / Huber | Yes; Huber currently aliases SmoothL1 | Yes | No | Pending, including alias semantics |
| KLDiv | Yes | No graph node/runtime route | No | Pending |
| CosineEmbedding | Yes | Intentionally compile-blocked | No | Pending |
| Focal | Yes | Yes | Yes | Focused backend checks exist; full parity pending |
| SoftDice | Yes | Yes | Yes | Focused backend checks exist; full parity pending |
| Tversky | Yes | Yes | Yes | Focused backend checks exist; full parity pending |
| Jaccard / IoU | Yes | Yes | Yes | Focused backend checks exist; full parity pending |
| Triplet | Yes | Intentionally compile-blocked | Yes | Pending |
| Contrastive | Yes | Intentionally compile-blocked | Yes | Pending |

CosineEmbedding, Triplet, and Contrastive must remain blocked in ordinary graph
training even though backend classes exist. Correct execution requires typed
pair/triplet batch payloads, shared encoder ownership, mining/sampling rules,
multi-input backward routing, and embedding outputs. A scalar loss class alone
does not make a valid metric-learning training workflow.

KLDiv also needs an explicit prediction/target-distribution contract. It must
not reuse the ordinary scalar class-label loader merely because a backend
factory case exists.

### Candidate missing losses

These are candidates, not a promise to implement every item. Selection must be
driven by an executable CyxWiz use case and a reference implementation.

Higher-value candidates:

- `CTCLoss` for speech recognition and unaligned sequence targets;
- `PoissonNLLLoss` for count regression;
- `GaussianNLLLoss` for probabilistic regression;
- quantile/pinball loss for forecasting and prediction intervals;
- margin/hinge loss for ranking and maximum-margin classification;
- multi-label soft-margin loss for multi-label classification.

Deferred specialized candidates:

- Lovasz-style segmentation losses;
- ranking losses that require grouped/query-aware batches;
- custom distance losses requiring sparse or ragged payloads;
- adversarial, diffusion, or composite multi-loss orchestration.

Composite training objectives should be implemented through a typed loss
composition contract, not by adding a separate monolithic node for every
possible weighted combination.

## Current optimizer inventory

The C++ backend and Python bindings currently expose eight optimizers:

| Optimizer | Backend | Python | Graph node/compiler | Computation truth |
| --- | --- | --- | --- | --- |
| SGD | Yes | Yes | Yes | Used in a training sanity check; full state parity pending |
| Adam | Yes | Yes | Yes | One-step and multi-step parity pending |
| AdamW | Yes | Yes | Yes | One-step deterministic parity exists; multi-step/state parity pending |
| RMSprop | Yes | Yes | Yes | Pending |
| AdaGrad | Yes | Yes | Yes | Pending |
| NAdam | Yes | Yes | Yes | Pending |
| Adadelta | Yes | Yes | No | Pending |
| LAMB | Yes | Yes | No | Pending |

### Verified optimizer wiring defect

`TrainingConfiguration` stores `momentum`, `beta1`, `beta2`, and
`weight_decay`, and the compiler reads some of these values from optimizer
nodes. `ModelBuilder` currently calls `CreateOptimizer(type, learning_rate)`.
The backend factory therefore constructs the selected optimizer with only the
learning rate plus hard-coded constructor defaults.

Consequences:

- compiled optimizer hyperparameters other than learning rate are not applied;
- AdamW uses its backend default weight decay rather than a graph-owned value;
- SGD momentum, Adam-family betas, RMSprop alpha/momentum, epsilon values, and
  other optimizer-specific settings do not have an end-to-end typed contract;
- checkpoint v1 does not persist optimizer state, so exact training resume is
  impossible even when the algorithm itself works.

This is a priority parity defect. It must be fixed before adding more optimizer
names to the GUI.

### Candidate missing optimizers

After current optimizer parity is complete, evaluate only use-case-backed
candidates:

- Adamax and RAdam for additional dense-training choices;
- Adafactor for memory-sensitive transformer training;
- ASGD or Rprop for specific classical/dense workloads;
- Lion only if a reproducible project use case justifies it.

`LBFGS` requires a closure/re-evaluation training-loop contract and must not be
implemented as a normal one-call `Step`. `SparseAdam` requires real sparse
gradient ownership. Both remain deferred until those primitives exist.

## Cross-computation truth contract

Extend `cyxwiz-engine/tests/computation_truth/` with small deterministic cases.
Reference PyTorch where available; otherwise use an exact documented hand
reference. Optional external reference scripts must not be required for the
default C++ test suite.

### Loss matrix

For every loss marked graph-supported, verify:

1. forward value;
2. backward gradient;
3. `none`, `mean`, and `sum` reduction semantics where meaningful;
4. scalar, batch, class, and sequence shape contracts;
5. invalid-shape and invalid-target failures;
6. weights, ignore indices, smoothing, margins, and stability parameters;
7. extreme-logit numerical stability and zero/empty-edge behavior;
8. CPU/GPU parity where the backend claims both placements;
9. C++ factory, Python binding, and graph-builder parameter equivalence.

### Optimizer matrix

For every optimizer marked graph-supported, verify:

1. one-step parameter update;
2. multi-step moment/accumulator progression;
3. all exposed hyperparameters;
4. zero-gradient and missing-gradient behavior;
5. weight-decay semantics, including coupled versus decoupled decay;
6. parameter-name/state ownership across multiple tensors;
7. CPU/GPU parity where supported;
8. state save/load round-trip for checkpoint v2;
9. identical configuration through C++, Python, and graph training.

Tests must fail with expected-versus-actual values and identify the algorithm,
parameter, tensor, step, device, and reduction involved.

### Classical-model matrix

For native Gradient Boosting and a future XGBoost adapter, verify separately:

- deterministic seeded training;
- binary and multiclass labels;
- probability and class-output semantics;
- missing-value behavior;
- feature-name and feature-order preservation;
- artifact save/load prediction parity;
- train/test schema mismatch failures;
- bounded memory and cancellation/progress reporting;
- representative small reference cases;
- an APS-scale benchmark without embedding APS-specific logic.

Native Gradient Boosting and XGBoost are different algorithms. Their numeric
predictions are not expected to match each other. Each must match its own
documented/reference contract.

## Proposed implementation phases

### Phase 0 - One support manifest

- Generate or maintain one typed inventory for backend, C++ factory, Python,
  graph compiler, model builder, device placement, checkpoint state, and GUI.
- Make metadata derive `Implemented`, `Blocked`, or `Unavailable` from that
  truth instead of independent labels.
- Add drift tests so a GUI node cannot be implemented while its runtime route
  defaults to another algorithm.

### Phase 1 - Prove existing backend algorithms

- Add the loss and optimizer cross-computation matrices.
- Fix numerical or reduction defects before adding new algorithms.
- Record CPU/GPU support explicitly.
- Keep unproven algorithms usable only at their currently verified API layer;
  do not broaden GUI claims.

### Phase 2 - Complete existing engine wiring

- Replace the learning-rate-only optimizer factory call with a typed optimizer
  configuration that forwards every supported hyperparameter.
- Add validation and plain-language logs for ignored/incompatible settings.
- Decide whether Adadelta and LAMB should receive graph nodes based on proven
  use cases; otherwise mark them backend/Python-only.
- Add KLDiv only with a distribution-target contract.
- Preserve fail-closed metric-learning nodes until typed pair/triplet training
  is implemented.
- Reconcile missing Python loss bindings without creating a second semantics
  layer.

### Phase 3 - Add selected missing primitives

- Select a small first set from CTC, probabilistic/quantile regression, and
  one justified optimizer.
- Implement backend, bindings, graph wiring, logs, and parity together.
- Do not add metadata or GUI nodes before executable truth exists.

### Phase 4 - Optional XGBoost adapter/plugin

Recommended boundary:

```text
Registered Arrow/Parquet dataset
  -> typed feature/target selection
  -> optional XGBoost adapter
       -> bounded DMatrix construction
       -> objective + parameter validation
       -> train/evaluate/predict
       -> native XGBoost artifact + CyxWiz manifest
  -> registered prediction dataset / model handle
```

Requirements:

- optional build/runtime capability probe;
- no mandatory XGBoost load cost when unused;
- binary classification first, followed by multiclass and regression only
  after the first slice passes;
- deterministic seed, thread, missing-value, and objective settings;
- validation/evaluation-set support without data leakage;
- class/probability output schema;
- native XGBoost artifact identity, never `cyxwiz_tree_model` masquerading as
  another format;
- `TreeModelPredictor` extension or a generic typed model-predictor boundary;
- actionable unavailable/dependency/version diagnostics;
- plugin/package version recorded with the artifact.

The integration should reuse DatasetAsset/DataRegistry, tasks, progress,
artifacts, and capability truth. It must not create a second dataset registry,
training dashboard, or project format.

### Phase 5 - Production lifecycle

- checkpoint v2 optimizer state;
- exact resume compatibility checks;
- persisted run metadata and algorithm/plugin version;
- benchmark thresholds and cancellation tests;
- documentation and one small reference graph per accepted algorithm family.

## Acceptance criteria

### Truth and safety

- No XGBoost label appears as implemented unless a real XGBoost adapter is
  available and tested.
- Native Gradient Boosting remains correctly named and retains passing
  operator, artifact, and predictor tests.
- Unsupported loss/optimizer routes fail closed; none silently default to
  CrossEntropy or Adam.
- One support manifest detects drift across backend, bindings, compiler,
  builder, runtime, and GUI metadata.

### Losses and optimizers

- Every graph-supported loss passes deterministic forward/backward parity for
  its supported reductions and parameters.
- Every graph-supported optimizer passes one-step and multi-step parity.
- Graph optimizer hyperparameters produce the same backend object semantics as
  equivalent C++ and Python construction.
- Adadelta, LAMB, KLDiv, and metric-learning losses have truthful supported or
  blocked states with specific reasons.
- Checkpoint v2 round-trips optimizer type, parameters, and internal state
  before exact resume is advertised.

### XGBoost

- The optional adapter can be absent without breaking the core engine.
- When present, it trains, evaluates, saves, loads, and predicts through the
  normal CyxWiz dataset/task/artifact boundaries.
- Cross-computation tests validate fixed-seed outputs against the selected
  XGBoost reference version.
- Binary APS classification may be used as a benchmark, but ticket completion
  does not depend on hard-coded APS behavior.

## Non-goals

- Replacing the native Gradient Boosting implementation with XGBoost.
- Shipping every loss or optimizer found in another framework.
- Treating a Python subprocess as an invisible always-on core dependency.
- Exposing metric-learning losses before pair/triplet data and shared-weight
  training contracts exist.
- Implementing LBFGS without closure support or SparseAdam without sparse
  gradients.
- Combining all classical ML and neural-network training into one monolithic
  model interface.

## Relevant implementation areas

- `cyxwiz-backend/include/cyxwiz/losses/`
- `cyxwiz-backend/src/algorithms/losses/`
- `cyxwiz-backend/include/cyxwiz/optimizers/`
- `cyxwiz-backend/src/algorithms/optimizers/`
- `cyxwiz-backend/python/bindings.cpp`
- `cyxwiz-engine/src/core/model_builder.cpp`
- `cyxwiz-engine/src/core/graph_compiler.cpp`
- `cyxwiz-engine/src/core/node_metadata_registry.cpp`
- `cyxwiz-engine/src/core/node_executors/gradient_boosting_*`
- `cyxwiz-engine/src/core/node_executors/tree_model_*`
- `cyxwiz-engine/tests/computation_truth/`
- `docs/Data Studio/tofix39.md`

## First implementation slice

The smallest production slice after Track70 stabilization is:

1. create the typed loss/optimizer support manifest;
2. add drift tests for backend/factory/compiler/builder/metadata agreement;
3. wire optimizer hyperparameters end to end;
4. add Adam, AdamW multi-step, SGD momentum, and RMSprop parity tests;
5. add BCE/BCEWithLogits, MSE/L1/SmoothL1, and NLL parity tests;
6. leave XGBoost optional and begin it only after this foundation passes.

This ordering improves correctness for every existing training graph before
adding another model family.
