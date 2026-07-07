## 21) Phase 2 design register (training contract & runtime semantics)

### 21.1 Goal
- make all training data and execution contracts explicit and machine-checkable:
  - what enters training,
  - how labels/tensors are formatted,
  - where shape failures are raised,
  - how sequence and external modes diverge.

### 21.2 Compile-to-executor contract

```text
GraphCompiler
   |
   +--> TrainingConfiguration
          - dataset_source_id
          - loss / optimizer / metrics ids
          - backendPlacement
          - preprocessingDomain
          - class_weight policy
          - sequence settings
          - split + batch + epoch params
          |
          +--> CompiledGraphPlan
          |      - data labels predictions loss optimizer pin ids
          |      - model_path_node_ids (ordered)
          |      - compiled edges/pin wiring
          |
          +--> PreflightValidator
                 - readiness checks
                 - shape/loss hints
                 - validation summary
                 |
                 +--> TrainingExecutor init
                        - BuildExecutableFromConfig
                        - batcher setup
                        - materialized plan binding
                        - callback wiring
                        |
                        +--> train loops
```

### 21.3 Training execution modes
- legacy mode:
  - older dataset/iterator path,
  - used when source does not route through Arrow/Parquet materializer path.
- Arrow mode:
  - source-driven pipeline creation through `BuildArrowTrainingBatchers`.
- Parquet mode:
  - source-driven pipeline creation through `BuildParquetTrainingBatchers`.
- external mode:
  - user-defined external batch source integration.
- sequence external mode:
  - explicit time-series/sequence branch.

### 21.4 Batcher contract
- `TrainingBatcherSet` holds:
  - batch size,
  - workers,
  - shuffling and split policies,
  - prefetching characteristics.
- `TrainingInputSizeResolution` resolves shape expectations from compiler graph and batch source.
- Arrow/Parquet builders share:
  - source column mapping to model data/target fields,
  - per-step batch tuple layout,
  - optional balancing/class-weight routing.
- Failure behavior:
  - builder errors are surfaced before training loop start,
  - mis-sized input signatures stop launch.

### 21.5 Data contract across domains

#### General (tabular / dense)
- data tensor shape must be compatible with input layer width and activation/flatten assumptions.
- label tensor for classification must respect loss contract:
  - CE/Focal require target categories and class-count checks,
  - BCE/BCEWithLogits require binary-compatible target format.
- optimizer/loss nodes are selected by resolved compiler ids only, not by free node name.

#### Image
- image path checks enforce resize/transform chain expectations before Dense transitions where required.
- normalization and channel expectations come from preprocessing nodes attached to data path.
- label format follows dataset and final loss contract.

#### Text
- tokenizer/vectorizer/text-clean nodes are normalized into preprocessing domain config.
- sequence-like text models inherit sequence training branch when enabled by graph path.
- padding and truncation settings are validated through text config aggregation.

#### Sequence / time-series
- `TimeSeriesWindow` can override split behavior and windowing assumptions.
- sequence external mode is treated distinctly in executor construction.
- recurrent/loss compatibility is guarded by layer support checks.

#### Audio
- audio nodes mostly remain in compatibility layer until full executor/path support is confirmed.
- compile preflight should block unsupported pure-audio materialization for training launches.

### 21.6 Node-to-training contract details (critical edges)
- data source -> training graph path:
  - exactly one active dataset source candidate preferred;
  - split chain attached through `DataSplit` metadata when present.
- labels path:
  - graph must connect loss target input to labels semantic source.
- predictions path:
  - model output chain must connect into loss predictions input pin.
- optimizer path:
  - optimizer node must reach loss graph input according to `GraphCompiler` connectivity model.
- training loop consumes:
  - pin role ids from `CompiledGraphPlan`
  - model/runtime handle from executable builder.

ASCII mapping:

```text
DataSource
  --> Preprocess chain
       --> Model input path (layers)
            --> Loss prediction input
            --> Loss target input
                 --> Optimizer input
```

### 21.7 Loss and output-size checks
From compile-level policy:
- `CrossEntropyLoss`, `FocalLoss`:
  - class count and output dimension alignment checked.
- `BCELoss`, `BCEWithLogits`:
  - binary output assumptions checked for final activations.
- regression losses (`MSE`, `L1`, `SmoothL1`, `Huber`):
  - dimensional alignment against target shape and model output checked.
- invalid pairings are compile/preflight blocking issues.

### 21.8 Class imbalance support
- class-weighting and class balancing are fed into training config before batcher construction.
- balancing path remains optional and depends on loss/output category.
- for multi-class pipelines, class weight source is expected from node config when present.

### 21.9 Checkpoint and metric policy
- checkpoint interval and restore policy are part of executor control.
- best metric capture can trigger restore logic after validation checkpoints.
- resume/stop interacts with:
  - active loop state,
  - callback channel progress,
  - final result summary.

### 21.10 Runtime state transitions during training

```text
CREATE_EXECUTOR
  -> PREPARE_BATCHERS
    -> INIT_EPOCH
      -> FOR_EACH_EPOCH
            -> FOR_EACH_BATCH
                  -> FORWARD_BACKWARD
                  -> METRICS
                  -> VALIDATION_GATE
            -> CHECKPOINT_GATE
            -> EARLY_STOP_CHECK
      -> DONE
```

Where:
- `VALIDATION_GATE` depends on configured validation cadence.
- `CHECKPOINT_GATE` can trigger best-checkpoint snapshot.
- every transition emits callbacks unless stop/pause state forces break.

### 21.11 Error model in training contract
- compile/preflight failures block launch before threads spawn.
- runtime exceptions:
  - convert to training failure result,
- validation failures while running:
  - can stop current epoch/branch with deterministic partial-state result.
- pause/stop are explicit control paths, not restart-as-default behavior.

### 21.12 Phase 2 completion criteria
- all training modes documented with accepted data tuple schema,
- all major loss types mapped to required tensor shape and class cardinality checks,
- sequence and non-sequence launch divergence documented with source-level guard points,
- materialized operator path and batcher path references added in file-level tracker list.
