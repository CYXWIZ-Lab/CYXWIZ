## 32) Node contract deep catalog (source -> model -> objective -> training control)

### 32.1 Canonical node role grammar (for compiler+runtime alignment)

```text
node      := source | preprocess | model | objective | metric | optimizer | scheduler | artifact | control
source    := dataset | parquet_loader | arrow_source | synthetic_input
preprocess:= resize | normalize | encode | tokenize | window | augment | split | shuffle | batch_transform
model     := layer_stack | graph_layer | dense | conv | rnn | attention | head | output
objective := loss_class | metric_loss | custom_objective
metric    := scalar_metric | ranking_metric | segmentation_metric | aux_metric
optimizer := sgd | adam | rmsprop | custom_optimizer
scheduler := static_lr | step_lr | plateau | cosine | custom_scheduler
artifact  := checkpoint | model_snapshot | tensorboard_sink
control   := validation | stopping | callback | seed | determinism
```

### 32.2 Minimum viable contract by role family

```text
Role       | Runtime required pins            | Optional pins             | Contract checks
-----------+---------------------------------|---------------------------+-----------------------------
source     | Data                            | Labels, Weights           | source type, schema, index policy
preprocess | DataIn -> DataOut                | LabelsIn/LabelsOut        | domain checks, arity, shape legality
model      | in/out feature tensors           | activations/bypass         | layer support matrix + backend placement
loss       | Predictions + Targets -> Scalar   | weights/class_weights      | output class/shape compatibility
metric     | Predictions + Targets -> Scalar   | thresholds                 | mode compatibility + reduce policy
optimizer  | Loss, parameters                 | grad_clip, momentum, decay | hyperparam bounds + mode capability
executor   | Config + plan                    | profiler, hooks            | mode init + source compatibility
artifact   | training state                   | metadata                   | path and serialization format checks
control    | callback channel                 | policy thresholds          | deterministic mode + timeout handling
```

### 32.3 Node transition invariants

```text
[source] --Data--> [preprocess]* --Data/Loss Inputs--> [model] --Predictions--> [loss] --Loss--> [metric]
                                                                     \
                                                                      +--> [optimizer] ---> [control]
                                                                      \
                                                                       +--> [scheduler]
```

### 32.4 Hard failure families by contract

- C_FAMILY_UNKNOWN: node family has no compiler runtime support
- C_PIN_MISMATCH: required pin direction/type missing
- C_CLASS_SHAPE_MISMATCH: class count / target tensor rank incompatible with selected loss
- P_DOMAIN_MISMATCH: preprocessing chain incompatible with domain (text/image/sequence/audio)
- M_OP_UNSUPPORTED: source operator cannot be materialized for chosen backend/source
- E_MODE_INIT_FAIL: training executor cannot bind mode (arrow/parquet/external)

### 32.5 Design principle for new node addition

For any new node family:
1. Define editor descriptor in `node_editor` with explicit pin contract.
2. Add/extend runtime capability support in `pipeline_runtime_capabilities`.
3. Add node executor mapping in `node_executor_factory`.
4. Add compile extraction path in `graph_compiler`.
5. Add preflight guard in `preflight_validator` when graph-level assumptions change.
6. Add executor/materializer behavior or explicit blocker with clear message.
7. Add failure-path mapping in one row of this tracker and section 24.

### 32.6 ASCII end-to-end node contract chain (stable, non-negotiable)

```text
1) Build graph IR
2) Infer roles -> validate pin graph
3) Materialize training dataset operators (if enabled)
4) Build compiled plan
5) Build executable model
6) Build mode executor + batcher
7) Run loop -> emit callbacks
8) Persist checkpoint -> emit run summary
```

No optional hop exists between step 5 and step 6:
- if build fails at any step, launch is blocked and returned as explicit issue code.
- define versioned JSON export for compile/preflight summaries.
- finalize an acceptance checklist for every training launch:
  - compile pass,
  - preflight pass,
  - materialization pass,
  - executor mode pass,
  - observability pass.
