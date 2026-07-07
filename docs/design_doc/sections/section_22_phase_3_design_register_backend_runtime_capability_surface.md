## 22) Phase 3 design register (backend + runtime capability surface)

### 22.1 Objective
Define the exact runtime envelope and guarantee that unsupported nodes fail loudly at the right layer:
- runtime resolver behavior (operator-backed vs legacy vs fail-closed),
- materialization storage matrix by backend,
- training backend support and blocker contracts.

### 22.2 Capability model
Resolver contract is implemented as a fixed three-step fallback:
1) operator-backed lookup
2) fail-closed lookup
3) legacy lookup
4) unknown/implicit default

Observed support model:
- `PipelineRuntimeSupportMode::OperatorBacked` / `::FailClosed` / `::LegacyExecutor`
- `PipelineRuntimeFailMode::Real` / `::HardFail`
- `PipelineMaterializerStorageSupport::ArrowTableOnly` / `::None`
- `PipelineTrainingBackendSupportMode::Allowed` / `UnsupportedSequentialModelLayer` / `UnsupportedTrainingControl`

The same resolver populates required-input and validation metadata (allowed values,
integer constraints, float constraints, required parameters) from shared
capability tables.

### 22.3 Runtime resolver behavior

```text
legacy_name
   |
   +-- operator backed? -> OperatorBacked
   |      -> materializer: ArrowTableOnly
   |      -> executor: PipelineOperatorFactory
   |
   +-- fail-closed? -> FailClosed + HardFail
   |      -> materializer: none
   |      -> executor: disabled
   |
   +-- legacy alias/runtime? -> LegacyExecutor
   |      -> materializer: none
   |      -> executor: PipelineExecutor
   |
   \-- unknown -> default Unknown
```

Current implementation sets this precedence directly inside
`ResolvePipelineRuntimeSupport(const std::string&)`.

### 22.4 Backend surface by dataset and materializer support

```text
Source kind                Source query            Materializer result
-------------------------  ---------------------- --------------------------------------
ArrowTable                 registry.IsArrowDataset  supported:
                                                 operators run, result can register
                                                 <dataset>__materialized

ParquetBacked              registry.IsParquet...    unsupported:
                                                 unsupported reason: cannot rewrite row groups

ImageDataset               registry.IsImageDataset   unsupported:
                                                 no table rewrite path; external image batcher used

AudioDataset               registry.IsAudioDataset   unsupported:
                                                 no table rewrite path; external audio batcher used

TextDataset                registry.IsTextDataset    unsupported:
                                                 no table rewrite path; external text batcher used

Unknown                    no backend flag          unsupported + diagnostic
```

Materializer behavior:
- returns `MaterializeResult` with:
  - `source_kind`
  - `skipped_unsupported_source`
  - `unsupported_source_reason`
- registers `<dataset>__materialized` only when `operators_applied > 0`.
- leaves `operators_applied == 0` for pass-through cases.

### 22.5 Backend matrix by execution path

```text
+--------------------------+-----------------------------+------------------------------+
| Path                    | Runtime mode                 | Training entrypoint            |
+--------------------------+-----------------------------+------------------------------+
| Arrow source             | Operator / legacy resolution  | StartTrainingArrow             |
| Parquet source           | Operator / legacy resolution  | StartTrainingParquet           |
| Image source             | Operator / legacy resolution  | StartTrainingImage (external)  |
| Audio source             | Operator / legacy resolution  | StartTrainingAudio (external)  |
| Text source              | Operator / legacy resolution  | StartTrainingText (external)   |
| Legacy DatasetHandle     | Legacy                       | StartTraining (legacy)         |
| Sequence enabled (Arrow)  | Any resolved source          | BuildSequenceBatcher + StartTrainingSequence |
+--------------------------+-----------------------------+------------------------------+
```

Sequence launch is enforced separately from source backend by checking
`config.sequence_batch.enabled`.

### 22.6 Failure policy by phase

```text
Compile phase:
  - unsupported training role/control/sequential nodes -> compile/graph issue
Materialization phase:
  - unsupported backend support -> skipped_unsupported_source + continue
Launch/Dispatch phase:
  - hard blockers return LaunchResult blocked and no async run
Runtime phase:
  - dataset/batcher constructor failures -> false from StartTraining* and async completion failure
```

Important nuance:
- materialization skip is not a hard fail by itself; hard fail happens if a required
  runtime contract downstream cannot be honored.

### 22.7 Runtime taxonomy (contract lists)
- Operator-backed list includes preprocessing/transform nodes that map to `gui::NodeType`
  and are expected to have no pass-through legacy behavior.
- Fail-closed list includes nodes whose runtime contracts exist in metadata but are not executed:
  activations, many losses/optimizers, tensor ops, classic ML models, RL nodes, I/O and export nodes.
- Legacy list maps historical names to canonical node runtime types.
- Alias list contains compatibility decisions:
  - canonicalization
  - hidden compatibility aliases
  - reason metadata for migration cost

### 22.8 Training-capability role matrix

```text
Node role                  Status source                     Behavior
-------------------------  --------------------------------  -------------------------
Model layer                 Supported model node table         Compiled into sequential path
Activation                  Supported as model role table      Compiled into activation layer
Loss                        Supported as model role table      Compiled into loss configuration
Optimizer                   Supported as model role table      Compiled into optimizer configuration
Sequential model             Unsupported list                  Hard/contract stop in training backend
Training-control scheduler   Unsupported list                  Not wired into executor config
```

Unsupported examples in training backend:
- `Conv2D`, `MaxPool2D`, `RNN`, `PolicyNetwork`, `ValueNetwork`
- schedulers: `StepLR`, `CosineAnnealing`, `ReduceOnPlateau`, etc.

### 22.9 Memory and safety contracts
- materializer writes only Arrow-derived tables.
- sequence launch validates required columns before start.
- `TrainingExecutor` receives an explicit dataset mode (`Arrow`, `Parquet`, `External`,
  `SequenceExternal`, `Legacy`) and selects batcher mode deterministically.

### 22.10 Known debt
- runtime still has compatibility layers for broad legacy surface in graph UI.
- many legacy nodes remain UI-only placeholders by design.
- some training-control nodes remain in the node catalog before being wired in runtime.

### 22.11 Phase 3 completion criteria
- complete evidence-backed matrix for resolver order and per-backend materializer support.
- complete evidence-backed table for training support roles and unsupported categories.
- convert each support contract item to a section-to-file/claim-to-source evidence line in follow-up section.
