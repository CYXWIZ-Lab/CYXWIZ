## 4) Graph compilation deep dive

### 4.1 Compile input -> output contract
Input:
- editor graph nodes and links, plus project-level context.

Output:
- `CompileResult`:
  - `TrainingConfiguration` (main runtime blueprint),
  - `CompiledGraphPlan` (ordered plan),
  - issue list with severity.

### 4.2 `TrainingConfiguration` responsibilities
`TrainingConfiguration` is the central immutable-ish plan-of-record for training launches.
From the code it carries:
- model-level selectors (loss, optimizer, metrics, scheduler candidates),
- dataset + labels + data split + augmentation/preprocessing policy,
- domain context (`PreprocessingDomain`, image/audio/text flags),
- backend placement hints (CPU/CUDA/ArrayFire),
- validation issues and state flags.

### 4.3 Compiler pipeline

ASCII:

```
[Graph + links]
      |
      v
[GraphCompiler::Compile]
  - node collection and topological order
  - pin/edge normalization
  - source/loss/optimizer walk
      |
      +--> [Role extraction]
      |      - data source
      |      - model layer set
      |      - loss node
      |      - optimizer node
      |      - pre-processing chain
      |
      +--> [Config assembly]
      |      - training hyperparams
      |      - preprocessing/domain configs
      |      - backend/placement hints
      |
      +--> [CompileGraphPlan]
      |      - selected node list
      |      - pin-id semantics
      |
      +--> [Graph-specific checks]
             - cycles
             - required pin reachability
             - single source/target consistency
             - backend compatibility
             - domain constraints
```

### 4.4 Structural validation sequence (from compiler)
The compiler enforces:
- DAG shape of the graph.
- presence of required anchors (dataset, model entry, loss, optimizer depending on mode).
- required pin direction + reachability:
  - `Loss Targets -> Labels`
  - `Loss Predictions -> Model output path`
  - `Optimizer -> Loss`
- single active dataset source policy for training path.
- single-loss restrictions in some branch checks.

### 4.5 Domain-aware extraction
Domain logic adjusts compile behavior for:
- text preprocessing nodes (`Text*`) and tokenizer/padding metadata,
- image-specific checks (e.g. size transform requirements),
- time-series settings (`TimeSeriesWindow`, lag, split),
- class imbalance handling (`class_weights` and balancing flag).

### 4.6 Layer/model graph extraction and capability checks
The compiler:
- traverses model layer chain from input/loss ancestors.
- applies supported-node filters for graph runtime compilation.
- marks unsupported sequences as warnings/errors depending on strictness and execution path.
- builds backend placement report (`BackendPlacement` status objects).

### 4.7 Shape and metric-learning guards
Observed explicit checks include:
- output-class-size constraints for BCE/CE family losses,
- metric-learning contract probing (`AnalyzeMetricLearningGraphContract`) for pair/triplet-style graphs,
- rough memory usage checks to avoid pathological allocations.

### 4.8 Compile result semantics
- Compile issues are categorized with explicit severity.
- `TrainingConfiguration::is_valid` reflects compile blocking-state.
- `error_message` acts as human-readable aggregation of blocking issues.

---
