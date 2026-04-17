# Local Debug Mode — Plan

**Filed:** 2026-04-17
**Source:** Plan agent
**Scope:** Third validation tier between Compile and Train — runs one
forward + one backward pass on synthetic data to catch runtime shape /
NaN / dead-gradient bugs before real training
**Size:** 4 commits, ~half day

## What it does

- After Compile passes, Local Debug takes the compiled `TrainingConfiguration`
- Builds the same model `TrainingExecutor` would
- Feeds synthetic tensors matching each layer's expected input shape
- Runs ONE forward pass, ONE backward pass, ONE optimizer step
- Reports per-layer shape + forward time, loss value + finiteness, gradient
  norms per learnable parameter, NaN/Inf flags
- Does NOT touch the user's real dataset — isolates graph wiring bugs from
  data bugs
- Does NOT trigger a reservation / Central Server call — fully local

## DebugResult Struct

```cpp
enum class DebugStage { NotRun, BuildModel, Forward, Loss, Backward, OptimizerStep, Complete };

struct LayerTrace {
    int node_id;
    std::string name;                     // "Dense_3"
    gui::NodeType type;
    std::vector<size_t> predicted_shape;  // from CompiledLayer::output_shape
    std::vector<size_t> actual_shape;     // observed at runtime
    float forward_ms = 0.0f;
    bool shape_matches = false;
    bool has_nan = false;
    bool has_inf = false;
};

struct GradNormEntry {
    std::string param_name;
    int layer_index = -1;
    float l2_norm = 0.0f;
    bool is_nan = false;
    bool is_zero = false;                 // dead subgraph indicator
};

struct DebugResult {
    bool success = false;
    DebugStage reached = DebugStage::NotRun;
    std::string failure_summary;
    std::vector<LayerTrace> layer_traces;
    float forward_total_ms = 0.0f;
    float backward_total_ms = 0.0f;
    float loss_value = NaN;
    bool loss_finite = false;
    std::vector<GradNormEntry> grad_norms;
    size_t params_with_grad = 0;
    size_t params_missing_grad = 0;
    std::vector<cyxwiz::ValidationIssue> issues;  // reuse existing severity
    std::chrono::steady_clock::time_point timestamp;
};
```

## SyntheticBatch

File: `cyxwiz-engine/src/core/synthetic_batch.h/.cpp`

```cpp
struct SyntheticBatch {
    Tensor features;
    Tensor labels;
};
SyntheticBatch MakeSyntheticBatch(const TrainingConfiguration& config,
                                  uint32_t seed = 1337);
```

Dispatch off `config.preprocessing_domain`:
- Tabular → `[1, input_size]` float32 in `[0, 1)`
- Image → `[1, C, H, W]` float32 in `[0, 1)`
- Text → `[1, seq_len]` int64 token IDs clamped to
  `[0, Embedding.num_embeddings - 1)` (reads embedding size from config.layers)
- TimeSeries → `[1, input_width, 1 + num_feature_cols]`
- Audio → `[1, n_mels, n_frames]`

Labels off `config.loss_type`:
- CrossEntropy/NLL → int64 `[1]` in `[0, num_classes)`
- BCE/BCEWithLogits → float `[1, num_classes]` in `{0, 1}`
- MSE/L1/SmoothL1 → float `[1, output_size]` in `[0, 1)`

Seeded for reproducibility.

## DebugExecutor

Path: `cyxwiz-engine/src/core/debug_executor.h/.cpp`.

**Does NOT inherit from `TrainingExecutor`** — too much dataset-backed state.
Instead: extract `BuildModelFromConfig` (`training_executor.cpp:77-411`) +
loss/optimizer setup (`:413-463`) into a free function
`BuildSequentialFromConfig(config)` callable from both. TrainingExecutor calls
the free function; DebugExecutor calls the same function.

```cpp
class DebugExecutor {
public:
    explicit DebugExecutor(TrainingConfiguration config);
    DebugResult Run();  // synchronous, ~200ms target
private:
    DebugResult result_;
    TrainingConfiguration config_;
    std::unique_ptr<SequentialModel> model_;
    std::unique_ptr<Optimizer> optimizer_;
    std::unique_ptr<Loss> loss_;
    bool RunForward(const Tensor& features);
    bool RunLoss(const Tensor& predictions, const Tensor& targets);
    bool RunBackward(const Tensor& predictions, const Tensor& targets);
    void CollectGradNorms();
};
```

## UI Surface (v1)

- New button in `node_editor.cpp::ShowToolbar` (insert after Compile ~line 887)
- Callback pattern mirrors `SetCompileCallback` → add `SetDebugCallback`
- Keyboard shortcut: **F6** (sits between F5=Train and F7=Compile)
- **Reuse the Compile popup** for v1. Extend `RenderCompileResultPopup` with
  a `compile_result_mode_` enum `{Compile, Debug, BlockedTrain}` switching
  the title. Header: "Local Debug Result — 1 forward + 1 backward pass,
  synthetic data"
- Populate `compile_result_issues_` from `DebugResult::issues` so all the
  existing issue-list rendering infrastructure applies unchanged.

v2 (out of scope): dedicated `DebugResultsPanel` with per-layer shape timeline,
grad norm histogram, NaN heatmap.

## File Layout

New files:
- `cyxwiz-engine/src/core/debug_executor.h` — structs + DebugExecutor class
- `cyxwiz-engine/src/core/debug_executor.cpp`
- `cyxwiz-engine/src/core/synthetic_batch.h`
- `cyxwiz-engine/src/core/synthetic_batch.cpp`
- `cyxwiz-engine/src/core/model_builder.h/.cpp` — extracted free function
- `cyxwiz-engine/tests/test_debug_executor.cpp`

Edits:
- `training_executor.cpp` — call `BuildSequentialFromConfig` (pure move)
- `node_editor.h/.cpp` — `DebugCallback` + toolbar button
- `main_window.h/.cpp` — `LocalDebugGraph()` mirroring `StartTrainingFromGraph`,
  `compile_result_mode_` field, F6 handler, popup title switch
- `CMakeLists.txt` — add new `.cpp` files

## Migration — 4 Commits

### Commit 1 — Structs + SyntheticBatch + model-builder extraction
- Add `debug_executor.h` (structs only, no executor yet)
- Add `synthetic_batch.h/.cpp` for Tabular + Text first (hardest cases)
  - Image/TimeSeries/Audio fall back to Tabular in v1
- Refactor: extract `BuildModelFromConfig` + loss/optimizer setup into
  `model_builder.cpp` as `BuildSequentialFromConfig(config)`. Pure move —
  existing training tests cover regression.
- Unit test: minimal Dense(10→32) → ReLU → Dense(32→4), CrossEntropy, Adam.
  Feed synthetic tabular batch, assert no throw.

### Commit 2 — DebugExecutor::Run
- Implement forward with per-layer shape capture (may need thin
  `TracingSequential` wrapper, or manual module loop reading `config.layers`)
- Collect grad norms via `SequentialModel::GetParameters()` pattern
- NaN/Inf detection on forward output, loss, each grad tensor
- Pathological test: hand-crafted all-zero-weights config →
  `params_missing_grad > 0`, Warning issue emitted

### Commit 3 — UI wiring
- `DebugCallback` in `node_editor.h/.cpp`
- "Local Debug" toolbar button (yellow-green, distinct from blue Compile)
- F6 handler in `main_window.cpp:3491`
- `MainWindow::LocalDebugGraph()` — runs Compile first (gate), then
  DebugExecutor, translates result into compile popup
- Extend popup title switch via `compile_result_mode_`

### Commit 4 — Optional: strict mode + staleness tracking
- Cache `last_debug_result_` keyed by `hash(nodes, links)`
- Warning in `StartTrainingFromGraph` if graph hash changed since last
  successful debug: "Consider F6 before F5"
- Engine-config flag `require_debug_before_train` upgrades warning to Error

## Risks

- **Half-initialized model on forward throw**: `BuildModelFromConfig`
  allocates CUDA/OpenCL buffers. A throw mid-build leaves `model_`
  partially populated. Wrap `Run()` in try/catch; destruct `model_`/
  `optimizer_` explicitly before returning failure.
- **Recurrent / Embedding shape awareness**: Text dispatch in
  `BuildModelFromConfig` (:140-167) does lookahead. `SyntheticBatch` must
  match — text input is `[1, seq_len]` int64 IDs, not flat floats.
  `seq_len` comes from `config.input_size` for text. Explicit unit test
  per domain.
- **Pathological synthetic gradients**: Random inputs + untrained weights
  can yield near-zero or NaN grads. Treat "grad == 0 for all params of a
  learnable layer" as a **Warning**. NaN in any grad → **Error**.
  Distinguishes dead subgraphs from real bugs.
- **Registry isolation**: DO NOT register synthetic batch with
  `DataRegistry`. Tensors live on stack. Unit test asserts registry state
  unchanged after Run.
- **Compile popup overload**: Three modes (Compile, Debug, BlockedTrain)
  risks UX confusion. Distinct header + status color per mode.
- **BuildModel refactor regression**: Lifting the 334-line switch into a
  free function must preserve identical logging + side-effects. Keep the
  refactor a pure move in commit 1.
- **Threading**: v1 is synchronous on UI thread (~200ms target). If real
  workloads exceed ~500ms (large CNNs), move to `AsyncTaskManager` task
  in v2. Don't prematurely thread.
