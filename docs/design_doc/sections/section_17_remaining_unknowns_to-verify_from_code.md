## 17) Remaining unknowns / to-verify from code


## 17.0 Current closure checkpoint (2026-07-04)

- Open verification unknowns requiring targeted evidence/tests: `7`.
- Documented closure notes already in active gap-tracking: `8` files under `sections/missing` (listed in section `54`).
- No additional engine design unknowns are currently discovered outside the items below.

## 17.1 What is now verified from source

Verified from `cyxwiz-engine/src/core/graph_compiler.cpp` and related compiler/runtime headers:

- Compile-time contract completeness for graph shape and connectivity is implemented in `GraphCompiler::Compile`, with:
  - one dataset source requirement,
  - loss + optimizer presence,
  - at least one model layer,
  - cycle detection,
  - required pin checks for labels / predictions / optimizer flow, and
  - explicit warning path for missing pre-train inspection nodes.
- Data loading truth is resolved by registry at compile time, not by stale UI hint flags. This is explicitly documented in the dataset block where stale `data_loaded` is overridden when registry has data.
- `DataLoader` contract is normalized during compile:
  - numeric parsing with defaults and clamping,
  - pinning flag unsupported warning (`pin_memory=true`),
  - checkpoint/patience/split/seed/balance policy extraction.
- Sequence and time-series branches are explicit:
  - `config.sequence_batch.enabled`,
  - `TimeSeriesWindow` switches `config.is_time_series`,
  - input sizing override for sequence windows.
- Training runtime control contract includes:
  - pause/stop/resume public methods,
  - atomic stop/pause state,
  - explicit wait loop while paused.
- Materialization/training compatibility is contract-based through
  `pipeline_runtime_capabilities.[h|cpp]`:
  - explicit operator-backed list,
  - explicit fail-closed list with hard reasons,
  - explicit legacy alias/source/input-arity/parameter capability catalogs.

## 17.2 Verified status by high-risk areas

| Area | Risk | Evidence now present | Remaining open?
|------|------|---------------------|--------------------|
| Core compile graph path (`GraphCompiler`) | Low | Complete checks and configuration contract extraction. | No |
| Training launch orchestration (`StartGraphTrainingFromCompiledConfig`) | Low | Explicit launcher result object, dispatch callback, and label resolution. | No |
| Arrow/Parquet batcher mode selection | Medium | Materializer/source-mode handling present in launcher + executor constructors. | No |
| Pause / stop ordering around callbacks | Medium | Pause/Stop signals are atomic and honored in training loops (`ShouldStop`, `WaitWhilePaused`). | Need deterministic callback ordering sample matrix in tests |
| RL nodes execution path | High | Multiple nodes are currently listed as fail-closed/unimplemented in pipeline runtime capability map. | Yes |
| Advanced audio path execution | High | Feature nodes exist in preprocessing and batcher config structs, but audio tensor ops are not all implemented in PipelineExecutor. | Yes |
| Python script ownership boundary for normal training | Low | Design sections separate script-runtime planes from C++ executor path. | No |
| `GraphExecutableModel` full consumption of `CompiledGraphPlan` | Medium | Plan object exists and is populated, but legacy sequential path is still primary executor path in current code. | Yes |

## 17.3 Remaining unknowns / next verification backlog

- Exact callback temporal ordering for every callback registration under nested pause/resume with validation gate transitions (test coverage is partial; needs one deterministic trace test).
- RL execution path for `PolicyNetwork/ValueNetwork/ReplayBufferNode` in a real launch loop.
- Advanced audio execution path for `Spectrogram/MelSpectrogram/MFCC` in full end-to-end launcher + batcher path.
- Provider/plugin lifecycle around custom training hooks under restart/reload boundaries.
- End-to-end guarantee that `materializer_source_kind == Unknown` only appears for explicitly intended legacy branches.
- Whether all node parameters in legacy aliases are normalized exactly once (audit needed for alias collisions).
- Python scripting ownership boundary is now documented in `docs/design_doc/sections/missing/missing_01_python_scripting_does_not_own_graph_training_model_build.md`.
- Callback ordering gap details are now documented in `docs/design_doc/sections/missing/missing_02_pause_stop_validation_callback_order.md`.
- RL node execution gap is now documented in `docs/design_doc/sections/missing/missing_03_rl_nodes_policy_value_replaybuffer_runtime_gap.md`.
- Advanced audio execution gap is now documented in `docs/design_doc/sections/missing/missing_04_audio_spectrogram_pipeline_gap.md`.
- Plugin restart/reload hook lifecycle gap is now documented in `docs/design_doc/sections/missing/missing_05_plugin_hook_restart_reload_lifecycle.md`.
- Unknown source-kind guard gap is now documented in `docs/design_doc/sections/missing/missing_06_materializer_source_kind_unknown_guard.md`.
- Legacy alias normalization idempotence gap is now documented in `docs/design_doc/sections/missing/missing_07_legacy_alias_param_normalization_once.md`.
- Graph executable consumption path gap is now documented in `docs/design_doc/sections/missing/missing_08_graph_executable_model_contract_consumption.md`.

## 17.4 Evidence-closure matrix (docs-ready, verification-open)

| Unknown area | Design note file | Closure state |
|---|---|---|
| Callback ordering under pause/stop/validation gates | [missing_02](/D:/Dev/CyxWiz_Claude/docs/design_doc/sections/missing/missing_02_pause_stop_validation_callback_order.md) | Design documented; execution-order proof needed |
| RL node runtime execution path (`PolicyNetwork`, `ValueNetwork`, `ReplayBufferNode`) | [missing_03](/D:/Dev/CyxWiz_Claude/docs/design_doc/sections/missing/missing_03_rl_nodes_policy_value_replaybuffer_runtime_gap.md) | Design documented; end-to-end launch proof needed |
| Advanced audio execution (`Spectrogram`, `MelSpectrogram`, `MFCC`) | [missing_04](/D:/Dev/CyxWiz_Claude/docs/design_doc/sections/missing/missing_04_audio_spectrogram_pipeline_gap.md) | Design documented; runtime proof needed |
| Training hook restart/reload lifecycle | [missing_05](/D:/Dev/CyxWiz_Claude/docs/design_doc/sections/missing/missing_05_plugin_hook_restart_reload_lifecycle.md) | Design documented; restart/reload matrix needed |
| `materializer_source_kind == Unknown` edge guarantee | [missing_06](/D:/Dev/CyxWiz_Claude/docs/design_doc/sections/missing/missing_06_materializer_source_kind_unknown_guard.md) | Design documented; explicit contract hardening needed |
| Legacy alias parameter normalization exactly-once | [missing_07](/D:/Dev/CyxWiz_Claude/docs/design_doc/sections/missing/missing_07_legacy_alias_param_normalization_once.md) | Design documented; idempotence proof needed |
| GraphExecutableModel contract consumption | [missing_08](/D:/Dev/CyxWiz_Claude/docs/design_doc/sections/missing/missing_08_graph_executable_model_contract_consumption.md) | Design documented; branch coverage needed |

## 17.5 Suggested closure plan for section

1. Add focused tests for each `GraphExecutor`-path risk above.
2. Gate release readiness by section 28 when all “Remaining open?” entries become `No`.
3. Keep this section as one per-version migration tracker, not permanent design canon.




