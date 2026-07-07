## 30) Claim-to-source evidence index format (exact traceability)

Use this format for all future hardening updates:

```text
Claim-ID           | Section | Claim text                               | Evidence files                    | Evidence symbol                          | Owner
------------------+---------+------------------------------------------+-----------------------------------+----------------------------------------+--------------
C-01-compile-gate  | 4.3/27.2| Compile rejects cycles                     | core/graph_compiler.cpp           | core/graph_compiler.cpp:2749 GraphCompiler::Compile ; core/graph_compiler.cpp:2811 HasCycle(nodes, links) ; core/graph_compiler.cpp:4801 HasCycle | Engine
C-02-plan-ids       | 5/21.6  | Role pins are materialized in plan         | core/compiled_graph_plan.*         | core/compiled_graph_plan.h:36 CompiledGraphPlan ; core/compiled_graph_plan.h:51 BuildCompiledGraphPlan ; core/compiled_graph_plan.cpp:76 BuildCompiledGraphPlan | Engine
C-03-batcher-arrow  | 9.4/21.2| Arrow batch path selected                  | core/training_batcher_setup.cpp    | core/training_batcher_setup.cpp:301 BuildArrowTrainingBatchers ; core/training_batcher_setup.cpp:330 GetPreprocessorConfig | Engine
C-04-batcher-parquet| 9.4/21.2| Parquet batch path selected                | core/training_batcher_setup.cpp    | core/training_batcher_setup.cpp:416 BuildParquetTrainingBatchers | Engine
C-05-materialize    | 8/9.2  | Materializer applies source operator chain   | core/pipeline_materializer.cpp     | core/pipeline_materializer.cpp:74 PipelineMaterializer::Materialize ; core/pipeline_materializer.cpp:126 MaterializeTable | Core
C-06-runtime-gpu    | 25.3   | GPU/placement policy is validated            | core/graph_compiler.cpp           | core/graph_compiler.cpp:1453 AddBackendPlacementReports ; core/graph_compiler.cpp:3919 config.SummarizeBackendPlacements | Core
C-07-callback       | 24.2   | Run event stream includes run/component id  | core/training_executor.cpp, gui/main_window.cpp | core/training_executor.cpp:478 last_run->run_id ; core/training_executor.cpp:480 RecordRuntimeEvent ; core/training_executor.cpp:639 RecordValidationMetrics ; core/training_executor.cpp:661 RecordCheckpointSaved ; gui/main_window.cpp:3373 run_id pass-through | Runtime
C-08-ui-launch-gates | 24.3/28   | UI blocks blocked launch paths                  | gui/main_window.cpp               | gui/main_window.cpp:2977 blocked by graph errors ; gui/main_window.cpp:3010 blocked by warning ; gui/main_window.cpp:3237 Validate(config...) ; gui/main_window.cpp:3488 preflight_validator.Validate | GUI
C-09-trace-instrument| 24.2/30 | Runtime trace events emitted with context         | core/training_executor.cpp, gui/main_window.cpp | core/training_executor.cpp:468 StartRun ; core/training_executor.cpp:480 RecordRuntimeEvent ; core/training_executor.cpp:919 RecordStage/GetNextBatch ; core/training_executor.cpp:1066 RecordStage/BatchCallback ; core/training_executor.cpp:1881 RecordStage/BatchCallback ; gui/main_window.cpp:3070 run callback dispatch | Runtime
C-10-state-contract | 33 | State transitions are deterministic and failure coded | core/graph_compiler.cpp, core/pipeline_materializer.cpp, core/training_manager.cpp, core/training_executor.cpp, gui/main_window.cpp | core/graph_compiler.cpp:2749 GraphCompiler::Compile ; core/pipeline_materializer.cpp:74 PipelineMaterializer::Materialize ; core/training_manager.cpp:54 StartTrainingCommon ; core/training_manager.cpp:562 StopTraining ; core/training_manager.cpp:602 PauseTraining ; core/training_manager.cpp:609 ResumeTraining ; core/training_executor.cpp:168 Initialize ; core/training_executor.cpp:217 Train ; core/training_executor.cpp:1311 Stop ; core/training_executor.cpp:1316 Pause ; core/training_executor.cpp:1324 Resume ; gui/main_window.cpp:2963 MainWindow::StartTrainingFromGraph ; gui/main_window.cpp:3021 GraphCompiler Compile call ; gui/main_window.cpp:3062 materialization-failure branch ; gui/main_window.cpp:3235 PreflightValidator::Validate | Runtime
``` 

Rule:
- every entry must have at least one function symbol and a planned maintenance owner.
- every `Claim-ID` must be referenced in one paragraph before it is considered complete.

### 30.1 Suggested upkeep script

When you have time to collect exact line spans:
- use `rg -n "symbol|function|class"` over the relevant file list,
- then replace each Evidence symbol with `path:line`.

### 30.2 Acceptance gate
- No release-ready claim in Sections 4-30 without a `Claim-ID`.
- No `Claim-ID` remains without an owning file symbol and owner.
