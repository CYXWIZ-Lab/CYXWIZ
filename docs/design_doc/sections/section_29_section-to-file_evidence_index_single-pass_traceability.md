## 29) Section-to-file evidence index (single-pass traceability)

Purpose:
- enforce that every numbered section has implementation anchor(s),
- keep the design tracker audit-friendly,
- keep future edits local and reversible.

```text
Section               Primary evidence files
-------------------  ----------------------------------------------
0) Versioned intent    docs/design_doc/*
1) Architecture       cyxwiz-engine/src/application.* 
                      cyxwiz-engine/src/gui/main_window.*
2) Layer responsibilities   same as above + gui/main_window.h/cpp
3) Core data contracts cyxwiz-engine/src/gui/node_editor.h
                      cyxwiz-engine/src/core/node_executors/*
4) Graph compilation   cyxwiz-engine/src/core/graph_compiler.*
                      cyxwiz-engine/src/core/compiled_graph_plan.*
5) Compiled plan       cyxwiz-engine/src/core/compiled_graph_plan.*
6) Executable model    cyxwiz-engine/src/core/executable_model.*
                      cyxwiz-engine/src/core/graph_executable_model.*
                      cyxwiz-engine/src/core/model_builder.*
7) Preflight gates     cyxwiz-engine/src/core/preflight_validator.*
8) Materialization     cyxwiz-engine/src/core/pipeline_materializer.*
                      cyxwiz-engine/src/core/pipeline_runtime_capabilities.*
9) Training orchestration cyxwiz-engine/src/core/training_manager.*
                      cyxwiz-engine/src/core/training_executor.*
10) Data-domain paths  cyxwiz-engine/src/core/*_config.*
                      cyxwiz-engine/src/core/graph_compiler.*
11) Error model       cyxwiz-engine/src/gui/main_window.*
                      cyxwiz-engine/src/core/preflight_validator.*
                      cyxwiz-engine/src/core/training_executor.*
12) Invariants         cross-cutting from 0-11 sections
13) Node matrix       cyxwiz-engine/src/gui/node_editor.h
                      cyxwiz-engine/src/core/pipeline_runtime_capabilities.*
14) State machine      cyxwiz-engine/src/core/training_executor.*
                      cyxwiz-engine/src/gui/main_window.*
15) Lean assessment    this document + all referenced headers
16) Phase plan        this document
17) Unknowns          code review pass + all above modules
18) Maintenance rules  repository docs conventions + this file
19) Phase 1 register  node editor + capabilities + compiler
20) Phase 2 register  training_executor + batcher + preflight + model_builder
21) Phase 3 register  pipeline_runtime_capabilities + compiler + training_executor
22) Phase 4 register  gui/graph_training_launcher.cpp
                      cyxwiz-engine/src/gui/main_window.cpp
23) Phase 5 register  callbacks/training logs in training_executor
                      + main_window + compile/preflight summaries
24) Phase 6 register  application/main_window/training_manager
25) Consolidation     this section + all referenced sections
26) Parity evidence   matrix and files listed in 27.2
27) Phase 7 register  matrix rows and claim checks in all above modules
28) Release gates     section 28 + all gate files above + log/checkpoint paths
29) Section-to-file evidence index  this section
30) Claim-to-source evidence index format this section
33) Execution state contract section  this section
```

### 29.1 Traceability maintenance rule
- when adding any claim, immediately append the owning file in this matrix.
- if a file is added here, add at least one issue or behavior anchor in the relevant phase section.

### 29.2 Immediate follow-up actions
- update section 0b snapshot after each review cycle.
- replace any remaining `U` statuses in matrix rows with evidence-backed symbols.
- align sign-off names with team ownership in Section 28.6.
