# CyxWiz Engine Track Design

## Design split index (one source of truth)

This file points to the full sectionized design documentation in `docs/design_doc/sections`.

### Current phase status snapshot and source preface

# CyxWiz Engine Track Design (Comprehensive)

Created: `2026-07-03`  
Scope: End-to-end design, from computer layer and graph model down to materialization, training execution, node contracts, and diagnostics.

This document follows lean-engineering guardrails:
- keep the stable core small,
- make mandatory behavior explicit,
- isolate optional compatibility and adapters,
- expose interfaces through concrete data contracts.

### 0a) Quick navigation (ASCII TOC)

```text
0) Versioned design intent
1) Architecture map
2) Layer responsibilities
3) Core data contracts
4) Graph compilation deep dive
5) Compiled graph plan contract
6) Executable model abstraction
7) Preflight validation and safety gates
8) Materialization architecture
9) Training orchestration and execution
10) Data-domain paths
11) Error model and observability
12) Critical invariants
13) Node-contract matrix (high-level)
14) End-to-end dataflow state machine
15) Lean guardrail assessment
16) Phase plan (lean staged documentation)
17) Remaining unknowns / to-verify
18) Maintenance rules for this file
19) Phase 1 design register (node-contract audit)
20) Phase 1 completion criteria
21) Phase 2 design register (training contract & runtime semantics)
22) Phase 3 design register (backend + runtime capability surface)
23) Phase 4 design register (walkthrough examples)
24) Phase 5 design register (observability, ops, and lifecycle quality)
25) Phase 6 design register (compute layer + frontend/backend synchronization)
26) Consolidation register (status, risks, and governance)
27) Phase 7 design register (parity evidence + sign-off model)
28) Phase 8 design register (release readiness + hard gates)
29) Section-to-file evidence index
30) Claim-to-source evidence index format
31) Pending claim completion backlog
35) Runtime diagnostics, traceability, and failure contracts
36) Training dashboard telemetry, UI contracts, and Python bridge
37) Async control-plane, task lifecycle, and UI visibility contracts
38) Lifecycle and shutdown safety contracts
39) Project session, Python environment, and registry hygiene contracts
40) Startup/bootstrap, process spawn, and close-confirmation contracts
41) Console and observability contracts
42) Crash and runtime trace contracts
43) Trace persistence and replay contracts
44) Studio Debugger runtime-trace recovery contracts
45) Debug diagnostics import/export and support-bundle contracts
46) Studio Debugger trace visibility and filtering contracts
47) Studio Debugger run orchestration and persistence contracts
48) Studio Debugger training trace contracts
49) Plugin extensibility and compatibility contracts
50) Data I/O, dataset loading, and training backend contracts
51) Runtime fail-closed behavior and training-capability gap register
52) Training lifecycle and callback-order contracts
53) Python scripting and model-build boundary contracts
54) Missing design closure notes index
``` 

### 0b) Current phase status snapshot

```text
Phase | Status     | Evidence maturity | Risk
------+------------+------------------+-----------
0     | DONE       | High             | Low
1     | COMPLETE   | High             | Low
2     | DONE       | High             | Medium
3     | DONE       | Medium           | Medium
4     | DONE       | Medium           | Medium
5     | COMPLETE   | High             | Low
6     | DONE       | Medium           | Medium
7     | INPROGRESS | Medium           | Medium
8     | DONE       | Low-Medium       | Low
```

> Snapshot is refreshed manually; align with Section 26 for governance workflow.

---

## Section index

| # | Section | File |
|---|---------|------|
| 0 | Versioned design intent | [docs/design_doc/sections/section_00_versioned_design_intent.md](sections/section_00_versioned_design_intent.md) |
| 1 | High-level architecture (ASCII) | [docs/design_doc/sections/section_01_high-level_architecture_ascii.md](sections/section_01_high-level_architecture_ascii.md) |
| 2 | Layer-by-layer design responsibilities | [docs/design_doc/sections/section_02_layer-by-layer_design_responsibilities.md](sections/section_02_layer-by-layer_design_responsibilities.md) |
| 3 | Core data contracts | [docs/design_doc/sections/section_03_core_data_contracts.md](sections/section_03_core_data_contracts.md) |
| 4 | Graph compilation deep dive | [docs/design_doc/sections/section_04_graph_compilation_deep_dive.md](sections/section_04_graph_compilation_deep_dive.md) |
| 5 | Compiled graph plan contract (pin-aware) | [docs/design_doc/sections/section_05_compiled_graph_plan_contract_pin-aware.md](sections/section_05_compiled_graph_plan_contract_pin-aware.md) |
| 6 | Executable model abstraction and building | [docs/design_doc/sections/section_06_executable_model_abstraction_and_building.md](sections/section_06_executable_model_abstraction_and_building.md) |
| 7 | Preflight validation and safety gates | [docs/design_doc/sections/section_07_preflight_validation_and_safety_gates.md](sections/section_07_preflight_validation_and_safety_gates.md) |
| 8 | Materialization architecture | [docs/design_doc/sections/section_08_materialization_architecture.md](sections/section_08_materialization_architecture.md) |
| 9 | Training orchestration and execution | [docs/design_doc/sections/section_09_training_orchestration_and_execution.md](sections/section_09_training_orchestration_and_execution.md) |
| 10 | Data-domain paths (from node graph to runtime behavior) | [docs/design_doc/sections/section_10_data-domain_paths_from_node_graph_to_runtime_behavior.md](sections/section_10_data-domain_paths_from_node_graph_to_runtime_behavior.md) |
| 11 | Error model and observability | [docs/design_doc/sections/section_11_error_model_and_observability.md](sections/section_11_error_model_and_observability.md) |
| 12 | Critical invariants (must remain true) | [docs/design_doc/sections/section_12_critical_invariants_must_remain_true.md](sections/section_12_critical_invariants_must_remain_true.md) |
| 13 | Node-contract matrix (high-level) | [docs/design_doc/sections/section_13_node-contract_matrix_high-level.md](sections/section_13_node-contract_matrix_high-level.md) |
| 14 | End-to-end dataflow state machine (training path) | [docs/design_doc/sections/section_14_end-to-end_dataflow_state_machine_training_path.md](sections/section_14_end-to-end_dataflow_state_machine_training_path.md) |
| 15 | Lean guardrail assessment (applied to this design) | [docs/design_doc/sections/section_15_lean_guardrail_assessment_applied_to_this_design.md](sections/section_15_lean_guardrail_assessment_applied_to_this_design.md) |
| 16 | Phase plan (lean, staged documentation) | [docs/design_doc/sections/section_16_phase_plan_lean_staged_documentation.md](sections/section_16_phase_plan_lean_staged_documentation.md) |
| 17 | Remaining unknowns / to-verify from code | [docs/design_doc/sections/section_17_remaining_unknowns_to-verify_from_code.md](sections/section_17_remaining_unknowns_to-verify_from_code.md) |
| 18 | Maintenance rules for this file | [docs/design_doc/sections/section_18_maintenance_rules_for_this_file.md](sections/section_18_maintenance_rules_for_this_file.md) |
| 19 | Phase 1 design register (node-contract audit) | [docs/design_doc/sections/section_19_phase_1_design_register_node-contract_audit.md](sections/section_19_phase_1_design_register_node-contract_audit.md) |
| 20 | Phase 1 completion criteria | [docs/design_doc/sections/section_20_phase_1_completion_criteria.md](sections/section_20_phase_1_completion_criteria.md) |
| 21 | Phase 2 design register (training contract & runtime semantics) | [docs/design_doc/sections/section_21_phase_2_design_register_training_contract_runtime_semantics.md](sections/section_21_phase_2_design_register_training_contract_runtime_semantics.md) |
| 22 | Phase 3 design register (backend + runtime capability surface) | [docs/design_doc/sections/section_22_phase_3_design_register_backend_runtime_capability_surface.md](sections/section_22_phase_3_design_register_backend_runtime_capability_surface.md) |
| 23 | Phase 4 design register (walkthrough examples) | [docs/design_doc/sections/section_23_phase_4_design_register_walkthrough_examples.md](sections/section_23_phase_4_design_register_walkthrough_examples.md) |
| 24 | Phase 5 design register (observability, ops, and lifecycle quality) | [docs/design_doc/sections/section_24_phase_5_design_register_observability_ops_and_lifecycle_quality.md](sections/section_24_phase_5_design_register_observability_ops_and_lifecycle_quality.md) |
| 25 | Phase 6 design register (compute layer + frontend/backend synchronization) | [docs/design_doc/sections/section_25_phase_6_design_register_compute_layer_frontendbackend_synchronization.md](sections/section_25_phase_6_design_register_compute_layer_frontendbackend_synchronization.md) |
| 26 | Consolidation register (status, risks, and governance) | [docs/design_doc/sections/section_26_consolidation_register_status_risks_and_governance.md](sections/section_26_consolidation_register_status_risks_and_governance.md) |
| 27 | Phase 7 design register (parity evidence + sign-off model) | [docs/design_doc/sections/section_27_phase_7_design_register_parity_evidence_sign-off_model.md](sections/section_27_phase_7_design_register_parity_evidence_sign-off_model.md) |
| 28 | Phase 8 design register (release readiness + hard gates) | [docs/design_doc/sections/section_28_phase_8_design_register_release_readiness_hard_gates.md](sections/section_28_phase_8_design_register_release_readiness_hard_gates.md) |
| 29 | Section-to-file evidence index (single-pass traceability) | [docs/design_doc/sections/section_29_section-to-file_evidence_index_single-pass_traceability.md](sections/section_29_section-to-file_evidence_index_single-pass_traceability.md) |
| 30 | Claim-to-source evidence index format (exact traceability) | [docs/design_doc/sections/section_30_claim-to-source_evidence_index_format_exact_traceability.md](sections/section_30_claim-to-source_evidence_index_format_exact_traceability.md) |
| 31 | Pending claim completion backlog (next proceed-ready cycle) | [docs/design_doc/sections/section_31_pending_claim_completion_backlog_next_proceed-ready_cycle.md](sections/section_31_pending_claim_completion_backlog_next_proceed-ready_cycle.md) |
| 32 | Node contract deep catalog (source -> model -> objective -> training control) | [docs/design_doc/sections/section_32_node_contract_deep_catalog_source_-_model_-_objective_-_training_control.md](sections/section_32_node_contract_deep_catalog_source_-_model_-_objective_-_training_control.md) |
| 33 | Cross-layer execution contract (compiler -> materializer -> executor -> runtime events) | [docs/design_doc/sections/section_33_cross-layer_execution_contract_compiler_-_materializer_-_executor_-_runtime_events.md](sections/section_33_cross-layer_execution_contract_compiler_-_materializer_-_executor_-_runtime_events.md) |
| 34 | Runtime control-plane and state lifecycle register | [docs/design_doc/sections/section_34_phase_9_design_register_runtime_control_plane_and_state_lifecycle.md](sections/section_34_phase_9_design_register_runtime_control_plane_and_state_lifecycle.md) |
| 35 | Runtime diagnostics, traceability, and failure contracts | [docs/design_doc/sections/section_35_phase_9_design_register_runtime_diagnostics_traceability_and_failure_contracts.md](sections/section_35_phase_9_design_register_runtime_diagnostics_traceability_and_failure_contracts.md) |
| 36 | Training dashboard telemetry, UI contracts, and Python bridge | [docs/design_doc/sections/section_36_phase_9_design_register_training_dashboard_telemetry_ui_contracts_and_python_bridge.md](sections/section_36_phase_9_design_register_training_dashboard_telemetry_ui_contracts_and_python_bridge.md) |
| 37 | Async control-plane, task lifecycle, and UI visibility contracts | [docs/design_doc/sections/section_37_phase_9_design_register_async_control_plane_task_lifecycle_and_ui_visibility_contracts.md](sections/section_37_phase_9_design_register_async_control_plane_task_lifecycle_and_ui_visibility_contracts.md) |
| 38 | Lifecycle and shutdown safety contracts | [docs/design_doc/sections/section_38_phase_9_design_register_lifecycle_and_shutdown_safety_contracts.md](sections/section_38_phase_9_design_register_lifecycle_and_shutdown_safety_contracts.md) |
| 39 | Project session, Python environment, and registry hygiene contracts | [docs/design_doc/sections/section_39_phase_9_design_register_project_session_and_python_environment_and_registry_hygiene_contracts.md](sections/section_39_phase_9_design_register_project_session_and_python_environment_and_registry_hygiene_contracts.md) |
| 40 | Startup/bootstrap, process spawn, and close-confirmation contracts | [docs/design_doc/sections/section_40_phase_9_design_register_startup_bootstrap_process_spawn_and_close_confirmation_contracts.md](sections/section_40_phase_9_design_register_startup_bootstrap_process_spawn_and_close_confirmation_contracts.md) |
| 41 | Console and observability contracts | [docs/design_doc/sections/section_41_phase_9_design_register_console_and_observability_contracts.md](sections/section_41_phase_9_design_register_console_and_observability_contracts.md) |
| 42 | Crash and runtime trace contracts | [docs/design_doc/sections/section_42_phase_9_design_register_crash_and_runtime_trace_contracts.md](sections/section_42_phase_9_design_register_crash_and_runtime_trace_contracts.md) |
| 43 | Trace persistence and replay contracts | [docs/design_doc/sections/section_43_phase_9_design_register_trace_persistence_and_replay_contracts.md](sections/section_43_phase_9_design_register_trace_persistence_and_replay_contracts.md) |
| 44 | Studio Debugger runtime-trace recovery contracts | [docs/design_doc/sections/section_44_phase_9_design_register_studio_debugger_runtime_trace_recovery_contracts.md](sections/section_44_phase_9_design_register_studio_debugger_runtime_trace_recovery_contracts.md) |
| 45 | Debug diagnostics import/export and support-bundle contracts | [docs/design_doc/sections/section_45_phase_9_design_register_debug_diagnostics_import_export_and_support_bundle_contracts.md](sections/section_45_phase_9_design_register_debug_diagnostics_import_export_and_support_bundle_contracts.md) |
| 46 | Studio Debugger trace visibility and filtering contracts | [docs/design_doc/sections/section_46_phase_9_design_register_studio_debugger_trace_visibility_and_filtering_contracts.md](sections/section_46_phase_9_design_register_studio_debugger_trace_visibility_and_filtering_contracts.md) |
| 47 | Studio Debugger run orchestration and persistence contracts | [docs/design_doc/sections/section_47_phase_9_design_register_studio_debugger_run_orchestration_and_persistence_contracts.md](sections/section_47_phase_9_design_register_studio_debugger_run_orchestration_and_persistence_contracts.md) |
| 48 | Studio Debugger training trace contracts | [docs/design_doc/sections/section_48_phase_9_design_register_studio_debugger_training_trace_contracts.md](sections/section_48_phase_9_design_register_studio_debugger_training_trace_contracts.md) |
| 49 | Plugin extensibility and compatibility contracts | [docs/design_doc/sections/section_49_phase_10_design_register_plugin_extensibility_and_compatibility_contracts.md](sections/section_49_phase_10_design_register_plugin_extensibility_and_compatibility_contracts.md) |
| 50 | Data I/O, dataset loading, and training backend contracts | [docs/design_doc/sections/section_50_phase_10_design_register_data_io_dataset_contracts_and_training_backends.md](sections/section_50_phase_10_design_register_data_io_dataset_contracts_and_training_backends.md) |
| 51 | Runtime fail-closed behavior and training-capability gap register | [docs/design_doc/sections/section_51_phase_10_design_register_runtime_fail-closed_training_gap_contracts.md](sections/section_51_phase_10_design_register_runtime_fail-closed_training_gap_contracts.md) |
| 52 | Training lifecycle and callback-order contracts | [docs/design_doc/sections/section_52_phase_10_design_register_training_lifecycle_and_callback_order_contracts.md](sections/section_52_phase_10_design_register_training_lifecycle_and_callback_order_contracts.md) |
| 53 | Python scripting and model-build boundary contracts | [docs/design_doc/sections/section_53_phase_10_design_register_python_scripting_and_model_build_boundaries.md](sections/section_53_phase_10_design_register_python_scripting_and_model_build_boundaries.md) |
| 54 | Missing design closure notes index | [docs/design_doc/sections/section_54_missing_design_closure_notes_index.md](sections/section_54_missing_design_closure_notes_index.md) |

## Notes
- Each top-level `## N)` section is now stored in its own file.
- Claim and backlog sections remain source-linked and now inherit traceability from their own files.







