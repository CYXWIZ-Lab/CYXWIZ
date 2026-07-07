## 31) Pending claim completion backlog (next proceed-ready cycle)

### 31.1 Open items after current pass
- `C-01` through `C-09` now have exact `path:line` anchors in the claim table.
- `C-10-state-contract` now has concrete `path:line` anchors in the claim table.
- Rows in Section 19 and Section 22 critical families have explicit `S/B/L` normalization recorded in section 19.2 and 22.4.
- `Phase 1` and `Phase 5` are marked COMPLETE in 0b and Section 26.
- `Phase 6` and `Phase 7` remain `INPROGRESS` and are excluded from current close.
- Node surface/runtime mismatch rows should add explicit migration and deprecation notes for UI-only nodes.
- Python boundary/ownership gap note added in `sections/missing/missing_01_python_scripting_does_not_own_graph_training_model_build.md` for closure traceability and lean-gap tracking.

### 31.2 Close list for this backlog
```text
Task-ID   | Scope        | Owner    | Deliverable                                | Status    | Deadline
----------+--------------+----------+--------------------------------------------+-----------+----------
D-01      | 30.1/30.2   | Engine   | exact symbol pins for C-01..C-09            | DONE      | Next review
D-02      | 29          | Engine   | replace shorthand section labels in 29.1/29.2  | DONE      | Next review
D-03      | 19/22/26    | PM/Arch  | clear all open U statuses in critical rows      | DONE      | Next release cut
D-04      | 27/28       | QA       | add one concrete pass/fail example per gate     | DONE      | Next release cut
D-05      | 24/28       | Ops      | document recovery actions for failure codes     | DONE      | Next review
D-06      | 30/31/33    | Runtime  | finalize C-10 state-transition claim anchors   | DONE      | Next review
D-07      | 17 / 53     | Engine   | document explicit Python execution boundary vs C++ TrainingExecutor in missing design note | DONE      | Next review
D-08      | 17 / 52     | Engine   | add callback ordering determinism design note for pause/stop/validation | DONE      | Next review
D-09      | 17 / 53     | Engine   | add RL execution boundary note for Policy/Value/ReplayBuffer | DONE      | Next review
D-10      | 17 / 53     | Engine   | add advanced audio E2E execution gap note (Spectrogram/MFCC) | DONE      | Next review
D-11      | 17 / 53     | Engine   | add plugin restart/reload hook lifecycle design note | DONE      | Next review
D-12      | 17 / 53     | Engine   | add materializer Unknown source-kind guard design note | DONE      | Next review
D-13      | 17 / 53     | Engine   | add legacy alias normalization idempotence design note | DONE      | Next review
D-14      | 17 / 53     | Engine   | add GraphExecutableModel consumption path design note | DONE      | Next review
``` 

### 31.2b Closure mode for this cycle

- This cycle currently closes documentation gaps and defines execution-ready evidence packages.
- No runtime or integration evidence is attached yet; all [31.4](#314-execution-closure-verification-backlog) items remain at `READY` until trace artifacts are collected and signed.

### 31.3 Definition of done
- claim text maps to a stable file symbol and owner,
- each evidence symbol is a concrete `path:line` tuple,
- each backlog row above moves to DONE with completion timestamp,
- section 0b snapshot and Section 26 status are updated in the same pass,
- claim IDs in `33` use only concrete `path:line` symbols before any release-ready sign-off.

### 31.4 Execution-closure verification backlog

| Task-ID | Scope | Owner | Deliverable | Status |
|---------|-------|-------|------------|--------|
| V-01 | 17.2 / 17.5 | Engine | Deterministic callback-order trace for pause/stop/validation nested transitions | READY |
| V-02 | 17.2 / 17.4 | Runtime | End-to-end RL launch proof for `PolicyNetwork`, `ValueNetwork`, `ReplayBufferNode` | READY |
| V-03 | 17.2 / 17.4 | Runtime | End-to-end audio feature execution proof (`Spectrogram`, `MelSpectrogram`, `MFCC`) | READY |
| V-04 | 17.2 / 17.4 | Plugin | Hook lifecycle proof for restart/reload (register/unregister/reload order) | READY |
| V-05 | 17.2 / 17.4 | Materializer | Explicit contract test for `materializer_source_kind == Unknown` allowed only for approved legacy branches | READY |
| V-06 | 17.2 / 17.4 | Compiler | Alias normalization idempotence test against alias collision inputs | READY |
| V-07 | 17.2 / 17.4 | Compiler/Runtime | Contract-evidence test for `GraphExecutableModel` consumption branch selection | READY |

### 31.5 Required closure evidence per item

| Task-ID | Acceptance target | Evidence format |
|---------|------------------|----------------|
| V-01 | A deterministic callback-order trace must show the exact callback transition sequence for nested pause/stop/validation gate changes. | Single run log with timestamped callback events and deterministic expected-order assertions. |
| V-02 | RL launch must execute graph-derived RL request through the Python bridge using current node config, and emit the expected train-loop metrics callbacks. | One reproducible graph + RL config + success/failure log sequence proving script generation + execution path. |
| V-03 | Audio preprocessing feature nodes must complete an end-to-end training batch path with corresponding feature transform outputs in the active batcher. | One run artifact containing generated batch schema and sample transform outputs for each feature node. |
| V-04 | Restart/reload transitions must not duplicate hook invocations and must re-register in deterministic order. | One lifecycle matrix test (start → hook register → reload → unload → reload) with before/after callback list. |
| V-05 | `materializer_source_kind == Unknown` must occur only in explicitly approved legacy branches and be rejected otherwise. | A matrix test with approved/disallowed node/domain inputs and resulting materializer reason strings. |
| V-06 | Legacy alias normalization must be idempotent across alias-collision cases and only emit one normalized form. | Property-style test output verifying output contract for repeated alias inputs and deterministic canonical node mapping. |
| V-07 | `GraphExecutableModel` branch selection must be proven for graph-capable and non-graph-capable paths. | One execution trace proving graph path and sequential fallback path with contract fields and model artifact classification. |

### 31.6 Evidence status board

| Task-ID | Status | Owner | Planned window | Planned evidence artifact |
|---------|--------|-------|---------------|------------------------|
| V-01 | READY | Engine | S1-01 | `evidence/V01_pause_stop_callback_trace.md` |
| V-02 | READY | Runtime | S1-02 | `evidence/V02_rl_launch_execution_trace.md` |
| V-03 | READY | Runtime | S1-03 | `evidence/V03_audio_spectrogram_trace.md` |
| V-04 | READY | Plugin | S1-04 | `evidence/V04_hook_restart_reload_trace.md` |
| V-05 | READY | Materializer | S1-05 | `evidence/V05_materializer_unknown_guard_matrix.md` |
| V-06 | READY | Compiler | S1-06 | `evidence/V06_alias_idempotence_matrix.md` |
| V-07 | READY | Compiler/Runtime | S1-07 | `evidence/V07_graph_executable_branch_matrix.md` |

### 31.7 Verification runbooks (implementation-ready)

| Task-ID | Test fixture | Execution steps | Success criteria |
|---------|--------------|-----------------|-----------------|
| V-01 | A graph with alternating pause and stop conditions around validation epochs (single training run) | 1) Start training from graph. 2) Trigger pause at batch boundary. 3) Trigger resume, then validation. 4) Trigger stop during validation. 5) Capture ordered callback log from all plugin hook points. | Callback order log is deterministic, exactly matches expected sequence, and no callback is emitted after terminal stop except completion/cleanup hooks. |
| V-02 | Graph containing `PolicyNetwork`, `ValueNetwork`, `ReplayBufferNode` with minimal RL dataset and reward/filter nodes | 1) Launch RL training path through UI `Train RL`. 2) Capture generated script and metrics callback stream. 3) Verify policy save path and stop behavior. | Script generation path is invoked once, `ScriptingEngine` executes with RL bridge events, and RL model artifact is produced by Python runtime. |
| V-03 | Graph with `Spectrogram`/`MelSpectrogram`/`MFCC` dataset preprocessing chain and audio-compatible dataset source | 1) Run normal (non-RL) training launch. 2) Capture batch schema and runtime batcher transform outputs. 3) Verify capability map indicates unsupported status and runtime path behavior. | For each feature, emitted batch schema includes expected transform metadata and no silent fallback changes feature semantics. |
| V-04 | Two mock training plugins with deterministic registration IDs loaded in sequence plus restart/reload toggles | 1) Start run and register hooks. 2) Trigger plugin reload path. 3) Trigger unload and reload. 4) Capture hook invocation list before and after each transition. | Hook registration order and identity are deterministic, no duplicate callbacks after reload, and plugin removal is complete at unload/reload boundaries. |
| V-05 | Node set where `materializer_source_kind` can be `Known`, `Unknown-legacy`, and invalid | 1) Compile each fixture. 2) Capture materializer source-kind and reason. 3) Compare against expected list of approved legacy branches. | `Unknown` appears only for approved legacy branches; all other invalid inputs produce explicit failure/skip reasons and non-legacy path. |
| V-06 | Legacy alias collision corpus (`legacy`, `canonical`, mixed-case, duplicate aliases) with repeated compile passes | 1) Build graph twice with the same alias input set. 2) Serialize `compiled_model.config.alias_resolution` and parameter set for each pass. 3) Compare outputs for monotonicity and duplicates. | Canonical node mapping is stable between passes, no duplicate normalized nodes are introduced, and parameter normalization is identical each pass. |
| V-07 | Two graphs: one graph-executable-capable and one forcing legacy-only runtime path | 1) Launch TrainingExecutor for both graphs. 2) Capture `BuildExecutableFromConfig` branch decision. 3) Capture produced model artifact metadata. | Capable graph selects `BuildGraphExecutableFromConfig` and emits graph model artifact classification; non-capable graph follows legacy path with explicit fallback evidence. |

### 31.8 Suggested execution sequence

- Recommended order: `V-01 -> V-02 -> V-03 -> V-04 -> V-05 -> V-06 -> V-07`.
- Suggested execution mode:
  - `V-01` and `V-04` (callback/lifecycle determinism) first, because they constrain expected event ordering for later runs.
  - `V-02`, `V-03`, and `V-07` in parallel once runtime logging is stable.
  - `V-05` and `V-06` after runtime path validation to confirm materializer and alias contract baselines.
- Completion rule for this cycle:
  - All rows in [31.4](#314-execution-closure-verification-backlog) must move from `READY` to `DONE`.
  - [section_54_missing_design_closure_notes_index.md](section_54_missing_design_closure_notes_index.md) should drop the corresponding note only after implementation/signature evidence is attached.

### 31.9 Execution-ready package list

| Item | Artifact type | Required fields |
|------|---------------|-----------------|
| `evidence/V01_pause_stop_callback_trace.md` | log + ordering table | fixture id, timestamp, callback event sequence, expected sequence match assertion |
| `evidence/V02_rl_launch_execution_trace.md` | run transcript + generated script + RL metrics timeline | graph id, script SHA, bridge call log, policy save path, stop/pause behavior |
| `evidence/V03_audio_spectrogram_trace.md` | batch schema + transform artifact snapshot | feature node set, input audio source metadata, batch-level transform outputs |
| `evidence/V04_hook_restart_reload_trace.md` | lifecycle matrix log | plugin registration IDs, pre/post callback list, reload/unload ordering, duplicate-callback check |
| `evidence/V05_materializer_unknown_guard_matrix.md` | compile matrix table + failure reasons | fixture set, source-kind output, reason string, approval list comparison |
| `evidence/V06_alias_idempotence_matrix.md` | property-run report | repeated-input corpus, canonical mapping snapshot (run 1/2), duplicate/extra-mapping diff |
| `evidence/V07_graph_executable_branch_matrix.md` | execution trace + artifact metadata | graph-capable flag, branch decision, emitted model class, fallback evidence |

### 31.10 Closure state transition convention

- `READY` → `IN_PROGRESS` (execution started, evidence collected partially)
- `IN_PROGRESS` → `DONE` (all acceptance criteria met + evidence artifacts signed)
- `DONE` entries should be reflected by:
  - updating [section 17] to mark the unknown as closed,
  - moving/removing the corresponding missing-note entry after creating a permanent evidence-backed permanent section.




