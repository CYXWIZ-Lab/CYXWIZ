## 28) Phase 8 design register (release readiness + hard gates)

### 28.1 Objective
convert design coverage into launch-ready engineering gates:
- no unexpected behavior at launch.
- deterministic reasons for all blocked runs.
- reproducible operational behavior for troubleshooting.

### 28.2 Readiness architecture matrix

```text
Gate                  | Required evidence                  | Blocking threshold
---------------------+-----------------------------------+-------------------------------
Compile gate          | C_* issues and is_valid false     | any severity=error
Preflight gate        | preflight summary and runtime path | any severity=error
Materialization gate   | materialization result             | blocked for active source/op pair
Executor gate         | training mode init success          | failure to construct executable/batcher
Runtime gate          | first N epochs + callback output    | unhandled exception or divergence
Observability gate    | event stream + run summary          | missing run metadata / IDs
```

### 28.3 Hard stop criteria (must block launch)
- unresolved compile error (`C_*`)
- unresolved preflight error (`P_*`)
- materializer unsupported required path (`M_*`)
- executor init failure (`E_*`)
- missing determinism metadata for reproducibility-sensitive launch

### 28.4 Soft warning criteria (must surface to user)
- large graph-to-runtime mismatch in node support (alias fallback path),
- potential OOM risk (`R_OOM_RISK`),
- mixed backend placement warnings,
- non-deterministic mode due to missing seeds.

### 28.5 Release checklist
- [ ] `Phase 7` evidence matrix complete and owner-assigned.
- [ ] `Phase 1` and `Phase 3` matrices contain no unresolved `U` in critical families.
- [ ] Every blocker in matrix has recovery note and source-path.
- [ ] At least one smoke test path per execution mode: arrow, parquet, external, sequence.
- [ ] Logs include component + severity + run_id + source layer.
- [ ] Checkpoint policy and restore mode are documented and operational.
- [ ] Failure taxonomy is exposed in compile/preflight/runner UI flow.

### 28.6 Per-gate pass/fail examples

```text
Gate             Pass example                                          Fail example
---------------  ---------------------------------------------------  ------------------------------------------
Compile gate     Dense + CE + Adam with one data+labels path compiles | C_MISSING_SOURCE / C_PIN_MISMATCH
Preflight gate   Text chain with valid labels resolves preflight pass    | P_LABEL_MISSING, CYCLE/shape mismatch
Materialization  Arrow+Resize+Normalize applies and materializes        | M_OP_MAPPING_FAIL / M_UNSUPPORTED_NODE
Executor gate    Arrow mode constructor succeeds and training loop starts  | E_BATCHER_FAIL / E_BACKEND_FAIL
Runtime gate     First N epochs stream callbacks + checkpoint save       | E_EPOCH_FAIL / R_IO_FAILURE
Observability    run_id present in session summary and callback stream    | Observability gate: missing run/component metadata
```

- Bind each row to one walkthrough in Section 23 where possible.

### 28.7 Sign-off model
- Engineering lead: confirms phase evidence and hard gate definitions.
- Runtime lead: confirms executable/mode parity and placement behavior.
- UX/tooling lead: confirms launch blockers and recovery guidance are user-visible.
- Release lead: confirms checklist complete and signed.

### 28.8 Final completion criteria
- all phases 0-8 have explicit status and concrete owner.
- document can be used as single source for architecture onboarding.
- all code-path claims in phases 0-7 have file-level evidence.
- remaining `B`/`L` items have scheduled migration or deprecation action.
