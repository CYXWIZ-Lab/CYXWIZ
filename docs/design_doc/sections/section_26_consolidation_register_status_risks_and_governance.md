## 26) Consolidation register (status, risks, and governance)

### 26.1 Cross-document status table

```text
Phase | Status    | Coverage completeness | Risk level | Next review
------+-----------+----------------------|------------+-------------------
0    | COMPLETE  | High                 | Low        | On demand
1    | COMPLETE  | High                 | Low        | Next commit
2    | COMPLETE  | High                 | Medium     | Verify against executor modes
3    | COMPLETE  | Medium               | Medium     | Add backend file evidence
4    | COMPLETE  | Medium               | Medium     | Add concrete example outputs
5    | COMPLETE  | High                 | Low        | Add gate example linkage
6    | COMPLETE  | Medium               | Medium     | Add thread-state ownership map
7    | PENDING   | Low                  | Low        | Create this phase
```

### 26.2 Governance rules for this document
1. Every new section must link to at least one concrete implementation file.
2. Any `U` (unknown) state in matrices must become `S/B/L` before production acceptance.
3. Every `B` blocker must include user-facing recovery guidance.
4. New node/runtime features must append both:
   - guard path (compiler/preflight/materializer/training),
   - fallback behavior.
5. If behavior changes, create a migration note in the relevant phase section.

### 26.3 High-priority risks (current)
- node surface > runtime support gap can increase false user expectations.
- alias compatibility currently smooths migration but may hide deprecations if untracked.
- some domain branches (audio/legacy visualization/advanced analytics) are cataloged before stable runtime semantics.
- matrix `U` rows can mask critical blockers when users test large graphs quickly.

### 26.4 Design debt closure plan
- close critical `U` entries in Phase 1 and Phase 3 matrices with deterministic status.
- document exact alias behavior and deprecation date per entry.
- create smoke-train paths for:
  - image->compile->arrow->train,
  - text->compile->materializer->train,
  - sequence->compile->sequence external->train,
  - at least one blocked legacy case.

### 26.5 Phase 7 proposal (next step)
Phase 7 should lock documentation-to-runtime parity:
- add a concrete, per-file trace matrix:
- `graph_compiler.cpp` -> compile gates,
- `training_executor.cpp` -> mode-specific loops,
- `pipeline_materializer.*` -> source/operator limits,
- `main_window.cpp` -> event handoff.
