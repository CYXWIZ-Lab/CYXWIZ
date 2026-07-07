## 20) Phase 1 completion criteria

Phase 1 is complete for release-critical scope when all rows in the node-contract register are normalized and no unresolved open items remain.

| Artifact | Required state for completion |
|----------|-----------------------------|
| Node contract matrix (`section_19`) | every row is exactly `S`, `B`, or `L` |
| Node coverage | includes any new `NodeType` introduced since last version |
| Source anchors | each row has at least one source file path or explicit reason for absence |
| Launch impact | each row has one of `allowed`, `warn`, `blocked` |
| Unknowns | `section_17` has no unresolved high-risk items |

## 20.1 Phase-1 gate checklist

- `GraphCompiler` contract extraction paths are present (`graph_compiler.h/.cpp`).
- Node execution support is traceable via `pipeline_runtime_capabilities.h/.cpp`.
- At least one section has compile-path evidence tied to source identifiers.
- Front-to-back launch flow has a defined path in:
  - `section_04` (compile),
  - `section_08` (materialization),
  - `section_09` (execution),
  - `section_11` (error model).
- A short walkthrough (`section_23`) references the phase-1 launch path and blocker handling.

## 20.2 Completion outcomes

- **DONE** when every item above is satisfied and no critical unknown remains.
- **CONDITIONAL** when all non-blocking nodes are covered but failclosed entries exist.
- **BLOCKED** when any unresolved required behavior has no source contract or deterministic reason.

## 20.3 Suggested acceptance statement

> “Phase 1 is complete when compile contract extraction, runtime capability mapping, and launch path behavior are all represented as explicit, source-backed node contracts with no unresolved critical unknowns.”
