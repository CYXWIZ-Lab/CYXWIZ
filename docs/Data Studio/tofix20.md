# To Fix 20 - Runtime Architecture Follow-Ups From To Fix 5

Source: carried forward from `done5.md` closeout.

This document owns the broad architecture work that remained after the
concrete `tofix5` correctness pass was completed. Do not use this file for
small DataInput/DataOutput alias bugs, stale picker filters, or individual
placeholder branches; those were closed in `done5.md`.

## Boundary

Keep this focused on structural runtime ownership and support presentation:

- canonical execution ownership between `PipelineExecutor`,
  `PipelineMaterializer`, and `PipelineOperatorFactory`
- materializer v2 graph/storage scope
- remaining string-only legacy alias migration
- frontend presentation of centralized support axes
- full pre-execution schema/type validation parity

## Current Truth Already Achieved

- Runtime support truth lives in `pipeline_runtime_capabilities.{h,cpp}`.
- Runtime support carries mode, fail mode, PipelineExecutor support,
  materializer scope, implementation owner, source role, arity, required
  parameters, enum values, and numeric validation axes.
- `PipelineExecutor` no longer relies on a raw `node.type` string-dispatch
  chain for active routing.
- Known unsupported legacy nodes fail closed with central reasons.
- Exact operator-backed runtime names route through `PipelineOperatorFactory`.
- Browser-visible metadata and Node Info/Browser support summaries consume
  centralized support axes.
- `PipelineMaterializer` uses central support truth for Arrow-table operator
  applicability and fails closed on unsupported v1 graph shapes.

## Priority 1 - Choose Canonical Runtime Ownership

Problem:

- `PipelineExecutor` still owns legacy graph execution and compatibility
  aliases.
- `PipelineOperatorFactory` owns real operator implementations for many
  transforms.
- `PipelineMaterializer` applies only a narrow Arrow-table operator slice.
- The support registry says who owns each node, but there is still no single
  graph execution plan/owner.

Next work:

- Define the canonical graph execution plan shape.
- Migrate one legacy-dispatched node family at a time to operator ownership.
- Keep compatibility aliases explicit until migrated or retired.
- Remove legacy executor branches only after parity tests prove the operator
  path is equivalent or intentionally different.

Definition of done:

- Each executable Data Studio node has one runtime owner.
- Compatibility aliases resolve to canonical nodes before execution.
- Drift tests fail if a node is executable through two competing owners without
  an explicit migration exception.

## Priority 2 - Materializer v2 Scope

Problem:

- `PipelineMaterializer` is intentionally v1: linear Arrow-table operator
  chains only.
- It does not rewrite Parquet row groups.
- It does not materialize image, audio, or text domain datasets.
- It fails closed on cycles and branched operator paths instead of planning a
  full graph.

Next work:

- Decide whether materializer v2 is a full graph planner or remains a
  preprocessing adapter.
- Add a typed topological plan for materializable graphs if general coverage is
  required.
- Add typed dataset adapters only where they preserve domain semantics.
- Keep unsupported storage/domain modes explicit in runtime support truth.

Definition of done:

- Materializer support is either explicitly Arrow-table-only by design, or v2
  has tests for branching, selected source identity, and supported storage
  backends.
- Parquet/image/audio/text behavior is documented as supported, adapted, or
  intentionally skipped with user-visible reasons.

## Priority 3 - Finish String-Only Alias Migration

Problem:

- Some compatibility names still exist without first-class `gui::NodeType`
  ownership.
- Examples include legacy names such as `SaveDataset`,
  `DeployToNodeEditor`, text aliases, old time-series aliases,
  `PolynomialFeatures`, and `Binning`.

Next work:

- For each alias, choose one of:
  - migrate to a first-class typed metadata node
  - map to an existing canonical typed node
  - keep as a documented compatibility alias with no browser-visible surface
  - remove if no supported graph depends on it

Definition of done:

- No executable runtime alias is string-only unless it has an explicit
  compatibility justification and dispatch kind.
- Drift tests list every remaining exception.

Status 2026-06-08:

- Started. Remaining string-only legacy aliases now have explicit compatibility
  reasons in `PipelineLegacyRuntimeCapability`.
- Metadata drift tests pin the current exception list:
  `SaveDataset`, `DeployToNodeEditor`, `TextClean`, `TextTokenize`,
  `TextVectorize`, `TSWindow`, `TSFeatures`, `TSLag`, `TSDiff`,
  `PolynomialFeatures`, and `Binning`.
- Adding a new executable string-only alias now requires both a dispatch kind
  and a compatibility reason, and the drift test must name it explicitly.

## Priority 4 - Broader Frontend Support-Axis Presentation

Problem:

- Node Info and Node Browser hover already expose structured support axes.
- The broader UI still needs richer filtering and clearer availability
  presentation for users building graphs.

Next work:

- Add support-aware filtering in node browser/search.
- Distinguish real, blocked, UI-only, training-only, and pipeline-only nodes
  without parsing help text.
- Keep frontend display backed by `support_axes`, not a second support list.

Definition of done:

- Users can filter/hide blocked or unsupported runtime nodes.
- Support badges come from central support axes.
- There is no parallel frontend support matrix.

## Priority 5 - Complete Schema/Type Validation Parity

Problem:

- Many high-risk required parameters, enum values, numeric bounds, and column
  existence/type checks now fail before execution.
- Some less-used node families still rely on late execution-time errors.

Next work:

- Audit remaining active executor/operator families for missing pre-execution
  schema/type checks.
- Add table-driven validation where parameters are static.
- Keep node-specific validation local only when it truly depends on loaded
  table schema or operator semantics.

Definition of done:

- Unsupported graph shapes and unsupported column types fail before query or
  operator execution for every active Data Studio runtime node.
- Regression tests cover at least one bad-schema case per executable family.
