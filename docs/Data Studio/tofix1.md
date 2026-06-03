# To Fix 1 - Active Backlog

Migrated from the oversized `tofix.md` on 2026-06-03. Keep this file
small and current: active work only, grouped by priority. Completed
items should get a one-line status update here, then be removed or moved
to the archive when the next `tofixN.md` file is opened.

## Migration Rule

- `tofix.md` is closed. Do not append new work there.
- Keep active backlog files below roughly 300-500 lines.
- When this file gets too large, open `tofix2.md` for new pending work
  and mark this file closed with a short summary.
- Preserve details in commits and implementation docs; backlog entries
  should say what remains, why it matters, and the next verification.

## Priority 1 - Text Fix B

### Text preprocessing nodes must become real dataflow operations

**Status 2026-06-03:** In progress. `TextTokenizerOperator` exists and
is registered as a Cat-1 operator. The frontend dialog now writes
canonical params and is exposed for `TextTokenizer`, `TextVocabulary`,
and `TextPadding`. Single-file text CSV/TSV now dual-registers raw
Arrow plus legacy `TextDatasetEntry`, and
`test_text_tokenizer_operator` validates the Arrow operator contract.
The Arrow and Parquet batchers now fall back to common label-name
auto-detection when a stale explicit label column is missing, so a
materialized tokenizer table with `tok_*` and `y` does not silently use
`y` as a feature. The shared label-name resolver is centralized and
covered by `test_label_column_resolver`. `test_text_arrow_materializer`
now proves a graph-shaped Arrow table flow from `DataInput` through
`TextTokenizer` into a tokenized table, then exports and reloads that
tokenized table as Parquet. It also verifies that the materialized
`tok_*` plus `y` table feeds `ArrowDatasetBatcher` as training-ready
feature tensors with one-hot labels. `ArrowDatasetBatcher` was split
into its own translation unit so this boundary can be tested without
linking unrelated legacy preprocessing code. Text CSV/TSV with a
`TextTokenizer` graph now has a clear canonical training route:
DataInput registers raw Arrow plus legacy metadata, the Cat-1
materializer writes `<dataset>__materialized` as Arrow-only output, and
loader dispatch routes that materialized table through Arrow training.
If no materialized Cat-1 text table is selected, `TextLoader` logs that
it is intentionally using the legacy `TextDatasetBatcher` fallback.
Raw Arrow text backing for JSON, TXT, and folder corpora remains
deferred because those sources need an explicit synthetic table schema
for text and labels. The materializer test now runs the materialized
Arrow batch through a real model forward, CrossEntropy loss, backward
pass, and optimizer update, proving the tokenized table is compatible
with the engine's training-step contract.

**What remains:**
- Make Arrow the default text training path when a materialized Cat-1
  text table exists; keep legacy fallback only for unmaterialized text.
- Implement raw table adapters for JSON, TXT, and folder corpora after
  defining their synthetic text/label schema.
- Rewrite or replace `TextDatasetBatcher` so text training consumes
  pre-tokenized Arrow data instead of tokenizing inside the batcher.
- Remove `ExtractTextTokenizer`, `ExtractTextVocabulary`, and
  `ExtractTextPadding` from `graph_compiler.cpp` after no consumers
  need `TrainingConfiguration::text_preprocessing`.
- Decide whether `TextVocabulary` and `TextPadding` stay separate real
  operators or remain folded into the v1 combined tokenizer operator.
- Run a GUI/runtime smoke graph for one minimal text training launch
  once legacy text path removal is planned. The headless model-step
  smoke is covered; this remaining item is the threaded GUI launch path.
- Update and rerun existing text smoke graphs:
  `test_01`, `v1`, `v2`, and `test_02_lstm`.

**Sweet spot for next slice:** do not delete the legacy path yet. First
prove one complete Arrow graph from DataInput text CSV through
TextTokenizer into training or export, then remove extractors in a
separate cleanup commit.

## Priority 2 - Graph Honesty And Runtime Contracts

### TrainingExecutor should walk pin connections

The compile gate now enforces much of the graph contract, but runtime
training still leans on registry lookups and graph-wide assumptions.
Runtime execution should follow pin connections so data, labels, split
outputs, and preprocessing order are derived from the graph shape, not
from global dataset conventions.

**Next step:** audit `MainWindow::StartTrainingFromGraph`,
`GraphCompiler`, `PipelineMaterializer`, and `TrainingExecutor` for
places where graph topology is ignored after compile.

### Normalize in DataInput vs graph

Normalization remains split between DataInput options and graph nodes.
The long-term direction is one honest preprocessing path where graph
nodes own transformations and DataInput owns loading/profiling.

**Next step:** decide which DataInput normalization controls become
deprecated aliases and which graph node owns the canonical behavior.

### DataLoader pin and label flow follow-up

Pin pass and compile-gate enforcement landed, but label/data flow should
continue moving toward explicit graph contracts rather than hidden
DataInput assumptions.

**Next step:** after Text Fix B, audit DataLoader/DataSplit/Loss label
paths against actual pin connections.

## Priority 3 - Data Quality And Debuggability

### Apply-time dataset audit follow-ups

Compact Apply-time audit is done. Deferred items remain:
- Parquet data-page/sample audit for null/constant/NaN/label checks,
  but only after a bounded worker/timeout design exists.
- Text vocabulary coverage drill-down belongs in the richer text-node
  dialog/debugger flow, not the compact Apply audit.
- More detailed audit issue drill-down UI remains pending.

### Local Debug and synthetic data expansion

Local Debug infrastructure landed, but concrete synthetic batch
generation remains thin across loaders.

**Next step:** implement realistic `MakeSynthetic` per loader only where
it directly supports compile/debug workflows.

### Training logs and sub-epoch validation

Training can stay silent for long stretches within an epoch, and
fractional validation cadence is still pending.

**Next step:** add lightweight per-batch or rate-limited training
progress updates before broad validation-cadence changes.

## Priority 4 - Studio And Workflow UX

### Query Console in CyxWiz Studio

Open Node Editor landed. Query Console remains a larger bridge:
grammar/parser, engine-to-NodeEditor bridge, mutation safety, and undo
integration.

**Next step:** start read-only query inspection before allowing graph
mutation commands.

### Properties panel generic parameter editor

The generic parameter editor is still basic. Complex nodes should keep
dedicated dialogs; the generic editor should become safer for simple
typed params.

**Next step:** add typed validation and avoid editing hidden/internal
params that should only be owned by dialogs.

### Memory estimation is approximate

Memory tab estimates are useful but still approximate across lazy and
streaming datasets.

**Next step:** keep estimates honest in UI and improve per-loader
metadata where it is cheap to compute.

## Priority 5 - Model Coverage And Performance

### GRU verification

GRU CPU forward/BPTT landed, but smoke verification and ArrayFire path
cleanup remain.

**Next step:** run focused GRU smoke tests, then remove stale one-shot
warnings only after verification.

### Transformer verification

Transformer text path is still unverified relative to LSTM.

**Next step:** create or update one small text graph that exercises
Transformer training after Text Fix B stabilizes.

### LSTM ArrayFire performance follow-up

LSTM correctness is resolved. AF performance optimizations remain
deferred.

**Next step:** profile before changing. Avoid speculative rewrites.

### `num_workers` support is partial

Some loaders/batchers honor worker settings better than others.

**Next step:** document which data paths are actually parallel and make
unsupported paths explicit rather than pretending all loaders scale.

## Priority 6 - Tool-To-Node Migration

### Remaining standalone panels / dead NodeTypes

The TimeSeries block and multiple Cat-1 operators landed, but remaining
panel-only tools still need either:
- a real graph operator,
- a Cat-2 introspection panel contract, or
- explicit removal/defer status.

**Next step:** continue in small groups. Do not migrate a node unless
its data contract is clear and testable.

### Known dual-maintained registration lists

Node registration unification mostly landed, but some lists remain
dual-maintained.

**Next step:** only consolidate lists that are actively causing drift;
avoid broad registry rewrites during Text Fix B.

## Priority 7 - Future Architecture

### Variable-shape Sample type

Deferred v3 consideration. Needed only when fixed-shape tensor batches
are no longer enough.

### Explicit Decode node

Deferred v3 consideration. Useful for generated/tokenized outputs, but
not required for the current Text Fix B path.

## Closed Source

The full historical context was in `tofix.md` before the 2026-06-03
backlog migration. Use git history for completed details instead of
copying old resolved sections forward.
