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
with the engine's training-step contract. `StartTrainingFromGraph` now
resolves the runtime label column against the materialized Arrow schema
before loader dispatch, so a tokenized text table uses `y` explicitly
instead of relying on `ArrowDatasetBatcher` fallback from the original
DataInput label name. Local Debug's text Smoke Run now also tries the
Cat-1 materialized Arrow path first and reports whether it used
`materialized Arrow text` or the legacy text fallback.
`test_text_arrow_training_launch` covers the worker-thread runtime
contract for materialized text: `DataInput -> TextTokenizer` produces
Arrow `tok_*` plus `y`, `ArrowDatasetBatcher` consumes it on a worker
thread, and the model runs forward/loss/backward/update successfully.
Raw text Arrow backing now covers JSON/JSONL, TXT, and folder corpora
through a small `TextDataset -> Arrow` adapter. CSV/TSV keep Arrow's
native CSV reader so original file columns are preserved; adapter-backed
sources expose a stable `text_column` UTF-8 column and, when the
`TextDataset` reports classes, an int32 `label_column`. This is covered
by `test_text_arrow_adapter`, including a folder-corpus table feeding
`TextTokenizerOperator`. Legacy text training compatibility now routes
`TextDatasetBatcher` through raw Arrow materialization,
`TextTokenizerOperator`, and `ArrowDatasetBatcher`; `GetNextBatch()` no
longer tokenizes samples itself. `test_text_dataset_batcher_arrow`
covers train, validation, and test batches from that delegated path.
`TextDataset` now supports an explicit raw-load mode so the compatibility
wrapper can parse JSON/TXT/folder corpora without building a duplicate
vocabulary before `TextTokenizerOperator` tokenizes.
`TextVocabulary` and `TextPadding` stay folded into tokenizer v1:
`PipelineMaterializer` folds their supported params into the reachable
`TextTokenizer` operator instead of registering separate Arrow
operators. `TextPadding.pad_value` remains fixed to the tokenizer PAD id
0 in v1.
`GraphCompiler` no longer extracts TextTokenizer/TextVocabulary/
TextPadding into `TrainingConfiguration::text_preprocessing`; it only
recognizes them as preprocessing nodes so they do not compile as model
layers. Text node validation now reads node params directly in preflight.
Saved text smoke graph templates `test_01`, `v1`, `v2`, and
`test_02_lstm` were updated to the current contract: clean ASCII JSON,
explicit Data/Labels pin indices, canonical `TextTokenizer` Arrow params
(`text_col`, `label_col`, `min_word_freq`), and no stale extractor
wording. `test_text_smoke_graph_patterns` now validates those templates
load as known node types, keep required pins connected, route label flow
to loss targets, route model predictions to loss predictions, route loss
to optimizer, and keep tokenizer params complete.
`StartTrainingFromGraph` now delegates the runtime launch tail to
`graph_training_launcher`, so the GUI button path and focused tests use
the same dataset lookup, Cat-1 materialization, runtime label resolution,
and dispatch handoff. `test_text_gui_training_launch` registers a raw
text Arrow table, runs the shared launch helper over
`DataInput -> TextTokenizer`, verifies dispatch receives the
materialized Arrow dataset and `y` label, then runs one real
batch/model/loss/backward/update step. This closes the untested
GUI/runtime launch branch without adding brittle desktop click
automation.

**Status:** complete for Text Fix B. Optional manual desktop smoke can
still be run before a release build, but no known Text Fix B runtime
branch remains untested.

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

**Follow-up from text Arrow smoke:** core `DataRegistry` ownership
methods (`Instance`, `GenerateUniqueName`, `UnloadDataset`,
`UnloadAll`) were split into `data_registry_core.cpp`, so Arrow-only
runtime tests can link the singleton and Arrow registry utilities
without pulling `data_registry.cpp` and its legacy image/Kaggle/STB/
OpenCV loader dependencies. `test_text_gui_training_launch` and
`test_arrow_integration` now build against that smaller boundary.
`TrainingExecutor::Train` has also started moving toward smaller runtime
boundaries: Arrow and Parquet batcher construction now lives in
`training_batcher_setup.cpp`, preserving time-series partition handling,
normalization, one-hot/regression mode, and split warnings outside the
epoch loop. `test_training_batcher_setup` covers the Arrow helper path
and proves the returned owners, `IBatcher` pointers, split count, feature
shape, and one-hot label shape. Runtime launch now also treats the
compiled `TrainingConfiguration::dataset_name` as the dispatch source
instead of rescanning node order, and `PipelineMaterializer` starts from
the DataInput/DatasetInput that owns that source dataset. This prevents a
stale or unrelated first DataInput from stealing materialization and
label resolution. `test_text_gui_training_launch` covers that regression
by putting an unused loaded DataInput first in the node list.
`GraphCompiler` has also started using graph reachability for data-path
configuration: source selection now prefers a DataInput/DatasetInput that
can reach a loss node, then falls back to connected/incomplete sources,
and DataSplit/DataLoader extraction only reads nodes reachable from that
selected source instead of the first matching node in raw node order.
Compiler preprocessing/model extraction now uses the selected
source-to-loss path instead of the full topological graph, so stale
branches do not silently add preprocessing flags or model layers.
Optimizer extraction now prefers the supported optimizer reachable from
the selected loss, and CrossEntropy output class-count extraction now
prefers an Output node on the selected source-to-loss path.
The compiled config now carries selected data source, loss, and optimizer
node ids; runtime launch uses those ids for label-column lookup and
legacy optimizer `epochs`/`batch_size` fallback instead of rescanning
raw node order. The legacy `TrainingExecutor` branch also uses the
compiled dataset name for preprocessing and augmentation registry lookup,
falling back to `DatasetHandle::GetName()` only for older callers.
`test_graph_topology_utils` covers the helper behavior with a stale first
DataInput, stale first DataLoader, reverse loss reachability, and a
connected branch that does not reach the loss. The remaining
`TrainingExecutor` work is the legacy/external image/audio/text setup and
the epoch loop topology; those should be split only after the graph
contract is pinned down so the next refactor does not mix behavior
changes with file organization.

**Still open for Priority 2:** runtime label/data flow and the
TrainingExecutor external setup and epoch loop still need pin-walked
ownership inside the executor. Whole-graph validation still audits all
nodes, so stale disconnected training branches may remain a separate UX
decision from compile extraction.

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

## Build Health Follow-Up

### Full engine Debug build duplicate DataSources case

**Status 2026-06-03:** fixed. The affected category display helpers in
`node_editor_context_menu.cpp` and `node_browser_panel.cpp` no longer use
`switch` statements, so the build is not blocked by duplicate enum case
values. Full `cyxwiz-engine` Debug rebuild now passes.

## Closed Source

The full historical context was in `tofix.md` before the 2026-06-03
backlog migration. Use git history for completed details instead of
copying old resolved sections forward.
