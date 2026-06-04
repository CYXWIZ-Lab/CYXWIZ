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
External executor mode was audited and renamed internally from image-only
wording to generic external `IBatcher` ownership; image/audio/text
batchers are constructed by `TrainingManager` from the compiled graph
config before entering the shared epoch loop.
`test_graph_topology_utils` covers the helper behavior with a stale first
DataInput, stale first DataLoader, reverse loss reachability, and a
connected branch that does not reach the loss.

**Decision 2026-06-03:** the current runtime boundary is compile-time
pin validation plus a compiled dataset/batcher contract, not a full
per-batch tensor router. `GraphCompiler` owns graph honesty for data,
labels, predictions, loss, and optimizer reachability. Runtime launch
uses the selected source/loss/optimizer ids and compiled dataset name,
then `TrainingExecutor` consumes `IBatcher::Batch{data, labels}` from
the selected batcher. A deeper tensor-router epoch loop is deferred until
there is a concrete multi-output runtime need.

**Deferred beyond Priority 2:** whole-graph validation still audits all
nodes, so stale disconnected training branches may remain a separate UX
decision from compile extraction.

### Normalize in DataInput vs graph

Normalization remains split between DataInput options and graph nodes.
The long-term direction is one honest preprocessing path where graph
nodes own transformations and DataInput owns loading/profiling.

**Decision 2026-06-03:** keep DataInput `normalize` as a loader/decode
option, for example uint8 image/HDF5/custom data scaled into `[0,1]`
during loading. It is not the canonical training preprocessing path.
Graph `Normalize` remains the canonical train-time preprocessing node
and is the path that sets `TrainingConfiguration::preprocessing` and
batcher normalization. Do not add a second graph transform behind the
DataInput checkbox; future UI wording can make the loader/decode scope
clearer without changing runtime behavior.

### DataLoader pin and label flow follow-up

Pin pass and compile-gate enforcement landed, but label/data flow should
continue moving toward explicit graph contracts rather than hidden
DataInput assumptions.

**Status 2026-06-03:** audited after Text Fix B. DataInput, DataLoader,
DataSplit, and Loss now use explicit label/data pin contracts at compile
time: required pins must be connected, loss Targets must trace upstream
to a Labels-typed pin, loss Predictions must trace to a model/output
path, and optimizer loss input must trace to a loss node. Runtime label
selection follows the selected data source id and compiled dataset name.
This completes the Priority 2 label-flow contract for the current
batcher-based executor.

## Priority 3 - Data Quality And Debuggability

### Apply-time dataset audit follow-ups

Compact Apply-time audit is done. Deferred items remain:
- Parquet data-page/sample audit for null/constant/NaN/label checks,
  but only after a bounded worker/timeout design exists.
- Text vocabulary coverage drill-down belongs in the richer text-node
  dialog/debugger flow, not the compact Apply audit.

**Status 2026-06-03:** the DataInput Audit tab now renders
current-session audit issues as a structured drill-down table with
severity, issue code, detail text, and bounded examples instead of a
flat string list.

### Local Debug and synthetic data expansion

Local Debug infrastructure landed, but concrete synthetic batch
generation remains thin across loaders.

**Status 2026-06-03:** loader-level `MakeSynthetic` now returns a real
single-sample synthetic batch for Tabular/TimeSeries, Image, Audio, and
Text by delegating to the shared core `MakeSyntheticBatch` helper with
the selected preprocessing domain. The returned loader batch includes
feature/label tensors, sample count, and a compact summary. This keeps
Local Debug shape generation in one core path while avoiding another
category switch in the dialog.

**Still deferred:** true image/audio/time-series shaped synthetic
tensors remain folded into the current core flat-feature fallback until
the debug executor has a concrete need to execute loader-native shapes.

### Training logs and sub-epoch validation

Training can stay silent for long stretches within an epoch, and
fractional validation cadence is still pending.

**Status 2026-06-03:** lightweight per-batch progress has already
landed. `TrainingExecutor` emits batch callbacks and periodic progress
logs on both legacy and `IBatcher` paths, `TrainingManager` forwards the
batch state to the dashboard, and `TrainingPlotPanel` renders batch
count plus running loss and progressive fractional-epoch curves.

**Still deferred:** fractional validation cadence is a separate training
policy change and should not be mixed with progress visibility.

## Priority 4 - Studio And Workflow UX

### Query Console in CyxWiz Studio

Open Node Editor landed. Query Console remains a larger bridge:
grammar/parser, engine-to-NodeEditor bridge, mutation safety, and undo
integration.

**Next step:** start read-only query inspection before allowing graph
mutation commands.

**2026-06-03 update:** first read-only slice landed. Query Console now
parses CyxQL before execution and rejects `CREATE`, `DELETE`, and `SET`
clauses with a clear error instead of passing them to the executor.
Read-only graph inspection queries continue to run through the existing
CyxQL parser/executor path. Mutation commands remain deferred until they
have graph mutation safety and undo integration.

### Properties panel generic parameter editor

The generic parameter editor is still basic. Complex nodes should keep
dedicated dialogs; the generic editor should become safer for simple
typed params.

**Next step:** add typed validation and avoid editing hidden/internal
params that should only be owned by dialogs.

**2026-06-03 update:** first hardening slice landed. The metadata-driven
Properties panel now filters dialog-owned/internal state such as loaded
dataset status, audit counters, source type, and DataInput/DataOutput
file fields from the generic editor. Simple typed params use strict
integer/float parsing, range validation/clamping where metadata provides
a range, and dropdown aliases share enum handling. Invalid stored values
surface as validation errors instead of throwing during render. Complex
data nodes remain owned by their dedicated dialogs, while `DatasetInput`
keeps its registry dataset name editable because it has no dedicated
dialog owner.

### Memory estimation is approximate

Memory tab estimates are useful but still approximate across lazy and
streaming datasets.

**Next step:** keep estimates honest in UI and improve per-loader
metadata where it is cheap to compute.

**2026-06-03 update:** loader metadata audit found the useful cheap
metadata already present: tabular loaders report concrete in-memory or
Parquet-cache bytes, while image, audio, and text loaders mark their
reported byte count as an if-fully-cached estimate. The DataInput Memory
tab now makes lazy-loader estimates explicit as not reserved RAM and
labels pre-load file size as disk size, not a RAM reservation. Deeper
per-loader accounting can stay deferred until a loader has a concrete
cache metric to expose.

**Status:** complete for current Priority 4 scope. Remaining Query
Console graph mutation work is intentionally deferred behind undo and
mutation-safety design.

## Priority 5 - Model Coverage And Performance

### GRU verification

GRU CPU forward/BPTT landed, but smoke verification and ArrayFire path
cleanup remain.

**Next step:** run focused GRU smoke tests, then remove stale one-shot
warnings only after verification.

**2026-06-04 update:** focused GRU verification passed for the current
scope. The low-level `[gru]` suite passes, text smoke graph pattern
contracts pass, and the slow DebugExecutor path now exercises
synthetic sentiment-shaped `Embedding -> BiGRU -> Dense` graphs for
both single-layer and two-layer bidirectional GRU. The smoke verifies
finite loss, complete forward/backward execution, no missing gradients,
and the expected 192-feature bidirectional output width.

The Debug CRT assertion seen during the old slow runtime check happened
before GRU execution while loading an external sentiment CSV/vocab
fixture. That loader fixture belongs to dataset-loader robustness, not
GRU model verification, so the GRU smoke no longer depends on
`D:/demo/...` files. No stale GRU ArrayFire one-shot warning was observed
in the focused smoke run.

**Status:** complete for current GRU verification scope.

### Transformer verification

Transformer text path is still unverified relative to LSTM.

**Next step:** create or update one small text graph that exercises
Transformer training after Text Fix B stabilizes.

**2026-06-04 update:** TransformerEncoder text classification is now
verified for the current small-scope path. Added a backend
`TransformerEncoderModule` wrapper so `ModelBuilder` can build
`Embedding -> TransformerEncoder -> Flatten -> Dense` classifier graphs,
and added `examples/cyxgraph/text/test_05_sentiment_transformer_mini.cyxgraph`
as the graph-level smoke fixture.

The verification uncovered and fixed three backend bugs that only show
up once Transformer runs as a real sequence model:
- TransformerEncoder FFN now flattens `[batch, seq, d_model]` to
  `[batch * seq, d_model]` around Dense feed-forward layers and restores
  sequence shape afterwards.
- legacy `DenseLayer` bias broadcast now handles multi-row 2D input
  instead of only batch-size-one paths.
- MultiHeadAttention backward now uses a consistent attention-weight
  cache layout and computes `grad_attn` with the correct
  `[seq_q, seq_kv]` shape.

Slow DebugExecutor smoke now completes a tiny synthetic text Transformer
classifier with finite loss, complete backward, 19 gradient entries, and
zero missing gradients. Graph-pattern contracts include the new
Transformer mini graph.

**Status:** complete for current TransformerEncoder text-classification
scope. TransformerDecoder, generation, pretrained transformer import,
and full LLM fine-tuning remain out of scope and should not be presented
as completed.

### LSTM ArrayFire performance follow-up

LSTM correctness is resolved. AF performance optimizations remain
deferred.

**Next step:** profile before changing. Avoid speculative rewrites.

**2026-06-04 update:** focused Debug backend LSTM tests pass
(`cyxwiz-tests.exe "[lstm]"`: 8 test cases, 16 assertions) with a Debug
baseline of about 3.2 seconds on this workstation. No `[lstm][arrayfire]`
tagged test exists, so no AF-specific rewrite was made. Keep this as a
profiling baseline and only optimize after a real workload identifies a
recurrent hotspot.

**Status:** complete for current no-speculative-rewrite scope.
Deferred LSTM AF optimization work has been moved to
`docs/Data Studio/tofix16.md`.

### `num_workers` support is partial

Some loaders/batchers honor worker settings better than others.

**Next step:** document which data paths are actually parallel and make
unsupported paths explicit rather than pretending all loaders scale.

**2026-06-04 update:** `num_workers` is now treated as a bounded,
explicit training-batcher budget rather than a blanket async loader
promise. Graph compile and TrainingManager clamp requested values to the
platform default, and Arrow/Parquet batcher factories clamp again for
direct callers.

Current support is synchronous per batch, not background prefetch:
- legacy `DatasetBatcher` splits sample loading across workers.
- `ImageDatasetBatcher` and `AudioDatasetBatcher` split per-sample
  decode/feature extraction across workers.
- `ArrowDatasetBatcher`, text's Arrow-backed compatibility path, and
  `ParquetArrowBatcher` split feature-column extraction across workers
  when there is more than one feature column.

Unsupported/reserved fields are now explicit in UI/logs:
- `prefetch_factor` is serialized for future compatibility but ignored
  by current training batchers.
- `pin_memory` is serialized for future compatibility but ignored by
  current training batchers.

**Status:** complete for current explicit-contract scope. True async
prefetch queues and pinned host-memory transfers remain deferred
performance work and have been moved to `docs/Data Studio/tofix16.md`.

## Priority 6 - Tool-To-Node Migration

### Remaining standalone panels / dead NodeTypes

The TimeSeries block and multiple Cat-1 operators landed, but remaining
panel-only tools still need either:
- a real graph operator,
- a Cat-2 introspection panel contract, or
- explicit removal/defer status.

**Next step:** continue in small groups. Do not migrate a node unless
its data contract is clear and testable.

**2026-06-04 update:** first Priority 6 audit slice reviewed the pending
tensor shape/merge/reduction node exposure already present in the
worktree. The C++ editor integration builds, but these nodes are not yet
backend training-runtime operators and are not part of
`GraphCompiler::IsModelLayer`. They should therefore stay as
template/deferred metadata until a real runtime contract and tests exist.
Existing codegen/serialization groundwork can remain as scaffolding, but
the UI must not present Tensor* nodes as completed executable operators.

**Next step:** either add runtime/test contracts for one small tensor
operator group, or leave the group explicitly deferred and move to the
next panel-only/dead-node group.

**2026-06-04 update:** completed the next small group for time-series
analysis. `TimeSeriesDecomposition`, `ARIMAForecaster`, and
`ExponentialSmoothing` are real Cat-1 in-sample table operators already
registered in `PipelineOperatorFactory`; metadata and editor pins now
match that contract instead of describing old panel-style forecast
outputs. Added focused operator tests that verify each preserves row
count and appends its expected columns. Future-row forecasting remains
deferred because it changes table row count and needs a separate schema
contract.

**2026-06-04 update:** completed the remaining time-series analysis
nodes as Cat-1 table-output operators. `ACFNode` and `PACFNode` emit
lag-indexed statistic/confidence/significance tables.
`StationarityTest` emits a one-row ADF/KPSS summary table, and
`SeasonalityDetector` emits candidate periods with strengths and primary
detection fields. These nodes are now registered in the runtime factory,
metadata, editor creation path, graph load/type maps, pattern maps, and
CyxQL string conversion. The focused time-series operator test covers all
seven implemented analysis operators. Rich plot/dialog views can still
be added later as Cat-2 inspection surfaces, but the dead-node exposure
gap for this time-series group is closed.

**Status:** complete for the bounded time-series analysis group.

### Known dual-maintained registration lists

Node registration unification mostly landed, but some lists remain
dual-maintained.

**Next step:** only consolidate lists that are actively causing drift;
avoid broad registry rewrites during Text Fix B.

**2026-06-04 update:** added a focused drift guard,
`test_pipeline_operator_metadata`, that checks every
`PipelineOperatorFactory` node has metadata and is marked implemented.
The audit found real status drift: factory-backed `LogTransform`,
`Differencing`, `PolynomialRegressionNode`, `RobustScaler`,
`LabelEncoder`, `OrdinalEncoder`, `TargetEncoder`, and
`OutlierDetector` had no metadata, while `GMMCluster` and
`Convolution1D` were still template metadata despite registered runtime
operators. Metadata is now aligned for those executable Cat-1 operators.
This intentionally does not rewrite all enum/string/category maps.

**Remaining:** a deeper pin/parameter schema audit is still useful for
older factory-backed nodes whose metadata/frontend pins predate their
Arrow table operators, but that should be a separate small slice.

**2026-06-04 update:** completed the first pin/parameter schema slice
for factory-backed Cat-1 nodes. `LinearRegressionNode`,
`PolynomialRegressionNode`, `TargetEncoder`, `OutlierDetector`,
`FFTNode`, `FilterDesigner`, and `Convolution1D` now create dataset
input/output pins that match their Arrow table operators instead of old
model/tensor/panel-era shapes. `StandardScaler`, `MinMaxScaler`,
`LinearRegressionNode`, `FFTNode`, and `FilterDesigner` metadata now
uses the canonical operator params and one-table output contracts.

**Remaining:** continue schema audit for other older factory-backed
nodes only where drift is demonstrated by metadata/frontend/operator
mismatches.

## Priority 7 - Future Architecture

### Variable-shape Sample type

Deferred v3 consideration. Needed only when fixed-shape tensor batches
are no longer enough.

### Explicit Decode node

Deferred v3 consideration. Useful for generated/tokenized outputs, but
not required for the current Text Fix B path.

## Priority 8 - Deferred Bug And Warning Debt

Track issues observed while working Priority 6 but intentionally not
mixed into those slices:
- Build warning cleanup: unused `sigma` in backend `time_series.cpp`;
  numeric narrowing warnings in `data_analyzer.cpp`,
  `preprocessing_operators.cpp`, and `signal_processing_operators.cpp`;
  unreferenced stub parameters in `data_table.cpp`.
- `test_pipeline_operator_metadata` is intentionally useful now, but it
  links many operator translation units. Consider a lighter registry
  boundary if this starts slowing normal focused test builds.
- Some older Cat-1 operators still have legacy params retained for
  saved-graph compatibility. Audit before removing any persisted param
  names.

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
