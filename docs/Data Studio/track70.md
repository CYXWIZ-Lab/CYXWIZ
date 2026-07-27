# Track 70 - Code Audit and Implementation Log for To Fix 70

## Purpose

Living source map and implementation record for `tofix70.md`. This file
records current engine truth before changes are made. It is not a second design
document.

## Audit snapshot - 2026-07-20

### Confirmed legacy bypass

The supplied log confirms Asset Browser direct loading:

```text
[2026-07-20 22:49:54.763] [cyxwiz] [info]
Dataset loaded successfully: aps_failure_test_set_1 (16014 samples)
```

Source confirms the bypass:

- `src/gui/panels/asset_browser.cpp:654-659` exposes `Load Dataset`.
- `src/gui/panels/asset_browser.cpp:1626-1680` calls
  `DataRegistry::LoadDataset(path)` asynchronously.
- `src/gui/main_window.cpp:1895-1915` receives the resulting handle, opens
  explorer UI, and updates a node name, but does not create/configure a Data
  Input graph source or assign a dataset role.

Result: the registry holds a dataset with no graph provenance, no Train/Dev/
Test role, and no reliable selected compiler path. This must be removed, not
extended.

### Current foundations to reuse

| Concern | Existing source truth | Ticket use |
| --- | --- | --- |
| Data Input async tabular load | `src/gui/data_input_dialog_apply.cpp`; `src/gui/loaders/tabular_loader.cpp:189+` | Retain as the only training-data load path. |
| CSV backing selection | `src/core/data_registry.cpp:1429-1518` | Preserve in-memory Arrow vs disk-backed Parquet choice. |
| Data Split config discovery | `src/core/graph_compiler.cpp:3078-3104` | Replace one-source settings extraction with role-aware resolved partitions. |
| Train/dev/test runtime | `src/core/training_batcher_setup.cpp:301+` | Reuse three internal batchers; do not add visible model branches. |
| Disk-backed training | `ParquetBackedDataset`, `ParquetArrowBatcher` | Partition views must not duplicate large data in RAM. |
| Completed-run ledger | `src/core/training_run_comparison.h`; `TrainingPlotPanel` | Later record partition-manifest fingerprint. |

### Current mismatches with To Fix 70

1. **Single selected dataset:** compiler configuration carries one
   `dataset_name`/data source, not explicit Train, Dev, and Test source roles.
2. **Misleading Data Split pins:** `src/gui/node_editor_nodes.cpp:920+` creates
   tensor/label Train, Val, and Test outputs. The compiler only reads split
   parameters; runtime creates internal batchers. The pins are not real
   independent evaluation routes.
3. **Duplicate preview stacks:**
   - Asset Browser `View in Table` dispatches separately in
     `src/gui/main_window.cpp:1917-1948` to `TableViewer` loaders.
   - Asset Browser Quick Preview uses `PreviewRenderer` in
     `src/gui/panels/asset_browser.cpp:1584+`.
   - Data Input has loading/audit state but no shared data-preview contract.
4. **Format truth differs by path:** Tabular CSV/TSV uses automatic
   memory/disk dispatch; Parquet and other loaders can materialize Arrow tables.
   Do not advertise universal streaming.
5. **No partition manifest:** no typed record currently proves external versus
   derived roles, source fingerprints, or split comparability.

## Required implementation order

1. Add focused characterization tests for current Data Input registration,
   source identity, and the Asset Browser bypass. Do not change UI first.
2. Remove Asset Browser direct registry loading; replace it with
   `Create Data Input from this source` plumbing only.
3. Introduce one shared bounded Data Preview service and route Data Input,
   Asset Browser dataset preview, and Table Viewer through it.
4. Define typed `DatasetRole`, `ResolvedDatasetPartitions`, and
   `PartitionManifest`; preserve backing-store identity.
5. Make compiler/batcher setup consume resolved partitions. Prove APS external
   Test remains untouched while Dev derives from Train.
6. Migrate Data Split from six legacy tensor outputs to a Dataset-oriented
   contract without deleting old graph links.
7. Add run-ledger provenance/comparison compatibility after runtime behavior is
   correct.

## First implementation slice

Asset Browser direct-load removal is the first safe slice. Acceptance evidence:

- right-clicking a dataset cannot call `DataRegistry::LoadDataset`;
- creating a Data Input source does not register/load it until Data Input Apply
  succeeds;
- no orphan dataset appears in the registry;
- existing non-dataset file actions remain unchanged.

## Code-review notes (unrelated dirty tree)

Do not include unrelated changes in Track 70 implementation commits. Current
review findings:

- **High:** root `README.md` has UTF-8/mojibake corruption and a BOM in its
  uncommitted edit; restore valid UTF-8 before commit.
- **Medium:** root `LICENSE` and `legal/LICENSE.md` duplicate the same draft
  license; establish one authoritative path before publication.
- **Medium:** `external/imnodes` has uncommitted submodule modifications;
  decide whether they belong in the submodule and keep it synchronized with the
  bundled engine copy before commit.
- **Low:** `resume.txt` appears unrelated and should not be committed.

The transformer, metadata-catalog, legal-draft, and model-comparison document
changes are outside Track 70 scope and must remain isolated.

## Log

- 2026-07-20: Created after source audit. No Track 70 engine code changed.
- 2026-07-21: Completed the Asset Browser boundary slice. Dataset activation now
  creates a configured, unloaded Data Input node; it no longer calls
  `DataRegistry::LoadDataset` or registers an orphan dataset. The obsolete
  table-view callback now uses the existing bounded quick-preview renderer.
  Release target `cyxwiz-engine` builds successfully.
- Remaining: make Data Input Apply the authoritative loader with preview and
  source metadata, then implement Data Split's external dev/test routing and
  Data Loader runtime hand-off.
- 2026-07-21: Data Input audit confirms that Data Input Apply is already the
  authoritative asynchronous loading and registry-registration boundary. It
  sets `data_loaded=false` while loading and only marks it true after loader
  completion; reopening reconciles against the registry.
- 2026-07-21: Preview audit found two different layers: Data Input has a
  source-aware pre-Apply preview (including import options and label mapping),
  while Asset Browser uses `PreviewRenderer` for raw file viewing. They cannot
  be merged by replacing one with the other without losing import-aware
  behavior; extract a bounded shared preview model before deduplicating UI.
- 2026-07-21: Data Split audit confirms a migration is required before pin
  changes. Its metadata exposes legacy Tensor/Labels inputs and six tensor
  outputs, yet runtime treats it as graph-level partition configuration.
  Normalizer still accepts Tensor input, so changing only Data Split metadata
  to Dataset pins would break new default graphs. Define the Dataset-to-tensor
  boundary and add migration coverage before changing those pins.
- 2026-07-21: Found and corrected the separate Run Test path for modern Arrow
  Data Input datasets. Training and its held-out test already used the Arrow
  registry/batcher path, while `StartTestingFromGraph` incorrectly required a
  legacy `DatasetHandle`. Run Test now dispatches Arrow sources through
  `TestManager` and an Arrow test batcher built by the same modern batcher
  setup. Release build passed. Manual MNIST GUI verification remains pending.

- 2026-07-21: Manual MNIST verification passed after the Arrow Run Test fix. The Run Test command completed through the modern path with 97.64% accuracy and no legacy-registry error.

- 2026-07-21: Generic training-runtime audit: Arrow and Parquet already expose separate train/dev/test IBatchers through TrainingBatcherSet, while the external image/audio/text path currently assigns one external batcher to both train and validation. Role-aware work must introduce a common resolved-role batcher contract for every modality; it cannot be implemented as an Arrow-only feature.

- 2026-07-21: Added the compiler-owned `ResolvedDatasetRoles` boundary. The
  currently selected Data Input resolves explicitly as Train; Dev/Test stay
  absent rather than being fabricated. Added a narrow compiler assertion for
  that contract. The focused graph-compiler executable reaches this assertion
  but the overall executable currently fails later on an unrelated existing
  TransformerDecoder expectation. The six focused Release data-loader tests
  pass (6/6).

- 2026-07-21: Release `test_training_batcher_setup` passed. It confirms the
  current Arrow/Parquet `TrainingBatcherSet` partitions Train/Dev/Test and
  applies train-only balancing. Implementation decision: retain one
  role-neutral batcher-set boundary at the executor; each loader must produce
  its role batchers before Data Input exposes separate external Dev/Test
  sources. This prevents an Arrow-only compiler/runtime branch.

- 2026-07-21: Role-aware foundation implemented and Release-built:
  `ResolvedDatasetRoles` resolves Train plus explicit `dataset_role=dev` /
  `dataset_role=test` Data Inputs; Dev/Test sources are permitted off the
  Train tensor path and preflight-validated as registered. The launcher keeps
  supplied Dev/Test identities while materialization updates Train. Each role
  now carries its own resolved label column. `TrainingExecutor` and
  `TrainingManager::StartTrainingExternal` accept `ResolvedExternalBatchers`.
  `TakeResolvedExternalBatchers` transfers existing Arrow/Parquet batcher-set
  ownership into that runtime container without copying data. Release engine
  builds passed after these batches.

- 2026-07-21: **Next implementation batch (do this first):** guarded
  tabular Arrow role dispatch in `src/gui/loaders/tabular_loader.cpp`. Keep the
  present sequence and single-source paths unchanged. When explicit role
  sources are present and all are Arrow-backed: build Train with
  `BuildArrowTrainingBatchers`; replace supplied Dev/Test with full-source
  `ArrowDatasetBatcher`s (`train_split=1.0`, `split_phase=Train`, their own
  `ResolvedDatasetRole::label_column`); apply the same normalization,
  regression, and one-hot settings; call
  `TrainingManager::StartTrainingExternal`. Reject or fall back clearly for
  mixed backing stores until their adapter is implemented. Then add focused
  coverage and validate APS Train + external Test end-to-end.

- 2026-07-21: Do not expose a Data Input Train/Dev/Test role selector until
  the guarded tabular Arrow path above is proven. Image, audio, text, and
  mixed Arrow/Parquet role dispatch remain pending. The shared preview/Table
  Viewer and legacy Data Split pin migration are separate remaining Track 70
  work.

- 2026-07-21: Existing Release 	est_training_batcher_setup.exe passed (Arrow/Parquet partition and balancing coverage). Full engine regeneration is currently blocked by access-denied writes under uild/ and cpkg during CMake configure. The guarded tabular Arrow explicit-role dispatch is present but awaits a successful rebuild.

- 2026-07-21: Explicit tabular Arrow Dev/Test batchers now receive train-fitted normalization and output encoding/regression mode; class balancing remains Train-only. Elevated Release engine rebuild passed.

- 2026-07-21: APS acceptance fixture confirmed at D:\Use_cases\APS\APS\datasets; training CSV header uses class as label after the license preamble, with 60,000 Train and 16,000 external Test rows. No APS-specific implementation is planned.

- 2026-07-21: Added generic supplied-role schema validation at graph-training launch. Explicit Dev/Test roles must now be registered Arrow/Parquet tabular sources before dispatch; their label column must resolve, label type must match Train, and ordered numeric feature columns/types must match Train after excluding labels/internal columns. Validation runs before the async launch and again after preprocessing materialization updates the effective Train dataset. Added focused launcher coverage for missing external Test label and mismatched external Test feature type. Release `test_graph_training_sequence_preflight`, `test_training_batcher_setup`, `test_training_executor_arrow_parquet`, and `cyxwiz-engine` build passed.

- 2026-07-21: Extended explicit tabular role dispatch beyond Arrow-only. `TabularLoader` now accepts Train/Dev/Test roles backed by Arrow or Parquet in the same launch, builds Train through the matching batcher setup, and replaces supplied Dev/Test with full-source role batchers for either backend. Fixed `ParquetArrowBatcher` so `train_split=1.0` preserves every row group for externally supplied full-source roles. Added focused coverage that external Parquet Test preserves all row groups. Release `test_training_batcher_setup`, `test_training_executor_arrow_parquet`, `test_graph_training_sequence_preflight`, and `cyxwiz-engine` build passed.

- 2026-07-21: Added first run-ledger partition provenance fields. Training run comparison records and CSV export now include Train/Dev/Test source names, role origins (`external` vs `derived`), role label columns, resolved row counts already present in metrics, and a stable Track70 partition-manifest fingerprint computed from role sources/origins/labels, split ratios, seed/shuffle policy, and resolved row counts. This is intentionally a flat comparison-record contract, not a separate manifest database. Focused run-comparison coverage passed, along with Release `test_text_gui_training_launch`, `test_graph_training_sequence_preflight`, `test_training_batcher_setup`, and `cyxwiz-engine` build.

- 2026-07-21: Surfaced Track70 partition provenance in the Training Plot run-comparison table. The panel now shows role source names, role origins, role label columns, and a short partition ID with the full fingerprint available as a tooltip. This uses the flat comparison-record fields added in the prior batch and avoids introducing a separate manifest UI before the runtime contract is fully stable. Release `cyxwiz-engine` build passed.
- 2026-07-22: Made run-comparison partition compatibility explicit. The Training Plot table now compares each row's partition-manifest fingerprint with the top-ranked reference run and labels the result as same, different, or unknown; different manifests are identified as not directly comparable. Added focused coverage for equal, changed, and missing fingerprints without adding a second manifest store or duplicating provenance fields. Release test_text_gui_training_launch and cyxwiz-engine build passed.
- 2026-07-21: Exposed the explicit tabular Data Input dataset role selector now
  that the guarded Arrow/Parquet backend path is proven. The Settings tab
  persists `dataset_role=train|dev|test`; Dev/Test continue through the
  compiler-owned validation and full-source role batcher path rather than a UI
  shortcut. APS remains an acceptance fixture only, not a special case.
- 2026-07-21: Added role overlap preflight for externally supplied Dev/Test
  tabular sources. When Train and a supplied role share a stable identifier
  column such as `sample_id`/`*_id`, launch now blocks on detected overlap with
  a role-specific leakage diagnostic. For small tables without such an
  identifier, the launcher falls back to bounded exact-row comparison; for
  larger tables it logs that overlap verification was not practical instead of
  pretending the source is leak-free.
- 2026-07-21: Added the first shared bounded tabular data preview service in
  `src/core/data_preview_service.*`. It previews already-registered Arrow and
  Parquet-backed datasets by dataset identity with offset/row-limit/selected
  column requests, returns schema/backend/cursor metadata plus typed failure
  reasons, and does not load files or register datasets as a side effect. This
  is the replacement boundary for Data Input and Asset Browser preview wiring;
  the old file-path preview helpers remain temporarily as callers are migrated.
  Release `test_data_preview_service` passed.
- 2026-07-22: Wired Data Input's loaded tabular preview path to the shared
  `DataPreviewService`. After Apply, Preview reads a bounded registered
  Arrow/Parquet page by dataset identity and displays backend, total row count,
  and next cursor metadata. The older source-file preview remains only as the
  pre-Apply fallback while Asset Browser/Table Viewer callers are migrated.
- 2026-07-22: Migrated the Asset Browser dataset preview pane off legacy
  `DataRegistry::GetPreview(path, ...)` for registered tabular data. The
  registry now records source/cache path provenance for Arrow/Parquet-backed
  tabular datasets, and the pane resolves a selected file to its registered
  dataset name before calling `DataPreviewService`. Raw file peeking remains
  in Quick Preview; the side pane no longer silently reloads or reparses files.
- 2026-07-22: Removed the now-orphan legacy preview API and renderer hooks:
  `DataRegistry::GetPreview(path, ...)`, `DatasetPreview`,
  `PreviewRenderer::RenderDatasetPreview(...)`, and
  `TablePreviewRenderer::LoadFromDataset(...)`. Raw Quick Preview remains a
  file-only preview path, while registered Arrow/Parquet previews flow through
  `DataPreviewService`.
- 2026-07-22: Wired graph testing dispatch for registered Arrow and Parquet
  tabular datasets. `StartTestingFromGraph` now prefers the compiler-resolved
  dataset name, resolves the Data Input label column, and dispatches to
  `TestManager::StartTestingArrow` or `StartTestingParquet` instead of falling
  through to the legacy DatasetHandle registry path.
- 2026-07-22: Removed the remaining misleading Asset Browser `Preview Data`
  menu path that opened the raw Quick Preview parser for table-like files.
  `Preview Data` is now dataset-only and refreshes the registered-dataset side
  pane backed by `DataPreviewService`; raw file peeking stays under Quick
  Preview. Updated the orphan Table Viewer empty-state copy so it no longer
  points users at the removed `View in Table` action.
- 2026-07-22: Removed the orphan Table Viewer source-file loading stack.
  `TableViewerPanel` no longer exposes CSV/TXT/HDF5/Excel file-loader methods,
  no longer compiles `table_viewer_loading.cpp`, and no longer retains its old
  `LazyDataTable` file-index/cache path. Table Viewer remains only an
  already-materialized `DataTable` display shell; registered/source previews
  stay on the shared bounded `DataPreviewService` path.
- 2026-07-22: Hid raw Quick Preview for dataset assets in Asset Browser.
  Dataset files now expose `Preview Data` and `Create Data Input` only, so
  dataset inspection cannot silently fall back to the old raw file table parser;
  Quick Preview remains available for non-dataset files such as code, Markdown,
  JSON, images, and opaque binary peeks.
- 2026-07-22: Added the first Data Split legacy-contract notice without
  changing saved-graph pin shape. The Data Split dialog now states that
  Train/Val/Test tensor outputs are compatibility pins, while runtime
  validation/test execution is created from compiler-resolved dataset
  partitions rather than separate canvas branches. The node pin descriptions
  now carry the same warning until the Dataset-oriented pin migration lands.
- 2026-07-22: Mirrored the Data Split legacy-contract notice into the
  Properties panel and node documentation. Users now see the same truth from
  the quick inspector, rich dialog, node pins, and docs: legacy Train/Val/Test
  pins remain loadable, but runtime evaluation follows resolved dataset
  partitions.
- 2026-07-22: Added a compiler info diagnostic for reachable Data Split nodes
  that still expose Train/Val/Test tensor pins. Compile now records the same
  compatibility truth as the UI: those pins remain loadable, while validation
  and held-out testing come from compiler-resolved dataset partitions.
- 2026-07-22: Repaired Asset Browser `Show in Explorer` to launch the OS file
  browser without building a shell command string. Windows now asks Explorer to
  select the exact asset path and logs a typed failure when the OS launch fails.
- 2026-07-22: Fixed the modern node catalog blocker before Data Split pin
  migration. Catalog preview placeholders no longer overwrite real implemented
  metadata, and supported Transformer/activation nodes now remain searchable as
  implemented rather than being downgraded to template/deferred previews.
- 2026-07-22: Aligned Data Split's modern metadata catalog descriptions with
  the legacy-pin contract. Node Browser/search metadata now says Train/Val/Test
  pins are compatibility pins and runtime uses compiler-resolved partitions.
- 2026-07-22: Started the actual Data Split pin migration. Newly created
  Data Split nodes and modern catalog metadata now expose Dataset role inputs
  (Training/Validation/Test) plus one `Partitions` Dataset output. Legacy
  Train/Val/Test Tensor/Labels pins remain supported for saved graph fixtures.
- 2026-07-22: Aligned new Data Loader nodes with the Dataset partition
  boundary. New Data Loader nodes now consume a single `Partitions` Dataset
  input and emit model-facing batched `Data`/`Labels`; legacy saved graphs
  with raw Data/Labels inputs remain loadable.
- 2026-07-22: Aligned new Data Input nodes with the Dataset artifact boundary.
  New Data Input nodes now emit one `Dataset` output matching the modern
  catalog, and the startup showcase now routes `Data Input -> Data Split ->
  Data Loader -> Normalize -> Model` with labels supplied by `DataLoader.Labels`.
- 2026-07-22: Fixed legacy/no-editor pattern insertion for the role-aware data
  boundary. The fallback Pattern Library path now creates modern DataInput,
  DataSplit, and DataLoader pins instead of generic Tensor Input/Output pins,
  with a focused pattern guard covering the fallback.
- 2026-07-22: Stopped pattern link pin-index clamping. Pattern links with
  stale legacy pin indices are now skipped with a warning instead of being
  silently rewired to the wrong modern Dataset role pin. The pattern guard now
  covers stale DataInput/DataSplit/DataLoader label-link indices in both the
  creator and no-editor fallback insertion paths.
- 2026-07-22: Applied the same stale-pin rule to direct pattern-file graph
  loading. `NodeEditor::LoadPatternFromFile` now skips out-of-range template
  pin indices with a warning instead of falling back to the first pin and
  possibly changing Dataset role semantics.
- 2026-07-22: Updated binary-model conversion graph generation for the modern
  data boundary. Generated `graph.cyxgraph` files now include DataInput ->
  DataSplit -> DataLoader before model layers, and loss targets are wired from
  `DataLoader.Labels` instead of the removed `DataInput.Labels` output.
- 2026-07-22: Cleaned stale user-facing `DataInput.Data` / `DataInput.Labels`
  guidance after the Dataset boundary migration. OneHotEncode, BarChart, and
  the loss-connectivity diagnostic now point new graphs at `DataLoader.Data` /
  `DataLoader.Labels`, while naming DataInput/DataSplit label pins as legacy
  compatibility sources only where relevant.
- 2026-07-22: Migrated the remaining saved graphs and text smoke templates
  off the legacy four-wire DataInput/DataSplit/DataLoader boundary. Sentiment
  and causal-LM examples now keep only Dataset -> split and Partitions ->
  loader links, while model targets continue to come from `DataLoader.Labels`.
  Focused guards now use the modern boundary and reject stale saved-example pin
  indices.
- 2026-07-22: Hardened saved-graph link restoration in both file and in-memory
  load paths. Explicit negative, malformed, or out-of-range pin indices are now
  skipped with a warning instead of being silently clamped to pin 0; truly
  index-less legacy links retain their first-pin fallback. Removed the unused
  legacy pin-ID heuristic and added focused index-resolution coverage.
- 2026-07-22: Finished the template-format label-source migration. MNIST,
  BERT encoder, and time-series examples now route targets from
  `DataLoader.Labels`; BERT templates gained the minimal Data Split/Data Loader
  boundary. Pin-connectivity negative fixtures were rewritten to preserve each
  intended compile failure without stale DataInput label pins, and the example
  guard now scans both saved-graph and template formats.
- 2026-07-22: Reproduced the APS CSV preamble failure against the supplied
  Train/Test files. Their real 171-column header follows 20 license/metadata
  rows. The Apply loader already honors skip_rows=20, but source preview did
  not; preview now skips the same physical rows, changing the setting
  invalidates stale preview state, and the UI clarifies that the next row is
  treated as the header. Added a generic preambled-CSV regression fixture so
  no APS-specific filename or schema logic enters the engine. CSV failures now
  also point users to delimiter/header settings and explicit source-row skipping.
  Release test_data_preview_service and cyxwiz-engine build passed.
- 2026-07-23: Completed the production parser-state refresh behind the
  preambled-source fix. Data Input now restores header, delimiter, skip-row,
  row-limit, and encoding settings before initial schema discovery; changes to
  path, format, delimiter, header, or skip rows rebuild the column/label list
  immediately while preserving a still-valid selection. Async tabular loading
  now resolves `Auto` from the source extension, so `.csv` does not fall into
  the Arrow loader. The supplied APS Train file was rechecked generically:
  line 21 is a 171-column header from `class` through `eg_000`, and line 22 is
  the first `neg` record. Release `test_data_preview_service`,
  `test_tabular_loader_apply_context`, and `cyxwiz-engine` passed.
- 2026-07-23: Clarified the learning-objective contract in To Fix 70. A Data
  Input continues to emit one generic Dataset with optional target metadata.
  Supervised plans require dataset targets; self-supervised plans derive
  targets; unsupervised plans use no target or an explicit derived objective;
  reinforcement learning consumes typed transition experience rather than
  treating rewards/actions as a label column. The existing compiler remains a
  supervised single-batch executor and must gain separate typed training-plan
  validation before unsupervised or RL sketches are executable.
- 2026-07-23: Moved the existing Data Input source preview out of the content
  rendered below Settings and into a dedicated `Preview` tab beside Settings.
  The same bounded preview loader and table/image/text renderers are reused;
  the buried duplicate placement was removed so the grid receives the tab's
  usable viewport instead of appearing as a separator below the dialog fold.
- 2026-07-23: Added bounded lazy scrolling for registered Arrow and
  Parquet-backed tabular previews. The Preview tab exposes the full virtual row
  count, uses ImGui row clipping, requests 100-row pages on an
  `AsyncTaskManager` worker, prefetches near page boundaries, and retains at
  most five LRU pages (500 rows) regardless of dataset size. Hidden columns do
  not render cell text. Page failures stop automatic retries and offer an
  explicit retry action. Paging is invalidated on Apply, unload, reset, or
  parser changes, and is used only when the registered source and applied
  parser contract still match. Raw pre-Apply delimited preview remains a
  bounded source sample until a streaming cursor/index adapter is implemented.
- 2026-07-23: Corrected the Limit Rows checkbox so its visible state is derived
  from persisted `max_rows`; disabling it now writes `0` (all rows), enabling
  it defaults to 1,000, and changing it invalidates preview pages from the
  previously applied dataset. The UI now states the current backend truth:
  in-memory CSV produces a final capped dataset but still parses the full file
  before slicing, while the disk-backed Parquet path does not yet enforce
  `max_rows`. Streaming CSV row caps and a logical disk-backed row limit remain
  production gaps and must not be confused with bounded Preview paging.
- 2026-07-23: Implemented versioned fitted preprocessing artifacts for the
  generic `FillMissing` and `StandardScaler` nodes. Node properties now expose
  Fit + Transform / Transform Only, feature and label columns, state path,
  save, and overwrite policy. Fit persists training-derived imputation values
  or scaler mean/scale; Transform Only validates operator/configuration/schema
  and reuses the artifact without fitting on evaluation data. Pipeline
  validation rejects Fit + Transform behind explicit Dev/Validation/Test/
  Inference Data Input roles and logs corrective guidance for missing paths,
  unsaved fits, bad artifacts, all-null columns, and schema mismatches. Focused
  routing regressions prove that test imputation uses training mean 3 rather
  than test mean 9 and that a test value equal to the training scaler mean maps
  to zero. Debug routing suite passed (122.8 s) and the Release engine build
  passed. Stateful scaler caching is disabled until artifact content identity
  participates in cache keys. File-based state reuse is the intentional first slice; typed state
  ports and same-graph multi-input coordination remain deferred.
- 2026-07-23: Verified the fitted-preprocessing contract end to end with the
  APS Train/external-Test acceptance case. Separate v0.2 graphs fitted
  `FillMissing` and `StandardScaler` on all 60,000 training rows and transformed
  all 16,000 held-out test rows from those saved states. Both operators covered
  170 sensor features and excluded `class`; the test execution loaded artifacts
  reporting `fit_rows=60000` and did not rewrite them. The resulting Train and
  Test Parquet artifacts are distinct. Preprocessing schema validation now also
  treats Arrow numeric-width variants as compatible numeric types (and
  string-width variants as compatible string types), preventing legitimate
  Train/Test integer compaction differences from rejecting Transform Only.
  Focused routing coverage forces different integer widths across fit and
  transform inputs and passed.
- 2026-07-23: Fixed the supervised tabular label warning and first-batch crash
  exposed by the APS classifier acceptance graph. `DataLoader.Labels` remains
  the correct target-flow connection, while Data Input/schema metadata now
  identifies the physical target column. Apply no longer clears a stored label
  merely because an async/disk-backed selector is temporarily empty. Compile
  auto-resolves conventional labels such as `class`, derives the exact numeric
  feature width, and blocks missing explicit labels with `CW-D-0102` instead of
  falling through to `Linear(1, ...)`. Launch reconciles the effective schema
  again after materialization, and Arrow/Parquet batchers share the numeric
  feature contract. Resolver, compiler, batcher-setup, sequence-preflight, and
  Release engine builds passed; APS remains only an acceptance dataset.
- 2026-07-23: Traced the subsequent APS Train access violation to resolved-role
  prefetch ownership, not labels or model shape. Compile and batch setup were
  correct (`class`, 170 features, `Linear(170 -> 128)`), but the role handoff
  retained non-owning prefetch wrappers after destroying their Arrow/Parquet
  sources. The first wrapper call dereferenced freed memory and Windows
  reported `0xc0000005`. `TakeResolvedExternalBatchers` now transfers source
  ownership into each wrapper before temporary owners expire. The focused
  batcher suite covers post-handoff sample access and batch consumption with
  prefetch enabled.
- 2026-07-23: The new ownership invariant then exposed a late external-role
  replacement mismatch: the external Test batcher replaced the derived Test
  owner after its prefetch wrapper had already captured the old source.
  Explicit tabular roles are now assembled before prefetch attachment, the
  displaced Arrow/Parquet owner is cleared, and runtime split ratios enforce
  the resolved mapping even without a DataSplit node. Train plus external Test
  derives only Train/Dev from the Train source and preserves all external Test
  rows; externally supplied Dev and Test leave the entire Train source for
  training. The replacement-source prefetch handoff regression and focused
  batcher suite pass.
- 2026-07-23: Fixed the false 100% accuracy contract for scalar binary models.
  The loss implementation was functional, but a one-output model incorrectly
  received width-one one-hot targets (`0 -> [1]`, `1 -> [0]`) and every metric
  used `argmax` on a one-value output. BCE/BCEWithLogits now receive preserved
  scalar `[batch, 1]` targets; probability/logit decisions use 0.5/0.0
  thresholds, while multiclass retains argmax. One shared decision utility is
  used by train, validation, smoke, and test execution, and scalar binary
  testing uses a two-class confusion matrix. Arrow and Parquet label-shape
  regressions plus a deterministic anti-argmax accuracy fixture pass. The
  change applies across supported modalities and contains no APS-specific
  logic; checkpoints from the invalid run are not valid acceptance artifacts.
- 2026-07-24: Fixed Tools > Test ignoring a supplied external Test role. The
  accepted training run had already restored the epoch-6 best checkpoint and
  evaluated all 16,000 external rows (`test_loss=0.2848`, `test_acc=96.11%`),
  but the separate command selected Train and reconstructed its `90/10/0`
  split, producing zero test batches. Test selection now uses the compiled
  Test role identity and consumes a supplied Test dataset in full without
  repartitioning, shuffling, dropping, stratifying, or balancing it. Derived
  test splits retain existing behavior. TestExecutor also uses the shared loss
  builder, preserving settings such as `BCEWithLogits pos_weight=59`.
  Role-selection, whole-dataset, and configured-loss regressions plus the full
  Debug engine build pass; the contract is generic across Arrow and Parquet.
- 2026-07-27: Wired a production-safe local checkpoint workflow into the GUI.
  The current graph is compiled through the shared model builder, checkpoint
  tensors are validated by count/name/shape/type before atomic installation,
  loading runs as a visible task, and the restored shared model is available to
  Tools > Test. The Training Dashboard reports it as the active loaded model;
  Run Comparison remains truthful session-only training history. Exact optimizer
  resume and persisted run history are explicitly deferred to checkpoint v2.
- 2026-07-27: Standardized project graph storage on `cyxgraph`. New projects
  create it and register `.cyxgraph`; older projects are upgraded additively,
  preserving custom filters and legacy folders. Save/Open graph dialogs now use
  that project directory, and an unspecified training checkpoint path resolves
  to the active project's `checkpoints` directory.
- 2026-07-27: Closed the remaining `Limit Rows` ingestion gap. Bounded
  in-memory CSV loads now stop the Arrow streaming reader at `max_rows`, and
  disk-backed conversion stops writing Parquet after the same cap. The cap is
  part of the Parquet cache identity, so an earlier full/unbounded cache cannot
  be reused for a bounded load. Focused Arrow and forced disk-backed
  regressions preserve exactly the requested number of rows.
- 2026-07-27: Fixed lazy-preview paging falsely reporting `preview schema
  changed while paging`. The dialog now requests the complete registered
  schema by stable ordinal on every page instead of round-tripping mutable
  column names. If a registered dataset is genuinely refreshed while a page
  is in flight, the bounded cache adopts the new schema as a new generation
  rather than terminating preview. This is format- and dataset-neutral.
- 2026-07-27: To Fix 70 is complete for the production Arrow/Parquet tabular
  scope. The accepted contract covers one-source derived partitions, explicit
  Train/Dev/Test sources, held-out Test preservation, Train-fitted preprocessing,
  role-specific validation, partition fingerprints/run comparison, modern
  Dataset pins with legacy compatibility, bounded lazy preview, streaming row
  limits, and checkpoint loading for later testing. Non-tabular modality
  adapters, typed preprocessing-state ports, and exact training resume remain
  separately scoped extensions rather than hidden Track 70 work.
- 2026-07-27: Removed the redundant Asset Browser right-click `Preview Data`
  command. Selecting an already registered dataset continues to show the
  shared bounded Dataset Preview pane; raw sources use `Create Data Input`,
  where parser options and the authoritative Preview tab are available. Data
  Input role help is now contextual: Train explains fitting and derived
  partitions, Dev/Validation explains validation and early stopping, and Test
  explains final held-out evaluation and the prohibition on fitting or model
  selection.
- 2026-07-27: Made Data Input column inclusion truthful. The Transformation
  tab persists included columns by name, keeps a selected label included, and
  applies projection during CSV ingestion for both in-memory Arrow and
  disk-backed Parquet caches. Other Arrow-backed tabular formats use the same
  named projection after load. Limit Rows remains row-specific and now points
  users to Transformation for columns; positional `skip first N columns` was
  intentionally rejected because schema changes would silently select the
  wrong features.
- 2026-07-27: Reconciled the Asset Browser acceptance contract with the final
  implementation. The redundant right-click `Preview Data` command is not a
  second workflow: selecting an already registered DatasetAsset shows the
  shared bounded side-pane preview, while an unregistered dataset source goes
  through `Create Data Input` and Data Input's authoritative parser/Preview
  tabs. The ticket now records non-tabular roles, typed training plans, typed
  preprocessing-state ports, checkpoint v2 exact resume, and persisted run
  history as explicit follow-up tickets rather than hidden closure work.
- 2026-07-27: Added a fast Track 70-only execution mode to the existing
  pipeline routing regression. Running
  `test_pipeline_executor_operator_routing --track70-ingestion-limits`
  terminates independently and verifies named projection plus row caps on both
  backends: in-memory Arrow returns 2 rows containing only `label`; forced
  disk-backed Parquet returns 3 rows containing only `feature`. The Debug test
  passed in 0.7 seconds.
- 2026-07-27: Fixed the Debug-only ImNodes scope assertion exposed while
  stabilizing the Track 70 UI. Both the primary node editor and the legacy
  Data Studio canvas now query `IsNodeHovered` exactly once after
  `EndNodeEditor`, cache the node ID, and reuse it for tooltip/context-menu
  routing. The assertion-enabled Debug engine rendered the restored graph for
  more than 20 seconds with empty stderr and remained responsive; Debug and
  Release engine builds passed.
- 2026-07-27: The closure review found and fixed a scalar-binary balancing
  reset regression. Scalar `[batch, 1]` targets are required by BCE losses, but
  they are not regression targets and must not disable a requested training
  sampler. `ArrowDatasetBatcher::Reset` now rebuilds balanced indices for
  binary classification. The batcher regression resets an oversampled 8:1
  fixture and verifies the next epoch still contains 8 negative and 8 positive
  scalar labels; the complete Debug batcher suite passed.
- 2026-07-27: Hardened checkpoint-to-test state during the closure review.
  Cancellation is checked before model construction, after parameter loading,
  and immediately before atomic installation, so a cancelled task cannot
  replace the active model invisibly. A loaded checkpoint also retains the
  graph fingerprint from its compatibility check; Tools > Test refuses to use
  it after the canvas graph changes and directs the user to reload. The Debug
  engine rebuilt successfully after both guards.
- 2026-07-27: Final focused verification passed:
  `test_pipeline_executor_operator_routing --track70-ingestion-limits` proved
  named projection and bounded rows on in-memory Arrow and forced disk-backed
  Parquet, and `test_training_batcher_setup` covered resolved roles, prefetch
  ownership, scalar BCE targets, class-balancing reset, held-out Test selection,
  configured binary loss, and Arrow/Parquet training batches.
