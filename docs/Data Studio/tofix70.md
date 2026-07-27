# To Fix 70 - Role-Aware Datasets and Truthful Data Input / Split / Loader Architecture

## Status

Completed for the production tabular workflow - 2026-07-27.

The Train/Dev/Test role resolver, Dataset-oriented split contract, external
held-out evaluation, fitted preprocessing reuse, truthful lazy preview,
streaming row limits, partition provenance, and checkpoint-to-test workflow
are implemented and covered by focused regressions. APS was used only as an
acceptance fixture; no APS path, schema, row count, or model rule is embedded
in the engine.

Completion boundary: Track 70 closes the Arrow/Parquet tabular training path
that motivated this ticket. Extending the same typed role contract to image,
audio, text, time-series, self-supervised, or reinforcement-learning runtimes
is follow-up work, as are typed preprocessing-state ports and exact optimizer
resume. Those extensions do not change the completed tabular contract.

## Tracked follow-ups - not Track 70 closure blockers

The following production extensions remain intentionally outside the completed
tabular contract and require their own scoped tickets and acceptance tests:

1. Extend resolved Train/Dev/Test roles to image, audio, text, and time-series
   adapters without adding modality-specific branches to Data Split.
2. Introduce typed training plans for self-supervised, unsupervised, and
   reinforcement-learning execution instead of treating every objective as
   supervised labels.
3. Replace file-mediated fitted preprocessing handoff with typed state ports
   and pin-aware same-graph Train/Dev/Test coordination.
4. Add checkpoint v2 with optimizer, scheduler, RNG, sampler, graph, and
   dataset identity required for exact training resume.
5. Persist complete run-comparison history rather than presenting loaded
   checkpoints as fabricated training runs.

These follow-ups may reuse the Track 70 DatasetAsset, role resolver, manifest,
and checkpoint-validation boundaries. They must not broaden or duplicate the
core tabular loader merely to share the same UI.

## Decision statement

Separate the responsibilities currently blurred across Data Input, Data Split,
and Data Loader:

```text
Data Input  = load a physical source into a CyxWiz dataset asset
Data Split  = resolve Train / Dev / Test roles and partition policy
Data Loader = create runtime batchers from resolved partitions
```

CyxWiz must support both:

1. one source dataset that needs an internal Train/Dev/Test split; and
2. externally supplied Train, Dev, and/or Test datasets, which are preserved
   exactly as supplied.

This is a data-role and provenance feature, not a new file format, compute
backend, or second dataset storage system. Data Input continues to use its
existing supported physical formats and Arrow-backed dataset representation.

## Why this exists

The current one-source workflow is useful:

```text
Data Input -> Data Split (80/10/10) -> training runtime
```

However, datasets such as APS are delivered as separate files:

```text
aps_failure_training_set.csv
aps_failure_test_set.csv
```

Treating both as one source and splitting again would contaminate the supplied
test set and make final evaluation misleading. The engine needs to know the
difference between a file's physical format and its machine-learning role.

## Existing foundation and implementation boundary

This is not a from-scratch training-data rewrite. The engine already has the
important foundations:

- Data Input imports its currently supported dataset formats;
- Data Split contributes train/dev/test split settings to compiled training
  configuration;
- the training runtime creates Train, Dev, and Test batchers;
- training validates during the run and evaluates Test after training.

The delivery must reuse those foundations and add a narrow role-resolution
boundary between them. Do not duplicate loaders, batchers, or the training
loop.

### Mandatory Data Input audit before implementation

Before changing data roles, audit the existing Data Input node end-to-end:

1. canvas pins and their declared types versus the actual loaded dataset
   artifact;
2. multi-source behavior: two independently loaded datasets must not overwrite
   one another in registry state, labels, paths, or UI selection;
3. dataset registration, source identity, schema, row count, label/target
   metadata, and error reporting;
4. compiler path discovery from each Data Input node through Partition Policy
   and Data Loader;
5. supported-format claims versus real importer/runtime support.

Fix Data Input defects exposed by this audit before adding external Test or Dev
role selection. A role-aware policy cannot be reliable if two source nodes do
not retain distinct, truthful dataset identities.

### Mandatory Data Split pin remediation

Current Data Split pins visually present six routed tensors for Train, Dev,
and Test, but runtime partitioning is internal and does not execute separate
graph branches from those pins. This contract must be corrected as part of the
work:

- new Data Split nodes use Dataset-oriented input/output contracts;
- legacy six-pin graphs remain loadable and receive a clear migration notice;
- compiler/runtime uses the resolved partition result, not apparent legacy
  tensor-link wiring;
- pin migration must not silently delete user links or alter a saved graph's
  effective split ratios.
## Target architecture

```text
                    physical source loading
┌──────────────────────────────────────────────────────────────┐
│ Data Input                                                     │
│ CSV / TSV / Parquet / Feather / Arrow / IPC -> DatasetAsset   │
└──────────────────────────────┬───────────────────────────────┘
                               │
                    one or more DatasetAssets
                               │
┌──────────────────────────────▼───────────────────────────────┐
│ Data Split / Partition Policy                                 │
│                                                              │
│ Training Dataset      required                                │
│ Validation Dataset    optional                                │
│ Test Dataset          optional                                │
│ ratios / seed / shuffle / stratification                      │
│                                                              │
│ -> ResolvedDatasetPartitions + PartitionManifest              │
└──────────────────────────────┬───────────────────────────────┘
                               │
┌──────────────────────────────▼───────────────────────────────┐
│ Data Loader                                                    │
│ batch size / training shuffle / workers / prefetch / balance  │
│ -> Train batcher, Dev batcher, Test batcher                   │
└───────────────┬───────────────────┬──────────────────────────┘
                │                   │
                ▼                   ▼
         weight updates       validation during training
                                    │
                                    ▼
                          final held-out test evaluation
```

Train, Dev, and Test are real runtime paths. They do not need to be three
separate tensor-routing branches on the canvas at this stage.

## Node responsibilities

### Agreed normalization and externally supplied role sources

Data Input loads physical sources independently. A training file, validation
file, and test file are therefore separate Data Input nodes; they are not
concatenated or re-split merely because they share a project.

```text
external train file -> Data Input --\
external dev file   -> Data Input ----> Data Split / Partition Policy
external test file  -> Data Input --/                 |
                                                      v
                                           Normalizer -> Data Loader -> Model
```

Data Split resolves the roles as follows:

- supplied Train, Dev, and Test sources keep their respective roles exactly;
- an absent Dev and/or Test role is derived only from the Train source using
  the selected ratios, seed, shuffle, and stratification policy;
- a supplied Test source is never merged into or split with Train.

Normalization is an explicit **Normalizer node**, not an invisible training
runtime behavior. It is ordered after role resolution and before batching:

1. fit learned normalization statistics from the Train partition only;
2. apply that fitted transform unchanged to Train, Dev, and Test;
3. create batches only after transformation.

Fixed, non-learned conversions such as image `uint8 -> [0, 1]` may be source
adapter options, but any transform that learns mean, standard deviation,
min/max, vocabulary, imputation values, or similar corpus statistics must fit
from Train only. This prevents validation/test leakage.

Before training, role resolution must validate schema compatibility: feature
names, order, types, selected target/label definition, and feature count. A
test source may omit labels only when it is explicitly used for inference;
evaluation requires labels. Errors must name the incompatible source and
columns rather than silently aligning by position.

The old Data Split tensor-pin facade is not a compatibility contract. New
Data Split semantics must represent Dataset/partition policy, with the
Dataset-to-batch/tensor boundary owned explicitly by Data Loader. Do not keep
unused Val/Test canvas routes merely to preserve unfinished graphs.

### Data Input - physical loading only

Data Input owns:

- source location, source format, parsing, and import errors;
- schema inspection and user-selected label/target column;
- creation or lookup of a `DatasetAsset` backed by the existing data layer;
- dataset identity/fingerprint: source identity, schema, row count, and
  content/version identity where practical.

Data Input does **not** decide whether its output is Train, Dev, or Test. A
single Data Input node produces one generic `Dataset` artifact that can later
be assigned a role by the partition policy.

The target canvas port is a Dataset artifact, not separate feature and label
tensor outputs. Feature and label selection is dataset metadata.

### Universal source boundary

All externally supplied data enters the engine through a Data Input boundary:

```text
file / folder / remote source -> Data Input or source adapter -> DatasetAsset
DatasetAsset                 -> Partition Policy -> Data Loader
```

The role-aware architecture is modality-agnostic: tabular, text, image, audio,
and time-series sources all become the same DatasetAsset contract before
partitioning. Data Split and Data Loader must not parse files, make network
requests, or contain format-specific loading paths.

This does **not** claim that every physical source format is implemented now.
The current generic Data Input supports only formats with a verified importer,
including CSV, TSV, Parquet, Feather, Arrow, and IPC. A future Excel reader,
JSON reader, image-folder loader, Hugging Face source, Kaggle source, database
connector, or remote adapter must first implement the narrow source-adapter
contract and then produce a DatasetAsset. It must not bypass Data Input by
placing file-loading behavior in Data Split or Data Loader.

```text
ISourceAdapter
  probe(source) -> supported | unsupported(reason)
  load(source, import_options) -> DatasetAsset | typed failure
  describe_schema(DatasetAsset) -> dataset metadata
```

The Data Input UI owns adapter selection and truthfully exposes only adapters
available in the current build. This keeps one source boundary while allowing
new formats/categories to be added independently of partition and training
logic.
### Storage and memory-backing contract

Loading a dataset does not always mean materializing every sample in RAM. The
existing tabular path already distinguishes small in-memory Arrow datasets from
large or user-forced disk-backed Parquet datasets. Other modalities may have
their own metadata and sample-cache behavior.

DatasetAsset must therefore carry a backing/storage capability, for example:

```text
InMemoryArrow | DiskBackedParquet | ImageCached | AudioCached | AdapterDefined
```

Partition Policy creates logical Train/Dev/Test views over that backing. It
must not copy a source dataset three times into memory merely to assign roles.
Data Loader creates bounded batches from each resolved view. Preview is also
bounded and must not force full materialization of a disk-backed dataset.

The implementation audit must record and expose the actual backing selected
for each loaded source, including any cache location and reason. In particular,
the current CSV policy may choose in-memory loading for files below its RAM
safety threshold or disk-backed Parquet caching for large/forced-disk loads;
format-specific paths must be audited rather than assumed to stream.
### Mandatory Data Preview

Data Input must provide a read-only preview after a source is loaded. Users
must be able to inspect what CyxWiz actually recognized before assigning a
Train, Dev, or Test role. The Asset Browser should expose the same preview for
an already registered DatasetAsset.

This is a verification experience comparable to opening a file in an editor,
not a promise to build a full Visual Studio Code clone or to load an entire
large dataset into the UI.

```text
source -> Data Input -> DatasetAsset
                       |-> Schema / metadata
                       |-> Format-aware preview
                       |-> Role assignment and partition policy
```

Minimum preview behavior by supported modality:

| DatasetAsset modality | Preview |
| --- | --- |
| Tabular: CSV, TSV, Parquet, Feather, Arrow, IPC | Paginated row/column grid; column names, types, null counts, row count, selected label, and limited sample values. |
| Text records | Document/row list with safely truncated text, selected text/label fields, encoding and record-count information. |
| Image dataset | Thumbnail grid plus selected-image inspector: dimensions, channels, dtype, label/class, and source path/identifier. |
| Audio dataset | Metadata and selected-item preview; waveform/spectrogram only when the audio adapter and decoder support it. |
| Unsupported or opaque binary source | Source metadata and a typed `preview unsupported` reason; never pretend the content was understood. |

An adapter advertises preview capability alongside load capability:

```text
ISourceAdapter
  probe(source) -> supported | unsupported(reason)
  load(source, import_options) -> DatasetAsset | typed failure
  preview(DatasetAsset, PreviewRequest) -> PreviewPage | typed failure
  describe_schema(DatasetAsset) -> dataset metadata
```

`PreviewRequest` is bounded: page/cursor, row or item limit, selected columns,
and a modality-specific item identifier. Preview must be lazy, paginated,
cancellable, read-only, and memory-bounded. It must not train a model, fit
normalization, alter the source, download arbitrary additional content, or
materialize a whole huge dataset merely for display.

The UI must truthfully distinguish:

```text
Format support: loaded and previewable
Format support: loaded; preview unavailable (reason)
Format support: unavailable in this build
```

File extensions appear in the picker only when a current adapter genuinely
supports loading them. A preview renderer may be introduced later, but the UI
must not claim a format is fully inspectable until that renderer works.
### Asset Browser reconciliation - remove the obsolete load path

The Asset Browser is a project-file discovery and organization surface. It
must not be a second dataset-loading path. The current legacy `Load Dataset`
action bypasses Data Input and directly registers a dataset in DataRegistry.
That loaded object has no Data Input node, no graph source identity, no
explicit Train/Dev/Test role, and no reliable route into the compiler. It is
therefore orphaned runtime state.

Remove the Asset Browser `Load Dataset` action and its direct
`DataRegistry::LoadDataset` callback path. Loading data for analysis or
training must begin from Data Input.

The replacement Asset Browser behavior is:

| Current action | Target action/behavior |
| --- | --- |
| `Load Dataset` | Remove. Offer `Create Data Input from this source` as a convenience only; it creates/configures a Data Input node, then Data Input performs the load. |
| `View in Table` / `Preview Data` | Remove the redundant context command. Selecting an already registered DatasetAsset displays the bounded side-pane preview backed by the shared Data Preview service. |
| `Quick Preview` on a supported dataset | Hide it for dataset sources so it cannot invoke a second parser. Keep generic file preview only for non-dataset assets such as code/Markdown. |
| `Show in Explorer` | Keep, but repair it as a file-system action with the exact selected file/folder and a visible typed failure if the OS launch fails. |

The Asset Browser does not parse an unregistered dataset source. The user
chooses `Create Data Input from this source`, then uses Data Input's
authoritative parser options and Preview tab. This keeps raw-source parsing in
one place. Selecting a dataset already registered by Data Input shows its real
schema and bounded page in the Asset Browser side pane without reloading it.

```text
selected registered DatasetAsset -> Asset Browser side pane -> DataPreviewService
raw dataset source -> Create Data Input -> Data Input Preview -> DatasetAsset
```

There must be one implementation for tabular table preview, one for text,
one for images, and so on�not separate Table Viewer, Quick Preview, and Data
Input preview parsers that silently disagree on extensions or schema. Existing
Table Viewer functionality may survive only as a panel shell around the shared
DataPreviewService; it must not own another CSV/Excel/HDF5 loader stack.

### Asset Browser acceptance checks

1. Right-clicking a dataset file cannot silently load/register it outside a
   Data Input node.
2. `Create Data Input from this source` creates a configured graph source but
   does not mark it loaded until Data Input succeeds.
3. Selecting an already registered dataset in Asset Browser shows the same
   schema, bounded sample, support state, and renderer as Data Input.
4. Inspecting a registered dataset never creates duplicate or orphaned
   DataRegistry training state.
5. Unsupported formats display one clear support/preview reason rather than a
   failed CSV fallback in Data Input.
6. `Show in Explorer` selects/opens the exact target and reports OS failures.
### Data Split - role resolution and partition provenance

Data Split becomes the authoritative partition-policy node. It owns:

- role assignment for input datasets;
- internal derivation of missing partitions from the training source;
- split ratios, seed, shuffle, stratification, and time-series constraints;
- schema/label compatibility checks;
- production of a stable partition manifest.

Target input contract:

```text
Training Dataset      required
Validation Dataset    optional
Test Dataset          optional
```

Target output contract:

```text
Partitioned Dataset Set
  train
  dev
  test
  manifest
```

Do not retain six apparent tensor outputs (`Train Data`, `Train Labels`,
`Val Data`, and so on) as normal active routing semantics while the runtime
does not consume separate graph branches. Old graphs must remain loadable, but
their legacy layout should migrate to this truthful Dataset-based contract.

### Data Loader - runtime iteration only

Data Loader receives resolved partitions. It owns:

- batch size and training-epoch policy;
- training-only shuffle and class balancing;
- worker/pre-fetch policy where actually supported;
- construction of one batcher for each resolved runtime phase.

Data Loader does **not** decide data roles or split raw datasets. It cannot
resplit an external Dev or Test dataset.

```text
Train batcher -> updates weights
Dev batcher   -> validation at the configured training point
Test batcher  -> final evaluation after selected checkpoint restoration
```

### Learning objective and target contract

The single Data Input `Dataset` output remains the correct source boundary for
supervised, unsupervised, self-supervised, and reinforcement-learning work.
Do not add a permanent Data Input label pin: a label column is optional dataset
metadata, and not every training plan has a dataset label.

The compiler must resolve a typed training plan from the objective/training
node, not infer supervision from the model architecture:

```text
TrainingPlan
  Supervised
    target_source: DatasetTarget
    target_requirement: Required

  SelfSupervised
    target_source: DerivedTarget
    target_requirement: Derived

  Unsupervised
    target_source: None | DerivedTarget
    target_requirement: None | Derived

  Reinforcement
    target_source: Experience
    required_fields: observation, action, reward,
                     next_observation, terminated/truncated
```

For supervised training, the selected loss/objective requires a resolved
target for every evaluated role. The compiler must fail before launch when the
target metadata is missing, the target column does not exist, role target
types disagree, or `DataLoader.Labels` does not reach the loss target input.
Inference-only Test data may omit a target; evaluation Test data may not.

For self-supervised training, the target is produced by a declared transform
or objective (for example masked tokens, shifted tokens, augmented views, or
autoencoder reconstruction). The source dataset does not need a label column.

For unsupervised training, clustering, embedding extraction, density
estimation, and inference can consume features without a target. An objective
that derives a comparison tensor must declare that derivation explicitly; the
compiler must not silently use the last table column as a label.

Reinforcement learning is a separate execution plan, not a special case of a
supervised `Labels` tensor. Online RL receives transition experience from an
environment/rollout buffer. Offline RL may use the same generic Dataset source,
but maps dataset fields into a typed experience schema. Rewards, actions, and
termination signals remain distinct semantic fields rather than being folded
into one label column.

The current `DataLoader.Data` plus optional `DataLoader.Labels` boundary remains
the compatibility surface for supervised dataset training. A future generic
batch artifact may carry optional targets and auxiliary fields, while RL uses
an `ExperienceBatch`/rollout boundary. This avoids adding task-specific pins to
Data Input and keeps the physical source layer independent of learning mode.

## Resolution rules

External partitions always win. Only the Training Dataset is eligible for
internal derivation of a missing partition.

| Supplied sources | Resolved partitions |
| --- | --- |
| Train | Derive Train, Dev, Test from Train using the configured policy. |
| Train + Test | Preserve Test; derive Train and Dev from Train. |
| Train + Dev | Preserve Dev; derive Test from Train only when explicitly enabled. |
| Train + Dev + Test | Preserve all sources; do not internally split them. |

The default one-file compatibility policy remains 80/10/10. If users supply
an external Test source, the UI must show that it is preserved rather than
quietly applying the previous default to all rows.

## APS example

```text
Data Input A: aps_failure_training_set.csv -> Training Dataset
Data Input B: aps_failure_test_set.csv     -> Test Dataset

Data Split / Partition Policy:
  Train source: external A
  Dev source: derive 10% from A
  Test source: external B, preserved

Data Loader:
  Train batcher: remaining 90% of A
  Dev batcher: 10% of A
  Test batcher: all of B
```

The external APS test CSV is never mixed into the training source, shuffled,
balanced, augmented, or internally split.

## Partition manifest

The smallest durable contract is a typed `PartitionManifest`, owned by the
partition policy and attached to the compiled training configuration and run
record.

```text
PartitionManifest
  version
  training_source_fingerprint
  validation_source_fingerprint?
  test_source_fingerprint?
  train_origin: external | derived
  dev_origin: external | derived
  test_origin: external | derived
  label_column
  feature_schema_fingerprint
  split_method: random | stratified | time_ordered | none
  train_ratio / dev_ratio / test_ratio when derived
  seed / shuffle
  resolved_row_counts
  manifest_fingerprint
```

It must be enough to answer, after a run:

```text
Which rows/data source were used for train, dev, and test?
Were these model results directly comparable to another run?
```

Do not build a second dataset database, duplicate Arrow tables, or serialize
every row index into the graph file in the first delivery. Store source and
policy identity; materialize row membership only through the existing dataset
partition/batcher mechanisms.

## Validation and leakage rules

Before launch, Partition Policy must validate:

1. Train source exists and label column is defined.
2. Externally supplied Dev/Test schemas are compatible with Train features and
   the expected label semantics.
3. Ratios are valid only for partitions that are derived.
4. An external partition is never resplit.
5. Duplicate IDs or exact duplicate rows across partitions are detected when a
   stable identifier or practical comparison is available; otherwise disclose
   that overlap verification was not possible.
6. Normalization, encoding, imputation fitting, class balancing, and training
   augmentation learn from or modify Train only. Dev/Test consume fitted
   training state without fitting their own.
7. Time-series policies may forbid random shuffle/stratification and require
   chronological partition boundaries.

Failures must name the role and source, for example:

```text
Test Dataset schema mismatch: missing feature 'sensor_14' present in Training Dataset.
```

## Canvas UX

The common one-source graph remains small:

```text
Data Input -> Partition Policy -> Data Loader -> Model -> Loss
```

The explicit-source graph is equally understandable:

```text
Training Data Input --\
Dev Data Input ------- > Partition Policy -> Data Loader -> Model -> Loss
Test Data Input -----/
```

Partition Policy inspector must show resolved, not merely configured, truth:

```text
Training: 44,000 rows, derived from training source
Dev:       4,669 rows, derived from training source, seed 42
Test:     16,000 rows, external source preserved
```

It must also show source names/fingerprints, label column, compatibility state,
and a concise reason for every derived or preserved partition.

## Runtime sequence

```text
load DatasetAssets
  -> validate and resolve PartitionManifest
  -> compile selected model path with manifest identity
  -> create train/dev/test batchers
  -> train on train batcher
  -> validate on dev batcher during training
  -> save/restore best dev checkpoint when configured
  -> run test batcher after training when available and requested
  -> publish metrics plus manifest fingerprint to the run ledger
```

The final test phase should be visibly distinguished from validation. Dev
metrics guide model selection; test metrics are final evaluation, not the
default ranking signal for repeated candidate selection.

## Compatibility and migration

- Existing one-Data-Input graphs load as `Training source only` and retain
  their current 80/10/10 derived behavior.
- Existing Data Split nodes with legacy tensor pins continue to load. New
  nodes use Dataset-oriented ports and truthful inspector language.
- Old graphs may be migrated on save after presenting the resolved policy; do
  not silently delete legacy pins or links.
- A graph with an external test file but no explicit role must require a user
  assignment. Filename guessing is advisory only and never changes behavior.
- Serialized source roles use stable node/dataset IDs, not display text or
  filesystem names alone.

## Relationship to model comparison

`tofix69.md` adds one active model branch at a time. All candidate models must
use the same resolved PartitionManifest to be directly comparable. The run
comparison ledger must include the manifest fingerprint and label records with
different manifests as not directly comparable.

## Non-goals

- A new physical data format or another loader framework.
- Separate visible tensor graph branches for validation and test.
- Automatic data-source role guessing based on names such as `train.csv`.
- Automatically treating every provided test file as safe or leak-free.
- Cross-validation, k-fold search, or automatic hyperparameter optimization.
- Distributed dataset coordination or a persistent experiment database.

## Acceptance criteria

1. One-source datasets retain the existing 80/10/10 workflow.
2. A user can load external Train, Dev, and Test sources using existing Data
   Input formats and assign their roles explicitly.
3. External Dev/Test data is preserved and never internally split.
4. Missing Dev/Test partitions can be derived only from Training data through
   an explicit policy.
5. Data Loader receives resolved partitions rather than raw data plus ratios.
6. Runtime creates separate Train/Dev/Test batchers without requiring three
   visible model graph branches.
7. Schema, label, and leakage checks produce role-specific diagnostics.
8. Training transforms fit Train only and are applied consistently to Dev/Test.
9. Run records include a stable PartitionManifest fingerprint and resolved row
   counts.
10. Legacy graphs load safely and new graph UX does not claim unsupported
    routing semantics.

## Test plan

- Train-only CSV: deterministic derived Train/Dev/Test row counts and seed.
- Train + external Test APS fixture: Test remains unchanged; Dev derives only
  from Train.
- Train + external Dev + external Test: no internal split occurs.
- Compatible and incompatible feature schemas; missing/mismatched labels.
- Row/ID overlap warning and unavailable-overlap-check disclosure.
- Verify normalization/encoding/balancing/augmentation fit Train only.
- Arrow and Parquet batcher tests for resolved external and derived roles.
- Graph save/load migration tests for legacy Data Split layouts and new role
  assignments.
- Run-ledger test for manifest fingerprint and comparison compatibility.
- Regression test for current one-model, one-dataset training.

## Delivery phases

1. Define typed DatasetRole, ResolvedDatasetPartitions, PartitionManifest, and
   graph serialization/migration rules. No new visual ports yet.
2. Add resolver and validation logic with Train-only and APS external-Test
   tests; keep the existing batchers behind the resolved contract.
3. Make batcher setup consume resolved partitions and prove external Test
   preservation.
4. Update Data Split/Data Loader compiler contracts and diagnostics.
5. Deliver the Dataset-oriented canvas UX and legacy-node migration notice.
6. Add run-ledger manifest provenance and comparison compatibility status.
7. Consider time-series-specific resolution and cross-validation only after
   the simple role-aware path is stable.

## Verified reference flow: MNIST MLP

The current `examples/cyxgraph/mnist_mlp.cyxgraph` is a valid one-source
reference flow and must remain a functional acceptance scenario while the
role-aware path is added.

Runtime evidence from the 2026-07-21 Release run:

- a 70,000-row, 784-feature `uint8` MNIST Arrow source was loaded;
- the internal training partition contained 55,996 samples and validation ran
  in 55 batches;
- fixed `mean=0, std=255` batch normalization was applied, which is a
  non-learned pixel-scale conversion and is safe for every partition;
- epoch 4 reported 97.86% training accuracy and 97.21% validation accuracy;
- the best validation checkpoint was written by the training runtime.

The graph's visible Data Split wiring is historical rather than an execution
contract: only its first output is linked to Data Loader, labels bypass it to
Loss, and none of its Val/Test output pins are linked. The runtime performs
partitioning and validation internally. The role-aware design therefore adds
external role resolution around this proven one-source path; it must not
replace the existing internal-split behavior.

### Smallest implementation sequence

1. Introduce a typed resolved-role configuration in the graph compiler:
   required Train dataset plus optional externally supplied Dev/Test dataset
   names and an internal-split policy for missing roles.
2. Resolve and schema-validate roles before materialization/training; keep the
   current one-source ratios as the fallback implementation of missing roles.
3. Extend batcher setup to consume the resolved role datasets, without adding
   loader logic to Asset Browser, Data Split UI, or Data Loader configuration.
4. Make Normalizer fit/train-apply semantics explicit for learned transforms.
5. Only after runtime role resolution is covered by tests, simplify the canvas
   Data Split contract to Dataset/partition-policy pins.

### Generic resolved-role runtime contract

Before changing individual batchers, introduce one typed, modality-neutral
runtime value owned by the compiler:

```text
ResolvedDatasetRoles
  train: DatasetSourceRef            required
  dev:   DatasetSourceRef | Derived  optional
  test:  DatasetSourceRef | Derived  optional
  policy: SplitPolicy                used only for Derived roles
```

`DatasetSourceRef` identifies the Data Input node, registered dataset identity,
label/target metadata, storage kind, and source modality. It is not a file
path and it is not a tensor pin. `Derived` means a logical partition of Train
with the existing ratios, seed, shuffle, and stratification policy; it must not
copy the source dataset.

The compiler owns resolution and validation. Each runtime adapter receives the
same resolved roles and produces role-specific batchers. Arrow, Parquet,
image, audio, text, and time-series implementations may differ internally,
but none may reinterpret roles or load a separate physical source themselves.

Acceptance requirements:

- one-source graphs resolve exactly as today: Train plus internally derived
  Dev/Test;
- supplied Dev/Test identities remain distinct through compilation, batching,
  run logging, and Run Test;
- every supplied source is loaded through Data Input and verified compatible
  before any training batcher is created;
- no role may silently fall back to "first dataset in graph" or a legacy
  registry handle.

## Implemented fitted-preprocessing slice - 2026-07-23

`FillMissing` and `StandardScaler` now implement the first generic production
contract for learned tabular preprocessing:

- `Fit + Transform` learns values from the current input and can persist a
  versioned `.cyxstate.json` artifact;
- `Transform Only` requires that artifact and reuses its training values
  without reading evaluation statistics;
- artifacts record operator identity/version, fit row count, selected feature
  names and Arrow types, configuration, and an ordered schema fingerprint;
- `FillMissing` persists mean/median/mode/constant values and excludes the
  configured label column; `StandardScaler` persists per-feature mean and
  scale;
- missing, unreadable, malformed, wrong-operator, empty, schema-incompatible,
  option-incompatible, and overwrite-conflict states fail closed with a
  corrective user message;
- graph validation blocks `Fit + Transform` when an upstream Data Input is
  explicitly assigned Dev, Validation, Test, or Inference role.
- stateful StandardScaler materialization is non-cacheable until artifact
  content identity is included in cache keys, preventing a replaced artifact
  at the same path from returning stale transformed data.

Legacy graphs remain `fit_transform`, do not save state, and execute as before.
This slice deliberately uses a file artifact across separate train and
evaluation executions. Typed preprocessing-state pins and pin-aware multi-input
execution remain future work; the runtime must not pretend that two visible
dataset inputs currently coordinate fit/apply state inside one node.

Schema validation compares semantic type families for fitted preprocessing:
numeric Arrow widths are mutually compatible, as are string-width variants.
This allows role-specific integer compaction without weakening name, feature
order, operator, or configuration checks. Numeric-to-string and other
cross-family changes still fail closed.

The verified APS acceptance run uses two explicit graphs rather than a
dataset-specific engine path:

- Train: 60,000 rows, `Fit + Transform`, 170 sensor features, `class` excluded.
- External Test: 16,000 rows, `Transform Only`, the same two training artifacts,
  170 sensor features, `class` excluded.
- Outputs: distinct Train and Test Parquet artifacts; neither evaluation
  operator computes replacement statistics from Test.

APS remains an acceptance dataset only. No APS filename, column list, row count,
or label value is embedded in the runtime.

## Implemented supervised tabular label/shape contract - 2026-07-23

The DataLoader `Labels -> Loss.Targets` edge expresses target flow, but it does
not identify which source-table field contains those targets. The selected
Data Input label metadata and the registered Arrow/Parquet schema now complete
that contract generically:

- Data Input Apply preserves an existing label while an asynchronous or
  disk-backed source has not yet populated the column selector;
- when no explicit label is stored, the compiler resolves conventional label
  names from the registered schema and records the effective Train role label;
- an explicit label absent from the schema fails closed with `CW-D-0102`;
- an unresolved supervised tabular label fails compile instead of silently
  treating the final column as the target;
- the compiler derives the model input width from numeric, non-internal
  feature fields after excluding the resolved label;
- the training launcher repeats label and feature-width reconciliation against
  the effective post-materialization schema before dispatch;
- Arrow and disk-backed Parquet batchers now use the same numeric-feature
  selection rule.

This removes the unsafe `input_size=1` fallback that could construct
`Linear(1, ...)` while a tabular batcher emitted a wider tensor. The fix is
schema-based and contains no APS-specific column or file logic.

### Resolved-role prefetch ownership fix - 2026-07-23

The first APS classifier run after the label/shape fix exposed an independent
generic crash when both an external dataset role and `prefetch_factor > 0`
were active. `TrainingBatcherSet` initially keeps concrete Arrow/Parquet
batchers beside non-owning `PrefetchBatcher` wrappers for inspection. During
the resolved-role handoff, the wrappers were retained but their concrete
sources were destroyed, leaving dangling pointers. Training then failed with
Windows access violation `0xc0000005` before the first batch.

The handoff now transfers each concrete source into its prefetch wrapper before
the temporary batcher set is destroyed. A focused regression moves an Arrow
prefetch set through `TakeResolvedExternalBatchers`, reads its sample count,
and consumes a valid batch. This lifetime correction applies equally to
Train, Dev, and Test roles and to Arrow and Parquet sources.

The follow-up APS run was stopped safely by the ownership invariant with
`PrefetchBatcher ownership source must match its wrapped source`. The external
Test batcher had replaced the derived Test batcher after prefetch wrappers were
created, so the retained wrapper still named the displaced source. Explicit
role assembly now:

- derives only roles that are not externally supplied;
- clears the displaced Arrow or Parquet role owner;
- applies preprocessing to the final Dev/Test sources;
- attaches prefetch wrappers only after role replacement is complete; and
- transfers the matching concrete source into each wrapper at task handoff.

Role ratios are also normalized independently of DataSplit-node presence.
Train plus external Test maps the Train source to Train/Dev only and preserves
all external Test rows. Train plus external Dev plus external Test maps 100% of
the Train source to Train. The focused batcher regression replaces a derived
Test source, attaches prefetching, performs resolved-role ownership handoff,
and consumes a valid batch from all six rows of the replacement source.

## Implemented binary classification target/decision contract - 2026-07-23

The first APS classifier run reported a constant 100% accuracy even while its
loss changed. `BCEWithLogits` itself was functional; the fault was the runtime
contract on either side of the loss:

- a one-output binary model caused the loader to one-hot encode labels with
  width one, mapping source label `0` to `[1]` and source label `1` to `[0]`;
- train, validation, smoke, and test metrics then applied multiclass `argmax`
  to a one-value output, which always returns index zero.

Binary losses now opt into scalar float targets with shape `[batch, 1]` and
preserve the encoded `0/1` values. `BCEWithLogits` decisions threshold logits
at zero; `BCELoss` decisions threshold probabilities at 0.5. Multiclass losses
retain score/one-hot tensors and `argmax`. One shared classification-decision
utility owns prediction, target, accuracy, and confidence semantics so train,
validation, smoke-run, and Run Test cannot drift independently. A scalar
binary test result also allocates a two-class confusion matrix rather than a
one-class matrix.

The scalar-target contract is forwarded through legacy, Arrow, disk-backed
Parquet, prefetch, image, audio, and text batchers. Focused regressions verify
that Arrow and Parquet preserve `0,1` scalar labels and that a deterministic
four-logit fixture reports 2/4 correct rather than the old false 4/4. The
broader training-batcher model-step suite passes on both Arrow and Parquet.

This is loss- and shape-driven engine behavior. APS remains only an acceptance
dataset; no APS filename, class ratio, or schema is embedded in the fix.
Checkpoints produced by the invalid one-hot-width-one run must not be used.

## Implemented Tools > Test external-role selection - 2026-07-24

The corrected APS run early-stopped after five non-improving epochs, restored
the best checkpoint from epoch 6, and successfully evaluated all 16,000
external Test rows inside `TrainingExecutor` (`test_loss=0.2848`,
`test_acc=96.11%`). Invoking Tools > Test afterward failed independently:
the command recompiled the graph, recognized `external test=true`, but selected
the Train dataset and reconstructed its `90/10/0` partitions. Its newly built
test batcher therefore contained zero rows.

Tools > Test now resolves the same semantic dataset roles as graph training:

- when an explicit Test role exists, its dataset identity, label metadata, and
  source node are selected instead of the Train source;
- an explicitly supplied Test dataset is consumed in full and is not split,
  shuffled, dropped, stratified, or class-balanced again;
- graphs without an explicit Test source preserve their configured derived
  test-split behavior;
- Arrow and disk-backed Parquet follow the same typed scope contract;
- Test loss is constructed through the same model-builder function as
  training, preserving reduction, weights, smoothing, and binary
  `pos_weight` settings.

Focused coverage proves that a Train plus external-Test configuration selects
the Test identity, retains every supplied row, and rebuilds
`BCEWithLogits(pos_weight=59)`. The full Debug engine build passes. The fix is
role-driven and contains no APS-specific name, path, row count, or schema.

## Implemented local checkpoint load for testing - 2026-07-27

The engine can now restore a completed model without retraining it first.
`Tools > Checkpoints > Load Checkpoint for Testing...` accepts either a concrete
checkpoint directory or a training-run directory containing `best`. The active
graph is compiled through the normal compiler and model builder, then the load
runs as a visible background task so the GUI remains responsive.

Loading is transactional. The checkpoint parameter count, names, shapes, and
data types must exactly match the model built from the current graph before any
parameter is changed. A successful load installs a shared active model for
`Tools > Test`; a failed load leaves the previous model untouched and reports a
specific task error. Cancelling the background task is also transactional: a
model that finished reading after cancellation is not installed. `Tools > Test`
compares the current graph fingerprint with the graph used for checkpoint load
and requires an explicit reload if the canvas changed. The Training Dashboard
identifies the active model as a checkpoint loaded for testing, including its
source path and recorded metrics.

Run Comparison remains a comparison of completed training runs from the current
session. A loaded checkpoint is not inserted as a fabricated training run,
because checkpoint format v1 does not contain the dataset, timing, partition,
and complete run metadata required by that table. Persisted run history and
exact training resume remain follow-up work. Exact resume requires a checkpoint
format that also stores optimizer/scheduler state, RNG and sampler progress,
the graph snapshot, and dataset identity; the current command is intentionally
named load-for-testing rather than resume.

The APS classifier checkpoint signature (`170 -> 128 -> 64 -> 1`) matches the
current classifier graph and can exercise this workflow, but compatibility is
validated generically and contains no APS-specific logic.

## Implemented canonical project graph directory - 2026-07-27

New projects now create a `cyxgraph` directory and register the `.cyxgraph`
asset filter. Opening an older project additively creates the missing directory
and merges the missing default filter without removing custom filters or the
user-created legacy `node` directory. Graph Save/Open dialogs start in the
active project's `cyxgraph` directory. When checkpoint output is not explicitly
configured, graph training uses the active project's `checkpoints` directory;
the runtime `.cyxwiz/checkpoints` path remains the fallback outside a project.
