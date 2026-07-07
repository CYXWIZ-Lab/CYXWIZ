# 50) Data I/O, dataset loading, and training backend contracts

## 50.1 Scope and boundary

This section records the complete data path from user selection -> registry -> compile-time validation -> runtime training dispatch for all supported domains (tabular, image, audio, text, and timeseries). It focuses on contracts, not implementation internals.

- Source entry points:
  - `DataInputDialog` drives apply-time loading for user workflows.
  - `AssetBrowser` and legacy asset import still call `DataRegistry::LoadDataset`.
- Data routing is now split by domain through `DataLoader` polymorphism.
- The compile and launch path is now coupled to runtime registry presence (`GetByRegisteredDataset`) rather than dialog cache fields.

## 50.2 Core data registry contract surface

`DataRegistry` remains the global ownership center for data registrations.

- Primary single entry points:
  - legacy generic loads (`LoadDataset`, `LoadMNIST`, `LoadImageFolder`, ...)
  - modern Arrow loads (`LoadArrowTable`, `LoadParquetToArrow`, `LoadCSVToArrow`)
  - tabular backend-specific paths (`LoadTabularCSV`, `RegisterParquetBacked`, `UnregisterTabularDataset`)
  - domain-specific registries for non-arrow runtime metadata (`ImageDatasetEntry`, `AudioDatasetEntry`, `TextDatasetEntry`)
- Domain map split contract:
  - `arrow_datasets_`: in-memory tabular tables.
  - `parquet_backed_datasets_`: disk-backed tables.
  - image/audio/text registry maps: metadata entries that batchers consume at training time.

ASCII contract:

```text
file path + user category
  -> Loader (category-specific)
     -> DataRegistry entry APIs
        -> training dispatch chooses one of:
           - Arrow in-memory map
           - Parquet-backed map
           - Image/Audio/Text entry maps
```

### 50.2.1 Plugin extension integration point

`DataRegistry::LoadDataset` includes plugin extension awareness:
- detects file extension and checks `PluginDataLoaderRegistry::HasLoaderForExtension`
- logs `bridge not yet implemented`
- returns failure for the unknown branch today

This is a contract hole from user perspective:
- plugin data providers can advertise loaders
- core `LoadDataset` currently does not instantiate an engine dataset for plugin results

## 50.2.2 Tabular backend auto-selection contract

`LoadTabularCSV` decides the backend at load time:
- read file size
- compare against `available_ram * 0.75`
- if override flag true or dataset too large -> Parquet path, else Arrow path
- logs decision and cache behavior

Before registration:
- `UnregisterTabularDataset(name)` is called so only one backend is active per name.

Backend outcomes:
- `TabularLoadBackend::InMemory`: registers through `LoadCSVToArrow` into `arrow_datasets_`
- `TabularLoadBackend::DiskBacked`: registers through `RegisterParquetBacked` into `parquet_backed_datasets_`
- `max_rows` is intentionally ignored on the disk-backed path (logged warning), with note for future row-group based support.

## 50.3 Loader abstraction contract

`DataLoader` is the runtime polymorphic contract in `gui/loaders/data_loader.h`.

Must provide:
- category identity (`Category`, `CategoryName`, `BackendTag`, `IsLazyLoaded`)
- preflight validation (`ValidateApplyContext`)
- async loading (`LaunchAsyncLoad` returns async task id)
- registry operations (`IsRegistered`, `Unregister`)
- restore & description (`RestoreFromRegistry`, `DescribeCompletedLoad`)
- launch (`LaunchTraining`)
- domain + behavior flags (`Domain`, `LabelsFromStructure`)
- compile/node schema metadata (`NodeParams`)
- synthetic sample generation (`MakeSynthetic`)

Global helpers:
- `GetByCategory`: category -> loader
- `GetByRegisteredDataset`: scan all loaders and ask each `IsRegistered(name)`
- `GetByBackendTag`: back-compat mapping from UI/backend tag

Loader registration order:
- Tabular, Text, Image, Audio

## 50.4 Tabular contract (Arrow + Parquet)

`TabularLoader` encapsulates the cross-backend tabular design.

Load contract:
- async thread resolves backend by calling `LoadTabularCSV`
- backend tag in state:
  - `1` => in-memory Arrow
  - `2` => disk-backed Parquet
- reads/updates row/col/bytes into `AsyncLoadState`
- writes dataset audit metadata into state for UI

Memory contract:
- not lazy (full memory estimate is real materialized memory for both paths at this level)
- row-level validation and audit run in loader before registry register is finalized for UI

Training contract (`LaunchTraining`):
- If `sequence_batch.enabled`, build sequence rows and start sequence path.
- Else Arrow-first probe then Parquet fallback:
  - `StartTrainingArrow` if `GetArrowDataset` succeeds
  - `StartTrainingParquet` if `GetParquetBackedDataset` succeeds
  - hard error if neither path available.

## 50.5 Image/audio/text loader contracts

All three follow the same contract with domain-specific entry fields:

### ImageLoader
- input: folder path + layout + size hints
- metadata contract:
  - registers `ImageDatasetEntry` with `num_images`, `num_classes`, folder path, label source
- lazy memory estimate:
  - estimate `rows * w*h*c*sizeof(float)` for UI and compile display
- training contract:
  - `StartTrainingImage(ImageDatasetEntry, ...)`

### AudioLoader
- input: folder + layout + optional CSV + feature config (sr, fft, mfcc/mel)
- metadata contract:
  - probes by constructing a temporary `AudioDataset` and storing `num_samples`, `num_classes`, feature shape
- lazy estimate: `num_samples * (feature_rows*feature_cols or fallback)`
- training contract:
  - `StartTrainingAudio(AudioDatasetEntry, ...)`

### TextLoader
- supports:
  - CSV/TSV raw text
  - arrow-native text files
  - non-native text with adapter conversion
- registers both:
  - raw `ArrowDataset` for textual columns (`RegisterArrowTable`)
  - `TextDatasetEntry` with tokenizer settings, class info, and vocab size
- lazy estimate:
  - `num_samples * max_length * sizeof(float)` upper bound
- training contract:
  - `StartTrainingText(TextDatasetEntry, ...)`

All category loaders:
- clear stale cross-category registrations before re-loads to prevent stale map collisions.
- `Unregister` removes only their domain metadata by default, with some defensive tabular cleanup.

## 50.6 Compile-gate to launch contract

### Compile gate (`GraphCompiler`)

- Loads `dataset_name` from node parameters (`dataset_name` then legacy `dataset` fallback).
- checks if loaded via:
  `loaders::GetByRegisteredDataset(config.dataset_name) != nullptr`
- ignores stale `data_loaded` hint and trusts registry truth.
- asks domain loader for `LabelsFromStructure` before warning about missing label column.
- derives preprocessing domain with `loader->Domain(file_category)`.

### Runtime launch (`MainWindow::StartTrainingFromGraph`)

- sequence preflight branch for explicit sequence dataset batching.
- generic branch:
  - resolve dataset loader by registry name via `GetByRegisteredDataset`
  - call loader-owned `LaunchTraining(...)`
  - fallback to `TrainingManager::StartTraining` only when no loader claimed the dataset name.

## 50.7 Training execution contract (mode lattice)

`TrainingManager` owns domain-specific start methods:
- `StartTraining` (legacy `DatasetHandle`)
- `StartTrainingArrow`
- `StartTrainingParquet`
- `StartTrainingImage`
- `StartTrainingAudio`
- `StartTrainingText`
- `StartTrainingSequence`

`TrainingExecutor` owns the runtime mode lattice:
- `Legacy`, `Arrow`, `Parquet`, `External`, `SequenceExternal`
- training loop dispatches to:
  - `BuildArrowTrainingBatchers`
  - `BuildParquetTrainingBatchers`
  - external IBatcher path (image/audio/text)
  - sequence batcher path

`BuildArrowTrainingBatchers` supports:
- stratified split via partition column
- class-weight balancing path when requested
- prefetch and normalization/one-hot wiring

`BuildParquetTrainingBatchers` currently logs capability gaps:
- no stratified split for disk-backed parquet
- no class balancing for disk-backed parquet
- drop_last unsupported (`last` partial batch retained)

## 50.8 Data lifecycle and cleanup contracts

- Cross-load cleanup:
  - loaders call `UnregisterTabularDataset` before re-registering same/related names.
- cross-domain cleanup:
  - image/audio loaders unregister stale own-domain entries before re-load.
  - `TextLoader` also drops stale tabular name entries before replacement.
- `DataRegistry::UnregisterTabularDataset`:
  - removes from both Arrow and parquet maps
  - cascades to `__materialized` companion dataset name to avoid stale compiled outputs.
- `ClearAllTabularDatasets` clears Arrow/Parquet/image/audio/text collections at project lifecycle boundaries.

## 50.9 Plugin data provider contract

`IDataProvider` contract includes:
- `GetLoaders()`
- `CanLoad(file_path)`
- `LoadDataset(file_path, dataset_name)`

`PluginDataLoaderRegistry`:
- stores loaders by `loader_id`
- indexes loader ids per extension
- validates load attempts by extension at call time
- `TryLoadDataset` returns `std::shared_ptr<PluginDataset>` with opaque pointer semantics

Current coupling to `DataRegistry`:
- plugin data provider discovery works
- adapter bridge from `PluginDataset` to engine dataset is incomplete

## 50.10 ASCII end-to-end contract (load -> train)

```text
User apply
 -> DataInputDialog.Apply
 -> DataLoader::LaunchAsyncLoad
   -> DataRegistry load call
      -> (Tabular) LoadTabularCSV -> Arrow or Parquet
      -> (Image/Audio/Text) domain-specific probe + metadata entry
   -> Async state -> dialog restore/status/params
 -> GraphCompiler: dataset_name in registry ? pass : block
 -> StartTrainingFromGraph:
    -> loader->LaunchTraining
       -> TrainingManager::StartTraining*
         -> TrainingExecutor with DatasetMode
            -> batcher build
               -> batch loop
               -> plugin hooks
               -> metrics + trace
```

## 50.11 Evidence anchors

| Claim family | Source |
|---|---|
| Loader polymorphism and dispatch helpers (`DataLoader`, `GetByRegisteredDataset`) | `cyxwiz-engine/src/gui/loaders/data_loader.h:221-366`, `cyxwiz-engine/src/gui/loaders/data_loader.cpp:33-136` |
| Tabular loader backend decision + async state + training dispatch | `cyxwiz-engine/src/gui/loaders/tabular_loader.h:19-88`, `cyxwiz-engine/src/gui/loaders/tabular_loader.cpp:64-523` |
| Image loader metadata register + batcher-based training | `cyxwiz-engine/src/gui/loaders/image_loader.h:12-46`, `cyxwiz-engine/src/gui/loaders/image_loader.cpp:24-255` |
| Audio loader metadata register + batcher-based training | `cyxwiz-engine/src/gui/loaders/audio_loader.h:12-46`, `cyxwiz-engine/src/gui/loaders/audio_loader.cpp:23-230` |
| Text loader dual representation (raw Arrow + text metadata) + training dispatch | `cyxwiz-engine/src/gui/loaders/text_loader.cpp:30-446`, `cyxwiz-engine/src/core/text_dataset_batcher.h:24-90` |
| Registry core maps, tabular dispatch, and cleanup lifecycle | `cyxwiz-engine/src/core/data_registry.h:276-463`, `cyxwiz-engine/src/core/data_registry.cpp:91-110`, `cyxwiz-engine/src/core/data_registry.cpp:1442-1513`, `cyxwiz-engine/src/core/data_registry_utils.cpp:287-365` |
| Compiler and launch path registry truth / no stale `data_loaded` trust | `cyxwiz-engine/src/core/graph_compiler.cpp:2857-2925`, `cyxwiz-engine/src/core/graph_compiler.cpp:3238-3708` |
| Runtime launch fallback to loader or legacy path | `cyxwiz-engine/src/gui/main_window.cpp:3176-3206`, `cyxwiz-engine/src/core/graph_compiler.cpp:3635-3710` |
| TrainingManager start matrix and dataset mode entry points | `cyxwiz-engine/src/core/training_manager.h:55-123`, `cyxwiz-engine/src/core/training_manager.cpp:168-520` |
| Executor mode dispatch and plugin hook points | `cyxwiz-engine/src/core/training_executor.h:222-248`, `cyxwiz-engine/src/core/training_executor.cpp:276-322`, `cyxwiz-engine/src/core/training_executor.cpp:415-460` |
| Parquet/backed vs arrow batcher capability limits | `cyxwiz-engine/src/core/training_batcher_setup.cpp:288-440` |
| Plugin provider registration + permission and TODO bridge | `cyxwiz-engine/src/plugin/interfaces/i_data_provider.h:18-40`, `cyxwiz-engine/src/plugin/registries/plugin_data_loader_registry.cpp:10-120`, `cyxwiz-engine/src/plugin/plugin_context.cpp:63-72`, `cyxwiz-engine/src/core/data_registry.cpp:91-118` |
