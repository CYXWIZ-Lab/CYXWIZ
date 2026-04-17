# DataLoader Polymorphic Refactor — Plan

**Filed:** 2026-04-17
**Source:** Plan agent (expanded scope per user feedback)
**Scope:** Every per-category switch in the codebase (not just async)
**Size:** 8 commits, ~8-12 hours total, multi-session

## Problem

Every new data type in CyxWiz requires hand-editing ~5-10 unrelated files:

| Concern | Hand-coded per category today |
|---|---|
| Async dispatch (`DataInputDialog::Apply`) | 4 near-identical ~150-line blocks |
| Dialog restore (ctor probes registry per type) | 5 parallel if-blocks (~130 lines) |
| Memory tab rendering | 3-way switch on `loaded_backend_` |
| Training dispatch (`StartTrainingFromGraph`) | 5-way if/else, ~140 lines |
| `TrainingManager::StartTraining{X}` (5 methods) | 5 copies of ~90-line near-identical code |
| Compile gate dataset resolution | 5-way `IsXDataset` OR |
| Registry coordination | 5 parallel `Register/Unregister/Get/Is/Clear` sets |
| Memory estimation | Duplicated in Apply path + restore path |
| LRU / lazy-load semantics | Implicit per backend class |
| Synthetic data shape (Local Debug) | Would be yet another switch |
| Node-param schema | Categories leak stale params on switch |

Surfaced by this session's BarChart + image async + audio async work, all of
which duplicated 80% of pre-existing category code.

## Interface

Full `DataLoader` (place at `cyxwiz-engine/src/gui/loaders/data_loader.h`,
namespace `cyxwiz::loaders`):

```cpp
class DataLoader {
public:
    // Identity
    virtual FileCategory Category() const = 0;
    virtual const char*  CategoryName() const = 0;
    virtual int          BackendTag() const = 0;          // 1..5 for persistence

    // Apply-time async
    virtual bool ValidateApplyContext(const ApplyContext&, std::string& err) const = 0;
    virtual uint64_t LaunchAsyncLoad(const ApplyContext&,
                                     std::shared_ptr<AsyncLoadState>) = 0;
    virtual std::string FinalizeNodeParams(const AsyncLoadState&, gui::MLNode&) const = 0;

    // Restore on dialog reopen
    virtual bool RestoreFromRegistry(const std::string& name,
                                     const gui::MLNode& node,
                                     RestoreState& out) const = 0;

    // Memory tab
    virtual bool IsLazyLoaded() const = 0;
    virtual std::string MemoryStatusLabel() const = 0;
    virtual std::string MemoryBytesFormat(size_t bytes) const = 0;

    // Registry
    virtual bool IsRegistered(const std::string& name) const = 0;
    virtual void Unregister(const std::string& name) = 0;

    // Training dispatch
    virtual bool LaunchTraining(TrainingConfiguration config,
                                const std::string& dataset_name,
                                const std::string& label_column,
                                int epochs, int batch_size,
                                TrainingPlotPanel* plot,
                                std::function<void(bool)> node_editor_cb) = 0;

    // Compile gate
    virtual PreprocessingDomain Domain() const = 0;
    virtual bool LabelsFromStructure() const = 0;

    // Synthetic data (Local Debug)
    virtual SyntheticBatch MakeSynthetic(const TrainingConfiguration&, uint32_t seed) const = 0;

    // Node params schema
    virtual std::vector<ParamSchema> NodeParams() const = 0;
};
```

Supporting structs: `ApplyContext`, `AsyncLoadState`, `RestoreState`,
`ParamSchema`, `SyntheticBatch`.

## DataRegistry Strategy

**Option A (recommended):** Keep 5 maps. Add
`std::unordered_map<std::string, FileCategory> name_to_category_`
populated by each `Register*` method. Expose
`std::optional<FileCategory> ResolveCategory(const std::string& name)`.

Option B (unified `unordered_map<string, unique_ptr<DatasetHandle>>`)
touches 16 callsites across Data Studio, properties, pipeline — invasive
and orthogonal to the loader refactor. Defer to a separate quarter.

## File Layout

New files under `cyxwiz-engine/src/gui/loaders/`:
- `data_loader.h` — interface + structs + factory decls
- `data_loader.cpp` — factory, `GetByCategory`, `GetByRegisteredDataset`, `All`
- `tabular_loader.{h,cpp}` — CSV + Parquet (secondary `GetStorageKind()` enum)
- `image_loader.{h,cpp}`
- `audio_loader.{h,cpp}`
- `text_loader.{h,cpp}`
- `timeseries_loader.{h,cpp}` — subclass of TabularLoader

## Migration — 8 Commits

Each commit compiles + runs. Parallelizable groups noted.

1. **Foundation + TabularLoader** — interface + structs + factory + TabularLoader (handles Arrow + Parquet). Apply's tabular branch replaced. Other 4 branches stay inline.
2. **TextLoader** *(parallel after 1)*
3. **ImageLoader** *(parallel after 1)*
4. **AudioLoader** *(parallel after 1)* — Apply now 40 lines
5. **Restore + Memory tab + DataRegistry routing sidecar** *(after 1-4)* — collapses ctor restore block, Memory tab switch, PollAsyncLoadResult switch
6. **Training dispatch unification** *(parallel with 5)* — extract `TrainingManager::StartTrainingCommon` taking `IBatcher*`. 5 `StartTraining{X}` methods → 1. `StartTrainingFromGraph` collapses to 8 lines
7. **Compile gate + Domain + Labels** *(parallel with 5/6)* — `graph_compiler.cpp:150-158` + `:521-560` + `:198-205` all collapse to loader calls
8. **Node-param schema + Properties filter + synthetic data** — fixes stale params on category switch, MakeSynthetic powers Local Debug

## Open Questions

1. `TimeSeriesLoader` = TabularLoader subclass with overridden `Domain()` + `NodeParams()`? **Recommend yes.**
2. `VideoLoader` always-fails stub or keep category-switch guard? **Recommend stub.**
3. Plugin data loaders (`PluginDataLoaderRegistry`) join the refactor? **Recommend defer.**
4. `MakeSynthetic` takes full config or trimmed? **Recommend full.**
5. `IBatcher` abstract base in Commit 6 or 6b? **Recommend in-scope Commit 6.**
6. Post-refactor `loaded_backend_` handling: tag int vs TabularLoader `GetStorageKind()` enum? **Recommend enum.**

## Risks

- Plugin data loaders: separate registry today; plugin-owned loaders can't train through the refactored dispatch without explicit opt-in.
- Existing cyxgraph JSON files in `examples/cyxgraph/` persist `loaded_backend` int + `file_category` string — both survive the refactor (loaders produce same keys).
- `previous_dataset_name` unregister asymmetry: today only Tabular + Text clean old names; Image + Audio don't. Commit 1 extraction will force consistency — fixing a silent bug for free.
- Async publication order: `state->done.store(true)` must stay last in every worker lambda. Document in the `LaunchAsyncLoad` contract.
- `ApplyContext` copy is ~400 bytes — fine for per-Apply, not in any hot path.

## CMake

Add after `src/gui/data_input_dialog.cpp` (around line 195):

```cmake
    src/gui/loaders/data_loader.cpp
    src/gui/loaders/tabular_loader.cpp
    src/gui/loaders/image_loader.cpp
    src/gui/loaders/audio_loader.cpp
    src/gui/loaders/text_loader.cpp
    src/gui/loaders/timeseries_loader.cpp
```

## Execution Order Within a Session

Within a single session, land commits in this order:
- Session 1: Commits 1 + 2 (TabularLoader + TextLoader) — ~3h, 2 new loaders proven
- Session 2: Commits 3 + 4 (ImageLoader + AudioLoader) — ~2h, Apply shrinks to 40 lines
- Session 3: Commits 5 + 6 (Restore/Memory + Training dispatch) — ~3h, biggest wins
- Session 4: Commits 7 + 8 (Compile gate + Node-param schema) — ~2h, cleanup
