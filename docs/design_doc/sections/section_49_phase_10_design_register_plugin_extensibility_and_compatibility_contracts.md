# 49) Plugin extensibility and compatibility contracts

## 49.1 Scope and boundary

This section documents the extension seam between the core engine and externally loaded plugins, including:

- plugin discovery, manifest loading, signature/version gates,
- lifecycle state transitions and crash-isolated initialization/shutdown,
- runtime extension surfaces (nodes, panels, data loaders, hooks, analytics),
- runtime hook invocation contract into `TrainingExecutor`,
- compatibility and migration policy for legacy/dead nodes.

Contracts in this section are evidence-backed from:

- `cyxwiz-engine/src/plugin`
- `cyxwiz-engine/src/plugin/security`
- `cyxwiz-engine/src/gui/panels`
- `cyxwiz-engine/src/core/training_executor.cpp`
- `cyxwiz-engine/src/core/pipeline_runtime_capabilities.cpp` (compatibility policy context)

## 49.2 Plugin subsystem contract model

```text
Plugin assets (plugin_dir + plugin.json)
  -> PluginLoader::ParseManifest + API compatibility check
  -> PluginManager::LoadPlugin
     -> duplicate guard + permission/version/sig checks
  -> PluginManager::InitializePlugin
     -> OnLoad -> OnInitialize under crash isolation
     -> QueryInterface registration fan-out
  -> runtime-visible registries and plugin panels
```

Contract obligations:

- every plugin has an `IPlugin` instance created through the declared plugin entry-point functions,
- `OnLoad` then `OnInitialize` must run before registration side effects are considered active,
- plugin state transitions are explicit and surfaced via `PluginManager::GetPluginState`.

## 49.3 Runtime safety and state machine

Core runtime model:

- `PluginState` defines:
  - `Unloaded`, `Loaded`, `Initialized`, `Active`, `Failed`, `Disabled`.
- `LoadFromDirectory` moves plugin to `Loaded` after manifest parse, API check, and dynamic library initialization.
- `InitializePlugin` moves to `Initialized` only if lifecycle callbacks succeed.
- `ShutdownPlugin` returns registration state by removing interface registrations and dropping context.
- `UnloadPlugin` calls `OnShutdown` if needed and then `OnUnload`, then erases plugin entry.

Failure policy:

- any fail path sets `Failed` and keeps engine stable by preserving non-plugin execution,
- startup and lifecycle callbacks are wrapped in crash isolation (`SafeExecute` / `SafeExecuteBool`) so native crashes or exceptions do not abort process.

Permission policy:

- dangerous permission prompts can defer initialization through the permission store/dialog system,
- undecided dangerous permissions cause initialization retries after approval is captured.

## 49.4 Extension surfaces and API contract

### 49.4.1 Primary plugin interfaces

`IPlugin` defines:

- `OnLoad`, `OnInitialize`, `OnShutdown`, `OnUnload`,
- `GetManifest`, `GetRequiredPermissions`, `GetState`,
- `QueryInterface` for cross-DLL typed extension discovery.

### 49.4.2 Capability flags and manifest contract

- compile-time compatibility uses `api_version` major/minor compatibility.
- capability bits define what extension points plugin can provide (`Nodes`, `Panels`, `Data`, `Training`, etc.).
- permissions bits provide governance metadata for gating dangerous operations.

### 49.4.3 Interface-specific contracts

- `INodeProvider`: node catalog + pins + defaults + code generation + dynamic pin updates.
- `IPanelProvider`: panel metadata and render hooks (registered as plugin UI panels).
- `ITrainingHook`: optional training lifecycle callbacks + early-stop request.
- `IDataProvider`: file-format capability + load API for custom datasets.
- `IAnalyticsProvider`: optional metric/external analytics interface (registered when present).

Contract boundary rule:

- all extension objects are stored as non-owning pointers in plugin registries; engine owns registration lifecycle and cleanup.

## 49.5 Plugin registration and ownership contract

`PluginManager::InitializePlugin` performs bounded, crash-isolated registration by interface:

1. register plugin panels in `PluginPanelRegistry`,
2. register plugin nodes in `PluginNodeRegistry` (C-string callback safe path),
3. register training hook(s) in `PluginTrainingHookManager`,
4. register data loaders in `PluginDataLoaderRegistry`,
5. register analytics providers.

`PluginManager::ShutdownPlugin` and `ShutdownAll` must reverse every registered extension for deterministic teardown:

- remove panel registrations,
- remove node registrations,
- remove data-loader registrations,
- remove training hooks,
- remove analytics registrations.

## 49.6 Training hook contract into execution loop

`TrainingExecutor` consumes hooks at fixed points:

```text
Train() start:
  NotifyTrainingStart
Each epoch:
  epoch start -> NotifyEpochStart
  epoch end   -> NotifyEpochEnd
Before stop check:
  ShouldStopEarly (plugin can request early stop)
Train() finish:
  NotifyTrainingEnd
```

Execution contract:

- callbacks receive `TrainingContext` carrying epoch/batch counts and metric telemetry,
- `ShouldStopEarly` is consulted before epoch body execution,
- hook exceptions are isolated and do not crash the process.

## 49.7 Compatibility and migration policy (legacy/alias/dead-node surface)

Compatibility policy in the core runtime remains explicit and table-driven:

- operator-backed runtime contracts,
- fail-closed contracts for unsupported but documented node families,
- legacy executor fallback,
- alias decisions with `NormalizeToCanonical` vs `HiddenCompatibilityAlias`,
- source-kind compatibility through materializer/runtime capability tables.

This means compatibility behavior is explicit at compile and launch boundaries,
rather than implicit graph mutation in execution:

- unsupported/legacy domains can still flow with compatibility paths,
- stable training semantics are maintained only for nodes with explicit support mode.

Dead/legacy operators are still represented in headers and factory paths as compatibility surfaces, enabling controlled migration rather than silent failure.

## 49.8 Startup, UI, and lifecycle control contract

Plugin UX control points:

- Refresh/Load/Initialize/Enable/Disable/Unload actions via Plugin Manager panel,
- plugin state visualization and error messaging,
- plugin search/install flow can be async via task manager to keep app responsive,
- permission dialogs are rendered from main loop.

UI/plugin registry rendering contract:

- core + plugin UI panels are rendered via:
  - built-in plugin manager panel and standard panel tree,
  - `PluginPanelRegistry::RenderAllVisible` for plugin injected UI.

## 49.9 Runtime extension registry trace (ASCII)

```text
plugin.json
  -> PluginLoader::ParseManifest
  -> API/Signature checks
  -> LoadFromDirectory/CreatePlugin
  -> InitializePlugin
      -> OnLoad (Isolated)
      -> OnInitialize (Isolated)
      -> QueryInterface fan-out
          -> node/panel/data/hook registrations
  -> runtime usage
      -> TrainingExecutor hook callbacks
      -> UI render / panel registry
  -> Shutdown/Unload
      -> remove registrations
      -> OnShutdown/OnUnload
      -> dll release
```

## 49.10 Evidence anchors

| Claim family | Source |
|---|---|
| Manifest + version/permission/capability model | `cyxwiz-engine/src/plugin/plugin_types.h:16-177`, `cyxwiz-engine/src/plugin/plugin_types.h:178-320` |
| Entry API compatibility and signature checks for plugin binaries | `cyxwiz-engine/src/plugin/plugin_loader.h:32-48`, `cyxwiz-engine/src/plugin/plugin_loader.cpp:155-189`, `cyxwiz-engine/src/plugin/plugin_loader.cpp:241-279` |
| Plugin lifecycle load/initialize/shutdown/unload sequencing | `cyxwiz-engine/src/plugin/plugin_manager.h:37-72`, `cyxwiz-engine/src/plugin/plugin_manager.cpp:83-95`, `cyxwiz-engine/src/plugin/plugin_manager.cpp:133-161`, `cyxwiz-engine/src/plugin/plugin_manager.cpp:340-390`, `cyxwiz-engine/src/plugin/plugin_manager.cpp:394-433` |
| Crash isolation and permission-gated initialization | `cyxwiz-engine/src/plugin/plugin_manager.cpp:178-222`, `cyxwiz-engine/src/plugin/plugin_manager.cpp:249-276`, `cyxwiz-engine/src/plugin/security/permission_store.h:12-60`, `cyxwiz-engine/src/plugin/security/safe_execute.h:6-24` |
| Extension registration via QueryInterface and interface registries | `cyxwiz-engine/src/plugin/plugin_manager.cpp:305-336`, `cyxwiz-engine/src/plugin/interfaces/i_node_provider.h:76-127`, `cyxwiz-engine/src/plugin/interfaces/i_panel_provider.h:1-42`, `cyxwiz-engine/src/plugin/interfaces/i_data_provider.h:26-48`, `cyxwiz-engine/src/plugin/interfaces/i_training_hook.h:21-36` |
| Registry contracts for nodes/panels/data/training hooks | `cyxwiz-engine/src/plugin/registries/plugin_node_registry.cpp:11-45`, `cyxwiz-engine/src/plugin/registries/plugin_panel_registry.cpp:11-57`, `cyxwiz-engine/src/plugin/registries/plugin_data_loader_registry.cpp:14-44`, `cyxwiz-engine/src/plugin/registries/plugin_training_hook_manager.cpp:12-80` |
| Plugin UI control plane and async install flow | `cyxwiz-engine/src/gui/panels/plugin_manager_panel.cpp:122-125`, `cyxwiz-engine/src/gui/panels/plugin_manager_panel.cpp:251-272`, `cyxwiz-engine/src/gui/panels/plugin_manager_panel.cpp:444-500`, `cyxwiz-engine/src/gui/main_window.cpp:2598-2603` |
| Training loop hook invocation contract | `cyxwiz-engine/src/core/training_executor.cpp:456-516`, `cyxwiz-engine/src/core/training_executor.cpp:543`, `cyxwiz-engine/src/core/training_executor.cpp:712-713`, `cyxwiz-engine/src/core/training_executor.cpp:842-842` |
| Compatibility table policy (operator/legacy/fail-closed/alias) | `cyxwiz-engine/src/core/pipeline_runtime_capabilities.cpp:54-99`, `cyxwiz-engine/src/core/pipeline_runtime_capabilities.cpp:363-408`, `cyxwiz-engine/src/core/pipeline_runtime_capabilities.cpp:888-908`, `cyxwiz-engine/src/core/pipeline_runtime_capabilities.cpp:957-1049`, `cyxwiz-engine/src/core/pipeline_runtime_capabilities.h:17-86`, `cyxwiz-engine/src/core/pipeline_runtime_capabilities.h:117-156` |
| Data registry plugin-loader integration (currently bridge TODO) | `cyxwiz-engine/src/core/data_registry.cpp:107-118` |
