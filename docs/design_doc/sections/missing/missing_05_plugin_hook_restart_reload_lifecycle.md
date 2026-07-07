# M-05) Plugin hook lifecycle around restart / reload

## M-05.1 Question
How are training hooks removed and restored when a plugin is disabled, restarted, or reloaded?

## M-05.2 Lifecycle boundaries

```text
LoadFromDirectory
  -> InitializePlugin
     -> Register nodes / panels / data loaders / analytics / training hooks

ShutdownPlugin / DisablePlugin
  -> remove training hook registrations by plugin
  -> remove node/panel/data/analytics registrations
  -> plugin.OnShutdown if loaded

UnloadPlugin
  -> OnShutdown(if needed)
  -> OnUnload
  -> context cleanup

InitializePlugin again
  -> re-run registration
```

## M-05.3 Training-hook contract
- Hooks are registered through `PluginTrainingHookManager` and resolved by plugin id.
- Shutdown path removes hooks via `RemoveByPlugin`, preventing stale callbacks from dead plugin instances.
- Restart/reinitialize therefore relies on plugin-context rebind, not persistent global hook pointers.

## M-05.4 Risk
- If plugin state transitions are interrupted (permission deferral, failed OnInitialize, partial unload), hook registration must remain consistent with active registry snapshot. Existing `SafeExecute*` and failure-state transitions are designed to avoid hard crash impact but need lifecycle-order regression tests.

## M-05.5 Evidence anchors
- `cyxwiz-engine/src/plugin/plugin_manager.cpp:133` (plugin initialization and registration)
- `cyxwiz-engine/src/plugin/plugin_manager.cpp:149` (plugin instance validation before registration)
- `cyxwiz-engine/src/plugin/plugin_manager.cpp:313` (training hook registration via SafeExecute)
- `cyxwiz-engine/src/plugin/plugin_manager.cpp:354` (ShutdownPlugin orchestrates hook/panel/data loader cleanup)
- `cyxwiz-engine/src/plugin/plugin_manager.cpp:375` (training hook deregistration call during shutdown)
- `cyxwiz-engine/src/plugin/plugin_manager.cpp:381` (ShutdownPlugin completion)
- `cyxwiz-engine/src/plugin/plugin_manager.cpp:451` (DisablePlugin route)
- `cyxwiz-engine/src/plugin/plugin_manager.cpp:416` (UnloadPlugin keeps context during OnUnload)
- `cyxwiz-engine/src/plugin/plugin_manager.cpp:432` (UnloadAll orchestration)
- `cyxwiz-engine/src/plugin/registries/plugin_training_hook_manager.cpp:12` (RegisterHook)
- `cyxwiz-engine/src/plugin/registries/plugin_training_hook_manager.cpp:19` (RemoveByPlugin)
- `cyxwiz-engine/src/plugin/registries/plugin_training_hook_manager.cpp:45` (NotifyTrainingStart)
- `cyxwiz-engine/src/plugin/registries/plugin_training_hook_manager.cpp:54` (ShouldStopEarly path)
- `cyxwiz-engine/src/plugin/registries/plugin_node_registry.cpp:44` (node registration)
- `cyxwiz-engine/src/plugin/registries/plugin_node_registry.cpp:54` (RemoveByPlugin cleanup)
