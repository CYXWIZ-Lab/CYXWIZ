# Plugin API reference (C++)

Every type a C++ plugin sees, what the Engine does with it, and what works
today. For a walk-through read the [quickstart](plugin-quickstart.md) and the
[developer guide](plugin-developer-guide.md); for the rules a trainable
module must keep, the [node contract](node-contract.md).

> Checked 2026-10-17 against plugin API 1.2.0 and CyxWiz SDK 0.2.

## 1. Headers

Include the plugin headers as `"plugin/..."` and the backend headers as
`<cyxwiz/...>`. All of them need only the C++ standard library; nothing
from ArrayFire, spdlog, ImGui or nlohmann/json is required.

| Header | Holds |
|---|---|
| `plugin/plugin_types.h` | `IPlugin`, `PluginManifest`, versions, permissions, capabilities, states, `CYXWIZ_PLUGIN_ENTRY` |
| `plugin/plugin_context.h` | `PluginContext`, `AssistantContextSnapshot` |
| `plugin/plugin_fingerprint.h` | `CurrentBuildFingerprint()` (used by `CYXWIZ_PLUGIN_ENTRY`) |
| `plugin/interfaces/i_tensor_node_provider.h` | `ITensorNodeProvider`, `TensorNodeInfo` (trainable nodes) |
| `plugin/interfaces/i_node_provider.h` | `INodeProvider`, `PluginNodeTypeInfo` (signal nodes) |
| `plugin/interfaces/i_panel_provider.h` | `IPanelProvider`, `PluginPanelInfo` |
| `plugin/interfaces/i_training_hook.h` | `ITrainingHook`, `TrainingContext` |
| `plugin/interfaces/i_data_provider.h` | `IDataProvider`, `PluginDataLoaderInfo`, `PluginDataset` |
| `plugin/interfaces/i_analytics_provider.h` | `IAnalyticsProvider`, `PluginAnalyticsInfo`, `AnalyticsResult` |
| `plugin/interfaces/i_assistant_provider.h` | `IAssistantProvider` (used by the CyxWiz assistant plugin) |
| `core/extension_node_contract.h`, `core/node_metadata.h`, `core/graph_model.h` | `ExtensionShapeResult` and the types it builds on |
| `<cyxwiz/sequential.h>` | `cyxwiz::Module`, `cyxwiz::Tensor` and the built-in layers |

Namespaces: `cyxwiz::plugin` for the plugin API, `cyxwiz` for `Module`,
`Tensor` and `ExtensionShapeResult`.

## 2. Entry point

```cpp
CYXWIZ_PLUGIN_ENTRY(my_nodes::MyNodesPlugin)
```

Write it once, after your plugin class, outside any namespace. It exports
three C functions:

| Function | Called |
|---|---|
| `const char* CyxwizPluginBuildFingerprint()` | First, before any other plugin code. A missing or different fingerprint refuses the load (developer guide, *Build fingerprint*) |
| `IPlugin* CreatePlugin()` | Once, to create your plugin (`new PluginClass()`) |
| `void DestroyPlugin(IPlugin*)` | Once, after `OnUnload` |

`CYXWIZ_PLUGIN_EXPORT` (`extern "C"` with dllexport or default visibility)
is available for other exports. The function types are `CreatePluginFunc`,
`DestroyPluginFunc` and `BuildFingerprintFunc`.

## 3. `plugin.json`

The manifest sits in the plugin folder; the Engine reads it before it loads
the library.

| Field | Type | Meaning |
|---|---|---|
| `id` | string | Unique id: `a-z`, `0-9`, `.`, `-`, `_`. Part of every node's saved type id (`<id>:<TypeName>`), so do not change it once graphs use it |
| `name` | string | Shown in the Plugin Manager (default: `id`) |
| `version` | `"major.minor.patch"` | Your plugin's version |
| `api_version` | `"major.minor.patch"` | The plugin API you built against. Loads when the major equals the Engine's and the minor is not higher (1.2 today) |
| `description`, `author`, `license`, `homepage`, `repository` | string | Shown in the Plugin Manager |
| `capabilities` | array of strings | What the plugin provides (table below) |
| `permissions` | array of strings | What it may do; the user approves them on first load (table below) |
| `dependencies` | array of `{"id", "version"}` | Plugins that must load first; `version` is a constraint such as `">=1.0.0"` |
| `platforms` | `{"windows": {"library": ...}, "linux": ..., "macos": ...}` | The library path for each platform, relative to the folder |
| `library` | string | Used when `platforms` has no entry for this platform |
| `signature` | hex string | Ed25519 signature of the library; an unsigned plugin loads with a warning |

| Capability | Meaning |
|---|---|
| `ProvidesTensorNodes` | Trainable nodes (`ITensorNodeProvider`, API 1.1) |
| `ProvidesNodes` | Signal nodes (`INodeProvider`) |
| `ProvidesPanels` | ImGui panels (`IPanelProvider`) |
| `ProvidesTraining` | Training hooks (`ITrainingHook`) |
| `ProvidesData`, `ProvidesAnalytics` | Data loaders, analytics (see section 9) |
| `RequiresPython`, `RequiresGPU`, `RequiresNetwork` | Requirements shown to the user |

| Permission | Allows |
|---|---|
| `Training` | Taking part in training: trainable nodes, training hooks |
| `UIModify` | Adding nodes and panels |
| `DataRegistry` | Data loaders and analytics |
| `FileSystem`, `Network`, `SystemCommands`, `Python`, `GPU` | Declared for the user's approval |

Permissions are a declaration the user approves; the Engine does not
sandbox native code (developer guide, *Threads, crashes and limits*).

## 4. `IPlugin`

| Member | You return or do |
|---|---|
| `bool OnLoad(PluginContext&)` | First call after `CreatePlugin`. Prepare; `false` fails the load |
| `bool OnInitialize(PluginContext&)` | After `OnLoad`. After it returns `true` the Engine asks `QueryInterface` for each interface in section 6 and registers what you return |
| `void OnShutdown(PluginContext&)` | Before unload, or when the user disables the plugin |
| `void OnUnload(PluginContext&)` | Last call; your nodes and panels are already removed |
| `const PluginManifest& GetManifest() const` | A manifest. The Engine uses `plugin.json`; it reads this only when a library is loaded without one |
| `PluginPermissionFlags GetRequiredPermissions() const` | The permissions your code needs, for example `static_cast<uint32_t>(PluginPermission::Training)` |
| `PluginState GetState() const` | Your state (`Unloaded`, `Loaded`, `Initialized`, `Active`, `Failed`, `Disabled`) |
| `void* QueryInterface(const char* name)` | A pointer to the interface called `name`, cast from your class, or `nullptr` |
| `void SetImGuiContext(void* ctx)` | Called before `OnLoad`. A plugin that draws calls `ImGui::SetCurrentContext(static_cast<ImGuiContext*>(ctx))` |

Lifecycle and registration calls run inside the Engine's crash guard: an
exception or a crash there marks the plugin *Failed* instead of closing the
Engine. During training, an exception from `Forward` or `Backward` stops the
run with your node named; a memory fault is not caught.

## 5. `PluginContext`

Passed to each lifecycle call; implemented by the Engine. Keep the reference
only until `OnUnload` returns.

| Member | Does |
|---|---|
| `const std::string& GetPluginId() const` | Your manifest id |
| `const std::filesystem::path& GetPluginDir() const` | Your plugin folder |
| `void LogInfo / LogWarn / LogError / LogDebug(const std::string&)` | Writes to the Engine's log, prefixed with your id |
| `bool RegisterNodeProvider(INodeProvider*)` | Registers signal nodes now. Needs `UIModify` |
| `bool RegisterPanelProvider(IPanelProvider*)` | Registers panels now. Needs `UIModify` |
| `bool RegisterDataProvider(IDataProvider*)` | Needs `DataRegistry` |
| `bool RegisterTrainingHook(ITrainingHook*)` | Needs `Training` |
| `bool RegisterAnalyticsProvider(IAnalyticsProvider*)` | Needs `DataRegistry` |
| `AssistantContextSnapshot GetAssistantContextSnapshot() const` | What the user is looking at (workspace, graph, selection); for the assistant plugin |

Each `Register*` returns `false` when the permission was not granted. Use
either `QueryInterface` (section 6) or `Register*` for an interface, not
both: both would register it twice.

## 6. Interfaces found through `QueryInterface`

After `OnInitialize` the Engine asks for each name and registers what you
return, inside the crash guard. Unloading removes all of it.

| Name | Interface | What the Engine does with it today |
|---|---|---|
| `"ITensorNodeProvider"` | `ITensorNodeProvider` | Lists, compiles and trains your nodes (section 7) |
| `"INodeProvider"` | `INodeProvider` | Lists your signal nodes; calls `EvaluateNode` in live graph execution, and `GenerateCode` in code export, where the text is written as a comment (a simulation node is not part of a trained model) |
| `"IPanelProvider"` | `IPanelProvider` | Adds your panels to the menus and draws them every frame they are open |
| `"ITrainingHook"` | `ITrainingHook` | Calls it at training start and end, each epoch's start and end, and asks `ShouldStopEarly` (section 8) |
| `"IDataProvider"` | `IDataProvider` | Registers the loaders; the Engine does not load datasets through them yet |
| `"IAnalyticsProvider"` | `IAnalyticsProvider` | Registers the analytics; nothing in the Engine runs them yet |
| `"IAssistantProvider"` | `IAssistantProvider` | Runs assistant commands (the CyxWiz assistant plugin) |

```cpp
void* QueryInterface(const char* name) override {
    if (std::string(name) == "ITensorNodeProvider") return static_cast<ITensorNodeProvider*>(this);
    return nullptr;
}
```

## 7. Trainable nodes: `ITensorNodeProvider`

| Member | Contract |
|---|---|
| `std::vector<TensorNodeInfo> GetTensorNodes()` | Your nodes. Called once at initialisation |
| `ExtensionShapeResult InferOutputShape(type_name, input_shape, parameters)` | The output shape for one sample (no batch dimension), or `{false, {}, "reason"}` to refuse the settings. Pure and cheap: called after every graph edit, from any thread |
| `std::unique_ptr<cyxwiz::Module> CreateModule(type_name, input_shape, parameters, error)` | A new module for one sample's input shape; `nullptr` with `error` set when it cannot be made. Any thread |
| `void EnumerateTensorNodes(const TensorNodeCallback&)` | Do not override: it hands the Engine your nodes as C strings |

`parameters` maps each setting's name to its text value (`"0.8"`, `"true"`).

### `TensorNodeInfo`

| Field | Meaning |
|---|---|
| `type_name` | Unique in the plugin; letters, digits, `_`. Saved in graphs as `<plugin id>:<type_name>` |
| `display_name` | Shown on the node (default: `type_name`) |
| `category` | Your group in the add-node menu and the Node Browser (`"Activations"`) |
| `description` | One line, shown in tooltips and Properties |
| `version` | The version of this node's pins and settings, saved with each node |
| `color` | Header colour, `IM_COL32` layout (`0xAABBGGRR`); `0` uses the theme's plugin colour |
| `icon` | Icon font codepoint; empty uses the plug icon |
| `is_activation` | `true` places it with the activations |
| `pins` | `{name, is_input}`; exactly one input and one output in API 1.2 |
| `parameters` | `{name, type, default_value, description, choices}`; `type` is `"int"`, `"float"`, `"bool"`, `"string"` or `"enum"` (with `choices`) |

### `ExtensionShapeResult`

| Field | Meaning |
|---|---|
| `bool ok` | `true` when the settings and input are accepted |
| `std::vector<size_t> output_shape` | Per sample, without the batch dimension |
| `std::string error` | Shown on the node and in the compile errors when `ok` is `false` |

### The module: `cyxwiz::Module`

| Member | Override when |
|---|---|
| `Tensor Forward(const Tensor& input)` | Always. `input` holds the whole batch; dimension 0 is the batch |
| `Tensor Backward(const Tensor& grad_output)` | Always. Returns the gradient with the shape of the last `Forward` input; store gradients for your weights |
| `std::string GetName() const` | Always |
| `std::map<std::string, Tensor> GetParameters()`, `SetParameters(...)`, `GetGradients()` | The module has weights. The three use the same keys |
| `bool HasParameters() const` | The module has weights (return `true`) |
| `IncrementalDecoding GetIncrementalDecoding() const` | Return `PositionWise` when each output position depends only on the same input position (activations, per-position layers) |
| `void SetTraining(bool)` | Behaviour differs between training and evaluation |

`input_cache_` (protected) is a convenient place to keep the input for
`Backward`. The Engine wraps each module in a guard that checks the shapes
it returns and names your node in any error.

## 8. Training hooks: `ITrainingHook`

All members have empty defaults; override what you need. The
`TrainingContext` holds `current_epoch`, `total_epochs`, `current_batch`,
`total_batches`, `train_loss`, `train_accuracy`, `val_loss`,
`val_accuracy`, `learning_rate` and `custom_metrics`.

| Member | Called today |
|---|---|
| `OnTrainingStart`, `OnTrainingEnd` | Yes |
| `OnEpochStart`, `OnEpochEnd` | Yes |
| `bool ShouldStopEarly(const TrainingContext&)` | Yes, after each epoch; `true` stops training |
| `OnBatchStart`, `OnBatchEnd` | Not called yet |

## 9. Other interfaces

### `INodeProvider` (signal nodes)

Nodes that pass values (scalars, vectors, strings) in live graph execution,
not trainable tensors.

| Member | Contract |
|---|---|
| `std::vector<PluginNodeTypeInfo> GetNodeTypes()` | Your nodes: `type_name`, `display_name`, `category`, `description`, `color`, `icon`, `pins` (`{name, type, is_input}`), `default_parameters`, `supports_dynamic_pins`, `dynamic_pin_trigger` |
| `std::string GenerateCode(type_name, parameters, framework)` | Text for `"pytorch"`, `"tensorflow"`, `"keras"` or `"pycyxwiz"` code export, written into the script as a comment beside the model |
| `DynamicPinResult ResolveDynamicPins(type_name, parameters)` | New pins when the `dynamic_pin_trigger` setting changes (default: none) |
| `PluginNodeEvalResult EvaluateNode(const PluginNodeEvalContext&)` | One tick of live execution: `input_values` by pin name in, `output_values` out (default: not supported) |

### `IPanelProvider`

| Member | Contract |
|---|---|
| `std::vector<PluginPanelInfo> GetPanels()` | `panel_id`, `title`, `category` (menu path), `icon`, `show_by_default` |
| `void RenderPanel(const std::string& panel_id, bool* visible)` | Draws the panel's contents every frame it is open; set `*visible = false` to close it |

Panels need `SetImGuiContext` (section 4) and the plugin must link ImGui
itself.

### `IDataProvider`, `IAnalyticsProvider`

Declared and registered, but not used by the Engine yet (section 6).
`IDataProvider` has `GetLoaders()`, `CanLoad(path)` and
`LoadDataset(path, name)` returning a `PluginDataset`;
`IAnalyticsProvider` has `GetAnalytics()` and `Compute(id, dataset)`
returning an `AnalyticsResult`.

## 10. Small types

| Type | Members |
|---|---|
| `PluginVersion` | `major`, `minor`, `patch`; `Parse("1.2.0")`, `ToString()`, `IsCompatibleWith(engine_api)`, comparisons |
| `PLUGIN_API_VERSION_MAJOR`, `_MINOR`, `_PATCH` | 1, 2, 0 |
| `PluginPermission`, `PluginPermissionFlags` | The permission bits; `HasPermission(flags, p)`, `ParsePermissions`, `PermissionsToStrings` |
| `PluginCapability`, `PluginCapabilityFlags` | The capability bits |
| `PluginState` | `Unloaded`, `Loaded`, `Initialized`, `Active`, `Failed`, `Disabled`; `PluginStateToString` |
| `PluginDependency` | `plugin_id`, `version_constraint` |
| `PluginManifest` | The `plugin.json` fields of section 3, plus `plugin_dir` (set by the Engine) and `IsValid()` |

## 11. CMake

| Name | What it is |
|---|---|
| `find_package(CyxWizSDK)` | Finds the SDK; configure with `-DCMAKE_PREFIX_PATH=<SDK folder>`. A version request matches the same major and minor (`0.2`) |
| `CyxWiz::PluginSDK` | The one target to link: headers, the backend's import library, C++20 |
| `CyxWiz::Backend` | The backend's import library alone |
| `cyxwiz_plugin_folder(<target> [FOLDER <dir>])` | Puts the library in `<dir>/bin` and copies `plugin.json` to `<dir>` (default `<build>/<target>`). With MSVC every configuration uses the release C++ runtime, so Debug builds load in the Engine |
| `CyxWizSDK_VERSION` | The SDK version (0.2.0) |
| `CyxWizSDK_PLUGIN_API_VERSION` | The plugin API version (1.2) |
| `CyxWizSDK_TEMPLATE_DIR` | The template plugin's folder |

The SDK is in every installation (`runtime/base/<base>/sdk`, see the
quickstart) and is licensed under the Apache License 2.0 (`sdk/LICENSE`,
`sdk/NOTICE`); every SDK header starts with
`SPDX-License-Identifier: Apache-2.0`.

Inside the CyxWiz source tree `CyxWiz::PluginSDK` and
`cyxwiz_plugin_folder` exist too, so one `CMakeLists.txt` builds a plugin in
the tree and against the SDK (the template and
`plugins/examples/activation_nodes` do).

## 12. `cyxwiz-plugin-check`

```
cyxwiz-runtime-bootstrapper --plugin-check <plugin folder> [<plugin folder> ...]
```

Ships with the Engine (`runtime/base/<base>/cyxwiz-plugin-check`); start it
through the launcher in the installation folder, as above, which gives it the
Engine's runtime libraries and keeps your current folder for relative paths.
In a source build run `bin/Release/cyxwiz-plugin-check` directly. For each folder it reads
`plugin.json`, loads the library (fingerprint first), initialises the plugin,
lists the nodes it registers and runs the gradient check on each trainable
node with its default settings. Exit code 0 when every folder loads and
every node passes, 1 otherwise, 2 for no arguments.
