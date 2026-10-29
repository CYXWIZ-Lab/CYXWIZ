# Plugin API reference (C++)

Every type a C++ plugin sees, what the Engine does with it, and what works
today. For a walk-through read the [quickstart](plugin-quickstart.md) and the
[developer guide](plugin-developer-guide.md); for the rules a trainable
module must keep, the [node contract](node-contract.md).

> Checked against plugin API 1.3.0 and pycyxwiz 0.2.0.
> The CyxWiz SDK is version 0.2.

## 1. Headers

Include the plugin headers as `"plugin/..."` and the backend headers as
`<cyxwiz/...>`. They need only the C++ standard library. Every header the
SDK installs, with the types it declares, is listed in
[sdk-contents.md](sdk-contents.md); the tensor operations in
[tensor-operations.md](tensor-operations.md).

| Header | Holds |
|---|---|
| `plugin/plugin_types.h` | `IPlugin`, `PluginManifest`, versions, permissions, capabilities, states, `CYXWIZ_PLUGIN_ENTRY` |
| `plugin/plugin_context.h` | `PluginContext`, `AssistantContextSnapshot` |
| `plugin/plugin_fingerprint.h` | `CurrentBuildFingerprint()` (used by `CYXWIZ_PLUGIN_ENTRY`) |
| `plugin/interfaces/i_tensor_node_provider.h` | `ITensorNodeProvider`, `TensorNodeInfo` (trainable nodes) |
| `plugin/interfaces/i_node_provider.h` | `INodeProvider`, `PluginNodeTypeInfo` (signal nodes) |
| `plugin/interfaces/i_panel_provider.h` | `IPanelProvider`, `PluginPanelInfo` |
| `plugin/interfaces/i_training_hook.h` | `ITrainingHook`, `TrainingContext` |
| `plugin/interfaces/arrow_c_stream.h` | `ArrowSchema`, `ArrowArray`, `ArrowArrayStream` (the Arrow C data and C stream interfaces) |
| `plugin/interfaces/i_data_provider.h` | `IDataProvider`, `PluginDataLoaderInfo` |
| `plugin/interfaces/i_analytics_provider.h` | `IAnalyticsProvider`, `PluginAnalyticsInfo`, `AnalyticsColumns`, `AnalyticsResult` |
| `plugin/interfaces/i_assistant_provider.h` | `IAssistantProvider` (used by the CyxWiz assistant plugin) |
| `core/extension_node_contract.h`, `core/node_metadata.h`, `core/graph_model.h` | `ExtensionShapeResult` and the types it builds on |
| `<cyxwiz/sequential.h>` | `cyxwiz::Module`, `cyxwiz::Tensor` and the built-in layers |

Namespaces: `cyxwiz::plugin` for the plugin API, `cyxwiz` for `Module`,
`Tensor` and `ExtensionShapeResult`.

The SDK has no ArrayFire, ImGui, spdlog or JSON headers. Members whose
signature uses `af::` types are declared in `cyxwiz/tensor.h` but cannot be
called from a plugin built against the SDK.

## 2. Entry point

<!-- excerpt: plugins/template/my_nodes_plugin.cpp -->
```cpp
CYXWIZ_PLUGIN_ENTRY(my_nodes::MyNodesPlugin)
```

Write it once, after your plugin class, outside any namespace, with the
class's full name. It exports three C functions:

| Function | Called |
|---|---|
| `const char* CyxwizPluginBuildFingerprint()` | Right after the library is loaded, before `CreatePlugin`. A missing or different fingerprint refuses the load and unloads the library ([developer guide](plugin-developer-guide.md#10-build-fingerprint)). Loading the library has already run its global constructors (and `DllMain` on Windows) |
| `IPlugin* CreatePlugin()` | Once, to create your plugin (`new PluginClass()`) |
| `void DestroyPlugin(IPlugin*)` | Once, after `OnUnload`, before the library is freed |

`CYXWIZ_PLUGIN_EXPORT` (`extern "C"` with `__declspec(dllexport)` or default
visibility) is available for other exports. The function types are
`CreatePluginFunc`, `DestroyPluginFunc` and `BuildFingerprintFunc`.

The load order is: read `plugin.json`, check `api_version`, find the library,
check `signature` when there is one, load the library, check the
fingerprint, `CreatePlugin`, then (when the plugin is initialised)
`SetImGuiContext`, `OnLoad`, `OnInitialize`, and the `QueryInterface`
calls of section 6.

## 3. `plugin.json`

The manifest sits in the plugin folder; the Engine reads it before it loads
the library. Unknown fields, capabilities and permissions are ignored.

| Field | Type | Meaning |
|---|---|---|
| `id` | string | Required. `a-z`, `0-9`, `.`, `-`, `_`; a plugin whose id has other characters loads, but all its nodes are refused. Part of every node's saved type id (`<id>:<TypeName>`), so do not change it once graphs use it. A plugin whose id is already loaded is not loaded again: **Refresh** skips it, **Install...** says it is already loaded |
| `name` | string | Shown in the Plugin Manager (default: `id`) |
| `version` | `"major.minor.patch"` | Your plugin's version (default `0.0.0`). Permission answers are stored per version |
| `api_version` | `"major.minor.patch"` | The plugin API you built against (default `1.0.0`). Loads when the major equals the Engine's and the minor is not higher (1.3 today) |
| `description`, `author`, `license`, `homepage`, `repository` | string | Stored; the Plugin Manager card shows `description` and `author` |
| `capabilities` | array of strings | What the plugin says it provides (table below). Informational: the Engine finds interfaces through `QueryInterface` whatever this says |
| `permissions` | array of strings | What it may do (table below) |
| `dependencies` | array of `{"id", "version"}` | Plugins that must run before yours. `version`: empty or `*` (any), `1.2.0` (exactly), `>=`, `>`, `<=`, `<`, `^1.2.0` (at least, same major; same minor below 1.0), `~1.2.0` (at least, same major.minor); several separated by spaces must all hold (`>=1.0 <2.0`). Your plugin does not start when a dependency is not installed, has another version or failed; the reason shows in the Install dialog, Plugin Manager and the log ("needs com.acme.base >=2.0.0; version 1.4.0 is installed"). When a dependency has not started yet (it waits for its permissions), yours waits and starts right after it |
| `platforms` | `{"windows": {"library": ...}, "linux": ..., "macos": ...}` | The library path for each platform, relative to the folder |
| `library` | string | Used when `platforms` has no entry for this platform. Without either, the first `.dll` (`.so`, `.dylib`) directly in the folder is used |
| `signature` | hex string | Leave it out. A value is checked as an Ed25519 signature, by CyxWiz's own key, of the library's SHA-512 hash; a value that does not verify refuses the load (*Invalid signature: ...*). Without it the plugin loads, with *unsigned* in the log |

| Capability | Meaning |
|---|---|
| `ProvidesTensorNodes` | Trainable nodes (`ITensorNodeProvider`, API 1.1) |
| `ProvidesNodes` | Signal nodes (`INodeProvider`) |
| `ProvidesPanels` | ImGui panels (`IPanelProvider`) |
| `ProvidesTraining` | Training hooks (`ITrainingHook`) |
| `ProvidesData`, `ProvidesAnalytics` | Data loaders, analytics (section 9) |
| `RequiresPython` | A plugin with this capability and no library file loads as Python-only |
| `RequiresGPU`, `RequiresNetwork` | Stored only |

| Permission | Granted | Checked by |
|---|---|---|
| `Training` | Without asking | `ITensorNodeProvider` (trainable nodes), `ITrainingHook`, `RegisterTrainingHook` |
| `UIModify` | Without asking | `INodeProvider`, `IPanelProvider`, `RegisterNodeProvider`, `RegisterPanelProvider` |
| `DataRegistry` | Without asking | `IDataProvider`, `IAnalyticsProvider`, `RegisterDataProvider`, `RegisterAnalyticsProvider` |
| `GPU` | Without asking | Nothing |
| `FileSystem`, `Network`, `SystemCommands`, `Python` | The user is asked to allow or deny each before the plugin starts; the answer is stored per plugin id and version (`%APPDATA%\cyxwiz\plugin_permissions.json`, `~/.cyxwiz/plugin_permissions.json`), so a new version asks again. A denied permission does not stop the plugin | Nothing in the Engine today |

The interfaces your `QueryInterface` returns and the `Register*` calls check
the flags `GetRequiredPermissions()` returns, not `plugin.json`: declare a
permission in both. Without it that interface is not registered, and the
Install dialog, Plugin Manager and `cyxwiz-plugin-check` say so
("ITrainingHook: GetRequiredPermissions() does not include the Training
permission"). Offering one provider both ways (through `QueryInterface` and a
`Register*` call) registers it once. The
approval dialog asks about the permissions in `plugin.json`. Permissions are
a statement to the user: the Engine does not sandbox native code.
`cyxwiz-plugin-check` grants the four asked permissions for its own run
only (section 12).

## 4. `IPlugin`

All members are pure virtual except `QueryInterface` and `SetImGuiContext`.

| Member | You return or do |
|---|---|
| `bool OnLoad(PluginContext&)` | First lifecycle call. Prepare; `false` fails the plugin (*OnLoad() returned false*) |
| `bool OnInitialize(PluginContext&)` | After `OnLoad`. After it returns `true` the Engine asks `QueryInterface` for each interface of section 6 and registers what you return |
| `void OnShutdown(PluginContext&)` | When the plugin is disabled or unloaded, and when the Engine closes |
| `void OnUnload(PluginContext&)` | Last call before `DestroyPlugin`; your nodes and panels are already removed |
| `const PluginManifest& GetManifest() const` | Any manifest; an empty `PluginManifest` member is enough. The Engine uses `plugin.json` |
| `PluginPermissionFlags GetRequiredPermissions() const` | The permissions your `Register*` calls need, for example `static_cast<uint32_t>(PluginPermission::Training)` |
| `PluginState GetState() const` | Your own state. The Engine keeps its own and does not call this |
| `void* QueryInterface(const char* name)` | A pointer to the interface called `name`, cast from your class (`static_cast<ITensorNodeProvider*>(this)`), or `nullptr` |
| `void SetImGuiContext(void* ctx)` | Called before `OnLoad`. A plugin that draws calls `ImGui::SetCurrentContext(static_cast<ImGuiContext*>(ctx))` (section 9) |

Lifecycle and registration calls run inside the Engine's crash guard: an
exception or a crash there marks the plugin *Failed* instead of closing the
Engine. During training, an exception from `Forward` or `Backward` stops the
run with your node named; a memory fault there is not caught.

## 5. `PluginContext`

Passed to each lifecycle call; implemented by the Engine. Keep the reference
only until `OnUnload` returns.

| Member | Does |
|---|---|
| `const std::string& GetPluginId() const` | Your manifest id |
| `const std::filesystem::path& GetPluginDir() const` | Your plugin folder (the one with `plugin.json`) |
| `void LogInfo / LogWarn / LogError / LogDebug(const std::string&)` | Writes to the Engine's Console and `engine_log.txt`, prefixed with `[Plugin:<id>]` |
| `bool RegisterNodeProvider(INodeProvider*)` | Registers signal nodes now. Needs `UIModify` |
| `bool RegisterPanelProvider(IPanelProvider*)` | Registers panels now. Needs `UIModify` |
| `bool RegisterDataProvider(IDataProvider*)` | Needs `DataRegistry` |
| `bool RegisterTrainingHook(ITrainingHook*)` | Needs `Training` |
| `bool RegisterAnalyticsProvider(IAnalyticsProvider*)` | Needs `DataRegistry` |
| `AssistantContextSnapshot GetAssistantContextSnapshot() const` | What the user is looking at (workspace, graph, selection); for the assistant plugin |

Each `Register*` returns `false` when the permission is missing (section 3)
and logs why. Use either `QueryInterface` (section 6) or `Register*` for an
interface, not both: both would register it twice.

## 6. Interfaces found through `QueryInterface`

After `OnInitialize` the Engine asks for each name and registers what you
return, inside the crash guard. Unloading removes all of it.

| Name | Interface | What the Engine does with it today |
|---|---|---|
| `"ITensorNodeProvider"` | `ITensorNodeProvider` | Lists, compiles and trains your nodes (section 7) |
| `"INodeProvider"` | `INodeProvider` | Lists your signal nodes; calls `EvaluateNode` in live graph execution, and `GenerateCode` in code export, where the text is written as a comment (a simulation node is not part of a trained model) |
| `"IPanelProvider"` | `IPanelProvider` | Adds your panels to the menus and draws them every frame they are open |
| `"ITrainingHook"` | `ITrainingHook` | Calls it at training start and end, each epoch's and each training batch's start and end, and asks `ShouldStopEarly` (section 8) |
| `"IDataProvider"` | `IDataProvider` | Registers the dataset formats (plugin API 1.3 or newer): Data Input lists and opens them |
| `"IAnalyticsProvider"` | `IAnalyticsProvider` | Registers the analytics (plugin API 1.3 or newer); Data Studio does not run them yet |
| `"IAssistantProvider"` | `IAssistantProvider` | Runs assistant commands (the CyxWiz assistant plugin) |

<!-- excerpt: plugins/template/my_nodes_plugin.cpp -->
```cpp
void* QueryInterface(const char* name) override {
    if (std::string(name) == "ITensorNodeProvider") return static_cast<ITensorNodeProvider*>(this);
    return nullptr;
}
```

## 7. Trainable nodes: `ITensorNodeProvider`

<!-- excerpt: cyxwiz-engine/src/plugin/interfaces/i_tensor_node_provider.h -->
```cpp
class ITensorNodeProvider {
public:
    virtual ~ITensorNodeProvider() = default;

    virtual std::vector<TensorNodeInfo> GetTensorNodes() = 0;

    // Output shape per sample (without the batch dimension). Pure.
    virtual ExtensionShapeResult InferOutputShape(
        const std::string& type_name,
        const std::vector<size_t>& input_shape,
        const std::map<std::string, std::string>& parameters) = 0;

    // input_shape is one sample's shape (no batch dimension), as for
    // InferOutputShape. nullptr with `error` set when the module cannot be
    // created.
    virtual std::unique_ptr<cyxwiz::Module> CreateModule(
        const std::string& type_name,
        const std::vector<size_t>& input_shape,
        const std::map<std::string, std::string>& parameters,
        std::string& error) = 0;
    // ...
    virtual void EnumerateTensorNodes(const TensorNodeCallback& cb) {
```

| Member | Contract |
|---|---|
| `GetTensorNodes()` | Your nodes. Called once, when the plugin initialises |
| `InferOutputShape(type_name, input_shape, parameters)` | The output shape for one sample (no batch dimension), or `{false, {}, "reason"}` to refuse the settings. Pure and cheap: called after every graph edit, from any thread, possibly at the same time |
| `CreateModule(type_name, input_shape, parameters, error)` | A new module for one sample's input shape; `nullptr` with `error` set when it cannot be made. Any thread |
| `EnumerateTensorNodes(const TensorNodeCallback&)` | Do not override: it hands the Engine your nodes as C strings |

`parameters` maps each setting's name to its text value (`"0.8"`, `"true"`);
a setting can be missing or empty.

### `TensorNodeInfo`

| Field | Meaning |
|---|---|
| `type_name` | Unique in the plugin; letters, digits, `_`. Saved in graphs as `<plugin id>:<type_name>` |
| `display_name` | Shown on the node (empty: `type_name`) |
| `category` | Your group in the add-node menu and the Node Browser (`"Activations"`) |
| `description` | One line, shown in tooltips and Properties |
| `version` | The version of this node's pins and settings, saved with each node |
| `color` | Header colour, `IM_COL32` layout (`0xAABBGGRR`); a colour with alpha 0 (such as `0`) uses the theme's plugin colour |
| `icon` | Icon font codepoint; empty uses the plug icon |
| `is_activation` | `true` places it with the activations |
| `pins` | `{name, is_input}`; exactly one input and one output in API 1.2 |
| `parameters` | `{name, type, default_value, description, choices}`; `type` is `"int"`, `"float"`, `"bool"`, `"string"` or `"enum"` (with `choices`). Choices are passed joined by commas: a choice containing a comma is split, an empty one dropped |

A node is refused, and the rest of the plugin loads, when it does not have
exactly one input and one output pin, when `type_name` or the plugin id
breaks its character rule, or when its type id is already registered. The
reasons are listed in the
[developer guide](plugin-developer-guide.md#nodes-the-engine-refuses); the
Plugin Manager's **Install...** and `cyxwiz-plugin-check` show them.

### `ExtensionShapeResult`

| Field | Meaning |
|---|---|
| `bool ok` | `true` when the settings and input are accepted |
| `std::vector<size_t> output_shape` | Per sample, without the batch dimension |
| `std::string error` | Shown on the node and in the compile errors when `ok` is `false` |

### The module: `cyxwiz::Module`

| Member | Override when |
|---|---|
| `Tensor Forward(const Tensor& input)` | Always (pure virtual). `input` holds the whole batch; dimension 0 is the batch |
| `Tensor Backward(const Tensor& grad_output)` | Always (pure virtual). Returns the gradient with the shape of the last `Forward` input; store gradients for your weights |
| `std::string GetName() const` | Always (pure virtual) |
| `std::map<std::string, Tensor> GetParameters()`, `SetParameters(...)`, `GetGradients()` | The module has weights. The three use the same keys; `SetParameters` may get only some keys |
| `bool HasParameters() const` | The module has weights (return `true`) |
| `IncrementalDecoding GetIncrementalDecoding() const` | Return `PositionWise` when each output position depends only on the same input position (activations, per-position layers) |
| `void SetTraining(bool)` | Behaviour differs between training and evaluation; keep `is_training_` set |

`input_cache_` (protected) is a convenient place to keep the input for
`Backward`. The Engine wraps each module in a guard that checks the shapes
it returns and names your node in any error.

Tensors reaching a module are `float32` and row-major. `ReadData<T>()`
copies device values to host memory (slow; `T` is not checked against the
tensor's type). The operations are listed in
[tensor-operations.md](tensor-operations.md).

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
| `OnBatchStart`, `OnBatchEnd` | Yes, around every training batch (not validation): `current_batch` counts 1 to `total_batches` in each epoch; `OnBatchEnd` has the batch's `train_loss` and the running `train_accuracy`. They run on the training thread for every batch, so keep them fast |

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

A panel draws with ImGui inside the Engine's ImGui context
(`SetImGuiContext`, section 4), so the plugin must be compiled against
exactly the ImGui version and configuration the Engine uses. The SDK has no
ImGui: panels can only be built inside the CyxWiz source tree.

### `IDataProvider`, `IAnalyticsProvider`

A dataset format and a dataset analytic. Data crosses in both directions as
the Arrow C stream interface (`arrow_c_stream.h`, the structs of the Apache
Arrow specification): a stream of record batches with one schema. The SDK
ships no Arrow library and needs none. Fill or read the structs by hand, or
with any Arrow library of any version (Arrow C++ `arrow::ExportRecordBatchReader`
/ `ImportRecordBatchReader`, nanoarrow, arrow-rs).
`plugins/examples/data_formats` does both by hand, with no Arrow library.

Both interfaces took this shape in plugin API 1.3. The Engine calls them
only in a plugin whose `plugin.json` declares `api_version` 1.3 or newer. An
older plugin loads and its other interfaces work, but the Plugin Manager and
the check say *IDataProvider: built for plugin API 1.2.0; it needs 1.3.0 or
newer (rebuild the plugin)*.

`IDataProvider`:

| Method | Called | Contract |
|---|---|---|
| `std::vector<PluginDataLoaderInfo> GetLoaders()` | Once, when the plugin starts | One entry per format: `loader_id` (unique across plugins), `format_name` (shown to the user), `extensions` (any case, with or without the dot), `description`. An entry without an id, a name or an extension is skipped; an id another plugin already registered is skipped with a warning |
| `bool CanLoad(loader_id, file_path)` | When a file's extension matches the loader | Whether this loader reads the file. It may read the file's beginning. Return false for a file of another format with the same extension |
| `bool OpenDataset(loader_id, file_path, ArrowArrayStream* out, std::string& error)` | On a worker thread | On success fill `*out` and return true. The Engine reads the stream maybe after `OpenDataset` returned, copies each batch into its own memory, and calls `out->release` once. On failure return false with `error` (the user sees *&lt;format_name&gt; could not open &lt;file&gt;: &lt;error&gt;*) and leave `*out` untouched. An error while streaming: return non-zero from `get_next` and the message from `get_last_error` |

Give numbers a numeric type (`g` float64, `f` float32, `l` int64, `i`
int32) and text `u` (UTF-8): training and the Explore tabs work on numeric
and text columns. Return batches of a bounded size (the example uses
50,000 rows), so a file larger than memory streams.

Data Input lists each loader in its **Format** list as *&lt;format_name&gt;
(plugin)* and in the file types of **Browse...**; a Data Input stores it as
`file_type` `plugin:<loader_id>` (lower case). Detection by extension: a
format the Engine reads itself wins, then a loader whose `CanLoad` accepts
the file. On Apply the Engine streams the file once into a Parquet copy in
the project's cache and reads that copy until the file is newer or your
plugin's `version` (plugin.json) changes: raise the version when your reader
returns different data for the same file.

`IAnalyticsProvider`:

| Member | Contract |
|---|---|
| `std::vector<PluginAnalyticsInfo> GetAnalytics()` | One entry per analytic: `analytics_id` (unique across plugins), `name`, `description`, `category` (groups analytics, e.g. *Distribution*), `columns` (`AnalyticsColumns::Any`, `Numeric` or `Text`: the columns the user may pick), `min_columns` (at least 1), `max_columns` (0: no limit) |
| `AnalyticsResult Compute(analytics_id, ArrowArrayStream* data)` | On a worker thread. `data` streams the columns the user picked, every row. The Engine owns the stream and releases it after `Compute` returns: read it during the call, do not keep it |

`AnalyticsResult`: a number (`scalar_value` with `has_scalar`), labelled
values (`vector_value` with `vector_labels`, shown as bars), and `text_value`;
any combination. On failure set `success = false` and `error_message`. An
empty `name` or `category` is filled from the analytic's info.

Unloading a plugin waits until its streams are read and its analytics have
returned, then frees its library. Data already read stays readable.

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
| `find_package(CyxWizSDK)` | Finds the SDK; configure with `-DCMAKE_PREFIX_PATH=<SDK folder>`. A version request matches the same major and minor (`find_package(CyxWizSDK 0.2 REQUIRED)`) |
| `CyxWiz::PluginSDK` | The one target to link: the headers, the backend's import library, C++20 |
| `CyxWiz::Backend` | The backend's import library alone |
| `cyxwiz_plugin_folder(<target> [FOLDER <dir>])` | Puts the library in `<dir>/bin` for every configuration and copies `plugin.json` from the current source folder to `<dir>` (default `<build>/<target>`). With MSVC every configuration uses the release C++ runtime (`MultiThreadedDLL`), so Debug builds load. Needs policy CMP0091, which `cmake_minimum_required(VERSION 3.15)` or later turns on |
| `CyxWizSDK_VERSION` | The SDK version (0.2.0) |
| `CyxWizSDK_PLUGIN_API_VERSION` | The plugin API version (1.3) |
| `CyxWizSDK_TEMPLATE_DIR` | The template plugin's folder |

<!-- excerpt: cmake/sdk/CyxWizPluginHelpers.cmake -->
```cmake
function(cyxwiz_plugin_folder target)
    cmake_parse_arguments(PARSE_ARGV 1 _cyxwiz "" "FOLDER" "")
    if(NOT _cyxwiz_FOLDER)
        set(_cyxwiz_FOLDER "${CMAKE_BINARY_DIR}/${target}")
    endif()
    # ...
    if(MSVC)
        set_target_properties(${target} PROPERTIES MSVC_RUNTIME_LIBRARY "MultiThreadedDLL")
    endif()
```

`lib/cmake/CyxWizSDK/CyxWizSDKConfig.cmake` also records, as a comment,
the compiler the Engine was built with (`# Built with: MSVC 19.44... on
Windows`); build your plugin with that compiler
([quickstart](plugin-quickstart.md#the-compiler)).

The SDK is in every installation (`runtime/base/<base>/sdk`, see the
quickstart) and is licensed under the Apache License 2.0 (`sdk/LICENSE`,
`sdk/NOTICE`); every SDK header starts with
`SPDX-License-Identifier: Apache-2.0`.

Inside the CyxWiz source tree `CyxWiz::PluginSDK` and
`cyxwiz_plugin_folder` exist too, so one `CMakeLists.txt` builds a plugin in
the tree and against the SDK (the template and
`plugins/examples/activation_nodes` do, with `if(NOT TARGET
CyxWiz::PluginSDK)` around `find_package`). From a Release build of the
sources, `cmake --install <build folder> --config Release --component
plugin_sdk --prefix <folder>` installs the SDK.

## 12. `cyxwiz-plugin-check`

```
cyxwiz-plugin-check <plugin folder> [<plugin folder> ...]
```

The tool is `runtime/base/<base>/cyxwiz-plugin-check` (`.exe`). Start it
with `cyxwiz-cli plugin-check <plugin folder> ...` from the installation
folder, on every platform: `cyxwiz-cli` finds the active base, sets the
library path, keeps your current folder for relative paths, and returns the
tool's exit code ([quickstart](plugin-quickstart.md#3-check-it) has the
PowerShell, Command Prompt and shell forms, and the CyxWiz 1.0.7 commands,
which has no `cyxwiz-cli`). In a source build run
`bin/Release/cyxwiz-plugin-check` directly, with ArrayFire on `PATH`.

For each folder it:

1. reads `plugin.json` (*FAIL plugin.json: ...*);
2. loads the library the way the Engine does, API version, signature and
   fingerprint included (*FAIL the plugin did not load: &lt;reason&gt;*);
3. grants `FileSystem`, `Network`, `SystemCommands` and `Python`, when
   `plugin.json` declares them, for this run only, saving nothing
   (*permission X: granted for this check only (the Engine asks the user)*);
4. initialises the plugin (*FAIL the plugin did not initialise: ...*);
5. for each trainable node: calls `InferOutputShape` with the default
   settings for each sample shape the gradient check uses (`[3]` and
   `[4, 5]`), prints *node ...: refuses a [4, 5] input ...* for one it
   refuses, then runs the gradient check on the accepted shapes
   ([developer guide](plugin-developer-guide.md#8-the-gradient-check));
   prints *ok node ...* or *FAIL node ...* (*refuses every input shape the
   check uses* when it accepts neither);
6. lists the plugin's dataset formats and analytics (*ok dataset format
   ...*, *ok analytic ...*), prints *FAIL not added: &lt;reason&gt;* for each
   node, format or analytic the Engine refused, and *FAIL the plugin
   registered no nodes, dataset formats or analytics* when there is none;
7. unloads the plugin.

The last line is *cyxwiz-plugin-check: all plugins pass* or
*cyxwiz-plugin-check: FAILED*. Exit code 0 when every folder passes, 1
otherwise, 2 for no arguments. The Engine's log lines are printed too.
