# Plugin API reference (C++)

Every type a C++ plugin sees, what the Engine does with it, and what works
today. For a walk-through read the [quickstart](plugin-quickstart.md) and the
[developer guide](plugin-developer-guide.md); for the rules a trainable
module must keep, the [node contract](node-contract.md).

> Checked against plugin API 1.4.0 and pycyxwiz 0.2.0.
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
| `plugin/plugin_ui.h` | `PluginUi`, `UiColors`, `UiColor`, `UiButtonSize` (the Engine's widgets for panels, API 1.4) |
| `plugin/plugin_fingerprint.h` | `CurrentBuildFingerprint()` (used by `CYXWIZ_PLUGIN_ENTRY`) |
| `plugin/interfaces/i_tensor_node_provider.h` | `ITensorNodeProvider`, `TensorNodeInfo` (trainable nodes) |
| `plugin/interfaces/i_node_provider.h` | `INodeProvider`, `PluginNodeTypeInfo` (signal nodes) |
| `plugin/interfaces/i_panel_provider.h` | `IPanelProvider`, `PluginPanelInfo` |
| `plugin/interfaces/i_command_provider.h` | `ICommandProvider`, `PluginCommandInfo`, `PluginCommandState` (menu commands, API 1.4) |
| `plugin/interfaces/i_preferences_provider.h` | `IPreferencesProvider`, `PluginPreferencesSection` (settings in Preferences, API 1.4) |
| `plugin/interfaces/i_simulation_provider.h` | `ISimulationProvider`, `PluginSimulationNodeInfo`, `PluginSimulationGraphNode`, `PluginSimulationBeginResult` (graph simulation runs, API 1.4) |
| `plugin/interfaces/i_environment_provider.h` | `IEnvironmentProvider`, `IEnvironmentBatch`, `PluginEnvironmentInfo`, `PluginEnvironmentSetting`, `PluginSettingKind`, `PluginSettingOption`, `PluginEnvironmentStat`, `PluginEnvironmentStepBuffers`, `PluginEnvironmentCreateResult` (environments for reinforcement learning, API 1.4) |
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
| `api_version` | `"major.minor.patch"` | The plugin API you built against (default `1.0.0`). Loads when the major equals the Engine's and the minor is not higher (1.4 today) |
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
| `Training` | Without asking | `ITensorNodeProvider` (trainable nodes), `IEnvironmentProvider`, `ITrainingHook`, `RegisterTrainingHook` |
| `UIModify` | Without asking | `INodeProvider`, `IPanelProvider`, `ICommandProvider`, `IPreferencesProvider`, `ISimulationProvider`, `RegisterNodeProvider`, `RegisterPanelProvider` |
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
| `AssistantContextSnapshot GetAssistantContextSnapshot() const` | What the user is looking at (workspace, graph, selection); for the assistant plugin. The Engine fills it in when the user sends an assistant command, just before your `RunAssistantCommand`, so read it there |
| `PluginUi* GetUi()` | The Engine's buttons, fields and theme colours for your panels (API 1.4, section 9). Null only in hosts without windows (`cyxwiz-plugin-check`) |
| `bool ShowPanel(const std::string& panel_id)` | Opens one of your own panels and brings it to the front, for a button in one panel that leads to another (API 1.4). `false` when you have no panel with that id |
| `void RequestFullFrameRate()` | Keeps the Engine drawing at full frame rate for the next second (API 1.4). Without mouse or key input the Engine draws 10 frames a second, so call it every frame while something in your panel moves, such as a running simulation. Stop calling it and the Engine goes back to the idle rate |
| `std::filesystem::path GetProjectDir() const` | The open project's folder (API 1.4), which holds `exports/`, `models/` and the rest; empty when no project is open. Save what the user makes there, such as pictures and recordings in `exports/` |
| `std::string GetSetting(const std::string& key, const std::string& fallback = {}) const` | One of the user's settings for your plugin (API 1.4), kept by the Engine between sessions with its own settings; `fallback` when the key was never set. Values are text: read numbers and switches back from the text you wrote |
| `void SetSetting(const std::string& key, const std::string& value)` | Saves one setting at once (API 1.4). `cyxwiz-plugin-check` and other hosts without a settings store keep them for the session only |

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
| `"ICommandProvider"` | `ICommandProvider` | Adds your commands to the menu they name and to the Command Palette (plugin API 1.4, section 9) |
| `"IPreferencesProvider"` | `IPreferencesProvider` | Draws your settings at the end of the Preferences tab they name (plugin API 1.4, section 9) |
| `"ISimulationProvider"` | `ISimulationProvider` | Names your nodes that run in a graph simulation and starts and ends each run (plugin API 1.4, section 9) |
| `"IEnvironmentProvider"` | `IEnvironmentProvider` | Lists your environments; creates copies of one for reinforcement learning, resets and steps them (plugin API 1.4, section 9) |
| `"ITrainingHook"` | `ITrainingHook` | Calls it at training start and end, each epoch's and each training batch's start and end, and asks `ShouldStopEarly` (section 8) |
| `"IDataProvider"` | `IDataProvider` | Registers the dataset formats (plugin API 1.3 or newer): Data Input lists and opens them |
| `"IAnalyticsProvider"` | `IAnalyticsProvider` | Registers the analytics (plugin API 1.3 or newer); Data Studio's Explore > Plugins tab runs them |
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
| `std::vector<std::string> GetStateNames() const` | `GetParameters` also returns state the module keeps but does not train, such as BatchNorm's running statistics: name those keys. They need no gradient; the optimizer leaves them, the gradient check skips them, and saved models keep them |
| `size_t CachedElementsPerSample(const std::vector<size_t>& sample_input_shape) const` | Your forward keeps more than its output for the backward pass (attention weights, gate values, a mask): return how many elements, per sample of that shape (no batch axis). The Engine's training-memory estimate (Compile, F7) adds it |
| `size_t BackwardWorkingElementsPerSample(const std::vector<size_t>& sample_input_shape) const` | Your backward allocates large temporaries while it runs (unfolded input columns): return their elements per sample. Only the largest over the model counts |
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

#### Drawing with the Engine's widgets: `PluginUi` (API 1.4)

Plain ImGui calls in a panel follow the theme's ImGui colours, but not the
Engine's button styles, its brand colour or its status colours. Get those
from `PluginContext::GetUi()` (keep the context pointer from `OnLoad`; call
the kit only inside `RenderPanel`, on the UI thread). Declare
`"api_version": "1.4.0"` or newer to use it.

| Member | Draws |
|---|---|
| `UiColors Colors() const` | The theme's colours: `text`, `text_dim`, `text_bright`, `bg_window`, `bg_panel`, `bg_raised`, `bg_input`, `accent`, `accent_text`, `success`, `warning`, `caution`, `error`, `pending`, `running`. Read them every frame: the user can switch themes |
| `bool PrimaryButton(label, enabled, disabled_reason, size, width)` | The filled brand button: one per area, for its main action |
| `bool SecondaryButton(...)`, `bool LinkButton(label, enabled)`, `bool DangerButton(...)` | Supporting actions, navigation, removal. A disabled button stays visible and shows `disabled_reason` on hover |
| `float ButtonWidth(label, size) const`, `float ButtonHeight(size) const` | To right-align buttons or centre a row |
| `bool SegmentedControl(id, labels, count, &selected)`, `float SegmentedControlWidth(...)` | One of several options side by side |
| `bool StatusPill(id, text, dot)`, `float StatusPillWidth(text)` | A state with a coloured dot (Running, Paused) |
| `bool NumberField(id, &value, min, max, step, format, width, hint)` | A typed number with - / + buttons when `step` > 0. CyxWiz uses no sliders or drag fields: a typed value is exact |
| `bool SearchField(id, buffer, size, hint, width)` | A search box with its icon |
| `void EmptyState(title, hint)` | The centred message of an empty panel; put its actions after it |
| `std::string OpenFileDialog(title, filter_name, extensions)` | The system's open-file dialog; empty when cancelled |
| `bool ToggleChip(id, label, bool* on)`, `float ToggleChipWidth(label)` | A pill that turns a choice on or off (API 1.4): accent text when on, dimmed when off; flips `*on` and returns true when clicked |

<!-- not compiled: a sketch; the full panel is plugins/simulation/mujoco/src/mj_viewport_panel.cpp -->
```cpp
void MyPlugin::RenderPanel(const std::string& panel_id, bool* visible) {
    if (!ImGui::Begin("My Panel", visible)) { ImGui::End(); return; }
    PluginUi* ui = ctx_->GetUi();
    if (ui->PrimaryButton("Run", can_run, "Load a model first")) Run();
    ImGui::SameLine();
    if (ui->SecondaryButton("Open library")) ctx_->ShowPanel("my_library_panel");
    ImGui::End();
}
```

The MuJoCo plugin (`plugins/simulation/mujoco`, `mj_viewport_panel.cpp`)
draws its whole viewport this way.

### `ICommandProvider` (menu commands, API 1.4)

A plugin adds actions to the Engine's menus: the MuJoCo plugin fills the
**Simulation** menu this way (Run Simulation, Pause, Stop, Step Once, Load
MJCF Model...). Needs `UIModify`.

| Member | Contract |
|---|---|
| `std::vector<PluginCommandInfo> GetCommands()` | Asked once when the plugin starts. Each: `command_id` (unique in your plugin), `menu` (a top menu by its title, such as `"Simulation"` or `"Tools"`; a title the Engine has no menu for puts the command in **Apps** under that title), `group` (a separator between groups), `label` (end it with `...` when it opens a dialog), `hint` (one sentence for the status bar and the Command Palette), `checkable`, `replaced_by_graph_run` (your own run control, such as Run or Stop of your viewport: hidden while the open graph is a simulation graph, whose run commands take its place) |
| `PluginCommandState GetCommandState(const std::string& command_id)` | Asked every frame a menu or the palette shows the command, so keep it cheap: `enabled`, `disabled_reason` (shown on hover and in the status bar; say what to do), `checked` |
| `void RunCommand(const std::string& command_id)` | The user chose it. Runs on the UI thread, between frames: open a panel with `PluginContext::ShowPanel` to show the result |

The **Simulation** menu starts with the open graph's run (Run Sim or
Resume, Pause, Step Once, Stop, Clear Traces) when the graph has simulation
nodes. Then come the commands that name the menu, under each plugin's name,
leaving out those marked `replaced_by_graph_run` while such a graph is open.
Then **Simulation Monitor** and the panels whose `category` is
`"Simulation"`, as show/hide entries. Without a simulation plugin it says
so. Everything is called on the UI thread; a
command that throws is logged and the Engine goes on.

<!-- not compiled: a sketch; the full provider is plugins/simulation/mujoco/src/mujoco_plugin.cpp -->
```cpp
std::vector<PluginCommandInfo> GetCommands() override {
    return {{"run", "Simulation", "run", "Run Simulation", "Runs the loaded model.", false}};
}
PluginCommandState GetCommandState(const std::string& id) override {
    if (!model_loaded_) return {false, "Load a model first.", false};
    return {};
}
void RunCommand(const std::string& id) override {
    if (id == "run") StartRun();
    ctx_->ShowPanel("my_viewport");
}
```

### `IPreferencesProvider` (settings in Preferences, API 1.4)

A plugin's own defaults belong in the Engine's **Preferences** dialog, next
to the Engine's: the MuJoCo plugin puts the Viewport's (the camera a policy
playback starts with, the live numbers, the Record defaults, the Stress
panel's choices and the Scene options) at the end of
**Preferences > Simulation** this way. Needs `UIModify`.

| Member | Contract |
|---|---|
| `std::vector<PluginPreferencesSection> GetPreferenceSections()` | Asked once when the plugin starts. Each: `section_id` (unique in your plugin), `tab` (`"General"`, `"Editor"`, `"Files"`, `"Python"` or `"Simulation"`; any other title puts the section in General), `title` (its heading, such as your panel's name), `hint` (one muted sentence under the heading, or empty) |
| `void RenderPreferences(const std::string& section_id)` | Draws the section's settings below its heading with ImGui and `GetUi()` (typed number fields, segmented controls, checkboxes; no sliders). Called every frame the tab is open, on the UI thread: keep it cheap. Save a change at once with `PluginContext::SetSetting`, and read the setting where your panel starts |

A section that throws is logged and the dialog goes on. The Engine's own
convention: Preferences hold how a panel starts, and a change made in the
panel lasts for the session only (offer a "Keep these as the default" item
when a panel's own choices should become the default).

<!-- not compiled: a sketch; the full provider is plugins/simulation/mujoco/src/mujoco_plugin.cpp -->
```cpp
std::vector<PluginPreferencesSection> GetPreferenceSections() override {
    return {{"viewer", "Simulation", "My Viewer", "How the viewer starts."}};
}
void RenderPreferences(const std::string& id) override {
    bool grid = ctx_->GetSetting("viewer.grid", "on") == "on";
    if (ImGui::Checkbox("Show the grid", &grid)) ctx_->SetSetting("viewer.grid", grid ? "on" : "off");
}
```

### `ISimulationProvider` (graph simulation runs, API 1.4)

A plugin node runs in a graph simulation (**Run Sim** on the canvas) only
when its plugin names it here; without this interface its nodes are not
offered for a simulation run, even with `EvaluateNode`. The MuJoCo plugin
uses it so that a run with a **MuJoCo Plant** holds the physics and shows the
**MuJoCo Viewport**. Needs `UIModify`.

| Member | Contract |
|---|---|
| `std::vector<PluginSimulationNodeInfo> GetSimulationNodes()` | Asked once when the plugin starts. Each: `node_type` (one of your `INodeProvider` node types), `panel_id` (one of your panels the Engine shows when a run with this node starts, or empty) |
| `PluginSimulationBeginResult BeginGraphRun(const std::vector<PluginSimulationGraphNode>& nodes)` | The user runs a graph holding your nodes; `nodes` are yours, each with `node_id`, `node_type` and its `parameters`. Check them and take what the run needs (the right to step a model, which nothing else may hold meanwhile). Return `{false, reason}` to refuse: the run does not start, the toolbar shows the reason, and `EndGraphRun` is not called. Say what to do in the reason |
| `void EndGraphRun()` | The run ended (Stop, a failed tick, the graph closed, your plugin stopping) and its worker has finished: release what `BeginGraphRun` took |

Both are called on the UI thread, before the first tick and after the last
one; between them `EvaluateNode` is called from the run's worker thread. A
graph with a plugin node that no loaded plugin runs in a simulation does not
start, and the reason names the node. Disabling or unloading your plugin
stops a run that uses it before `OnShutdown`.

<!-- not compiled: a sketch; the full provider is plugins/simulation/mujoco/src/mujoco_plugin.cpp -->
```cpp
std::vector<PluginSimulationNodeInfo> GetSimulationNodes() override {
    return {{"Plant", "my_viewport"}};
}
PluginSimulationBeginResult BeginGraphRun(const std::vector<PluginSimulationGraphNode>& nodes) override {
    if (viewport_running_) return {false, "The viewport is running the model. Stop it, then run the graph."};
    graph_owns_model_ = true;
    return {};
}
void EndGraphRun() override { graph_owns_model_ = false; }
```

### `IEnvironmentProvider` (environments for reinforcement learning, API 1.4)

An environment the Engine can train a policy in. The Engine is the learner:
it asks for many copies of one environment, sends one action per copy and
reads back what each copy observed and earned; your plugin only simulates.
The MuJoCo plugin offers its tasks this way. Needs `Training`.

| Member | Contract |
|---|---|
| `std::vector<PluginEnvironmentInfo> GetEnvironments()` | Asked once when the plugin starts. Each: `id` (unique in your plugin), `name`, `description`, `category`, `observation_size`, the action space (continuous: `action_size` with `action_low` / `action_high` per value; discrete: `discrete_actions`, `action_size` 1 and `action_count` choices), `max_episode_steps` (0 = no limit), `step_seconds` (simulated time of one step), `view_panel_id` (your panel that shows a copy, see `ShowCopy`; empty: none) and `settings` (below). An environment that observes nothing, takes no action, has bounds that do not match its action size, or one discrete choice, is not added (the log says why) |
| `bool DescribeEnvironment(const std::string& id, const std::map<std::string, std::string>& settings, PluginEnvironmentInfo& info, std::string& error)` | Optional (default: the environment as `GetEnvironments` lists it). The environment as these settings make it, when what it observes follows them (the MuJoCo plugin's Look ahead adds 10 values). The Engine asks while it draws the Environment node and before it creates copies, so answer from what you already know (no file loading). False with `error` for settings you cannot use |
| `PluginEnvironmentCreateResult CreateEnvironments(const std::string& id, const std::map<std::string, std::string>& settings, int count)` | `count` copies of environment `id`. Settings you are not given keep their defaults; a value you cannot use is an error (`{nullptr, "why and what to do"}`), never a silent default. The batch must report the count and sizes `DescribeEnvironment` gives for these settings, or the Engine releases it and refuses it |

A `PluginEnvironmentSetting` is one setting on the Environment node: `key`,
`label`, `default_value` and `description`, and how the Engine draws it.
Every value travels as text.

| Field | Meaning |
|---|---|
| `kind` | `Number` (default): a typed number field, kept within `min` .. `max` when `min < max`, with `unit` after the value ("m/s", "%"). `Flag`: a checkbox, "1" or "0". `Choice`: one of `options` as a segmented control, its `value`. `Choices`: any of `options` as chips, their values joined by "," (empty: none) |
| `options` | For `Choice` and `Choices`: `{value, label}` pairs, in the order shown |
| `section` | Settings with the same section are drawn together under it as a heading, after the main ones (empty) |

The `IEnvironmentBatch` you return:

| Member | Contract |
|---|---|
| `int Count()`, `int ObservationSize()`, `int ActionSize()` | What the batch holds |
| `bool Reset(uint64_t seed, float* observations)` | Starts every copy and writes `[count x observation_size]`. Copy `i`'s randomness comes from `(seed, i)` alone, so one seed gives the same runs on any number of threads |
| `bool Step(const float* actions, const PluginEnvironmentStepBuffers& out)` | `actions` is `[count x action_size]` (discrete: the choice's index as a float). Advance every copy one step and write `observations`, `rewards`, `terminated` (a terminal state) and `truncated` (the step limit). A copy whose episode ends starts its next episode in the same call: its row of `final_observations` gets the observation it ended on, and its row of `observations` the first one of the next episode. Rows of copies that did not end are left alone in `final_observations`. Return false with `LastError()` when a step fails |
| `std::string LastError()` | Why the last `Reset` or `Step` failed |
| `std::vector<PluginEnvironmentStat> Stats()` | Optional (default: none). Numbers about the copies so far, `{label, value, note}` ("Difficulty", "0.35", "curriculum, rising"); the RL Training Dashboard shows them as cards beside the return, read once per update between steps |
| `void LabelShownCopy(const PluginShownCopyLabel& label)` | Optional (default: ignored; API 1.4). How your view names the copy `ShowCopy` shows, each part optional: `subtitle` (the line under the task's name, "Preview of update 1,580"), `status` ("Training preview") and `rows` put first in your live numbers (`{"Networks of", "update 1,580"}`). `watch_only`: the copy is a training's own, so your view must not change it (drop the user's pushes, throws and ground changes for it). `ShowCopy` starts it empty (your view's own words). The Engine labels its training preview and a training copy shown while it trains |
| `bool ShowCopyAsGhost(int copy, const std::string& name)` | Optional (default: false, no ghost; API 1.4). Shows copy `copy` see-through in your view beside the copy another batch shows (`ShowCopy`), as later Resets and Steps move it, until `ShowCopyAsGhost(-1, "")` or `Release`: a second policy to compare. `name` names it (the policy). The Engine calls it when Policy Playback plays a second saved policy beside the first (**Show as a ghost**): that batch has the same environment and seed, and the played copy is labelled with a `Policy` row. Draw it only when it plays the same model and scene as the shown copy; return false when your view draws no ghost (the Engine then shows its numbers only) |
| `bool ShowCopy(int copy)` | Optional (default: false, no picture). Name the panel that shows it in `PluginEnvironmentInfo::view_panel_id`; the Engine opens that panel when it plays a policy. Show copy `copy` in your own view as later `Reset`s and `Step`s move it, until `ShowCopy(-1)` or `Release`; the Engine calls it to play a trained policy. Return false for a copy out of range. The batch must still not use your plugin object: share a small state object with your view instead (the MuJoCo plugin publishes the copy's `mj_getState` into a slot its viewport reads) |
| `void Release()` | Free the batch in your own module (`delete this`) |

Every buffer is flat, row-major, one row per copy, so the Engine moves all
copies' data in one piece per step. A batch stands alone: it must not use
your plugin object, because it may outlive your `OnShutdown` (the Engine keeps
your library loaded until the last batch is released). One thread at a time
calls a batch; step the copies on your own threads if you like. Every call
is crash-isolated: a crash comes back as an error instead of ending the Engine.
`cyxwiz-plugin-check` creates two copies of each environment, resets them and
steps them 20 times with the middle of each action range.

<!-- not compiled: a sketch; the full provider is plugins/simulation/mujoco -->
```cpp
std::vector<PluginEnvironmentInfo> GetEnvironments() override {
    PluginEnvironmentInfo pendulum;
    pendulum.id = "Pendulum-v0";
    pendulum.name = "Pendulum";
    pendulum.observation_size = 3;
    pendulum.action_size = 1;
    pendulum.action_low = {-2.0f};
    pendulum.action_high = {2.0f};
    pendulum.max_episode_steps = 200;
    return {pendulum};
}
PluginEnvironmentCreateResult CreateEnvironments(const std::string& id,
                                                 const std::map<std::string, std::string>& settings,
                                                 int count) override {
    return {new PendulumBatch(count), {}};  // PendulumBatch : IEnvironmentBatch
}
```

### `IDataProvider`, `IAnalyticsProvider`

A dataset format and a dataset analytic. Data crosses in both directions as
the Arrow C stream interface (`arrow_c_stream.h`, the structs of the Apache
Arrow specification): a stream of record batches with one schema. The SDK
ships no Arrow library and needs none. Fill or read the structs by hand, or
with any Arrow library of any version (Arrow C++ `arrow::ExportRecordBatchReader`
/ `ImportRecordBatchReader`, nanoarrow, arrow-rs).
`plugins/examples/data_formats` (in the SDK `examples/plugins/data-formats`)
does both by hand, with no Arrow library; [Data plugins](data-plugin-guide.md)
walks through it.

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
| `AnalyticsResult Compute(analytics_id, ArrowArrayStream* data)` | On a worker thread, when the user presses Run in Explore > Plugins. `data` streams the columns the user picked, in the order picked, every row: in batches, a table kept on disk one row group at a time. The Engine owns the stream and releases it after `Compute` returns: read it during the call, do not keep it. If the user cancels, `get_next` fails and the answer is dropped. A crash or exception inside `Compute` is caught: the user sees its message and the Engine keeps running |

`AnalyticsResult`: a number (`scalar_value` with `has_scalar`, shown large),
labelled values (`vector_value` with `vector_labels`, shown as bars and as a
table the user can copy or export as CSV), and `text_value` (shown as a note
under them); any combination. On failure set `success = false` and `error_message`. An
empty `name` or `category` is filled from the analytic's info.

Unloading a plugin waits until its streams are read and its analytics have
returned, then frees its library. Data already read stays readable.

## 10. Small types

| Type | Members |
|---|---|
| `PluginVersion` | `major`, `minor`, `patch`; `Parse("1.2.0")`, `ToString()`, `IsCompatibleWith(engine_api)`, comparisons |
| `PLUGIN_API_VERSION_MAJOR`, `_MINOR`, `_PATCH` | 1, 3, 0 |
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
| `CyxWizSDK_PLUGIN_API_VERSION` | The plugin API version (1.4) |
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
cyxwiz-plugin-check [--open <file>]... <plugin folder> [<plugin folder> ...]
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
7. for each `--open <file>`, when the plugin adds a dataset format: picks
   the format as Data Input does (extension, then `CanLoad`), reads every
   batch, validates the table, prints *ok open &lt;file&gt; as &lt;format&gt;:
   &lt;rows&gt; rows, &lt;columns&gt; columns in &lt;n&gt; batches* and the column
   types, then runs each analytic on the columns it reads and prints its
   answer (*ok analytic &lt;name&gt; on &lt;file&gt; (&lt;columns&gt;): ...*), or a
   *FAIL open ...* / *FAIL analytic ...* line with the reason
   ([data plugin guide](data-plugin-guide.md#7-test-and-debug));
8. unloads the plugin.

The last line is *cyxwiz-plugin-check: all plugins pass* or
*cyxwiz-plugin-check: FAILED*. Exit code 0 when every folder and file
passes, 1 otherwise, 2 for no folder or an unknown option. The Engine's log lines are printed too.
