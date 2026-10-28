# Plugin developer guide: trainable nodes

How to write a C++ plugin whose nodes train inside CyxWiz. Build and load the
template first with the [quickstart](plugin-quickstart.md). Every type,
`plugin.json` field and CMake name is listed in the
[API reference](plugin-api-reference.md); the rules every node keeps are in
the [node contract](node-contract.md).

> Checked against plugin API 1.2.0 and pycyxwiz 0.2.0.

The code in this guide is taken from the SDK's template
(`plugins/template/my_nodes_plugin.cpp`, in the SDK as
`template/my_nodes_plugin.cpp`) and from the example
`plugins/examples/activation_nodes/activation_nodes_plugin.cpp` in the
CyxWiz repository, which the test `test_plugin_tensor_nodes` builds, loads
and trains.

## 1. What a plugin is

A folder with a manifest and a shared library:

```
my_nodes/
  plugin.json
  bin/my_nodes.dll        (Linux: bin/libmy_nodes.so, macOS: bin/libmy_nodes.dylib)
```

### The manifest

The template's `plugin.json`:

```json
{
    "id": "com.example.my-nodes",
    "name": "My Nodes",
    "version": "1.0.0",
    "api_version": "1.2.0",
    "description": "A trainable activation written in C++: Bent Identity",
    "author": "",
    "license": "",
    "capabilities": ["ProvidesTensorNodes"],
    "permissions": ["Training"],
    "platforms": {
        "windows": { "library": "bin/my_nodes.dll" },
        "linux": { "library": "bin/libmy_nodes.so" },
        "macos": { "library": "bin/libmy_nodes.dylib" }
    }
}
```

| Field | Rule |
|---|---|
| `id` | Required. `a-z`, `0-9`, `.`, `-`, `_` only (a plugin with other characters loads, but every node is refused). It is the first half of every node's saved type id, so do not change it once graphs use your nodes |
| `name`, `version` | Shown on the Plugin Manager card. `version` is `major.minor.patch`; dangerous permissions are approved per version (section 4) |
| `api_version` | The plugin API you built against, `1.2.0` with this SDK. The Engine loads the plugin when the major number equals its own and the minor number is not higher |
| `capabilities` | `ProvidesTensorNodes`. Shown only; the Engine finds your nodes through `QueryInterface`, whatever this says |
| `permissions` | `Training`, as the template has. Nothing checks it for trainable nodes today. `FileSystem`, `Network`, `SystemCommands` and `Python` make the Engine ask the user (section 4) |
| `platforms` | The library path per platform, relative to the folder |
| `signature` | Leave it out. Only plugins signed with CyxWiz's own key pass; a value that does not verify refuses the load. Unsigned plugins load, with a warning in the log |

All fields are in the [API reference](plugin-api-reference.md#3-pluginjson).

## 2. The plugin class

A plugin implements `IPlugin` and, for trainable nodes,
`ITensorNodeProvider`, and exports itself with `CYXWIZ_PLUGIN_ENTRY`. Every
`IPlugin` member below is pure virtual except `QueryInterface`: leave one
out and the class does not compile. The template's class:

<!-- excerpt: plugins/template/my_nodes_plugin.cpp -->
```cpp
#include "plugin/plugin_types.h"
#include "plugin/plugin_context.h"
#include "plugin/interfaces/i_tensor_node_provider.h"

#include <cyxwiz/sequential.h>

#include <map>
#include <memory>
#include <string>
#include <vector>

namespace my_nodes {

using namespace cyxwiz;
using namespace cyxwiz::plugin;

// ...

class MyNodesPlugin : public IPlugin, public ITensorNodeProvider {
public:
    bool OnLoad(PluginContext& ctx) override {
        ctx.LogInfo("My Nodes loaded");
        state_ = PluginState::Loaded;
        return true;
    }
    bool OnInitialize(PluginContext&) override {
        state_ = PluginState::Active;
        return true;
    }
    void OnShutdown(PluginContext&) override { state_ = PluginState::Loaded; }
    void OnUnload(PluginContext&) override { state_ = PluginState::Unloaded; }

    const PluginManifest& GetManifest() const override { return manifest_; }
    PluginPermissionFlags GetRequiredPermissions() const override {
        return static_cast<uint32_t>(PluginPermission::Training);
    }
    PluginState GetState() const override { return state_; }
    void* QueryInterface(const char* name) override {
        if (std::string(name) == "ITensorNodeProvider") return static_cast<ITensorNodeProvider*>(this);
        return nullptr;
    }

    // ...

private:
    PluginManifest manifest_;
    PluginState state_ = PluginState::Unloaded;
};

}  // namespace my_nodes

CYXWIZ_PLUGIN_ENTRY(my_nodes::MyNodesPlugin)
```

- The Engine reads `plugin.json`, not `GetManifest()`; an empty
  `PluginManifest` is enough. It does not call `GetState()` either.
- `CYXWIZ_PLUGIN_ENTRY` goes once, after the class, outside any namespace,
  with the class's full name.
- `QueryInterface` must return the pointer cast to the interface
  (`static_cast<ITensorNodeProvider*>(this)`), not `this`.

### Building it

Link the one target `CyxWiz::PluginSDK` and call `cyxwiz_plugin_folder`, as
the template's `CMakeLists.txt` does:

<!-- excerpt: plugins/template/CMakeLists.txt -->
```cmake
if(NOT TARGET CyxWiz::PluginSDK)
    find_package(CyxWizSDK REQUIRED)
endif()

add_library(my_nodes SHARED my_nodes_plugin.cpp)
target_link_libraries(my_nodes PRIVATE CyxWiz::PluginSDK)
# ...
cyxwiz_plugin_folder(my_nodes)
```

The installed SDK defines two targets: `CyxWiz::PluginSDK` (the headers, the
backend's import library, C++20) and `CyxWiz::Backend` (the import library
alone). Keep `cmake_minimum_required(VERSION 3.20)` and `project()` at the
top of the file ([quickstart](plugin-quickstart.md#2-build-the-template)).

Include the plugin headers as `"plugin/..."` and the backend headers as
`<cyxwiz/...>`. They need only the C++ standard library.

### Other libraries

The SDK ships no spdlog, ImGui, JSON or ArrayFire. Log through the context
(`ctx.LogInfo`, below) instead of spdlog. Any other library is yours to find
(`find_package`) and link, with these limits:

- Never pass that library's objects to the Engine or take them from it.
- **Windows:** the Engine loads your DLL's own dependencies from the folder
  of your DLL (`bin/`), the Engine's folder and the Windows system folders.
  `PATH` is not searched. Copy every DLL your plugin needs into `bin/`.
- **Linux, macOS:** the system loader uses your library's run path. Copy the
  libraries into `bin/` and set the run path to it:
  `set_target_properties(my_nodes PROPERTIES BUILD_RPATH "$ORIGIN")` on
  Linux, `"@loader_path"` on macOS.

### The context

The `PluginContext&` passed to each lifecycle call is implemented by the
Engine:

| Call | What it does |
|---|---|
| `GetPluginId()`, `GetPluginDir()` | Your manifest id and your plugin folder (the one with `plugin.json`) |
| `LogInfo`, `LogWarn`, `LogError`, `LogDebug` | Write to the Engine's log, prefixed with `[Plugin:<id>]` (section 11) |
| `RegisterNodeProvider`, `RegisterPanelProvider`, `RegisterDataProvider`, `RegisterTrainingHook`, `RegisterAnalyticsProvider` | Not needed for trainable nodes ([API reference](plugin-api-reference.md#5-plugincontext)) |

Keep the reference only until `OnUnload` returns. Trainable nodes need none
of the `Register*` calls: the Engine asks
`QueryInterface("ITensorNodeProvider")` after `OnInitialize`.

## 3. Describing your nodes: `GetTensorNodes`

Return one `TensorNodeInfo` per node. From the example:

<!-- excerpt: plugins/examples/activation_nodes/activation_nodes_plugin.cpp -->
```cpp
TensorNodeInfo snake;
snake.type_name = "Snake";
snake.display_name = "Snake";
snake.category = "Activations";
snake.description = "x + sin^2(a x) / a, with a learnable frequency a; suits periodic signals";
snake.version = "1.0.0";
snake.color = 0xFF3C8F4A;  // IM_COL32(74, 143, 60)
snake.is_activation = true;
snake.pins = {{"Input", true}, {"Output", false}};
snake.parameters = {{"initial_a", "float", "1.0", "Starting value of the frequency a (greater than 0)", {}}};
```

| Field | Rule and where the user sees it |
|---|---|
| `type_name` | Letters `A-Z a-z`, digits, `_`. With the plugin id it forms the type id `com.cyxwiz.examples.activation-nodes:Snake`, saved in graphs. Do not rename it |
| `display_name` | The node's title on the canvas and in the Node Browser (empty: `type_name`) |
| `category` | Your own group name; the Node Browser and search group by it |
| `description` | Shown in the Node Browser, Info panel and Properties |
| `version` | The version of this node's pins and settings. Saved with the graph; a different version when a graph is opened gives a warning on the node |
| `color` | Header colour, `IM_COL32` layout (`0xAABBGGRR`). A colour with alpha 0, such as `0`, uses the Engine's plugin node colour |
| `icon` | Leave empty: the Engine uses its plug icon |
| `is_activation` | `true` files the node with the activations |
| `pins` | Exactly one input and one output, in API 1.2 |
| `parameters` | The node's settings: `{name, type, default, description, choices}` |

Setting types and their editors in **Properties**:

| `type` | Editor |
|---|---|
| `float` | Number field |
| `int` | Whole-number field with steps |
| `bool` | Checkbox (values `true` / `false`) |
| `enum` | Drop-down of `choices` |
| `string` | Text field |

Under each field Properties shows your description, then the type and
default: *Starting value of the frequency a (greater than 0). Number.
Default 1.0*. Settings reach your code as a
`std::map<std::string, std::string>` of text values (`"0.8"`, `"true"`).

`enum` choices cross to the Engine as one text joined by commas: a choice
that contains a comma is split into two, and an empty choice is dropped.

### Nodes the Engine refuses

The Engine checks each node when the plugin initialises and leaves out the
ones it refuses. The rest of the plugin loads. **Install...** then shows
*Plugin installed, but N of its nodes were not added:* with one reason per
node; `cyxwiz-plugin-check` prints *FAIL node not added: &lt;reason&gt;*.

| Reason | Cause |
|---|---|
| `<type>: it needs exactly one input and one output pin` | `pins` is not one input and one output |
| `provider id '...' may only use a-z, 0-9, '.', '-' and '_'` | The `id` in `plugin.json` |
| `node type name '...' may only use letters, digits and '_'` | `type_name` |
| `the node type name is empty` | `type_name` is empty |
| `node type '...' is already registered by '...' (<folder>)` | Two nodes with the same type name, or another loaded plugin with the same id |

## 4. Permissions

| Permission | What happens |
|---|---|
| `Training`, `UIModify`, `DataRegistry`, `GPU` | Granted without asking. Trainable nodes need none of them |
| `FileSystem`, `Network`, `SystemCommands`, `Python` | Before the plugin starts, the Engine asks the user to allow or deny each one. The answer is stored per plugin id and version, so a new `version` asks again |

A permission is a statement to the user, not a sandbox: your library runs
with the Engine's rights whatever was approved, and a denied permission does
not stop the plugin. What you register checks the permission for it
(Training for trainable nodes and training hooks, UIModify for panels and
signal nodes, DataRegistry for data and analytics providers; none of the four
asked ones), against the flags your `GetRequiredPermissions()` returns, not
`plugin.json`. Declare a permission in both places.

## 5. Shapes: `InferOutputShape`

<!-- excerpt: plugins/examples/activation_nodes/activation_nodes_plugin.cpp -->
```cpp
ExtensionShapeResult InferOutputShape(const std::string& type_name,
                                      const std::vector<size_t>& input_shape,
                                      const std::map<std::string, std::string>& parameters) override {
    if (type_name == "Snake") {
        float a = 0.0f;
        try {
            a = ParseFloat(parameters, "initial_a", 1.0f);
        } catch (const std::exception&) {
            return {false, {}, "initial_a must be a number"};
        }
        if (!(a > 0.0f)) return {false, {}, "initial_a must be greater than 0"};
    }
    // Elementwise: the output has the input's shape.
    return {true, input_shape, {}};
}
```

- `input_shape` is **one sample's shape, without the batch dimension**.
  Return the output shape the same way.
- This is where you refuse bad settings. Return `{false, {}, "message"}`.
  The message is shown on the node when the graph compiles, and in
  Properties under the setting it names: write the setting's name in it
  (`initial_a must be greater than 0`).
- Keep it pure and cheap. The Engine calls it after every graph edit, from
  any thread, possibly from two threads at once.
- `parameters` may lack a setting, or hold an empty text: fall back to the
  default, as `ParseFloat` in the example does.

## 6. Modules: `CreateModule`

<!-- excerpt: plugins/examples/activation_nodes/activation_nodes_plugin.cpp -->
```cpp
std::unique_ptr<Module> CreateModule(const std::string& type_name,
                                     const std::vector<size_t>&,
                                     const std::map<std::string, std::string>& parameters,
                                     std::string& error) override {
    if (type_name == "Softsign") return std::make_unique<SoftsignModule>();
    if (type_name == "Snake") return std::make_unique<SnakeModule>(ParseFloat(parameters, "initial_a", 1.0f));
    error = "unknown node type " + type_name;
    return nullptr;
}
```

The second argument is again one sample's input shape (unused by these
elementwise nodes); size your weights from it. On failure return `nullptr`
and set `error`. Like `InferOutputShape`, it can be called from any thread.

### A module without weights

<!-- excerpt: plugins/examples/activation_nodes/activation_nodes_plugin.cpp -->
```cpp
class SoftsignModule : public Module {
public:
    Tensor Forward(const Tensor& x) override {
        input_cache_ = x;
        return x / (x.Abs() + 1.0f);
    }
    Tensor Backward(const Tensor& grad) override {
        const Tensor d = input_cache_.Abs() + 1.0f;
        return grad / (d * d);
    }
    IncrementalDecoding GetIncrementalDecoding() const override { return IncrementalDecoding::PositionWise; }
    std::string GetName() const override { return "Softsign"; }
};
```

### A module with a learnable weight

<!-- excerpt: plugins/examples/activation_nodes/activation_nodes_plugin.cpp -->
```cpp
float HostScalar(const Tensor& tensor) {
    return tensor.ReadData<float>()[0];
}

Tensor Scalar(float value) {
    return Tensor(std::vector<size_t>{1}, &value);
}

// ...

class SnakeModule : public Module {
public:
    explicit SnakeModule(float initial_a) : a_(Scalar(initial_a)), grad_a_(Scalar(0.0f)) {}

    Tensor Forward(const Tensor& x) override {
        input_cache_ = x;
        const float a = HostScalar(a_);
        const Tensor s = (x * a).Sin();
        return x + (s * s) / a;
    }

    Tensor Backward(const Tensor& grad) override {
        const float a = HostScalar(a_);
        const Tensor& x = input_cache_;
        const Tensor s = (x * a).Sin();
        const Tensor sin2 = (x * (2.0f * a)).Sin();
        // dy/da = x sin(2ax) / a - sin^2(ax) / a^2, summed against grad.
        const Tensor da = grad * ((x * sin2) / a - (s * s) / (a * a));
        const float* values = da.ReadData<float>();
        double sum = 0.0;
        for (size_t i = 0; i < da.NumElements(); ++i) sum += values[i];
        grad_a_ = Scalar(static_cast<float>(sum));
        // dy/dx = 1 + sin(2ax).
        return grad * (sin2 + 1.0f);
    }

    std::map<std::string, Tensor> GetParameters() override { return {{"a", a_}}; }
    void SetParameters(const std::map<std::string, Tensor>& params) override {
        const auto it = params.find("a");
        if (it != params.end()) a_ = it->second;
    }
    std::map<std::string, Tensor> GetGradients() override { return {{"a", grad_a_}}; }
    bool HasParameters() const override { return true; }
    IncrementalDecoding GetIncrementalDecoding() const override { return IncrementalDecoding::PositionWise; }
    std::string GetName() const override { return "Snake"; }

private:
    Tensor a_;
    Tensor grad_a_;
};
```

The rules these follow are in the [node contract](node-contract.md). In
short:

- `Forward` gets the whole batch; dimension 0 is the batch, and its size can
  change from call to call. Do not change the input tensor.
- `Backward` returns the gradient for the input of the last `Forward`, with
  that input's shape, and stores the gradients of your weights.
- `GetParameters`, `GetGradients` and `SetParameters` use the same keys;
  `HasParameters` returns `true` when there are any.
- `SetParameters` may receive only some of your keys (the gradient check
  sets one at a time): update the ones it holds and keep the others, as
  `SnakeModule` does.
- Keep your parameter keys stable across versions: saved weights are loaded
  by key (section 9).
- `SetTraining(bool)` tells training from evaluation (the gradient check runs
  in evaluation). If you override it, set the base class's `is_training_`
  too, or call `Module::SetTraining`.

`GetIncrementalDecoding` matters for language models that generate text one
token at a time. Return `PositionWise` when each output position depends only
on the same input position (activations, per-feature layers); leave the
default (`Unsupported`) otherwise, and the model recomputes the whole
sequence.

### Tensors

| Topic | Rule |
|---|---|
| Type | Tensors reaching your module are `float32` (`DataType::Float32`) |
| Layout | Row-major; dimension 0 is the batch |
| Operations | Arithmetic with tensors and scalars, `Abs`, `Sin`, `Exp`, `Sum`, `Mean`, `Dot`, `BatchMatMul` and more run on the Engine's compute device. The full list, with the Python names beside them: [tensor-operations.md](tensor-operations.md) |
| Host access | `ReadData<float>()` copies the values from the device to host memory when the host copy is stale: it is slow inside `Forward` and `Backward`, so use it for scalars, as `HostScalar` does. `T` must match the tensor's type; nothing checks it |
| New tensors | `Tensor(shape, const void* data)` copies host values (`float` by default); `Tensor::Zeros(shape)`, `Tensor::Ones(shape)` |
| ArrayFire | `Tensor(const af::array&)`, `GetArray()` and other members with `af::` types are declared but unusable from the SDK: it has no ArrayFire headers. Do not use raw ArrayFire arrays: they are column-major, `Tensor` is row-major |

## 7. What the Engine checks for you

Every module from a plugin is wrapped by a guard. When a check fails,
training stops with a message that names your node, for example
*Extension node 'Snake' (com.cyxwiz.examples.activation-nodes:Snake):
backward returned gradient shape [8, 16], expected the input shape [8, 32]*.

| Check | Message ends with |
|---|---|
| Forward returned nothing | `returned an empty tensor` |
| Output shape per sample is not what `InferOutputShape` said | `expected output shape [..] per sample, got [..] for input [..]` |
| Batch size changed | `changed the batch size from N to M` |
| Forward or Backward threw | `forward failed: ...`, `backward failed: ...` |
| Gradient shape differs from the input | `backward returned gradient shape [..], expected the input shape [..]` |
| Parameters and gradients do not pair up | `its gradients do not match its parameters (..)` |
| A gradient's shape differs from its parameter | `gradient of 'a' has shape [..], the parameter has [..]` |

## 8. The gradient check

The Engine does not differentiate your code: you write `Backward`, and the
Engine checks it. **Check gradient** in Properties uses the node's current
settings; `cyxwiz-plugin-check` uses the default settings.

| Step | Detail |
|---|---|
| Modules | One module per shape, from your `CreateModule`, with the sample shapes `[3]` and `[4, 5]`, in evaluation mode (`SetTraining(false)`). Properties checks both; `cyxwiz-plugin-check` skips a shape your `InferOutputShape` refuses |
| Inputs | Batches of shape `[2, 3]` and `[3, 4, 5]`, random values in [-2, 2]; values closer to zero than 0.05 are moved to ±0.05, away from the kink of `\|x\|`-style functions |
| Loss | sum(Forward(x) · r) for a fixed random r in [-1, 1] |
| Numerical gradient | Central differences with step 0.01, on up to 48 evenly spread elements per tensor |
| Compared | Your `Backward(r)` for the input, and `GetGradients()` for every parameter (each parameter is moved with `SetParameters`, one key at a time) |
| Passes when | The largest relative error is at most 0.02. Relative error is \|analytic − numeric\| / max(0.05, \|analytic\| + \|numeric\|) |

The 0.05 floor keeps 32-bit rounding on a gradient close to zero from
reading as a large error; a wrong gradient is typically off by 0.1 or more.

- In Properties the check runs only on settings your `InferOutputShape`
  accepts; while it refuses them, Properties shows *Could not check* with
  your message. The result goes stale when you change a setting.
- `cyxwiz-plugin-check` asks `InferOutputShape`, with the default settings,
  about each of the two sample shapes, says which it refuses and why, and
  runs the check above on the shapes it accepts; a node that accepts
  neither fails.
- The check does not go through the guard of section 7: a wrong output shape
  shows up only in training.

## 9. Saved graphs and trained models

A graph stores each of your nodes with its type id, your node version, a
fingerprint of your node's description, its settings and its pins. When the
graph opens:

| Situation | What the user sees |
|---|---|
| Your plugin is loaded, same version | The node, as saved |
| Your plugin is loaded, different version or pins | The node, with a warning that it is not the one the graph was saved with |
| Your plugin is not loaded | The node in the not-installed look, its saved settings shown read-only, and **Open Plugin Manager** |

A not-installed node on the path from the data to the loss stops the compile
(*Extension node '...' is not installed. Install and approve the extension
that provides it, or remove the node.*); off the path it is a warning.

| Path | What happens |
|---|---|
| Export a CyxWiz package (`.cyxmodel`) | Your module's `GetParameters` are saved with the other weights; the graph is always included; the manifest's `extensions` lists your node types with their versions |
| Import the package | Your plugin builds the module again (`CreateModule`) and `SetParameters` gets the saved weights. Plugin not loaded: refused, "is not installed". Another version: loads, with a warning |
| ONNX export, generated code | Refused with the node's name: they cannot contain your module's code |
| Server Node deployment, remote training | Refused: the Server Node does not load plugins yet |

## 10. Build fingerprint

A trainable node's module is a real `cyxwiz::Module` that the Engine calls
directly, so your plugin must be built the way the Engine was.
`CYXWIZ_PLUGIN_ENTRY` exports your build's fingerprint, computed by the
compiler that builds your plugin, for example

```
sdk=0.2;api=1.2;compiler=msvc-19;stdlib=msvc-stl;build=release;ptr=64;arrayfire=none
```

The Engine loads your library, then compares the fingerprint with its own
before it calls `CreatePlugin`. Loading the library has already run your
global constructors (and `DllMain` on Windows): keep them trivial.

| Item | Must match | Values |
|---|---|---|
| `sdk` | The CyxWiz SDK major and minor version | `0.2` |
| `api` | Same major; your minor not newer than the Engine's | `1.2` |
| `compiler` | Compiler family and major version | `msvc-19` (every Visual Studio from 2015 to 2026), `gcc-11`, `clang-16`. clang-cl reports `clang-N`, MinGW `gcc-N`: both differ from a Visual Studio Engine |
| `stdlib` | The C++ standard library | `msvc-stl`, `libstdc++`, `libc++` |
| `build` | `debug` with the debug Visual C++ runtime (`_DEBUG`) or `_GLIBCXX_DEBUG`, otherwise `release` | The Engine is `release` |
| `ptr` | Pointer size | `64` |
| `arrayfire` | The ArrayFire version, compared only when both sides name one | A plugin built against the SDK has no ArrayFire headers and reports `none`, so this item is not compared |

A difference refuses the load with both values, for example *The plugin was
built with build=debug; this Engine with build=release. Rebuild the plugin
against the CyxWiz SDK 0.2 with the same settings.* A plugin built before
API 1.2 has no fingerprint and is refused with *The plugin has no build
fingerprint*. You write nothing for this: building with the current SDK and
`cyxwiz_plugin_folder` is enough.

What the fingerprint cannot see:

- **Compiler minor version.** On Windows every Visual Studio from 2015 on
  reports `msvc-19`. Build with the Visual Studio of the SDK's
  "Built with" line or an older one (2019 at least, for C++20), never a
  newer one: the Engine ships its own Visual C++ runtime, and code from a
  newer compiler can need a newer runtime and crash
  ([quickstart](plugin-quickstart.md#the-compiler)).
- **Architecture and operating system.** A library for another architecture
  or system fails in the operating system's loader, with its error after
  *Failed to load DLL:*.
- **Compiler options** that change object layouts (packing, a different
  `_ITERATOR_DEBUG_LEVEL`). Keep CMake's defaults.

## 11. Loading, unloading and debugging

- **Install...** loads one plugin from its folder and remembers the folder;
  **Refresh** loads every plugin in the two plugin folders
  ([quickstart](plugin-quickstart.md#4-load-it-in-the-engine)). Loaded
  plugins load again when the Engine starts, unless the user unticks
  **Load when the Engine starts**; **Unload** takes the plugin off that
  list. The list is `%APPDATA%\cyxwiz\plugin_startup.json`
  (`~/.cyxwiz/plugin_startup.json` on Linux and macOS); permission answers
  are in `plugin_permissions.json` beside it.
- A plugin id loads once. To load a new build, follow
  [Change, rebuild, reload](plugin-quickstart.md#6-change-rebuild-reload).
- When a plugin is unloaded while a built model still uses its nodes, the
  Engine keeps the library loaded until the last such module is gone. You do
  not need to handle this.
- The **Apps** menu lists your plugin under *Loaded plugins* with what it
  adds (*2 nodes*); choosing it opens the Node Browser on your nodes.

### Logs

`ctx.LogInfo` and the other log calls write to the Engine's **Console** and
to `engine_log.txt` in the Engine's folder (`runtime/base/<base>/`), prefixed
with `[Plugin:<your id>]`. The load reasons, refused nodes and permission
decisions are there too.

### Debugging (Windows)

1. Build Debug (`cmake --build my_nodes\build --config Debug`) or
   RelWithDebInfo; both load. The `.pdb` file lands beside the DLL in
   `bin/`.
2. Start CyxWiz as usual. The launcher starts the Engine as a separate
   process, `cyxwiz-engine.exe`.
3. In Visual Studio: **Debug > Attach to Process...**, choose
   `cyxwiz-engine.exe`, and set breakpoints in your source.
4. **Install...** the plugin (or let it load at start) and use a node.

On Linux attach with `gdb -p <pid of cyxwiz-engine>`, on macOS with
`lldb -n cyxwiz-engine`. Do not start `cyxwiz-engine` directly under the
debugger: it needs the runtime libraries the launcher sets up.

## 12. Threads, crashes and limits

| Topic | Rule |
|---|---|
| Threads | `InferOutputShape` and `CreateModule` may be called from any thread, at the same time; keep them free of shared state. A module's `Forward` and `Backward` run on the thread of the run that owns it (the training thread; the Engine's window thread for **Check gradient**). Several of your modules can be in use on different threads at once: keep state in the module, and lock anything shared |
| Crash isolation | Lifecycle calls, `GetTensorNodes`, `InferOutputShape` and `CreateModule` run inside the Engine's crash guard: an exception or a memory fault there fails the plugin or the node instead of closing the Engine. During training, C++ exceptions from `Forward` and `Backward` are caught by the guard of section 7; a memory fault in them is not, and closes the Engine |
| Global state | The library stays loaded while any of its modules lives, even after **Unload** |
| Nodes | One tensor input and one tensor output. No losses from C++ yet (Python nodes can be losses) |
| Remote training | Graphs with extension nodes do not train on a Server Node yet; the Engine refuses with a message |
| Export | ONNX export and generated code refuse a model with extension nodes, naming the node |
| Panels | `IPanelProvider` needs ImGui built exactly as in the Engine (same version and configuration). The SDK has no ImGui, so a plugin built against it cannot draw panels; panels are for plugins built inside the CyxWiz source tree |

## 13. Distributing your plugin

- Ship the plugin folder: `plugin.json` and `bin/` with your library and any
  library it needs, for example as a zip file.
- The user unzips it into the user plugin folder
  (`%APPDATA%\cyxwiz\plugins\`, `~/.cyxwiz/plugins/`) and presses
  **Refresh**, or uses **Install...** on the folder.
- Build once per platform, and again whenever CyxWiz moves to another SDK
  minor version or compiler (the fingerprint refuses the old build). Name
  the CyxWiz version you built against in your release notes.
- Leave `signature` out: only CyxWiz's own key can sign. Unsigned plugins
  load.
- Raise `version` for each release. A plugin with `FileSystem`, `Network`,
  `SystemCommands` or `Python` asks the user to approve them again for each
  new version.
- Keep `id`, every `type_name` and every parameter key unchanged: graphs and
  trained models refer to them.
- Licence: the SDK and the template are under the Apache License 2.0
  (`sdk/LICENSE`, `sdk/NOTICE`). Your plugin may use any licence; replace
  the template's licence line in your sources.

These pages are `docs/extending/` in the CyxWiz repository on GitHub
(<https://github.com/CYXWIZ-Lab/CYXWIZ>).
