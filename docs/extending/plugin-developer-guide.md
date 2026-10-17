# Plugin developer guide: trainable nodes

How to write a C++ plugin whose nodes train inside CyxWiz. Start with the
[quickstart](plugin-quickstart.md) if you have not built the example yet.

> Checked 2026-10-17 against plugin API 1.2.0. This guide grows as the
> feature is built; sections marked *Later* describe what is planned.

Code in this guide is taken from
`plugins/examples/activation_nodes/activation_nodes_plugin.cpp`, which the
test `test_plugin_tensor_nodes` builds, loads and trains.

## 1. What a plugin is

A folder with a manifest and a shared library:

```
my_nodes/
  plugin.json
  bin/my_nodes.dll        (libmy_nodes.so, libmy_nodes.dylib)
```

### The manifest

```json
{
    "id": "com.cyxwiz.examples.activation-nodes",
    "name": "Example Activation Nodes",
    "version": "1.0.0",
    "api_version": "1.2.0",
    "description": "Trainable example nodes written in C++: Softsign, and Snake with a learnable frequency",
    "author": "CyxWiz Lab",
    "license": "MIT",
    "capabilities": ["ProvidesTensorNodes"],
    "permissions": ["Training"],
    "platforms": {
        "windows": { "library": "bin/activation_nodes.dll" },
        "linux": { "library": "bin/libactivation_nodes.so" },
        "macos": { "library": "bin/libactivation_nodes.dylib" }
    }
}
```

| Field | Rule |
|---|---|
| `id` | Unique. `a-z`, `0-9`, `.`, `-`, `_` only. It becomes the first half of every node's type id, so do not change it once graphs use your nodes |
| `api_version` | The plugin API you built against. The Engine loads the plugin when the major number equals its own and the minor number is not higher. Trainable nodes need 1.1; the build fingerprint (section 9) came with 1.2 |
| `capabilities` | `ProvidesTensorNodes` for trainable nodes |
| `permissions` | `Training` for trainable nodes. File system, network, system commands and Python are dangerous permissions: the user is asked to approve them |
| `platforms` | The library path per platform, relative to the folder |

## 2. The plugin class

A plugin implements `IPlugin` and, for trainable nodes,
`ITensorNodeProvider`, and exports itself with `CYXWIZ_PLUGIN_ENTRY`:

```cpp
#include "plugin/plugin_types.h"
#include "plugin/plugin_context.h"
#include "plugin/interfaces/i_tensor_node_provider.h"

#include <cyxwiz/sequential.h>

class ActivationNodesPlugin : public IPlugin, public ITensorNodeProvider {
public:
    bool OnLoad(PluginContext& ctx) override { ...; return true; }
    bool OnInitialize(PluginContext&) override { ...; return true; }
    void OnShutdown(PluginContext&) override { ... }
    void OnUnload(PluginContext&) override { ... }

    PluginPermissionFlags GetRequiredPermissions() const override {
        return static_cast<uint32_t>(PluginPermission::Training);
    }
    void* QueryInterface(const char* name) override {
        if (std::string(name) == "ITensorNodeProvider") return static_cast<ITensorNodeProvider*>(this);
        return nullptr;
    }
    // GetTensorNodes, InferOutputShape, CreateModule: below.
};

CYXWIZ_PLUGIN_ENTRY(cyxwiz::plugin::examples::ActivationNodesPlugin)
```

Link `cyxwiz-plugin-sdk` and `cyxwiz-backend` (the backend gives you
`cyxwiz::Tensor` and `cyxwiz::Module`):

```cmake
add_library(my_nodes SHARED my_nodes_plugin.cpp)
target_link_libraries(my_nodes PRIVATE cyxwiz-plugin-sdk cyxwiz-backend)
```

`cyxwiz-plugin-sdk` is headers only: it brings the plugin headers, which
need only the C++ standard library, and no library. Its include folder holds only the public
headers (`plugin/...` and the three `core/` headers they use), not the
Engine's sources, so include them as `"plugin/plugin_types.h"`. A plugin that logs with spdlog, draws
ImGui panels or reads JSON links `spdlog::spdlog`, `imgui::imgui` or
`nlohmann_json::nlohmann_json` itself (see the `image_nodes` example).

Write the library into `bin/` beside `plugin.json` (see the example's
`CMakeLists.txt`).

### The context

The `PluginContext&` passed to each lifecycle call is an interface the
Engine implements. Every call on it runs in the Engine:

| Call | What it does |
|---|---|
| `GetPluginId()`, `GetPluginDir()` | Your manifest id and your plugin folder |
| `LogInfo`, `LogWarn`, `LogError`, `LogDebug` | Write to the Engine's log, prefixed with your id |
| `RegisterNodeProvider`, `RegisterPanelProvider`, `RegisterDataProvider`, `RegisterTrainingHook`, `RegisterAnalyticsProvider` | Register with the Engine's own registries, after checking your permissions; returns `false` when a permission is missing |

Keep the reference only for the plugin's lifetime: the Engine destroys the
context after `OnUnload`. Tensor nodes need none of the `Register*` calls;
the Engine finds them through `QueryInterface("ITensorNodeProvider")`.

## 3. Describing your nodes: `GetTensorNodes`

Return one `TensorNodeInfo` per node:

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
| `type_name` | Letters, digits, `_`. With the plugin id it forms the type id `com.cyxwiz.examples.activation-nodes:Snake`, saved in graphs. Do not rename it |
| `display_name` | The node's title on the canvas and in the Node Browser |
| `category` | Your own group name; the Node Browser and search group by it |
| `description` | Shown in the Node Browser, Info panel and Properties |
| `version` | The version of this node's pins and settings. Saved with the graph; a different version when a graph is opened gives a warning on the node |
| `color` | Header colour (`IM_COL32` layout, ABGR); a colour with alpha 0 (such as 0) uses the Engine's plugin node colour |
| `icon` | Leave empty: the Engine uses its plug icon |
| `pins` | v1: exactly one input and one output |
| `parameters` | The node's settings: name, type, default, description, choices |

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
Default 1.0*. Settings reach your code as a `std::map<std::string,
std::string>` of text values.

## 4. Shapes: `InferOutputShape`

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
  any thread.

## 5. Modules: `CreateModule`

```cpp
std::unique_ptr<Module> CreateModule(const std::string& type_name,
                                     const std::vector<size_t>& input_shape,
                                     const std::map<std::string, std::string>& parameters,
                                     std::string& error) override {
    if (type_name == "Softsign") return std::make_unique<SoftsignModule>();
    if (type_name == "Snake") return std::make_unique<SnakeModule>(ParseFloat(parameters, "initial_a", 1.0f));
    error = "unknown node type " + type_name;
    return nullptr;
}
```

`input_shape` is again one sample's shape. Size your weights from it. On
failure return `nullptr` and set `error`.

### A module without weights

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

```cpp
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
short: the batch is dimension 0; `Backward` returns the gradient for the
input of the last `Forward`; parameters, gradients and `SetParameters` use the
same keys; `HasParameters` is true when there are any.

`GetIncrementalDecoding` matters for language models that generate text one
token at a time. Return `PositionWise` when each output position depends only
on the same input position (activations, per-feature layers); leave the
default (`Unsupported`) otherwise, and the model recomputes the whole
sequence.

## 6. What the Engine checks for you

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

## 7. The gradient check

The Engine does not differentiate your code: you write `Backward`, and the
Engine checks it. In **Properties**, **Check gradient** runs the check with
the node's current settings:

| Step | Detail |
|---|---|
| Inputs | Random values in [-2, 2], batch shapes [2, 3] and [3, 4, 5] (your module gets the sample shapes [3] and [4, 5]) |
| Loss | sum(Forward(x) · r) for a fixed random r |
| Numerical gradient | Central differences with step 0.01, up to 48 elements per tensor |
| Compared | Your `Backward(r)` for the input, and `GetGradients()` for every parameter |
| Passes when | The largest relative error is at most 0.02. Relative error is \|analytic − numeric\| / max(0.05, \|analytic\| + \|numeric\|) |

The check runs only on settings your `InferOutputShape` accepts; while it
refuses them, Properties shows *Could not check* with your message.

The 0.05 floor keeps 32-bit rounding on a gradient close to zero from
reading as a large error; a wrong gradient is typically off by 0.1 or more.
The result goes stale when you change a setting; check again before
training.

## 8. Saved graphs

A graph stores each of your nodes with its type id, your node version, a
fingerprint of your node's description, its settings and its pins. When the
graph opens:

| Situation | What the user sees |
|---|---|
| Your plugin is loaded, same version | The node, as saved |
| Your plugin is loaded, different version or pins | The node, with a warning that it is not the one the graph was saved with |
| Your plugin is not loaded | The node in the not-installed look, its saved settings shown read-only, and **Open Plugin Manager** |

A not-installed node on the path from the data to the loss stops the
compile (*Extension node '...' is not installed. Install and approve the
extension that provides it, or remove the node.*); off the path it is a
warning.

## 9. Loading, unloading and startup

- **Install...** in the Plugin Manager loads one plugin from its folder; **Refresh** loads every plugin in the two plugin folders (see the [quickstart](plugin-quickstart.md#3-load-it-in-the-engine)).
  Loaded plugins are remembered and load again when the Engine starts, unless
  the user unticks **Load when the Engine starts**. **Unload** takes the
  plugin off that list.
- When a plugin is unloaded while a built model still uses its nodes, the
  Engine keeps the library loaded until the last such module is gone. You do
  not need to handle this.
- **Apps > Loaded plugins** lists your plugin with what it adds
  (*2 nodes*); choosing it opens the Node Browser on your nodes.

### Build fingerprint

A trainable node's module is a real `cyxwiz::Module` that the Engine calls
directly, so your plugin must be built the way the Engine was. Since plugin
API 1.2, `CYXWIZ_PLUGIN_ENTRY` also exports your build's fingerprint, for
example

```
sdk=0.2;api=1.2;compiler=msvc-19;stdlib=msvc-stl;build=release;ptr=64;arrayfire=3.10
```

The Engine compares it with its own before it runs any of your code:

| Item | Must match |
|---|---|
| `sdk` | The CyxWiz SDK major and minor version (0.2) |
| `api` | Same major; your minor not newer than the Engine's |
| `compiler` | Compiler family and major version (MSVC 19 covers Visual Studio 2015-2022) |
| `stdlib` | The C++ standard library |
| `build` | Release or Debug: build your plugin in **Release** for a Release Engine |
| `ptr` | 64-bit |
| `arrayfire` | The ArrayFire version, when your plugin includes the backend |

A difference refuses the load with both values, for example *The plugin was
built with build=debug; this Engine with build=release. Rebuild the plugin
against the CyxWiz SDK 0.2 with the same settings.* A plugin built before
API 1.2 has no fingerprint and is refused with *The plugin has no build
fingerprint*: rebuild it. You do not write anything for this: rebuilding
with the current SDK is enough.

## 10. Threads, crashes and limits

| Topic | Rule |
|---|---|
| Threads | `InferOutputShape` and `CreateModule` may be called from any thread; keep them free of shared state |
| Crash isolation | Plugin lifecycle calls, `InferOutputShape` and `CreateModule` run inside the Engine's crash guard. During training, C++ exceptions from `Forward` and `Backward` are caught by the guard above; a memory fault in your code is not, and can close the Engine |
| Tensors | Use `cyxwiz::Tensor` operations. Do not mix in raw ArrayFire arrays: they are column-major, `Tensor` is row-major |
| Remote training | Graphs with extension nodes do not train on a Server Node yet; the Engine refuses with a message |
| Export | Models with extension nodes cannot be exported yet |

## 11. Later

| Topic | Stage |
|---|---|
| The SDK as an optional installer component and a download (today: `cmake --install` from a CyxWiz build, see the quickstart) | P5 |
| API reference for every interface | P5 |
