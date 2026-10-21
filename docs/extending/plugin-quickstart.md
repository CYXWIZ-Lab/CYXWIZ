# Plugin quickstart: trainable nodes from C++

Build the example plugin that ships with CyxWiz, load it, and train a model
with its nodes. Then copy it to start your own.

> Checked 2026-10-17 against plugin API 1.2.0. Sections 1-4 use the example
> inside the CyxWiz source tree; section 5 builds your own plugin anywhere,
> against the CyxWiz plugin SDK.

## 1. The example plugin

`plugins/examples/activation_nodes/` adds two activations CyxWiz does not
ship:

| Node | Formula | Settings | Learns |
|---|---|---|---|
| Softsign | x / (1 + \|x\|) | none | nothing |
| Snake | x + sin²(a·x) / a | `initial_a` (number, default 1.0, must be greater than 0) | `a` |

The folder holds:

| File | What it is |
|---|---|
| `plugin.json` | The manifest: id, name, version, API version, capability, permission, library path |
| `activation_nodes_plugin.cpp` | The plugin and its two modules |
| `CMakeLists.txt` | Builds `bin/activation_nodes.dll` (`.so`, `.dylib`) beside `plugin.json` |

The same source is loaded by the test `test_plugin_tensor_nodes`, so the code
shown in these guides is code that builds and passes.

## 2. Build it

The root `CMakeLists.txt` adds the example when the Engine is built:

```cmake
if(CYXWIZ_BUILD_ENGINE)
    add_subdirectory(plugins/examples/activation_nodes)
```

So a normal Engine build also builds the plugin:

```bash
cmake --build <build folder> --config Release --target activation_nodes
```

The library lands in `plugins/examples/activation_nodes/bin/`.

## 3. Load it in the Engine

1. Open the **Plugin Manager** (sidebar, or **Apps > Plugin Manager**).
2. Press **Install...**, enter the plugin folder
   (`<repository>/plugins/examples/activation_nodes`, the folder that holds
   `plugin.json`) or pick it with **Browse**, and press **Install**.
3. The card **Example Activation Nodes** appears. It says *Adds: 2 trainable
   nodes (Activations)* and has **Load when the Engine starts** ticked.

**Refresh** loads every plugin found, at any depth, in the two plugin
folders:

| Folder | Where |
|---|---|
| `plugins` beside the Engine program | In a source build `<build folder>/bin/Release/plugins/`; in an installed Engine, its install folder. The Engine always works from its own program folder, whatever folder you start it from |
| Your user plugin folder | `%APPDATA%\cyxwiz\plugins\` on Windows, `~/.cyxwiz/plugins/` elsewhere |

Copying the plugin folder into one of them also works: press **Refresh**.

From now on the plugin loads by itself when the Engine starts. Untick the box
to stop that; **Unload** also takes it off the list.

## 4. Use the nodes

- **Apps > Loaded plugins > Example Activation Nodes** opens the Node
  Browser on the plugin's two nodes.
- Drag **Snake** onto the canvas between two layers, as you would ReLU.
- Select it. **Properties** shows the plugin and version, the `initial_a`
  field with its description, and a **Gradient check** card.
- Press **Check gradient**. The Engine compares the node's backward step with
  a numerical gradient: *Verified* with the largest error, or *Failed* with
  what to do. **Details** shows the numbers.
- Train as usual. Snake's `a` is updated by the optimizer like any weight.

Save the graph. If you open it on a computer without the plugin, Snake shows
as **Not installed**, keeps its saved settings and pins, and the graph does
not train until the plugin is loaded.

## 5. Start your own plugin

Your plugin does not need the CyxWiz sources: it builds against the plugin
SDK, a folder with the headers, the backend's import library, a CMake
package and a template.

### Get the SDK

Every CyxWiz installation carries it, in the `sdk` folder of the Engine's
runtime:

| Installed for | SDK folder |
|---|---|
| You (Windows default) | `%LOCALAPPDATA%\Programs\CyxWiz\runtime\base\<base>\sdk` |
| All users (Windows) | `%ProgramFiles%\CyxWiz\runtime\base\<base>\sdk` |
| Linux, macOS | `runtime/base/<base>/sdk` under the CyxWiz installation folder |

`<base>` is the one folder in `runtime\base`, for example
`cyxwiz-base-0.2.0-6-win64-x86_64`. Copy the SDK somewhere of your own if you
like: it is a plain folder, under the Apache License 2.0 (its `LICENSE` and
`NOTICE`), so plugins you build with it can use any licence.

From a Release build of the CyxWiz sources you can also install it yourself:

```bash
cmake --install <build folder> --config Release --component plugin_sdk --prefix <SDK folder>
```

| In the SDK folder | What it is |
|---|---|
| `include/` | The plugin headers (`plugin/...`) and the backend headers (`cyxwiz/...`). They need only the C++ standard library |
| `lib/` | The backend's import library; at run time your plugin uses the Engine's copy |
| `lib/cmake/CyxWizSDK/` | `find_package(CyxWizSDK)`: the target `CyxWiz::PluginSDK` and `cyxwiz_plugin_folder()` |
| `template/` | My Nodes: one trainable node (Bent Identity) that builds as delivered |

### Build the template

```bash
cp -r <SDK folder>/template my_nodes
cmake -S my_nodes -B my_nodes/build -DCMAKE_PREFIX_PATH=<SDK folder>
cmake --build my_nodes/build --config Release
```

The template's `CMakeLists.txt` is the whole build:

```cmake
find_package(CyxWizSDK REQUIRED)
add_library(my_nodes SHARED my_nodes_plugin.cpp)
target_link_libraries(my_nodes PRIVATE CyxWiz::PluginSDK)
cyxwiz_plugin_folder(my_nodes)
```

`cyxwiz_plugin_folder` makes `build/my_nodes/` the plugin folder:
`plugin.json` and `bin/my_nodes.dll`. Use the Engine's compiler family
(Visual Studio 2019 or later on Windows). A **Debug** build also loads: the
SDK builds every configuration with the release C++ runtime, as the Engine
uses, and keeps the debug information.

### Check it, load it

In an installed CyxWiz, run the check through the CyxWiz launcher, which
gives it the Engine's runtime libraries:

```bash
"%LOCALAPPDATA%\Programs\CyxWiz\cyxwiz-runtime-bootstrapper.exe" --plugin-check my_nodes/build/my_nodes
```

(with a source build: `<build folder>/bin/Release/cyxwiz-plugin-check my_nodes/build/my_nodes`).
It loads the folder the way the Engine does (manifest, build fingerprint,
permissions) and runs the gradient check on every trainable node:

```
  ok   node com.example.my-nodes:BentIdentity: Passed: largest relative error 5.8e-06 in input (tolerance 2.0e-02)
cyxwiz-plugin-check: all plugins pass
```

Then in the Engine: **Plugin Manager > Install...** and choose
`my_nodes/build/my_nodes`. Bent Identity appears under Activations.

### Make it yours

1. In `plugin.json` give it its own `id` (lower case: `a-z`, `0-9`, `.`,
   `-`, `_`; for example `com.yourname.my-nodes`), `name` and `version`; if
   you rename the library target, change the library paths too.
2. Replace `BentIdentityModule` with your module, following the
   [developer guide](plugin-developer-guide.md) and the
   [node contract](node-contract.md). The example plugin
   (`plugins/examples/activation_nodes/`) shows a node with a setting and a
   learnable weight; it builds against the SDK the same way.
3. Rebuild, run the check, and in the Engine press **Refresh**.
