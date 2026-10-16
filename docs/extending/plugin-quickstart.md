# Plugin quickstart: trainable nodes from C++

Build the example plugin that ships with CyxWiz, load it, and train a model
with its nodes. Then copy it to start your own.

> Checked 2026-10-16 against plugin API 1.1.0. Today a plugin is built
> inside the CyxWiz source tree; a packaged SDK for outside builds comes
> later (stage P5).

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

1. Copy `plugins/examples/activation_nodes/` to a new folder under
   `plugins/`.
2. In `plugin.json` give it its own `id` (lower case: `a-z`, `0-9`, `.`,
   `-`, `_`; for example `com.yourname.my-nodes`), `name`, `version` and
   library path.
3. In `CMakeLists.txt` rename the library target.
4. Add `add_subdirectory(plugins/<your folder>)` next to the example's line
   in the root `CMakeLists.txt`.
5. Replace the two modules with yours, following the
   [developer guide](plugin-developer-guide.md) and the
   [node contract](node-contract.md).
6. Build, press **Refresh**, and run **Check gradient** on each node before
   you train.
