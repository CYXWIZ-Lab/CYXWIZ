# Plugin quickstart: trainable nodes from C++

Build the template plugin from an installed CyxWiz, check it, load it, train
with its node, and update it after a change. You need no CyxWiz sources:
every installation carries the plugin SDK and the check tool.

> Checked against plugin API 1.2.0 and pycyxwiz 0.2.0.
> Windows x64 is the main platform; Linux and macOS differences are given
> where they exist.

## 1. What you need

| Need | Detail |
|---|---|
| An installed CyxWiz | From the [Releases page](https://github.com/CYXWIZ-Lab/CYXWIZ/releases) (root README, *Install a pre-release*) |
| CMake 3.20 or newer | On `PATH` |
| The C++ compiler the Engine was built with | See [The compiler](#the-compiler). Another compiler builds, but the Engine refuses to load the result |

### Where CyxWiz is installed

| Installed for | Installation folder |
|---|---|
| You, Windows (default) | `%LOCALAPPDATA%\Programs\CyxWiz` |
| All users, Windows | `%ProgramFiles%\CyxWiz` |
| You, Linux | `$XDG_DATA_HOME/cyxwiz`, or `~/.local/share/cyxwiz` when that variable is not set |
| All users, Linux | `/opt/cyxwiz` |
| You, macOS | `~/Applications/CyxWiz` |
| All users, macOS | `/Applications/CyxWiz` |

The installation folder holds the launcher, `cyxwiz-runtime-bootstrapper.exe`
(Linux and macOS: `cyxwiz-runtime-bootstrapper`). The Start Menu entry and
the macOS app start it. Everything else is in `runtime/base/<base>/`:

| In `runtime/base/<base>/` | What it is |
|---|---|
| `cyxwiz-engine` (`.exe`) | The Engine. Start it through the launcher, not directly: the launcher gives it its runtime libraries |
| `cyxwiz-plugin-check` (`.exe`) | Loads a plugin folder the way the Engine does and checks its nodes. Start it through the launcher too ([section 3](#3-check-it)) |
| `sdk/` | The plugin SDK |
| `plugins/` | The Engine's own plugin folder. Do not put your plugins here: CyxWiz replaces this folder when it updates |
| `engine_log.txt` | The Engine's log of the current launch (`engine_log.1.txt` to `engine_log.4.txt` are the four launches before) |

`<base>` is the active base, named by `base_pack_id` in
`runtime/active-runtime.json`; usually it is the only folder in
`runtime/base`, for example `cyxwiz-base-0.2.0-6-win64-x86_64`.

### Find the SDK

Set these shell variables; every command below uses them. Change the first
line for an all-users install or another platform (table above).

PowerShell:

```powershell
$cyxwiz = "$env:LOCALAPPDATA\Programs\CyxWiz"
$base = (Get-Content "$cyxwiz\runtime\active-runtime.json" | ConvertFrom-Json).base_pack_id
$sdk = "$cyxwiz\runtime\base\$base\sdk"
```

Command Prompt (`cmd`) cannot read JSON: list the base folders and type the
name.

```bat
set CYXWIZ=%LOCALAPPDATA%\Programs\CyxWiz
dir /b "%CYXWIZ%\runtime\base"
set SDK=%CYXWIZ%\runtime\base\<the folder name>\sdk
```

Linux and macOS (`bash`, `zsh`):

```bash
cyxwiz=~/.local/share/cyxwiz          # macOS: cyxwiz=~/Applications/CyxWiz
base=$(sed -n 's/.*"base_pack_id": *"\([^"]*\)".*/\1/p' "$cyxwiz/runtime/active-runtime.json")
sdk="$cyxwiz/runtime/base/$base/sdk"
```

The SDK folder holds:

| In the SDK | What it is |
|---|---|
| `include/plugin/`, `include/core/` | The plugin headers |
| `include/cyxwiz/` | The backend headers: `cyxwiz::Tensor`, `cyxwiz::Module`, the built-in layers |
| `lib/` | The backend's import library (`cyxwiz-backend.lib`; `libcyxwiz-backend.so`, `.dylib`). At run time your plugin uses the Engine's own copy |
| `lib/cmake/CyxWizSDK/` | The CMake package: `find_package(CyxWizSDK)`, the target `CyxWiz::PluginSDK`, the function `cyxwiz_plugin_folder()` |
| `template/` | My Nodes: a plugin with one trainable node, Bent Identity, that builds as delivered |
| `examples/python-nodes/` | Python example nodes ([python-nodes.md](python-nodes.md)) |
| `LICENSE`, `NOTICE` | The Apache License 2.0, which covers every file of the SDK |

Every header is listed in [sdk-contents.md](sdk-contents.md). The SDK has
no ArrayFire, ImGui, spdlog or JSON headers, and a trainable node needs
none.

### The compiler

Your plugin shares C++ objects with the Engine, so it must be built with the
Engine's compiler family, its major version and its C++ standard library.
The SDK records the Engine's compiler in its CMake package. Read it with one
of:

```powershell
Select-String "Built with" "$sdk\lib\cmake\CyxWizSDK\CyxWizSDKConfig.cmake"
```

```bat
findstr /C:"Built with" "%SDK%\lib\cmake\CyxWizSDK\CyxWizSDKConfig.cmake"
```

```bash
grep "Built with" "$sdk/lib/cmake/CyxWizSDK/CyxWizSDKConfig.cmake"
```

It prints a line such as `# Built with: MSVC 19.44.35211.0 on Windows`.

| Platform | Use | Refused or unsafe |
|---|---|---|
| Windows | x64 only. Visual Studio (MSVC) 2019 or later, and **not newer than the "Built with" version**. MSVC 19.44 is Visual Studio 2022 version 17.14; 19.50 is Visual Studio 2026 | clang-cl and MinGW are refused at load. A newer Visual Studio than the Engine's builds and loads, but your plugin may then need a newer Visual C++ runtime than the one the Engine ships, and can crash |
| Linux | GCC with the same major version (`GNU 11.4.0` means `g++-11`), libstdc++ | Clang; another GCC major |
| macOS | Apple clang with the same major version (`AppleClang 16.0.0` means Xcode 16), libc++, and the architecture of your CyxWiz download (`x86_64` for macOS Intel, `arm64` for Apple Silicon) | Another clang major. A library of the other architecture fails to load |

The [developer guide](plugin-developer-guide.md#10-build-fingerprint)
explains what the Engine compares.

## 2. Build the template

Copy the template out of the installation (an all-users installation is
read-only) and build it, in a folder of your own.

PowerShell:

```powershell
Copy-Item -Recurse "$sdk\template" my_nodes
cmake -S my_nodes -B my_nodes\build -DCMAKE_PREFIX_PATH="$sdk"
cmake --build my_nodes\build --config Release
```

Command Prompt:

```bat
xcopy /E /I "%SDK%\template" my_nodes
cmake -S my_nodes -B my_nodes\build -DCMAKE_PREFIX_PATH="%SDK%"
cmake --build my_nodes\build --config Release
```

Linux and macOS:

```bash
cp -r "$sdk/template" my_nodes
cmake -S my_nodes -B my_nodes/build -DCMAKE_PREFIX_PATH="$sdk" -DCMAKE_BUILD_TYPE=Release
cmake --build my_nodes/build
```

- **Windows:** CMake finds Visual Studio by itself; any shell works. With
  several versions installed it takes the newest. Choose one with `-G` on
  the first `cmake` line, for example `-G "Visual Studio 17 2022"`; to change
  it later, delete `my_nodes\build` first.
- **Linux, macOS:** choose the compiler on the first `cmake` line, for
  example `-DCMAKE_CXX_COMPILER=g++-11`.

The template's `CMakeLists.txt` is the whole build:

<!-- excerpt: plugins/template/CMakeLists.txt -->
```cmake
cmake_minimum_required(VERSION 3.20)
project(my_nodes LANGUAGES CXX)

# Inside the CyxWiz source tree the SDK targets already exist; elsewhere,
# configure with -DCMAKE_PREFIX_PATH=<the CyxWiz SDK folder>.
if(NOT TARGET CyxWiz::PluginSDK)
    find_package(CyxWizSDK REQUIRED)
endif()

add_library(my_nodes SHARED my_nodes_plugin.cpp)
target_link_libraries(my_nodes PRIVATE CyxWiz::PluginSDK)

# <build>/my_nodes: plugin.json and bin/my_nodes.dll, ready for the Engine's
# Plugin Manager > Install...
cyxwiz_plugin_folder(my_nodes)
```

| Line | Why it is there |
|---|---|
| `cmake_minimum_required(VERSION 3.20)` | Also turns on the CMake behaviour (policy CMP0091) that lets `cyxwiz_plugin_folder` choose the Visual C++ runtime. Without it a Debug build is refused |
| `if(NOT TARGET CyxWiz::PluginSDK)` | The same file builds inside the CyxWiz source tree, where the target already exists |
| `CyxWiz::PluginSDK` | The headers, the backend's import library, and C++20 |
| `cyxwiz_plugin_folder(my_nodes)` | Makes `build/my_nodes/` a plugin folder (below) and builds every configuration with the release Visual C++ runtime, which the Engine uses |

The result is the plugin folder `my_nodes/build/my_nodes/`:

```
my_nodes/build/my_nodes/
  plugin.json
  bin/my_nodes.dll          (Linux: bin/libmy_nodes.so, macOS: bin/libmy_nodes.dylib)
```

**Debug builds.** `cmake --build my_nodes\build --config Debug` builds
without optimisation and with debug information (`my_nodes.pdb` beside the
DLL), and still loads: `cyxwiz_plugin_folder` gives it the release C++
runtime. A library built without `cyxwiz_plugin_folder` uses the debug
runtime in Debug and is refused (*built with build=debug*). On Linux and
macOS a Debug build (`-DCMAKE_BUILD_TYPE=Debug`) loads unless you define
`_GLIBCXX_DEBUG`.

## 3. Check it

Run the check through the launcher, which gives the check tool the Engine's
runtime libraries and keeps your current folder, so relative paths work:

```powershell
& "$cyxwiz\cyxwiz-runtime-bootstrapper.exe" --plugin-check my_nodes\build\my_nodes
```

```bat
"%CYXWIZ%\cyxwiz-runtime-bootstrapper.exe" --plugin-check my_nodes\build\my_nodes
```

```bash
"$cyxwiz/cyxwiz-runtime-bootstrapper" --plugin-check my_nodes/build/my_nodes
```

PowerShell needs the `&` to run a quoted path. Started directly,
`cyxwiz-plugin-check` does not find the Engine's libraries. Several folders
may follow `--plugin-check`.

For each folder the check reads `plugin.json`, loads the library the way the
Engine does (API version, build fingerprint), initialises the plugin, and
checks each trainable node with its default settings. The Engine's own log
lines, with a time stamp, are mixed in; the result lines are:

```
D:\work\my_nodes\build\my_nodes
  plugin com.example.my-nodes 1.0.0 (API 1.2.0)
  ok   node com.example.my-nodes:BentIdentity: Passed: largest relative error 5.8e-06 in input (tolerance 2.0e-02)
cyxwiz-plugin-check: all plugins pass
```

| Line | Meaning |
|---|---|
| `FAIL plugin.json: ...` | The manifest is missing, has no `id`, or is not valid JSON |
| `FAIL the plugin did not load: ...` | The reason: API version, build fingerprint, a missing library file, or the operating system's error for the library |
| `permission X: granted for this check only (the Engine asks the user)` | `plugin.json` declares a permission the user must approve in the Engine; the check grants it for this run only and saves nothing |
| `FAIL the plugin did not initialise: ...` | `OnLoad` or `OnInitialize` returned `false`, threw or crashed |
| `FAIL node not added: ...` | The Engine refused one of your nodes; the reason says why ([developer guide](plugin-developer-guide.md#nodes-the-engine-refuses)) |
| `FAIL node ...: refuses a [3] input with its default settings: ...` | `InferOutputShape` refused one sample of shape `[3]` |
| `ok   node ...: Passed ...`, `FAIL node ...: Failed ...` | The gradient check ([developer guide](plugin-developer-guide.md#8-the-gradient-check)) |
| `FAIL the plugin registered no nodes` | The check expects at least one node |

The exit code is 0 when every folder passes, 1 otherwise, 2 without a
folder.

## 4. Load it in the Engine

1. Start CyxWiz and open the **Plugin Manager** (sidebar, or
   **Apps > Plugin Manager**).
2. Press **Install...**, enter the plugin folder (the one that holds
   `plugin.json`, here `my_nodes/build/my_nodes`) or pick it with **Browse**,
   and press **Install**.
3. The card **My Nodes** appears: *Adds: 1 trainable node (Activations)*,
   with **Load when the Engine starts** ticked.

**Install...** loads the plugin from the folder you name and remembers that
folder; it copies nothing. Keep the folder where it is, or the plugin is not
found at the next start.

When something goes wrong, Install says what:

| Message | What to do |
|---|---|
| *Failed to load plugin from &lt;folder&gt;: &lt;reason&gt;* | Fix the reason (fingerprint, API version, missing library); the check prints the same |
| *Plugin loaded but could not be initialized: &lt;folder&gt;* and the error | `OnLoad` or `OnInitialize` failed |
| *Plugin installed, but N of its nodes were not added:* and one line per node | The plugin loaded; those nodes were refused ([developer guide](plugin-developer-guide.md#nodes-the-engine-refuses)) |
| *&lt;name&gt; &lt;version&gt; is already loaded from &lt;folder&gt;. To load a new build: Unload it, rebuild, then Install again. If a model still uses its nodes, restart the Engine instead of Unload.* | See [section 6](#6-change-rebuild-reload) |

A plugin that declares a permission the user must approve (file system,
network, system commands, Python) shows an approval dialog before it starts.

### The user plugin folder

**Refresh** in the Plugin Manager loads every plugin it finds, at any depth,
in two folders, and skips plugins that are already loaded:

| Folder | Where |
|---|---|
| Your user plugin folder | `%APPDATA%\cyxwiz\plugins\` on Windows; `~/.cyxwiz/plugins/` on Linux and macOS. Create it if it does not exist |
| The Engine's own `plugins` folder | `runtime/base/<base>/plugins/`, for CyxWiz's own plugins; replaced when CyxWiz updates |

To make a plugin independent of your build folder, copy the plugin folder
(`plugin.json` and `bin/`) into your user plugin folder and press
**Refresh**, or **Install...** from there. Plugins loaded by **Refresh** are
remembered for the next start too. At start the Engine loads the plugins it
remembers; it does not search the two folders by itself.

## 5. Use the node

- In the **Apps** menu, under *Loaded plugins*, **My Nodes** opens the Node
  Browser on the plugin's node. **Bent Identity** is also under
  **Activations** in the node editor.
- Drag it onto the canvas between two layers, as you would ReLU.
- Select it. **Properties** shows the plugin and version and a
  **Gradient check** card. Press **Check gradient**: *Verified* with the
  largest error, or *Failed* with what to do. **Details** shows the numbers.
- Train as usual.

Save the graph. Opened on a computer without the plugin, the node shows as
**Not installed**, keeps its saved settings and pins, and the graph does not
train until the plugin is loaded.

## 6. Change, rebuild, reload

The Engine keeps a loaded library in memory, and Windows locks the file while
it is loaded: building into the same folder then fails with
*LNK1104: cannot open file '...\bin\my_nodes.dll'*. **Refresh** does not
reload a plugin that is already loaded, and **Install...** refuses it with
the *already loaded* message. The loop is:

1. In the Plugin Manager, press **Unload** on the plugin's card. This also
   takes it off the plugins loaded at start.
2. Rebuild (`cmake --build my_nodes\build --config Release`).
3. Run the check ([section 3](#3-check-it)).
4. **Install...** the same folder again.

While a model built from a graph with the plugin's nodes still exists (a
training run, a trained model kept for testing or export), **Unload** removes
the nodes but the Engine keeps the library loaded until that model is gone,
so the rebuild still fails. Then close the Engine, rebuild and start it
again: the plugin loads at start with the new build, as long as you did not
press **Unload**.

On Linux and macOS the file is not locked and the rebuild succeeds, but the
Engine runs the old code until you Unload and Install again.

## 7. Make it yours

1. In `plugin.json` set your own `id` (lower case: `a-z`, `0-9`, `.`, `-`,
   `_`; for example `com.yourname.my-nodes`), `name`, `version`, `author`
   and `license`. The `id` and each node's type name are saved in every graph
   that uses your nodes: choose them once.
2. To rename the library, change `my_nodes` in `CMakeLists.txt` (`project`,
   `add_library`, `target_link_libraries`, `cyxwiz_plugin_folder`) and the
   three library paths in `plugin.json` (`bin/<name>.dll`,
   `bin/lib<name>.so`, `bin/lib<name>.dylib`).
3. Replace `BentIdentityModule` with your module, following the
   [developer guide](plugin-developer-guide.md) and the
   [node contract](node-contract.md).
4. The template's source starts with `SPDX-License-Identifier: Apache-2.0`;
   replace it with your own licence. Your plugin may use any licence.

A second example, with a setting and a learnable weight (Softsign and Snake),
is in the CyxWiz repository on GitHub
(<https://github.com/CYXWIZ-Lab/CYXWIZ>), folder
`plugins/examples/activation_nodes`. It is not in the SDK. Copy its three
files (`CMakeLists.txt`, `plugin.json`, `activation_nodes_plugin.cpp`) into a
folder and build that folder like the template; its plugin folder is
`build/activation_nodes/`.
