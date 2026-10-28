# My Nodes: a CyxWiz plugin template

The smallest C++ plugin with a trainable node (Bent Identity). Copy this
folder, rename it, and change the node.

## Build

```
cmake -S . -B build -DCMAKE_PREFIX_PATH=<CyxWiz SDK folder> -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```

(`CMAKE_BUILD_TYPE` matters on Linux and macOS, whose generators build one
configuration; Visual Studio uses `--config`.)

Use the compiler the Engine was built with: the SDK names it in
`<CyxWiz SDK folder>/lib/cmake/CyxWizSDK/CyxWizSDKConfig.cmake`, on the line
`# Built with:`. On Windows that is Visual Studio (MSVC), x64, no newer
than that version. A Debug build works too: the SDK builds it with the
release C++ runtime, so it loads in the Engine and you can debug it.

`build/my_nodes/` is the plugin folder: `plugin.json` and
`bin/my_nodes.dll` (`bin/libmy_nodes.so` on Linux, `.dylib` on macOS).

## Check

Run the plugin check of your CyxWiz installation on the plugin folder:

```
<CyxWiz installation>/cyxwiz-cli plugin-check build/my_nodes
```

It loads the folder the way the Engine does and runs the gradient check on
each node (in a source build: `bin/Release/cyxwiz-plugin-check build/my_nodes`).
The quickstart gives the command for each shell and explains every line the
check prints.

## Load

In the Engine: Plugin Manager > Install..., choose `build/my_nodes`. The
Engine remembers that folder and loads the plugin from it at every start
(nothing is installed elsewhere), so keep it there, or copy the folder into
your user plugin folder
(`%APPDATA%\cyxwiz\plugins\`, `~/.cyxwiz/plugins/`). Bent Identity appears
under Activations in the node editor. To load a new build: rebuild (the
Engine runs a copy of the library, so the build output is never locked),
then Unload the plugin and Install the folder again.

## Documentation

The guides are in the CyxWiz repository on GitHub,
<https://github.com/CYXWIZ-Lab/CYXWIZ>, folder `docs/extending`: start with
`plugin-quickstart.md`, then `plugin-developer-guide.md`.

The template is licensed under the Apache License 2.0 (the SDK's `LICENSE`
and `NOTICE`). Your plugin may use any licence you choose.
