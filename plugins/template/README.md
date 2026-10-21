# My Nodes: a CyxWiz plugin template

The smallest C++ plugin with a trainable node (Bent Identity). Copy this
folder, rename it, and change the node.

## Build

```
cmake -S . -B build -DCMAKE_PREFIX_PATH=<CyxWiz SDK folder>
cmake --build build --config Release
```

Use the same compiler family as the Engine (Visual Studio 2019 or 2022 on
Windows). A Debug build works too: the SDK builds it with the release C++
runtime, so it loads in the Engine and you can debug it.

`build/my_nodes/` is the plugin folder: `plugin.json` and
`bin/my_nodes.dll`.

## Check

```
<CyxWiz installation>/cyxwiz-runtime-bootstrapper --plugin-check build/my_nodes
```

loads the folder the way the Engine does and runs the gradient check on
each node (in a source build: `bin/Release/cyxwiz-plugin-check build/my_nodes`).

## Load

In the Engine: Plugin Manager > Install..., choose `build/my_nodes`. Bent
Identity appears under Activations in the node editor.

The full guide is `docs/extending/plugin-developer-guide.md` in the CyxWiz
repository.

The template is licensed under the Apache License 2.0 (the SDK's `LICENSE`
and `NOTICE`). Your plugin may use any licence you choose.
