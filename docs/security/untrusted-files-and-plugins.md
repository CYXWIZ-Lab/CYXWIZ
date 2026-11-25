---
title: Plugins, Python nodes and opened files
description: What a C++ plugin, a Python node bundle, a script and an opened graph, project or weight file can do on your computer, and how to stay safe.
type: explanation
area: security
audience: user
status: stable
since: 1.0.7
owner: security
last_reviewed: 2026-11-25
related:
  - index.md
  - ../extending/python-nodes.md
  - ../extending/plugin-developer-guide.md
  - ../user-guide/training/start-from-pretrained-weights.md
redirect_from: []
---

Some things you load into CyxWiz carry code; most carry only data. Code runs
with your user's rights. Data is parsed and never run. This page says which
is which and what the Engine checks first.

## At a glance

| You load | Carries code? | Runs when | Can |
| --- | --- | --- | --- |
| C++ plugin (folder with `plugin.json` and a DLL, `.so` or `.dylib`) | Yes, native | When it is loaded (Plugin Manager, or at startup if you set it to) | Anything your user can do |
| Python node bundle (`<name>.cyxnode/` with `node.json` and `node.py`) | Yes, Python | Only after you approve that exact code | Anything your user can do |
| Python script, notebook cell, console command | Yes, Python | When you run it | Anything your user can do |
| Python startup scripts (Python settings) | Yes, Python | At the start of each Python session | Anything your user can do |
| `.cyxgraph` graph, project folder | No | Never | Name nodes, settings and data files |
| Dataset files (CSV, Parquet, images, audio, and so on) | No | Never | Hold data |
| `.pth` / `.pt` PyTorch state-dict, `.safetensors` | No | Never | Hold tensors |
| `.cyxmodel` trained model | No | Never | Hold the model and its weights |

## C++ plugins

A plugin is a native library that the Engine loads into its own process.

**Where the Engine looks:** the `plugins` folder next to the Engine,
`%APPDATA%\cyxwiz\plugins` on Windows (`~/.cyxwiz/plugins` elsewhere), and
the plugin packs the installer activated. Finding a plugin does not load it.
Only plugins you load in the Plugin Manager, or set to load at startup, are
loaded.

**What the Engine checks, in order:**

1. It reads `plugin.json` (id, version, the permissions it asks for).
2. It copies the library to a private location and loads that copy.
3. It checks the signature in `plugin.json`. A signature that does not
   verify stops the load ("Invalid signature"). A missing signature loads
   with a warning in the log ("unsigned").
4. It checks that the plugin was built the way this Engine was (the build
   fingerprint). A mismatch refuses the plugin before any other plugin
   function is called.
5. Before the plugin starts, it asks you to approve the dangerous permissions
   it requests. Your choice is saved per plugin id and version in
   `%APPDATA%\cyxwiz\plugin_permissions.json` (`~/.cyxwiz/` elsewhere).
6. Each call into the plugin (load, render, training hooks) runs inside a
   guard that catches C++ exceptions and crashes (access violations, stack
   overflow), so a faulty plugin is reported instead of closing the Engine.

**Permissions:**

| Permission | Shown as | Kind |
| --- | --- | --- |
| `FileSystem` | File System: "Read and write files on your computer" | Asks you |
| `Network` | Network Access: "Make network connections to remote servers" | Asks you |
| `SystemCommands` | System Commands: "Execute system commands and processes" | Asks you |
| `Python` | Python Execution: "Run arbitrary Python code in the embedded interpreter" | Asks you |
| `GPU` | GPU Compute | Granted |
| `DataRegistry` | Dataset Access | Granted |
| `Training` | Training Hooks | Granted |
| `UIModify` | UI Modification: panels, nodes, commands, preferences | Granted |

A permission decides which **Engine services** the plugin may register with:
without `UIModify` it cannot add panels or nodes, without `DataRegistry` it
cannot add dataset formats or analyses, without `Training` it cannot add
training hooks or trainable layers.

What permissions do not do: they do not limit the plugin's own native code.
A loaded plugin can open files and sockets through the operating system
whatever it declared. Code in a library can also run while the library is
being loaded, before you are asked about permissions.

**Signatures today:** the Engine's signature key is not set yet, so no
signature can verify. Every plugin you use loads as unsigned, and a plugin
that carries a signature is refused. Trust a plugin because you trust where
it came from, not because the Engine accepted it.

## Python node bundles

A bundle is a folder `<name>.cyxnode/` holding `node.json` (the description
of its nodes) and `node.py` (the code).

**Where the Engine looks:** the open project's `nodes` folder,
`%APPDATA%\cyxwiz\nodes` (`~/.cyxwiz/nodes` elsewhere), and the `nodes`
folder of each loaded plugin.

- Reading a bundle never runs its code: the palette lists its nodes from
  `node.json`, and graphs that use them open and save.
- Its code runs only after you approve it in **Run this Python code?** The
  dialog says: "Its code runs inside CyxWiz with your user's rights: it can
  read and change your files. Approve only code you wrote or trust."
- The approval is for that exact code: the bundle id and the SHA-256 hash of
  `node.py`. If `node.py` changes, the Engine asks again. Bundles you create
  in the Engine are approved when they are written. Approvals are kept in
  `%APPDATA%\cyxwiz\node_approvals.json` (`~/.cyxwiz/` elsewhere).

A project you receive can contain a `nodes` folder. Its bundles show as
**Not approved yet** until you approve them.

## Python scripts and the console

Scripts, notebook cells and console commands run in the Engine's embedded
Python with full access, when you run them. `pip install` runs only when you
type it.

The Script Editor's **Security > Enable Sandbox** (off by default) blocks some
built-ins (`exec`, `eval`, `compile`, `input`), limits imports to a list,
denies writing files and stops a script after 60 seconds. It runs in the same
interpreter and its allowed list includes `os` and `sys`, so treat it as a
guard against mistakes, not as protection against hostile code.

Python startup scripts set in the Python settings run at the start of every
Python session. They are part of your own Engine settings, not of a project.

## Graphs, projects and data files

A `.cyxgraph` file is JSON: nodes, their settings and links. A project is a
folder of such files, scripts, data and settings. Opening either runs no
code. A graph can name files to load; they load when you click **Apply** or
start a run.

A graph that uses a Python node runs that node's code only if its bundle is
approved on your computer.

## Weight and model files

PyTorch `.pth` and `.pt` files hold a Python pickle. CyxWiz never runs it:
its own reader knows the few pickle instructions and constructors a
state-dict uses and never imports or calls anything the file names. A file
holding anything other than a plain state-dict is refused with the reason
("the file holds something other than a plain state-dict"). `.safetensors`
files hold no code at all. See
[Start from pretrained weights](../user-guide/training/start-from-pretrained-weights.md).

## Staying safe

- Install plugins and Python bundles only from people you trust; read
  `node.py` before you approve it.
- Approve plugin permissions you understand. Deny **System Commands** and
  **Python Execution** unless the plugin's purpose needs them.
- Try unknown code under a separate user account or in a virtual machine.
- To withdraw your choices, close the Engine and remove the plugin's or
  bundle's entries from the permission and approval files named above.

## Related

- [How CyxWiz handles security](index.md)
- [Python nodes](../extending/python-nodes.md)
- [Plugin developer guide](../extending/plugin-developer-guide.md)
