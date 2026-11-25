---
title: Menus
description: The twelve menus of the Engine's menu bar, what each holds, and the project and account chips.
type: reference
area: general
audience: user
status: stable
since: 1.0.7
owner: main-window
last_reviewed: 2026-11-24
related:
  - index.md
  - toolbar.md
  - status-bar.md
  - ../reference/shortcuts.md
redirect_from:
  - menus.md
---

The menu bar sits under the window's title: twelve menus on the left, the
open project and your account at the right end.

## How menu items behave

- An item that cannot run now is greyed out. Hover it for the reason; the
  [status bar](status-bar.md) shows the same reason.
- An item marked **(planned)** is not built yet and stays greyed out.
- Every menu command is also in the **Command Palette** (Ctrl+P), with its
  shortcut and menu path.
- Shortcuts are listed beside each item and in
  [Keyboard shortcuts](../reference/shortcuts.md).

## The menus at a glance

| Menu | What it holds |
| --- | --- |
| [File](#file) | Projects, windows, scripts, saving, model files, Preferences, Restart, Exit |
| [Edit](#edit) | Undo and redo, the clipboard, find and replace, line tools for scripts |
| [View](#view) | Command Palette, panels, layout, windows, themes, icons, minimaps, full screen |
| [Nodes](#nodes) | Add layers, group, duplicate, delete, freeze, make a Python node |
| [Train](#train) | Compile, Local Debug, train, test, evaluate, dashboards, search |
| [Data](#data) | Import data, statistics, Explore and Transform shortcuts into Data Studio |
| [Tools](#tools) | Model, evaluation, statistics, maths, signal, time series, text, utilities, monitoring |
| [Script](#script) | New, open, run and stop scripts, the Python console |
| [Deploy](#deploy) | Server connection, deploy, model serving, export and convert |
| [Simulation](#simulation) | The open graph's simulation run and simulation plugins |
| [Apps](#apps) | Plugin panels, loaded plugins, Python node bundles, Plugin Manager |
| [Help](#help) | Tutorials, shortcuts, documentation, logs, About |

## File

- **New Project...** (Ctrl+Shift+N), **Open Project...** (Ctrl+Shift+O),
  **Open Recent** (with **Clear Recent Projects**), **Close Project**,
  **Show Project Folder**. The **Create a new project** dialog is described
  on [The start page](start-page.md#create-a-new-project).
- **New Window** starts a second Engine window; **Open Project in New
  Window** opens a recent project in one.
- **New Script...** (Ctrl+N) and **Open Script...** (Ctrl+O) for the
  Script Editor.
- **Save** (Ctrl+S) saves the project settings and layout, every open
  script, and the graph (asking for a name the first time). **Save As...**
  (Ctrl+Shift+S) copies the project under a new name. **Save All**
  (Ctrl+Alt+S) saves every open script and the graph. **Auto Save** (a
  check item) saves open scripts and a saved graph every 60 seconds; an
  untitled graph waits until you save it once.
- **Open Model File...** (Ctrl+I) reads a model file and opens its graph.
- **Export Model**: **CyxWiz package (.cyxmodel)** (Ctrl+E), **Weights only
  (.safetensors)**, **ONNX model (.onnx)** (greyed out when the build has no
  ONNX support).
- **Preferences...** (Ctrl+,), see [Preferences](preferences.md).
- **Restart** saves, then restarts the Engine with the project.
- **Exit** (Alt+F4).

**Save**, **Save As...**, **Close Project**, **Show Project Folder** and
**Restart** need an open project. Greyed out, they say "Open or create a
project first."

## Edit

The header line says which editor the commands act on: **Editing: Script
Editor**, **Editing: Studio canvas** or **Editing: nothing focused**. Click
in the Script Editor or on the canvas first.

- **Undo** (Ctrl+Z), **Redo** (Ctrl+Y; on the canvas also Ctrl+Shift+Z).
- **Cut**, **Copy**, **Paste**, **Delete**, **Select All**: text in the
  Script Editor, nodes on the canvas.
- **Find...** (Ctrl+F): inline find in a script, the find bar on the
  canvas.
- Script Editor only: **Replace...** (Ctrl+H), **Go to Line...** (Ctrl+G),
  **Lines** (duplicate, move, indent, join, sort), **Transform Text**
  (UPPERCASE, lowercase, Title Case), **Toggle Line Comment** (Ctrl+/),
  **Toggle Block Comment** (Shift+Alt+A).
- **Find in Files...** (Ctrl+Shift+F) and **Replace in Files...**
  (Ctrl+Shift+H) search a folder. Replace skips files with unsaved edits.

## View

- **Command Palette...** (Ctrl+P).
- **Panels**: every panel by group (Workspace, Data, Training, Network,
  Extensions), the same list as the tool sidebar. See
  [The main window](index.md).
- **Layout**: **Save Layout**, **Reset to Default**. **Window**: **Move to
  its own window**, **Dock back into the main window**. See
  [Layouts and docking](layouts-and-docking.md).
- **Theme** (colour presets), **Theme Editor...**, **Node Icon Pack**
  (also in **Preferences > Appearance**).
- **Minimaps**: **Studio Minimap** (M on the canvas) and **Script Editor
  Minimap**.
- **Full Screen** (F11): the Engine window without its frame over the
  whole monitor. F11 again puts it back. It is not exclusive full screen, so
  a window of its own can sit on top.

## Nodes

- **Add Layer**: Dense / Linear, Convolutional, Pooling, Dropout, Batch
  Normalization, Attention.
- **Group Selected** (Ctrl+G), **Ungroup** (Ctrl+Shift+G), **Duplicate**
  (Ctrl+D), **Delete Selected** (Delete).
- **Freeze**: keep chosen layers' weights while the rest trains
  (**Freeze Selected**, **Unfreeze Selected**, **Freeze All but the Head**,
  **Freeze All but the Last N Layers...**, **Unfreeze All**, and
  **Freeze Every Layer Loaded from the File** for a graph that loads
  weights).
- **Make a Python node...** opens the Python node maker.

More ways to add nodes:
[Find and add nodes](../user-guide/graph-editor/find-and-add-nodes.md).

## Train

- **Compile Graph** (F7), **Local Debug** (F6), **Start Training** (F5),
  **Pause Training** / **Resume Training**, **Stop Training**. See
  [Compile and run](../user-guide/graph-editor/compile-and-run.md).
- **Test**: **Run Test** (F8), **Run Quick Test** (the first 10 batches),
  **View Test Results**, **Compare Test Results...**, **Export Test
  Report...**, **Load Checkpoint for Testing...**. See
  [Testing a model](../user-guide/evaluation/test-a-trained-model.md).
- **Evaluate**: **Cross-validate**, **Learning Curve**.
- **Training Dashboard**, **RL Training Dashboard**, **Optimizer
  Settings...** (selects the graph's optimizer node in Properties),
  **Hyperparameter Search**.

## Data

- **Import Dataset...** and **Create Custom Dataset...** both add a Data
  Input node and open its dialog.
- **Dataset Statistics** opens Data Studio's profile.
- **Explore**: **Missing Value Analysis**, **Outlier Detection**,
  **Correlation Matrix** (Data Studio tabs).
- **Transform**: each item adds a step to Data Studio's **Prepare**
  (Min-Max, Z-Score, Log, Box-Cox, or a comparison of every scaling method).
- **Tokenization** opens Data Studio's Text tab.
- **Augment (planned)**.

See [Data Studio](../user-guide/data-studio/overview.md).

## Tools

Each item opens a tool window or a Data Studio tab on the right view:

- **Model** (the Model window: summary, learning rate finder, explanations,
  training health, embedding projector, and **Neural Architecture
  Search**), **Evaluation**, **Clustering and Projection**, **Statistics**.
- **Linear Algebra**, **Signal Processing**, **Optimization and
  Calculus**, **Time Series**, **Text** (including **Language Model
  Generation**), **Utilities** (calculator, unit converter, random, hash,
  JSON, regex).
- **Monitoring**: **System**, **Memory**, **Loaded Data**, **Frames**,
  **Performance Profiler**, **Clear Caches...**, **Free Unused Memory**.
- **Diagnostics**: **Log Frame-rate Changes**, **Log Every Python
  Command**, **Frame Time Overlay** (check items, saved).

Pages: [Model tools](model-window.md),
[Linear algebra](../usage/linear_algebra.md),
[Monitoring](../usage/monitoring.md), [Utilities](../usage/utilities.md).

## Script

**New Script...** (Ctrl+N), **Open Script...** (Ctrl+O), **Run Script** (F5
in the Script Editor), **Stop Script** (Shift+F5 in the Script Editor),
**Open Python Console** (F12).

## Deploy

- **Connect to Server...** and **Deploy to Server Node...** need you to be
  signed in; signed out, **Login Required** asks you to sign in first.
- **Model Serving** serves the trained model on this computer.
- **Export Model** (the same submenu as in File).
- **Convert Model**: **Binary to Directory...**, **Directory to
  Binary...**.
- **Quantize** (INT8, INT4, FP16) is planned.

## Simulation

When the open graph has simulation nodes, the menu starts with **Graph
run: <graph name>** and **Run Sim** (or **Resume**), **Pause**, **Step
Once**, **Stop**, **Clear Traces**. Each loaded simulation plugin adds its
own commands under its name. **Simulation Monitor** shows or hides that
panel. See [Graph simulation](../usage/graph_simulation.md).

## Apps

Plugin panels by category (check items show or hide them), **Loaded
plugins** (a plugin with nodes shows them in the Nodes panel), **Python
node bundles** (with their node count and approval), and **Plugin
Manager**. **No plugins loaded** when there are none. See
[Your own nodes](../usage/extension_nodes.md).

## Help

- **Interactive Tutorials**: **Getting Started**, **Your First Model**,
  **Training Basics**, **Browse All Tutorials...** (a tick marks the ones
  you finished).
- **Keyboard Shortcuts...** opens **Preferences > Shortcuts**.
- **Documentation** (F1), **API Reference**, **What's New**: open the user
  guide, the Python API reference and the release notes in your browser,
  from the CyxWiz repository on GitHub.
- **Known Issues** opens `KNOWN_ISSUES.md`, which ships beside the Engine: a short list that points to [Known issues](../known-issues.md)
  (greyed out when it is not there).
- **Report Issue...** opens the GitHub issues page.
- **Open Logs Folder** shows `engine_log.txt` in the file manager, to
  attach to a bug report.
- **Check for Updates... (planned)**.
- **About CyxWiz...**: version, build, compute device, Python and licence,
  with **Copy details**.

## The right end of the menu bar

- **The project chip** (when a project is open): a folder icon and the
  project's name. Hover for the project file's path. Click it for **Show in
  Explorer** (the project folder in the file manager), **Copy Path** and
  **Close Project**.
- **The account chip**, top-right corner:
  - signed out, **Sign in**: its menu has **Sign in...**, which opens the
    sign-in dialog;
  - signed in, your initials, your name and your plan. Hover for your
    email. The menu has **Account Settings...**, **Servers...**
    (**Preferences > Servers**) and **Sign out**.

Of the menu commands, only **Deploy > Connect to Server...** and **Deploy >
Deploy to Server Node...** ask you to sign in. Building and training graphs
on your computer works signed out.
