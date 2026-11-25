---
title: The start page
description: The Get started page the Engine shows at launch, its starter graphs, recent projects and project actions.
type: reference
area: general
audience: user
status: stable
since: 1.0.7
owner: main-window
last_reviewed: 2026-11-24
related:
  - index.md
  - ../install/first-launch.md
  - ../user-guide/graph-editor/index.md
redirect_from: []
---

The start page (**Get started**) fills the Engine window when it starts.
From it you open a starter graph, a recent project, or create a new
project. When you pick something, the [main window](index.md) opens.

The Engine skips the start page when you open it with a project on the
command line, or when **Preferences > General > When the Engine starts** is
set to **Reopen the last project** (the default is **Show the start page**).

## The header

- **Get started** and the Engine's **Version**.
- **The Python chip**, top right: whether the Engine found a Python for
  scripts, the Python console and project environments (**Checking
  Python...**, a ready Python with its version, **Choose a Python**, or
  **Python not found**). Click it to open **Python for scripting**, where
  you pick or install an interpreter. Graphs build and train without
  Python; see [First launch](../install/first-launch.md).
- **The search box** under the title filters **Recent Projects** by name or
  folder.

## Task starter graphs

The left column lists five working graphs, each with a small sample
dataset, so **Train** works at once and offline:

| Starter | What it trains on |
| --- | --- |
| **Binary image classification** | Cats or dogs, 60 sample photos |
| **Multiclass image classification** | Handwritten digits 0-9, 1,797 sample 8x8 images, an MLP |
| **Text classification** | Positive or negative review sentences, 3,000 samples |
| **Audio classification** | Spoken digits 0-9, 400 sample clips |
| **Time-series forecasting** | Monthly airline passengers, 144 months |

Select **Open** on a row to load it into CyxWiz Studio **without a
project**. It opens as *Untitled graph*, so **Save** asks for a file name
and the example files are never overwritten. Hover **Open** for the file
it reads.

The starters come from the `examples/cyxgraph` folder beside the Engine,
and their data from `examples/datasets` (sources listed in
`examples/datasets/README.md`). Packages built from the current code
include both folders; the 1.0.7 release packages do not, so there the
column is empty. When the folder is missing, the column says "No starter graphs found: the
examples/cyxgraph folder is not next to this Engine." The same five graphs
are under **Open an example graph** on an empty canvas.

## Recent Projects

Projects you opened before, grouped **This week**, **This month** and
**Older**.

- Click a row to select it. Double-click it, press **Enter**, or select
  **Open** to open it.
- **Actions** on the selected row:
  - **Open in new window**: opens the project in a second Engine window;
  - **Show in folder**: shows the project file in the file manager;
  - **Remove from this list**: forgets it here. The project's files are not
    touched.

## Launch workspace

The right column:

- **Create a new project** opens **Create a new project** on the **Blank
  project** template.
- **Workflow lanes**: **Classic ML workflow**, **Deep Learning workflow**.
- **Domain starters**: **Tabular project**, **Vision project**, **NLP
  project**.
- **File actions**: **Open a project** (a `.cyxwiz` project file) and
  **Open a project folder** (the folder that holds one). **Clone a
  repository** is shown but not available.

Every lane and domain starter opens the same **Create a new project**
dialog with that template picked. A template sets only the suggested
project name and description: the new project starts with an empty canvas,
whatever the lane's text says. To start from a working graph, use a task
starter or the [Pattern Browser](pattern-browser.md).

### Create a new project

| Field | Meaning |
| --- | --- |
| **Template** | Blank project, Classic ML workflow, Deep Learning workflow, Tabular project, Vision project, NLP project |
| **Name** | The project and folder name. It cannot contain characters a file name cannot use, end with a dot or a space, or be a name Windows reserves |
| **Location** | Where the project folder goes (default `Documents/CyxWiz Projects`); **Browse...** picks a folder |
| **Creates** | The folder that will be made |

**Create** stays disabled with the reason ("Enter a project name.",
"Choose a location.", "A folder with this name already exists here.")
until the fields are valid. The project folder holds the project file
(`<name>.cyxwiz`) and the folders `cyxgraph`, `scripts`, `models`,
`datasets`, `artifacts`, `checkpoints`, `exports`, `plugins` and `cache`.

**File > New Project...** (Ctrl+Shift+N) opens the same dialog from the
main window.

## Continue without project

**Continue without project**, bottom right, opens the main window with no
project. You can build, train and save graphs. Project commands such as
**File > Save** (the project) and **File > Close Project** stay disabled
until you open or create a project.

## Keys on the start page

| Key | On the start page |
| --- | --- |
| Ctrl+N | **Create a new project** |
| Ctrl+O | **Open a project** |
| Enter | Opens the selected recent project |

In the main window, Ctrl+N and Ctrl+O mean **New Script...** and **Open
Script...** instead. See [Keyboard shortcuts](../reference/shortcuts.md).

## If something goes wrong

- **"Could not open <file>. The file may be missing or damaged; details are
  in the log."**: the project file could not be read. **Help > Open Logs
  Folder** shows the log.
- **"No .cyxwiz project file in <folder>."**: **Open a project folder**
  needs the folder that holds the `.cyxwiz` file.
