---
title: Asset Browser
description: The panel that shows the open project's folders and files, previews them, opens them and creates new files.
type: reference
area: general
audience: user
status: stable
since: 1.0.7
owner: main-window
last_reviewed: 2026-11-25
related:
  - table-viewer.md
  - data-input-dialog.md
  - ../user-guide/scripting/script-editor.md
redirect_from:
  - ../usage/asset_browser.md
---

The **Asset Browser** (left dock) shows the open project's folders and
files as a tree. The **Search...** field above the tree filters it by name,
ignoring case, and opens every folder that holds a match. Click a file to
preview it in the pane under the tree.

## Double-click

| File | What happens |
| --- | --- |
| A table file (CSV, TSV, Parquet, JSON Lines, Feather, Arrow IPC, ARFF, NumPy, and HDF5 or Excel when the build has them) | Opens in the **Table Viewer**: the loaded dataset when a Data Input loaded the file, otherwise the file's first 200,000 rows, read in a task. Nothing is loaded into the graph. A file over 1 GB is refused with a note: create a Data Input, which reads it from disk. |
| Another data file (images, audio, ...) | Creates a Data Input node for it. |
| A graph (`.cyxgraph`) | Opens it in CyxWiz Studio. |
| A text file | Opens it in the Script Editor. |

The Script Editor opens `.py`, `.pyw`, `.cyx`, `.ipynb`, `.csv`, `.tsv`,
`.json`, `.yaml`, `.yml`, `.toml`, `.ini`, `.cfg`, `.conf`, `.txt`, `.md`,
`.log` and `.rst` files.

## The preview pane

Under a file's preview, buttons take it further:

- a dataset a Data Input has loaded: **Open in Data Studio** and **Create
  Data Input**;
- another data file: **Create Data Input**;
- a table file: **Open in Table Viewer**;
- a graph: **Open in CyxWiz Studio**;
- a Python file: **Open in Script Editor**; another text file: **Open as
  text**.

## Creating files

Right-click a folder or file and choose **New**, or right-click the empty
space below the tree (the **Project** menu, which works on the project
folder):

| Item | What it creates |
| --- | --- |
| **Script...** | A Python (`.py`) or CyxWiz (`.cyx`) script. Same dialog as **File > New Script** and Ctrl+N. |
| **File...** | A text file: **Text** (`.txt`), **Markdown** (`.md`), **JSON** (`.json`), **YAML** (`.yaml`) or **CSV** (`.csv`). |
| **Folder...** | A folder. |

The **New file** dialog:

- **Name**: the extension of the chosen type is added unless the name
  already has one. For another kind, type its extension in the name:
  `config.toml`, `settings.ini`. `.env` is a name, not an extension, so it
  becomes `.env.txt`.
- **Folder**: the folder you right-clicked (a file's own folder when you
  right-clicked a file), or the project folder from the Project menu.
  **Browse...** picks another.
- **Creates** shows the full path. Characters Windows rejects
  (`< > : " / \ | ? *`) and reserved names (`nul`, `con`, ...) are refused.
  An existing file of that name is replaced, with a warning first.
- **Create and open** writes the file and opens it in the Script Editor. A
  Markdown file starts with a `# title` line from its name, a JSON file with
  `{ }`; the others start empty. A file the Script Editor does not open is
  created but not opened.

## The right-click menu

| On | Items |
| --- | --- |
| Any folder or file | **New**, **Copy**, **Cut**, **Paste**, **Rename**, **Delete**, **Show in Explorer**, **Open in Terminal** |
| A table file | **Open in Table Viewer** |
| A file the Script Editor opens | **Open**, or **Open as text** for a data file |
| A data file | **Create Data Input** (a configured Data Input node) |
| Another file that has a preview | **Quick Preview** |
| A graph | **Open in CyxWiz Studio** |
| The empty space below the tree | The **Project** menu, with **Refresh** |

Drag a file or folder onto a folder to move it there. When the name is
already taken there, the moved item gets `_1`, `_2`, ... added to its name.

## Shortcuts

While the Asset Browser has focus:

| Key | Action |
| --- | --- |
| F2 | Rename the selected item |
| Del | Delete the selection (asks first) |
| Ctrl+C / Ctrl+X / Ctrl+V | Copy, cut, paste |
| Ctrl+N | New script |
| Ctrl+A | Select everything |
| F5 | Refresh |
| Esc | Clear the selection |

They do not fire while you type in a text field (the search box, a dialog's
fields) or while one of its dialogs is open: there Ctrl+A and Del edit the
text.

## Related

- [Table Viewer](table-viewer.md)
- [The Data Input dialog](data-input-dialog.md)
- [Write and run scripts](../user-guide/scripting/script-editor.md)
