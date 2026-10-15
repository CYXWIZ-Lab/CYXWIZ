# Asset Browser

The Asset Browser shows the open project's folders and files. Click a file to
preview it; double-click a text file to open it in the Script Editor.

## Creating files

Right-click a folder or file and choose **New**, or right-click the empty
space below the tree (the **Project** menu, which works on the project
folder):

| Item | What it creates |
| --- | --- |
| **Script...** | A Python (`.py`) or CyxWiz (`.cyx`) script. Same dialog as File > New Script and Ctrl+N. |
| **File...** | A text file: **Text** (`.txt`), **Markdown** (`.md`), **JSON** (`.json`), **YAML** (`.yaml`) or **CSV** (`.csv`). |
| **Folder...** | A folder. |

The **New file** dialog:

- **Name**: the extension of the chosen type is added unless the name already
  has one. For another kind, type its extension in the name: `config.toml`,
  `settings.ini`. `.env` is a name, not an extension, so it becomes `.env.txt`.
- **Folder**: the folder you right-clicked (a file's own folder when you
  right-clicked a file), or the project folder from the Project menu.
  **Browse...** picks another.
- **Creates** shows the full path. Characters Windows rejects
  (`< > : " / \ | ? *`) and reserved names (`nul`, `con`, ...) are refused.
  An existing file of that name is replaced, with a warning first.
- **Create and open** writes the file and opens it in the Script Editor. A
  Markdown file starts with a `# title` line from its name, a JSON file with
  `{ }`; the others start empty.

The Script Editor opens `.py .cyx .txt .md .json .csv .tsv .yaml .yml .toml
.ini .cfg .conf`; a file with another extension is created but not opened.

## Other actions

Right-click a folder or file for **Open**, **Copy**, **Cut**, **Paste**,
**Rename**, **Delete**, **Show in Explorer** and **Open in Terminal**. Data
files also offer **Create Data Input** (a configured Data Input node), other
previewable files **Quick Preview**, and graphs **Open in CyxWiz Studio**. The
Project menu adds **Refresh**.

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
