---
title: Keyboard shortcuts
description: Every keyboard and mouse shortcut in the CyxWiz Engine, grouped by the window that has focus.
type: reference
area: general
audience: user
status: stable
since: 1.0.7
owner: main-window
last_reviewed: 2026-11-24
related:
  - ../main-window/preferences.md
  - ../main-window/menus.md
  - ../user-guide/graph-editor/canvas.md
  - ../user-guide/graph-editor/find-and-add-nodes.md
redirect_from: []
---

A shortcut belongs to the window that has keyboard focus. The same key can
do a different, matching action in another window: **F5** runs the script in
the Script Editor and starts training everywhere else. A key the focused
window handles itself wins over the global key.

The Engine lists the same table in **Help > Keyboard Shortcuts...**, which
opens **File > Preferences... > Shortcuts**. There you can filter by window and
search with **Search actions or keys**. You can't change the keys today.

This page is hand-checked against the code; it is not generated yet.

## How shortcuts are handled

- While a modal dialog or the code-completion list is open, global shortcuts
  are off. In the Engine's dialogs, **Esc** chooses **Cancel**, and in a
  one-button message both **Enter** and **Esc** choose **OK**.
- While you type in a text field on the canvas, the canvas keys (**M**,
  **F**, **Ctrl+A** and the others) type into the field instead.
- A global shortcut runs even when its menu item is greyed out. If nothing
  happens, open the menu and hover the item: the status bar says why it is
  not available.

## Everywhere

| Keys | Action | Menu |
| --- | --- | --- |
| **Ctrl+Shift+N** | New Project... | File |
| **Ctrl+Shift+O** | Open Project... | File |
| **Ctrl+N** | New Script... | File, Script |
| **Ctrl+O** | Open Script... | File, Script |
| **Ctrl+S** | Save (the script or graph when the Script Editor or canvas has focus) | File |
| **Ctrl+Shift+S** | Save As... (the script or graph when the Script Editor or canvas has focus) | File |
| **Ctrl+Alt+S** | Save All | File |
| **Ctrl+I** | Open Model File... | File |
| **Ctrl+E** | Export Model as CyxWiz package | File > Export Model |
| **Ctrl+,** | Preferences... | File |
| **Ctrl+P** | Command Palette | View |
| **Ctrl+Shift+P** | Show or hide the Pattern Browser panel | View > Panels |
| **Ctrl+Shift+F** | Find in Files... | Edit |
| **Ctrl+Shift+H** | Replace in Files... | Edit |
| **F5** | Start training (runs the script when the Script Editor has focus) | Train |
| **F6** | Local Debug | Train |
| **F7** | Compile Graph | Train |
| **F8** | Run Test | Train > Test |
| **F12** | Open Python Console (go to definition when the Script Editor has focus) | Script |
| **F1** | Documentation | Help |
| **F11** | Full Screen | View |
| **Alt+F4** | Exit | File |

In the **Command Palette**, type to filter, use **Up** and **Down** to move,
**Enter** to run the selected command and **Esc** to close.

## Studio canvas

These keys work while the **CyxWiz Studio** canvas has focus. Click an empty
part of the canvas first.

| Keys | Action |
| --- | --- |
| **Ctrl+Z** | Undo (up to 50 steps) |
| **Ctrl+Y** or **Ctrl+Shift+Z** | Redo |
| **Ctrl+X** / **Ctrl+C** / **Ctrl+V** | Cut, copy, paste the selected nodes with the links between them |
| **Ctrl+D** | Duplicate the selected nodes |
| **Ctrl+A** | Select all nodes |
| **Delete** | Delete the selected nodes, or the selected note |
| **Ctrl+F** | Open or close the find bar |
| **F3** / **Shift+F3** | Next / previous match (find bar open); **Enter** and **Shift+Enter** in the bar do the same |
| **Esc** | Close the find bar, else close the menu, else clear the selection |
| **Ctrl+G** | Group the selected nodes |
| **Ctrl+Shift+G** | Ungroup the selected nodes |
| **Ctrl+Shift+U** | Create a subgraph from the selection (two or more nodes) |
| **Ctrl+S** | Save the graph to its file (asks for a name the first time) |
| **Ctrl+Shift+S** | Save the graph as a new file |
| **+** / **-** (also on the keypad) | Zoom in or out around the centre of the view |
| **M** | Show or hide the minimap |
| **F** | Fit the selected nodes in the view, or all nodes when none is selected |

Mouse on the canvas:

| Mouse | Action |
| --- | --- |
| Wheel | Zoom toward the pointer, from 25 % to 200 % |
| Left-drag on empty canvas | Pan the view. There is no box selection. |
| Click a node | Select it |
| **Ctrl**+click a node | Add it to the selection, or remove it |
| Double-click a node | Open its dialog, or expand a subgraph |
| Double-click a group band, frame title or note | Rename the group, edit the frame, or edit the note |
| **Ctrl**+drag the input end of a link | Detach the link; drop it on empty canvas to remove it |
| Right-click | The canvas, node, frame or note menu |

In the canvas **Search nodes...** box: **Up** and **Down** move through the
results, **Enter** adds the selected node and **Esc** closes the list. No key
moves the focus into this box; click it. See
[Find and add nodes](../user-guide/graph-editor/find-and-add-nodes.md).

## Nodes panel

| Keys | Action |
| --- | --- |
| **/** | Move to the panel's search field |
| **Esc** | Clear the search |

## Start page

| Keys | Action |
| --- | --- |
| **Ctrl+N** | Create a new project |
| **Ctrl+O** | Open a project |
| **Enter** | Open the selected recent project |

These differ from the main window, where **Ctrl+N** and **Ctrl+O** are New
Script and Open Script, and projects use **Ctrl+Shift+N** and
**Ctrl+Shift+O**.

## Script Editor

| Keys | Action |
| --- | --- |
| **Ctrl+N** | New script tab |
| **Ctrl+O** | Open Script... |
| **Ctrl+S** / **Ctrl+Shift+S** | Save script / Save script as... |
| **Ctrl+W** | Close tab |
| **Ctrl+Shift+M** | Turn notebook cell mode on or off |
| **F5** / **Shift+F5** | Run script / Stop script |
| **F9** | Run selection |
| **Ctrl+Enter** | Run current cell |
| **F10** | Start debugging |
| **Ctrl+Space** | Completion (with the list open: show or hide details) |
| **Ctrl+Shift+Space** | Show the signature of the call (also opens when you type `(` or `,`) |
| **F12** or **Ctrl**+click | Go to definition |
| **Ctrl+F** / **Ctrl+H** / **Ctrl+G** | Find... / Replace... / Go to Line... |
| **F3** / **Shift+F3** | Next / previous match (find open) |
| **Ctrl+D** | Duplicate Line |
| **Alt+Up** / **Alt+Down** | Move Line Up / Move Line Down |
| **Ctrl+J** | Join Lines |
| **Tab** / **Shift+Tab** | Indent / Outdent |
| **Ctrl+/** | Toggle Line Comment |
| **Shift+Alt+A** | Toggle Block Comment |
| **Ctrl+Z** / **Ctrl+Y** | Undo / Redo |
| **Ctrl+X** / **Ctrl+C** / **Ctrl+V** / **Delete** / **Ctrl+A** | Cut, copy, paste, delete, select all |
| **Ctrl+Alt+Up** / **Ctrl+Alt+Down** | Add a cursor on the line above / below |
| **Alt**+click | Add a cursor where you click |
| **Ctrl+Shift+L** | Select every occurrence of the word |
| **Escape** | Back to one cursor; close the completion list or signature |
| **Ctrl+Shift+[** / **Ctrl+Shift+]** | Fold / unfold the block at the cursor |
| Right-click the gutter | Breakpoint settings: condition, hit count, enabled |

### While debugging a script

| Keys | Action |
| --- | --- |
| **F5** | Continue |
| **Shift+F5** | Stop debugging |
| **F10** | Step over |
| **F11** | Step into |
| **Shift+F11** | Step out |
| **F9** | Toggle breakpoint |
| **Enter** in the Console REPL | Run Python in the paused frame |

**F11** while debugging also switches the Engine to full screen, because the
global **F11** still runs. Press **F11** again outside the debugger, or use
**View > Full Screen**, to leave full screen.

### Notebook mode

The letter keys work in command mode, when no cell is being edited.

| Keys | Action |
| --- | --- |
| **Enter** | Edit the selected cell |
| **Escape** | Stop editing (command mode) |
| **Shift+Enter** | Run the cell and select the next |
| **Ctrl+Enter** | Run the cell and stay on it |
| **Up** / **Down** | Select the cell above / below |
| **A** / **B** | Add a code cell above / below |
| **D**, **D** | Delete the cell (press **D** twice) |
| **M** / **Y** | Make the cell markdown / code |
| **C** / **O** | Fold or unfold the cell / its output |
| **Ctrl+Shift+M** | Leave notebook mode |
| **Ctrl+Space**, **Ctrl+Shift+Space**, **F12** | Completion, signature, go to definition while a code cell is edited |

## Variable Explorer

| Keys | Action |
| --- | --- |
| **Up** / **Down** | Select the variable above or below |
| **Right** / **Left** | Expand or collapse the selected value |
| **Enter** | View data (tables, arrays, lists), or expand |
| **Ctrl+C** | Copy the value |
| **Delete** | Delete the variable |

## Table Viewer

| Keys | Action |
| --- | --- |
| Double-click a cell | Edit the cell |
| **Escape** | Cancel editing the cell |
| **Ctrl+S** | Save the table's edits |
| **Ctrl**+click / **Shift**+click | Add a cell to the selection / extend the selection |
| **Up** / **Down** | Step the first index of an array slice |

**Ctrl+S** in the Table Viewer also runs **File > Save**, so the project's
open scripts and graph are saved at the same time.

## Data Studio

| Keys | Action |
| --- | --- |
| **Ctrl+Enter** | Run the query (Query tab, editor focused) |

## Same key, different action

| Keys | Where | Action |
| --- | --- | --- |
| **Ctrl+N**, **Ctrl+O** | Start page / elsewhere | Create or open a project / new or open a script |
| **Ctrl+S**, **Ctrl+Shift+S** | Canvas, Script Editor / elsewhere | Save the graph or script / save the project. Clicking **File > Save As...** always copies the whole project, whichever window has focus. |
| **F5** | Script Editor / debugging / elsewhere | Run script / Continue / start training |
| **F9**, **F10**, **F11** | Script Editor / debugging | Run selection, start debugging / toggle breakpoint, step over, step into |
| **F12** | Script Editor / elsewhere | Go to definition / Python console |
| **Ctrl+F**, **F3** | Canvas / Script Editor | Find a node / find text |
| **Ctrl+G** | Canvas / Script Editor | Group Selected / Go to Line... |
| **Ctrl+D** | Canvas / Script Editor | Duplicate nodes / Duplicate Line |
| **M** | Canvas / notebook | Minimap / make the cell markdown |

## Changes

- Changed in 1.0.8: on the canvas, **Ctrl+S** and **Ctrl+Shift+S** save the
  graph, and **+** and **-** zoom around the centre of the view.
