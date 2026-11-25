---
title: The Variable Explorer
description: See every value in the Engine's Python session or a notebook, filter and expand them, and open tables and arrays in the Table Viewer.
type: how-to
area: scripting
audience: user
status: stable
since: 1.0.7
owner: scripting
last_reviewed: 2026-11-25
related:
  - python-console.md
  - notebooks.md
  - ../../main-window/table-viewer.md
  - ../../reference/shortcuts.md
---

Use this to see what your code made: the names in the Python session (the
one scripts and the Python console share) or in an open notebook, with
their type, size, memory and value. Tables and arrays open in the Table
Viewer.

## Open it

Choose **View > Panels > Variable Explorer**. It opens next to the Console
the first time. A notebook has the same view under its cells: select
**Variables** on the notebook bar.

## Pick whose variables

The pill at the top left names the scope: **Python session · scripts and
Console**, or one of the open notebooks. Click it to pick another. Each
notebook has its own variables; scripts and the Python console share the
session's.

## Read the list

| Column | What it shows |
| --- | --- |
| **Name** | The variable. A value with parts (a list, a dict, an object, a DataFrame's columns) has an arrow to expand it |
| **Type** | Its Python type |
| **Size** | Its shape or length |
| **Memory** | The memory it uses, where Python can tell |
| **Value** | A short preview |

Click a column header to sort. The chips above the table filter by kind,
each with its count: **All**, **Tables**, **Arrays**, **Numbers**, **Text**,
**Collections**. **Filter by name, type or value** narrows the list as you
type. **Show modules, functions and classes** also lists those (off by
default).

## When it reads

The view reads the values when a run finishes, when the view opens, and when
you select **Refresh**. It never polls. While code runs, the header says
"Running... values from before the run", and Refresh waits until the run
ends. Before Python starts it says "Python has not started yet."

## Act on a value

Double-click a row or press **Enter** to view its data; the button at the
end of the row does the same. Right-click a row for:

- **View data**: tables, arrays and lists open in the
  [Table Viewer](../../main-window/table-viewer.md), with **Read again**.
- **Plot**: a table or array opens as a plot.
- **Expand** / **Collapse**.
- **Copy name**, **Copy value** (Ctrl+C).
- **Insert name in the editor**: types the name at the Script Editor's
  cursor.
- **Save as CSV...**: writes a table or array to a `.csv` file you choose.
- **Delete variable** (Delete): removes the name from the namespace.

| Key | What it does |
| --- | --- |
| Up, Down | Select the variable above or below |
| Right, Left | Expand or collapse the selected value |
| Enter | View data, or expand |
| Ctrl+C | Copy the value |
| Delete | Delete the variable |

Reading a large table runs on a Python worker and is listed in Background
Tasks; a Python call cannot be stopped part way.

## Related

- [Use the Python console](python-console.md)
- [Notebooks](notebooks.md)
- [Debug a script](debug-a-script.md): the debugger's own Variables view of
  the paused frame
