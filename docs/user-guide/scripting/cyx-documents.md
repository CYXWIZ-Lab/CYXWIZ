---
title: The .cyx document format
description: The plain-text .cyx format of CyxWiz scripts and notebooks, its cell markers, and how the Engine reads, runs and writes it.
type: reference
area: scripting
audience: user
status: stable
since: 1.0.7
owner: scripting
last_reviewed: 2026-11-25
related:
  - notebooks.md
  - script-editor.md
---

A `.cyx` file is a Python script that can also hold notebook cells. It is
plain UTF-8 text, so it diffs and merges like any source file. The Script
Editor's **New Script...** dialog offers it as **CyxWiz script (.cyx)**, and
the Asset Browser opens it in the Script Editor. Unlike a Jupyter `.ipynb`
file, a `.cyx` notebook does not keep cell outputs.

## Without cell markers

A `.cyx` file without any cell marker is an ordinary Python script: it
opens as text and **Run** (F5) runs it as Python. Lines that are exactly
`%%` are left out when it runs, so `%%` lines can split the file into
sections for **Run section** (Ctrl+Enter).

## Cell markers

A line whose text starts with `%%` opens a cell. The word after it gives
the cell's kind; case and a space after `%%` do not matter:

| Marker | Cell |
| --- | --- |
| `%%code` | Code cell |
| `%%markdown` or `%%md` | Text cell, written in Markdown |
| `%%raw` | Raw cell (kept, never run) |
| `%%` or `%%` followed by any other word | Code cell (the word is ignored) |

Text before the first marker becomes a code cell. Trailing blank lines of
each cell are dropped.

A `.cyx` file with at least one marker opens as a notebook. **View >
Notebook Mode** (Ctrl+Shift+M) switches between the cells and the text.

## What the Engine writes

When the Engine saves a `.cyx` notebook it writes two header lines, then
each cell as its marker, its text and one blank line:

```text
# CyxWiz Script v0.3.0
# Cell markers: %%code, %%markdown, %%raw

%%markdown
# Loss curve
Plot the loss of a short run.

%%code
import numpy as np
loss = np.exp(-np.arange(20) / 5.0)

%%code
print(loss.round(3))

```

Reading this file gives the same cells back, so saving again does not
change it. The two header lines are skipped when the file is read.

## How a notebook runs

In notebook mode, **Run All** and the cell buttons run the code cells only;
text and raw cells are skipped. Each notebook has its own variables. See
[Notebooks](notebooks.md).

## Limits

- No outputs, execution counts or metadata are stored. Use `.ipynb` when you
  need them kept.
- A cell marker must be on its own line; `%%` inside a Python line (a
  string, for example) does not open a cell, but in text mode **Run
  section** treats any line that contains `%%` as a section break.

## Related

- [Notebooks](notebooks.md)
- [Write and run scripts](script-editor.md)
