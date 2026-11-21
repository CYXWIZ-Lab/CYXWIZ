# Preferences

**File > Preferences...** (Ctrl+,). Every setting here is Engine-wide and is
saved the moment you change it.

## General

### Starting

- **When the Engine starts**: **Show the start page** (the default) or
  **Reopen the last project**. Reopening opens the newest project in the
  recent list whose folder still exists, with the scripts that were open and
  the graph that was on the canvas; if no recent project is left, the start
  page shows. A project named on the command line (`cyxwiz-engine
  myproject.cyxwiz`) always wins, and a graph named there wins over the
  project's.
- **Recent projects kept**: how many projects **File > Open Recent** and the
  start page list, 1 to 50 (10 by default). A smaller number trims the list
  at once.

A project remembers its open scripts and graph when you save it (File >
Save) and when the Engine closes.

### Closing

- **Ask before closing**: **When something would be lost** (the default) or
  **Always**. Unsaved scripts, an unsaved graph, a running script or data in
  memory always bring up the close dialog, which lists them under **Unsaved
  work**, **Running script** and **Datasets in memory**, with **Save and
  close**, **Close without saving** and **Keep working**. **Always** shows it
  also when nothing is open.

### Logs, Windows, Data

- **Record Debug messages**: see [Console logs](console_logs.md).
- **Any window can have a window of its own**: see [Windows of their own](windows.md).
- **Data budget (MB)**: see [Big data](big_data.md).

## Script Editor

For every project (they used to be saved into the open project only, and
only on OK). The Script Editor's **View** menu and **View > Minimaps >
Script Editor Minimap** change the same settings and they are saved the same
way.

- **Text**: **Tab size** (2, 4 or 8 spaces), **Show whitespace**, **Syntax
  colours**, **Minimap**, **Word wrap**, **Auto indent**. The code text size
  is set under **Appearance** (the line shows it, with a link). Line numbers
  are always shown and the current line is highlighted; code colours follow
  the Engine theme.
- **Saving files**:
  - **Line endings of new files**: as the system (CRLF on Windows, LF
    elsewhere), LF or CRLF. A file you open keeps the line endings it has;
    the Script Editor's status bar changes them for that one file.
  - **Trim trailing whitespace on save** (off by default): spaces and tabs at
    the end of each line are not written.
  - **Insert a final newline on save** (on by default): the file ends with
    exactly one line break.
  - Trimming and the final newline happen in the editor too, as one step
    that **Ctrl+Z** takes back, so the editor shows what the file holds.
  - Files are written as UTF-8; a file with a byte-order mark keeps it.
    Notebooks (`.ipynb`) are saved as they are.

## Python

- **Interpreter**: the Python the Engine uses (empty: the one shipped with
  the Engine, when there is one). Type a path and press Enter, **Browse...**,
  or **Find Python** (looks on this PC for a supported version; pick one in
  the list). The line under the field says whether it can be used; **Use
  this Python** takes it. Project environments are made from it, and Python
  runs in it when a project has none. Python that is already running keeps
  its interpreter until the Engine restarts. **Runtime details** shows the
  running Python (paths, version, packages folder), with **Copy**.
- **New projects**:
  - **Create a virtual environment for each new project** (on by default):
    in the project's `python` folder. Off, a new project (and one opened
    without an environment) runs in the interpreter above.
  - **Packages to install**: names separated by spaces (or commas), for
    example `numpy pandas`, installed with pip into each new environment
    before Python starts there. A version stays with its name
    (`numpy==2.1`). Words starting with `-` are pip options and are not
    passed. A package that fails to install is reported in the Console log;
    the environment is still used.
- **When Python starts** (each Python session: before its first command or
  script, and again after **Restart**; a change applies to the next session):
  - **Startup script**: a `.py` file run in the session's namespace, so what
    it defines is there in the Python console and in scripts. If it is
    missing or stops with an error, the session starts anyway and the
    Console log says why.
  - **Imports**: **import numpy as np**, and **import pycyxwiz** (with its
    short names: `cyx`, `svd`, `eig`, `zeros`...). Both on by default.
- **Python console > Commands kept**: how many commands, each with its
  output, the Python console keeps (100 to 10,000; 500 by default).
