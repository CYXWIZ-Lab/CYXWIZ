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
