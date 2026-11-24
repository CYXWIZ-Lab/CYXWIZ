---
title: The status bar
description: What each part of the status bar at the bottom of the Engine window shows.
type: reference
area: general
audience: user
status: stable
since: 1.0.7
owner: main-window
last_reviewed: 2026-11-24
related:
  - index.md
  - menus.md
redirect_from: []
---

The status bar runs along the bottom of the [main window](index.md). It
shows, from left to right, the open project, Python environment messages, a
hint about the menu item under the pointer, and running background tasks.

| Part | Where | What it shows |
| --- | --- | --- |
| Project | Left | The open project's name with a folder icon, or **No Project** |
| Python environment | After the project | The project's Python environment setup (see below) |
| Menu hint | Centre | One sentence about the menu item under the pointer |
| Tasks | Right end | Running background tasks, or **Ready** |

## Project

The name of the open project. **No Project** when you chose **Continue
without project** on the start page or closed the project. The full path
of the project file is on the project chip at the right end of the menu bar
(hover it); see [Menus](menus.md#the-right-end-of-the-menu-bar).

## Python environment

When a project opens, the Engine prepares the project's Python
environment in the background:

- a spinner with what it is doing while it works;
- a green tick with the result, for 8 seconds, when it is ready;
- **Python environment setup failed** in red when it failed. Hover it for
  the reason.

When Python is already running another project's environment (you opened
a second project without restarting), the bar says **Restart to use this
project's Python**, with a **Restart** link. Hover the warning for the
reason. **Restart** is the same as **File > Restart**: it saves, then
restarts the Engine with this project. Until you restart, scripts and the
Python console keep the other project's environment; graphs train as
usual.

## Menu hint

While the pointer is over a menu item, the centre of the bar says what the
item does. For an item that is greyed out it gives the reason, for example
"Not available: Open or create a project first." The same reason shows in
the item's tooltip.

## Tasks

- **Ready** when no background task runs.
- While tasks run (loading a dataset, a pipeline, a long check), an
  indicator and a summary of the running tasks. Click the indicator to
  open **Background Tasks** (in **View > Panels** it is called **Tasks**),
  which lists each task with its progress.

## Related

- [The main window](index.md)
- [Console and logs](../usage/console_logs.md): the full messages behind
  a status
