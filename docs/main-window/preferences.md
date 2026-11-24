---
title: Preferences
description: The Preferences dialog, its pages, search, restoring defaults and where the Engine saves your settings.
type: reference
area: general
audience: user
status: experimental
since: 1.0.8
owner: main-window
last_reviewed: 2026-11-24
related:
  - ../usage/preferences.md
  - ./editing-values.md
  - ../reference/shortcuts.md
  - ../concepts/gpu-providers.md
redirect_from: []
---

**Preferences** holds the Engine's own settings: how it starts, how it
looks, the Studio, Python, training, devices and tools. Every setting here is
Engine-wide: it applies to every project.

Each option is described in [Preferences options](../usage/preferences.md).
This page describes the dialog itself.

## Open Preferences

- Select **File > Preferences...**, or press **Ctrl+,**.
- **Help > Keyboard Shortcuts...** opens Preferences at the **Shortcuts**
  page.
- **Servers...** in the account menu (at the right end of the menu bar)
  opens Preferences at the **Servers** page.

Preferences is a modal dialog: while it is open, the rest of the Engine
does not take clicks or shortcuts.

## Layout

The dialog has a rail of pages on the left, the selected page on the right,
and **Close** at the bottom.

There is no **OK**, **Apply** or **Cancel**. Every change applies and is saved
the moment you make it. Two choices have their own button because the
Engine checks them first:

- **Use this device** on the **Devices** page.
- **Use this Python** on the **Python** page.

## Pages

The rail groups the pages under four headings.

| Group | Page | What it holds |
| --- | --- | --- |
| ENGINE | **General** | What the Engine shows at start, the recent-projects list, when to ask before closing, Debug messages in the logs, separate windows, the data memory budget |
| ENGINE | **Appearance** | Theme (and a link to the **Theme Editor...**), interface and code text sizes, the side of the tool sidebar, node icons, a live preview |
| ENGINE | **Shortcuts** | The read-only list of keyboard shortcuts by window, with **Search actions or keys** |
| ENGINE | **Storage** | Caches of prepared data with their sizes, **Show in Explorer** and **Clear...** |
| ENGINE | **Servers** | The CyxWiz server address, what the Engine connects to and its status, deployment address and P2P port, the signed-in account |
| STUDIO | **Studio** | Auto Save at start, the canvas minimap, requiring a passing Local Debug run before Train |
| STUDIO | **Script Editor** | Tab size, whitespace, syntax colours, minimap, word wrap, auto indent, and how new files are saved |
| STUDIO | **Python** | The Python interpreter, virtual environments for new projects, startup imports and scripts, the Python console |
| STUDIO | **Nodes panel** | Grid or list view, and the **Recent** list |
| TRAINING | **Training** | Fine-tuning with a frozen BatchNorm, and how the Training Dashboard charts behave |
| TRAINING | **Search** | Defaults for Hyperparameter Search and Neural Architecture Search |
| TRAINING | **Studio Debugger** | How many layer-capture runs to keep, and how the layer views draw |
| TRAINING | **Simulation** | Tick rate, the Simulation Monitor, and policy playback |
| TRAINING | **Devices** | The training backend, the execution policy, installed backend packs, and the compute device to use |
| TOOLS | **Tools** | Defaults for Utilities and Monitoring, and diagnostics logging |

Installed plugins can add their own sections at the end of the **General**,
**Script Editor**, **Python** and **Simulation** pages.

For the **Devices** page, see also [GPU providers](../concepts/gpu-providers.md).

## Search the pages

Type in **Search settings** at the top of the rail. The rail keeps only the
pages whose names or settings match every word you typed, in any order. For
example, `minimap` leaves **Studio** and **Script Editor**. When nothing
matches, the rail says **No page matches.** Clear the field to see every page
again.

The search finds pages, not single settings: open the page and look for the
setting there.

## Restore a page's defaults

Every page with settings ends with **Restore this page's defaults**.

1. Select **Restore this page's defaults**.
2. Read the question **Restore the shipped values on this page?** and, where
   shown, what is kept.
3. Select **Restore defaults**, or **Keep my settings** to leave the page as
   it is.

Some things are yours and are kept:

| Page | Kept or set |
| --- | --- |
| **Appearance** | Goes back to CyxWiz Dark, 15 px interface text, 16 px code text, sidebar on the right, Font Awesome node icons |
| **Python** | The interpreter is kept; **Packages to install** is emptied |
| **Nodes panel** | Favourites and Recent are kept |
| **Simulation** | Plugin sections keep their own settings |

**Shortcuts**, **Storage**, **Servers** and **Devices** have no restore
button.

## How fields behave

The number fields in Preferences are typed fields with **-** and **+**
steps. A typed number applies when you press **Enter** or leave the field.
See [Editing values](./editing-values.md).

## Where settings are saved

The Engine saves Preferences in `engine_config.json` in your user settings
folder:

| System | Folder |
| --- | --- |
| Windows | `%APPDATA%\CyxWiz` |
| Linux and macOS | `~/.cyxwiz` |

If a file named `engine_config.json` exists next to the Engine program, or in
a `config` folder next to it, the Engine reads and saves that file instead.

Other things the Engine remembers live in the same folder, for example the
recent-projects list and your saved node presets. The window layout is saved
separately; see [Layouts and docking](./layouts-and-docking.md).

## User-set defaults

Some views have choices you can change while you work, such as a chart's
scale or a panel's view mode. Where the Engine keeps a saved default for
such a choice, it is on the matching Preferences page: for example the
Training Dashboard chart options (**Auto scale**, **Follow the latest
epoch**, **Log loss axis**, **Smooth**) on **Training**, and the Nodes panel's
**Grid** or **List** view on **Nodes panel**. A choice that has no entry in
Preferences has no saved default.

## Related

- [Preferences options](../usage/preferences.md)
- [Keyboard shortcuts](../reference/shortcuts.md)
- [Editing values](./editing-values.md)
