# The menu bar

The bar under the window's title: twelve menus on the left, the open
project and your account at the right end.

## The menus

| Menu | What it holds |
| --- | --- |
| File | Projects (new, open, recent, close, **Show Project Folder**), windows, scripts, Save / Save As / Save All, model files, Preferences, Restart, Exit |
| Edit | Undo and Redo, the clipboard, find and replace (in the file or across files), line and text tools for the Script Editor |
| View | Command Palette (Ctrl+P), panels, layouts, a window of its own, themes, the node icon pack (kept for the next launch, also in Preferences > Appearance), minimaps, **Full Screen** (F11) |
| Nodes | Add a layer, group and ungroup, duplicate, delete, the Custom Node Editor |
| Train | Compile (F7), Local Debug (F6), Start (F5), Pause, Stop, Test, Evaluate, the dashboard, optimizer settings, hyperparameter search |
| Data | Import, custom datasets, statistics, Explore and Transform shortcuts into Data Studio |
| Tools | Model, evaluation, clustering, statistics, linear algebra, signal, optimization, time series, text, utilities, monitoring, diagnostics |
| Script | New, open, run, stop, the Python console |
| Deploy | Connect to a server, deploy, model serving, export and convert models |
| Simulation | The Simulation Monitor and the commands of simulation plugins |
| Apps | Plugin panels and the Plugin Manager |
| Help | Tutorials, keyboard shortcuts, **Documentation** (F1), **API Reference**, **What's New**, **Known Issues**, Report Issue, **Open Logs Folder**, About |

An item that cannot run now is greyed; hover it for the reason. Items marked
*(planned)* are not built yet (Check for Updates, Deploy > Quantize, Publish
to Marketplace, Data > Augment).

## The right end

- **The project** (when one is open): its folder icon and name. Hover for the
  project file's path; click for **Show in Explorer** (the project's folder in
  the file manager), **Copy Path** (the project file's path to the
  clipboard) and **Close Project**.
- **Your account**, in the window's top-right corner:
  - signed out, **Sign in**: its menu opens the sign-in dialog;
  - signed in, your initials in a circle, your name (the first name when the
    whole name is long) and your plan (*Pro*, *Admin*). Hover for your email.
    The menu starts with your name, email and plan, then **Account
    Settings...**, **Wallet**, **Servers...** (Preferences > Servers: the
    central server and your nodes) and **Sign out**.

## New in the menus

- **File > Show Project Folder**: the project's folder in Explorer, Finder
  or the file manager. Greyed with "Open or create a project first." when no
  project is open.
- **View > Full Screen** (F11): the Engine window without its frame over the
  whole monitor it is on. F11 again puts it back as it was (maximised or not).
  It is not exclusive full screen, so a panel in a window of its own can sit
  on top without the Engine minimising.
- **Help > Documentation** (F1), **API Reference**, **What's New**: this user
  guide, the pycyxwiz Python reference and the release notes, opened in your
  browser (from the CyxWiz repository on GitHub).
- **Help > Known Issues**: `KNOWN_ISSUES.md`, which ships beside the Engine,
  in the program that opens Markdown files (or selected in the file manager
  when none does).
- **Help > Open Logs Folder**: `engine_log.txt` selected in the file manager,
  to attach to a bug report.
