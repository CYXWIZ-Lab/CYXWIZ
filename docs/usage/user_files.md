# Where the Engine keeps your files

The Engine only reads its own folder. Your settings, layout, logs and caches
are kept in your user folders, so an update or a rollback of the Engine never
loses them, and an Engine installed for all users (in a folder you cannot
write) keeps them too.

This page lists each file and where it goes, for an installed Engine and for
a development build.

## Your user folders

| Folder | Windows | macOS | Linux |
| --- | --- | --- | --- |
| Settings | `%APPDATA%\CyxWiz` | `~/.cyxwiz` | `~/.cyxwiz` |
| Machine state | `%LOCALAPPDATA%\CyxWiz` | `~/Library/Application Support/CyxWiz` | `~/.local/state/cyxwiz` |
| Logs | `%LOCALAPPDATA%\CyxWiz\Engine\logs` | `~/Library/Logs/CyxWiz` | `~/.local/state/cyxwiz/engine/logs` |
| Caches | `%LOCALAPPDATA%\CyxWiz\cache` | `~/Library/Caches/CyxWiz` | `~/.cache/cyxwiz` |

On Linux, `XDG_STATE_HOME` and `XDG_CACHE_HOME` replace `~/.local/state` and
`~/.cache` when they are set. With no user folder at all (no `APPDATA` or
`HOME`), the Engine uses a `cyxwiz` folder in the temporary folder.

## What goes where

| File | Folder | What it is |
| --- | --- | --- |
| `engine_config.json` | Settings | Your settings (Preferences). The only config file: none is read beside the Engine |
| `imgui.ini` | Settings | The window layout, the open panels and the windows of their own |
| `tutorial_progress.json` | Settings | Which tutorials you finished |
| `recent_projects.json`, `node_presets.json`, `node_approvals.json`, `plugin_startup.json`, `plugin_permissions.json` | Settings | Recent projects (on macOS and Linux in `~/.config/cyxwiz`), node presets, approved Python nodes, plugins loaded at start and their permissions |
| `python_scan.json` | Settings | The Python interpreters found at the last Python check |
| `plugins/`, `nodes/`, `patterns/` | Settings | Your own plugins, Python nodes and patterns |
| `engine_log.txt` | Logs | The log of this launch; `engine_log.1.txt` to `engine_log.4.txt` are the four launches before |
| `compute/` | Machine state | The saved compute device and the route qualification evidence |
| `Engine/debug_runs/` (`engine/debug_runs/` on Linux) | Machine state | Studio Debugger runs and the last training trace |
| `cache/plugins/` | Caches | Copies of the plugin libraries the Engine loads |
| `python/pycache/` | Caches | An installed Engine's compiled Python bytecode: of its own Python and of every Python it starts (the Python check, new environments, a project's environment, pip) |
| `cloud/` | Caches | Cached CyxCloud dataset files |
| `no-project/` | Caches | What a run writes when no project is open: `artifacts/`, `checkpoints/`, `prepared/` (prepared data) |
| `cyxwiz/exports/` | The temporary folder | Files an export node writes when no project is open (`no-project/exports/` in Caches if there is no temporary folder) |
| `.cyxwiz/menagerie/`, `.cyxwiz/imported/` | Your home folder (`%USERPROFILE%` on Windows) | The MuJoCo add-on: downloaded Menagerie models and models imported from a URL |

With a project open, runs write into the project folder as before
(`artifacts/`, `checkpoints/`, the prepared-data cache).

On Linux the per-user installation folder (`~/.local/share/cyxwiz`) is never
used for your files: uninstalling removes it.

## An installed Engine

An installed Engine (started from the CyxWiz launcher, or from
`runtime/base/<base>/` of an installation) writes nothing into its own folder:

- The log is in the Logs folder above. **Help > Open Logs Folder** and the
  Console's **Open log file** open it there.
- The layout and the tutorial state are in the Settings folder.
- The Engine's working folder is `Engine` in the Machine state folder, so a
  Python script run with no project open that writes a file by a relative
  name writes it there.
- Python keeps its compiled bytecode in the Caches folder, never as
  `__pycache__` beside the bundled `.py` files. This holds for every Python the
  Engine starts too: the Python check, a new project environment, the
  project's environment and pip (the Engine sets `PYTHONPYCACHEPREFIX` for
  them). A terminal opened from the Engine inherits it, so a Python you run
  there keeps its bytecode in the same folder.
- The MuJoCo add-on writes no `MUJOCO_LOG.TXT`: MuJoCo's warnings go to the
  Engine log and the Console, each kind once; the number of repeats is logged
  when the add-on unloads. A MuJoCo error is logged before the Engine closes.

## A development build

An Engine run from a build folder (`build/bin/Release` and the like) keeps
`engine_log.txt`, `imgui.ini` and `tutorial_progress.json` in that build
folder and works in it, so each build keeps its own log and layout. Python
writes its bytecode as it always does (`__pycache__` beside the `.py` files).
Your settings, presets, plugins and caches are in your user folders, as for an
installed Engine.

## Starting fresh

To reset the layout, close the Engine and delete `imgui.ini`. To reset every
setting, delete `engine_config.json`. Caches can be deleted at any time while
the Engine is closed; they are rebuilt when needed.

## Related

- [Console and logs](console_logs.md)
- [Preferences](preferences.md)
- [Windows of their own](windows.md)
