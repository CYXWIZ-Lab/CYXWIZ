# Python Interpreter Design (Engine)

This document defines how the CyxWiz Engine selects and isolates the Python runtime, with a focus on
virtual environments and avoiding accidental dependency leakage from system Python.

---

**Goals**
1. Respect user-selected interpreter (including venv/conda) consistently across runs.
2. Avoid accidental fallback to system Python unless explicitly requested.
3. Ensure packages resolve from the selected environment only.
4. Provide clear diagnostics when the configured interpreter cannot be used.

**Non-goals**
- Managing Python package installation inside the engine.
- Supporting multiple interpreters in a single engine session.

---

## Behavior (as of 2026-03-17)
- Config sources:
  - Global: `engine_config.json` (`python.use_bundled`, `python.interpreter_path`).
  - Project override: `python_env.json` (relative paths resolve from project root).
- Python initializes lazily on first Python use (console/script), not at app launch.
- On project open/close, the engine checks for interpreter mismatch if Python is already initialized.
- After new project venv creation completes, the engine checks again if Python is already initialized
  to detect mismatch with the newly written `python_env.json`.
- If Python is not initialized yet, the first Python use will pick up the current project override.

---

## Interpreter Selection Flow
1. Resolve global settings from `engine_config.json`.
2. If a project is open and it has a valid `python_env.json`, that overrides the global settings.
   - Relative paths are resolved against the project root.
3. Validate the selection:
   - `use_bundled = true` requires a bundled runtime next to the engine.
   - `use_bundled = false` with `interpreter_path` requires that path to exist.
   - `use_bundled = false` with no `interpreter_path` uses system Python.
4. Initialization happens only on first Python use:
   - If `Py_IsInitialized()` is already true, initialization fails with a hard error.
   - The engine config is applied before initialization, then the interpreter is started.
5. If Python is already initialized and the selection changes, the engine logs a mismatch and
   requires restart. No hot swap is performed.

---

## Configuration Locations
Search order for `engine_config.json`:
1. `<exe_dir>/engine_config.json`
2. `<exe_dir>/config/engine_config.json`
3. User config:
   - Windows: `%APPDATA%/CyxWiz/engine_config.json`
   - Linux/macOS: `~/.cyxwiz/engine_config.json`
Fallback template (used only when no config is found):
- `<exe_dir>/resources/engine_config.json`

First run behavior:
- If no config is found, the engine creates a default config in the user config directory.
- A template file is also written next to the executable as:
  `<exe_dir>/engine_config.template.json`
  - For portable installs, copy this template to `<exe_dir>/engine_config.json`
    and edit it directly.
Notes:
- On startup the engine sets the current working directory to `<exe_dir>`, so the
  search order effectively uses the executable directory first.

---

## Venv Creation for New Projects
- A `python/` venv is created asynchronously under the project root.
- The interpreter used to create the venv is chosen in this order:
  - If `python.use_bundled = true`, only the bundled runtime is used.
  - Else if `python.interpreter_path` is set, only that interpreter is used.
  - Else fallback to system Python:
    - Windows: `python`, then `py -3`.
    - Linux/macOS: `python3`, then `python`.
- The venv command uses isolated mode (`-I`) to avoid leaking user site packages.
- If venv creation fails, it retries once with `--without-pip`.
- After venv creation, `python_env.json` is written to point at the venv interpreter.
- When venv creation finishes:
  - If Python is not initialized yet, the next Python use will start with the venv interpreter.
  - If Python is already initialized to a different interpreter, a mismatch is logged and a
    restart is required.

---

## Interpreter Path Conventions (Reference)
- Bundled:
  - Windows: `<exe_dir>/python/python.exe`
  - Linux/macOS: `<exe_dir>/python/bin/python3`
- Project venv:
  - Windows: `<project>/python/Scripts/python.exe`
  - Linux/macOS: `<project>/python/bin/python`
- Custom: absolute path from `engine_config.json`.
- `python_env.json` supports absolute or project-relative paths.

---

## Environment Isolation
- Clear `PYTHONHOME` and `PYTHONPATH` before initialization to avoid leakage.
- For venv interpreters, leave `PYTHONHOME` unset (venv `pyvenv.cfg` drives resolution).
- For custom system installs, set `PYTHONHOME` to the interpreter root.
- For bundled Python, leave `PYTHONHOME` unset; ensure the bundled directory is on PATH/LD_LIBRARY_PATH
  so the runtime can locate its DLLs/shared libraries.
- Rebuild `sys.path` to include:
  - Standard library paths under the selected runtime.
  - The selected `site-packages` path.
  - The engine module directory (so `pycyxwiz` can be imported from next to the exe).
  - Optional scripts directory.
- User site-packages are not added, to avoid cross-version leakage.
- Log final `sys.executable`, `sys.prefix`, `sys.base_prefix`, and a sample of `sys.path`.

Status: Implemented (clears env vars and rebuilds `sys.path` to the selected runtime + engine module dir).

---

## Diagnostics
Preferences -> Python -> Show Runtime Details displays:
- `sys.executable`, `sys.prefix`, `sys.base_prefix`
- `PYTHONHOME`, `PYTHONPATH`
- `site.getsitepackages()` and `sys.path`

---

## CLI Command Window (Quick Check)
In the engine command window, run:

```python
import sys, site, os

print("exe:", sys.executable)
print("prefix:", sys.prefix)
print("base_prefix:", sys.base_prefix)
print("PYTHONHOME:", os.getenv("PYTHONHOME"))
print("PYTHONPATH:", os.getenv("PYTHONPATH"))
print("site-packages:", site.getsitepackages() if hasattr(site, "getsitepackages") else "n/a")
print("user-site:", site.getusersitepackages())
print("sys.path:")
for p in sys.path:
    print("  ", p)
```

This triggers lazy initialization (if not already initialized) and prints the active interpreter and
effective search paths.

---

## Troubleshooting Checklist
- Python fails to initialize with "already initialized before engine configuration": restart the engine
  and avoid initializing Python from any other component before the engine applies config.
- Interpreter mismatch warning after project open or venv creation: restart to pick up the new interpreter.
- Bundled Python missing: confirm `<exe_dir>/python/` exists and contains the expected runtime.
- Custom interpreter path invalid: update `engine_config.json` or Preferences -> Python to a valid path.
- Venv not detected after project creation: confirm `python_env.json` exists and the venv interpreter path
  is valid under `<project>/python/`.
- Imports resolve from the wrong environment: run the CLI quick check above and verify `sys.path` only
  includes the intended `site-packages` and scripts directory.
- Venv creation fails: check logs for the attempted interpreter list and ensure a working Python is on PATH.
- Interpreter path ignored: if `python.use_bundled = true`, the engine will ignore
  `python.interpreter_path` and always use the bundled runtime.
- Venv creation fails with "Bundled Python is missing the venv module":
  set `python.use_bundled = false` and point `python.interpreter_path` to a full Python install.
- Error: "failed to get the Python codec of the filesystem encoding":
  - Cause: stdlib `encodings` is missing, or `PYTHONHOME` was set to a venv root.
  - Fix: ensure the base Python install has a full `Lib` (or `python*.zip` with stdlib). For venv
    interpreters, leave `PYTHONHOME` unset and recreate the venv if it was built from a broken base.
- Error: "ModuleNotFoundError: No module named 'threading'":
  - Cause: stdlib missing (bundled Python missing `Lib` or `python*.zip`).
  - Fix: point `engine_config.json` to a full Python install or copy the full stdlib into the bundled runtime.
- Error: "SRE module mismatch":
  - Cause: mixing stdlib/extension modules from different Python versions (often from a partial install
    or a polluted `sys.path`).
  - Fix: use a clean Python install, avoid mixing versions, and rebuild the venv from that base.
- Error: "The filename, directory name, or volume label syntax is incorrect":
  - Cause: invalid `python.interpreter_path` (bad quoting, stray characters, or a path that no longer exists).
  - Fix: edit `engine_config.json` to a valid absolute path without surrounding quotes.
- Error: "No module named 'pycyxwiz'":
  - Cause: the `pycyxwiz` extension module is not on `sys.path` or was not built for this Python version.
  - Fix: ensure `pycyxwiz.pyd` (Windows) or `pycyxwiz*.so` (Linux/macOS) is present next to the engine
    binary, or install it into the selected interpreter's `site-packages`. Rebuild the `pycyxwiz` target
    if needed and make sure Python version matches.

---

## Logging
The engine logs interpreter selection, initialization, venv creation, and mismatch errors using
`spdlog` through the standard logging API.

---

## Known Limitations / Open Items
- Python cannot be swapped once initialized; interpreter changes require a restart.
- If Python initializes before a new project's venv is ready, a restart is required to use the venv.
- A blocking UI confirmation for missing bundled Python is not implemented; initialization fails
  with an error instead of silently falling back.
- Bundled Python must include the standard library (`Lib` or `python*.zip` with stdlib). If missing,
  initialization fails and the user must point to a full Python install.
