# Python Interpreter Design (Engine)

This document defines how the CyxWiz Engine should select and isolate the Python runtime, with a focus on
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

## Current Behavior (as of 2026-03-17)
- Reads `engine_config.json` for:
  - `python.use_bundled`
  - `python.interpreter_path`
- Optional per-project override `python_env.json` (relative paths resolve from project root).
- Python initializes lazily on first Python use (console/script), not at app launch.
- On project open/close, the engine checks for interpreter mismatch if Python is already initialized.
- After new project venv creation completes, the engine reloads again if the project is still active
  and Python is already initialized, to detect mismatch with the freshly written `python_env.json`.
- Windows venv detection: if the interpreter is in `...\\Scripts\\python.exe` and `pyvenv.cfg`
  exists at the venv root, the venv root is used for `PYTHONHOME` and `Lib/site-packages`.
- Linux/macOS venv detection: if the interpreter is in `.../bin/python` and `pyvenv.cfg`
  exists at the venv root, that root is used for `PYTHONHOME` and `lib/python*/site-packages`.
- If Python is already initialized by another component, the engine still reuses that interpreter
  and only reconfigures paths. In that case `sys.base_prefix` may still reflect the original
  interpreter even though package resolution favors the selected environment.

---

## Proposed Design

### 1) Single Authoritative Initialization Path
All Python initialization must go through `PythonEngine::Initialize()` and nowhere else.
If any module needs Python, it must request it through the engine.

Rule: No component is allowed to call `Py_Initialize()` directly or load a Python DLL ahead of the engine.

### 2) Interpreter Source of Truth
Engine config file: `engine_config.json`

```json
{
  "python": {
    "use_bundled": true,
    "interpreter_path": "C:\\Path\\To\\venv\\Scripts\\python.exe"
  }
}
```

Behavior:
- If `use_bundled = true`:
  - Require bundled Python next to the engine (`<exe_dir>/python/`).
  - If missing, show a blocking warning and fall back only after explicit user confirmation.
- If `use_bundled = false`:
  - Require `interpreter_path`.
  - If invalid, show a blocking warning and provide “Reset to bundled” option.

### 3) Venv Awareness
Optional per-project override: `python_env.json`

```json
{
  "python": {
    "interpreter_path": "C:\\Path\\To\\venv\\Scripts\\python.exe"
  }
}
```

Behavior:
- Project-specific interpreter overrides global config.
- If project interpreter invalid, fall back to global config.
- On new project creation, create a `python/` virtual environment and write `python_env.json`
  to point at that interpreter.

### 3a) Interpreter Path Conventions (Reference)
- Bundled:
  - Windows: `<exe_dir>/python/python.exe`
  - Linux/macOS: `<exe_dir>/python/bin/python3`
- Project venv:
  - Windows: `<project>/python/Scripts/python.exe`
  - Linux/macOS: `<project>/python/bin/python`
- Custom: absolute path from `engine_config.json`.
- `python_env.json` supports absolute or project-relative paths.

### 4) Environment Isolation
Add a small guard in initialization:
- Clear `PYTHONHOME` and `PYTHONPATH` unless explicitly set in config.
- Ensure `sys.path` is set only to the selected interpreter’s `site-packages`
  plus the engine’s script directory.
- Log final `sys.executable`, `sys.prefix`, `sys.base_prefix`, and `sys.path[0..N]`.

Status: Implemented (clears env vars and filters `sys.path` to the selected runtime + scripts).

### 5) Explicit Diagnostics
Add a diagnostics view in Preferences → Python:
- Interpreter source: bundled / custom / system
- `sys.executable`
- `sys.prefix`
- `sys.base_prefix`
- `site.getsitepackages()`
- `sys.path`

Status: Implemented (Preferences -> Python -> "Show Runtime Details").

### 6) Manual Check in Engine Command Window
Use this snippet in the Engine Command Window to confirm the active interpreter and search paths:

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

---

## Implementation Checklist

1. Guard against early initialization:
   - Replace any direct Python initialization calls with `PythonEngine::Initialize()`.
   - Add a hard error if `Py_IsInitialized()` is true before engine config is applied.
   - Status: Implemented (hard error on early init).

2. Stronger configuration validation:
   - Validate interpreter path on startup.
   - For bundled Python, verify `python312.dll` or `python.exe` exists.
   - Status: Partially implemented (bundled existence checks in config/UI; blocking UX not yet added).

3. Optional project override:
   - Add `python_env.json` in project root.
   - On project creation, create `python/` venv and set `python_env.json` to its interpreter.
   - Status: Implemented.

4. Lazy initialization gate:
   - Initialize Python only on first Python use (console/script), not at app launch or project open.
   - Status: Implemented.

5. Mismatch detection on project changes:
   - If Python is already initialized, project open/close/venv-ready events check for mismatch and require restart.
   - Status: Implemented.

6. Add UI diagnostics (Preferences -> Python):
   - "Show runtime details" button.
   - "Reload Python settings (requires restart)" note.
   - Status: Runtime details button implemented; restart note remains.

---

## Remaining Risk
Python cannot be safely swapped once initialized. If any component initializes Python
before the engine applies configuration, the engine will hard-fail initialization and
require a restart to select the correct interpreter.

• Here’s the end‑to‑end selection flow, grounded in the current code.

  At Engine Launch (no project yet)

  1. EngineConfig loads engine_config.json (search order):
      - ./engine_config.json
      - ./config/engine_config.json
      - %APPDATA%/CyxWiz/engine_config.json on Windows (or ~/.cyxwiz/engine_config.json on Linux/macOS)
      - fallback template: ./resources/engine_config.json (used only if no config found)
        cyxwiz-engine/src/core/engine_config.h, cyxwiz-engine/src/core/engine_config.cpp
  2. MainWindow constructs ScriptingEngine, which constructs PythonEngine, which immediately calls Initialize().
     cyxwiz-engine/src/gui/main_window.cpp, cyxwiz-engine/src/scripting/scripting_engine.cpp, cyxwiz-engine/src/
     scripting/python_engine.cpp
  3. PythonEngine::Initialize():
      - If Py_IsInitialized() is already true, it reuses the existing interpreter and skips config.
      - Otherwise it calls ResolvePythonConfig() then configures Python home and initializes the interpreter.
        cyxwiz-engine/src/scripting/python_engine.cpp

  How the Interpreter Is Chosen
  Selection happens inside ResolvePythonConfig() in this order:

  1. Start with global settings from engine_config.json:
      - python.use_bundled
      - python.interpreter_path
  2. If there’s an active project and it has a valid python_env.json, that overrides the global settings.
     cyxwiz-engine/src/scripting/python_engine.cpp

  Bundled vs Custom vs System
  After ResolvePythonConfig():

  - Bundled Python (use_bundled = true)
      - Uses <exe_dir>/python if present.
      - On Windows, it adds that directory to PATH so python*.dll can be found.
      - PYTHONHOME is not set for bundled.
        cyxwiz-engine/src/scripting/python_engine.cpp, cyxwiz-engine/src/core/engine_config.cpp
  - Custom Python (use_bundled = false and interpreter_path set)
      - If the interpreter exists, PYTHONHOME is set to its Python home.
      - On Windows, if the path is a venv interpreter (…/Scripts/python.exe) and pyvenv.cfg is present, the venv root is
        used as Python home.
        cyxwiz-engine/src/scripting/python_engine.cpp, cyxwiz-engine/src/core/engine_config.cpp
  - System Python
      - Happens when bundled is missing/disabled and no valid custom path exists.
      - PYTHONHOME is not set.
      - The runtime is whatever Python is discoverable by the OS loader (DLL/shared library in PATH / standard
        locations).
        cyxwiz-engine/src/scripting/python_engine.cpp

  Where python.exe Is Expected
  These are the concrete locations the engine looks for or creates:

  - Bundled:
      - Windows: <exe_dir>/python/python.exe
      - Linux/macOS: <exe_dir>/python/bin/python3
  - Custom (global):
      - Whatever the user sets in Preferences → Python (engine_config.json).
  - Project override (python_env.json):
      - Absolute path, or relative to the project root.
      - Relative paths are resolved against the project root.
  - New project venv:
      - Windows: <project>/python/Scripts/python.exe
      - Linux/macOS: <project>/python/bin/python

  When a New Project Is Created
  2. A venv is created asynchronously in <project>/python/.
  3. python_env.json is written to point at that venv’s python.exe.
     cyxwiz-engine/src/core/project_manager.cpp

  The venv creation tries these interpreters in order:

  - Bundled Python (if enabled and exists)
  - Custom interpreter (if configured)
  - System fallback via python / py -3 on Windows, or python3 / python on Linux/macOS
  Python selection is resolved only when PythonEngine initializes. If a project opens after the interpreter is
  initialized, the project override won’t apply until the engine is restarted (current design).

  Runtime Diagnostics (New UI)
  Preferences → Python → Show Runtime Details shows:

  - sys.executable, sys.prefix, sys.base_prefix
  - PYTHONHOME, PYTHONPATH
  - site.getsitepackages(), sys.path
    cyxwiz-engine/src/scripting/scripting_engine.cpp, cyxwiz-engine/src/gui/panels/toolbar.cpp

## Risk
 High‑risk gaps

  1. Runtime mismatch still possible.
     If Python initializes before python_env.json is ready (or before project open), we can only “soft reload” paths.
     sys.base_prefix can still point to the original runtime. That means true isolation isn’t guaranteed.
  2. No controlled initialization gate.
     We still initialize Python eagerly in PythonEngine (and CyxWizApp also creates a PythonEngine separately). That
     makes it easy for the “wrong” interpreter to be locked in early.
  3. Finalization/reinit is intentionally avoided.
     That’s safer, but it means you can’t actually swap runtimes mid‑process without a restart. So runtime switching is
     only partial.

  Medium‑risk gaps
  4. Reload skipped when Python is busy.
     CyxWizApp creates one and ScriptingEngine creates another. We rely on Py_IsInitialized() to reuse, but this can
     hide early init problems.
  6. Environment isolation not enforced.
     We don’t clear PYTHONHOME/PYTHONPATH, and we only prepend site‑packages rather than resetting sys.path. That can
  Lower‑risk gaps
  7. Bundled missing falls back silently (UI warning only).
  Design calls for a blocking warning/confirmation, not implemented.

  8. Windows venv path fix added (good), but Linux/macOS venv detection is still naive (assumes python3 layout).

###

 Current Handling (Code)
  Here’s how the interpreter is selected and configured today:

  - Global config is loaded from engine_config.json via EngineConfig (default python.use_bundled = true). UI writes
    these settings in Preferences → Python. cyxwiz-engine/src/core/engine_config.cpp, cyxwiz-engine/src/gui/panels/
    toolbar.cpp
  - Project override is read from python_env.json if a project is open. If present and valid, it overrides global
    config. cyxwiz-engine/src/scripting/python_engine.cpp
  - New projects create a python/ venv asynchronously and write python_env.json pointing at that interpreter. cyxwiz-
    engine/src/core/project_manager.cpp
  - PythonEngine initializes the interpreter on construction (it’s created in ScriptingEngine’s constructor). If
    Py_IsInitialized() is already true, it reuses the existing interpreter and skips config/application of settings.
    cyxwiz-engine/src/scripting/python_engine.cpp, cyxwiz-engine/src/scripting/scripting_engine.cpp
  - Environment setup before init:
      - Bundled: adds <exe_dir>/python to PATH (or LD_LIBRARY_PATH), does not set PYTHONHOME.
      - Custom: sets PYTHONHOME to the interpreter’s parent directory.
      - System: leaves env untouched.
        cyxwiz-engine/src/scripting/python_engine.cpp
  - After init, it prepends a derived site-packages path to sys.path (bundled or custom). It does not clear sys.path.
    cyxwiz-engine/src/scripting/python_engine.cpp
  - Shutdown does not call py::finalize_interpreter() to avoid pybind11 teardown crashes. cyxwiz-engine/src/scripting/
    python_engine.cpp

  Alignment vs docs/python_interpreter_design.md
  What matches:

  - Project override python_env.json exists and overrides global config. cyxwiz-engine/src/scripting/python_engine.cpp
  - New project venv creation and python_env.json write are implemented. cyxwiz-engine/src/core/project_manager.cpp

  Gaps vs the design doc:

  - The design requires a hard error if Python is initialized before config is applied. Current behavior silently reuses
    the pre-initialized interpreter. This is a direct mismatch. cyxwiz-engine/src/scripting/python_engine.cpp
  - The design requires a blocking warning and explicit user confirmation before falling back when bundled Python is
  - The design calls for clearing PYTHONHOME/PYTHONPATH unless explicitly set. Current code does not clear either, and
    doesn’t restrict sys.path to only the selected env. cyxwiz-engine/src/scripting/python_engine.cpp
  - The design calls for runtime diagnostics (sys.executable, sys.prefix, sys.base_prefix, sys.path). There’s no
    diagnostics view yet. cyxwiz-engine/src/gui/panels/toolbar.cpp

  Potential Venv Path Bug (Windows)
  Two places derive site-packages from the interpreter path using parent / "Lib" / "site-packages", which is wrong for
  venvs because venv python.exe lives in .../venv/Scripts/. This yields .../venv/Scripts/Lib/site-packages (non-
  existent). That likely prevents the venv packages from being added and aligns with the doc’s “wrong interpreter”
  symptom.
  - EngineConfig::GetPythonPackagesDir in cyxwiz-engine/src/core/engine_config.cpp
  - ConfigurePythonHome also sets PYTHONHOME to the interpreter’s parent directory, which is .../venv/Scripts for venvs.

