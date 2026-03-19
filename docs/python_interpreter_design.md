# Python Interpreter Design (Engine)

This document defines how the CyxWiz Engine selects and isolates the Python runtime, with a focus on
virtual environments and avoiding accidental dependency leakage from system Python.

---

**Goals**
1. Respect user-selected interpreter (including venv/conda) consistently across runs.
2. Automatically create and use project-specific virtual environments.
3. Ensure packages resolve from the selected environment only.
4. Provide clear, user-friendly diagnostics during Python initialization.
5. Support Python 3.12.x exclusively (avoid 3.14+ and < 3.12).

**Non-goals**
- Managing Python package installation inside the engine (use pip manually).
- Supporting multiple interpreters in a single engine session.
- Bundled Python distributions (removed as of 2026-03-19).

---

## Behavior (as of 2026-03-19)

### Configuration Sources
- **Global**: `engine_config.json` (`system_python_path`)
- **Project override**: `python_env.json` (relative paths resolve from project root)

### Initialization
- Python initializes **lazily** on first Python use (console/script), not at app launch
- On project open/close, the engine checks for interpreter mismatch if Python is already initialized
- After new project venv creation completes, the engine detects the new environment automatically
- If Python is not initialized yet, the first Python use will pick up the current project override

### User-Friendly Logging
The engine now provides clear status messages during Python initialization:
```
[info] ✓ Using project virtual environment
[info]   Project: my_project
[info]   Python: C:\Projects\my_project\python\Scripts\python.exe
[info] → Initializing Python with project environment...
[info]   Environment: Virtual environment (venv)
[info]   Location: C:\Projects\my_project\python
[info]   Site-packages: C:\Projects\my_project\python\Lib\site-packages
[info]   Project scripts: C:\Projects\my_project\scripts
[info]   Engine modules: D:\CyxWiz\build\bin\Release
[info] ✓ Python environment ready (project venv)
```

---

## Interpreter Selection Flow

1. **Resolve configuration**:
   - Check for project `python_env.json` if a project is open
   - Fall back to global `system_python_path` from `engine_config.json`

2. **Validate selection**:
   - Interpreter path must exist and be executable
   - Must be Python 3.12.x (3.12.0 - 3.12.99)
   - Must have standard library (`threading` module test)
   - For venvs, base Python (from `pyvenv.cfg`) must also be valid

3. **Initialization** (on first Python use):
   - If `Py_IsInitialized()` is already true, initialization fails with a hard error
   - Clear environment variables (`PYTHONHOME`, `PYTHONPATH`, set `PYTHONNOUSERSITE=1`)
   - For venvs: Leave `PYTHONHOME` unset (let `pyvenv.cfg` drive resolution)
   - For system Python: Set `PYTHONHOME` to interpreter root
   - Call `py::initialize_interpreter()`
   - Configure `sys.path` with proper isolation

4. **Mismatch handling**:
   - If Python is already initialized and selection changes, the engine logs a mismatch
   - **Restart required** - no hot swap is performed

---

## Configuration Locations

### engine_config.json Search Order
1. `<exe_dir>/engine_config.json`
2. `<exe_dir>/config/engine_config.json`
3. User config:
   - Windows: `%APPDATA%/CyxWiz/engine_config.json`
   - Linux/macOS: `~/.cyxwiz/engine_config.json`

### First Run Behavior
- If no config is found, the engine scans for Python 3.12.x installations
- A default config is created in the user config directory with the best Python found
- A template file is written to `<exe_dir>/engine_config.template.json` for portable installs

**Notes**:
- On startup, the engine sets CWD to `<exe_dir>`, so the search order uses the executable directory first
- Legacy `use_bundled_python` configs are automatically migrated to `system_python_path`

---

## Venv Creation for New Projects

### Automatic Venv Creation
- A `python/` venv is created **asynchronously** under the project root when:
  - Creating a new project
  - Opening a legacy project (missing `python_env.json`)
- Race condition protection: Only one venv creation attempt per project (checks for existing `python/` folder)

### Interpreter Selection for Venv
The system Python from `engine_config.json` is used to create the venv:
1. Verify system Python is valid (3.12.x, has venv module)
2. Run: `<system_python> -m venv <project>/python`
3. Write `python_env.json` with relative path: `python/Scripts/python.exe` (Windows) or `python/bin/python` (Linux/macOS)

### Post-Creation Behavior
- When venv creation finishes:
  - If Python is not initialized yet, the next Python use will start with the venv interpreter
  - If Python is already initialized to a different interpreter, a mismatch is logged and a restart is required
- Venv creation status is shown in the async task manager with progress updates

**Important**: The `-I` (isolated mode) flag is **NOT** used for venv creation, as it causes incomplete venv setup missing standard library links.

---

## Interpreter Path Conventions

- **Project venv**:
  - Windows: `<project>/python/Scripts/python.exe`
  - Linux/macOS: `<project>/python/bin/python`
- **System Python**:
  - Windows: `C:\Python312\python.exe`
  - Linux/macOS: `/usr/bin/python3.12` or `/usr/local/bin/python3.12`
- **python_env.json** supports absolute or project-relative paths

---

## Environment Isolation

### Environment Variables
- Clear `PYTHONHOME`, `PYTHONPATH` before initialization to avoid leakage
- Set `PYTHONNOUSERSITE=1` to prevent loading incompatible user site-packages from other Python versions
- For venv interpreters: Leave `PYTHONHOME` unset (venv `pyvenv.cfg` drives resolution)
- For system Python: Set `PYTHONHOME` to the interpreter root

### sys.path Configuration
Rebuild `sys.path` to include:
1. **Venv site-packages**: `<project>/python/Lib/site-packages`
2. **Project scripts**: `<project>/scripts/`
3. **Engine modules**: `<exe_dir>/` (for `pycyxwiz` module)
4. **Standard library paths**: From both venv and base Python

### Base Prefix Detection (Critical Fix - 2026-03-19)
For venvs, `sys.base_prefix` is read from `pyvenv.cfg` file instead of relying on Python's runtime value:
```python
# pyvenv.cfg example:
home = C:\Python312
include-system-site-packages = false
version = 3.12.8
```

**Why this matters**:
- pybind11's embedded interpreter incorrectly sets `sys.base_prefix` to the engine build directory
- Reading from `pyvenv.cfg` ensures the correct base Python path is used
- Allows access to base Python's standard library (`threading`, `io`, etc.)

### Path Filtering
- Filter `sys.path` to only include paths under:
  - The venv root (`<project>/python/`)
  - The base Python root (from `pyvenv.cfg`)
  - The engine module directory
- User site-packages are **not** added to prevent cross-version leakage

---

## Python Version Requirements

### Supported Versions
- **Required**: Python 3.12.0 - 3.12.99
- **Not supported**: Python < 3.12 or Python 3.14+

### CMake Configuration
The engine build system explicitly finds Python 3.12:
```cmake
# Find Python 3.12 specifically (avoid 3.14+)
find_package(Python3 3.12 EXACT COMPONENTS Interpreter Development QUIET)
if(NOT Python3_FOUND)
    # Fallback to Python 3.12-3.13 range
    find_package(Python3 3.12...3.13 COMPONENTS Interpreter Development QUIET)
endif()
```

This ensures:
- `pycyxwiz` module builds as `pycyxwiz.cp312-win_amd64.pyd` (Python 3.12)
- No ABI compatibility issues with Python 3.14+
- Consistent behavior across venvs created from the same base Python

---

## Diagnostics

### Preferences -> Python -> Show Runtime Details
Displays:
- `sys.executable`, `sys.prefix`, `sys.base_prefix`
- `PYTHONHOME`, `PYTHONPATH`, `PYTHONNOUSERSITE`
- `site.getsitepackages()` and `sys.path`

### Quick Check (Console Command)
Run this in the Python console to verify environment:

```python
import sys, site, os

print("=== Python Environment ===")
print(f"Version: {sys.version}")
print(f"Prefix: {sys.prefix}")
print(f"Base Prefix: {sys.base_prefix}")
print(f"Executable: {sys.executable}")
print(f"\n=== Site Packages ===")
print(f"User Site: {site.getusersitepackages()}")
print(f"Site Packages: {site.getsitepackages()}")
print(f"\n=== OS Info ===")
print(f"CWD: {os.getcwd()}")
print(f"Platform: {sys.platform}")
print(f"\n=== sys.path (first 5) ===")
for i, p in enumerate(sys.path[:5], 1):
    print(f"{i}. {p}")
```

**Expected output for project venv**:
```
Prefix: C:\Projects\my_project\python
Base Prefix: C:\Python312
Executable: C:\Projects\my_project\python\Scripts\python.exe
```

---

## Troubleshooting Checklist

### Python Initialization
- **"Python fails to initialize with 'already initialized before engine configuration'"**:
  - Restart the engine and avoid initializing Python from any other component before config is applied

- **"Interpreter mismatch warning after project open or venv creation"**:
  - Restart the engine to pick up the new interpreter

- **"ModuleNotFoundError: No module named 'threading'"**:
  - **FIXED (2026-03-19)**: Base prefix now correctly detected from `pyvenv.cfg`
  - If still occurring: Verify system Python has complete standard library
  - Check logs for base prefix path (should be `C:\Python312`, not the build directory)

### Venv Creation
- **"Permission denied" during venv creation**:
  - **FIXED (2026-03-19)**: Race condition now prevented with folder existence check
  - If still occurring: Close other processes using the `python/` directory

- **"Venv creation fails"**:
  - Check logs for the attempted system Python path
  - Ensure system Python is 3.12.x and has the `venv` module
  - Verify system Python path in `engine_config.json` is correct

- **"Venv not detected after project creation"**:
  - Confirm `python_env.json` exists in project root
  - Verify venv interpreter path is valid: `<project>/python/Scripts/python.exe`

### Configuration
- **"Custom interpreter path invalid"**:
  - Update `engine_config.json` -> `system_python_path` to a valid Python 3.12.x path
  - Use absolute paths without quotes

- **"No system Python configured"**:
  - Run the engine once to trigger automatic Python detection
  - Manually set `system_python_path` in `engine_config.json`

### Import Errors
- **"Imports resolve from the wrong environment"**:
  - Run the CLI quick check above and verify `sys.path` only includes the intended `site-packages`
  - Check that `PYTHONNOUSERSITE=1` is set (prevents user site-packages leakage)

- **"No module named 'pycyxwiz'"**:
  - Ensure `pycyxwiz.cp312-win_amd64.pyd` (Windows) or `pycyxwiz.cpython-312-*.so` (Linux/macOS) is present next to the engine binary
  - Verify Python version matches (3.12.x only)
  - Rebuild the `pycyxwiz` target if needed

### Version Errors
- **"Error: 'SRE module mismatch'"**:
  - Mixing stdlib/extension modules from different Python versions
  - Ensure system Python is 3.12.x
  - Rebuild the engine with Python 3.12 (delete `build/` directory and reconfigure)

- **"Python 3.14 detected, expected 3.12"**:
  - Install Python 3.12.x
  - Update `system_python_path` in `engine_config.json` to point to Python 3.12

### Filesystem Errors
- **"Error: 'failed to get the Python codec of the filesystem encoding'"**:
  - Cause: stdlib `encodings` is missing, or `PYTHONHOME` was set to a venv root
  - Fix: Ensure base Python has a full `Lib` directory with all standard library modules
  - For venv interpreters: Leave `PYTHONHOME` unset and recreate the venv

- **"The filename, directory name, or volume label syntax is incorrect"**:
  - Invalid `system_python_path` (bad quoting, stray characters, or path doesn't exist)
  - Edit `engine_config.json` to a valid absolute path without surrounding quotes

---

## Logging

The engine logs interpreter selection, initialization, venv creation, and mismatch errors using `spdlog`.

### Log Format (2026-03-19 Update)
- **✓** Success indicators
- **→** Progress indicators
- **⚠** Warning indicators
- Indented details for better readability

### Example Log Sequence
```
[info] ✓ Using project virtual environment
[info]   Project: my_ml_project
[info]   Python: C:\Projects\my_ml_project\python\Scripts\python.exe
[info] → Initializing Python with project environment...
[info]   Environment: Virtual environment (venv)
[info]   Location: C:\Projects\my_ml_project\python
[info]   Site-packages: C:\Projects\my_ml_project\python\Lib\site-packages
[info]   Project scripts: C:\Projects\my_ml_project\scripts
[info]   Engine modules: D:\CyxWiz\build\bin\Release
[info] ✓ Python environment ready (project venv)
```

---

## Known Limitations / Open Items

- **Python cannot be swapped once initialized**: Interpreter changes require a restart
- **Venv creation during initialization**: If Python initializes before a new project's venv is ready, a restart is required to use the venv
- **Python version locked to 3.12.x**: No support for 3.13+ or 3.11 and below
- **No bundled Python**: System Python 3.12.x is required (must be installed separately)
- **Windows Store Python not supported**: Use the official Python.org installer
- **Conda environments**: Supported via manual `system_python_path` configuration

---

## Implementation Notes

### Key Files
- **CMakeLists.txt**: Python 3.12 finding before pybind11
- **cyxwiz-engine/src/core/engine_config.h/cpp**: System Python path configuration
- **cyxwiz-engine/src/core/project_manager.cpp**: Venv creation with race condition protection
- **cyxwiz-engine/src/scripting/python_engine.cpp**:
  - `ReadVenvBasePython()`: Read base Python from pyvenv.cfg
  - `ConfigureCustomPythonPath()`: sys.path isolation with base prefix support
  - `ResolvePythonConfig()`: Interpreter selection with user-friendly logging

### Recent Fixes (2026-03-19)
1. **Race Condition Fix**: Check for existing `python/` folder before creating venv
2. **Base Prefix Detection**: Read from `pyvenv.cfg` instead of `sys.base_prefix`
3. **User-Friendly Logging**: Clear status messages with visual indicators
4. **PYTHONNOUSERSITE**: Prevent user site-packages leakage across Python versions
5. **Removed `-I` flag**: Venv creation no longer uses isolated mode (caused incomplete stdlib setup)

---

**Document Version**: 2026-03-19
**Last Updated**: After Python initialization fixes and logging improvements
**Author**: code3hr (https://github.com/code3hr)
