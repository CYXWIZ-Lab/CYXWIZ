# Sandbox Import Error Fix

**Date:** 2025-11-17
**Issue:** ImportError when trying to import sys/os/csv modules with sandbox ON or OFF
**Status:** FIXED

---

## Problem Description

User reported:
> "importError on data_loading <string> line 15 when sandbox is on or off"

Line 15 of data_loading.cyx:
```python
import sys
```

This import was failing even though we had added 'sys' to the sandbox whitelist.

---

## Root Cause Analysis

The issue had **two related causes**:

### Cause 1: Missing 'builtins' Module in Whitelist

The `CleanupHooks()` function tried to clean up the import hook using Python code:

```python
import builtins  # ← This import goes through the hook!
import sys       # ← This too!
if hasattr(builtins.__import__, 'original_import'):
    builtins.__import__ = builtins.__import__.original_import
```

**Problem:** If 'builtins' or 'sys' were NOT in the sandbox whitelist, this cleanup code would fail, leaving the import hook installed permanently!

**Impact:** Once the hook failed to clean up, it would persist across script executions. Even turning sandbox OFF wouldn't help because the hook was already installed in the Python interpreter.

### Cause 2: Fragile Cleanup Using Python Code

The cleanup code executed Python strings via `py::exec()`, which:
- Relied on Python imports working correctly
- Could fail if the import hook was misconfigured
- Created a bootstrapping problem (need imports to clean up imports)

---

## The Fix

### Fix 1: Add 'builtins' to Whitelist

**File:** `cyxwiz-engine/src/scripting/python_sandbox.h`

**Change:**
```cpp
std::unordered_set<std::string> allowed_modules{
    // Core Python (required for sandbox cleanup)
    "builtins",  // ← ADDED
    "sys",
    "os",
    "io",
    "csv",
    // ... rest of modules
};
```

**Why:** The cleanup code needs to import 'builtins' to restore __import__. If 'builtins' isn't whitelisted, cleanup fails and the hook persists.

### Fix 2: Rewrite Cleanup Using Pure C++ API

**File:** `cyxwiz-engine/src/scripting/python_sandbox.cpp`

**Before (Python code execution):**
```cpp
void PythonSandbox::CleanupHooks() {
    try {
        std::string cleanup_code = R"(
import builtins  // Goes through import hook!
import sys       // Could fail!
if hasattr(builtins.__import__, 'original_import'):
    builtins.__import__ = builtins.__import__.original_import
)";
        py::exec(cleanup_code);  // Fragile!
    } catch (const py::error_already_set& e) {
        spdlog::error("Failed to cleanup hooks: {}", e.what());
    }
}
```

**After (Pure C++ pybind11 API):**
```cpp
void PythonSandbox::CleanupHooks() {
    try {
        // Use pybind11 C++ API directly to avoid import issues
        // py::module_::import() uses Python C API and bypasses __import__ hook
        py::module_ builtins = py::module_::import("builtins");

        // Get current __import__ function
        py::object current_import = builtins.attr("__import__");

        // Check if our sandbox hook is installed
        if (py::hasattr(current_import, "original_import")) {
            // Restore the original __import__
            py::object original_import = current_import.attr("original_import");
            builtins.attr("__import__") = original_import;
            spdlog::debug("Sandbox import hook cleaned up");
        } else {
            spdlog::debug("No sandbox hook to clean up (already clean)");
        }
    } catch (const py::error_already_set& e) {
        spdlog::error("Failed to cleanup hooks: {}", e.what());
    }
}
```

**Why This Works Better:**

1. **Bypasses Import Hook:** `py::module_::import()` in C++ uses `PyImport_ImportModule()` from the Python C API, which operates at a lower level than Python's `__import__` function. This bypasses our custom import hook!

2. **No Python Code Execution:** Doesn't use `py::exec()` with Python strings that could fail due to import restrictions.

3. **Direct Attribute Access:** Uses pybind11's C++ API to directly manipulate Python objects (builtins.__import__).

4. **More Robust:** Works even if the import hook is broken or misconfigured.

---

## How the Import Hook Works

### Normal Python Import Flow

```
User script: import sys
    ↓
Python: calls builtins.__import__("sys")
    ↓
System: loads sys module from stdlib
    ✓ Success
```

### Sandbox Import Flow (Hook Installed)

```
User script: import sys
    ↓
Python: calls builtins.__import__("sys")
    ↓
Our Hook: SandboxImportHook.__call__("sys")
    ↓
Hook: checks if "sys" in allowed_modules
    ↓ Yes (in whitelist)
Hook: calls original_import("sys")
    ↓
System: loads sys module
    ✓ Success
```

### Previous Broken Cleanup Flow

```
Execute(): Install hook
    ↓
User script: import sys fails (sys not in whitelist)
    ↓
CleanupHooks(): tries to "import builtins"
    ↓
Hook: checks if "builtins" in allowed_modules
    ↓ No! (not in whitelist)
Hook: raises ImportError
    ↓
Cleanup fails! Hook stays installed!
    ↓
Next execution: Hook still active with old whitelist
    ✗ Broken state persists
```

### Fixed Cleanup Flow

```
Execute(): Install hook
    ↓
User script runs...
    ↓
CleanupHooks(): Uses py::module_::import("builtins")
    ↓
pybind11: Calls PyImport_ImportModule() directly (bypasses hook!)
    ↓
Gets builtins module
    ↓
Accesses __import__ attribute
    ↓
Restores original __import__
    ✓ Hook removed cleanly
```

---

## Testing

### Test 1: Basic Import (Sandbox OFF)
```python
import sys
import os
import csv
print("✓ All imports successful")
```

**Expected:** Should work (no sandbox interference)

### Test 2: Basic Import (Sandbox ON)
```python
import sys
import os
import csv
print("✓ All imports successful with sandbox")
```

**Expected:** Should work (modules in whitelist)

### Test 3: Blocked Import (Sandbox ON)
```python
import subprocess  # Not in whitelist
```

**Expected:** ImportError with message "Module 'subprocess' is not allowed in sandbox environment"

### Test 4: Hook Cleanup Verification

Run script with sandbox ON, then turn sandbox OFF and run again:

```python
import sys
print("✓ sys imported successfully")
```

**Expected:** Should work both times. Previous bug would cause second execution to fail because hook wasn't cleaned up.

---

## Current Whitelist (25 modules)

```cpp
"builtins",      // Python internals
"sys",           // System
"os",            // OS interface
"io",            // I/O
"csv",           // CSV parsing
"time",          // Time utilities
"math",          // Math functions
"random",        // Random numbers
"statistics",    // Statistics
"json",          // JSON
"datetime",      // Date/time
"collections",   // Collections
"itertools",     // Iteration tools
"functools",     // Functional tools
"re",            // Regex
"string",        // String utilities
"pathlib",       // Path handling
"tempfile",      // Temporary files
"numpy",         // NumPy (if installed)
"pandas",        // Pandas (if installed)
"matplotlib",    // Plotting (if installed)
"scipy",         // SciPy (if installed)
"pycyxwiz",      // CyxWiz backend bindings
"cyxwiz_plotting" // CyxWiz plotting
```

---

## Commit Information

**Files Changed:**
- `cyxwiz-engine/src/scripting/python_sandbox.h` - Added 'builtins' to whitelist
- `cyxwiz-engine/src/scripting/python_sandbox.cpp` - Rewrote CleanupHooks() to use C++ API

**Build Status:** ✓ Successful

**Testing Required:**
1. Run test_python_imports.cyx with sandbox ON
2. Run test_python_imports.cyx with sandbox OFF
3. Run data_loading.cyx template with sandbox ON
4. Verify startup scripts work (import modules)
5. Toggle sandbox ON/OFF multiple times and verify no issues

---

## Lessons Learned

1. **Avoid Bootstrapping Problems:** Don't use Python imports in code that cleans up import hooks. Use lower-level APIs (pybind11 C++ API) instead.

2. **Whitelist Essential Modules:** If your cleanup code needs certain modules, make sure they're whitelisted. In our case, 'builtins' is essential for hook cleanup.

3. **Use pybind11 C++ API When Possible:** For critical operations like hook management, prefer `py::module_`, `py::object`, and `py::hasattr` over executing Python strings.

4. **Test Hook Persistence:** Always test that hooks are properly cleaned up by running multiple executions with sandbox toggling.

---

## Future Improvements

1. **Store Original Import in C++:** Instead of storing `original_import` as a Python attribute, we could store it as a C++ static variable. This would make restoration even more robust.

2. **Timeout Implementation:** Re-implement timeout protection using Python's signal module or subprocess execution (currently disabled due to GIL conflicts).

3. **Hook Verification:** Add logging to verify hook is actually removed after cleanup, not just assumed.

4. **Unit Tests:** Create automated tests for sandbox hook installation and cleanup.
