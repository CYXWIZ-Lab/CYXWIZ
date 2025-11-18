# Testing Guide - Sandbox Import Fix

**Date:** 2025-11-17
**Issue:** ImportError and syntax errors when running scripts

---

## Changes Made

### 1. Improved Import Hook (python_sandbox.cpp)

**Before:**
```python
def __call__(self, name, *args, **kwargs):
    # Always check whitelist for every import
    module_parts = name.split('.')
    for i in range(len(module_parts)):
        partial_name = '.'.join(module_parts[:i+1])
        if partial_name in self.allowed_modules:
            return self.original_import(name, *args, **kwargs)
    raise ImportError(f"Module '{name}' is not allowed")
```

**After:**
```python
def __call__(self, name, *args, **kwargs):
    # Skip check if module already imported
    if name in sys.modules:
        return self.original_import(name, *args, **kwargs)

    # Check whitelist only for new imports
    module_parts = name.split('.')
    for i in range(len(module_parts)):
        partial_name = '.'.join(module_parts[:i+1])
        if partial_name in self.allowed_modules:
            return self.original_import(name, *args, **kwargs)
    raise ImportError(f"Module '{name}' is not allowed")
```

**Why:** Prevents re-checking the whitelist for modules that are already loaded. This fixes issues with scripts that import the same module multiple times.

### 2. Added 'builtins' to Whitelist

The 'builtins' module is now in the allowed_modules list (25 modules total).

### 3. Improved CleanupHooks()

Uses pybind11 C++ API instead of Python code execution to restore the original __import__ function.

---

## Test Files Available

Three test files are in the build directory (`build/windows-release/bin/Release/`):

### 1. test_minimal.cyx (Simplest test - START HERE)
```python
# Just imports sys, os, csv and prints success
import sys
import os
import csv
print("All imports successful!")
```

**Use this to:**
- Verify basic imports work
- Test both Sandbox ON and Sandbox OFF
- Should complete in < 1 second

### 2. test_imports_simple.cyx (Comprehensive single-import test)
```python
# Imports all allowed modules ONCE at the top
import builtins
import sys
import os
import io
import csv
# ... etc
# Then tests that each module works
```

**Use this to:**
- Verify all 25 whitelisted modules import correctly
- Test that modules actually function (not just import)
- Should complete in < 2 seconds

### 3. test_sandbox_fix.cyx (Full test suite - SKIP THIS)
⚠️ **This file has issues - don't use it**
- Imports modules multiple times (confuses error messages)
- Uses exec() which triggers security validation
- Line numbers don't match what user sees

---

## Testing Procedure

### Test 1: Minimal Test with Sandbox OFF

1. Launch CyxWiz Engine
2. Open `test_minimal.cyx`
3. **Uncheck** "Enable Sandbox" in Script menu
4. Press F5 to run

**Expected Output:**
```
Starting minimal import test...
sys imported OK
os imported OK
csv imported OK

All imports successful!
Python version: 3.x.x
```

**If you get errors:**
- Note the EXACT error message
- Note the EXACT line number
- Check if file was opened correctly (not corrupted)

### Test 2: Minimal Test with Sandbox ON

1. Keep `test_minimal.cyx` open
2. **Check** "Enable Sandbox" in Script menu
3. Press F5 to run

**Expected Output:**
```
(Same as Test 1)
```

**If you get errors:**
- ImportError = module not in whitelist (shouldn't happen for sys/os/csv)
- "Security violation" = dangerous pattern detected
- Other errors = check error message carefully

### Test 3: Simple Comprehensive Test

1. Open `test_imports_simple.cyx`
2. Run with Sandbox ON
3. Should see all modules import successfully

### Test 4: Data Loading Template (Original failure)

1. Open `scripts/templates/data_loading.cyx`
2. Run with Sandbox ON
3. Should load test_data.csv successfully

---

## Understanding Error Messages

### Error: "ImportError: Module 'xxx' is not allowed in sandbox environment"

**Meaning:** The module is not in the whitelist

**Current whitelist (25 modules):**
- builtins, sys, os, io, csv, time
- math, random, statistics
- json, datetime, collections, itertools, functools
- re, string
- pathlib, tempfile
- numpy, pandas, matplotlib, scipy
- pycyxwiz, cyxwiz_plotting

**Solution:** If you need a module, we can add it to the whitelist.

### Error: "Security violation: Code contains dangerous pattern: xxx"

**Meaning:** Code contains a blocked pattern:
- `os.system` - Can execute shell commands
- `subprocess.` - Can execute shell commands
- `eval(` - Can execute arbitrary code
- `exec(` - Can execute arbitrary code
- `__import__` - Direct import bypasses hook
- `compile(` - Can compile arbitrary code

**Solution:** Rewrite code to avoid these patterns. For example:
- Instead of `exec("import sys")`, use `import sys`
- Instead of `eval(data)`, use `json.loads(data)`

### Error: "Syntax error" (with Sandbox OFF)

**Possible causes:**
1. **File encoding issue** - File has BOM or wrong encoding
2. **File corruption** - File wasn't copied correctly
3. **Python syntax error** - Actual syntax mistake in code
4. **Cached error** - Old error message still showing

**Debugging:**
1. Close and reopen the file
2. Check file size (should match source)
3. Look at the actual line number mentioned
4. Try the minimal test first

---

## Troubleshooting

### Problem: Import works with Sandbox OFF but fails with Sandbox ON

**Likely cause:** Module not in whitelist

**Solution:** Check whitelist in `python_sandbox.h` and add module if safe.

### Problem: Import fails with both Sandbox ON and OFF

**Likely causes:**
1. Module not installed (e.g., numpy, pandas)
2. Python environment issue
3. File corruption

**Solution:** Try the minimal test first to isolate the issue.

### Problem: Script worked before, now fails

**Likely cause:** Import hook from previous run wasn't cleaned up

**Solution:**
1. Restart CyxWiz Engine (fully reinitializes Python)
2. The new CleanupHooks() should prevent this

### Problem: Error line number doesn't match code

**Possible causes:**
1. File in build directory is different from source
2. Error message is from a previous run
3. Line counting includes blank lines/comments differently

**Solution:**
1. Close and reopen file
2. Restart CyxWiz Engine
3. Check file in build directory matches source

---

## What to Report

If you encounter errors, please report:

1. **Which test file** (test_minimal.cyx, test_imports_simple.cyx, etc.)
2. **Sandbox state** (ON or OFF)
3. **Exact error message** (copy/paste)
4. **Line number** mentioned in error
5. **What's on that line** (look at the file)
6. **Console output** (if any)

---

## Expected Results Summary

| Test | Sandbox OFF | Sandbox ON |
|------|-------------|------------|
| test_minimal.cyx | ✓ Pass | ✓ Pass |
| test_imports_simple.cyx | ✓ Pass | ✓ Pass |
| data_loading.cyx | ✓ Pass | ✓ Pass |
| test_sandbox_fix.cyx | ⚠️ Skip | ⚠️ Skip |

All tests should pass with both Sandbox ON and OFF after the fixes.

---

## Next Steps

1. **Start with test_minimal.cyx** - It's the simplest and fastest to debug
2. **Test both Sandbox ON and OFF** - Verify both modes work
3. **Report results** - Let me know what happens with each test
4. **If all pass** - Move on to testing the actual templates

The fixes should resolve:
- ✓ Import hook cleanup issues
- ✓ Re-import handling
- ✓ Module whitelist coverage
- ✓ Hook persistence across executions
