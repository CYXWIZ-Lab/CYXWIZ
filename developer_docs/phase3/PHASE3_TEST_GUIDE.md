# Phase 3: Python Sandbox Security - Testing Guide

## Build Status

✅ **Build Successful**: `cyxwiz-engine.exe (1.7 MB)`

## Testing Overview

This guide will walk you through testing the Python Sandbox security system to verify:
1. Dangerous operations are blocked
2. Safe operations are allowed
3. Timeout mechanism works
4. Module whitelist enforcement
5. UI controls function correctly

---

## Step 1: Launch the Application

```bash
cd D:\Dev\CyxWiz_Claude
build\windows-release\bin\Release\cyxwiz-engine.exe
```

---

## Step 2: Enable the Sandbox

### Method 1: Via Security Menu

1. Open **Script Editor** tab (should be visible in center panel)
2. Click **Security** menu in the menu bar
3. Check **"Enable Sandbox"**
4. You should see:
   - ✅ **"Active - Scripts are sandboxed"** in green
   - Status bar shows **"SANDBOX ON"** in green at bottom

### Method 2: Verify Status

The status bar (bottom of Script Editor) will show:
- **Green "SANDBOX ON"** - Sandbox is active
- **Gray "Sandbox Off"** - Sandbox is disabled

---

## Step 3: Open the Security Test File

1. In Script Editor, click **File** → **Open** (or press Ctrl+O)
2. Navigate to: `D:\Dev\CyxWiz_Claude\test_sandbox_security.cyx`
3. Open the file

You should see 11 test sections separated by `%%` markers.

---

## Step 4: Run Individual Security Tests

### Test 1: Block eval()

1. Place cursor anywhere in **Section 1** (lines 4-11)
2. Press **Ctrl+Enter**
3. Check **Command Window** tab for output

**Expected Result**:
```
=== Test 1: Attempting to use eval() ===
✅ PASS: eval() blocked - NameError
```

### Test 2: Block exec()

1. Place cursor in **Section 2** (lines 13-20)
2. Press **Ctrl+Enter**

**Expected Result**:
```
=== Test 2: Attempting to use exec() ===
✅ PASS: exec() blocked - NameError
```

### Test 3: Block __import__()

1. Place cursor in **Section 3** (lines 22-29)
2. Press **Ctrl+Enter**

**Expected Result**:
```
=== Test 3: Attempting to use __import__() ===
✅ PASS: __import__() blocked - NameError
```

### Test 4: Block os module

1. Place cursor in **Section 4** (lines 31-38)
2. Press **Ctrl+Enter**

**Expected Result**:
```
=== Test 4: Attempting to import os module ===
✅ PASS: os module blocked - Module 'os' is not allowed in sandbox environment
```

### Test 5: Block subprocess module

1. Place cursor in **Section 5** (lines 40-47)
2. Press **Ctrl+Enter**

**Expected Result**:
```
=== Test 5: Attempting to import subprocess ===
✅ PASS: subprocess module blocked - Module 'subprocess' is not allowed
```

### Test 6: Block open()

1. Place cursor in **Section 6** (lines 49-56)
2. Press **Ctrl+Enter**

**Expected Result**:
```
=== Test 6: Attempting to use open() ===
✅ PASS: open() blocked - NameError
```

### Test 7: Allow math module

1. Place cursor in **Section 7** (lines 58-65)
2. Press **Ctrl+Enter**

**Expected Result**:
```
=== Test 7: Testing allowed module - math ===
✅ PASS: math module allowed, sqrt(16) = 4.0
```

### Test 8: Allow random module

1. Place cursor in **Section 8** (lines 67-74)
2. Press **Ctrl+Enter**

**Expected Result**:
```
=== Test 8: Testing allowed module - random ===
✅ PASS: random module allowed, random int = [some number 1-10]
```

### Test 9: Allow json module

1. Place cursor in **Section 9** (lines 76-83)
2. Press **Ctrl+Enter**

**Expected Result**:
```
=== Test 9: Testing allowed module - json ===
✅ PASS: json module allowed, serialized = {"test": "value"}
```

### Test 10: Basic Python Operations

1. Place cursor in **Section 10** (lines 85-100)
2. Press **Ctrl+Enter**

**Expected Result**:
```
=== Test 10: Basic Python operations ===
✅ PASS: Arithmetic works - 10 + 20 = 30
✅ PASS: List comprehension works - [1, 4, 9, 16, 25]
✅ PASS: Functions work - Hello, Sandbox!
```

### Test 11: Timeout Test (OPTIONAL - Takes 60s)

**⚠️ WARNING**: This test will run for 60 seconds before timing out. Only run if you want to test the timeout mechanism.

1. **Uncomment** lines 104-108 in the file
2. Place cursor in Section 11
3. Press **Ctrl+Enter**
4. Wait 60 seconds

**Expected Result**:
```
=== Test 11: Timeout Test ===
Starting infinite loop (should timeout after 60s)...
Error: Execution timeout exceeded (60s)
```

---

## Step 5: Run All Tests at Once

1. Press **F5** to run the entire file
2. Check **Command Window** for complete output
3. All tests 1-10 should pass

**Expected Summary**:
```
✅ All 10 tests passed (Test 11 commented out)
```

---

## Step 6: Test Sandbox Disable

1. Click **Security** → Uncheck **"Enable Sandbox"**
2. Status bar should show **"Sandbox Off"** in gray
3. Re-run Test 7 (math module) - should still work
4. Re-run Test 4 (os module) - **should now SUCCEED** (os module imports successfully)

**Expected Result with Sandbox Off**:
```
=== Test 4: Attempting to import os module ===
❌ FAIL: os module was not blocked!
```

This confirms the sandbox is actually controlling access.

---

## Step 7: Test Command Window Execution

The sandbox should also work in the Command Window REPL:

1. Click **Command Window** tab (bottom-left panel)
2. **Enable sandbox** in Script Editor Security menu
3. Type in Command Window: `import os`
4. Press Enter

**Expected Result**:
```
f:> import os
Error: Module 'os' is not allowed in sandbox environment
```

5. Type: `import math`
6. Press Enter

**Expected Result**:
```
f:> import math
[Success - no error]
```

---

## Verification Checklist

Use this checklist to verify all features:

### Security Features
- [ ] eval() is blocked
- [ ] exec() is blocked
- [ ] __import__() is blocked
- [ ] os module is blocked
- [ ] subprocess module is blocked
- [ ] open() is blocked
- [ ] math module is allowed
- [ ] random module is allowed
- [ ] json module is allowed
- [ ] Basic Python operations work

### UI Features
- [ ] Security menu is visible in Script Editor
- [ ] "Enable Sandbox" toggle works
- [ ] Status text changes (Active/Inactive)
- [ ] Status bar shows "SANDBOX ON" when enabled
- [ ] Status bar shows "Sandbox Off" when disabled
- [ ] Color coding works (green = on, gray = off)

### Integration
- [ ] Sandbox works with F5 (Run Script)
- [ ] Sandbox works with Ctrl+Enter (Run Section)
- [ ] Sandbox works with F9 (Run Selection)
- [ ] Sandbox works in Command Window REPL
- [ ] Output appears in Command Window tab
- [ ] Errors are displayed with red text

### Optional Advanced Tests
- [ ] Timeout test works (60s limit)
- [ ] Turning sandbox off allows blocked operations
- [ ] Re-enabling sandbox blocks operations again

---

## Expected Test Results Summary

| Test # | Description | Sandbox ON | Sandbox OFF |
|--------|-------------|------------|-------------|
| 1 | eval() | ✅ BLOCKED | ❌ ALLOWED |
| 2 | exec() | ✅ BLOCKED | ❌ ALLOWED |
| 3 | __import__() | ✅ BLOCKED | ❌ ALLOWED |
| 4 | os module | ✅ BLOCKED | ❌ ALLOWED |
| 5 | subprocess | ✅ BLOCKED | ❌ ALLOWED |
| 6 | open() | ✅ BLOCKED | ❌ ALLOWED |
| 7 | math module | ✅ ALLOWED | ✅ ALLOWED |
| 8 | random module | ✅ ALLOWED | ✅ ALLOWED |
| 9 | json module | ✅ ALLOWED | ✅ ALLOWED |
| 10 | Basic Python | ✅ WORKS | ✅ WORKS |
| 11 | Timeout (60s) | ✅ TIMEOUT | ⏱️ RUNS FOREVER |

---

## Troubleshooting

### "Security menu not visible"
- Ensure you're in Script Editor panel (not Command Window)
- Menu bar should show: File | Edit | Run | **Security**

### "Sandbox ON but tests fail (os module imports)"
- Check Command Window for actual error messages
- Verify spdlog output shows "Sandbox enabled"
- Try restarting the application

### "All tests show FAIL"
- Sandbox might not be enabled - check status bar
- Click Security → Enable Sandbox
- Re-run tests

### "Timeout test never completes"
- This is expected if sandbox is OFF
- Enable sandbox and try again
- Wait full 60 seconds for timeout

### "Module 'math' blocked even with sandbox"
- This shouldn't happen - math is whitelisted
- Check console logs for errors
- Report as bug

---

## Success Criteria

Phase 3 testing is **SUCCESSFUL** if:

1. ✅ All 10 security tests pass (Tests 1-10)
2. ✅ Sandbox can be enabled/disabled via UI
3. ✅ Status indicator updates correctly
4. ✅ Blocked operations fail with sandbox ON
5. ✅ Blocked operations succeed with sandbox OFF
6. ✅ Allowed operations work with sandbox ON/OFF
7. ✅ Output appears in Command Window
8. ✅ No crashes or exceptions

---

## Next Steps After Testing

Once testing is complete:

1. **Document results** - Note any failures or issues
2. **Commit UI changes** - Sandbox UI additions
3. **Update README** - Add test results
4. **Move to Phase 4** - Node↔Script Conversion
5. **Or merge to master** - If all phases complete

---

## Quick Test (5 minutes)

If you're short on time, run this abbreviated test:

1. Launch application
2. Open `test_sandbox_security.cyx`
3. Enable sandbox (Security → Enable Sandbox)
4. Verify status bar shows "SANDBOX ON" (green)
5. Press **F5** to run entire file
6. Check Command Window - all tests 1-10 should pass
7. Disable sandbox
8. Run Test 4 (os module) - should now succeed
9. Re-enable sandbox
10. Run Test 4 again - should be blocked

If all 10 steps work, Phase 3 is functional! ✅

---

## Reporting Issues

If you encounter issues, please provide:

1. Which test failed
2. Expected vs actual output
3. Sandbox status (ON/OFF)
4. Screenshot of Command Window output
5. Console/spdlog output if available

---

**Good luck testing!** 🧪🔒
