# Phase 5 Testing Checklist

## Build Status
✅ **Engine built successfully**
- Location: `build\windows-release\bin\Release\cyxwiz-engine.exe`
- Size: 1.8 MB
- Build time: Nov 17 07:19

## Test Plan

### Test 1: Startup Scripts
**Objective:** Verify startup scripts execute automatically on launch

**Steps:**
1. ✅ Enable startup scripts in `startup_scripts.txt`
2. Launch `cyxwiz-engine.exe`
3. Check CommandWindow for output
4. Verify welcome message appears
5. Verify imports are loaded

**Expected Output in CommandWindow:**
```
=== Running startup scripts ===
[scripts/startup/welcome.cyx]
============================================================
   Welcome to CyxWiz Engine!
============================================================

Quick Tips:
  • Press Tab for auto-completion
  • Use %% to mark code sections
  • F5 to run entire script
  • Ctrl+Enter to run current section
  • Security menu to enable/disable sandbox

Try these commands:
  import math; math.sqrt(16)
  help(math)
  dir()

Happy coding! 🚀
============================================================

[scripts/startup/init_imports.cyx]
=== Initializing common imports ===
  ✓ math
  ✓ random
  ✓ json
  ✗ numpy not available
  ✗ pandas not available
  ✗ pycyxwiz not available
=== Imports complete ===

=== Startup scripts completed ===
Executed: 2 scripts
Time: ~0.5 seconds
```

### Test 2: Auto-Completion (Tab Key)
**Objective:** Verify Tab completion works in CommandWindow

**Steps:**
1. Open CommandWindow
2. Type: `mat` and press Tab
3. Verify completion suggests `math`
4. Type: `math.sq` and press Tab
5. Verify completion suggests `math.sqrt`

**Expected:**
- Completion popup appears
- Arrow keys navigate options
- Enter/Tab accepts selection

### Test 3: Script Templates - Data Loading
**Objective:** Test data_loading.cyx template

**Steps:**
1. Open Script Editor
2. Copy contents of `scripts/templates/data_loading.cyx`
3. Verify TODO markers are present
4. Update file path to `test_data.csv`
5. Run script (F5)
6. Check output shows data loaded

**Expected:**
```
============================================================
Data Loading Script
============================================================

Loading data from: test_data.csv

✓ Loaded 10 rows, 4 columns

=== Dataset Information ===
Rows: 10
Columns: 4

Column Names:
  1. Name
  2. Age
  3. Score
  4. City

First 5 rows:
  Row 1: ['Alice', '25', '95.5', 'New York']
  Row 2: ['Bob', '30', '88.2', 'Los Angeles']
  ...

============================================================
Data loading complete!
============================================================
```

### Test 4: Script Templates - Plotting
**Objective:** Test plotting.cyx template (if matplotlib installed)

**Steps:**
1. Check if matplotlib is available
2. Copy `scripts/templates/plotting.cyx`
3. Run script
4. Verify plots are generated
5. Check for PNG files in current directory

**Expected:**
- line_plot.png created
- scatter_plot.png created
- bar_chart.png created
- histogram.png created
- multiple_series.png created

### Test 5: Script Templates - Custom Function
**Objective:** Test custom_function.cyx template

**Steps:**
1. Copy `scripts/templates/custom_function.cyx`
2. Uncomment `test_functions()` in main
3. Run script
4. Verify all test functions execute

**Expected:**
```
============================================================
Testing Custom Functions
============================================================

[Test 1] my_function:
  Result: 30

[Test 2] typed_function:
  Filtered: [0.6, 0.8]

[Test 3] safe_divide:
  10 / 2 = 5.0
  Warning: Division by zero attempted
  10 / 0 = None

[Test 4] calculate_statistics:
  Mean: 5.5, Median: 5, Min: 1, Max: 10

[Test 5] MyUtilityClass:
  Processed: [11, 12, 13]

============================================================
All tests complete!
============================================================
```

### Test 6: File Format Handlers
**Objective:** Test DataTable with TableViewer

**Steps:**
1. Open Asset Browser or File menu
2. Load `test_data.csv`
3. Verify TableViewer opens
4. Check pagination works
5. Test column sorting/filtering

**Expected:**
- Data loads into table
- 10 rows visible
- Column headers show: Name, Age, Score, City
- Can scroll and navigate

### Test 7: Startup Script Error Handling
**Objective:** Verify error handling with bad script

**Steps:**
1. Create `bad_script.cyx` with syntax error:
   ```python
   print("Starting"
   # Missing closing parenthesis
   ```
2. Add to `startup_scripts.txt`
3. Launch engine
4. Verify error is caught and displayed
5. Verify engine continues to load

**Expected:**
```
=== Running startup scripts ===
[bad_script.cyx]
✗ Script execution failed
Error: SyntaxError: invalid syntax

=== Startup scripts completed ===
Executed: 2 scripts (1 failed)
Time: ~0.5 seconds
```

### Test 8: Startup Script Timeout
**Objective:** Verify timeout protection

**Steps:**
1. Create `slow_script.cyx`:
   ```python
   import time
   print("Starting slow script...")
   time.sleep(35)  # Exceeds 30s timeout
   print("Done")
   ```
2. Add to startup_scripts.txt
3. Launch engine
4. Verify script times out after 30s

**Expected:**
```
[slow_script.cyx]
⏱ Script timeout after 30 seconds
```

## Test Results

| Test | Status | Notes |
|------|--------|-------|
| 1. Startup Scripts | ⏳ Pending | - |
| 2. Auto-Completion | ⏳ Pending | - |
| 3. Data Loading Template | ⏳ Pending | - |
| 4. Plotting Template | ⏳ Pending | Requires matplotlib |
| 5. Custom Function Template | ⏳ Pending | - |
| 6. File Format Handlers | ⏳ Pending | - |
| 7. Error Handling | ⏳ Pending | - |
| 8. Timeout Protection | ⏳ Pending | - |

## Manual Testing Instructions

**Launch Engine:**
```bash
cd D:\Dev\CyxWiz_Claude
.\build\windows-release\bin\Release\cyxwiz-engine.exe
```

**Check CommandWindow:**
- Look for startup script output immediately after launch
- Verify welcome message and imports

**Test Templates:**
1. Navigate to Script Editor panel
2. File → Open → `scripts/templates/data_loading.cyx`
3. Customize TODO sections
4. Press F5 to run

**Test Auto-Completion:**
1. Click in CommandWindow input area
2. Type partial identifier (e.g., "mat")
3. Press Tab key
4. Observe completion suggestions

## Known Limitations

**Current Implementation:**
- ✅ Startup scripts execute automatically
- ✅ Templates available for manual use
- ⏳ No GUI template browser yet (planned)
- ⏳ No File → New → From Template menu yet (planned)
- ⏳ No safe mode (hold Shift to skip) yet (planned)

**Dependencies:**
- numpy/pandas may not be installed (expected warnings)
- matplotlib needed for plotting template
- pycyxwiz may not be built yet

## Notes for User

**First Launch:**
The engine will execute startup scripts and display output in CommandWindow. You should see:
1. Welcome message with tips
2. Import status (some may fail if libraries not installed)
3. Total execution time

**Using Templates:**
Templates are in `scripts/templates/` directory:
- data_loading.cyx
- model_training.cyx
- plotting.cyx
- custom_function.cyx
- data_processing.cyx

Copy to your workspace and customize TODO sections.

**Customizing Startup Scripts:**
Edit `startup_scripts.txt` to add/remove scripts. Comment lines with #.

---

**Testing Date:** 2025-11-17
**Build Version:** Phase 5 Complete (scripting branch)
**Tester:** Ready for user testing
