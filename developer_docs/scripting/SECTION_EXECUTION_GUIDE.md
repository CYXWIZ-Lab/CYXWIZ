# Section Execution Guide

## How to Run a Section with Ctrl+Enter

### ✅ Correct Cursor Placement

Place your cursor **anywhere BETWEEN the %% markers** (on any line with actual code or comments):

```python
%%
# Section 1: Basic Math          ← ✅ Cursor here works
print("=== Section 1 ===")       ← ✅ Cursor here works
x = 10                            ← ✅ Cursor here works
y = 20                            ← ✅ Cursor here works
result = x + y                    ← ✅ Cursor here works
print(f"x + y = {result}")        ← ✅ Cursor here works
%%
```

### ❌ Incorrect Cursor Placement

**DO NOT** place cursor on blank lines before the first `%%`:

```python
# CyxWiz Script Editor Test File  ← ❌ No section here
# This file demonstrates...        ← ❌ No section here
                                   ← ❌ No section here
%%                                 ← ⚠️ On marker - will try next section
# Section 1: Basic Math           ← ✅ This is where section starts
```

## Visual Example

```python
# ===== NO SECTION HERE =====
# File header comments
# More comments
                                   ← Blank lines have no section
%%  ← SECTION BOUNDARY
# ===== SECTION 1 STARTS HERE =====
print("Section 1")                 ← Press Ctrl+Enter here
x = 10                             ← Or here
%%  ← SECTION BOUNDARY
# ===== SECTION 2 STARTS HERE =====
print("Section 2")                 ← Press Ctrl+Enter here
y = 20                             ← Or here
%%  ← SECTION BOUNDARY
```

## Testing Steps

1. Open `test_script_sections.cyx` in Script Editor
2. Click on **line 5** (the comment `# Section 1: Basic Math`)
3. Press **Ctrl+Enter**
4. Check the **Command Window** tab for output:
   ```
   f:> Running script: test_script_sections.cyx (lines 4-9)
   === Section 1: Basic Math ===
   x + y = 30
   ```

## Troubleshooting

### "No section found at cursor"
- **Cause**: Cursor is on a blank line or before the first `%%`
- **Fix**: Move cursor to a line with code between `%%` markers

### Section runs wrong code
- **Cause**: `%%` markers are not properly paired
- **Fix**: Ensure every section has both opening and closing `%%`

### Syntax error
- **Cause**: `%%` is being sent to Python
- **Fix**: This should be fixed now - `%%` lines are automatically filtered

## Debug Mode

If you're having issues, check the console output for debug messages like:
```
GetCurrentSection: cursor at line 5, found 5 sections
  Section 0: lines 4-9
  Section 1: lines 13-18
  -> Found section containing cursor at lines 4-9
```

This shows:
- What line your cursor is on
- How many sections were detected
- The line ranges for each section
- Which section was selected
