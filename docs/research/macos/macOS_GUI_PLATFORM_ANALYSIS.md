# macOS Platform Support Analysis - CyxWiz GUI Panels

## Executive Summary
The GUI implementation contains several platform-specific code issues that can impact macOS functionality. While the basic framework has macOS considerations, there are notable gaps and inconsistencies, particularly around URL launching functionality and file path handling.

---

## Critical Issues Found

### 1. WALLET PANEL - Missing macOS URL Launcher Implementation
**File:** `/Volumes/Work/cyxwiz_lab/CYXWIZ/cyxwiz-server-node/src/gui/panels/wallet_panel.cpp`
**Severity:** HIGH

**Issue:** Lines 100-105
The "Explorer" button that opens the Solana blockchain explorer only implements Windows functionality. macOS and Linux versions are completely missing.

```cpp
if (ImGui::SmallButton(ICON_FA_ARROW_UP_RIGHT_FROM_SQUARE " Explorer")) {
    std::string url = "start https://explorer.solana.com/address/" + current_address + "?cluster=devnet";
#ifdef _WIN32
    std::system(url.c_str());
#endif  // <- No macOS or Linux implementation!
}
```

**Impact on macOS:**
- Clicking "Explorer" button does nothing on macOS
- Windows command "start" is hardcoded in the URL string itself, not in platform-specific code
- No fallback for other platforms

**Missing Code:**
```cpp
#elif __APPLE__
    std::string cmd = "open " + url;
    std::system(cmd.c_str());
#else
    std::string cmd = "xdg-open " + url;
    std::system(cmd.c_str());
#endif
```

---

### 2. ALLOCATION PANEL - Windows Registry Paths with Backslashes
**File:** `/Volumes/Work/cyxwiz_lab/CYXWIZ/cyxwiz-server-node/src/gui/panels/allocation_panel.cpp`
**Severity:** HIGH

**Issue:** Lines 608, 663, 665, 670
File paths use Windows-style backslashes which only work on Windows. macOS and Linux will fail silently when trying to create directories.

```cpp
// Lines 608, 663-665:
config_path = std::string(path) + "\\CyxWiz\\allocations.json";  // Windows only
config_path = std::string(path) + "\\CyxWiz";
CreateDirectoryA(config_path.c_str(), nullptr);
config_path += "\\allocations.json";
```

**Platform Coverage:**
- Line 113-148: Windows CPU allocation uses `GetSystemInfo()` and registry queries ✓
- Line 149-185: macOS CPU allocation uses `sysctlbyname()` ✓
- Line 186-199: Linux fallback with `hardware_concurrency()` ✓
- **BUT:** File path handling only correct on Windows; macOS uses wrong separators

**macOS Path Generated:**
`/Users/username\\CyxWiz\\allocations.json` (INCORRECT - mixed separators)

**Expected macOS Path:**
`/Users/username/.config/cyxwiz/allocations.json`

---

### 3. DASHBOARD PANEL - Windows API for CPU Metrics
**File:** `/Volumes/Work/cyxwiz_lab/CYXWIZ/cyxwiz-server-node/src/gui/panels/dashboard_panel.cpp`
**Severity:** MEDIUM

**Issues:**
1. Line 70-109: Windows CPU initialization uses registry queries (`RegOpenKeyExA`, `RegQueryValueExA`)
2. Line 110-167: macOS implementation provided but may have incomplete metrics collection
3. No CPU per-core metrics collection on macOS (only Windows implements `per_core_usage_` tracking)

**Windows-specific code (lines 70-109):**
```cpp
#ifdef _WIN32
    SYSTEM_INFO sysInfo;
    GetSystemInfo(&sysInfo);
    // ... registry access for CPU name and speed ...
#elif defined(__APPLE__)
    // macOS initialization (partial)
    size_t size = sizeof(cpu_logical_);
    if (sysctlbyname("hw.logicalcpu", &cpu_logical_, &size, nullptr, 0) == 0) {
        // ...
    }
```

**Missing on macOS:**
- Performance counter initialization (uses PDH on Windows)
- Real-time CPU usage metrics collection
- Per-core usage tracking for detailed graphs

---

### 4. HARDWARE PANEL - Registry Dependency
**File:** `/Volumes/Work/cyxwiz_lab/CYXWIZ/cyxwiz-server-node/src/gui/panels/hardware_panel.cpp`
**Severity:** MEDIUM

**Issue:** Line 67-108
Windows registry path used directly in CPU initialization:
```cpp
#ifdef _WIN32
    // Get CPU info from registry (Windows)
    if (RegOpenKeyExA(HKEY_LOCAL_MACHINE,
                      "HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0",
                      0, KEY_READ, &hKey) == ERROR_SUCCESS) {
```

**Registry Path Issue:**
- Uses hardcoded path: `"HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0"`
- This is Windows-specific and cannot be used on macOS
- macOS implementation provided (sysctlbyname) but less detailed

**macOS Gaps:**
- No vendor identification code for macOS (Windows queries "VendorIdentifier" from registry)
- CPU architecture detection differs between platforms (lines 182-188)

---

### 5. ACCOUNT SETTINGS PANEL - Inconsistent URL Launcher
**File:** `/Volumes/Work/cyxwiz_lab/CYXWIZ/cyxwiz-server-node/src/gui/panels/account_settings_panel.cpp`
**Severity:** MEDIUM

**Issues:**
1. Lines 497-503: Link Wallet button - properly implemented for all platforms ✓
2. Lines 569-575: Change Password button - properly implemented for all platforms ✓
3. Lines 585-591: Manage 2FA button - properly implemented for all platforms ✓
4. **BUT** Line 460-467: Generic wallet explorer launch has platform-agnostic URL construction issue

**Issue Details (lines 460-467):**
```cpp
#ifdef _WIN32
    std::string cmd = "start " + url;
#elif __APPLE__
    std::string cmd = "open " + url;
#else
    std::string cmd = "xdg-open " + url;
#endif
std::system(cmd.c_str());
```

This implementation is CORRECT, BUT there are other URL launches that lack consistency:
- Some use this pattern (correctly)
- Some hardcode "start" command (wallet_panel.cpp)

**Status:** Mostly correct but lacks consistency across codebase.

---

### 6. LOGIN PANEL - Platform Identifier Retrieval
**File:** `/Volumes/Work/cyxwiz_lab/CYXWIZ/cyxwiz-server-node/src/gui/panels/login_panel.cpp`
**Severity:** LOW

**Issue:** Lines 11-15
Includes platform-specific headers but actual hostname retrieval may differ:

```cpp
#ifdef _WIN32
#include <Windows.h>
#else
#include <unistd.h>  // for gethostname on Unix/macOS
#endif
```

**Status:** Actually implemented correctly for both platforms using standard functions.

---

## File Path Issues Summary

### Backslash Paths Found:
1. `allocation_panel.cpp:608` - `"\\CyxWiz\\allocations.json"`
2. `allocation_panel.cpp:663` - `"\\CyxWiz"`
3. `allocation_panel.cpp:665` - `"\\allocations.json"`
4. `dashboard_panel.cpp` - Registry path `"HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0"`
5. `hardware_panel.cpp` - Registry path `"HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0"`
6. `allocation_panel.cpp` - Registry path `"HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0"`

### Note on Registry Paths:
Registry paths use backslashes, which is correct for Windows API, but the concern is that these are only executed within `#ifdef _WIN32` blocks, which is proper.

**Real concern:** File system paths (allocations.json) use backslashes outside of Windows-specific code sections.

---

## GUI Panels with Platform-Specific Code

### 1. **AllocationPanel** - HIGH IMPACT
- CPU allocation: 3-way split ✓
- File path handling: WINDOWS ONLY ✗
- Config file loading: Different paths per platform (mostly correct with exceptions)

### 2. **DashboardPanel** - MEDIUM IMPACT
- CPU initialization: 3-way split ✓
- Metrics collection: Incomplete on macOS
- Per-core tracking: Windows only

### 3. **HardwarePanel** - MEDIUM IMPACT
- CPU info retrieval: 3-way split ✓
- Architecture detection: Platform-specific ✓
- Memory detection: Platform-specific ✓
- BUT: Vendor detection only on Windows

### 4. **AccountSettingsPanel** - LOW-MEDIUM IMPACT
- URL launching: Mostly consistent
- Clipboard operations: Uses ImGui (cross-platform) ✓
- File operations: None

### 5. **WalletPanel** - HIGH IMPACT
- URL launching: Inconsistent (missing macOS for Explorer button)
- Wallet address display: Cross-platform ✓

### 6. **LoginPanel** - LOW IMPACT
- Hostname retrieval: Properly handled
- Graphics: Cross-platform ImGui ✓

---

## Detailed Functionality Gaps on macOS

| Feature | Windows | macOS | Linux |
|---------|---------|-------|-------|
| CPU name retrieval | Registry query | sysctl | fallback |
| CPU cores detection | GetSystemInfo | sysctl | hardware_concurrency |
| Memory info | GlobalMemoryStatusEx | Mach API | fallback |
| Per-core CPU metrics | Windows API (PDH) | NOT IMPL | NOT IMPL |
| CPU frequency | Registry | sysctl | NOT IMPL |
| Config file path | AppData with `\\` | Home with `/` | Home with `/` |
| Link wallet URL open | START command | OPEN command | xdg-open |
| Explorer URL open | START (hardcoded) | NOT IMPL | NOT IMPL |
| GPU detection | Via daemon | Via daemon | Via daemon |
| Dark title bar | DwmSetWindowAttribute | NOT IMPL | NOT IMPL |

---

## TODO Comments Analysis

**Result:** No TODO or FIXME comments related to cross-platform support were found in the GUI code.

---

## Recommendations

### CRITICAL (Fix Immediately):
1. **wallet_panel.cpp:100-105** - Implement macOS/Linux URL opening for Explorer button
   ```cpp
   std::string url = "https://explorer.solana.com/address/" + current_address + "?cluster=devnet";
   #ifdef _WIN32
       std::string cmd = "start " + url;
   #elif __APPLE__
       std::string cmd = "open " + url;
   #else
       std::string cmd = "xdg-open " + url;
   #endif
   std::system(cmd.c_str());
   ```

2. **allocation_panel.cpp:608, 663-665, 670** - Fix Windows-style path separators
   Use `std::filesystem::path` or platform-independent path construction:
   ```cpp
   #ifdef _WIN32
       config_path = std::string(path) + "\\CyxWiz\\allocations.json";
   #else
       config_path = std::string(getenv("HOME")) + "/.config/cyxwiz/allocations.json";
   #endif
   ```

### HIGH PRIORITY (Within Sprint):
3. **dashboard_panel.cpp** - Implement macOS CPU metrics collection
   - Add real-time CPU usage tracking for macOS
   - Implement per-core metrics using Mach APIs

4. **hardware_panel.cpp** - Add vendor string detection for macOS
   - Use sysctl `machdep.cpu.vendor` for CPU vendor detection

### MEDIUM PRIORITY (Next Sprint):
5. Establish consistent URL-opening pattern across all panels
6. Add tests for platform-specific code paths on macOS
7. Document platform-specific limitations in code comments

---

## Files Requiring Changes

1. `/Volumes/Work/cyxwiz_lab/CYXWIZ/cyxwiz-server-node/src/gui/panels/wallet_panel.cpp`
2. `/Volumes/Work/cyxwiz_lab/CYXWIZ/cyxwiz-server-node/src/gui/panels/allocation_panel.cpp`
3. `/Volumes/Work/cyxwiz_lab/CYXWIZ/cyxwiz-server-node/src/gui/panels/dashboard_panel.cpp`
4. `/Volumes/Work/cyxwiz_lab/CYXWIZ/cyxwiz-server-node/src/gui/panels/hardware_panel.cpp`

