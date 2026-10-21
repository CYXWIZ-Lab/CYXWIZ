# Detailed macOS Platform Issues - Code Level Analysis

## Issue #1: Wallet Panel Explorer Button (CRITICAL)

**Location:** `/Volumes/Work/cyxwiz_lab/CYXWIZ/cyxwiz-server-node/src/gui/panels/wallet_panel.cpp`
**Lines:** 100-105
**Severity:** CRITICAL - Feature completely broken on macOS/Linux

### Current Code:
```cpp
if (ImGui::SmallButton(ICON_FA_ARROW_UP_RIGHT_FROM_SQUARE " Explorer")) {
    std::string url = "start https://explorer.solana.com/address/" + current_address + "?cluster=devnet";
#ifdef _WIN32
    std::system(url.c_str());
#endif
}
```

### Problems:
1. `"start"` command is hardcoded in the URL string, not in platform-specific section
2. No implementation for macOS (`#elif __APPLE__`)
3. No fallback for Linux
4. On macOS, clicking button produces no effect
5. The URL string is malformed for non-Windows - it concatenates "start" with URL

### Expected Result on macOS:
Should execute: `open https://explorer.solana.com/address/ADDR?cluster=devnet`
Actual Result: No action

### Correct Implementation:
```cpp
if (ImGui::SmallButton(ICON_FA_ARROW_UP_RIGHT_FROM_SQUARE " Explorer")) {
    std::string url = "https://explorer.solana.com/address/" + current_address + "?cluster=devnet";
#ifdef _WIN32
    std::string cmd = "start " + url;
#elif __APPLE__
    std::string cmd = "open " + url;
#else
    std::string cmd = "xdg-open " + url;
#endif
    std::system(cmd.c_str());
}
```

---

## Issue #2: Allocation Panel File Paths (CRITICAL)

**Location:** `/Volumes/Work/cyxwiz_lab/CYXWIZ/cyxwiz-server-node/src/gui/panels/allocation_panel.cpp`
**Lines:** 608, 663, 665, 670
**Severity:** CRITICAL - Config files cannot be saved/loaded on macOS

### Current Code (LoadAllocations):
```cpp
// Lines 601-612
void AllocationPanel::LoadAllocations() {
    std::string config_path;

#ifdef _WIN32
    char path[MAX_PATH];
    if (SUCCEEDED(SHGetFolderPathA(nullptr, CSIDL_APPDATA, nullptr, 0, path))) {
        config_path = std::string(path) + "\\CyxWiz\\allocations.json";  // <-- CORRECT (inside #ifdef)
    }
#else
    config_path = std::string(getenv("HOME")) + "/.config/cyxwiz/allocations.json";  // <-- CORRECT
#endif
```

**Wait - This section IS correct!**

### Current Code (SaveAllocations) - THE PROBLEM:
```cpp
// Lines 657-672
void AllocationPanel::SaveAllocations() {
    std::string config_path;

#ifdef _WIN32
    char path[MAX_PATH];
    if (SUCCEEDED(SHGetFolderPathA(nullptr, CSIDL_APPDATA, nullptr, 0, path))) {
        config_path = std::string(path) + "\\CyxWiz";           // Line 663: WINDOWS ONLY
        CreateDirectoryA(config_path.c_str(), nullptr);
        config_path += "\\allocations.json";                     // Line 665: WINDOWS ONLY
    }
#else
    config_path = std::string(getenv("HOME")) + "/.config/cyxwiz";
    mkdir(config_path.c_str(), 0755);                           // CORRECT for Unix/macOS
    config_path += "/allocations.json";                         // CORRECT for Unix/macOS
#endif
```

### The Issue:
The backslash paths are correctly wrapped in `#ifdef _WIN32`, so actually this is CORRECT!

**Re-analysis needed:** Looking more carefully:
- Lines 608-609 (LoadAllocations): Properly guarded ✓
- Lines 663-665 (SaveAllocations): Properly guarded with `#ifdef _WIN32` ✓
- The paths are NOT actually incorrectly used outside the guards

**Status Update:** This file appears to be CORRECT upon closer inspection. The backslashes are properly contained within Windows-specific blocks.

---

## Issue #3: Dashboard Panel CPU Metrics (MEDIUM)

**Location:** `/Volumes/Work/cyxwiz_lab/CYXWIZ/cyxwiz-server-node/src/gui/panels/dashboard_panel.cpp`
**Lines:** 70-167
**Severity:** MEDIUM - Incomplete functionality on macOS

### Windows Implementation (Lines 70-109):
```cpp
#ifdef _WIN32
    SYSTEM_INFO sysInfo;
    GetSystemInfo(&sysInfo);
    cpu_logical_ = sysInfo.dwNumberOfProcessors;
    cpu_cores_ = cpu_logical_ / 2;
    
    // Get CPU name from registry
    HKEY hKey;
    if (RegOpenKeyExA(HKEY_LOCAL_MACHINE,
                      "HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0",
                      0, KEY_READ, &hKey) == ERROR_SUCCESS) {
        char cpuName[256] = {0};
        DWORD bufSize = sizeof(cpuName);
        if (RegQueryValueExA(hKey, "ProcessorNameString", nullptr, nullptr,
                            (LPBYTE)cpuName, &bufSize) == ERROR_SUCCESS) {
            cpu_name_ = cpuName;
        }
        // ... frequency and more registry queries ...
        RegCloseKey(hKey);
    }
    // Initialize per-core history with PDH (Performance Data Helper)
    per_core_usage_.resize(cpu_logical_, 0.0f);
    per_core_history_.resize(cpu_logical_);
```

### macOS Implementation (Lines 110-167):
```cpp
#elif defined(__APPLE__)
    // macOS CPU initialization
    size_t size = sizeof(cpu_logical_);
    if (sysctlbyname("hw.logicalcpu", &cpu_logical_, &size, nullptr, 0) == 0) {
        // Successfully got logical CPU count
    } else {
        cpu_logical_ = 1;
    }
    
    // ... similar basic initialization ...
    
    // Initialize per-core history
    per_core_usage_.resize(cpu_logical_, 0.0f);
    per_core_history_.resize(cpu_logical_);
    for (auto& hist : per_core_history_) {
        hist.resize(HISTORY_SIZE, 0.0f);
    }
```

### Gap Analysis:
Both initialize the data structures correctly, BUT:

1. **Windows has PDH (Performance Data Helper)** initialization that macOS lacks
   - Windows: `#include <pdh.h>` with `#pragma comment(lib, "pdh.lib")`
   - macOS: No equivalent real-time performance counter API

2. **Real-time metrics collection** is not shown in the init, but would use:
   - Windows: PDH APIs to query CPU usage
   - macOS: procfs or Mach APIs (not found in code)

3. **Missing headers on macOS:**
   - `#include <mach/mach.h>` exists in HardwarePanel but not DashboardPanel
   - CPU usage tracking needs Mach kernel interfaces

### Consequence:
- Dashboard displays CPU usage metrics on Windows (collected via PDH)
- Dashboard on macOS shows CPU count and name but not actual usage
- Per-core graphs show zeros on macOS

---

## Issue #4: Hardware Panel CPU Vendor Detection (MEDIUM)

**Location:** `/Volumes/Work/cyxwiz_lab/CYXWIZ/cyxwiz-server-node/src/gui/panels/hardware_panel.cpp`
**Lines:** 67-108 (Windows), 133-213 (macOS)
**Severity:** MEDIUM - Missing vendor information on macOS

### Windows CPU Vendor Detection (Lines 93-98):
```cpp
char vendorId[256] = {0};
bufSize = sizeof(vendorId);
if (RegQueryValueExA(hKey, "VendorIdentifier", nullptr, nullptr,
                    (LPBYTE)vendorId, &bufSize) == ERROR_SUCCESS) {
    cpu_info_.vendor = vendorId;
}
```

### macOS CPU Vendor Detection (Lines 162-168):
```cpp
// Get CPU vendor
char cpu_vendor[256] = {0};
size = sizeof(cpu_vendor);
if (sysctlbyname("machdep.cpu.vendor", cpu_vendor, &size, nullptr, 0) == 0) {
    cpu_info_.vendor = cpu_vendor;
} else {
    cpu_info_.vendor = "Unknown";
}
```

### Status:
Actually BOTH platforms have vendor detection implemented!
- Windows uses registry path: `"VendorIdentifier"`
- macOS uses sysctl: `"machdep.cpu.vendor"`

**This issue is actually resolved.** macOS does have vendor detection.

---

## Issue #5: Account Settings Panel URL Opening (LOW)

**Location:** `/Volumes/Work/cyxwiz_lab/CYXWIZ/cyxwiz-server-node/src/gui/panels/account_settings_panel.cpp`
**Lines:** 460-467, 497-503, 569-575, 585-591
**Severity:** LOW - Consistency issue, but functional

### Inconsistent Implementation #1 (Lines 460-467) - CORRECT PATTERN:
```cpp
if (ImGui::Button("View Wallet Details")) {
    std::string url = "https://cyxwiz.com/dashboard/wallet";
#ifdef _WIN32
    std::string cmd = "start " + url;
#elif __APPLE__
    std::string cmd = "open " + url;
#else
    std::string cmd = "xdg-open " + url;
#endif
    std::system(cmd.c_str());
}
```

### Consistent Implementation #2 (Lines 497-503) - CORRECT PATTERN:
```cpp
if (ImGui::Button(ICON_FA_ARROW_UP_RIGHT_FROM_SQUARE " Link Wallet on Web Dashboard")) {
    // Open web dashboard
#ifdef _WIN32
    std::system("start https://cyxwiz.com/dashboard/wallet");
#elif __APPLE__
    std::system("open https://cyxwiz.com/dashboard/wallet");
#else
    std::system("xdg-open https://cyxwiz.com/dashboard/wallet");
#endif
}
```

### Analysis:
- Implementation #1 constructs URL separately, then command
- Implementation #2 builds entire command including URL
- Both work correctly, just inconsistent style
- macOS gets "open" command in both cases ✓

**Status:** Functional on macOS but code consistency could be improved.

---

## Issue #6: Server Application Dark Title Bar (LOW)

**Location:** `/Volumes/Work/cyxwiz_lab/CYXWIZ/cyxwiz-server-node/src/gui/server_application.cpp`
**Lines:** 180-187
**Severity:** LOW - Visual polish missing

### Current Code:
```cpp
#ifdef _WIN32
    // Enable dark title bar on Windows 10/11
    HWND hwnd = glfwGetWin32Window(window_);
    if (hwnd) {
        BOOL dark = TRUE;
        DwmSetWindowAttribute(hwnd, 20, &dark, sizeof(dark));  // DWMWA_USE_IMMERSIVE_DARK_MODE
    }
#endif
```

### Issue:
- No macOS equivalent for dark title bar
- macOS handles this automatically based on system appearance
- No Linux support needed (Linux WMs vary)

### Impact:
- Windows gets dark title bar automatically
- macOS title bar may not match theme (not critical)

---

## Summary of Actual vs Reported Issues

| Issue | Initial Assessment | Actual Status | Action Needed |
|-------|-------------------|---------------|---------------|
| Wallet Explorer | CRITICAL | CRITICAL | FIX NOW |
| Allocation Paths | CRITICAL | Actually OK | Verify/Document |
| Dashboard Metrics | MEDIUM | MEDIUM | Add Mach API support |
| Hardware Vendor | MEDIUM | Actually OK | None |
| Settings URLs | LOW | LOW | Minor cleanup |
| Dark Title Bar | LOW | LOW | Cosmetic only |

---

## Code Changes Required

### PRIORITY 1: wallet_panel.cpp (Lines 100-105)
**Estimated effort:** 2 minutes

```diff
         if (ImGui::SmallButton(ICON_FA_ARROW_UP_RIGHT_FROM_SQUARE " Explorer")) {
-            std::string url = "start https://explorer.solana.com/address/" + current_address + "?cluster=devnet";
+            std::string url = "https://explorer.solana.com/address/" + current_address + "?cluster=devnet";
 #ifdef _WIN32
-            std::system(url.c_str());
+            std::string cmd = "start " + url;
+            std::system(cmd.c_str());
+#elif __APPLE__
+            std::string cmd = "open " + url;
+            std::system(cmd.c_str());
+#else
+            std::string cmd = "xdg-open " + url;
+            std::system(cmd.c_str());
 #endif
         }
```

### PRIORITY 2: dashboard_panel.cpp (CPU Metrics Collection)
**Estimated effort:** 4-6 hours

Needs:
1. Add `#include <mach/mach.h>` for macOS
2. Implement Mach kernel API calls for CPU usage
3. Add periodic update thread or callback for metrics
4. Store historical data in `cpu_history_` vector

### PRIORITY 3: Account Settings Consistency (Optional)
**Estimated effort:** 30 minutes
Standardize URL opening pattern across all 4 URL buttons

---

## Testing Recommendations

1. **Test wallet_panel Explorer button on macOS**
   - Button should open Safari with Solana explorer
   - Verify URL is correctly formatted

2. **Test allocation_panel config file I/O**
   - Verify allocations.json is created in `~/.config/cyxwiz/`
   - Verify settings persist after restart

3. **Test dashboard_panel on macOS**
   - Monitor CPU usage display
   - Check if metrics update in real-time

4. **Add CI tests for platform-specific code**
   - Run on macOS runner
   - Verify all URL launches work
   - Verify config file I/O works
