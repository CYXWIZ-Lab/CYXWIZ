# CyxWiz Engine Docking Layout - Code Reference

Quick reference for the docking layout implementation.

## File Locations

```
cyxwiz-engine/src/gui/
├── main_window.h           // Layout structure and state
├── main_window.cpp         // Layout construction logic
└── panels/
    ├── toolbar.h           // Menu callback definition
    └── toolbar.cpp         // Menu implementation
```

## Key Code Snippets

### 1. Docking Layout Construction

**Location:** `cyxwiz-engine/src/gui/main_window.cpp` → `BuildInitialDockLayout()`

```cpp
void MainWindow::BuildInitialDockLayout() {
    ImGuiID dockspace_id = ImGui::GetID("CyxWizDockSpace");

    // Clear and initialize
    ImGui::DockBuilderRemoveNode(dockspace_id);
    ImGui::DockBuilderAddNode(dockspace_id, ImGuiDockNodeFlags_DockSpace);
    ImGui::DockBuilderSetNodeSize(dockspace_id, ImGui::GetMainViewport()->Size);

    // Create dock hierarchy
    ImGuiID dock_id_left = 0;
    ImGuiID dock_id_right = 0;
    ImGuiID dock_id_center = dockspace_id;
    ImGuiID dock_id_bottom = 0;
    ImGuiID dock_id_bottom_left = 0;
    ImGuiID dock_id_bottom_right = 0;
    ImGuiID dock_id_bottom_bottom = 0;

    // Split operations (order matters!)
    dock_id_left = ImGui::DockBuilderSplitNode(dock_id_center, ImGuiDir_Left, 0.15f, nullptr, &dock_id_center);
    dock_id_right = ImGui::DockBuilderSplitNode(dock_id_center, ImGuiDir_Right, 0.25f, nullptr, &dock_id_center);
    dock_id_bottom = ImGui::DockBuilderSplitNode(dock_id_center, ImGuiDir_Down, 0.30f, nullptr, &dock_id_center);
    dock_id_bottom_right = ImGui::DockBuilderSplitNode(dock_id_bottom, ImGuiDir_Right, 0.25f, nullptr, &dock_id_bottom_left);
    dock_id_bottom_bottom = ImGui::DockBuilderSplitNode(dock_id_bottom_left, ImGuiDir_Down, 0.40f, nullptr, &dock_id_bottom_left);

    // Dock windows (names must match ImGui::Begin() calls)
    ImGui::DockBuilderDockWindow("Asset Browser", dock_id_left);
    ImGui::DockBuilderDockWindow("Node Editor", dock_id_center);
    ImGui::DockBuilderDockWindow("Properties", dock_id_right);
    ImGui::DockBuilderDockWindow("Console", dock_id_bottom_left);
    ImGui::DockBuilderDockWindow("Training Dashboard", dock_id_bottom_right);
    ImGui::DockBuilderDockWindow("Viewport", dock_id_bottom_bottom);

    // Finalize
    ImGui::DockBuilderFinish(dockspace_id);
}
```

### 2. First-Run Check

**Location:** `cyxwiz-engine/src/gui/main_window.cpp` → `RenderDockSpace()`

```cpp
void MainWindow::RenderDockSpace() {
    // ... window setup ...

    ImGuiIO& io = ImGui::GetIO();
    if (io.ConfigFlags & ImGuiConfigFlags_DockingEnable) {
        ImGuiID dockspace_id = ImGui::GetID("CyxWizDockSpace");

        // Build layout only once on first run
        if (first_time_layout_) {
            BuildInitialDockLayout();
            first_time_layout_ = false;
        }

        ImGui::DockSpace(dockspace_id, ImVec2(0.0f, 0.0f), dockspace_flags);
    }

    ImGui::End();
}
```

### 3. Reset Layout Function

**Location:** `cyxwiz-engine/src/gui/main_window.cpp`

```cpp
void MainWindow::ResetDockLayout() {
    first_time_layout_ = true;  // Force rebuild on next frame
    spdlog::info("Dock layout reset requested");
}
```

### 4. Callback Setup

**Location:** `cyxwiz-engine/src/gui/main_window.cpp` → `MainWindow::MainWindow()`

```cpp
MainWindow::MainWindow()
    : show_about_dialog_(false), show_demo_window_(false), first_time_layout_(true) {

    // ... panel creation ...

    // Connect toolbar reset button to main window function
    toolbar_->SetResetLayoutCallback([this]() {
        this->ResetDockLayout();
    });

    spdlog::info("MainWindow initialized with docking layout system");
}
```

### 5. Toolbar Menu Integration

**Location:** `cyxwiz-engine/src/gui/panels/toolbar.cpp` → `RenderViewMenu()`

```cpp
void ToolbarPanel::RenderViewMenu() {
    if (ImGui::BeginMenu("View")) {
        // Panel toggles
        ImGui::MenuItem("Asset Browser", nullptr, true);
        ImGui::MenuItem("Node Editor", nullptr, true);
        ImGui::MenuItem("Properties", nullptr, true);
        ImGui::MenuItem("Console", nullptr, true);
        ImGui::MenuItem("Training Dashboard", nullptr, true);
        ImGui::MenuItem("Viewport (Profiler)", nullptr, true);

        ImGui::Separator();

        if (ImGui::BeginMenu("Layout")) {
            if (ImGui::MenuItem("Reset to Default Layout")) {
                if (reset_layout_callback_) {
                    reset_layout_callback_();  // Call the callback
                }
            }
            ImGui::Separator();
            if (ImGui::MenuItem("Save Layout...")) {
                // TODO: Save layout
            }
            if (ImGui::MenuItem("Load Layout...")) {
                // TODO: Load layout
            }
            ImGui::EndMenu();
        }

        ImGui::Separator();
        if (ImGui::MenuItem("Fullscreen", "F11")) {
            // TODO: Toggle fullscreen
        }

        ImGui::EndMenu();
    }
}
```

## Split Node Parameters Explained

### `ImGui::DockBuilderSplitNode()` signature:
```cpp
ImGuiID DockBuilderSplitNode(
    ImGuiID node_id,              // Node to split
    ImGuiDir split_dir,           // Direction: Left/Right/Up/Down
    float size_ratio_for_node_at_dir,  // Size ratio (0.0 to 1.0)
    ImGuiID* out_id_at_dir,       // Output: ID of new node in split direction
    ImGuiID* out_id_at_opposite_dir  // Output: ID of remaining node
);
```

### Example: Split Left 15%
```cpp
// Before: dock_id_center is full width
ImGuiID dock_id_left = ImGui::DockBuilderSplitNode(
    dock_id_center,    // Split this node
    ImGuiDir_Left,     // Create new node on left
    0.15f,             // New node gets 15% of width
    nullptr,           // We don't need the left ID separately (return value has it)
    &dock_id_center    // Update center to be the remaining 85%
);
// After: dock_id_left = 15% left side, dock_id_center = 85% right side
```

### Split Ratio Guidelines

| Panel Position | Ratio | Explanation |
|---------------|-------|-------------|
| Left (Asset Browser) | 0.15 | 15% of total width |
| Right (Properties) | 0.25 | 25% of remaining width after left split (~21% total) |
| Bottom | 0.30 | 30% of center height |
| Bottom-Right (Inspector) | 0.25 | 25% of bottom width |
| Bottom-Bottom (Profiler) | 0.40 | 40% of bottom-left height |

## Window Name Mapping

**CRITICAL:** Window names in `DockBuilderDockWindow()` must EXACTLY match `ImGui::Begin()` calls.

| C++ Class | ImGui::Begin() Name | DockBuilder Name | Location |
|-----------|--------------------|--------------------|----------|
| AssetBrowserPanel | "Asset Browser" | "Asset Browser" | Left |
| NodeEditor | "Node Editor" | "Node Editor" | Center |
| Properties | "Properties" | "Properties" | Right |
| Console | "Console" | "Console" | Bottom-Left |
| TrainingDashboardPanel | "Training Dashboard" | "Training Dashboard" | Bottom-Right |
| Viewport | "Viewport" | "Viewport" | Bottom-Bottom |

### Verifying Window Names

Search each panel's `.cpp` file for `ImGui::Begin()`:

```bash
# Example for Console
grep "ImGui::Begin" cyxwiz-engine/src/gui/console.cpp
# Output: ImGui::Begin("Console", &show_window_)
```

## Layout Hierarchy Diagram

```
dockspace_id (CyxWizDockSpace)
├─ dock_id_left (15%)
│  └─ "Asset Browser"
│
├─ dock_id_center (remaining width, 70% height)
│  └─ "Node Editor"
│
├─ dock_id_right (25% of remaining width)
│  └─ "Properties"
│
└─ dock_id_bottom (30% of center height)
   ├─ dock_id_bottom_left (75% of bottom width)
   │  ├─ top part (60% height)
   │  │  └─ "Console"
   │  │
   │  └─ dock_id_bottom_bottom (40% height)
   │     └─ "Viewport"
   │
   └─ dock_id_bottom_right (25% of bottom width)
      └─ "Training Dashboard"
```

## Debugging Tips

### 1. Check Window Names
If a panel doesn't dock correctly, verify names match:
```cpp
// In panel .cpp file
ImGui::Begin("Window Name", &show_window_);

// In BuildInitialDockLayout()
ImGui::DockBuilderDockWindow("Window Name", dock_id);
```

### 2. Enable ImGui Demo Window
Uncomment in main_window.cpp to see dock IDs:
```cpp
show_demo_window_ = true;  // In constructor
```

### 3. Add Logging
```cpp
spdlog::info("Dock ID left: {}", dock_id_left);
spdlog::info("Dock ID center: {}", dock_id_center);
```

### 4. Clear ImGui.ini
Delete `imgui.ini` to force fresh layout:
```bash
rm imgui.ini
```

## Common Modifications

### Change Panel Size

**Make Asset Browser wider (20% instead of 15%):**
```cpp
dock_id_left = ImGui::DockBuilderSplitNode(dock_id_center, ImGuiDir_Left, 0.20f, nullptr, &dock_id_center);
```

### Add New Panel

**Add a new panel to bottom-right area:**
```cpp
// 1. Split the bottom-right dock
ImGuiID dock_id_new = ImGui::DockBuilderSplitNode(
    dock_id_bottom_right,
    ImGuiDir_Down,
    0.50f,  // 50/50 split
    nullptr,
    &dock_id_bottom_right
);

// 2. Dock the window
ImGui::DockBuilderDockWindow("New Panel Name", dock_id_new);
```

### Move Panel to Different Location

**Move Viewport to right column instead of bottom:**
```cpp
// Change from:
ImGui::DockBuilderDockWindow("Viewport", dock_id_bottom_bottom);

// To:
ImGuiID dock_id_right_bottom = ImGui::DockBuilderSplitNode(
    dock_id_right,
    ImGuiDir_Down,
    0.50f,
    nullptr,
    &dock_id_right
);
ImGui::DockBuilderDockWindow("Viewport", dock_id_right_bottom);
```

## ImGui Flags Reference

### DockSpace Flags
```cpp
ImGuiDockNodeFlags_None              // No special flags
ImGuiDockNodeFlags_KeepAliveOnly     // Don't display, only keep alive
ImGuiDockNodeFlags_NoDockingOverCentralNode  // Prevent docking in center
ImGuiDockNodeFlags_PassthruCentralNode       // Make center node transparent
```

### Window Flags for DockSpace Window
```cpp
ImGuiWindowFlags_NoDocking           // Can't dock into this window
ImGuiWindowFlags_NoTitleBar          // No title bar
ImGuiWindowFlags_NoCollapse          // No collapse button
ImGuiWindowFlags_NoResize            // Can't resize
ImGuiWindowFlags_NoMove              // Can't move
ImGuiWindowFlags_NoBringToFrontOnFocus  // Don't bring to front on focus
ImGuiWindowFlags_NoNavFocus          // Don't take navigation focus
```

## Performance Notes

- Layout construction: ~1ms on first frame only
- DockSpace rendering: <0.1ms per frame
- No impact on panel rendering performance
- Layout state cached by ImGui in .ini file

## Dependencies

### Headers Required
```cpp
#include <imgui.h>          // Main ImGui API
#include <imgui_internal.h> // DockBuilder API (internal API)
#include <spdlog/spdlog.h>  // Logging
```

### ImGui Configuration Required
```cpp
// In application.cpp or main.cpp
ImGuiIO& io = ImGui::GetIO();
io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
io.ConfigFlags |= ImGuiConfigFlags_ViewportsEnable;
```

## Testing Checklist

- [ ] Build succeeds on all platforms (Windows/macOS/Linux)
- [ ] Initial layout appears correctly on first run
- [ ] All panels visible and in expected positions
- [ ] Panel sizes approximately match specified ratios
- [ ] Can resize panels by dragging splitters
- [ ] Can undock panels to floating windows
- [ ] Reset Layout menu item works
- [ ] Layout persists after restarting application
- [ ] No console errors or warnings
- [ ] Window names match between Begin() and DockBuilderDockWindow()

---

**Quick Command Reference:**
```bash
# Build
cmake --build build/windows-release --config Release --target cyxwiz-engine

# Run
./build/windows-release/bin/Release/cyxwiz-engine.exe

# Clean layout (delete imgui.ini)
rm imgui.ini

# Check window names
grep -r "ImGui::Begin" cyxwiz-engine/src/gui/ | grep -v "Child"
```
