# CyxWiz Plugin Developer Guide

This guide explains how to create plugins for the CyxWiz Engine. Plugins can add custom node types, UI panels, data loaders, training hooks, and analytics computations.

## Quick Start

### 1. Create a plugin directory

```
my_plugin/
  plugin.json
  my_plugin.h
  my_plugin.cpp
```

### 2. Write `plugin.json`

```json
{
  "id": "com.yourname.my-plugin",
  "name": "My Plugin",
  "version": "1.0.0",
  "api_version": "1.0.0",
  "description": "A minimal CyxWiz plugin",
  "author": "Your Name",
  "license": "MIT",
  "capabilities": ["ProvidesTraining"],
  "permissions": ["Training"],
  "platforms": {
    "windows": { "library": "bin/my_plugin.dll" },
    "linux": { "library": "bin/libmy_plugin.so" },
    "macos": { "library": "bin/libmy_plugin.dylib" }
  }
}
```

### 3. Implement `IPlugin`

```cpp
#include <cyxwiz/plugin/plugin_types.h>
#include <cyxwiz/plugin/plugin_context.h>

class MyPlugin : public cyxwiz::plugin::IPlugin {
public:
    bool OnLoad(cyxwiz::plugin::PluginContext& ctx) override {
        ctx.LogInfo("Hello from MyPlugin!");
        return true;
    }
    bool OnInitialize(cyxwiz::plugin::PluginContext& ctx) override { return true; }
    void OnShutdown(cyxwiz::plugin::PluginContext& ctx) override {}
    void OnUnload(cyxwiz::plugin::PluginContext& ctx) override {}

    const cyxwiz::plugin::PluginManifest& GetManifest() const override { return manifest_; }
    cyxwiz::plugin::PluginPermissionFlags GetRequiredPermissions() const override { return 0; }
    cyxwiz::plugin::PluginState GetState() const override { return state_; }

private:
    cyxwiz::plugin::PluginManifest manifest_;
    cyxwiz::plugin::PluginState state_ = cyxwiz::plugin::PluginState::Unloaded;
};

CYXWIZ_PLUGIN_ENTRY(MyPlugin)
```

### 4. Build as a shared library

```cmake
add_library(my_plugin SHARED my_plugin.cpp)
target_link_libraries(my_plugin PRIVATE cyxwiz-plugin-sdk)
```

### 5. Install

Copy the plugin directory to one of:
- `<project>/plugins/` (project-specific)
- `%APPDATA%\cyxwiz\plugins\` (Windows user)
- `~/.cyxwiz/plugins/` (Linux/macOS user)

Open the Plugin Manager panel and click Refresh, or use Install to point to the directory.

---

## Plugin Directory Structure

```
my_plugin/
  plugin.json              # Required: manifest
  bin/
    my_plugin.dll          # Windows binary
    libmy_plugin.so        # Linux binary
    libmy_plugin.dylib     # macOS binary
  mlflow_config.json       # Optional: plugin-specific config
  README.md                # Optional: documentation
```

The engine loads `plugin.json` first, then resolves the platform-specific library path from the `"platforms"` section.

---

## plugin.json Reference

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `id` | string | Yes | Reverse-domain ID (e.g. `com.vendor.my-plugin`) |
| `name` | string | Yes | Display name |
| `version` | string | Yes | Semantic version (`major.minor.patch`) |
| `api_version` | string | Yes | Required engine API version (currently `1.0.0`) |
| `description` | string | No | Short description |
| `author` | string | No | Author name/email |
| `license` | string | No | License identifier (e.g. `MIT`, `Apache-2.0`) |
| `homepage` | string | No | URL |
| `repository` | string | No | Source code URL |
| `capabilities` | string[] | No | What the plugin provides (see below) |
| `permissions` | string[] | No | What the plugin needs (see below) |
| `dependencies` | object[] | No | Other plugins required |
| `platforms` | object | Yes | Platform-specific library paths |
| `signature` | string | No | Ed25519 hex-encoded signature of the DLL |

### Capabilities

| Value | Description |
|-------|-------------|
| `ProvidesNodes` | Custom node types for the visual editor |
| `ProvidesPanels` | Custom ImGui UI panels |
| `ProvidesData` | Custom dataset loaders |
| `ProvidesTraining` | Training lifecycle hooks |
| `ProvidesAnalytics` | Data quality/profiling tools |
| `RequiresPython` | Needs Python interpreter |
| `RequiresGPU` | Needs GPU compute |
| `RequiresNetwork` | Needs network access |

### Permissions

| Permission | Dangerous? | Description |
|-----------|------------|-------------|
| `FileSystem` | Yes | Read/write files |
| `Network` | Yes | Network access |
| `SystemCommands` | Yes | Execute system commands |
| `Python` | Yes | Execute Python code |
| `GPU` | No | Use GPU compute |
| `DataRegistry` | No | Access dataset registry |
| `Training` | No | Access training system |
| `UIModify` | No | Add panels/nodes to the UI |

**Dangerous permissions** require explicit user approval via a dialog the first time the plugin loads. Safe permissions are auto-granted.

---

## Plugin Lifecycle

```
                LoadPlugin()
Unloaded ─────────────────────> Loaded
                                  │
                        OnLoad() + OnInitialize()
                                  │
                                  v
                              Initialized ──────> Active
                                  │
                          OnShutdown()
                                  │
                                  v
                               Loaded
                                  │
                           OnUnload()
                                  │
                                  v
                              Unloaded
```

**Method call order:** `OnLoad` -> `OnInitialize` -> (running) -> `OnShutdown` -> `OnUnload`

- **OnLoad**: Called after DLL is loaded. Read config, validate prerequisites. Return `false` to fail.
- **OnInitialize**: Called after permissions are approved. Register providers here. Return `false` to fail.
- **OnShutdown**: Called when plugin is disabled or engine is shutting down. Cleanup resources.
- **OnUnload**: Called just before DLL is unloaded. Final cleanup.

If the user hasn't approved dangerous permissions yet, `OnInitialize` is deferred until approval.

---

## Interfaces

### IPlugin (required)

Every plugin must implement this base interface. See Quick Start above.

### ITrainingHook

Hook into the training lifecycle to log metrics, implement early stopping, etc.

```cpp
class ITrainingHook {
public:
    virtual void OnTrainingStart(TrainingContext& ctx) {}
    virtual void OnTrainingEnd(TrainingContext& ctx) {}
    virtual void OnEpochStart(TrainingContext& ctx) {}
    virtual void OnEpochEnd(TrainingContext& ctx) {}
    virtual void OnBatchStart(TrainingContext& ctx) {}
    virtual void OnBatchEnd(TrainingContext& ctx) {}
    virtual bool ShouldStopEarly(const TrainingContext& ctx) { return false; }
};
```

`TrainingContext` provides: `current_epoch`, `total_epochs`, `current_batch`, `total_batches`, `train_loss`, `train_accuracy`, `val_loss`, `val_accuracy`, `learning_rate`, `custom_metrics`.

**Register in OnInitialize:**
```cpp
ctx.RegisterTrainingHook(this);  // Requires Training permission
```

See `plugins/examples/mlflow_logger/` for a complete example.

### INodeProvider

Add custom node types to the visual graph editor.

```cpp
class INodeProvider {
public:
    virtual std::vector<PluginNodeTypeInfo> GetNodeTypes() = 0;
    virtual std::string GenerateCode(
        const std::string& node_type_name,
        const std::map<std::string, std::string>& parameters,
        const std::string& framework) = 0;
};
```

Each `PluginNodeTypeInfo` defines:
- `type_name` / `display_name` / `category` / `description`
- `color` (ABGR uint32)
- `pins` (vector of `{name, type, is_input}`)
- `default_parameters` (map of string key/value)

`GenerateCode` is called when exporting the graph. The `framework` parameter is `"pytorch"`, `"tensorflow"`, or `"keras"`.

**Register in OnInitialize:**
```cpp
ctx.RegisterNodeProvider(this);  // Requires UIModify permission
```

See `plugins/examples/image_nodes/` for a complete example.

### IPanelProvider

Add custom ImGui panels to the engine.

```cpp
class IPanelProvider {
public:
    virtual std::vector<PluginPanelInfo> GetPanels() = 0;
    virtual void RenderPanel(const std::string& panel_id, bool* visible) = 0;
};
```

`RenderPanel` is called every frame while the panel is visible. Use `ImGui::Begin(title, visible)` / `ImGui::End()`. Set `*visible = false` to close.

**Register in OnInitialize:**
```cpp
ctx.RegisterPanelProvider(this);  // Requires UIModify permission
```

### IDataProvider

Add custom dataset loaders (e.g. Parquet, Arrow, custom binary formats).

```cpp
class IDataProvider {
public:
    virtual std::vector<PluginDataLoaderInfo> GetLoaders() = 0;
    virtual bool CanLoad(const std::string& file_path) = 0;
    virtual std::shared_ptr<PluginDataset> LoadDataset(
        const std::string& file_path,
        const std::string& dataset_name) = 0;
};
```

**Register in OnInitialize:**
```cpp
ctx.RegisterDataProvider(this);  // Requires DataRegistry permission
```

### IAnalyticsProvider

Add custom data quality/profiling computations.

```cpp
class IAnalyticsProvider {
public:
    virtual std::vector<PluginAnalyticsInfo> GetAnalytics() = 0;
    virtual AnalyticsResult Compute(
        const std::string& analytics_id,
        std::shared_ptr<PluginDataset> dataset) = 0;
};
```

`AnalyticsResult` can return scalar values, vectors with labels, or text.

**Register in OnInitialize:**
```cpp
ctx.RegisterAnalyticsProvider(this);  // Requires DataRegistry permission
```

---

## PluginContext API

The `PluginContext` is the sole interaction surface between plugins and the engine. It is passed to all lifecycle methods.

| Method | Permission Required | Description |
|--------|-------------------|-------------|
| `LogInfo(msg)` | None | Log info message |
| `LogWarn(msg)` | None | Log warning |
| `LogError(msg)` | None | Log error |
| `LogDebug(msg)` | None | Log debug message |
| `GetPluginId()` | None | Get this plugin's ID |
| `RegisterNodeProvider(ptr)` | UIModify | Register custom nodes |
| `RegisterPanelProvider(ptr)` | UIModify | Register custom panels |
| `RegisterDataProvider(ptr)` | DataRegistry | Register data loaders |
| `RegisterTrainingHook(ptr)` | Training | Register training hooks |
| `RegisterAnalyticsProvider(ptr)` | DataRegistry | Register analytics |

All registration methods return `false` if the required permission is not granted.

---

## Permissions & Security

### Permission Approval Flow

1. Plugin declares permissions in `plugin.json`
2. On first load, engine checks if dangerous permissions are undecided
3. If undecided, a permission dialog appears asking the user to approve/deny each
4. Decisions are persisted to `plugin_permissions.json` (keyed by `plugin_id:version`)
5. On subsequent loads, stored decisions are used without prompting

### Ed25519 Signature Verification

Plugins can be signed for integrity verification:

1. Generate a keypair (external tool)
2. Sign the DLL file with Ed25519
3. Add hex-encoded signature to `plugin.json` `"signature"` field
4. Engine verifies signature on load; invalid signatures block loading

Unsigned plugins load with a warning. This is optional but recommended for distribution.

### Crash Isolation

Plugin lifecycle methods (`OnLoad`, `OnInitialize`, etc.) are wrapped in crash isolation:
- **Windows**: SEH (`__try`/`__except`) catches access violations
- **Unix**: `sigsetjmp`/`siglongjmp` catches SIGSEGV/SIGFPE

If a plugin crashes, it is marked as `Failed` with an error message rather than crashing the engine.

---

## Building Plugins

### CMake Template

```cmake
cmake_minimum_required(VERSION 3.20)
project(my_plugin LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)

# Find CyxWiz plugin SDK headers
# Adjust path to your CyxWiz Engine source
set(CYXWIZ_ENGINE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../cyxwiz-engine")

add_library(my_plugin SHARED
    my_plugin.cpp
)

target_include_directories(my_plugin PRIVATE
    ${CYXWIZ_ENGINE_DIR}/src
)

# Link against ImGui if providing panels
# target_link_libraries(my_plugin PRIVATE imgui::imgui)

# Output to bin/ subdirectory
set_target_properties(my_plugin PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/bin"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/bin"
)
```

### Cross-Platform Notes

- Use `CYXWIZ_PLUGIN_EXPORT` macro (auto-detects `__declspec(dllexport)` vs `__attribute__((visibility))`)
- Use `CYXWIZ_PLUGIN_ENTRY(ClassName)` for the factory functions
- Avoid static initializers that depend on engine state
- All strings are `std::string` (UTF-8)
- File paths use `std::filesystem::path`

---

## Troubleshooting

### "Failed to load plugin"
- Check that the DLL exists at the path specified in `plugin.json` `"platforms"` section
- On Windows: ensure all DLL dependencies are available (use `dumpbin /dependents` to check)
- On Linux: check with `ldd`; ensure `LD_LIBRARY_PATH` includes dependency locations

### "Permission denied" on registration
- The required permission is not declared in `plugin.json` `"permissions"` array
- Or the user denied the permission in the approval dialog
- Check the console for `PluginContext: Permission denied` messages

### "API version incompatible"
- Your plugin's `api_version` major must match the engine's (currently `1`)
- Your plugin's minor must be `<=` the engine's minor version

### Plugin loads but nodes/panels don't appear
- Ensure capabilities are declared: `"capabilities": ["ProvidesNodes"]`
- Ensure `RegisterNodeProvider()`/`RegisterPanelProvider()` is called in `OnInitialize` and returns `true`
- Check console output for registration errors

### Plugin crashes the engine
- Plugin lifecycle methods are crash-isolated, but `RenderPanel()` and `GenerateCode()` are not
- Avoid null pointer dereferences and out-of-bounds access in render code
- Test thoroughly before distribution

---

## Example Plugins

Two complete example plugins are provided in `plugins/examples/`:

| Plugin | Interfaces | Demonstrates |
|--------|-----------|--------------|
| `mlflow_logger/` | IPlugin, ITrainingHook | Training metric logging, early stopping, config files |
| `image_nodes/` | IPlugin, INodeProvider, IPanelProvider | Custom nodes, code generation, settings panel |

These serve as templates for building your own plugins.
