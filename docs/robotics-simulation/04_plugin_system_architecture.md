# CyxWiz Plugin System Architecture

> Status: **DESIGN COMPLETE** - Ready for implementation when prioritized
> Version: 1.0.0
> Last Updated: 2026-01-22

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Architecture Overview](#2-architecture-overview)
3. [Core Interfaces](#3-core-interfaces)
4. [PluginContext API](#4-plugincontext-api)
5. [Plugin Manifest Schema](#5-plugin-manifest-schema)
6. [Plugin Discovery & Loading](#6-plugin-discovery--loading)
7. [Dependency Resolution](#7-dependency-resolution)
8. [Registration Systems](#8-registration-systems)
9. [Versioning & Compatibility](#9-versioning--compatibility)
10. [Security Model](#10-security-model)
11. [Error Handling & Crash Isolation](#11-error-handling--crash-isolation)
12. [Example Plugins](#12-example-plugins)
13. [Implementation Roadmap](#13-implementation-roadmap)

---

## 1. Executive Summary

### Purpose

The CyxWiz Plugin System enables third-party developers to extend CyxWiz Engine with:
- **Custom Node Types** - New layers, operations, and algorithms for the visual graph editor
- **UI Panels** - Custom interfaces for specialized workflows
- **Data Loaders** - Support for additional data formats and sources
- **Training Hooks** - Integration with experiment tracking (MLflow, W&B, etc.)
- **Analytics Tools** - Data quality validation (Great Expectations, etc.)

### Design Goals

| Goal | Description |
|------|-------------|
| **Minimal Intrusion** | Existing code requires minimal changes |
| **Safety First** | Crash isolation, permission system, resource limits |
| **Cross-Platform** | Works on Windows, macOS, Linux |
| **Hot Reload** | Developers can reload plugins without restarting |
| **Versioned API** | Clear compatibility rules and deprecation policy |
| **Familiar Patterns** | Builds on existing CyxWiz patterns (singletons, callbacks, factories) |

### Key Principles

1. **Plugins are DLLs/SOs** - Native code for performance, loaded dynamically
2. **Declarative Manifests** - `plugin.json` describes capabilities and requirements
3. **Capability-Based Security** - Plugins declare what they need, users approve
4. **Registry Pattern** - Central registries for nodes, panels, hooks
5. **Loose Coupling** - Plugins communicate via PluginContext, not direct access

---

## 2. Architecture Overview

### 2.1 High-Level Component Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            CyxWiz Engine                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────┐                                                        │
│  │   Application   │                                                        │
│  │   (main.cpp)    │                                                        │
│  └────────┬────────┘                                                        │
│           │                                                                  │
│           ▼                                                                  │
│  ┌─────────────────┐     ┌─────────────────────────────────────────────┐   │
│  │  PluginManager  │────▶│              Plugin Registries               │   │
│  │   (Singleton)   │     ├─────────────────────────────────────────────┤   │
│  └────────┬────────┘     │  NodeRegistry      │ Custom node types      │   │
│           │              │  PanelRegistry     │ Custom UI panels       │   │
│           │              │  DataRegistry*     │ Custom data loaders    │   │
│           │              │  TrainingHooks     │ Training callbacks     │   │
│           │              │  AnalyticsRegistry │ Data quality tools     │   │
│           │              └─────────────────────────────────────────────┘   │
│           │                                                                  │
│           ▼                                                                  │
│  ┌─────────────────┐     ┌─────────────────┐                               │
│  │  PluginLoader   │     │  PluginSandbox  │                               │
│  │  (DLL loading)  │     │ (crash isolation)│                               │
│  └────────┬────────┘     └─────────────────┘                               │
│           │                                                                  │
└───────────┼──────────────────────────────────────────────────────────────────┘
            │
            ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          Plugin Directory Structure                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  plugins/                                                                    │
│  ├── mlflow-integration/                                                     │
│  │   ├── plugin.json              ← Manifest                                │
│  │   ├── mlflow_plugin.dll        ← Windows binary                          │
│  │   ├── libmlflow_plugin.so      ← Linux binary                            │
│  │   └── resources/               ← Icons, configs                          │
│  │                                                                           │
│  ├── great-expectations/                                                     │
│  │   ├── plugin.json                                                         │
│  │   ├── ge_plugin.dll                                                       │
│  │   └── python/                  ← Python scripts                          │
│  │                                                                           │
│  └── custom-nodes/                                                           │
│      ├── plugin.json                                                         │
│      └── custom_nodes.dll                                                    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘

* DataRegistry already exists - extended to support plugin data loaders
```

### 2.2 Plugin Lifecycle

```
┌──────────────────────────────────────────────────────────────────────────┐
│                         Plugin Lifecycle States                           │
└──────────────────────────────────────────────────────────────────────────┘

    ┌─────────┐      LoadLibrary()     ┌─────────┐
    │UNLOADED │ ────────────────────▶  │ LOADED  │
    └─────────┘                        └────┬────┘
         ▲                                  │
         │                                  │ Initialize(context)
         │ UnloadLibrary()                  ▼
         │                             ┌─────────────┐
         │                             │ INITIALIZED │
         │                             └──────┬──────┘
         │                                    │
         │                                    │ OnActivate()
         │                                    ▼
         │     OnDeactivate()          ┌─────────┐
         │◀─────────────────────────── │ ACTIVE  │ ◀──┐
         │                             └────┬────┘    │
         │                                  │         │ Update() each frame
    ┌─────────┐                             │         │
    │ FAILED  │ ◀── Error at any stage     └─────────┘
    └─────────┘

    ┌──────────┐
    │ DISABLED │ ◀── User disabled in settings
    └──────────┘
```

### 2.3 Data Flow

```
┌──────────────────────────────────────────────────────────────────────────┐
│                    Plugin Registration Flow                               │
└──────────────────────────────────────────────────────────────────────────┘

1. Application Startup
   │
   ▼
2. PluginManager::Initialize()
   │
   ├──▶ Scan plugin directories
   │    └──▶ Parse plugin.json manifests
   │
   ├──▶ Resolve dependencies (topological sort)
   │
   ├──▶ Load plugins in dependency order
   │    │
   │    └──▶ For each plugin:
   │         ├── LoadLibrary(plugin.dll)
   │         ├── GetSymbol("CreatePlugin")
   │         ├── plugin = CreatePlugin()
   │         ├── Verify API version
   │         └── plugin->Initialize(context)
   │
   └──▶ Activate all plugins
        └──▶ plugin->OnActivate()

3. Runtime
   │
   ├──▶ Each frame: plugin->Update(delta_time)
   │
   ├──▶ Training hooks called at appropriate times
   │
   └──▶ Panels rendered when visible

4. Shutdown
   │
   ├──▶ plugin->OnDeactivate()
   ├──▶ plugin->Shutdown()
   ├──▶ DestroyPlugin(plugin)
   └──▶ UnloadLibrary()
```

---

## 3. Core Interfaces

### 3.1 IPlugin - Base Plugin Interface

```cpp
// ═══════════════════════════════════════════════════════════════════════════
// File: cyxwiz-engine/include/plugin/iplugin.h
// Purpose: Base interface that all plugins must implement
// ═══════════════════════════════════════════════════════════════════════════

#pragma once

#include <string>
#include <vector>
#include <memory>
#include <cstdint>

namespace cyxwiz::plugin {

// ───────────────────────────────────────────────────────────────────────────
// Forward Declarations
// ───────────────────────────────────────────────────────────────────────────

class PluginContext;
struct PluginVersion;

// ───────────────────────────────────────────────────────────────────────────
// Plugin API Version
// Increment MAJOR for breaking changes, MINOR for new features, PATCH for fixes
// ───────────────────────────────────────────────────────────────────────────

constexpr int CYXWIZ_PLUGIN_API_VERSION_MAJOR = 1;
constexpr int CYXWIZ_PLUGIN_API_VERSION_MINOR = 0;
constexpr int CYXWIZ_PLUGIN_API_VERSION_PATCH = 0;

constexpr uint32_t CYXWIZ_PLUGIN_API_VERSION =
    (CYXWIZ_PLUGIN_API_VERSION_MAJOR << 16) |
    (CYXWIZ_PLUGIN_API_VERSION_MINOR << 8) |
    CYXWIZ_PLUGIN_API_VERSION_PATCH;

// ───────────────────────────────────────────────────────────────────────────
// Plugin Lifecycle States
// ───────────────────────────────────────────────────────────────────────────

enum class PluginState {
    Unloaded,       // DLL not loaded into memory
    Loaded,         // DLL loaded, CreatePlugin() not called yet
    Initialized,    // Initialize() called successfully
    Active,         // OnActivate() called, plugin is running
    Failed,         // Error occurred during lifecycle
    Disabled        // User has disabled this plugin
};

// ───────────────────────────────────────────────────────────────────────────
// Plugin Capabilities (Bitmask)
// Declares what features this plugin provides
// ───────────────────────────────────────────────────────────────────────────

enum class PluginCapability : uint32_t {
    None              = 0,
    ProvidesNodes     = 1 << 0,   // Adds custom node types to graph editor
    ProvidesPanels    = 1 << 1,   // Adds UI panels
    ProvidesData      = 1 << 2,   // Adds data loaders/transformers
    ProvidesTraining  = 1 << 3,   // Hooks into training lifecycle
    ProvidesAnalytics = 1 << 4,   // Adds data quality/analytics tools
    RequiresPython    = 1 << 5,   // Needs Python interpreter access
    RequiresGPU       = 1 << 6,   // Needs GPU compute access
    RequiresNetwork   = 1 << 7,   // Needs network access
    SupportsHotReload = 1 << 8    // Can be reloaded without restart
};

// Bitwise operators for capability flags
inline PluginCapability operator|(PluginCapability a, PluginCapability b) {
    return static_cast<PluginCapability>(
        static_cast<uint32_t>(a) | static_cast<uint32_t>(b)
    );
}

inline bool HasCapability(PluginCapability caps, PluginCapability check) {
    return (static_cast<uint32_t>(caps) & static_cast<uint32_t>(check)) != 0;
}

// ───────────────────────────────────────────────────────────────────────────
// Version Structure
// ───────────────────────────────────────────────────────────────────────────

struct PluginVersion {
    int major = 0;
    int minor = 0;
    int patch = 0;

    // Parse from string "1.2.3"
    static PluginVersion Parse(const std::string& str) {
        PluginVersion v;
        sscanf(str.c_str(), "%d.%d.%d", &v.major, &v.minor, &v.patch);
        return v;
    }

    // Convert to string
    std::string ToString() const {
        return std::to_string(major) + "." +
               std::to_string(minor) + "." +
               std::to_string(patch);
    }

    // Comparison operators
    bool operator<(const PluginVersion& o) const {
        if (major != o.major) return major < o.major;
        if (minor != o.minor) return minor < o.minor;
        return patch < o.patch;
    }

    bool operator==(const PluginVersion& o) const {
        return major == o.major && minor == o.minor && patch == o.patch;
    }

    bool operator<=(const PluginVersion& o) const { return *this < o || *this == o; }
    bool operator>=(const PluginVersion& o) const { return !(*this < o); }
    bool operator>(const PluginVersion& o) const { return !(*this <= o); }
    bool operator!=(const PluginVersion& o) const { return !(*this == o); }
};

// ───────────────────────────────────────────────────────────────────────────
// IPlugin Interface
// ───────────────────────────────────────────────────────────────────────────

/**
 * @brief Base interface for all CyxWiz plugins
 *
 * Plugins must implement this interface and export factory functions.
 * The plugin system calls lifecycle methods in this order:
 *
 *   1. CreatePlugin()    - Factory creates plugin instance
 *   2. GetInfo methods   - Query metadata before initialization
 *   3. Initialize()      - One-time setup with context
 *   4. OnActivate()      - Called when plugin becomes active
 *   5. Update()          - Called each frame (optional)
 *   6. OnDeactivate()    - Called before shutdown
 *   7. Shutdown()        - Release all resources
 *   8. DestroyPlugin()   - Factory destroys plugin instance
 *
 * Thread Safety:
 *   - Initialize(), Shutdown() called from main thread
 *   - Update() called from main thread (render loop)
 *   - Other methods may be called from any thread
 */
class IPlugin {
public:
    virtual ~IPlugin() = default;

    // ═══════════════════════════════════════════════════════════════════════
    // METADATA - Query plugin information
    // ═══════════════════════════════════════════════════════════════════════

    /**
     * @brief Unique plugin identifier
     * @return Reverse-domain ID (e.g., "com.cyxwiz.mlflow-integration")
     *
     * Must be globally unique. Use reverse-domain notation.
     * This ID is used for dependency references and settings storage.
     */
    virtual const char* GetId() const = 0;

    /**
     * @brief Human-readable plugin name
     * @return Display name (e.g., "MLflow Integration")
     */
    virtual const char* GetName() const = 0;

    /**
     * @brief Plugin version
     * @return Semantic version (MAJOR.MINOR.PATCH)
     */
    virtual PluginVersion GetVersion() const = 0;

    /**
     * @brief Plugin description
     * @return Short description for UI display
     */
    virtual const char* GetDescription() const = 0;

    /**
     * @brief Author or organization
     * @return Author name/email/organization
     */
    virtual const char* GetAuthor() const = 0;

    /**
     * @brief Plugin capabilities
     * @return Bitmask of PluginCapability flags
     *
     * Used by the host to determine what interfaces to query
     * and what registries to update.
     */
    virtual PluginCapability GetCapabilities() const = 0;

    /**
     * @brief API version this plugin was built against
     * @return Plugin API version
     *
     * Host will reject plugins with incompatible API versions.
     * Major version must match exactly.
     * Minor version must be <= host version.
     */
    virtual PluginVersion GetApiVersion() const = 0;

    /**
     * @brief Dependencies on other plugins
     * @return List of plugin IDs this plugin requires
     *
     * Dependencies are loaded before this plugin.
     * Empty by default (no dependencies).
     */
    virtual std::vector<std::string> GetDependencies() const { return {}; }

    // ═══════════════════════════════════════════════════════════════════════
    // LIFECYCLE - Plugin state management
    // ═══════════════════════════════════════════════════════════════════════

    /**
     * @brief Initialize the plugin
     * @param context API surface for interacting with host
     * @return true if initialization succeeded, false to abort loading
     *
     * Called once after the plugin DLL is loaded.
     * Register node types, panels, hooks here.
     *
     * Thread: Main thread
     */
    virtual bool Initialize(PluginContext* context) = 0;

    /**
     * @brief Activate the plugin
     *
     * Called after all dependencies are initialized.
     * Plugin can start background tasks, connect to services, etc.
     *
     * Thread: Main thread
     */
    virtual void OnActivate() {}

    /**
     * @brief Per-frame update
     * @param delta_time Seconds since last frame
     *
     * Called every frame for plugins that need continuous updates.
     * Keep this fast - runs on render thread.
     *
     * Thread: Main thread (render loop)
     */
    virtual void Update(float delta_time) { (void)delta_time; }

    /**
     * @brief Deactivate the plugin
     *
     * Called before shutdown or when user disables plugin.
     * Stop background tasks, disconnect from services.
     *
     * Thread: Main thread
     */
    virtual void OnDeactivate() {}

    /**
     * @brief Shutdown and release resources
     *
     * Called during application shutdown or plugin unload.
     * Release all allocated resources.
     * After this, DestroyPlugin() will be called.
     *
     * Thread: Main thread
     */
    virtual void Shutdown() = 0;

    // ═══════════════════════════════════════════════════════════════════════
    // HOT RELOAD - Development support (optional)
    // ═══════════════════════════════════════════════════════════════════════

    /**
     * @brief Save plugin state for hot reload
     * @return JSON string with state to preserve
     *
     * Called before hot reload. Return any state that should
     * survive the reload (e.g., connection URLs, user settings).
     */
    virtual std::string SaveState() { return "{}"; }

    /**
     * @brief Restore plugin state after hot reload
     * @param state JSON string from SaveState()
     *
     * Called after hot reload completes. Restore state from
     * the JSON string saved before reload.
     */
    virtual void RestoreState(const std::string& state) { (void)state; }
};

// ───────────────────────────────────────────────────────────────────────────
// Factory Function Types
// These functions must be exported by plugin DLLs
// ───────────────────────────────────────────────────────────────────────────

/**
 * @brief Creates a new plugin instance
 * @return Pointer to IPlugin implementation
 *
 * The host owns this pointer and will call DestroyPlugin() to release it.
 */
using CreatePluginFunc = IPlugin* (*)();

/**
 * @brief Destroys a plugin instance
 * @param plugin Pointer returned by CreatePlugin()
 *
 * Must properly clean up the plugin instance.
 */
using DestroyPluginFunc = void (*)(IPlugin*);

/**
 * @brief Returns the plugin API version
 * @return API version as uint32_t
 *
 * Called before CreatePlugin() to verify compatibility.
 */
using GetApiVersionFunc = uint32_t (*)();

// ───────────────────────────────────────────────────────────────────────────
// Export Macros
// Use these in your plugin implementation
// ───────────────────────────────────────────────────────────────────────────

#ifdef _WIN32
    #define CYXWIZ_PLUGIN_EXPORT __declspec(dllexport)
#else
    #define CYXWIZ_PLUGIN_EXPORT __attribute__((visibility("default")))
#endif

/**
 * @brief Macro to define plugin entry points
 * @param PluginClass Your IPlugin implementation class
 *
 * Usage:
 *   class MyPlugin : public IPlugin { ... };
 *   CYXWIZ_PLUGIN(MyPlugin)
 */
#define CYXWIZ_PLUGIN(PluginClass) \
    extern "C" CYXWIZ_PLUGIN_EXPORT ::cyxwiz::plugin::IPlugin* CreatePlugin() { \
        return new PluginClass(); \
    } \
    extern "C" CYXWIZ_PLUGIN_EXPORT void DestroyPlugin(::cyxwiz::plugin::IPlugin* p) { \
        delete p; \
    } \
    extern "C" CYXWIZ_PLUGIN_EXPORT uint32_t GetPluginApiVersion() { \
        return CYXWIZ_PLUGIN_API_VERSION; \
    }

} // namespace cyxwiz::plugin
```

### 3.2 INodeProvider - Custom Node Types

```cpp
// ═══════════════════════════════════════════════════════════════════════════
// File: cyxwiz-engine/include/plugin/inode_provider.h
// Purpose: Interface for plugins that provide custom node types
// ═══════════════════════════════════════════════════════════════════════════

#pragma once

#include "iplugin.h"
#include <vector>
#include <map>
#include <functional>
#include <cstdint>

namespace cyxwiz::plugin {

// ───────────────────────────────────────────────────────────────────────────
// Pin Types (must match gui/node_editor.h PinType enum)
// ───────────────────────────────────────────────────────────────────────────

enum class PinType {
    Tensor,         // Multi-dimensional array data
    Labels,         // Classification labels
    Parameters,     // Layer parameters (weights, biases)
    Loss,           // Loss value
    Optimizer,      // Optimizer state
    Dataset,        // Dataset reference
    Scalar,         // Single numeric value
    String,         // Text data
    Any             // Accepts any type (for generic nodes)
};

// ───────────────────────────────────────────────────────────────────────────
// Pin Definition
// ───────────────────────────────────────────────────────────────────────────

/**
 * @brief Defines an input or output pin for a custom node
 */
struct PinDefinition {
    std::string name;           // Display name (e.g., "Input", "Output")
    PinType type;               // Data type
    bool is_input;              // true = input pin, false = output pin
    bool is_required = true;    // Must be connected for node to execute

    // Variadic pin support (for nodes like Concatenate that take N inputs)
    bool is_variadic = false;   // Can accept multiple connections
    int min_connections = 0;    // Minimum connections (if variadic)
    int max_connections = 1;    // Maximum connections (-1 = unlimited)

    // Documentation
    std::string tooltip;        // Hover tooltip text
    std::string default_value;  // Default value if not connected
};

// ───────────────────────────────────────────────────────────────────────────
// Parameter Definition
// ───────────────────────────────────────────────────────────────────────────

/**
 * @brief Defines a configurable parameter for a custom node
 *
 * Parameters appear in the Properties panel when a node is selected.
 */
struct ParameterDefinition {
    std::string key;            // Internal key (e.g., "units", "kernel_size")
    std::string display_name;   // UI label (e.g., "Units", "Kernel Size")
    std::string default_value;  // Default value as string

    enum class Type {
        Int,            // Integer input
        Float,          // Float input
        String,         // Text input
        Bool,           // Checkbox
        Enum,           // Dropdown selection
        IntArray,       // Array of ints (e.g., "[3, 3]")
        FloatArray,     // Array of floats
        FilePath,       // File picker
        Color           // Color picker (RGBA)
    };
    Type type;

    // Validation
    std::string min_value;      // Minimum (for numeric types)
    std::string max_value;      // Maximum (for numeric types)
    std::string regex_pattern;  // Validation regex (for strings)

    // For Enum type
    std::vector<std::string> enum_options;

    // Documentation
    std::string tooltip;
    std::string help_url;       // Link to documentation
};

// ───────────────────────────────────────────────────────────────────────────
// Code Generation Template
// ───────────────────────────────────────────────────────────────────────────

/**
 * @brief Template for generating code in different frameworks
 *
 * Use $param placeholders that get substituted with actual values.
 * Example: "nn.Linear($in_features, $out_features)"
 */
struct CodeTemplate {
    std::string framework;      // "PyTorch", "TensorFlow", "Keras", "PyCyxWiz"
    std::string code_template;  // Code with $param placeholders
    std::string import_line;    // Required import (e.g., "import torch.nn as nn")
};

// ───────────────────────────────────────────────────────────────────────────
// Shape Inference Function Type
// ───────────────────────────────────────────────────────────────────────────

/**
 * @brief Function to infer output shape from input shapes and parameters
 *
 * @param input_shapes Vector of input shapes (one per input pin)
 * @param params Map of parameter key -> value
 * @return Vector of output shapes (one per output pin)
 *
 * Return empty vector if shape cannot be determined.
 */
using ShapeInferenceFunc = std::function<
    std::vector<std::vector<size_t>>(
        const std::vector<std::vector<size_t>>& input_shapes,
        const std::map<std::string, std::string>& params
    )
>;

// ───────────────────────────────────────────────────────────────────────────
// Node Type Info
// ───────────────────────────────────────────────────────────────────────────

/**
 * @brief Complete definition of a custom node type
 */
struct NodeTypeInfo {
    // ─── Identity ───
    std::string type_id;        // Unique ID (e.g., "plugin.attention.sparse")
    std::string display_name;   // UI name (e.g., "Sparse Attention")
    std::string category;       // Palette category (e.g., "Attention", "Custom")

    // ─── Visual ───
    std::string icon;           // FontAwesome icon code (e.g., "\uf0e7")
    uint32_t header_color;      // Header color (0xRRGGBBAA)
    uint32_t body_color;        // Body color (0xRRGGBBAA)

    // ─── Structure ───
    std::vector<PinDefinition> inputs;
    std::vector<PinDefinition> outputs;
    std::vector<ParameterDefinition> parameters;

    // ─── Code Generation ───
    std::vector<CodeTemplate> code_templates;

    // ─── Shape Inference ───
    ShapeInferenceFunc shape_inference;

    // ─── Documentation ───
    std::string description;    // Tooltip/help text
    std::string author;         // Node author
    std::string help_url;       // Link to documentation
    std::vector<std::string> tags;  // Search tags

    // ─── Defaults ───
    static constexpr uint32_t DEFAULT_HEADER_COLOR = 0xFF5A5A5A;
    static constexpr uint32_t DEFAULT_BODY_COLOR = 0xFF3A3A3A;
};

// ───────────────────────────────────────────────────────────────────────────
// Tensor Data (for node execution)
// ───────────────────────────────────────────────────────────────────────────

/**
 * @brief Simple tensor structure for data passing
 *
 * Note: In production, this would interface with the actual Tensor class
 * from cyxwiz-backend. This is a simplified version for the plugin API.
 */
struct TensorData {
    std::vector<float> data;
    std::vector<size_t> shape;

    size_t NumElements() const {
        size_t n = 1;
        for (size_t s : shape) n *= s;
        return n;
    }
};

// ───────────────────────────────────────────────────────────────────────────
// INodeProvider Interface
// ───────────────────────────────────────────────────────────────────────────

/**
 * @brief Interface for plugins that provide custom node types
 *
 * Plugins implementing this interface can add new node types to the
 * visual graph editor. Nodes can represent layers, operations, or
 * any computation that fits the dataflow paradigm.
 *
 * Implementation requirements:
 *   1. Return PluginCapability::ProvidesNodes from GetCapabilities()
 *   2. Implement GetNodeTypes() to return node definitions
 *   3. Implement node instance lifecycle (Create/Execute/Destroy)
 *
 * Thread Safety:
 *   - GetNodeTypes() may be called from any thread
 *   - Create/Execute/Destroy called from training thread
 */
class INodeProvider : public virtual IPlugin {
public:
    virtual ~INodeProvider() = default;

    // ═══════════════════════════════════════════════════════════════════════
    // NODE TYPE REGISTRATION
    // ═══════════════════════════════════════════════════════════════════════

    /**
     * @brief Get all node types provided by this plugin
     * @return Vector of NodeTypeInfo definitions
     *
     * Called during plugin initialization to register nodes
     * with the NodeRegistry. Return all node types this plugin provides.
     */
    virtual std::vector<NodeTypeInfo> GetNodeTypes() const = 0;

    // ═══════════════════════════════════════════════════════════════════════
    // NODE INSTANCE LIFECYCLE
    // These are called during local training execution
    // ═══════════════════════════════════════════════════════════════════════

    /**
     * @brief Create a node instance for execution
     * @param type_id Node type ID (from NodeTypeInfo)
     * @param params Parameter values (key -> value)
     * @return Opaque handle to node instance
     *
     * The plugin owns this handle and must properly clean up
     * in DestroyNodeInstance().
     *
     * Thread: Training thread
     */
    virtual void* CreateNodeInstance(
        const std::string& type_id,
        const std::map<std::string, std::string>& params) = 0;

    /**
     * @brief Execute the node's computation
     * @param instance Handle from CreateNodeInstance()
     * @param inputs Map of input pin name -> tensor data
     * @param outputs Map of output pin name -> tensor data (to be filled)
     * @return true if execution succeeded
     *
     * Perform the node's computation. Read from inputs, write to outputs.
     *
     * Thread: Training thread
     */
    virtual bool ExecuteNode(
        void* instance,
        const std::map<std::string, TensorData>& inputs,
        std::map<std::string, TensorData>& outputs) = 0;

    /**
     * @brief Destroy a node instance
     * @param instance Handle from CreateNodeInstance()
     *
     * Release all resources associated with this instance.
     *
     * Thread: Training thread
     */
    virtual void DestroyNodeInstance(void* instance) = 0;

    // ═══════════════════════════════════════════════════════════════════════
    // OPTIONAL METHODS
    // ═══════════════════════════════════════════════════════════════════════

    /**
     * @brief Validate node configuration before execution
     * @param type_id Node type ID
     * @param params Parameter values
     * @param error_message Output: error message if invalid
     * @return true if configuration is valid
     *
     * Called before creating a node instance. Use to validate
     * parameter values and provide helpful error messages.
     */
    virtual bool ValidateNode(
        const std::string& type_id,
        const std::map<std::string, std::string>& params,
        std::string& error_message) {
        (void)type_id; (void)params; (void)error_message;
        return true;  // Default: all configurations valid
    }

    /**
     * @brief Get parameter suggestions based on current values
     * @param type_id Node type ID
     * @param param_key Parameter to get suggestions for
     * @param current_params Current parameter values
     * @return List of suggested values
     *
     * Used for autocomplete in the Properties panel.
     */
    virtual std::vector<std::string> GetParameterSuggestions(
        const std::string& type_id,
        const std::string& param_key,
        const std::map<std::string, std::string>& current_params) {
        (void)type_id; (void)param_key; (void)current_params;
        return {};  // Default: no suggestions
    }
};

} // namespace cyxwiz::plugin
```

### 3.3 IPanelProvider - Custom UI Panels

```cpp
// ═══════════════════════════════════════════════════════════════════════════
// File: cyxwiz-engine/include/plugin/ipanel_provider.h
// Purpose: Interface for plugins that provide custom UI panels
// ═══════════════════════════════════════════════════════════════════════════

#pragma once

#include "iplugin.h"
#include <vector>
#include <string>

// Forward declare ImGui types (plugins include imgui.h themselves)
struct ImVec2;

namespace cyxwiz::plugin {

// ───────────────────────────────────────────────────────────────────────────
// Panel Info
// ───────────────────────────────────────────────────────────────────────────

/**
 * @brief Defines a custom panel provided by a plugin
 */
struct PanelInfo {
    std::string id;             // Unique panel ID (e.g., "mlflow-dashboard")
    std::string display_name;   // UI name (e.g., "MLflow Dashboard")
    std::string icon;           // FontAwesome icon (e.g., "\uf080")
    std::string category;       // Menu category (e.g., "Training", "Data", "Tools")

    // Behavior
    bool visible_by_default = false;    // Show on first launch
    bool allow_multiple = false;        // Can open multiple instances
    bool save_state = true;             // Persist visibility across sessions

    // Keyboard shortcut (optional)
    std::string shortcut;       // e.g., "Ctrl+Shift+M"

    // Size hints
    float min_width = 200.0f;
    float min_height = 150.0f;
    float default_width = 400.0f;
    float default_height = 300.0f;

    // Documentation
    std::string tooltip;
    std::string help_url;
};

// ───────────────────────────────────────────────────────────────────────────
// IPanelProvider Interface
// ───────────────────────────────────────────────────────────────────────────

/**
 * @brief Interface for plugins that provide custom UI panels
 *
 * Plugins implementing this interface can add new panels to the
 * CyxWiz Engine UI. Panels use Dear ImGui for rendering.
 *
 * Implementation requirements:
 *   1. Return PluginCapability::ProvidesPanels from GetCapabilities()
 *   2. Implement GetPanels() to return panel definitions
 *   3. Implement panel instance lifecycle (Create/Render/Destroy)
 *   4. Include <imgui.h> in your implementation
 *
 * Thread Safety:
 *   - All methods called from main thread (render loop)
 */
class IPanelProvider : public virtual IPlugin {
public:
    virtual ~IPanelProvider() = default;

    // ═══════════════════════════════════════════════════════════════════════
    // PANEL REGISTRATION
    // ═══════════════════════════════════════════════════════════════════════

    /**
     * @brief Get all panels provided by this plugin
     * @return Vector of PanelInfo definitions
     *
     * Called during plugin initialization to register panels.
     */
    virtual std::vector<PanelInfo> GetPanels() const = 0;

    // ═══════════════════════════════════════════════════════════════════════
    // PANEL INSTANCE LIFECYCLE
    // ═══════════════════════════════════════════════════════════════════════

    /**
     * @brief Create a panel instance
     * @param panel_id Panel ID from PanelInfo
     * @return Opaque handle to panel instance
     *
     * Called when panel needs to be displayed for the first time,
     * or when allow_multiple is true and user opens another instance.
     *
     * Thread: Main thread
     */
    virtual void* CreatePanelInstance(const std::string& panel_id) = 0;

    /**
     * @brief Render the panel
     * @param instance Handle from CreatePanelInstance()
     *
     * Called every frame when the panel is visible.
     * Use ImGui calls to render panel content.
     *
     * The host wraps your content in ImGui::Begin()/End().
     * You only need to render the interior.
     *
     * Example:
     *   void RenderPanel(void* instance) override {
     *       auto* panel = static_cast<MyPanel*>(instance);
     *       ImGui::Text("Hello from plugin!");
     *       if (ImGui::Button("Click me")) {
     *           panel->OnButtonClick();
     *       }
     *   }
     *
     * Thread: Main thread (render loop)
     */
    virtual void RenderPanel(void* instance) = 0;

    /**
     * @brief Handle panel visibility change
     * @param instance Handle from CreatePanelInstance()
     * @param visible true if panel became visible, false if hidden
     *
     * Called when panel is shown or hidden. Use to start/stop
     * background work or refresh data.
     *
     * Thread: Main thread
     */
    virtual void OnPanelVisibilityChanged(void* instance, bool visible) {
        (void)instance; (void)visible;
    }

    /**
     * @brief Handle panel resize
     * @param instance Handle from CreatePanelInstance()
     * @param width New width
     * @param height New height
     *
     * Called when panel is resized. Use to adjust layout.
     *
     * Thread: Main thread
     */
    virtual void OnPanelResized(void* instance, float width, float height) {
        (void)instance; (void)width; (void)height;
    }

    /**
     * @brief Destroy a panel instance
     * @param instance Handle from CreatePanelInstance()
     *
     * Called when panel is closed or plugin is unloaded.
     * Release all resources associated with this instance.
     *
     * Thread: Main thread
     */
    virtual void DestroyPanelInstance(void* instance) = 0;

    // ═══════════════════════════════════════════════════════════════════════
    // PANEL STATE MANAGEMENT
    // ═══════════════════════════════════════════════════════════════════════

    /**
     * @brief Check if a panel is currently visible
     * @param panel_id Panel ID
     * @return true if panel is visible
     */
    virtual bool IsPanelVisible(const std::string& panel_id) const = 0;

    /**
     * @brief Set panel visibility
     * @param panel_id Panel ID
     * @param visible true to show, false to hide
     */
    virtual void SetPanelVisible(const std::string& panel_id, bool visible) = 0;

    // ═══════════════════════════════════════════════════════════════════════
    // OPTIONAL: PANEL STATE PERSISTENCE
    // ═══════════════════════════════════════════════════════════════════════

    /**
     * @brief Save panel state to JSON
     * @param instance Panel instance
     * @return JSON string with panel state
     *
     * Called when saving project or closing panel.
     * Save any state that should persist (scroll position, selections, etc.)
     */
    virtual std::string SavePanelState(void* instance) {
        (void)instance;
        return "{}";
    }

    /**
     * @brief Restore panel state from JSON
     * @param instance Panel instance
     * @param state JSON string from SavePanelState()
     */
    virtual void RestorePanelState(void* instance, const std::string& state) {
        (void)instance; (void)state;
    }
};

} // namespace cyxwiz::plugin
```

### 3.4 IDataProvider - Custom Data Loaders

```cpp
// ═══════════════════════════════════════════════════════════════════════════
// File: cyxwiz-engine/include/plugin/idata_provider.h
// Purpose: Interface for plugins that provide custom data loaders
// ═══════════════════════════════════════════════════════════════════════════

#pragma once

#include "iplugin.h"
#include <vector>
#include <map>
#include <string>

namespace cyxwiz::plugin {

// ───────────────────────────────────────────────────────────────────────────
// Data Format Info
// ───────────────────────────────────────────────────────────────────────────

/**
 * @brief Describes a data format supported by this plugin
 */
struct DataFormatInfo {
    std::string format_id;      // Unique ID (e.g., "parquet", "tfrecord")
    std::string display_name;   // UI name (e.g., "Apache Parquet")
    std::string description;    // Format description

    // File extensions (e.g., {".parquet", ".pq"})
    std::vector<std::string> file_extensions;

    // MIME types (e.g., {"application/vnd.apache.parquet"})
    std::vector<std::string> mime_types;

    // Capabilities
    bool supports_streaming = false;    // Can load in chunks
    bool supports_preview = true;       // Can show quick preview
    bool supports_write = false;        // Can export to this format
    bool supports_schema = true;        // Has queryable schema
};

// ───────────────────────────────────────────────────────────────────────────
// Dataset Info
// ───────────────────────────────────────────────────────────────────────────

/**
 * @brief Information about a loaded dataset
 */
struct PluginDatasetInfo {
    std::string name;               // Dataset name
    std::string source_path;        // Original file/URL path

    // Dimensions
    size_t num_samples;             // Total number of samples
    size_t num_classes;             // Number of classes (0 if regression)
    std::vector<size_t> sample_shape;   // Shape of each sample

    // Labels
    std::vector<std::string> class_names;   // Class label names
    std::map<int, size_t> class_distribution;   // class_id -> count

    // Memory
    size_t estimated_memory_bytes;  // Estimated memory when fully loaded
    size_t loaded_memory_bytes;     // Currently loaded memory

    // Metadata
    std::map<std::string, std::string> metadata;  // Arbitrary key-value pairs

    // Schema (for tabular data)
    struct ColumnInfo {
        std::string name;
        std::string dtype;  // "float32", "int64", "string", etc.
        bool is_nullable;
    };
    std::vector<ColumnInfo> columns;
};

// ───────────────────────────────────────────────────────────────────────────
// Data Sample
// ───────────────────────────────────────────────────────────────────────────

/**
 * @brief A single data sample
 */
struct DataSample {
    std::vector<float> data;        // Primary data (flattened)
    std::vector<size_t> shape;      // Data shape
    int label = -1;                 // Class label (-1 if no label)

    // Extra fields (for complex datasets)
    std::map<std::string, std::vector<float>> extra_tensors;
    std::map<std::string, std::string> extra_strings;

    // Metadata
    std::string sample_id;          // Unique sample identifier
    std::string source_file;        // Source file (for multi-file datasets)
};

// ───────────────────────────────────────────────────────────────────────────
// Load Options
// ───────────────────────────────────────────────────────────────────────────

/**
 * @brief Options for loading a dataset
 */
struct DataLoadOptions {
    // Subset loading
    size_t max_samples = 0;         // 0 = load all
    size_t skip_samples = 0;        // Skip first N samples

    // Preprocessing
    bool normalize = false;         // Normalize to [0, 1]
    bool shuffle = false;           // Shuffle on load
    int random_seed = 42;           // Seed for shuffling

    // Memory management
    bool lazy_load = true;          // Load samples on demand
    size_t cache_size = 1000;       // Number of samples to cache

    // Format-specific options
    std::map<std::string, std::string> options;
};

// ───────────────────────────────────────────────────────────────────────────
// IDataProvider Interface
// ───────────────────────────────────────────────────────────────────────────

/**
 * @brief Interface for plugins that provide custom data loaders
 *
 * Plugins implementing this interface can add support for new data
 * formats (Parquet, TFRecord, HDF5, etc.) or data sources (S3, databases).
 *
 * Implementation requirements:
 *   1. Return PluginCapability::ProvidesData from GetCapabilities()
 *   2. Implement GetSupportedFormats() to declare formats
 *   3. Implement dataset lifecycle (Open/Get/Close)
 *
 * Thread Safety:
 *   - GetSupportedFormats(), CanLoad() may be called from any thread
 *   - Open/Get/Close called from data loading thread
 */
class IDataProvider : public virtual IPlugin {
public:
    virtual ~IDataProvider() = default;

    // ═══════════════════════════════════════════════════════════════════════
    // FORMAT REGISTRATION
    // ═══════════════════════════════════════════════════════════════════════

    /**
     * @brief Get all data formats supported by this plugin
     * @return Vector of DataFormatInfo definitions
     */
    virtual std::vector<DataFormatInfo> GetSupportedFormats() const = 0;

    /**
     * @brief Check if this plugin can load the given path
     * @param path File path or URL
     * @return true if this plugin can handle this path
     *
     * Called to determine which plugin to use for a given file.
     * Check file extension, magic bytes, or URL scheme.
     */
    virtual bool CanLoad(const std::string& path) const = 0;

    // ═══════════════════════════════════════════════════════════════════════
    // DATASET LIFECYCLE
    // ═══════════════════════════════════════════════════════════════════════

    /**
     * @brief Get dataset info without fully loading
     * @param path File path or URL
     * @param out_info Output: dataset information
     * @return true if info was retrieved successfully
     *
     * Called to show dataset preview before loading.
     * Should be fast - don't read the entire file.
     */
    virtual bool GetDatasetInfo(
        const std::string& path,
        PluginDatasetInfo& out_info) = 0;

    /**
     * @brief Open a dataset
     * @param path File path or URL
     * @param options Load options
     * @return Opaque handle to dataset, nullptr on failure
     *
     * Thread: Data loading thread
     */
    virtual void* OpenDataset(
        const std::string& path,
        const DataLoadOptions& options) = 0;

    /**
     * @brief Get dataset size
     * @param handle Dataset handle from OpenDataset()
     * @return Number of samples in dataset
     */
    virtual size_t GetDatasetSize(void* handle) const = 0;

    /**
     * @brief Get full dataset info
     * @param handle Dataset handle
     * @param out_info Output: dataset information
     * @return true on success
     */
    virtual bool GetLoadedDatasetInfo(
        void* handle,
        PluginDatasetInfo& out_info) const = 0;

    /**
     * @brief Get a single sample
     * @param handle Dataset handle
     * @param index Sample index
     * @param out_sample Output: sample data
     * @return true on success
     *
     * Thread: Data loading thread
     */
    virtual bool GetSample(
        void* handle,
        size_t index,
        DataSample& out_sample) = 0;

    /**
     * @brief Get a batch of samples (more efficient for some formats)
     * @param handle Dataset handle
     * @param indices Sample indices
     * @param out_samples Output: sample data
     * @return true on success
     *
     * Default implementation calls GetSample() for each index.
     * Override for formats that support efficient batch loading.
     */
    virtual bool GetBatch(
        void* handle,
        const std::vector<size_t>& indices,
        std::vector<DataSample>& out_samples) {
        out_samples.resize(indices.size());
        for (size_t i = 0; i < indices.size(); i++) {
            if (!GetSample(handle, indices[i], out_samples[i])) {
                return false;
            }
        }
        return true;
    }

    /**
     * @brief Close a dataset
     * @param handle Dataset handle
     *
     * Release all resources associated with this dataset.
     */
    virtual void CloseDataset(void* handle) = 0;

    // ═══════════════════════════════════════════════════════════════════════
    // OPTIONAL: STREAMING SUPPORT
    // ═══════════════════════════════════════════════════════════════════════

    /**
     * @brief Create a streaming iterator
     * @param handle Dataset handle
     * @param batch_size Samples per batch
     * @param shuffle Shuffle samples
     * @return Iterator handle, nullptr if not supported
     */
    virtual void* CreateIterator(
        void* handle,
        size_t batch_size,
        bool shuffle) {
        (void)handle; (void)batch_size; (void)shuffle;
        return nullptr;  // Not supported by default
    }

    /**
     * @brief Get next batch from iterator
     * @param iterator Iterator handle
     * @param out_samples Output: batch of samples
     * @return true if batch was returned, false if exhausted
     */
    virtual bool GetNextBatch(
        void* iterator,
        std::vector<DataSample>& out_samples) {
        (void)iterator; (void)out_samples;
        return false;
    }

    /**
     * @brief Destroy iterator
     * @param iterator Iterator handle
     */
    virtual void DestroyIterator(void* iterator) {
        (void)iterator;
    }

    // ═══════════════════════════════════════════════════════════════════════
    // OPTIONAL: WRITE SUPPORT
    // ═══════════════════════════════════════════════════════════════════════

    /**
     * @brief Write samples to a file
     * @param path Output path
     * @param format_id Format to write (from GetSupportedFormats())
     * @param samples Samples to write
     * @param info Dataset info (metadata, class names, etc.)
     * @return true on success
     */
    virtual bool WriteDataset(
        const std::string& path,
        const std::string& format_id,
        const std::vector<DataSample>& samples,
        const PluginDatasetInfo& info) {
        (void)path; (void)format_id; (void)samples; (void)info;
        return false;  // Write not supported by default
    }
};

} // namespace cyxwiz::plugin
```

### 3.5 ITrainingHook - Training Lifecycle Callbacks

```cpp
// ═══════════════════════════════════════════════════════════════════════════
// File: cyxwiz-engine/include/plugin/itraining_hook.h
// Purpose: Interface for plugins that hook into the training lifecycle
// ═══════════════════════════════════════════════════════════════════════════

#pragma once

#include "iplugin.h"
#include <map>
#include <string>
#include <vector>

namespace cyxwiz::plugin {

// ───────────────────────────────────────────────────────────────────────────
// Training Configuration Snapshot
// ───────────────────────────────────────────────────────────────────────────

/**
 * @brief Snapshot of training configuration at start
 */
struct TrainingConfigSnapshot {
    // ─── Model ───
    std::string model_name;
    std::string model_type;         // "Sequential", "Functional", etc.
    std::vector<std::string> layer_names;
    size_t total_parameters;
    size_t trainable_parameters;

    // ─── Dataset ───
    std::string dataset_name;
    std::string dataset_path;
    size_t train_samples;
    size_t val_samples;
    size_t num_classes;
    std::vector<size_t> input_shape;

    // ─── Training ───
    int epochs;
    int batch_size;
    std::string optimizer;          // "Adam", "SGD", etc.
    std::string loss_function;      // "CrossEntropy", "MSE", etc.
    float learning_rate;

    // ─── Hyperparameters ───
    std::map<std::string, std::string> hyperparameters;

    // ─── Environment ───
    std::string device;             // "cuda:0", "cpu"
    std::string cyxwiz_version;
    std::string start_time;         // ISO 8601 timestamp
    std::string run_id;             // Unique run identifier
};

// ───────────────────────────────────────────────────────────────────────────
// Training Metrics Snapshot
// ───────────────────────────────────────────────────────────────────────────

/**
 * @brief Snapshot of training metrics at a point in time
 */
struct TrainingMetricsSnapshot {
    // ─── Progress ───
    int epoch;
    int batch;
    int total_epochs;
    int total_batches;
    int samples_processed;

    // ─── Training Metrics ───
    float train_loss;
    float train_accuracy;
    std::map<std::string, float> train_metrics;  // Custom metrics

    // ─── Validation Metrics ───
    float val_loss;
    float val_accuracy;
    std::map<std::string, float> val_metrics;

    // ─── Learning Rate ───
    float learning_rate;

    // ─── Performance ───
    float epoch_time_seconds;
    float batch_time_ms;
    float samples_per_second;
    float gpu_memory_used_mb;
    float gpu_utilization_percent;

    // ─── Best Metrics (so far) ───
    float best_val_loss;
    float best_val_accuracy;
    int best_epoch;
};

// ───────────────────────────────────────────────────────────────────────────
// ITrainingHook Interface
// ───────────────────────────────────────────────────────────────────────────

/**
 * @brief Interface for plugins that hook into the training lifecycle
 *
 * Plugins implementing this interface can:
 *   - Track experiments (MLflow, Weights & Biases, TensorBoard)
 *   - Log metrics and artifacts
 *   - Implement early stopping
 *   - Send notifications
 *
 * Implementation requirements:
 *   1. Return PluginCapability::ProvidesTraining from GetCapabilities()
 *   2. Implement OnTrainingStart() and OnTrainingEnd()
 *   3. Optionally implement other callbacks as needed
 *
 * Thread Safety:
 *   - All callbacks called from training thread
 *   - Must be thread-safe if accessing shared state
 */
class ITrainingHook : public virtual IPlugin {
public:
    virtual ~ITrainingHook() = default;

    // ═══════════════════════════════════════════════════════════════════════
    // TRAINING LIFECYCLE CALLBACKS
    // ═══════════════════════════════════════════════════════════════════════

    /**
     * @brief Called when training starts
     * @param config Training configuration
     *
     * Use to:
     *   - Create experiment run (MLflow, W&B)
     *   - Log hyperparameters
     *   - Initialize tracking state
     *
     * Thread: Training thread
     */
    virtual void OnTrainingStart(const TrainingConfigSnapshot& config) = 0;

    /**
     * @brief Called when training ends
     * @param final_metrics Final training metrics
     * @param success true if training completed, false if stopped/crashed
     * @param error_message Error message if success is false
     *
     * Use to:
     *   - Finalize experiment run
     *   - Log final metrics
     *   - Send completion notification
     *
     * Thread: Training thread
     */
    virtual void OnTrainingEnd(
        const TrainingMetricsSnapshot& final_metrics,
        bool success,
        const std::string& error_message) = 0;

    /**
     * @brief Called at the start of each epoch
     * @param epoch Epoch number (0-indexed)
     */
    virtual void OnEpochStart(int epoch) { (void)epoch; }

    /**
     * @brief Called at the end of each epoch
     * @param epoch Epoch number
     * @param metrics Metrics after this epoch
     *
     * Use to:
     *   - Log epoch metrics
     *   - Check early stopping conditions
     *   - Update learning rate schedules
     */
    virtual void OnEpochEnd(int epoch, const TrainingMetricsSnapshot& metrics) {
        (void)epoch; (void)metrics;
    }

    /**
     * @brief Called at the start of each batch
     * @param epoch Current epoch
     * @param batch Batch number within epoch
     */
    virtual void OnBatchStart(int epoch, int batch) {
        (void)epoch; (void)batch;
    }

    /**
     * @brief Called at the end of each batch
     * @param epoch Current epoch
     * @param batch Batch number
     * @param batch_loss Loss for this batch
     */
    virtual void OnBatchEnd(int epoch, int batch, float batch_loss) {
        (void)epoch; (void)batch; (void)batch_loss;
    }

    /**
     * @brief Called after validation run
     * @param epoch Current epoch
     * @param metrics Validation metrics
     */
    virtual void OnValidationEnd(int epoch, const TrainingMetricsSnapshot& metrics) {
        (void)epoch; (void)metrics;
    }

    /**
     * @brief Called when a checkpoint is saved
     * @param checkpoint_path Path to saved checkpoint
     * @param metrics Metrics at checkpoint time
     */
    virtual void OnCheckpointSaved(
        const std::string& checkpoint_path,
        const TrainingMetricsSnapshot& metrics) {
        (void)checkpoint_path; (void)metrics;
    }

    /**
     * @brief Called when learning rate changes
     * @param old_lr Previous learning rate
     * @param new_lr New learning rate
     * @param reason Reason for change (e.g., "scheduler", "manual")
     */
    virtual void OnLearningRateChanged(
        float old_lr,
        float new_lr,
        const std::string& reason) {
        (void)old_lr; (void)new_lr; (void)reason;
    }

    // ═══════════════════════════════════════════════════════════════════════
    // LOGGING METHODS (called by plugin to log custom data)
    // ═══════════════════════════════════════════════════════════════════════

    /**
     * @brief Log a scalar metric
     * @param name Metric name
     * @param value Metric value
     * @param step Step number (usually epoch or global step)
     */
    virtual void LogMetric(
        const std::string& name,
        float value,
        int step) {
        (void)name; (void)value; (void)step;
    }

    /**
     * @brief Log multiple metrics at once
     * @param metrics Map of metric name -> value
     * @param step Step number
     */
    virtual void LogMetrics(
        const std::map<std::string, float>& metrics,
        int step) {
        for (const auto& [name, value] : metrics) {
            LogMetric(name, value, step);
        }
    }

    /**
     * @brief Log a parameter (hyperparameter)
     * @param name Parameter name
     * @param value Parameter value
     */
    virtual void LogParam(
        const std::string& name,
        const std::string& value) {
        (void)name; (void)value;
    }

    /**
     * @brief Log an artifact (file)
     * @param name Artifact name
     * @param path Local path to file
     * @param artifact_type Type hint (e.g., "model", "plot", "data")
     */
    virtual void LogArtifact(
        const std::string& name,
        const std::string& path,
        const std::string& artifact_type) {
        (void)name; (void)path; (void)artifact_type;
    }

    /**
     * @brief Log a text note
     * @param name Note name
     * @param text Note content
     */
    virtual void LogText(
        const std::string& name,
        const std::string& text) {
        (void)name; (void)text;
    }

    // ═══════════════════════════════════════════════════════════════════════
    // CONTROL METHODS (plugin can influence training)
    // ═══════════════════════════════════════════════════════════════════════

    /**
     * @brief Check if training should stop early
     * @return true to stop training
     *
     * Called at the end of each epoch. Return true to trigger
     * early stopping (e.g., no improvement for N epochs).
     */
    virtual bool ShouldStopEarly() { return false; }

    /**
     * @brief Get suggested learning rate
     * @param current_lr Current learning rate
     * @param epoch Current epoch
     * @param metrics Current metrics
     * @return New learning rate, or current_lr to keep unchanged
     *
     * Called at the start of each epoch. Return a different value
     * to adjust the learning rate (e.g., for custom schedules).
     */
    virtual float GetSuggestedLearningRate(
        float current_lr,
        int epoch,
        const TrainingMetricsSnapshot& metrics) {
        (void)epoch; (void)metrics;
        return current_lr;
    }
};

} // namespace cyxwiz::plugin
```

### 3.6 IAnalyticsProvider - Data Quality & Analytics

```cpp
// ═══════════════════════════════════════════════════════════════════════════
// File: cyxwiz-engine/include/plugin/ianalytics_provider.h
// Purpose: Interface for data quality and analytics plugins
// ═══════════════════════════════════════════════════════════════════════════

#pragma once

#include "iplugin.h"
#include <vector>
#include <map>
#include <variant>
#include <string>

namespace cyxwiz::plugin {

// ───────────────────────────────────────────────────────────────────────────
// Analytics Value Type
// ───────────────────────────────────────────────────────────────────────────

/**
 * @brief Variant type for analytics results
 */
using AnalyticsValue = std::variant<
    double,                             // Single number
    std::string,                        // Single string
    std::vector<double>,                // Array of numbers
    std::vector<std::string>,           // Array of strings
    std::map<std::string, double>,      // Dictionary of numbers
    std::map<std::string, std::string>  // Dictionary of strings
>;

// ───────────────────────────────────────────────────────────────────────────
// Analytics Report
// ───────────────────────────────────────────────────────────────────────────

/**
 * @brief Comprehensive analytics report
 */
struct AnalyticsReport {
    // ─── Metadata ───
    std::string title;
    std::string summary;
    std::string timestamp;          // ISO 8601
    std::string tool_name;          // e.g., "Great Expectations"
    std::string tool_version;

    // ─── Results ───
    std::map<std::string, AnalyticsValue> metrics;

    // ─── Warnings & Issues ───
    struct Issue {
        enum class Level { Info, Warning, Error, Critical };
        Level level;
        std::string code;           // e.g., "DATA_001"
        std::string message;
        std::string suggestion;     // How to fix
        std::map<std::string, std::string> details;
    };
    std::vector<Issue> issues;

    // ─── Visualizations ───
    struct Chart {
        std::string title;
        std::string type;           // "histogram", "scatter", "bar", etc.
        std::string data_json;      // JSON data for chart
    };
    std::vector<Chart> charts;

    // ─── Export Formats ───
    std::string html_report;        // Full HTML report
    std::string json_report;        // Structured JSON
    std::string markdown_report;    // Markdown summary
};

// ───────────────────────────────────────────────────────────────────────────
// Expectation Result (for Great Expectations-style validation)
// ───────────────────────────────────────────────────────────────────────────

/**
 * @brief Result of a single data expectation check
 */
struct ExpectationResult {
    std::string expectation_type;   // e.g., "expect_column_values_to_not_be_null"
    std::string column;             // Column name (if applicable)
    bool success;                   // Did the expectation pass?

    // ─── Details ───
    std::string message;            // Human-readable result
    std::map<std::string, std::string> observed_value;  // What was found
    std::map<std::string, std::string> expected_value;  // What was expected

    // ─── Statistics ───
    size_t element_count;           // Total elements checked
    size_t unexpected_count;        // Elements that failed
    double unexpected_percent;      // Failure percentage

    // ─── Examples ───
    std::vector<std::string> unexpected_examples;  // Sample failing values
};

// ───────────────────────────────────────────────────────────────────────────
// Expectation Suite
// ───────────────────────────────────────────────────────────────────────────

/**
 * @brief A collection of expectations to validate
 */
struct ExpectationSuite {
    std::string name;
    std::string description;

    struct Expectation {
        std::string type;           // Expectation type
        std::string column;         // Column (if applicable)
        std::map<std::string, std::string> kwargs;  // Parameters
    };
    std::vector<Expectation> expectations;

    // Serialization
    static ExpectationSuite FromJson(const std::string& json);
    std::string ToJson() const;
};

// ───────────────────────────────────────────────────────────────────────────
// IAnalyticsProvider Interface
// ───────────────────────────────────────────────────────────────────────────

/**
 * @brief Interface for data quality and analytics plugins
 *
 * Plugins implementing this interface can provide:
 *   - Data profiling (statistics, distributions)
 *   - Data validation (Great Expectations-style)
 *   - Data comparison (drift detection)
 *   - Quality scoring
 *
 * Implementation requirements:
 *   1. Return PluginCapability::ProvidesAnalytics from GetCapabilities()
 *   2. Implement GetToolName() and ProfileDataset()
 *   3. Optionally implement validation and comparison
 *
 * Thread Safety:
 *   - All methods may be called from analytics worker thread
 */
class IAnalyticsProvider : public virtual IPlugin {
public:
    virtual ~IAnalyticsProvider() = default;

    // ═══════════════════════════════════════════════════════════════════════
    // TOOL INFO
    // ═══════════════════════════════════════════════════════════════════════

    /**
     * @brief Get the analytics tool name
     * @return Tool name (e.g., "Great Expectations", "pandas-profiling")
     */
    virtual const char* GetToolName() const = 0;

    /**
     * @brief Get supported analysis types
     * @return List of analysis types (e.g., ["profile", "validate", "compare"])
     */
    virtual std::vector<std::string> GetSupportedAnalyses() const {
        return {"profile"};
    }

    // ═══════════════════════════════════════════════════════════════════════
    // DATA PROFILING
    // ═══════════════════════════════════════════════════════════════════════

    /**
     * @brief Profile a dataset
     * @param dataset_handle Dataset handle from IDataProvider
     * @param options Analysis options
     * @param progress_callback Progress callback (0.0 to 1.0)
     * @param out_report Output: profiling report
     * @return true on success
     *
     * Generate comprehensive statistics about the dataset:
     *   - Row/column counts
     *   - Data types
     *   - Missing values
     *   - Distributions
     *   - Correlations
     *   - Outliers
     */
    virtual bool ProfileDataset(
        void* dataset_handle,
        const std::map<std::string, std::string>& options,
        std::function<void(float)> progress_callback,
        AnalyticsReport& out_report) = 0;

    // ═══════════════════════════════════════════════════════════════════════
    // DATA VALIDATION (Great Expectations-style)
    // ═══════════════════════════════════════════════════════════════════════

    /**
     * @brief Validate a dataset against expectations
     * @param dataset_handle Dataset handle
     * @param suite Expectation suite to validate
     * @param out_results Output: results for each expectation
     * @return true if all expectations passed
     */
    virtual bool ValidateDataset(
        void* dataset_handle,
        const ExpectationSuite& suite,
        std::vector<ExpectationResult>& out_results) {
        (void)dataset_handle; (void)suite; (void)out_results;
        return false;  // Not supported by default
    }

    /**
     * @brief Generate expectations from profiling
     * @param dataset_handle Dataset handle
     * @return Auto-generated expectation suite
     *
     * Analyze the data and generate a reasonable set of expectations
     * that would pass on this data (baseline expectations).
     */
    virtual ExpectationSuite GenerateExpectations(void* dataset_handle) {
        (void)dataset_handle;
        return {};
    }

    /**
     * @brief Get available expectation types
     * @return List of supported expectation types
     */
    virtual std::vector<std::string> GetExpectationTypes() const {
        return {};
    }

    // ═══════════════════════════════════════════════════════════════════════
    // DATA COMPARISON (Drift Detection)
    // ═══════════════════════════════════════════════════════════════════════

    /**
     * @brief Compare two datasets for drift
     * @param dataset_a Reference dataset (e.g., training data)
     * @param dataset_b Current dataset (e.g., production data)
     * @param options Comparison options
     * @param out_report Output: comparison report
     * @return true on success
     *
     * Detect data drift between two datasets:
     *   - Distribution changes
     *   - Missing columns
     *   - Type changes
     *   - Statistical differences
     */
    virtual bool CompareDatasets(
        void* dataset_a,
        void* dataset_b,
        const std::map<std::string, std::string>& options,
        AnalyticsReport& out_report) {
        (void)dataset_a; (void)dataset_b; (void)options; (void)out_report;
        return false;  // Not supported by default
    }

    // ═══════════════════════════════════════════════════════════════════════
    // REPORT EXPORT
    // ═══════════════════════════════════════════════════════════════════════

    /**
     * @brief Export report to file
     * @param report Report to export
     * @param output_path Output file path
     * @param format Format ("html", "json", "pdf", "markdown")
     * @return true on success
     */
    virtual bool ExportReport(
        const AnalyticsReport& report,
        const std::string& output_path,
        const std::string& format) {
        (void)report; (void)output_path; (void)format;
        return false;
    }

    /**
     * @brief Get supported export formats
     * @return List of format strings
     */
    virtual std::vector<std::string> GetExportFormats() const {
        return {"json"};
    }
};

} // namespace cyxwiz::plugin
```

---

## 4. PluginContext API

```cpp
// ═══════════════════════════════════════════════════════════════════════════
// File: cyxwiz-engine/include/plugin/plugin_context.h
// Purpose: API surface exposed to plugins
// ═══════════════════════════════════════════════════════════════════════════

#pragma once

#include <string>
#include <vector>
#include <map>
#include <functional>
#include <memory>

namespace cyxwiz::plugin {

// Forward declarations
struct NodeTypeInfo;
class ITrainingHook;

/**
 * @brief API surface exposed to plugins
 *
 * PluginContext is the plugin's gateway to the host application.
 * It provides controlled access to core systems with proper
 * permission checking and resource limits.
 *
 * Thread Safety:
 *   - All methods are thread-safe unless otherwise noted
 *   - UI methods must be called from main thread
 */
class PluginContext {
public:
    virtual ~PluginContext() = default;

    // ═══════════════════════════════════════════════════════════════════════
    // PLUGIN IDENTITY
    // ═══════════════════════════════════════════════════════════════════════

    /// Get this plugin's unique ID
    virtual const std::string& GetPluginId() const = 0;

    /// Get plugin's data directory (for config, cache)
    /// Returns: <user>/CyxWiz/plugins/<plugin_id>/data/
    virtual std::string GetPluginDataPath() const = 0;

    /// Get plugin's resource directory (bundled with plugin)
    virtual std::string GetPluginResourcePath() const = 0;

    // ═══════════════════════════════════════════════════════════════════════
    // LOGGING
    // ═══════════════════════════════════════════════════════════════════════

    /// Log info message (automatically prefixed with plugin name)
    virtual void LogInfo(const std::string& message) = 0;
    virtual void LogWarning(const std::string& message) = 0;
    virtual void LogError(const std::string& message) = 0;
    virtual void LogDebug(const std::string& message) = 0;

    // ═══════════════════════════════════════════════════════════════════════
    // UI INTEGRATION
    // ═══════════════════════════════════════════════════════════════════════

    /// Register a panel with the sidebar
    /// @param name Panel display name
    /// @param icon FontAwesome icon
    /// @param visible_ptr Pointer to visibility bool
    /// @param category Menu category (default: "Plugins")
    virtual void RegisterPanel(
        const std::string& name,
        const std::string& icon,
        bool* visible_ptr,
        const std::string& category = "Plugins") = 0;

    /// Unregister a panel
    virtual void UnregisterPanel(const std::string& name) = 0;

    /// Add menu item
    virtual void AddMenuItem(
        const std::string& menu,        // e.g., "Plugins", "Tools"
        const std::string& item,        // Item name
        std::function<void()> callback,
        const std::string& shortcut = "") = 0;

    /// Remove menu item
    virtual void RemoveMenuItem(
        const std::string& menu,
        const std::string& item) = 0;

    /// Show notification toast
    virtual void ShowNotification(
        const std::string& title,
        const std::string& message,
        const std::string& type = "info") = 0;  // "info", "warning", "error", "success"

    // ═══════════════════════════════════════════════════════════════════════
    // NODE GRAPH INTEGRATION
    // ═══════════════════════════════════════════════════════════════════════

    /// Register custom node type
    virtual void RegisterNodeType(const NodeTypeInfo& info) = 0;

    /// Unregister custom node type
    virtual void UnregisterNodeType(const std::string& type_id) = 0;

    // ═══════════════════════════════════════════════════════════════════════
    // TRAINING INTEGRATION
    // ═══════════════════════════════════════════════════════════════════════

    /// Register training hook
    virtual void RegisterTrainingHook(ITrainingHook* hook) = 0;

    /// Unregister training hook
    virtual void UnregisterTrainingHook(ITrainingHook* hook) = 0;

    /// Check if training is currently active
    virtual bool IsTrainingActive() const = 0;

    // ═══════════════════════════════════════════════════════════════════════
    // DATA ACCESS (Read-Only)
    // ═══════════════════════════════════════════════════════════════════════

    /// Get list of loaded dataset names
    virtual std::vector<std::string> GetLoadedDatasets() const = 0;

    // ═══════════════════════════════════════════════════════════════════════
    // ASYNC TASK EXECUTION
    // ═══════════════════════════════════════════════════════════════════════

    /// Submit an async task
    /// @param name Task name (shown in UI)
    /// @param task_func Task function (receives progress callback)
    /// @param completion_cb Completion callback
    /// @return Task ID
    virtual uint64_t SubmitTask(
        const std::string& name,
        std::function<void(std::function<void(float, const std::string&)>)> task_func,
        std::function<void(bool, const std::string&)> completion_cb = nullptr) = 0;

    /// Cancel a task
    virtual bool CancelTask(uint64_t task_id) = 0;

    // ═══════════════════════════════════════════════════════════════════════
    // PYTHON INTEGRATION
    // ═══════════════════════════════════════════════════════════════════════

    /// Execute Python code
    virtual bool ExecutePython(
        const std::string& code,
        std::string& out_result,
        std::string& out_error) = 0;

    /// Check if Python is available
    virtual bool IsPythonAvailable() const = 0;

    // ═══════════════════════════════════════════════════════════════════════
    // SETTINGS & PERSISTENCE
    // ═══════════════════════════════════════════════════════════════════════

    /// Get plugin setting (persisted across sessions)
    virtual std::string GetSetting(
        const std::string& key,
        const std::string& default_value = "") const = 0;

    /// Set plugin setting
    virtual void SetSetting(
        const std::string& key,
        const std::string& value) = 0;

    /// Get all settings for this plugin
    virtual std::map<std::string, std::string> GetAllSettings() const = 0;

    // ═══════════════════════════════════════════════════════════════════════
    // INTER-PLUGIN COMMUNICATION
    // ═══════════════════════════════════════════════════════════════════════

    /// Send message to another plugin
    virtual bool SendMessage(
        const std::string& target_plugin_id,
        const std::string& message_type,
        const std::string& payload) = 0;

    /// Register message handler
    virtual void RegisterMessageHandler(
        const std::string& message_type,
        std::function<void(const std::string& from_plugin,
                           const std::string& payload)> handler) = 0;

    // ═══════════════════════════════════════════════════════════════════════
    // RESOURCE LIMITS
    // ═══════════════════════════════════════════════════════════════════════

    /// Get memory limit for this plugin (0 = unlimited)
    virtual size_t GetMemoryLimit() const = 0;

    /// Get current memory usage
    virtual size_t GetMemoryUsage() const = 0;
};

} // namespace cyxwiz::plugin
```

---

## 5. Plugin Manifest Schema

### 5.1 Complete `plugin.json` Specification

```json
{
  "$schema": "https://cyxwiz.io/schemas/plugin-manifest-v1.json",

  "id": "com.example.my-plugin",
  "name": "My Awesome Plugin",
  "version": "1.2.3",
  "api_version": "1.0.0",

  "description": "A comprehensive description of what this plugin does",
  "author": "Your Name <your.email@example.com>",
  "license": "MIT",
  "homepage": "https://github.com/you/my-plugin",
  "repository": "https://github.com/you/my-plugin.git",
  "documentation": "https://my-plugin.docs.io",

  "capabilities": [
    "ProvidesNodes",
    "ProvidesPanels",
    "ProvidesTraining",
    "RequiresPython",
    "RequiresNetwork"
  ],

  "dependencies": [
    {
      "id": "com.cyxwiz.python-integration",
      "version": ">=1.0.0"
    }
  ],

  "optional_dependencies": [
    {
      "id": "com.cyxwiz.gpu-compute",
      "version": ">=1.0.0",
      "reason": "Enables GPU-accelerated processing"
    }
  ],

  "platforms": {
    "windows": {
      "library": "my_plugin.dll",
      "min_os_version": "10.0"
    },
    "linux": {
      "library": "libmy_plugin.so",
      "min_os_version": "5.0"
    },
    "macos": {
      "library": "libmy_plugin.dylib",
      "min_os_version": "11.0"
    }
  },

  "resources": {
    "icons": "resources/icons/",
    "templates": "resources/templates/",
    "python": "python/"
  },

  "permissions": [
    "filesystem:plugin_data",
    "network:read",
    "python:execute"
  ],

  "settings_schema": {
    "type": "object",
    "properties": {
      "api_endpoint": {
        "type": "string",
        "title": "API Endpoint",
        "description": "URL of the backend service",
        "default": "http://localhost:8080"
      },
      "auto_sync": {
        "type": "boolean",
        "title": "Auto Sync",
        "description": "Automatically sync data on startup",
        "default": true
      },
      "max_retries": {
        "type": "integer",
        "title": "Max Retries",
        "minimum": 0,
        "maximum": 10,
        "default": 3
      }
    }
  },

  "nodes": [
    {
      "type_id": "custom.attention.sparse",
      "display_name": "Sparse Attention",
      "category": "Attention"
    }
  ],

  "panels": [
    {
      "id": "my-dashboard",
      "display_name": "My Dashboard",
      "category": "Tools",
      "icon": "\\uf080"
    }
  ],

  "signature": {
    "algorithm": "ed25519",
    "public_key": "MCowBQYDK2VwAyEA...",
    "signature": "base64-encoded-signature"
  }
}
```

### 5.2 Field Reference

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `id` | string | Yes | Reverse-domain unique identifier |
| `name` | string | Yes | Human-readable name |
| `version` | semver | Yes | Plugin version (MAJOR.MINOR.PATCH) |
| `api_version` | semver | Yes | CyxWiz Plugin API version |
| `description` | string | Yes | Plugin description |
| `author` | string | Yes | Author name/email |
| `license` | string | No | License identifier (SPDX) |
| `homepage` | url | No | Plugin homepage |
| `repository` | url | No | Source repository |
| `capabilities` | array | Yes | Capability flags |
| `dependencies` | array | No | Required plugins |
| `platforms` | object | Yes | Platform-specific library paths |
| `permissions` | array | No | Required permissions |
| `settings_schema` | object | No | JSON Schema for settings UI |
| `signature` | object | No | Cryptographic signature |

---

## 6. Plugin Discovery & Loading

### 6.1 Directory Structure

```
Scan Order (first found wins):
1. <project>/plugins/        # Project-specific plugins
2. <user>/CyxWiz/plugins/    # User-installed plugins
3. <app>/plugins/            # Bundled plugins
```

### 6.2 Loading Flow

```cpp
// Pseudocode for plugin loading

class PluginLoader {
    std::vector<PluginManifest> DiscoverPlugins() {
        std::vector<PluginManifest> manifests;

        for (const auto& search_path : GetSearchPaths()) {
            for (const auto& entry : fs::directory_iterator(search_path)) {
                if (entry.is_directory()) {
                    auto manifest_path = entry.path() / "plugin.json";
                    if (fs::exists(manifest_path)) {
                        auto manifest = ParseManifest(manifest_path);
                        if (manifest.is_valid && IsApiCompatible(manifest)) {
                            manifests.push_back(manifest);
                        }
                    }
                }
            }
        }

        return manifests;
    }

    void* LoadPlugin(const PluginManifest& manifest) {
        // Platform-specific loading
        #ifdef _WIN32
            return LoadLibraryA(manifest.library_path.c_str());
        #else
            return dlopen(manifest.library_path.c_str(), RTLD_NOW);
        #endif
    }
};
```

---

## 7. Dependency Resolution

### 7.1 Algorithm (Kahn's Topological Sort)

```cpp
std::vector<std::string> ResolveDependencies(
    const std::vector<PluginManifest>& manifests)
{
    // Build adjacency list and in-degree map
    std::map<std::string, std::vector<std::string>> graph;
    std::map<std::string, int> in_degree;

    for (const auto& m : manifests) {
        graph[m.id] = {};
        in_degree[m.id] = 0;
    }

    for (const auto& m : manifests) {
        for (const auto& dep : m.dependencies) {
            graph[dep.id].push_back(m.id);
            in_degree[m.id]++;
        }
    }

    // Kahn's algorithm
    std::queue<std::string> queue;
    for (const auto& [id, degree] : in_degree) {
        if (degree == 0) queue.push(id);
    }

    std::vector<std::string> result;
    while (!queue.empty()) {
        auto id = queue.front();
        queue.pop();
        result.push_back(id);

        for (const auto& dependent : graph[id]) {
            if (--in_degree[dependent] == 0) {
                queue.push(dependent);
            }
        }
    }

    if (result.size() != manifests.size()) {
        throw CircularDependencyException();
    }

    return result;
}
```

### 7.2 Version Constraint Matching

| Constraint | Meaning | Example |
|------------|---------|---------|
| `1.0.0` | Exact match | Only 1.0.0 |
| `>=1.0.0` | Greater or equal | 1.0.0, 1.5.0, 2.0.0 |
| `^1.0.0` | Compatible | 1.0.0 to <2.0.0 |
| `~1.0.0` | Patch only | 1.0.0 to <1.1.0 |
| `>=1.0.0 <2.0.0` | Range | 1.0.0 to 1.x.x |

---

## 8. Registration Systems

### 8.1 NodeRegistry

```cpp
class NodeRegistry {
public:
    static NodeRegistry& Instance();

    void RegisterNodeType(const std::string& plugin_id, const NodeTypeInfo& info);
    void UnregisterPlugin(const std::string& plugin_id);

    std::vector<NodeTypeInfo> GetAllNodeTypes() const;
    const NodeTypeInfo* GetNodeType(const std::string& type_id) const;
    INodeProvider* GetProvider(const std::string& type_id) const;

private:
    std::map<std::string, NodeTypeInfo> node_types_;
    std::map<std::string, INodeProvider*> providers_;
    mutable std::mutex mutex_;
};
```

### 8.2 Integration with NodeEditor

```cpp
// In NodeEditor::InitializeSearchableNodes()
void NodeEditor::InitializeSearchableNodes() {
    // ... existing builtin nodes ...

    // Add plugin nodes
    for (const auto& info : NodeRegistry::Instance().GetAllNodeTypes()) {
        SearchableNode node;
        node.type = NodeType::PluginNode;
        node.plugin_type_id = info.type_id;
        node.name = info.display_name;
        node.category = "Plugin > " + info.category;
        all_searchable_nodes_.push_back(node);
    }
}
```

---

## 9. Versioning & Compatibility

### 9.1 Semantic Versioning Policy

| Change Type | Version Bump | Example |
|-------------|--------------|---------|
| Breaking API change | MAJOR | 1.x.x → 2.0.0 |
| New feature (backward-compatible) | MINOR | 1.0.x → 1.1.0 |
| Bug fix | PATCH | 1.0.0 → 1.0.1 |

### 9.2 Compatibility Rules

```cpp
bool IsApiCompatible(const PluginVersion& plugin_api) {
    // Major version must match exactly
    if (plugin_api.major != CYXWIZ_PLUGIN_API_VERSION_MAJOR) {
        return false;
    }

    // Plugin minor must be <= host minor
    // (older plugins work with newer hosts)
    if (plugin_api.minor > CYXWIZ_PLUGIN_API_VERSION_MINOR) {
        return false;
    }

    return true;
}
```

### 9.3 Deprecation Policy

1. Feature marked `[[deprecated]]` in release N
2. Warning shown when plugin uses deprecated feature
3. Feature removed in release N+2 (minimum)
4. Migration guide provided in release notes

---

## 10. Security Model

### 10.1 Permission System

| Permission | Description |
|------------|-------------|
| `filesystem:read` | Read files outside plugin directory |
| `filesystem:write` | Write files outside plugin directory |
| `filesystem:plugin_data` | Read/write plugin data directory |
| `network:read` | Make outbound network requests |
| `network:listen` | Listen on network ports |
| `python:execute` | Execute Python code |
| `gpu:compute` | Use GPU for computation |
| `training:modify` | Modify training parameters |
| `data:full_access` | Full access to loaded datasets |

### 10.2 Permission Checking

```cpp
class PermissionChecker {
public:
    PermissionChecker(const std::vector<std::string>& granted);

    bool HasPermission(const std::string& permission) const;
    void RequirePermission(const std::string& permission) const;

private:
    std::set<std::string> permissions_;
};
```

### 10.3 Plugin Signing

- **Algorithm**: Ed25519 (fast, secure)
- **Signed content**: Hash of manifest (excluding signature block)
- **Verification**: Against list of trusted public keys
- **Unsigned plugins**: Warning shown, user can allow/deny

---

## 11. Error Handling & Crash Isolation

### 11.1 SafeExecute Wrapper

```cpp
class PluginSandbox {
public:
    template<typename F>
    static std::optional<std::string> SafeExecute(
        const std::string& plugin_id,
        F&& func)
    {
        try {
            func();
            return std::nullopt;  // Success
        }
        catch (const std::exception& e) {
            return fmt::format("Plugin {} error: {}", plugin_id, e.what());
        }
        catch (...) {
            return fmt::format("Plugin {} unknown error", plugin_id);
        }
    }
};

// Usage
auto error = PluginSandbox::SafeExecute(plugin->GetId(), [&]() {
    plugin->Update(delta_time);
});

if (error) {
    spdlog::error(*error);
    DisablePlugin(plugin->GetId());
}
```

### 11.2 Exception Hierarchy

```cpp
class PluginException : public std::runtime_error { ... };
class PluginLoadException : public PluginException { ... };
class PluginDependencyException : public PluginException { ... };
class PluginSecurityException : public PluginException { ... };
```

---

## 12. Example Plugins

### 12.1 MLflow Integration Plugin

```cpp
class MLflowPlugin : public IPlugin, public ITrainingHook, public IPanelProvider {
public:
    const char* GetId() const override { return "com.cyxwiz.mlflow"; }
    const char* GetName() const override { return "MLflow Integration"; }

    PluginCapability GetCapabilities() const override {
        return PluginCapability::ProvidesTraining |
               PluginCapability::ProvidesPanels |
               PluginCapability::RequiresNetwork;
    }

    bool Initialize(PluginContext* context) override {
        context_ = context;
        tracking_uri_ = context->GetSetting("tracking_uri", "http://localhost:5000");
        context->RegisterTrainingHook(this);
        return true;
    }

    void OnTrainingStart(const TrainingConfigSnapshot& config) override {
        run_id_ = mlflow_.CreateRun(config.model_name);
        for (const auto& [k, v] : config.hyperparameters) {
            mlflow_.LogParam(k, v);
        }
    }

    void OnEpochEnd(int epoch, const TrainingMetricsSnapshot& m) override {
        mlflow_.LogMetric("train_loss", m.train_loss, epoch);
        mlflow_.LogMetric("val_accuracy", m.val_accuracy, epoch);
    }

    // ... panel methods ...

private:
    PluginContext* context_;
    std::string tracking_uri_;
    std::string run_id_;
    MLflowClient mlflow_;
};

CYXWIZ_PLUGIN(MLflowPlugin)
```

### 12.2 Custom Attention Node Plugin

```cpp
class SparseAttentionPlugin : public IPlugin, public INodeProvider {
public:
    const char* GetId() const override { return "com.example.sparse-attention"; }

    std::vector<NodeTypeInfo> GetNodeTypes() const override {
        return {{
            .type_id = "attention.sparse",
            .display_name = "Sparse Attention",
            .category = "Attention",
            .inputs = {
                {"query", PinType::Tensor, true},
                {"key", PinType::Tensor, true},
                {"value", PinType::Tensor, true}
            },
            .outputs = {
                {"output", PinType::Tensor, false}
            },
            .parameters = {
                {"heads", "Attention Heads", "8", ParameterDefinition::Type::Int}
            }
        }};
    }

    void* CreateNodeInstance(const std::string& type_id,
                              const std::map<std::string, std::string>& params) override {
        return new SparseAttentionLayer(std::stoi(params.at("heads")));
    }

    bool ExecuteNode(void* instance,
                     const std::map<std::string, TensorData>& inputs,
                     std::map<std::string, TensorData>& outputs) override {
        auto* layer = static_cast<SparseAttentionLayer*>(instance);
        outputs["output"] = layer->Forward(inputs.at("query"),
                                            inputs.at("key"),
                                            inputs.at("value"));
        return true;
    }

    void DestroyNodeInstance(void* instance) override {
        delete static_cast<SparseAttentionLayer*>(instance);
    }
};

CYXWIZ_PLUGIN(SparseAttentionPlugin)
```

### 12.3 Great Expectations Plugin

```cpp
class GreatExpectationsPlugin : public IPlugin,
                                 public IAnalyticsProvider,
                                 public IPanelProvider {
public:
    const char* GetId() const override { return "com.cyxwiz.great-expectations"; }
    const char* GetToolName() const override { return "Great Expectations"; }

    PluginCapability GetCapabilities() const override {
        return PluginCapability::ProvidesAnalytics |
               PluginCapability::ProvidesPanels |
               PluginCapability::RequiresPython;
    }

    bool ProfileDataset(void* handle,
                        const std::map<std::string, std::string>& options,
                        std::function<void(float)> progress,
                        AnalyticsReport& report) override {
        // Call Python Great Expectations via PluginContext
        std::string code = R"(
import great_expectations as ge
# ... profiling code ...
        )";

        std::string result, error;
        if (context_->ExecutePython(code, result, error)) {
            report = ParseReport(result);
            return true;
        }
        return false;
    }

    // ... validation, comparison, panel methods ...
};

CYXWIZ_PLUGIN(GreatExpectationsPlugin)
```

---

## 13. Implementation Roadmap

### Phase 1: Core Infrastructure (2-3 weeks)

| Task | Files | Priority |
|------|-------|----------|
| Define all interfaces | `include/plugin/*.h` | P0 |
| Implement PluginLoader | `src/plugin/plugin_loader.cpp` | P0 |
| Implement PluginManager | `src/plugin/plugin_manager.cpp` | P0 |
| Implement PluginContext | `src/plugin/plugin_context_impl.cpp` | P0 |
| Add plugin manifest parsing | `src/plugin/manifest_parser.cpp` | P0 |

### Phase 2: Registries (1-2 weeks)

| Task | Files | Priority |
|------|-------|----------|
| Implement NodeRegistry | `src/plugin/node_registry.cpp` | P0 |
| Integrate with NodeEditor | `src/gui/node_editor.cpp` | P0 |
| Implement PanelRegistry | `src/plugin/panel_registry.cpp` | P1 |
| Integrate with MainWindow | `src/gui/main_window.cpp` | P1 |
| Implement TrainingHookManager | `src/plugin/training_hooks.cpp` | P1 |
| Integrate with TrainingManager | `src/core/training_manager.cpp` | P1 |

### Phase 3: Security & Stability (1-2 weeks)

| Task | Files | Priority |
|------|-------|----------|
| Implement PermissionChecker | `src/plugin/permission_checker.cpp` | P1 |
| Add crash isolation (SafeExecute) | `src/plugin/plugin_sandbox.cpp` | P1 |
| Implement plugin signing | `src/plugin/signature_verifier.cpp` | P2 |
| Add resource limits | `src/plugin/resource_monitor.cpp` | P2 |

### Phase 4: Hot Reload & DX (1 week)

| Task | Files | Priority |
|------|-------|----------|
| Implement hot reload | `src/plugin/hot_reload.cpp` | P2 |
| Create plugin SDK headers | `sdk/include/cyxwiz/*` | P1 |
| Write developer docs | `docs/plugin-development.md` | P1 |
| Create example plugins | `examples/plugins/*` | P1 |

### Phase 5: Testing & Polish (1-2 weeks)

| Task | Priority |
|------|----------|
| Unit tests for all components | P0 |
| Integration tests with sample plugins | P1 |
| Performance benchmarks | P2 |
| Documentation review | P1 |

---

## References

- [VS Code Extension API](https://code.visualstudio.com/api)
- [Unity Package Manager](https://docs.unity3d.com/Manual/Packages.html)
- [Unreal Engine Plugins](https://docs.unrealengine.com/ProductionPipelines/Plugins/)
- [MATLAB Plugin Architecture](https://www.mathworks.com/help/matlab/matlab_external/)
- [Great Expectations](https://greatexpectations.io/)
- [MLflow](https://mlflow.org/)

---

*Document Version: 1.0.0*
*Last Updated: 2026-01-22*
