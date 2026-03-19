# CyxWiz Engine Plugin System Architecture

**Version:** 1.0.0
**Status:** Design Complete
**Last Updated:** 2026-01-29

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Design Goals & Principles](#2-design-goals--principles)
3. [Architecture Overview](#3-architecture-overview)
4. [Core Interfaces](#4-core-interfaces)
5. [PluginContext API](#5-plugincontext-api)
6. [Plugin Manifest Schema](#6-plugin-manifest-schema)
7. [Plugin Discovery & Loading](#7-plugin-discovery--loading)
8. [Dependency Resolution](#8-dependency-resolution)
9. [Registration Systems](#9-registration-systems)
10. [Engine Integration Points](#10-engine-integration-points)
11. [Security Model](#11-security-model)
12. [Python Plugin Support](#12-python-plugin-support)
13. [Plugin Manager UI](#13-plugin-manager-ui)
14. [Versioning & Compatibility](#14-versioning--compatibility)
15. [Example Plugins](#15-example-plugins)
16. [Implementation Roadmap](#16-implementation-roadmap)
17. [File Map](#17-file-map)

---

## 1. Executive Summary

The CyxWiz Engine Plugin System enables third-party developers to extend the engine with:

- **Custom Node Types** — New layers, operations, and algorithms for the visual graph editor
- **UI Panels** — Custom interfaces for specialized workflows
- **Data Loaders** — Support for additional data formats and sources
- **Training Hooks** — Integration with experiment tracking (MLflow, W&B, TensorBoard)
- **Analytics Tools** — Data quality validation and profiling

Plugins are native DLLs (C++) or Python scripts, loaded at runtime via declarative JSON manifests. The system uses capability-based security, crash isolation, and topological dependency resolution.

---

## 2. Design Goals & Principles

### Goals

| Goal | Description |
|------|-------------|
| Minimal Intrusion | Existing engine code requires minimal changes |
| Safety First | Crash isolation, permission system, resource limits |
| Cross-Platform | Windows, macOS, Linux support |
| Versioned API | Clear compatibility rules and deprecation policy |
| Familiar Patterns | Builds on existing CyxWiz singleton/callback patterns |

### Principles

1. **Plugins are DLLs/SOs** — Native code, dynamically loaded at runtime
2. **Declarative Manifests** — `plugin.json` describes capabilities, dependencies, permissions
3. **Capability-Based Security** — Plugins declare required permissions, users approve
4. **Registry Pattern** — Central registries for nodes, panels, hooks, data loaders, analytics
5. **Loose Coupling** — Plugins interact with the engine exclusively through `PluginContext`
6. **No Event Bus** — Uses existing `std::function` callback pattern consistent with engine codebase

---

## 3. Architecture Overview

### High-Level Component Diagram

```
CyxWiz Engine
├── Application (init/shutdown lifecycle)
│
├── PluginManager (Singleton)
│   ├── PluginLoader (DLL loading, symbol resolution)
│   ├── ManifestParser (plugin.json parsing)
│   ├── DependencyResolver (Kahn's topological sort)
│   ├── PermissionManager (security checks)
│   └── PluginContext (API surface for plugins)
│
├── Registration Systems
│   ├── PluginNodeRegistry (custom node types → NodeEditor)
│   ├── PluginPanelRegistry (custom panels → MainWindow)
│   ├── PluginTrainingHookManager (training callbacks → TrainingManager)
│   ├── PluginDataLoaderRegistry (data formats → DataRegistry)
│   └── PluginAnalyticsRegistry (analytics tools)
│
├── PythonPluginAdapter (wraps Python plugins via ScriptingEngine)
│
└── PluginManagerPanel (ImGui UI for managing plugins)
```

### Plugin Lifecycle

```
                    ┌──────────┐
                    │ Unloaded │
                    └────┬─────┘
                         │ LoadLibrary()
                    ┌────▼─────┐
                    │  Loaded  │
                    └────┬─────┘
                         │ CreatePlugin() + Initialize()
                    ┌────▼────────┐
                    │ Initialized │
                    └────┬────────┘
                         │ OnActivate()
                    ┌────▼────┐
                    │  Active │◄─── Update() called each frame
                    └────┬────┘
                         │ OnDeactivate()
                    ┌────▼────────┐
                    │ Initialized │
                    └────┬────────┘
                         │ Shutdown() + DestroyPlugin()
                    ┌────▼─────┐
                    │ Unloaded │
                    └──────────┘

Error at any stage ──► ┌────────┐
                       │ Failed │
                       └────────┘

User disables ──────► ┌──────────┐
                      │ Disabled │
                      └──────────┘
```

### Plugin Directory Structure

```
plugins/
├── mlflow-integration/
│   ├── plugin.json              # Manifest
│   ├── mlflow_plugin.dll        # Windows binary
│   ├── libmlflow_plugin.so      # Linux binary
│   └── resources/               # Icons, configs
├── great-expectations/
│   ├── plugin.json
│   ├── python/                  # Python scripts
│   │   └── ge_plugin.py
│   └── resources/
└── custom-nodes/
    ├── plugin.json
    └── custom_nodes.dll
```

---

## 4. Core Interfaces

### 4.1 Common Types

```cpp
// Plugin API version constants
constexpr int CYXWIZ_PLUGIN_API_VERSION_MAJOR = 1;
constexpr int CYXWIZ_PLUGIN_API_VERSION_MINOR = 0;
constexpr int CYXWIZ_PLUGIN_API_VERSION_PATCH = 0;
constexpr uint32_t CYXWIZ_PLUGIN_API_VERSION =
    (CYXWIZ_PLUGIN_API_VERSION_MAJOR << 16) |
    (CYXWIZ_PLUGIN_API_VERSION_MINOR << 8) |
    CYXWIZ_PLUGIN_API_VERSION_PATCH;

// Plugin capability flags (bitmask)
enum class PluginCapability : uint32_t {
    None             = 0,
    ProvidesNodes    = 1 << 0,   // Adds custom node types
    ProvidesPanels   = 1 << 1,   // Adds UI panels
    ProvidesData     = 1 << 2,   // Adds data loaders
    ProvidesTraining = 1 << 3,   // Hooks into training
    ProvidesAnalytics= 1 << 4,   // Data quality tools
    RequiresPython   = 1 << 5,   // Needs Python interpreter
    RequiresGPU      = 1 << 6,   // Needs GPU compute
    RequiresNetwork  = 1 << 7,   // Needs network access
    SupportsHotReload= 1 << 8    // Can be reloaded at runtime
};

// Plugin states
enum class PluginState {
    Unloaded,       // DLL not loaded
    Loaded,         // DLL loaded, CreatePlugin() not yet called
    Initialized,    // Initialize() called successfully
    Active,         // OnActivate() called, receiving Update()
    Failed,         // Error occurred during lifecycle
    Disabled        // User disabled
};

// Semantic version
struct PluginVersion {
    int major = 0, minor = 0, patch = 0;

    static PluginVersion Parse(const std::string& str);   // "1.2.3"
    std::string ToString() const;
    bool operator<(const PluginVersion& o) const;
    bool operator==(const PluginVersion& o) const;
    bool operator<=(const PluginVersion& o) const;
    // ... other comparison operators
};
```

### 4.2 IPlugin — Base Interface

All plugins must implement this interface.

```cpp
class IPlugin {
public:
    virtual ~IPlugin() = default;

    // --- Metadata ---
    virtual const char* GetId() const = 0;            // "com.cyxwiz.mlflow"
    virtual const char* GetName() const = 0;          // "MLflow Integration"
    virtual PluginVersion GetVersion() const = 0;     // {1, 0, 0}
    virtual const char* GetDescription() const = 0;
    virtual const char* GetAuthor() const = 0;
    virtual PluginCapability GetCapabilities() const = 0;
    virtual uint32_t GetApiVersion() const = 0;
    virtual std::vector<std::string> GetDependencies() const { return {}; }

    // --- Lifecycle ---
    virtual bool Initialize(PluginContext* context) = 0;  // Called once after DLL load
    virtual void OnActivate() {}                          // After all deps initialized
    virtual void Update(float delta_time) {}              // Called each frame (optional)
    virtual void OnDeactivate() {}                        // Before shutdown
    virtual void Shutdown() = 0;                          // Release all resources

    // --- State persistence (for hot reload) ---
    virtual std::string SaveState() { return "{}"; }
    virtual void RestoreState(const std::string& state) {}
};

// Factory functions exported by plugin DLLs
extern "C" IPlugin* CreatePlugin();
extern "C" void DestroyPlugin(IPlugin* plugin);
extern "C" uint32_t GetPluginApiVersion();

// Export macro for plugin authors
#ifdef _WIN32
    #define CYXWIZ_PLUGIN_EXPORT __declspec(dllexport)
#else
    #define CYXWIZ_PLUGIN_EXPORT __attribute__((visibility("default")))
#endif

#define CYXWIZ_PLUGIN(PluginClass) \
    extern "C" CYXWIZ_PLUGIN_EXPORT IPlugin* CreatePlugin() { \
        return new PluginClass(); \
    } \
    extern "C" CYXWIZ_PLUGIN_EXPORT void DestroyPlugin(IPlugin* p) { \
        delete p; \
    } \
    extern "C" CYXWIZ_PLUGIN_EXPORT uint32_t GetPluginApiVersion() { \
        return CYXWIZ_PLUGIN_API_VERSION; \
    }
```

### 4.3 INodeProvider — Custom Node Types

Plugins providing custom nodes for the visual graph editor.

```cpp
// Pin type for node connections
enum class PinType {
    Tensor, Labels, Parameters, Loss, Optimizer,
    Dataset, Scalar, String, Any
};

struct PinDefinition {
    std::string name;
    PinType type;
    bool is_input;
    bool is_required = true;
    bool is_variadic = false;
    int min_connections = 0;
    int max_connections = 1;    // -1 = unlimited
    std::string tooltip;
    std::string default_value;
};

struct ParameterDefinition {
    std::string key;            // Internal key
    std::string display_name;   // UI label
    std::string default_value;
    enum class Type {
        Int, Float, String, Bool, Enum,
        IntArray, FloatArray, FilePath, Color
    } type;
    std::string min_value, max_value, regex_pattern;
    std::vector<std::string> enum_options;  // For Enum type
    std::string tooltip, help_url;
};

struct CodeTemplate {
    std::string framework;      // "PyTorch", "TensorFlow", "Keras", "PyCyxWiz"
    std::string code_template;  // Code with $param placeholders
    std::string import_line;    // Required import statement
};

using ShapeInferenceFunc = std::function<
    std::vector<std::vector<size_t>>(
        const std::vector<std::vector<size_t>>& input_shapes,
        const std::map<std::string, std::string>& params
    )>;

struct NodeTypeInfo {
    std::string type_id;        // Unique ID (e.g., "com.example.sparse-attention")
    std::string display_name;
    std::string category;       // Palette category (e.g., "Attention")
    std::string icon;           // FontAwesome icon
    uint32_t header_color;      // 0xRRGGBBAA
    uint32_t body_color;

    std::vector<PinDefinition> inputs;
    std::vector<PinDefinition> outputs;
    std::vector<ParameterDefinition> parameters;
    std::vector<CodeTemplate> code_templates;
    ShapeInferenceFunc shape_inference;

    std::string description, author, help_url;
    std::vector<std::string> tags;
};

struct TensorData {
    std::vector<float> data;
    std::vector<size_t> shape;
    size_t NumElements() const;
};

class INodeProvider : public virtual IPlugin {
public:
    // Registration
    virtual std::vector<NodeTypeInfo> GetNodeTypes() const = 0;

    // Instance lifecycle
    virtual void* CreateNodeInstance(
        const std::string& type_id,
        const std::map<std::string, std::string>& params) = 0;

    virtual bool ExecuteNode(
        void* instance,
        const std::map<std::string, TensorData>& inputs,
        std::map<std::string, TensorData>& outputs) = 0;

    virtual void DestroyNodeInstance(void* instance) = 0;

    // Optional
    virtual bool ValidateNode(
        const std::string& type_id,
        const std::map<std::string, std::string>& params,
        std::string& error_message) { return true; }

    virtual std::vector<std::string> GetParameterSuggestions(
        const std::string& type_id,
        const std::string& param_key,
        const std::map<std::string, std::string>& current_params) { return {}; }
};
```

### 4.4 IPanelProvider — Custom UI Panels

```cpp
struct PanelInfo {
    std::string id;
    std::string display_name;
    std::string icon;           // FontAwesome icon
    std::string category;       // Menu category (default: "Plugins")

    bool visible_by_default = false;
    bool allow_multiple = false;
    bool save_state = true;

    std::string shortcut;       // e.g., "Ctrl+Shift+M"
    float min_width = 200.0f, min_height = 150.0f;
    float default_width = 400.0f, default_height = 300.0f;
    std::string tooltip, help_url;
};

class IPanelProvider : public virtual IPlugin {
public:
    virtual std::vector<PanelInfo> GetPanels() const = 0;

    virtual void* CreatePanelInstance(const std::string& panel_id) = 0;
    virtual void RenderPanel(void* instance) = 0;  // Called every frame
    virtual void DestroyPanelInstance(void* instance) = 0;

    virtual void OnPanelVisibilityChanged(void* instance, bool visible) {}
    virtual void OnPanelResized(void* instance, float width, float height) {}

    virtual bool IsPanelVisible(const std::string& panel_id) const = 0;
    virtual void SetPanelVisible(const std::string& panel_id, bool visible) = 0;

    // State persistence
    virtual std::string SavePanelState(void* instance) { return "{}"; }
    virtual void RestorePanelState(void* instance, const std::string& state) {}
};
```

### 4.5 IDataProvider — Custom Data Loaders

```cpp
struct DataFormatInfo {
    std::string format_id;
    std::string display_name;
    std::string description;
    std::vector<std::string> file_extensions;   // ".parquet", ".pq"
    std::vector<std::string> mime_types;
    bool supports_streaming = false;
    bool supports_preview = true;
    bool supports_write = false;
    bool supports_schema = true;
};

struct PluginDatasetInfo {
    std::string name, source_path;
    size_t num_samples, num_classes;
    std::vector<size_t> sample_shape;
    std::vector<std::string> class_names;
    std::map<int, size_t> class_distribution;
    size_t estimated_memory_bytes, loaded_memory_bytes;
    std::map<std::string, std::string> metadata;

    struct ColumnInfo {
        std::string name, dtype;
        bool is_nullable;
    };
    std::vector<ColumnInfo> columns;  // For tabular data
};

struct DataSample {
    std::vector<float> data;
    std::vector<size_t> shape;
    int label = -1;
    std::map<std::string, std::vector<float>> extra_tensors;
    std::map<std::string, std::string> extra_strings;
    std::string sample_id, source_file;
};

struct DataLoadOptions {
    size_t max_samples = 0;
    size_t skip_samples = 0;
    bool normalize = false;
    bool shuffle = false;
    int random_seed = 42;
    bool lazy_load = true;
    size_t cache_size = 1000;
    std::map<std::string, std::string> options;  // Format-specific
};

class IDataProvider : public virtual IPlugin {
public:
    virtual std::vector<DataFormatInfo> GetSupportedFormats() const = 0;
    virtual bool CanLoad(const std::string& path) const = 0;

    virtual bool GetDatasetInfo(const std::string& path, PluginDatasetInfo& out) = 0;
    virtual void* OpenDataset(const std::string& path, const DataLoadOptions& opts) = 0;
    virtual size_t GetDatasetSize(void* handle) const = 0;
    virtual bool GetSample(void* handle, size_t index, DataSample& out) = 0;
    virtual void CloseDataset(void* handle) = 0;

    // Optional batch access
    virtual bool GetBatch(void* handle, const std::vector<size_t>& indices,
                         std::vector<DataSample>& out) {
        out.reserve(indices.size());
        for (auto idx : indices) {
            DataSample s;
            if (!GetSample(handle, idx, s)) return false;
            out.push_back(std::move(s));
        }
        return true;
    }

    // Optional streaming
    virtual void* CreateIterator(void* handle, size_t batch_size, bool shuffle) { return nullptr; }
    virtual bool GetNextBatch(void* iterator, std::vector<DataSample>& out) { return false; }
    virtual void DestroyIterator(void* iterator) {}

    // Optional write support
    virtual bool WriteDataset(const std::string& path, const std::string& format_id,
                             const std::vector<DataSample>& samples,
                             const PluginDatasetInfo& info) { return false; }
};
```

### 4.6 ITrainingHook — Training Lifecycle Callbacks

```cpp
struct TrainingConfigSnapshot {
    std::string model_name, model_type;
    std::vector<std::string> layer_names;
    size_t total_parameters, trainable_parameters;

    std::string dataset_name, dataset_path;
    size_t train_samples, val_samples, num_classes;
    std::vector<size_t> input_shape;

    int epochs, batch_size;
    std::string optimizer, loss_function;
    float learning_rate;
    std::map<std::string, std::string> hyperparameters;

    std::string device;          // "cuda:0", "cpu"
    std::string cyxwiz_version;
    std::string start_time;      // ISO 8601
    std::string run_id;
};

struct TrainingMetricsSnapshot {
    int epoch, batch, total_epochs, total_batches;
    int samples_processed;

    float train_loss, train_accuracy;
    std::map<std::string, float> train_metrics;
    float val_loss, val_accuracy;
    std::map<std::string, float> val_metrics;

    float learning_rate;
    float epoch_time_seconds, batch_time_ms, samples_per_second;
    float gpu_memory_used_mb, gpu_utilization_percent;
    float best_val_loss, best_val_accuracy;
    int best_epoch;
};

class ITrainingHook : public virtual IPlugin {
public:
    virtual void OnTrainingStart(const TrainingConfigSnapshot& config) = 0;
    virtual void OnTrainingEnd(const TrainingMetricsSnapshot& final_metrics,
                              bool success, const std::string& error_message) = 0;

    virtual void OnEpochStart(int epoch) {}
    virtual void OnEpochEnd(int epoch, const TrainingMetricsSnapshot& metrics) {}
    virtual void OnBatchStart(int epoch, int batch) {}
    virtual void OnBatchEnd(int epoch, int batch, float batch_loss) {}
    virtual void OnValidationEnd(int epoch, const TrainingMetricsSnapshot& metrics) {}
    virtual void OnCheckpointSaved(const std::string& path, const TrainingMetricsSnapshot& m) {}
    virtual void OnLearningRateChanged(float old_lr, float new_lr, const std::string& reason) {}

    // Logging
    virtual void LogMetric(const std::string& name, float value, int step) {}
    virtual void LogParam(const std::string& name, const std::string& value) {}
    virtual void LogArtifact(const std::string& name, const std::string& path,
                            const std::string& type) {}

    // Control
    virtual bool ShouldStopEarly() { return false; }
    virtual float GetSuggestedLearningRate(float current_lr, int epoch,
                                           const TrainingMetricsSnapshot& m) { return current_lr; }
};
```

### 4.7 IAnalyticsProvider — Data Quality & Analytics

```cpp
using AnalyticsValue = std::variant<
    double, std::string,
    std::vector<double>, std::vector<std::string>,
    std::map<std::string, double>, std::map<std::string, std::string>
>;

struct AnalyticsReport {
    std::string title, summary, timestamp;
    std::string tool_name, tool_version;
    std::map<std::string, AnalyticsValue> metrics;

    struct Issue {
        enum class Level { Info, Warning, Error, Critical };
        Level level;
        std::string code, message, suggestion;
        std::map<std::string, std::string> details;
    };
    std::vector<Issue> issues;

    struct Chart {
        std::string title, type;    // "histogram", "scatter", "bar"
        std::string data_json;
    };
    std::vector<Chart> charts;

    std::string html_report, json_report, markdown_report;
};

struct ExpectationResult {
    std::string expectation_type;   // "expect_column_values_to_not_be_null"
    std::string column;
    bool success;
    std::string message;
    size_t element_count, unexpected_count;
    double unexpected_percent;
    std::vector<std::string> unexpected_examples;
};

struct ExpectationSuite {
    std::string name, description;
    struct Expectation {
        std::string type, column;
        std::map<std::string, std::string> kwargs;
    };
    std::vector<Expectation> expectations;

    static ExpectationSuite FromJson(const std::string& json);
    std::string ToJson() const;
};

class IAnalyticsProvider : public virtual IPlugin {
public:
    virtual const char* GetToolName() const = 0;
    virtual std::vector<std::string> GetSupportedAnalyses() const { return {"profile"}; }

    virtual bool ProfileDataset(void* dataset_handle,
                                const std::map<std::string, std::string>& options,
                                std::function<void(float)> progress,
                                AnalyticsReport& out) = 0;

    virtual bool ValidateDataset(void* handle, const ExpectationSuite& suite,
                                 std::vector<ExpectationResult>& out) { return false; }

    virtual ExpectationSuite GenerateExpectations(void* handle) { return {}; }

    virtual bool CompareDatasets(void* a, void* b,
                                const std::map<std::string, std::string>& options,
                                AnalyticsReport& out) { return false; }

    virtual bool ExportReport(const AnalyticsReport& report,
                             const std::string& path, const std::string& format) { return false; }

    virtual std::vector<std::string> GetExportFormats() const { return {"json"}; }
};
```

---

## 5. PluginContext API

The `PluginContext` is the sole API surface exposed to plugins. Plugins never access `MainWindow`, `NodeEditor`, or other engine internals directly.

```cpp
class PluginContext {
public:
    // --- Identity ---
    virtual const std::string& GetPluginId() const = 0;
    virtual std::string GetPluginDataPath() const = 0;      // Persistent storage
    virtual std::string GetPluginResourcePath() const = 0;   // Read-only resources

    // --- Logging ---
    virtual void LogInfo(const std::string& msg) = 0;
    virtual void LogWarning(const std::string& msg) = 0;
    virtual void LogError(const std::string& msg) = 0;
    virtual void LogDebug(const std::string& msg) = 0;

    // --- UI Integration ---
    virtual void RegisterPanel(const std::string& name, const std::string& icon,
                              bool* visible_ptr, const std::string& category = "Plugins") = 0;
    virtual void UnregisterPanel(const std::string& name) = 0;
    virtual void AddMenuItem(const std::string& menu, const std::string& item,
                            std::function<void()> callback, const std::string& shortcut = "") = 0;
    virtual void RemoveMenuItem(const std::string& menu, const std::string& item) = 0;
    virtual void ShowNotification(const std::string& title, const std::string& message,
                                 const std::string& type = "info") = 0;

    // --- Node Graph ---
    virtual void RegisterNodeType(const NodeTypeInfo& info) = 0;
    virtual void UnregisterNodeType(const std::string& type_id) = 0;

    // --- Training ---
    virtual void RegisterTrainingHook(ITrainingHook* hook) = 0;
    virtual void UnregisterTrainingHook(ITrainingHook* hook) = 0;
    virtual bool IsTrainingActive() const = 0;

    // --- Data Access (Read-Only) ---
    virtual std::vector<std::string> GetLoadedDatasets() const = 0;

    // --- Async Tasks ---
    virtual uint64_t SubmitTask(const std::string& name,
        std::function<void(std::function<void(float, const std::string&)>)> task_func,
        std::function<void(bool, const std::string&)> completion_cb = nullptr) = 0;
    virtual bool CancelTask(uint64_t task_id) = 0;

    // --- Python ---
    virtual bool ExecutePython(const std::string& code,
                              std::string& out_result, std::string& out_error) = 0;
    virtual bool IsPythonAvailable() const = 0;

    // --- Settings ---
    virtual std::string GetSetting(const std::string& key,
                                  const std::string& default_value = "") const = 0;
    virtual void SetSetting(const std::string& key, const std::string& value) = 0;
    virtual std::map<std::string, std::string> GetAllSettings() const = 0;

    // --- Inter-Plugin Communication ---
    virtual bool SendMessage(const std::string& target_plugin_id,
                            const std::string& message_type, const std::string& payload) = 0;
    virtual void RegisterMessageHandler(const std::string& message_type,
        std::function<void(const std::string& from_plugin, const std::string& payload)> handler) = 0;

    // --- Resource Limits ---
    virtual size_t GetMemoryLimit() const = 0;
    virtual size_t GetMemoryUsage() const = 0;
};
```

---

## 6. Plugin Manifest Schema

Every plugin must include a `plugin.json` manifest in its root directory.

### Complete Example

```json
{
  "$schema": "https://cyxwiz.io/schemas/plugin-manifest-v1.json",

  "id": "com.example.my-plugin",
  "name": "My Plugin",
  "version": "1.2.3",
  "api_version": "1.0.0",

  "description": "What this plugin does",
  "author": "Your Name <email@example.com>",
  "license": "MIT",
  "homepage": "https://github.com/you/my-plugin",
  "repository": "https://github.com/you/my-plugin.git",

  "capabilities": [
    "ProvidesNodes",
    "ProvidesPanels",
    "ProvidesTraining",
    "RequiresPython",
    "RequiresNetwork"
  ],

  "dependencies": [
    { "id": "com.cyxwiz.python-integration", "version": ">=1.0.0" }
  ],

  "optional_dependencies": [
    { "id": "com.cyxwiz.gpu-compute", "version": ">=1.0.0",
      "reason": "Enables GPU-accelerated processing" }
  ],

  "platforms": {
    "windows": { "library": "my_plugin.dll", "min_os_version": "10.0" },
    "linux":   { "library": "libmy_plugin.so", "min_os_version": "5.0" },
    "macos":   { "library": "libmy_plugin.dylib", "min_os_version": "11.0" }
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
        "type": "string", "title": "API Endpoint",
        "default": "http://localhost:8080"
      },
      "auto_sync": {
        "type": "boolean", "title": "Auto Sync",
        "default": true
      }
    }
  },

  "signature": {
    "algorithm": "ed25519",
    "public_key": "MCowBQYDK2VwAyEA...",
    "signature": "base64-encoded-signature"
  }
}
```

### Field Reference

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `id` | string | Yes | Reverse-domain unique identifier |
| `name` | string | Yes | Human-readable display name |
| `version` | semver | Yes | Plugin version (MAJOR.MINOR.PATCH) |
| `api_version` | semver | Yes | CyxWiz Plugin API version required |
| `description` | string | Yes | Plugin description |
| `author` | string | Yes | Author name and/or email |
| `license` | string | No | SPDX license identifier |
| `homepage` | url | No | Plugin homepage |
| `repository` | url | No | Source code repository |
| `capabilities` | array | Yes | List of PluginCapability strings |
| `dependencies` | array | No | Required plugins with version constraints |
| `optional_dependencies` | array | No | Optional plugins with reason |
| `platforms` | object | Yes | Platform-specific library paths |
| `resources` | object | No | Resource directory paths |
| `permissions` | array | No | Required permissions |
| `settings_schema` | object | No | JSON Schema for plugin settings UI |
| `signature` | object | No | Cryptographic signature |

---

## 7. Plugin Discovery & Loading

### Discovery Order (first found wins)

```
1. <project>/plugins/           # Project-specific plugins
2. ~/.cyxwiz/plugins/           # User-installed plugins
3. <executable>/plugins/        # Bundled plugins
```

### Loading Flow

```cpp
// Pseudocode
void PluginManager::Initialize() {
    // 1. Discover manifests
    auto manifests = DiscoverPlugins(GetSearchPaths());

    // 2. Validate manifests
    manifests = FilterValid(manifests);  // API compatible, platform match

    // 3. Resolve dependencies
    auto load_order = ResolveDependencies(manifests);  // Kahn's toposort

    // 4. Load in order
    for (auto& id : load_order) {
        auto& manifest = FindManifest(id, manifests);

        // Platform-specific DLL loading
        #ifdef _WIN32
            HMODULE handle = LoadLibraryA(manifest.library_path.c_str());
        #else
            void* handle = dlopen(manifest.library_path.c_str(), RTLD_NOW);
        #endif

        // Resolve factory symbols
        auto create = (CreatePluginFunc)GetSymbol(handle, "CreatePlugin");
        auto destroy = (DestroyPluginFunc)GetSymbol(handle, "DestroyPlugin");
        auto version = (GetApiVersionFunc)GetSymbol(handle, "GetPluginApiVersion");

        // Verify API version
        if (!IsApiCompatible(version())) { UnloadLibrary(handle); continue; }

        // Create plugin instance
        IPlugin* plugin = create();

        // Check permissions
        if (!PermissionManager::Approve(manifest.permissions)) { destroy(plugin); continue; }

        // Initialize
        auto context = CreateContext(manifest);
        if (plugin->Initialize(context.get())) {
            plugin->OnActivate();
            RegisterPlugin(id, plugin, destroy, handle, std::move(context));
        }
    }
}
```

---

## 8. Dependency Resolution

### Kahn's Topological Sort

```cpp
std::vector<std::string> ResolveDependencies(
    const std::vector<PluginManifest>& manifests)
{
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

    std::queue<std::string> queue;
    for (const auto& [id, degree] : in_degree) {
        if (degree == 0) queue.push(id);
    }

    std::vector<std::string> result;
    while (!queue.empty()) {
        auto id = queue.front(); queue.pop();
        result.push_back(id);
        for (const auto& dependent : graph[id]) {
            if (--in_degree[dependent] == 0) queue.push(dependent);
        }
    }

    if (result.size() != manifests.size()) {
        throw PluginDependencyException("Circular dependency detected");
    }
    return result;
}
```

### Version Constraint Matching

| Constraint | Meaning | Example Match |
|------------|---------|---------------|
| `1.0.0` | Exact match | Only 1.0.0 |
| `>=1.0.0` | Greater or equal | 1.0.0, 1.5.0, 2.0.0 |
| `^1.0.0` | Compatible (same major) | 1.0.0 to <2.0.0 |
| `~1.0.0` | Patch updates only | 1.0.0 to <1.1.0 |
| `>=1.0.0 <2.0.0` | Range | 1.0.0 through 1.x.x |

---

## 9. Registration Systems

### PluginNodeRegistry

```cpp
class PluginNodeRegistry {
public:
    static PluginNodeRegistry& Instance();

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

### PluginPanelRegistry

```cpp
class PluginPanelRegistry {
public:
    static PluginPanelRegistry& Instance();

    void RegisterPanel(const std::string& plugin_id, const PanelInfo& info, IPanelProvider* provider);
    void UnregisterPlugin(const std::string& plugin_id);

    std::vector<PanelInfo> GetAllPanels() const;
    IPanelProvider* GetProvider(const std::string& panel_id) const;
};
```

### PluginTrainingHookManager

```cpp
class PluginTrainingHookManager {
public:
    static PluginTrainingHookManager& Instance();

    void RegisterHook(ITrainingHook* hook);
    void UnregisterHook(ITrainingHook* hook);

    // Called by TrainingManager at appropriate lifecycle points
    void NotifyTrainingStart(const TrainingConfigSnapshot& config);
    void NotifyEpochEnd(int epoch, const TrainingMetricsSnapshot& metrics);
    void NotifyTrainingEnd(const TrainingMetricsSnapshot& metrics, bool success, const std::string& error);

private:
    std::vector<ITrainingHook*> hooks_;
    mutable std::mutex mutex_;
};
```

### PluginDataLoaderRegistry

```cpp
class PluginDataLoaderRegistry {
public:
    static PluginDataLoaderRegistry& Instance();

    void RegisterLoader(const std::string& format_id, IDataProvider* provider);
    void UnregisterPlugin(const std::string& plugin_id);

    IDataProvider* FindLoaderForFile(const std::string& path) const;
    std::vector<DataFormatInfo> GetAllFormats() const;
};
```

### PluginAnalyticsRegistry

```cpp
class PluginAnalyticsRegistry {
public:
    static PluginAnalyticsRegistry& Instance();

    void RegisterProvider(const std::string& plugin_id, IAnalyticsProvider* provider);
    void UnregisterPlugin(const std::string& plugin_id);

    std::vector<std::string> GetAvailableTools() const;
    IAnalyticsProvider* GetProvider(const std::string& tool_name) const;
};
```

---

## 10. Engine Integration Points

This section describes how the plugin system connects to the existing CyxWiz Engine architecture.

### 10.1 PluginManager Singleton

Follows the same Meyer's Singleton pattern as `ProjectManager`, `DataRegistry`, `TrainingManager`, and `AsyncTaskManager`:

```cpp
class PluginManager {
public:
    static PluginManager& Instance();
    PluginManager(const PluginManager&) = delete;
    PluginManager& operator=(const PluginManager&) = delete;

    void Initialize();                // Called after singletons ready
    void Update(float delta_time);    // Called each frame in main loop
    void Shutdown();                  // Called before app cleanup

    // Plugin management
    bool LoadPlugin(const std::string& plugin_dir);
    bool UnloadPlugin(const std::string& plugin_id);
    bool EnablePlugin(const std::string& plugin_id);
    bool DisablePlugin(const std::string& plugin_id);

    // Query
    std::vector<PluginInfo> GetLoadedPlugins() const;
    PluginState GetPluginState(const std::string& plugin_id) const;

private:
    PluginManager() = default;
};
```

### 10.2 Application Lifecycle Hooks

Integration into `application.cpp` init/shutdown order:

```
Existing order:
1. GLFW init
2. OpenGL window
3. ImGui/ImPlot/ImNodes contexts
4. Singletons (on demand via Instance())
5. MainWindow (creates all panels)
6. ScriptingEngine
7. Networking

Plugin system hooks:
- After step 5: PluginManager::Instance().Initialize()
- In main loop:  PluginManager::Instance().Update(delta_time)
- Before cleanup: PluginManager::Instance().Shutdown()
```

### 10.3 MainWindow Panel Registration

Existing pattern — panels created via `std::make_unique` in MainWindow constructor:

```cpp
// MainWindow constructor (existing pattern)
dataset_panel_ = std::make_unique<DatasetPanel>();
training_plot_panel_ = std::make_unique<TrainingPlotPanel>();
// ...

// Plugin panels registered dynamically after Initialize():
// PluginPanelRegistry provides panel info + render callbacks
// MainWindow iterates registered plugin panels in Render()
```

### 10.4 NodeEditor Integration

The engine has a 170+ entry `NodeType` enum. Plugin nodes use a sentinel value to avoid modifying this enum:

```cpp
// Add to NodeType enum:
enum class NodeType {
    // ... existing 170+ types ...
    PluginCustom    // Sentinel — actual type resolved via string ID
};

// In NodeEditor::InitializeSearchableNodes():
for (const auto& info : PluginNodeRegistry::Instance().GetAllNodeTypes()) {
    SearchableNode node;
    node.type = NodeType::PluginCustom;
    node.plugin_type_id = info.type_id;  // String-based lookup
    node.name = info.display_name;
    node.category = "Plugin > " + info.category;
    all_searchable_nodes_.push_back(node);
}

// In NodeEditor::CreateNode():
if (type == NodeType::PluginCustom) {
    auto* provider = PluginNodeRegistry::Instance().GetProvider(plugin_type_id);
    // Use provider to create node instance
}
```

### 10.5 TrainingManager Hook Vectors

Existing callbacks in `TrainingManager`:

```cpp
// Existing (single callback each)
TrainingStartCallback on_training_start_;
TrainingEndCallback on_training_end_;
ProgressCallback on_progress_;
```

Plugin hooks are managed by `PluginTrainingHookManager` which is invoked alongside existing callbacks:

```cpp
// In TrainingManager::TrainingThreadFunc():
// Existing: if (on_training_start_) on_training_start_(desc);
// Added:    PluginTrainingHookManager::Instance().NotifyTrainingStart(config_snapshot);

// In epoch callback:
// Added: PluginTrainingHookManager::Instance().NotifyEpochEnd(epoch, metrics_snapshot);

// After training completes:
// Added: PluginTrainingHookManager::Instance().NotifyTrainingEnd(metrics, success, error);
```

### 10.6 DataRegistry Plugin Loader Registration

```cpp
// In DataRegistry, when loading a file with unknown extension:
auto* loader = PluginDataLoaderRegistry::Instance().FindLoaderForFile(path);
if (loader && loader->CanLoad(path)) {
    // Use plugin loader instead of built-in
}
```

### 10.7 Toolbar "Plugins" Menu

The toolbar uses callback setters. Add a "Plugins" menu section:

```cpp
// In ToolbarPanel::Render():
if (ImGui::BeginMenu("Plugins")) {
    if (ImGui::MenuItem("Plugin Manager...")) {
        // Open PluginManagerPanel
    }
    ImGui::Separator();
    // List plugin-registered menu items
    ImGui::EndMenu();
}
```

### 10.8 ScriptingEngine for Python Plugins

Python plugins are loaded via `PythonPluginAdapter` which wraps Python classes using the existing `ScriptingEngine`:

```cpp
// ScriptingEngine provides:
ExecutionResult ExecuteScript(const std::string& script);
ExecutionResult ExecuteCommand(const std::string& command);
void ExecuteScriptAsync(const std::string& script);
bool IsScriptRunning() const;
```

---

## 11. Security Model

### Permission System

| Permission | Description |
|------------|-------------|
| `filesystem:read` | Read files outside plugin directory |
| `filesystem:write` | Write files outside plugin directory |
| `filesystem:plugin_data` | Read/write plugin's own data directory |
| `network:read` | Make outbound HTTP/gRPC requests |
| `network:listen` | Listen on network ports |
| `python:execute` | Execute Python code |
| `gpu:compute` | Use GPU for computation |
| `training:modify` | Modify training parameters |
| `data:full_access` | Full access to loaded datasets |

### Plugin Signing

- **Algorithm**: Ed25519
- **Signed content**: SHA-256 hash of manifest (excluding `signature` block)
- **Verification**: Against list of trusted public keys stored in engine settings
- **Unsigned plugins**: Warning dialog shown, user can allow or deny

### Crash Isolation (SafeExecute)

```cpp
template<typename F>
static std::optional<std::string> SafeExecute(
    const std::string& plugin_id, F&& func)
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

// Usage — wraps every plugin callback:
auto error = SafeExecute(plugin->GetId(), [&]() {
    plugin->Update(delta_time);
});
if (error) {
    spdlog::error(*error);
    DisablePlugin(plugin->GetId());  // Auto-disable on crash
}
```

---

## 12. Python Plugin Support

### PythonPluginAdapter

Wraps a Python plugin class in the `IPlugin` interface using the existing `ScriptingEngine`:

```cpp
class PythonPluginAdapter : public IPlugin {
public:
    PythonPluginAdapter(const std::string& script_path, const PluginManifest& manifest);

    const char* GetId() const override { return manifest_.id.c_str(); }
    const char* GetName() const override { return manifest_.name.c_str(); }

    bool Initialize(PluginContext* context) override {
        // Execute Python script to load plugin class
        std::string code = "import importlib.util\n"
            "spec = importlib.util.spec_from_file_location('plugin', '" + script_path_ + "')\n"
            "mod = importlib.util.module_from_spec(spec)\n"
            "spec.loader.exec_module(mod)\n"
            "plugin = mod.create_plugin()\n"
            "plugin.initialize(context)\n";
        std::string result, error;
        return context->ExecutePython(code, result, error);
    }

    // ... other IPlugin methods delegate to Python object
};
```

### Python Plugin Author API

```python
# my_plugin.py
class MyPlugin:
    def initialize(self, context):
        self.context = context
        context.log_info("MyPlugin initialized!")

    def on_activate(self):
        pass

    def update(self, dt):
        pass

    def shutdown(self):
        pass

def create_plugin():
    return MyPlugin()
```

### Detection

The `PluginLoader` detects Python plugins via the manifest:
- If `capabilities` includes `RequiresPython` and no native library is specified for the current platform
- Or if a `"type": "python"` field is present in the manifest

---

## 13. Plugin Manager UI

ImGui panel for managing plugins:

```
+------------------------------------------------------------------+
| Plugin Manager                                            [x]    |
+------------------------------------------------------------------+
| [Search plugins...]                          [Install Plugin...] |
+----------+-------------------------------------------------------+
|          | MLflow Integration              v1.2.0    [Enabled ▼]  |
| Installed| Experiment tracking and metrics logging                |
|          | Permissions: network:read, training:modify              |
|          +-------------------------------------------------------+
|          | Great Expectations              v2.0.1    [Disabled ▼] |
| Updates  | Data quality validation                                |
|          | Requires: Python                                       |
|          +-------------------------------------------------------+
|          | Custom Attention Nodes          v0.5.0    [Enabled ▼]  |
| Store    | Sparse and linear attention node types                 |
|          | No special permissions required                        |
+----------+-------------------------------------------------------+
| Status: 2 plugins active, 1 disabled | Memory: 12 MB             |
+------------------------------------------------------------------+
```

Features:
- Plugin list with enable/disable toggles
- Permission display per plugin
- Error log viewer for failed plugins
- Install from directory button
- Search/filter plugins
- Plugin details (version, author, description, dependencies)

---

## 14. Versioning & Compatibility

### Semantic Versioning Policy

| Change Type | Version Bump | Example |
|-------------|-------------|---------|
| Breaking API change | MAJOR | 1.x.x -> 2.0.0 |
| New feature (backward-compatible) | MINOR | 1.0.x -> 1.1.0 |
| Bug fix | PATCH | 1.0.0 -> 1.0.1 |

### API Compatibility Rules

```cpp
bool IsApiCompatible(const PluginVersion& plugin_api) {
    // Major version must match exactly
    if (plugin_api.major != CYXWIZ_PLUGIN_API_VERSION_MAJOR) return false;
    // Plugin minor must be <= host minor (older plugins work with newer hosts)
    if (plugin_api.minor > CYXWIZ_PLUGIN_API_VERSION_MINOR) return false;
    return true;
}
```

### Deprecation Policy

1. Feature marked `[[deprecated]]` in release N
2. Warning shown when plugin uses deprecated feature
3. Feature removed in release N+2 (minimum)
4. Migration guide provided in release notes

---

## 15. Example Plugins

### 15.1 MLflow Integration

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
        for (const auto& [k, v] : config.hyperparameters)
            mlflow_.LogParam(k, v);
    }

    void OnEpochEnd(int epoch, const TrainingMetricsSnapshot& m) override {
        mlflow_.LogMetric("train_loss", m.train_loss, epoch);
        mlflow_.LogMetric("val_accuracy", m.val_accuracy, epoch);
    }

    void OnTrainingEnd(const TrainingMetricsSnapshot& m, bool success,
                      const std::string& error) override {
        mlflow_.EndRun(success ? "FINISHED" : "FAILED");
    }

    void Shutdown() override { context_ = nullptr; }

private:
    PluginContext* context_ = nullptr;
    std::string tracking_uri_, run_id_;
    MLflowClient mlflow_;
};

CYXWIZ_PLUGIN(MLflowPlugin)
```

### 15.2 Custom Attention Node

```cpp
class SparseAttentionPlugin : public IPlugin, public INodeProvider {
public:
    const char* GetId() const override { return "com.example.sparse-attention"; }
    const char* GetName() const override { return "Sparse Attention Nodes"; }

    PluginCapability GetCapabilities() const override {
        return PluginCapability::ProvidesNodes;
    }

    std::vector<NodeTypeInfo> GetNodeTypes() const override {
        return {{
            .type_id = "attention.sparse",
            .display_name = "Sparse Attention",
            .category = "Attention",
            .inputs = {
                {"query", PinType::Tensor, true, true},
                {"key",   PinType::Tensor, true, true},
                {"value", PinType::Tensor, true, true}
            },
            .outputs = {{"output", PinType::Tensor, false}},
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

    bool Initialize(PluginContext* ctx) override { return true; }
    void Shutdown() override {}
};

CYXWIZ_PLUGIN(SparseAttentionPlugin)
```

### 15.3 Great Expectations (Python)

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
        std::string code = R"(
import great_expectations as ge
import json
# ... profiling logic ...
result = json.dumps(profile_result)
        )";
        std::string result, error;
        if (context_->ExecutePython(code, result, error)) {
            report = ParseReport(result);
            return true;
        }
        return false;
    }

    bool Initialize(PluginContext* ctx) override { context_ = ctx; return true; }
    void Shutdown() override {}

    // IPanelProvider methods omitted for brevity
private:
    PluginContext* context_ = nullptr;
};

CYXWIZ_PLUGIN(GreatExpectationsPlugin)
```

---

## 16. Implementation Roadmap

### Phase 1: Core Infrastructure

| Task | Files |
|------|-------|
| Define all 6 interfaces + common types | `src/core/plugin/plugin_interface.h` |
| Implement PluginContext | `src/core/plugin/plugin_context.h` |
| Implement manifest parsing | `src/core/plugin/plugin_manifest.h/cpp` |
| Implement DLL loader | `src/core/plugin/plugin_loader.h/cpp` |
| Implement PluginManager singleton | `src/core/plugin/plugin_manager.h/cpp` |
| Update CMakeLists.txt | `cyxwiz-engine/CMakeLists.txt` |
| Wire Initialize/Update/Shutdown | `src/application.cpp` |

### Phase 2: Registration Systems & Engine Integration

| Task | Files |
|------|-------|
| Implement all 5 registries | `src/core/plugin/plugin_registry.h/cpp` |
| Integrate with NodeEditor | `src/gui/node_editor.h/cpp` |
| Integrate with MainWindow | `src/gui/main_window.h/cpp` |
| Integrate with TrainingManager | `src/core/training_manager.h/cpp` |
| Integrate with DataRegistry | `src/core/data_registry.h/cpp` |
| Add Plugins menu to toolbar | `src/gui/panels/toolbar.h/cpp` |

### Phase 3: Security & Permissions

| Task | Files |
|------|-------|
| Permission system | `src/core/plugin/plugin_security.h/cpp` |
| SafeExecute crash isolation | (in plugin_security) |
| Ed25519 signature verification | (in plugin_security) |

### Phase 4: Python Plugin Support

| Task | Files |
|------|-------|
| PythonPluginAdapter | `src/core/plugin/python_plugin_adapter.h/cpp` |
| Python detection in PluginLoader | `src/core/plugin/plugin_loader.cpp` |

### Phase 5: Plugin Manager UI

| Task | Files |
|------|-------|
| ImGui panel | `src/gui/panels/plugin_manager_panel.h/cpp` |
| Register in MainWindow + toolbar | `src/gui/main_window.cpp`, `toolbar.cpp` |

### Phase 6: Examples & Documentation

| Task | Output |
|------|--------|
| MLflow plugin example | `plugins/examples/mlflow_plugin/` |
| Python plugin example | `plugins/examples/data_validation_plugin/` |
| Developer guide | `docs/plugin_developer_guide.md` |

---

## 17. File Map

### Files to Create

| File | Purpose |
|------|---------|
| `src/core/plugin/plugin_interface.h` | All 6 interfaces, enums, structs, export macros |
| `src/core/plugin/plugin_context.h` | PluginContext abstract class |
| `src/core/plugin/plugin_manifest.h` | PluginManifest struct |
| `src/core/plugin/plugin_manifest.cpp` | JSON parsing with nlohmann::json |
| `src/core/plugin/plugin_loader.h` | NativePluginLoader, PythonPluginLoader |
| `src/core/plugin/plugin_loader.cpp` | LoadLibrary/dlopen, symbol resolution |
| `src/core/plugin/plugin_registry.h` | All 5 registry classes |
| `src/core/plugin/plugin_registry.cpp` | Registry implementations |
| `src/core/plugin/plugin_manager.h` | PluginManager singleton |
| `src/core/plugin/plugin_manager.cpp` | Discovery, dependency resolution, lifecycle |
| `src/core/plugin/plugin_security.h` | PermissionManager, SafeExecute |
| `src/core/plugin/plugin_security.cpp` | Permission checking, Ed25519 verification |
| `src/core/plugin/python_plugin_adapter.h` | PythonPluginAdapter class |
| `src/core/plugin/python_plugin_adapter.cpp` | Python wrapping via ScriptingEngine |
| `src/gui/panels/plugin_manager_panel.h` | Plugin Manager ImGui panel |
| `src/gui/panels/plugin_manager_panel.cpp` | Plugin list, enable/disable, install UI |

### Files to Modify

| File | Changes |
|------|---------|
| `cyxwiz-engine/CMakeLists.txt` | Add all `src/core/plugin/*.cpp` and `plugin_manager_panel.cpp` |
| `src/application.cpp` | Call Initialize/Update/Shutdown on PluginManager |
| `src/gui/main_window.h/cpp` | Add plugin_manager_panel_, render plugin panels |
| `src/gui/node_editor.h/cpp` | Add NodeType::PluginCustom, PluginNodeRegistry fallback |
| `src/core/training_manager.h/cpp` | Add PluginTrainingHookManager notifications |
| `src/core/data_registry.h/cpp` | Add PluginDataLoaderRegistry fallback |
| `src/gui/panels/toolbar.h/cpp` | Add "Plugins" menu section |
