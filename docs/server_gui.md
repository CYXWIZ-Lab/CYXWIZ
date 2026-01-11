# CyxWiz Server Node GUI Design Document

## Executive Summary

This document outlines the comprehensive design for adding dual-mode user interfaces to the CyxWiz Server Node. The Server Node currently operates as a CLI-only service, but we are expanding it to support both a terminal user interface (TUI) inspired by btop and a graphical user interface (GUI) inspired by LM Studio. This enables both power users and casual miners to participate in the decentralized ML compute network.

**Target Users:**
1. **Training Providers** - Users contributing GPU/CPU power for distributed ML training
2. **Model Deployers** - Users hosting models with API endpoints for inference
3. **Pool Miners** - Low-power PC users joining pools for passive CyxWiz coin rewards

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Technology Choices](#2-technology-choices)
3. [UI/UX Design](#3-uiux-design)
4. [Component Design](#4-component-design)
5. [API Design](#5-api-design)
6. [Model Deployment Flow](#6-model-deployment-flow)
7. [Marketplace Integration](#7-marketplace-integration)
8. [Security Considerations](#8-security-considerations)
9. [Implementation Roadmap](#9-implementation-roadmap)

---

## 1. Architecture Overview

### 1.1 High-Level Architecture

```
                            CyxWiz Server Node
                    +-------------------------------+
                    |                               |
                    |   +-------------------------+ |
                    |   |    Interface Manager    | |
                    |   |  (Mode Selection/Routing)| |
                    |   +-------------------------+ |
                    |          |         |          |
                    |   +------+         +------+   |
                    |   |                      |   |
                    |   v                      v   |
                    | +-------+          +-------+ |
                    | |  TUI  |          |  GUI  | |
                    | | (btop)| <------> |(LM    | |
                    | |       |   Shared |Studio)| |
                    | +-------+   State  +-------+ |
                    |      |               |       |
                    |      +-------+-------+       |
                    |              |               |
                    |   +----------v----------+    |
                    |   |   Core Services     |    |
                    |   |  (Backend Manager)  |    |
                    |   +----------+----------+    |
                    |              |               |
                    +-------------------------------+
                               |
        +----------------------+----------------------+
        |                      |                      |
        v                      v                      v
+---------------+    +------------------+    +----------------+
|  JobExecutor  |    | DeploymentManager|    |  NodeClient    |
| (ML Training) |    | (Model Serving)  |    | (Central Svr)  |
+---------------+    +------------------+    +----------------+
```

### 1.2 Mode Selection Strategy

The Server Node supports three operational modes, selectable at startup:

```cpp
enum class InterfaceMode {
    Headless,   // No UI, pure service mode (default for servers)
    TUI,        // Terminal UI (btop-style)
    GUI         // Desktop GUI (LM Studio-style)
};
```

**Mode Detection Logic:**
1. Command-line argument: `--mode=[headless|tui|gui]`
2. Config file preference: `config.yaml -> interface.mode`
3. Auto-detection: If `$DISPLAY`/`$WAYLAND_DISPLAY` is set, prefer GUI; otherwise TUI
4. Fallback: Headless mode for servers without display capability

### 1.3 Component Coexistence

Both TUI and GUI share the same underlying services through a `BackendManager` singleton:

```cpp
namespace cyxwiz::servernode {

class BackendManager {
public:
    static BackendManager& Instance();

    // Core services (shared between TUI/GUI)
    JobExecutor* GetJobExecutor();
    DeploymentManager* GetDeploymentManager();
    NodeClient* GetNodeClient();
    MetricsCollector* GetMetricsCollector();

    // State observers (for reactive UI updates)
    void AddObserver(BackendObserver* observer);
    void RemoveObserver(BackendObserver* observer);

    // Configuration
    const NodeConfig& GetConfig() const;
    void SaveConfig(const NodeConfig& config);

private:
    std::shared_ptr<JobExecutor> job_executor_;
    std::shared_ptr<DeploymentManager> deployment_manager_;
    std::shared_ptr<NodeClient> node_client_;
    std::shared_ptr<MetricsCollector> metrics_collector_;
    std::vector<BackendObserver*> observers_;
};

} // namespace cyxwiz::servernode
```

### 1.4 Process Model Options

**Option A: Single Process (Recommended)**
```
cyxwiz-server-node --mode=gui
    |
    +-- Main thread: GUI event loop (ImGui)
    +-- Worker threads: Job execution, gRPC services
    +-- Watchdog thread: Health monitoring
```

**Option B: Dual Process (Advanced)**
```
cyxwiz-server-daemon (headless service)
    |
    +-- gRPC server on localhost:50054 (IPC)

cyxwiz-server-gui (GUI client)
    |
    +-- Connects to daemon via gRPC
    +-- Can run on different machine (remote management)
```

We recommend **Option A** for simplicity, with Option B as a future enhancement for remote management scenarios.

---

## 2. Technology Choices

### 2.1 GUI Framework: ImGui (Recommended)

**Why ImGui:**
1. **Consistency** - Already used in cyxwiz-engine; shared codebase and theming
2. **Performance** - Immediate-mode rendering is lightweight and GPU-efficient
3. **Cross-platform** - Works on Windows, macOS, Linux with minimal changes
4. **Integration** - Easy to embed in existing C++ applications
5. **Customization** - Full control over rendering and behavior

**Reusable Components from cyxwiz-engine:**
- `Theme` system (`gui/theme.h`) - All theme presets
- `Panel` base class (`gui/panel.h`) - Standard panel interface
- `DockStyle` (`gui/dock_style.h`) - Docking configuration
- `IconsFontAwesome6` (`gui/IconsFontAwesome6.h`) - Icon definitions
- `PlotManager` integration with ImPlot for real-time charts

**Alternative Considered: Qt**
- Pros: More mature, better native look
- Cons: Large dependency, licensing complexity, inconsistent with Engine

**Alternative Considered: Electron/React**
- Pros: Web tech familiarity, rich ecosystem
- Cons: Memory overhead, performance concerns for real-time metrics

### 2.2 TUI Framework: FTXUI (Recommended)

**Why FTXUI:**
1. **Modern C++** - Uses C++17/20 features, clean API
2. **Component-based** - Similar mental model to ImGui (immediate-mode)
3. **No curses dependency** - Pure C++ with minimal dependencies
4. **Rich widgets** - Gauges, graphs, flexbox layout
5. **Cross-platform** - Works on all terminals

**Installation via vcpkg:**
```bash
vcpkg install ftxui
```

**Alternative Considered: btop library extraction**
- Pros: Exact btop appearance
- Cons: Tightly coupled to btop, difficult to extract

**Alternative Considered: ncurses + CDK**
- Pros: Widely available, mature
- Cons: C API, complex, cross-platform issues on Windows

### 2.3 Shared Rendering Components

Create abstraction layer for metrics visualization:

```cpp
namespace cyxwiz::servernode::ui {

// Abstract chart data provider
class ChartDataSource {
public:
    virtual ~ChartDataSource() = default;
    virtual std::vector<float> GetData(int count) const = 0;
    virtual float GetMin() const = 0;
    virtual float GetMax() const = 0;
    virtual std::string GetLabel() const = 0;
};

// Used by both TUI and GUI
class CPUUsageSource : public ChartDataSource { /* ... */ };
class GPUUsageSource : public ChartDataSource { /* ... */ };
class MemoryUsageSource : public ChartDataSource { /* ... */ };
class NetworkThroughputSource : public ChartDataSource { /* ... */ };

} // namespace cyxwiz::servernode::ui
```

---

## 3. UI/UX Design

### 3.1 TUI Mode (btop-inspired)

```
+------------------------------------------------------------------------------+
|                      CyxWiz Server Node v0.3.0                               |
|                  Node: node_1733580000 | Central: Connected                  |
+------------------------------------------------------------------------------+
| CPU [|||||||||||||||||||         ] 72.3%  | GPU 0 [||||||||||||||||||   ] 85%|
| MEM [||||||||||                  ] 45.2%  | VRAM  [||||||||||||||||||||| ] 92%|
+------------------------------------------------------------------------------+
|                           ACTIVE JOBS (2)                                    |
+------------------------------------------------------------------------------+
| ID          | Type     | Progress | ETA     | Epoch    | Loss    | Client    |
|-------------|----------|----------|---------|----------|---------|-----------|
| job_abc123  | Training | [====   ]| 2h 15m  | 45/100   | 0.0234  | Engine001 |
| job_def456  | Training | [=======]| 12m     | 89/100   | 0.0089  | Engine042 |
+------------------------------------------------------------------------------+
|                         DEPLOYED MODELS (3)                                  |
+------------------------------------------------------------------------------+
| Model              | Format | Status  | Requests | Latency | Port  | Memory  |
|--------------------|--------|---------|----------|---------|-------|---------|
| llama-2-7b-chat    | GGUF   | Running | 1,234    | 45ms    | 8080  | 6.2 GB  |
| whisper-large-v3   | ONNX   | Running | 892      | 120ms   | 8081  | 3.1 GB  |
| sdxl-turbo         | SafeT  | Stopped | 0        | -       | -     | 0 B     |
+------------------------------------------------------------------------------+
|                            EARNINGS TODAY                                    |
+------------------------------------------------------------------------------+
| Training: 12.45 CYXWIZ | Inference: 8.23 CYXWIZ | Pool: 0.12 CYXWIZ          |
| Total: 20.80 CYXWIZ (~$4.16 USD)                                             |
+------------------------------------------------------------------------------+
| [Q]uit | [S]ettings | [D]eploy Model | [J]obs | [L]ogs | [H]elp              |
+------------------------------------------------------------------------------+
```

**Key TUI Features:**
1. **Compact Information Density** - All critical metrics visible at once
2. **Color Coding** - Green (healthy), Yellow (warning), Red (critical)
3. **Real-time Updates** - 1-second refresh for metrics, 100ms for active jobs
4. **Keyboard Navigation** - Single-key shortcuts for common actions
5. **Responsive Layout** - Adapts to terminal size (min 80x24)

### 3.2 GUI Mode (LM Studio-inspired)

**Main Window Layout:**

```
+-----------------------------------------------------------------------------------+
|  File  Settings  Help                                         [_][O][X]           |
+-----------------------------------------------------------------------------------+
|  +----------------+  +----------------------------------------------------------+ |
|  |                |  |                                                          | |
|  |   [Dashboard]  |  |  Dashboard                                               | |
|  |   [Models]     |  |  +----------------------+  +---------------------------+ | |
|  |   [Jobs]       |  |  | System Resources     |  | Earnings                   | | |
|  |   [Deploy]     |  |  | CPU [======    ] 60% |  | Today:    20.80 CYXWIZ    | | |
|  |   [API Keys]   |  |  | GPU [=========] 92%  |  | This Week: 145.2 CYXWIZ   | | |
|  |   [Marketplace]|  |  | RAM [====     ] 40%  |  | This Month: 892.5 CYXWIZ  | | |
|  |   [Pool Mining]|  |  | VRAM[========= ] 87% |  +---------------------------+ | |
|  |   [Logs]       |  |  +----------------------+                                | |
|  |   [Settings]   |  |                                                          | |
|  |                |  |  +------------------------------------------------------+| |
|  |                |  |  | Active Jobs                                          || |
|  |                |  |  | +--------------------------------------------------+ || |
|  |                |  |  | | job_abc123 | Training | 45% | Epoch 45/100       | || |
|  |                |  |  | | [===================>                          ] | || |
|  |                |  |  | +--------------------------------------------------+ || |
|  |                |  |  +------------------------------------------------------+| |
|  |                |  |                                                          | |
|  | [Connection]   |  |  +------------------------------------------------------+| |
|  | * Connected    |  |  | Deployed Models                                      || |
|  |                |  |  | llama-2-7b    GGUF    Running    localhost:8080      || |
|  | [Wallet]       |  |  | whisper-v3    ONNX    Running    localhost:8081      || |
|  | 5FL3...8dK2    |  |  +------------------------------------------------------+| |
|  +----------------+  +----------------------------------------------------------+ |
+-----------------------------------------------------------------------------------+
|  Status: Ready | Jobs: 2 active | Models: 2 deployed | Network: 125 MB/s          |
+-----------------------------------------------------------------------------------+
```

**GUI Views:**

#### 3.2.1 Dashboard View
- System resource gauges (CPU, GPU, RAM, VRAM)
- Active job summary cards
- Deployed model summary
- Earnings overview
- Network status indicator

#### 3.2.2 Models View
```
+-----------------------------------------------------------------------------------+
|  Models                                                          [Search...] [+]  |
+-----------------------------------------------------------------------------------+
|  [Local Models]  [Downloaded]  [Marketplace]                                      |
+-----------------------------------------------------------------------------------+
|  +-----------------------------------------------------------------------------+ |
|  | llama-2-7b-chat.gguf                                          [Deploy] [X] | |
|  | Size: 6.2 GB | Format: GGUF | Quantization: Q4_K_M                         | |
|  | Downloaded: Nov 15, 2024 | Source: HuggingFace                             | |
|  +-----------------------------------------------------------------------------+ |
|  | mistral-7b-instruct-v0.2.gguf                                 [Deploy] [X] | |
|  | Size: 7.1 GB | Format: GGUF | Quantization: Q5_K_M                         | |
|  | Downloaded: Nov 20, 2024 | Source: CyxWiz Hub                              | |
|  +-----------------------------------------------------------------------------+ |
+-----------------------------------------------------------------------------------+
```

#### 3.2.3 Deploy View
```
+-----------------------------------------------------------------------------------+
|  Deploy Model                                                                     |
+-----------------------------------------------------------------------------------+
|  Model: [llama-2-7b-chat.gguf              v]                                     |
|                                                                                   |
|  Deployment Type:                                                                 |
|  (*) Local (serve on this machine)                                                |
|  ( ) Network (serve on CyxWiz network for rewards)                                |
|                                                                                   |
|  Server Configuration:                                                            |
|  +-------------------------------------------------------------------------+     |
|  | Host:      [0.0.0.0          ]  Port: [8080    ]                        |     |
|  | Context:   [4096             ]  Threads: [8    ]                        |     |
|  | GPU Layers:[35               ]  Batch Size: [512]                       |     |
|  +-------------------------------------------------------------------------+     |
|                                                                                   |
|  API Configuration:                                                               |
|  [x] Enable OpenAI-compatible API                                                 |
|  [x] Enable streaming responses                                                   |
|  [ ] Require API key authentication                                               |
|                                                                                   |
|  Estimated Resources:                                                             |
|  VRAM: ~5.8 GB | RAM: ~1.2 GB | CPU: 4 threads minimum                           |
|                                                                                   |
|  [Cancel]                                                     [Deploy Model]      |
+-----------------------------------------------------------------------------------+
```

#### 3.2.4 API Keys View
```
+-----------------------------------------------------------------------------------+
|  API Keys Management                                             [+ Generate Key] |
+-----------------------------------------------------------------------------------+
|  Your API keys grant access to your deployed models.                              |
|                                                                                   |
|  +-----------------------------------------------------------------------------+ |
|  | Key Name     | Created      | Last Used    | Requests | Rate Limit | Status | |
|  |--------------|--------------|--------------|----------|------------|--------| |
|  | prod-key-1   | Nov 10, 2024 | 2 min ago    | 12,456   | 100/min    | Active | |
|  |              | cyx_sk_live_Ax8d...F3k2                              [Revoke]| |
|  +-----------------------------------------------------------------------------+ |
|  | test-key-1   | Nov 15, 2024 | 1 hour ago   | 234      | 10/min     | Active | |
|  |              | cyx_sk_test_Bx9e...G4l3                              [Revoke]| |
|  +-----------------------------------------------------------------------------+ |
|                                                                                   |
|  Endpoint Information:                                                            |
|  Base URL: http://your-ip:8080/v1                                                 |
|  Example: curl -H "Authorization: Bearer cyx_sk_live_..." http://...              |
+-----------------------------------------------------------------------------------+
```

#### 3.2.5 Pool Mining View
```
+-----------------------------------------------------------------------------------+
|  Pool Mining                                                                      |
+-----------------------------------------------------------------------------------+
|  Join a mining pool to earn CYXWIZ rewards with low-power hardware.               |
|                                                                                   |
|  Current Pool: [CyxWiz Official Pool          v]                                  |
|  Pool Address: pool.cyxwiz.io:3333                                                |
|  Pool Fee: 1%                                                                     |
|                                                                                   |
|  +-------------------------------------------------------------------------+     |
|  | Your Stats                                                              |     |
|  | Hashrate: 125 MH/s | Shares: 1,234 | Rejected: 12 (0.97%)              |     |
|  | Est. Daily: 0.45 CYXWIZ | Est. Monthly: 13.5 CYXWIZ                     |     |
|  +-------------------------------------------------------------------------+     |
|                                                                                   |
|  [x] Auto-start mining on startup                                                 |
|  [x] Mine only when GPU is idle                                                   |
|  [ ] Low-power mode (reduce hashrate by 50%)                                      |
|                                                                                   |
|  Intensity: [==========|         ] 50%                                            |
|                                                                                   |
|  [Stop Mining]                                          Mining for 2h 34m         |
+-----------------------------------------------------------------------------------+
```

### 3.3 User Journeys

#### Journey 1: First-Time Training Provider
```
1. Install CyxWiz Server Node
2. Launch in GUI mode
3. Connect wallet (Solana) via QR code or paste address
4. Register with Central Server (automatic)
5. Configure GPU allocation (slider: 0-100%)
6. Wait for job assignments
7. Monitor earnings in Dashboard
```

#### Journey 2: Model Deployment for Inference
```
1. Go to Models view
2. Search/browse marketplace for model
3. Download model (progress indicator)
4. Click "Deploy"
5. Configure: port, API key requirement, rate limits
6. Deploy locally or to network
7. Copy API endpoint URL
8. Use in ChatCyxWiz/agents/third-party apps
```

#### Journey 3: Casual Pool Miner
```
1. Launch in GUI mode
2. Go to Pool Mining view
3. Select pool from dropdown
4. Configure intensity (low for laptop, high for desktop)
5. Click "Start Mining"
6. Minimize to system tray
7. Check earnings periodically
```

---

## 4. Component Design

### 4.1 GUI Panel Architecture

Following the pattern established in cyxwiz-engine:

```cpp
// cyxwiz-server-node/src/gui/server_panel.h
namespace cyxwiz::servernode::gui {

class ServerPanel : public cyxwiz::Panel {
public:
    ServerPanel(const std::string& name, bool visible = true)
        : Panel(name, visible), backend_(BackendManager::Instance()) {}

protected:
    BackendManager& backend_;
};

} // namespace cyxwiz::servernode::gui
```

**Panel Implementations:**

```cpp
// Dashboard Panel
class DashboardPanel : public ServerPanel {
public:
    DashboardPanel();
    void Render() override;

private:
    void RenderSystemResources();
    void RenderActiveJobs();
    void RenderDeployedModels();
    void RenderEarnings();

    // Cached data for smooth rendering
    struct SystemMetrics {
        float cpu_usage;
        float gpu_usage;
        float ram_usage;
        float vram_usage;
        float network_in;
        float network_out;
    } cached_metrics_;

    std::chrono::steady_clock::time_point last_update_;
};

// Model Browser Panel
class ModelBrowserPanel : public ServerPanel {
public:
    void Render() override;

private:
    void RenderLocalModels();
    void RenderDownloadedModels();
    void RenderMarketplace();
    void RenderModelCard(const ModelInfo& model);
    void DownloadModel(const std::string& model_id);

    std::string search_query_;
    int selected_tab_;
    std::vector<ModelInfo> local_models_;
    std::vector<ModelInfo> marketplace_models_;
    bool is_loading_;
};

// Job Monitor Panel
class JobMonitorPanel : public ServerPanel {
public:
    void Render() override;

private:
    void RenderJobCard(const JobState& job);
    void RenderJobDetails(const std::string& job_id);
    void RenderTrainingChart(const std::string& job_id);

    std::string selected_job_id_;
    std::map<std::string, std::vector<float>> loss_history_;
};

// Deployment Panel
class DeploymentPanel : public ServerPanel {
public:
    void Render() override;

private:
    void RenderModelSelector();
    void RenderDeploymentConfig();
    void RenderResourceEstimate();
    void DeployModel();

    DeploymentConfig current_config_;
    std::string selected_model_;
};

// API Keys Panel
class APIKeysPanel : public ServerPanel {
public:
    void Render() override;

private:
    void RenderKeyList();
    void RenderGenerateDialog();
    void GenerateKey(const std::string& name);
    void RevokeKey(const std::string& key_id);

    std::vector<APIKey> api_keys_;
    bool show_generate_dialog_;
    char new_key_name_[64];
};

// Pool Mining Panel
class PoolMiningPanel : public ServerPanel {
public:
    void Render() override;

private:
    void RenderPoolSelector();
    void RenderMiningStats();
    void RenderMiningConfig();
    void StartMining();
    void StopMining();

    bool is_mining_;
    float mining_intensity_;
    PoolMiningStats stats_;
};

// Settings Panel
class SettingsPanel : public ServerPanel {
public:
    void Render() override;

private:
    void RenderGeneralSettings();
    void RenderNetworkSettings();
    void RenderGPUSettings();
    void RenderWalletSettings();
    void RenderAdvancedSettings();

    NodeConfig config_;
    bool config_changed_;
};

// Logs Panel
class LogsPanel : public ServerPanel {
public:
    void Render() override;

private:
    void RenderLogFilter();
    void RenderLogView();

    std::vector<LogEntry> logs_;
    int log_level_filter_;
    std::string search_filter_;
    bool auto_scroll_;
};
```

### 4.2 TUI Panel Architecture

Using FTXUI components:

```cpp
// cyxwiz-server-node/src/tui/tui_app.h
namespace cyxwiz::servernode::tui {

class TUIApp {
public:
    TUIApp();
    void Run();  // Blocking main loop
    void Stop();

private:
    ftxui::Component CreateUI();
    ftxui::Component CreateHeader();
    ftxui::Component CreateResourceGauges();
    ftxui::Component CreateJobsTable();
    ftxui::Component CreateModelsTable();
    ftxui::Component CreateEarnings();
    ftxui::Component CreateFooter();

    // Screen components
    ftxui::ScreenInteractive screen_;
    std::atomic<bool> running_;

    // Data refresh
    std::thread refresh_thread_;
    void RefreshLoop();
};

// Resource Gauge Component
ftxui::Element CreateGauge(const std::string& label, float value, ftxui::Color color) {
    return ftxui::hbox({
        ftxui::text(label) | ftxui::size(ftxui::WIDTH, ftxui::EQUAL, 6),
        ftxui::gauge(value) | ftxui::flex | ftxui::color(color),
        ftxui::text(fmt::format("{:5.1f}%", value * 100)) | ftxui::size(ftxui::WIDTH, ftxui::EQUAL, 7),
    });
}

// Jobs Table Component
ftxui::Element CreateJobRow(const JobState& job) {
    auto progress_bar = ftxui::gauge(job.progress);
    auto status_color = job.is_running ? ftxui::Color::Green : ftxui::Color::Yellow;

    return ftxui::hbox({
        ftxui::text(job.id) | ftxui::size(ftxui::WIDTH, ftxui::EQUAL, 12),
        ftxui::text(job.type) | ftxui::size(ftxui::WIDTH, ftxui::EQUAL, 10),
        progress_bar | ftxui::flex | ftxui::color(status_color),
        ftxui::text(fmt::format("{:.4f}", job.metrics.loss)) | ftxui::size(ftxui::WIDTH, ftxui::EQUAL, 10),
    });
}

} // namespace cyxwiz::servernode::tui
```

### 4.3 Main Window (GUI Mode)

```cpp
// cyxwiz-server-node/src/gui/server_main_window.h
namespace cyxwiz::servernode::gui {

class ServerMainWindow {
public:
    ServerMainWindow();
    ~ServerMainWindow();

    void Render();
    void ResetDockLayout();

private:
    void RenderDockSpace();
    void RenderMenuBar();
    void RenderSidebar();
    void RenderStatusBar();
    void BuildInitialDockLayout();

    // Panels
    std::unique_ptr<DashboardPanel> dashboard_;
    std::unique_ptr<ModelBrowserPanel> model_browser_;
    std::unique_ptr<JobMonitorPanel> job_monitor_;
    std::unique_ptr<DeploymentPanel> deployment_;
    std::unique_ptr<APIKeysPanel> api_keys_;
    std::unique_ptr<PoolMiningPanel> pool_mining_;
    std::unique_ptr<MarketplacePanel> marketplace_;
    std::unique_ptr<SettingsPanel> settings_;
    std::unique_ptr<LogsPanel> logs_;
    std::unique_ptr<WalletPanel> wallet_;

    // State
    int selected_sidebar_item_;
    bool show_about_dialog_;
    bool first_time_layout_;
};

} // namespace cyxwiz::servernode::gui
```

### 4.4 Shared State Management

```cpp
// cyxwiz-server-node/src/core/state_manager.h
namespace cyxwiz::servernode {

// Observer pattern for reactive UI updates
class StateObserver {
public:
    virtual ~StateObserver() = default;
    virtual void OnJobsChanged() {}
    virtual void OnDeploymentsChanged() {}
    virtual void OnMetricsUpdated() {}
    virtual void OnWalletChanged() {}
    virtual void OnConnectionStatusChanged() {}
};

class StateManager {
public:
    static StateManager& Instance();

    // Registration
    void AddObserver(StateObserver* observer);
    void RemoveObserver(StateObserver* observer);

    // State queries
    const std::vector<JobState>& GetActiveJobs() const;
    const std::vector<DeploymentInstance>& GetDeployments() const;
    const SystemMetrics& GetMetrics() const;
    const WalletState& GetWallet() const;
    ConnectionStatus GetConnectionStatus() const;

    // State mutations (triggers observer notifications)
    void UpdateJobs(const std::vector<JobState>& jobs);
    void UpdateDeployments(const std::vector<DeploymentInstance>& deployments);
    void UpdateMetrics(const SystemMetrics& metrics);
    void UpdateWallet(const WalletState& wallet);
    void UpdateConnectionStatus(ConnectionStatus status);

private:
    void NotifyJobsChanged();
    void NotifyDeploymentsChanged();
    void NotifyMetricsUpdated();
    void NotifyWalletChanged();
    void NotifyConnectionStatusChanged();

    std::vector<StateObserver*> observers_;
    std::mutex mutex_;

    // Cached state
    std::vector<JobState> jobs_;
    std::vector<DeploymentInstance> deployments_;
    SystemMetrics metrics_;
    WalletState wallet_;
    ConnectionStatus connection_status_;
};

} // namespace cyxwiz::servernode
```

---

## 5. API Design

### 5.1 Internal Service API

The GUI/TUI communicates with backend services through a clean API layer:

```cpp
// cyxwiz-server-node/src/api/node_api.h
namespace cyxwiz::servernode::api {

// Training Provider API
class TrainingAPI {
public:
    // Query active jobs
    std::vector<JobInfo> GetActiveJobs();
    JobInfo GetJobDetails(const std::string& job_id);

    // Control jobs
    bool CancelJob(const std::string& job_id);
    bool PauseJob(const std::string& job_id);
    bool ResumeJob(const std::string& job_id);

    // Training configuration
    void SetGPUAllocation(float percentage);  // 0.0 - 1.0
    void SetMaxConcurrentJobs(int count);

    // Metrics
    TrainingMetrics GetJobMetrics(const std::string& job_id);
    std::vector<float> GetLossHistory(const std::string& job_id, int count);
};

// Model Deployment API
class DeploymentAPI {
public:
    // Model management
    std::vector<ModelInfo> GetLocalModels();
    bool DownloadModel(const std::string& model_id, ProgressCallback callback);
    bool DeleteModel(const std::string& model_id);

    // Deployment lifecycle
    std::string CreateDeployment(const DeploymentConfig& config);
    bool StartDeployment(const std::string& deployment_id);
    bool StopDeployment(const std::string& deployment_id);
    bool DeleteDeployment(const std::string& deployment_id);

    // Deployment info
    std::vector<DeploymentInfo> GetDeployments();
    DeploymentInfo GetDeploymentDetails(const std::string& deployment_id);
    DeploymentMetrics GetDeploymentMetrics(const std::string& deployment_id);
};

// API Key Management
class APIKeyAPI {
public:
    std::vector<APIKey> GetAPIKeys();
    APIKey GenerateKey(const std::string& name, int rate_limit);
    bool RevokeKey(const std::string& key_id);
    bool UpdateKeyRateLimit(const std::string& key_id, int rate_limit);
};

// Pool Mining API
class PoolMiningAPI {
public:
    std::vector<PoolInfo> GetAvailablePools();
    bool JoinPool(const std::string& pool_address);
    bool LeavePool();

    void SetMiningIntensity(float intensity);  // 0.0 - 1.0
    void StartMining();
    void StopMining();

    PoolMiningStats GetStats();
    bool IsMining();
};

// Wallet API
class WalletAPI {
public:
    bool ConnectWallet(const std::string& address);
    bool DisconnectWallet();

    WalletInfo GetWalletInfo();
    double GetCyxwizBalance();
    double GetSolBalance();

    std::vector<Transaction> GetTransactions(int limit);

    // Earnings
    EarningsInfo GetEarningsToday();
    EarningsInfo GetEarningsThisWeek();
    EarningsInfo GetEarningsThisMonth();
};

// System API
class SystemAPI {
public:
    SystemMetrics GetMetrics();
    NodeInfo GetNodeInfo();
    ConnectionStatus GetConnectionStatus();

    // Configuration
    NodeConfig GetConfig();
    void SaveConfig(const NodeConfig& config);

    // Central Server
    bool ConnectToCentralServer(const std::string& address);
    bool DisconnectFromCentralServer();
    bool IsConnectedToCentralServer();
};

} // namespace cyxwiz::servernode::api
```

### 5.2 External HTTP API (for deployed models)

Server Node exposes OpenAI-compatible API for deployed models:

```yaml
# OpenAPI 3.0 Specification
openapi: 3.0.3
info:
  title: CyxWiz Server Node Model API
  version: 1.0.0

servers:
  - url: http://localhost:8080/v1

security:
  - ApiKeyAuth: []

paths:
  /models:
    get:
      summary: List deployed models
      responses:
        '200':
          content:
            application/json:
              schema:
                type: object
                properties:
                  data:
                    type: array
                    items:
                      $ref: '#/components/schemas/Model'

  /chat/completions:
    post:
      summary: Create chat completion
      requestBody:
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/ChatCompletionRequest'
      responses:
        '200':
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/ChatCompletionResponse'

  /completions:
    post:
      summary: Create text completion
      requestBody:
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/CompletionRequest'
      responses:
        '200':
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/CompletionResponse'

  /embeddings:
    post:
      summary: Create embeddings
      requestBody:
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/EmbeddingRequest'
      responses:
        '200':
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/EmbeddingResponse'

components:
  securitySchemes:
    ApiKeyAuth:
      type: http
      scheme: bearer

  schemas:
    Model:
      type: object
      properties:
        id:
          type: string
        object:
          type: string
          enum: [model]
        created:
          type: integer
        owned_by:
          type: string

    ChatCompletionRequest:
      type: object
      required:
        - model
        - messages
      properties:
        model:
          type: string
        messages:
          type: array
          items:
            type: object
            properties:
              role:
                type: string
                enum: [system, user, assistant]
              content:
                type: string
        temperature:
          type: number
          default: 0.7
        max_tokens:
          type: integer
          default: 256
        stream:
          type: boolean
          default: false
```

### 5.3 gRPC Extensions for GUI

Add new messages to `cyxwiz-protocol/proto/node.proto`:

```protobuf
// GUI-specific messages
message GetSystemMetricsRequest {
    string node_id = 1;
}

message SystemMetricsResponse {
    double cpu_usage = 1;
    double gpu_usage = 2;
    double ram_usage = 3;
    double vram_usage = 4;
    double network_in_mbps = 5;
    double network_out_mbps = 6;
    double disk_usage = 7;
    double power_watts = 8;
    double temperature_celsius = 9;
}

message GetEarningsRequest {
    string node_id = 1;
    EarningsPeriod period = 2;
}

enum EarningsPeriod {
    EARNINGS_TODAY = 0;
    EARNINGS_THIS_WEEK = 1;
    EARNINGS_THIS_MONTH = 2;
    EARNINGS_ALL_TIME = 3;
}

message EarningsResponse {
    double training_earnings = 1;
    double inference_earnings = 2;
    double pool_mining_earnings = 3;
    double total_earnings = 4;
    string currency = 5;  // "CYXWIZ"
    double usd_equivalent = 6;
}

// Pool Mining
message JoinPoolRequest {
    string node_id = 1;
    string pool_address = 2;
    string wallet_address = 3;
}

message JoinPoolResponse {
    StatusCode status = 1;
    string worker_id = 2;
    Error error = 3;
}

message PoolMiningStatsRequest {
    string node_id = 1;
}

message PoolMiningStatsResponse {
    double hashrate = 1;
    uint64 shares_submitted = 2;
    uint64 shares_accepted = 3;
    uint64 shares_rejected = 4;
    double estimated_daily = 5;
    double estimated_monthly = 6;
    uint64 uptime_seconds = 7;
}

// API Key Management
message GenerateAPIKeyRequest {
    string node_id = 1;
    string key_name = 2;
    int32 rate_limit_per_minute = 3;
    repeated string allowed_models = 4;
}

message GenerateAPIKeyResponse {
    StatusCode status = 1;
    string key_id = 2;
    string api_key = 3;  // Only returned once
    Error error = 4;
}

message ListAPIKeysRequest {
    string node_id = 1;
}

message ListAPIKeysResponse {
    repeated APIKeyInfo keys = 1;
    Error error = 2;
}

message APIKeyInfo {
    string key_id = 1;
    string name = 2;
    string key_prefix = 3;  // First 8 chars for identification
    int32 rate_limit_per_minute = 4;
    uint64 total_requests = 5;
    int64 created_at = 6;
    int64 last_used_at = 7;
    bool is_active = 8;
}

message RevokeAPIKeyRequest {
    string node_id = 1;
    string key_id = 2;
}

message RevokeAPIKeyResponse {
    StatusCode status = 1;
    Error error = 2;
}
```

---

## 6. Model Deployment Flow

### 6.1 Local Deployment Flow

```
User                  GUI                   DeploymentManager        ModelLoader
  |                    |                          |                      |
  |-- Select Model --->|                          |                      |
  |-- Configure ------>|                          |                      |
  |-- Click Deploy --->|                          |                      |
  |                    |-- AcceptDeployment() --->|                      |
  |                    |                          |-- Load() ----------->|
  |                    |                          |<-- Success ----------|
  |                    |                          |-- StartHTTPServer -->|
  |                    |<-- deployment_id --------|                      |
  |<-- Show Endpoint --|                          |                      |
  |                    |                          |                      |
```

### 6.2 Network Deployment Flow

```
User         GUI         NodeClient       CentralServer      OtherNodes
  |           |              |                 |                  |
  |-- Deploy->|              |                 |                  |
  |           |-- Register ->|-- Register ---->|                  |
  |           |              |                 |-- Notify ------->|
  |           |              |<-- ACK ---------|                  |
  |           |              |                 |                  |
  |           |  [Model is now discoverable]   |                  |
  |           |              |                 |                  |
  |           |              |<-- Request -----|<-- Inference ----|
  |           |              |-- Response ---->|-- Response ----->|
  |           |              |                 |                  |
```

### 6.3 API Key Generation Flow

```cpp
// cyxwiz-server-node/src/api/api_key_manager.h
namespace cyxwiz::servernode {

struct APIKey {
    std::string id;              // UUID
    std::string name;
    std::string key_hash;        // SHA256 of actual key
    std::string key_prefix;      // First 8 chars for display
    int rate_limit_per_minute;
    std::vector<std::string> allowed_models;  // Empty = all models
    int64_t created_at;
    int64_t last_used_at;
    uint64_t total_requests;
    bool is_active;
};

class APIKeyManager {
public:
    APIKeyManager(const std::string& storage_path);

    // Generate new API key
    // Returns: { key_id, full_key } - full_key is only returned once!
    std::pair<std::string, std::string> GenerateKey(
        const std::string& name,
        int rate_limit_per_minute,
        const std::vector<std::string>& allowed_models = {}
    );

    // Validate incoming request
    bool ValidateKey(const std::string& key, const std::string& model_id);

    // Check rate limit
    bool CheckRateLimit(const std::string& key);

    // Management
    std::vector<APIKey> GetAllKeys();
    bool RevokeKey(const std::string& key_id);
    bool UpdateRateLimit(const std::string& key_id, int new_limit);

    // Persistence
    void Save();
    void Load();

private:
    std::string GenerateSecureKey();
    std::string HashKey(const std::string& key);

    std::string storage_path_;
    std::unordered_map<std::string, APIKey> keys_;
    std::unordered_map<std::string, int> rate_limit_counters_;  // key_hash -> count
    std::mutex mutex_;
};

} // namespace cyxwiz::servernode
```

### 6.4 Endpoint Management

```cpp
// cyxwiz-server-node/src/api/endpoint_manager.h
namespace cyxwiz::servernode {

struct Endpoint {
    std::string id;
    std::string model_id;
    std::string host;
    int port;
    std::string protocol;  // "http", "grpc"
    bool require_auth;
    int64_t started_at;
    uint64_t total_requests;
    double avg_latency_ms;
};

class EndpointManager {
public:
    // Create endpoint for model
    Endpoint CreateEndpoint(const std::string& model_id, int port = 0);

    // Start serving
    bool StartEndpoint(const std::string& endpoint_id);
    bool StopEndpoint(const std::string& endpoint_id);

    // Query
    std::vector<Endpoint> GetActiveEndpoints();
    Endpoint GetEndpoint(const std::string& endpoint_id);

    // Get URL for external access
    std::string GetPublicURL(const std::string& endpoint_id);

private:
    // HTTP server for OpenAI-compatible API
    std::unique_ptr<httplib::Server> http_server_;

    // Endpoint registry
    std::unordered_map<std::string, Endpoint> endpoints_;
    std::mutex mutex_;
};

} // namespace cyxwiz::servernode
```

---

## 7. Marketplace Integration

### 7.1 "Airbnb for Models" Concept

The marketplace allows model owners to list their models for rental:

```
Model Owner (Host)                    Model User (Guest)
      |                                     |
      |-- List Model on Marketplace ------->|
      |   - Set price per request           |
      |   - Set availability hours          |
      |   - Configure hardware requirements |
      |                                     |
      |<-- Request to Deploy ---------------|
      |                                     |
      |-- Accept/Auto-accept -------------->|
      |                                     |
      |<-- Inference Requests --------------|
      |-- Responses ------------------------>|
      |                                     |
      |<-- Payment (CYXWIZ tokens) ---------|
      |                                     |
```

### 7.2 Listing Models

```cpp
// cyxwiz-server-node/src/marketplace/model_listing.h
namespace cyxwiz::servernode::marketplace {

enum class PricingModel {
    PerRequest,      // Pay per API call
    PerToken,        // Pay per input/output token (for LLMs)
    PerMinute,       // Time-based pricing
    PerCompute,      // GPU compute time
};

struct ModelListing {
    std::string listing_id;
    std::string model_id;
    std::string owner_id;
    std::string owner_wallet;

    // Pricing
    PricingModel pricing_model;
    double price;              // In CYXWIZ tokens
    double min_deposit;        // Required deposit

    // Availability
    std::vector<int> available_hours;  // 0-23 UTC
    int max_concurrent_users;

    // Hardware requirements
    protocol::HardwareRequirements requirements;

    // Stats
    double rating;
    uint64_t total_uses;
    uint64_t total_earnings;

    // Metadata
    std::string description;
    std::vector<std::string> tags;
    std::string thumbnail_url;
    int64_t listed_at;
};

class MarketplaceClient {
public:
    MarketplaceClient(const std::string& central_server_address);

    // Listing management
    std::string CreateListing(const ModelListing& listing);
    bool UpdateListing(const std::string& listing_id, const ModelListing& listing);
    bool RemoveListing(const std::string& listing_id);

    // Browse marketplace
    std::vector<ModelListing> SearchListings(
        const std::string& query,
        const std::vector<std::string>& tags,
        protocol::ModelFormat format_filter,
        double max_price
    );

    // Rental
    std::string RequestAccess(const std::string& listing_id, double deposit);
    bool ReleaseAccess(const std::string& access_id);

    // Earnings
    EarningsReport GetEarningsReport(int64_t start_time, int64_t end_time);

private:
    std::unique_ptr<protocol::MarketplaceService::Stub> stub_;
};

} // namespace cyxwiz::servernode::marketplace
```

### 7.3 Revenue Sharing Model

```
Inference Request Revenue Flow:

  User Payment (100%)
       |
       v
  +----------+
  |  Escrow  | (Smart Contract)
  +----------+
       |
       +-- 90% --> Model Host (Server Node)
       |
       +-- 7% --> CyxWiz Platform Fee
       |
       +-- 2% --> Model Creator (if different)
       |
       +-- 1% --> Network Validators
```

### 7.4 Marketplace Panel UI

```cpp
// cyxwiz-server-node/src/gui/panels/marketplace_panel.h
namespace cyxwiz::servernode::gui {

class MarketplacePanel : public ServerPanel {
public:
    void Render() override;

private:
    void RenderSearchBar();
    void RenderFilters();
    void RenderListings();
    void RenderListingCard(const marketplace::ModelListing& listing);
    void RenderListingDetails(const marketplace::ModelListing& listing);
    void RenderMyListings();
    void RenderCreateListingDialog();

    // Search state
    char search_query_[256];
    int selected_format_filter_;
    double max_price_filter_;
    std::vector<std::string> selected_tags_;

    // Listings
    std::vector<marketplace::ModelListing> search_results_;
    std::vector<marketplace::ModelListing> my_listings_;
    marketplace::ModelListing* selected_listing_;

    // Create listing dialog
    bool show_create_dialog_;
    marketplace::ModelListing new_listing_;
};

} // namespace cyxwiz::servernode::gui
```

---

## 8. Security Considerations

### 8.1 Authentication Flow

```
Client                     Server Node                   Central Server
   |                           |                              |
   |-- API Key -------------->|                              |
   |                          |-- Verify Key (local) ------->|
   |                          |                              |
   |                          |-- Check Reputation ---------->|
   |                          |<-- Reputation OK ------------|
   |                          |                              |
   |<-- 200 OK ---------------|                              |
   |                          |                              |
```

### 8.2 API Key Security

```cpp
namespace cyxwiz::servernode::security {

class APIKeyValidator {
public:
    // Key format: cyx_sk_[live|test]_[32 random chars]
    static constexpr char KEY_PREFIX_LIVE[] = "cyx_sk_live_";
    static constexpr char KEY_PREFIX_TEST[] = "cyx_sk_test_";

    // Validate key format
    static bool ValidateFormat(const std::string& key);

    // Hash key for storage (never store raw keys)
    static std::string HashKey(const std::string& key);

    // Generate secure random key
    static std::string GenerateSecureKey(bool is_live);

private:
    static std::string GenerateRandomString(int length);
};

// Rate limiter with sliding window
class RateLimiter {
public:
    RateLimiter(int requests_per_minute);

    // Returns true if request is allowed
    bool Allow(const std::string& key_id);

    // Get remaining quota
    int GetRemainingQuota(const std::string& key_id);

private:
    struct Window {
        std::deque<int64_t> timestamps;
    };

    int limit_per_minute_;
    std::unordered_map<std::string, Window> windows_;
    std::mutex mutex_;
};

} // namespace cyxwiz::servernode::security
```

### 8.3 Model Sandboxing

For untrusted models or custom Python scripts:

```cpp
namespace cyxwiz::servernode::security {

enum class SandboxLevel {
    None,           // Full access (trusted models)
    Basic,          // Limited file system access
    Strict,         // Docker container isolation
    MaxSecurity     // gVisor/Firecracker microVM
};

class ModelSandbox {
public:
    ModelSandbox(SandboxLevel level);

    // Create isolated environment for model
    bool CreateEnvironment(const std::string& model_id);

    // Execute inference in sandbox
    bool ExecuteInference(
        const std::string& model_id,
        const std::string& input,
        std::string& output,
        int timeout_ms
    );

    // Resource limits
    void SetMemoryLimit(size_t bytes);
    void SetCPUQuota(float percentage);
    void SetGPUMemoryLimit(size_t bytes);
    void SetNetworkAccess(bool allowed);

    // Cleanup
    void DestroyEnvironment(const std::string& model_id);

private:
    SandboxLevel level_;
    std::unordered_map<std::string, std::string> container_ids_;
};

} // namespace cyxwiz::servernode::security
```

### 8.4 Network Security

```cpp
namespace cyxwiz::servernode::security {

class TLSConfig {
public:
    // Generate self-signed certificate (development)
    static void GenerateSelfSigned(
        const std::string& cert_path,
        const std::string& key_path
    );

    // Load certificates
    void LoadCertificate(const std::string& cert_path);
    void LoadPrivateKey(const std::string& key_path);
    void LoadCACertificate(const std::string& ca_path);

    // Apply to gRPC server
    std::shared_ptr<grpc::ServerCredentials> GetServerCredentials();

    // Apply to gRPC client
    std::shared_ptr<grpc::ChannelCredentials> GetChannelCredentials();
};

// Firewall rules for model endpoints
class EndpointFirewall {
public:
    // IP allowlist/blocklist
    void AllowIP(const std::string& ip);
    void BlockIP(const std::string& ip);
    void AllowIPRange(const std::string& cidr);

    // Check if request is allowed
    bool IsAllowed(const std::string& ip, const std::string& api_key);

private:
    std::set<std::string> allowed_ips_;
    std::set<std::string> blocked_ips_;
};

} // namespace cyxwiz::servernode::security
```

### 8.5 Wallet Security

```cpp
namespace cyxwiz::servernode::security {

class WalletSecurity {
public:
    // Never store private keys directly
    // Only store public address and encrypted auth tokens

    // Wallet connection via browser extension
    struct WalletConnection {
        std::string public_address;
        std::string auth_token;  // Encrypted session token
        int64_t expires_at;
    };

    // Sign transactions using hardware wallet or extension
    static std::string SignMessage(
        const std::string& message,
        const WalletConnection& wallet
    );

    // Verify wallet ownership
    static bool VerifyOwnership(
        const std::string& address,
        const std::string& signature,
        const std::string& challenge
    );
};

} // namespace cyxwiz::servernode::security
```

---

## 9. Implementation Roadmap

### Phase 1: Core Infrastructure (4-6 weeks)

**Week 1-2: Backend Manager & State Management**
- [ ] Implement `BackendManager` singleton
- [ ] Implement `StateManager` with observer pattern
- [ ] Create internal API layer (`TrainingAPI`, `DeploymentAPI`, etc.)
- [ ] Add system metrics collection

**Week 3-4: TUI Foundation**
- [ ] Add FTXUI dependency via vcpkg
- [ ] Implement `TUIApp` main loop
- [ ] Create resource gauges (CPU, GPU, RAM, VRAM)
- [ ] Create jobs table component
- [ ] Create deployments table component
- [ ] Implement keyboard navigation

**Week 5-6: GUI Foundation**
- [ ] Set up ImGui rendering for server-node
- [ ] Port theme system from cyxwiz-engine
- [ ] Implement `ServerMainWindow` with docking
- [ ] Create sidebar navigation
- [ ] Implement `DashboardPanel`

### Phase 2: Model Management (3-4 weeks)

**Week 7-8: Model Browser & Deployment**
- [ ] Implement `ModelBrowserPanel`
- [ ] Add local model scanning
- [ ] Implement `DeploymentPanel`
- [ ] Add deployment configuration UI
- [ ] Integrate with `DeploymentManager`

**Week 9-10: HTTP API & API Keys**
- [ ] Implement OpenAI-compatible HTTP API
- [ ] Add API key generation and management
- [ ] Implement `APIKeysPanel`
- [ ] Add rate limiting
- [ ] Add request logging

### Phase 3: Network Features (3-4 weeks)

**Week 11-12: Central Server Integration**
- [ ] Implement marketplace client
- [ ] Add model listing functionality
- [ ] Implement `MarketplacePanel`
- [ ] Add model search and filtering

**Week 13-14: Pool Mining**
- [ ] Implement pool mining protocol
- [ ] Add `PoolMiningPanel`
- [ ] Integrate with mining pools
- [ ] Add earnings tracking

### Phase 4: Security & Polish (2-3 weeks)

**Week 15-16: Security Hardening**
- [ ] Implement TLS for all connections
- [ ] Add model sandboxing (Docker)
- [ ] Implement wallet security
- [ ] Add audit logging

**Week 17: Testing & Documentation**
- [ ] Write unit tests for API layer
- [ ] Write integration tests
- [ ] Create user documentation
- [ ] Performance optimization

### Phase 5: Advanced Features (Ongoing)

**Future Enhancements:**
- [ ] Remote management (dual-process mode)
- [ ] Mobile companion app
- [ ] Advanced analytics dashboard
- [ ] Multi-GPU management
- [ ] Cluster mode (multiple nodes)
- [ ] Model fine-tuning UI
- [ ] A/B testing for deployed models
- [ ] Auto-scaling configuration

---

## Appendix A: File Structure

```
cyxwiz-server-node/
├── CMakeLists.txt
├── src/
│   ├── main.cpp                          # Entry point with mode selection
│   ├── core/
│   │   ├── backend_manager.cpp/h         # Shared services manager
│   │   ├── state_manager.cpp/h           # Reactive state management
│   │   ├── config.cpp/h                  # Configuration handling
│   │   └── metrics_collector.cpp/h       # System metrics
│   ├── api/
│   │   ├── training_api.cpp/h            # Training provider API
│   │   ├── deployment_api.cpp/h          # Model deployment API
│   │   ├── api_key_manager.cpp/h         # API key management
│   │   ├── endpoint_manager.cpp/h        # HTTP endpoint management
│   │   └── pool_mining_api.cpp/h         # Pool mining API
│   ├── gui/
│   │   ├── server_main_window.cpp/h      # Main GUI window
│   │   ├── server_panel.h                # Base panel class
│   │   └── panels/
│   │       ├── dashboard_panel.cpp/h     # System overview
│   │       ├── model_browser_panel.cpp/h # Model management
│   │       ├── job_monitor_panel.cpp/h   # Active jobs
│   │       ├── deployment_panel.cpp/h    # Deploy models
│   │       ├── api_keys_panel.cpp/h      # API key management
│   │       ├── pool_mining_panel.cpp/h   # Pool mining
│   │       ├── marketplace_panel.cpp/h   # Model marketplace
│   │       ├── settings_panel.cpp/h      # Configuration
│   │       ├── logs_panel.cpp/h          # Log viewer
│   │       └── wallet_panel.cpp/h        # Wallet management
│   ├── tui/
│   │   ├── tui_app.cpp/h                 # TUI application
│   │   └── components/
│   │       ├── resource_gauges.cpp/h     # CPU/GPU/RAM gauges
│   │       ├── jobs_table.cpp/h          # Jobs display
│   │       ├── models_table.cpp/h        # Models display
│   │       └── footer.cpp/h              # Status bar
│   ├── marketplace/
│   │   ├── model_listing.cpp/h           # Listing data structures
│   │   ├── marketplace_client.cpp/h      # Central server client
│   │   └── revenue_tracker.cpp/h         # Earnings tracking
│   ├── security/
│   │   ├── api_key_validator.cpp/h       # Key validation
│   │   ├── rate_limiter.cpp/h            # Request rate limiting
│   │   ├── model_sandbox.cpp/h           # Docker sandboxing
│   │   └── tls_config.cpp/h              # TLS configuration
│   └── http/
│       ├── openai_api_server.cpp/h       # OpenAI-compatible API
│       └── routes/
│           ├── models.cpp/h              # /v1/models
│           ├── completions.cpp/h         # /v1/completions
│           ├── chat.cpp/h                # /v1/chat/completions
│           └── embeddings.cpp/h          # /v1/embeddings
├── tests/
│   ├── test_backend_manager.cpp
│   ├── test_api_key_manager.cpp
│   ├── test_rate_limiter.cpp
│   └── test_marketplace_client.cpp
└── resources/
    ├── fonts/                            # UI fonts
    ├── icons/                            # Application icons
    └── themes/                           # Theme configurations
```

---

## Appendix B: Configuration Schema

```yaml
# config.yaml - Server Node configuration

# General settings
node:
  id: "auto"                    # Auto-generate or specify
  name: "My CyxWiz Node"
  region: "us-west-2"

# Interface settings
interface:
  mode: "gui"                   # gui, tui, or headless
  theme: "CyxWizDark"
  start_minimized: false
  system_tray: true

# Network settings
network:
  central_server: "central.cyxwiz.io:50051"
  p2p_port: 50052
  http_api_port: 8080
  enable_tls: true
  cert_path: "./certs/server.crt"
  key_path: "./certs/server.key"

# Training provider settings
training:
  enabled: true
  max_concurrent_jobs: 2
  gpu_allocation: 0.8           # 80% of GPU for training
  accepted_job_types:
    - "training"
    - "fine_tuning"

# Model deployment settings
deployment:
  enabled: true
  models_directory: "./models"
  max_loaded_models: 3
  default_port_range: [8080, 8090]

# API settings
api:
  require_authentication: true
  default_rate_limit: 100       # requests per minute
  cors_origins:
    - "https://chat.cyxwiz.io"
    - "http://localhost:*"

# Pool mining settings
pool_mining:
  enabled: false
  pool_address: "pool.cyxwiz.io:3333"
  intensity: 0.5                # 0.0 to 1.0
  auto_start: false
  mine_when_idle_only: true

# Wallet settings
wallet:
  address: ""                   # Solana wallet address
  auto_withdraw_threshold: 100  # CYXWIZ tokens

# Resource limits
resources:
  max_memory_percent: 80
  max_gpu_memory_percent: 90
  max_disk_usage_gb: 100

# Logging
logging:
  level: "info"                 # debug, info, warn, error
  file: "./logs/server-node.log"
  max_size_mb: 100
  max_files: 5

# Security
security:
  sandbox_level: "basic"        # none, basic, strict
  allowed_ip_ranges: []         # Empty = allow all
  blocked_ips: []
```

---

## Appendix C: Key Dependencies

Add to `vcpkg.json`:

```json
{
  "dependencies": [
    "ftxui",
    "imgui",
    "imgui[docking-experimental]",
    "imgui[glfw-binding]",
    "imgui[opengl3-binding]",
    "implot",
    "glfw3",
    "glad",
    "grpc",
    "protobuf",
    "spdlog",
    "nlohmann-json",
    "cpp-httplib",
    "openssl",
    "yaml-cpp",
    "catch2"
  ]
}
```

---

## Appendix D: Command-Line Interface

```bash
# Start in GUI mode (default if display available)
cyxwiz-server-node --mode=gui

# Start in TUI mode
cyxwiz-server-node --mode=tui

# Start in headless mode (servers)
cyxwiz-server-node --mode=headless

# Specify config file
cyxwiz-server-node --config=/path/to/config.yaml

# Override specific settings
cyxwiz-server-node --port=8080 --gpu-allocation=0.9

# Show version and exit
cyxwiz-server-node --version

# Show help
cyxwiz-server-node --help
```

---

## References

1. **btop**: https://github.com/aristocratos/btop - Terminal resource monitor inspiration
2. **LM Studio**: https://lmstudio.ai/ - Desktop LLM interface inspiration
3. **FTXUI**: https://github.com/ArthurSonzogni/FTXUI - C++ TUI library
4. **Dear ImGui**: https://github.com/ocornut/imgui - Immediate mode GUI library
5. **OpenAI API Spec**: https://platform.openai.com/docs/api-reference - API compatibility target
6. **Solana Docs**: https://docs.solana.com/ - Blockchain integration reference


 cyxwiz-server-node --mode=gui      # GUI mode
  cyxwiz-server-node --mode=tui      # TUI mode (default on Windows/display)
  cyxwiz-server-node --tui           # Short form
  cyxwiz-server-node --mode=headless # Original headless mode
  cyxwiz-server-node                 # Auto-detect

    Dual-Process Architecture

  Process 1: cyxwiz-server-daemon (Headless Service)

  cyxwiz-server-daemon [options]
    --ipc-address=ADDR     IPC for GUI (default: localhost:50054)
    --central-server=ADDR  Central server (default: localhost:50051)
    --config=PATH          Config file path

  Provides:
  - gRPC IPC service on localhost:50054 for GUI connections
  - P2P service for Engine connections
  - Node service for Central Server job assignment
  - Model deployment with OpenAI-compatible API
  - Job execution for distributed training

  Process 2: cyxwiz-server-gui (GUI/TUI Client)

  cyxwiz-server-gui [options]
    --daemon=ADDR   Daemon to connect to (default: localhost:50054)
    --mode=gui|tui  Interface mode (default: gui)
    --gui           Use GUI mode
    --tui           Use TUI mode

  Features:
  - Connects to daemon via gRPC IPC
  - Can run on a different machine (remote management)
  - Full GUI with ImGui or TUI with FTXUI

  Key Files Created

  | File                         | Purpose                                  |
  |------------------------------|------------------------------------------|
  | proto/daemon.proto           | IPC service definition (50+ RPC methods) |
  | src/ipc/daemon_service.h/cpp | gRPC server for daemon                   |
  | src/ipc/daemon_client.h/cpp  | Client for GUI to connect                |
  | src/daemon_main.cpp          | Daemon entry point                       |
  | src/gui_main.cpp             | GUI client entry point                   |

  Architecture Diagram

  cyxwiz-server-daemon (headless service)
      |
      +-- gRPC IPC on localhost:50054
      |       |
      |       +-- GetStatus, GetMetrics, StreamMetrics
      |       +-- ListJobs, CancelJob, StreamJobUpdates
      |       +-- DeployModel, UndeployModel, ListDeployments
      |       +-- CreateAPIKey, RevokeAPIKey, ListAPIKeys
      |       +-- GetConfig, SetConfig
      |       +-- Shutdown, Restart
      |
      +-- P2P service on :50052 (Engine)
      +-- Node service on :50055 (Central Server)
      +-- Deployment API on :50056

  cyxwiz-server-gui (GUI/TUI client)
      |
      +-- Connects to daemon via gRPC
      +-- Can run on different machine (remote management)
      +-- Renders ImGui/FTXUI interface

        1. Model Browser Panel (model_browser_panel.cpp/h)
    - DaemonClient integration for listing models
    - Deploy dialog with port/GPU layers/context size configuration
    - Delete confirmation dialog with local fallback
    - Connection status indicator
  2. Deployment Panel (deployment_panel.cpp/h)
    - Tab-based UI (Deploy New / Active Deployments)
    - Model dropdown populated from daemon
    - Server configuration form with validation
    - Active deployments table with status colors
    - Undeploy confirmation dialog
  3. API Keys Panel (api_keys_panel.cpp/h)
    - Generate new key dialog with name and rate limit
    - One-time key display with copy-to-clipboard
    - Keys table with prefix, usage stats, last used
    - Revoke confirmation dialog
  4. Job Monitor Panel (job_monitor_panel.cpp/h)
    - Job cards with status, progress, metrics
    - Loss mini-charts with ImPlot sparklines
    - Status filtering (show completed/failed)
    - Cancel job functionality with confirmation
  5. Logs Panel (logs_panel.cpp/h)
    - Level filtering (Debug/Info/Warning/Error)
    - Text search filtering
    - Auto-scroll with toggle
    - Circular buffer (max 1000 entries)
    - ImGuiListClipper for performance

     Phase 4: Security & Polish - Complete ✅

  All security components have been implemented and tested. Here's a summary of what was accomplished:

  Implemented Components

  | Component         | Files                 | Status                                              |
  |-------------------|-----------------------|-----------------------------------------------------|
  | TLS Configuration | tls_config.h/cpp      | ✅ Self-signed cert generation, gRPC credentials     |
  | API Key Manager   | api_key_manager.h/cpp | ✅ SHA256 hashing, persistence, model access control |
  | Rate Limiter      | rate_limiter.h/cpp    | ✅ Token bucket algorithm per-key                    |
  | Audit Logger      | audit_logger.h/cpp    | ✅ 16 event types, JSON logging                      |
  | Docker Manager    | docker_manager.h/cpp  | ✅ Container sandboxing interface                    |
  | Wallet Manager    | wallet_manager.h/cpp  | ✅ Solana JSON-RPC, earnings tracking                |
  | TLS in Daemon     | daemon_main.cpp       | ✅ CLI args, auto-gen certs, gRPC servers            |
  | Security Tests    | test_security.cpp     | ✅ 24 test cases, 123 assertions                     |

  Test Results

  All tests passed (123 assertions in 24 test cases)

  Test coverage includes:
  - API Key generation, validation, revocation, model access, persistence
  - Rate limiting with quotas and reset times
  - Audit logging with all event types
  - TLS certificate loading and generation
  - Wallet address validation, connection state, earnings
  - Docker manager availability and configuration
phase 5 result
  Daemon TLS Usage

  The daemon now supports:
  cyxwiz-server-daemon --tls --tls-cert=cert.crt --tls-key=key.key
  cyxwiz-server-daemon --tls-auto  # Auto-generate self-signed cert

    Daemon (ID: 3a6d87):
  - DevicePool detected 2 GPUs: NVIDIA GTX 1050 Ti + Intel UHD 630
  - IPC service on localhost:50054
  - P2P service on 0.0.0.0:50052

  GUI (ID: a1bb2f):
  - Connected to daemon successfully
  - ImGui window initialized
  - All panels available including the new Fine-tuning panel

  The Server Node GUI window should now be open. You can navigate to the Fine-tuning panel in the sidebar to see the
   new model fine-tuning configuration UI with:
  - Model/Dataset selection
  - Layer freezing options (None, Except Last N, Up To Layer, Custom)
  - LR Schedule configuration (Constant, Step, Exponential, Cosine, OneCycle, CosineWarmRestart)
  - Early stopping settings
  - Gradient clipping options
  - Training parameters

    Key Use Cases

  1. Transfer Learning Service
  - Node operators can offer fine-tuning as a paid service
  - Users submit pre-trained models + custom datasets → get specialized models back
  - Much faster/cheaper than training from scratch

  2. Marketplace Model Customization
  - Download a general model from the CyxWiz Marketplace
  - Fine-tune it on your domain-specific data (e.g., medical images, legal documents)
  - Deploy the customized model for inference

  3. Federated Learning
  - In decentralized ML, nodes fine-tune local model copies on local data
  - Updates are aggregated without sharing raw data (privacy-preserving)

  4. Model-as-a-Service Enhancement
  - Node operators hosting inference endpoints can improve their deployed models
  - Fine-tune on user feedback or new data without full retraining

  Architecture Flow

  Engine (Design) → Central Server (Dispatch) → Server Node (Execute)
                                                      ↓
                                                Fine-tune job
                                                      ↓
                                                Return improved model

  The Engine designs models visually, but the Server Node has the GPU horsepower to actually run compute-intensive
  fine-tuning jobs. This separation allows users without powerful hardware to leverage the decentralized network for
   model customization.

  Essentially: Server Nodes monetize their GPUs by offering training/fine-tuning/inference services to the network.