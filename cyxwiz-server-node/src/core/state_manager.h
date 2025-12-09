// state_manager.h - Observable state with observer pattern for reactive UI
#pragma once

#include <vector>
#include <string>
#include <mutex>
#include <chrono>
#include <atomic>
#include <unordered_map>

namespace cyxwiz::servernode::core {

// Forward declarations
struct SystemMetrics;

// Job state for UI display
struct JobState {
    std::string id;
    std::string type;  // "Training", "Inference", "Fine-tuning"
    std::string client_id;
    float progress = 0.0f;  // 0.0 - 1.0
    int current_epoch = 0;
    int total_epochs = 0;
    double loss = 0.0;
    double accuracy = 0.0;
    double learning_rate = 0.0;
    int64_t samples_processed = 0;
    int64_t time_elapsed_ms = 0;
    bool is_running = false;
    bool is_paused = false;
    std::chrono::system_clock::time_point start_time;
    std::unordered_map<std::string, double> custom_metrics;
};

// Deployment state for UI display
struct DeploymentState {
    std::string id;
    std::string model_id;
    std::string model_name;
    std::string format;  // "GGUF", "ONNX", "PyTorch"
    std::string status;  // "Loading", "Running", "Stopped", "Failed"
    std::string endpoint;
    int port = 0;
    uint64_t request_count = 0;
    double avg_latency_ms = 0.0;
    size_t memory_usage = 0;
    std::chrono::system_clock::time_point started_at;
};

// Per-GPU metrics
struct GPUMetrics {
    int device_id = 0;
    std::string name;
    std::string vendor;  // "NVIDIA", "Intel", "AMD"
    float usage_3d = 0.0f;      // 3D/Compute engine utilization (0-1)
    float usage_copy = 0.0f;    // Copy engine utilization (0-1)
    float usage_video_decode = 0.0f;  // Video decode engine (0-1)
    float usage_video_encode = 0.0f;  // Video encode engine (0-1)
    float memory_usage = 0.0f;  // VRAM usage (0-1)
    size_t vram_used_bytes = 0;
    size_t vram_total_bytes = 0;
    float temperature_celsius = 0.0f;
    float power_watts = 0.0f;
    bool is_nvidia = false;  // NVML available
};

// System metrics for UI display
struct SystemMetrics {
    float cpu_usage = 0.0f;
    float gpu_usage = 0.0f;         // Primary GPU usage (for backward compat)
    float ram_usage = 0.0f;
    float vram_usage = 0.0f;        // Primary GPU VRAM (for backward compat)
    float network_in_mbps = 0.0f;
    float network_out_mbps = 0.0f;
    float temperature_celsius = 0.0f;  // Primary GPU temp
    float power_watts = 0.0f;          // Primary GPU power
    size_t ram_used_bytes = 0;
    size_t ram_total_bytes = 0;
    size_t vram_used_bytes = 0;        // Primary GPU VRAM
    size_t vram_total_bytes = 0;       // Primary GPU VRAM

    // Per-GPU metrics
    std::vector<GPUMetrics> gpus;
    int gpu_count = 0;
};

// Earnings info
struct EarningsInfo {
    double training_earnings = 0.0;
    double inference_earnings = 0.0;
    double pool_mining_earnings = 0.0;
    double total_earnings = 0.0;
    std::string currency = "CYXWIZ";
    double usd_equivalent = 0.0;
};

// Connection status
enum class ConnectionStatus {
    Disconnected,
    Connecting,
    Connected,
    Reconnecting,
    Error
};

// Observer interface
class StateObserver {
public:
    virtual ~StateObserver() = default;
    virtual void OnJobsChanged() {}
    virtual void OnDeploymentsChanged() {}
    virtual void OnMetricsUpdated() {}
    virtual void OnConnectionStatusChanged() {}
    virtual void OnEarningsChanged() {}
    virtual void OnWalletChanged() {}
};

// State manager singleton
class StateManager {
public:
    StateManager();
    ~StateManager() = default;

    // Observer management
    void AddObserver(StateObserver* observer);
    void RemoveObserver(StateObserver* observer);

    // State getters (thread-safe, returns copies)
    std::vector<JobState> GetActiveJobs() const;
    std::vector<DeploymentState> GetDeployments() const;
    SystemMetrics GetMetrics() const;
    EarningsInfo GetEarningsToday() const;
    EarningsInfo GetEarningsThisWeek() const;
    EarningsInfo GetEarningsThisMonth() const;
    ConnectionStatus GetConnectionStatus() const;
    std::string GetWalletAddress() const;
    double GetWalletBalance() const;

    // State setters (trigger observer notifications)
    void UpdateJobs(const std::vector<JobState>& jobs);
    void UpdateJob(const JobState& job);
    void RemoveJob(const std::string& job_id);

    void UpdateDeployments(const std::vector<DeploymentState>& deployments);
    void UpdateDeployment(const DeploymentState& deployment);
    void RemoveDeployment(const std::string& deployment_id);

    void UpdateMetrics(const SystemMetrics& metrics);
    void UpdateConnectionStatus(ConnectionStatus status);
    void UpdateEarnings(const EarningsInfo& today, const EarningsInfo& week, const EarningsInfo& month);
    void UpdateWallet(const std::string& address, double balance);

private:
    void NotifyJobsChanged();
    void NotifyDeploymentsChanged();
    void NotifyMetricsUpdated();
    void NotifyConnectionStatusChanged();
    void NotifyEarningsChanged();
    void NotifyWalletChanged();

    // Thread safety
    mutable std::mutex mutex_;
    std::vector<StateObserver*> observers_;

    // Cached state
    std::vector<JobState> jobs_;
    std::vector<DeploymentState> deployments_;
    SystemMetrics metrics_;
    EarningsInfo earnings_today_;
    EarningsInfo earnings_week_;
    EarningsInfo earnings_month_;
    ConnectionStatus connection_status_ = ConnectionStatus::Disconnected;
    std::string wallet_address_;
    double wallet_balance_ = 0.0;
};

} // namespace cyxwiz::servernode::core
