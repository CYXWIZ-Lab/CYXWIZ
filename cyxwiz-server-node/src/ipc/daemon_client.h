// daemon_client.h - Client for connecting GUI/TUI to daemon via gRPC
#pragma once

#include <grpcpp/grpcpp.h>
#include <memory>
#include <string>
#include <vector>
#include <functional>
#include <atomic>
#include <thread>
#include <mutex>

namespace cyxwiz::servernode::ipc {

// Per-GPU metrics matching GPUMetrics proto message
struct GPUInfo {
    int device_id = 0;
    std::string name;
    std::string vendor;  // "NVIDIA", "Intel", "AMD"
    float usage_3d = 0.0f;
    float usage_copy = 0.0f;
    float usage_video_decode = 0.0f;
    float usage_video_encode = 0.0f;
    float memory_usage = 0.0f;
    uint64_t vram_used = 0;
    uint64_t vram_total = 0;
    float temperature = 0.0f;
    float power_watts = 0.0f;
    bool is_nvidia = false;
};

// Data structures matching daemon.proto (avoids exposing protobuf in headers)
struct SystemMetrics {
    float cpu_usage = 0.0f;
    float gpu_usage = 0.0f;
    float ram_usage = 0.0f;
    float vram_usage = 0.0f;
    uint64_t ram_total = 0;
    uint64_t ram_used = 0;
    uint64_t vram_total = 0;
    uint64_t vram_used = 0;
    float network_rx_mbps = 0.0f;
    float network_tx_mbps = 0.0f;
    std::vector<GPUInfo> gpus;  // Per-GPU metrics
    int gpu_count = 0;
};

struct DaemonStatus {
    std::string node_id;
    std::string version;
    bool connected_to_central = false;
    bool auth_required = false;  // Token invalid/revoked - user needs to re-login
    int64_t uptime_seconds = 0;
    int active_jobs = 0;
    int active_deployments = 0;
    SystemMetrics metrics;
    std::string gpu_name;
    int gpu_count = 0;
};

struct JobInfo {
    std::string id;
    std::string type;
    int status = 0;  // 1=pending, 2=running, 3=paused, 4=completed, 5=failed, 6=cancelled
    float progress = 0.0f;
    int current_epoch = 0;
    int total_epochs = 0;
    float loss = 0.0f;
    float accuracy = 0.0f;
    int64_t started_at = 0;
    std::string model_name;
    double earnings = 0.0;
};

struct DeploymentInfo {
    std::string id;
    std::string model_name;
    std::string model_path;
    std::string format;
    int status = 0;  // 1=loading, 2=running, 3=stopped, 4=error
    int port = 0;
    int gpu_layers = 0;
    int context_size = 0;
    int64_t request_count = 0;
    double earnings = 0.0;
};

struct ModelInfo {
    std::string name;
    std::string path;
    std::string format;
    int64_t size_bytes = 0;
    int64_t modified_at = 0;
    bool is_deployed = false;
    std::string architecture;
};

struct APIKeyInfo {
    std::string id;
    std::string name;
    std::string key_prefix;
    int64_t created_at = 0;
    int64_t last_used_at = 0;
    int64_t request_count = 0;
    int rate_limit_rpm = 0;
    bool is_active = true;
};

struct EarningsInfo {
    double today = 0.0;
    double this_week = 0.0;
    double this_month = 0.0;
    double all_time = 0.0;
    double pending_payout = 0.0;
    int64_t jobs_completed = 0;
};

struct NodeConfig {
    std::string node_name;
    std::string central_server_address;
    std::vector<std::string> model_directories;
    int max_concurrent_jobs = 4;
    int default_gpu_layers = 35;
    int default_context_size = 4096;
    std::string log_level = "info";
};

struct LogEntry {
    int64_t timestamp = 0;
    std::string level;
    std::string message;
    std::string source;
};

// Pool Mining
struct MiningStats {
    float hashrate_mhs = 0.0f;
    int64_t shares_submitted = 0;
    int64_t shares_accepted = 0;
    int64_t shares_rejected = 0;
    double estimated_daily = 0.0;
    double estimated_monthly = 0.0;
    int64_t mining_uptime_seconds = 0;
};

struct PoolStatus {
    std::string pool_id;
    std::string pool_name;
    std::string pool_address;
    bool is_joined = false;
    bool is_mining = false;
    float mining_intensity = 0.5f;
    double pool_earnings = 0.0;
    int active_miners = 0;
    float pool_hashrate = 0.0f;
    MiningStats stats;
};

// Marketplace
enum class ModelCategory {
    All = 0,
    LLM = 1,
    Vision = 2,
    Audio = 3,
    Embedding = 4,
    Multimodal = 5
};

struct MarketplaceListing {
    std::string id;
    std::string name;
    std::string description;
    std::string format;
    int64_t size_bytes = 0;
    double price_per_request = 0.0;
    float rating = 0.0f;
    int64_t download_count = 0;
    std::string owner_id;
    std::vector<std::string> tags;
    ModelCategory category = ModelCategory::All;
    std::string architecture;
    int64_t parameter_count = 0;
    std::string thumbnail_url;
    int64_t created_at = 0;
};

struct DownloadProgress {
    std::string model_id;
    int64_t bytes_downloaded = 0;
    int64_t total_bytes = 0;
    float progress = 0.0f;
    bool completed = false;
    std::string error_message;
    std::string local_path;
};

// Device Allocation for Central Server connection
enum class AllocDeviceType {
    Gpu = 0,
    Cpu = 1
};

enum class AllocPriority {
    Low = 0,
    Medium = 1,
    High = 2
};

struct DeviceAllocationInfo {
    AllocDeviceType device_type = AllocDeviceType::Gpu;
    int device_id = 0;
    std::string device_name;         // Device name (e.g., "NVIDIA GeForce GTX 1050 Ti")
    bool is_enabled = true;
    int vram_total_mb = 0;           // Total VRAM in MB (for GPU)
    int vram_allocation_mb = 0;      // Allocated VRAM in MB (for GPU)
    int cpu_cores_allocation = 0;    // For CPU
    AllocPriority priority = AllocPriority::Medium;
};

struct SetAllocationsResult {
    bool success = false;
    std::string message;
    bool connected_to_central = false;
    std::string node_id;
};

struct RetryConnectionResult {
    bool success = false;
    std::string message;
    bool connected = false;
    std::string node_id;
};

struct DisconnectResult {
    bool success = false;
    std::string message;
};

// Callbacks for streaming updates
using MetricsCallback = std::function<void(const SystemMetrics&)>;
using JobUpdateCallback = std::function<void(const std::string& job_id, const JobInfo& job, const std::string& update_type)>;
using LogCallback = std::function<void(const LogEntry&)>;
using DownloadCallback = std::function<void(const DownloadProgress&)>;

// TLS connection settings for remote daemon connections
struct TLSConnectionSettings {
    bool enabled = false;              // Enable TLS encryption
    std::string ca_cert_path;          // Path to CA certificate (required for TLS)
    std::string client_cert_path;      // Path to client certificate (optional, for mTLS)
    std::string client_key_path;       // Path to client private key (optional, for mTLS)
    std::string target_name_override;  // Override server name for verification (optional)
    bool skip_verification = false;    // Skip server cert verification (development only!)
};

class DaemonClient {
public:
    DaemonClient();
    ~DaemonClient();

    // Connection management
    bool Connect(const std::string& address);

    // Connect with TLS settings (for remote secure connections)
    bool Connect(const std::string& address, const TLSConnectionSettings& tls_settings);

    void Disconnect();
    bool IsConnected() const { return connected_.load(); }
    bool IsTLSEnabled() const { return tls_enabled_.load(); }
    std::string GetAddress() const { return address_; }
    void SetTargetAddress(const std::string& address) { address_ = address; }

    // Async connection - returns immediately, connects in background
    void ConnectAsync(const std::string& address);
    
    // Test connection without fully connecting (for settings validation)
    static bool TestConnection(const std::string& address,
                               const TLSConnectionSettings& tls_settings,
                               std::string& error_message,
                               int timeout_seconds = 5);

    // Status & Metrics
    bool GetStatus(DaemonStatus& status);
    bool GetMetrics(SystemMetrics& metrics,
                    std::vector<float>& cpu_history,
                    std::vector<float>& gpu_history,
                    std::vector<float>& ram_history,
                    std::vector<float>& vram_history);
    void StartMetricsStream(MetricsCallback callback, int interval_ms = 1000);
    void StopMetricsStream();

    // Jobs
    bool ListJobs(std::vector<JobInfo>& jobs, bool include_completed = false);
    bool GetJob(const std::string& job_id, JobInfo& job);
    bool CancelJob(const std::string& job_id, std::string& error);
    void StartJobUpdatesStream(JobUpdateCallback callback);
    void StopJobUpdatesStream();

    // Deployments
    bool ListDeployments(std::vector<DeploymentInfo>& deployments);
    bool DeployModel(const std::string& model_path, int port, int gpu_layers,
                     int context_size, std::string& deployment_id, std::string& error);
    bool UndeployModel(const std::string& deployment_id, std::string& error);

    // Models
    bool ListLocalModels(std::vector<ModelInfo>& models, const std::string& directory = "");
    bool ScanModels(const std::vector<std::string>& directories, int& models_found);
    bool DeleteModel(const std::string& model_path, std::string& error);

    // API Keys
    bool ListAPIKeys(std::vector<APIKeyInfo>& keys);
    bool CreateAPIKey(const std::string& name, int rate_limit_rpm,
                      std::string& full_key, std::string& error);
    bool RevokeAPIKey(const std::string& key_id, std::string& error);

    // Configuration
    bool GetConfig(NodeConfig& config);
    bool SetConfig(const NodeConfig& config, bool& restart_required, std::string& error);

    // Earnings & Wallet
    bool GetEarnings(EarningsInfo& earnings);
    bool GetWalletAddress(std::string& address, double& balance, bool& is_connected);
    bool SetWalletAddress(const std::string& address, std::string& error);

    // Logs
    bool GetLogs(std::vector<LogEntry>& entries, int limit = 100,
                 const std::string& level_filter = "");
    void StartLogStream(LogCallback callback, const std::string& level_filter = "");
    void StopLogStream();

    // Pool Mining
    bool GetPoolStatus(PoolStatus& status);
    bool JoinPool(const std::string& pool_address, std::string& error);
    bool LeavePool(std::string& error);
    bool SetMiningIntensity(float intensity, std::string& error);
    bool StartMining(std::string& error);
    bool StopMining(std::string& error);

    // Marketplace
    bool ListMarketplaceModels(std::vector<MarketplaceListing>& listings,
                               const std::string& query = "",
                               ModelCategory category = ModelCategory::All,
                               int limit = 50, int offset = 0,
                               const std::string& sort_by = "rating");
    bool GetMarketplaceModel(const std::string& model_id, MarketplaceListing& listing);
    void DownloadMarketplaceModel(const std::string& model_id,
                                  const std::string& target_dir,
                                  DownloadCallback callback);
    void CancelMarketplaceDownload(const std::string& model_id);

    // Daemon Control
    bool Shutdown(bool graceful, std::string& error);
    bool Restart(std::string& error);

    // Resource Allocation & Central Server Connection
    SetAllocationsResult SetAllocations(const std::vector<DeviceAllocationInfo>& allocations,
                                         const std::string& jwt_token,
                                         bool connect_to_central);
    RetryConnectionResult RetryConnection();
    DisconnectResult DisconnectFromCentral();

private:
    std::shared_ptr<grpc::Channel> channel_;
    std::unique_ptr<class DaemonServiceStub> stub_;
    std::string address_;
    std::atomic<bool> connected_{false};
    std::atomic<bool> tls_enabled_{false};
    TLSConnectionSettings tls_settings_;

    // Streaming threads
    std::thread metrics_stream_thread_;
    std::thread job_updates_thread_;
    std::thread log_stream_thread_;
    std::thread download_thread_;
    std::atomic<bool> metrics_streaming_{false};
    std::atomic<bool> job_updates_streaming_{false};
    std::atomic<bool> log_streaming_{false};
    std::atomic<bool> download_active_{false};
    std::string active_download_model_id_;

    std::mutex stream_mutex_;
};

} // namespace cyxwiz::servernode::ipc
