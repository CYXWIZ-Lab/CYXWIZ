// deployment_dialog.h - UI dialog for deploying models
#pragma once

#include "../panel.h"
#include "../../network/deployment_client.h"
#include "../../inference/local_inference_server.h"
#include <string>
#include <memory>
#include <atomic>
#include <thread>
#include <mutex>
#include <chrono>

namespace cyxwiz {

/**
 * Deployment mode selection
 */
enum class DeploymentMode {
    Embedded,    // Run inference server in-process (no daemon needed)
    ServerNode   // Deploy to Server Node daemon via gRPC
};

/**
 * DeploymentDialog - UI for deploying trained models
 *
 * Supports two modes:
 * 1. Embedded: Runs HTTP inference server directly in the Engine process
 * 2. Server Node: Deploys to a Server Node daemon via gRPC
 */
class DeploymentDialog : public Panel {
public:
    DeploymentDialog();
    ~DeploymentDialog() override;

    void Render() override;

    // Dialog control
    void Open();
    void Close();
    bool IsOpen() const { return is_open_; }

    // Pre-fill model path
    void SetModelPath(const std::string& path);

    // Access to embedded server (for status display in main window)
    LocalInferenceServer* GetEmbeddedServer() { return embedded_server_.get(); }

    // Check if any deployment is active
    bool HasActiveDeployment() const;

private:
    // Render sections
    void RenderModeSelector();
    void RenderModelSection();
    void RenderEmbeddedConfig();
    void RenderServerNodeConfig();
    void RenderGGUFConfig();  // GGUF/LLM-specific options
    void RenderDeployButton();
    void RenderActiveDeployments();
    void RenderStatusMessages();

    // Actions
    void StartEmbeddedDeployment();
    void StopEmbeddedDeployment();
    void StartServerNodeDeployment();
    void StopServerNodeDeployment(const std::string& deployment_id);
    void DeleteServerNodeDeployment(const std::string& deployment_id);
    void ConnectToServerNode();
    void DisconnectFromServerNode();

    // File dialog helpers
    std::string OpenModelFile();    // For binary .cyxmodel files
    std::string OpenModelFolder();  // For directory .cyxmodel folders

private:
    // Dialog state
    bool is_open_ = false;
    DeploymentMode mode_ = DeploymentMode::Embedded;

    // Model settings
    char model_path_[512] = "";

    // Embedded mode config
    int embedded_port_ = 8080;

    // Server Node mode config
    char server_address_[256] = "localhost:50055";
    int server_port_ = 8080;
    bool enable_terminal_ = false;

    // GGUF/LLM mode config (shown when .gguf model detected)
    bool is_gguf_model_ = false;           // Auto-detected from file extension
    int n_gpu_layers_ = 0;                 // GPU offloading (0 = CPU only)
    int context_size_ = 2048;              // Context window size
    float temperature_ = 0.8f;             // Sampling temperature
    int max_tokens_ = 256;                 // Max generation length
    float top_p_ = 0.95f;                  // Top-p (nucleus) sampling
    int top_k_ = 40;                       // Top-k sampling
    float repeat_penalty_ = 1.1f;          // Repetition penalty
    bool enable_embeddings_ = false;       // Run in embedding mode

    // Embedded server instance
    std::unique_ptr<LocalInferenceServer> embedded_server_;

    // Server Node client
    std::unique_ptr<network::DeploymentClient> deployment_client_;

    // Server Node deployments
    std::vector<network::DeploymentSummary> server_deployments_;
    std::string active_server_deployment_id_;

    // Status
    std::string status_message_;
    std::string error_message_;
    std::atomic<bool> is_deploying_{false};

    // Thread for async operations
    std::unique_ptr<std::thread> operation_thread_;
    std::mutex state_mutex_;

    // Polling control for Server Node deployments
    std::chrono::steady_clock::time_point last_refresh_time_;
    static constexpr float refresh_interval_seconds_ = 2.0f;  // Refresh every 2 seconds
    bool last_refresh_failed_ = false;  // Backoff on failures
};

} // namespace cyxwiz
