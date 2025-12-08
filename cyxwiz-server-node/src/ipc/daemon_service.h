// daemon_service.h - gRPC service for daemon management (IPC)
#pragma once

#include <grpcpp/grpcpp.h>
#include <daemon.grpc.pb.h>
#include <memory>
#include <string>
#include <atomic>
#include <mutex>
#include <vector>
#include <functional>

namespace cyxwiz::servernode {

class JobExecutor;
class DeploymentManager;
class NodeClient;

namespace core {
    class MetricsCollector;
    class StateManager;
    class ConfigManager;
}

namespace ipc {

// Callback for shutdown request from GUI
using ShutdownCallback = std::function<void(bool graceful)>;

class DaemonServiceImpl final : public cyxwiz::daemon::DaemonService::Service {
public:
    DaemonServiceImpl(
        const std::string& node_id,
        JobExecutor* job_executor,
        DeploymentManager* deployment_manager,
        NodeClient* node_client,
        core::MetricsCollector* metrics,
        core::StateManager* state,
        core::ConfigManager* config
    );

    ~DaemonServiceImpl();

    // Set shutdown callback
    void SetShutdownCallback(ShutdownCallback callback);

    // Start the gRPC server
    bool Start(const std::string& address);
    void Stop();
    bool IsRunning() const { return running_.load(); }

    // gRPC Service Methods - Status & Metrics
    grpc::Status GetStatus(grpc::ServerContext* context,
        const cyxwiz::daemon::GetStatusRequest* request,
        cyxwiz::daemon::GetStatusResponse* response) override;

    grpc::Status GetMetrics(grpc::ServerContext* context,
        const cyxwiz::daemon::GetMetricsRequest* request,
        cyxwiz::daemon::GetMetricsResponse* response) override;

    grpc::Status StreamMetrics(grpc::ServerContext* context,
        const cyxwiz::daemon::StreamMetricsRequest* request,
        grpc::ServerWriter<cyxwiz::daemon::MetricsUpdate>* writer) override;

    // Jobs
    grpc::Status ListJobs(grpc::ServerContext* context,
        const cyxwiz::daemon::ListJobsRequest* request,
        cyxwiz::daemon::ListJobsResponse* response) override;

    grpc::Status GetJob(grpc::ServerContext* context,
        const cyxwiz::daemon::GetJobRequest* request,
        cyxwiz::daemon::GetJobResponse* response) override;

    grpc::Status CancelJob(grpc::ServerContext* context,
        const cyxwiz::daemon::CancelJobRequest* request,
        cyxwiz::daemon::CancelJobResponse* response) override;

    grpc::Status StreamJobUpdates(grpc::ServerContext* context,
        const cyxwiz::daemon::StreamJobUpdatesRequest* request,
        grpc::ServerWriter<cyxwiz::daemon::JobUpdate>* writer) override;

    // Deployments
    grpc::Status ListDeployments(grpc::ServerContext* context,
        const cyxwiz::daemon::ListDeploymentsRequest* request,
        cyxwiz::daemon::ListDeploymentsResponse* response) override;

    grpc::Status DeployModel(grpc::ServerContext* context,
        const cyxwiz::daemon::DeployModelRequest* request,
        cyxwiz::daemon::DeployModelResponse* response) override;

    grpc::Status UndeployModel(grpc::ServerContext* context,
        const cyxwiz::daemon::UndeployModelRequest* request,
        cyxwiz::daemon::UndeployModelResponse* response) override;

    grpc::Status GetDeploymentStatus(grpc::ServerContext* context,
        const cyxwiz::daemon::GetDeploymentStatusRequest* request,
        cyxwiz::daemon::GetDeploymentStatusResponse* response) override;

    // Models
    grpc::Status ListLocalModels(grpc::ServerContext* context,
        const cyxwiz::daemon::ListLocalModelsRequest* request,
        cyxwiz::daemon::ListLocalModelsResponse* response) override;

    grpc::Status ScanModels(grpc::ServerContext* context,
        const cyxwiz::daemon::ScanModelsRequest* request,
        cyxwiz::daemon::ScanModelsResponse* response) override;

    grpc::Status DeleteModel(grpc::ServerContext* context,
        const cyxwiz::daemon::DeleteModelRequest* request,
        cyxwiz::daemon::DeleteModelResponse* response) override;

    // API Keys
    grpc::Status ListAPIKeys(grpc::ServerContext* context,
        const cyxwiz::daemon::ListAPIKeysRequest* request,
        cyxwiz::daemon::ListAPIKeysResponse* response) override;

    grpc::Status CreateAPIKey(grpc::ServerContext* context,
        const cyxwiz::daemon::CreateAPIKeyRequest* request,
        cyxwiz::daemon::CreateAPIKeyResponse* response) override;

    grpc::Status RevokeAPIKey(grpc::ServerContext* context,
        const cyxwiz::daemon::RevokeAPIKeyRequest* request,
        cyxwiz::daemon::RevokeAPIKeyResponse* response) override;

    // Configuration
    grpc::Status GetConfig(grpc::ServerContext* context,
        const cyxwiz::daemon::GetConfigRequest* request,
        cyxwiz::daemon::GetConfigResponse* response) override;

    grpc::Status SetConfig(grpc::ServerContext* context,
        const cyxwiz::daemon::SetConfigRequest* request,
        cyxwiz::daemon::SetConfigResponse* response) override;

    // Earnings & Wallet
    grpc::Status GetEarnings(grpc::ServerContext* context,
        const cyxwiz::daemon::GetEarningsRequest* request,
        cyxwiz::daemon::GetEarningsResponse* response) override;

    grpc::Status GetWalletInfo(grpc::ServerContext* context,
        const cyxwiz::daemon::GetWalletInfoRequest* request,
        cyxwiz::daemon::GetWalletInfoResponse* response) override;

    grpc::Status SetWalletAddress(grpc::ServerContext* context,
        const cyxwiz::daemon::SetWalletAddressRequest* request,
        cyxwiz::daemon::SetWalletAddressResponse* response) override;

    // Logs
    grpc::Status GetLogs(grpc::ServerContext* context,
        const cyxwiz::daemon::GetLogsRequest* request,
        cyxwiz::daemon::GetLogsResponse* response) override;

    grpc::Status StreamLogs(grpc::ServerContext* context,
        const cyxwiz::daemon::StreamLogsRequest* request,
        grpc::ServerWriter<cyxwiz::daemon::LogEntry>* writer) override;

    // Pool Mining
    grpc::Status GetPoolStatus(grpc::ServerContext* context,
        const cyxwiz::daemon::GetPoolStatusRequest* request,
        cyxwiz::daemon::GetPoolStatusResponse* response) override;

    grpc::Status JoinPool(grpc::ServerContext* context,
        const cyxwiz::daemon::JoinPoolRequest* request,
        cyxwiz::daemon::JoinPoolResponse* response) override;

    grpc::Status LeavePool(grpc::ServerContext* context,
        const cyxwiz::daemon::LeavePoolRequest* request,
        cyxwiz::daemon::LeavePoolResponse* response) override;

    grpc::Status SetMiningIntensity(grpc::ServerContext* context,
        const cyxwiz::daemon::SetMiningIntensityRequest* request,
        cyxwiz::daemon::SetMiningIntensityResponse* response) override;

    // Daemon Control
    grpc::Status Shutdown(grpc::ServerContext* context,
        const cyxwiz::daemon::ShutdownRequest* request,
        cyxwiz::daemon::ShutdownResponse* response) override;

    grpc::Status Restart(grpc::ServerContext* context,
        const cyxwiz::daemon::RestartRequest* request,
        cyxwiz::daemon::RestartResponse* response) override;

private:
    std::string node_id_;
    JobExecutor* job_executor_;
    DeploymentManager* deployment_manager_;
    NodeClient* node_client_;
    core::MetricsCollector* metrics_;
    core::StateManager* state_;
    core::ConfigManager* config_;

    std::unique_ptr<grpc::Server> server_;
    std::atomic<bool> running_{false};
    int64_t start_time_;

    ShutdownCallback shutdown_callback_;
    std::mutex callback_mutex_;

    // Log buffer for streaming
    std::vector<std::pair<int64_t, std::string>> log_buffer_;
    std::mutex log_mutex_;
    static constexpr size_t MAX_LOG_BUFFER = 1000;
};

} // namespace ipc
} // namespace cyxwiz::servernode
