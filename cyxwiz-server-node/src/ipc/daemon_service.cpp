// daemon_service.cpp - gRPC service implementation for daemon management
#include "ipc/daemon_service.h"

// Generated protobuf/gRPC headers
#include <daemon.grpc.pb.h>
#include <daemon.pb.h>
#include "job_executor.h"
#include "deployment_manager.h"
#include "node_client.h"
#include "core/metrics_collector.h"
#include "core/state_manager.h"
#include "core/config_manager.h"

#include <spdlog/spdlog.h>
#include <chrono>
#include <thread>

namespace cyxwiz::servernode::ipc {

using namespace cyxwiz::daemon;

DaemonServiceImpl::DaemonServiceImpl(
    const std::string& node_id,
    JobExecutor* job_executor,
    DeploymentManager* deployment_manager,
    NodeClient* node_client,
    core::MetricsCollector* metrics,
    core::StateManager* state,
    core::ConfigManager* config)
    : node_id_(node_id)
    , job_executor_(job_executor)
    , deployment_manager_(deployment_manager)
    , node_client_(node_client)
    , metrics_(metrics)
    , state_(state)
    , config_(config)
    , start_time_(std::chrono::system_clock::now().time_since_epoch().count()) {
}

DaemonServiceImpl::~DaemonServiceImpl() {
    Stop();
}

void DaemonServiceImpl::SetShutdownCallback(ShutdownCallback callback) {
    std::lock_guard<std::mutex> lock(callback_mutex_);
    shutdown_callback_ = std::move(callback);
}

bool DaemonServiceImpl::Start(const std::string& address) {
    if (running_.load()) {
        spdlog::warn("Daemon service already running");
        return false;
    }

    grpc::ServerBuilder builder;
    builder.AddListeningPort(address, grpc::InsecureServerCredentials());
    builder.RegisterService(this);

    server_ = builder.BuildAndStart();
    if (!server_) {
        spdlog::error("Failed to start daemon service on {}", address);
        return false;
    }

    running_.store(true);
    spdlog::info("Daemon IPC service started on {}", address);
    return true;
}

void DaemonServiceImpl::Stop() {
    if (!running_.load()) return;

    running_.store(false);
    if (server_) {
        server_->Shutdown();
        server_.reset();
    }
    spdlog::info("Daemon IPC service stopped");
}

// ============================================================================
// Status & Metrics
// ============================================================================

grpc::Status DaemonServiceImpl::GetStatus(
    grpc::ServerContext* context,
    const GetStatusRequest* request,
    GetStatusResponse* response) {

    response->set_node_id(node_id_);
    response->set_version("0.3.0");

    // Connection status - check actual registration state
    if (node_client_ && node_client_->IsRegistered()) {
        response->set_central_server_status(ConnectionStatus::CONNECTION_STATUS_CONNECTED);
    } else {
        response->set_central_server_status(ConnectionStatus::CONNECTION_STATUS_DISCONNECTED);
    }

    // Uptime
    auto now = std::chrono::system_clock::now().time_since_epoch().count();
    response->set_uptime_seconds((now - start_time_) / 1000000000);

    // Job/deployment counts
    if (job_executor_) {
        response->set_active_jobs(job_executor_->GetActiveJobCount());
    }
    if (deployment_manager_) {
        response->set_active_deployments(deployment_manager_->GetActiveDeploymentCount());
    }

    // Metrics
    if (metrics_) {
        auto m = metrics_->GetCurrentMetrics();
        auto* metrics_pb = response->mutable_metrics();
        metrics_pb->set_cpu_usage(m.cpu_usage);
        metrics_pb->set_gpu_usage(m.gpu_usage);
        metrics_pb->set_ram_usage(m.ram_usage);
        metrics_pb->set_vram_usage(m.vram_usage);
        metrics_pb->set_ram_total_bytes(m.ram_total_bytes);
        metrics_pb->set_ram_used_bytes(m.ram_used_bytes);
        metrics_pb->set_vram_total_bytes(m.vram_total_bytes);
        metrics_pb->set_vram_used_bytes(m.vram_used_bytes);
        metrics_pb->set_gpu_count(m.gpu_count);

        // Per-GPU metrics
        for (const auto& gpu : m.gpus) {
            auto* gpu_pb = metrics_pb->add_gpus();
            gpu_pb->set_device_id(gpu.device_id);
            gpu_pb->set_name(gpu.name);
            gpu_pb->set_vendor(gpu.vendor);
            gpu_pb->set_usage_3d(gpu.usage_3d);
            gpu_pb->set_usage_copy(gpu.usage_copy);
            gpu_pb->set_usage_video_decode(gpu.usage_video_decode);
            gpu_pb->set_usage_video_encode(gpu.usage_video_encode);
            gpu_pb->set_memory_usage(gpu.memory_usage);
            gpu_pb->set_vram_used_bytes(gpu.vram_used_bytes);
            gpu_pb->set_vram_total_bytes(gpu.vram_total_bytes);
            gpu_pb->set_temperature_celsius(gpu.temperature_celsius);
            gpu_pb->set_power_watts(gpu.power_watts);
            gpu_pb->set_is_nvidia(gpu.is_nvidia);
        }

        // GPU info from first GPU or default
        if (!m.gpus.empty()) {
            response->set_gpu_name(m.gpus[0].name);
            response->set_gpu_count(m.gpu_count);
        } else {
            response->set_gpu_name("Unknown GPU");
            response->set_gpu_count(m.gpu_count);
        }
    } else {
        // GPU info fallback
        response->set_gpu_name("Unknown GPU");
        response->set_gpu_count(0);
    }

    return grpc::Status::OK;
}

grpc::Status DaemonServiceImpl::GetMetrics(
    grpc::ServerContext* context,
    const GetMetricsRequest* request,
    GetMetricsResponse* response) {

    if (!metrics_) {
        return grpc::Status(grpc::StatusCode::UNAVAILABLE, "Metrics not available");
    }

    auto m = metrics_->GetCurrentMetrics();
    auto* metrics_pb = response->mutable_metrics();
    metrics_pb->set_cpu_usage(m.cpu_usage);
    metrics_pb->set_gpu_usage(m.gpu_usage);
    metrics_pb->set_ram_usage(m.ram_usage);
    metrics_pb->set_vram_usage(m.vram_usage);
    metrics_pb->set_ram_total_bytes(m.ram_total_bytes);
    metrics_pb->set_ram_used_bytes(m.ram_used_bytes);
    metrics_pb->set_vram_total_bytes(m.vram_total_bytes);
    metrics_pb->set_vram_used_bytes(m.vram_used_bytes);
    metrics_pb->set_gpu_count(m.gpu_count);

    // Per-GPU metrics
    for (const auto& gpu : m.gpus) {
        auto* gpu_pb = metrics_pb->add_gpus();
        gpu_pb->set_device_id(gpu.device_id);
        gpu_pb->set_name(gpu.name);
        gpu_pb->set_vendor(gpu.vendor);
        gpu_pb->set_usage_3d(gpu.usage_3d);
        gpu_pb->set_usage_copy(gpu.usage_copy);
        gpu_pb->set_usage_video_decode(gpu.usage_video_decode);
        gpu_pb->set_usage_video_encode(gpu.usage_video_encode);
        gpu_pb->set_memory_usage(gpu.memory_usage);
        gpu_pb->set_vram_used_bytes(gpu.vram_used_bytes);
        gpu_pb->set_vram_total_bytes(gpu.vram_total_bytes);
        gpu_pb->set_temperature_celsius(gpu.temperature_celsius);
        gpu_pb->set_power_watts(gpu.power_watts);
        gpu_pb->set_is_nvidia(gpu.is_nvidia);
    }

    // History (get 60 samples for graph)
    auto cpu_hist = metrics_->GetHistory(core::MetricType::CPU, 60);
    auto gpu_hist = metrics_->GetHistory(core::MetricType::GPU, 60);
    auto ram_hist = metrics_->GetHistory(core::MetricType::RAM, 60);
    auto vram_hist = metrics_->GetHistory(core::MetricType::VRAM, 60);

    for (float v : cpu_hist) response->add_cpu_history(v);
    for (float v : gpu_hist) response->add_gpu_history(v);
    for (float v : ram_hist) response->add_ram_history(v);
    for (float v : vram_hist) response->add_vram_history(v);

    return grpc::Status::OK;
}

grpc::Status DaemonServiceImpl::StreamMetrics(
    grpc::ServerContext* context,
    const StreamMetricsRequest* request,
    grpc::ServerWriter<MetricsUpdate>* writer) {

    int interval_ms = request->interval_ms() > 0 ? request->interval_ms() : 1000;

    while (!context->IsCancelled() && running_.load()) {
        if (metrics_) {
            auto m = metrics_->GetCurrentMetrics();

            MetricsUpdate update;
            update.set_timestamp(std::chrono::system_clock::now().time_since_epoch().count());

            auto* metrics_pb = update.mutable_metrics();
            metrics_pb->set_cpu_usage(m.cpu_usage);
            metrics_pb->set_gpu_usage(m.gpu_usage);
            metrics_pb->set_ram_usage(m.ram_usage);
            metrics_pb->set_vram_usage(m.vram_usage);

            if (!writer->Write(update)) {
                break;
            }
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(interval_ms));
    }

    return grpc::Status::OK;
}

// ============================================================================
// Jobs
// ============================================================================

grpc::Status DaemonServiceImpl::ListJobs(
    grpc::ServerContext* context,
    const ListJobsRequest* request,
    ListJobsResponse* response) {

    if (!state_) {
        return grpc::Status(grpc::StatusCode::UNAVAILABLE, "State not available");
    }

    auto jobs = state_->GetActiveJobs();
    for (const auto& job : jobs) {
        auto* job_pb = response->add_jobs();
        job_pb->set_id(job.id);
        job_pb->set_type(job.type);
        // Determine status from is_running/is_paused flags
        if (job.is_running && !job.is_paused) {
            job_pb->set_status(JobStatus::JOB_STATUS_RUNNING);
        } else if (job.is_paused) {
            job_pb->set_status(JobStatus::JOB_STATUS_PAUSED);
        } else {
            job_pb->set_status(JobStatus::JOB_STATUS_PENDING);
        }
        job_pb->set_progress(job.progress);
        job_pb->set_current_epoch(job.current_epoch);
        job_pb->set_total_epochs(job.total_epochs);
        job_pb->set_loss(job.loss);
        job_pb->set_accuracy(job.accuracy);
        // Convert time_point to timestamp
        auto started_ts = std::chrono::duration_cast<std::chrono::seconds>(
            job.start_time.time_since_epoch()).count();
        job_pb->set_started_at(started_ts);
        // Model name not available in JobState, use id as fallback
        job_pb->set_model_name(job.id);
    }

    return grpc::Status::OK;
}

grpc::Status DaemonServiceImpl::GetJob(
    grpc::ServerContext* context,
    const GetJobRequest* request,
    GetJobResponse* response) {

    // TODO: Implement job lookup
    return grpc::Status(grpc::StatusCode::UNIMPLEMENTED, "Not implemented");
}

grpc::Status DaemonServiceImpl::CancelJob(
    grpc::ServerContext* context,
    const CancelJobRequest* request,
    CancelJobResponse* response) {

    if (!job_executor_) {
        response->set_success(false);
        response->set_error_message("Job executor not available");
        return grpc::Status::OK;
    }

    bool success = job_executor_->CancelJob(request->job_id());
    response->set_success(success);
    if (!success) {
        response->set_error_message("Failed to cancel job");
    }

    return grpc::Status::OK;
}

grpc::Status DaemonServiceImpl::StreamJobUpdates(
    grpc::ServerContext* context,
    const StreamJobUpdatesRequest* request,
    grpc::ServerWriter<JobUpdate>* writer) {

    // TODO: Implement job update streaming via StateManager observer
    while (!context->IsCancelled() && running_.load()) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }

    return grpc::Status::OK;
}

// ============================================================================
// Deployments
// ============================================================================

grpc::Status DaemonServiceImpl::ListDeployments(
    grpc::ServerContext* context,
    const ListDeploymentsRequest* request,
    ListDeploymentsResponse* response) {

    if (!state_) {
        return grpc::Status(grpc::StatusCode::UNAVAILABLE, "State not available");
    }

    auto deployments = state_->GetDeployments();
    for (const auto& dep : deployments) {
        auto* dep_pb = response->add_deployments();
        dep_pb->set_id(dep.id);
        dep_pb->set_model_name(dep.model_name);
        dep_pb->set_model_path(dep.model_id);  // Use model_id as path
        dep_pb->set_format(dep.format);
        dep_pb->set_port(dep.port);
        dep_pb->set_request_count(dep.request_count);

        if (dep.status == "Running") {
            dep_pb->set_status(DeploymentStatus::DEPLOYMENT_STATUS_RUNNING);
        } else if (dep.status == "Loading") {
            dep_pb->set_status(DeploymentStatus::DEPLOYMENT_STATUS_LOADING);
        } else if (dep.status == "Stopped") {
            dep_pb->set_status(DeploymentStatus::DEPLOYMENT_STATUS_STOPPED);
        } else {
            dep_pb->set_status(DeploymentStatus::DEPLOYMENT_STATUS_ERROR);
        }
    }

    return grpc::Status::OK;
}

grpc::Status DaemonServiceImpl::DeployModel(
    grpc::ServerContext* context,
    const DeployModelRequest* request,
    DeployModelResponse* response) {

    if (!deployment_manager_) {
        response->set_success(false);
        response->set_error_message("Deployment manager not available");
        return grpc::Status::OK;
    }

    // TODO: Call deployment_manager to deploy the model
    response->set_success(true);
    response->set_deployment_id("dep_" + std::to_string(std::time(nullptr)));

    return grpc::Status::OK;
}

grpc::Status DaemonServiceImpl::UndeployModel(
    grpc::ServerContext* context,
    const UndeployModelRequest* request,
    UndeployModelResponse* response) {

    if (!deployment_manager_) {
        response->set_success(false);
        response->set_error_message("Deployment manager not available");
        return grpc::Status::OK;
    }

    // TODO: Call deployment_manager to undeploy
    response->set_success(true);

    return grpc::Status::OK;
}

grpc::Status DaemonServiceImpl::GetDeploymentStatus(
    grpc::ServerContext* context,
    const GetDeploymentStatusRequest* request,
    GetDeploymentStatusResponse* response) {

    // TODO: Implement
    return grpc::Status(grpc::StatusCode::UNIMPLEMENTED, "Not implemented");
}

// ============================================================================
// Models
// ============================================================================

grpc::Status DaemonServiceImpl::ListLocalModels(
    grpc::ServerContext* context,
    const ListLocalModelsRequest* request,
    ListLocalModelsResponse* response) {

    // TODO: Scan model directories
    return grpc::Status::OK;
}

grpc::Status DaemonServiceImpl::ScanModels(
    grpc::ServerContext* context,
    const ScanModelsRequest* request,
    ScanModelsResponse* response) {

    // TODO: Implement model scanning
    response->set_models_found(0);
    return grpc::Status::OK;
}

grpc::Status DaemonServiceImpl::DeleteModel(
    grpc::ServerContext* context,
    const DeleteModelRequest* request,
    DeleteModelResponse* response) {

    // TODO: Implement model deletion
    response->set_success(false);
    response->set_error_message("Not implemented");
    return grpc::Status::OK;
}

// ============================================================================
// API Keys
// ============================================================================

grpc::Status DaemonServiceImpl::ListAPIKeys(
    grpc::ServerContext* context,
    const ListAPIKeysRequest* request,
    ListAPIKeysResponse* response) {

    // TODO: Get from APIKeyManager
    return grpc::Status::OK;
}

grpc::Status DaemonServiceImpl::CreateAPIKey(
    grpc::ServerContext* context,
    const CreateAPIKeyRequest* request,
    CreateAPIKeyResponse* response) {

    // TODO: Create via APIKeyManager
    response->set_success(true);
    response->set_key("cyx_sk_live_" + std::to_string(std::time(nullptr)));

    return grpc::Status::OK;
}

grpc::Status DaemonServiceImpl::RevokeAPIKey(
    grpc::ServerContext* context,
    const RevokeAPIKeyRequest* request,
    RevokeAPIKeyResponse* response) {

    // TODO: Revoke via APIKeyManager
    response->set_success(true);
    return grpc::Status::OK;
}

// ============================================================================
// Configuration
// ============================================================================

grpc::Status DaemonServiceImpl::GetConfig(
    grpc::ServerContext* context,
    const GetConfigRequest* request,
    GetConfigResponse* response) {

    if (!config_) {
        return grpc::Status(grpc::StatusCode::UNAVAILABLE, "Config not available");
    }

    auto cfg = config_->GetConfig();
    response->set_node_name(cfg.node_name);
    response->set_central_server_address(cfg.central_server);
    response->set_max_concurrent_jobs(cfg.max_concurrent_jobs);
    response->set_default_gpu_layers(0);  // Not in NodeConfig, use default
    response->set_default_context_size(2048);  // Not in NodeConfig, use default
    response->set_log_level(cfg.log_level);

    // models_directory is a single path, add it as the only directory
    response->add_model_directories(cfg.models_directory);

    return grpc::Status::OK;
}

grpc::Status DaemonServiceImpl::SetConfig(
    grpc::ServerContext* context,
    const SetConfigRequest* request,
    SetConfigResponse* response) {

    if (!config_) {
        response->set_success(false);
        response->set_error_message("Config not available");
        return grpc::Status::OK;
    }

    // Get current config and update fields from request
    core::NodeConfig cfg = config_->GetConfig();
    cfg.node_name = request->node_name();
    cfg.central_server = request->central_server_address();
    cfg.max_concurrent_jobs = request->max_concurrent_jobs();
    // default_gpu_layers and default_context_size not in NodeConfig, ignore
    cfg.log_level = request->log_level();

    // Use first model directory if provided
    if (request->model_directories_size() > 0) {
        cfg.models_directory = request->model_directories(0);
    }

    config_->SetConfig(cfg);
    config_->Save();

    response->set_success(true);
    response->set_restart_required(false);

    return grpc::Status::OK;
}

// ============================================================================
// Earnings & Wallet
// ============================================================================

grpc::Status DaemonServiceImpl::GetEarnings(
    grpc::ServerContext* context,
    const GetEarningsRequest* request,
    GetEarningsResponse* response) {

    if (!state_) {
        return grpc::Status(grpc::StatusCode::UNAVAILABLE, "State not available");
    }

    auto today = state_->GetEarningsToday();
    auto week = state_->GetEarningsThisWeek();
    auto month = state_->GetEarningsThisMonth();

    auto* earnings = response->mutable_earnings();
    earnings->set_today(today.total_earnings);
    earnings->set_this_week(week.total_earnings);
    earnings->set_this_month(month.total_earnings);
    // jobs_completed not in EarningsInfo, use 0
    earnings->set_jobs_completed(0);

    return grpc::Status::OK;
}

grpc::Status DaemonServiceImpl::GetWalletInfo(
    grpc::ServerContext* context,
    const GetWalletInfoRequest* request,
    GetWalletInfoResponse* response) {

    // TODO: Get from wallet integration
    response->set_wallet_address("");
    response->set_balance(0.0);
    response->set_is_connected(false);

    return grpc::Status::OK;
}

grpc::Status DaemonServiceImpl::SetWalletAddress(
    grpc::ServerContext* context,
    const SetWalletAddressRequest* request,
    SetWalletAddressResponse* response) {

    // TODO: Validate and store wallet address
    response->set_success(true);
    return grpc::Status::OK;
}

// ============================================================================
// Logs
// ============================================================================

grpc::Status DaemonServiceImpl::GetLogs(
    grpc::ServerContext* context,
    const GetLogsRequest* request,
    GetLogsResponse* response) {

    std::lock_guard<std::mutex> lock(log_mutex_);

    int limit = request->limit() > 0 ? request->limit() : 100;
    int count = 0;

    for (auto it = log_buffer_.rbegin(); it != log_buffer_.rend() && count < limit; ++it, ++count) {
        auto* entry = response->add_entries();
        entry->set_timestamp(it->first);
        entry->set_message(it->second);
        entry->set_level("INFO"); // TODO: Parse level from message
    }

    return grpc::Status::OK;
}

grpc::Status DaemonServiceImpl::StreamLogs(
    grpc::ServerContext* context,
    const StreamLogsRequest* request,
    grpc::ServerWriter<cyxwiz::daemon::LogEntry>* writer) {

    // TODO: Implement log streaming with spdlog sink
    while (!context->IsCancelled() && running_.load()) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }

    return grpc::Status::OK;
}

// ============================================================================
// Pool Mining
// ============================================================================

grpc::Status DaemonServiceImpl::GetPoolStatus(
    grpc::ServerContext* context,
    const GetPoolStatusRequest* request,
    GetPoolStatusResponse* response) {

    // TODO: Get from pool mining manager
    response->set_is_mining(false);
    response->set_current_intensity(0.0f);

    return grpc::Status::OK;
}

grpc::Status DaemonServiceImpl::JoinPool(
    grpc::ServerContext* context,
    const JoinPoolRequest* request,
    JoinPoolResponse* response) {

    // TODO: Implement
    response->set_success(false);
    response->set_error_message("Pool mining not implemented");
    return grpc::Status::OK;
}

grpc::Status DaemonServiceImpl::LeavePool(
    grpc::ServerContext* context,
    const LeavePoolRequest* request,
    LeavePoolResponse* response) {

    // TODO: Implement
    response->set_success(true);
    return grpc::Status::OK;
}

grpc::Status DaemonServiceImpl::SetMiningIntensity(
    grpc::ServerContext* context,
    const SetMiningIntensityRequest* request,
    SetMiningIntensityResponse* response) {

    // TODO: Implement
    response->set_success(true);
    return grpc::Status::OK;
}

// ============================================================================
// Daemon Control
// ============================================================================

grpc::Status DaemonServiceImpl::Shutdown(
    grpc::ServerContext* context,
    const ShutdownRequest* request,
    ShutdownResponse* response) {

    spdlog::info("Shutdown requested via IPC (graceful={})", request->graceful());

    {
        std::lock_guard<std::mutex> lock(callback_mutex_);
        if (shutdown_callback_) {
            shutdown_callback_(request->graceful());
        }
    }

    response->set_success(true);
    return grpc::Status::OK;
}

grpc::Status DaemonServiceImpl::Restart(
    grpc::ServerContext* context,
    const RestartRequest* request,
    RestartResponse* response) {

    // TODO: Implement restart logic
    response->set_success(false);
    response->set_error_message("Restart not implemented");
    return grpc::Status::OK;
}

// ============================================================================
// Resource Allocation & Central Server Connection
// ============================================================================

grpc::Status DaemonServiceImpl::SetAllocations(
    grpc::ServerContext* context,
    const SetAllocationsRequest* request,
    SetAllocationsResponse* response) {

    spdlog::info("SetAllocations called with {} allocations, connect_to_central={}",
                 request->allocations_size(), request->connect_to_central());

    // Log the allocations
    for (const auto& alloc : request->allocations()) {
        spdlog::debug("  Device {}: {} enabled={} vram_allocated={} MB",
                      alloc.device_id(),
                      alloc.device_name(),
                      alloc.is_enabled(),
                      alloc.vram_allocated_mb());
    }

    // Store allocations (TODO: Apply to actual device pool)
    // For now, just log them

    // If requested to connect to Central Server
    if (request->connect_to_central()) {
        if (!node_client_) {
            response->set_success(false);
            response->set_error_message("NodeClient not initialized");
            response->set_connected_to_central(false);
            return grpc::Status::OK;
        }

        // Check if already registered - just update allocations and return success
        if (node_client_->IsRegistered()) {
            spdlog::info("Already registered with Central Server, updating allocations only");
            // TODO: Send updated allocations to Central Server via UpdateAllocations RPC
            response->set_success(true);
            response->set_connected_to_central(true);
            response->set_node_id(node_client_->GetNodeId());
            return grpc::Status::OK;
        }

        // Check if JWT token is provided for new registration
        if (request->jwt_token().empty()) {
            response->set_success(false);
            response->set_error_message("JWT token required for Central Server connection");
            response->set_connected_to_central(false);
            return grpc::Status::OK;
        }

        // Set JWT token on NodeClient for authentication
        std::string jwt = request->jwt_token();
        spdlog::info("Setting auth token on NodeClient (token length: {})", jwt.length());
        node_client_->SetAuthToken(jwt);

        // Convert protocol allocations to C++ DeviceAllocation structs
        std::vector<cyxwiz::servernode::DeviceAllocation> device_allocations;
        for (const auto& alloc : request->allocations()) {
            cyxwiz::servernode::DeviceAllocation da;
            // Map daemon DeviceType to our int: CPU=0, CUDA=1, OPENCL=2
            switch (alloc.device_type()) {
                case daemon::DEVICE_TYPE_CPU: da.device_type = 0; break;
                case daemon::DEVICE_TYPE_GPU: da.device_type = 1; break;  // Treat GPU as CUDA
                default: da.device_type = -1; break;
            }
            da.device_id = alloc.device_id();
            da.device_name = alloc.device_name();
            da.is_enabled = alloc.is_enabled();
            da.vram_total_mb = alloc.vram_total_mb();
            da.vram_allocated_mb = alloc.vram_allocated_mb();
            da.cores_allocated = alloc.cores_allocated();
            da.memory_total = alloc.vram_total_mb() * 1024 * 1024;  // Convert MB to bytes
            da.compute_units = 0;  // Not provided by daemon proto
            device_allocations.push_back(da);
        }

        // Attempt registration with device allocations
        spdlog::info("Attempting to register with Central Server with {} allocations...",
                     device_allocations.size());
        if (node_client_->RegisterWithAllocations(device_allocations)) {
            spdlog::info("Successfully registered with Central Server");
            node_client_->StartHeartbeat(10);

            response->set_success(true);
            response->set_connected_to_central(true);
            response->set_node_id(node_client_->GetNodeId());
        } else {
            spdlog::warn("Failed to register with Central Server");
            response->set_success(false);
            response->set_error_message("Failed to connect to Central Server. Check if server is running.");
            response->set_connected_to_central(false);
        }
    } else {
        // Just save allocations locally without connecting
        response->set_success(true);
        response->set_connected_to_central(node_client_ && node_client_->IsRegistered());
        if (node_client_) {
            response->set_node_id(node_client_->GetNodeId());
        }
    }

    return grpc::Status::OK;
}

grpc::Status DaemonServiceImpl::RetryConnection(
    grpc::ServerContext* context,
    const RetryConnectionRequest* request,
    RetryConnectionResponse* response) {

    spdlog::info("RetryConnection called");

    if (!node_client_) {
        response->set_success(false);
        response->set_error_message("NodeClient not initialized");
        response->set_connected_to_central(false);
        return grpc::Status::OK;
    }

    // Check if JWT token is provided
    if (request->jwt_token().empty()) {
        response->set_success(false);
        response->set_error_message("JWT token required for Central Server connection");
        response->set_connected_to_central(false);
        return grpc::Status::OK;
    }

    // TODO: Set JWT token on NodeClient for authentication
    // node_client_->SetAuthToken(request->jwt_token());

    // Attempt registration
    spdlog::info("Retrying connection to Central Server...");
    if (node_client_->Register()) {
        spdlog::info("Successfully registered with Central Server on retry");
        node_client_->StartHeartbeat(10);

        response->set_success(true);
        response->set_connected_to_central(true);
        response->set_node_id(node_client_->GetNodeId());
    } else {
        spdlog::warn("Failed to register with Central Server on retry");
        response->set_success(false);
        response->set_error_message("Failed to connect to Central Server. Please try again later.");
        response->set_connected_to_central(false);
    }

    return grpc::Status::OK;
}

grpc::Status DaemonServiceImpl::DisconnectFromCentral(
    grpc::ServerContext* context,
    const DisconnectFromCentralRequest* request,
    DisconnectFromCentralResponse* response) {

    spdlog::info("DisconnectFromCentral called");

    if (!node_client_) {
        response->set_success(true);  // Already disconnected
        return grpc::Status::OK;
    }

    // Stop heartbeat
    node_client_->StopHeartbeat();

    // TODO: Send unregister to Central Server
    // node_client_->Unregister();

    spdlog::info("Disconnected from Central Server");
    response->set_success(true);
    return grpc::Status::OK;
}

} // namespace cyxwiz::servernode::ipc
