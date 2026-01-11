// daemon_client.cpp - Client implementation for GUI/TUI to daemon connection
#include "ipc/daemon_client.h"

// Generated protobuf/gRPC headers
#include <daemon.grpc.pb.h>
#include <daemon.pb.h>

#include <spdlog/spdlog.h>
#include <chrono>
#include <thread>
#include <fstream>
#include <sstream>

namespace cyxwiz::servernode::ipc {

using namespace cyxwiz::daemon;

// Pimpl wrapper for gRPC stub
class DaemonServiceStub {
public:
    std::unique_ptr<DaemonService::Stub> stub;
};

// Helper to read file contents
static std::string ReadFileContents(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        spdlog::error("Failed to open file: {}", path);
        return "";
    }
    std::stringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

// Helper to create gRPC credentials from TLS settings
static std::shared_ptr<grpc::ChannelCredentials> CreateCredentials(
    const TLSConnectionSettings& tls_settings,
    std::string& error_message) {

    if (!tls_settings.enabled) {
        return grpc::InsecureChannelCredentials();
    }

    // For skip_verification mode (development only)
    if (tls_settings.skip_verification) {
        spdlog::warn("TLS verification disabled - NOT RECOMMENDED for production!");
        // Use SslCredentialsOptions with empty root certs to skip verification
        grpc::SslCredentialsOptions ssl_opts;
        ssl_opts.pem_root_certs = "";  // Empty = system roots (may still verify)
        return grpc::SslCredentials(ssl_opts);
    }

    // Load CA certificate (required for TLS)
    if (tls_settings.ca_cert_path.empty()) {
        error_message = "TLS enabled but CA certificate path not provided";
        spdlog::error("{}", error_message);
        return nullptr;
    }

    std::string ca_cert = ReadFileContents(tls_settings.ca_cert_path);
    if (ca_cert.empty()) {
        error_message = "Failed to read CA certificate: " + tls_settings.ca_cert_path;
        spdlog::error("{}", error_message);
        return nullptr;
    }

    grpc::SslCredentialsOptions ssl_opts;
    ssl_opts.pem_root_certs = ca_cert;

    // Optional: mutual TLS with client certificate
    if (!tls_settings.client_cert_path.empty() && !tls_settings.client_key_path.empty()) {
        std::string client_cert = ReadFileContents(tls_settings.client_cert_path);
        std::string client_key = ReadFileContents(tls_settings.client_key_path);

        if (client_cert.empty() || client_key.empty()) {
            error_message = "Failed to read client certificate or key";
            spdlog::error("{}", error_message);
            return nullptr;
        }

        ssl_opts.pem_cert_chain = client_cert;
        ssl_opts.pem_private_key = client_key;
        spdlog::info("Using mutual TLS with client certificate");
    }

    return grpc::SslCredentials(ssl_opts);
}

DaemonClient::DaemonClient() = default;

DaemonClient::~DaemonClient() {
    Disconnect();
}

bool DaemonClient::Connect(const std::string& address) {
    // Default to insecure connection for local daemon
    TLSConnectionSettings settings;
    settings.enabled = false;
    return Connect(address, settings);
}

bool DaemonClient::Connect(const std::string& address, const TLSConnectionSettings& tls_settings) {
    if (connected_.load()) {
        Disconnect();
    }

    address_ = address;
    tls_settings_ = tls_settings;

    // Create credentials based on TLS settings
    std::string error_message;
    auto credentials = CreateCredentials(tls_settings, error_message);
    if (!credentials) {
        spdlog::error("Failed to create credentials: {}", error_message);
        return false;
    }

    // Create channel arguments
    grpc::ChannelArguments args;
    args.SetInt(GRPC_ARG_KEEPALIVE_TIME_MS, 10000);
    args.SetInt(GRPC_ARG_KEEPALIVE_TIMEOUT_MS, 5000);

    // Override server name if specified (for TLS verification)
    if (tls_settings.enabled && !tls_settings.target_name_override.empty()) {
        args.SetSslTargetNameOverride(tls_settings.target_name_override);
        spdlog::debug("Using TLS target name override: {}", tls_settings.target_name_override);
    }

    channel_ = grpc::CreateCustomChannel(address, credentials, args);

    if (!channel_) {
        spdlog::error("Failed to create channel to {}", address);
        return false;
    }

    // Wait for connection with timeout
    auto deadline = std::chrono::system_clock::now() + std::chrono::seconds(5);
    if (!channel_->WaitForConnected(deadline)) {
        spdlog::error("Failed to connect to daemon at {} (timeout)", address);
        return false;
    }

    // Create stub
    stub_ = std::make_unique<DaemonServiceStub>();
    stub_->stub = DaemonService::NewStub(channel_);

    connected_.store(true);
    tls_enabled_.store(tls_settings.enabled);

    if (tls_settings.enabled) {
        spdlog::info("Connected to daemon at {} with TLS encryption", address);
    } else {
        spdlog::info("Connected to daemon at {} (insecure)", address);
    }

    return true;
}

bool DaemonClient::TestConnection(const std::string& address,
                                   const TLSConnectionSettings& tls_settings,
                                   std::string& error_message,
                                   int timeout_seconds) {
    // Create credentials
    auto credentials = CreateCredentials(tls_settings, error_message);
    if (!credentials) {
        return false;
    }

    // Create channel arguments
    grpc::ChannelArguments args;
    args.SetInt(GRPC_ARG_KEEPALIVE_TIME_MS, 5000);
    args.SetInt(GRPC_ARG_KEEPALIVE_TIMEOUT_MS, 2000);

    if (tls_settings.enabled && !tls_settings.target_name_override.empty()) {
        args.SetSslTargetNameOverride(tls_settings.target_name_override);
    }

    auto channel = grpc::CreateCustomChannel(address, credentials, args);
    if (!channel) {
        error_message = "Failed to create channel";
        return false;
    }

    // Wait for connection with timeout
    auto deadline = std::chrono::system_clock::now() + std::chrono::seconds(timeout_seconds);
    if (!channel->WaitForConnected(deadline)) {
        error_message = "Connection timeout";
        return false;
    }

    // Try to get status to verify the connection works
    auto stub = DaemonService::NewStub(channel);
    grpc::ClientContext context;
    context.set_deadline(std::chrono::system_clock::now() + std::chrono::seconds(3));

    GetStatusRequest request;
    GetStatusResponse response;

    auto status = stub->GetStatus(&context, request, &response);
    if (!status.ok()) {
        error_message = "RPC failed: " + status.error_message();
        return false;
    }

    return true;
}

void DaemonClient::ConnectAsync(const std::string& address) {
    // Launch connection in background thread
    std::thread([this, address]() {
        spdlog::info("Connecting to daemon at {} (background)...", address);
        if (Connect(address)) {
            spdlog::info("Connected to daemon successfully");
        } else {
            spdlog::warn("Failed to connect to daemon at {}", address);
        }
    }).detach();
}

void DaemonClient::Disconnect() {
    // Stop all streams
    StopMetricsStream();
    StopJobUpdatesStream();
    StopLogStream();

    connected_.store(false);
    tls_enabled_.store(false);
    stub_.reset();
    channel_.reset();

    spdlog::info("Disconnected from daemon");
}

// ============================================================================
// Status & Metrics
// ============================================================================

bool DaemonClient::GetStatus(DaemonStatus& status) {
    if (!connected_.load() || !stub_) return false;

    grpc::ClientContext context;
    context.set_deadline(std::chrono::system_clock::now() + std::chrono::seconds(5));

    GetStatusRequest request;
    GetStatusResponse response;

    auto grpc_status = stub_->stub->GetStatus(&context, request, &response);
    if (!grpc_status.ok()) {
        spdlog::error("GetStatus failed: {}", grpc_status.error_message());
        // If connection is lost (not just a temporary error), mark as disconnected
        if (grpc_status.error_code() == grpc::StatusCode::UNAVAILABLE ||
            grpc_status.error_code() == grpc::StatusCode::DEADLINE_EXCEEDED) {
            spdlog::warn("Daemon connection lost, marking as disconnected");
            connected_.store(false);
        }
        return false;
    }

    status.node_id = response.node_id();
    status.version = response.version();
    status.connected_to_central = response.central_server_status() == ConnectionStatus::CONNECTION_STATUS_CONNECTED;
    status.auth_required = response.central_server_status() == ConnectionStatus::CONNECTION_STATUS_AUTH_REQUIRED;
    status.uptime_seconds = response.uptime_seconds();
    status.active_jobs = response.active_jobs();
    status.active_deployments = response.active_deployments();
    status.gpu_name = response.gpu_name();
    status.gpu_count = response.gpu_count();

    if (response.has_metrics()) {
        auto& m = response.metrics();
        status.metrics.cpu_usage = m.cpu_usage();
        status.metrics.gpu_usage = m.gpu_usage();
        status.metrics.ram_usage = m.ram_usage();
        status.metrics.vram_usage = m.vram_usage();
        status.metrics.ram_total = m.ram_total_bytes();
        status.metrics.ram_used = m.ram_used_bytes();
        status.metrics.vram_total = m.vram_total_bytes();
        status.metrics.vram_used = m.vram_used_bytes();

        // Parse per-GPU metrics
        status.metrics.gpus.clear();
        for (const auto& gpu_pb : m.gpus()) {
            GPUInfo gpu;
            gpu.device_id = gpu_pb.device_id();
            gpu.name = gpu_pb.name();
            gpu.vendor = gpu_pb.vendor();
            gpu.usage_3d = gpu_pb.usage_3d();
            gpu.usage_copy = gpu_pb.usage_copy();
            gpu.usage_video_decode = gpu_pb.usage_video_decode();
            gpu.usage_video_encode = gpu_pb.usage_video_encode();
            gpu.memory_usage = gpu_pb.memory_usage();
            gpu.vram_used = gpu_pb.vram_used_bytes();
            gpu.vram_total = gpu_pb.vram_total_bytes();
            gpu.temperature = gpu_pb.temperature_celsius();
            gpu.power_watts = gpu_pb.power_watts();
            gpu.is_nvidia = gpu_pb.is_nvidia();
            status.metrics.gpus.push_back(gpu);
        }
        status.metrics.gpu_count = m.gpu_count();
    }

    return true;
}

bool DaemonClient::GetMetrics(SystemMetrics& metrics,
                               std::vector<float>& cpu_history,
                               std::vector<float>& gpu_history,
                               std::vector<float>& ram_history,
                               std::vector<float>& vram_history) {
    if (!connected_.load() || !stub_) return false;

    grpc::ClientContext context;
    context.set_deadline(std::chrono::system_clock::now() + std::chrono::seconds(5));

    GetMetricsRequest request;
    GetMetricsResponse response;

    auto status = stub_->stub->GetMetrics(&context, request, &response);
    if (!status.ok()) {
        spdlog::error("GetMetrics failed: {}", status.error_message());
        // If connection is lost, mark as disconnected
        if (status.error_code() == grpc::StatusCode::UNAVAILABLE ||
            status.error_code() == grpc::StatusCode::DEADLINE_EXCEEDED) {
            spdlog::warn("Daemon connection lost during GetMetrics, marking as disconnected");
            connected_.store(false);
        }
        return false;
    }

    if (response.has_metrics()) {
        auto& m = response.metrics();
        metrics.cpu_usage = m.cpu_usage();
        metrics.gpu_usage = m.gpu_usage();
        metrics.ram_usage = m.ram_usage();
        metrics.vram_usage = m.vram_usage();
        metrics.ram_total = m.ram_total_bytes();
        metrics.ram_used = m.ram_used_bytes();
        metrics.vram_total = m.vram_total_bytes();
        metrics.vram_used = m.vram_used_bytes();

        // Parse per-GPU metrics
        metrics.gpus.clear();
        for (const auto& gpu_pb : m.gpus()) {
            GPUInfo gpu;
            gpu.device_id = gpu_pb.device_id();
            gpu.name = gpu_pb.name();
            gpu.vendor = gpu_pb.vendor();
            gpu.usage_3d = gpu_pb.usage_3d();
            gpu.usage_copy = gpu_pb.usage_copy();
            gpu.usage_video_decode = gpu_pb.usage_video_decode();
            gpu.usage_video_encode = gpu_pb.usage_video_encode();
            gpu.memory_usage = gpu_pb.memory_usage();
            gpu.vram_used = gpu_pb.vram_used_bytes();
            gpu.vram_total = gpu_pb.vram_total_bytes();
            gpu.temperature = gpu_pb.temperature_celsius();
            gpu.power_watts = gpu_pb.power_watts();
            gpu.is_nvidia = gpu_pb.is_nvidia();
            metrics.gpus.push_back(gpu);
        }
        metrics.gpu_count = m.gpu_count();
    }

    cpu_history.assign(response.cpu_history().begin(), response.cpu_history().end());
    gpu_history.assign(response.gpu_history().begin(), response.gpu_history().end());
    ram_history.assign(response.ram_history().begin(), response.ram_history().end());
    vram_history.assign(response.vram_history().begin(), response.vram_history().end());

    return true;
}

void DaemonClient::StartMetricsStream(MetricsCallback callback, int interval_ms) {
    if (metrics_streaming_.load()) return;
    if (!connected_.load() || !stub_) return;

    metrics_streaming_.store(true);

    metrics_stream_thread_ = std::thread([this, callback, interval_ms]() {
        grpc::ClientContext context;
        StreamMetricsRequest request;
        request.set_interval_ms(interval_ms);

        auto reader = stub_->stub->StreamMetrics(&context, request);
        MetricsUpdate update;

        while (metrics_streaming_.load() && reader->Read(&update)) {
            SystemMetrics metrics;
            if (update.has_metrics()) {
                auto& m = update.metrics();
                metrics.cpu_usage = m.cpu_usage();
                metrics.gpu_usage = m.gpu_usage();
                metrics.ram_usage = m.ram_usage();
                metrics.vram_usage = m.vram_usage();

                // Parse per-GPU metrics
                for (const auto& gpu_pb : m.gpus()) {
                    GPUInfo gpu;
                    gpu.device_id = gpu_pb.device_id();
                    gpu.name = gpu_pb.name();
                    gpu.vendor = gpu_pb.vendor();
                    gpu.usage_3d = gpu_pb.usage_3d();
                    gpu.usage_copy = gpu_pb.usage_copy();
                    gpu.usage_video_decode = gpu_pb.usage_video_decode();
                    gpu.usage_video_encode = gpu_pb.usage_video_encode();
                    gpu.memory_usage = gpu_pb.memory_usage();
                    gpu.vram_used = gpu_pb.vram_used_bytes();
                    gpu.vram_total = gpu_pb.vram_total_bytes();
                    gpu.temperature = gpu_pb.temperature_celsius();
                    gpu.power_watts = gpu_pb.power_watts();
                    gpu.is_nvidia = gpu_pb.is_nvidia();
                    metrics.gpus.push_back(gpu);
                }
                metrics.gpu_count = m.gpu_count();
            }
            callback(metrics);
        }

        context.TryCancel();
    });
}

void DaemonClient::StopMetricsStream() {
    metrics_streaming_.store(false);
    if (metrics_stream_thread_.joinable()) {
        metrics_stream_thread_.join();
    }
}

// ============================================================================
// Jobs
// ============================================================================

bool DaemonClient::ListJobs(std::vector<JobInfo>& jobs, bool include_completed) {
    spdlog::info("DaemonClient::ListJobs called (connected={}, stub={})",
                 connected_.load(), stub_ ? "valid" : "null");

    if (!connected_.load() || !stub_) {
        spdlog::warn("ListJobs: Not connected or no stub");
        return false;
    }

    grpc::ClientContext context;
    context.set_deadline(std::chrono::system_clock::now() + std::chrono::seconds(5));

    ListJobsRequest request;
    request.set_include_completed(include_completed);
    ListJobsResponse response;

    spdlog::info("Calling gRPC ListJobs...");
    auto status = stub_->stub->ListJobs(&context, request, &response);
    if (!status.ok()) {
        spdlog::error("ListJobs gRPC failed: {} (code={})",
                     status.error_message(), static_cast<int>(status.error_code()));
        return false;
    }

    spdlog::info("ListJobs gRPC success: {} jobs in response", response.jobs_size());

    jobs.clear();
    for (const auto& j : response.jobs()) {
        spdlog::info("  Received job: id={}, status={}, progress={:.2f}",
                    j.id(), static_cast<int>(j.status()), j.progress());
        JobInfo job;
        job.id = j.id();
        job.type = j.type();
        job.status = static_cast<int>(j.status());
        job.progress = j.progress();
        job.current_epoch = j.current_epoch();
        job.total_epochs = j.total_epochs();
        job.loss = j.loss();
        job.accuracy = j.accuracy();
        job.started_at = j.started_at();
        job.model_name = j.model_name();
        job.earnings = j.earnings();
        jobs.push_back(job);
    }

    return true;
}

bool DaemonClient::GetJob(const std::string& job_id, JobInfo& job) {
    if (!connected_.load() || !stub_) return false;

    grpc::ClientContext context;
    context.set_deadline(std::chrono::system_clock::now() + std::chrono::seconds(5));

    GetJobRequest request;
    request.set_job_id(job_id);
    GetJobResponse response;

    auto status = stub_->stub->GetJob(&context, request, &response);
    if (!status.ok()) {
        return false;
    }

    if (response.has_job()) {
        auto& j = response.job();
        job.id = j.id();
        job.type = j.type();
        job.status = static_cast<int>(j.status());
        job.progress = j.progress();
        job.current_epoch = j.current_epoch();
        job.total_epochs = j.total_epochs();
        job.loss = j.loss();
        job.model_name = j.model_name();
    }

    return true;
}

bool DaemonClient::CancelJob(const std::string& job_id, std::string& error) {
    if (!connected_.load() || !stub_) {
        error = "Not connected";
        return false;
    }

    grpc::ClientContext context;
    context.set_deadline(std::chrono::system_clock::now() + std::chrono::seconds(5));

    CancelJobRequest request;
    request.set_job_id(job_id);
    CancelJobResponse response;

    auto status = stub_->stub->CancelJob(&context, request, &response);
    if (!status.ok()) {
        error = status.error_message();
        return false;
    }

    if (!response.success()) {
        error = response.error_message();
        return false;
    }

    return true;
}

void DaemonClient::StartJobUpdatesStream(JobUpdateCallback callback) {
    if (job_updates_streaming_.load()) return;
    if (!connected_.load() || !stub_) return;

    job_updates_streaming_.store(true);

    job_updates_thread_ = std::thread([this, callback]() {
        grpc::ClientContext context;
        StreamJobUpdatesRequest request;

        auto reader = stub_->stub->StreamJobUpdates(&context, request);
        JobUpdate update;

        while (job_updates_streaming_.load() && reader->Read(&update)) {
            JobInfo job;
            if (update.has_job()) {
                auto& j = update.job();
                job.id = j.id();
                job.type = j.type();
                job.status = static_cast<int>(j.status());
                job.progress = j.progress();
                job.loss = j.loss();
            }
            callback(update.job_id(), job, update.update_type());
        }

        context.TryCancel();
    });
}

void DaemonClient::StopJobUpdatesStream() {
    job_updates_streaming_.store(false);
    if (job_updates_thread_.joinable()) {
        job_updates_thread_.join();
    }
}

// ============================================================================
// Deployments
// ============================================================================

bool DaemonClient::ListDeployments(std::vector<DeploymentInfo>& deployments) {
    if (!connected_.load() || !stub_) return false;

    grpc::ClientContext context;
    context.set_deadline(std::chrono::system_clock::now() + std::chrono::seconds(5));

    ListDeploymentsRequest request;
    ListDeploymentsResponse response;

    auto status = stub_->stub->ListDeployments(&context, request, &response);
    if (!status.ok()) {
        return false;
    }

    deployments.clear();
    for (const auto& d : response.deployments()) {
        DeploymentInfo dep;
        dep.id = d.id();
        dep.model_name = d.model_name();
        dep.model_path = d.model_path();
        dep.format = d.format();
        dep.status = static_cast<int>(d.status());
        dep.port = d.port();
        dep.gpu_layers = d.gpu_layers();
        dep.context_size = d.context_size();
        dep.request_count = d.request_count();
        dep.earnings = d.earnings();
        deployments.push_back(dep);
    }

    return true;
}

bool DaemonClient::DeployModel(const std::string& model_path, int port, int gpu_layers,
                                int context_size, std::string& deployment_id, std::string& error) {
    if (!connected_.load() || !stub_) {
        error = "Not connected";
        return false;
    }

    grpc::ClientContext context;
    context.set_deadline(std::chrono::system_clock::now() + std::chrono::seconds(30));

    DeployModelRequest request;
    request.set_model_path(model_path);
    request.set_port(port);
    request.set_gpu_layers(gpu_layers);
    request.set_context_size(context_size);
    DeployModelResponse response;

    auto status = stub_->stub->DeployModel(&context, request, &response);
    if (!status.ok()) {
        error = status.error_message();
        return false;
    }

    if (!response.success()) {
        error = response.error_message();
        return false;
    }

    deployment_id = response.deployment_id();
    return true;
}

bool DaemonClient::UndeployModel(const std::string& deployment_id, std::string& error) {
    if (!connected_.load() || !stub_) {
        error = "Not connected";
        return false;
    }

    grpc::ClientContext context;
    context.set_deadline(std::chrono::system_clock::now() + std::chrono::seconds(10));

    UndeployModelRequest request;
    request.set_deployment_id(deployment_id);
    UndeployModelResponse response;

    auto status = stub_->stub->UndeployModel(&context, request, &response);
    if (!status.ok()) {
        error = status.error_message();
        return false;
    }

    if (!response.success()) {
        error = response.error_message();
        return false;
    }

    return true;
}

// ============================================================================
// Models
// ============================================================================

bool DaemonClient::ListLocalModels(std::vector<ModelInfo>& models, const std::string& directory) {
    if (!connected_.load() || !stub_) return false;

    grpc::ClientContext context;
    context.set_deadline(std::chrono::system_clock::now() + std::chrono::seconds(10));

    ListLocalModelsRequest request;
    if (!directory.empty()) {
        request.set_directory(directory);
    }
    ListLocalModelsResponse response;

    auto status = stub_->stub->ListLocalModels(&context, request, &response);
    if (!status.ok()) {
        return false;
    }

    models.clear();
    for (const auto& m : response.models()) {
        ModelInfo model;
        model.name = m.name();
        model.path = m.path();
        model.format = m.format();
        model.size_bytes = m.size_bytes();
        model.modified_at = m.modified_at();
        model.is_deployed = m.is_deployed();
        model.architecture = m.architecture();
        models.push_back(model);
    }

    return true;
}

bool DaemonClient::ScanModels(const std::vector<std::string>& directories, int& models_found) {
    if (!connected_.load() || !stub_) return false;

    grpc::ClientContext context;
    context.set_deadline(std::chrono::system_clock::now() + std::chrono::seconds(30));

    ScanModelsRequest request;
    for (const auto& dir : directories) {
        request.add_directories(dir);
    }
    ScanModelsResponse response;

    auto status = stub_->stub->ScanModels(&context, request, &response);
    if (!status.ok()) {
        return false;
    }

    models_found = response.models_found();
    return true;
}

bool DaemonClient::DeleteModel(const std::string& model_path, std::string& error) {
    if (!connected_.load() || !stub_) {
        error = "Not connected";
        return false;
    }

    grpc::ClientContext context;
    context.set_deadline(std::chrono::system_clock::now() + std::chrono::seconds(5));

    DeleteModelRequest request;
    request.set_model_path(model_path);
    DeleteModelResponse response;

    auto status = stub_->stub->DeleteModel(&context, request, &response);
    if (!status.ok()) {
        error = status.error_message();
        return false;
    }

    if (!response.success()) {
        error = response.error_message();
        return false;
    }

    return true;
}

// ============================================================================
// API Keys
// ============================================================================

bool DaemonClient::ListAPIKeys(std::vector<APIKeyInfo>& keys) {
    if (!connected_.load() || !stub_) return false;

    grpc::ClientContext context;
    context.set_deadline(std::chrono::system_clock::now() + std::chrono::seconds(5));

    ListAPIKeysRequest request;
    ListAPIKeysResponse response;

    auto status = stub_->stub->ListAPIKeys(&context, request, &response);
    if (!status.ok()) {
        return false;
    }

    keys.clear();
    for (const auto& k : response.keys()) {
        APIKeyInfo key;
        key.id = k.id();
        key.name = k.name();
        key.key_prefix = k.key_prefix();
        key.created_at = k.created_at();
        key.last_used_at = k.last_used_at();
        key.request_count = k.request_count();
        key.rate_limit_rpm = k.rate_limit_rpm();
        key.is_active = k.is_active();
        keys.push_back(key);
    }

    return true;
}

bool DaemonClient::CreateAPIKey(const std::string& name, int rate_limit_rpm,
                                 std::string& full_key, std::string& error) {
    if (!connected_.load() || !stub_) {
        error = "Not connected";
        return false;
    }

    grpc::ClientContext context;
    context.set_deadline(std::chrono::system_clock::now() + std::chrono::seconds(5));

    CreateAPIKeyRequest request;
    request.set_name(name);
    request.set_rate_limit_rpm(rate_limit_rpm);
    CreateAPIKeyResponse response;

    auto status = stub_->stub->CreateAPIKey(&context, request, &response);
    if (!status.ok()) {
        error = status.error_message();
        return false;
    }

    if (!response.success()) {
        error = response.error_message();
        return false;
    }

    full_key = response.key();
    return true;
}

bool DaemonClient::RevokeAPIKey(const std::string& key_id, std::string& error) {
    if (!connected_.load() || !stub_) {
        error = "Not connected";
        return false;
    }

    grpc::ClientContext context;
    context.set_deadline(std::chrono::system_clock::now() + std::chrono::seconds(5));

    RevokeAPIKeyRequest request;
    request.set_key_id(key_id);
    RevokeAPIKeyResponse response;

    auto status = stub_->stub->RevokeAPIKey(&context, request, &response);
    if (!status.ok()) {
        error = status.error_message();
        return false;
    }

    if (!response.success()) {
        error = response.error_message();
        return false;
    }

    return true;
}

// ============================================================================
// Configuration
// ============================================================================

bool DaemonClient::GetConfig(NodeConfig& config) {
    if (!connected_.load() || !stub_) return false;

    grpc::ClientContext context;
    context.set_deadline(std::chrono::system_clock::now() + std::chrono::seconds(5));

    GetConfigRequest request;
    GetConfigResponse response;

    auto status = stub_->stub->GetConfig(&context, request, &response);
    if (!status.ok()) {
        return false;
    }

    config.node_name = response.node_name();
    config.central_server_address = response.central_server_address();
    config.max_concurrent_jobs = response.max_concurrent_jobs();
    config.default_gpu_layers = response.default_gpu_layers();
    config.default_context_size = response.default_context_size();
    config.log_level = response.log_level();

    config.model_directories.clear();
    for (int i = 0; i < response.model_directories_size(); ++i) {
        config.model_directories.push_back(response.model_directories(i));
    }

    return true;
}

bool DaemonClient::SetConfig(const NodeConfig& config, bool& restart_required, std::string& error) {
    if (!connected_.load() || !stub_) {
        error = "Not connected";
        return false;
    }

    grpc::ClientContext context;
    context.set_deadline(std::chrono::system_clock::now() + std::chrono::seconds(5));

    SetConfigRequest request;
    request.set_node_name(config.node_name);
    request.set_central_server_address(config.central_server_address);
    request.set_max_concurrent_jobs(config.max_concurrent_jobs);
    request.set_default_gpu_layers(config.default_gpu_layers);
    request.set_default_context_size(config.default_context_size);
    request.set_log_level(config.log_level);

    for (const auto& dir : config.model_directories) {
        request.add_model_directories(dir);
    }

    SetConfigResponse response;

    auto status = stub_->stub->SetConfig(&context, request, &response);
    if (!status.ok()) {
        error = status.error_message();
        return false;
    }

    if (!response.success()) {
        error = response.error_message();
        return false;
    }

    restart_required = response.restart_required();
    return true;
}

// ============================================================================
// Earnings & Wallet
// ============================================================================

bool DaemonClient::GetEarnings(EarningsInfo& earnings) {
    if (!connected_.load() || !stub_) return false;

    grpc::ClientContext context;
    context.set_deadline(std::chrono::system_clock::now() + std::chrono::seconds(5));

    GetEarningsRequest request;
    GetEarningsResponse response;

    auto status = stub_->stub->GetEarnings(&context, request, &response);
    if (!status.ok()) {
        return false;
    }

    if (response.has_earnings()) {
        auto& e = response.earnings();
        earnings.today = e.today();
        earnings.this_week = e.this_week();
        earnings.this_month = e.this_month();
        earnings.all_time = e.all_time();
        earnings.pending_payout = e.pending_payout();
        earnings.jobs_completed = e.jobs_completed();
    }

    return true;
}

bool DaemonClient::GetWalletAddress(std::string& address, double& balance, bool& is_connected) {
    if (!connected_.load() || !stub_) return false;

    grpc::ClientContext context;
    context.set_deadline(std::chrono::system_clock::now() + std::chrono::seconds(5));

    GetWalletInfoRequest request;
    GetWalletInfoResponse response;

    auto status = stub_->stub->GetWalletInfo(&context, request, &response);
    if (!status.ok()) {
        return false;
    }

    address = response.wallet_address();
    balance = response.balance();
    is_connected = response.is_connected();

    return true;
}

bool DaemonClient::SetWalletAddress(const std::string& address, std::string& error) {
    if (!connected_.load() || !stub_) {
        error = "Not connected";
        return false;
    }

    grpc::ClientContext context;
    context.set_deadline(std::chrono::system_clock::now() + std::chrono::seconds(5));

    SetWalletAddressRequest request;
    request.set_wallet_address(address);
    SetWalletAddressResponse response;

    auto status = stub_->stub->SetWalletAddress(&context, request, &response);
    if (!status.ok()) {
        error = status.error_message();
        return false;
    }

    if (!response.success()) {
        error = response.error_message();
        return false;
    }

    return true;
}

// ============================================================================
// Logs
// ============================================================================

bool DaemonClient::GetLogs(std::vector<LogEntry>& entries, int limit, const std::string& level_filter) {
    if (!connected_.load() || !stub_) return false;

    grpc::ClientContext context;
    context.set_deadline(std::chrono::system_clock::now() + std::chrono::seconds(5));

    GetLogsRequest request;
    request.set_limit(limit);
    if (!level_filter.empty()) {
        request.set_level_filter(level_filter);
    }
    GetLogsResponse response;

    auto status = stub_->stub->GetLogs(&context, request, &response);
    if (!status.ok()) {
        return false;
    }

    entries.clear();
    for (const auto& e : response.entries()) {
        LogEntry entry;
        entry.timestamp = e.timestamp();
        entry.level = e.level();
        entry.message = e.message();
        entry.source = e.source();
        entries.push_back(entry);
    }

    return true;
}

void DaemonClient::StartLogStream(LogCallback callback, const std::string& level_filter) {
    if (log_streaming_.load()) return;
    if (!connected_.load() || !stub_) return;

    log_streaming_.store(true);

    log_stream_thread_ = std::thread([this, callback, level_filter]() {
        grpc::ClientContext context;
        StreamLogsRequest request;
        if (!level_filter.empty()) {
            request.set_level_filter(level_filter);
        }

        auto reader = stub_->stub->StreamLogs(&context, request);
        cyxwiz::daemon::LogEntry entry_pb;

        while (log_streaming_.load() && reader->Read(&entry_pb)) {
            LogEntry entry;
            entry.timestamp = entry_pb.timestamp();
            entry.level = entry_pb.level();
            entry.message = entry_pb.message();
            entry.source = entry_pb.source();
            callback(entry);
        }

        context.TryCancel();
    });
}

void DaemonClient::StopLogStream() {
    log_streaming_.store(false);
    if (log_stream_thread_.joinable()) {
        log_stream_thread_.join();
    }
}

// ============================================================================
// Pool Mining
// ============================================================================

bool DaemonClient::GetPoolStatus(PoolStatus& status) {
    if (!connected_.load() || !stub_) return false;

    grpc::ClientContext context;
    context.set_deadline(std::chrono::system_clock::now() + std::chrono::seconds(5));

    GetPoolStatusRequest request;
    GetPoolStatusResponse response;

    auto grpc_status = stub_->stub->GetPoolStatus(&context, request, &response);
    if (!grpc_status.ok()) {
        return false;
    }

    // Fill pool info
    if (response.has_pool()) {
        const auto& pool = response.pool();
        status.pool_id = pool.pool_id();
        status.pool_name = pool.pool_name();
        status.pool_address = pool.pool_address();
        status.is_joined = pool.is_joined();
        status.mining_intensity = pool.mining_intensity();
        status.pool_earnings = pool.pool_earnings();
        status.active_miners = pool.active_miners();
        status.pool_hashrate = pool.pool_hashrate();
    }

    status.is_mining = response.is_mining();
    status.mining_intensity = response.current_intensity();

    // Fill mining stats
    if (response.has_stats()) {
        const auto& stats_pb = response.stats();
        status.stats.hashrate_mhs = stats_pb.hashrate_mhs();
        status.stats.shares_submitted = stats_pb.shares_submitted();
        status.stats.shares_accepted = stats_pb.shares_accepted();
        status.stats.shares_rejected = stats_pb.shares_rejected();
        status.stats.estimated_daily = stats_pb.estimated_daily();
        status.stats.estimated_monthly = stats_pb.estimated_monthly();
        status.stats.mining_uptime_seconds = stats_pb.mining_uptime_seconds();
    }

    return true;
}

bool DaemonClient::StartMining(std::string& error) {
    if (!connected_.load() || !stub_) {
        error = "Not connected";
        return false;
    }

    grpc::ClientContext context;
    context.set_deadline(std::chrono::system_clock::now() + std::chrono::seconds(10));

    StartMiningRequest request;
    StartMiningResponse response;

    auto status = stub_->stub->StartMining(&context, request, &response);
    if (!status.ok()) {
        error = status.error_message();
        return false;
    }

    if (!response.success()) {
        error = response.error_message();
        return false;
    }

    return true;
}

bool DaemonClient::StopMining(std::string& error) {
    if (!connected_.load() || !stub_) {
        error = "Not connected";
        return false;
    }

    grpc::ClientContext context;
    context.set_deadline(std::chrono::system_clock::now() + std::chrono::seconds(5));

    StopMiningRequest request;
    StopMiningResponse response;

    auto status = stub_->stub->StopMining(&context, request, &response);
    if (!status.ok()) {
        error = status.error_message();
        return false;
    }

    if (!response.success()) {
        error = response.error_message();
        return false;
    }

    return true;
}

bool DaemonClient::JoinPool(const std::string& pool_address, std::string& error) {
    if (!connected_.load() || !stub_) {
        error = "Not connected";
        return false;
    }

    grpc::ClientContext context;
    context.set_deadline(std::chrono::system_clock::now() + std::chrono::seconds(10));

    JoinPoolRequest request;
    request.set_pool_address(pool_address);
    JoinPoolResponse response;

    auto status = stub_->stub->JoinPool(&context, request, &response);
    if (!status.ok()) {
        error = status.error_message();
        return false;
    }

    if (!response.success()) {
        error = response.error_message();
        return false;
    }

    return true;
}

bool DaemonClient::LeavePool(std::string& error) {
    if (!connected_.load() || !stub_) {
        error = "Not connected";
        return false;
    }

    grpc::ClientContext context;
    context.set_deadline(std::chrono::system_clock::now() + std::chrono::seconds(5));

    LeavePoolRequest request;
    LeavePoolResponse response;

    auto status = stub_->stub->LeavePool(&context, request, &response);
    if (!status.ok()) {
        error = status.error_message();
        return false;
    }

    if (!response.success()) {
        error = response.error_message();
        return false;
    }

    return true;
}

bool DaemonClient::SetMiningIntensity(float intensity, std::string& error) {
    if (!connected_.load() || !stub_) {
        error = "Not connected";
        return false;
    }

    grpc::ClientContext context;
    context.set_deadline(std::chrono::system_clock::now() + std::chrono::seconds(5));

    SetMiningIntensityRequest request;
    request.set_intensity(intensity);
    SetMiningIntensityResponse response;

    auto status = stub_->stub->SetMiningIntensity(&context, request, &response);
    if (!status.ok()) {
        error = status.error_message();
        return false;
    }

    if (!response.success()) {
        error = response.error_message();
        return false;
    }

    return true;
}

// ============================================================================
// Marketplace
// ============================================================================

bool DaemonClient::ListMarketplaceModels(std::vector<MarketplaceListing>& listings,
                                          const std::string& query,
                                          ModelCategory category,
                                          int limit, int offset,
                                          const std::string& sort_by) {
    if (!connected_.load() || !stub_) return false;

    grpc::ClientContext context;
    context.set_deadline(std::chrono::system_clock::now() + std::chrono::seconds(30));

    ListMarketplaceModelsRequest request;
    if (!query.empty()) {
        request.set_query(query);
    }
    request.set_category(static_cast<cyxwiz::daemon::ModelCategory>(category));
    request.set_limit(limit);
    request.set_offset(offset);
    if (!sort_by.empty()) {
        request.set_sort_by(sort_by);
    }

    ListMarketplaceModelsResponse response;

    auto status = stub_->stub->ListMarketplaceModels(&context, request, &response);
    if (!status.ok()) {
        return false;
    }

    listings.clear();
    listings.reserve(response.listings_size());

    for (const auto& listing_pb : response.listings()) {
        MarketplaceListing listing;
        listing.id = listing_pb.id();
        listing.name = listing_pb.name();
        listing.description = listing_pb.description();
        listing.format = listing_pb.format();
        listing.size_bytes = listing_pb.size_bytes();
        listing.price_per_request = listing_pb.price_per_request();
        listing.rating = listing_pb.rating();
        listing.download_count = listing_pb.download_count();
        listing.owner_id = listing_pb.owner_id();
        listing.category = static_cast<ModelCategory>(listing_pb.category());
        listing.architecture = listing_pb.architecture();
        listing.parameter_count = listing_pb.parameter_count();
        listing.thumbnail_url = listing_pb.thumbnail_url();
        listing.created_at = listing_pb.created_at();

        // Copy tags
        for (const auto& tag : listing_pb.tags()) {
            listing.tags.push_back(tag);
        }

        listings.push_back(std::move(listing));
    }

    return true;
}

bool DaemonClient::GetMarketplaceModel(const std::string& model_id, MarketplaceListing& listing) {
    if (!connected_.load() || !stub_) return false;

    grpc::ClientContext context;
    context.set_deadline(std::chrono::system_clock::now() + std::chrono::seconds(10));

    GetMarketplaceModelRequest request;
    request.set_model_id(model_id);

    GetMarketplaceModelResponse response;

    auto status = stub_->stub->GetMarketplaceModel(&context, request, &response);
    if (!status.ok()) {
        return false;
    }

    const auto& listing_pb = response.listing();
    listing.id = listing_pb.id();
    listing.name = listing_pb.name();
    listing.description = listing_pb.description();
    listing.format = listing_pb.format();
    listing.size_bytes = listing_pb.size_bytes();
    listing.price_per_request = listing_pb.price_per_request();
    listing.rating = listing_pb.rating();
    listing.download_count = listing_pb.download_count();
    listing.owner_id = listing_pb.owner_id();
    listing.category = static_cast<ModelCategory>(listing_pb.category());
    listing.architecture = listing_pb.architecture();
    listing.parameter_count = listing_pb.parameter_count();
    listing.thumbnail_url = listing_pb.thumbnail_url();
    listing.created_at = listing_pb.created_at();

    listing.tags.clear();
    for (const auto& tag : listing_pb.tags()) {
        listing.tags.push_back(tag);
    }

    return true;
}

void DaemonClient::DownloadMarketplaceModel(const std::string& model_id,
                                             const std::string& target_dir,
                                             DownloadCallback callback) {
    if (!connected_.load() || !stub_) {
        DownloadProgress progress;
        progress.model_id = model_id;
        progress.error_message = "Not connected";
        callback(progress);
        return;
    }

    // Cancel any existing download
    CancelMarketplaceDownload(model_id);

    download_active_.store(true);
    active_download_model_id_ = model_id;

    download_thread_ = std::thread([this, model_id, target_dir, callback]() {
        grpc::ClientContext context;
        // Long timeout for downloads
        context.set_deadline(std::chrono::system_clock::now() + std::chrono::hours(1));

        DownloadMarketplaceModelRequest request;
        request.set_model_id(model_id);
        request.set_target_directory(target_dir);

        auto reader = stub_->stub->DownloadMarketplaceModel(&context, request);
        cyxwiz::daemon::DownloadProgress progress_pb;

        while (download_active_.load() && reader->Read(&progress_pb)) {
            DownloadProgress progress;
            progress.model_id = progress_pb.model_id();
            progress.bytes_downloaded = progress_pb.bytes_downloaded();
            progress.total_bytes = progress_pb.total_bytes();
            progress.progress = progress_pb.progress();
            progress.completed = progress_pb.completed();
            progress.error_message = progress_pb.error_message();
            progress.local_path = progress_pb.local_path();
            callback(progress);

            if (progress.completed || !progress.error_message.empty()) {
                break;
            }
        }

        download_active_.store(false);
        active_download_model_id_.clear();
    });
}

void DaemonClient::CancelMarketplaceDownload(const std::string& model_id) {
    if (download_active_.load() && active_download_model_id_ == model_id) {
        download_active_.store(false);

        // Also send cancel request to daemon
        if (connected_.load() && stub_) {
            grpc::ClientContext context;
            context.set_deadline(std::chrono::system_clock::now() + std::chrono::seconds(5));

            CancelMarketplaceDownloadRequest request;
            request.set_model_id(model_id);
            CancelMarketplaceDownloadResponse response;

            stub_->stub->CancelMarketplaceDownload(&context, request, &response);
        }

        if (download_thread_.joinable()) {
            download_thread_.join();
        }
    }
}

// ============================================================================
// Daemon Control
// ============================================================================

bool DaemonClient::Shutdown(bool graceful, std::string& error) {
    if (!connected_.load() || !stub_) {
        error = "Not connected";
        return false;
    }

    grpc::ClientContext context;
    context.set_deadline(std::chrono::system_clock::now() + std::chrono::seconds(5));

    ShutdownRequest request;
    request.set_graceful(graceful);
    ShutdownResponse response;

    auto status = stub_->stub->Shutdown(&context, request, &response);
    if (!status.ok()) {
        error = status.error_message();
        return false;
    }

    if (!response.success()) {
        error = response.error_message();
        return false;
    }

    return true;
}

bool DaemonClient::Restart(std::string& error) {
    if (!connected_.load() || !stub_) {
        error = "Not connected";
        return false;
    }

    grpc::ClientContext context;
    context.set_deadline(std::chrono::system_clock::now() + std::chrono::seconds(5));

    RestartRequest request;
    RestartResponse response;

    auto status = stub_->stub->Restart(&context, request, &response);
    if (!status.ok()) {
        error = status.error_message();
        return false;
    }

    if (!response.success()) {
        error = response.error_message();
        return false;
    }

    return true;
}

// ============================================================================
// Resource Allocation & Central Server Connection
// ============================================================================

SetAllocationsResult DaemonClient::SetAllocations(
    const std::vector<DeviceAllocationInfo>& allocations,
    const std::string& jwt_token,
    bool connect_to_central) {

    SetAllocationsResult result;

    if (!connected_.load() || !stub_) {
        result.message = "Not connected to daemon";
        return result;
    }

    grpc::ClientContext context;
    context.set_deadline(std::chrono::system_clock::now() + std::chrono::seconds(30));

    SetAllocationsRequest request;

    // Convert allocations to proto format
    for (const auto& alloc : allocations) {
        auto* proto_alloc = request.add_allocations();
        // Map AllocDeviceType (Gpu=0, Cpu=1) to proto DeviceType (UNSPECIFIED=0, GPU=1, CPU=2)
        DeviceType proto_device_type;
        switch (alloc.device_type) {
            case AllocDeviceType::Gpu: proto_device_type = DEVICE_TYPE_GPU; break;
            case AllocDeviceType::Cpu: proto_device_type = DEVICE_TYPE_CPU; break;
            default: proto_device_type = DEVICE_TYPE_UNSPECIFIED; break;
        }
        proto_alloc->set_device_type(proto_device_type);
        proto_alloc->set_device_id(alloc.device_id);
        proto_alloc->set_device_name(alloc.device_name);
        proto_alloc->set_is_enabled(alloc.is_enabled);
        proto_alloc->set_vram_total_mb(alloc.vram_total_mb);
        proto_alloc->set_vram_allocated_mb(alloc.vram_allocation_mb);
        proto_alloc->set_cores_allocated(alloc.cpu_cores_allocation);
        proto_alloc->set_priority(static_cast<AllocationPriority>(alloc.priority));
    }

    request.set_jwt_token(jwt_token);
    request.set_connect_to_central(connect_to_central);

    SetAllocationsResponse response;

    auto status = stub_->stub->SetAllocations(&context, request, &response);
    if (!status.ok()) {
        result.message = "RPC failed: " + status.error_message();
        spdlog::error("SetAllocations RPC failed: {}", status.error_message());
        return result;
    }

    result.success = response.success();
    result.message = response.error_message();
    result.connected_to_central = response.connected_to_central();
    result.node_id = response.node_id();

    if (result.success) {
        spdlog::info("SetAllocations succeeded: connected={}, node_id={}",
                     result.connected_to_central, result.node_id);
    } else {
        spdlog::warn("SetAllocations failed: {}", result.message);
    }

    return result;
}

RetryConnectionResult DaemonClient::RetryConnection() {
    RetryConnectionResult result;

    if (!connected_.load() || !stub_) {
        result.message = "Not connected to daemon";
        return result;
    }

    grpc::ClientContext context;
    context.set_deadline(std::chrono::system_clock::now() + std::chrono::seconds(30));

    RetryConnectionRequest request;
    RetryConnectionResponse response;

    auto status = stub_->stub->RetryConnection(&context, request, &response);
    if (!status.ok()) {
        result.message = "RPC failed: " + status.error_message();
        spdlog::error("RetryConnection RPC failed: {}", status.error_message());
        return result;
    }

    result.success = response.success();
    result.message = response.error_message();
    result.connected = response.connected_to_central();
    result.node_id = response.node_id();

    if (result.success) {
        spdlog::info("RetryConnection succeeded: connected={}", result.connected);
    } else {
        spdlog::warn("RetryConnection failed: {}", result.message);
    }

    return result;
}

DisconnectResult DaemonClient::DisconnectFromCentral() {
    DisconnectResult result;

    if (!connected_.load() || !stub_) {
        result.message = "Not connected to daemon";
        return result;
    }

    grpc::ClientContext context;
    context.set_deadline(std::chrono::system_clock::now() + std::chrono::seconds(10));

    DisconnectFromCentralRequest request;
    DisconnectFromCentralResponse response;

    auto status = stub_->stub->DisconnectFromCentral(&context, request, &response);
    if (!status.ok()) {
        result.message = "RPC failed: " + status.error_message();
        spdlog::error("DisconnectFromCentral RPC failed: {}", status.error_message());
        return result;
    }

    result.success = response.success();
    result.message = response.error_message();

    if (result.success) {
        spdlog::info("DisconnectFromCentral succeeded");
    } else {
        spdlog::warn("DisconnectFromCentral failed: {}", result.message);
    }

    return result;
}

} // namespace cyxwiz::servernode::ipc
