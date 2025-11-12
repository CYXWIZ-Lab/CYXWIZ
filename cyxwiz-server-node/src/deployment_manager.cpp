#include "deployment_manager.h"
#include <spdlog/spdlog.h>
#include <chrono>
#include <random>
#include <sstream>
#include <iomanip>

namespace cyxwiz {
namespace servernode {

// ============================================================================
// DeploymentManager Implementation
// ============================================================================

DeploymentManager::DeploymentManager(const std::string& node_id)
    : node_id_(node_id) {
    spdlog::info("DeploymentManager initialized for node: {}", node_id);
}

DeploymentManager::~DeploymentManager() {
    shutdown_ = true;

    // Stop all deployments
    std::lock_guard<std::mutex> lock(deployments_mutex_);
    for (auto& [id, instance] : deployments_) {
        if (instance->worker_thread.joinable()) {
            instance->should_stop = true;
            instance->worker_thread.join();
        }
    }

    spdlog::info("DeploymentManager shutdown complete");
}

std::string DeploymentManager::AcceptDeployment(
    const std::string& model_id,
    protocol::DeploymentType type,
    const protocol::DeploymentConfig& config) {

    spdlog::info("Accepting deployment for model: {}", model_id);

    // Create deployment instance
    auto instance = std::make_unique<DeploymentInstance>();
    instance->id = GenerateDeploymentId();
    instance->model_id = model_id;
    instance->type = type;
    instance->status = protocol::DEPLOYMENT_STATUS_PENDING;
    instance->config = config;
    instance->created_at = std::chrono::system_clock::now();

    std::string deployment_id = instance->id;

    // Store instance
    {
        std::lock_guard<std::mutex> lock(deployments_mutex_);
        deployments_[deployment_id] = std::move(instance);
    }

    // Start worker thread
    {
        std::lock_guard<std::mutex> lock(deployments_mutex_);
        auto* inst_ptr = deployments_[deployment_id].get();
        inst_ptr->worker_thread = std::thread([this, inst_ptr]() {
            ExecuteDeployment(inst_ptr);
        });
    }

    spdlog::info("Deployment {} created and started", deployment_id);
    return deployment_id;
}

void DeploymentManager::StopDeployment(const std::string& deployment_id) {
    spdlog::info("Stopping deployment: {}", deployment_id);

    std::lock_guard<std::mutex> lock(deployments_mutex_);
    auto it = deployments_.find(deployment_id);
    if (it == deployments_.end()) {
        throw std::runtime_error("Deployment not found: " + deployment_id);
    }

    auto* instance = it->second.get();
    instance->should_stop = true;
    instance->status = protocol::DEPLOYMENT_STATUS_STOPPED;

    // Wait for worker thread to finish
    if (instance->worker_thread.joinable()) {
        deployments_mutex_.unlock();  // Unlock to avoid deadlock
        instance->worker_thread.join();
        deployments_mutex_.lock();
    }

    spdlog::info("Deployment {} stopped successfully", deployment_id);
}

protocol::DeploymentStatus DeploymentManager::GetDeploymentStatus(
    const std::string& deployment_id) const {

    std::lock_guard<std::mutex> lock(deployments_mutex_);
    auto it = deployments_.find(deployment_id);
    if (it == deployments_.end()) {
        throw std::runtime_error("Deployment not found: " + deployment_id);
    }

    return it->second->status;
}

std::vector<DeploymentMetrics> DeploymentManager::GetDeploymentMetrics(
    const std::string& deployment_id) const {

    std::lock_guard<std::mutex> lock(deployments_mutex_);
    auto it = deployments_.find(deployment_id);
    if (it == deployments_.end()) {
        throw std::runtime_error("Deployment not found: " + deployment_id);
    }

    const auto* instance = it->second.get();

    // TODO: Collect actual metrics from system
    // For now, return a single snapshot
    DeploymentMetrics metrics;
    metrics.cpu_usage = 0.0;  // TODO: Get from metrics collector
    metrics.gpu_usage = 0.0;  // TODO: Get from metrics collector
    metrics.memory_usage = 0; // TODO: Get from metrics collector
    metrics.request_count = instance->request_count.load();

    double total_latency = instance->total_latency_ms.load();
    uint64_t count = instance->request_count.load();
    metrics.avg_latency_ms = (count > 0) ? (total_latency / count) : 0.0;

    // ISO 8601 timestamp
    auto now = std::chrono::system_clock::now();
    auto tt = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    ss << std::put_time(std::gmtime(&tt), "%Y-%m-%dT%H:%M:%SZ");
    metrics.timestamp = ss.str();

    return {metrics};
}

size_t DeploymentManager::GetActiveDeploymentCount() const {
    std::lock_guard<std::mutex> lock(deployments_mutex_);
    return deployments_.size();
}

bool DeploymentManager::HasDeployment(const std::string& deployment_id) const {
    std::lock_guard<std::mutex> lock(deployments_mutex_);
    return deployments_.find(deployment_id) != deployments_.end();
}

// ============================================================================
// Private Methods
// ============================================================================

void DeploymentManager::ExecuteDeployment(DeploymentInstance* instance) {
    spdlog::info("Starting deployment execution: {}", instance->id);

    try {
        // Update status to loading
        instance->status = protocol::DEPLOYMENT_STATUS_LOADING;

        // Load model
        if (!LoadModel(instance)) {
            spdlog::error("Failed to load model for deployment: {}", instance->id);
            instance->status = protocol::DEPLOYMENT_STATUS_FAILED;
            return;
        }

        // Update status to running
        instance->status = protocol::DEPLOYMENT_STATUS_RUNNING;
        instance->started_at = std::chrono::system_clock::now();
        spdlog::info("Deployment {} is now running", instance->id);

        // Run inference loop
        RunInferenceLoop(instance);

        // Update status to stopped
        instance->status = protocol::DEPLOYMENT_STATUS_STOPPED;
        spdlog::info("Deployment {} finished execution", instance->id);

    } catch (const std::exception& e) {
        spdlog::error("Exception in deployment {}: {}", instance->id, e.what());
        instance->status = protocol::DEPLOYMENT_STATUS_FAILED;
    }

    // Cleanup
    CleanupDeployment(instance);
}

bool DeploymentManager::LoadModel(DeploymentInstance* instance) {
    spdlog::info("Loading model {} with format {}",
                 instance->model_id,
                 instance->config.model_format());

    try {
        // Create model loader based on format
        instance->model_loader = ModelLoaderFactory::Create(
            instance->config.model_format()
        );

        if (!instance->model_loader) {
            spdlog::error("Failed to create model loader for format: {}",
                         instance->config.model_format());
            return false;
        }

        // TODO: Get actual model path from model registry
        std::string model_path = "./models/" + instance->model_id;

        // Load model
        if (!instance->model_loader->Load(model_path)) {
            spdlog::error("Failed to load model from: {}", model_path);
            return false;
        }

        spdlog::info("Model loaded successfully: {}", instance->model_id);
        return true;

    } catch (const std::exception& e) {
        spdlog::error("Exception loading model: {}", e.what());
        return false;
    }
}

void DeploymentManager::RunInferenceLoop(DeploymentInstance* instance) {
    spdlog::debug("Starting inference loop for deployment: {}", instance->id);

    // Main loop
    while (!instance->should_stop && !shutdown_) {
        // TODO: Wait for inference requests
        // TODO: Process inference requests
        // TODO: Update metrics (request_count, total_latency_ms)

        // For now, just sleep
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    spdlog::debug("Inference loop stopped for deployment: {}", instance->id);
}

void DeploymentManager::CleanupDeployment(DeploymentInstance* instance) {
    spdlog::debug("Cleaning up deployment: {}", instance->id);

    try {
        if (instance->model_loader) {
            instance->model_loader->Unload();
            instance->model_loader.reset();
        }

        spdlog::info("Deployment {} cleaned up successfully", instance->id);

    } catch (const std::exception& e) {
        spdlog::error("Exception during cleanup of deployment {}: {}",
                     instance->id, e.what());
    }
}

std::string DeploymentManager::GenerateDeploymentId() const {
    // Generate UUID-like deployment ID
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_int_distribution<> dis(0, 15);
    static const char* hex = "0123456789abcdef";

    std::string id = "dep_";
    for (int i = 0; i < 32; ++i) {
        id += hex[dis(gen)];
        if (i == 7 || i == 11 || i == 15 || i == 19) {
            id += '-';
        }
    }

    return id;
}

} // namespace servernode
} // namespace cyxwiz
