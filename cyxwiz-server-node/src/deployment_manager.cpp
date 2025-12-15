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

    // Signal all deployments to stop
    {
        std::lock_guard<std::mutex> lock(deployments_mutex_);
        for (auto& [id, instance] : deployments_) {
            instance->should_stop = true;
            instance->queue_cv.notify_all();
        }
    }

    // Join all worker threads (outside the lock to avoid deadlock)
    std::vector<std::thread*> threads_to_join;
    {
        std::lock_guard<std::mutex> lock(deployments_mutex_);
        for (auto& [id, instance] : deployments_) {
            if (instance->worker_thread.joinable()) {
                threads_to_join.push_back(&instance->worker_thread);
            }
        }
    }

    for (auto* thread : threads_to_join) {
        thread->join();
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

    DeploymentInstance* instance = nullptr;

    {
        std::lock_guard<std::mutex> lock(deployments_mutex_);
        auto it = deployments_.find(deployment_id);
        if (it == deployments_.end()) {
            throw std::runtime_error("Deployment not found: " + deployment_id);
        }

        instance = it->second.get();
        instance->should_stop = true;
        instance->status = protocol::DEPLOYMENT_STATUS_STOPPED;
    }

    // Notify the inference loop to wake up and check should_stop
    instance->queue_cv.notify_all();

    // Wait for worker thread to finish
    if (instance->worker_thread.joinable()) {
        instance->worker_thread.join();
    }

    spdlog::info("Deployment {} stopped successfully", deployment_id);
}

bool DeploymentManager::RemoveDeployment(const std::string& deployment_id) {
    spdlog::info("Removing deployment: {}", deployment_id);

    std::lock_guard<std::mutex> lock(deployments_mutex_);
    auto it = deployments_.find(deployment_id);
    if (it == deployments_.end()) {
        spdlog::warn("Deployment not found for removal: {}", deployment_id);
        return false;
    }

    // Check if deployment is still running
    auto status = it->second->status;
    if (status == protocol::DEPLOYMENT_STATUS_RUNNING ||
        status == protocol::DEPLOYMENT_STATUS_LOADING ||
        status == protocol::DEPLOYMENT_STATUS_READY) {
        spdlog::error("Cannot remove active deployment: {} (status={})",
                     deployment_id, static_cast<int>(status));
        return false;
    }

    // Remove from map
    deployments_.erase(it);
    spdlog::info("Deployment {} removed successfully", deployment_id);
    return true;
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

    // Get timestamp in milliseconds since epoch
    auto now = std::chrono::system_clock::now();
    auto ms_since_epoch = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()
    ).count();
    metrics.timestamp_ms = ms_since_epoch;

    metrics.cpu_usage = 0.0;  // TODO: Get from metrics collector
    metrics.gpu_usage = 0.0;  // TODO: Get from metrics collector
    metrics.memory_usage = 0.0; // TODO: Get from metrics collector
    metrics.request_count = instance->request_count.load();

    double total_latency = instance->total_latency_ms.load();
    uint64_t count = instance->request_count.load();
    metrics.avg_latency_ms = (count > 0) ? (total_latency / count) : 0.0;

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

std::vector<DeploymentManager::DeploymentInfo> DeploymentManager::GetAllDeployments() const {
    std::lock_guard<std::mutex> lock(deployments_mutex_);
    std::vector<DeploymentInfo> result;
    result.reserve(deployments_.size());
    for (const auto& [id, instance] : deployments_) {
        result.push_back({
            instance->id,
            instance->model_id,
            instance->type,
            instance->status
        });
    }
    return result;
}

bool DeploymentManager::RunInference(
    const std::string& deployment_id,
    const std::unordered_map<std::string, cyxwiz::Tensor>& inputs,
    std::unordered_map<std::string, cyxwiz::Tensor>& outputs) {

    DeploymentInstance* instance = nullptr;

    // Find deployment
    {
        std::lock_guard<std::mutex> lock(deployments_mutex_);
        auto it = deployments_.find(deployment_id);
        if (it == deployments_.end()) {
            spdlog::error("RunInference: Deployment not found: {}", deployment_id);
            return false;
        }
        instance = it->second.get();
    }

    // Check deployment status
    if (instance->status != protocol::DEPLOYMENT_STATUS_RUNNING) {
        spdlog::error("RunInference: Deployment {} is not running (status={})",
                     deployment_id, static_cast<int>(instance->status));
        return false;
    }

    // Create inference request
    auto request = std::make_shared<InferenceRequest>();
    request->request_id = GenerateDeploymentId();  // Reuse ID generator
    request->inputs = inputs;

    // Get future before queuing
    auto future = request->promise.get_future();

    // Queue the request
    {
        std::lock_guard<std::mutex> lock(instance->queue_mutex);
        instance->request_queue.push(request);
    }
    instance->queue_cv.notify_one();

    spdlog::debug("RunInference: Queued request {} for deployment {}",
                 request->request_id, deployment_id);

    // Wait for result
    try {
        auto [success, result_outputs] = future.get();
        if (success) {
            outputs = std::move(result_outputs);
        }
        return success;
    } catch (const std::exception& e) {
        spdlog::error("RunInference: Exception waiting for result: {}", e.what());
        return false;
    }
}

std::vector<servernode::TensorSpec> DeploymentManager::GetInputSpecs(
    const std::string& deployment_id) const {

    std::lock_guard<std::mutex> lock(deployments_mutex_);
    auto it = deployments_.find(deployment_id);
    if (it == deployments_.end()) {
        return {};
    }

    const auto* instance = it->second.get();
    if (instance->model_loader && instance->model_loader->IsLoaded()) {
        return instance->model_loader->GetInputSpecs();
    }
    return {};
}

std::vector<servernode::TensorSpec> DeploymentManager::GetOutputSpecs(
    const std::string& deployment_id) const {

    std::lock_guard<std::mutex> lock(deployments_mutex_);
    auto it = deployments_.find(deployment_id);
    if (it == deployments_.end()) {
        return {};
    }

    const auto* instance = it->second.get();
    if (instance->model_loader && instance->model_loader->IsLoaded()) {
        return instance->model_loader->GetOutputSpecs();
    }
    return {};
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
    // Get model path - prefer local_path from config, fallback to model_id
    std::string model_path = instance->config.model().local_path();
    if (model_path.empty()) {
        model_path = "./models/" + instance->model_id;
    }

    // Get format string - detect from extension (factory expects lowercase format names)
    std::string format_str;

    // Check if it's a directory (directory format .cyxmodel)
    namespace fs = std::filesystem;
    if (fs::is_directory(model_path)) {
        format_str = "cyxmodel";
        spdlog::info("Detected directory format model");
    } else {
        // Detect format from file extension
        size_t dot_pos = model_path.rfind('.');
        if (dot_pos != std::string::npos) {
            std::string ext = model_path.substr(dot_pos);
            if (ext == ".cyxmodel") {
                format_str = "cyxmodel";
            } else if (ext == ".onnx") {
                format_str = "onnx";
            } else if (ext == ".gguf") {
                format_str = "gguf";
            } else if (ext == ".safetensors") {
                format_str = "safetensors";
            } else if (ext == ".pt" || ext == ".pth") {
                format_str = "pytorch";
            }
        }
    }

    // Fallback: check the proto format if extension detection failed
    if (format_str.empty()) {
        auto proto_format = instance->config.model().format();
        if (proto_format == protocol::MODEL_FORMAT_CYXMODEL) {
            format_str = "cyxmodel";
        } else if (proto_format == protocol::MODEL_FORMAT_ONNX) {
            format_str = "onnx";
        } else if (proto_format == protocol::MODEL_FORMAT_GGUF) {
            format_str = "gguf";
        } else if (proto_format == protocol::MODEL_FORMAT_SAFETENSORS) {
            format_str = "safetensors";
        }
    }

    spdlog::info("Loading model {} from path {} with format {}",
                 instance->model_id, model_path, format_str);

    try {
        // Create model loader based on format
        instance->model_loader = ModelLoaderFactory::Create(format_str);

        if (!instance->model_loader) {
            spdlog::error("Failed to create model loader for format: {}", format_str);
            return false;
        }

        // Apply GGUF-specific configuration from deployment config
        if (format_str == "gguf") {
            auto* gguf_loader = dynamic_cast<GGUFLoader*>(instance->model_loader.get());
            if (gguf_loader) {
                const auto& cfg = instance->config;
                if (cfg.gpu_layers() > 0) {
                    gguf_loader->SetGPULayers(cfg.gpu_layers());
                    spdlog::info("GGUF: Setting {} GPU layers", cfg.gpu_layers());
                }
                if (cfg.context_size() > 0) {
                    gguf_loader->SetContextSize(cfg.context_size());
                    spdlog::info("GGUF: Setting context size to {}", cfg.context_size());
                }
                // Note: temperature, max_tokens, top_p could be added to DeploymentConfig proto
            }
        }

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
    spdlog::info("Starting inference loop for deployment: {}", instance->id);

    // Main loop - wait for and process inference requests
    while (!instance->should_stop && !shutdown_) {
        std::shared_ptr<InferenceRequest> request;

        // Wait for a request
        {
            std::unique_lock<std::mutex> lock(instance->queue_mutex);

            // Wait with timeout to allow checking should_stop
            bool got_request = instance->queue_cv.wait_for(
                lock,
                std::chrono::milliseconds(100),
                [instance]() { return !instance->request_queue.empty(); }
            );

            if (!got_request || instance->request_queue.empty()) {
                continue;  // Timeout, check should_stop and loop again
            }

            request = instance->request_queue.front();
            instance->request_queue.pop();
        }

        // Process the request
        spdlog::debug("Processing inference request: {}", request->request_id);

        auto start_time = std::chrono::high_resolution_clock::now();

        try {
            std::unordered_map<std::string, cyxwiz::Tensor> outputs;
            bool success = instance->model_loader->Infer(request->inputs, outputs);

            auto end_time = std::chrono::high_resolution_clock::now();
            double latency_ms = std::chrono::duration<double, std::milli>(
                end_time - start_time
            ).count();

            // Update metrics
            instance->request_count.fetch_add(1);
            // Atomic add for double (not directly supported, use compare-exchange)
            double current = instance->total_latency_ms.load();
            while (!instance->total_latency_ms.compare_exchange_weak(
                current, current + latency_ms)) {}

            spdlog::debug("Inference completed in {:.2f}ms, success={}",
                         latency_ms, success);

            // Set the result
            request->promise.set_value({success, std::move(outputs)});

        } catch (const std::exception& e) {
            spdlog::error("Inference failed for request {}: {}",
                         request->request_id, e.what());

            // Set failure result
            request->promise.set_value({false, {}});
        }
    }

    // Drain remaining requests with error
    {
        std::lock_guard<std::mutex> lock(instance->queue_mutex);
        while (!instance->request_queue.empty()) {
            auto request = instance->request_queue.front();
            instance->request_queue.pop();
            try {
                request->promise.set_value({false, {}});
            } catch (...) {
                // Promise may already be set
            }
        }
    }

    spdlog::info("Inference loop stopped for deployment: {}", instance->id);
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
