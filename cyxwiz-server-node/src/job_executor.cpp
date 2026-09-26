#include "job_executor.h"
#include "node_client.h"
#include <spdlog/spdlog.h>
#include <nlohmann/json.hpp>
#include <chrono>
#include <cmath>
#include <random>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <type_traits>

// Backend includes for real training
#include <cyxwiz/cyxwiz.h>
#include <cyxwiz/layers/linear.h>
#include <cyxwiz/loss.h>

// Shared training core (after node_client.h: gRPC sets up winsock first)
#include "core/graph_training_job.h"

namespace cyxwiz {
namespace servernode {

JobExecutor::JobExecutor(const std::string& node_id, cyxwiz::Device* device)
    : node_id_(node_id)
    , device_(device)
    , node_client_(nullptr)
{
    spdlog::info("JobExecutor initialized for node: {}", node_id_);

    // Initialize device pool for multi-GPU support
    core::DevicePoolConfig pool_config;
    pool_config.include_cpu = false;  // GPU-only for training
    pool_config.include_cuda = true;
    pool_config.include_opencl = true;
    pool_config.strategy = core::DeviceSelectionStrategy::LeastUtilized;

    device_pool_ = std::make_unique<core::DevicePool>(pool_config);
    if (device_pool_->Initialize()) {
        use_device_pool_ = true;
        spdlog::info("DevicePool initialized with {} devices", device_pool_->GetDeviceCount());

        // Log device info
        for (const auto& dev : device_pool_->GetAllDeviceStates()) {
            spdlog::info("  Device {}: {} ({} MB)",
                         dev.device_id, dev.name,
                         dev.total_memory / (1024 * 1024));
        }
    } else {
        spdlog::warn("DevicePool initialization failed, falling back to single device mode");
        use_device_pool_ = false;
    }
}

JobExecutor::~JobExecutor() {
    spdlog::info("JobExecutor shutting down...");

    // Cancel all active jobs
    std::vector<std::string> job_ids;
    {
        std::lock_guard<std::mutex> lock(jobs_mutex_);
        for (const auto& pair : active_jobs_) {
            job_ids.push_back(pair.first);
        }
    }

    for (const auto& job_id : job_ids) {
        CancelJob(job_id);
    }

    // Wait for all threads to finish
    {
        std::lock_guard<std::mutex> lock(jobs_mutex_);
        for (auto& pair : active_jobs_) {
            if (pair.second->worker_thread.joinable()) {
                pair.second->worker_thread.join();
            }
        }
    }

    spdlog::info("JobExecutor shutdown complete");
}

bool JobExecutor::ExecuteJobAsync(const protocol::JobConfig& job_config) {
    std::string job_id = job_config.job_id();

    spdlog::info("Received job execution request: {}", job_id);

    // Check if job is already running
    {
        std::lock_guard<std::mutex> lock(jobs_mutex_);
        if (active_jobs_.find(job_id) != active_jobs_.end()) {
            spdlog::warn("Job {} is already running", job_id);
            return false;
        }
    }

    // Try to acquire a device from the pool
    int device_id = -1;
    if (use_device_pool_ && device_pool_) {
        // TODO: Parse memory requirement from job config
        size_t required_memory_mb = 0;  // 0 = use pool's minimum threshold

        device_id = device_pool_->AcquireDevice(job_id, required_memory_mb);

        if (device_id < 0) {
            // No device available, queue the job
            spdlog::info("No device available for job {}, adding to pending queue", job_id);
            {
                std::lock_guard<std::mutex> lock(pending_mutex_);
                pending_jobs_.push(job_config);
            }
            return true;  // Job accepted but queued
        }

        spdlog::info("Acquired device {} for job {}", device_id, job_id);
    }

    // Create job state
    auto job_state = std::make_unique<JobState>();
    job_state->config = job_config;
    job_state->is_running = true;
    job_state->should_cancel = false;
    job_state->start_time = std::chrono::steady_clock::now();
    job_state->assigned_device_id = device_id;

    // Store job state
    {
        std::lock_guard<std::mutex> lock(jobs_mutex_);
        active_jobs_[job_id] = std::move(job_state);
    }

    // Launch worker thread
    {
        std::lock_guard<std::mutex> lock(jobs_mutex_);
        auto& state = active_jobs_[job_id];
        state->worker_thread = std::thread(&JobExecutor::ExecuteJob, this, job_id);
    }

    spdlog::info("Job {} started on device {}", job_id, device_id);
    return true;
}

bool JobExecutor::CancelJob(const std::string& job_id) {
    std::lock_guard<std::mutex> lock(jobs_mutex_);

    auto it = active_jobs_.find(job_id);
    if (it == active_jobs_.end()) {
        spdlog::warn("Cannot cancel job {}: not found", job_id);
        return false;
    }

    spdlog::info("Cancelling job {}...", job_id);
    it->second->should_cancel = true;

    return true;
}

bool JobExecutor::PauseJob(const std::string& job_id) {
    std::lock_guard<std::mutex> lock(jobs_mutex_);

    auto it = active_jobs_.find(job_id);
    if (it == active_jobs_.end()) {
        spdlog::warn("Cannot pause job {}: not found", job_id);
        return false;
    }

    if (!it->second->is_running) {
        spdlog::warn("Cannot pause job {}: not running", job_id);
        return false;
    }

    if (it->second->is_paused) {
        spdlog::info("Job {} is already paused", job_id);
        return true;
    }

    spdlog::info("Pausing job {}...", job_id);
    it->second->is_paused = true;
    return true;
}

bool JobExecutor::ResumeJob(const std::string& job_id) {
    std::lock_guard<std::mutex> lock(jobs_mutex_);

    auto it = active_jobs_.find(job_id);
    if (it == active_jobs_.end()) {
        spdlog::warn("Cannot resume job {}: not found", job_id);
        return false;
    }

    if (!it->second->is_paused) {
        spdlog::warn("Cannot resume job {}: not paused", job_id);
        return false;
    }

    spdlog::info("Resuming job {}...", job_id);
    it->second->is_paused = false;

    // Notify waiting threads
    {
        std::lock_guard<std::mutex> pause_lock(it->second->pause_mutex);
        it->second->pause_cv.notify_all();
    }

    return true;
}

bool JobExecutor::StopJob(const std::string& job_id) {
    // StopJob is an alias for CancelJob for P2P API consistency
    return CancelJob(job_id);
}

std::vector<char> JobExecutor::GetCurrentWeights(const std::string& job_id) {
    std::lock_guard<std::mutex> lock(jobs_mutex_);

    auto it = active_jobs_.find(job_id);
    if (it == active_jobs_.end()) {
        spdlog::warn("Cannot get weights for job {}: not found", job_id);
        return {};
    }

    auto* state = it->second.get();
    std::lock_guard<std::mutex> model_lock(state->model_mutex);

    if (!state->model) {
        spdlog::warn("Cannot get weights for job {}: no model", job_id);
        return {};
    }

    // Serialize model weights to binary format using GetParameters()
    try {
        std::vector<char> weights_data;

        // Get all parameters as a map of name -> tensor
        auto params = state->model->GetParameters();
        uint32_t num_params = static_cast<uint32_t>(params.size());

        // Write header: magic + num_params
        const char* magic = "CYXW";
        weights_data.insert(weights_data.end(), magic, magic + 4);
        weights_data.insert(weights_data.end(),
                           reinterpret_cast<char*>(&num_params),
                           reinterpret_cast<char*>(&num_params) + sizeof(num_params));

        // For each parameter, write name and tensor data
        for (auto& [name, tensor] : params) {
            // Write parameter name (length + string)
            uint32_t name_len = static_cast<uint32_t>(name.size());
            weights_data.insert(weights_data.end(),
                               reinterpret_cast<char*>(&name_len),
                               reinterpret_cast<char*>(&name_len) + sizeof(name_len));
            weights_data.insert(weights_data.end(), name.begin(), name.end());

            // Write tensor shape
            auto shape = tensor.Shape();
            uint32_t num_dims = static_cast<uint32_t>(shape.size());
            weights_data.insert(weights_data.end(),
                               reinterpret_cast<char*>(&num_dims),
                               reinterpret_cast<char*>(&num_dims) + sizeof(num_dims));
            for (size_t dim : shape) {
                uint64_t d = dim;
                weights_data.insert(weights_data.end(),
                                   reinterpret_cast<char*>(&d),
                                   reinterpret_cast<char*>(&d) + sizeof(d));
            }

            // Write tensor data
            size_t data_size = tensor.NumElements() * sizeof(float);
            weights_data.insert(weights_data.end(),
                               reinterpret_cast<char*>(&data_size),
                               reinterpret_cast<char*>(&data_size) + sizeof(data_size));

            // Copy tensor data (assumes data is on CPU)
            const char* data_ptr = static_cast<const char*>(tensor.Data());
            if (data_ptr) {
                weights_data.insert(weights_data.end(), data_ptr, data_ptr + data_size);
            } else {
                // Tensor data not accessible, write zeros
                weights_data.insert(weights_data.end(), data_size, 0);
            }
        }

        spdlog::info("Serialized {} bytes of weights for job {} ({} params)",
                    weights_data.size(), job_id, num_params);
        return weights_data;

    } catch (const std::exception& e) {
        spdlog::error("Failed to serialize weights for job {}: {}", job_id, e.what());
        return {};
    }
}

bool JobExecutor::LoadWeights(const std::string& job_id, const std::vector<char>& weights) {
    std::lock_guard<std::mutex> lock(jobs_mutex_);

    auto it = active_jobs_.find(job_id);
    if (it == active_jobs_.end()) {
        spdlog::warn("Cannot load weights for job {}: not found", job_id);
        return false;
    }

    auto* state = it->second.get();
    std::lock_guard<std::mutex> model_lock(state->model_mutex);

    if (!state->model) {
        spdlog::warn("Cannot load weights for job {}: no model", job_id);
        return false;
    }

    if (weights.size() < 8) {
        spdlog::error("Invalid weights data for job {}: too small", job_id);
        return false;
    }

    try {
        // Verify magic header
        if (std::string(weights.data(), 4) != "CYXW") {
            spdlog::error("Invalid weights format for job {}: bad magic", job_id);
            return false;
        }

        size_t offset = 4;
        uint32_t num_params = *reinterpret_cast<const uint32_t*>(weights.data() + offset);
        offset += sizeof(uint32_t);

        // Deserialize parameters into a map
        std::map<std::string, cyxwiz::Tensor> params_map;

        for (uint32_t i = 0; i < num_params; ++i) {
            // Read parameter name
            uint32_t name_len = *reinterpret_cast<const uint32_t*>(weights.data() + offset);
            offset += sizeof(uint32_t);
            std::string name(weights.data() + offset, name_len);
            offset += name_len;

            // Read shape
            uint32_t num_dims = *reinterpret_cast<const uint32_t*>(weights.data() + offset);
            offset += sizeof(uint32_t);

            std::vector<size_t> shape(num_dims);
            for (uint32_t d = 0; d < num_dims; ++d) {
                shape[d] = *reinterpret_cast<const uint64_t*>(weights.data() + offset);
                offset += sizeof(uint64_t);
            }

            // Read data size and data
            size_t data_size = *reinterpret_cast<const size_t*>(weights.data() + offset);
            offset += sizeof(size_t);

            if (offset + data_size > weights.size()) {
                spdlog::error("Truncated weights data for job {}", job_id);
                return false;
            }

            // Create tensor from data
            const void* data_ptr = weights.data() + offset;
            cyxwiz::Tensor tensor(shape, data_ptr, cyxwiz::DataType::Float32);
            params_map[name] = std::move(tensor);

            offset += data_size;
        }

        // Set all parameters at once
        state->model->SetParameters(params_map);

        spdlog::info("Loaded {} bytes of weights for job {} ({} params)",
                    weights.size(), job_id, num_params);
        return true;

    } catch (const std::exception& e) {
        spdlog::error("Failed to load weights for job {}: {}", job_id, e.what());
        return false;
    }
}

bool JobExecutor::IsJobRunning(const std::string& job_id) const {
    std::lock_guard<std::mutex> lock(jobs_mutex_);
    auto it = active_jobs_.find(job_id);
    return it != active_jobs_.end() && it->second->is_running;
}

bool JobExecutor::IsJobPaused(const std::string& job_id) const {
    std::lock_guard<std::mutex> lock(jobs_mutex_);
    auto it = active_jobs_.find(job_id);
    return it != active_jobs_.end() && it->second->is_paused;
}

size_t JobExecutor::GetActiveJobCount() const {
    std::lock_guard<std::mutex> lock(jobs_mutex_);
    return active_jobs_.size();
}

std::vector<std::string> JobExecutor::GetActiveJobIds() const {
    std::lock_guard<std::mutex> lock(jobs_mutex_);
    std::vector<std::string> ids;
    ids.reserve(active_jobs_.size());
    for (const auto& pair : active_jobs_) {
        ids.push_back(pair.first);
    }
    return ids;
}

void JobExecutor::SetProgressCallback(ProgressCallback callback) {
    std::lock_guard<std::mutex> lock(callback_mutex_);
    progress_callback_ = std::move(callback);
}

void JobExecutor::SetCompletionCallback(CompletionCallback callback) {
    std::lock_guard<std::mutex> lock(callback_mutex_);
    completion_callback_ = std::move(callback);
}

void JobExecutor::SetNodeClient(NodeClient* client) {
    node_client_ = client;
}

void JobExecutor::ExecuteJob(const std::string& job_id) {
    spdlog::info("Worker thread started for job: {}", job_id);

    JobState* state = nullptr;
    int device_id = -1;
    {
        std::lock_guard<std::mutex> lock(jobs_mutex_);
        auto it = active_jobs_.find(job_id);
        if (it == active_jobs_.end()) {
            spdlog::error("Job {} not found in active jobs", job_id);
            return;
        }
        state = it->second.get();
        device_id = state->assigned_device_id;
    }

    bool success = false;
    std::string error_msg;

    try {
        // Set up device context if using device pool
        std::unique_ptr<core::ScopedDeviceContext> device_context;
        if (use_device_pool_ && device_id >= 0) {
            device_context = std::make_unique<core::ScopedDeviceContext>(device_id);
            if (!device_context->IsValid()) {
                spdlog::error("Failed to set device context for device {}", device_id);
                error_msg = "Failed to set device context";
                success = false;
            }
        }

        if (error_msg.empty()) {
            // Run the training
            success = RunTraining(job_id, state);

            if (!success && !state->should_cancel) {
                error_msg = "Training failed";
            } else if (state->should_cancel) {
                error_msg = "Job cancelled by user";
                success = false;
            }
        }

    } catch (const std::exception& e) {
        spdlog::error("Exception during job execution: {}", e.what());
        error_msg = std::string("Exception: ") + e.what();
        success = false;
    }

    // Mark as not running
    state->is_running = false;

    // Release the device back to the pool
    if (use_device_pool_ && device_pool_ && device_id >= 0) {
        device_pool_->ReleaseDevice(device_id, success);
        spdlog::info("Released device {} for job {} (success: {})", device_id, job_id, success);

        // Try to process any pending jobs
        ProcessPendingJobs();
    }

    // Report final result to Central Server
    if (node_client_ && node_client_->IsRegistered()) {
        // Build final metrics map
        std::map<std::string, double> final_metrics;
        final_metrics["loss"] = state->current_metrics.loss;
        final_metrics["accuracy"] = state->current_metrics.accuracy;

        // Calculate total compute time
        auto now = std::chrono::steady_clock::now();
        auto total_time = std::chrono::duration_cast<std::chrono::milliseconds>(
            now - state->start_time);

        // Send final result to Central Server
        node_client_->ReportJobResult(
            job_id,
            success ? protocol::STATUS_SUCCESS : protocol::STATUS_FAILED,
            final_metrics,
            "",  // model_weights_uri - TODO: implement model saving
            "",  // model_weights_hash - TODO: implement model saving
            0,   // model_size - TODO: implement model saving
            total_time.count(),
            error_msg
        );
    }

    // Call completion callback
    {
        std::lock_guard<std::mutex> lock(callback_mutex_);
        if (completion_callback_) {
            completion_callback_(job_id, success, error_msg);
        }
    }

    spdlog::info("Job {} finished. Success: {}", job_id, success);

    // Clean up job state (detach thread first)
    {
        std::lock_guard<std::mutex> lock(jobs_mutex_);
        auto it = active_jobs_.find(job_id);
        if (it != active_jobs_.end()) {
            if (it->second->worker_thread.joinable()) {
                it->second->worker_thread.detach();
            }
            active_jobs_.erase(it);
        }
    }
}

namespace {

// The job's dataset as Data Input files for RunGraphTrainingJob. Empty: the
// graph's own Data Input paths (files on this node). "file://<path>" or
// "file://parquet/<path>" / "file://arrow/<path>": that file replaces the
// graph's single Data Input. Legacy MNIST/CIFAR/CSV/mock URIs are refused.
bool ResolveJobDatasetFiles(const std::string& dataset_uri, const std::string& graph_json,
                            std::map<std::string, std::string>& files, std::string& error) {
    if (dataset_uri.empty()) return true;
    if (dataset_uri.rfind("mock://", 0) == 0) {
        error = "mock datasets are not trained (dataset_uri '" + dataset_uri + "')";
        return false;
    }
    std::string path = dataset_uri;
    if (path.rfind("file://", 0) == 0) {
        path = path.substr(7);
        for (const char* legacy : {"mnist/", "cifar10/", "csv/"}) {
            if (path.rfind(legacy, 0) == 0) {
                error = "dataset_uri '" + dataset_uri + "' uses a retired loader; send Parquet or Arrow IPC "
                        "(file://<path>) or leave it empty to use the graph's Data Input paths";
                return false;
            }
        }
        for (const char* typed : {"parquet/", "arrow/"}) {
            if (path.rfind(typed, 0) == 0) path = path.substr(std::string(typed).size());
        }
    } else if (dataset_uri.find("://") != std::string::npos) {
        error = "unsupported dataset_uri scheme: " + dataset_uri;
        return false;
    }
    std::vector<std::string> inputs;
    try {
        const auto graph = nlohmann::json::parse(graph_json);
        for (const auto& node : graph.at("nodes")) {
            if (node.value("type", -1) != static_cast<int>(gui::NodeType::DataInput)) continue;
            const auto params = node.value("parameters", nlohmann::json::object());
            inputs.push_back(params.value("dataset_name", std::string()));
        }
    } catch (const std::exception& e) {
        error = std::string("the job's graph is not valid JSON: ") + e.what();
        return false;
    }
    if (inputs.size() != 1 || inputs.front().empty()) {
        error = "dataset_uri names one file but the graph has " + std::to_string(inputs.size()) +
                " Data Inputs; leave dataset_uri empty to use the graph's own paths";
        return false;
    }
    files[inputs.front()] = path;
    return true;
}

}  // namespace

bool JobExecutor::RunTraining(const std::string& job_id, JobState* state) {
    spdlog::info("Starting training for job: {}", job_id);
    const auto& config = state->config;

    // TOFIX118 P2: the node trains with the Engine's compiler, preparation and
    // executor (RunGraphTrainingJob); unsupported graphs and inputs fail with
    // the reason instead of training something else.
    cyxwiz::GraphTrainingJobRequest request;
    request.graph_json = config.model_definition();
    if (request.graph_json.empty()) {
        throw std::runtime_error("the job has no model definition (graph)");
    }
    std::string error;
    if (!ResolveJobDatasetFiles(config.dataset_uri(), request.graph_json, request.dataset_files, error)) {
        throw std::runtime_error(error);
    }
    request.epochs_override = config.epochs();
    request.batch_size_override = config.batch_size();
    if (!config.hyperparameters().empty()) {
        spdlog::warn("Job {}: hyperparameters are not applied; the graph's optimizer and Data Loader "
                     "settings are used", job_id);
    }

    cyxwiz::GraphTrainingJobCallbacks callbacks;
    callbacks.on_start = [state](int epochs, int batch_size) {
        state->current_metrics.total_epochs = epochs;
        spdlog::info("Beginning training: {} epochs, batch size {}", epochs, batch_size);
    };
    callbacks.on_epoch = [this, &job_id, state](int epoch, float train_loss, float train_acc, float val_loss,
                                                float val_acc, float) {
        auto& metrics = state->current_metrics;
        metrics.current_epoch = epoch;
        metrics.loss = train_loss;
        metrics.accuracy = train_acc;
        if (val_loss >= 0.0f) metrics.custom_metrics["val_loss"] = val_loss;
        if (val_acc >= 0.0f) metrics.custom_metrics["val_accuracy"] = val_acc;
        metrics.time_elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                                      std::chrono::steady_clock::now() - state->start_time)
                                      .count();
        ReportProgress(job_id, state);
        spdlog::info("Job {} - Epoch {}/{}: Loss={:.4f}, Acc={:.2f}%", job_id, epoch, metrics.total_epochs,
                     metrics.loss, metrics.accuracy * 100.0);
    };
    callbacks.should_cancel = [state] { return state->should_cancel.load(); };

    auto result = cyxwiz::RunGraphTrainingJob(request, callbacks);
    if (result.cancelled) {
        spdlog::info("Training cancelled for job: {}", job_id);
        return false;
    }
    if (!result.ok) {
        throw std::runtime_error(result.error);
    }
    {
        std::lock_guard<std::mutex> lock(state->model_mutex);
        state->model = std::move(result.model);
    }
    spdlog::info("Training completed successfully for job: {}", job_id);
    return true;
}

bool JobExecutor::SaveResults(
    const std::string& job_id,
    [[maybe_unused]] cyxwiz::Model* model,
    [[maybe_unused]] const TrainingMetrics& final_metrics)
{
    spdlog::info("Saving results for job: {}", job_id);

    // TODO: Implement actual result saving
    // - Save trained model weights
    // - Save training metrics
    // - Generate result artifacts

    return true;
}

void JobExecutor::ReportProgress(const std::string& job_id, JobState* state) {
    double progress = static_cast<double>(state->current_metrics.current_epoch) /
                     state->current_metrics.total_epochs;

    // Call progress callback
    {
        std::lock_guard<std::mutex> lock(callback_mutex_);
        if (progress_callback_) {
            progress_callback_(job_id, progress, state->current_metrics);
        }
    }

    // Report to Central Server via NodeClient if available
    if (node_client_ && node_client_->IsRegistered()) {
        // Build metrics map
        std::map<std::string, double> metrics;
        metrics["loss"] = state->current_metrics.loss;
        metrics["accuracy"] = state->current_metrics.accuracy;
        metrics["learning_rate"] = state->current_metrics.learning_rate;

        // Add custom metrics
        for (const auto& [key, value] : state->current_metrics.custom_metrics) {
            metrics[key] = value;
        }

        // Send status update to Central Server
        node_client_->UpdateJobStatus(
            job_id,
            protocol::STATUS_IN_PROGRESS,
            progress,
            metrics,
            state->current_metrics.current_epoch,
            ""  // log_message - empty for now
        );
    }
}

void JobExecutor::ProcessPendingJobs() {
    // Try to start any pending jobs when a device becomes available
    if (!use_device_pool_ || !device_pool_) {
        return;
    }

    while (true) {
        protocol::JobConfig next_job;

        // Get next pending job
        {
            std::lock_guard<std::mutex> lock(pending_mutex_);
            if (pending_jobs_.empty()) {
                return;  // No more pending jobs
            }

            // Check if a device is available before dequeuing
            if (device_pool_->GetAvailableDeviceCount() == 0) {
                return;  // No device available
            }

            next_job = pending_jobs_.front();
            pending_jobs_.pop();
        }

        std::string job_id = next_job.job_id();
        spdlog::info("Processing pending job: {}", job_id);

        // Try to acquire device and start the job
        size_t required_memory_mb = 0;  // TODO: Parse from job config
        int device_id = device_pool_->AcquireDevice(job_id, required_memory_mb);

        if (device_id < 0) {
            // Device became unavailable, put job back in queue
            spdlog::warn("Device became unavailable for pending job {}, re-queueing", job_id);
            std::lock_guard<std::mutex> lock(pending_mutex_);
            pending_jobs_.push(next_job);
            return;
        }

        spdlog::info("Acquired device {} for pending job {}", device_id, job_id);

        // Create job state
        auto job_state = std::make_unique<JobState>();
        job_state->config = next_job;
        job_state->is_running = true;
        job_state->should_cancel = false;
        job_state->start_time = std::chrono::steady_clock::now();
        job_state->assigned_device_id = device_id;

        // Store and launch
        {
            std::lock_guard<std::mutex> lock(jobs_mutex_);
            active_jobs_[job_id] = std::move(job_state);
            active_jobs_[job_id]->worker_thread = std::thread(&JobExecutor::ExecuteJob, this, job_id);
        }

        spdlog::info("Pending job {} started on device {}", job_id, device_id);
    }
}

} // namespace servernode
} // namespace cyxwiz
