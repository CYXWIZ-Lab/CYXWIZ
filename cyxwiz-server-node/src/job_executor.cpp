#include "job_executor.h"
#include <spdlog/spdlog.h>
#include <nlohmann/json.hpp>
#include <chrono>
#include <cmath>
#include <random>

namespace cyxwiz {
namespace servernode {

JobExecutor::JobExecutor(const std::string& node_id, cyxwiz::Device* device)
    : node_id_(node_id)
    , device_(device)
    , node_client_(nullptr)
{
    spdlog::info("JobExecutor initialized for node: {}", node_id_);
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

    // Create job state
    auto job_state = std::make_unique<JobState>();
    job_state->config = job_config;
    job_state->is_running = true;
    job_state->should_cancel = false;
    job_state->start_time = std::chrono::steady_clock::now();

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

    spdlog::info("Job {} started successfully", job_id);
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

bool JobExecutor::IsJobRunning(const std::string& job_id) const {
    std::lock_guard<std::mutex> lock(jobs_mutex_);
    auto it = active_jobs_.find(job_id);
    return it != active_jobs_.end() && it->second->is_running;
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
    {
        std::lock_guard<std::mutex> lock(jobs_mutex_);
        auto it = active_jobs_.find(job_id);
        if (it == active_jobs_.end()) {
            spdlog::error("Job {} not found in active jobs", job_id);
            return;
        }
        state = it->second.get();
    }

    bool success = false;
    std::string error_msg;

    try {
        // Run the training
        success = RunTraining(job_id, state);

        if (!success && !state->should_cancel) {
            error_msg = "Training failed";
        } else if (state->should_cancel) {
            error_msg = "Job cancelled by user";
            success = false;
        }

    } catch (const std::exception& e) {
        spdlog::error("Exception during job execution: {}", e.what());
        error_msg = std::string("Exception: ") + e.what();
        success = false;
    }

    // Mark as not running
    state->is_running = false;

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

bool JobExecutor::LoadDataset(
    const std::string& dataset_uri,
    std::vector<cyxwiz::Tensor>& train_data,
    std::vector<cyxwiz::Tensor>& train_labels)
{
    spdlog::info("Loading dataset from: {}", dataset_uri);

    // Mock dataset loading - generate random data
    // In a real implementation, this would load from files, HTTP, etc.

    if (dataset_uri.find("mock://") == 0) {
        // Generate mock data
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(0.0f, 1.0f);

        // Create 100 samples of 784 features (28x28 images)
        int num_samples = 100;
        int input_dim = 784;
        int num_classes = 10;

        for (int i = 0; i < num_samples; ++i) {
            std::vector<size_t> shape = {1, static_cast<size_t>(input_dim)};
            cyxwiz::Tensor data(shape);

            std::vector<size_t> label_shape = {1, static_cast<size_t>(num_classes)};
            cyxwiz::Tensor label(label_shape);

            train_data.push_back(std::move(data));
            train_labels.push_back(std::move(label));
        }

        spdlog::info("Generated {} mock training samples", num_samples);
        return true;
    }

    spdlog::error("Unsupported dataset URI: {}", dataset_uri);
    return false;
}

std::unique_ptr<cyxwiz::Model> JobExecutor::BuildModel(const std::string& model_definition) {
    spdlog::info("Building model from definition");

    // TODO: Parse model_definition JSON and build actual model
    // For now, return nullptr as we're using mock training

    return nullptr;
}

bool JobExecutor::RunTraining(const std::string& job_id, JobState* state) {
    spdlog::info("Starting training for job: {}", job_id);

    const auto& config = state->config;
    int total_epochs = config.epochs();
    int batch_size = config.batch_size();

    // Parse hyperparameters
    auto hyperparams = ParseHyperparameters(config.hyperparameters());
    double learning_rate = 0.001;
    if (hyperparams.count("learning_rate") > 0) {
        learning_rate = hyperparams["learning_rate"];
    }

    // Load dataset
    std::vector<cyxwiz::Tensor> train_data;
    std::vector<cyxwiz::Tensor> train_labels;

    if (!LoadDataset(config.dataset_uri(), train_data, train_labels)) {
        spdlog::error("Failed to load dataset");
        return false;
    }

    // Mock training loop
    spdlog::info("Beginning training: {} epochs, batch size {}, lr {}",
                total_epochs, batch_size, learning_rate);

    // Initialize metrics
    state->current_metrics.total_epochs = total_epochs;
    state->current_metrics.learning_rate = learning_rate;

    // Simulate training with realistic loss decay
    double initial_loss = 2.3;  // Typical for random initialization
    double target_loss = 0.1;   // Good final loss

    for (int epoch = 1; epoch <= total_epochs; ++epoch) {
        // Check for cancellation
        if (state->should_cancel) {
            spdlog::info("Training cancelled at epoch {}", epoch);
            return false;
        }

        // Update metrics
        state->current_metrics.current_epoch = epoch;

        // Simulate loss decay (exponential)
        double progress = static_cast<double>(epoch) / total_epochs;
        double loss = initial_loss * std::exp(-3.0 * progress) + target_loss;
        state->current_metrics.loss = loss;

        // Simulate accuracy increase
        state->current_metrics.accuracy = 0.1 + 0.85 * progress;

        // Simulate samples processed
        state->current_metrics.samples_processed = epoch * static_cast<int64_t>(train_data.size());

        // Calculate elapsed time
        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            now - state->start_time);
        state->current_metrics.time_elapsed_ms = elapsed.count();

        // Report progress every 5 epochs or on last epoch
        if (epoch % 5 == 0 || epoch == total_epochs) {
            double progress_pct = static_cast<double>(epoch) / total_epochs;
            ReportProgress(job_id, state);

            spdlog::info("Job {} - Epoch {}/{}: Loss={:.4f}, Acc={:.2f}%",
                        job_id, epoch, total_epochs, loss,
                        state->current_metrics.accuracy * 100.0);
        }

        // Simulate epoch duration (100ms per epoch for demo)
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    // Save results
    // TODO: Implement actual model saving
    spdlog::info("Training completed successfully for job: {}", job_id);

    return true;
}

bool JobExecutor::SaveResults(
    const std::string& job_id,
    cyxwiz::Model* model,
    const TrainingMetrics& final_metrics)
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
    if (node_client_) {
        // TODO: Implement ReportJobProgress on NodeClient
        // node_client_->ReportJobProgress(job_id, status);
    }
}

std::unordered_map<std::string, double> JobExecutor::ParseHyperparameters(
    const google::protobuf::Map<std::string, std::string>& hyper_params)
{
    std::unordered_map<std::string, double> result;

    for (const auto& pair : hyper_params) {
        try {
            result[pair.first] = std::stod(pair.second);
        } catch (const std::exception& e) {
            spdlog::warn("Failed to parse hyperparameter {}: {}", pair.first, e.what());
        }
    }

    return result;
}

std::unique_ptr<cyxwiz::Optimizer> JobExecutor::CreateOptimizer(
    const std::unordered_map<std::string, double>& hyperparameters)
{
    // TODO: Implement optimizer creation
    // For now, return nullptr as we're using mock training
    return nullptr;
}

bool JobExecutor::LoadLocalDataset(
    const std::string& file_path,
    std::vector<cyxwiz::Tensor>& train_data,
    std::vector<cyxwiz::Tensor>& train_labels)
{
    spdlog::info("Loading local dataset from: {}", file_path);

    // TODO: Implement local file loading
    // - Support CSV, NPY, HDF5 formats
    // - Parse and convert to Tensor format

    return false;
}

bool JobExecutor::LoadRemoteDataset(
    const std::string& uri,
    std::vector<cyxwiz::Tensor>& train_data,
    std::vector<cyxwiz::Tensor>& train_labels)
{
    spdlog::info("Loading remote dataset from: {}", uri);

    // TODO: Implement HTTP/HTTPS dataset download
    // - Download file to temp location
    // - Parse and load

    return false;
}

bool JobExecutor::ParseCSVDataset(
    const std::string& csv_content,
    std::vector<cyxwiz::Tensor>& train_data,
    std::vector<cyxwiz::Tensor>& train_labels)
{
    spdlog::info("Parsing CSV dataset");

    // TODO: Implement CSV parsing
    // - Parse header
    // - Extract features and labels
    // - Convert to Tensor format

    return false;
}

} // namespace servernode
} // namespace cyxwiz
