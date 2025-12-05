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

// Backend includes for real training
#include <cyxwiz/cyxwiz.h>
#include <cyxwiz/layers/linear.h>
#include <cyxwiz/loss.h>

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

bool JobExecutor::LoadDataset(
    const std::string& dataset_uri,
    std::vector<cyxwiz::Tensor>& train_data,
    std::vector<cyxwiz::Tensor>& train_labels)
{
    spdlog::info("Loading dataset from: {}", dataset_uri);

    // Parse URI format: protocol://type/path
    // Examples:
    //   file://mnist/./data/mnist
    //   file://cifar10/./data/cifar10
    //   file://csv/./data/train.csv
    //   mock://random

    if (dataset_uri.find("mock://") == 0) {
        return LoadMockDataset(train_data, train_labels);
    }

    if (dataset_uri.find("file://") == 0) {
        std::string remainder = dataset_uri.substr(7); // Remove "file://"

        // Find first slash to separate type from path
        size_t slash_pos = remainder.find('/');
        if (slash_pos == std::string::npos) {
            spdlog::error("Invalid file URI format: {}", dataset_uri);
            return false;
        }

        std::string dataset_type = remainder.substr(0, slash_pos);
        std::string path = remainder.substr(slash_pos + 1);

        spdlog::info("Dataset type: {}, path: {}", dataset_type, path);

        if (dataset_type == "mnist") {
            return LoadMNISTDataset(path, train_data, train_labels);
        } else if (dataset_type == "cifar10") {
            return LoadCIFAR10Dataset(path, train_data, train_labels);
        } else if (dataset_type == "csv") {
            return LoadCSVDataset(path, train_data, train_labels);
        } else {
            spdlog::error("Unknown dataset type: {}", dataset_type);
            return false;
        }
    }

    spdlog::error("Unsupported dataset URI scheme: {}", dataset_uri);
    return false;
}

bool JobExecutor::LoadMockDataset(
    std::vector<cyxwiz::Tensor>& train_data,
    std::vector<cyxwiz::Tensor>& train_labels)
{
    spdlog::info("Generating mock dataset...");

    // Generate random data for testing
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);
    std::uniform_int_distribution<int> label_dis(0, 9);

    int num_samples = 1000;
    int input_dim = 784;  // 28x28 for MNIST-like
    int num_classes = 10;

    for (int i = 0; i < num_samples; ++i) {
        // Create input tensor
        std::vector<float> data(input_dim);
        for (int j = 0; j < input_dim; ++j) {
            data[j] = dis(gen);
        }

        std::vector<size_t> data_shape = {1, static_cast<size_t>(input_dim)};
        cyxwiz::Tensor tensor(data_shape, data.data(), cyxwiz::DataType::Float32);
        train_data.push_back(std::move(tensor));

        // Create one-hot label tensor
        std::vector<float> label(num_classes, 0.0f);
        label[label_dis(gen)] = 1.0f;

        std::vector<size_t> label_shape = {1, static_cast<size_t>(num_classes)};
        cyxwiz::Tensor label_tensor(label_shape, label.data(), cyxwiz::DataType::Float32);
        train_labels.push_back(std::move(label_tensor));
    }

    spdlog::info("Generated {} mock training samples ({} features, {} classes)",
                 num_samples, input_dim, num_classes);
    return true;
}

bool JobExecutor::LoadMNISTDataset(
    const std::string& path,
    std::vector<cyxwiz::Tensor>& train_data,
    std::vector<cyxwiz::Tensor>& train_labels)
{
    spdlog::info("Loading MNIST dataset from: {}", path);

    // MNIST binary format:
    // train-images-idx3-ubyte: magic (4 bytes), num_images (4), rows (4), cols (4), then pixels
    // train-labels-idx1-ubyte: magic (4 bytes), num_labels (4), then labels

    std::string images_file = path + "/train-images-idx3-ubyte";
    std::string labels_file = path + "/train-labels-idx1-ubyte";

    // Try alternate naming
    if (!std::filesystem::exists(images_file)) {
        images_file = path + "/train-images.idx3-ubyte";
        labels_file = path + "/train-labels.idx1-ubyte";
    }

    std::ifstream images(images_file, std::ios::binary);
    std::ifstream labels(labels_file, std::ios::binary);

    if (!images.is_open() || !labels.is_open()) {
        spdlog::error("Failed to open MNIST files in: {}", path);
        spdlog::error("Tried: {}", images_file);
        return false;
    }

    // Read image file header (big-endian)
    auto read_int32_be = [](std::ifstream& file) -> int32_t {
        unsigned char bytes[4];
        file.read(reinterpret_cast<char*>(bytes), 4);
        return (bytes[0] << 24) | (bytes[1] << 16) | (bytes[2] << 8) | bytes[3];
    };

    int32_t magic_images = read_int32_be(images);
    int32_t num_images = read_int32_be(images);
    int32_t num_rows = read_int32_be(images);
    int32_t num_cols = read_int32_be(images);

    int32_t magic_labels = read_int32_be(labels);
    int32_t num_labels = read_int32_be(labels);

    if (magic_images != 2051 || magic_labels != 2049) {
        spdlog::error("Invalid MNIST file format (magic: {}, {})", magic_images, magic_labels);
        return false;
    }

    spdlog::info("MNIST: {} images, {}x{} pixels", num_images, num_rows, num_cols);

    int image_size = num_rows * num_cols;
    std::vector<uint8_t> image_buffer(image_size);
    int num_classes = 10;

    // Limit for testing (use all for production)
    int max_samples = std::min(num_images, 60000);

    for (int i = 0; i < max_samples; ++i) {
        // Read image pixels
        images.read(reinterpret_cast<char*>(image_buffer.data()), image_size);

        // Convert to float and normalize
        std::vector<float> image_float(image_size);
        for (int j = 0; j < image_size; ++j) {
            image_float[j] = static_cast<float>(image_buffer[j]) / 255.0f;
        }

        std::vector<size_t> data_shape = {1, static_cast<size_t>(image_size)};
        cyxwiz::Tensor tensor(data_shape, image_float.data(), cyxwiz::DataType::Float32);
        train_data.push_back(std::move(tensor));

        // Read label and create one-hot encoding
        uint8_t label;
        labels.read(reinterpret_cast<char*>(&label), 1);

        std::vector<float> label_onehot(num_classes, 0.0f);
        if (label < num_classes) {
            label_onehot[label] = 1.0f;
        }

        std::vector<size_t> label_shape = {1, static_cast<size_t>(num_classes)};
        cyxwiz::Tensor label_tensor(label_shape, label_onehot.data(), cyxwiz::DataType::Float32);
        train_labels.push_back(std::move(label_tensor));
    }

    spdlog::info("Loaded {} MNIST samples", train_data.size());
    return true;
}

bool JobExecutor::LoadCIFAR10Dataset(
    const std::string& path,
    std::vector<cyxwiz::Tensor>& train_data,
    std::vector<cyxwiz::Tensor>& train_labels)
{
    spdlog::info("Loading CIFAR-10 dataset from: {}", path);

    // CIFAR-10 binary format:
    // Each file has 10000 samples
    // Each sample: 1 byte label + 3072 bytes image (32x32x3)

    std::vector<std::string> batch_files = {
        path + "/data_batch_1.bin",
        path + "/data_batch_2.bin",
        path + "/data_batch_3.bin",
        path + "/data_batch_4.bin",
        path + "/data_batch_5.bin"
    };

    const int image_size = 32 * 32 * 3;
    int num_classes = 10;

    for (const auto& batch_file : batch_files) {
        std::ifstream file(batch_file, std::ios::binary);

        if (!file.is_open()) {
            spdlog::warn("Could not open CIFAR-10 batch: {}", batch_file);
            continue;
        }

        spdlog::info("Reading batch: {}", batch_file);

        for (int i = 0; i < 10000; ++i) {
            // Read label
            uint8_t label;
            file.read(reinterpret_cast<char*>(&label), 1);
            if (file.eof()) break;

            // Read image
            std::vector<uint8_t> image_buffer(image_size);
            file.read(reinterpret_cast<char*>(image_buffer.data()), image_size);

            if (file.gcount() != image_size) break;

            // Convert to float and normalize
            std::vector<float> image_float(image_size);
            for (int j = 0; j < image_size; ++j) {
                image_float[j] = static_cast<float>(image_buffer[j]) / 255.0f;
            }

            std::vector<size_t> data_shape = {1, static_cast<size_t>(image_size)};
            cyxwiz::Tensor tensor(data_shape, image_float.data(), cyxwiz::DataType::Float32);
            train_data.push_back(std::move(tensor));

            // Create one-hot label
            std::vector<float> label_onehot(num_classes, 0.0f);
            if (label < num_classes) {
                label_onehot[label] = 1.0f;
            }

            std::vector<size_t> label_shape = {1, static_cast<size_t>(num_classes)};
            cyxwiz::Tensor label_tensor(label_shape, label_onehot.data(), cyxwiz::DataType::Float32);
            train_labels.push_back(std::move(label_tensor));
        }
    }

    if (train_data.empty()) {
        spdlog::error("No CIFAR-10 samples loaded");
        return false;
    }

    spdlog::info("Loaded {} CIFAR-10 samples", train_data.size());
    return true;
}

bool JobExecutor::LoadCSVDataset(
    const std::string& path,
    std::vector<cyxwiz::Tensor>& train_data,
    std::vector<cyxwiz::Tensor>& train_labels)
{
    spdlog::info("Loading CSV dataset from: {}", path);

    std::ifstream file(path);
    if (!file.is_open()) {
        spdlog::error("Failed to open CSV file: {}", path);
        return false;
    }

    std::string line;
    int line_num = 0;
    bool has_header = false;
    int num_features = -1;
    int max_label = 0;

    // First pass: determine structure and max label
    std::vector<std::vector<float>> all_features;
    std::vector<int> all_labels;

    while (std::getline(file, line)) {
        line_num++;
        if (line.empty()) continue;

        std::vector<float> features;
        std::stringstream ss(line);
        std::string value;

        while (std::getline(ss, value, ',')) {
            try {
                float f = std::stof(value);
                features.push_back(f);
            } catch (...) {
                if (line_num == 1) {
                    has_header = true;
                    break;
                }
            }
        }

        if (features.empty()) continue;
        if (has_header && line_num == 1) continue;

        // Last column is label
        int label = static_cast<int>(features.back());
        features.pop_back();

        if (num_features < 0) {
            num_features = static_cast<int>(features.size());
        }

        all_features.push_back(features);
        all_labels.push_back(label);
        max_label = std::max(max_label, label);
    }

    if (all_features.empty()) {
        spdlog::error("No samples loaded from CSV");
        return false;
    }

    int num_classes = max_label + 1;
    spdlog::info("CSV: {} samples, {} features, {} classes", all_features.size(), num_features, num_classes);

    // Convert to tensors
    for (size_t i = 0; i < all_features.size(); ++i) {
        std::vector<size_t> data_shape = {1, static_cast<size_t>(num_features)};
        cyxwiz::Tensor tensor(data_shape, all_features[i].data(), cyxwiz::DataType::Float32);
        train_data.push_back(std::move(tensor));

        // Create one-hot label
        std::vector<float> label_onehot(num_classes, 0.0f);
        if (all_labels[i] < num_classes) {
            label_onehot[all_labels[i]] = 1.0f;
        }

        std::vector<size_t> label_shape = {1, static_cast<size_t>(num_classes)};
        cyxwiz::Tensor label_tensor(label_shape, label_onehot.data(), cyxwiz::DataType::Float32);
        train_labels.push_back(std::move(label_tensor));
    }

    spdlog::info("Loaded {} CSV samples", train_data.size());
    return true;
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
    double learning_rate = 0.001;
    if (hyperparameters.count("learning_rate") > 0) {
        learning_rate = hyperparameters.at("learning_rate");
    }

    // Check for optimizer type in hyperparameters
    cyxwiz::OptimizerType opt_type = cyxwiz::OptimizerType::Adam; // Default

    // Note: optimizer type is passed as string in hyperparameters
    // For simplicity, we check if "sgd" or "adamw" keys exist with value 1.0
    // Actually, we need to check the string value from the original map

    // Create and return optimizer
    spdlog::info("Creating optimizer with learning_rate: {}", learning_rate);
    return cyxwiz::CreateOptimizer(opt_type, learning_rate);
}

} // namespace servernode
} // namespace cyxwiz
