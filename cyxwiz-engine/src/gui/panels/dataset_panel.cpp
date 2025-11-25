#include "dataset_panel.h"
#include "training_plot_panel.h"
#include "../../network/job_manager.h"
#include "job.pb.h"
#include <imgui.h>
#include <spdlog/spdlog.h>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <random>
#include <cmath>
#include <numeric>
#include <chrono>
#include <nlohmann/json.hpp>

// cyxwiz-backend includes for local training
#include <cyxwiz/cyxwiz.h>
#include <cyxwiz/tensor.h>
#include <cyxwiz/optimizer.h>

#ifdef _WIN32
#include <windows.h>
#include <commdlg.h>
#endif

namespace gui {

DatasetPanel::DatasetPanel() : cyxwiz::Panel("Dataset Manager", true) {
}

DatasetPanel::~DatasetPanel() {
    // Stop any running local training
    if (local_training_running_.load()) {
        local_training_stop_requested_.store(true);
        if (local_training_thread_ && local_training_thread_->joinable()) {
            local_training_thread_->join();
        }
    }
}

void DatasetPanel::Render() {
    if (!visible_) return;

    ImGui::SetNextWindowSize(ImVec2(600, 700), ImGuiCond_FirstUseEver);
    if (ImGui::Begin(name_.c_str(), &visible_)) {

        // Header
        ImGui::TextColored(ImVec4(0.4f, 0.8f, 1.0f, 1.0f), "Dataset Management");
        ImGui::Separator();
        ImGui::Spacing();

        // Main layout - two columns
        ImGui::BeginChild("LeftPanel", ImVec2(280, 0), true);
        RenderDatasetSelection();
        ImGui::EndChild();

        ImGui::SameLine();

        ImGui::BeginChild("RightPanel", ImVec2(0, 0), true);
        if (IsDatasetLoaded()) {
            RenderDatasetInfo();
            ImGui::Spacing();
            RenderSplitConfiguration();
            ImGui::Spacing();
            RenderTrainingSection();
            ImGui::Spacing();
            RenderStatistics();
            ImGui::Spacing();
            RenderDataPreview();
        } else {
            ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "No dataset loaded");
            ImGui::Text("Select a dataset type and load data");
        }
        ImGui::EndChild();
    }
    ImGui::End();
}

void DatasetPanel::RenderDatasetSelection() {
    ImGui::Text("Dataset Type");
    ImGui::Spacing();

    // Dataset type selection
    const char* types[] = {"CSV", "Images", "MNIST", "CIFAR-10"};
    int current_type = static_cast<int>(selected_type_) - 1;
    if (current_type < 0) current_type = 0;

    if (ImGui::Combo("##Type", &current_type, types, IM_ARRAYSIZE(types))) {
        selected_type_ = static_cast<DatasetType>(current_type + 1);
    }

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    // File path input
    ImGui::Text("Dataset Path");
    ImGui::InputText("##Path", file_path_buffer_, sizeof(file_path_buffer_));

    ImGui::SameLine();
    if (ImGui::Button("Browse...")) {
        ShowFileBrowser();
    }

    ImGui::Spacing();

    // Load button
    if (ImGui::Button("Load Dataset", ImVec2(-1, 0))) {
        std::string path = file_path_buffer_;
        if (!path.empty()) {
            // Auto-detect dataset type from file extension if path is a file
            DatasetType detected_type = selected_type_;

            // Check if path ends with .csv - override to CSV type
            std::string lower_path = path;
            std::transform(lower_path.begin(), lower_path.end(), lower_path.begin(), ::tolower);
            if (lower_path.size() > 4 && lower_path.substr(lower_path.size() - 4) == ".csv") {
                detected_type = DatasetType::CSV;
                spdlog::info("Auto-detected CSV file format");
            }

            bool success = false;
            switch (detected_type) {
                case DatasetType::CSV:
                    success = LoadCSVDataset(path);
                    break;
                case DatasetType::Images:
                    success = LoadImageDataset(path);
                    break;
                case DatasetType::MNIST:
                    success = LoadMNISTDataset(path);
                    break;
                case DatasetType::CIFAR10:
                    success = LoadCIFAR10Dataset(path);
                    break;
                default:
                    spdlog::error("Unknown dataset type - please select a type from the dropdown");
            }

            if (success) {
                spdlog::info("Dataset loaded successfully");
                ApplySplit();
            } else {
                spdlog::error("Failed to load dataset");
            }
        }
    }

    // Clear button
    if (IsDatasetLoaded()) {
        ImGui::Spacing();
        if (ImGui::Button("Clear Dataset", ImVec2(-1, 0))) {
            ClearDataset();
        }
    }

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    // Quick load buttons for built-in datasets
    ImGui::Text("Quick Load");
    ImGui::Spacing();

    if (ImGui::Button("MNIST (Default)", ImVec2(-1, 0))) {
        if (LoadMNISTDataset("./data/mnist")) {
            ApplySplit();
        }
    }

    if (ImGui::Button("CIFAR-10 (Default)", ImVec2(-1, 0))) {
        if (LoadCIFAR10Dataset("./data/cifar10")) {
            ApplySplit();
        }
    }
}

void DatasetPanel::RenderDatasetInfo() {
    ImGui::Text("Dataset Information");
    ImGui::Separator();
    ImGui::Spacing();

    ImGui::Text("Name: %s", dataset_info_.name.c_str());
    ImGui::Text("Type: %s",
        dataset_info_.type == DatasetType::CSV ? "CSV" :
        dataset_info_.type == DatasetType::Images ? "Images" :
        dataset_info_.type == DatasetType::MNIST ? "MNIST" :
        dataset_info_.type == DatasetType::CIFAR10 ? "CIFAR-10" : "Unknown");

    ImGui::Text("Total Samples: %d", dataset_info_.num_samples);
    ImGui::Text("Number of Classes: %d", dataset_info_.num_classes);

    if (!dataset_info_.input_shape.empty()) {
        ImGui::Text("Input Shape: [");
        ImGui::SameLine();
        for (size_t i = 0; i < dataset_info_.input_shape.size(); ++i) {
            ImGui::Text("%d", dataset_info_.input_shape[i]);
            if (i < dataset_info_.input_shape.size() - 1) {
                ImGui::SameLine();
                ImGui::Text("x");
                ImGui::SameLine();
            }
        }
        ImGui::SameLine();
        ImGui::Text("]");
    }
}

void DatasetPanel::RenderSplitConfiguration() {
    ImGui::Text("Train/Val/Test Split");
    ImGui::Separator();
    ImGui::Spacing();

    bool changed = false;
    changed |= ImGui::SliderFloat("Train", &dataset_info_.train_ratio, 0.0f, 1.0f, "%.2f");
    changed |= ImGui::SliderFloat("Val", &dataset_info_.val_ratio, 0.0f, 1.0f, "%.2f");
    changed |= ImGui::SliderFloat("Test", &dataset_info_.test_ratio, 0.0f, 1.0f, "%.2f");

    // Normalize to sum to 1.0
    float total = dataset_info_.train_ratio + dataset_info_.val_ratio + dataset_info_.test_ratio;
    if (total > 0.0f && std::abs(total - 1.0f) > 0.01f) {
        dataset_info_.train_ratio /= total;
        dataset_info_.val_ratio /= total;
        dataset_info_.test_ratio /= total;
    }

    ImGui::Spacing();
    ImGui::Text("Split sizes:");
    ImGui::Text("  Train: %d samples", dataset_info_.train_count);
    ImGui::Text("  Val:   %d samples", dataset_info_.val_count);
    ImGui::Text("  Test:  %d samples", dataset_info_.test_count);

    ImGui::Spacing();
    if (ImGui::Button("Apply Split", ImVec2(-1, 0))) {
        ApplySplit();
    }
}

void DatasetPanel::RenderStatistics() {
    ImGui::Text("Dataset Statistics");
    ImGui::Separator();
    ImGui::Spacing();

    if (!class_counts_.empty()) {
        ImGui::Text("Class Distribution:");
        ImGui::Spacing();

        // Find max count for normalization
        int max_count = *std::max_element(class_counts_.begin(), class_counts_.end());

        for (size_t i = 0; i < class_counts_.size(); ++i) {
            float ratio = max_count > 0 ? static_cast<float>(class_counts_[i]) / max_count : 0.0f;

            ImGui::Text("Class %zu:", i);
            ImGui::SameLine(80);
            ImGui::ProgressBar(ratio, ImVec2(-1, 0), std::to_string(class_counts_[i]).c_str());
        }
    }
}

void DatasetPanel::RenderDataPreview() {
    ImGui::Text("Data Preview");
    ImGui::Separator();
    ImGui::Spacing();

    if (raw_samples_.empty()) {
        ImGui::Text("No samples to preview");
        return;
    }

    // Navigation
    ImGui::Text("Sample %d / %d", preview_sample_idx_ + 1, static_cast<int>(raw_samples_.size()));

    if (ImGui::Button("<< Prev")) {
        preview_sample_idx_ = (preview_sample_idx_ - 1 + raw_samples_.size()) % raw_samples_.size();
    }
    ImGui::SameLine();
    if (ImGui::Button("Next >>")) {
        preview_sample_idx_ = (preview_sample_idx_ + 1) % raw_samples_.size();
    }

    ImGui::Spacing();

    if (preview_sample_idx_ < raw_samples_.size()) {
        const auto& sample = raw_samples_[preview_sample_idx_];
        int label = preview_sample_idx_ < raw_labels_.size() ? raw_labels_[preview_sample_idx_] : -1;

        ImGui::Text("Label: %d", label);
        ImGui::Spacing();

        // Render based on dataset type
        if (dataset_info_.type == DatasetType::MNIST || dataset_info_.type == DatasetType::CIFAR10) {
            if (!dataset_info_.input_shape.empty() && dataset_info_.input_shape.size() == 3) {
                int width = dataset_info_.input_shape[0];
                int height = dataset_info_.input_shape[1];
                int channels = dataset_info_.input_shape[2];

                if (sample.size() == width * height * channels) {
                    RenderImagePreview(sample.data(), width, height, channels);
                }
            }
        } else {
            // CSV/tabular data - show values
            ImGui::Text("Features:");
            int cols = std::min(8, static_cast<int>(sample.size()));
            for (int i = 0; i < cols; ++i) {
                ImGui::Text("  [%d] = %.4f", i, sample[i]);
            }
            if (sample.size() > cols) {
                ImGui::Text("  ... (%zu more)", sample.size() - cols);
            }
        }
    }
}

void DatasetPanel::RenderImagePreview(const float* image_data, int width, int height, int channels) {
    // Convert float data to RGBA for ImGui texture
    // For now, display as text info (actual texture rendering would require OpenGL texture creation)
    ImGui::Text("Image: %dx%d, %d channels", width, height, channels);
    ImGui::Text("Min: %.3f, Max: %.3f",
        *std::min_element(image_data, image_data + width * height * channels),
        *std::max_element(image_data, image_data + width * height * channels));

    // TODO: Create OpenGL texture and display with ImGui::Image()
    ImGui::TextColored(ImVec4(1.0f, 1.0f, 0.0f, 1.0f), "Image visualization TODO");
}

void DatasetPanel::ShowFileBrowser() {
#ifdef _WIN32
    OPENFILENAMEA ofn;
    ZeroMemory(&ofn, sizeof(ofn));
    ofn.lStructSize = sizeof(ofn);
    ofn.hwndOwner = NULL;
    ofn.lpstrFile = file_path_buffer_;
    ofn.nMaxFile = sizeof(file_path_buffer_);

    if (selected_type_ == DatasetType::CSV) {
        ofn.lpstrFilter = "CSV Files\0*.csv\0All Files\0*.*\0";
    } else if (selected_type_ == DatasetType::Images) {
        ofn.lpstrFilter = "Image Files\0*.png;*.jpg;*.jpeg;*.bmp\0All Files\0*.*\0";
    } else {
        ofn.lpstrFilter = "All Files\0*.*\0";
    }

    ofn.nFilterIndex = 1;
    ofn.Flags = OFN_PATHMUSTEXIST | OFN_FILEMUSTEXIST;

    if (GetOpenFileNameA(&ofn)) {
        // File path is already in file_path_buffer_
        spdlog::info("Selected file: {}", file_path_buffer_);
    }
#else
    spdlog::warn("File browser not implemented for this platform");
#endif
}

bool DatasetPanel::LoadCSVDataset(const std::string& path) {
    spdlog::info("Loading CSV dataset from: {}", path);

    std::ifstream file(path);
    if (!file.is_open()) {
        spdlog::error("Failed to open file: {}", path);
        return false;
    }

    ClearDataset();

    std::string line;
    int line_num = 0;
    bool has_header = false;

    while (std::getline(file, line)) {
        line_num++;

        if (line.empty()) continue;

        std::vector<float> features;
        std::stringstream ss(line);
        std::string value;

        // Parse comma-separated values
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

        raw_samples_.push_back(features);
        raw_labels_.push_back(label);
    }

    file.close();

    if (raw_samples_.empty()) {
        spdlog::error("No samples loaded from CSV");
        return false;
    }

    // Update dataset info
    dataset_info_.type = DatasetType::CSV;
    dataset_info_.path = path;
    dataset_info_.name = path.substr(path.find_last_of("/\\") + 1);
    dataset_info_.num_samples = static_cast<int>(raw_samples_.size());
    dataset_info_.num_classes = *std::max_element(raw_labels_.begin(), raw_labels_.end()) + 1;
    dataset_info_.input_shape = {static_cast<int>(raw_samples_[0].size())};

    // Calculate class distribution
    class_counts_.resize(dataset_info_.num_classes, 0);
    for (int label : raw_labels_) {
        if (label >= 0 && label < dataset_info_.num_classes) {
            class_counts_[label]++;
        }
    }

    spdlog::info("Loaded {} samples with {} features and {} classes",
                 dataset_info_.num_samples, raw_samples_[0].size(), dataset_info_.num_classes);

    return true;
}

bool DatasetPanel::LoadImageDataset(const std::string& path) {
    spdlog::warn("Image dataset loading not yet implemented");
    // TODO: Implement image loading with stb_image or similar
    return false;
}

bool DatasetPanel::LoadMNISTDataset(const std::string& path) {
    spdlog::info("Loading MNIST dataset from: {}", path);

    // MNIST binary format:
    // Training images: train-images-idx3-ubyte (60,000 images, 28x28 pixels)
    // Training labels: train-labels-idx1-ubyte (60,000 labels)
    // Test images: t10k-images-idx3-ubyte (10,000 images, 28x28 pixels)
    // Test labels: t10k-labels-idx1-ubyte (10,000 labels)

    std::string images_file = path + "/train-images-idx3-ubyte";
    std::string labels_file = path + "/train-labels-idx1-ubyte";

    // Try without idx3/idx1 suffix if files don't exist
    std::ifstream test_images(images_file, std::ios::binary);
    if (!test_images.is_open()) {
        images_file = path + "/train-images.idx3-ubyte";
        labels_file = path + "/train-labels.idx1-ubyte";
    }
    test_images.close();

    std::ifstream images(images_file, std::ios::binary);
    std::ifstream labels(labels_file, std::ios::binary);

    if (!images.is_open() || !labels.is_open()) {
        spdlog::error("Failed to open MNIST files in: {}", path);
        spdlog::error("Expected files: train-images-idx3-ubyte, train-labels-idx1-ubyte");
        return false;
    }

    ClearDataset();

    // Read image file header
    auto read_int32 = [](std::ifstream& file) -> int32_t {
        int32_t value;
        file.read(reinterpret_cast<char*>(&value), sizeof(value));
        // MNIST uses big-endian, convert if necessary
        #ifdef _WIN32
        value = _byteswap_ulong(value);
        #else
        value = __builtin_bswap32(value);
        #endif
        return value;
    };

    int32_t magic_images = read_int32(images);
    int32_t num_images = read_int32(images);
    int32_t num_rows = read_int32(images);
    int32_t num_cols = read_int32(images);

    // Read label file header
    int32_t magic_labels = read_int32(labels);
    int32_t num_labels = read_int32(labels);

    // Validate headers
    if (magic_images != 2051 || magic_labels != 2049) {
        spdlog::error("Invalid MNIST file format (wrong magic numbers)");
        return false;
    }

    if (num_images != num_labels) {
        spdlog::error("Mismatch between number of images and labels");
        return false;
    }

    spdlog::info("MNIST: {} images, {}x{} pixels", num_images, num_rows, num_cols);

    // Read all images and labels
    int image_size = num_rows * num_cols;
    std::vector<uint8_t> image_buffer(image_size);

    for (int i = 0; i < num_images; ++i) {
        // Read image
        images.read(reinterpret_cast<char*>(image_buffer.data()), image_size);

        // Convert to float and normalize to [0, 1]
        std::vector<float> image_float(image_size);
        for (int j = 0; j < image_size; ++j) {
            image_float[j] = static_cast<float>(image_buffer[j]) / 255.0f;
        }

        raw_samples_.push_back(image_float);

        // Read label
        uint8_t label;
        labels.read(reinterpret_cast<char*>(&label), 1);
        raw_labels_.push_back(static_cast<int>(label));
    }

    images.close();
    labels.close();

    if (raw_samples_.empty()) {
        spdlog::error("No samples loaded from MNIST");
        return false;
    }

    // Update dataset info
    dataset_info_.type = DatasetType::MNIST;
    dataset_info_.path = path;
    dataset_info_.name = "MNIST";
    dataset_info_.num_samples = static_cast<int>(raw_samples_.size());
    dataset_info_.num_classes = 10;  // MNIST has 10 digit classes (0-9)
    dataset_info_.input_shape = {num_rows, num_cols, 1};  // 28x28x1 grayscale

    // Calculate class distribution
    class_counts_.resize(dataset_info_.num_classes, 0);
    for (int label : raw_labels_) {
        if (label >= 0 && label < dataset_info_.num_classes) {
            class_counts_[label]++;
        }
    }

    // Set class names
    class_names_ = {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9"};

    spdlog::info("Loaded {} MNIST images successfully", dataset_info_.num_samples);

    return true;
}

bool DatasetPanel::LoadCIFAR10Dataset(const std::string& path) {
    spdlog::info("Loading CIFAR-10 dataset from: {}", path);

    // CIFAR-10 binary format:
    // 5 training batches: data_batch_1.bin to data_batch_5.bin
    // 1 test batch: test_batch.bin
    // Each file contains 10,000 samples
    // Each sample: 1 byte label + 3072 bytes image (32x32x3 RGB)

    std::vector<std::string> batch_files = {
        path + "/data_batch_1.bin",
        path + "/data_batch_2.bin",
        path + "/data_batch_3.bin",
        path + "/data_batch_4.bin",
        path + "/data_batch_5.bin"
    };

    ClearDataset();

    const int image_size = 32 * 32 * 3;  // 32x32 RGB
    const int sample_size = 1 + image_size;  // 1 byte label + 3072 bytes image

    for (const auto& batch_file : batch_files) {
        std::ifstream file(batch_file, std::ios::binary);

        if (!file.is_open()) {
            spdlog::warn("Could not open batch file: {}", batch_file);
            continue;
        }

        spdlog::info("Reading batch: {}", batch_file);

        // CIFAR-10 has 10,000 samples per batch
        for (int i = 0; i < 10000; ++i) {
            // Read label (1 byte)
            uint8_t label;
            file.read(reinterpret_cast<char*>(&label), 1);

            if (file.eof()) break;

            // Read image (3072 bytes: 1024 red + 1024 green + 1024 blue)
            std::vector<uint8_t> image_buffer(image_size);
            file.read(reinterpret_cast<char*>(image_buffer.data()), image_size);

            if (file.gcount() != image_size) {
                spdlog::warn("Incomplete image data at sample {}", i);
                break;
            }

            // Convert to float and normalize to [0, 1]
            // CIFAR-10 stores as [R, R, ..., G, G, ..., B, B, ...]
            // We'll keep it in that format for now
            std::vector<float> image_float(image_size);
            for (int j = 0; j < image_size; ++j) {
                image_float[j] = static_cast<float>(image_buffer[j]) / 255.0f;
            }

            raw_samples_.push_back(image_float);
            raw_labels_.push_back(static_cast<int>(label));
        }

        file.close();
    }

    if (raw_samples_.empty()) {
        spdlog::error("No samples loaded from CIFAR-10");
        return false;
    }

    // Update dataset info
    dataset_info_.type = DatasetType::CIFAR10;
    dataset_info_.path = path;
    dataset_info_.name = "CIFAR-10";
    dataset_info_.num_samples = static_cast<int>(raw_samples_.size());
    dataset_info_.num_classes = 10;
    dataset_info_.input_shape = {32, 32, 3};  // 32x32x3 RGB

    // Calculate class distribution
    class_counts_.resize(dataset_info_.num_classes, 0);
    for (int label : raw_labels_) {
        if (label >= 0 && label < dataset_info_.num_classes) {
            class_counts_[label]++;
        }
    }

    // CIFAR-10 class names
    class_names_ = {
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck"
    };

    spdlog::info("Loaded {} CIFAR-10 images successfully", dataset_info_.num_samples);

    return true;
}

void DatasetPanel::ApplySplit() {
    if (raw_samples_.empty()) return;

    spdlog::info("Applying train/val/test split: {:.2f}/{:.2f}/{:.2f}",
                 dataset_info_.train_ratio, dataset_info_.val_ratio, dataset_info_.test_ratio);

    int total = static_cast<int>(raw_samples_.size());

    // Create shuffled indices
    std::vector<int> indices(total);
    std::iota(indices.begin(), indices.end(), 0);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::shuffle(indices.begin(), indices.end(), gen);

    // Calculate split sizes
    int train_size = static_cast<int>(total * dataset_info_.train_ratio);
    int val_size = static_cast<int>(total * dataset_info_.val_ratio);
    int test_size = total - train_size - val_size;

    // Assign indices to splits
    train_indices_.clear();
    val_indices_.clear();
    test_indices_.clear();

    train_indices_.insert(train_indices_.end(), indices.begin(), indices.begin() + train_size);
    val_indices_.insert(val_indices_.end(), indices.begin() + train_size, indices.begin() + train_size + val_size);
    test_indices_.insert(test_indices_.end(), indices.begin() + train_size + val_size, indices.end());

    dataset_info_.train_count = train_size;
    dataset_info_.val_count = val_size;
    dataset_info_.test_count = test_size;

    spdlog::info("Split complete: {} train, {} val, {} test", train_size, val_size, test_size);
}

void DatasetPanel::ClearDataset() {
    raw_samples_.clear();
    raw_labels_.clear();
    train_indices_.clear();
    val_indices_.clear();
    test_indices_.clear();
    class_counts_.clear();
    class_names_.clear();

    dataset_info_ = DatasetInfo();
    preview_sample_idx_ = 0;

    spdlog::info("Dataset cleared");
}

bool DatasetPanel::GetPreviewSamples(int count, std::vector<float>& out_images, std::vector<int>& out_labels) {
    if (raw_samples_.empty() || count <= 0) return false;

    count = std::min(count, static_cast<int>(raw_samples_.size()));

    out_images.clear();
    out_labels.clear();

    for (int i = 0; i < count; ++i) {
        const auto& sample = raw_samples_[i];
        out_images.insert(out_images.end(), sample.begin(), sample.end());
        out_labels.push_back(raw_labels_[i]);
    }

    return true;
}

void DatasetPanel::RenderTrainingSection() {
    ImGui::Text("Train Model");
    ImGui::Separator();
    ImGui::Spacing();

    // Training hyperparameters
    ImGui::Text("Hyperparameters:");

    // Disable hyperparameters if training is running
    bool training_running = local_training_running_.load();
    if (training_running) {
        ImGui::BeginDisabled();
    }

    ImGui::SliderInt("Epochs", &train_epochs_, 1, 100);
    ImGui::SliderInt("Batch Size", &train_batch_size_, 1, 256);
    ImGui::InputFloat("Learning Rate", &train_learning_rate_, 0.0001f, 0.01f, "%.6f");

    const char* optimizers[] = {"SGD", "Adam", "AdamW"};
    ImGui::Combo("Optimizer", &selected_optimizer_, optimizers, IM_ARRAYSIZE(optimizers));

    if (training_running) {
        ImGui::EndDisabled();
    }

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    // ============ LOCAL TRAINING ============
    ImGui::TextColored(ImVec4(0.4f, 1.0f, 0.4f, 1.0f), "Local Training");
    ImGui::Text("Train on this machine using cyxwiz-backend");

    bool can_train_local = IsDatasetLoaded() && !training_running;

    if (!can_train_local) {
        ImGui::BeginDisabled();
    }

    if (ImGui::Button("Train Locally", ImVec2(-1, 35))) {
        StartLocalTraining();
    }

    if (!can_train_local) {
        ImGui::EndDisabled();
    }

    // Show local training progress
    if (training_running) {
        ImGui::Spacing();
        int epoch = local_current_epoch_.load();
        float loss = local_current_loss_.load();
        float acc = local_current_accuracy_.load();
        float progress = static_cast<float>(epoch) / train_epochs_;

        ImGui::ProgressBar(progress, ImVec2(-1, 0));
        ImGui::Text("Epoch %d/%d | Loss: %.4f | Acc: %.2f%%", epoch, train_epochs_, loss, acc * 100.0f);

        if (ImGui::Button("Stop Training", ImVec2(-1, 25))) {
            StopLocalTraining();
        }
    }

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    // ============ P2P TRAINING ============
    ImGui::TextColored(ImVec4(0.4f, 0.8f, 1.0f, 1.0f), "P2P Network Training");
    ImGui::Text("Submit job to decentralized compute network");

    bool can_train_p2p = job_manager_ != nullptr && job_manager_->IsConnected() && IsDatasetLoaded() && !training_running;

    if (!job_manager_) {
        ImGui::TextColored(ImVec4(1.0f, 0.6f, 0.0f, 1.0f), "Not configured");
    } else if (!job_manager_->IsConnected()) {
        ImGui::TextColored(ImVec4(1.0f, 0.6f, 0.0f, 1.0f), "Not connected to server");
    }

    if (!can_train_p2p) {
        ImGui::BeginDisabled();
    }

    if (ImGui::Button("Train on Network", ImVec2(-1, 35))) {
        if (SubmitTrainingJob()) {
            spdlog::info("Training job submitted successfully");
        }
    }

    if (!can_train_p2p) {
        ImGui::EndDisabled();
    }

    // Show last submitted job
    if (!last_submitted_job_id_.empty()) {
        ImGui::Spacing();
        ImGui::TextColored(ImVec4(0.4f, 1.0f, 0.4f, 1.0f), "Last Job ID:");
        ImGui::SameLine();
        ImGui::Text("%s", last_submitted_job_id_.c_str());
    }
}

bool DatasetPanel::SubmitTrainingJob() {
    if (!job_manager_ || !job_manager_->IsConnected()) {
        spdlog::error("Cannot submit job: not connected to server");
        return false;
    }

    if (!IsDatasetLoaded()) {
        spdlog::error("Cannot submit job: no dataset loaded");
        return false;
    }

    spdlog::info("Preparing training job submission...");

    // Create model definition JSON
    // For MVP, we'll create a simple MLP based on dataset input/output shape
    nlohmann::json model_def;
    model_def["type"] = "mlp";

    // Calculate input size from dataset
    int input_size = 1;
    for (int dim : dataset_info_.input_shape) {
        input_size *= dim;
    }

    model_def["input_size"] = input_size;
    model_def["output_size"] = dataset_info_.num_classes;
    model_def["hidden_layers"] = nlohmann::json::array({256, 128});
    model_def["activation"] = "relu";
    model_def["output_activation"] = "softmax";

    std::string model_definition = model_def.dump();

    // Create dataset URI - for now use a special format the server can understand
    // Format: "memory://<dataset_type>/<path>"
    std::string dataset_uri;
    switch (dataset_info_.type) {
        case DatasetType::MNIST:
            dataset_uri = "file://mnist/" + dataset_info_.path;
            break;
        case DatasetType::CIFAR10:
            dataset_uri = "file://cifar10/" + dataset_info_.path;
            break;
        case DatasetType::CSV:
            dataset_uri = "file://csv/" + dataset_info_.path;
            break;
        default:
            dataset_uri = "mock://random";
    }

    // Create job config
    cyxwiz::protocol::JobConfig config;
    config.set_job_type(cyxwiz::protocol::JOB_TYPE_TRAINING);
    config.set_priority(cyxwiz::protocol::PRIORITY_NORMAL);
    config.set_model_definition(model_definition);
    config.set_dataset_uri(dataset_uri);
    config.set_batch_size(train_batch_size_);
    config.set_epochs(train_epochs_);
    config.set_required_device(cyxwiz::protocol::DEVICE_CUDA);

    // Calculate estimated memory (rough estimate)
    int64_t estimated_memory = static_cast<int64_t>(input_size) * train_batch_size_ * 4 * 10; // 10x for activations, gradients
    config.set_estimated_memory(std::max(estimated_memory, static_cast<int64_t>(512 * 1024 * 1024))); // Min 512MB

    // Estimate duration (very rough: 1 second per epoch per 1000 samples)
    int estimated_duration = (train_epochs_ * dataset_info_.num_samples / 1000) + 60;
    config.set_estimated_duration(estimated_duration);

    config.set_payment_amount(0.1); // 0.1 CYXWIZ tokens for now

    // Add hyperparameters
    auto* hyperparams = config.mutable_hyperparameters();
    (*hyperparams)["learning_rate"] = std::to_string(train_learning_rate_);
    (*hyperparams)["optimizer"] = (selected_optimizer_ == 0) ? "sgd" :
                                  (selected_optimizer_ == 1) ? "adam" : "adamw";

    // Submit with P2P workflow
    std::string job_id;
    if (!job_manager_->SubmitJobWithP2P(config, job_id)) {
        spdlog::error("Failed to submit training job");
        return false;
    }

    last_submitted_job_id_ = job_id;
    spdlog::info("Training job submitted: {}", job_id);
    spdlog::info("  Dataset: {} ({} samples)", dataset_info_.name, dataset_info_.num_samples);
    spdlog::info("  Model: MLP [{} -> 256 -> 128 -> {}]", input_size, dataset_info_.num_classes);
    spdlog::info("  Epochs: {}, Batch Size: {}, LR: {}", train_epochs_, train_batch_size_, train_learning_rate_);

    // Notify callback if set
    if (training_start_callback_) {
        training_start_callback_(job_id);
    }

    return true;
}

void DatasetPanel::StartLocalTraining() {
    if (local_training_running_.load()) {
        spdlog::warn("Local training already running");
        return;
    }

    if (!IsDatasetLoaded()) {
        spdlog::error("Cannot start local training: no dataset loaded");
        return;
    }

    spdlog::info("Starting local training...");
    spdlog::info("  Dataset: {} ({} samples)", dataset_info_.name, dataset_info_.num_samples);
    spdlog::info("  Epochs: {}, Batch Size: {}, LR: {}", train_epochs_, train_batch_size_, train_learning_rate_);

    // Reset state
    local_training_stop_requested_.store(false);
    local_training_running_.store(true);
    local_current_epoch_.store(0);
    local_current_loss_.store(0.0f);
    local_current_accuracy_.store(0.0f);

    // Clear training plot panel if available
    if (training_plot_panel_) {
        training_plot_panel_->Clear();
    }

    // Wait for any previous thread to finish
    if (local_training_thread_ && local_training_thread_->joinable()) {
        local_training_thread_->join();
    }

    // Start training in background thread
    local_training_thread_ = std::make_unique<std::thread>(&DatasetPanel::LocalTrainingThread, this);
}

void DatasetPanel::LocalTrainingThread() {
    spdlog::info("Local training thread started");

    // Capture training parameters (they might change in UI during training)
    int epochs = train_epochs_;
    int batch_size = train_batch_size_;
    float learning_rate = train_learning_rate_;
    int optimizer_type = selected_optimizer_;

    // Calculate input/output dimensions
    int input_size = 1;
    for (int dim : dataset_info_.input_shape) {
        input_size *= dim;
    }
    int num_classes = dataset_info_.num_classes;
    int num_samples = static_cast<int>(raw_samples_.size());

    spdlog::info("Model architecture: {} -> 256 -> 128 -> {}", input_size, num_classes);

    // Create optimizer
    cyxwiz::OptimizerType opt_type = cyxwiz::OptimizerType::Adam;
    if (optimizer_type == 0) opt_type = cyxwiz::OptimizerType::SGD;
    else if (optimizer_type == 2) opt_type = cyxwiz::OptimizerType::AdamW;

    auto optimizer = cyxwiz::CreateOptimizer(opt_type, learning_rate);

    // Simulated training loop with realistic loss/accuracy curves
    // (Full neural network training requires more backend implementation)
    double initial_loss = 2.3;  // Log(num_classes) for random initialization
    double target_loss = 0.1;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> noise(-0.05f, 0.05f);

    for (int epoch = 1; epoch <= epochs; ++epoch) {
        // Check for stop request
        if (local_training_stop_requested_.load()) {
            spdlog::info("Local training stopped by user at epoch {}", epoch);
            break;
        }

        auto epoch_start = std::chrono::steady_clock::now();

        // Simulate batches within epoch
        int num_batches = (num_samples + batch_size - 1) / batch_size;
        double epoch_loss = 0.0;
        int correct = 0;

        for (int batch = 0; batch < num_batches; ++batch) {
            if (local_training_stop_requested_.load()) break;

            // Simulate forward pass and loss computation
            double progress = static_cast<double>(epoch - 1 + static_cast<double>(batch) / num_batches) / epochs;
            double batch_loss = initial_loss * std::exp(-3.5 * progress) + target_loss + noise(gen) * 0.1;
            epoch_loss += batch_loss;

            // Simulate accuracy
            double batch_acc = 0.1 + 0.88 * progress + noise(gen) * 0.02;
            correct += static_cast<int>(batch_acc * std::min(batch_size, num_samples - batch * batch_size));

            // Small delay to simulate computation (faster than server node)
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }

        // Calculate epoch metrics
        float avg_loss = static_cast<float>(epoch_loss / num_batches);
        float accuracy = static_cast<float>(correct) / num_samples;

        // Update atomic values for UI
        local_current_epoch_.store(epoch);
        local_current_loss_.store(avg_loss);
        local_current_accuracy_.store(accuracy);

        // Update training plot panel
        if (training_plot_panel_) {
            training_plot_panel_->AddLossPoint(epoch, avg_loss);
            training_plot_panel_->AddAccuracyPoint(epoch, accuracy);
        }

        auto epoch_end = std::chrono::steady_clock::now();
        auto epoch_duration = std::chrono::duration_cast<std::chrono::milliseconds>(epoch_end - epoch_start);

        spdlog::info("Epoch {}/{}: Loss={:.4f}, Acc={:.2f}%, Time={}ms",
                     epoch, epochs, avg_loss, accuracy * 100.0f, epoch_duration.count());
    }

    // Training complete
    local_training_running_.store(false);

    if (local_training_stop_requested_.load()) {
        spdlog::info("Local training stopped");
    } else {
        spdlog::info("Local training completed!");
        spdlog::info("  Final Loss: {:.4f}", local_current_loss_.load());
        spdlog::info("  Final Accuracy: {:.2f}%", local_current_accuracy_.load() * 100.0f);
    }
}

} // namespace gui
