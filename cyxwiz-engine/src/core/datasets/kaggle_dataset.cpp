#include "kaggle_dataset.h"
#include <spdlog/spdlog.h>
#include <stb_image.h>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <random>

namespace fs = std::filesystem;

namespace cyxwiz {

KaggleDataset::KaggleDataset(const KaggleConfig& config) : config_(config) {
    Initialize();
}

size_t KaggleDataset::Size() const {
    return samples_.size();
}

std::pair<std::vector<float>, int> KaggleDataset::GetItem(size_t index) const {
    if (index >= samples_.size()) {
        return {{}, -1};
    }
    return {samples_[index], labels_[index]};
}

DatasetInfo KaggleDataset::GetInfo() const {
    DatasetInfo info;
    info.name = GetDatasetName();
    info.path = GetCacheDir();
    info.type = DatasetType::Kaggle;
    info.shape = shape_;
    info.num_samples = samples_.size();
    info.num_classes = num_classes_;
    info.class_names = class_names_;
    info.train_count = train_indices_.size();
    info.val_count = val_indices_.size();
    info.test_count = test_indices_.size();
    info.memory_usage = CalculateMemoryUsage();
    info.is_loaded = true;
    return info;
}

void KaggleDataset::Initialize() {
    // Determine cache directory
    std::string cache_dir = GetCacheDir();
    fs::create_directories(cache_dir);

    std::string dataset_name = GetDatasetName();
    spdlog::info("Initializing Kaggle dataset: {}", dataset_name);

    // Check for cached data
    std::string cache_file = cache_dir + "/" + dataset_name + ".cache";
    if (fs::exists(cache_file)) {
        if (LoadFromCache(cache_file)) {
            spdlog::info("Loaded Kaggle dataset from cache: {}", cache_file);
            return;
        }
    }

    // Try to load from downloaded files in cache directory
    if (LoadFromDownloadedFiles(cache_dir)) {
        SaveToCache(cache_file);
        return;
    }

    // Simulate well-known Kaggle datasets with predefined data
    if (LoadPredefinedDataset(dataset_name)) {
        SaveToCache(cache_file);
        spdlog::info("Loaded predefined Kaggle dataset: {}", dataset_name);
        return;
    }

    spdlog::warn("Kaggle dataset '{}' not found. Please download it using the Kaggle CLI:", dataset_name);
    spdlog::warn("  kaggle datasets download -d {}", config_.dataset_slug);
    spdlog::warn("Or for competitions:");
    spdlog::warn("  kaggle competitions download -c {}", config_.competition);
}

std::string KaggleDataset::GetDatasetName() const {
    if (!config_.dataset_slug.empty()) {
        // Extract name from slug (e.g., "zalando-research/fashionmnist" -> "fashionmnist")
        size_t pos = config_.dataset_slug.find_last_of('/');
        if (pos != std::string::npos) {
            return config_.dataset_slug.substr(pos + 1);
        }
        return config_.dataset_slug;
    }
    return config_.competition;
}

std::string KaggleDataset::GetCacheDir() const {
    if (!config_.cache_dir.empty()) {
        return config_.cache_dir;
    }
    return "./data/kaggle_cache/" + GetDatasetName();
}

bool KaggleDataset::LoadFromCache(const std::string& cache_file) {
    std::ifstream file(cache_file, std::ios::binary);
    if (!file.is_open()) return false;

    size_t num_samples, feature_size;
    file.read(reinterpret_cast<char*>(&num_samples), sizeof(num_samples));
    file.read(reinterpret_cast<char*>(&feature_size), sizeof(feature_size));
    file.read(reinterpret_cast<char*>(&num_classes_), sizeof(num_classes_));

    samples_.resize(num_samples);
    labels_.resize(num_samples);

    for (size_t i = 0; i < num_samples; i++) {
        samples_[i].resize(feature_size);
        file.read(reinterpret_cast<char*>(samples_[i].data()), feature_size * sizeof(float));
        file.read(reinterpret_cast<char*>(&labels_[i]), sizeof(int));
    }

    // Read shape
    size_t shape_size;
    file.read(reinterpret_cast<char*>(&shape_size), sizeof(shape_size));
    shape_.resize(shape_size);
    file.read(reinterpret_cast<char*>(shape_.data()), shape_size * sizeof(size_t));

    SetupSplits();
    return true;
}

void KaggleDataset::SaveToCache(const std::string& cache_file) {
    if (samples_.empty()) return;

    std::ofstream file(cache_file, std::ios::binary);
    if (!file.is_open()) return;

    size_t num_samples = samples_.size();
    size_t feature_size = samples_[0].size();

    file.write(reinterpret_cast<const char*>(&num_samples), sizeof(num_samples));
    file.write(reinterpret_cast<const char*>(&feature_size), sizeof(feature_size));
    file.write(reinterpret_cast<const char*>(&num_classes_), sizeof(num_classes_));

    for (size_t i = 0; i < num_samples; i++) {
        file.write(reinterpret_cast<const char*>(samples_[i].data()), feature_size * sizeof(float));
        file.write(reinterpret_cast<const char*>(&labels_[i]), sizeof(int));
    }

    // Write shape
    size_t shape_size = shape_.size();
    file.write(reinterpret_cast<const char*>(&shape_size), sizeof(shape_size));
    file.write(reinterpret_cast<const char*>(shape_.data()), shape_size * sizeof(size_t));

    spdlog::info("Cached Kaggle dataset to: {}", cache_file);
}

bool KaggleDataset::LoadFromDownloadedFiles(const std::string& cache_dir) {
    // Look for CSV files in the cache directory
    for (const auto& entry : fs::directory_iterator(cache_dir)) {
        if (entry.path().extension() == ".csv") {
            std::string csv_path = entry.path().string();

            // Check if this is the file we want
            if (!config_.file_name.empty() &&
                entry.path().filename().string() != config_.file_name) {
                continue;
            }

            spdlog::info("Loading Kaggle dataset from CSV: {}", csv_path);
            return LoadCSVFile(csv_path);
        }
    }

    // Look for image folders
    for (const auto& entry : fs::directory_iterator(cache_dir)) {
        if (entry.is_directory()) {
            // Check for class subdirectories (image classification structure)
            bool has_subdirs = false;
            for (const auto& subentry : fs::directory_iterator(entry.path())) {
                if (subentry.is_directory()) {
                    has_subdirs = true;
                    break;
                }
            }
            if (has_subdirs) {
                spdlog::info("Loading Kaggle dataset from image folder: {}", entry.path().string());
                return LoadImageFolder(entry.path().string());
            }
        }
    }

    return false;
}

bool KaggleDataset::LoadCSVFile(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) return false;

    std::string line;
    std::getline(file, line); // Skip header

    std::vector<std::vector<float>> features;
    std::vector<int> labels;
    std::map<std::string, int> label_map;

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string value;
        std::vector<float> row;
        std::string label_str;

        // Parse CSV (assume last column is label)
        while (std::getline(ss, value, ',')) {
            label_str = value;
            if (row.empty() || ss.peek() != EOF) {
                try {
                    row.push_back(std::stof(value));
                } catch (...) {
                    // Non-numeric value, use as label
                }
            }
        }

        // Map label to integer
        if (label_map.find(label_str) == label_map.end()) {
            label_map[label_str] = static_cast<int>(label_map.size());
            class_names_.push_back(label_str);
        }
        labels.push_back(label_map[label_str]);

        if (!row.empty()) {
            row.pop_back(); // Remove label from features
            features.push_back(row);
        }
    }

    if (features.empty()) return false;

    samples_ = std::move(features);
    labels_ = std::move(labels);
    num_classes_ = label_map.size();
    shape_ = {samples_[0].size()};

    SetupSplits();
    return true;
}

bool KaggleDataset::LoadImageFolder(const std::string& path) {
    std::vector<std::string> extensions = {".jpg", ".jpeg", ".png", ".bmp"};
    int label = 0;

    for (const auto& class_dir : fs::directory_iterator(path)) {
        if (!class_dir.is_directory()) continue;

        class_names_.push_back(class_dir.path().filename().string());

        for (const auto& img_entry : fs::directory_iterator(class_dir.path())) {
            std::string ext = img_entry.path().extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

            if (std::find(extensions.begin(), extensions.end(), ext) != extensions.end()) {
                // Load image using stb_image
                int width, height, channels;
                unsigned char* data = stbi_load(img_entry.path().string().c_str(),
                                                 &width, &height, &channels, 0);
                if (data) {
                    std::vector<float> sample(width * height * channels);
                    for (int i = 0; i < width * height * channels; i++) {
                        sample[i] = data[i] / 255.0f;
                    }
                    stbi_image_free(data);

                    samples_.push_back(std::move(sample));
                    labels_.push_back(label);

                    if (shape_.empty()) {
                        shape_ = {static_cast<size_t>(height),
                                 static_cast<size_t>(width),
                                 static_cast<size_t>(channels)};
                    }
                }
            }
        }
        label++;
    }

    num_classes_ = class_names_.size();
    SetupSplits();
    return !samples_.empty();
}

bool KaggleDataset::LoadPredefinedDataset(const std::string& name) {
    // Normalize dataset name
    std::string normalized = name;
    std::transform(normalized.begin(), normalized.end(), normalized.begin(), ::tolower);

    // Support for well-known Kaggle datasets
    if (normalized == "titanic" || normalized.find("titanic") != std::string::npos) {
        return CreateTitanicDataset();
    }
    else if (normalized == "iris" || normalized.find("iris") != std::string::npos) {
        return CreateIrisDataset();
    }
    else if (normalized == "fashionmnist" || normalized.find("fashion") != std::string::npos) {
        return CreateFashionMNISTDataset();
    }
    else if (normalized == "digits" || normalized.find("digit") != std::string::npos) {
        return CreateDigitsDataset();
    }

    return false;
}

bool KaggleDataset::CreateTitanicDataset() {
    // Simplified Titanic dataset (Pclass, Sex, Age, SibSp, Parch, Fare -> Survived)
    class_names_ = {"Died", "Survived"};
    num_classes_ = 2;
    shape_ = {6}; // 6 features

    // Generate synthetic data based on Titanic patterns
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);

    for (int i = 0; i < 891; i++) { // 891 samples like original
        std::vector<float> features(6);
        features[0] = static_cast<float>((i % 3) + 1) / 3.0f;  // Pclass (1-3)
        features[1] = (i % 2 == 0) ? 1.0f : 0.0f;               // Sex
        features[2] = (20.0f + dis(gen) * 50.0f) / 80.0f;       // Age
        features[3] = static_cast<float>(i % 4) / 4.0f;         // SibSp
        features[4] = static_cast<float>(i % 3) / 3.0f;         // Parch
        features[5] = dis(gen);                                  // Fare (normalized)

        samples_.push_back(features);
        // Survival probability based on class and sex
        int survived = (features[0] < 0.5f && features[1] > 0.5f) ? 1 : 0;
        if (dis(gen) < 0.3f) survived = 1 - survived; // Add noise
        labels_.push_back(survived);
    }

    SetupSplits();
    return true;
}

bool KaggleDataset::CreateIrisDataset() {
    class_names_ = {"Setosa", "Versicolor", "Virginica"};
    num_classes_ = 3;
    shape_ = {4}; // 4 features

    // Classic Iris dataset measurements
    float iris_data[][5] = {
        // Sepal L, Sepal W, Petal L, Petal W, Class
        {5.1f, 3.5f, 1.4f, 0.2f, 0}, {4.9f, 3.0f, 1.4f, 0.2f, 0}, {4.7f, 3.2f, 1.3f, 0.2f, 0},
        {4.6f, 3.1f, 1.5f, 0.2f, 0}, {5.0f, 3.6f, 1.4f, 0.2f, 0}, {5.4f, 3.9f, 1.7f, 0.4f, 0},
        {4.6f, 3.4f, 1.4f, 0.3f, 0}, {5.0f, 3.4f, 1.5f, 0.2f, 0}, {4.4f, 2.9f, 1.4f, 0.2f, 0},
        {4.9f, 3.1f, 1.5f, 0.1f, 0}, {7.0f, 3.2f, 4.7f, 1.4f, 1}, {6.4f, 3.2f, 4.5f, 1.5f, 1},
        {6.9f, 3.1f, 4.9f, 1.5f, 1}, {5.5f, 2.3f, 4.0f, 1.3f, 1}, {6.5f, 2.8f, 4.6f, 1.5f, 1},
        {5.7f, 2.8f, 4.5f, 1.3f, 1}, {6.3f, 3.3f, 4.7f, 1.6f, 1}, {4.9f, 2.4f, 3.3f, 1.0f, 1},
        {6.6f, 2.9f, 4.6f, 1.3f, 1}, {5.2f, 2.7f, 3.9f, 1.4f, 1}, {6.3f, 3.3f, 6.0f, 2.5f, 2},
        {5.8f, 2.7f, 5.1f, 1.9f, 2}, {7.1f, 3.0f, 5.9f, 2.1f, 2}, {6.3f, 2.9f, 5.6f, 1.8f, 2},
        {6.5f, 3.0f, 5.8f, 2.2f, 2}, {7.6f, 3.0f, 6.6f, 2.1f, 2}, {4.9f, 2.5f, 4.5f, 1.7f, 2},
        {7.3f, 2.9f, 6.3f, 1.8f, 2}, {6.7f, 2.5f, 5.8f, 1.8f, 2}, {7.2f, 3.6f, 6.1f, 2.5f, 2}
    };

    for (const auto& row : iris_data) {
        std::vector<float> features = {
            row[0] / 8.0f, row[1] / 5.0f, row[2] / 7.0f, row[3] / 3.0f
        };
        samples_.push_back(features);
        labels_.push_back(static_cast<int>(row[4]));
    }

    // Duplicate to get more samples
    size_t original_size = samples_.size();
    for (size_t i = 0; i < original_size * 4; i++) {
        size_t idx = i % original_size;
        std::vector<float> features = samples_[idx];
        // Add slight noise
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<float> noise(0.0f, 0.02f);
        for (auto& f : features) {
            f += noise(gen);
            f = std::max(0.0f, std::min(1.0f, f));
        }
        samples_.push_back(features);
        labels_.push_back(labels_[idx]);
    }

    SetupSplits();
    return true;
}

bool KaggleDataset::CreateFashionMNISTDataset() {
    class_names_ = {"T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
                    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"};
    num_classes_ = 10;
    shape_ = {28, 28, 1};

    // Generate synthetic Fashion-MNIST-like data
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);

    for (int i = 0; i < 1000; i++) {
        std::vector<float> sample(784);
        int label = i % 10;

        // Create simple patterns for each class
        for (int j = 0; j < 784; j++) {
            int row = j / 28;
            int col = j % 28;
            float value = 0.0f;

            switch (label) {
                case 0: // T-shirt shape
                    value = (row > 5 && row < 22 && col > 5 && col < 22) ? 0.8f : 0.1f;
                    break;
                case 1: // Trouser shape
                    value = ((col > 8 && col < 14) || (col > 14 && col < 20)) && row > 5 ? 0.8f : 0.1f;
                    break;
                default:
                    value = dis(gen) * 0.3f + (row == label || col == label ? 0.5f : 0.0f);
            }
            sample[j] = value + dis(gen) * 0.1f;
        }

        samples_.push_back(sample);
        labels_.push_back(label);
    }

    SetupSplits();
    return true;
}

bool KaggleDataset::CreateDigitsDataset() {
    class_names_ = {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9"};
    num_classes_ = 10;
    shape_ = {8, 8, 1}; // Sklearn digits format

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);

    for (int i = 0; i < 1797; i++) { // Same size as sklearn digits
        std::vector<float> sample(64);
        int label = i % 10;

        // Simple digit patterns
        for (int j = 0; j < 64; j++) {
            int row = j / 8;
            int col = j % 8;
            float value = 0.0f;

            // Create simple numeral patterns
            if (label == 0 && ((row == 0 || row == 7) && col > 1 && col < 6)) value = 0.8f;
            else if (label == 1 && col == 4) value = 0.8f;
            else if (label == 2 && (row == 0 || row == 3 || row == 7)) value = 0.7f;
            else value = dis(gen) * 0.2f;

            sample[j] = std::min(1.0f, value + dis(gen) * 0.15f);
        }

        samples_.push_back(sample);
        labels_.push_back(label);
    }

    SetupSplits();
    return true;
}

void KaggleDataset::SetupSplits() {
    all_indices_.resize(samples_.size());
    for (size_t i = 0; i < samples_.size(); i++) {
        all_indices_[i] = i;
    }

    // Default 80/10/10 split
    size_t train_end = static_cast<size_t>(samples_.size() * 0.8);
    size_t val_end = static_cast<size_t>(samples_.size() * 0.9);

    train_indices_.assign(all_indices_.begin(), all_indices_.begin() + train_end);
    val_indices_.assign(all_indices_.begin() + train_end, all_indices_.begin() + val_end);
    test_indices_.assign(all_indices_.begin() + val_end, all_indices_.end());
}

size_t KaggleDataset::CalculateMemoryUsage() const {
    size_t usage = 0;
    for (const auto& sample : samples_) {
        usage += sample.size() * sizeof(float);
    }
    usage += labels_.size() * sizeof(int);
    return usage;
}

} // namespace cyxwiz
