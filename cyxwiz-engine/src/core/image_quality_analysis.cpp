#include "image_quality_analysis.h"

#include "datasets/image_csv_dataset.h"
#include "datasets/image_folder_dataset.h"
#include "graph_compiler_dataset_hooks.h"

#include <cyxwiz/device.h>
#include <cyxwiz/neural_provider.h>

#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <mutex>
#include <sstream>
#include <thread>

namespace cyxwiz {

namespace {

namespace fs = std::filesystem;
using json = nlohmann::json;

constexpr size_t kAnalysisBatch = 256;

uint64_t Fnv(uint64_t hash, const std::string& text) {
    for (unsigned char c : text) {
        hash ^= c;
        hash *= 1099511628211ull;
    }
    return hash;
}

std::string Hex(uint64_t value) {
    char text[17];
    std::snprintf(text, sizeof(text), "%016llx", static_cast<unsigned long long>(value));
    return text;
}

// Modification times of the folder and its direct subfolders: they change
// when files are added, removed or renamed.
std::string DirectorySignature(const std::string& folder) {
    std::error_code ec;
    std::ostringstream out;
    out << fs::last_write_time(folder, ec).time_since_epoch().count();
    for (const auto& entry : fs::directory_iterator(folder, ec)) {
        if (entry.is_directory(ec)) {
            out << '|' << entry.path().filename().string() << ':'
                << fs::last_write_time(entry.path(), ec).time_since_epoch().count();
        }
    }
    return out.str();
}

std::string ComputeKey(const DataRegistry::ImageDatasetEntry& entry, int width, int height) {
    const auto dataset = OpenImageQualityDataset(entry, width, height);
    std::vector<std::string> rows;
    rows.reserve(dataset->Size());
    std::error_code ec;
    for (size_t i = 0; i < dataset->Size(); ++i) {
        const std::string file = dataset->GetItemSource(i);
        const auto size = fs::file_size(file, ec);
        const auto time = fs::last_write_time(file, ec).time_since_epoch().count();
        rows.push_back(file + '|' + std::to_string(ec ? 0 : size) + '|' + std::to_string(time));
    }
    std::sort(rows.begin(), rows.end());
    uint64_t hash = 1469598103934665603ull;
    hash = Fnv(hash, "v" + std::to_string(image::kImageQualityVersion) + "|" + std::to_string(width) + "x" +
                         std::to_string(height) + "|" + entry.labels_csv);
    for (const auto& row : rows) hash = Fnv(hash, row + '\n');
    return Hex(hash);
}

float ParseFloat(const std::map<std::string, std::string>& parameters, const char* name, float fallback) {
    const auto it = parameters.find(name);
    if (it == parameters.end()) return fallback;
    try {
        return std::stof(it->second);
    } catch (...) {
        return fallback;
    }
}

bool ParseBool(const std::map<std::string, std::string>& parameters, const char* name, bool fallback) {
    const auto it = parameters.find(name);
    if (it == parameters.end()) return fallback;
    return it->second == "true" || it->second == "1";
}

std::string FloatText(float value) {
    std::ostringstream out;
    out << value;
    return out.str();
}

}  // namespace

std::string ImageQualityDeviceLabel() {
    std::string platform = NeuralDevicePlatformName(CaptureCurrentNeuralDeviceTarget().platform);
    std::transform(platform.begin(), platform.end(), platform.begin(),
                   [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
    if (platform == "OPENCL") platform = "OpenCL";
    if (platform == "ONEAPI") platform = "oneAPI";
    if (const Device* device = Device::GetCurrentDevice()) {
        const std::string name = device->GetInfo().name;
        if (!name.empty()) return platform + " - " + name;
    }
    return platform;
}

std::shared_ptr<Dataset> OpenImageQualityDataset(const DataRegistry::ImageDatasetEntry& entry, int width, int height) {
    if (entry.layout == 1 && !entry.labels_csv.empty()) {
        return std::make_shared<ImageCSVDataset>(entry.folder_path, entry.labels_csv, width, height, 1);
    }
    return std::make_shared<ImageFolderDataset>(entry.folder_path, width, height, 1);
}

std::string ImageQualityKey(const DataRegistry::ImageDatasetEntry& entry, int width, int height) {
    struct Remembered {
        std::string signature;
        std::string key;
    };
    static std::mutex mutex;
    static std::map<std::string, Remembered> remembered;

    const std::string id = entry.folder_path + '|' + entry.labels_csv + '|' + std::to_string(width) + 'x' +
                           std::to_string(height);
    std::string signature = DirectorySignature(entry.folder_path);
    if (!entry.labels_csv.empty()) {
        std::error_code ec;
        signature += '|' + std::to_string(fs::last_write_time(entry.labels_csv, ec).time_since_epoch().count());
    }
    {
        std::lock_guard<std::mutex> lock(mutex);
        const auto it = remembered.find(id);
        if (it != remembered.end() && it->second.signature == signature) return it->second.key;
    }
    std::string key = ComputeKey(entry, width, height);
    std::lock_guard<std::mutex> lock(mutex);
    remembered[id] = {std::move(signature), key};
    return key;
}

namespace {

std::mutex project_root_mutex;
std::string project_root;

}  // namespace

void SetImageQualityProjectRoot(const std::string& root) {
    std::lock_guard<std::mutex> lock(project_root_mutex);
    project_root = root;
}

fs::path ImageQualityCacheDir() {
    std::lock_guard<std::mutex> lock(project_root_mutex);
    if (!project_root.empty()) return fs::path(project_root) / "cache" / "quality";
    return fs::temp_directory_path() / "cyxwiz" / "quality";
}

std::optional<ImageQualityAnalysis> AnalyzeImageQuality(const DataRegistry::ImageDatasetEntry& entry,
                                                        int width, int height,
                                                        const std::function<void(size_t, size_t)>& progress,
                                                        const std::atomic<bool>* cancel) {
    const image::ImageShape shape{static_cast<size_t>(height), static_cast<size_t>(width), 3};
    if (const std::string why = image::ValidateImageQualityShape(shape); !why.empty()) {
        throw std::invalid_argument(why);
    }
    const auto started = std::chrono::steady_clock::now();
    const auto dataset = OpenImageQualityDataset(entry, width, height);
    const size_t total = dataset->Size();
    if (total == 0) throw std::runtime_error("the dataset has no images");

    ImageQualityAnalysis analysis;
    analysis.key = ImageQualityKey(entry, width, height);
    analysis.folder_path = entry.folder_path;
    analysis.width = width;
    analysis.height = height;
    analysis.class_names = dataset->GetInfo().class_names;
    analysis.device = ImageQualityDeviceLabel();
    analysis.files.resize(total);
    analysis.labels.resize(total);
    analysis.metrics.reserve(total);

    const size_t sample = shape.Size();
    const size_t workers = std::clamp<size_t>(std::thread::hardware_concurrency(), 2, 9) - 1;
    std::vector<float> rows;
    for (size_t begin = 0; begin < total; begin += kAnalysisBatch) {
        if (cancel && cancel->load()) return std::nullopt;
        const size_t count = (std::min)(kAnalysisBatch, total - begin);
        // An image that does not decode stays black and measures as too dark.
        rows.assign(count * sample, 0.0f);
        auto decode = [&](size_t from, size_t to) {
            for (size_t i = from; i < to; ++i) {
                auto [pixels, label] = dataset->GetItem(begin + i);
                if (pixels.size() == sample) std::copy(pixels.begin(), pixels.end(), rows.begin() + i * sample);
                analysis.files[begin + i] = dataset->GetItemSource(begin + i);
                analysis.labels[begin + i] = label;
            }
        };
        std::vector<std::thread> threads;
        const size_t chunk = (count + workers - 1) / workers;
        for (size_t from = 0; from < count; from += chunk) {
            threads.emplace_back(decode, from, (std::min)(count, from + chunk));
        }
        for (auto& thread : threads) thread.join();

        const Tensor batch({count, sample}, rows.data(), DataType::Float32);
        const auto measured = image::MeasureImageQuality(batch, shape);
        analysis.metrics.insert(analysis.metrics.end(), measured.begin(), measured.end());
        if (progress) progress(begin + count, total);
    }
    analysis.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    return analysis;
}

bool SaveImageQualityAnalysis(const ImageQualityAnalysis& analysis, std::string* error) {
    json doc;
    doc["version"] = image::kImageQualityVersion;
    doc["key"] = analysis.key;
    doc["folder_path"] = analysis.folder_path;
    doc["width"] = analysis.width;
    doc["height"] = analysis.height;
    doc["class_names"] = analysis.class_names;
    doc["files"] = analysis.files;
    doc["labels"] = analysis.labels;
    doc["device"] = analysis.device;
    doc["seconds"] = analysis.seconds;
    json blur = json::array(), brightness = json::array(), contrast = json::array(), hash = json::array();
    for (const auto& m : analysis.metrics) {
        blur.push_back(m.blur);
        brightness.push_back(m.brightness);
        contrast.push_back(m.contrast);
        hash.push_back(Hex(m.hash));
    }
    doc["blur"] = std::move(blur);
    doc["brightness"] = std::move(brightness);
    doc["contrast"] = std::move(contrast);
    doc["hash"] = std::move(hash);

    std::error_code ec;
    const fs::path dir = ImageQualityCacheDir();
    fs::create_directories(dir, ec);
    const fs::path path = dir / (analysis.key + ".json");
    const fs::path partial = dir / (analysis.key + ".json.partial");
    {
        std::ofstream out(partial, std::ios::binary | std::ios::trunc);
        out << doc.dump();
        if (!out) {
            if (error) *error = "could not write " + partial.string();
            return false;
        }
    }
    fs::rename(partial, path, ec);
    if (ec) {
        if (error) *error = "could not save " + path.string() + ": " + ec.message();
        return false;
    }
    return true;
}

std::shared_ptr<const ImageQualityAnalysis> LoadImageQualityAnalysis(const std::string& key) {
    static std::mutex mutex;
    static std::map<std::string, std::pair<fs::file_time_type, std::shared_ptr<const ImageQualityAnalysis>>> loaded;

    const fs::path path = ImageQualityCacheDir() / (key + ".json");
    std::error_code ec;
    const auto stamp = fs::last_write_time(path, ec);
    if (ec) return nullptr;
    {
        std::lock_guard<std::mutex> lock(mutex);
        const auto it = loaded.find(path.string());
        if (it != loaded.end() && it->second.first == stamp) return it->second.second;
    }
    try {
        std::ifstream in(path, std::ios::binary);
        const json doc = json::parse(in);
        if (doc.at("version").get<int>() != image::kImageQualityVersion || doc.at("key").get<std::string>() != key) {
            return nullptr;
        }
        auto analysis = std::make_shared<ImageQualityAnalysis>();
        analysis->key = key;
        analysis->folder_path = doc.at("folder_path").get<std::string>();
        analysis->width = doc.at("width").get<int>();
        analysis->height = doc.at("height").get<int>();
        analysis->class_names = doc.at("class_names").get<std::vector<std::string>>();
        analysis->files = doc.at("files").get<std::vector<std::string>>();
        analysis->labels = doc.at("labels").get<std::vector<int>>();
        analysis->device = doc.value("device", std::string());
        analysis->seconds = doc.value("seconds", 0.0);
        const auto blur = doc.at("blur").get<std::vector<float>>();
        const auto brightness = doc.at("brightness").get<std::vector<float>>();
        const auto contrast = doc.at("contrast").get<std::vector<float>>();
        const auto hash = doc.at("hash").get<std::vector<std::string>>();
        const size_t n = analysis->files.size();
        if (analysis->labels.size() != n || blur.size() != n || brightness.size() != n || contrast.size() != n ||
            hash.size() != n) {
            return nullptr;
        }
        analysis->metrics.resize(n);
        for (size_t i = 0; i < n; ++i) {
            analysis->metrics[i] = {blur[i], brightness[i], contrast[i], std::stoull(hash[i], nullptr, 16)};
        }
        std::lock_guard<std::mutex> lock(mutex);
        loaded[path.string()] = {stamp, analysis};
        return analysis;
    } catch (const std::exception& e) {
        spdlog::warn("Quality Analyzer: ignoring unreadable analysis {}: {}", path.string(), e.what());
        return nullptr;
    }
}

image::ImageQualityChecks ImageQualityChecksFromParameters(const std::map<std::string, std::string>& parameters) {
    image::ImageQualityChecks checks;
    checks.blur = ParseBool(parameters, "blur_check", checks.blur);
    checks.blur_min = ParseFloat(parameters, "blur_min", checks.blur_min);
    checks.brightness = ParseBool(parameters, "brightness_check", checks.brightness);
    checks.brightness_min = ParseFloat(parameters, "brightness_min", checks.brightness_min);
    checks.brightness_max = ParseFloat(parameters, "brightness_max", checks.brightness_max);
    checks.contrast = ParseBool(parameters, "contrast_check", checks.contrast);
    checks.contrast_min = ParseFloat(parameters, "contrast_min", checks.contrast_min);
    checks.duplicates = ParseBool(parameters, "duplicate_check", checks.duplicates);
    return checks;
}

void WriteImageQualityChecks(const image::ImageQualityChecks& checks, std::map<std::string, std::string>& parameters) {
    parameters["blur_check"] = checks.blur ? "true" : "false";
    parameters["blur_min"] = FloatText(checks.blur_min);
    parameters["brightness_check"] = checks.brightness ? "true" : "false";
    parameters["brightness_min"] = FloatText(checks.brightness_min);
    parameters["brightness_max"] = FloatText(checks.brightness_max);
    parameters["contrast_check"] = checks.contrast ? "true" : "false";
    parameters["contrast_min"] = FloatText(checks.contrast_min);
    parameters["duplicate_check"] = checks.duplicates ? "true" : "false";
}

ImageQualityVerdict JudgeImageQualityAnalysis(const ImageQualityAnalysis& analysis,
                                              const image::ImageQualityChecks& checks) {
    // The duplicate search depends only on the hashes and the distance:
    // remember the last one, so editing a threshold does not repeat it.
    static std::mutex mutex;
    static std::string remembered_id;
    static std::vector<int64_t> remembered;

    ImageQualityVerdict verdict;
    if (checks.duplicates) {
        const std::string id = analysis.key + '|' + std::to_string(checks.duplicate_bits);
        std::lock_guard<std::mutex> lock(mutex);
        if (remembered_id != id) {
            std::vector<uint64_t> hashes(analysis.metrics.size());
            std::transform(analysis.metrics.begin(), analysis.metrics.end(), hashes.begin(),
                           [](const image::ImageQualityMetrics& m) { return m.hash; });
            remembered = image::FindNearDuplicates(hashes, checks.duplicate_bits);
            remembered_id = id;
        }
        verdict.duplicate_of = remembered;
    } else {
        verdict.duplicate_of.assign(analysis.metrics.size(), -1);
    }
    verdict.reasons = image::JudgeImageQuality(analysis.metrics, verdict.duplicate_of, checks);

    verdict.class_total.assign(analysis.class_names.size(), 0);
    verdict.class_rejected.assign(analysis.class_names.size(), 0);
    for (size_t i = 0; i < verdict.reasons.size(); ++i) {
        const uint32_t r = verdict.reasons[i];
        const int label = i < analysis.labels.size() ? analysis.labels[i] : -1;
        const bool known = label >= 0 && static_cast<size_t>(label) < verdict.class_total.size();
        if (known) ++verdict.class_total[static_cast<size_t>(label)];
        if (r == 0) continue;
        ++verdict.rejected;
        if (known) ++verdict.class_rejected[static_cast<size_t>(label)];
        if (r & image::kQualityBlurry) ++verdict.blurry;
        if (r & image::kQualityDark) ++verdict.dark;
        if (r & image::kQualityBright) ++verdict.bright;
        if (r & image::kQualityLowContrast) ++verdict.low_contrast;
        if (r & image::kQualityDuplicate) ++verdict.duplicates;
        if ((r & (r - 1)) != 0) ++verdict.multiple;
    }
    return verdict;
}

GraphImageQualityResult ResolveImageQuality(const std::string& dataset_name, int width, int height,
                                            const std::map<std::string, std::string>& parameters) {
    GraphImageQualityResult result;
    const auto* entry = DataRegistry::Instance().GetImageDatasetEntry(dataset_name);
    if (!entry) {
        result.error = "the image dataset is not loaded";
        return result;
    }
    const image::ImageShape shape{static_cast<size_t>(height), static_cast<size_t>(width), 3};
    if (std::string why = image::ValidateImageQualityShape(shape); !why.empty()) {
        result.error = std::move(why);
        return result;
    }
    const auto checks = ImageQualityChecksFromParameters(parameters);
    if (std::string why = image::ValidateImageQualityChecks(checks); !why.empty()) {
        result.error = std::move(why);
        return result;
    }
    try {
        const std::string key = ImageQualityKey(*entry, width, height);
        const auto analysis = LoadImageQualityAnalysis(key);
        if (!analysis) {
            const auto previous = parameters.find("analysis_key");
            result.error = previous != parameters.end() && !previous->second.empty()
                ? "the images or the Resize size changed since the analysis: open Quality Analyzer and click Analyze"
                : "the images are not analyzed yet: open Quality Analyzer and click Analyze";
            return result;
        }
        const auto verdict = JudgeImageQualityAnalysis(*analysis, checks);
        result.total = analysis->files.size();
        result.rejected = verdict.rejected;
        for (size_t i = 0; i < verdict.reasons.size(); ++i) {
            if (verdict.reasons[i] != 0) result.excluded.push_back(analysis->files[i]);
        }
        if (result.rejected == result.total) {
            result.error = "the checks leave out all " + std::to_string(result.total) + " images";
        }
    } catch (const std::exception& e) {
        result.error = e.what();
    }
    return result;
}

namespace {

// Hosts that compile this file can judge image quality at compile time; the
// other catalog lookups keep what data_registry_utils.cpp installed.
const bool kImageQualityHookInstalled = [] {
    GraphDatasetCatalog catalog = GetGraphDatasetCatalog();
    catalog.image_quality = ResolveImageQuality;
    SetGraphDatasetCatalog(std::move(catalog));
    return true;
}();

}  // namespace

}  // namespace cyxwiz
