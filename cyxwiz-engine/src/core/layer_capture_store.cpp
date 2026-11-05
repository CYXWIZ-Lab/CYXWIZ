#include "layer_capture_store.h"

#include "algorithms/arrayfire_backend_utils.h"
#include "debug_run_paths.h"
#include "tensor_file.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstdio>
#include <exception>
#include <fstream>
#include <optional>
#include <system_error>
#include <utility>

namespace cyxwiz {

namespace fs = std::filesystem;

namespace {

constexpr const char* kSchema = "cyxwiz.debug.layer_capture.v1";
constexpr const char* kIndexFile = "capture.json";

// Run ids and labels name folders: letters, digits, '.', '_' and '-' only.
bool SafeFolderName(const std::string& name) {
    if (name.empty() || name == "." || name == "..") return false;
    return std::all_of(name.begin(), name.end(), [](char c) {
        return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '.' ||
               c == '_' || c == '-';
    });
}

template <typename T>
T Fail(std::string* error, std::string message, T result) {
    if (error) *error = std::move(message);
    return result;
}

std::string FileName(size_t module_index, CapturedTensorRole role, const std::string& name) {
    std::string file = std::to_string(module_index) + "_" + CapturedTensorRoleName(role) + "_";
    for (char c : name)
        file += ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9')) ? c : '_';
    return file + ".bin";
}

std::optional<CapturedTensorRole> RoleFromName(const std::string& name) {
    for (auto role : {CapturedTensorRole::Input, CapturedTensorRole::Output, CapturedTensorRole::Parameter,
                      CapturedTensorRole::State, CapturedTensorRole::Gradient})
        if (name == CapturedTensorRoleName(role)) return role;
    return std::nullopt;
}

}  // namespace

std::string EpochCaptureLabel(int epoch) {
    char label[32];
    std::snprintf(label, sizeof(label), "epoch_%04d", epoch);
    return label;
}

fs::path LayerCaptureDir(const std::string& run_id, const std::string& label) {
    return GetDebugRunStoreRoot() / run_id / "layers" / label;
}

bool SaveLayerCapture(const LayerCapture& capture, const std::string& run_id, const std::string& label,
                      std::string* error) {
    if (!SafeFolderName(run_id) || !SafeFolderName(label))
        return Fail(error, "the run id or capture label is not a folder name", false);
    try {
        const fs::path dir = LayerCaptureDir(run_id, label);
        std::error_code ec;
        fs::remove_all(dir, ec);
        fs::create_directories(dir);
        nlohmann::json layers = nlohmann::json::array();
        for (const auto& layer : capture.Layers()) {
            nlohmann::json tensors = nlohmann::json::array();
            for (const auto& t : layer.tensors) {
                const std::string file = FileName(layer.module_index, t.role, t.name);
                std::string written;
                if (!WriteTensorFile(dir / file, t.tensor, ArrayFireHostSyncCategory::DebugSampleDump,
                                     "SaveLayerCapture", &written))
                    return Fail(error, written, false);
                tensors.push_back({{"role", CapturedTensorRoleName(t.role)}, {"name", t.name}, {"file", file}});
            }
            layers.push_back({{"module_index", layer.module_index},
                              {"node_id", layer.node_id},
                              {"node_name", layer.node_name},
                              {"tensors", std::move(tensors)}});
        }
        const auto& options = capture.Options();
        const nlohmann::json index = {
            {"schema", kSchema},
            {"options",
             {{"activations", options.activations},
              {"parameters", options.parameters},
              {"gradients", options.gradients},
              {"byte_budget", options.byte_budget}}},
            {"skipped_tensors", capture.SkippedTensors()},
            {"skipped_bytes", capture.SkippedBytes()},
            {"layers", std::move(layers)}};
        std::ofstream file(dir / kIndexFile, std::ios::trunc);
        file << index.dump(1) << '\n';
        if (!file.good()) return Fail(error, "writing " + (dir / kIndexFile).string() + " failed", false);
        return true;
    } catch (const std::exception& e) {
        return Fail(error, std::string("saving the layer capture failed: ") + e.what(), false);
    }
}

std::shared_ptr<LayerCapture> LoadLayerCapture(const std::string& run_id, const std::string& label,
                                               std::string* error) {
    const std::shared_ptr<LayerCapture> none;
    if (!SafeFolderName(run_id) || !SafeFolderName(label))
        return Fail(error, "the run id or capture label is not a folder name", none);
    try {
        const fs::path dir = LayerCaptureDir(run_id, label);
        std::ifstream file(dir / kIndexFile);
        if (!file.is_open()) return Fail(error, "no layer capture is saved at " + dir.string(), none);
        nlohmann::json index;
        file >> index;
        if (index.value("schema", std::string{}) != kSchema)
            return Fail(error, "the layer capture at " + dir.string() + " has an unknown schema", none);
        LayerCaptureOptions options;
        const auto& o = index.at("options");
        options.activations = o.value("activations", options.activations);
        options.parameters = o.value("parameters", options.parameters);
        options.gradients = o.value("gradients", options.gradients);
        options.byte_budget = o.value("byte_budget", options.byte_budget);
        std::vector<CapturedLayer> layers;
        for (const auto& l : index.at("layers")) {
            CapturedLayer layer;
            layer.module_index = l.at("module_index").get<size_t>();
            layer.node_id = l.value("node_id", -1);
            layer.node_name = l.value("node_name", std::string{});
            for (const auto& t : l.at("tensors")) {
                const auto role = RoleFromName(t.at("role").get<std::string>());
                const std::string name = t.at("name").get<std::string>();
                const std::string file_name = t.at("file").get<std::string>();
                if (!role || !SafeFolderName(file_name))
                    return Fail(error, "the layer capture at " + dir.string() + " lists an invalid tensor", none);
                std::string read;
                auto tensor = ReadTensorFile(dir / file_name, &read);
                if (!tensor) return Fail(error, read, none);
                layer.tensors.push_back({*role, name, std::move(*tensor)});
            }
            layers.push_back(std::move(layer));
        }
        return std::make_shared<LayerCapture>(options, std::move(layers), index.value("skipped_tensors", size_t{0}),
                                              index.value("skipped_bytes", size_t{0}));
    } catch (const std::exception& e) {
        return Fail(error, std::string("reading the layer capture failed: ") + e.what(), none);
    }
}

std::vector<std::string> ListLayerCaptures(const std::string& run_id) {
    std::vector<std::string> labels;
    if (!SafeFolderName(run_id)) return labels;
    std::error_code ec;
    for (const auto& entry : fs::directory_iterator(GetDebugRunStoreRoot() / run_id / "layers", ec))
        if (entry.is_directory(ec) && fs::exists(entry.path() / kIndexFile, ec))
            labels.push_back(entry.path().filename().string());
    std::sort(labels.begin(), labels.end());
    return labels;
}

size_t PruneLayerCaptures(size_t keep_runs) {
    std::vector<std::pair<fs::file_time_type, fs::path>> runs;
    std::error_code ec;
    for (const auto& entry : fs::directory_iterator(GetDebugRunStoreRoot(), ec)) {
        const fs::path layers = entry.path() / "layers";
        if (!entry.is_directory(ec) || !fs::is_directory(layers, ec)) continue;
        const auto stamp = fs::last_write_time(layers, ec);
        if (!ec) runs.emplace_back(stamp, layers);
    }
    if (runs.size() <= keep_runs) return 0;
    std::sort(runs.begin(), runs.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
    size_t cleared = 0;
    for (size_t i = keep_runs; i < runs.size(); ++i) {
        fs::remove_all(runs[i].second, ec);
        if (!ec) ++cleared;
    }
    return cleared;
}

}  // namespace cyxwiz
