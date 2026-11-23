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
constexpr const char* kTrainingSchema = "cyxwiz.debug.training_run.v1";
constexpr const char* kTrainingFile = "training.json";

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

// A capture whose first layer took an image batch first [N, H, W, C] and
// returned rows [N, H x W x C] (a Flatten-first model, saved before the
// image layout was recorded).
static bool LooksBatchFirst(const std::vector<CapturedLayer>& layers) {
    const CapturedLayer* first = nullptr;
    for (const auto& layer : layers)
        if (layer.module_index == 0) first = &layer;
    if (!first) return false;
    const CapturedTensor* in = first->Find(CapturedTensorRole::Input);
    const CapturedTensor* out = first->Find(CapturedTensorRole::Output);
    if (!in || !out) return false;
    const auto& i = in->tensor.Shape();
    const auto& o = out->tensor.Shape();
    return i.size() == 4 && o.size() == 2 && o[0] == i[0] && o[1] == i[1] * i[2] * i[3];
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
              {"byte_budget", options.byte_budget},
              {"images_batch_first", options.images_batch_first},
              {"input_channel_names", options.input_channels.names},
              {"input_colour_channels", options.input_channels.colour}}},
            // Every rank-4 input / output is kept as [H, W, C, N] (TOFIX207).
            {"image_layout", "HWCN"},
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
        options.images_batch_first = o.value("images_batch_first", options.images_batch_first);
        options.input_channels.names = o.value("input_channel_names", std::vector<std::string>{});
        options.input_channels.colour = o.value("input_colour_channels", size_t{0});
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
        // Saved before the layout was recorded: a Flatten-first model kept
        // its image batch first. Its first layer turns [N, H, W, C] into
        // [N, H x W x C] rows (a spatial model's Flatten or pool comes after
        // a spatial layer); convert those captures to [H, W, C, N].
        if (!index.contains("image_layout") && LooksBatchFirst(layers)) {
            options.images_batch_first = true;
            for (auto& layer : layers)
                for (auto& t : layer.tensors)
                    if (t.role == CapturedTensorRole::Input || t.role == CapturedTensorRole::Output)
                        t.tensor = ImagesBatchLast(t.tensor);
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

std::vector<LayerCaptureInfo> ListLayerCaptureInfo(const std::string& run_id) {
    std::vector<LayerCaptureInfo> infos;
    for (const auto& label : ListLayerCaptures(run_id)) {
        LayerCaptureInfo info;
        info.label = label;
        int epoch = 0;
        if (std::sscanf(label.c_str(), "epoch_%d", &epoch) == 1) info.epoch = epoch;
        std::error_code ec;
        for (const auto& entry : fs::directory_iterator(LayerCaptureDir(run_id, label), ec))
            if (entry.is_regular_file(ec)) info.bytes += entry.file_size(ec);
        infos.push_back(std::move(info));
    }
    return infos;
}

std::vector<std::string> ListRunsWithLayerCaptures() {
    std::vector<std::pair<fs::file_time_type, std::string>> runs;
    std::error_code ec;
    for (const auto& entry : fs::directory_iterator(GetDebugRunStoreRoot(), ec)) {
        const std::string run_id = entry.path().filename().string();
        if (!entry.is_directory(ec) || !SafeFolderName(run_id) || ListLayerCaptures(run_id).empty()) continue;
        const auto stamp = fs::last_write_time(entry.path() / "layers", ec);
        runs.emplace_back(ec ? fs::file_time_type{} : stamp, run_id);
    }
    std::sort(runs.begin(), runs.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
    std::vector<std::string> ids;
    for (auto& run : runs) ids.push_back(std::move(run.second));
    return ids;
}

bool SaveTrainingRunRecord(const TrainingRunRecord& record, std::string* error) {
    if (!SafeFolderName(record.run_id)) return Fail(error, "the run id is not a folder name", false);
    try {
        const fs::path dir = GetDebugRunStoreRoot() / record.run_id / "layers";
        fs::create_directories(dir);
        nlohmann::json epochs = nlohmann::json::array();
        for (const auto& e : record.epochs) {
            nlohmann::json row = {{"epoch", e.epoch}, {"train_loss", e.train_loss}, {"train_metric", e.train_metric}};
            if (e.val_loss) row["val_loss"] = *e.val_loss;
            if (e.val_metric) row["val_metric"] = *e.val_metric;
            epochs.push_back(std::move(row));
        }
        const nlohmann::json json = {{"schema", kTrainingSchema},
                                     {"graph_name", record.graph_name},
                                     {"total_epochs", record.total_epochs},
                                     {"regression", record.regression},
                                     {"epochs", std::move(epochs)},
                                     {"sample",
                                      {{"item_names", record.item_names},
                                       {"class_names", record.class_names},
                                       {"label_values", record.label_values},
                                       {"label_shape", record.label_shape},
                                       {"output_rule", record.output_rule}}}};
        // Written beside, then moved over the old one: a reader never sees half a file.
        const fs::path next = dir / (std::string(kTrainingFile) + ".next");
        {
            std::ofstream file(next, std::ios::trunc);
            file << json.dump(1) << '\n';
            if (!file.good()) return Fail(error, "writing " + next.string() + " failed", false);
        }
        fs::rename(next, dir / kTrainingFile);
        return true;
    } catch (const std::exception& e) {
        return Fail(error, std::string("saving the training record failed: ") + e.what(), false);
    }
}

std::optional<TrainingRunRecord> LoadTrainingRunRecord(const std::string& run_id) {
    if (!SafeFolderName(run_id)) return std::nullopt;
    try {
        std::ifstream file(GetDebugRunStoreRoot() / run_id / "layers" / kTrainingFile);
        if (!file.is_open()) return std::nullopt;
        nlohmann::json json;
        file >> json;
        if (json.value("schema", std::string{}) != kTrainingSchema) return std::nullopt;
        TrainingRunRecord record;
        record.run_id = run_id;
        record.graph_name = json.value("graph_name", std::string{});
        record.total_epochs = json.value("total_epochs", 0);
        record.regression = json.value("regression", false);
        for (const auto& row : json.at("epochs")) {
            TrainingEpochRecord e;
            e.epoch = row.at("epoch").get<int>();
            e.train_loss = row.value("train_loss", 0.0f);
            e.train_metric = row.value("train_metric", 0.0f);
            if (row.contains("val_loss")) e.val_loss = row.at("val_loss").get<float>();
            if (row.contains("val_metric")) e.val_metric = row.at("val_metric").get<float>();
            record.epochs.push_back(e);
        }
        if (json.contains("sample")) {
            const auto& s = json.at("sample");
            record.item_names = s.value("item_names", std::vector<std::string>{});
            record.class_names = s.value("class_names", std::vector<std::string>{});
            record.label_values = s.value("label_values", std::vector<float>{});
            record.label_shape = s.value("label_shape", std::vector<size_t>{});
            record.output_rule = s.value("output_rule", std::string{});
        }
        return record;
    } catch (const std::exception&) {
        return std::nullopt;
    }
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
