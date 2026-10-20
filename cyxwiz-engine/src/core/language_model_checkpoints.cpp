#include "language_model_checkpoints.h"

#include "graph_document.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>

namespace cyxwiz {
namespace {

std::string ReadFile(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("Cannot read " + path.string());
    std::ostringstream text;
    text << input.rdbuf();
    return text.str();
}

std::string Param(const gui::MLNode& node, const char* key) {
    const auto it = node.parameters.find(key);
    return it == node.parameters.end() ? std::string() : it->second;
}

std::string Fixed(double value, int decimals) {
    char buffer[32];
    std::snprintf(buffer, sizeof(buffer), "%.*f", decimals, value);
    return buffer;
}

}  // namespace

std::string LanguageModelThousands(size_t value) {
    std::string digits = std::to_string(value);
    for (int at = static_cast<int>(digits.size()) - 3; at > 0; at -= 3) digits.insert(static_cast<size_t>(at), ",");
    return digits;
}

// "2026-10-14T13:26:55Z" -> "2026-10-14 13:26".
std::string CheckpointSavedLabel(const std::string& timestamp) {
    if (timestamp.size() >= 16 && timestamp[10] == 'T') return timestamp.substr(0, 10) + " " + timestamp.substr(11, 5);
    return timestamp;
}

std::vector<CheckpointRunRow> ListCheckpointRuns(const std::filesystem::path& folder) {
    namespace fs = std::filesystem;
    std::vector<CheckpointRunRow> rows;
    std::vector<std::pair<fs::file_time_type, CheckpointRunRow>> found;
    std::error_code ec;
    const auto add = [&](const fs::path& checkpoint, const std::string& run) {
        std::error_code inner;
        const auto metadata_path = checkpoint / "metadata.json";
        if (!fs::is_regular_file(metadata_path, inner)) return;
        CheckpointRunRow row;
        row.run = run;
        row.weights = checkpoint.filename().string();
        row.folder = checkpoint;
        try {
            const auto metadata = nlohmann::json::parse(ReadFile(metadata_path));
            row.epoch = metadata.value("epoch", 0);
            const double loss = metadata.value("val_loss", 0.0);
            if (loss > 0.0) row.validation_loss = loss;
            row.saved = CheckpointSavedLabel(metadata.value("timestamp", std::string()));
        } catch (const std::exception&) {
            return;
        }
        found.push_back({fs::last_write_time(metadata_path, inner), std::move(row)});
    };
    for (fs::directory_iterator it(folder, ec), end; !ec && it != end; it.increment(ec)) {
        std::error_code inner;
        if (!it->is_directory(inner)) continue;
        add(it->path() / "best", it->path().filename().string());
        add(it->path() / "latest", it->path().filename().string());
    }
    std::sort(found.begin(), found.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
    for (auto& [time, row] : found) rows.push_back(std::move(row));
    return rows;
}

std::filesystem::path GraphCheckpointFolder(const std::string& graph_path, const std::string& project_root,
                                            const std::string& project_checkpoints) {
    namespace fs = std::filesystem;
    try {
        GraphDocument graph;
        std::string error;
        if (ParseGraphDocument(ReadFile(graph_path), graph, error)) {
            for (const auto& node : graph.nodes) {
                if (node.type != gui::NodeType::DataLoader) continue;
                const fs::path folder = Param(node, "checkpoint_dir");
                if (folder.empty()) break;
                return folder.is_relative() && !project_root.empty() ? fs::path(project_root) / folder : folder;
            }
        }
    } catch (const std::exception&) {
    }
    return project_checkpoints;
}

std::string LanguageModelSummary(int epoch, std::optional<double> validation_loss, int context, size_t vocabulary,
                                 const std::string& tokenizer_kind) {
    std::vector<std::string> parts;
    if (epoch > 0) parts.push_back("Epoch " + std::to_string(epoch));
    if (validation_loss) parts.push_back("validation loss " + Fixed(*validation_loss, 2));
    if (context > 0) parts.push_back("context " + std::to_string(context));
    parts.push_back("vocabulary " + LanguageModelThousands(vocabulary) + " (" + tokenizer_kind + ")");
    std::string text;
    for (const auto& part : parts) text += (text.empty() ? "" : " \xC2\xB7 ") + part;
    if (!text.empty()) text[0] = static_cast<char>(std::toupper(static_cast<unsigned char>(text[0])));
    return text;
}

bool SaveGenerationSettingsToGraph(const std::string& graph_path, int loader_node_id,
                                   const TrainingGenerationPreviewSettings& settings, std::string& error) {
    try {
        auto document = nlohmann::json::parse(ReadFile(graph_path));
        for (auto& node : document.at("nodes")) {
            if (node.value("id", -1) != loader_node_id) continue;
            auto& parameters = node["parameters"];
            std::map<std::string, std::string> current;
            for (auto it = parameters.begin(); it != parameters.end(); ++it)
                if (it.value().is_string()) current[it.key()] = it.value().get<std::string>();
            // Keep on/off, prompts, cadence and weights; take the rest from `settings`.
            auto merged = ParseTrainingGenerationPreview(current);
            const bool enabled = merged.enabled;
            const auto prompts = merged.prompts;
            const int every = merged.every;
            const auto cadence = merged.cadence;
            const bool baseline = merged.baseline, after_last = merged.after_last;
            const auto weights = merged.weights;
            const int samples = merged.samples_per_prompt;
            merged = settings;
            merged.enabled = enabled;
            merged.prompts = prompts;
            merged.every = every;
            merged.cadence = cadence;
            merged.baseline = baseline;
            merged.after_last = after_last;
            merged.weights = weights;
            merged.samples_per_prompt = merged.decoding == GenerationPreviewDecoding::Greedy ? 1 : samples;
            if (const auto problem = ValidateTrainingGenerationPreview(merged); !problem.empty()) {
                error = problem;
                return false;
            }
            for (const auto& [key, value] : SerializeTrainingGenerationPreview(merged)) parameters[key] = value;
            std::ofstream output(graph_path, std::ios::binary);
            output << document.dump(2) << "\n";
            if (!output) {
                error = "Cannot write " + graph_path;
                return false;
            }
            return true;
        }
        error = "The graph has no Data Loader " + std::to_string(loader_node_id);
    } catch (const std::exception& exception) {
        error = exception.what();
    }
    return false;
}

}  // namespace cyxwiz
