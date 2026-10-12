// Warm start from a checkpoint (TOFIX143 P2): weights only, tokenizer identity,
// exact parameter compatibility, and provenance written by later checkpoints.
#include "core/checkpoint_manager.h"
#include "core/model_builder.h"
#include <nlohmann/json.hpp>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <stdexcept>
#include <vector>

using namespace cyxwiz;
namespace fs = std::filesystem;

namespace {
int checks = 0;
void Require(bool ok, const std::string& why) { ++checks; if (!ok) throw std::runtime_error(why); }

TrainingConfiguration Config(int hidden) {
    TrainingConfiguration cfg;
    cfg.input_size = 4; cfg.output_size = 2;
    cfg.preprocessing_domain = PreprocessingDomain::Tabular;
    CompiledLayer a; a.type = gui::NodeType::Dense; a.node_id = 1; a.name = "Hidden"; a.units = hidden;
    a.input_shape = {4}; a.output_shape = {static_cast<size_t>(hidden)};
    CompiledLayer b; b.type = gui::NodeType::Dense; b.node_id = 2; b.name = "Output"; b.units = 2;
    b.input_shape = {static_cast<size_t>(hidden)}; b.output_shape = {2};
    cfg.layers = {a, b};
    cfg.loss_type = gui::NodeType::CrossEntropyLoss; cfg.optimizer_type = gui::NodeType::Adam;
    return cfg;
}

std::unique_ptr<SequentialModel> Model(int hidden) {
    auto built = BuildSequentialFromConfig(Config(hidden));
    Require(built.ok(), "model build");
    return std::move(built.model);
}

std::map<std::string, std::vector<float>> Weights(SequentialModel& model) {
    std::map<std::string, std::vector<float>> result;
    for (const auto& [name, tensor] : model.GetParameters()) {
        const auto* data = tensor.ReadData<float>();
        result[name] = std::vector<float>(data, data + tensor.NumElements());
    }
    return result;
}

nlohmann::json ReadJson(const fs::path& path) { std::ifstream file(path); return nlohmann::json::parse(file); }
void WriteJson(const fs::path& path, const nlohmann::json& value) { std::ofstream file(path); file << value.dump(2); }
}

int main() {
    try {
        const fs::path run = fs::temp_directory_path() / "cyxwiz_warm_start_test" / "train-1";
        fs::remove_all(run.parent_path());
        auto source = Model(3);
        {
            CheckpointManager manager(run.string());
            manager.SetProvenance("abc123", 256, "");
            Require(!manager.SaveCheckpoint(*source, nullptr, TrainingMetrics{}, "best").empty(), "save");
        }
        const fs::path best = run / "best";
        auto metadata = ReadJson(best / "metadata.json");
        Require(metadata.value("tokenizer_sha256", "") == "abc123", "tokenizer provenance written");
        Require(metadata.value("context", 0) == 256, "context provenance written");
        Require(!metadata.contains("initialized_from"), "no warm-start provenance on a fresh run");

        // Weights arrive exactly; the source checkpoint's learning rate is not applied.
        std::string error;
        auto target = Model(3);
        Require(Weights(*target) != Weights(*source), "fresh model differs before warm start");
        auto loaded = LoadWarmStartCheckpoint(best, *target, "abc123", error);
        Require(loaded.has_value(), "warm start: " + error);
        Require(Weights(*target) == Weights(*source), "weights copied exactly");
        auto trailing = Model(3);
        Require(LoadWarmStartCheckpoint(best.string() + "/", *trailing, "abc123", error).has_value(), "trailing slash: " + error);

        // A tokenizer mismatch fails before the model is touched.
        auto untouched = Model(3);
        const auto before = Weights(*untouched);
        Require(!LoadWarmStartCheckpoint(best, *untouched, "xyz789", error), "tokenizer mismatch must fail");
        Require(error.find("tokenizer differs") != std::string::npos, "mismatch message: " + error);
        Require(Weights(*untouched) == before, "failed warm start leaves the model unchanged");

        // Shape and existence errors name the problem.
        auto wider = Model(5);
        Require(!LoadWarmStartCheckpoint(best, *wider, "abc123", error), "shape mismatch must fail");
        Require(error.find("shape mismatch") != std::string::npos || error.find("count") != std::string::npos,
                "shape message: " + error);
        Require(!LoadWarmStartCheckpoint(run / "missing", *target, "abc123", error), "missing folder must fail");

        // Checkpoints from before TOFIX143: identity from the same run's previews.
        metadata.erase("tokenizer_sha256");
        WriteJson(best / "metadata.json", metadata);
        Require(!LoadWarmStartCheckpoint(best, *Model(3), "abc123", error), "no identity must fail");
        Require(error.find("no tokenizer identity") != std::string::npos, "no-identity message: " + error);
        fs::create_directories(run / "generation_previews");
        WriteJson(run / "generation_previews" / "epoch_1.json", {{"tokenizer_sha256", "abc123"}});
        Require(LoadWarmStartCheckpoint(best, *Model(3), "abc123", error).has_value(), "preview identity: " + error);
        Require(!LoadWarmStartCheckpoint(best, *Model(3), "xyz789", error), "preview identity mismatch must fail");
        WriteJson(run / "generation_previews" / "epoch_2.json", {{"tokenizer_sha256", "other"}});
        Require(!LoadWarmStartCheckpoint(best, *Model(3), "abc123", error), "disagreeing previews must fail");
        // Runs without a tokenizer (non-language models) need no identity.
        Require(LoadWarmStartCheckpoint(best, *Model(3), "", error).has_value(), "no tokenizer expected: " + error);

        // The new run records where it started.
        {
            CheckpointManager manager((run.parent_path() / "train-2").string());
            manager.SetProvenance("abc123", 512, best.string());
            Require(!manager.SaveCheckpoint(*target, nullptr, TrainingMetrics{}, "best").empty(), "save warm-started");
            const auto reloaded = manager.LoadCheckpoint(*Model(3), nullptr, "best");
            Require(reloaded && reloaded->initialized_from == best.string() && reloaded->context == 512 &&
                    reloaded->tokenizer_sha256 == "abc123", "provenance round trip");
        }
        fs::remove_all(run.parent_path());
        std::cout << "Warm start checkpoint passed: " << checks << " checks\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "FAIL: " << e.what() << "\n";
        return 1;
    }
}
