// Warm start from a checkpoint (TOFIX143 P2): weights only, tokenizer identity,
// exact parameter compatibility, and provenance written by later checkpoints.
#include "core/checkpoint_manager.h"
#include "core/model_builder.h"
#include <nlohmann/json.hpp>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
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
        // BatchNorm's running statistics are state, not trained: named as
        // such, saved with the weights, and a checkpoint from before they
        // were saved still loads (the state starts from the model's own).
        {
            const auto bn_model = [] {
                auto model = std::make_unique<SequentialModel>();
                model->Add<LinearModule>(4, 3);
                model->Add<BatchNormModule>(3);
                return model;
            };
            auto bn = bn_model();
            const std::set<std::string> state{"layer1.running_mean", "layer1.running_var"};
            Require(bn->GetStateParameterNames() == state, "the BatchNorm running statistics are its state");
            Require(bn->GetParameters().count("layer1.running_mean") == 1, "state is saved with the weights");
            Require(bn->GetGradients().count("layer1.running_mean") == 0, "state has no gradient");
            const fs::path bn_run = run.parent_path() / "train-bn";
            CheckpointManager manager(bn_run.string());
            Require(!manager.SaveCheckpoint(*bn, nullptr, TrainingMetrics{}, "best").empty(), "save BatchNorm model");
            const fs::path manifest_path = bn_run / "best" / "model" / "manifest.json";
            auto manifest = ReadJson(manifest_path);
            manifest.erase("layer1_running_mean.bin");
            manifest.erase("layer1_running_var.bin");
            WriteJson(manifest_path, manifest);
            Require(manager.LoadCheckpoint(*bn_model(), nullptr, "best").has_value(),
                    "a checkpoint without the running statistics loads");
        }
        // A frozen layer stays part of the model: saved and loaded with the
        // rest, never stepped (weight decay included), and without a gradient.
        {
            auto frozen = Model(3);
            const size_t all = frozen->GetParameters().size();
            frozen->FreezeLayer(0);
            const auto frozen_names = frozen->GetFrozenParameterNames();
            Require(!frozen_names.empty(), "the frozen layer's parameters are named");
            for (const auto& name : frozen_names) Require(name.rfind("layer0.", 0) == 0, "frozen name " + name);
            Require(frozen->GetParameters().size() == all, "a frozen layer stays in the model's parameters");
            Require(frozen->GetTrainableParameters().size() == all - frozen_names.size(),
                    "the optimizer is handed the other layers only");

            const std::vector<float> rows{0.5f, -1.0f, 2.0f, 0.25f, -0.75f, 1.5f, 0.0f, 3.0f};
            const std::vector<float> ones(4, 1.0f);
            AdamWOptimizer optimizer(0.05, 0.9, 0.999, 1e-8, 0.1);
            const auto frozen_before = Weights(*frozen);
            for (int step = 0; step < 3; ++step) {
                frozen->Forward(Tensor({2, 4}, rows.data()));
                frozen->Backward(Tensor({2, 2}, ones.data()));
                for (const auto& [name, gradient] : frozen->GetGradients())
                    Require(!frozen_names.count(name), "no gradient is reported for " + name);
                frozen->UpdateParameters(&optimizer);
            }
            const auto after = Weights(*frozen);
            bool others_moved = false;
            for (const auto& [name, values] : after) {
                if (frozen_names.count(name)) Require(values == frozen_before.at(name), name + " is unchanged, bit for bit");
                else others_moved = others_moved || values != frozen_before.at(name);
            }
            Require(others_moved, "the layers that are not frozen still train");

            const fs::path frozen_run = run.parent_path() / "train-frozen";
            CheckpointManager manager(frozen_run.string());
            Require(!manager.SaveCheckpoint(*frozen, nullptr, TrainingMetrics{}, "best").empty(), "save frozen model");
            auto reloaded = Model(3);
            Require(manager.LoadCheckpoint(*reloaded, nullptr, "best").has_value(),
                    "the checkpoint loads into a model with nothing frozen: " + manager.GetLastError());
            Require(Weights(*reloaded) == after, "the checkpoint holds the frozen layer's weights too");
        }
        fs::remove_all(run.parent_path());
        std::cout << "Warm start checkpoint passed: " << checks << " checks\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "FAIL: " << e.what() << "\n";
        return 1;
    }
}
