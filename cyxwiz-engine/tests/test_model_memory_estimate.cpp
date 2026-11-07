// The training-step memory estimate for every layer kind (TOFIX178 6.0d).
// Each example graph is compiled without its data, built, and estimated;
// then one training step (forward, backward, optimizer update) runs on a
// full batch and ArrayFire's allocated bytes are read before and after, so
// the estimate is checked against what the device really holds.
#include "core/graph_compiler.h"
#include "core/graph_compiler_dataset_hooks.h"
#include "core/graph_document.h"
#include "core/model_builder.h"
#include "core/model_memory_estimate.h"
#include "core/synthetic_batch.h"
#include "causal_lm_token_window_fixture.h"

#include <arrayfire.h>
#include <spdlog/spdlog.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <optional>
#include <sstream>
#include <set>
#include <string>

namespace fs = std::filesystem;

namespace {

int g_failures = 0;

void Check(bool ok, const std::string& what) {
    std::cout << (ok ? "  ok   " : "  FAIL ") << what << "\n";
    if (!ok) ++g_failures;
}

double Mb(uint64_t bytes) { return static_cast<double>(bytes) / (1024.0 * 1024.0); }

std::string MbText(uint64_t bytes) {
    std::ostringstream out;
    out.precision(1);
    out << std::fixed << Mb(bytes) << " MB";
    return out.str();
}

uint64_t AllocatedBytes() {
    size_t alloc_bytes = 0, alloc_buffers = 0, lock_bytes = 0, lock_buffers = 0;
    af::deviceMemInfo(&alloc_bytes, &alloc_buffers, &lock_bytes, &lock_buffers);
    return alloc_bytes;
}

cyxwiz::TrainingConfiguration CompileText(const std::string& relative, const std::string& text) {
    cyxwiz::GraphDocument graph;
    std::string error;
    if (!cyxwiz::ParseGraphDocument(text, graph, error)) {
        Check(false, relative + " loads: " + error);
        return {};
    }
    auto config = cyxwiz::GraphCompiler{}.Compile(graph.nodes, graph.links, /*allow_unloaded_data=*/true);
    if (!config.is_valid) {
        for (const auto& issue : config.issues) {
            if (issue.level == cyxwiz::IssueLevel::Error) std::cout << "       issue: " << issue.message << "\n";
        }
    }
    return config;
}

std::string ExampleText(const std::string& relative) {
    std::ifstream in(fs::path(CYXWIZ_SOURCE_ROOT) / "examples" / "cyxgraph" / relative, std::ios::binary);
    return std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}

// One training step on a full batch: growth of ArrayFire's allocation over
// it (gradients, optimizer state, activations); the weights exist before.
uint64_t MeasureStep(cyxwiz::BuiltExecutableModel& built, const cyxwiz::TrainingConfiguration& config,
                     const std::vector<size_t>& sample_override) {
    const auto sample = cyxwiz::MakeSyntheticBatch(config);
    // Images come as [H, W, C, N] (batch last); the rest is the compiled
    // per-sample input with the batch first.
    const size_t batch_size = static_cast<size_t>(std::max(1, config.batch_size));
    std::vector<size_t> shape = sample.features.Shape();
    if (shape.size() == 4 && shape.back() == 1) {
        shape[3] = batch_size;
    } else {
        const std::vector<size_t>& sample_shape = !sample_override.empty() ? sample_override : config.input_shape;
        shape = {batch_size};
        shape.insert(shape.end(), sample_shape.begin(), sample_shape.end());
    }
    // Zeros: a valid token id for embeddings; the values do not change the memory.
    const cyxwiz::Tensor batch = cyxwiz::Tensor::Zeros(shape, sample.features.GetDataType());
    built.model->SetTraining(true);
    af::sync();
    af::deviceGC();
    const uint64_t before = AllocatedBytes();
    const cyxwiz::Tensor output = built.model->Forward(batch);
    built.model->Backward(cyxwiz::Tensor::Ones(output.Shape()));
    built.model->UpdateParameters(built.optimizer.get());
    af::sync();
    const uint64_t after = AllocatedBytes();
    return after > before ? after - before : 0;
}

struct Expect {
    std::vector<size_t> sample_shape;  // the batch's per-sample shape when the compiled input is flat
    bool has_state = false;            // BatchNorm running statistics
};

void RunChecked(const std::string& name, const std::string& relative, const std::string& text, const Expect& expect);

void Run(const std::string& name, const std::string& relative, const std::string& text, const Expect& expect = {}) {
    try {
        RunChecked(name, relative, text, expect);
    } catch (const std::exception& error) {
        Check(false, name + " threw: " + error.what());
    }
}

void Run(const std::string& name, const std::string& relative, const Expect& expect = {}) {
    Run(name, relative, ExampleText(relative), expect);
}

void RunChecked(const std::string& name, const std::string& relative, const std::string& text, const Expect& expect) {
    std::cout << name << " (" << relative << ")\n";
    const auto config = CompileText(relative, text);
    Check(config.is_valid, "compiles without its data");
    if (!config.is_valid) return;
    auto built = cyxwiz::BuildExecutableFromConfig(config);
    Check(built.ok(), "builds" + (built.error_message.empty() ? "" : ": " + built.error_message));
    if (!built.ok()) return;
    const auto estimate = cyxwiz::EstimateTrainingMemory(built, config.input_shape, config.batch_size);
    Check(estimate.available, "estimated");

    // The parameters are the built model's, trained and kept apart.
    uint64_t all = 0, state = 0;
    const auto state_names = built.model->GetStateParameterNames();
    for (const auto& [key, tensor] : built.model->GetParameters()) {
        all += tensor.NumElements();
        if (state_names.count(key)) state += tensor.NumElements();
    }
    Check(estimate.trained_elements + estimate.state_elements == all && estimate.state_elements == state,
          "parameters " + std::to_string(estimate.trained_elements) + " trained + " +
              std::to_string(estimate.state_elements) + " kept = the built model's " + std::to_string(all));
    Check((estimate.state_elements > 0) == expect.has_state,
          expect.has_state ? "BatchNorm running statistics counted as kept state" : "no kept state");
    Check(estimate.optimizer_tensors_per_parameter == built.optimizer->StateTensorsPerParameter() &&
              estimate.optimizer_bytes == estimate.trained_elements * estimate.optimizer_tensors_per_parameter * 4,
          "optimizer state: " + std::to_string(estimate.optimizer_tensors_per_parameter) + " per parameter");

    // What a layer capture keeps: the input and each compiled layer's output
    // once, without the modules' caches.
    uint64_t outputs = 1;
    for (size_t d : config.input_shape) outputs *= d;
    std::set<size_t> layers;
    for (const auto& entry : built.module_provenance) {
        if (!layers.insert(entry.compiled_layer_index).second) continue;
        uint64_t n = entry.output_shape.empty() ? 0 : 1;
        for (size_t d : entry.output_shape) n *= d;
        outputs += n;
    }
    Check(estimate.output_elements_per_sample == outputs &&
              estimate.output_elements_per_sample <= estimate.activation_elements_per_sample,
          "outputs per sample " + std::to_string(estimate.output_elements_per_sample) + " = input + layer outputs " +
              std::to_string(outputs));

    // Against the device: the step's growth over the weights already held.
    const uint64_t step_estimate = estimate.total_bytes - estimate.parameter_bytes - estimate.state_bytes;
    const uint64_t measured = MeasureStep(built, config, expect.sample_shape);
    const std::string numbers = "estimate " + MbText(step_estimate) + ", measured " + MbText(measured) +
                                " (batch " + std::to_string(estimate.batch_size) + ")";
    if (measured < 8ull * 1024 * 1024) {
        // A small model's step is dominated by allocator rounding.
        std::cout << "       " << numbers << "\n";
        return;
    }
    const double ratio = static_cast<double>(step_estimate) / static_cast<double>(measured);
    Check(ratio > 0.6 && ratio < 1.6, "within the measured step: " + numbers);
}

}  // namespace

int main() {
    std::cout << std::unitbuf;
    spdlog::set_level(spdlog::level::err);
    cyxwiz::GraphCompilerDatasetHooks hooks;
    hooks.is_dataset_registered = [](const std::string&) { return false; };
    hooks.labels_from_structure = [](const std::string& category) -> std::optional<bool> {
        return category == "image";
    };
    hooks.preprocessing_domain = [](const std::string& category) -> std::optional<cyxwiz::PreprocessingDomain> {
        if (category == "image") return cyxwiz::PreprocessingDomain::Image;
        if (category == "text") return cyxwiz::PreprocessingDomain::Text;
        return cyxwiz::PreprocessingDomain::Tabular;
    };
    cyxwiz::SetGraphCompilerDatasetHooks(std::move(hooks));

    Run("ResNet (conv, BatchNorm, skips)", "cats_dogs_resnet_scratch.cyxgraph", Expect{{}, true});
    // The tagger reads word and tag ids per position: [seq, 2] per sample.
    Run("BiLSTM tagger (embedding, recurrent)", "NER/ner_bilstm_sequence_tagger.cyxgraph", Expect{{96, 2}, false});
    Run("decoder LM (embedding, attention, transformer blocks)", "text/causal_lm_decoder_generation_experiment",
        cyxwiz::test::LoadTokenWindowGraph(CYXWIZ_SOURCE_ROOT).dump());
    {
        // The same decoder at 128 tokens and batch 32: attention's
        // [heads, seq, seq] weights dominate.
        auto graph = cyxwiz::test::LoadTokenWindowGraph(CYXWIZ_SOURCE_ROOT);
        for (auto& node : graph["nodes"]) {
            auto& params = node["parameters"];
            if (params.contains("shape")) params["shape"] = "[128]";
            for (const char* key : {"max_len", "max_length"}) {
                if (params.contains(key)) params[key] = "128";
            }
            if (params.contains("batch_size")) params["batch_size"] = "32";
        }
        Run("decoder LM at 128 tokens", "text/causal_lm_decoder_generation_experiment", graph.dump());
    }

    std::cout << (g_failures == 0 ? "all checks passed\n" : std::to_string(g_failures) + " checks failed\n");
    return g_failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
