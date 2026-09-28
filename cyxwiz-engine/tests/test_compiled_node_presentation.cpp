// The Properties "AS COMPILED" card (TOFIX123) on real graphs: the NER
// example (word + POS fusion) and the Berean full-data decoder, compiled and
// built exactly as the Engine does, with no dataset loaded.
#include "../src/core/compiled_node_presentation.h"
#include "../src/core/graph_compiler.h"
#include "../src/core/graph_document.h"
#include "../src/core/model_builder.h"
#include "../src/gui/loaders/data_loader.h"

#include <nlohmann/json.hpp>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

// Graph-only: no file loaders, so datasets are not read.
namespace cyxwiz::loaders {
DataLoader* GetByCategory(FileCategory) { return nullptr; }
DataLoader* GetByRegisteredDataset(const std::string&) { return nullptr; }
FileCategory FileCategoryFromString(const std::string&) { return FileCategory::Tabular; }
}  // namespace cyxwiz::loaders

namespace {

void Check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        std::exit(1);
    }
}

std::filesystem::path RepoRoot() {
    auto dir = std::filesystem::current_path();
    while (!dir.empty()) {
        if (std::filesystem::exists(dir / "examples" / "cyxgraph" / "NER")) return dir;
        const auto parent = dir.parent_path();
        if (parent == dir) break;
        dir = parent;
    }
    return std::filesystem::current_path();
}

struct Compiled {
    cyxwiz::GraphDocument graph;
    cyxwiz::TrainingConfiguration config;
    std::map<size_t, long long> layer_parameters;
    bool parameters_counted = false;
};

Compiled CompileAndCount(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    Check(in.is_open(), "could not open " + path.string());
    std::stringstream text;
    text << in.rdbuf();
    Compiled out;
    std::string error;
    Check(cyxwiz::ParseGraphDocument(text.str(), out.graph, error), "graph should load: " + error);
    out.config = cyxwiz::GraphCompiler{}.Compile(out.graph.nodes, out.graph.links, true);
    auto built = cyxwiz::BuildExecutableFromConfig(out.config);
    if (built.ok()) {
        std::vector<std::pair<std::string, long long>> named;
        for (const auto& [name, tensor] : built.model->GetParameters()) {
            named.emplace_back(name, static_cast<long long>(tensor.NumElements()));
        }
        out.layer_parameters = cyxwiz::CountParametersPerLayer(named);
        out.parameters_counted = true;
    }
    return out;
}

int IdNamed(const cyxwiz::GraphDocument& graph, const std::string& name) {
    for (const auto& node : graph.nodes) {
        if (node.name == name) return node.id;
    }
    Check(false, "missing node " + name);
    return -1;
}

cyxwiz::CompiledNodeCard Card(const Compiled& compiled, const std::string& name, const std::string& type_label,
                              bool data_loaded = true) {
    cyxwiz::CompiledNodeInputs inputs;
    inputs.state = cyxwiz::LiveCompileState::Compiled;
    inputs.config = &compiled.config;
    inputs.data_loaded = data_loaded;
    inputs.layer_parameters = compiled.layer_parameters;
    inputs.parameters_counted = compiled.parameters_counted;
    inputs.type_label = type_label;
    return cyxwiz::BuildCompiledNodeCard(compiled.graph.nodes, compiled.graph.links,
                                         IdNamed(compiled.graph, name), inputs);
}

void CheckFormatting() {
    Check(cyxwiz::FormatCompiledCount(3002048) == "3,002,048" && cyxwiz::FormatCompiledCount(0) == "0" &&
              cyxwiz::FormatCompiledCount(256) == "256",
          "counts are grouped by thousands");
    Check(cyxwiz::FormatCompiledBytes(1425408) == "1.36 MB" && cyxwiz::FormatCompiledBytes(791808) == "773.2 KB" &&
              cyxwiz::FormatCompiledBytes(0) == "0 bytes",
          "memory uses 1024-based units");
    Check(cyxwiz::FormatCompiledShape({96, 116}) == "[96, 116]", "shapes print as [a, b]");
    const auto per_layer = cyxwiz::CountParametersPerLayer(
        {{"layer0.weight", 10}, {"layer2.layer0.forward.W_ih", 3}, {"layer2.layer1.forward.W_ih", 4}, {"other", 9}});
    Check(per_layer.at(0) == 10 && per_layer.at(2) == 7 && per_layer.size() == 2,
          "parameter names are summed per top-level layer index");
}

void CheckNer() {
    const auto ner = CompileAndCount(RepoRoot() / "examples" / "cyxgraph" / "NER" / "ner_bilstm_sequence_tagger.cyxgraph");
    Check(ner.config.is_valid, "NER compiles offline");

    const auto concat = Card(ner, "Concat Word + POS", "Concatenate");
    Check(concat.kind == cyxwiz::CompiledStatusKind::Ok && concat.status == "Compiled", "NER concat compiled");
    Check(concat.role == "Layer 1 of 4: word + POS fusion (with Word Embedding and POS Embedding)",
          "fused layer role names both Embeddings: " + concat.role);
    Check(concat.input_sample == "[96, 2]" && concat.output_sample == "[96, 116]" &&
              concat.input_batch == "[32, 96, 2]" && concat.output_batch == "[32, 96, 116]",
          "fused layer shapes per sample and batch");
    Check(concat.output_memory == "1.36 MB", "fused layer output memory per batch");
    Check(concat.parameters == "3,002,048" && concat.parameter_memory == "11.45 MB",
          "fused layer parameters from the built model");
    Check(concat.parameter_note.find("Word 30,000 x 100 (3,000,000) + POS 128 x 16 (2,048)") == 0,
          "fused layer parameter breakdown: " + concat.parameter_note);

    const auto word = Card(ner, "Word Embedding", "Embedding");
    Check(word.role == "Part of layer 1 (word + POS fusion); the layer is built on Concat Word + POS",
          "word Embedding is part of the fused layer: " + word.role);
    Check(word.select_node_id == IdNamed(ner.graph, "Concat Word + POS") &&
              word.select_label == "Select Concat Word + POS",
          "word Embedding offers to select the fused layer's node");
    Check(word.input_sample == "[96]" && word.output_sample == "[96, 100]" && word.parameters == "3,000,000",
          "word Embedding shapes and parameters");

    const auto lstm = Card(ner, "BiLSTM Sequence Encoder", "LSTM");
    Check(lstm.role == "Layer 2 of 4: LSTM" && lstm.input_sample == "[96, 116]" && lstm.output_sample == "[96, 256]" &&
              lstm.parameters == "647,168" && lstm.output_memory == "3.00 MB",
          "BiLSTM compiled facts");

    const auto dropout = Card(ner, "Dropout", "Dropout");
    Check(dropout.parameters == "0" && dropout.output_sample == "[96, 256]", "Dropout has no parameters");

    const auto data = Card(ner, "NER Sentence CSV", "Data Input", false);
    Check(data.role == "Data source for the model" && data.output_sample == "[96]",
          "the Data Input shows the model input it provides");
    Check(data.status_note == "Data not loaded: shapes come from the saved data contract.",
          "unloaded data is said plainly");

    const auto split = Card(ner, "Split 80/10/10", "Data Split");
    Check(split.role == "On the training path; not a model layer" && !split.has_shapes, "Split is not a layer");

    // dim = 1: the Concatenate has its own error, so it shows no shapes.
    auto broken = ner;
    for (auto& node : broken.graph.nodes) {
        if (node.name == "Concat Word + POS") node.parameters["dim"] = "1";
    }
    broken.config = cyxwiz::GraphCompiler{}.Compile(broken.graph.nodes, broken.graph.links, true);
    const auto failed = Card(broken, "Concat Word + POS", "Concatenate");
    Check(failed.kind == cyxwiz::CompiledStatusKind::Failed && !failed.has_shapes,
          "a node with its own compile error shows no shapes");
    Check(!failed.issues.empty() && failed.issues.front().error &&
              failed.issues.front().message.find("Concatenate dim is 1") != std::string::npos,
          "the node's own compiler error is listed");
    Check(failed.status_note == "1 error on this node. Shapes appear once the graph compiles.",
          "status names the node's error count: " + failed.status_note);

    cyxwiz::CompiledNodeInputs counting;
    counting.state = cyxwiz::LiveCompileState::Compiled;
    counting.config = &ner.config;
    counting.parameters_counting = true;
    const auto pending = cyxwiz::BuildCompiledNodeCard(ner.graph.nodes, ner.graph.links,
                                                        IdNamed(ner.graph, "BiLSTM Sequence Encoder"), counting);
    Check(pending.parameters == "counting..." && pending.output_sample == "[96, 256]",
          "shapes show at once while parameters are still counting");

    cyxwiz::CompiledNodeInputs none;
    const auto not_compiled = cyxwiz::BuildCompiledNodeCard(ner.graph.nodes, ner.graph.links,
                                                             IdNamed(ner.graph, "Dropout"), none);
    Check(not_compiled.status == "Not compiled yet" && !not_compiled.has_shapes, "before the first compile");
}

void CheckBerean() {
    const auto path = RepoRoot() / "examples" / "thinking_llm" / "Project Berean" / "cyxgraph" /
                      "berean_base_corpus_qknorm_T7recipe_train_v1.cyxgraph";
    if (!std::filesystem::exists(path)) {
        std::cout << "Berean graph not present (untracked example); skipped\n";
        return;
    }
    const auto berean = CompileAndCount(path);
    const auto block = Card(berean, "llama_style + QK-norm block 2", "TransformerDecoder", false);
    Check(block.role == "Layer 3 of 7: TransformerDecoder" && block.input_sample == "[256, 128]" &&
              block.output_batch == "[8, 256, 128]" && block.parameters == "197,952" &&
              block.parameter_memory == "773.2 KB" && block.output_memory == "1.00 MB",
          "Berean decoder block compiled facts");
    if (!berean.config.is_valid) {
        // Offline the generation-preview dataset is missing: errors elsewhere,
        // the block still shows the shapes that compile gave it.
        Check(block.kind == cyxwiz::CompiledStatusKind::Failed && block.has_shapes,
              "errors on other nodes keep this node's compiled shapes");
    }
}

}  // namespace

int main() {
    CheckFormatting();
    CheckNer();
    CheckBerean();
    std::cout << "Compiled node presentation passed\n";
    return 0;
}
