// The Model window's Summary (TOFIX129 D11) against PyTorch: graphs that
// cover the layer kinds a summary gets wrong (valid and same padding,
// strides, depthwise and transposed convolutions, a residual branch, stacked
// bidirectional recurrent layers, attention and a transformer block) are
// compiled and built by the Engine, then every layer's output shape,
// parameters and multiply-adds are compared with the same model in PyTorch
// (fixture from computation_truth/reference/generate_model_summary_fixtures.py).
// The words of the summary (tiles, advice, text and CSV) are checked on the
// MNIST CNN.
#include "../src/core/compiled_node_presentation.h"
#include "../src/core/graph_compiler.h"
#include "../src/core/graph_compiler_dataset_hooks.h"
#include "../src/core/graph_node_factory.h"
#include "../src/core/model_builder.h"
#include "../src/core/model_summary_presentation.h"
#include "../src/gui/loaders/data_loader.h"
#include "computation_truth/test_device_selection.h"

#include <nlohmann/json.hpp>

#include <cstdlib>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <string>
#include <vector>

// Graph-only: no file loaders, so datasets are not read.
namespace cyxwiz::loaders {
DataLoader* GetByCategory(FileCategory) { return nullptr; }
DataLoader* GetByRegisteredDataset(const std::string&) { return nullptr; }
FileCategory FileCategoryFromString(const std::string& text) {
    return text == "image" ? FileCategory::Image : FileCategory::Tabular;
}
}  // namespace cyxwiz::loaders

namespace {

using json = nlohmann::json;

int failures = 0;
int checks = 0;

void Check(bool condition, const std::string& message) {
    ++checks;
    if (!condition) {
        ++failures;
        std::cerr << "FAIL: " << message << "\n";
    }
}

std::string ShapeText(const std::vector<size_t>& shape) { return cyxwiz::FormatCompiledShape(shape); }

// A graph built with the editor's own node factory (registry pins).
struct Graph {
    std::vector<gui::MLNode> nodes;
    std::vector<gui::NodeLink> links;
    int next_node = 1;
    int next_pin = 1;
    int next_link = 1;

    int Add(gui::NodeType type, const std::string& name, std::map<std::string, std::string> parameters = {}) {
        auto node = gui::CreateGraphNode(type, name, next_node, next_pin);
        for (auto& [key, value] : parameters) node.parameters[key] = value;
        nodes.push_back(node);
        return node.id;
    }
    gui::MLNode& Node(int id) {
        for (auto& node : nodes)
            if (node.id == id) return node;
        std::cerr << "no node " << id << "\n";
        std::exit(1);
    }
    int PinOf(int node, bool input, gui::PinType type, int nth) {
        const auto& pins = input ? Node(node).inputs : Node(node).outputs;
        for (const auto& pin : pins)
            if (pin.type == type && nth-- == 0) return pin.id;
        std::cerr << Node(node).name << ": no " << (input ? "input" : "output") << " pin of type "
                  << static_cast<int>(type) << "\n";
        std::exit(1);
    }
    void Connect(int from, int to, gui::PinType type = gui::PinType::Tensor, int to_nth = 0,
                 gui::PinType to_type = gui::PinType::Tensor) {
        gui::NodeLink link;
        link.id = next_link++;
        link.from_node = from;
        link.from_pin = PinOf(from, false, type, 0);
        link.to_node = to;
        link.to_pin = PinOf(to, true, type == gui::PinType::Tensor ? to_type : type, to_nth);
        links.push_back(link);
    }
    // data -> split -> loader; returns the loader (the layers read its Data output).
    int Data(const std::string& category, const std::string& shape) {
        const int data = Add(gui::NodeType::DataInput, "Data",
                             {{"dataset_name", "summary_rows"}, {"file_category", category}, {"shape", shape},
                              {"label_column", "label"}, {"data_loaded", "true"}});
        const int split = Add(gui::NodeType::DataSplit, "Split");
        const int loader = Add(gui::NodeType::DataLoader, "Loader", {{"batch_size", "64"}});
        Connect(data, split, gui::PinType::Dataset);
        Connect(split, loader, gui::PinType::Dataset);
        return loader;
    }
    // A chain of layers after `from`; returns the last.
    int Chain(int from, const std::vector<int>& layers) {
        for (int layer : layers) {
            Connect(from, layer);
            from = layer;
        }
        return from;
    }
    void Finish(int loader, int last) {
        const int loss = Add(gui::NodeType::CrossEntropyLoss, "Loss");
        const int adam = Add(gui::NodeType::Adam, "Adam");
        Connect(last, loss);
        Connect(loader, loss, gui::PinType::Labels);
        Connect(loss, adam, gui::PinType::Loss);
    }
};

using Builder = std::function<Graph()>;

Graph MnistCnn() {
    Graph g;
    const int loader = g.Data("tabular", "[28, 28, 1]");
    const int last = g.Chain(loader, {
        g.Add(gui::NodeType::Conv2D, "Conv2D 32", {{"filters", "32"}, {"kernel_size", "3"}, {"padding", "valid"}}),
        g.Add(gui::NodeType::ReLU, "ReLU 1"),
        g.Add(gui::NodeType::MaxPool2D, "MaxPool 1", {{"pool_size", "2"}, {"stride", "2"}}),
        g.Add(gui::NodeType::Conv2D, "Conv2D 64", {{"filters", "64"}, {"kernel_size", "3"}, {"padding", "valid"}}),
        g.Add(gui::NodeType::ReLU, "ReLU 2"),
        g.Add(gui::NodeType::MaxPool2D, "MaxPool 2", {{"pool_size", "2"}, {"stride", "2"}}),
        g.Add(gui::NodeType::Flatten, "Flatten"),
        g.Add(gui::NodeType::Dense, "Dense 128", {{"units", "128"}}),
        g.Add(gui::NodeType::ReLU, "ReLU 3"),
        g.Add(gui::NodeType::Dropout, "Dropout", {{"rate", "0.5"}}),
        g.Add(gui::NodeType::Dense, "Dense 10", {{"units", "10"}}),
    });
    g.Finish(loader, last);
    return g;
}

Graph ImageMix() {
    Graph g;
    const int loader = g.Data("tabular", "[32, 32, 3]");
    const int last = g.Chain(loader, {
        g.Add(gui::NodeType::Conv2D, "Conv same", {{"filters", "16"}, {"kernel_size", "3"}, {"padding", "same"}}),
        g.Add(gui::NodeType::BatchNorm, "BatchNorm"),
        g.Add(gui::NodeType::ReLU, "ReLU"),
        g.Add(gui::NodeType::DepthwiseConv2D, "Depthwise s2",
              {{"kernel_size", "3"}, {"stride", "2"}, {"padding", "same"}, {"depth_multiplier", "2"}}),
        g.Add(gui::NodeType::Conv2D, "Pointwise", {{"filters", "8"}, {"kernel_size", "1"}, {"padding", "valid"}}),
        g.Add(gui::NodeType::ConvTranspose2D, "Up",
              {{"out_channels", "4"}, {"kernel_size", "2"}, {"stride", "2"}, {"padding", "0"}, {"output_padding", "0"}}),
        g.Add(gui::NodeType::GlobalAvgPool, "Global pool"),
        g.Add(gui::NodeType::Dense, "Head", {{"units", "3"}}),
    });
    g.Finish(loader, last);
    return g;
}

Graph Residual() {
    Graph g;
    const int loader = g.Data("tabular", "[8]");
    const int a = g.Add(gui::NodeType::Dense, "Dense a", {{"units", "6"}});
    const int b = g.Add(gui::NodeType::Dense, "Dense b", {{"units", "6"}});
    const int add = g.Add(gui::NodeType::Add, "Add");
    const int head = g.Add(gui::NodeType::Dense, "Head", {{"units", "3"}});
    g.Connect(loader, a);
    g.Connect(loader, b);
    g.Connect(a, add, gui::PinType::Tensor, 0);
    g.Connect(b, add, gui::PinType::Tensor, 1);
    g.Connect(add, head);
    g.Finish(loader, head);
    return g;
}

Graph Recurrent() {
    Graph g;
    const int loader = g.Data("tabular", "[20]");
    const int last = g.Chain(loader, {
        g.Add(gui::NodeType::Embedding, "Embedding", {{"num_embeddings", "1000"}, {"embedding_dim", "32"}}),
        g.Add(gui::NodeType::LSTM, "LSTM",
              {{"hidden_size", "24"}, {"num_layers", "2"}, {"bidirectional", "true"}, {"return_sequences", "true"}}),
        g.Add(gui::NodeType::GRU, "GRU", {{"hidden_size", "16"}, {"return_sequences", "true"}}),
        g.Add(gui::NodeType::RNN, "RNN", {{"hidden_size", "12"}, {"return_sequences", "false"}}),
        g.Add(gui::NodeType::Dense, "Head", {{"units", "5"}}),
    });
    g.Finish(loader, last);
    return g;
}

Graph Attention() {
    Graph g;
    const int loader = g.Data("tabular", "[16]");
    const int last = g.Chain(loader, {
        g.Add(gui::NodeType::Embedding, "Embedding", {{"num_embeddings", "500"}, {"embedding_dim", "32"}}),
        g.Add(gui::NodeType::MultiHeadAttention, "Attention", {{"embed_dim", "32"}, {"num_heads", "4"}, {"dropout", "0"}}),
        g.Add(gui::NodeType::TransformerEncoder, "Encoder",
              {{"d_model", "32"}, {"num_heads", "4"}, {"dim_feedforward", "64"}, {"dropout", "0"}}),
        g.Add(gui::NodeType::Flatten, "Flatten"),
        g.Add(gui::NodeType::Dense, "Head", {{"units", "2"}}),
    });
    g.Finish(loader, last);
    return g;
}

struct Compiled {
    cyxwiz::TrainingConfiguration config;
    std::map<size_t, long long> parameters;
    bool built = false;
};

Compiled CompileAndCount(const std::string& name, const Graph& graph) {
    Compiled out;
    out.config = cyxwiz::GraphCompiler{}.Compile(graph.nodes, graph.links, true);
    for (const auto& issue : out.config.issues) {
        if (issue.level == cyxwiz::IssueLevel::Error)
            std::cerr << "  " << name << " issue: " << issue.node_name << ": " << issue.message << "\n";
    }
    Check(out.config.is_valid, name + ": the graph compiles");
    auto built = cyxwiz::BuildExecutableFromConfig(out.config);
    Check(built.ok(), name + ": the model builds: " + built.error_message);
    if (!built.ok()) return out;
    std::vector<std::pair<std::string, long long>> named;
    for (const auto& [param, tensor] : built.model->GetParameters())
        named.emplace_back(param, static_cast<long long>(tensor.NumElements()));
    out.parameters = cyxwiz::CountParametersPerLayer(named);
    out.built = true;
    return out;
}

cyxwiz::ModelSummary Summarize(const Compiled& compiled) {
    cyxwiz::ModelSummaryInputs inputs;
    inputs.state = compiled.config.is_valid ? cyxwiz::LiveCompileState::Compiled : cyxwiz::LiveCompileState::Failed;
    inputs.config = &compiled.config;
    inputs.layer_parameters = compiled.parameters;
    inputs.parameters_counted = compiled.built;
    for (const auto& layer : compiled.config.layers) inputs.type_labels[layer.node_id] = layer.name;
    return cyxwiz::BuildModelSummary(inputs);
}

void CheckAgainstTorch(const std::string& name, const Builder& build, const json& truth, bool print) {
    const Compiled compiled = CompileAndCount(name, build());
    const auto summary = Summarize(compiled);
    if (print) {
        std::cout << "== " << name << "\n";
        for (const auto& row : summary.rows)
            std::cout << "  " << row.name << " " << row.output << " params " << row.parameters << " macs "
                      << row.multiply_adds.value << (row.multiply_adds.counted ? "" : " (not counted)") << "\n";
    }
    if (truth.is_null()) return;
    std::map<std::string, const cyxwiz::ModelSummaryRow*> by_name;
    for (const auto& row : summary.rows) by_name[row.name] = &row;
    for (const auto& layer : truth.at("layers")) {
        const std::string layer_name = layer.at("name");
        const auto found = by_name.find(layer_name);
        Check(found != by_name.end(), name + ": the summary has a row for " + layer_name);
        if (found == by_name.end()) continue;
        const auto& row = *found->second;
        const auto shape = layer.at("output").get<std::vector<size_t>>();
        Check(row.output_shape == shape,
              name + " / " + layer_name + ": output " + row.output + ", PyTorch " + ShapeText(shape));
        const long long params = layer.at("parameters");
        Check(row.parameters_known && row.parameters == params,
              name + " / " + layer_name + ": parameters " + std::to_string(row.parameters) + ", PyTorch " +
                  std::to_string(params));
        const long long macs = layer.at("multiply_adds");
        Check(row.multiply_adds.counted && row.multiply_adds.value == macs,
              name + " / " + layer_name + ": multiply-adds " + std::to_string(row.multiply_adds.value) +
                  ", PyTorch " + std::to_string(macs) + " (" + row.multiply_adds.formula + ")");
    }
    const long long params = truth.at("parameters");
    const long long macs = truth.at("multiply_adds");
    Check(summary.parameters == params, name + ": total parameters " + std::to_string(summary.parameters) +
                                            ", PyTorch " + std::to_string(params));
    Check(summary.multiply_adds == macs, name + ": total multiply-adds " + std::to_string(summary.multiply_adds) +
                                             ", PyTorch " + std::to_string(macs));
    Check(summary.uncounted_layers == 0, name + ": every layer's multiply-adds are counted");
}

void CheckWords() {
    const Compiled compiled = CompileAndCount("mnist_cnn", MnistCnn());
    const auto summary = Summarize(compiled);
    Check(summary.has_model && summary.kind == cyxwiz::CompiledStatusKind::Ok, "the MNIST CNN summary is shown");
    Check(summary.parameters_text == "225,034", "parameters tile: " + summary.parameters_text);
    Check(summary.multiply_adds_text == "2.63 M", "multiply-adds tile: " + summary.multiply_adds_text);
    Check(summary.multiply_adds_note == "5.26 M FLOPs (2 per multiply-add)", "FLOPs line: " + summary.multiply_adds_note);
    Check(summary.weights_text == "879.0 KB" && summary.weights_note == "float32; 439.5 KB as float16",
          "weights tile: " + summary.weights_text + " / " + summary.weights_note);
    // Input 784 + every layer's output, float32, batch 64.
    const unsigned long long per_sample = (784ULL + 21632 * 2 + 5408 + 7744 * 2 + 1600 * 2 + 128 * 3 + 10) * 4;
    Check(summary.activation_bytes_per_sample == per_sample,
          "activations a sample: " + std::to_string(summary.activation_bytes_per_sample));
    Check(summary.batch == 64 && summary.activations_text == cyxwiz::FormatCompiledBytes(per_sample * 64),
          "activations at batch 64: " + summary.activations_text);
    Check(summary.rows.size() == 12 && summary.rows[0].name == "Data" && summary.rows[0].output == "[28, 28, 1]",
          "the data row comes first");
    Check(summary.advice.size() == 2, "two advice cards");
    if (summary.advice.size() == 2) {
        Check(summary.advice[0].title == "Conv2D 64 does 85 % of the work", "advice: " + summary.advice[0].title);
        Check(summary.advice[1].title == "Dense 128 holds 91 % of the parameters", "advice: " + summary.advice[1].title);
    }
    const auto* conv = &summary.rows[4];
    Check(conv->name == "Conv2D 64" &&
              conv->how == "[13, 13, 32] in, [11, 11, 64] out. Parameters 18,496. Multiply-adds 11·11·64 · 3·3·32 = 2,230,272.",
          "Conv2D 64 explained: " + conv->how);
    const std::string text = cyxwiz::ModelSummaryText(summary);
    Check(text.find("Conv2D 64") != std::string::npos && text.find("2,230,272") != std::string::npos &&
              text.find("Parameters: 225,034") != std::string::npos,
          "copy as text names the layers and totals");
    const std::string csv = cyxwiz::ModelSummaryCsv(summary);
    Check(csv.rfind("index,layer,type,output_shape,parameters,multiply_adds,activation_bytes\n", 0) == 0 &&
              csv.find("5,\"Conv2D 64\",\"Conv2D 64\",\"[11, 11, 64]\",18496,2230272,30976\n") != std::string::npos,
          "CSV rows");

    // Before the first compile, and while the parameters are counted.
    cyxwiz::ModelSummaryInputs none;
    const auto empty = cyxwiz::BuildModelSummary(none);
    Check(!empty.has_model && !empty.empty_text.empty(), "no compile: an empty state that says what to do");
    cyxwiz::ModelSummaryInputs counting;
    counting.state = cyxwiz::LiveCompileState::Compiled;
    counting.config = &compiled.config;
    counting.parameters_counting = true;
    const auto pending = cyxwiz::BuildModelSummary(counting);
    Check(pending.has_model && pending.parameters_text == "counting..." && pending.multiply_adds_text == "2.63 M",
          "multiply-adds show while the parameters are still counted");
}

}  // namespace

int main(int argc, char** argv) {
    Check(cyxwiz::test::SelectTestDeviceFromEnvironment(), "requested test device");
    cyxwiz::GraphCompilerDatasetHooks hooks;
    hooks.preprocessing_domain = [](const std::string& category) -> std::optional<cyxwiz::PreprocessingDomain> {
        if (category == "image") return cyxwiz::PreprocessingDomain::Image;
        return std::nullopt;
    };
    hooks.labels_from_structure = [](const std::string& category) -> std::optional<bool> {
        if (category == "image") return true;
        return std::nullopt;
    };
    hooks.is_dataset_registered = [](const std::string&) { return false; };
    cyxwiz::SetGraphCompilerDatasetHooks(hooks);

    const bool print = argc > 1 && std::string(argv[1]) == "--print";
    json fixture;
    std::ifstream file(CYXWIZ_MODEL_SUMMARY_FIXTURE);
    Check(file.good(), "the PyTorch fixture opens");
    if (file.good()) file >> fixture;
    const std::vector<std::pair<std::string, Builder>> cases = {
        {"mnist_cnn", MnistCnn}, {"image_mix", ImageMix}, {"residual", Residual},
        {"recurrent", Recurrent}, {"attention", Attention}};
    for (const auto& [name, build] : cases) {
        const json truth = fixture.contains(name) ? fixture.at(name) : json();
        Check(!truth.is_null() || print, name + ": PyTorch truth present");
        CheckAgainstTorch(name, build, truth, print);
    }
    CheckWords();

    if (failures > 0) {
        std::cerr << failures << " of " << checks << " checks failed\n";
        return 1;
    }
    std::cout << "model summary: " << checks << " checks passed against PyTorch\n";
    return 0;
}
