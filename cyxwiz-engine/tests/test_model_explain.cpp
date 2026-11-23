// The Model window's Explain tab (TOFIX129 D11) against PyTorch
// (generate_model_explain_fixtures.py): Engine graphs with PyTorch's weights,
// explained by core/model_explain through the SDK's ModelInterpretability.
// A CNN (layer stack): scores, saliency, Integrated Gradients and Grad-CAM at
// the last convolution (at the layer's size and resized to the input). A
// graph with branches: saliency and Integrated Gradients through the graph
// model, and Grad-CAM said not to apply. Plus the words over the maps.
#include "../src/core/compiled_node_presentation.h"
#include "../src/core/graph_compiler.h"
#include "../src/core/graph_compiler_dataset_hooks.h"
#include "../src/core/graph_node_factory.h"
#include "../src/core/model_builder.h"
#include "../src/core/model_explain.h"
#include "../src/core/model_projector.h"
#include "../src/core/spatial_batch_layout.h"
#include "../src/gui/loaders/data_loader.h"
#include "computation_truth/test_device_selection.h"

#include <cyxwiz/model_interpretability.h>
#include <arrow/api.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
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

void CheckClose(const std::vector<float>& actual, const std::vector<double>& expected, double tolerance,
                const std::string& what) {
    Check(actual.size() == expected.size(),
          what + ": " + std::to_string(actual.size()) + " values, PyTorch " + std::to_string(expected.size()));
    double worst = 0.0;
    size_t at = 0;
    for (size_t i = 0; i < actual.size() && i < expected.size(); ++i) {
        const double error = std::abs(actual[i] - expected[i]) / std::max(1.0, std::abs(expected[i]));
        if (error > worst) {
            worst = error;
            at = i;
        }
    }
    Check(worst <= tolerance, what + ": worst difference " + std::to_string(worst) + " at " + std::to_string(at));
}

std::filesystem::path FixturePath(const char* argv0) {
    const auto beside = std::filesystem::path(argv0).parent_path() / "computation_truth_fixtures" /
                        "model_explain_pytorch.json";
    if (std::filesystem::exists(beside)) return beside;
    return CYXWIZ_MODEL_EXPLAIN_FIXTURE;
}

cyxwiz::Tensor ReadTensor(const json& value) {
    const auto shape = value.at("shape").get<std::vector<size_t>>();
    const auto values = value.at("values").get<std::vector<float>>();
    return cyxwiz::Tensor(shape, values.data());
}

struct Graph {
    std::vector<gui::MLNode> nodes;
    std::vector<gui::NodeLink> links;
    int next_node = 1, next_pin = 1, next_link = 1;

    int Add(gui::NodeType type, const std::string& name, std::map<std::string, std::string> parameters = {}) {
        auto node = gui::CreateGraphNode(type, name, next_node, next_pin);
        for (auto& [key, value] : parameters) node.parameters[key] = value;
        nodes.push_back(node);
        return node.id;
    }
    gui::MLNode& Node(int id) {
        for (auto& node : nodes)
            if (node.id == id) return node;
        std::exit(1);
    }
    int PinOf(int node, bool input, gui::PinType type, int nth) {
        const auto& pins = input ? Node(node).inputs : Node(node).outputs;
        for (const auto& pin : pins)
            if (pin.type == type && nth-- == 0) return pin.id;
        std::cerr << Node(node).name << ": no pin\n";
        std::exit(1);
    }
    void Connect(int from, int to, gui::PinType type = gui::PinType::Tensor, int to_nth = 0) {
        gui::NodeLink link;
        link.id = next_link++;
        link.from_node = from;
        link.from_pin = PinOf(from, false, type, 0);
        link.to_node = to;
        link.to_pin = PinOf(to, true, type, to_nth);
        links.push_back(link);
    }
    int Data(const std::string& shape) {
        const int data = Add(gui::NodeType::DataInput, "Data",
                             {{"dataset_name", "explain_rows"}, {"file_category", "tabular"}, {"shape", shape},
                              {"label_column", "label"}, {"data_loaded", "true"}});
        const int split = Add(gui::NodeType::DataSplit, "Split");
        const int loader = Add(gui::NodeType::DataLoader, "Loader", {{"batch_size", "8"}});
        Connect(data, split, gui::PinType::Dataset);
        Connect(split, loader, gui::PinType::Dataset);
        return loader;
    }
    void Finish(int loader, int last) {
        const int loss = Add(gui::NodeType::CrossEntropyLoss, "Loss");
        const int adam = Add(gui::NodeType::Adam, "Adam");
        Connect(last, loss);
        Connect(loader, loss, gui::PinType::Labels);
        Connect(loss, adam, gui::PinType::Loss);
    }
};

Graph Cnn() {
    Graph g;
    const int loader = g.Data("[8, 8, 1]");
    int last = loader;
    for (int layer : {g.Add(gui::NodeType::Conv2D, "Conv 4", {{"filters", "4"}, {"kernel_size", "3"}, {"padding", "same"}}),
                      g.Add(gui::NodeType::ReLU, "ReLU 1"),
                      g.Add(gui::NodeType::Conv2D, "Conv 6", {{"filters", "6"}, {"kernel_size", "3"}, {"padding", "valid"}}),
                      g.Add(gui::NodeType::ReLU, "ReLU 2"), g.Add(gui::NodeType::GlobalAvgPool, "Pool"),
                      g.Add(gui::NodeType::Dense, "Head", {{"units", "3"}})}) {
        g.Connect(last, layer);
        last = layer;
    }
    g.Finish(loader, last);
    return g;
}

Graph Residual() {
    Graph g;
    const int loader = g.Data("[6]");
    const int a = g.Add(gui::NodeType::Dense, "Dense a", {{"units", "5"}});
    const int b = g.Add(gui::NodeType::Dense, "Dense b", {{"units", "5"}});
    const int add = g.Add(gui::NodeType::Add, "Add");
    const int relu = g.Add(gui::NodeType::ReLU, "ReLU");
    const int head = g.Add(gui::NodeType::Dense, "Head", {{"units", "3"}});
    g.Connect(loader, a);
    g.Connect(loader, b);
    g.Connect(a, add, gui::PinType::Tensor, 0);
    g.Connect(b, add, gui::PinType::Tensor, 1);
    g.Connect(add, relu);
    g.Connect(relu, head);
    g.Finish(loader, head);
    return g;
}

struct Built {
    cyxwiz::TrainingConfiguration config;
    std::unique_ptr<cyxwiz::IExecutableModel> model;
};

Built BuildWithWeights(const std::string& name, const Graph& graph, const json& c) {
    Built built;
    built.config = cyxwiz::GraphCompiler{}.Compile(graph.nodes, graph.links, true);
    for (const auto& issue : built.config.issues)
        if (issue.level == cyxwiz::IssueLevel::Error) std::cerr << "  issue: " << issue.node_name << ": " << issue.message << "\n";
    Check(built.config.is_valid, name + ": compiles");
    auto result = cyxwiz::BuildExecutableFromConfig(built.config);
    Check(result.ok(), name + ": builds: " + result.error_message);
    if (!result.ok()) return built;
    built.model = std::move(result.model);
    std::map<std::string, cyxwiz::Tensor> parameters;
    for (const auto& [layer_name, layer] : c.at("weights").items()) {
        long long index = -1;
        for (size_t i = 0; i < built.config.layers.size(); ++i)
            if (built.config.layers[i].name == layer_name) index = static_cast<long long>(i);
        Check(index >= 0, name + ": a layer named " + layer_name);
        for (const auto& [key, value] : layer.items())
            parameters["layer" + std::to_string(index) + "." + key] = ReadTensor(value);
    }
    const auto before = built.model->GetParameters();
    for (const auto& [key, tensor] : parameters)
        Check(before.count(key) == 1 && before.at(key).Shape() == tensor.Shape(), name + ": parameter " + key + " fits");
    built.model->SetParameters(parameters);
    return built;
}

std::vector<double> Absolute(std::vector<double> values) {
    for (auto& v : values) v = std::abs(v);
    return values;
}

void CheckCnn(const json& c) {
    Built built = BuildWithWeights("cnn", Cnn(), c);
    if (!built.model) return;
    cyxwiz::ExplainRequest request;
    request.sample = c.at("input").get<std::vector<float>>();
    request.ig_steps = 64;
    const auto out = cyxwiz::ExplainSample(*built.model, built.config, request);
    Check(out.ok, "cnn: explained: " + out.error);
    if (!out.ok) return;
    Check(out.predicted == c.at("target").get<int>() && out.target == out.predicted,
          "cnn: the predicted class is PyTorch's: " + std::to_string(out.predicted));
    Check(out.height == 8 && out.width == 8 && out.channels == 1, "cnn: maps are 8 x 8");
    CheckClose(out.saliency, Absolute(c.at("saliency").get<std::vector<double>>()), 1e-4, "cnn saliency");
    CheckClose(out.integrated, c.at("integrated").get<std::vector<double>>(), 1e-4, "cnn Integrated Gradients");
    Check(std::abs(out.score - c.at("score").get<double>()) < 1e-4 &&
              std::abs(out.baseline_score - c.at("baseline_score").get<double>()) < 1e-4,
          "cnn: scores at the input and the baseline");
    CheckClose(out.grad_cam, c.at("grad_cam").get<std::vector<double>>(), 1e-4, "cnn Grad-CAM resized to 8 x 8");
    Check(out.grad_cam_layer == "Conv 6", "cnn: Grad-CAM at the last convolution: " + out.grad_cam_layer);

    // The SDK's map at the layer's own size.
    auto* stack = built.model->AsSequentialModel();
    Check(stack != nullptr, "cnn: a layer stack");
    if (stack) {
        const auto& input = request.sample;
        const cyxwiz::Tensor rows({1, input.size()}, input.data());
        const cyxwiz::Tensor spatial = cyxwiz::SpatialBatchFromRows(rows, {8, 8, 1});
        int layer = -1;
        cyxwiz::Tensor current = spatial;
        for (size_t i = 0; i < stack->Size(); ++i) {
            current = stack->GetModule(i)->Forward(current);
            if (current.Shape().size() == 4 && stack->GetModule(i)->GetName().find("Conv") != std::string::npos)
                layer = static_cast<int>(i);
        }
        const auto cam = cyxwiz::ModelInterpretability::ComputeGradCAM(*stack, spatial, layer, -1);
        Check(cam.success && cam.layer_shape == std::vector<size_t>{6, 6}, "cnn: Grad-CAM map 6 x 6: " + cam.error_message);
        CheckClose(cam.layer_heatmap, c.at("grad_cam_layer").get<std::vector<double>>(), 1e-4, "cnn Grad-CAM at 6 x 6");
        Check(!cam.heatmap.empty() && std::abs(*std::max_element(cam.heatmap.begin(), cam.heatmap.end()) - 1.0f) < 1e-6f,
              "cnn: the 0-1 map reaches 1: " + (cam.heatmap.empty() ? std::string("empty") : std::to_string(*std::max_element(cam.heatmap.begin(), cam.heatmap.end())) + " of " + std::to_string(cam.heatmap.size())));
    }

    const auto words = cyxwiz::DescribeExplanation(out, 12, out.predicted, {"zero", "one", "two"});
    Check(words.headline.find("Row 12 is a ") == 0 && !words.wrong, "words for a right answer: " + words.headline);
    const auto wrong = cyxwiz::DescribeExplanation(out, 12, (out.predicted + 1) % 3, {"zero", "one", "two"});
    Check(wrong.wrong && wrong.headline.find(", but the model says ") != std::string::npos,
          "words for a mistake: " + wrong.headline);
    Check(words.completeness.find("They add up to ") == 0, "completeness: " + words.completeness);
}

void CheckResidual(const json& c) {
    Built built = BuildWithWeights("residual", Residual(), c);
    if (!built.model) return;
    Check(built.model->AsSequentialModel() == nullptr, "residual: a graph model");
    cyxwiz::ExplainRequest request;
    request.sample = c.at("input").get<std::vector<float>>();
    const auto out = cyxwiz::ExplainSample(*built.model, built.config, request);
    Check(out.ok, "residual: explained: " + out.error);
    if (!out.ok) return;
    Check(out.predicted == c.at("target").get<int>(), "residual: the predicted class is PyTorch's");
    CheckClose(out.saliency, Absolute(c.at("saliency").get<std::vector<double>>()), 1e-4, "residual saliency");
    CheckClose(out.integrated, c.at("integrated").get<std::vector<double>>(), 1e-4, "residual Integrated Gradients");
    Check(out.grad_cam.empty() && out.grad_cam_note.find("image-shaped") != std::string::npos,
          "residual: Grad-CAM said not to apply: " + out.grad_cam_note);

    // Another class than the predicted one.
    request.target_class = (out.predicted + 1) % 3;
    const auto other = cyxwiz::ExplainSample(*built.model, built.config, request);
    Check(other.ok && other.target == request.target_class && other.predicted == out.predicted,
          "residual: explains the score of another class");
}

void CheckBaseline() {
    cyxwiz::TrainingConfiguration config;
    config.input_shape = {2, 2, 3};
    config.preprocessing.has_normalization = true;
    config.preprocessing.norm_channel_mean = {0.5f, 0.4f, 0.3f};
    config.preprocessing.norm_channel_std = {0.25f, 0.2f, 0.1f};
    const auto baseline = cyxwiz::ExplainBaseline(config, 12);
    Check(baseline.size() == 12 && std::abs(baseline[0] + 2.0f) < 1e-6f && std::abs(baseline[1] + 2.0f) < 1e-6f &&
              std::abs(baseline[2] + 3.0f) < 1e-6f && std::abs(baseline[5] + 3.0f) < 1e-6f,
          "the baseline is a black image after a per-channel Normalize");
    config.preprocessing.has_normalization = false;
    Check(cyxwiz::ExplainBaseline(config, 4) == std::vector<float>(4, 0.0f), "without Normalize the baseline is 0");
}

// Projector on a test with two clear groups in a layer's 3 features.
void CheckProjector() {
    cyxwiz::TestingMetrics test;
    cyxwiz::TestingMetrics::LayerFeatures features;
    features.layer = 4;
    features.width = 3;
    for (int i = 0; i < 40; ++i) {
        const int group = i % 2;
        const float spread = 0.05f * static_cast<float>(i % 5);
        features.values.insert(features.values.end(), {group ? 5.0f + spread : -5.0f - spread, spread, -spread});
        test.ground_truth.push_back(group);
        test.predictions.push_back(i == 7 ? 0 : group);  // one mistake
        test.confidences.push_back(0.9f);
        test.row_ids.push_back(100 + i);
    }
    test.layer_features.push_back(features);
    const auto layers = cyxwiz::ProjectorLayers(test);
    Check(layers.size() == 1 && layers[0].layer == 4 && layers[0].width == 3, "projector: the layer recorded for every row");
    cyxwiz::ProjectorRequest request;
    request.layer = 4;
    const auto pca = cyxwiz::ProjectTestRows(test, request);
    Check(pca.ok && pca.points.size() == 40 && pca.x_label.rfind("PC 1 ", 0) == 0, "projector: PCA of 40 rows: " + pca.error);
    bool apart = true;
    for (const auto& p : pca.points) apart = apart && (p.truth == 1 ? p.x : -p.x) * (pca.points[1].x > 0 ? 1 : -1) > 0;
    Check(apart, "projector: the two groups fall on either side of PC 1");
    const auto wrong = std::count_if(pca.points.begin(), pca.points.end(), [](const cyxwiz::ProjectorPoint& p) { return p.wrong; });
    Check(wrong == 1 && pca.points[7].row == 107, "projector: the mistake is marked, rows named");
    request.max_rows = 10;
    const auto thinned = cyxwiz::ProjectTestRows(test, request);
    Check(thinned.ok && thinned.points.size() == 10 && thinned.note.find("every 4th") != std::string::npos,
          "projector: more rows than drawn are thinned evenly: " + thinned.note);
    request.max_rows = 2000;
    request.method = cyxwiz::ProjectorMethod::UMAP;
    const auto umap = cyxwiz::ProjectTestRows(test, request);
    Check(umap.ok && umap.points.size() == 40 && umap.x_label == "UMAP 1", "projector: UMAP: " + umap.error);
    const auto table = cyxwiz::ProjectorFeaturesTable(test, 4);
    Check(table && table->num_rows() == 40 && table->num_columns() == 6 &&
              table->schema()->field(0)->name() == "feature_1" && table->schema()->field(5)->name() == "row",
          "projector: the features table for the Projection tab");
    request.layer = 9;
    Check(!cyxwiz::ProjectTestRows(test, request).ok, "projector: a layer without features is refused");
    test.predictions.push_back(0);  // a sample without features: the layer is not complete
    Check(cyxwiz::ProjectorLayers(test).empty(), "projector: a layer short of rows is not offered");
}

}  // namespace

int main(int, char** argv) {
    Check(cyxwiz::test::SelectTestDeviceFromEnvironment(), "requested test device");
    cyxwiz::GraphCompilerDatasetHooks hooks;
    hooks.is_dataset_registered = [](const std::string&) { return false; };
    cyxwiz::SetGraphCompilerDatasetHooks(hooks);

    std::ifstream in(FixturePath(argv[0]));
    Check(in.good(), "the PyTorch fixture opens");
    if (in.good()) {
        const json fixture = json::parse(in);
        CheckCnn(fixture.at("cases").at("cnn"));
        CheckResidual(fixture.at("cases").at("residual"));
    }
    CheckBaseline();
    CheckProjector();
    if (failures > 0) {
        std::cerr << failures << " of " << checks << " checks failed\n";
        return 1;
    }
    std::cout << "model explain: " << checks << " checks passed against PyTorch\n";
    return 0;
}
