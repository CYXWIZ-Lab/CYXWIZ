// Neural Architecture Search's presentation (TOFIX178 T4 H2): the input and
// output read from the compiled graph, the head named in the note, the score
// summary, the model's layers in order, what a change did, the search's
// ranked versions and its stop, and the saved defaults.

#include "core/nas_presentation.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <set>
#include <string>
#include <vector>

using namespace cyxwiz;
namespace np = cyxwiz::nas_presentation;
using gui::MLNode;
using gui::NodeLink;
using gui::NodeType;

namespace {

int g_failures = 0;

void Check(bool ok, const std::string& what) {
    if (!ok) {
        std::printf("FAIL: %s\n", what.c_str());
        ++g_failures;
    }
}

MLNode Node(int id, NodeType type, const char* name, std::map<std::string, std::string> parameters = {}) {
    MLNode node;
    node.id = id;
    node.type = type;
    node.name = name;
    node.parameters = std::move(parameters);
    gui::NodePin in;
    in.id = id * 10;
    in.is_input = true;
    gui::NodePin out;
    out.id = id * 10 + 1;
    node.inputs.push_back(in);
    node.outputs.push_back(out);
    return node;
}

NodeLink Link(int id, int from, int to) { return {id, from, from * 10 + 1, to, to * 10}; }

// The chain as the canvas holds a training graph, for the scorer's compile:
// the Data Input gives Data and Labels (no input pin), the model's last layer
// (or the Output node's place) feeds a CrossEntropy loss with the labels, and
// Adam takes the loss.
std::pair<std::vector<MLNode>, std::vector<NodeLink>> Trainable(std::vector<MLNode> nodes, std::vector<NodeLink> links) {
    std::set<int> outputs;
    for (const auto& n : nodes)
        if (n.type == NodeType::Output) outputs.insert(n.id);
    int head = -1;
    for (const auto& l : links)
        if (outputs.count(l.to_node)) head = l.from_node;
    nodes.erase(std::remove_if(nodes.begin(), nodes.end(), [&](const MLNode& n) { return outputs.count(n.id) > 0; }),
                nodes.end());
    links.erase(std::remove_if(links.begin(), links.end(), [&](const NodeLink& l) { return outputs.count(l.to_node) > 0; }),
                links.end());
    if (head < 0) {  // no Output node: the layer nothing reads from
        std::set<int> read;
        for (const auto& l : links) read.insert(l.from_node);
        for (const auto& n : nodes)
            if (!read.count(n.id) && n.type != NodeType::DataInput) head = n.id;
    }
    gui::NodePin labels;
    for (auto& n : nodes) {
        if (n.type != NodeType::DataInput) continue;
        n.inputs.clear();
        labels.id = n.id * 10 + 2;
        labels.type = gui::PinType::Labels;
        n.outputs.push_back(labels);
        if (!n.parameters.count("label_column")) n.parameters["label_column"] = "label";
    }
    MLNode loss;
    loss.id = 90;
    loss.type = NodeType::CrossEntropyLoss;
    loss.name = "CrossEntropy";
    gui::NodePin predictions, targets, loss_out;
    predictions.id = 900; predictions.is_input = true;
    targets.id = 902; targets.is_input = true; targets.type = gui::PinType::Labels;
    loss_out.id = 901; loss_out.type = gui::PinType::Loss;
    loss.inputs = {predictions, targets};
    loss.outputs = {loss_out};
    MLNode adam;
    adam.id = 91;
    adam.type = NodeType::Adam;
    adam.name = "Adam";
    gui::NodePin loss_in;
    loss_in.id = 910; loss_in.is_input = true; loss_in.type = gui::PinType::Loss;
    adam.inputs = {loss_in};
    nodes.push_back(loss);
    nodes.push_back(adam);
    links.push_back({900, head, head * 10 + 1, 90, 900});
    links.push_back({901, 1, 12, 90, 902});
    links.push_back({902, 90, 901, 91, 910});
    return {nodes, links};
}

}  // namespace

int main() {
    // mnist_mlp: Input -> Dense 1 (128) -> ReLU 1 -> Dense 2 (64) -> ReLU 2 ->
    // Dense 3 (10). The nodes are listed out of order: the layers follow the
    // links.
    std::vector<MLNode> nodes = {
        Node(6, NodeType::Dense, "Dense 3", {{"units", "10"}}),
        Node(1, NodeType::DataInput, "Input", {{"shape", "[784]"}}),
        Node(2, NodeType::Dense, "Dense 1", {{"units", "128"}}),
        Node(3, NodeType::ReLU, "ReLU 1"),
        Node(4, NodeType::Dense, "Dense 2", {{"units", "64"}}),
        Node(5, NodeType::ReLU, "ReLU 2"),
    };
    std::vector<NodeLink> links;
    for (int i = 1; i < 6; ++i) links.push_back(Link(i, i, i + 1));

    // Input and output from the compile.
    TrainingConfiguration compiled;
    compiled.input_shape = {784};
    compiled.output_size = 10;
    const auto shape = np::ReadShape(compiled);
    Check(shape.ok, "features read from the compile");
    Check(shape.text == "784 features in, 10 classes out", "shape text: " + shape.text);
    Check(shape.task == "classification", "a classifier");
    TrainingConfiguration images;
    images.input_shape = {28, 28, 1};
    images.output_size = 10;
    const auto image_shape = np::ReadShape(images);
    Check(image_shape.task == "image_classification", "images");
    Check(image_shape.text == "28 x 28 x 1 images in, 10 classes out", "image text: " + image_shape.text);
    TrainingConfiguration numbers;
    numbers.input_shape = {8};
    numbers.output_size = 1;
    numbers.loss_type = NodeType::MSELoss;
    const auto number_shape = np::ReadShape(numbers);
    Check(number_shape.task == "regression" && number_shape.text == "8 features in, 1 value out",
          "regression: " + number_shape.text);
    const auto unloaded = np::ReadShape(TrainingConfiguration{});
    Check(!unloaded.ok && unloaded.why.find("load the graph's data") != std::string::npos,
          "no input size: the data is not loaded: " + unloaded.why);

    Check(np::HeadNote(nodes, links, shape) == "The head (Dense 3, the 10 outputs) is never removed, resized or swapped.",
          "head note: " + np::HeadNote(nodes, links, shape));

    // The score card.
    const auto score = NASEvaluator::ScoreArchitecture(Trainable(nodes, links).first, Trainable(nodes, links).second);
    Check(np::ScoreSummary(score, nodes, links) ==
              "Parameters 109,386 \xC2\xB7 model layers: 3x Dense, 2x ReLU \xC2\xB7 no skip connections",
          "summary: " + np::ScoreSummary(score, nodes, links));
    Check(np::ScoreParts(score).size() == 5 && np::ScoreParts(score)[0].label == "Size", "five score parts");
    Check(np::LayersText(nodes, links) == "Dense 128, ReLU, Dense 64, ReLU, Dense 10", "layers: " + np::LayersText(nodes, links));
    Check(np::Mutations().size() == 6, "six change buttons");

    // What a change did.
    {
        auto added = nodes;
        auto added_links = links;
        added.push_back(Node(7, NodeType::BatchNorm, "BatchNorm"));
        added_links.back() = Link(5, 5, 7);
        added_links.push_back(Link(6, 7, 6));
        Check(np::DescribeChange(nodes, links, added, added_links) == "Added BatchNorm before Dense 3",
              "add: " + np::DescribeChange(nodes, links, added, added_links));
    }
    {
        std::vector<MLNode> removed;
        for (const auto& n : nodes)
            if (n.id != 4) removed.push_back(n);
        std::vector<NodeLink> removed_links = {Link(1, 1, 2), Link(2, 2, 3), Link(3, 3, 5), Link(5, 5, 6)};
        Check(np::DescribeChange(nodes, links, removed, removed_links) == "Removed Dense 2 (ReLU 1 feeds ReLU 2)",
              "remove: " + np::DescribeChange(nodes, links, removed, removed_links));
    }
    {
        auto resized = nodes;
        resized[2].parameters["units"] = "192";
        resized[3].type = NodeType::GELU;
        Check(np::DescribeChange(nodes, links, resized, links) == "Dense 1 units 128 to 192; ReLU 1 became GELU",
              "resize and swap: " + np::DescribeChange(nodes, links, resized, links));
        Check(np::DescribeChange(nodes, links, nodes, links).empty(), "no change, no text");
        // Layers with one name (mnist_mlp names all three "Dense") get their number.
        auto same_names = nodes;
        for (auto& n : same_names)
            if (n.type == NodeType::Dense) n.name = "Dense";
        auto wider = same_names;
        wider[4].parameters["units"] = "96";
        Check(np::DescribeChange(same_names, links, wider, links) == "Dense #4 units 64 to 96",
              "a repeated name gets its number: " + np::DescribeChange(same_names, links, wider, links));
        Check(np::HeadNote(same_names, links, shape).find("(Dense #6, the 10 outputs)") != std::string::npos,
              "head with its number: " + np::HeadNote(same_names, links, shape));
    }

    // The search keeps the best different versions, best first, and stops
    // when asked.
    {
        NASSearchConfig config;
        config.population_size = 12;
        config.generations = 6;
        int rounds_seen = 0;
        const auto result = NASEvaluator::EvolveArchitecture(Trainable(nodes, links).first, Trainable(nodes, links).second, config,
                                                             [&](int, const ArchitectureScore&) { return ++rounds_seen < 2; });
        Check(result.success && result.stopped && result.total_generations == 2, "stopped after round 2");
        Check(!result.ranked.empty() && result.ranked.size() <= static_cast<size_t>(kNASRanked), "ranked versions kept");
        std::set<std::string> layers;
        bool ordered = true;
        for (size_t i = 0; i < result.ranked.size(); ++i) {
            layers.insert(np::LayersText(result.ranked[i].nodes, result.ranked[i].links));
            if (i > 0 && result.ranked[i].score.overall_score > result.ranked[i - 1].score.overall_score) ordered = false;
        }
        Check(layers.size() == result.ranked.size(), "every ranked version is different");
        Check(ordered, "best first");
    }

    // Trained results.
    TrialScore trained;
    trained.has_validation = true;
    trained.val_loss = 0.079f;
    trained.val_accuracy = 0.977f;
    Check(np::LossText(trained) == "0.079" && np::AccuracyText(trained) == "97.7 %", "trained text");
    trained.regression = true;
    trained.val_mae = 0.5f;
    Check(np::AccuracyText(trained) == "MAE 0.5", "regression shows MAE");

    // The saved defaults: set on a search, read back, saved and reloaded,
    // clamped.
    {
        core::ArchitectureSearchDefaults defaults;
        defaults.versions = 30;
        defaults.rounds = 5;
        defaults.kept = 3;
        defaults.changed_percent = 40;
        defaults.mixed_percent = 20;
        defaults.train_best = 4;
        NASSearchConfig config;
        int train_best = 0;
        np::ApplyDefaults(config, train_best, defaults);
        Check(config.population_size == 30 && config.generations == 5 && config.elite_count == 3 && train_best == 4,
              "the defaults set the search");
        Check(np::DefaultsOf(config, train_best) == defaults, "and read back the same");

        auto& engine_config = core::EngineConfig::Instance();
        const auto path = std::filesystem::temp_directory_path() / "cyxwiz_architecture_defaults_test.json";
        engine_config.SetArchitectureSearchDefaults(defaults);
        Check(engine_config.Save(path), "the settings save");
        engine_config.SetArchitectureSearchDefaults({});
        Check(engine_config.Load(path), "the settings load");
        Check(engine_config.GetArchitectureSearchDefaults() == defaults, "the defaults survive a save and load");
        std::error_code error;
        std::filesystem::remove(path, error);

        core::ArchitectureSearchDefaults wrong;
        wrong.versions = 1;
        wrong.kept = 9;
        wrong.changed_percent = 150;
        wrong.train_best = 50;
        engine_config.SetArchitectureSearchDefaults(wrong);
        const auto clamped = engine_config.GetArchitectureSearchDefaults();
        Check(clamped.versions == 2 && clamped.kept == 1 && clamped.changed_percent == 100 &&
                  clamped.train_best == core::kArchitectureSearchMaxTrain,
              "out-of-range defaults are clamped");
    }

    if (g_failures) {
        std::printf("%d NAS presentation checks failed\n", g_failures);
        return EXIT_FAILURE;
    }
    std::printf("NAS presentation passed\n");
    return EXIT_SUCCESS;
}
