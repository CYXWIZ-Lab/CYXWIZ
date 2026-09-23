void RunGraphCompileTaskTests();
#include "../src/core/graph_compiler.h"
#include "../src/gui/loaders/data_loader.h"

#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

namespace cyxwiz::loaders {

DataLoader* GetByCategory(FileCategory) {
    return nullptr;
}

DataLoader* GetByRegisteredDataset(const std::string&) {
    return nullptr;
}

FileCategory FileCategoryFromString(const std::string&) {
    return FileCategory::Tabular;
}

} // namespace cyxwiz::loaders

namespace {

void Check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << "\n";
        std::exit(1);
    }
}

gui::NodePin Pin(int id,
                 gui::PinType type,
                 const std::string& name,
                 bool is_input,
                 bool required = true) {
    gui::NodePin pin;
    pin.id = id;
    pin.type = type;
    pin.name = name;
    pin.is_input = is_input;
    pin.is_required = required;
    return pin;
}

gui::MLNode Node(int id,
                 gui::NodeType type,
                 const std::string& name,
                 std::vector<gui::NodePin> inputs,
                 std::vector<gui::NodePin> outputs) {
    gui::MLNode node;
    node.id = id;
    node.type = type;
    node.name = name;
    node.inputs = std::move(inputs);
    node.outputs = std::move(outputs);
    return node;
}

gui::NodeLink Link(int id, int from_node, int from_pin, int to_node, int to_pin) {
    gui::NodeLink link;
    link.id = id;
    link.from_node = from_node;
    link.from_pin = from_pin;
    link.to_node = to_node;
    link.to_pin = to_pin;
    return link;
}

bool HasErrorText(const cyxwiz::TrainingConfiguration& config,
                  const std::string& text) {
    for (const auto& issue : config.issues) {
        if (issue.level == cyxwiz::IssueLevel::Error &&
            issue.message.find(text) != std::string::npos) {
            return true;
        }
    }
    return false;
}

} // namespace

int main() {
    auto data = Node(1,
                     gui::NodeType::DataInput,
                     "Token features",
                     {},
                     {Pin(101, gui::PinType::Tensor, "Data", false),
                      Pin(102, gui::PinType::Labels, "Labels", false)});
    data.parameters["dataset_name"] = "causal_lm_shape_dataset";
    data.parameters["data_loaded"] = "true";
    data.parameters["file_category"] = "tabular";
    data.parameters["shape"] = "[4]";
    data.parameters["create_causal_lm_targets"] = "true";
    data.parameters["max_sequence_length"] = "7";

    auto decoder = Node(2,
                        gui::NodeType::TransformerDecoder,
                        "Decoder",
                        {Pin(201, gui::PinType::Tensor, "Input", true)},
                        {Pin(202, gui::PinType::Tensor, "Output", false)});
    decoder.parameters["d_model"] = "4";
    decoder.parameters["num_heads"] = "2";
    decoder.parameters["dim_feedforward"] = "8";
    decoder.parameters["dropout"] = "0";

    auto head = Node(3,
                     gui::NodeType::TimeDistributed,
                     "Token Head",
                     {Pin(301, gui::PinType::Tensor, "Input", true)},
                     {Pin(302, gui::PinType::Tensor, "Output", false)});
    head.parameters["units"] = "6";

    auto output = Node(4,
                       gui::NodeType::Output,
                       "Sequence Logits",
                       {Pin(401, gui::PinType::Tensor, "Input", true)},
                       {});
    output.parameters["num_classes"] = "6";

    auto loss = Node(5,
                     gui::NodeType::CrossEntropyLoss,
                     "Token Loss",
                     {Pin(501, gui::PinType::Tensor, "Predictions", true),
                      Pin(502, gui::PinType::Labels, "Targets", true)},
                     {Pin(503, gui::PinType::Tensor, "Loss", false, false)});

    auto optimizer = Node(6,
                          gui::NodeType::Adam,
                          "Adam",
                          {},
                          {Pin(601, gui::PinType::Optimizer, "Optimizer", false)});

    const std::vector<gui::MLNode> nodes = {
        data, decoder, head, output, loss, optimizer};
    const std::vector<gui::NodeLink> links = {
        Link(1, 1, 101, 2, 201),
        Link(2, 2, 202, 3, 301),
        Link(3, 3, 302, 4, 401),
        Link(4, 3, 302, 5, 501),
        Link(5, 1, 102, 5, 502),
        Link(6, 6, 601, 5, 503),
    };

    cyxwiz::GraphCompiler compiler;
    const cyxwiz::TrainingConfiguration config =
        compiler.Compile(nodes, links, true);

    Check(config.is_valid,
          "causal LM graph should compile without shape errors: " +
              config.error_message);
    Check(config.input_shape == std::vector<size_t>({7, 4}),
          "causal LM DatasetInput should preserve [seq, features] shape");
    Check(config.input_size == 4,
          "causal LM input_size should be per-token feature width");
    Check(config.layers.size() == 2,
          "causal LM graph should compile decoder plus token head");
    Check(config.layers[0].input_shape == std::vector<size_t>({7, 4}),
          "decoder should receive sequence-shaped input");
    Check(config.layers[0].output_shape == std::vector<size_t>({7, 4}),
          "decoder should preserve sequence-shaped output");
    Check(config.layers[1].input_shape == std::vector<size_t>({7, 4}),
          "TimeDistributed should receive sequence-shaped input");
    Check(config.layers[1].output_shape == std::vector<size_t>({7, 6}),
          "TimeDistributed should produce [seq, vocab] output shape");
    Check(!HasErrorText(config, "TimeDistributed requires sequence input shape"),
          "causal LM graph should not report TimeDistributed shape error");

    Check(config.model_seed == -1, "old graph without model seed must remain unset");
    auto seeded_nodes = nodes;
    auto seeded_links = links;
    auto loader = Node(90, gui::NodeType::DataLoader, "Seeded loader",
        {Pin(901, gui::PinType::Tensor, "Input", true)},
        {Pin(902, gui::PinType::Tensor, "Data", false)});
    loader.parameters["seed"] = "97";
    seeded_nodes.push_back(loader);
    seeded_links[0] = Link(1, 1, 101, 90, 901);
    seeded_links.push_back(Link(90, 90, 902, 2, 201));
    for (const auto& value : {"-1", "0", "52", "2147483647"}) {
        seeded_nodes.back().parameters["model_seed"] = value;
        // Graph serialization preserves parameter strings, including the unset sentinel.
        const nlohmann::json saved = seeded_nodes.back().parameters;
        seeded_nodes.back().parameters = nlohmann::json::parse(saved.dump()).get<decltype(loader.parameters)>();
        const auto seeded = compiler.Compile(seeded_nodes, seeded_links, true);
        Check(seeded.is_valid, "seeded graph should compile: " + seeded.error_message);
        Check(seeded.model_seed == std::stoi(value) && seeded.dataloader_seed == 97,
              "model and data seeds must propagate independently");
    }
    for (const auto& value : {"-2", "2147483648", "1x", "", "1.5"}) {
        seeded_nodes.back().parameters["model_seed"] = value;
        const auto invalid = compiler.Compile(seeded_nodes, seeded_links, true);
        Check(!invalid.is_valid && HasErrorText(invalid, "Model RNG seed"), "invalid model seed must fail compilation");
    }
    // Preview settings participate in real graph compilation and parameter persistence.
    seeded_nodes.back().parameters["model_seed"] = "52";
    seeded_nodes.back().parameters["generation_preview_enabled"] = "false";
    seeded_nodes.back().parameters["generation_preview_every_epochs"] = "invalid";
    Check(compiler.Compile(seeded_nodes, seeded_links, true).is_valid,
          "disabled previews must preserve legacy graph behavior");
    seeded_nodes.back().parameters["generation_preview_enabled"] = "true";
    seeded_nodes.back().parameters["generation_preview_prompts"] = "one\ntwo";
    const auto bad_preview = compiler.Compile(seeded_nodes, seeded_links, true);
    Check(!bad_preview.is_valid && HasErrorText(bad_preview, "Generation preview:"),
          "invalid enabled preview cadence must fail graph compilation");
    seeded_nodes.back().parameters["generation_preview_every_epochs"] = "20";
    const nlohmann::json saved_preview = seeded_nodes.back().parameters;
    seeded_nodes.back().parameters = nlohmann::json::parse(saved_preview.dump()).get<decltype(loader.parameters)>();
    Check(seeded_nodes.back().parameters.at("generation_preview_prompts") == "one\ntwo",
          "fixed multiline prompts must survive graph parameter serialization");
    const auto noncausal = compiler.Compile(seeded_nodes, seeded_links, true);
    Check(!noncausal.is_valid && HasErrorText(noncausal, "causal next-token targets"),
          "previews must reject a noncausal training contract");
    seeded_nodes.back().parameters["create_causal_lm_targets"] = "true";
    seeded_nodes.back().parameters["max_sequence_length"] = "7";
    const auto missing_artifact = compiler.Compile(seeded_nodes, seeded_links, true);
    Check(!missing_artifact.is_valid && HasErrorText(missing_artifact, "vocabulary metadata"),
          "enabled previews must reject a dataset without the training vocabulary artifact");
    RunGraphCompileTaskTests();
    std::cout << "Graph compiler causal LM shape test passed\n";
    return 0;
}
