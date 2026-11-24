#include "../src/core/graph_node_factory.h"
#include "../src/core/node_metadata_registry.h"
#include "../src/core/pipeline_runtime_capabilities.h"
#include "../src/gui/patterns/pattern_library.h"
#include "../src/gui/node_import_guardrails.h"
#include "../src/gui/node_type_import_registry.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstdlib>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

void Check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << "\n";
        std::exit(1);
    }
}

std::filesystem::path WritePatternWithNode(const std::string& id,
                                           const std::string& type,
                                           const std::string& name,
                                           const std::string& params_json) {
    const auto dir = std::filesystem::temp_directory_path() / "cyxwiz_pattern_guard";
    std::filesystem::create_directories(dir);

    const auto path = dir / (id + ".cyxgraph");
    std::ofstream out(path, std::ios::binary);
    Check(out.is_open(), "could not write pattern file");
    out << "{\n"
        << "  \"id\": \"" << id << "\",\n"
        << "  \"name\": \"" << id << "\",\n"
        << "  \"category\": \"Basic\",\n"
        << "  \"template\": {\n"
        << "    \"nodes\": [\n"
        << "      {\"id\": \"n1\", \"type\": \"" << type
        << "\", \"name\": \"" << name << "\", \"pos_x\": 0, \"pos_y\": 0";
    if (!params_json.empty()) {
        out << ", \"params\": " << params_json;
    }
    out << "}\n"
        << "    ],\n"
        << "    \"links\": []\n"
        << "  }\n"
        << "}\n";
    return path;
}

std::filesystem::path WritePattern(const std::string& id, const std::string& type) {
    return WritePatternWithNode(id, type, type, "");
}

std::filesystem::path WriteDataBoundaryPattern(const std::string& id) {
    const auto dir = std::filesystem::temp_directory_path() / "cyxwiz_pattern_guard";
    std::filesystem::create_directories(dir);

    const auto path = dir / (id + ".cyxgraph");
    std::ofstream out(path, std::ios::binary);
    Check(out.is_open(), "could not write data-boundary pattern file");
    out << "{\n"
        << "  \"id\": \"" << id << "\",\n"
        << "  \"name\": \"" << id << "\",\n"
        << "  \"category\": \"Training Pipeline\",\n"
        << "  \"template\": {\n"
        << "    \"nodes\": [\n"
        << "      {\"id\": \"input\", \"type\": \"DataInput\", \"name\": \"Data Input\", \"pos_x\": 0, \"pos_y\": 0},\n"
        << "      {\"id\": \"split\", \"type\": \"DataSplit\", \"name\": \"Split\", \"pos_x\": 200, \"pos_y\": 0},\n"
        << "      {\"id\": \"loader\", \"type\": \"DataLoader\", \"name\": \"Loader\", \"pos_x\": 400, \"pos_y\": 0}\n"
        << "    ],\n"
        << "    \"links\": [\n"
        << "      {\"from\": \"input\", \"to\": \"split\", \"from_pin\": 0, \"to_pin\": 0},\n"
        << "      {\"from\": \"input\", \"to\": \"split\", \"from_pin\": 1, \"to_pin\": 1},\n"
        << "      {\"from\": \"split\", \"to\": \"loader\", \"from_pin\": 0, \"to_pin\": 0},\n"
        << "      {\"from\": \"split\", \"to\": \"loader\", \"from_pin\": 1, \"to_pin\": 1}\n"
        << "    ]\n"
        << "  }\n"
        << "}\n";
    return path;
}

gui::MLNode MinimalNode(gui::NodeType type, const std::string& name) {
    gui::MLNode node;
    node.type = type;
    node.name = name;

    gui::NodePin input;
    input.id = 1;
    input.type = gui::PinType::Tensor;
    input.name = "Input";
    input.is_input = true;
    node.inputs.push_back(input);

    gui::NodePin output;
    output.id = 2;
    output.type = gui::PinType::Tensor;
    output.name = "Output";
    output.is_input = false;
    node.outputs.push_back(output);

    return node;
}

std::filesystem::path FindRepoRoot() {
    auto dir = std::filesystem::current_path();
    while (!dir.empty()) {
        if (std::filesystem::exists(dir / "cyxwiz-engine" / "CMakeLists.txt") &&
            std::filesystem::exists(dir / "examples" / "cyxgraph")) {
            return dir;
        }

        const auto parent = dir.parent_path();
        if (parent == dir) {
            break;
        }
        dir = parent;
    }

    return std::filesystem::current_path();
}

std::string ReadFile(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    Check(in.is_open(), "could not open " + path.string());

    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

bool LooksLikeLocalAbsolutePath(const std::string& value) {
    if (value.size() >= 3 &&
        std::isalpha(static_cast<unsigned char>(value[0])) &&
        value[1] == ':' &&
        (value[2] == '\\' || value[2] == '/')) {
        return true;
    }

    return value.rfind("/home/", 0) == 0 ||
           value.rfind("/Users/", 0) == 0 ||
           value.rfind("/mnt/", 0) == 0;
}

void CheckNoLocalAbsolutePaths(const nlohmann::json& value, const std::string& context) {
    if (value.is_string()) {
        const auto text = value.get<std::string>();
        Check(!LooksLikeLocalAbsolutePath(text),
              context + " should not contain local absolute path '" + text + "'");
        return;
    }

    if (value.is_array()) {
        for (size_t i = 0; i < value.size(); ++i) {
            CheckNoLocalAbsolutePaths(value[i], context + "[" + std::to_string(i) + "]");
        }
        return;
    }

    if (value.is_object()) {
        for (auto it = value.begin(); it != value.end(); ++it) {
            CheckNoLocalAbsolutePaths(it.value(), context + "." + it.key());
        }
    }
}

// Every parameter a shipped pattern sets on a node is one that node defines:
// an unknown key (the Embedding's "vocab_size" for num_embeddings) is silently
// ignored, and the node falls back to its default.
// The example templates too: their Adam nodes set "epochs", which only the
// DataLoader reads, so the epoch count was ignored (TOFIX178).
void CheckShippedPatternParameterNames(const std::filesystem::path& dir) {
    Check(std::filesystem::is_directory(dir), "shipped pattern folder not found: " + dir.string());
    auto& registry = cyxwiz::NodeMetadataRegistry::Instance();
    std::string unknown;
    size_t checked = 0;
    for (const auto& entry : std::filesystem::directory_iterator(dir)) {
        if (entry.path().extension() != ".cyxgraph") continue;
        const auto pattern = nlohmann::json::parse(ReadFile(entry.path()));
        if (!pattern.contains("template") || !pattern["template"].contains("nodes")) continue;
        for (const auto& node : pattern["template"]["nodes"]) {
            if (!node.contains("params") || !node["params"].is_object()) continue;
            const auto type = gui::ResolveNodeTypeImportName(node.value("type", ""));
            if (!type) {
                unknown += "\n  " + entry.path().filename().string() + ": unknown node type " + node.value("type", "");
                continue;
            }
            const auto* metadata = registry.GetMetadata(*type);
            if (metadata == nullptr) continue;
            // Data Input's settings depend on its source and file kind; its
            // metadata does not list them.
            if (*type == gui::NodeType::DataInput) continue;
            for (const auto& [key, value] : node["params"].items()) {
                (void)value;
                ++checked;
                const bool defined = std::any_of(metadata->parameters.begin(), metadata->parameters.end(),
                                                 [&](const auto& p) { return p.name == key; });
                if (!defined)
                    unknown += "\n  " + entry.path().filename().string() + ": " + node.value("type", "") + "." + key;
            }
        }
    }
    Check(checked > 0, "no shipped pattern parameters were checked");
    Check(unknown.empty(), "shipped patterns set parameters their nodes do not define:" + unknown);
}

void CheckSavedNERGraphUsesFirstClassSequenceNodes() {
    const auto path = FindRepoRoot() /
        "examples" / "cyxgraph" / "NER" / "ner_bilstm_sequence_tagger.cyxgraph";
    const auto graph = nlohmann::json::parse(ReadFile(path));

    Check(graph.contains("nodes") && graph["nodes"].is_array(),
          "NER graph should contain saved nodes");
    CheckNoLocalAbsolutePaths(graph, "NER graph");

    const std::unordered_map<int, gui::NodeType> expected_types = {
        // The sequence Data Input carries the tagging contract; word + POS
        // Embeddings join on the Concatenate (TOFIX112 word + POS fusion).
        {1, gui::NodeType::DataInput},
        {7, gui::NodeType::DataSplit},
        {8, gui::NodeType::DataLoader},
        {9, gui::NodeType::Embedding},
        {10, gui::NodeType::Embedding},
        {11, gui::NodeType::Concatenate},
        {12, gui::NodeType::LSTM},
        {13, gui::NodeType::Dropout},
        {14, gui::NodeType::TimeDistributed},
        {15, gui::NodeType::CrossEntropyLoss},
        {16, gui::NodeType::Adam},
        {18, gui::NodeType::SequenceTagOutput},
    };

    int checked_nodes = 0;
    for (const auto& node_json : graph["nodes"]) {
        Check(node_json.contains("id") && node_json["id"].is_number_integer(),
              "NER graph node should have integer id");
        Check(node_json.contains("type") && node_json["type"].is_number_integer(),
              "NER graph node should have integer type");

        const int id = node_json["id"].get<int>();
        const auto type = static_cast<gui::NodeType>(node_json["type"].get<int>());
        const std::string name = node_json.value("name", "");

        auto expected_it = expected_types.find(id);
        if (expected_it != expected_types.end()) {
            Check(type == expected_it->second,
                  "NER graph node '" + name + "' has stale serialized node type");
            ++checked_nodes;
        }

        gui::MLNode node;
        node.type = type;
        node.name = name;
        if (node_json.contains("parameters") && node_json["parameters"].is_object()) {
            node.parameters =
                node_json["parameters"].get<std::map<std::string, std::string>>();
        }

        if (id == 1) {
            Check(node.parameters["file_path"] ==
                      "examples/cyxgraph/NER/generated/ner_sentences.csv",
                  "NER graph DataInput should use repo-relative generated sentence CSV");
            Check(node.parameters["raw_source_path"] ==
                      "examples/cyxgraph/NER/sample_ner.csv",
                  "NER graph DataInput should use repo-relative sample CSV");
            Check(node.parameters["token_column"] == "tokens" &&
                      node.parameters["pos_column"] == "pos_tags" &&
                      node.parameters["tag_column"] == "ner_tags",
                  "NER graph DataInput should declare the token, POS and tag columns");
        } else if (id == 18) {
            Check(node.parameters["tag_vocab_file"] ==
                      "examples/cyxgraph/NER/generated/ner_tag_vocab.txt",
                  "NER graph output should use repo-relative tag vocab path");
        }

        std::string marker;
        Check(!gui::detail::IsDenseEncodedSequencePlaceholder(node, marker),
              "NER graph node '" + name +
              "' is still encoded as a Dense sequence placeholder");
    }

    Check(checked_nodes == static_cast<int>(expected_types.size()),
          "NER graph should include every expected first-class sequence node");
}

void CheckExampleDataBoundaryPins() {
    const auto examples_root = FindRepoRoot() / "examples" / "cyxgraph";

    for (const auto& entry : std::filesystem::recursive_directory_iterator(examples_root)) {
        if (!entry.is_regular_file() || entry.path().extension() != ".cyxgraph") {
            continue;
        }

        const auto graph = nlohmann::json::parse(ReadFile(entry.path()));
        if (graph.contains("template") && graph["template"].is_object()) {
            const auto& pattern = graph["template"];
            if (!pattern.contains("nodes") || !pattern["nodes"].is_array() ||
                !pattern.contains("links") || !pattern["links"].is_array()) {
                continue;
            }

            std::unordered_map<std::string, std::string> node_types;
            for (const auto& node_json : pattern["nodes"]) {
                if (node_json.contains("id") && node_json["id"].is_string() &&
                    node_json.contains("type") && node_json["type"].is_string()) {
                    node_types[node_json["id"].get<std::string>()] =
                        node_json["type"].get<std::string>();
                }
            }

            std::size_t link_number = 0;
            for (const auto& link_json : pattern["links"]) {
                const std::string from = link_json.value("from", "");
                const std::string to = link_json.value("to", "");
                const auto from_it = node_types.find(from);
                const auto to_it = node_types.find(to);
                const std::string context = entry.path().string() +
                    " template link " + std::to_string(link_number++);
                if (from_it == node_types.end() || to_it == node_types.end()) {
                    continue;
                }

                if (from_it->second == "DataInput" || from_it->second == "DataSplit") {
                    Check(link_json.value("from_pin", 0) == 0,
                          context + " uses a stale data-boundary output pin index");
                }
                if (to_it->second == "DataLoader") {
                    Check(link_json.value("to_pin", 0) == 0,
                          context + " uses a stale DataLoader input pin index");
                }
            }
            continue;
        }

        if (!graph.contains("nodes") || !graph["nodes"].is_array() ||
            !graph.contains("links") || !graph["links"].is_array()) {
            continue;
        }

        std::unordered_map<int, gui::NodeType> node_types;
        for (const auto& node_json : graph["nodes"]) {
            if (node_json.contains("id") && node_json["id"].is_number_integer() &&
                node_json.contains("type") && node_json["type"].is_number_integer()) {
                node_types[node_json["id"].get<int>()] =
                    static_cast<gui::NodeType>(node_json["type"].get<int>());
            }
        }

        for (const auto& link_json : graph["links"]) {
            if (!link_json.contains("from_node") ||
                !link_json["from_node"].is_number_integer() ||
                !link_json.contains("to_node") ||
                !link_json["to_node"].is_number_integer()) {
                continue;
            }

            const auto from_it = node_types.find(link_json["from_node"].get<int>());
            const auto to_it = node_types.find(link_json["to_node"].get<int>());
            if (from_it == node_types.end() || to_it == node_types.end()) {
                continue;
            }

            const std::string context = entry.path().string() + " link " +
                std::to_string(link_json.value("id", -1));

            if (link_json.contains("from_pin_index") &&
                link_json["from_pin_index"].is_number_integer()) {
                const int index = link_json["from_pin_index"].get<int>();
                if (from_it->second == gui::NodeType::DataInput ||
                    from_it->second == gui::NodeType::DataSplit) {
                    Check(index == 0,
                          context + " uses a stale data-boundary output pin index");
                }
            }

            if (link_json.contains("to_pin_index") &&
                link_json["to_pin_index"].is_number_integer() &&
                to_it->second == gui::NodeType::DataLoader) {
                Check(link_json["to_pin_index"].get<int>() == 0,
                      context + " uses a stale DataLoader input pin index");
            }
        }
    }
}

void CheckModernPinConnectivityFixtures() {
    const auto root = FindRepoRoot() / "examples" / "cyxgraph";
    auto load = [&root](const std::filesystem::path& relative_path) {
        return nlohmann::json::parse(ReadFile(root / relative_path));
    };
    auto has_link = [](const nlohmann::json& graph,
                       const char* from,
                       const char* to,
                       int from_pin,
                       int to_pin) {
        for (const auto& link : graph["template"]["links"]) {
            if (link.value("from", "") == from &&
                link.value("to", "") == to &&
                link.value("from_pin", 0) == from_pin &&
                link.value("to_pin", 0) == to_pin) {
                return true;
            }
        }
        return false;
    };

    const auto mnist = load("mnist_mlp.cyxgraph");
    Check(has_link(mnist, "data_loader", "loss", 1, 1),
          "MNIST should route DataLoader.Labels to Loss.Targets");

    const auto disconnected = load("test_pin_connectivity/01_targets_disconnected.cyxgraph");
    Check(!has_link(disconnected, "data_loader", "loss", 1, 1),
          "disconnected-target fixture should keep Loss.Targets unconnected");

    const auto wrong_target = load("test_pin_connectivity/02_targets_wrong_source.cyxgraph");
    Check(has_link(wrong_target, "fc3", "loss", 0, 1),
          "wrong-target fixture should route model output to Loss.Targets");

    const auto wrong_predictions =
        load("test_pin_connectivity/03_predictions_wrong_source.cyxgraph");
    Check(has_link(wrong_predictions, "data_loader", "loss", 1, 0) &&
              has_link(wrong_predictions, "data_loader", "loss", 1, 1),
          "wrong-predictions fixture should route labels to both loss inputs");

    const auto wrong_optimizer =
        load("test_pin_connectivity/04_optimizer_loss_wrong_source.cyxgraph");
    Check(has_link(wrong_optimizer, "data_loader", "loss", 1, 1) &&
              has_link(wrong_optimizer, "fc3", "optimizer", 0, 0),
          "wrong-optimizer fixture should keep valid targets and invalid loss source");
}

void CheckSerializedPinIndexGuard() {
    int index = -1;
    nlohmann::json link = nlohmann::json::object();
    Check(gui::detail::ResolveSerializedPinIndex(link, "from_pin_index", 1, index) &&
              index == 0,
          "index-less legacy links should resolve to the first available pin");
    Check(!gui::detail::ResolveSerializedPinIndex(link, "from_pin_index", 0, index),
          "index-less legacy links should reject nodes without pins");

    link["from_pin_index"] = 1;
    Check(gui::detail::ResolveSerializedPinIndex(link, "from_pin_index", 2, index) &&
              index == 1,
          "explicit in-range pin indices should be preserved");

    link["from_pin_index"] = -1;
    Check(!gui::detail::ResolveSerializedPinIndex(link, "from_pin_index", 2, index),
          "negative pin indices should be rejected");

    link["from_pin_index"] = 2;
    Check(!gui::detail::ResolveSerializedPinIndex(link, "from_pin_index", 2, index),
          "out-of-range pin indices should be rejected");

    link["from_pin_index"] = "1";
    Check(!gui::detail::ResolveSerializedPinIndex(link, "from_pin_index", 2, index),
          "non-integer pin indices should be rejected");
}

void CheckDataBoundaryVersionGuard() {
    for (const auto* format : {"1.0", "2.1"}) {
        auto document = gui::detail::CreateSerializedGraphDocument(format);
        document["nodes"] = nlohmann::json::array();
        document["links"] = nlohmann::json::array();
        const auto loaded = nlohmann::json::parse(document.dump());
        Check(loaded.at("version") == format,
              "graph format and pin contract versions must remain independent");
        Check(!gui::detail::PreserveLegacyDataBoundaryPins(loaded) &&
              !gui::detail::PreserveLegacyDataValidatorOutputs(loaded) &&
              !gui::detail::PreserveLegacyEvaluationTableInputs(loaded) &&
              !gui::detail::PreserveLegacyClassicalTreeTablePins(loaded),
              "new serialized graph documents must use all current GUI pin contracts");
        const auto preserved = nlohmann::json::parse(
            gui::detail::CreateSerializedGraphDocument(format, true).dump());
        Check(gui::detail::PreserveLegacyDataBoundaryPins(preserved),
              "saving a legacy data boundary must not silently migrate its pins");
    }
    nlohmann::json legacy_graph = nlohmann::json::object();
    Check(gui::detail::PreserveLegacyDataBoundaryPins(legacy_graph),
          "unversioned saved graphs must preserve legacy data-boundary pins");

    legacy_graph["data_boundary_version"] = 1;
    Check(gui::detail::PreserveLegacyDataBoundaryPins(legacy_graph),
          "explicit Data Boundary v1 must remain preserved");

    nlohmann::json modern_graph = {
        {"data_boundary_version", gui::detail::kCurrentDataBoundaryVersion}};
    Check(!gui::detail::PreserveLegacyDataBoundaryPins(modern_graph),
          "Data Boundary v2 must recreate modern Dataset pins");

    nlohmann::json malformed_graph = {{"data_boundary_version", "2"}};
    Check(gui::detail::PreserveLegacyDataBoundaryPins(malformed_graph),
          "malformed boundary versions must fail safe to preserved legacy pins");

    nlohmann::json future_graph = {{"data_boundary_version", 3}};
    Check(gui::detail::PreserveLegacyDataBoundaryPins(future_graph),
          "unknown future boundary versions must not be reinterpreted as v2");

    nlohmann::json legacy_test_link = {{"from_pin_index", 5}};
    int index = -1;
    Check(gui::detail::ResolveSerializedPinIndex(
              legacy_test_link, "from_pin_index", 6, index) && index == 5,
          "preserved six-output Data Split must retain its Test Labels link");
    Check(!gui::detail::ResolveSerializedPinIndex(
              legacy_test_link, "from_pin_index", 1, index),
          "the same legacy link must not be silently applied to Dataset v2");
}

} // namespace

// Save selection as pattern: a Dense -> ReLU -> Dropout row fed by a Flatten
// outside the selection saves as a block with a Block input, keeps its
// settings, places and pins, and inserts again as it was saved.
void CheckSaveSelectionRoundTrip() {
    auto& library = gui::patterns::PatternLibrary::Instance();
    const auto dir = std::filesystem::temp_directory_path() / "cyxwiz_pattern_guard_user";
    std::filesystem::remove_all(dir);
    const std::string kept_dir = library.GetUserPatternsDirectory();
    library.SetUserPatternsDirectory(dir.string());

    int pin = 1;
    const auto make = [&pin](int id, gui::NodeType type, const std::string& name) {
        gui::MLNode node = MinimalNode(type, name);
        node.id = id;
        node.inputs[0].id = pin++;
        node.outputs[0].id = pin++;
        return node;
    };
    std::vector<gui::MLNode> nodes = {make(1, gui::NodeType::Flatten, "Flatten"),
                                      make(2, gui::NodeType::Dense, "Dense 512"),
                                      make(3, gui::NodeType::ReLU, "ReLU 1"),
                                      make(4, gui::NodeType::Dropout, "Dropout 0.4")};
    nodes[1].parameters["units"] = "512";
    nodes[3].parameters["rate"] = "0.4";
    const auto wire = [&nodes](int id, int from, int to) {
        gui::NodeLink link;
        link.id = id;
        link.from_node = from;
        link.to_node = to;
        link.from_pin = nodes[from - 1].outputs[0].id;
        link.to_pin = nodes[to - 1].inputs[0].id;
        return link;
    };
    const std::vector<gui::NodeLink> links = {wire(1, 1, 2), wire(2, 2, 3), wire(3, 3, 4)};
    const std::map<int, ImVec2> positions = {
        {1, ImVec2(0, 100)}, {2, ImVec2(300, 100)}, {3, ImVec2(500, 120)}, {4, ImVec2(700, 100)}};
    using gui::patterns::PatternCategory;

    std::string error;
    const std::string id = library.SavePatternFromSelection(nodes, links, positions, {2, 3, 4}, "Dense-ReLU-Dropout",
                                                            "Saved by the guard", PatternCategory::BuildingBlocks,
                                                            &error);
    Check(id == "user:dense-relu-dropout", "the saved pattern is known by its file name: " + id + " " + error);
    Check(std::filesystem::exists(dir / "dense-relu-dropout.cyxgraph"), "the file is in the user's patterns folder");
    const auto* saved = library.GetPattern(id);
    Check(saved != nullptr && saved->user, "the saved pattern is one of My patterns");
    if (saved) {
        Check(saved->kind == gui::patterns::PatternKind::Block, "a selection without Data Input and Output is a block");
        Check(saved->category == PatternCategory::BuildingBlocks, "the category is kept");
        Check(saved->template_data.nodes.size() == 4, "three nodes and a Block input");
        Check(saved->template_data.links.size() == 3, "the Block input link and the two inside links");
        bool input_first = false;
        for (const auto& node : saved->template_data.nodes) {
            if (node.name == "Block input") input_first = node.type == "Identity" && node.pos_x == 0.0f;
            if (node.name == "Dense 512")
                Check(node.type == "Dense" && node.params.at("units") == "512" && node.pos_x == 220.0f &&
                          node.pos_y == 0.0f,
                      "Dense keeps its file type name, settings and place after the Block input");
            if (node.name == "ReLU 1") Check(node.pos_x == 420.0f && node.pos_y == 20.0f, "places are relative");
        }
        Check(input_first, "the Block input is an Identity at the left");
    }

    std::vector<gui::MLNode> out_nodes;
    std::vector<gui::NodeLink> out_links;
    int next_node = 1, next_link = 1;
    const auto creator = [](gui::NodeType type, const std::string& name) { return MinimalNode(type, name); };
    Check(library.InstantiatePatternWithCreator(id, {}, out_nodes, out_links, next_node, next_link, ImVec2(0, 0),
                                                creator, &error),
          "the saved pattern inserts: " + error);
    Check(out_nodes.size() == 4 && out_links.size() == 3, "it inserts as it was saved");

    // The same name again replaces it; an empty selection or a blank name is refused.
    Check(library.SavePatternFromSelection(nodes, links, positions, {2, 3}, "Dense-ReLU-Dropout", "",
                                           PatternCategory::BuildingBlocks, &error) == id,
          "saving under the same name replaces the pattern");
    Check(library.GetPattern(id) && library.GetPattern(id)->template_data.nodes.size() == 3,
          "the replaced one is read");
    Check(library.SavePatternFromSelection(nodes, links, positions, {}, "Empty", "", PatternCategory::Basic, &error)
                  .empty() &&
              error == "no nodes are selected",
          "an empty selection is refused: " + error);
    Check(library.SavePatternFromSelection(nodes, links, positions, {2}, "   ", "", PatternCategory::Basic, &error)
                  .empty() &&
              error == "it needs a name",
          "a blank name is refused: " + error);

    // Delete takes it out of the library; a shipped pattern is not deleted. The
    // file is removed first, so the test never fills the developer's Recycle
    // Bin: a pattern whose file is already gone still leaves My patterns.
    std::filesystem::remove(dir / "dense-relu-dropout.cyxgraph");
    Check(library.DeleteUserPattern(id, &error), "a pattern whose file is gone is deleted: " + error);
    Check(library.GetPattern(id) == nullptr, "the deleted pattern is gone from the library");
    Check(!library.DeleteUserPattern(id, &error), "it cannot be deleted twice");
    Check(!library.DeleteUserPattern("guard_dense", &error), "a shipped pattern is not deleted");

    library.SetUserPatternsDirectory(kept_dir);
    std::filesystem::remove_all(dir);
}

// Every shipped pattern opens as a graph the way the Engine opens one (File >
// Open and the headless tools read the file and instantiate it through the
// library): read from its file, and, unless it says it is not ready, every
// node made by the node factory with its real pins and every link on a pin
// the node has. Node types given by a pattern parameter ("$activation")
// resolve to its default (TOFIX233 A1: they did not open before).
void CheckShippedPatternsOpen(const std::filesystem::path& dir) {
    auto& library = gui::patterns::PatternLibrary::Instance();
    size_t opened = 0, parameter_typed = 0;
    std::string failed;
    for (const auto& entry : std::filesystem::directory_iterator(dir)) {
        if (entry.path().extension() != ".cyxgraph") continue;
        gui::patterns::Pattern pattern;
        std::string error;
        if (!library.ReadPatternFile(entry.path().string(), pattern, &error)) {
            failed += "\n  " + entry.path().filename().string() + ": not read: " + error;
            continue;
        }
        if (!pattern.needs.empty()) continue;
        int factory_node_id = 1, next_pin_id = 1, next_node_id = 1, next_link_id = 1;
        const auto create = [&](gui::NodeType type, const std::string& name) {
            return gui::CreateGraphNode(type, name, factory_node_id, next_pin_id);
        };
        std::vector<gui::MLNode> nodes;
        std::vector<gui::NodeLink> links;
        if (!library.InstantiatePatternWithCreator(pattern, {}, nodes, links, next_node_id, next_link_id,
                                                   ImVec2(0, 0), create, &error)) {
            failed += "\n  " + entry.path().filename().string() + ": " + error;
            continue;
        }
        ++opened;
        if (std::any_of(pattern.template_data.nodes.begin(), pattern.template_data.nodes.end(),
                        [](const auto& node) { return node.type.rfind('$', 0) == 0; }))
            ++parameter_typed;
    }
    Check(failed.empty(), "shipped patterns that do not open as a graph:" + failed);
    Check(opened > 0 && parameter_typed > 0,
          "shipped patterns open, those whose node types are parameters among them (" + std::to_string(opened) +
              " opened, " + std::to_string(parameter_typed) + " with parameter types)");

    gui::patterns::Pattern not_a_pattern;
    std::string error;
    Check(!library.ReadPatternFile((FindRepoRoot() / "examples" / "cyxgraph" / "model_tools" / "mnist_cnn.cyxgraph")
                                       .string(),
                                   not_a_pattern, &error) &&
              error == "it has no template nodes",
          "a saved graph is not read as a pattern: " + error);
}

int main() {
    const auto training_graph = [](std::initializer_list<gui::NodeType> types) {
        std::vector<gui::MLNode> nodes;
        for (auto type : types) {
            gui::MLNode node;
            node.type = type;
            nodes.push_back(node);
        }
        return cyxwiz::HasTrainingGraphStructure(nodes);
    };
    using gui::NodeType;
    Check(training_graph({NodeType::DataInput, NodeType::DataSplit, NodeType::DataLoader,
          NodeType::Embedding, NodeType::PositionalEncoding, NodeType::TransformerDecoder,
          NodeType::TransformerDecoder, NodeType::LayerNorm, NodeType::TimeDistributed,
          NodeType::Output, NodeType::CrossEntropyLoss, NodeType::Adam}),
          "transformer language-model graphs must enable the training entry point");
    Check(training_graph({NodeType::DataInput, NodeType::Dense, NodeType::MSELoss}),
          "a data input, a dense layer and a loss enable training");
    Check(!training_graph({NodeType::DatasetInput, NodeType::Dense, NodeType::MSELoss}),
          "the retired Dataset Input is no data source");
    Check(!training_graph({NodeType::DataInput, NodeType::DataSplit, NodeType::DataLoader}),
          "data-only pipelines must not enable training");
    Check(!training_graph({NodeType::DataInput, NodeType::CrossEntropyLoss}),
          "a loss without a model must not enable training");
    Check(!training_graph({NodeType::Embedding, NodeType::CrossEntropyLoss}),
          "a graph without a dataset source must not enable training");
    Check(!training_graph({NodeType::DataInput, NodeType::TransformerDecoder}),
          "a graph without a loss must not enable training");
    Check(!training_graph({}), "empty graphs must not enable training");
    auto& library = gui::patterns::PatternLibrary::Instance();
    Check(library.LoadPatternFromFile(WritePattern("guard_dense", "Dense").string()),
          "failed to load implemented-node pattern");
    Check(library.LoadPatternFromFile(WritePattern("guard_add", "Add").string()),
          "failed to load graph-runtime merge pattern");
    Check(library.LoadPatternFromFile(WritePattern("guard_concat", "Concatenate").string()),
          "failed to load deferred merge pattern");
    Check(library.LoadPatternFromFile(WritePattern("guard_tensor_dot", "TensorDot").string()),
          "failed to load graph-runtime dot pattern");
    Check(library.LoadPatternFromFile(WritePattern("guard_lambda", "Lambda").string()),
          "failed to load template-node pattern");
    Check(library.LoadPatternFromFile(WritePattern("guard_multihead_attention", "MultiHeadAttention").string()),
          "failed to load implemented attention-node pattern");
    Check(library.LoadPatternFromFile(WritePattern("guard_cosine_scheduler", "CosineAnnealing").string()),
          "failed to load implemented scheduler-node pattern");
    Check(library.LoadPatternFromFile(WritePattern("guard_typo", "DefinitelyNotANode").string()),
          "failed to load unknown-node pattern");
    Check(library.LoadPatternFromFile(
              WritePatternWithNode("guard_ner_name", "Dense", "NERSequenceBuilder", "").string()),
          "failed to load Dense-encoded NER placeholder-name pattern");
    Check(library.LoadPatternFromFile(
              WritePatternWithNode("guard_ner_param",
                                   "Dense",
                                   "Custom Sequence Builder",
                                   "{\"bio_scheme\": \"BIO\", \"units\": \"128\"}").string()),
          "failed to load Dense-encoded NER parameter-marker pattern");
    Check(library.LoadPatternFromFile(WriteDataBoundaryPattern("guard_data_boundary").string()),
          "failed to load data-boundary pattern");

    std::vector<gui::MLNode> nodes;
    std::vector<gui::NodeLink> links;
    int next_node_id = 100;
    int next_link_id = 1000;
    int creator_calls = 0;
    auto creator = [&creator_calls](gui::NodeType type, const std::string& name) {
        ++creator_calls;
        return MinimalNode(type, name);
    };

    Check(library.InstantiatePatternWithCreator(
              "guard_dense", {}, nodes, links, next_node_id, next_link_id, ImVec2(0, 0), creator),
          "implemented node pattern should instantiate");
    Check(creator_calls == 1, "implemented pattern should call node creator once");
    Check(nodes.size() == 1 && nodes.front().type == gui::NodeType::Dense,
          "implemented pattern created wrong node type");

    nodes.clear();
    links.clear();
    Check(library.InstantiatePatternWithCreator(
              "guard_add", {}, nodes, links, next_node_id, next_link_id, ImVec2(0, 0), creator),
          "implemented graph-runtime Add pattern should instantiate");
    Check(creator_calls == 2, "Add pattern should call node creator once");
    Check(nodes.size() == 1 && nodes.front().type == gui::NodeType::Add,
          "Add pattern created wrong node type");

    nodes.clear();
    links.clear();
    Check(library.InstantiatePatternWithCreator(
              "guard_concat", {}, nodes, links, next_node_id, next_link_id, ImVec2(0, 0), creator),
          "implemented Concatenate pattern should instantiate");
    Check(creator_calls == 3, "Concatenate pattern should call node creator once");
    Check(nodes.size() == 1 && nodes.front().type == gui::NodeType::Concatenate,
          "Concatenate pattern created wrong node type");

    nodes.clear();
    links.clear();
    Check(library.InstantiatePatternWithCreator(
              "guard_tensor_dot", {}, nodes, links, next_node_id, next_link_id, ImVec2(0, 0), creator),
          "implemented TensorDot pattern should instantiate");
    Check(creator_calls == 4, "TensorDot pattern should call node creator once");
    Check(nodes.size() == 1 && nodes.front().type == gui::NodeType::TensorDot,
          "TensorDot pattern created wrong node type");

    nodes.clear();
    links.clear();
    Check(!library.InstantiatePatternWithCreator(
              "guard_lambda", {}, nodes, links, next_node_id, next_link_id, ImVec2(0, 0), creator),
          "template Lambda pattern should be rejected");
    Check(nodes.empty() && links.empty(), "template rejection should leave no partial graph");
    Check(creator_calls == 4, "template rejection should not call node creator");

    nodes.clear();
    links.clear();
    Check(library.InstantiatePatternWithCreator(
              "guard_multihead_attention", {}, nodes, links, next_node_id, next_link_id, ImVec2(0, 0), creator),
          "implemented MultiHeadAttention pattern should instantiate");
    Check(creator_calls == 5, "MultiHeadAttention pattern should call node creator once");
    Check(nodes.size() == 1 && nodes.front().type == gui::NodeType::MultiHeadAttention,
          "MultiHeadAttention pattern created wrong node type");

    nodes.clear();
    links.clear();
    // Learning-rate schedulers are implemented since TOFIX140 A3 (db4ebe36).
    Check(library.InstantiatePatternWithCreator(
              "guard_cosine_scheduler", {}, nodes, links, next_node_id, next_link_id, ImVec2(0, 0), creator),
          "implemented CosineAnnealing pattern should instantiate");
    Check(creator_calls == 6, "CosineAnnealing pattern should call node creator once");
    Check(nodes.size() == 1 && nodes.front().type == gui::NodeType::CosineAnnealing,
          "CosineAnnealing pattern created wrong node type");

    nodes.clear();
    links.clear();
    Check(!library.InstantiatePatternWithCreator(
              "guard_typo", {}, nodes, links, next_node_id, next_link_id, ImVec2(0, 0), creator),
          "unknown node pattern should be rejected");
    Check(nodes.empty() && links.empty(), "unknown rejection should leave no partial graph");
    Check(creator_calls == 6, "unknown rejection should not call node creator");

    // Its second link leaves output 2 of Data Input, which has one output: the whole
    // pattern is refused with that reason, not inserted without the link.
    nodes.clear();
    links.clear();
    std::string error;
    Check(!library.InstantiatePatternWithCreator(
              "guard_data_boundary", {}, nodes, links, next_node_id,
              next_link_id, ImVec2(0, 0), creator, &error),
          "a pattern with a link to a pin its node lacks is refused");
    Check(nodes.empty() && links.empty(), "a refused pattern leaves no partial graph");
    Check(error == "a link comes from output 2 of \"Data Input\", which has 1 output",
          "the refusal names the pin and the node: " + error);

    nodes.clear();
    links.clear();
    Check(!library.InstantiatePatternWithCreator(
              "guard_typo", {}, nodes, links, next_node_id, next_link_id, ImVec2(0, 0), creator, &error),
          "unknown node pattern is refused again");
    Check(error.find("DefinitelyNotANode") != std::string::npos && error.find("does not know") != std::string::npos,
          "an unknown type is named in the reason: " + error);
    Check(!library.InstantiatePatternWithCreator(
              "guard_lambda", {}, nodes, links, next_node_id, next_link_id, ImVec2(0, 0), creator, &error),
          "template pattern is refused again");
    Check(error.find("template") != std::string::npos, "a template node is named in the reason: " + error);
    Check(!library.InstantiatePatternWithCreator(
              "no_such_pattern", {}, nodes, links, next_node_id, next_link_id, ImVec2(0, 0), creator, &error),
          "a pattern the library lacks is refused");
    Check(error == "the pattern is not in the library", "a missing pattern says so: " + error);
    const int calls_before_refusals = creator_calls;

    nodes.clear();
    links.clear();
    Check(!library.InstantiatePatternWithCreator(
              "guard_ner_name", {}, nodes, links, next_node_id, next_link_id, ImVec2(0, 0), creator),
          "Dense-encoded NER placeholder-name pattern should be rejected");
    Check(nodes.empty() && links.empty(), "Dense-encoded NER name rejection should leave no partial graph");
    Check(creator_calls == calls_before_refusals, "Dense-encoded NER name rejection should not call node creator");
    Check(!library.InstantiatePatternWithCreator(
              "guard_ner_param", {}, nodes, links, next_node_id, next_link_id, ImVec2(0, 0), creator, &error),
          "Dense-encoded NER parameter-marker pattern should be rejected");
    Check(nodes.empty() && links.empty(), "Dense-encoded NER parameter rejection should leave no partial graph");
    Check(creator_calls == calls_before_refusals, "a refused pattern calls the node creator for none of its nodes");

    const auto placeholder_name = [](gui::NodeType type, const std::string& name) {
        std::string marker;
        return gui::detail::IsDenseEncodedSequencePlaceholderName(type, name, marker);
    };
    Check(placeholder_name(gui::NodeType::Dense, "NERSequenceBuilder"),
          "Dense-encoded NERSequenceBuilder placeholder should be rejected");
    Check(placeholder_name(gui::NodeType::Dense, "Sentence Sequences"),
          "Dense-encoded friendly NER placeholder name should be rejected");
    Check(placeholder_name(gui::NodeType::Dense, "TokenCrossEntropyLoss"),
          "Dense-encoded token loss placeholder should be rejected");
    Check(!placeholder_name(gui::NodeType::Dense, "Dense (128)"),
          "ordinary Dense layer names should not be rejected");
    Check(!placeholder_name(gui::NodeType::Embedding, "NERSequenceBuilder"),
          "first-class non-Dense node types should not be rejected by the Dense placeholder guard");

    gui::MLNode dense_encoded_ner;
    dense_encoded_ner.type = gui::NodeType::Dense;
    dense_encoded_ner.name = "Custom Sequence Builder";
    dense_encoded_ner.parameters["bio_scheme"] = "BIO";
    std::string matched_marker;
    Check(gui::detail::IsDenseEncodedSequencePlaceholder(dense_encoded_ner, matched_marker),
          "Dense-encoded NER parameter marker should be rejected");
    Check(matched_marker == "bio_scheme",
          "Dense-encoded NER parameter marker should report the matching parameter");

    CheckSavedNERGraphUsesFirstClassSequenceNodes();
    CheckShippedPatternParameterNames(FindRepoRoot() / "cyxwiz-engine" / "resources" / "patterns");
    CheckShippedPatternParameterNames(FindRepoRoot() / "examples" / "cyxgraph");
    CheckExampleDataBoundaryPins();
    CheckModernPinConnectivityFixtures();
    CheckSerializedPinIndexGuard();
    CheckDataBoundaryVersionGuard();
    CheckSaveSelectionRoundTrip();
    CheckShippedPatternsOpen(FindRepoRoot() / "cyxwiz-engine" / "resources" / "patterns");

    std::cout << "Pattern template guard passed\n";
    return 0;
}
