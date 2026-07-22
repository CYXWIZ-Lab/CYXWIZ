#include "../src/gui/patterns/pattern_library.h"
#include "../src/gui/node_import_guardrails.h"

#include <nlohmann/json.hpp>

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

void CheckSavedNERGraphUsesFirstClassSequenceNodes() {
    const auto path = FindRepoRoot() /
        "examples" / "cyxgraph" / "NER" / "ner_bilstm_sequence_tagger.cyxgraph";
    const auto graph = nlohmann::json::parse(ReadFile(path));

    Check(graph.contains("nodes") && graph["nodes"].is_array(),
          "NER graph should contain saved nodes");
    CheckNoLocalAbsolutePaths(graph, "NER graph");

    const std::unordered_map<int, gui::NodeType> expected_types = {
        {1, gui::NodeType::DataInput},
        {2, gui::NodeType::NERSequenceBuilder},
        {3, gui::NodeType::TokenVocabulary},
        {4, gui::NodeType::POSVocabulary},
        {5, gui::NodeType::NERTagVocabulary},
        {6, gui::NodeType::TextPadding},
        {7, gui::NodeType::DataSplit},
        {8, gui::NodeType::DataLoader},
        {11, gui::NodeType::Concatenate},
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
        } else if (id == 3) {
            Check(node.parameters["vocab_file"] ==
                      "examples/cyxgraph/NER/generated/ner_word_vocab.txt",
                  "NER graph TokenVocabulary should use repo-relative word vocab path");
        } else if (id == 4) {
            Check(node.parameters["vocab_file"] ==
                      "examples/cyxgraph/NER/generated/ner_pos_vocab.txt",
                  "NER graph POSVocabulary should use repo-relative POS vocab path");
        } else if (id == 5) {
            Check(node.parameters["vocab_file"] ==
                      "examples/cyxgraph/NER/generated/ner_tag_vocab.txt",
                  "NER graph NERTagVocabulary should use repo-relative tag vocab path");
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

} // namespace

int main() {
    auto& library = gui::patterns::PatternLibrary::Instance();
    Check(library.LoadPatternFromFile(WritePattern("guard_dense", "Dense").string()),
          "failed to load implemented-node pattern");
    Check(library.LoadPatternFromFile(WritePattern("guard_add", "Add").string()),
          "failed to load graph-runtime merge pattern");
    Check(library.LoadPatternFromFile(WritePattern("guard_concat", "Concatenate").string()),
          "failed to load deferred merge pattern");
    Check(library.LoadPatternFromFile(WritePattern("guard_tensor_dot", "TensorDot").string()),
          "failed to load graph-runtime dot pattern");
    Check(library.LoadPatternFromFile(WritePattern("guard_batch_matmul", "TensorBatchMatMul").string()),
          "failed to load template-node pattern");
    Check(library.LoadPatternFromFile(WritePattern("guard_multihead_attention", "MultiHeadAttention").string()),
          "failed to load implemented attention-node pattern");
    Check(library.LoadPatternFromFile(WritePattern("guard_cosine_scheduler", "CosineAnnealing").string()),
          "failed to load scheduler template-node pattern");
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
              "guard_batch_matmul", {}, nodes, links, next_node_id, next_link_id, ImVec2(0, 0), creator),
          "template TensorBatchMatMul pattern should be rejected");
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
    Check(!library.InstantiatePatternWithCreator(
              "guard_cosine_scheduler", {}, nodes, links, next_node_id, next_link_id, ImVec2(0, 0), creator),
          "template CosineAnnealing pattern should be rejected");
    Check(nodes.empty() && links.empty(), "scheduler template rejection should leave no partial graph");
    Check(creator_calls == 5, "scheduler template rejection should not call node creator");

    nodes.clear();
    links.clear();
    Check(!library.InstantiatePatternWithCreator(
              "guard_typo", {}, nodes, links, next_node_id, next_link_id, ImVec2(0, 0), creator),
          "unknown node pattern should be rejected");
    Check(nodes.empty() && links.empty(), "unknown rejection should leave no partial graph");
    Check(creator_calls == 5, "unknown rejection should not call node creator");

    nodes.clear();
    links.clear();
    Check(library.InstantiatePatternWithCreator(
              "guard_data_boundary", {}, nodes, links, next_node_id,
              next_link_id, ImVec2(0, 0), creator),
          "creator data-boundary pattern should instantiate");
    Check(creator_calls == 8,
          "data-boundary pattern should call node creator for each node");
    Check(nodes.size() == 3 && links.size() == 2,
          "creator data-boundary pattern should skip stale legacy pin links");

    int next_pin_id = 2000;

    nodes.clear();
    links.clear();
    Check(library.InstantiatePattern(
              "guard_data_boundary", {}, nodes, links, next_node_id,
              next_pin_id, next_link_id, ImVec2(0, 0)),
          "legacy/no-editor data-boundary pattern should instantiate");
    Check(nodes.size() == 3 && links.size() == 2,
          "data-boundary fallback should create three nodes and two links");
    const gui::MLNode* data_input = nullptr;
    const gui::MLNode* data_split = nullptr;
    const gui::MLNode* data_loader = nullptr;
    for (const auto& node : nodes) {
        if (node.type == gui::NodeType::DataInput) data_input = &node;
        if (node.type == gui::NodeType::DataSplit) data_split = &node;
        if (node.type == gui::NodeType::DataLoader) data_loader = &node;
    }
    Check(data_input != nullptr && data_split != nullptr && data_loader != nullptr,
          "data-boundary fallback should create DataInput/DataSplit/DataLoader");
    Check(data_input->outputs.size() == 1 &&
              data_input->outputs[0].name == "Dataset" &&
              data_input->outputs[0].type == gui::PinType::Dataset,
          "fallback DataInput should expose one Dataset output");
    Check(data_split->inputs.size() == 3 && data_split->outputs.size() == 1 &&
              data_split->inputs[0].name == "Training Dataset" &&
              data_split->inputs[0].type == gui::PinType::Dataset &&
              data_split->outputs[0].name == "Partitions" &&
              data_split->outputs[0].type == gui::PinType::Dataset,
          "fallback DataSplit should expose Dataset role inputs and Partitions output");
    Check(data_loader->inputs.size() == 1 && data_loader->outputs.size() == 2 &&
              data_loader->inputs[0].name == "Partitions" &&
              data_loader->inputs[0].type == gui::PinType::Dataset &&
              data_loader->outputs[0].name == "Data" &&
              data_loader->outputs[0].type == gui::PinType::Tensor &&
              data_loader->outputs[1].name == "Labels" &&
              data_loader->outputs[1].type == gui::PinType::Labels,
          "fallback DataLoader should consume Partitions and emit Data/Labels");

    nodes.clear();
    links.clear();
    Check(!library.InstantiatePatternWithCreator(
              "guard_ner_name", {}, nodes, links, next_node_id, next_link_id, ImVec2(0, 0), creator),
          "Dense-encoded NER placeholder-name pattern should be rejected");
    Check(nodes.empty() && links.empty(), "Dense-encoded NER name rejection should leave no partial graph");
    Check(creator_calls == 8, "Dense-encoded NER name rejection should not call node creator");

    Check(!library.InstantiatePattern(
              "guard_ner_name", {}, nodes, links, next_node_id, next_pin_id, next_link_id, ImVec2(0, 0)),
          "legacy instantiation should reject Dense-encoded NER placeholder names");
    Check(nodes.empty() && links.empty(), "legacy Dense-encoded NER name rejection should leave no partial graph");

    Check(!library.InstantiatePattern(
              "guard_ner_param", {}, nodes, links, next_node_id, next_pin_id, next_link_id, ImVec2(0, 0)),
          "legacy instantiation should reject Dense-encoded NER parameter markers");
    Check(nodes.empty() && links.empty(), "legacy Dense-encoded NER parameter rejection should leave no partial graph");

    Check(!library.InstantiatePattern(
              "guard_batch_matmul", {}, nodes, links, next_node_id, next_pin_id, next_link_id, ImVec2(0, 0)),
          "legacy instantiation should also reject template nodes");

    Check(gui::detail::IsDenseEncodedSequencePlaceholder(gui::NodeType::Dense, "NERSequenceBuilder"),
          "Dense-encoded NERSequenceBuilder placeholder should be rejected");
    Check(gui::detail::IsDenseEncodedSequencePlaceholder(gui::NodeType::Dense, "Sentence Sequences"),
          "Dense-encoded friendly NER placeholder name should be rejected");
    Check(gui::detail::IsDenseEncodedSequencePlaceholder(gui::NodeType::Dense, "TokenCrossEntropyLoss"),
          "Dense-encoded token loss placeholder should be rejected");
    Check(!gui::detail::IsDenseEncodedSequencePlaceholder(gui::NodeType::Dense, "Dense (128)"),
          "ordinary Dense layer names should not be rejected");
    Check(!gui::detail::IsDenseEncodedSequencePlaceholder(gui::NodeType::Embedding, "NERSequenceBuilder"),
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

    std::cout << "Pattern template guard passed\n";
    return 0;
}
