#include "../src/gui/node_editor.h"

#include <nlohmann/json.hpp>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <queue>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace {

using json = nlohmann::json;

void Check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << "\n";
        std::exit(1);
    }
}

struct Graph {
    std::vector<gui::MLNode> nodes;
    std::vector<gui::NodeLink> links;
};

std::filesystem::path FindRepoRoot() {
    auto dir = std::filesystem::current_path();
    while (!dir.empty()) {
        if (std::filesystem::exists(dir / "examples" / "cyxgraph") &&
            std::filesystem::exists(dir / "cyxwiz-engine" / "CMakeLists.txt")) {
            return dir;
        }
        auto parent = dir.parent_path();
        if (parent == dir) break;
        dir = parent;
    }
    return std::filesystem::current_path();
}

std::string ReadFile(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    Check(in.is_open(), "cannot open " + path.string());
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

void CheckAscii(const std::filesystem::path& path, const std::string& content) {
    for (unsigned char ch : content) {
        Check(ch <= 0x7f,
              path.string() + " contains non-ASCII byte " +
              std::to_string(static_cast<int>(ch)));
    }
}

gui::NodeType TypeFromString(const std::string& type) {
    static const std::map<std::string, gui::NodeType> types = {
        {"DataInput", gui::NodeType::DataInput},
        {"DataSplit", gui::NodeType::DataSplit},
        {"DataLoader", gui::NodeType::DataLoader},
        {"TextTokenizer", gui::NodeType::TextTokenizer},
        {"TFIDFVectorizer", gui::NodeType::TFIDFVectorizer},
        {"TextVocabulary", gui::NodeType::TextVocabulary},
        {"TextPadding", gui::NodeType::TextPadding},
        {"Normalize", gui::NodeType::Normalize},
        {"Embedding", gui::NodeType::Embedding},
        {"Flatten", gui::NodeType::Flatten},
        {"Dense", gui::NodeType::Dense},
        {"ReLU", gui::NodeType::ReLU},
        {"Dropout", gui::NodeType::Dropout},
        {"LSTM", gui::NodeType::LSTM},
        {"GRU", gui::NodeType::GRU},
        {"TransformerEncoder", gui::NodeType::TransformerEncoder},
        {"CrossEntropyLoss", gui::NodeType::CrossEntropyLoss},
        {"Adam", gui::NodeType::Adam},
        {"AdamW", gui::NodeType::AdamW},
        {"Output", gui::NodeType::Output},
    };
    auto it = types.find(type);
    Check(it != types.end(), "unknown node type in text smoke graph: " + type);
    return it->second;
}

gui::NodePin Pin(int id, gui::PinType type, const std::string& name,
                 bool is_input, bool required = true) {
    gui::NodePin pin;
    pin.id = id;
    pin.type = type;
    pin.name = name;
    pin.is_input = is_input;
    pin.is_required = required;
    return pin;
}

gui::MLNode MakeNode(int id, int& next_pin_id, gui::NodeType type,
                     const std::string& name,
                     const std::map<std::string, std::string>& params) {
    gui::MLNode node;
    node.id = id;
    node.type = type;
    node.name = name;
    node.parameters = params;

    auto one_in_out = [&](const std::string& in_name = "Input",
                          const std::string& out_name = "Output") {
        node.inputs.push_back(Pin(next_pin_id++, gui::PinType::Tensor,
                                  in_name, true));
        node.outputs.push_back(Pin(next_pin_id++, gui::PinType::Tensor,
                                   out_name, false));
    };

    switch (type) {
        case gui::NodeType::DataInput:
            node.outputs.push_back(Pin(next_pin_id++, gui::PinType::Dataset,
                                       "Dataset", false));
            break;
        case gui::NodeType::DataSplit:
            node.inputs.push_back(Pin(next_pin_id++, gui::PinType::Dataset,
                                      "Training Dataset", true));
            node.inputs.push_back(Pin(next_pin_id++, gui::PinType::Dataset,
                                      "Validation Dataset", true, false));
            node.inputs.push_back(Pin(next_pin_id++, gui::PinType::Dataset,
                                      "Test Dataset", true, false));
            node.outputs.push_back(Pin(next_pin_id++, gui::PinType::Dataset,
                                       "Partitions", false));
            break;
        case gui::NodeType::DataLoader:
            node.inputs.push_back(Pin(next_pin_id++, gui::PinType::Dataset,
                                      "Partitions", true));
            node.outputs.push_back(Pin(next_pin_id++, gui::PinType::Tensor,
                                       "Data", false));
            node.outputs.push_back(Pin(next_pin_id++, gui::PinType::Labels,
                                       "Labels", false, false));
            break;
        case gui::NodeType::TextTokenizer:
            one_in_out("Text Data", "Token IDs");
            break;
        case gui::NodeType::TFIDFVectorizer:
            one_in_out("Text Data", "TF-IDF Features");
            break;
        case gui::NodeType::TextVocabulary:
            one_in_out("Text Data", "Vocabulary");
            break;
        case gui::NodeType::TextPadding:
            one_in_out("Sequences", "Padded");
            break;
        case gui::NodeType::Embedding:
            one_in_out("Indices", "Embeddings");
            break;
        case gui::NodeType::LSTM:
        case gui::NodeType::GRU:
        case gui::NodeType::TransformerEncoder:
            node.inputs.push_back(Pin(next_pin_id++, gui::PinType::Tensor,
                                      "Input", true));
            node.outputs.push_back(Pin(next_pin_id++, gui::PinType::Tensor,
                                       "Output", false));
            node.outputs.push_back(Pin(next_pin_id++, gui::PinType::Tensor,
                                       "Hidden", false, false));
            break;
        case gui::NodeType::CrossEntropyLoss:
            node.inputs.push_back(Pin(next_pin_id++, gui::PinType::Tensor,
                                      "Predictions", true));
            node.inputs.push_back(Pin(next_pin_id++, gui::PinType::Labels,
                                      "Targets", true));
            node.outputs.push_back(Pin(next_pin_id++, gui::PinType::Loss,
                                       "Loss", false));
            break;
        case gui::NodeType::Adam:
        case gui::NodeType::AdamW:
            node.inputs.push_back(Pin(next_pin_id++, gui::PinType::Loss,
                                      "Loss", true));
            node.outputs.push_back(Pin(next_pin_id++, gui::PinType::Optimizer,
                                       "State", false, false));
            break;
        case gui::NodeType::Output:
            node.inputs.push_back(Pin(next_pin_id++, gui::PinType::Tensor,
                                      "Input", true));
            node.outputs.push_back(Pin(next_pin_id++, gui::PinType::Tensor,
                                       "Predictions", false, false));
            break;
        default:
            one_in_out();
            break;
    }

    return node;
}

std::map<std::string, std::string> ParamsFromJson(const json& node_json,
                                                  const json& root) {
    std::map<std::string, std::string> params;
    if (!node_json.contains("params") && !node_json.contains("parameters")) {
        return params;
    }

    std::unordered_map<std::string, std::string> defaults;
    if (root.contains("parameters")) {
        for (const auto& p : root["parameters"]) {
            defaults[p.value("name", "")] = p.value("default_value", "");
        }
    }

    const auto& param_json = node_json.contains("params")
        ? node_json["params"]
        : node_json["parameters"];

    for (const auto& item : param_json.items()) {
        std::string value;
        if (item.value().is_string()) {
            value = item.value().get<std::string>();
            if (!value.empty() && value[0] == '$') {
                const std::string key = value.substr(1);
                Check(defaults.count(key) > 0,
                      "missing pattern default for parameter $" + key);
                value = defaults.at(key);
            }
        } else if (item.value().is_number_integer()) {
            value = std::to_string(item.value().get<int>());
        } else if (item.value().is_number_float()) {
            value = std::to_string(item.value().get<double>());
        } else if (item.value().is_boolean()) {
            value = item.value().get<bool>() ? "true" : "false";
        }
        params[item.key()] = value;
    }
    return params;
}

gui::NodeType TypeFromJson(const json& node_json) {
    if (node_json["type"].is_string()) {
        return TypeFromString(node_json.value("type", ""));
    }
    return static_cast<gui::NodeType>(node_json.value("type", 0));
}

Graph LoadPatternGraph(const std::filesystem::path& path) {
    const std::string content = ReadFile(path);
    CheckAscii(path, content);
    auto root = json::parse(content);

    Graph graph;
    std::unordered_map<std::string, int> id_by_name;
    int next_node_id = 1;
    int next_pin_id = 1;

    if (root.contains("template")) {
        for (const auto& node_json : root["template"]["nodes"]) {
            const std::string string_id = node_json.value("id", "");
            const auto type = TypeFromJson(node_json);
            auto node = MakeNode(next_node_id++, next_pin_id, type,
                                 node_json.value("name", string_id),
                                 ParamsFromJson(node_json, root));
            id_by_name[string_id] = node.id;
            graph.nodes.push_back(std::move(node));
        }
    } else {
        Check(root.contains("nodes"), "missing nodes in " + path.string());
        for (const auto& node_json : root["nodes"]) {
            const int node_id = node_json.value("id", next_node_id++);
            next_node_id = std::max(next_node_id, node_id + 1);
            const auto type = TypeFromJson(node_json);
            auto node = MakeNode(node_id, next_pin_id, type,
                                 node_json.value("name", ""),
                                 ParamsFromJson(node_json, root));
            graph.nodes.push_back(std::move(node));
        }
    }

    auto node_by_id = [&](int id) -> const gui::MLNode* {
        for (const auto& node : graph.nodes) {
            if (node.id == id) return &node;
        }
        return nullptr;
    };

    int next_link_id = 1;
    if (root.contains("template")) {
        for (const auto& link_json : root["template"]["links"]) {
            const std::string from = link_json.value("from", "");
            const std::string to = link_json.value("to", "");
            Check(id_by_name.count(from) > 0, "unknown link source: " + from);
            Check(id_by_name.count(to) > 0, "unknown link target: " + to);

            const auto* from_node = node_by_id(id_by_name.at(from));
            const auto* to_node = node_by_id(id_by_name.at(to));
            const int from_pin_idx = link_json.value("from_pin", 0);
            const int to_pin_idx = link_json.value("to_pin", 0);
            Check(from_pin_idx >= 0 &&
                  from_pin_idx < static_cast<int>(from_node->outputs.size()),
                  "from_pin out of range for " + from);
            Check(to_pin_idx >= 0 &&
                  to_pin_idx < static_cast<int>(to_node->inputs.size()),
                  "to_pin out of range for " + to);

            gui::NodeLink link;
            link.id = next_link_id++;
            link.from_node = from_node->id;
            link.to_node = to_node->id;
            link.from_pin = from_node->outputs[from_pin_idx].id;
            link.to_pin = to_node->inputs[to_pin_idx].id;
            graph.links.push_back(link);
        }
        return graph;
    }

    Check(root.contains("links"), "missing links in " + path.string());
    for (const auto& link_json : root["links"]) {
        const int from_id = link_json.value("from_node", -1);
        const int to_id = link_json.value("to_node", -1);
        const auto* from_node = node_by_id(from_id);
        const auto* to_node = node_by_id(to_id);
        Check(from_node != nullptr,
              "unknown link source node: " + std::to_string(from_id));
        Check(to_node != nullptr,
              "unknown link target node: " + std::to_string(to_id));

        const int from_pin_idx = link_json.value("from_pin_index", 0);
        const int to_pin_idx = link_json.value("to_pin_index", 0);
        Check(from_pin_idx >= 0 &&
              from_pin_idx < static_cast<int>(from_node->outputs.size()),
              "from_pin_index out of range for " + from_node->name);
        Check(to_pin_idx >= 0 &&
              to_pin_idx < static_cast<int>(to_node->inputs.size()),
              "to_pin_index out of range for " + to_node->name);

        gui::NodeLink link;
        link.id = link_json.value("id", next_link_id++);
        link.from_node = from_node->id;
        link.to_node = to_node->id;
        link.from_pin = from_node->outputs[from_pin_idx].id;
        link.to_pin = to_node->inputs[to_pin_idx].id;
        graph.links.push_back(link);
    }

    return graph;
}

bool IsModelOrActivation(gui::NodeType type) {
    return type == gui::NodeType::Dense ||
           type == gui::NodeType::Embedding ||
           type == gui::NodeType::Flatten ||
           type == gui::NodeType::LSTM ||
           type == gui::NodeType::GRU ||
           type == gui::NodeType::TransformerEncoder ||
           type == gui::NodeType::ReLU ||
           type == gui::NodeType::Dropout;
}

void ValidateConnectivity(const Graph& graph, const std::string& name) {
    std::unordered_set<int> connected_inputs;
    std::unordered_set<int> connected_outputs;
    for (const auto& link : graph.links) {
        connected_inputs.insert(link.to_pin);
        connected_outputs.insert(link.from_pin);
    }

    for (const auto& node : graph.nodes) {
        for (const auto& pin : node.inputs) {
            Check(!pin.is_required || connected_inputs.count(pin.id) > 0,
                  name + ": required input not connected: " + node.name +
                  "." + pin.name);
        }
        for (const auto& pin : node.outputs) {
            Check(!pin.is_required || connected_outputs.count(pin.id) > 0,
                  name + ": required output not connected: " + node.name +
                  "." + pin.name);
        }
    }
}

void ValidateSemanticFlows(const Graph& graph, const std::string& name) {
    std::unordered_map<int, const gui::NodePin*> pin_by_id;
    std::unordered_map<int, int> pin_to_node;
    std::unordered_map<int, const gui::MLNode*> node_by_id;
    for (const auto& node : graph.nodes) {
        node_by_id[node.id] = &node;
        for (const auto& pin : node.inputs) {
            pin_by_id[pin.id] = &pin;
            pin_to_node[pin.id] = node.id;
        }
        for (const auto& pin : node.outputs) {
            pin_by_id[pin.id] = &pin;
            pin_to_node[pin.id] = node.id;
        }
    }

    std::unordered_map<int, std::vector<int>> upstream_outputs;
    for (const auto& link : graph.links) {
        upstream_outputs[link.to_pin].push_back(link.from_pin);
    }

    std::unordered_map<int, std::vector<int>> node_input_pins;
    for (const auto& node : graph.nodes) {
        for (const auto& pin : node.inputs) {
            node_input_pins[node.id].push_back(pin.id);
        }
    }

    auto reaches = [&](int start_input_pin,
                       const auto& accepts_output) {
        std::queue<int> queue;
        std::unordered_set<int> visited;
        queue.push(start_input_pin);
        visited.insert(start_input_pin);
        while (!queue.empty()) {
            const int input_pin_id = queue.front();
            queue.pop();
            auto up = upstream_outputs.find(input_pin_id);
            if (up == upstream_outputs.end()) continue;
            for (int out_pin_id : up->second) {
                const auto* pin = pin_by_id.at(out_pin_id);
                const auto* owner = node_by_id.at(pin_to_node.at(out_pin_id));
                if (accepts_output(*pin, *owner)) return true;
                for (int next_input : node_input_pins[owner->id]) {
                    if (visited.insert(next_input).second) {
                        queue.push(next_input);
                    }
                }
            }
        }
        return false;
    };

    for (const auto& node : graph.nodes) {
        if (node.type == gui::NodeType::CrossEntropyLoss) {
            Check(node.inputs.size() >= 2, name + ": loss pins missing");
            Check(reaches(node.inputs[0].id,
                          [](const gui::NodePin&, const gui::MLNode& owner) {
                              return IsModelOrActivation(owner.type);
                          }),
                  name + ": loss predictions do not reach a model output");
            Check(reaches(node.inputs[1].id,
                          [](const gui::NodePin& pin, const gui::MLNode&) {
                              return pin.type == gui::PinType::Labels;
                          }),
                  name + ": loss targets do not reach a Labels pin");
        } else if (node.type == gui::NodeType::Adam ||
                   node.type == gui::NodeType::AdamW) {
            Check(!node.inputs.empty(), name + ": optimizer loss pin missing");
            Check(reaches(node.inputs[0].id,
                          [](const gui::NodePin& pin, const gui::MLNode&) {
                              return pin.type == gui::PinType::Loss;
                          }),
                  name + ": optimizer loss input does not reach a loss output");
        } else if (node.type == gui::NodeType::TextTokenizer) {
            auto text_col = node.parameters.find("text_col");
            auto label_col = node.parameters.find("label_col");
            auto min_word_freq = node.parameters.find("min_word_freq");
            Check(text_col != node.parameters.end() && !text_col->second.empty(),
                  name + ": TextTokenizer missing text_col");
            Check(label_col != node.parameters.end() && !label_col->second.empty(),
                  name + ": TextTokenizer missing label_col");
            Check(min_word_freq != node.parameters.end() &&
                  !min_word_freq->second.empty(),
                  name + ": TextTokenizer missing min_word_freq");
        } else if (node.type == gui::NodeType::TFIDFVectorizer) {
            auto text_col = node.parameters.find("text_col");
            auto label_col = node.parameters.find("label_col");
            auto max_features = node.parameters.find("max_features");
            Check(text_col != node.parameters.end() && !text_col->second.empty(),
                  name + ": TFIDFVectorizer missing text_col");
            Check(label_col != node.parameters.end() && !label_col->second.empty(),
                  name + ": TFIDFVectorizer missing label_col");
            Check(max_features != node.parameters.end() &&
                  !max_features->second.empty(),
                  name + ": TFIDFVectorizer missing max_features");
        }
    }
}

} // namespace

int main() {
    const auto root = FindRepoRoot();
    const std::vector<std::filesystem::path> graphs = {
        root / "examples/cyxgraph/text/test_01_sentiment_basic.cyxgraph",
        root / "examples/cyxgraph/mental_health_sentiment_classifier.cyxgraph",
        root / "examples/cyxgraph/mental_health_sentiment_classifier_v2.cyxgraph",
        root / "examples/cyxgraph/Sentiment analysis/sentiment_analysis_gru_classifier.cyxgraph",
        root / "examples/cyxgraph/Sentiment analysis/sentiment_analysis_tfidf_mlp_classifier.cyxgraph",
        root / "examples/cyxgraph/text/test_02_sentiment_lstm.cyxgraph",
        root / "examples/cyxgraph/text/test_05_sentiment_transformer_mini.cyxgraph",
    };

    for (const auto& path : graphs) {
        auto graph = LoadPatternGraph(path);
        ValidateConnectivity(graph, path.filename().string());
        ValidateSemanticFlows(graph, path.filename().string());
    }

    std::cout << "Text smoke graph pattern contracts passed\n";
    return 0;
}
