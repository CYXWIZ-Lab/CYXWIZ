#include "pattern_library.h"
#include "../node_editor.h"
#include <imgui.h>
#include <nlohmann/json.hpp>
#include <fstream>
#include <filesystem>
#include <spdlog/spdlog.h>
#include <regex>
#include <algorithm>
#include <set>

namespace gui::patterns {

using json = nlohmann::json;
namespace fs = std::filesystem;

// Singleton instance
PatternLibrary& PatternLibrary::Instance() {
    static PatternLibrary instance;
    return instance;
}

PatternLibrary::PatternLibrary() {
    // Default directories
    builtin_patterns_dir_ = "resources/patterns";
    user_patterns_dir_ = "user_patterns";
}

void PatternLibrary::Initialize() {
    if (initialized_) return;

    spdlog::info("Initializing Pattern Library...");

    LoadBuiltinPatterns();
    LoadUserPatterns(user_patterns_dir_);

    initialized_ = true;
    spdlog::info("Pattern Library initialized with {} patterns", patterns_.size());
}

void PatternLibrary::LoadBuiltinPatterns() {
    // Try multiple possible locations for builtin patterns
    std::vector<std::string> search_paths = {
        builtin_patterns_dir_,
        "../resources/patterns",
        "../../resources/patterns",
        "../../../cyxwiz-engine/resources/patterns"
    };

    for (const auto& path : search_paths) {
        if (fs::exists(path) && fs::is_directory(path)) {
            spdlog::info("Loading builtin patterns from: {}", path);
            for (const auto& entry : fs::directory_iterator(path)) {
                if (entry.path().extension() == ".json") {
                    LoadPatternFromFile(entry.path().string());
                }
            }
            return;
        }
    }

    spdlog::warn("Could not find builtin patterns directory");
}

void PatternLibrary::LoadUserPatterns(const std::string& directory) {
    if (!fs::exists(directory)) {
        spdlog::debug("User patterns directory does not exist: {}", directory);
        return;
    }

    if (!fs::is_directory(directory)) {
        spdlog::warn("User patterns path is not a directory: {}", directory);
        return;
    }

    spdlog::info("Loading user patterns from: {}", directory);
    for (const auto& entry : fs::directory_iterator(directory)) {
        if (entry.path().extension() == ".json") {
            LoadPatternFromFile(entry.path().string());
        }
    }
}

bool PatternLibrary::LoadPatternFromFile(const std::string& filepath) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        spdlog::error("Failed to open pattern file: {}", filepath);
        return false;
    }

    try {
        std::string content((std::istreambuf_iterator<char>(file)),
                            std::istreambuf_iterator<char>());

        Pattern pattern;
        if (ParsePatternJson(content, pattern)) {
            // Use filename (without extension) as ID if not specified
            if (pattern.id.empty()) {
                pattern.id = fs::path(filepath).stem().string();
            }

            patterns_[pattern.id] = pattern;
            spdlog::debug("Loaded pattern: {} ({})", pattern.name, pattern.id);
            return true;
        }
    } catch (const std::exception& e) {
        spdlog::error("Error parsing pattern file {}: {}", filepath, e.what());
    }

    return false;
}

bool PatternLibrary::ParsePatternJson(const std::string& json_content, Pattern& out_pattern) {
    try {
        json j = json::parse(json_content);

        // Required fields
        out_pattern.id = j.value("id", "");
        out_pattern.name = j.value("name", "Unnamed Pattern");
        out_pattern.description = j.value("description", "");

        // Category
        std::string category_str = j.value("category", "Custom");
        out_pattern.category = StringToPatternCategory(category_str);

        // Tags
        if (j.contains("tags") && j["tags"].is_array()) {
            for (const auto& tag : j["tags"]) {
                out_pattern.tags.push_back(tag.get<std::string>());
            }
        }

        // Parameters
        if (j.contains("parameters") && j["parameters"].is_array()) {
            for (const auto& param_json : j["parameters"]) {
                PatternParameter param;
                param.name = param_json.value("name", "");
                param.type = StringToParameterType(param_json.value("type", "string"));
                param.default_value = param_json.value("default_value", "");
                param.description = param_json.value("description", "");
                param.min_value = param_json.value("min_value", "");
                param.max_value = param_json.value("max_value", "");

                if (param_json.contains("options") && param_json["options"].is_array()) {
                    for (const auto& opt : param_json["options"]) {
                        param.options.push_back(opt.get<std::string>());
                    }
                }

                out_pattern.parameters.push_back(param);
            }
        }

        // Template
        if (j.contains("template") && j["template"].is_object()) {
            const auto& tmpl = j["template"];

            // Parse nodes
            if (tmpl.contains("nodes") && tmpl["nodes"].is_array()) {
                for (const auto& node_json : tmpl["nodes"]) {
                    PatternNode node;
                    node.id = node_json.value("id", "");
                    node.type = node_json.value("type", "Dense");
                    node.name = node_json.value("name", node.id);
                    node.pos_x = node_json.value("pos_x", 0.0f);
                    node.pos_y = node_json.value("pos_y", 0.0f);

                    // Alternative position format: pos: [x, y]
                    if (node_json.contains("pos") && node_json["pos"].is_array() && node_json["pos"].size() >= 2) {
                        node.pos_x = node_json["pos"][0].get<float>();
                        node.pos_y = node_json["pos"][1].get<float>();
                    }

                    // Node parameters
                    if (node_json.contains("params") && node_json["params"].is_object()) {
                        for (auto& [key, value] : node_json["params"].items()) {
                            if (value.is_string()) {
                                node.params[key] = value.get<std::string>();
                            } else if (value.is_number_integer()) {
                                node.params[key] = std::to_string(value.get<int>());
                            } else if (value.is_number_float()) {
                                node.params[key] = std::to_string(value.get<float>());
                            } else if (value.is_boolean()) {
                                node.params[key] = value.get<bool>() ? "true" : "false";
                            }
                        }
                    }

                    out_pattern.template_data.nodes.push_back(node);
                }
            }

            // Parse links
            if (tmpl.contains("links") && tmpl["links"].is_array()) {
                for (const auto& link_json : tmpl["links"]) {
                    PatternLink link;
                    link.from_node = link_json.value("from", "");
                    link.to_node = link_json.value("to", "");
                    link.from_pin = link_json.value("from_pin", 0);
                    link.to_pin = link_json.value("to_pin", 0);

                    out_pattern.template_data.links.push_back(link);
                }
            }
        }

        // Store raw JSON for saving
        out_pattern.template_json = json_content;

        return true;
    } catch (const json::exception& e) {
        spdlog::error("JSON parse error: {}", e.what());
        return false;
    }
}

std::vector<Pattern> PatternLibrary::GetAllPatterns() const {
    std::vector<Pattern> result;
    result.reserve(patterns_.size());
    for (const auto& [id, pattern] : patterns_) {
        result.push_back(pattern);
    }
    return result;
}

std::vector<Pattern> PatternLibrary::GetByCategory(PatternCategory category) const {
    std::vector<Pattern> result;
    for (const auto& [id, pattern] : patterns_) {
        if (pattern.category == category) {
            result.push_back(pattern);
        }
    }
    return result;
}

std::vector<Pattern> PatternLibrary::Search(const std::string& query) const {
    if (query.empty()) {
        return GetAllPatterns();
    }

    std::string lower_query = query;
    std::transform(lower_query.begin(), lower_query.end(), lower_query.begin(), ::tolower);

    std::vector<Pattern> result;
    for (const auto& [id, pattern] : patterns_) {
        // Search in name
        std::string lower_name = pattern.name;
        std::transform(lower_name.begin(), lower_name.end(), lower_name.begin(), ::tolower);
        if (lower_name.find(lower_query) != std::string::npos) {
            result.push_back(pattern);
            continue;
        }

        // Search in description
        std::string lower_desc = pattern.description;
        std::transform(lower_desc.begin(), lower_desc.end(), lower_desc.begin(), ::tolower);
        if (lower_desc.find(lower_query) != std::string::npos) {
            result.push_back(pattern);
            continue;
        }

        // Search in tags
        for (const auto& tag : pattern.tags) {
            std::string lower_tag = tag;
            std::transform(lower_tag.begin(), lower_tag.end(), lower_tag.begin(), ::tolower);
            if (lower_tag.find(lower_query) != std::string::npos) {
                result.push_back(pattern);
                break;
            }
        }
    }

    return result;
}

const Pattern* PatternLibrary::GetPattern(const std::string& id) const {
    auto it = patterns_.find(id);
    if (it != patterns_.end()) {
        return &it->second;
    }
    return nullptr;
}

std::vector<PatternCategory> PatternLibrary::GetAvailableCategories() const {
    std::set<PatternCategory> categories;
    for (const auto& [id, pattern] : patterns_) {
        categories.insert(pattern.category);
    }
    return std::vector<PatternCategory>(categories.begin(), categories.end());
}

std::string PatternLibrary::SubstituteParams(
    const std::string& template_str,
    const std::map<std::string, std::string>& params
) const {
    std::string result = template_str;

    // Replace $param_name with actual value
    for (const auto& [name, value] : params) {
        std::string placeholder = "$" + name;
        size_t pos = 0;
        while ((pos = result.find(placeholder, pos)) != std::string::npos) {
            result.replace(pos, placeholder.length(), value);
            pos += value.length();
        }
    }

    return result;
}

NodeType PatternLibrary::StringToNodeType(const std::string& type_str) const {
    // Map string to NodeType enum
    static const std::map<std::string, NodeType> type_map = {
        // Core Layers
        {"Dense", NodeType::Dense},
        {"Conv1D", NodeType::Conv1D},
        {"Conv2D", NodeType::Conv2D},
        {"Conv3D", NodeType::Conv3D},
        {"DepthwiseConv2D", NodeType::DepthwiseConv2D},

        // Pooling
        {"MaxPool2D", NodeType::MaxPool2D},
        {"AvgPool2D", NodeType::AvgPool2D},
        {"GlobalMaxPool", NodeType::GlobalMaxPool},
        {"GlobalAvgPool", NodeType::GlobalAvgPool},
        {"AdaptiveAvgPool", NodeType::AdaptiveAvgPool},

        // Normalization
        {"BatchNorm", NodeType::BatchNorm},
        {"LayerNorm", NodeType::LayerNorm},
        {"GroupNorm", NodeType::GroupNorm},
        {"InstanceNorm", NodeType::InstanceNorm},

        // Regularization
        {"Dropout", NodeType::Dropout},
        {"Flatten", NodeType::Flatten},

        // Recurrent
        {"RNN", NodeType::RNN},
        {"LSTM", NodeType::LSTM},
        {"GRU", NodeType::GRU},
        {"Bidirectional", NodeType::Bidirectional},
        {"TimeDistributed", NodeType::TimeDistributed},
        {"Embedding", NodeType::Embedding},

        // Attention & Transformer
        {"MultiHeadAttention", NodeType::MultiHeadAttention},
        {"SelfAttention", NodeType::SelfAttention},
        {"CrossAttention", NodeType::CrossAttention},
        {"LinearAttention", NodeType::LinearAttention},
        {"TransformerEncoder", NodeType::TransformerEncoder},
        {"TransformerDecoder", NodeType::TransformerDecoder},
        {"PositionalEncoding", NodeType::PositionalEncoding},

        // Activations
        {"ReLU", NodeType::ReLU},
        {"LeakyReLU", NodeType::LeakyReLU},
        {"PReLU", NodeType::PReLU},
        {"ELU", NodeType::ELU},
        {"SELU", NodeType::SELU},
        {"GELU", NodeType::GELU},
        {"Swish", NodeType::Swish},
        {"Mish", NodeType::Mish},
        {"Sigmoid", NodeType::Sigmoid},
        {"Tanh", NodeType::Tanh},
        {"Softmax", NodeType::Softmax},

        // Shape Operations
        {"Reshape", NodeType::Reshape},
        {"Permute", NodeType::Permute},
        {"Squeeze", NodeType::Squeeze},
        {"Unsqueeze", NodeType::Unsqueeze},
        {"View", NodeType::View},
        {"Split", NodeType::Split},

        // Merge Operations
        {"Concatenate", NodeType::Concatenate},
        {"Add", NodeType::Add},
        {"Multiply", NodeType::Multiply},
        {"Average", NodeType::Average},

        // Output
        {"Output", NodeType::Output},

        // Loss Functions
        {"MSELoss", NodeType::MSELoss},
        {"CrossEntropyLoss", NodeType::CrossEntropyLoss},
        {"BCELoss", NodeType::BCELoss},
        {"BCEWithLogits", NodeType::BCEWithLogits},
        {"L1Loss", NodeType::L1Loss},
        {"SmoothL1Loss", NodeType::SmoothL1Loss},
        {"HuberLoss", NodeType::HuberLoss},
        {"NLLLoss", NodeType::NLLLoss},

        // Optimizers
        {"SGD", NodeType::SGD},
        {"Adam", NodeType::Adam},
        {"AdamW", NodeType::AdamW},
        {"RMSprop", NodeType::RMSprop},
        {"Adagrad", NodeType::Adagrad},
        {"NAdam", NodeType::NAdam},

        // Schedulers
        {"StepLR", NodeType::StepLR},
        {"CosineAnnealing", NodeType::CosineAnnealing},
        {"ReduceOnPlateau", NodeType::ReduceOnPlateau},
        {"ExponentialLR", NodeType::ExponentialLR},
        {"WarmupScheduler", NodeType::WarmupScheduler},

        // Regularization Nodes
        {"L1Regularization", NodeType::L1Regularization},
        {"L2Regularization", NodeType::L2Regularization},
        {"ElasticNet", NodeType::ElasticNet},

        // Utility
        {"Lambda", NodeType::Lambda},
        {"Identity", NodeType::Identity},
        {"Constant", NodeType::Constant},
        {"Parameter", NodeType::Parameter},

        // Data Pipeline
        {"DatasetInput", NodeType::DatasetInput},
        {"DataLoader", NodeType::DataLoader},
        {"Augmentation", NodeType::Augmentation},
        {"DataSplit", NodeType::DataSplit},
        {"TensorReshape", NodeType::TensorReshape},
        {"Normalize", NodeType::Normalize},
        {"OneHotEncode", NodeType::OneHotEncode}
    };

    auto it = type_map.find(type_str);
    if (it != type_map.end()) {
        return it->second;
    }

    spdlog::warn("Unknown node type: {}, defaulting to Dense", type_str);
    return NodeType::Dense;
}

bool PatternLibrary::InstantiatePattern(
    const std::string& pattern_id,
    const std::map<std::string, std::string>& params,
    std::vector<MLNode>& out_nodes,
    std::vector<NodeLink>& out_links,
    int& next_node_id,
    int& next_pin_id,
    int& next_link_id,
    ImVec2 base_position
) {
    const Pattern* pattern = GetPattern(pattern_id);
    if (!pattern) {
        spdlog::error("Pattern not found: {}", pattern_id);
        return false;
    }

    // Merge default parameters with provided parameters
    std::map<std::string, std::string> merged_params;
    for (const auto& param : pattern->parameters) {
        merged_params[param.name] = param.default_value;
    }
    for (const auto& [key, value] : params) {
        merged_params[key] = value;
    }

    // Map from pattern node ID to actual node ID
    std::map<std::string, int> node_id_map;

    // Create nodes
    for (const auto& pattern_node : pattern->template_data.nodes) {
        // Substitute parameters in type (for dynamic types like $activation)
        std::string resolved_type = SubstituteParams(pattern_node.type, merged_params);
        NodeType node_type = StringToNodeType(resolved_type);

        // Create the node using NodeEditor's CreateNode logic
        MLNode node;
        node.id = next_node_id++;
        node.type = node_type;
        node.name = SubstituteParams(pattern_node.name, merged_params);

        // Store mapping
        node_id_map[pattern_node.id] = node.id;

        // Store position in node for later application by NodeEditor
        node.initial_pos_x = base_position.x + pattern_node.pos_x;
        node.initial_pos_y = base_position.y + pattern_node.pos_y;
        node.has_initial_position = true;

        // Create pins based on node type (simplified - NodeEditor has more complex logic)
        // Input pin
        NodePin input_pin;
        input_pin.id = next_pin_id++;
        input_pin.type = PinType::Tensor;
        input_pin.name = "Input";
        input_pin.is_input = true;
        node.inputs.push_back(input_pin);

        // Output pin
        NodePin output_pin;
        output_pin.id = next_pin_id++;
        output_pin.type = PinType::Tensor;
        output_pin.name = "Output";
        output_pin.is_input = false;
        node.outputs.push_back(output_pin);

        // Copy parameters with substitution
        for (const auto& [key, value] : pattern_node.params) {
            node.parameters[key] = SubstituteParams(value, merged_params);
        }

        out_nodes.push_back(node);
    }

    // Create links
    for (const auto& pattern_link : pattern->template_data.links) {
        auto from_it = node_id_map.find(pattern_link.from_node);
        auto to_it = node_id_map.find(pattern_link.to_node);

        if (from_it == node_id_map.end() || to_it == node_id_map.end()) {
            spdlog::warn("Invalid link in pattern: {} -> {}", pattern_link.from_node, pattern_link.to_node);
            continue;
        }

        int from_node_id = from_it->second;
        int to_node_id = to_it->second;

        // Find the actual pin IDs
        int from_pin_id = -1;
        int to_pin_id = -1;

        for (const auto& node : out_nodes) {
            if (node.id == from_node_id && !node.outputs.empty()) {
                int pin_idx = std::min(pattern_link.from_pin, (int)node.outputs.size() - 1);
                from_pin_id = node.outputs[pin_idx].id;
            }
            if (node.id == to_node_id && !node.inputs.empty()) {
                int pin_idx = std::min(pattern_link.to_pin, (int)node.inputs.size() - 1);
                to_pin_id = node.inputs[pin_idx].id;
            }
        }

        if (from_pin_id != -1 && to_pin_id != -1) {
            NodeLink link;
            link.id = next_link_id++;
            link.from_node = from_node_id;
            link.from_pin = from_pin_id;
            link.to_node = to_node_id;
            link.to_pin = to_pin_id;
            out_links.push_back(link);
        }
    }

    spdlog::info("Instantiated pattern '{}' with {} nodes and {} links",
                 pattern->name, out_nodes.size(), out_links.size());

    return true;
}

bool PatternLibrary::SavePatternFromSelection(
    const std::vector<MLNode>& nodes,
    const std::vector<NodeLink>& links,
    const std::vector<int>& selected_ids,
    const std::string& name,
    const std::string& description,
    PatternCategory category,
    const std::string& save_path
) {
    if (selected_ids.empty()) {
        spdlog::warn("No nodes selected for pattern creation");
        return false;
    }

    // Build pattern template from selected nodes
    json pattern_json;
    pattern_json["id"] = name;  // Use name as ID (sanitized)
    pattern_json["name"] = name;
    pattern_json["description"] = description;
    pattern_json["category"] = PatternCategoryToString(category);
    pattern_json["tags"] = json::array();
    pattern_json["parameters"] = json::array();

    json template_json;
    json nodes_json = json::array();
    json links_json = json::array();

    // Find bounding box for positioning
    float min_x = std::numeric_limits<float>::max();
    float min_y = std::numeric_limits<float>::max();

    std::set<int> selected_set(selected_ids.begin(), selected_ids.end());
    std::map<int, std::string> node_to_id_map;

    // First pass: find min position and create ID map
    int node_counter = 1;
    for (const auto& node : nodes) {
        if (selected_set.count(node.id)) {
            node_to_id_map[node.id] = "node_" + std::to_string(node_counter++);
            // Note: We'd need position info from ImNodes, which isn't stored in MLNode
            // For now, use default positions
        }
    }

    // Create node entries
    for (const auto& node : nodes) {
        if (!selected_set.count(node.id)) continue;

        json node_json;
        node_json["id"] = node_to_id_map[node.id];
        // Convert NodeType to string (would need reverse mapping)
        node_json["type"] = node.name;  // Simplified
        node_json["name"] = node.name;
        node_json["pos_x"] = 0.0f;  // Would need actual position
        node_json["pos_y"] = 0.0f;

        if (!node.parameters.empty()) {
            node_json["params"] = node.parameters;
        }

        nodes_json.push_back(node_json);
    }

    // Create link entries (only internal links)
    for (const auto& link : links) {
        if (selected_set.count(link.from_node) && selected_set.count(link.to_node)) {
            json link_json;
            link_json["from"] = node_to_id_map[link.from_node];
            link_json["to"] = node_to_id_map[link.to_node];
            links_json.push_back(link_json);
        }
    }

    template_json["nodes"] = nodes_json;
    template_json["links"] = links_json;
    pattern_json["template"] = template_json;

    // Determine save path
    std::string filepath = save_path;
    if (filepath.empty()) {
        filepath = user_patterns_dir_ + "/" + name + ".json";
    }

    // Ensure directory exists
    fs::path dir = fs::path(filepath).parent_path();
    if (!dir.empty() && !fs::exists(dir)) {
        fs::create_directories(dir);
    }

    // Write to file
    std::ofstream file(filepath);
    if (!file.is_open()) {
        spdlog::error("Failed to create pattern file: {}", filepath);
        return false;
    }

    file << pattern_json.dump(2);
    file.close();

    // Reload the pattern into memory
    LoadPatternFromFile(filepath);

    spdlog::info("Saved pattern '{}' to {}", name, filepath);
    return true;
}

} // namespace gui::patterns
