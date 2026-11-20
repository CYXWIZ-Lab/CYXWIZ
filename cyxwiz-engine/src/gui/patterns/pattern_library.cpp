#include "pattern_library.h"
#include "../node_editor.h"
#include "../node_import_guardrails.h"
#include "../node_type_import_registry.h"
#include "../../core/node_metadata_registry.h"
#include "../../core/recycle_bin.h"
#include <imgui.h>
#include <nlohmann/json.hpp>
#include <fstream>
#include <filesystem>
#include <spdlog/spdlog.h>
#include <regex>
#include <algorithm>
#include <set>

#ifdef __APPLE__
#include <mach-o/dyld.h>  // For _NSGetExecutablePath
#include <climits>         // For PATH_MAX
#include <libgen.h>        // For dirname
#endif

namespace gui::patterns {

using json = nlohmann::json;
namespace fs = std::filesystem;

namespace {

// Why a pattern node of this type cannot be created; empty when it can.
std::string NotInstantiableReason(NodeType type, const std::string& resolved_type, const std::string& node_name) {
    if (type == NodeType::Unknown)
        return "node \"" + node_name + "\" has the type \"" + resolved_type + "\", which the Engine does not know";

    auto& registry = cyxwiz::NodeMetadataRegistry::Instance();
    if (!registry.IsInitialized()) {
        registry.Initialize();
    }
    const auto* metadata = registry.GetMetadata(type);
    if (metadata == nullptr)
        return "node \"" + node_name + "\" has the type \"" + resolved_type + "\", which has no node contract";
    if (metadata->IsTemplate())
        return "node \"" + node_name + "\" is a " + resolved_type + " node, a template that cannot be inserted";
    return {};
}

} // namespace

// Singleton instance
PatternLibrary& PatternLibrary::Instance() {
    static PatternLibrary instance;
    return instance;
}

PatternLibrary::PatternLibrary() {
    // Default directories
    builtin_patterns_dir_ = "resources/patterns";
}

void PatternLibrary::Initialize() {
    if (initialized_) return;

    spdlog::debug("Initializing Pattern Library...");

    LoadBuiltinPatterns();
    if (!user_patterns_dir_.empty()) LoadUserPatterns(user_patterns_dir_);

    initialized_ = true;
    spdlog::debug("Pattern Library initialized with {} patterns", patterns_.size());
}

void PatternLibrary::LoadBuiltinPatterns() {
    // Try multiple possible locations for builtin patterns
    std::vector<std::string> search_paths = {
        builtin_patterns_dir_,
        "cyxwiz-engine/resources/patterns",  // From project root
        "../resources/patterns",
        "../../resources/patterns",
        "../../../cyxwiz-engine/resources/patterns"
    };

#ifdef __APPLE__
    // On macOS, also check paths relative to the executable
    char exec_path[PATH_MAX];
    uint32_t size = sizeof(exec_path);
    if (_NSGetExecutablePath(exec_path, &size) == 0) {
        std::string exec_dir = fs::path(exec_path).parent_path().string();
        search_paths.insert(search_paths.begin(), exec_dir + "/resources/patterns");
        search_paths.insert(search_paths.begin(), exec_dir + "/../resources/patterns");
        search_paths.insert(search_paths.begin(), exec_dir + "/../Resources/patterns");  // App bundle
        spdlog::debug("macOS executable dir: {}", exec_dir);
    }
#endif

    for (const auto& path : search_paths) {
        if (fs::exists(path) && fs::is_directory(path)) {
            spdlog::debug("Loading builtin patterns from: {}", path);
            for (const auto& entry : fs::directory_iterator(path)) {
                if (entry.path().extension() == ".cyxgraph") {
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
        if (entry.path().extension() == ".cyxgraph") {
            LoadPatternFile(entry.path().string(), true);
        }
    }
}

bool PatternLibrary::LoadPatternFromFile(const std::string& filepath) {
    // A file in the user's patterns folder is one of My patterns, however it is read.
    std::error_code ec;
    const bool user = !user_patterns_dir_.empty() &&
                      fs::equivalent(fs::path(filepath).parent_path(), user_patterns_dir_, ec);
    return LoadPatternFile(filepath, user);
}

bool PatternLibrary::LoadPatternFile(const std::string& filepath, bool user) {
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
            // A user pattern is known by its file, so it never replaces a shipped one of the same id.
            if (user) pattern.id = "user:" + fs::path(filepath).stem().string();
            pattern.file_path = filepath;
            pattern.user = user;

            patterns_[pattern.id] = pattern;
            ++revision_;
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
        out_pattern.kind = StringToPatternKind(j.value("kind", "block"));
        out_pattern.needs = j.value("needs", "");
        out_pattern.sample_shape = j.value("sample_shape", "");

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

bool PatternLibrary::DeleteUserPattern(const std::string& id, std::string* error) {
    const auto it = patterns_.find(id);
    if (it == patterns_.end() || !it->second.user) {
        if (error) *error = "it is not one of My patterns";
        return false;
    }
    if (!cyxwiz::MoveToRecycleBin(it->second.file_path, error)) return false;
    spdlog::info("Moved pattern '{}' to the {}: {}", it->second.name, cyxwiz::RecycleBinName(), it->second.file_path);
    patterns_.erase(it);
    ++revision_;
    return true;
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
    const auto resolved = ResolveNodeTypeImportName(type_str);
    if (resolved.has_value()) {
        return *resolved;
    }

    spdlog::warn("Unknown node type: {}", type_str);
    return NodeType::Unknown;
}

bool PatternLibrary::InstantiatePatternWithCreator(
    const std::string& pattern_id,
    const std::map<std::string, std::string>& params,
    std::vector<MLNode>& out_nodes,
    std::vector<NodeLink>& out_links,
    int& next_node_id,
    int& next_link_id,
    ImVec2 base_position,
    NodeCreatorCallback node_creator,
    std::string* error
) {
    // Nothing goes out unless the whole pattern goes out.
    const auto refuse = [&](std::string reason) {
        out_nodes.clear();
        out_links.clear();
        spdlog::error("Pattern '{}' was not inserted: {}", pattern_id, reason);
        if (error) *error = std::move(reason);
        return false;
    };
    out_nodes.clear();
    out_links.clear();
    if (error) error->clear();

    const Pattern* pattern = GetPattern(pattern_id);
    if (!pattern) return refuse("the pattern is not in the library");
    if (!pattern->needs.empty()) return refuse("it is not ready yet: " + pattern->needs);
    if (!node_creator) return refuse("there is no canvas to create its nodes on");

    // Merge default parameters with provided parameters
    std::map<std::string, std::string> merged_params;
    for (const auto& param : pattern->parameters) {
        merged_params[param.name] = param.default_value;
    }
    for (const auto& [key, value] : params) {
        merged_params[key] = value;
    }

    // Every node type is checked before the first node is created.
    std::vector<std::pair<NodeType, std::string>> resolved;
    for (const auto& pattern_node : pattern->template_data.nodes) {
        // Substitute parameters in type (for dynamic types like $activation)
        const std::string resolved_type = SubstituteParams(pattern_node.type, merged_params);
        const std::string resolved_name = SubstituteParams(pattern_node.name, merged_params);
        const NodeType node_type = StringToNodeType(resolved_type);
        if (auto reason = NotInstantiableReason(node_type, resolved_type, resolved_name); !reason.empty())
            return refuse(reason);
        std::string marker;
        MLNode probe;
        probe.type = node_type;
        probe.name = resolved_name;
        for (const auto& [key, value] : pattern_node.params) probe.parameters[key] = SubstituteParams(value, merged_params);
        if (detail::IsDenseEncodedSequencePlaceholder(probe, marker))
            return refuse("node \"" + resolved_name + "\" is a Dense node standing in for a sequence step (" + marker +
                          "); the pattern needs that step's own node");
        resolved.emplace_back(node_type, resolved_name);
    }

    // Map from pattern node ID to actual node ID
    std::map<std::string, int> node_id_map;
    std::map<int, size_t> node_index;
    for (size_t i = 0; i < pattern->template_data.nodes.size(); ++i) {
        const auto& pattern_node = pattern->template_data.nodes[i];
        // The callback creates the node with its real pins.
        MLNode node = node_creator(resolved[i].first, resolved[i].second);
        node.id = next_node_id++;
        node_id_map[pattern_node.id] = node.id;
        node_index[node.id] = out_nodes.size();

        // Store position for later application by NodeEditor
        node.initial_pos_x = base_position.x + pattern_node.pos_x;
        node.initial_pos_y = base_position.y + pattern_node.pos_y;
        node.has_initial_position = true;

        for (const auto& [key, value] : pattern_node.params) {
            node.parameters[key] = SubstituteParams(value, merged_params);
        }
        out_nodes.push_back(std::move(node));
    }

    for (const auto& pattern_link : pattern->template_data.links) {
        const auto from_it = node_id_map.find(pattern_link.from_node);
        const auto to_it = node_id_map.find(pattern_link.to_node);
        if (from_it == node_id_map.end())
            return refuse("a link names node \"" + pattern_link.from_node + "\", which the pattern does not have");
        if (to_it == node_id_map.end())
            return refuse("a link names node \"" + pattern_link.to_node + "\", which the pattern does not have");

        const MLNode& from = out_nodes[node_index.at(from_it->second)];
        const MLNode& to = out_nodes[node_index.at(to_it->second)];
        if (pattern_link.from_pin < 0 || pattern_link.from_pin >= static_cast<int>(from.outputs.size()))
            return refuse("a link comes from output " + std::to_string(pattern_link.from_pin + 1) + " of \"" +
                          from.name + "\", which has " + std::to_string(from.outputs.size()) + " output" +
                          (from.outputs.size() == 1 ? "" : "s"));
        if (pattern_link.to_pin < 0 || pattern_link.to_pin >= static_cast<int>(to.inputs.size()))
            return refuse("a link goes to input " + std::to_string(pattern_link.to_pin + 1) + " of \"" + to.name +
                          "\", which has " + std::to_string(to.inputs.size()) + " input" +
                          (to.inputs.size() == 1 ? "" : "s"));

        NodeLink link;
        link.id = next_link_id++;
        link.from_node = from.id;
        link.from_pin = from.outputs[pattern_link.from_pin].id;
        link.to_node = to.id;
        link.to_pin = to.inputs[pattern_link.to_pin].id;
        link.type = LinkType::TensorFlow;
        out_links.push_back(link);
    }

    spdlog::info("Instantiated pattern '{}' with {} nodes and {} links", pattern->name, out_nodes.size(),
                 out_links.size());
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

    std::set<int> selected_set(selected_ids.begin(), selected_ids.end());
    std::map<int, std::string> node_to_id_map;

    // First pass: create ID map
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
        filepath = user_patterns_dir_ + "/" + name + ".cyxgraph";
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
