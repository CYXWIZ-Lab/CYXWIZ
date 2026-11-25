#include "pattern_library.h"
#include "../node_editor.h"
#include "../node_import_guardrails.h"
#include "../node_type_import_registry.h"
#include "../../core/node_metadata_registry.h"
#include "../../core/pattern_browser_presentation.h"
#include "../../core/recycle_bin.h"
#include <imgui.h>
#include <nlohmann/json.hpp>
#include <fstream>
#include <filesystem>
#include <spdlog/spdlog.h>
#include <system_error>
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
        if (entry.path().extension() != ".cyxgraph") continue;
        try {
            LoadPatternFile(entry.path().string(), true);
        } catch (const std::system_error&) {
            // A name the system code page cannot spell: skip that file, keep the rest.
            spdlog::warn("Skipped a pattern file in {}: its name cannot be used", directory);
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
    Pattern pattern;
    std::string error;
    if (!ReadPatternFile(filepath, pattern, &error)) {
        spdlog::error("Pattern file {} not read: {}", filepath, error);
        return false;
    }
    // A user pattern is known by its file, so it never replaces a shipped one of the same id.
    if (user) pattern.id = "user:" + fs::path(filepath).stem().string();
    pattern.user = user;

    patterns_[pattern.id] = pattern;
    ++revision_;
    spdlog::debug("Loaded pattern: {} ({})", pattern.name, pattern.id);
    return true;
}

bool PatternLibrary::ReadPatternFile(const std::string& filepath, Pattern& out_pattern, std::string* error) const {
    const auto fail = [error](std::string reason) {
        if (error) *error = std::move(reason);
        return false;
    };
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) return fail("the file cannot be opened");
    const std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    out_pattern = {};
    if (!ParsePatternJson(content, out_pattern)) return fail("it is not valid pattern JSON");
    if (out_pattern.template_data.nodes.empty()) return fail("it has no template nodes");
    // Use filename (without extension) as ID if not specified
    if (out_pattern.id.empty()) out_pattern.id = fs::path(filepath).stem().string();
    out_pattern.file_path = filepath;
    return true;
}

bool PatternLibrary::ParsePatternJson(const std::string& json_content, Pattern& out_pattern) const {
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
    std::error_code ec;
    if (!fs::exists(it->second.file_path, ec) && !ec) {
        // Its file was moved or deleted outside the Engine: only the entry is left.
        spdlog::info("Pattern '{}' had no file left ({}); removed from My patterns", it->second.name,
                     it->second.file_path);
    } else {
        if (!cyxwiz::MoveToRecycleBin(it->second.file_path, error)) return false;
        spdlog::info("Moved pattern '{}' to the {}: {}", it->second.name, cyxwiz::RecycleBinName(),
                     it->second.file_path);
    }
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
    const Pattern* pattern = GetPattern(pattern_id);
    if (pattern)
        return InstantiatePatternWithCreator(*pattern, params, out_nodes, out_links, next_node_id, next_link_id,
                                             base_position, std::move(node_creator), error);
    out_nodes.clear();
    out_links.clear();
    spdlog::error("Pattern '{}' was not inserted: the pattern is not in the library", pattern_id);
    if (error) *error = "the pattern is not in the library";
    return false;
}

bool PatternLibrary::InstantiatePatternWithCreator(
    const Pattern& source,
    const std::map<std::string, std::string>& params,
    std::vector<MLNode>& out_nodes,
    std::vector<NodeLink>& out_links,
    int& next_node_id,
    int& next_link_id,
    ImVec2 base_position,
    NodeCreatorCallback node_creator,
    std::string* error
) const {
    const Pattern* pattern = &source;
    // Nothing goes out unless the whole pattern goes out.
    const auto refuse = [&](std::string reason) {
        out_nodes.clear();
        out_links.clear();
        spdlog::error("Pattern '{}' was not inserted: {}", source.id, reason);
        if (error) *error = std::move(reason);
        return false;
    };
    out_nodes.clear();
    out_links.clear();
    if (error) error->clear();

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

std::filesystem::path PatternLibrary::UserPatternFile(const std::string& name) const {
    return fs::path(user_patterns_dir_) / (cyxwiz::pattern_browser::FileStem(name) + ".cyxgraph");
}

std::string PatternLibrary::SavePatternFromSelection(
    const std::vector<MLNode>& nodes,
    const std::vector<NodeLink>& links,
    const std::map<int, ImVec2>& positions,
    const std::vector<int>& selected_ids,
    const std::string& name,
    const std::string& description,
    PatternCategory category,
    std::string* error
) {
    namespace pb = cyxwiz::pattern_browser;
    const auto fail = [error](std::string reason) {
        spdlog::error("Pattern not saved: {}", reason);
        if (error) *error = std::move(reason);
        return std::string();
    };
    if (error) error->clear();
    if (user_patterns_dir_.empty()) return fail("there is no folder for My patterns");
    if (name.find_first_not_of(" \t") == std::string::npos) return fail("it needs a name");
    const std::set<int> chosen(selected_ids.begin(), selected_ids.end());

    // The same summary the dialog showed: Block or Full graph, where Block input goes.
    std::vector<pb::SelectionNode> summary_nodes;
    std::vector<pb::SelectionLink> summary_links;
    for (const auto& node : nodes)
        summary_nodes.push_back({node.id, node.name, node.type == NodeType::DataInput, node.type == NodeType::Output});
    for (const auto& link : links) summary_links.push_back({link.from_node, link.to_node});
    const auto summary = pb::SummarizeSelection(summary_nodes, summary_links, selected_ids, "");
    if (summary.nodes == 0) return fail("no nodes are selected");

    // The selection's top-left is the pattern's origin.
    bool placed = false;
    ImVec2 origin(0.0f, 0.0f);
    for (int id : selected_ids) {
        const auto at = positions.find(id);
        if (at == positions.end()) continue;
        origin.x = placed ? std::min(origin.x, at->second.x) : at->second.x;
        origin.y = placed ? std::min(origin.y, at->second.y) : at->second.y;
        placed = true;
    }
    const auto place_of = [&](int id) {
        const auto at = positions.find(id);
        return at == positions.end() ? ImVec2(0.0f, 0.0f) : ImVec2(at->second.x - origin.x, at->second.y - origin.y);
    };
    const auto type_name = [](NodeType type) -> std::string {
        for (const auto& entry : GetNodeTypeImportNames())
            if (entry.node_type == type && !entry.legacy_import_compatibility_only) return std::string(entry.name);
        return {};
    };
    const auto pin_index = [](const std::vector<NodePin>& pins, int pin_id) {
        for (size_t i = 0; i < pins.size(); ++i)
            if (pins[i].id == pin_id) return static_cast<int>(i);
        return -1;
    };

    constexpr float kGap = 220.0f;
    const float shift = summary.entries.empty() ? 0.0f : kGap;  // room for Block input at the left
    std::map<int, std::string> ids;
    std::map<int, const MLNode*> by_id;
    json nodes_json = json::array();
    for (const auto& node : nodes) {
        if (!chosen.count(node.id)) continue;
        const std::string type = type_name(node.type);
        if (type.empty()) return fail("\"" + node.name + "\" is a node a pattern cannot hold yet");
        const std::string id = "n" + std::to_string(ids.size() + 1);
        ids[node.id] = id;
        by_id[node.id] = &node;
        const ImVec2 at = place_of(node.id);
        json entry = {{"id", id}, {"type", type}, {"name", node.name}, {"pos", {at.x + shift, at.y}}};
        if (!node.parameters.empty()) entry["params"] = node.parameters;
        nodes_json.push_back(std::move(entry));
    }

    json links_json = json::array();
    int inputs = 0;
    std::map<std::pair<int, int>, std::string> block_inputs;  // (node, input pin) -> its Block input
    for (const auto& link : links) {
        const bool from_in = chosen.count(link.from_node) > 0, to_in = chosen.count(link.to_node) > 0;
        if (!to_in) continue;  // a link out of the selection is not kept
        const MLNode& to = *by_id.at(link.to_node);
        const int to_pin = pin_index(to.inputs, link.to_pin);
        if (to_pin < 0) return fail("a link into \"" + to.name + "\" names a pin it does not have");
        if (from_in) {
            const MLNode& from = *by_id.at(link.from_node);
            const int from_pin = pin_index(from.outputs, link.from_pin);
            if (from_pin < 0) return fail("a link from \"" + from.name + "\" names a pin it does not have");
            links_json.push_back(
                {{"from", ids[from.id]}, {"to", ids[to.id]}, {"from_pin", from_pin}, {"to_pin", to_pin}});
            continue;
        }
        // A link from outside: a full graph drops it; a block gets a Block input there.
        if (summary.kind == pb::Kind::Graph) continue;
        auto& input = block_inputs[{to.id, to_pin}];
        if (input.empty()) {
            input = "block_input_" + std::to_string(++inputs);
            const ImVec2 at = place_of(to.id);
            const std::string input_name = inputs == 1 ? "Block input" : "Block input " + std::to_string(inputs);
            nodes_json.push_back(
                {{"id", input}, {"type", "Identity"}, {"name", input_name}, {"pos", {at.x + shift - kGap, at.y}}});
        }
        links_json.push_back({{"from", input}, {"to", ids[to.id]}, {"from_pin", 0}, {"to_pin", to_pin}});
    }

    const json pattern_json = {
        {"name", name},
        {"kind", summary.kind == pb::Kind::Graph ? "graph" : "block"},
        {"description", description},
        {"category", PatternCategoryToString(category)},
        {"tags", json::array()},
        {"parameters", json::array()},
        {"template", {{"nodes", nodes_json}, {"links", links_json}}},
    };

    const fs::path file = UserPatternFile(name);
    std::error_code ec;
    fs::create_directories(file.parent_path(), ec);
    if (ec) return fail("the folder " + file.parent_path().string() + " could not be made: " + ec.message());
    {
        std::ofstream out(file, std::ios::binary | std::ios::trunc);
        if (!out) return fail("the file " + file.string() + " could not be written");
        out << pattern_json.dump(2) << "\n";
        if (!out) return fail("the file " + file.string() + " could not be written");
    }
    if (!LoadPatternFile(file.string(), true)) return fail("the saved file could not be read back");
    spdlog::info("Saved pattern '{}' ({} nodes) to {}", name, nodes_json.size(), file.string());
    return "user:" + file.stem().string();
}

} // namespace gui::patterns
