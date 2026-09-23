#include "subgraph_document.h"
#include <algorithm>
#include <limits>
#include <set>
#include <stdexcept>

namespace gui::detail {
namespace {
using json = nlohmann::json;
[[noreturn]] void Invalid(const std::string& reason) {
    throw std::runtime_error("Subgraph document: " + reason);
}
const MLNode& Node(const std::vector<MLNode>& nodes, int id) {
    const auto it = std::find_if(nodes.begin(), nodes.end(),
        [id](const MLNode& n) { return n.id == id; });
    if (it == nodes.end()) Invalid("missing node " + std::to_string(id));
    return *it;
}
json Nodes(const std::vector<MLNode>& nodes, const std::map<int, ImVec2>& positions) {
    json result = json::array();
    for (const auto& node : nodes) {
        const auto it = positions.find(node.id);
        const auto pos = it == positions.end()
            ? ImVec2(node.initial_pos_x, node.initial_pos_y) : it->second;
        result.push_back({{"id", node.id}, {"type", static_cast<int>(node.type)},
            {"category", static_cast<int>(node.category)}, {"name", node.name},
            {"description", node.description}, {"parameters", node.parameters},
            {"pos_x", pos.x}, {"pos_y", pos.y}});
    }
    return result;
}
int PinIndex(const std::vector<NodePin>& pins, int id) {
    for (size_t i = 0; i < pins.size(); ++i)
        if (pins[i].id == id) return static_cast<int>(i);
    Invalid("link or boundary references missing pin " + std::to_string(id));
}
json Links(const std::vector<NodeLink>& links, const std::vector<MLNode>& nodes) {
    json result = json::array();
    for (const auto& link : links) {
        result.push_back({{"id", link.id}, {"from_node", link.from_node},
            {"from_pin", link.from_pin}, {"to_node", link.to_node},
            {"to_pin", link.to_pin}, {"link_type", static_cast<int>(link.type)},
            {"from_pin_index", PinIndex(Node(nodes, link.from_node).outputs, link.from_pin)},
            {"to_pin_index", PinIndex(Node(nodes, link.to_node).inputs, link.to_pin)}});
    }
    return result;
}
json Bindings(const std::vector<int>& mappings, const std::vector<MLNode>& nodes, bool input) {
    json result = json::array();
    for (int pin : mappings) {
        bool found = false;
        for (const auto& node : nodes) {
            const auto& pins = input ? node.inputs : node.outputs;
            for (size_t i = 0; i < pins.size(); ++i) {
                if (pins[i].id != pin) continue;
                if (found) Invalid("ambiguous boundary pin");
                result.push_back({{"node_id", node.id}, {"pin_index", i}});
                found = true;
            }
        }
        if (!found) Invalid("boundary target no longer exists");
    }
    return result;
}
const NodePin& BoundPin(const json& binding, const std::vector<MLNode>& nodes, bool input) {
    const auto& node = Node(nodes, binding.at("node_id").get<int>());
    const auto& pins = input ? node.inputs : node.outputs;
    const int index = binding.at("pin_index").get<int>();
    if (index < 0 || static_cast<size_t>(index) >= pins.size()) Invalid("boundary pin index out of range");
    return pins[static_cast<size_t>(index)];
}
std::set<int> InternalIds(const json& record) {
    std::set<int> result;
    for (const auto& node : record.at("nodes")) result.insert(node.at("id").get<int>());
    return result;
}
}

void CaptureExpandedSubgraph(SubgraphData& data, const std::vector<MLNode>& nodes,
                             const std::vector<NodeLink>& links) {
    if (!data.expanded) return;
    std::set<int> ids;
    for (auto& internal : data.internal_nodes) {
        internal = Node(nodes, internal.id);
        ids.insert(internal.id);
    }
    data.internal_links.clear();
    for (const auto& link : links) {
        const bool from = ids.count(link.from_node) != 0;
        const bool to = ids.count(link.to_node) != 0;
        if (from != to) Invalid("connect external nodes through the subgraph boundary pins");
        if (from && to) data.internal_links.push_back(link);
    }
}

void WriteEditorGraphContent(json& document, const std::vector<MLNode>& nodes,
    const std::vector<NodeLink>& links, const std::vector<SubgraphData>& subgraphs,
    const std::map<int, ImVec2>& positions) {
    std::set<int> internal_ids, internal_link_ids;
    json records = json::array();
    for (auto data : subgraphs) {
        const auto& wrapper = Node(nodes, data.subgraph_node_id);
        if (wrapper.type != NodeType::Subgraph) Invalid("owner is not a Subgraph node");
        CaptureExpandedSubgraph(data, nodes, links);
        for (const auto& node : data.internal_nodes) internal_ids.insert(node.id);
        for (const auto& link : data.internal_links) internal_link_ids.insert(link.id);
        records.push_back({{"node_id", data.subgraph_node_id}, {"expanded", data.expanded},
            {"nodes", Nodes(data.internal_nodes, positions)},
            {"links", Links(data.internal_links, data.internal_nodes)},
            {"inputs", Bindings(data.input_pin_mappings, data.internal_nodes, true)},
            {"outputs", Bindings(data.output_pin_mappings, data.internal_nodes, false)}});
        if (wrapper.inputs.size() != data.input_pin_mappings.size() ||
            wrapper.outputs.size() != data.output_pin_mappings.size()) Invalid("boundary count mismatch");
    }
    std::vector<MLNode> outer_nodes;
    std::vector<NodeLink> outer_links;
    for (const auto& node : nodes) if (!internal_ids.count(node.id)) outer_nodes.push_back(node);
    for (const auto& link : links) if (!internal_link_ids.count(link.id)) outer_links.push_back(link);
    document["nodes"] = Nodes(outer_nodes, positions);
    document["links"] = Links(outer_links, outer_nodes);
    if (!records.empty()) {
        document["subgraph_contract_version"] = 1;
        document["subgraphs"] = std::move(records);
    }
    // Validate before opening/truncating the destination file.
    (void)FlattenSubgraphDocument(document);
}

json FlattenSubgraphDocument(const json& document) {
    json flat = document;
    // Leave legacy non-composite graph validation to the existing loader.
    const bool has_wrapper = std::any_of(document.at("nodes").begin(), document.at("nodes").end(),
        [](const json& node) { return node.value("type", -1) == static_cast<int>(NodeType::Subgraph); });
    if (!has_wrapper && !document.contains("subgraphs") && !document.contains("subgraph_contract_version")) return flat;
    std::set<int> wrappers, node_ids, link_ids, owners;
    auto unique_id = [](const json& object, std::set<int>& ids) {
        const int id = object.at("id").get<int>();
        if (id <= 0 || id == std::numeric_limits<int>::max() || !ids.insert(id).second)
            Invalid("invalid or duplicate node/link id");
        return id;
    };
    for (const auto& node : document.at("nodes")) {
        const int id = unique_id(node, node_ids);
        if (node.at("type").get<int>() == static_cast<int>(NodeType::Subgraph)) wrappers.insert(id);
    }
    for (const auto& link : document.at("links")) unique_id(link, link_ids);
    if (!document.contains("subgraphs")) {
        if (!wrappers.empty() || document.contains("subgraph_contract_version"))
            Invalid("missing subgraph contents; this legacy file saved only a wrapper. Recreate the subgraph from the original nodes.");
        return flat;
    }
    if (document.value("subgraph_contract_version", 0) != 1) Invalid("unsupported contract version");
    if (!document.at("subgraphs").is_array()) Invalid("subgraphs must be an array");
    for (const auto& record : document.at("subgraphs")) {
        const int owner = record.at("node_id").get<int>();
        if (!wrappers.count(owner) || !owners.insert(owner).second) Invalid("missing or duplicate subgraph owner");
        (void)record.at("expanded").get<bool>();
        if (!record.at("nodes").is_array() || record.at("nodes").empty() ||
            !record.at("links").is_array()) Invalid("expected internal nodes and links arrays");
        for (const auto& node : record.at("nodes")) {
            unique_id(node, node_ids);
            if (node.at("type").get<int>() == static_cast<int>(NodeType::Subgraph))
                Invalid("nested subgraphs are not supported by contract version 1");
            flat["nodes"].push_back(node);
        }
        const auto ids = InternalIds(record);
        for (const auto& link : record.at("links")) {
            unique_id(link, link_ids);
            if (!ids.count(link.at("from_node").get<int>()) || !ids.count(link.at("to_node").get<int>()))
                Invalid("internal link crosses a subgraph boundary");
            flat["links"].push_back(link);
        }
        for (const char* side : {"inputs", "outputs"}) {
            if (!record.at(side).is_array()) Invalid("boundary must be an array");
            for (const auto& binding : record.at(side)) {
                if (!ids.count(binding.at("node_id").get<int>()) || binding.at("pin_index").get<int>() < 0)
                    Invalid("invalid boundary target");
            }
        }
    }
    if (owners != wrappers) Invalid("missing subgraph contents");
    // Parent edges may only address parent nodes. Internal edges were checked above.
    std::set<int> outer;
    for (const auto& node : document.at("nodes")) outer.insert(node.at("id").get<int>());
    for (const auto& link : document.at("links"))
        if (!outer.count(link.at("from_node").get<int>()) || !outer.count(link.at("to_node").get<int>()))
            Invalid("parent link crosses a subgraph boundary");
    return flat;
}

void RestoreSubgraphPins(const json& document, std::vector<MLNode>& nodes, int& next_pin_id) {
    if (!document.contains("subgraphs")) return;
    for (const auto& record : document.at("subgraphs")) {
        auto it = std::find_if(nodes.begin(), nodes.end(), [&](const MLNode& node) {
            return node.id == record.at("node_id").get<int>();
        });
        if (it == nodes.end()) Invalid("missing loaded wrapper");
        it->inputs.clear();
        it->outputs.clear();
        for (bool input : {true, false}) {
            auto& pins = input ? it->inputs : it->outputs;
            for (const auto& binding : record.at(input ? "inputs" : "outputs")) {
                NodePin pin = BoundPin(binding, nodes, input);
                if (next_pin_id == std::numeric_limits<int>::max()) Invalid("pin ID capacity exceeded");
                pin.id = next_pin_id++;
                pins.push_back(std::move(pin));
            }
        }
    }
}

std::vector<SubgraphData> RestoreSubgraphContents(const json& document,
    std::vector<MLNode>& nodes, std::vector<NodeLink>& links) {
    std::vector<SubgraphData> result;
    std::set<int> hidden_nodes, hidden_links;
    if (!document.contains("subgraphs")) return result;
    for (const auto& record : document.at("subgraphs")) {
        SubgraphData data{};
        data.subgraph_node_id = record.at("node_id").get<int>();
        data.expanded = record.at("expanded").get<bool>();
        for (const auto& node : record.at("nodes")) {
            const int id = node.at("id").get<int>();
            data.internal_nodes.push_back(Node(nodes, id));
            if (!data.expanded) hidden_nodes.insert(id);
        }
        for (const auto& entry : record.at("links")) {
            const int id = entry.at("id").get<int>();
            const auto it = std::find_if(links.begin(), links.end(), [id](const NodeLink& link) { return link.id == id; });
            if (it == links.end()) Invalid("internal link failed to load");
            data.internal_links.push_back(*it);
            if (!data.expanded) hidden_links.insert(id);
        }
        for (const auto& binding : record.at("inputs")) data.input_pin_mappings.push_back(BoundPin(binding, nodes, true).id);
        for (const auto& binding : record.at("outputs")) data.output_pin_mappings.push_back(BoundPin(binding, nodes, false).id);
        result.push_back(std::move(data));
    }
    std::erase_if(nodes, [&](const MLNode& node) { return hidden_nodes.count(node.id); });
    std::erase_if(links, [&](const NodeLink& link) { return hidden_links.count(link.id); });
    return result;
}
}
