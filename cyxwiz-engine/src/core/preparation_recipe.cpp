#include "preparation_recipe.h"

#include "graph_model.h"
#include "sql_step_contract.h"

#include <map>
#include <set>
#include <stdexcept>

namespace cyxwiz {
namespace {

using json = nlohmann::json;

[[noreturn]] void Fail(const std::string& reason) {
    throw std::runtime_error("Preparation Recipe: " + reason);
}

std::string NodeLabel(const json& node) {
    const std::string name = node.value("name", std::string());
    return name.empty() ? "node " + std::to_string(node.value("id", 0)) : "'" + name + "'";
}

std::string Parameter(const json& node, const char* key) {
    if (!node.contains("parameters") || !node.at("parameters").is_object()) return {};
    const auto& params = node.at("parameters");
    return params.contains(key) && params.at(key).is_string()
        ? params.at(key).get<std::string>() : std::string();
}

bool IsRecipeWrapper(const json& wrapper) {
    return Parameter(wrapper, kRecipeRoleParameter) == kPreparationRecipeRole;
}

std::map<std::string, std::string> Parameters(const json& node) {
    std::map<std::string, std::string> params;
    if (!node.contains("parameters") || !node.at("parameters").is_object()) return params;
    for (const auto& [key, value] : node.at("parameters").items()) {
        if (value.is_string()) params[key] = value.get<std::string>();
    }
    return params;
}

// Inputs a step needs bound inside the recipe: two for table combiners, one
// per named input for a SQL step.
int RequiredInputs(const json& node) {
    switch (static_cast<gui::NodeType>(node.at("type").get<int>())) {
    case gui::NodeType::JoinTables:
    case gui::NodeType::RowAppender:
    case gui::NodeType::ColumnAppender:
        return 2;
    case gui::NodeType::SQLQuery:
        return static_cast<int>(SqlStepInputAliases(Parameters(node)).size());
    default:
        return 1;
    }
}

bool IsKnownContract(const std::string& version) {
    return version == "1" || version == "2";
}

}  // namespace

bool IsPreparationRecipeStepType(gui::NodeType type) {
    switch (type) {
    // Row operations
    case gui::NodeType::FilterRows:
    case gui::NodeType::SortRows:
    case gui::NodeType::RemoveDuplicateRows:
    case gui::NodeType::SampleRows:
    case gui::NodeType::FillMissingValues:
    case gui::NodeType::TableCropper:
    case gui::NodeType::RowToColumnNames:
    // Column operations
    case gui::NodeType::SelectColumns:
    case gui::NodeType::RenameColumns:
    case gui::NodeType::ColumnAppender:
    case gui::NodeType::CellExtractor:
    case gui::NodeType::CellUpdater:
    case gui::NodeType::StringManipulation:
    case gui::NodeType::MathFormula:
    case gui::NodeType::RuleEngine:
    case gui::NodeType::Unpivot:
    // Text operations
    case gui::NodeType::TextCleanNode:
    case gui::NodeType::JSONPathExtractor:
    case gui::NodeType::RegexTester:
    // Aggregation and combination
    case gui::NodeType::GroupByAggregate:
    case gui::NodeType::JoinTables:
    case gui::NodeType::RowAppender:
    // SQL step (contract 1)
    case gui::NodeType::SQLQuery:
    // Checks
    case gui::NodeType::RowCountCheck:
    case gui::NodeType::DataValidator:
        return true;
    default:
        return false;
    }
}

std::string PreparationRecipeRejection(const json& wrapper, const json& record) {
    const std::string label = NodeLabel(wrapper);
    if (!record.contains("nodes") || !record.at("nodes").is_array() ||
        record.at("nodes").empty()) {
        return label + " has no steps";
    }
    const auto& inputs = record.at("inputs");
    const auto& outputs = record.at("outputs");
    // Unmarked subgraphs are validated against the version marking would write.
    std::string contract = Parameter(wrapper, kRecipeContractParameter);
    if (contract.empty()) contract = kPreparationRecipeContractVersion;
    const size_t max_inputs = contract == "1" ? 1 : kPreparationRecipeMaxInputs;
    if (outputs.size() != 1) {
        return label + " has " + std::to_string(outputs.size()) +
               " output(s); a Preparation Recipe produces exactly one dataset output";
    }
    if (inputs.empty() || inputs.size() > max_inputs) {
        return label + " has " + std::to_string(inputs.size()) + " input(s); a Preparation "
               "Recipe (contract " + contract + ") takes " +
               (contract == "1" ? std::string("exactly one dataset input")
                                : "1 to " + std::to_string(kPreparationRecipeMaxInputs) +
                                      " dataset inputs");
    }

    std::map<int, const json*> steps;
    for (const auto& node : record.at("nodes")) {
        const auto type = static_cast<gui::NodeType>(node.at("type").get<int>());
        if (!IsPreparationRecipeStepType(type)) {
            return "step " + NodeLabel(node) + " cannot be inside a Preparation Recipe: "
                   "sources, exports, splits, fitted transforms and models are pipeline "
                   "stages; keep them outside the recipe";
        }
        steps[node.at("id").get<int>()] = &node;
    }

    std::map<int, int> incoming;
    std::map<int, std::set<int>> downstream;
    for (const auto& link : record.at("links")) {
        const int from = link.at("from_node").get<int>();
        const int to = link.at("to_node").get<int>();
        ++incoming[to];
        downstream[from].insert(to);
    }
    for (const auto& binding : inputs) {
        ++incoming[binding.at("node_id").get<int>()];  // a recipe input feeds this step
    }
    const int exit = outputs.at(0).at("node_id").get<int>();

    for (const auto& [id, node] : steps) {
        const int needed = RequiredInputs(*node);
        if (incoming[id] < needed) {
            return "step " + NodeLabel(*node) + " has " + std::to_string(incoming[id]) +
                   " connected input(s) but needs " + std::to_string(needed) +
                   "; every step input must come from another step or a recipe input";
        }
    }

    // Every step must contribute to the output: a dangling branch would run
    // and be discarded, which hides mistakes.
    std::set<int> reaches_exit{exit};
    bool grew = true;
    while (grew) {
        grew = false;
        for (const auto& [from, targets] : downstream) {
            if (reaches_exit.count(from)) continue;
            for (int to : targets) {
                if (reaches_exit.count(to)) {
                    reaches_exit.insert(from);
                    grew = true;
                    break;
                }
            }
        }
    }
    for (const auto& [id, node] : steps) {
        if (!reaches_exit.count(id)) {
            return "step " + NodeLabel(*node) +
                   " does not lead to the recipe output; remove it or connect it";
        }
    }
    return {};
}

json LowerPreparationRecipes(const json& document) {
    if (!document.contains("nodes") || !document.at("nodes").is_array() ||
        !document.contains("links") || !document.at("links").is_array()) {
        Fail("graph document needs nodes and links arrays");
    }
    json lowered;
    lowered["nodes"] = json::array();
    lowered["links"] = json::array();
    lowered["recipe_steps"] = json::object();

    std::map<int, const json*> wrappers;
    for (const auto& node : document.at("nodes")) {
        if (node.at("type").get<int>() == static_cast<int>(gui::NodeType::Subgraph)) {
            wrappers[node.at("id").get<int>()] = &node;
        } else {
            lowered["nodes"].push_back(node);
        }
    }
    if (wrappers.empty()) {
        lowered["links"] = document.at("links");
        return lowered;
    }
    if (!document.contains("subgraphs") || !document.at("subgraphs").is_array()) {
        Fail("the graph has a Subgraph node without saved contents; recreate it from "
             "the original nodes");
    }

    std::set<int> ids;
    for (const auto& node : lowered["nodes"]) ids.insert(node.at("id").get<int>());

    struct Boundary { int node_id; int pin_index; };
    std::map<int, std::vector<Boundary>> recipe_inputs;
    std::map<int, Boundary> recipe_output;
    for (const auto& record : document.at("subgraphs")) {
        const int owner = record.at("node_id").get<int>();
        const auto it = wrappers.find(owner);
        if (it == wrappers.end()) Fail("saved contents reference a missing Subgraph node");
        const json& wrapper = *it->second;
        if (!IsRecipeWrapper(wrapper)) {
            Fail("Subgraph " + NodeLabel(wrapper) +
                 " is a visual group, not a Preparation Recipe, and cannot run. Expand "
                 "it, or mark it as a Preparation Recipe (right-click > Make Preparation "
                 "Recipe)");
        }
        if (!IsKnownContract(Parameter(wrapper, kRecipeContractParameter))) {
            Fail(NodeLabel(wrapper) + " uses recipe contract version '" +
                 Parameter(wrapper, kRecipeContractParameter) +
                 "'; this Engine runs versions 1 and 2");
        }
        if (const auto why = PreparationRecipeRejection(wrapper, record); !why.empty()) {
            Fail(why);
        }
        for (const auto& step : record.at("nodes")) {
            const int id = step.at("id").get<int>();
            if (!ids.insert(id).second) Fail("duplicate node id " + std::to_string(id));
            lowered["nodes"].push_back(step);
            lowered["recipe_steps"][std::to_string(id)] = owner;
        }
        for (const auto& link : record.at("links")) lowered["links"].push_back(link);
        for (const auto& in : record.at("inputs")) {
            recipe_inputs[owner].push_back({in.at("node_id").get<int>(), in.at("pin_index").get<int>()});
        }
        const auto& out = record.at("outputs").at(0);
        recipe_output[owner] = {out.at("node_id").get<int>(), out.at("pin_index").get<int>()};
        wrappers.erase(it);
    }
    if (!wrappers.empty()) {
        Fail("Subgraph " + NodeLabel(*wrappers.begin()->second) + " has no saved contents");
    }

    for (json link : document.at("links")) {
        const int from = link.at("from_node").get<int>();
        const int to = link.at("to_node").get<int>();
        if (const auto out = recipe_output.find(from); out != recipe_output.end()) {
            if (link.value("from_pin_index", 0) != 0) Fail("recipe output pin out of range");
            link["from_node"] = out->second.node_id;
            link["from_pin_index"] = out->second.pin_index;
        }
        if (const auto in = recipe_inputs.find(to); in != recipe_inputs.end()) {
            // The wrapper's input pin i is recipe input i.
            const int pin = link.value("to_pin_index", 0);
            if (pin < 0 || static_cast<size_t>(pin) >= in->second.size()) {
                Fail("a link reaches recipe input " + std::to_string(pin) + ", but the recipe has " +
                     std::to_string(in->second.size()) + " input(s)");
            }
            link["to_node"] = in->second[static_cast<size_t>(pin)].node_id;
            link["to_pin_index"] = in->second[static_cast<size_t>(pin)].pin_index;
        }
        lowered["links"].push_back(std::move(link));
    }
    return lowered;
}

}  // namespace cyxwiz
