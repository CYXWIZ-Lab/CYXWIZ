#include "node_result_plan.h"

#include "../pipeline_runtime_capabilities.h"
#include "../pipeline_type_names.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <deque>
#include <map>
#include <set>

namespace cyxwiz::plot {

namespace {

const gui::MLNode* Find(const std::vector<gui::MLNode>& nodes, int id) {
    for (const auto& n : nodes)
        if (n.id == id) return &n;
    return nullptr;
}

int PinIndex(const std::vector<gui::NodePin>& pins, int pin_id) {
    for (size_t i = 0; i < pins.size(); ++i)
        if (pins[i].id == pin_id) return static_cast<int>(i);
    return -1;
}

// The first node feeding `node_id` (its first connected input).
int FirstInput(int node_id, const std::vector<gui::NodeLink>& links) {
    for (const auto& l : links)
        if (l.to_node == node_id) return l.from_node;
    return -1;
}

bool Runnable(const gui::MLNode& n, std::string* why) {
    // A Data Input with no file cannot be read (the run would fail with a
    // validation code): say what to do instead.
    if (n.type == gui::NodeType::DataInput) {
        auto it = n.parameters.find("file_path");
        if (it == n.parameters.end() || it->second.empty()) {
            *why = n.name + " has no data yet: open it, choose a file and Apply.";
            return false;
        }
    }
    if (n.type == gui::NodeType::Subgraph) {
        *why = n.name + " is a Preparation Recipe; recipes cannot be plotted from inside yet.";
        return false;
    }
    const PipelineRuntimeSupport support = ResolvePipelineRuntimeSupport(n.type);
    if (support.pipeline_executor_supported && PipelineTypeName(n.type) != "Unknown") return true;
    *why = n.name + " runs only inside training, so its output cannot be plotted yet.";
    return false;
}

std::string Hex(size_t h) {
    char buf[24];
    std::snprintf(buf, sizeof(buf), "%016zx", h);
    return buf;
}

}  // namespace

NodeResultPlan PlanNodeResult(int plot_node_id, const std::vector<gui::MLNode>& nodes,
                              const std::vector<gui::NodeLink>& links,
                              const std::function<bool(const std::string&)>& dataset_loaded) {
    NodeResultPlan plan;
    plan.feeder_id = FirstInput(plot_node_id, links);
    const gui::MLNode* feeder = Find(nodes, plan.feeder_id);
    if (!feeder) {
        plan.state = NodeResultPlan::State::NotConnected;
        plan.feeder_id = -1;
        return plan;
    }
    plan.feeder_name = feeder->name;
    for (const auto& l : links)
        if (l.to_node == plot_node_id && l.from_node == feeder->id) {
            plan.feeder_pin = PinIndex(feeder->outputs, l.from_pin);
            break;
        }
    plan.model = plan.feeder_pin >= 0 && feeder->outputs[static_cast<size_t>(plan.feeder_pin)].type == gui::PinType::Parameters;

    // Everything above the plot, nearest first (breadth-first upstream).
    std::vector<int> order;
    std::set<int> seen;
    std::deque<int> queue{feeder->id};
    while (!queue.empty()) {
        const int id = queue.front();
        queue.pop_front();
        if (!seen.insert(id).second) continue;
        order.push_back(id);
        for (const auto& l : links)
            if (l.to_node == id && !seen.count(l.from_node)) queue.push_back(l.from_node);
    }

    // Fingerprint of the closure: types, names, parameters and links.
    std::string key;
    std::vector<int> sorted(order.begin(), order.end());
    std::sort(sorted.begin(), sorted.end());
    for (int id : sorted) {
        const gui::MLNode* n = Find(nodes, id);
        if (!n) continue;
        key += std::to_string(id) + "|" + std::to_string(static_cast<int>(n->type)) + "|" + n->name + "|";
        std::map<std::string, std::string> params(n->parameters.begin(), n->parameters.end());
        for (const auto& [k, v] : params) key += k + "=" + v + ";";
        key += "\n";
    }
    std::vector<std::string> link_keys;
    for (const auto& l : links)
        if (seen.count(l.from_node) && seen.count(l.to_node))
            link_keys.push_back(std::to_string(l.from_node) + ">" + std::to_string(l.to_node) + ":" +
                                std::to_string(l.from_pin) + ":" + std::to_string(l.to_pin));
    std::sort(link_keys.begin(), link_keys.end());
    for (const auto& lk : link_keys) key += lk + "\n";
    key += "out " + std::to_string(plan.feeder_pin) + "\n";  // the trainer's table or its model
    plan.fingerprint = Hex(std::hash<std::string>{}(key));

    // A loaded Data Input is read as it is (no run).
    if (feeder->type == gui::NodeType::DataInput) {
        auto it = feeder->parameters.find("dataset_name");
        if (it != feeder->parameters.end() && !it->second.empty() && dataset_loaded && dataset_loaded(it->second)) {
            plan.state = NodeResultPlan::State::Loaded;
            plan.dataset_name = it->second;
            return plan;
        }
    }

    // The nearest node above that cannot run decides.
    for (int id : order) {
        const gui::MLNode* n = Find(nodes, id);
        std::string why;
        if (n && !Runnable(*n, &why)) {
            plan.state = NodeResultPlan::State::Unavailable;
            plan.reason = why;
            plan.alternative_id = FirstInput(id, links);
            if (const gui::MLNode* alt = Find(nodes, plan.alternative_id)) plan.alternative_name = alt->name;
            else plan.alternative_id = -1;
            return plan;
        }
    }

    // A sparse vectorizer: its Data Input runs alone; the materializer does the rest.
    if (feeder->type == gui::NodeType::CountVectorizer || feeder->type == gui::NodeType::TFIDFVectorizer) {
        auto format = feeder->parameters.find("output_format");
        std::string f = format == feeder->parameters.end() ? std::string() : format->second;
        for (char& c : f) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        if (f == "sparse") {
            plan.sparse = true;
            plan.closure_ids = sorted;
            for (int id : order) {
                const gui::MLNode* n = Find(nodes, id);
                if (n && n->type == gui::NodeType::DataInput) {
                    plan.source_input_id = id;
                    break;
                }
            }
            if (plan.source_input_id < 0) {
                plan.state = NodeResultPlan::State::Unavailable;
                plan.reason = feeder->name + " needs a Data Input above it.";
                return plan;
            }
            const gui::MLNode* input = Find(nodes, plan.source_input_id);
            auto ds = input->parameters.find("dataset_name");
            if (ds != input->parameters.end() && !ds->second.empty() && dataset_loaded && dataset_loaded(ds->second))
                plan.dataset_name = ds->second;
            nlohmann::json ij;
            ij["nodes"] = nlohmann::json::array();
            nlohmann::json nj;
            nj["id"] = input->id;
            nj["type"] = PipelineTypeName(input->type);
            nj["name"] = input->name;
            nj["parameters"] = nlohmann::json::object();
            for (const auto& [k, v] : input->parameters) nj["parameters"][k] = v;
            ij["nodes"].push_back(std::move(nj));
            ij["links"] = nlohmann::json::array();
            plan.input_pipeline_json = ij.dump();
            plan.state = NodeResultPlan::State::Run;
            plan.run_node_count = static_cast<int>(sorted.size());
            return plan;
        }
    }

    // Run only the closure: its nodes and the links between them.
    nlohmann::json j;
    j["nodes"] = nlohmann::json::array();
    for (int id : sorted) {
        const gui::MLNode* n = Find(nodes, id);
        nlohmann::json nj;
        nj["id"] = n->id;
        nj["type"] = PipelineTypeName(n->type);
        nj["name"] = n->name;
        nj["parameters"] = nlohmann::json::object();
        for (const auto& [k, v] : n->parameters) nj["parameters"][k] = v;
        j["nodes"].push_back(std::move(nj));
    }
    j["links"] = nlohmann::json::array();
    for (const auto& l : links) {
        if (!seen.count(l.from_node) || !seen.count(l.to_node)) continue;
        const gui::MLNode* from = Find(nodes, l.from_node);
        const gui::MLNode* to = Find(nodes, l.to_node);
        const int from_pin = PinIndex(from->outputs, l.from_pin), to_pin = PinIndex(to->inputs, l.to_pin);
        if (from_pin < 0 || to_pin < 0) continue;
        j["links"].push_back({{"start_node", l.from_node},
                              {"end_node", l.to_node},
                              {"start_pin_index", from_pin},
                              {"end_pin_index", to_pin}});
    }
    plan.state = NodeResultPlan::State::Run;
    plan.pipeline_json = j.dump();
    plan.run_node_count = static_cast<int>(sorted.size());
    return plan;
}

}  // namespace cyxwiz::plot
