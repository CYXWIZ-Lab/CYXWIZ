#include "debug_training_graph_diff.h"

#include <algorithm>
#include <map>
#include <set>
#include <string>
#include <tuple>
#include <utility>

namespace cyxwiz {

namespace {

constexpr size_t kMaxStructuralDetails = 128;
constexpr size_t kMaxParameterKeysPerNode = 64;

void PushStructuralDetail(nlohmann::json& details,
                          nlohmann::json detail,
                          size_t& retained_count) {
    if (retained_count >= kMaxStructuralDetails) {
        return;
    }
    details.push_back(std::move(detail));
    ++retained_count;
}

const DebugTraceRecord* FindGraphSnapshot(
    const DebugRunStoreRecord& record) {
    const auto it = std::find_if(
        record.traces.begin(), record.traces.end(),
        [](const DebugTraceRecord& trace) {
            return trace.phase == "GraphSnapshot" &&
                trace.payload.contains("nodes") &&
                trace.payload.contains("links");
        });
    return it == record.traces.end() ? nullptr : &*it;
}

std::map<int, nlohmann::json> ReadNodes(const DebugTraceRecord& trace) {
    std::map<int, nlohmann::json> nodes;
    const auto it = trace.payload.find("nodes");
    if (it == trace.payload.end() || !it->is_array()) {
        return nodes;
    }
    for (const auto& node : *it) {
        if (!node.is_object()) {
            continue;
        }
        const auto id = node.find("id");
        if (id == node.end() || !id->is_number_integer()) {
            continue;
        }
        nodes[id->get<int>()] = node;
    }
    return nodes;
}

struct LinkKey {
    int from_node = -1;
    int from_pin = -1;
    int to_node = -1;
    int to_pin = -1;
    int type = 0;

    bool operator<(const LinkKey& other) const {
        return std::tie(from_node, from_pin, to_node, to_pin, type) <
            std::tie(other.from_node, other.from_pin, other.to_node,
                     other.to_pin, other.type);
    }
};

std::set<LinkKey> ReadLinks(const DebugTraceRecord& trace) {
    std::set<LinkKey> links;
    const auto it = trace.payload.find("links");
    if (it == trace.payload.end() || !it->is_array()) {
        return links;
    }
    for (const auto& link : *it) {
        if (!link.is_object()) {
            continue;
        }
        links.insert({
            link.value("from_node", -1),
            link.value("from_pin", -1),
            link.value("to_node", -1),
            link.value("to_pin", -1),
            link.value("type", 0),
        });
    }
    return links;
}

nlohmann::json NodeIdentity(int id, const nlohmann::json& node) {
    return {
        {"node_id", id},
        {"name", node.value("name", std::string{})},
        {"type", node.value("type", -1)},
    };
}

nlohmann::json LinkIdentity(const LinkKey& link) {
    return {
        {"from_node", link.from_node},
        {"from_pin", link.from_pin},
        {"to_node", link.to_node},
        {"to_pin", link.to_pin},
        {"type", link.type},
    };
}

std::set<std::string> JsonObjectKeys(const nlohmann::json& value) {
    std::set<std::string> keys;
    if (!value.is_object()) {
        return keys;
    }
    for (auto it = value.begin(); it != value.end(); ++it) {
        keys.insert(it.key());
    }
    return keys;
}

struct ParameterChanges {
    nlohmann::json keys = nlohmann::json::array();
    size_t count = 0;
};

ParameterChanges ChangedParameterKeys(const nlohmann::json& baseline,
                                      const nlohmann::json& current) {
    const nlohmann::json empty = nlohmann::json::object();
    const auto& baseline_parameters =
        baseline.contains("parameters") && baseline["parameters"].is_object()
            ? baseline["parameters"]
            : empty;
    const auto& current_parameters =
        current.contains("parameters") && current["parameters"].is_object()
            ? current["parameters"]
            : empty;
    auto keys = JsonObjectKeys(baseline_parameters);
    const auto current_keys = JsonObjectKeys(current_parameters);
    keys.insert(current_keys.begin(), current_keys.end());

    ParameterChanges changed;
    for (const auto& key : keys) {
        const auto baseline_it = baseline_parameters.find(key);
        const auto current_it = current_parameters.find(key);
        if (baseline_it == baseline_parameters.end() ||
            current_it == current_parameters.end() ||
            *baseline_it != *current_it) {
            ++changed.count;
            if (changed.keys.size() < kMaxParameterKeysPerNode) {
                changed.keys.push_back(key);
            }
        }
    }
    return changed;
}

nlohmann::json CompiledConfigJson(
    const DebugReplayCompiledConfigSummary& config) {
    return {
        {"valid", config.valid},
        {"layer_count", config.layer_count},
        {"input_shape", config.input_shape},
        {"input_size", config.input_size},
        {"output_size", config.output_size},
        {"batch_size", config.batch_size},
        {"epochs", config.epochs},
        {"shuffle", config.shuffle},
        {"drop_last", config.drop_last},
        {"num_workers", config.num_workers},
        {"prefetch_factor", config.prefetch_factor},
        {"log_interval", config.log_interval},
        {"validation_freq", config.validation_freq},
        {"grad_accum_steps", config.grad_accum_steps},
        {"train_ratio", config.train_ratio},
        {"val_ratio", config.val_ratio},
        {"test_ratio", config.test_ratio},
        {"stratified", config.stratified},
        {"loss", config.loss},
        {"optimizer", config.optimizer},
        {"learning_rate", config.learning_rate},
        {"momentum", config.momentum},
        {"beta1", config.beta1},
        {"beta2", config.beta2},
        {"epsilon", config.epsilon},
        {"rmsprop_alpha", config.rmsprop_alpha},
        {"weight_decay", config.weight_decay},
        {"compiler_placement_fingerprint",
         config.compiler_placement_fingerprint},
        {"backend_placement_count", config.backend_placement_count},
        {"forbid_native_cpu_fallback",
         config.forbid_native_cpu_fallback},
    };
}

nlohmann::json BackendJson(const DebugRunExecutionSummary& execution) {
    return {
        {"requested_backend", execution.requested_backend},
        {"requested_device_id", execution.requested_device_id},
        {"effective_backend", execution.effective_backend},
        {"effective_device_id", execution.effective_device_id},
        {"effective_device_name", execution.effective_device_name},
        {"placement_fingerprint", execution.placement_fingerprint},
        {"residency_verdict", execution.residency_verdict},
        {"native_cpu_fallback_count",
         execution.native_cpu_fallback_count},
    };
}

nlohmann::json ObjectChanges(const nlohmann::json& baseline,
                             const nlohmann::json& current) {
    auto keys = JsonObjectKeys(baseline);
    const auto current_keys = JsonObjectKeys(current);
    keys.insert(current_keys.begin(), current_keys.end());

    nlohmann::json changes = nlohmann::json::array();
    for (const auto& key : keys) {
        const auto baseline_it = baseline.find(key);
        const auto current_it = current.find(key);
        if (baseline_it != baseline.end() && current_it != current.end() &&
            *baseline_it == *current_it) {
            continue;
        }
        changes.push_back({
            {"field", key},
            {"baseline", baseline_it == baseline.end()
                ? nlohmann::json(nullptr) : *baseline_it},
            {"current", current_it == current.end()
                ? nlohmann::json(nullptr) : *current_it},
        });
    }
    return changes;
}

} // namespace

DebugTraceRecord DebugTrainingGraphDiff::BuildTrace(
    const DebugRunStoreRecord& baseline,
    const DebugRunStoreRecord& current) const {
    DebugTraceRecord trace = DebugNodeTraceContract::Make(
        current.summary.run_id,
        -1,
        "Training Graph Diff",
        "TrainingDiagnostics",
        "TrainingGraphDiff",
        DebugTraceRole::CompileArtifact,
        {}, {}, "graph_diff", "PersistedDebugRuns", "captured");
    DebugNodeTraceContract::AttachDiagnosticContext(
        trace,
        "training_graph_diff",
        "DebugTrainingGraphDiff",
        "cyxwiz-engine/src/core/debug_training_graph_diff.cpp",
        "cyxwiz::DebugTrainingGraphDiff::BuildTrace");

    auto& payload = trace.payload;
    payload["training_graph_diff_schema"] = kSchema;
    payload["baseline_run_id"] = baseline.summary.run_id;
    payload["current_run_id"] = current.summary.run_id;
    payload["baseline_graph_hash"] = baseline.summary.graph_hash;
    payload["current_graph_hash"] = current.summary.graph_hash;
    payload["graph_hash_comparison_available"] =
        baseline.summary.graph_hash != 0 && current.summary.graph_hash != 0;
    payload["graph_hash_changed"] =
        baseline.summary.graph_hash != current.summary.graph_hash;
    payload["raw_graph_content_included"] = false;
    payload["raw_parameter_values_included"] = false;
    payload["link_identity_scope"] = "endpoints_pins_type";
    payload["link_ids_ignored"] = true;

    const auto* baseline_snapshot = FindGraphSnapshot(baseline);
    const auto* current_snapshot = FindGraphSnapshot(current);
    const bool graph_available = baseline_snapshot && current_snapshot;
    payload["graph_comparison_available"] = graph_available;

    nlohmann::json added_nodes = nlohmann::json::array();
    nlohmann::json removed_nodes = nlohmann::json::array();
    nlohmann::json changed_nodes = nlohmann::json::array();
    nlohmann::json added_links = nlohmann::json::array();
    nlohmann::json removed_links = nlohmann::json::array();
    size_t added_node_count = 0;
    size_t removed_node_count = 0;
    size_t changed_node_count = 0;
    size_t added_link_count = 0;
    size_t removed_link_count = 0;
    size_t retained_structural_details = 0;

    if (graph_available) {
        const auto baseline_nodes = ReadNodes(*baseline_snapshot);
        const auto current_nodes = ReadNodes(*current_snapshot);
        for (const auto& [id, node] : baseline_nodes) {
            const auto current_it = current_nodes.find(id);
            if (current_it == current_nodes.end()) {
                ++removed_node_count;
                PushStructuralDetail(removed_nodes, NodeIdentity(id, node),
                                     retained_structural_details);
                continue;
            }
            const auto changed_parameters = ChangedParameterKeys(
                node, current_it->second);
            const bool name_changed = node.value("name", std::string{}) !=
                current_it->second.value("name", std::string{});
            const bool type_changed = node.value("type", -1) !=
                current_it->second.value("type", -1);
            const bool pin_contract_changed =
                node.value("input_count", static_cast<size_t>(0)) !=
                    current_it->second.value(
                        "input_count", static_cast<size_t>(0)) ||
                node.value("output_count", static_cast<size_t>(0)) !=
                    current_it->second.value(
                        "output_count", static_cast<size_t>(0));
            if (name_changed || type_changed || pin_contract_changed ||
                changed_parameters.count > 0) {
                ++changed_node_count;
                PushStructuralDetail(changed_nodes, {
                    {"node_id", id},
                    {"baseline_name", node.value("name", std::string{})},
                    {"current_name", current_it->second.value(
                        "name", std::string{})},
                    {"baseline_type", node.value("type", -1)},
                    {"current_type", current_it->second.value("type", -1)},
                    {"name_changed", name_changed},
                    {"type_changed", type_changed},
                    {"pin_contract_changed", pin_contract_changed},
                    {"changed_parameter_keys", changed_parameters.keys},
                    {"parameter_change_count", changed_parameters.count},
                    {"parameter_keys_truncated",
                     changed_parameters.count > changed_parameters.keys.size()},
                }, retained_structural_details);
            }
        }
        for (const auto& [id, node] : current_nodes) {
            if (baseline_nodes.count(id) == 0) {
                ++added_node_count;
                PushStructuralDetail(added_nodes, NodeIdentity(id, node),
                                     retained_structural_details);
            }
        }

        const auto baseline_links = ReadLinks(*baseline_snapshot);
        const auto current_links = ReadLinks(*current_snapshot);
        for (const auto& link : baseline_links) {
            if (current_links.count(link) == 0) {
                ++removed_link_count;
                PushStructuralDetail(removed_links, LinkIdentity(link),
                                     retained_structural_details);
            }
        }
        for (const auto& link : current_links) {
            if (baseline_links.count(link) == 0) {
                ++added_link_count;
                PushStructuralDetail(added_links, LinkIdentity(link),
                                     retained_structural_details);
            }
        }
        payload["baseline_node_count"] = baseline_nodes.size();
        payload["current_node_count"] = current_nodes.size();
        payload["baseline_link_count"] = baseline_links.size();
        payload["current_link_count"] = current_links.size();
    }

    payload["added_nodes"] = std::move(added_nodes);
    payload["removed_nodes"] = std::move(removed_nodes);
    payload["changed_nodes"] = std::move(changed_nodes);
    payload["added_links"] = std::move(added_links);
    payload["removed_links"] = std::move(removed_links);
    payload["added_node_count"] = added_node_count;
    payload["removed_node_count"] = removed_node_count;
    payload["changed_node_count"] = changed_node_count;
    payload["added_link_count"] = added_link_count;
    payload["removed_link_count"] = removed_link_count;
    payload["structural_detail_limit"] = kMaxStructuralDetails;
    payload["retained_structural_detail_count"] =
        retained_structural_details;
    payload["structural_details_truncated"] =
        added_node_count + removed_node_count + changed_node_count +
            added_link_count + removed_link_count >
        retained_structural_details;

    const bool replay_available = baseline.replay_capsule.available &&
        current.replay_capsule.available;
    payload["replay_comparison_available"] = replay_available;
    payload["dataset_comparison_available"] = replay_available;
    payload["baseline_dataset_reference"] =
        baseline.replay_capsule.dataset_reference;
    payload["current_dataset_reference"] =
        current.replay_capsule.dataset_reference;
    const bool dataset_changed = replay_available &&
        baseline.replay_capsule.dataset_reference !=
            current.replay_capsule.dataset_reference;
    payload["dataset_changed"] = dataset_changed;
    payload["selected_sample_changed"] = replay_available &&
        baseline.replay_capsule.selected_sample_index !=
            current.replay_capsule.selected_sample_index;

    const bool config_available =
        baseline.replay_capsule.compiled_config.available &&
        current.replay_capsule.compiled_config.available;
    payload["compiled_config_comparison_available"] = config_available;
    const nlohmann::json config_changes = config_available
        ? ObjectChanges(
            CompiledConfigJson(baseline.replay_capsule.compiled_config),
            CompiledConfigJson(current.replay_capsule.compiled_config))
        : nlohmann::json::array();
    payload["compiled_config_changes"] = config_changes;
    payload["compiled_config_change_count"] = config_changes.size();

    const bool backend_available = baseline.summary.execution.available &&
        current.summary.execution.available;
    payload["backend_comparison_available"] = backend_available;
    const nlohmann::json backend_changes = backend_available
        ? ObjectChanges(BackendJson(baseline.summary.execution),
                        BackendJson(current.summary.execution))
        : nlohmann::json::array();
    payload["backend_changes"] = backend_changes;
    payload["backend_change_count"] = backend_changes.size();

    const size_t structural_change_count =
        payload["added_node_count"].get<size_t>() +
        payload["removed_node_count"].get<size_t>() +
        payload["changed_node_count"].get<size_t>() +
        payload["added_link_count"].get<size_t>() +
        payload["removed_link_count"].get<size_t>();
    const bool graph_hash_changed =
        payload["graph_hash_changed"].get<bool>();
    const bool selected_sample_changed =
        payload["selected_sample_changed"].get<bool>();
    const bool differences_found = structural_change_count > 0 ||
        dataset_changed || selected_sample_changed ||
        !config_changes.empty() || !backend_changes.empty() ||
        (payload["graph_hash_comparison_available"].get<bool>() &&
         graph_hash_changed);
    const bool comparison_available = graph_available || replay_available ||
        config_available || backend_available;
    payload["structural_change_count"] = structural_change_count;
    payload["differences_found"] = differences_found;
    payload["comparison_available"] = comparison_available;
    payload["success"] = true;
    if (!comparison_available) {
        payload["comparison_outcome"] = "unobserved";
        payload["comparison_reason"] =
            "Neither run contains comparable graph, replay, or backend evidence.";
        trace.status = "unobserved";
    } else if (differences_found) {
        payload["comparison_outcome"] = "changed";
        trace.status = "changed";
    } else {
        payload["comparison_outcome"] = "unchanged";
        trace.status = "unchanged";
    }
    return trace;
}

} // namespace cyxwiz
