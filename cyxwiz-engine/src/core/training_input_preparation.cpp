#include "training_input_preparation.h"
#include "training_input_graph.h"

#include "materialization_memory_guard.h"
#include "node_executors/pipeline_operator_factory.h"
#include "pipeline_runtime_capabilities.h"
#include "sparse_feature_dataset.h"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <cctype>
#include <set>

namespace cyxwiz {

using namespace training_input;

std::optional<TrainingSplitPolicy> TrainingSplitPolicyOf(const std::vector<gui::MLNode>& nodes,
                                                         const std::vector<gui::NodeLink>& links) {
    const gui::MLNode* split = nullptr;
    const gui::MLNode* loader = nullptr;
    for (const auto& node : nodes) {
        if (node.type == gui::NodeType::TimeSeriesWindow || node.type == gui::NodeType::TimeSeriesSplit) return std::nullopt;
        if (node.type == gui::NodeType::DataSplit && !split) split = &node;
        if (node.type == gui::NodeType::DataLoader && !loader) loader = &node;
    }
    if (!split) return std::nullopt;

    const auto param = [](const gui::MLNode& node, const char* key) -> std::string {
        const auto it = node.parameters.find(key);
        return it == node.parameters.end() ? std::string() : it->second;
    };
    const auto number = [&param](const gui::MLNode& node, const char* key, float fallback) {
        try {
            const std::string value = param(node, key);
            return value.empty() ? fallback : std::stof(value);
        } catch (...) {
            return fallback;
        }
    };
    TrainingSplitPolicy policy;
    policy.train = number(*split, "train_ratio", policy.train);
    policy.val = number(*split, "val_ratio", policy.val);
    policy.test = number(*split, "test_ratio", policy.test);
    policy.seed = static_cast<uint32_t>(std::max(0.0f, number(*split, "seed", 42.0f)));
    const std::string stratified = param(*split, "stratified");
    policy.stratified = stratified == "true" || stratified == "1";
    policy.shuffle = !loader || param(*loader, "shuffle").empty() || param(*loader, "shuffle") == "true";
    policy.label_column = param(*split, "label_column");
    std::vector<const gui::MLNode*> training_sources;
    for (const int id : Upstream(links, TrainingInputFeeder(nodes, links)))
        if (const gui::MLNode* node = FindNode(nodes, id); node && IsSource(*node)) training_sources.push_back(node);
    if (policy.label_column.empty())
        for (const gui::MLNode* node : training_sources) policy.label_column = param(*node, "label_column");

    // Supplied Validation / Test inputs: a Data Input straight on pin 1 / 2
    // (the compiler refuses steps there).
    for (const auto& link : links) {
        if (link.to_node != split->id) continue;
        const int pin = PinIndex(split->inputs, link.to_pin);
        if (pin != 1 && pin != 2) continue;
        const gui::MLNode* source = FindNode(nodes, link.from_node);
        if (!source || !IsSource(*source)) return std::nullopt;
        (pin == 1 ? policy.validation_input : policy.test_input) = source->id;
        (pin == 1 ? policy.validation_label_column : policy.test_label_column) = param(*source, "label_column");
    }
    if (policy.Supplies()) {
        if (training_sources.size() != 1) return std::nullopt;
        policy.training_input = training_sources.front()->id;
        // The compiler's shares: a supplied role takes no share of the training rows.
        if (policy.validation_input >= 0) policy.val = 0.0f;
        if (policy.test_input >= 0) policy.test = 0.0f;
        policy.train = std::max(0.0f, 1.0f - policy.val - policy.test);
        return policy;
    }
    if (!policy.stratified && !policy.shuffle) return std::nullopt;
    return policy;
}

int TrainingInputFeeder(const std::vector<gui::MLNode>& nodes, const std::vector<gui::NodeLink>& links) {
    const auto entry_of = [&](gui::NodeType type) -> int {
        for (const auto& node : nodes) {
            if (node.type != type) continue;
            for (const auto& link : links) {
                if (link.to_node != node.id || PinIndex(node.inputs, link.to_pin) != 0) continue;
                const gui::MLNode* source = FindNode(nodes, link.from_node);
                return source && PinIndex(source->outputs, link.from_pin) == 0 ? link.from_node : -1;
            }
        }
        return -1;
    };
    const int feeder = entry_of(gui::NodeType::DataSplit);
    return feeder >= 0 ? feeder : entry_of(gui::NodeType::DataLoader);
}

namespace {
struct EstimatePublished {};
}  // namespace

StepMemoryEstimate EstimateFirstStepMemory(const std::vector<gui::MLNode>& nodes,
                                           const std::vector<gui::NodeLink>& links,
                                           int node,
                                           const std::shared_ptr<arrow::Table>& source_table,
                                           const MaterializationMemoryContext& memory_context) {
    StepMemoryEstimate out;
    if (!source_table) return out;
    const std::set<int> slice = Upstream(links, node);
    const gui::MLNode* first = nullptr;
    for (const auto& link : links) {
        const gui::MLNode* from = FindNode(nodes, link.from_node);
        if (!from || !IsSource(*from) || !slice.count(link.to_node)) continue;
        first = FindNode(nodes, link.to_node);
        break;
    }
    if (!first) return out;
    const auto support = ResolvePipelineRuntimeSupport(first->type);
    if (support.mode != PipelineRuntimeSupportMode::OperatorBacked) return out;
    auto op = PipelineOperatorFactory::Instance().Create(first->type);
    if (!op) return out;
    if (!op->Configure(first->parameters, out.error)) {
        out.error = "'" + first->name + "': " + out.error;
        return out;
    }
    PipelineOperatorExecutionContext context;
    context.memory = memory_context;
    context.stop_after_memory_preflight = true;
    op->SetExecutionContext(context);
    const auto snapshot = memory_context.snapshot_override.value_or(DetectMaterializationMemorySnapshot());
    op->SetProgressCallback([&out, first, snapshot, &memory_context](const PipelineOperatorProgress& event) {
        if (event.memory_risk_level.empty()) return;
        out.available = true;
        out.evidence = event;
        out.evidence.node_id = first->id;
        out.evidence.node_name = first->name;
        out.evidence.available_memory_bytes = snapshot.available_bytes;
        out.evidence.safe_memory_budget_bytes =
            SaturatingScaleBytes(snapshot.available_bytes, memory_context.policy.blocked_fraction);
        throw EstimatePublished{};
    });
    // A vectorizer set to Sparse publishes from its sparse path (Apply refuses
    // that setting). A step that fails before its estimate says why (a
    // missing fitted state, a bad column), instead of "no estimate".
    std::string format = first->parameters.count("output_format") ? first->parameters.at("output_format") : "";
    std::transform(format.begin(), format.end(), format.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    try {
        const arrow::Status status = format == "sparse" && op->SupportsSparseFeatureOutput()
                                         ? op->ApplySparse(source_table, "memory_estimate").status()
                                         : op->Apply(source_table).status();
        if (!status.ok() && !out.available) out.error = "'" + first->name + "': " + status.message();
    } catch (const EstimatePublished&) {
    } catch (const std::exception& error) {
        out.error = "'" + first->name + "': " + error.what();
    }
    return out;
}

}  // namespace cyxwiz
