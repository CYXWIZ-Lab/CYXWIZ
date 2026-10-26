#include "training_input_preparation.h"
#include "training_input_graph.h"

#include "materialization_memory_guard.h"
#include "node_executors/pipeline_operator_factory.h"
#include "pipeline_runtime_capabilities.h"

#include <spdlog/spdlog.h>

#include <set>

namespace cyxwiz {

using namespace training_input;

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
    try {
        (void)op->Apply(source_table);
    } catch (const EstimatePublished&) {
    }
    return out;
}

}  // namespace cyxwiz
