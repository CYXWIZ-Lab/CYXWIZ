#pragma once

#include "backend_placement_capabilities.h"
#include "execution_device_context.h"
#include "graph_compiler.h"

#include <cstdint>
#include <iomanip>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace cyxwiz {

namespace ExecutionPlacementStatus {
inline constexpr const char* ArrayFire = "arrayfire";
inline constexpr const char* DeclaredHostBoundary = "declared_host_boundary";
} // namespace ExecutionPlacementStatus

struct ExecutionPlacementPlan {
    std::vector<BackendPlacementEntry> entries;
    std::string compiler_fingerprint;
    std::string fingerprint;
    std::string summary;
    std::vector<std::string> fatal_blockers;
    std::vector<std::string> strict_blockers;

    bool IsExecutable() const { return fatal_blockers.empty(); }
    bool IsStrictlyExecutable() const { return strict_blockers.empty(); }
    std::string FatalBlockerSummary() const {
        std::ostringstream out;
        for (size_t i = 0; i < fatal_blockers.size(); ++i) {
            if (i > 0) {
                out << "; ";
            }
            out << fatal_blockers[i];
        }
        return out.str();
    }
    std::string StrictBlockerSummary() const {
        std::ostringstream out;
        for (size_t i = 0; i < strict_blockers.size(); ++i) {
            if (i > 0) {
                out << "; ";
            }
            out << strict_blockers[i];
        }
        return out.str();
    }
};

inline void HashExecutionPlacementField(uint64_t& hash,
                                        const std::string& value) {
    constexpr uint64_t kFnvPrime = 1099511628211ull;
    for (const unsigned char ch : value) {
        hash ^= static_cast<uint64_t>(ch);
        hash *= kFnvPrime;
    }
    hash ^= static_cast<uint64_t>('\x1f');
    hash *= kFnvPrime;
}

inline std::string FingerprintPlacementEntries(
    const std::vector<BackendPlacementEntry>& placements) {
    uint64_t hash = 14695981039346656037ull;
    HashExecutionPlacementField(hash, std::to_string(placements.size()));
    for (const auto& placement : placements) {
        HashExecutionPlacementField(hash, std::to_string(placement.node_id));
        HashExecutionPlacementField(hash, placement.node_name);
        HashExecutionPlacementField(hash, placement.node_type);
        HashExecutionPlacementField(hash, placement.requested_backend);
        HashExecutionPlacementField(hash, placement.expected_backend);
        HashExecutionPlacementField(hash, placement.fallback_backend);
        HashExecutionPlacementField(hash, placement.status);
        HashExecutionPlacementField(hash, placement.reason_code);
        HashExecutionPlacementField(hash, placement.observation_device);
        HashExecutionPlacementField(hash, placement.observation_dtype);
        HashExecutionPlacementField(
            hash, placement.observation_shape_signature);
        HashExecutionPlacementField(
            hash, placement.observation_probe_outcome);
        HashExecutionPlacementField(hash, placement.observation_probe_scope);
    }

    std::ostringstream out;
    out << "placement:" << std::hex << std::setw(16) << std::setfill('0')
        << hash;
    return out.str();
}

inline std::string SummarizePlacementEntries(
    const std::vector<BackendPlacementEntry>& placements) {
    constexpr size_t kMaxSummaryChars = 900;
    std::ostringstream out;
    size_t summary_chars = 0;
    bool first = true;
    for (const auto& placement : placements) {
        std::string item = placement.node_name + ":" +
            placement.node_type + "=" + placement.expected_backend +
            "(" + placement.reason_code + ")";
        if (!first) {
            item = "; " + item;
        }
        if (summary_chars + item.size() > kMaxSummaryChars) {
            out << "; ...";
            break;
        }
        out << item;
        summary_chars += item.size();
        first = false;
    }
    return out.str();
}

inline BackendPlacementEntry MakeExecutionPlacementEntry(
    int node_id,
    const std::string& name,
    const std::string& type,
    const ExecutionDeviceContext& context,
    const std::string& status,
    const std::string& reason,
    const std::string& expected_backend,
    const std::string& detail) {
    BackendPlacementEntry placement;
    placement.node_id = node_id;
    placement.node_name = name;
    placement.node_type = type;
    placement.requested_backend = context.requested_backend.empty()
        ? "arrayfire"
        : context.requested_backend;
    placement.expected_backend = expected_backend.empty()
        ? context.effective_backend
        : expected_backend;
    placement.fallback_backend.clear();
    placement.status = status;
    placement.reason_code = reason;
    placement.observation_device = context.device_name;
    placement.observation_dtype = "float32";
    placement.observation_probe_outcome = "runtime_trace_declared";
    placement.observation_probe_scope = "execution_placement_plan";
    placement.explanation = detail;
    return placement;
}

inline bool IsCompiledModelLayerPlacement(
    const TrainingConfiguration& config,
    const BackendPlacementEntry& placement) {
    for (const auto& layer : config.layers) {
        if (layer.node_id == placement.node_id) {
            return true;
        }
    }
    return false;
}

inline bool HasCompiledPlacementForLayer(
    const TrainingConfiguration& config,
    const CompiledLayer& layer) {
    for (const auto& placement : config.backend_placements) {
        if (placement.node_id == layer.node_id) {
            return true;
        }
    }
    return false;
}

inline void AddStrictPlacementBlocker(ExecutionPlacementPlan& plan,
                                      const BackendPlacementEntry& placement) {
    if (placement.status == ExecutionPlacementStatus::ArrayFire ||
        placement.status == ExecutionPlacementStatus::DeclaredHostBoundary) {
        return;
    }
    plan.strict_blockers.push_back(
        placement.node_name + ":" + placement.node_type + "=" +
        placement.status + "(" + placement.reason_code + ")");
}

inline void AddFatalPlacementBlocker(ExecutionPlacementPlan& plan,
                                     const BackendPlacementEntry& placement) {
    if (placement.status != BackendPlacementStatus::Unsupported) {
        return;
    }
    plan.fatal_blockers.push_back(
        placement.node_name + ":" + placement.node_type + "=" +
        placement.status + "(" + placement.reason_code + ")");
}

inline void AppendExecutionPlacement(
    ExecutionPlacementPlan& plan,
    BackendPlacementEntry placement) {
    AddFatalPlacementBlocker(plan, placement);
    AddStrictPlacementBlocker(plan, placement);
    plan.entries.push_back(std::move(placement));
}

inline bool IsRegressionTrainingLoss(gui::NodeType type) {
    return type == gui::NodeType::MSELoss ||
           type == gui::NodeType::L1Loss ||
           type == gui::NodeType::SmoothL1Loss ||
           type == gui::NodeType::HuberLoss;
}

inline bool IsArrayFireTrainingLoss(gui::NodeType type) {
    switch (type) {
        case gui::NodeType::MSELoss:
        case gui::NodeType::CrossEntropyLoss:
        case gui::NodeType::FocalLoss:
        case gui::NodeType::BCELoss:
        case gui::NodeType::BCEWithLogits:
        case gui::NodeType::L1Loss:
        case gui::NodeType::SmoothL1Loss:
        case gui::NodeType::HuberLoss:
        case gui::NodeType::NLLLoss:
            return true;
        default:
            return false;
    }
}

inline bool IsNativeCpuTrainingLoss(gui::NodeType type) {
    return type == gui::NodeType::SoftDiceLoss ||
           type == gui::NodeType::TverskyLoss ||
           type == gui::NodeType::JaccardLoss;
}

inline bool IsSupportedTrainingLoss(gui::NodeType type) {
    return IsArrayFireTrainingLoss(type) || IsNativeCpuTrainingLoss(type);
}

inline bool IsSupportedTrainingOptimizer(gui::NodeType type) {
    switch (type) {
        case gui::NodeType::SGD:
        case gui::NodeType::Adam:
        case gui::NodeType::AdamW:
        case gui::NodeType::RMSprop:
        case gui::NodeType::Adagrad:
        case gui::NodeType::NAdam:
            return true;
        default:
            return false;
    }
}

inline bool IsSupportedTrainingMetricContract(
    const TrainingConfiguration& config) {
    if (!IsSupportedTrainingLoss(config.loss_type)) {
        return false;
    }
    if (config.target.value_kind == TargetValueKind::Unspecified) {
        return true;
    }
    if (config.target.value_kind == TargetValueKind::Continuous) {
        return IsRegressionTrainingLoss(config.loss_type);
    }
    if (config.target.value_kind == TargetValueKind::Categorical ||
        config.target.value_kind == TargetValueKind::TokenIds) {
        return !IsRegressionTrainingLoss(config.loss_type);
    }
    return false;
}

inline std::string TrainingLossPlacementName(
    const TrainingConfiguration& config) {
    return IsSupportedTrainingLoss(config.loss_type)
        ? config.GetLossName()
        : "Unsupported(" +
              std::to_string(static_cast<int>(config.loss_type)) + ")";
}

inline std::string TrainingOptimizerPlacementName(
    const TrainingConfiguration& config) {
    return IsSupportedTrainingOptimizer(config.optimizer_type)
        ? config.GetOptimizerName()
        : "Unsupported(" +
              std::to_string(static_cast<int>(config.optimizer_type)) + ")";
}

inline ExecutionPlacementPlan BuildExecutionPlacementPlan(
    const TrainingConfiguration& config,
    const ExecutionDeviceContext& context) {
    ExecutionPlacementPlan plan;
    plan.compiler_fingerprint =
        FingerprintPlacementEntries(config.backend_placements);

    const std::string backend = context.effective_backend.empty()
        ? "arrayfire_unknown"
        : context.effective_backend;
    plan.entries.reserve(
        config.backend_placements.size() + config.layers.size() + 9);

    for (const auto& compiler_placement : config.backend_placements) {
        auto placement = compiler_placement;
        if (IsCompiledModelLayerPlacement(config, placement) &&
            placement.node_type.rfind("ModelForward.", 0) != 0) {
            placement.node_type = "ModelForward." + placement.node_type;
        }
        if (placement.reason_code ==
                BackendPlacementReason::ArrayFireTensorOpCapable &&
            (placement.status == BackendPlacementStatus::Cpu ||
             placement.status == BackendPlacementStatus::Gpu)) {
            placement.requested_backend = context.requested_backend.empty()
                ? backend
                : context.requested_backend;
            placement.expected_backend = backend;
            placement.status = ExecutionPlacementStatus::ArrayFire;
            placement.fallback_backend.clear();
            placement.explanation +=
                " Executable placement resolved from the bound execution "
                "context.";
        }
        // Compiler capability entries may advertise a possible compatibility
        // route. The executable plan records intended placement; only an
        // observed runtime event may claim that fallback actually occurred.
        placement.fallback_backend.clear();
        AppendExecutionPlacement(plan, std::move(placement));
    }

    AppendExecutionPlacement(plan, MakeExecutionPlacementEntry(
        config.data_source_node_id >= 0 ? config.data_source_node_id : -1001,
        "dataset_ingress",
        "DatasetIngress",
        context,
        ExecutionPlacementStatus::DeclaredHostBoundary,
        "dataset_tensor_ingress",
        backend,
        "Dataset batches originate on host and are converted to tensors for "
        "the selected ArrayFire backend."));

    for (size_t i = 0; i < config.layers.size(); ++i) {
        const auto& layer = config.layers[i];
        if (HasCompiledPlacementForLayer(config, layer)) {
            continue;
        }
        const int node_id = layer.node_id >= 0
            ? layer.node_id
            : static_cast<int>(-2000 - static_cast<int>(i));
        const std::string name = layer.name.empty()
            ? "layer_" + std::to_string(i)
            : layer.name;
        AppendExecutionPlacement(plan, MakeExecutionPlacementEntry(
            node_id,
            name,
            std::string("ModelForward.") +
                backend_placement::LayerTypeName(layer.type),
            context,
            ExecutionPlacementStatus::ArrayFire,
            "model_forward_arrayfire",
            backend,
            "Model forward is assigned to the selected ArrayFire backend."));
    }

    const bool supported_loss = IsSupportedTrainingLoss(config.loss_type);
    const bool arrayfire_loss = IsArrayFireTrainingLoss(config.loss_type);
    AppendExecutionPlacement(plan, MakeExecutionPlacementEntry(
        config.loss_node_id >= 0 ? config.loss_node_id : -3001,
        "loss",
        "Loss." + TrainingLossPlacementName(config),
        context,
        !supported_loss
            ? BackendPlacementStatus::Unsupported
            : (arrayfire_loss
                   ? ExecutionPlacementStatus::ArrayFire
                   : BackendPlacementStatus::Cpu),
        !supported_loss
            ? "loss_unsupported"
            : (arrayfire_loss
                   ? "loss_arrayfire"
                   : "loss_native_cpu_compatibility"),
        !supported_loss
            ? "unavailable"
            : (arrayfire_loss ? backend : "native_cpu"),
        !supported_loss
            ? "TrainingExecutor has no configured loss implementation for "
              "this node type."
            : (arrayfire_loss
                   ? "Loss computation is assigned to the selected ArrayFire "
                     "backend."
                   : "This loss currently executes through its declared "
                     "native CPU compatibility implementation.")));

    const bool supported_metrics =
        IsSupportedTrainingMetricContract(config);
    const bool sequence_metrics =
        supported_metrics && config.sequence_batch.enabled;
    AppendExecutionPlacement(plan, MakeExecutionPlacementEntry(
        -3002,
        "metrics",
        !supported_metrics
            ? "Metrics.UnsupportedContract"
            : (sequence_metrics
                   ? "Metrics.SequenceTokenAccuracy"
                   : (UsesContinuousTargetMetrics(config)
                          ? "Metrics.Regression"
                          : "Metrics.Classification")),
        context,
        !supported_metrics
            ? BackendPlacementStatus::Unsupported
            : (sequence_metrics
                   ? BackendPlacementStatus::Cpu
                   : ExecutionPlacementStatus::ArrayFire),
        supported_metrics
            ? (sequence_metrics
                   ? "metrics_sequence_native_cpu_compatibility"
                   : "metrics_arrayfire_scalar_reduction")
            : "metrics_loss_target_contract_unsupported",
        !supported_metrics
            ? "unavailable"
            : (sequence_metrics ? "native_cpu" : backend),
        !supported_metrics
            ? "The configured loss and target value kind do not select a "
              "supported training metric contract."
            : (sequence_metrics
                   ? "Sequence token accuracy currently materializes logits "
                     "and targets for its declared native CPU compatibility "
                     "implementation."
                   : "Metrics use ArrayFire reductions with bounded scalar "
                     "reporting.")));
    AppendExecutionPlacement(plan, MakeExecutionPlacementEntry(
        -3003,
        "backward",
        "Backward",
        context,
        ExecutionPlacementStatus::ArrayFire,
        "backward_arrayfire",
        backend,
        "Backward and gradient tensor operations are assigned to the selected "
        "ArrayFire backend."));
    AppendExecutionPlacement(plan, MakeExecutionPlacementEntry(
        -3004,
        "gradient_accumulation",
        "GradientTransform.Accumulate",
        context,
        ExecutionPlacementStatus::ArrayFire,
        "gradient_accumulation_arrayfire",
        backend,
        "Per-parameter gradients are cloned or added as device-current "
        "ArrayFire tensors."));
    AppendExecutionPlacement(plan, MakeExecutionPlacementEntry(
        -3005,
        "gradient_averaging",
        "GradientTransform.Average",
        context,
        ExecutionPlacementStatus::ArrayFire,
        "gradient_averaging_arrayfire",
        backend,
        "Accumulated gradients are scaled on the selected ArrayFire backend "
        "before the optimizer step."));

    const bool supported_optimizer =
        IsSupportedTrainingOptimizer(config.optimizer_type);
    const std::string optimizer_name =
        TrainingOptimizerPlacementName(config);
    AppendExecutionPlacement(plan, MakeExecutionPlacementEntry(
        config.optimizer_node_id >= 0 ? config.optimizer_node_id : -3006,
        "optimizer_state",
        "OptimizerState." + optimizer_name,
        context,
        supported_optimizer
            ? ExecutionPlacementStatus::ArrayFire
            : BackendPlacementStatus::Unsupported,
        supported_optimizer
            ? "optimizer_state_arrayfire"
            : "optimizer_unsupported",
        supported_optimizer ? backend : "unavailable",
        supported_optimizer
            ? "Optimizer state tensors are created and updated on the "
              "selected ArrayFire backend."
            : "TrainingExecutor has no optimizer implementation for this "
              "node type."));
    AppendExecutionPlacement(plan, MakeExecutionPlacementEntry(
        config.optimizer_node_id >= 0 ? config.optimizer_node_id : -3007,
        "optimizer_update",
        "OptimizerUpdate." + optimizer_name,
        context,
        supported_optimizer
            ? ExecutionPlacementStatus::ArrayFire
            : BackendPlacementStatus::Unsupported,
        supported_optimizer
            ? "optimizer_update_arrayfire"
            : "optimizer_unsupported",
        supported_optimizer ? backend : "unavailable",
        supported_optimizer
            ? "Parameter updates are assigned to the selected ArrayFire "
              "backend; any runtime native CPU fallback is recorded as an "
              "event."
            : "TrainingExecutor has no optimizer implementation for this "
              "node type."));
    AppendExecutionPlacement(plan, MakeExecutionPlacementEntry(
        -3008,
        "loss_scalar_readback",
        "OutputBoundary.LossScalar",
        context,
        ExecutionPlacementStatus::DeclaredHostBoundary,
        "loss_scalar_readback",
        "host_scalar",
        "One scalar loss value is read back for reporting."));

    plan.fingerprint = FingerprintPlacementEntries(plan.entries);
    plan.summary = SummarizePlacementEntries(plan.entries);
    return plan;
}

} // namespace cyxwiz
