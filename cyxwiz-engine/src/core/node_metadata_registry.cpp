#include "node_metadata_registry.h"
#include "pipeline_runtime_capabilities.h"
#include "simulation_runtime_capabilities.h"
#include "../gui/icons.h"
#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <string_view>
#include <utility>
#include <spdlog/spdlog.h>
#include <nlohmann/json.hpp>

namespace cyxwiz {

namespace {

NodeCategory ParseTemplateCategory(const std::string& category_name) {
    if (category_name == "I/O" || category_name == "Data Sources") return NodeCategory::DataSources;
    if (category_name == "DB" || category_name == "Database") return NodeCategory::Database;
    if (category_name == "Cloud" || category_name == "Cloud Storage") return NodeCategory::CloudStorage;
    if (category_name == "Manipulation" || category_name == "Data Transform") return NodeCategory::DataTransform;
    if (category_name == "Analytics") return NodeCategory::Analytics;
    if (category_name == "Visualization") return NodeCategory::Visualization;
    if (category_name == "ML Layers" || category_name == "Layers") return NodeCategory::Layers;
    if (category_name == "Activation") return NodeCategory::Activation;
    if (category_name == "Pooling") return NodeCategory::Pooling;
    if (category_name == "Normalization") return NodeCategory::Normalization;
    if (category_name == "Attention") return NodeCategory::Attention;
    if (category_name == "Recurrent") return NodeCategory::Recurrent;
    if (category_name == "ShapeOps" || category_name == "Shape Ops" || category_name == "Shape Operations") return NodeCategory::ShapeOps;
    if (category_name == "MergeOps" || category_name == "Merge Ops" || category_name == "Merge Operations") return NodeCategory::MergeOps;
    if (category_name == "Training") return NodeCategory::Training;
    if (category_name == "Regularization") return NodeCategory::Regularization;
    if (category_name == "Model I/O") return NodeCategory::ModelIO;
    if (category_name == "ML Services") return NodeCategory::MLServices;
    if (category_name == "ML Advanced" || category_name == "Explainability") return NodeCategory::Explainability;
    if (category_name == "Preprocessing") return NodeCategory::Preprocessing;
    if (category_name == "Data Pipeline") return NodeCategory::DataPipeline;
    if (category_name == "Text" || category_name == "Text Processing") return NodeCategory::TextProcessing;
    if (category_name == "Time Series") return NodeCategory::TimeSeries;
    if (category_name == "Audio") return NodeCategory::Audio;
    if (category_name == "JSON/XML") return NodeCategory::JsonXml;
    if (category_name == "DNN" || category_name == "DNN Models") return NodeCategory::DNN;
    if (category_name == "RL" || category_name == "Reinforcement Learning") return NodeCategory::RL;
    if (category_name == "Big Data") return NodeCategory::BigData;
    if (category_name == "Workflow") return NodeCategory::Workflow;
    if (category_name == "Widgets") return NodeCategory::Widgets;
    if (category_name == "Reporting") return NodeCategory::Reporting;
    if (category_name == "Signal") return NodeCategory::Signal;
    return NodeCategory::Utility;
}

PinType ParseTemplatePinType(const std::string& type_name) {
    if (type_name == "Tensor") return PinType::Tensor;
    if (type_name == "Labels") return PinType::Labels;
    if (type_name == "Parameters") return PinType::Parameters;
    if (type_name == "Loss") return PinType::Loss;
    if (type_name == "Optimizer") return PinType::Optimizer;
    return PinType::Dataset;
}

NodeMetadata WithPropertiesEditor(NodeMetadata metadata,
                                  NodePropertiesEditor editor) {
    metadata.properties_editor = editor;
    return metadata;
}

void UpsertSupportAxis(NodeMetadata& metadata,
                       std::string name,
                       std::string value,
                       bool supported,
                       std::string reason = {}) {
    auto existing = std::find_if(
        metadata.support_axes.begin(),
        metadata.support_axes.end(),
        [&name](const SupportAxisDefinition& axis) {
            return axis.name == name;
        });
    if (existing != metadata.support_axes.end()) {
        existing->value = std::move(value);
        existing->supported = supported;
        existing->reason = std::move(reason);
        return;
    }
    metadata.support_axes.push_back({
        std::move(name),
        std::move(value),
        supported,
        std::move(reason),
    });
}

void ApplySupportState(NodeMetadata& metadata,
                       std::string state,
                       bool supported,
                       std::string reason = {}) {
    UpsertSupportAxis(metadata, "Support State", state, supported, reason);
}

void ApplyRuntimeSupportAxes(NodeMetadata& metadata,
                             const PipelineRuntimeSupport& support,
                             std::string reason = {}) {
    UpsertSupportAxis(
        metadata,
        "Runtime",
        PipelineRuntimeSupportModeName(support.mode),
        support.mode != PipelineRuntimeSupportMode::FailClosed,
        reason);
    UpsertSupportAxis(
        metadata,
        "Fail Mode",
        PipelineRuntimeFailModeName(support.fail_mode),
        support.fail_mode == PipelineRuntimeFailMode::Real,
        reason);
    UpsertSupportAxis(
        metadata,
        "Pipeline Executor",
        support.pipeline_executor_supported ? "supported" : "unsupported",
        support.pipeline_executor_supported,
        reason);
    UpsertSupportAxis(
        metadata,
        "Materializer",
        PipelineMaterializerStorageSupportName(
            support.materializer_storage_support),
        support.materializer_arrow_table_supported,
        reason);
    UpsertSupportAxis(
        metadata,
        "Implementation Owner",
        PipelineRuntimeImplementationOwnerName(
            support.implementation_owner),
        support.implementation_owner != PipelineRuntimeImplementationOwner::None,
        reason);

    if (support.fail_mode == PipelineRuntimeFailMode::Real &&
        support.pipeline_executor_supported) {
        ApplySupportState(metadata, "real", true, reason);
    }
}

bool IsTrainingMetadataCategory(NodeCategory category) {
    switch (category) {
        case NodeCategory::Layers:
        case NodeCategory::Activation:
        case NodeCategory::Pooling:
        case NodeCategory::Normalization:
        case NodeCategory::Attention:
        case NodeCategory::Recurrent:
        case NodeCategory::ShapeOps:
        case NodeCategory::MergeOps:
        case NodeCategory::Training:
        case NodeCategory::Regularization:
            return true;
        default:
            return false;
    }
}

const char* DefaultWorkflowLane(NodeCategory category) {
    switch (category) {
        case NodeCategory::DataSources:
        case NodeCategory::Database:
        case NodeCategory::CloudStorage:
            return "data_ingestion";
        case NodeCategory::DataTransform:
        case NodeCategory::Preprocessing:
        case NodeCategory::DataPipeline:
        case NodeCategory::JsonXml:
        case NodeCategory::BigData:
            return "data_transformation";
        case NodeCategory::Analytics: return "data_analytics";
        case NodeCategory::Visualization: return "visualization";
        case NodeCategory::Layers:
        case NodeCategory::Activation:
        case NodeCategory::Pooling:
        case NodeCategory::Normalization:
        case NodeCategory::Attention:
        case NodeCategory::Recurrent:
        case NodeCategory::ShapeOps:
        case NodeCategory::MergeOps:
        case NodeCategory::Upsampling:
        case NodeCategory::DNN:
            return "deep_learning";
        case NodeCategory::Training:
        case NodeCategory::Regularization:
            return "training_control";
        case NodeCategory::ModelIO:
        case NodeCategory::MLServices:
            return "model_lifecycle";
        case NodeCategory::Explainability: return "explainability";
        case NodeCategory::TextProcessing: return "text";
        case NodeCategory::TimeSeries: return "time_series";
        case NodeCategory::Audio: return "audio";
        case NodeCategory::RL: return "reinforcement_learning";
        case NodeCategory::Workflow: return "workflow";
        case NodeCategory::Widgets: return "widgets";
        case NodeCategory::Reporting: return "reporting";
        case NodeCategory::Utility: return "utility";
        case NodeCategory::Signal: return "simulation";
        case NodeCategory::Plugin: return "plugin_external";
        case NodeCategory::Unknown: return "unclassified";
    }
    return "unclassified";
}

} // namespace

// Category display order
const std::vector<NodeCategory> NodeMetadataRegistry::category_order_ = {
    // Data I/O
    NodeCategory::DataSources,
    NodeCategory::Database,
    NodeCategory::CloudStorage,
    NodeCategory::DataTransform,

    // Analytics & Visualization
    NodeCategory::Analytics,
    NodeCategory::Visualization,

    // ML Layers
    NodeCategory::Layers,
    NodeCategory::Activation,
    NodeCategory::Normalization,
    NodeCategory::Pooling,
    NodeCategory::Attention,
    NodeCategory::Recurrent,
    NodeCategory::Upsampling,
    NodeCategory::ShapeOps,
    NodeCategory::MergeOps,

    // Training & Models
    NodeCategory::Training,
    NodeCategory::Regularization,
    NodeCategory::ModelIO,
    NodeCategory::MLServices,
    NodeCategory::Explainability,

    // Data Processing
    NodeCategory::Preprocessing,
    NodeCategory::DataPipeline,
    NodeCategory::TextProcessing,
    NodeCategory::TimeSeries,
    NodeCategory::Audio,
    NodeCategory::JsonXml,

    // Specialized
    NodeCategory::DNN,
    NodeCategory::RL,
    NodeCategory::BigData,

    // Workflow & UI
    NodeCategory::Workflow,
    NodeCategory::Widgets,
    NodeCategory::Reporting,
    NodeCategory::Utility,
    NodeCategory::Signal,

    NodeCategory::Plugin
};

const char* GetCategoryIcon(NodeCategory category) {
    switch (category) {
        // Data I/O
        case NodeCategory::DataSources:     return ICON_FA_FILE_IMPORT;
        case NodeCategory::Database:        return ICON_FA_DATABASE;
        case NodeCategory::CloudStorage:    return ICON_FA_CLOUD;
        case NodeCategory::DataTransform:   return ICON_FA_WAND_MAGIC_SPARKLES;

        // Analytics & Visualization
        case NodeCategory::Analytics:       return ICON_FA_MAGNIFYING_GLASS_CHART;
        case NodeCategory::Visualization:   return ICON_FA_CHART_PIE;

        // ML Layers
        case NodeCategory::Layers:          return ICON_FA_LAYER_GROUP;
        case NodeCategory::Activation:      return ICON_FA_BOLT;
        case NodeCategory::Normalization:   return ICON_FA_SCALE_BALANCED;
        case NodeCategory::Pooling:         return ICON_FA_COMPRESS;
        case NodeCategory::Attention:       return ICON_FA_BULLSEYE;
        case NodeCategory::Recurrent:       return ICON_FA_REPEAT;
        case NodeCategory::Upsampling:      return ICON_FA_EXPAND;
        case NodeCategory::ShapeOps:        return ICON_FA_CUBES;
        case NodeCategory::MergeOps:        return ICON_FA_CODE_BRANCH;

        // Training & Models
        case NodeCategory::Training:        return ICON_FA_GRADUATION_CAP;
        case NodeCategory::Regularization:  return ICON_FA_SHIELD_HALVED;
        case NodeCategory::ModelIO:         return ICON_FA_FLOPPY_DISK;
        case NodeCategory::MLServices:      return ICON_FA_CLOUD_ARROW_UP;
        case NodeCategory::Explainability:  return ICON_FA_LIGHTBULB;

        // Data Processing
        case NodeCategory::Preprocessing:   return ICON_FA_FILTER;
        case NodeCategory::DataPipeline:    return ICON_FA_BARS;
        case NodeCategory::TextProcessing:  return ICON_FA_ALIGN_LEFT;
        case NodeCategory::TimeSeries:      return ICON_FA_CHART_LINE;
        case NodeCategory::Audio:           return ICON_FA_WAVE_SQUARE;
        case NodeCategory::JsonXml:         return ICON_FA_CODE;

        // Specialized
        case NodeCategory::DNN:             return ICON_FA_BRAIN;
        case NodeCategory::RL:              return ICON_FA_ROCKET;
        case NodeCategory::BigData:         return ICON_FA_SERVER;

        // Workflow & UI
        case NodeCategory::Workflow:        return ICON_FA_DIAGRAM_PROJECT;
        case NodeCategory::Widgets:         return ICON_FA_SLIDERS;
        case NodeCategory::Reporting:       return ICON_FA_FILE_EXPORT;
        case NodeCategory::Utility:         return ICON_FA_TOOLBOX;
        case NodeCategory::Signal:          return ICON_FA_WAVE_SQUARE;

        case NodeCategory::Plugin:          return ICON_FA_PLUG;
        default:                            return ICON_FA_CUBE;
    }
}

NodeMetadataRegistry& NodeMetadataRegistry::Instance() {
    static NodeMetadataRegistry instance;
    return instance;
}

void NodeMetadataRegistry::Initialize() {
    if (initialized_) return;
    std::lock_guard<std::mutex> lock(mutex_);
    if (initialized_) return;
    spdlog::info("NodeMetadataRegistry: Initializing node metadata...");
    InitializeBuiltinNodes();

    // Load template nodes from JSON files
    // Try to find templates in resources directory relative to executable
    std::filesystem::path templates_path = "resources/node_templates";
    if (std::filesystem::exists(templates_path)) {
        LoadTemplates(templates_path.string());
    }

    // Apply owner/support/lane truth after optional resource templates are in
    // the registry so generated inventory covers both sources consistently.
    ApplyRuntimeCapabilityStatus();

    LoadUserPreferences();
    initialized_ = true;
    spdlog::info("NodeMetadataRegistry: Registered {} nodes", metadata_.size());
}

void NodeMetadataRegistry::InitializeBuiltinNodes() {
    InitializeDataSourceNodes();
    InitializeDataTransformNodes();
    InitializeAnalyticsNodes();
    InitializeLayerNodes();
    InitializeActivationNodes();
    InitializeTrainingNodes();
    InitializeDNNNodes();
    InitializeTextNodes();
    InitializeTimeSeriesNodes();
    InitializeAudioNodes();
    InitializeRLNodes();
    InitializeExportNodes();
    InitializeLinearAlgebraNodes();
    InitializeTimeSeriesAnalysisNodes();
    InitializeStatisticsNodes();
    InitializeInterpretationNodes();
    InitializeOptimizationNodes();
    InitializeAdditionalTextNodes();
    InitializeVisualizationNodes();
    InitializeCatalogPreviewNodes();

    InitializeKNIMENodes();
    InitializeUtilityNodes();
}

void NodeMetadataRegistry::ApplyRuntimeCapabilityStatus() {
    const auto has_runtime_axis = [](const NodeMetadata& metadata) {
        return std::any_of(
            metadata.support_axes.begin(),
            metadata.support_axes.end(),
            [](const SupportAxisDefinition& axis) {
                return axis.name == "Runtime";
            });
    };

    for (const auto& capability : GetPipelineOperatorRuntimeCapabilities()) {
        auto it = metadata_.find(capability.node_type);
        if (it == metadata_.end()) {
            continue;
        }

        const auto support =
            ResolvePipelineRuntimeSupport(capability.legacy_type_name);
        ApplyRuntimeSupportAxes(it->second, support);
    }

    for (const auto& capability : GetPipelineLegacyRuntimeCapabilities()) {
        if (!capability.node_type.has_value()) {
            continue;
        }

        auto it = metadata_.find(*capability.node_type);
        if (it == metadata_.end()) {
            continue;
        }
        if (has_runtime_axis(it->second)) {
            continue;
        }

        const auto support =
            ResolvePipelineRuntimeSupport(capability.legacy_type_name);
        ApplyRuntimeSupportAxes(it->second, support);
    }

    for (const auto& capability : GetPipelineFailClosedRuntimeCapabilities()) {
        auto metadata_node_type = capability.metadata_node_type;
        if (!metadata_node_type.has_value()) {
            metadata_node_type = capability.node_type;
        }
        if (!metadata_node_type.has_value()) {
            continue;
        }

        auto it = metadata_.find(*metadata_node_type);
        if (it == metadata_.end()) {
            continue;
        }

        auto& metadata = it->second;
        if (capability.blocks_metadata_status) {
            metadata.status = NodeImplementationStatus::Template;
            metadata.badge = "Blocked";
        }

        const std::string reason =
            capability.reason != nullptr ? capability.reason : "";
        const auto support =
            ResolvePipelineRuntimeSupport(capability.legacy_type_name);
        ApplyRuntimeSupportAxes(metadata, support, reason);
        ApplySupportState(
            metadata,
            capability.blocks_metadata_status ? "blocked" : "partial",
            !capability.blocks_metadata_status,
            reason);
    }

    const auto apply_training_backend_status =
        [this](gui::NodeType node_type) {
            auto it = metadata_.find(node_type);
            if (it == metadata_.end()) {
                return;
            }

            const auto support = ResolvePipelineTrainingBackendSupport(node_type);
            if (support.mode == PipelineTrainingBackendSupportMode::Allowed) {
                return;
            }

            auto& metadata = it->second;
            metadata.status = NodeImplementationStatus::Template;
            metadata.badge = "Blocked";

            const std::string reason =
                support.reason != nullptr ? support.reason : "";
            UpsertSupportAxis(
                metadata,
                "Training Backend",
                PipelineTrainingBackendSupportModeName(support.mode),
                false,
                reason);
            if (support.mode ==
                PipelineTrainingBackendSupportMode::UnsupportedSequentialModelLayer) {
                UpsertSupportAxis(
                    metadata,
                    "Model Builder",
                    "unsupported",
                    false,
                    reason);
            }
            if (support.mode ==
                PipelineTrainingBackendSupportMode::UnsupportedTrainingControl) {
                UpsertSupportAxis(
                    metadata,
                    "Training Role",
                    PipelineTrainingSupportRoleName(
                        PipelineTrainingSupportRole::TrainingControl),
                    false,
                    reason);
            }
            if (support.mode ==
                PipelineTrainingBackendSupportMode::UnsupportedTrainingWorkflow) {
                UpsertSupportAxis(
                    metadata,
                    "Training Role",
                    PipelineTrainingSupportRoleName(
                        PipelineTrainingSupportRole::TrainingWorkflow),
                    false,
                    reason);
            }
            UpsertSupportAxis(
                metadata,
                "Compile",
                support.compile_supported ? "supported" : "unsupported",
                support.compile_supported,
                reason);
            UpsertSupportAxis(
                metadata,
                "Training",
                support.training_supported ? "supported" : "unsupported",
                support.training_supported,
                reason);
            const bool workflow_unowned =
                support.mode == PipelineTrainingBackendSupportMode::
                                    UnsupportedTrainingWorkflow;
            UpsertSupportAxis(
                metadata,
                "Implementation Owner",
                workflow_unowned ? "unowned_training_workflow"
                                 : "training_backend",
                !workflow_unowned,
                reason);

            ApplySupportState(metadata, "blocked", false, reason);
        };

    for (const auto& capability :
         GetPipelineUnsupportedSequentialModelLayerCapabilities()) {
        apply_training_backend_status(capability.node_type);
    }

    for (const auto& capability :
         GetPipelineUnsupportedTrainingControlCapabilities()) {
        apply_training_backend_status(capability.node_type);
    }

    for (const auto& capability :
         GetPipelineUnsupportedTrainingWorkflowCapabilities()) {
        apply_training_backend_status(capability.node_type);
    }

    for (const auto& capability :
         GetPipelineSupportedTrainingBackendCapabilities()) {
        auto it = metadata_.find(capability.node_type);
        if (it == metadata_.end()) {
            continue;
        }

        const auto support =
            ResolvePipelineTrainingBackendSupport(capability.node_type);
        if (support.mode != PipelineTrainingBackendSupportMode::Allowed) {
            continue;
        }

        auto& metadata = it->second;
        const std::string reason =
            support.reason != nullptr ? support.reason : "";
        UpsertSupportAxis(
            metadata,
            "Training Backend",
            PipelineTrainingBackendSupportModeName(support.mode),
            true,
            reason);
        UpsertSupportAxis(
            metadata,
            "Compile",
            support.compile_supported ? "supported" : "unsupported",
            support.compile_supported,
            reason);
        UpsertSupportAxis(
            metadata,
            "Training",
            support.training_supported ? "supported" : "unsupported",
            support.training_supported,
            reason);
        UpsertSupportAxis(
            metadata,
            "Implementation Owner",
            "training_backend",
            true,
            reason);

        ApplySupportState(metadata, "real", true, reason);
    }

    for (const auto& capability :
         GetPipelineSupportedTrainingRoleCapabilities()) {
        auto it = metadata_.find(capability.node_type);
        if (it == metadata_.end()) {
            continue;
        }

        auto& metadata = it->second;
        const char* role = PipelineTrainingSupportRoleName(capability.role);
        const char* reason =
            capability.reason != nullptr ? capability.reason : "";
        UpsertSupportAxis(metadata, "Training Role", role, true, reason);
        if (capability.role == PipelineTrainingSupportRole::ModelLayer) {
            UpsertSupportAxis(
                metadata,
                "Model Builder",
                "supported",
                true,
                reason);
        } else if (capability.role == PipelineTrainingSupportRole::Activation) {
            UpsertSupportAxis(
                metadata,
                "Activation",
                "supported",
                true,
                reason);
        } else if (capability.role == PipelineTrainingSupportRole::Loss) {
            UpsertSupportAxis(
                metadata,
                "Loss",
                "supported",
                true,
                reason);
        } else if (capability.role == PipelineTrainingSupportRole::Optimizer) {
            UpsertSupportAxis(
                metadata,
                "Optimizer",
                "supported",
                true,
                reason);
        } else if (capability.role == PipelineTrainingSupportRole::DataSource) {
            UpsertSupportAxis(
                metadata,
                "Data Source",
                "supported",
                true,
                reason);
        } else if (capability.role == PipelineTrainingSupportRole::Preprocessing) {
            UpsertSupportAxis(
                metadata,
                "Preprocessing",
                "supported",
                true,
                reason);
        }
        UpsertSupportAxis(metadata, "Compile", "supported", true, reason);
        UpsertSupportAxis(metadata, "Training", "supported", true, reason);
        const auto owner_it = std::find_if(
            metadata.support_axes.begin(),
            metadata.support_axes.end(),
            [](const SupportAxisDefinition& axis) {
                return axis.name == "Implementation Owner";
            });
        if (owner_it == metadata.support_axes.end() ||
            !owner_it->supported || owner_it->value == "none" ||
            owner_it->value == "unknown" ||
            owner_it->value == "unowned_training_workflow") {
            UpsertSupportAxis(
                metadata,
                "Implementation Owner",
                "training_backend",
                true,
                reason);
        }
        ApplySupportState(metadata, "real", true, reason);

    }

    const auto apply_task_type_guidance =
        [this](NodeType node_type, const char* task_type, const char* guidance) {
            auto it = metadata_.find(node_type);
            if (it == metadata_.end()) {
                return;
            }
            UpsertSupportAxis(
                it->second,
                "Task Type",
                task_type,
                true,
                guidance);
        };

    apply_task_type_guidance(
        NodeType::MSELoss,
        "regression",
        "Use for numeric targets and continuous-value prediction.");
    apply_task_type_guidance(
        NodeType::L1Loss,
        "regression",
        "Use for robust numeric-target regression.");
    apply_task_type_guidance(
        NodeType::SmoothL1Loss,
        "regression",
        "Use for robust numeric-target regression with smoother gradients.");
    apply_task_type_guidance(
        NodeType::HuberLoss,
        "regression",
        "Use for numeric-target regression when outliers should have limited influence.");
    apply_task_type_guidance(
        NodeType::CrossEntropyLoss,
        "multiclass_classification",
        "Use for mutually exclusive class labels with logits.");
    apply_task_type_guidance(
        NodeType::FocalLoss,
        "multiclass_classification",
        "Use for imbalanced mutually exclusive class labels with logits.");
    apply_task_type_guidance(
        NodeType::SoftDiceLoss,
        "segmentation",
        "Use for probability masks and same-shaped Float32 target masks.");
    apply_task_type_guidance(
        NodeType::TverskyLoss,
        "segmentation",
        "Use for imbalanced probability masks with tunable false-positive and false-negative penalties.");
    apply_task_type_guidance(
        NodeType::JaccardLoss,
        "segmentation",
        "Use for IoU-style probability masks and same-shaped Float32 target masks.");
    apply_task_type_guidance(
        NodeType::NLLLoss,
        "multiclass_classification",
        "Use for class labels when the model outputs log probabilities.");
    apply_task_type_guidance(
        NodeType::BCELoss,
        "binary_classification",
        "Use for binary targets when predictions are probabilities.");
    apply_task_type_guidance(
        NodeType::BCEWithLogits,
        "binary_classification",
        "Use for binary targets when the model outputs logits.");

    const auto apply_workflow_lane_guidance =
        [this](NodeType node_type, const char* lane, const char* guidance) {
            auto it = metadata_.find(node_type);
            if (it == metadata_.end()) {
                return;
            }
            UpsertSupportAxis(
                it->second,
                "Workflow Lane",
                lane,
                true,
                guidance);
        };

    apply_workflow_lane_guidance(
        NodeType::LinearRegressionNode,
        "classic_ml",
        "Table-path classical ML baseline for numeric regression.");
    apply_workflow_lane_guidance(
        NodeType::PolynomialRegressionNode,
        "classic_ml",
        "Table-path classical ML baseline for polynomial regression.");
    apply_workflow_lane_guidance(
        NodeType::DecisionTreeClassifier,
        "classic_ml",
        "Table-path classical ML classifier for numeric tabular features.");
    apply_workflow_lane_guidance(
        NodeType::RandomForestClassifier,
        "classic_ml",
        "Table-path classical ML ensemble classifier for numeric tabular features.");
    apply_workflow_lane_guidance(
        NodeType::GradientBoostingClassifier,
        "classic_ml",
        "Table-path boosted classical ML classifier for numeric tabular features.");
    apply_workflow_lane_guidance(
        NodeType::TreeModelPredictor,
        "classic_ml",
        "Table-path classical ML inference node for saved tree-family model artifacts.");
    apply_workflow_lane_guidance(
        NodeType::RegressionModelPredictor,
        "classic_ml",
        "Table-path inference node for fitted linear and polynomial regression artifacts.");

    const std::string simulation_reason =
        "GraphExecutor owns bounded scalar control evaluation for the live "
        "simulation lane; PipelineExecutor and the training backend reject it.";
    for (const auto& capability : kBuiltInSimulationRuntimeCapabilities) {
        auto it = metadata_.find(capability.node_type);
        if (it == metadata_.end()) continue;
        apply_workflow_lane_guidance(
            capability.node_type, "simulation", simulation_reason.c_str());
        UpsertSupportAxis(it->second, "Simulation Runtime", "supported", true,
                          simulation_reason);
        UpsertSupportAxis(it->second, "Implementation Owner", "graph_executor",
                          true, simulation_reason);
        ApplySupportState(it->second, "real", true, simulation_reason);
    }

    for (NodeType node_type : {
             NodeType::Dense,
             NodeType::Dropout,
             NodeType::BatchNorm,
             NodeType::LSTM,
             NodeType::GRU,
         }) {
        apply_workflow_lane_guidance(
            node_type,
            "deep_learning",
            "Training-backend deep learning architecture node.");
    }

    const auto has_workflow_lane = [](const NodeMetadata& metadata) {
        return std::any_of(
            metadata.support_axes.begin(),
            metadata.support_axes.end(),
            [](const SupportAxisDefinition& axis) {
                return axis.name == "Workflow Lane";
            });
    };

    for (const auto& capability : GetPipelineOperatorRuntimeCapabilities()) {
        auto it = metadata_.find(capability.node_type);
        if (it == metadata_.end() || has_workflow_lane(it->second)) {
            continue;
        }
        apply_workflow_lane_guidance(
            capability.node_type,
            "data_studio_analytics",
            "PipelineExecutor-backed Data Studio analytics or preprocessing node.");
    }

    for (auto& [node_type, metadata] : metadata_) {
        (void)node_type;
        if (has_workflow_lane(metadata)) continue;
        const char* lane = DefaultWorkflowLane(metadata.category);
        UpsertSupportAxis(
            metadata,
            "Workflow Lane",
            lane,
            std::string_view(lane) != "unclassified",
            "Category-derived fallback used only when no narrower runtime "
            "workflow lane is declared.");
    }

    const std::string ui_only_reason =
        "No graph runtime or training backend owner is registered; this node "
        "is currently a UI/panel workflow surface.";
    for (auto& [node_type, metadata] : metadata_) {
        const bool has_implementation_owner = std::any_of(
            metadata.support_axes.begin(),
            metadata.support_axes.end(),
            [](const SupportAxisDefinition& axis) {
                return axis.name == "Implementation Owner";
            });
        if (!metadata.IsImplemented() ||
            has_implementation_owner ||
            IsTrainingMetadataCategory(metadata.category)) {
            continue;
        }

        UpsertSupportAxis(
            metadata,
            "Implementation Owner",
            "ui_only",
            true,
            ui_only_reason);
        ApplySupportState(metadata, "partial", true, ui_only_reason);
    }
}

void NodeMetadataRegistry::RegisterNode(NodeMetadata metadata) {
    metadata_[metadata.type] = std::move(metadata);
}

const NodeMetadata* NodeMetadataRegistry::GetMetadata(NodeType type) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = metadata_.find(type);
    return it != metadata_.end() ? &it->second : nullptr;
}

std::vector<const NodeMetadata*> NodeMetadataRegistry::GetByCategory(NodeCategory category, bool include_templates) const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<const NodeMetadata*> result;
    for (const auto& [type, meta] : metadata_) {
        if (meta.category == category) {
            if (include_templates || meta.IsImplemented()) {
                result.push_back(&meta);
            }
        }
    }
    std::sort(result.begin(), result.end(), [](const NodeMetadata* a, const NodeMetadata* b) {
        if (a->usage_count != b->usage_count) return a->usage_count > b->usage_count;
        return a->name < b->name;
    });
    return result;
}

std::vector<const NodeMetadata*> NodeMetadataRegistry::GetAllMetadata() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<const NodeMetadata*> result;
    result.reserve(metadata_.size());
    for (const auto& [type, meta] : metadata_) {
        (void)type;
        result.push_back(&meta);
    }
    std::sort(
        result.begin(),
        result.end(),
        [](const NodeMetadata* lhs, const NodeMetadata* rhs) {
            const int lhs_type = static_cast<int>(lhs->type);
            const int rhs_type = static_cast<int>(rhs->type);
            if (lhs_type != rhs_type) {
                return lhs_type < rhs_type;
            }
            return lhs->name < rhs->name;
        });
    return result;
}

std::vector<const NodeMetadata*> NodeMetadataRegistry::Search(const std::string& query, bool include_templates) const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<const NodeMetadata*> result;
    if (query.empty()) return result;
    for (const auto& [type, meta] : metadata_) {
        if (!include_templates && meta.IsTemplate()) continue;
        if (MatchesQuery(meta, query)) result.push_back(&meta);
    }
    std::string lower_query = query;
    std::transform(lower_query.begin(), lower_query.end(), lower_query.begin(), ::tolower);
    std::sort(result.begin(), result.end(), [&lower_query](const NodeMetadata* a, const NodeMetadata* b) {
        std::string name_a = a->name, name_b = b->name;
        std::transform(name_a.begin(), name_a.end(), name_a.begin(), ::tolower);
        std::transform(name_b.begin(), name_b.end(), name_b.begin(), ::tolower);
        bool exact_a = (name_a == lower_query), exact_b = (name_b == lower_query);
        if (exact_a != exact_b) return exact_a > exact_b;
        bool starts_a = (name_a.find(lower_query) == 0), starts_b = (name_b.find(lower_query) == 0);
        if (starts_a != starts_b) return starts_a > starts_b;
        return a->usage_count > b->usage_count;
    });
    return result;
}

std::vector<NodeCategory> NodeMetadataRegistry::GetCategories() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<NodeCategory> result;
    for (auto category : category_order_) {
        for (const auto& [type, meta] : metadata_) {
            if (meta.category == category) { result.push_back(category); break; }
        }
    }
    return result;
}

std::vector<const NodeMetadata*> NodeMetadataRegistry::GetMostUsed(size_t count) const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<const NodeMetadata*> result;
    for (const auto& [type, meta] : metadata_) {
        if (meta.usage_count > 0 && meta.IsImplemented()) result.push_back(&meta);
    }
    std::sort(result.begin(), result.end(), [](const NodeMetadata* a, const NodeMetadata* b) {
        return a->usage_count > b->usage_count;
    });
    if (result.size() > count) result.resize(count);
    return result;
}

std::vector<const NodeMetadata*> NodeMetadataRegistry::GetRecent(size_t count) const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<const NodeMetadata*> result;
    for (size_t i = 0; i < std::min(count, recent_nodes_.size()); ++i) {
        auto it = metadata_.find(recent_nodes_[i]);
        if (it != metadata_.end()) result.push_back(&it->second);
    }
    return result;
}

std::vector<const NodeMetadata*> NodeMetadataRegistry::GetFavorites() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<const NodeMetadata*> result;
    for (NodeType type : favorites_) {
        auto it = metadata_.find(type);
        if (it != metadata_.end()) result.push_back(&it->second);
    }
    return result;
}

void NodeMetadataRegistry::RecordUsage(NodeType type) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = metadata_.find(type);
    if (it != metadata_.end()) { it->second.usage_count++; AddToRecent(type); }
}

void NodeMetadataRegistry::AddToRecent(NodeType type) {
    recent_nodes_.erase(std::remove(recent_nodes_.begin(), recent_nodes_.end(), type), recent_nodes_.end());
    recent_nodes_.insert(recent_nodes_.begin(), type);
    if (recent_nodes_.size() > MAX_RECENT) recent_nodes_.resize(MAX_RECENT);
}

void NodeMetadataRegistry::ToggleFavorite(NodeType type) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = favorites_.find(type);
    if (it != favorites_.end()) {
        favorites_.erase(it);
        auto meta_it = metadata_.find(type);
        if (meta_it != metadata_.end()) meta_it->second.is_favorite = false;
    } else {
        favorites_.insert(type);
        auto meta_it = metadata_.find(type);
        if (meta_it != metadata_.end()) meta_it->second.is_favorite = true;
    }
}

void NodeMetadataRegistry::LoadTemplates(const std::string& directory) {
    namespace fs = std::filesystem;

    if (!fs::exists(directory)) {
        spdlog::warn("NodeMetadataRegistry: Template directory does not exist: {}", directory);
        return;
    }

    int loaded_count = 0;

    // Resource template IDs must not depend on filesystem enumeration order.
    std::vector<fs::path> template_files;
    for (const auto& entry : fs::directory_iterator(directory)) {
        if (entry.path().extension() == ".json") {
            template_files.push_back(entry.path());
        }
    }
    std::sort(template_files.begin(), template_files.end());

    for (const auto& template_file : template_files) {
            try {
                std::ifstream file(template_file);
                if (!file.is_open()) {
                    spdlog::warn("NodeMetadataRegistry: Could not open template file: {}", template_file.string());
                    continue;
                }

                nlohmann::json root = nlohmann::json::parse(file);

                // Get category info
                std::string category_name = root.value("category", "Unknown");
                std::string category_description = root.value("description", "");

                // Parse templates array
                if (root.contains("templates") && root["templates"].is_array()) {
                    for (const auto& tmpl : root["templates"]) {
                        NodeMetadata meta;

                        // Basic info
                        meta.type = NodeType::Unknown;  // Template nodes use Unknown type
                        meta.name = tmpl.value("name", "Unknown");
                        meta.icon = ICON_FA_CUBE;  // Default icon for templates
                        meta.status = NodeImplementationStatus::Template;
                        meta.badge = tmpl.value("badge", "Coming Soon");  // Badge text for display

                        meta.category = ParseTemplateCategory(category_name);

                        // Keywords
                        if (tmpl.contains("keywords") && tmpl["keywords"].is_array()) {
                            for (const auto& kw : tmpl["keywords"]) {
                                meta.keywords.push_back(kw.get<std::string>());
                            }
                        }

                        // Documentation
                        meta.brief_description = tmpl.value("brief_description", "");
                        meta.help_text = tmpl.value("help_text", "");
                        meta.example_usage = tmpl.value("implementation_notes", "");  // Show implementation notes as example

                        // Ports - inputs
                        if (tmpl.contains("inputs") && tmpl["inputs"].is_array()) {
                            for (const auto& inp : tmpl["inputs"]) {
                                PortDefinition port;
                                port.name = inp.value("name", "Input");
                                port.type = ParseTemplatePinType(inp.value("type", "Dataset"));
                                port.required = inp.value("required", true);
                                port.description = inp.value("description", "");
                                meta.inputs.push_back(port);
                            }
                        }

                        // Ports - outputs
                        if (tmpl.contains("outputs") && tmpl["outputs"].is_array()) {
                            for (const auto& out : tmpl["outputs"]) {
                                PortDefinition port;
                                port.name = out.value("name", "Output");
                                port.type = ParseTemplatePinType(out.value("type", "Dataset"));
                                port.required = true;
                                port.description = out.value("description", "");
                                meta.outputs.push_back(port);
                            }
                        }

                        // Parameters
                        if (tmpl.contains("parameters") && tmpl["parameters"].is_array()) {
                            for (const auto& param : tmpl["parameters"]) {
                                ParameterDefinition pdef;
                                pdef.name = param.value("name", "param");
                                pdef.type = param.value("type", "string");
                                pdef.default_value = param.value("default", "");
                                pdef.description = param.value("description", "");
                                pdef.validation = param.value("validation", "");
                                pdef.display_name = param.value("display_name", param.value("label", ""));
                                pdef.group = param.value("group", "");
                                pdef.required = param.value("required", false);
                                pdef.advanced = param.value("advanced", false);

                                if (param.contains("enum_values") && param["enum_values"].is_array()) {
                                    for (const auto& ev : param["enum_values"]) {
                                        pdef.enum_values.push_back(ev.get<std::string>());
                                    }
                                }

                                meta.parameters.push_back(pdef);
                            }
                        }

                        // User votes for template prioritization
                        meta.user_votes = tmpl.value("user_votes", 0);

                        // Assign a unique NodeType ID for this template (starting at 10000)
                        // This ensures ImGui::PushID() gets a unique value for each template node
                        NodeType template_type = static_cast<NodeType>(10000 + loaded_count);
                        meta.type = template_type;

                        // Store the template with its unique type
                        metadata_[template_type] = meta;

                        loaded_count++;
                    }
                }
            } catch (const std::exception& e) {
                spdlog::error("NodeMetadataRegistry: Error parsing template file {}: {}", template_file.string(), e.what());
            }
    }

    spdlog::info("NodeMetadataRegistry: Loaded {} template nodes from {}", loaded_count, directory);
}
void NodeMetadataRegistry::SaveUserPreferences() {
    spdlog::debug("NodeMetadataRegistry: SaveUserPreferences not yet implemented");
}
void NodeMetadataRegistry::LoadUserPreferences() {
    spdlog::debug("NodeMetadataRegistry: LoadUserPreferences not yet implemented");
}

bool NodeMetadataRegistry::MatchesQuery(const NodeMetadata& metadata, const std::string& query) const {
    std::string lower_query = query;
    std::transform(lower_query.begin(), lower_query.end(), lower_query.begin(), ::tolower);
    std::string lower_name = metadata.name;
    std::transform(lower_name.begin(), lower_name.end(), lower_name.begin(), ::tolower);
    if (lower_name.find(lower_query) != std::string::npos) return true;
    for (const auto& keyword : metadata.keywords) {
        std::string lower_keyword = keyword;
        std::transform(lower_keyword.begin(), lower_keyword.end(), lower_keyword.begin(), ::tolower);
        if (lower_keyword.find(lower_query) != std::string::npos) return true;
    }
    std::string lower_desc = metadata.brief_description;
    std::transform(lower_desc.begin(), lower_desc.end(), lower_desc.begin(), ::tolower);
    if (lower_desc.find(lower_query) != std::string::npos) return true;
    return false;
}

// =============================================================================
// Data Source Nodes (I/O)
// =============================================================================
void NodeMetadataRegistry::InitializeCatalogPreviewNodes() {
    struct PreviewDefinition {
        NodeType type;
        NodeCategory category;
        const char* name;
        std::initializer_list<const char*> keywords;
    };

    const std::initializer_list<PreviewDefinition> previews = {
        {NodeType::TransformerEncoder, NodeCategory::Attention, "Transformer Encoder", {"transformer", "attention", "encoder"}},
        {NodeType::TransformerDecoder, NodeCategory::Attention, "Transformer Decoder", {"transformer", "attention", "decoder"}},
        {NodeType::PositionalEncoding, NodeCategory::Attention, "Positional Encoding", {"transformer", "position", "encoding"}},
        {NodeType::PReLU, NodeCategory::Activation, "PReLU", {"activation", "relu"}},
        {NodeType::ELU, NodeCategory::Activation, "ELU", {"activation"}},
        {NodeType::SELU, NodeCategory::Activation, "SELU", {"activation"}},
        {NodeType::Swish, NodeCategory::Activation, "Swish", {"activation"}},
        {NodeType::Mish, NodeCategory::Activation, "Mish", {"activation"}},
        {NodeType::Parameter, NodeCategory::Utility, "Parameter", {"parameter", "constant"}},
        {NodeType::SineWave, NodeCategory::Signal, "Sine Wave", {"signal", "sine", "wave"}},
        {NodeType::StepSignal, NodeCategory::Signal, "Step Signal", {"signal", "step"}},
        {NodeType::RampSignal, NodeCategory::Signal, "Ramp Signal", {"signal", "ramp"}},
        {NodeType::Augmentation, NodeCategory::Preprocessing, "Augmentation", {"augmentation", "transform"}},
        {NodeType::TensorReshape, NodeCategory::ShapeOps, "Tensor Reshape", {"tensor", "reshape", "legacy"}},
        {NodeType::Resize, NodeCategory::Preprocessing, "Resize", {"image", "resize"}},
        {NodeType::CenterCrop, NodeCategory::Preprocessing, "Center Crop", {"image", "crop"}},
        {NodeType::RandomCrop, NodeCategory::Preprocessing, "Random Crop", {"image", "crop", "augmentation"}},
        {NodeType::HorizontalFlip, NodeCategory::Preprocessing, "Horizontal Flip", {"image", "flip", "augmentation"}},
        {NodeType::VerticalFlip, NodeCategory::Preprocessing, "Vertical Flip", {"image", "flip", "augmentation"}},
        {NodeType::ImageRotate, NodeCategory::Preprocessing, "Image Rotate", {"image", "rotate", "augmentation"}},
        {NodeType::ColorJitter, NodeCategory::Preprocessing, "Color Jitter", {"image", "color", "augmentation"}},
        {NodeType::ImageGaussianBlur, NodeCategory::Preprocessing, "Image Gaussian Blur", {"image", "blur"}},
        {NodeType::Grayscale, NodeCategory::Preprocessing, "Grayscale", {"image", "grayscale"}},
        {NodeType::Subgraph, NodeCategory::Workflow, "Subgraph", {"workflow", "subgraph"}},
        {NodeType::DNNClassify, NodeCategory::DNN, "DNN Classify", {"dnn", "classification"}},
        {NodeType::DNNPoseEstimate, NodeCategory::DNN, "DNN Pose Estimate", {"dnn", "pose"}},
        {NodeType::DNNFaceDetect, NodeCategory::DNN, "DNN Face Detect", {"dnn", "face", "detection"}},
        {NodeType::DNNPreprocess, NodeCategory::DNN, "DNN Preprocess", {"dnn", "preprocess"}},
        {NodeType::PretrainedMobileNet, NodeCategory::DNN, "Pretrained MobileNet", {"pretrained", "mobilenet"}},
        {NodeType::PretrainedOpenPose, NodeCategory::DNN, "Pretrained OpenPose", {"pretrained", "pose"}},
        {NodeType::PretrainedFaceNet, NodeCategory::DNN, "Pretrained FaceNet", {"pretrained", "face"}},
        {NodeType::NonMaxSuppression, NodeCategory::DNN, "Non-Max Suppression", {"detection", "nms"}},
        {NodeType::ArgMax, NodeCategory::Utility, "ArgMax", {"tensor", "argmax"}},
        {NodeType::TopK, NodeCategory::Utility, "Top K", {"tensor", "topk"}},
        {NodeType::ThresholdFilter, NodeCategory::Preprocessing, "Threshold Filter", {"threshold", "filter"}},
        {NodeType::AudioAugmentation, NodeCategory::Audio, "Audio Augmentation", {"audio", "augmentation"}},
        {NodeType::RLTraining, NodeCategory::RL, "RL Training", {"reinforcement", "training"}},
        {NodeType::PivotTable, NodeCategory::DataTransform, "Pivot Table", {"table", "pivot"}},
        {NodeType::UnionTables, NodeCategory::DataTransform, "Union Tables", {"table", "union"}},
        {NodeType::CrossTabulation, NodeCategory::DataTransform, "Cross Tabulation", {"table", "crosstab"}},
        {NodeType::UMAPNode, NodeCategory::Analytics, "UMAP", {"umap", "dimension", "embedding"}},
        {NodeType::SVMRegressor, NodeCategory::Analytics, "SVM Regressor", {"svm", "regression"}},
        {NodeType::ImagePreprocessor, NodeCategory::Preprocessing, "Image Preprocessor", {"image", "preprocess"}},
        {NodeType::QualityAnalyzer, NodeCategory::Preprocessing, "Quality Analyzer", {"image", "quality"}},
        {NodeType::ImageFolderDataset, NodeCategory::DataSources, "Image Folder Dataset", {"image", "folder", "dataset"}},
        {NodeType::MNISTDataset, NodeCategory::DataSources, "MNIST Dataset", {"mnist", "dataset"}},
        {NodeType::CIFAR10Dataset, NodeCategory::DataSources, "CIFAR-10 Dataset", {"cifar", "dataset"}},
        {NodeType::HuggingFaceDataset, NodeCategory::DataSources, "Hugging Face Dataset", {"huggingface", "dataset", "hub"}},
        {NodeType::KaggleDataset, NodeCategory::DataSources, "Kaggle Dataset", {"kaggle", "dataset"}},
        {NodeType::AugmentationPreset, NodeCategory::Preprocessing, "Augmentation Preset", {"augmentation", "preset"}},
        {NodeType::GeometricTransform, NodeCategory::Preprocessing, "Geometric Transform", {"image", "geometry", "augmentation"}},
        {NodeType::ColorTransform, NodeCategory::Preprocessing, "Color Transform", {"image", "color", "augmentation"}},
        {NodeType::MorphologyTransform, NodeCategory::Preprocessing, "Morphology Transform", {"image", "morphology"}},
        {NodeType::AdvancedAugment, NodeCategory::Preprocessing, "Advanced Augment", {"image", "augmentation"}},
        {NodeType::LinePlot, NodeCategory::Visualization, "Line Plot", {"chart", "line", "plot"}},
        {NodeType::ScatterPlot, NodeCategory::Visualization, "Scatter Plot", {"chart", "scatter", "plot"}},
        {NodeType::Histogram, NodeCategory::Visualization, "Histogram", {"chart", "distribution"}},
        {NodeType::PieChart, NodeCategory::Visualization, "Pie Chart", {"chart", "pie"}},
        {NodeType::AreaPlot, NodeCategory::Visualization, "Area Plot", {"chart", "area", "plot"}},
        {NodeType::BoxPlot, NodeCategory::Visualization, "Box Plot", {"chart", "box", "distribution"}},
        {NodeType::ViolinPlot, NodeCategory::Visualization, "Violin Plot", {"chart", "violin", "distribution"}},
        {NodeType::ErrorBarPlot, NodeCategory::Visualization, "Error Bar Plot", {"chart", "error", "plot"}},
        {NodeType::StepPlot, NodeCategory::Visualization, "Step Plot", {"chart", "step", "plot"}},
        {NodeType::HexbinPlot, NodeCategory::Visualization, "Hexbin Plot", {"chart", "hexbin", "plot"}},
        {NodeType::Heatmap, NodeCategory::Visualization, "Heatmap", {"chart", "heatmap"}},
        {NodeType::ContourPlot, NodeCategory::Visualization, "Contour Plot", {"chart", "contour", "plot"}},
        {NodeType::Imshow, NodeCategory::Visualization, "Image Display", {"image", "imshow", "visualization"}},
        {NodeType::Plot3D, NodeCategory::Visualization, "3D Plot", {"chart", "3d", "plot"}},
        {NodeType::Scatter3D, NodeCategory::Visualization, "3D Scatter Plot", {"chart", "scatter", "3d"}},
        {NodeType::SurfacePlot, NodeCategory::Visualization, "Surface Plot", {"chart", "surface", "3d"}},
        {NodeType::WireframePlot, NodeCategory::Visualization, "Wireframe Plot", {"chart", "wireframe", "3d"}},
        {NodeType::PolarPlot, NodeCategory::Visualization, "Polar Plot", {"chart", "polar", "plot"}},
        {NodeType::QuiverPlot, NodeCategory::Visualization, "Quiver Plot", {"chart", "vector", "plot"}},
        {NodeType::StreamPlot, NodeCategory::Visualization, "Stream Plot", {"chart", "stream", "plot"}},
        {NodeType::SpectrogramPlot, NodeCategory::Visualization, "Spectrogram", {"audio", "spectrogram", "plot"}},
        {NodeType::NetworkGraph, NodeCategory::Visualization, "Network Graph", {"graph", "network", "visualization"}},
        {NodeType::PluginCustom, NodeCategory::Plugin, "Custom Plugin Node", {"plugin", "custom", "extension"}},
    };

    for (const auto& preview : previews) {
        if (metadata_.find(preview.type) != metadata_.end()) {
            continue;
        }

        NodeMetadata metadata;
        metadata.type = preview.type;
        metadata.category = preview.category;
        metadata.name = preview.name;
        metadata.icon = GetCategoryIcon(preview.category);
        metadata.keywords.assign(preview.keywords.begin(), preview.keywords.end());
        metadata.brief_description =
            std::string(preview.name) + " is planned and not yet available in the runtime.";
        metadata.status = NodeImplementationStatus::Template;
        metadata.badge = "Coming Soon";
        RegisterNode(std::move(metadata));
    }
}
void NodeMetadataRegistry::InitializeDataSourceNodes() {
    // ===== Smart I/O Nodes (Universal - replaces individual format nodes) =====
    RegisterNode(WithPropertiesEditor({NodeType::DataInput, NodeCategory::DataSources, "Data Input", ICON_FA_FILE_IMPORT,
        {"csv", "tsv", "parquet", "feather", "arrow", "ipc", "input", "load", "read", "import", "file"}, 0, false,
        "Configure and load a project dataset through the Data Input dialog",
        "The dialog owns source discovery, parsing options, schema inspection, "
        "column roles, preview, and loading. The node emits one Dataset artifact.",
        "",
        {}, {{"Dataset", PinType::Dataset, true,
              "Loaded dataset with source, schema, column-role, row-count, and backing-store metadata."}},
        {{"file_path", "file", "", "Initial file selected by the dialog", {}, "*.csv;*.tsv;*.parquet;*.feather;*.fea;*.arrow;*.ipc"},
         {"file_type", "enum", "auto", "Input format", {"auto", "csv", "tsv", "parquet", "feather", "arrow", "ipc"}, ""},
         {"configured", "bool", "false", "Whether the dialog has applied a source contract", {}, "", "", "", false, false,
          ParameterConsumption::UiOnly}},
        NodeImplementationStatus::Implemented, 0}, NodePropertiesEditor::Dialog));

    RegisterNode(WithPropertiesEditor({NodeType::DataOutput, NodeCategory::DataSources, "Data Output", ICON_FA_FILE_EXPORT,
        {"csv", "parquet", "output", "save", "write", "export", "file"}, 0, false,
        "Universal data exporter - supports CSV and Parquet", "", "",
        {{"Data", PinType::Dataset, true, "Input dataset"}}, {},
        {{"file_path", "file", "", "Output file", {}, "*.csv;*.parquet"},
         {"file_type", "enum", "csv", "Output format", {"csv", "parquet"}, ""},
         {"configured", "bool", "false", "Dialog configured", {}, "", "", "", false, false,
          ParameterConsumption::UiOnly}},
        NodeImplementationStatus::Implemented, 0}, NodePropertiesEditor::Dialog));

    RegisterNode({NodeType::DataLoader, NodeCategory::DataPipeline, "Data Loader", ICON_FA_DATABASE,
        {"batch", "loader", "dataloader", "shuffle", "epoch", "training"}, 0, false,
        "Creates runtime batchers from resolved dataset partitions", "", "",
        {{"Partitions", PinType::Dataset, true, "Resolved partition set from Data Split"}},
        {{"Data", PinType::Tensor, true, "Batched training features"},
         {"Labels", PinType::Labels, false, "Batched training labels"}},
        {{"epochs", "int", "10", "Training epochs", {}, "1-10000", "", "Training", false, false},
         {"batch_size", "int", "32", "Samples per batch", {}, "1-100000", "", "Training", false, false},
         {"shuffle", "bool", "true", "Shuffle training batches", {}, "", "", "Training", false, false},
         {"drop_last", "bool", "false", "Drop incomplete final batch", {}, "", "", "Training", false, false},
         {"num_workers", "int", "0", "Worker count for batch preparation", {}, "0-128", "", "Runtime", false, true},
         {"prefetch_factor", "int", "2", "Prefetch factor when workers are active", {}, "0-64", "", "Runtime", false, true},
         {"log_interval", "int", "10", "Training metric and log interval in batches", {}, "0-100000", "", "Training", false, true},
         {"validation_freq", "int", "1", "Validation frequency in epochs", {}, "1-10000", "", "Training", false, true},
         {"seed", "int", "42", "Batching random seed", {}, "0-2147483647", "", "Training", false, true},
         {"grad_accum_steps", "int", "1", "Gradient accumulation steps", {}, "1-10000", "", "Training", false, true},
         {"balance_classes", "bool", "false", "Apply training-only class balancing", {}, "", "", "Balancing", false, true},
         {"balance_mode", "enum", "none", "Class balancing mode", {"none", "oversample", "undersample", "weighted_sampler"}, "", "", "Balancing", false, true},
         {"balance_target", "string", "max", "Class balancing target", {}, "", "", "Balancing", false, true},
         {"balance_seed", "int", "42", "Class balancing seed", {}, "0-2147483647", "", "Balancing", false, true},
         {"save_best_checkpoint", "bool", "true", "Save best validation checkpoint", {}, "", "", "Checkpoint", false, true},
         {"early_stopping_patience", "int", "5", "Early stopping patience in validation checks", {}, "0-10000", "", "Checkpoint", false, true},
         {"checkpoint_dir", "directory", "", "Checkpoint output directory", {}, "", "", "Checkpoint", false, true}},
        NodeImplementationStatus::Implemented, 0});

    // DataSplit is a graph-level training configuration node. It has a
    // dedicated dialog and compiler support, so it must remain in the modern
    // metadata catalog used by both the Node Browser and the add-node search.
    RegisterNode({NodeType::DataSplit, NodeCategory::DataPipeline, "Data Split", ICON_FA_CODE_BRANCH,
        {"split", "train", "validation", "test", "partition", "stratified", "dataset"}, 0, false,
        "Resolves train, validation, and held-out test partition policy", "", "",
        {{"Training Dataset", PinType::Dataset, true, "Required Dataset asset used as the Training source"},
         {"Validation Dataset", PinType::Dataset, false, "Optional externally supplied Validation/Dev source"},
         {"Test Dataset", PinType::Dataset, false, "Optional externally supplied held-out Test source"}},
        {{"Partitions", PinType::Dataset, true, "Resolved Train/Validation/Test partition set plus manifest identity"}},
        {{"train_ratio", "float", "0.8", "Training fraction", {}, "0-1", "", "Split", false, false},
         {"val_ratio", "float", "0.1", "Validation fraction", {}, "0-1", "", "Split", false, false},
         {"test_ratio", "float", "0.1", "Held-out test fraction", {}, "0-1", "", "Split", false, false},
         {"stratified", "bool", "true", "Preserve class proportions when labels allow it", {}, "", "", "Split", false, false},
         {"seed", "int", "42", "Deterministic split seed", {}, "0-2147483647", "", "Split", false, false}},
        NodeImplementationStatus::Implemented, 0});
    RegisterNode({NodeType::DeployToNodeEditorNode, NodeCategory::DataSources, "Deploy to Node Editor", ICON_FA_FILE_EXPORT,
        {"deploy", "node", "editor", "handoff", "dataset"}, 0, false,
        "Mark a dataset ready for Node Editor deployment", "", "",
        {{"Data", PinType::Dataset, true, "Input dataset"}},
        {{"Dataset", PinType::Dataset, true, "Deployment-ready dataset"}},
        {{"name", "string", "", "Deployment dataset name", {}, ""}},
        NodeImplementationStatus::Implemented, 0});

    RegisterNode(WithPropertiesEditor({NodeType::DataConvert, NodeCategory::DataSources, "Data Convert", ICON_FA_RIGHT_LEFT,
        {"csv", "tsv", "parquet", "feather", "arrow", "ipc", "convert", "conversion", "format", "cache", "file"}, 0, false,
        "Convert datasets between supported table file formats", "", "",
        {{"Input", PinType::Dataset, false, "Optional input dataset artifact; input_path is used when disconnected"}},
        {{"Output", PinType::Dataset, true, "Converted dataset artifact"}},
        {{"input_path", "file", "", "Input data file", {}, "*.csv;*.tsv;*.parquet;*.pq;*.feather;*.fea;*.arrow;*.ipc"},
         {"input_format", "enum", "auto", "Input format", {"auto", "csv", "tsv", "parquet", "feather", "arrow", "ipc"}, ""},
         {"output_path", "file", "", "Output data file", {}, "*.csv;*.tsv;*.parquet;*.pq;*.feather;*.fea;*.arrow;*.ipc"},
         {"output_format", "enum", "auto", "Output format", {"auto", "csv", "tsv", "parquet", "feather", "arrow", "ipc"}, ""},
         {"delimiter", "enum", "auto", "CSV delimiter", {"auto", ",", "\\t", ";", "|"}, ""},
         {"decimal_point", "enum", ".", "Input decimal separator", {".", ","}, ""},
         {"header", "bool", "true", "Treat the first delimited row as column names", {}, ""},
         {"allow_newlines_in_values", "bool", "true", "Allow quoted multiline CSV values", {}, ""},
         {"skip_rows", "int", "0", "Rows skipped before parsing", {}, ">=0"},
         {"compression", "enum", "snappy", "Parquet compression", {"none", "snappy", "gzip", "zstd", "brotli"}, ""},
         {"row_group_size", "int", "1048576", "Parquet rows per row group", {}, ">0"},
         {"overwrite", "bool", "false", "Allow replacing an existing output file", {}, ""},
         {"create_parent_dirs", "bool", "true", "Create missing output folders", {}, ""},
         {"write_manifest", "bool", "true", "Write the conversion manifest", {}, ""},
         {"configured", "bool", "false", "Dialog configured", {}, "", "", "", false, false,
          ParameterConsumption::UiOnly},
         {"status", "string", "Not run", "Last conversion status", {}, "", "", "", false, false,
          ParameterConsumption::UiOnly},
         {"rows_written", "int", "0", "Rows written by the last conversion", {}, "", "", "", false, false,
          ParameterConsumption::UiOnly}},
        NodeImplementationStatus::Implemented, 0}, NodePropertiesEditor::Dialog));

    // ===== Legacy File Format Nodes (hidden - use DataInput/DataOutput instead) =====
    // Note: Commented out to clean up Node Browser - functionality consolidated into DataInput/DataOutput
    /*
    RegisterNode({NodeType::CSVFile, NodeCategory::DataSources, "CSV Reader", ICON_FA_FILE_CSV,
        {"csv", "file", "read", "import"}, 0, false, "Read CSV file into Arrow table", "", "",
        {}, {{"Table", PinType::Dataset, true, "Arrow table"}},
        {{"file_path", "file", "", "CSV file path", {}, "*.csv"},
         {"delimiter", "enum", ",", "Separator", {",", "\t", ";"}, ""},
         {"has_header", "bool", "true", "First row is header", {}, ""}},
        NodeImplementationStatus::Implemented, 0});

    RegisterNode({NodeType::ExcelFile, NodeCategory::DataSources, "Excel Reader", ICON_FA_FILE_EXCEL,
        {"excel", "xlsx", "spreadsheet"}, 0, false, "Read Excel file", "", "",
        {}, {{"Table", PinType::Dataset, true, "Arrow table"}},
        {{"file_path", "file", "", "Excel file", {}, "*.xlsx;*.xls"}},
        NodeImplementationStatus::Template, 0, "Blocked"});

    RegisterNode({NodeType::ParquetFile, NodeCategory::DataSources, "Parquet Reader", ICON_FA_DATABASE,
        {"parquet", "columnar"}, 0, false, "Read Parquet file", "", "",
        {}, {{"Table", PinType::Dataset, true, "Arrow table"}},
        {{"file_path", "file", "", "Parquet file", {}, "*.parquet"}},
        NodeImplementationStatus::Implemented, 0});

    RegisterNode({NodeType::JSONFile, NodeCategory::DataSources, "JSON Reader", ICON_FA_BRACKETS_CURLY,
        {"json", "javascript"}, 0, false, "Read JSON file", "", "",
        {}, {{"Table", PinType::Dataset, true, "Arrow table"}},
        {{"file_path", "file", "", "JSON file", {}, "*.json"}},
        NodeImplementationStatus::Template, 0, "Blocked"});

    RegisterNode({NodeType::SQLQuery, NodeCategory::DataSources, "SQL Query", ICON_FA_DATABASE,
        {"sql", "query", "database"}, 0, false, "Execute SQL query", "", "",
        {{"Source", PinType::Dataset, false, "Input table"}},
        {{"Result", PinType::Dataset, true, "Query result"}},
        {{"query", "string", "SELECT * FROM data", "SQL query", {}, ""}},
        NodeImplementationStatus::Template, 0, "Blocked"});

    RegisterNode({NodeType::HDF5Dataset, NodeCategory::DataSources, "HDF5 Reader", ICON_FA_HARD_DRIVE,
        {"hdf5", "h5", "scientific"}, 0, false, "Read HDF5 dataset", "", "",
        {}, {{"Table", PinType::Dataset, true, "Arrow table"}},
        {{"file_path", "file", "", "HDF5 file", {}, "*.h5;*.hdf5"}},
        NodeImplementationStatus::Template, 0, "Blocked"});

    RegisterNode({NodeType::RESTAPISource, NodeCategory::DataSources, "REST API", ICON_FA_GLOBE,
        {"rest", "api", "http"}, 0, false, "Fetch from REST API", "", "",
        {}, {{"Response", PinType::Dataset, true, "API response"}},
        {{"url", "string", "", "API URL", {}, ""},
         {"method", "enum", "GET", "HTTP method", {"GET", "POST"}, ""}},
        NodeImplementationStatus::Template, 0, "Blocked"});

    RegisterNode({NodeType::DatasetInput, NodeCategory::DataSources, "Dataset Input", ICON_FA_DATABASE,
        {"dataset", "input", "load"}, 0, false, "Load from Data Registry", "", "",
        {}, {{"Data", PinType::Tensor, true, "Samples"}, {"Labels", PinType::Labels, true, "Labels"}},
        {{"dataset_name", "string", "", "Dataset name", {}, ""}},
        NodeImplementationStatus::Implemented, 0});

    RegisterNode({NodeType::TSVFile, NodeCategory::DataSources, "TSV Reader", ICON_FA_FILE_LINES,
        {"tsv", "tab", "separated", "import", "load"}, 0, false,
        "Load TSV (tab-separated) file", "", "",
        {}, {{"Table", PinType::Dataset, false, "Table"}},
        {{"file_path", "file", "", "TSV file path", {}, "*.tsv"}},
        NodeImplementationStatus::Implemented, 0});

    RegisterNode({NodeType::TXTFile, NodeCategory::DataSources, "Text Reader", ICON_FA_FILE_LINES,
        {"txt", "text", "plain", "import", "load"}, 0, false,
        "Load plain text file", "", "",
        {}, {{"Text", PinType::Dataset, false, "Text"}},
        {{"file_path", "file", "", "Text file path", {}, "*.txt"}},
        NodeImplementationStatus::Implemented, 0});

    RegisterNode({NodeType::ImageCSVDataset, NodeCategory::DataSources, "Image+CSV Dataset", ICON_FA_IMAGES,
        {"image", "csv", "labels", "folder", "dataset"}, 0, false,
        "Load images from folder with CSV labels", "", "",
        {}, {{"Dataset", PinType::Dataset, false, "Dataset"}},
        {{"image_folder", "folder", "", "Image folder", {}, ""},
         {"labels_csv", "file", "", "Labels CSV", {}, "*.csv"}},
        NodeImplementationStatus::Implemented, 0});

    RegisterNode({NodeType::StreamingDataset, NodeCategory::DataSources, "Streaming Dataset", ICON_FA_WATER,
        {"streaming", "large", "lazy", "dataset"}, 0, false,
        "Stream large datasets without full memory load", "", "",
        {}, {{"Dataset", PinType::Dataset, false, "Dataset"}},
        {{"path", "folder", "", "Dataset path", {}, ""},
         {"batch_size", "int", "32", "Batch size", {}, ""}},
        NodeImplementationStatus::Implemented, 0});

    RegisterNode({NodeType::ARFFFile, NodeCategory::DataSources, "ARFF Reader", ICON_FA_FILE_CODE,
        {"arff", "weka", "import", "load"}, 0, false,
        "Load Weka ARFF file", "", "",
        {}, {{"Table", PinType::Dataset, false, "Table"}},
        {{"file_path", "file", "", "ARFF file path", {}, "*.arff"}},
        NodeImplementationStatus::Implemented, 0});

    RegisterNode({NodeType::FashionMNISTDataset, NodeCategory::DataSources, "Fashion-MNIST", ICON_FA_TAG,
        {"fashion", "mnist", "clothing", "dataset", "benchmark"}, 0, false,
        "Load Fashion-MNIST dataset", "", "",
        {}, {{"Dataset", PinType::Dataset, false, "Dataset"}},
        {{"split", "dropdown", "train", "Split", {"train", "test"}, ""}},
        NodeImplementationStatus::Implemented, 0});

    RegisterNode({NodeType::CIFAR100Dataset, NodeCategory::DataSources, "CIFAR-100", ICON_FA_IMAGE,
        {"cifar", "100", "dataset", "classification", "benchmark"}, 0, false,
        "Load CIFAR-100 dataset (100 classes)", "", "",
        {}, {{"Dataset", PinType::Dataset, false, "Dataset"}},
        {{"split", "dropdown", "train", "Split", {"train", "test"}, ""},
         {"label_type", "dropdown", "fine", "Labels", {"fine", "coarse"}, ""}},
        NodeImplementationStatus::Implemented, 0});

    RegisterNode({NodeType::FeatherFile, NodeCategory::DataSources, "Feather Reader", ICON_FA_FILE_CODE,
        {"feather", "arrow", "columnar", "import"}, 0, false,
        "Load Apache Arrow Feather file", "", "",
        {}, {{"Table", PinType::Dataset, false, "Table"}},
        {{"file_path", "file", "", "Feather file", {}, "*.feather;*.fea"}},
        NodeImplementationStatus::Implemented, 0});

    RegisterNode({NodeType::ArrowIPCFile, NodeCategory::DataSources, "Arrow IPC Reader", ICON_FA_FILE_IMPORT,
        {"arrow", "ipc", "binary", "import"}, 0, false,
        "Load Arrow IPC binary file", "", "",
        {}, {{"Table", PinType::Dataset, false, "Table"}},
        {{"file_path", "file", "", "Arrow file", {}, "*.arrow;*.ipc"}},
        NodeImplementationStatus::Implemented, 0});

    RegisterNode({NodeType::NumPyFile, NodeCategory::DataSources, "NumPy Reader", ICON_FA_CUBES,
        {"numpy", "npy", "npz", "array", "import"}, 0, false,
        "Load NumPy array file", "", "",
        {}, {{"Array", PinType::Tensor, false, "Array"}},
        {{"file_path", "file", "", "NumPy file", {}, "*.npy;*.npz"}},
        NodeImplementationStatus::Implemented, 0});

    RegisterNode({NodeType::AudioFolderDataset, NodeCategory::DataSources, "Audio Folder", ICON_FA_FOLDER_OPEN,
        {"audio", "wav", "flac", "ogg", "sound", "folder"}, 0, false,
        "Load audio files from class folders", "", "",
        {}, {{"Dataset", PinType::Dataset, false, "Dataset"}},
        {{"folder", "folder", "", "Audio folder", {}, ""},
         {"feature", "dropdown", "mfcc", "Feature", {"waveform", "spectrogram", "mel_spectrogram", "mfcc"}, ""},
         {"sample_rate", "int", "16000", "Sample rate", {}, ""}},
        NodeImplementationStatus::Implemented, 0});

    RegisterNode({NodeType::TimeSeriesCSV, NodeCategory::DataSources, "Time Series Reader", ICON_FA_CHART_LINE,
        {"timeseries", "csv", "temporal", "sequence", "forecast"}, 0, false,
        "Load time series CSV with windowing", "", "",
        {}, {{"Dataset", PinType::Dataset, false, "Dataset"}},
        {{"file_path", "file", "", "CSV file", {}, "*.csv"},
         {"target_column", "string", "", "Target column", {}, ""},
         {"lookback", "int", "10", "Lookback window", {}, ""},
         {"horizon", "int", "1", "Forecast horizon", {}, ""}},
        NodeImplementationStatus::Implemented, 0});

    RegisterNode({NodeType::TextCorpusDataset, NodeCategory::DataSources, "Text Corpus", ICON_FA_BOOK,
        {"text", "corpus", "nlp", "tokenize", "language"}, 0, false,
        "Load text corpus for NLP", "", "",
        {}, {{"Dataset", PinType::Dataset, false, "Dataset"}},
        {{"path", "file", "", "Text file/folder", {}, "*.txt;*.csv;*.json"},
         {"text_column", "string", "text", "Text column (CSV)", {}, ""},
         {"label_column", "string", "label", "Label column (CSV)", {}, ""},
         {"max_length", "int", "512", "Max sequence length", {}, ""}},
        NodeImplementationStatus::Implemented, 0});
    */  // End of legacy file format nodes comment

}

// =============================================================================
// Data Transform Nodes (Manipulation)
// =============================================================================
void NodeMetadataRegistry::InitializeDataTransformNodes() {
    RegisterNode({NodeType::FilterRows, NodeCategory::DataTransform, "Row Filter", ICON_FA_FILTER,
        {"filter", "where", "condition"}, 0, false, "Filter rows by condition", "", "",
        {{"Table", PinType::Dataset, true, "Input"}},
        {{"Filtered", PinType::Dataset, true, "Filtered"}},
        {{"condition", "string", "", "Comparison expression, for example: amount >= 100", {}, "", "Condition", "Filter", true}},
        NodeImplementationStatus::Implemented, 0});

    RegisterNode({NodeType::SelectColumns, NodeCategory::DataTransform, "Column Filter", ICON_FA_TABLE_COLUMNS,
        {"select", "columns", "project"}, 0, false, "Select columns", "", "",
        {{"Table", PinType::Dataset, true, "Input"}},
        {{"Selected", PinType::Dataset, true, "Selected columns"}},
        {{"columns", "string", "", "Comma-separated columns to retain, in output order", {}, "", "Columns", "Selection", true}},
        NodeImplementationStatus::Implemented, 0});

    RegisterNode({NodeType::JoinTables, NodeCategory::DataTransform, "Joiner", ICON_FA_CODE_BRANCH,
        {"join", "merge", "combine"}, 0, false,
        "Join two tables by matching key columns", "", "",
        {{"Left", PinType::Dataset, true, "Left table"},
         {"Right", PinType::Dataset, true, "Right table"}},
        {{"Joined", PinType::Dataset, true, "Joined table"}},
        {{"left_on", "string", "", "Key column in the Left table", {}, "", "Left key", "Join keys", true},
         {"right_on", "string", "", "Key column in the Right table", {}, "", "Right key", "Join keys", true},
         {"join_type", "enum", "inner", "Rows retained from the two inputs", {"inner", "left", "right", "outer"}, "", "Join type", "Join behavior"}},
        NodeImplementationStatus::Implemented, 0});

    RegisterNode({NodeType::GroupByAggregate, NodeCategory::DataTransform, "GroupBy", ICON_FA_OBJECT_GROUP,
        {"group", "aggregate", "sum"}, 0, false, "Group and aggregate", "", "",
        {{"Table", PinType::Dataset, true, "Input"}},
        {{"Grouped", PinType::Dataset, true, "Aggregated"}},
        {{"group_columns", "string", "", "Comma-separated grouping columns", {}, "", "Group columns", "Grouping", true},
         {"aggregations", "string", "", "Comma-separated COUNT, SUM, AVG, MIN, or MAX expressions with optional aliases", {}, "", "Aggregations", "Grouping", true}},
        NodeImplementationStatus::Implemented, 0});

    RegisterNode({NodeType::SortRows, NodeCategory::DataTransform, "Sorter", ICON_FA_ARROW_DOWN_LONG,
        {"sort", "order"}, 0, false, "Sort rows", "", "",
        {{"Table", PinType::Dataset, true, "Input"}},
        {{"Sorted", PinType::Dataset, true, "Sorted"}},
        {{"columns", "string", "", "Comma-separated columns used as sort keys", {}, "", "Sort columns", "Sorting", true},
         {"order", "enum", "asc", "Sort order", {"asc", "desc"}, ""},
         {"ascending", "bool", "true", "Compatibility ascending flag", {}, "", "", "", false, true}},
        NodeImplementationStatus::Implemented, 0});

    RegisterNode({NodeType::FillMissingValues, NodeCategory::DataTransform, "Missing Value", ICON_FA_ERASER,
        {"missing", "null", "fill"}, 0, false, "Handle missing values", "", "",
        {{"Table", PinType::Dataset, true, "Input"}},
        {{"Filled", PinType::Dataset, true, "Filled"}},
        {{"strategy", "enum", "mean", "Statistic or value used to replace nulls", {"mean", "median", "mode", "constant"}, "", "Strategy", "Transformation"},
         {"value", "string", "0", "Value used when Strategy is constant", {}, "", "Constant value", "Transformation"},
         {"columns", "string", "", "Comma-separated feature columns; empty selects all except Label column", {}, "", "Feature columns", "Columns"},
         {"label_col", "string", "", "Label/target column excluded from fitting", {}, "", "Label column", "Columns"},
         {"operation_mode", "enum", "fit_transform", "Fit on this input or reuse a saved training state", {"fit_transform", "transform_only"}, "", "Mode", "Fitted preprocessing state"},
         {"save_state", "bool", "false", "Persist fitted values to an engine-managed project artifact for validation, test, and inference", {}, "", "Save fitted state", "Fitted preprocessing state"},
         {"state_path", "file", "", "Existing training .cyxstate.json artifact required by Transform Only", {}, "*.cyxstate.json", "State artifact path", "Fitted preprocessing state"},
         {"state_overwrite", "bool", "false", "Allow replacing an existing state artifact", {}, "", "Allow state overwrite", "Fitted preprocessing state", false, true}},
        NodeImplementationStatus::Implemented, 0});

    RegisterNode({NodeType::RemoveDuplicateRows, NodeCategory::DataTransform, "Duplicate Remover", ICON_FA_COPY,
        {"duplicate", "unique"}, 0, false, "Remove duplicates", "", "",
        {{"Table", PinType::Dataset, true, "Input"}},
        {{"Unique", PinType::Dataset, true, "Unique rows"}},
        {{"columns", "string", "", "Comma-separated comparison columns; empty compares every column", {}, "", "Comparison columns", "Duplicates"}},
        NodeImplementationStatus::Implemented, 0});

    RegisterNode({NodeType::RenameColumns, NodeCategory::DataTransform, "Rename Columns", ICON_FA_TABLE_COLUMNS,
        {"rename", "columns", "mapping"}, 0, false, "Rename one or more columns", "", "",
        {{"Table", PinType::Dataset, true, "Input"}},
        {{"Renamed", PinType::Dataset, true, "Table with updated column names"}},
        {{"mapping", "string", "", "Comma-separated old_name:new_name pairs", {}, "", "Rename mapping", "Columns", true}},
        NodeImplementationStatus::Implemented, 0});

    RegisterNode({NodeType::BinningNode, NodeCategory::DataTransform, "Binning", ICON_FA_CHART_COLUMN,
        {"bin", "binning", "bucket", "discretize"}, 0, false, "Bin one numeric column", "", "",
        {{"Table", PinType::Dataset, true, "Input"}},
        {{"Binned", PinType::Dataset, true, "Dataset with bin column"}},
        {{"columns", "string", "", "Numeric column", {}, ""},
         {"method", "enum", "equal_width", "Method", {"equal_width", "equal_freq", "equal_frequency"}, ""},
         {"n_bins", "int", "10", "Number of bins", {}, ""}},
        NodeImplementationStatus::Implemented, 0});

    RegisterNode({NodeType::PolynomialFeaturesNode, NodeCategory::DataTransform, "Polynomial Features", ICON_FA_CHART_LINE,
        {"polynomial", "features", "power", "squared", "cubed"}, 0, false, "Generate polynomial features for one numeric column", "", "",
        {{"Table", PinType::Dataset, true, "Input"}},
        {{"Features", PinType::Dataset, true, "Dataset with polynomial feature columns"}},
        {{"columns", "string", "", "Numeric column", {}, ""},
         {"degree", "int", "2", "Maximum polynomial degree", {}, ""}},
        NodeImplementationStatus::Implemented, 0});

    RegisterNode({NodeType::Normalize, NodeCategory::Preprocessing, "Normalize", ICON_FA_SCALE_BALANCED,
        {"normalize", "scale", "mean", "standard deviation"}, 0, false,
        "Normalize a batched feature tensor with fixed configured statistics", "", "",
        {{"Input", PinType::Tensor, true, "Batched feature tensor to normalize"}},
        {{"Output", PinType::Tensor, true, "Tensor centered by mean and scaled by standard deviation"}},
        {{"mean", "float", "0.0", "Mean subtracted from each feature", {}, ""},
         {"std", "float", "1.0", "Positive standard deviation used to scale each feature", {}, ">0"}},
        NodeImplementationStatus::Implemented, 0});

    RegisterNode({NodeType::OneHotEncode, NodeCategory::Preprocessing, "One-Hot Encode", ICON_FA_TH,
        {"onehot", "encode", "categorical", "labels"}, 0, false,
        "Convert integer class indices into one-hot vectors", "", "",
        {{"Labels", PinType::Labels, true, "Integer class indices in [0, num_classes)"}},
        {{"OneHot", PinType::Tensor, true, "One-hot tensor with shape [batch, num_classes]"}},
        {{"num_classes", "int", "10", "Total number of target classes", {}, "1-100000"}},
        NodeImplementationStatus::Implemented, 0});
}

// =============================================================================
// Analytics Nodes
// =============================================================================
void NodeMetadataRegistry::InitializeAnalyticsNodes() {
    RegisterNode({NodeType::DescribeStats, NodeCategory::Analytics, "Descriptive Stats", ICON_FA_CHART_SIMPLE,
        {"statistics", "describe", "summary"}, 0, false, "Summarize numeric columns",
        "Emits one row per numeric column with its non-null count, mean, minimum, and maximum.", "",
        {{"Table", PinType::Dataset, true, "Input table"}},
        {{"Stats", PinType::Dataset, true, "Numeric summary table"}},
        {}, NodeImplementationStatus::Implemented, 0});

    RegisterNode({NodeType::CorrelationMatrix, NodeCategory::Analytics, "Correlation Matrix", ICON_FA_TABLE_CELLS,
        {"correlation", "pearson"}, 0, false, "Compute Pearson correlations",
        "Emits a long-form table containing every pair of numeric columns and their Pearson correlation.", "",
        {{"Table", PinType::Dataset, true, "Input table"}},
        {{"Matrix", PinType::Dataset, true, "Long-form Pearson correlation matrix"}},
        {},
        NodeImplementationStatus::Implemented, 0});

    RegisterNode({NodeType::VisualizeData, NodeCategory::Analytics, "Visualizer", ICON_FA_CHART_LINE,
        {"plot", "chart", "visualize"}, 0, false, "Create visualizations", "", "",
        {{"Table", PinType::Dataset, true, "Input"}}, {},
        {{"chart_type", "enum", "scatter", "Type", {"scatter", "bar", "line", "histogram"}, ""}},
        NodeImplementationStatus::Template, 0, "UI-only"});

    RegisterNode({NodeType::SampleRows, NodeCategory::Analytics, "Row Sampler", ICON_FA_DICE,
        {"sample", "head", "first", "limit"}, 0, false, "Take the first rows in source order",
        "Emits up to Count leading rows without shuffling or changing their order.", "",
        {{"Table", PinType::Dataset, true, "Input table"}},
        {{"Sample", PinType::Dataset, true, "Leading rows"}},
        {{"count", "int", "100", "Maximum rows to emit", {}, "0-2147483647"}},
        NodeImplementationStatus::Implemented, 0});

    RegisterNode({NodeType::ValueCounts, NodeCategory::Analytics, "Value Counts", ICON_FA_LIST_OL,
        {"count", "frequency"}, 0, false, "Count distinct values in one column",
        "Emits value and count columns ordered by descending frequency, then ascending value.", "",
        {{"Table", PinType::Dataset, true, "Input table"}},
        {{"Counts", PinType::Dataset, true, "Value-frequency table"}},
        {{"column", "string", "", "Column to count", {}, "", "", "", true, false}},
        NodeImplementationStatus::Implemented, 0});

    // ===== Machine Learning Algorithms (Phase 4) =====

    // Clustering
    RegisterNode({NodeType::KMeansCluster, NodeCategory::Analytics, "K-Means", ICON_FA_CIRCLE_NODES,
        {"kmeans", "cluster", "clustering"}, 0, false,
        "Assign rows to K-Means clusters",
        "Fits K-Means to selected numeric features and appends an Int32 cluster_id column to the input table.", "",
        {{"Data", PinType::Dataset, true, "Input table containing numeric features"}},
        {{"Clustered", PinType::Dataset, true, "Input table plus cluster_id"}},
        {{"feature_cols", "string", "", "Comma-separated numeric features; empty auto-detects numeric columns", {}, "", "Feature columns", "Columns"},
         {"label_col", "string", "", "Optional label column excluded from automatic feature detection", {}, "", "Label column", "Columns"},
         {"n_clusters", "int", "8", "Number of clusters", {}, "1-2147483647"},
         {"max_iter", "int", "300", "Maximum fit iterations", {}, "1-2147483647"},
         {"init", "enum", "kmeans++", "Centroid initialization", {"kmeans++", "random"}, ""},
         {"n_init", "int", "10", "Independent initializations; retains the lowest-inertia fit", {}, "1-2147483647"},
         {"tol", "float", "0.0001", "Convergence tolerance", {}, ""},
         {"seed", "int", "0", "Random seed; 0 selects nondeterministic initialization", {}, "0-2147483647"}},
        NodeImplementationStatus::Implemented, 0});

    RegisterNode({NodeType::DBSCANCluster, NodeCategory::Analytics, "DBSCAN", ICON_FA_CIRCLE_NODES,
        {"dbscan", "density", "cluster"}, 0, false,
        "Assign rows to density-based clusters",
        "Runs DBSCAN on selected numeric features and appends an Int32 cluster_id column; noise rows receive -1.", "",
        {{"Data", PinType::Dataset, true, "Input table containing numeric features"}},
        {{"Clustered", PinType::Dataset, true, "Input table plus cluster_id"}},
        {{"feature_cols", "string", "", "Comma-separated numeric features; empty auto-detects numeric columns", {}, "", "Feature columns", "Columns"},
         {"label_col", "string", "", "Optional label column excluded from automatic feature detection", {}, "", "Label column", "Columns"},
         {"eps", "float", "0.5", "Positive neighborhood radius", {}, ""},
         {"min_samples", "int", "5", "Minimum neighborhood size for a core point", {}, "1-2147483647"},
         {"metric", "enum", "euclidean", "Distance metric", {"euclidean", "manhattan", "cosine"}, ""}},
        NodeImplementationStatus::Implemented, 0});

    RegisterNode({NodeType::HierarchicalCluster, NodeCategory::Analytics, "Hierarchical Clustering", ICON_FA_SITEMAP,
        {"hierarchical", "dendrogram", "agglomerative"}, 0, false,
        "Assign rows using agglomerative clustering",
        "Clusters selected numeric features and appends an Int32 cluster_id column. Ward linkage requires Euclidean distance.", "",
        {{"Data", PinType::Dataset, true, "Input table containing numeric features"}},
        {{"Clustered", PinType::Dataset, true, "Input table plus cluster_id"}},
        {{"feature_cols", "string", "", "Comma-separated numeric features; empty auto-detects numeric columns", {}, "", "Feature columns", "Columns"},
         {"label_col", "string", "", "Optional label column excluded from automatic feature detection", {}, "", "Label column", "Columns"},
         {"n_clusters", "int", "3", "Number of clusters", {}, "1-2147483647"},
         {"linkage", "enum", "ward", "Linkage method", {"ward", "complete", "average", "single"}, ""},
         {"metric", "enum", "euclidean", "Distance metric", {"euclidean", "manhattan", "cosine"}, ""}},
        NodeImplementationStatus::Implemented, 0});

    // Dimensionality Reduction
    RegisterNode({NodeType::PCANode, NodeCategory::Analytics, "PCA", ICON_FA_COMPRESS,
        {"pca", "principal", "dimensionality"}, 0, false,
        "Project numeric features onto principal components",
        "Emits Float32 pc_0..pc_n columns and, when Label column is set, passes labels through as Int32 y.", "",
        {{"Data", PinType::Dataset, true, "Input table containing numeric features"}},
        {{"Transformed", PinType::Dataset, true, "Principal-component columns plus optional y"}},
        {{"feature_cols", "string", "", "Comma-separated numeric features; empty auto-detects numeric columns", {}, "", "Feature columns", "Columns"},
         {"label_col", "string", "", "Optional label column passed through as y and excluded from automatic features", {}, "", "Label column", "Columns"},
         {"n_components", "int", "2", "Number of principal components", {}, "1-2147483647"},
         {"center", "bool", "true", "Subtract each feature mean before projection", {}, ""},
         {"scale", "bool", "false", "Scale each feature to unit variance before projection", {}, ""}},
        NodeImplementationStatus::Implemented, 0});

    RegisterNode({NodeType::TSNENode, NodeCategory::Analytics, "t-SNE", ICON_FA_DIAGRAM_PROJECT,
        {"tsne", "visualization", "embedding"}, 0, false, "t-SNE visualization",
        "Non-linear dimensionality reduction for visualization.", "",
        {{"Data", PinType::Dataset, true, "Input data matrix"}},
        {{"Embedding", PinType::Dataset, true, "2D/3D embedding"}},
        {{"n_components", "int", "2", "Output dimensions", {}, "2-3"},
         {"perplexity", "float", "30.0", "Perplexity", {}, "5-50"},
         {"learning_rate", "float", "200.0", "Learning rate", {}, ""}},
        NodeImplementationStatus::Template, 0, "Blocked"});

    // Classification
    RegisterNode({NodeType::DecisionTreeClassifier, NodeCategory::Analytics, "Decision Tree", ICON_FA_SITEMAP,
        {"decision", "tree", "classifier", "classification", "classic ml"}, 0, false, "Decision Tree classifier",
        "Learns decision rules from tabular features and appends a prediction column.", "",
        {{"Data", PinType::Dataset, true, "Input table with features and target column"}},
        {{"Predictions", PinType::Dataset, true, "Input table plus prediction column"}},
        {{"target_col", "string", "", "Target label column in the input table", {}, "", "Target column", "Columns", true},
         {"feature_cols", "string", "", "Comma-separated numeric features; empty auto-detects numeric columns except the target", {}, "", "Feature columns", "Columns"},
         {"prediction_col", "string", "prediction", "Name of the appended prediction column", {}, "", "Prediction column", "Output"},
         {"model_path", "string", "", "Optional JSON path used to save the fitted tree artifact", {}, "", "Save model path", "Artifact"},
         {"max_depth", "int", "10", "Maximum tree depth", {}, "1-2147483647"},
         {"min_samples_split", "int", "2", "Minimum samples required to split a node", {}, "2-2147483647"},
         {"min_samples_leaf", "int", "1", "Minimum samples retained in a leaf", {}, "1-2147483647"},
         {"criterion", "enum", "gini", "Split criterion", {"gini", "entropy"}, ""}},
        NodeImplementationStatus::Implemented, 0});

    RegisterNode({NodeType::RandomForestClassifier, NodeCategory::Analytics, "Random Forest", ICON_FA_CUBES,
        {"random", "forest", "ensemble", "classification", "classic ml"}, 0, false, "Random Forest ensemble",
        "Fits a bagged ensemble of decision trees and appends a prediction column.", "",
        {{"Data", PinType::Dataset, true, "Input table with features and target column"}},
        {{"Predictions", PinType::Dataset, true, "Input table plus prediction column"}},
        {{"target_col", "string", "", "Target label column in the input table", {}, "", "Target column", "Columns", true},
         {"feature_cols", "string", "", "Comma-separated numeric features; empty auto-detects numeric columns except the target", {}, "", "Feature columns", "Columns"},
         {"prediction_col", "string", "prediction", "Name of the appended prediction column", {}, "", "Prediction column", "Output"},
         {"model_path", "string", "", "Optional JSON path used to save the fitted forest artifact", {}, "", "Save model path", "Artifact"},
         {"n_estimators", "int", "100", "Number of trees", {}, "1-2147483647"},
         {"max_depth", "int", "10", "Maximum depth per tree", {}, "1-2147483647"},
         {"min_samples_split", "int", "2", "Minimum samples required to split a node", {}, "2-2147483647"},
         {"min_samples_leaf", "int", "1", "Minimum samples retained in a leaf", {}, "1-2147483647"},
         {"criterion", "enum", "gini", "Split criterion", {"gini", "entropy"}, ""},
         {"max_features", "enum", "sqrt", "Features per tree", {"sqrt", "log2", "all"}, ""},
         {"seed", "int", "42", "Random seed", {}, "0-2147483647"}},
        NodeImplementationStatus::Implemented, 0});

    RegisterNode({NodeType::GradientBoostingClassifier, NodeCategory::Analytics, "Gradient Boosting", ICON_FA_CHART_LINE,
        {"gradient", "boosting", "classifier", "trees", "classic ml"}, 0, false, "Gradient Boosted Trees classifier",
        "Fits one-vs-rest boosted regression trees and appends a prediction column.", "",
        {{"Data", PinType::Dataset, true, "Input table with features and target column"}},
        {{"Predictions", PinType::Dataset, true, "Input table plus prediction column"}},
        {{"target_col", "string", "", "Target label column in the input table", {}, "", "Target column", "Columns", true},
         {"feature_cols", "string", "", "Comma-separated numeric features; empty auto-detects numeric columns except the target", {}, "", "Feature columns", "Columns"},
         {"prediction_col", "string", "prediction", "Name of the appended prediction column", {}, "", "Prediction column", "Output"},
         {"model_path", "string", "", "Optional JSON path used to save the fitted boosting artifact", {}, "", "Save model path", "Artifact"},
         {"n_estimators", "int", "100", "Number of boosting rounds", {}, "1-2147483647"},
         {"learning_rate", "float", "0.1", "Positive shrinkage applied to each boosting round", {}, ""},
         {"max_depth", "int", "3", "Maximum depth of each regression tree", {}, "1-2147483647"},
         {"min_samples_split", "int", "2", "Minimum samples required to split a node", {}, "2-2147483647"},
         {"min_samples_leaf", "int", "1", "Minimum samples retained in a leaf", {}, "1-2147483647"}},
        NodeImplementationStatus::Implemented, 0});

    RegisterNode({NodeType::TreeModelPredictor, NodeCategory::Analytics, "Tree Model Predictor", ICON_FA_FILE_IMPORT,
        {"tree", "model", "predict", "predictor", "inference", "artifact", "classic ml"}, 0, false, "Saved tree model inference",
        "Loads a native CyxWiz tree-family model artifact and appends a prediction column.", "",
        {{"Data", PinType::Dataset, true, "Input table with compatible numeric feature columns"}},
        {{"Predictions", PinType::Dataset, true, "Input table plus prediction column"}},
        {{"model_path", "string", "", "Saved JSON tree-family artifact to load", {}, "", "Model path", "Artifact", true},
         {"feature_cols", "string", "", "Comma-separated numeric features; empty uses the artifact's saved feature order", {}, "", "Feature columns", "Columns"},
         {"prediction_col", "string", "prediction", "Name of the appended prediction column", {}, "", "Prediction column", "Output"}},
        NodeImplementationStatus::Implemented, 0});

    RegisterNode({NodeType::SVMClassifier, NodeCategory::Analytics, "SVM Classifier", ICON_FA_BORDER_ALL,
        {"svm", "support", "vector"}, 0, false,
        "Blocked legacy SVM classifier preview",
        "Saved-graph compatibility contract only. No classifier executor, fitted model artifact, or predictor is implemented.", "",
        {{"Train Data", PinType::Dataset, true, "Legacy feature-table input"},
         {"Labels", PinType::Labels, true, "Legacy label input"}},
        {{"Model", PinType::Parameters, true, "Reserved fitted-model output"},
         {"Predictions", PinType::Labels, true, "Reserved prediction output"}},
        {{"kernel", "enum", "rbf", "Legacy kernel selection retained for saved graphs", {"linear", "rbf", "poly", "sigmoid"}, "", "Kernel", "Compatibility"},
         {"C", "float", "1.0", "Legacy regularization value retained for saved graphs", {}, "", "Regularization (C)", "Compatibility"},
         {"gamma", "enum", "scale", "Legacy kernel coefficient retained for saved graphs", {"scale", "auto"}, "", "Gamma", "Compatibility"}},
        NodeImplementationStatus::Template, 0, "Blocked"});

    RegisterNode({NodeType::KNNClassifier, NodeCategory::Analytics, "KNN Classifier", ICON_FA_USERS,
        {"knn", "nearest", "neighbor"}, 0, false,
        "Blocked legacy K-nearest-neighbors classifier preview",
        "Saved-graph compatibility contract only. No classifier executor, fitted model artifact, or predictor is implemented.", "",
        {{"Train Data", PinType::Dataset, true, "Legacy feature-table input"},
         {"Labels", PinType::Labels, true, "Legacy label input"}},
        {{"Model", PinType::Parameters, true, "Reserved fitted-model output"},
         {"Predictions", PinType::Labels, true, "Reserved prediction output"}},
        {{"n_neighbors", "int", "5", "Legacy neighbor count retained for saved graphs", {}, "1-100", "Neighbors", "Compatibility"},
         {"weights", "enum", "uniform", "Legacy neighbor weighting retained for saved graphs", {"uniform", "distance"}, "", "Weights", "Compatibility"},
         {"metric", "enum", "euclidean", "Legacy distance metric retained for saved graphs", {"euclidean", "manhattan", "cosine"}, "", "Distance metric", "Compatibility"}},
        NodeImplementationStatus::Template, 0, "Blocked"});

    RegisterNode({NodeType::NaiveBayesClassifier, NodeCategory::Analytics, "Naive Bayes", ICON_FA_PERCENT,
        {"naive", "bayes", "classifier"}, 0, false,
        "Blocked legacy Gaussian Naive Bayes preview",
        "Saved-graph compatibility contract only. No classifier executor, fitted model artifact, or predictor is implemented.", "",
        {{"Train Data", PinType::Dataset, true, "Legacy feature-table input"},
         {"Labels", PinType::Labels, true, "Legacy label input"}},
        {{"Model", PinType::Parameters, true, "Reserved fitted-model output"},
         {"Predictions", PinType::Labels, true, "Reserved prediction output"}},
        {{"var_smoothing", "float", "1e-9", "Legacy Gaussian variance smoothing retained for saved graphs", {}, "", "Variance smoothing", "Compatibility"}},
        NodeImplementationStatus::Template, 0, "Blocked"});

    RegisterNode({NodeType::LogisticRegressionNode, NodeCategory::Analytics, "Logistic Regression", ICON_FA_PERCENT,
        {"logistic", "regression", "classifier"}, 0, false,
        "Blocked legacy logistic-regression classifier preview",
        "Saved-graph compatibility contract only. No classifier executor, fitted model artifact, probability output, or predictor is implemented.", "",
        {{"Train Data", PinType::Dataset, true, "Legacy feature-table input"},
         {"Labels", PinType::Labels, true, "Legacy label input"}},
        {{"Model", PinType::Parameters, true, "Reserved fitted-model output"},
         {"Predictions", PinType::Labels, true, "Reserved prediction output"}},
        {{"C", "float", "1.0", "Legacy inverse regularization value retained for saved graphs", {}, "", "Regularization (C)", "Compatibility"},
         {"solver", "enum", "lbfgs", "Legacy solver selection retained for saved graphs", {"lbfgs"}, "", "Solver", "Compatibility"},
         {"max_iter", "int", "100", "Legacy iteration limit retained for saved graphs", {}, "1-2147483647", "Maximum iterations", "Compatibility"}},
        NodeImplementationStatus::Template, 0, "Blocked"});

    // Regression
    RegisterNode({NodeType::LinearRegressionNode, NodeCategory::Analytics, "Linear Regression", ICON_FA_CHART_LINE,
        {"linear", "regression"}, 0, false, "Linear Regression",
        "Fits ordinary least squares from numeric table columns and appends Float32 prediction and residual columns.", "",
        {{"Data", PinType::Dataset, true, "Table containing numeric predictor and target columns"}},
        {{"Fitted", PinType::Dataset, true, "Input table plus Float32 prediction and residual columns"},
         {"Model", PinType::Parameters, true, "Reusable fitted linear-regression artifact"}},
        {{"feature_cols", "string", "", "Comma-separated numeric predictor columns", {}, "", "Predictor columns", "Columns", true, false},
         {"target_col", "string", "", "Numeric response column", {}, "", "Target column", "Columns", true, false},
         {"fit_intercept", "bool", "true", "Include a fitted intercept term", {}, "", "Fit intercept", "Fit Options", false, false}},
        NodeImplementationStatus::Implemented, 0});

    RegisterNode({NodeType::PolynomialRegressionNode, NodeCategory::Analytics, "Polynomial Regression", ICON_FA_CHART_LINE,
        {"polynomial", "regression", "curve", "fit"}, 0, false, "Polynomial Regression",
        "Fits one numeric predictor as powers from degree one through the selected degree, then appends Float32 prediction and residual columns.", "",
        {{"Data", PinType::Dataset, true, "Table containing one numeric predictor and a numeric target column"}},
        {{"Fitted", PinType::Dataset, true, "Input table plus Float32 prediction and residual columns"},
         {"Model", PinType::Parameters, true, "Reusable fitted polynomial-regression artifact"}},
        {{"feature_col", "string", "", "Single numeric predictor column", {}, "", "Predictor column", "Columns", true, false},
         {"target_col", "string", "", "Numeric response column", {}, "", "Target column", "Columns", true, false},
         {"degree", "int", "2", "Highest polynomial power; requires at least degree plus two rows", {}, "1-2147483647", "Degree", "Fit Options", false, false}},
        NodeImplementationStatus::Implemented, 0});

    RegisterNode({NodeType::RegressionModelPredictor, NodeCategory::Analytics, "Regression Model Predictor", ICON_FA_FILE_IMPORT,
        {"regression", "model", "predict", "inference", "artifact", "classic ml"}, 0, false, "Fitted regression inference",
        "Applies a fitted Linear or Polynomial Regression model to compatible numeric table columns.", "",
        {{"Data", PinType::Dataset, true, "Input table containing the artifact's predictor columns"},
         {"Model", PinType::Parameters, true, "Fitted regression artifact from a regression node"}},
        {{"Predictions", PinType::Dataset, true, "Input table plus a Float32 prediction column"}},
        {{"prediction_col", "string", "prediction", "Name of the appended prediction column", {}, "", "Prediction column", "Output"}},
        NodeImplementationStatus::Implemented, 0});

    // Model Evaluation
    RegisterNode({NodeType::ConfusionMatrixNode, NodeCategory::Analytics, "Confusion Matrix", ICON_FA_TABLE_CELLS,
        {"confusion", "matrix", "evaluation"}, 0, false, "Count actual/predicted label pairs",
        "Emits observed label pairs with raw counts and values normalized by actual label, predicted label, or all valid rows.", "",
        {{"Data", PinType::Dataset, true, "Input table"}},
        {{"Matrix", PinType::Dataset, true, "Observed actual/predicted pairs with count and value"}},
        {{"actual_col", "string", "", "Actual label column", {}, "", "", "", true, false},
         {"predicted_col", "string", "", "Predicted label column", {}, "", "", "", true, false},
         {"normalize", "enum", "none", "none=count; true=by actual; pred=by predicted; all=by all rows", {"none", "true", "pred", "all"}, ""}},
        NodeImplementationStatus::Implemented, 0});

    RegisterNode({NodeType::ROCCurveNode, NodeCategory::Analytics, "ROC Curve", ICON_FA_CHART_AREA,
        {"roc", "auc", "curve"}, 0, false, "ROC curve and AUC",
        "Computes ROC curve points and AUC from binary labels and prediction scores.", "",
        {{"Data", PinType::Dataset, true, "Input table"}},
        {{"Curve", PinType::Dataset, true, "Threshold, FPR, TPR, and AUC table"}},
        {{"actual_col", "string", "", "Binary actual label column", {}, "", "", "", true, false},
         {"score_col", "string", "", "Numeric positive-class score column", {}, "", "", "", true, false},
         {"positive_label", "string", "1", "Positive class label", {}, ""}},
        NodeImplementationStatus::Implemented, 0});

    RegisterNode({NodeType::PRCurveNode, NodeCategory::Analytics, "Precision-Recall Curve", ICON_FA_CHART_AREA,
        {"precision", "recall", "pr", "average precision"}, 0, false, "Precision-recall curve",
        "Computes precision-recall curve points and average precision from binary labels and prediction scores.", "",
        {{"Data", PinType::Dataset, true, "Input table"}},
        {{"Curve", PinType::Dataset, true, "Threshold, precision, recall, and average-precision table"}},
        {{"actual_col", "string", "", "Binary actual label column", {}, "", "", "", true, false},
         {"score_col", "string", "", "Numeric positive-class score column", {}, "", "", "", true, false},
         {"positive_label", "string", "1", "Positive class label", {}, ""}},
        NodeImplementationStatus::Implemented, 0});

    RegisterNode({NodeType::LearningCurvesNode, NodeCategory::Analytics, "Learning Curves", ICON_FA_CHART_LINE,
        {"learning", "curves", "training"}, 0, false, "Training/validation learning curves",
        "Plot train and validation scores vs training set size.", "",
        {{"Model", PinType::Parameters, true, "Model"}, {"X", PinType::Dataset, true, "Features"}, {"y", PinType::Labels, true, "Labels"}},
        {{"TrainScores", PinType::Dataset, true, "Train scores"}, {"ValScores", PinType::Dataset, true, "Validation scores"}},
        {{"cv", "int", "5", "Cross-validation folds", {}, ""},
         {"train_sizes", "string", "0.1,0.3,0.5,0.7,0.9,1.0", "Training sizes", {}, ""}},
        NodeImplementationStatus::Template, 0, "UI-only"});

    RegisterNode({NodeType::CrossValidationNode, NodeCategory::Analytics, "Cross-Validation", ICON_FA_ROTATE,
        {"cross", "validation", "cv"}, 0, false, "K-Fold cross-validation",
        "Evaluate model performance using k-fold cross-validation.", "",
        {{"Model", PinType::Parameters, true, "Model"}, {"X", PinType::Dataset, true, "Features"}, {"y", PinType::Labels, true, "Labels"}},
        {{"Scores", PinType::Dataset, true, "CV scores"}, {"Mean", PinType::Dataset, true, "Mean score"}, {"Std", PinType::Dataset, true, "Std deviation"}},
        {{"cv", "int", "5", "Number of folds", {}, "2-20"},
         {"scoring", "enum", "accuracy", "Scoring metric", {"accuracy", "f1", "precision", "recall", "roc_auc"}, ""}},
        NodeImplementationStatus::Template, 0, "UI-only"});

    RegisterNode({NodeType::FeatureImportanceNode, NodeCategory::Analytics, "Feature Importance", ICON_FA_RANKING_STAR,
        {"feature", "importance", "selection"}, 0, false, "Feature importance analysis",
        "Compute and visualize feature importances from tree-based models.", "",
        {{"Model", PinType::Parameters, true, "Trained tree model"}},
        {{"Importances", PinType::Dataset, true, "Feature importance scores"}},
        {{"method", "enum", "impurity", "Importance method", {"impurity", "permutation"}, ""}},
        NodeImplementationStatus::Template, 0, "UI-only"});

    RegisterNode({NodeType::RegressionMetricsNode, NodeCategory::Analytics, "Regression Metrics", ICON_FA_CHART_SIMPLE,
        {"regression", "metrics", "mse", "rmse", "mae", "r2"}, 0, false, "Regression metrics",
        "Computes regression metrics from numeric actual and predicted columns.", "",
        {{"Data", PinType::Dataset, true, "Input table"}},
        {{"Metrics", PinType::Dataset, true, "Metric/value table"}},
        {{"actual_col", "string", "", "Numeric actual value column", {}, "", "", "", true, false},
         {"predicted_col", "string", "", "Numeric predicted value column", {}, "", "", "", true, false},
         {"metrics", "string", "mse,rmse,mae,r2", "Comma-separated: mse, rmse, mae, r2, count", {}, ""}},
        NodeImplementationStatus::Implemented, 0});

    RegisterNode({NodeType::ClassificationMetricsNode, NodeCategory::Analytics, "Classification Metrics", ICON_FA_CHART_SIMPLE,
        {"classification", "metrics", "accuracy", "precision", "recall", "f1", "macro", "weighted"}, 0, false,
        "Classification metrics",
        "Computes accuracy, macro precision/recall/F1, weighted F1, and count from actual and predicted label columns.", "",
        {{"Data", PinType::Dataset, true, "Input table"}},
        {{"Metrics", PinType::Dataset, true, "Metric/value table"}},
        {{"actual_col", "string", "", "Actual label column", {}, "", "", "", true, false},
         {"predicted_col", "string", "", "Predicted label column", {}, "", "", "", true, false},
         {"metrics", "string", "accuracy,precision,recall,f1,weighted_f1,count", "Comma-separated: accuracy, precision, recall, f1, weighted_f1, count, class_count", {}, ""}},
        NodeImplementationStatus::Implemented, 0});

    // Preprocessing (Phase 4)
    RegisterNode({NodeType::StandardScaler, NodeCategory::Preprocessing, "Standard Scaler", ICON_FA_SCALE_BALANCED,
        {"standardize", "zscore", "scaler"}, 0, false, "Z-score standardization",
        "Transforms features to have mean=0 and std=1.", "",
        {{"Data", PinType::Dataset, true, "Input data"}},
        {{"Scaled", PinType::Dataset, true, "Standardized data"}},
        {{"columns", "string", "", "Columns to scale (empty = numeric auto-detect)", {}, ""},
           {"label_col", "string", "", "Label column to exclude", {}, ""},
           {"exclude_columns", "string", "", "Additional columns to exclude from numeric auto-detect (comma-separated)", {}, ""},
         {"with_mean", "bool", "true", "Center data", {}, ""},
         {"with_std", "bool", "true", "Scale to unit variance", {}, ""},
         {"transform_role", "enum", "features", "Declare whether this scaler transforms features or continuous regression targets", {"features", "regression_target"}, "", "Transform role", "Training semantics"},
         {"operation_mode", "enum", "fit_transform", "Fit on this input or reuse saved training statistics", {"fit_transform", "transform_only"}, "", "Mode", "Fitted preprocessing state"},
         {"save_state", "bool", "false", "Persist mean and scale to an engine-managed project artifact for validation, test, and inference", {}, "", "Save fitted state", "Fitted preprocessing state"},
         {"state_path", "file", "", "Existing training .cyxstate.json artifact required by Transform Only", {}, "*.cyxstate.json", "State artifact path", "Fitted preprocessing state"},
         {"state_overwrite", "bool", "false", "Allow replacing an existing state artifact", {}, "", "Allow state overwrite", "Fitted preprocessing state", false, true}},
        NodeImplementationStatus::Implemented, 0});

    RegisterNode({NodeType::MinMaxScaler, NodeCategory::Preprocessing, "MinMax Scaler", ICON_FA_ARROWS_LEFT_RIGHT,
        {"minmax", "normalize", "scaler"}, 0, false, "Min-Max normalization",
        "Scales features to [0, 1] or custom range.", "",
        {{"Data", PinType::Dataset, true, "Input data"}},
        {{"Scaled", PinType::Dataset, true, "Normalized data"}},
        {{"columns", "string", "", "Columns to scale (empty = numeric auto-detect)", {}, ""},
         {"label_col", "string", "", "Label column to exclude", {}, ""},
         {"min", "float", "0.0", "Target range minimum", {}, ""},
         {"max", "float", "1.0", "Target range maximum", {}, ""}},
        NodeImplementationStatus::Implemented, 0});

    // (TrainTestSplit registration removed — use NodeType::DataSplit, which
    //  supports 3-way train/val/test split and is the single source of truth
    //  for dataset partitioning. See node_editor_add_search.cpp for its entry.)

    RegisterNode({NodeType::RobustScaler, NodeCategory::Preprocessing, "Robust Scaler", ICON_FA_SCALE_BALANCED,
        {"robust", "scaler", "median", "iqr"}, 0, false, "Robust scaling",
        "Scales numeric columns using median and interquartile range.", "",
        {{"Data", PinType::Dataset, true, "Input data"}},
        {{"Scaled", PinType::Dataset, true, "Scaled data"}},
        {{"columns", "string", "", "Columns to scale (empty = numeric auto-detect)", {}, ""},
         {"label_col", "string", "", "Label column to exclude", {}, ""},
         {"with_centering", "bool", "true", "Subtract median", {}, ""},
         {"with_scaling", "bool", "true", "Scale by IQR", {}, ""},
         {"quantile_min", "float", "25", "Lower quantile", {}, "0-100"},
         {"quantile_max", "float", "75", "Upper quantile", {}, "0-100"}},
        NodeImplementationStatus::Implemented, 0});

    RegisterNode({NodeType::LabelEncoder, NodeCategory::Preprocessing, "Label Encoder", ICON_FA_TAG,
        {"label", "encoder", "categorical"}, 0, false, "Encode one categorical column",
        "Replaces one string column with stable int32 category codes.", "",
        {{"Data", PinType::Dataset, true, "Input data"}},
        {{"Encoded", PinType::Dataset, true, "Encoded data"}},
        {{"column", "string", "", "String column to encode", {}, "", "", "", true}},
        NodeImplementationStatus::Implemented, 0});

    RegisterNode({NodeType::OrdinalEncoder, NodeCategory::Preprocessing, "Ordinal Encoder", ICON_FA_LIST_OL,
        {"ordinal", "encoder", "categorical"}, 0, false, "Encode categorical columns",
        "Replaces one or more string columns with alphabetical int32 category codes.", "",
        {{"Data", PinType::Dataset, true, "Input data"}},
        {{"Encoded", PinType::Dataset, true, "Encoded data"}},
        {{"columns", "string", "", "Comma-separated string columns to encode", {}, "", "", "", true},
         {"categories", "enum", "auto", "Category ordering (auto only in v1)", {"auto"}, ""}},
        NodeImplementationStatus::Implemented, 0});

    RegisterNode({NodeType::TargetEncoder, NodeCategory::Preprocessing, "Target Encoder", ICON_FA_PERCENT,
        {"target", "encoder", "categorical", "mean"}, 0, false, "Target mean encoding",
        "Replaces categorical columns with smoothed target means.", "",
        {{"Data", PinType::Dataset, true, "Input data"}},
        {{"Encoded", PinType::Dataset, true, "Encoded data"}},
        {{"columns", "string", "", "Comma-separated categorical columns to encode", {}, "", "", "", true},
         {"target_col", "string", "", "Numeric target column used to compute category means", {}, "", "", "", true},
         {"smoothing", "float", "1.0", "Smoothing pseudo-count", {}, ""}},
        NodeImplementationStatus::Implemented, 0});

    RegisterNode({NodeType::OutlierDetector, NodeCategory::Preprocessing, "Outlier Detector", ICON_FA_FILTER,
        {"outlier", "iqr", "zscore", "detect"}, 0, false, "Flag numeric outliers",
        "Adds an is_outlier column using IQR or Z-score detection.", "",
        {{"Data", PinType::Dataset, true, "Input data"}},
        {{"Flagged", PinType::Dataset, true, "Input data plus is_outlier"}},
        {{"columns", "string", "all", "Columns to inspect", {}, ""},
         {"label_col", "string", "", "Label column to exclude", {}, ""},
         {"method", "dropdown", "iqr", "Method", {"iqr", "zscore"}, ""},
         {"threshold", "float", "1.5", "Positive IQR multiplier or Z-score threshold", {}, ""}},
        NodeImplementationStatus::Implemented, 0});

    RegisterNode({NodeType::DataProfiler, NodeCategory::Analytics, "Data Profiler", ICON_FA_MAGNIFYING_GLASS_CHART,
        {"profiler", "eda", "exploration"}, 0, false, "Comprehensive data profiling",
        "Generate detailed data quality report: types, missing, distributions, correlations.", "",
        {{"Data", PinType::Dataset, true, "Input dataset"}},
        {{"Report", PinType::Dataset, true, "Profiling report"}},
        {},
        NodeImplementationStatus::Implemented, 0});

    RegisterNode({NodeType::DataValidator, NodeCategory::Analytics, "Data Validator", ICON_FA_CHECK_DOUBLE,
        {"validate", "schema", "quality", "rules"}, 0, false, "Data validation report",
        "Validates required, not-null, and unique column constraints and emits an issue report.", "",
        {{"Data", PinType::Dataset, true, "Input dataset"}},
        {{"Issues", PinType::Dataset, true, "Validation issues"}},
        {{"required_columns", "string", "", "Comma-separated required columns", {}, ""},
         {"not_null_columns", "string", "", "Columns that cannot contain nulls", {}, ""},
         {"unique_columns", "string", "", "Columns that must be unique", {}, ""}},
        NodeImplementationStatus::Implemented, 0});
}

// =============================================================================
// ML Layer Nodes
// =============================================================================
void NodeMetadataRegistry::InitializeLayerNodes() {
    RegisterNode({NodeType::Dense, NodeCategory::Layers, "Dense", ICON_FA_LAYER_GROUP,
        {"dense", "linear", "fc"}, 0, false,
        "Linear projection over the final input dimension",
        "Maps input features to a configurable output width. Add a separate "
        "activation node when nonlinear behavior is required.",
        "",
        {{"Input", PinType::Tensor, true,
          "Input features [batch, in_features]. Flatten higher-rank inputs first."}},
        {{"Output", PinType::Tensor, true,
          "Linear projection [batch, units]."}},
        {{"units", "int", "64", "Number of output features", {}, "1-1048576",
          "Output Units", "Layer", true, false}},
        NodeImplementationStatus::Implemented, 0});

    RegisterNode({NodeType::Conv2D, NodeCategory::Layers, "Conv2D", ICON_FA_BORDER_ALL,
        {"conv", "convolution", "cnn"}, 0, false,
        "Blocked 2D convolution layer retained for graph compatibility",
        "Saved Conv2D nodes remain visible and inspectable, but cannot compile "
        "or train until ModelBuilder owns a supported Conv2D module path. Use "
        "separate activation nodes; the legacy inline activation field is not "
        "an executable contract.",
        "",
        {{"Input", PinType::Tensor, true,
          "Input feature map [batch, channels, height, width]."}},
        {{"Output", PinType::Tensor, true,
          "Convolved feature map; unavailable at runtime while this node is blocked."}},
        {{"filters", "int", "32", "Number of output channels", {}, "1-1048576",
          "Output Channels", "Convolution", true, false},
         {"kernel_size", "int", "3", "Square kernel width and height", {}, "1-1048576",
          "Kernel Size", "Convolution", true, false},
         {"stride", "int", "1", "Spatial step between kernel applications", {}, "1-1048576",
          "Stride", "Convolution", true, false},
         {"padding", "enum", "same", "Legacy padding policy preserved for saved graphs",
          {"same", "valid"}, "", "Padding", "Convolution", true, false}},
        NodeImplementationStatus::Template, 0, "Blocked"});

    RegisterNode({NodeType::Conv1D, NodeCategory::Layers, "Conv1D", ICON_FA_BORDER_ALL,
        {"conv", "convolution", "1d", "sequence"}, 0, false,
        "Blocked 1D convolution layer retained for graph compatibility",
        "The backend contains a native Conv1D primitive, but GraphCompiler, "
        "ModelBuilder, and SequentialModel do not own an executable Conv1D "
        "path. Its Tensor layout, ArrayFire-first execution, fallback, and "
        "training integration must be proven before this node is enabled.",
        "",
        {{"Input", PinType::Tensor, true,
          "Legacy sequence feature-map input; no executable Engine layout contract exists yet."}},
        {{"Output", PinType::Tensor, true,
          "Convolved feature map; unavailable at runtime while this node is blocked."}},
        {{"filters", "int", "32", "Legacy output-channel count", {}, "1-1048576",
          "Output Channels", "Convolution", true, false},
         {"kernel_size", "int", "3", "Legacy kernel width", {}, "1-1048576",
          "Kernel Size", "Convolution", true, false},
         {"stride", "int", "1", "Legacy convolution stride", {}, "1-1048576",
          "Stride", "Convolution", true, false},
         {"padding", "enum", "same", "Legacy padding policy preserved for saved graphs",
          {"same", "valid"}, "", "Padding", "Convolution", true, false}},
        NodeImplementationStatus::Template, 0, "Blocked"});

    RegisterNode({NodeType::Conv3D, NodeCategory::Layers, "Conv3D", ICON_FA_BORDER_ALL,
        {"conv", "convolution", "3d", "volume"}, 0, false,
        "Blocked 3D convolution design node retained for graph compatibility",
        "No backend Conv3D layer, GraphCompiler extraction, ModelBuilder module, "
        "or SequentialModel execution path currently owns this node.",
        "",
        {{"Input", PinType::Tensor, true,
          "Legacy volume feature-map input; no executable Engine layout contract exists yet."}},
        {{"Output", PinType::Tensor, true,
          "Convolved volume; unavailable at runtime while this node is blocked."}},
        {{"filters", "int", "32", "Legacy output-channel count", {}, "1-1048576",
          "Output Channels", "Convolution", true, false},
         {"kernel_size", "int", "3", "Legacy cubic-kernel size", {}, "1-1048576",
          "Kernel Size", "Convolution", true, false},
         {"stride", "int", "1", "Legacy convolution stride", {}, "1-1048576",
          "Stride", "Convolution", true, false},
         {"padding", "enum", "same", "Legacy padding policy preserved for saved graphs",
          {"same", "valid"}, "", "Padding", "Convolution", true, false}},
        NodeImplementationStatus::Template, 0, "Blocked"});

    RegisterNode({NodeType::DepthwiseConv2D, NodeCategory::Layers,
        "Depthwise Conv2D", ICON_FA_BORDER_ALL,
        {"conv", "convolution", "depthwise", "image"}, 0, false,
        "Blocked depthwise convolution design node retained for graph compatibility",
        "No backend depthwise layer, GraphCompiler extraction, ModelBuilder "
        "module, or SequentialModel execution path currently owns this node.",
        "",
        {{"Input", PinType::Tensor, true,
          "Legacy image feature-map input; no executable Engine layout contract exists yet."}},
        {{"Output", PinType::Tensor, true,
          "Depthwise feature map; unavailable at runtime while this node is blocked."}},
        {{"filters", "int", "32", "Legacy output-channel count", {}, "1-1048576",
          "Output Channels", "Convolution", true, false},
         {"kernel_size", "int", "3", "Legacy square-kernel size", {}, "1-1048576",
          "Kernel Size", "Convolution", true, false},
         {"stride", "int", "1", "Legacy convolution stride", {}, "1-1048576",
          "Stride", "Convolution", true, false},
         {"padding", "enum", "same", "Legacy padding policy preserved for saved graphs",
          {"same", "valid"}, "", "Padding", "Convolution", true, false},
         {"depth_multiplier", "int", "1", "Legacy channel multiplier", {}, "1-1048576",
          "Depth Multiplier", "Convolution", true, false}},
        NodeImplementationStatus::Template, 0, "Blocked"});

    RegisterNode({NodeType::LSTM, NodeCategory::Recurrent, "LSTM", ICON_FA_REPEAT,
        {"lstm", "recurrent", "sequence"}, 0, false,
        "Trainable long short-term memory sequence layer",
        "Engine training supports unidirectional LSTM with dropout=0.0. "
        "Bidirectional forward exists, but reverse-direction backward gradients "
        "are not implemented, so bidirectional training fails closed.", "",
        {{"Input", PinType::Tensor, true,
          "Sequence tensor [batch, sequence, features]; features is derived as input_size."}},
        {{"Output", PinType::Tensor, true,
          "Full sequence [batch, sequence, hidden] when return_sequences=true; otherwise [batch, hidden]."},
         {"Hidden", PinType::Tensor, false,
          "Legacy compatibility pin only. Engine SequentialModel does not route a separate h_n output; leave disconnected."}},
        {{"input_size", "int", "0", "Auto-derived input feature count", {}, "Derived from the previous layer output.",
          "Input Size", "Recurrent", true, true},
         {"hidden_size", "int", "256", "Hidden-state width", {}, "1-1048576",
          "Hidden Size", "Recurrent", true, false},
         {"num_layers", "int", "1", "Number of stacked recurrent layers", {}, "1-1048576",
          "Layers", "Recurrent", true, false},
         {"bidirectional", "bool", "false", "Two-direction intent; unavailable for Engine training", {}, "",
          "Bidirectional", "Recurrent", true, false},
         {"return_sequences", "bool", "false", "Return every timestep instead of the final timestep", {}, "",
          "Return Sequences", "Output", true, false},
         {"dropout", "float", "0.0", "Must remain 0.0; use an explicit Dropout node", {}, "0.0-0.0",
          "Dropout", "Regularization", true, false}},
        NodeImplementationStatus::Implemented, 0});

    RegisterNode({NodeType::GRU, NodeCategory::Recurrent, "GRU", ICON_FA_REPEAT,
        {"gru", "recurrent", "sequence"}, 0, false,
        "Trainable gated recurrent unit sequence layer",
        "Engine training supports unidirectional and split-path bidirectional "
        "GRU with dropout=0.0. Bidirectional GRU currently uses the declared "
        "native CPU recurrent path.", "",
        {{"Input", PinType::Tensor, true,
          "Sequence tensor [batch, sequence, features]; features is derived as input_size."}},
        {{"Output", PinType::Tensor, true,
          "Full sequence [batch, sequence, hidden * directions] when return_sequences=true; otherwise [batch, hidden * directions]."},
         {"Hidden", PinType::Tensor, false,
          "Legacy compatibility pin only. Engine SequentialModel does not route a separate h_n output; leave disconnected."}},
        {{"input_size", "int", "0", "Auto-derived input feature count", {}, "Derived from the previous layer output.",
          "Input Size", "Recurrent", true, true},
         {"hidden_size", "int", "256", "Hidden-state width", {}, "1-1048576",
          "Hidden Size", "Recurrent", true, false},
         {"num_layers", "int", "1", "Number of stacked recurrent layers", {}, "1-1048576",
          "Layers", "Recurrent", true, false},
         {"bidirectional", "bool", "false", "Run explicit forward and reverse GRU branches", {}, "",
          "Bidirectional", "Recurrent", true, false},
         {"return_sequences", "bool", "false", "Return every timestep instead of the final timestep", {}, "",
          "Return Sequences", "Output", true, false},
         {"dropout", "float", "0.0", "Must remain 0.0; use an explicit Dropout node", {}, "0.0-0.0",
          "Dropout", "Regularization", true, false}},
        NodeImplementationStatus::Implemented, 0});

    RegisterNode({NodeType::RNN, NodeCategory::Recurrent, "RNN", ICON_FA_REPEAT,
        {"rnn", "recurrent", "sequence"}, 0, false,
        "Blocked simple-RNN compatibility node",
        "Studio has no RNN backend layer, Python binding, ModelBuilder module, "
        "or training owner. It must not be approximated with a GRU.", "",
        {{"Input", PinType::Tensor, true, "Legacy sequence input [batch, sequence, features]."}},
        {{"Output", PinType::Tensor, true, "Legacy recurrent output; unavailable while blocked."},
         {"Hidden", PinType::Tensor, false, "Optional legacy hidden state; unavailable while blocked."}},
        {{"input_size", "int", "0", "Legacy auto-derived feature count", {}, "Derived from the previous layer output.",
          "Input Size", "Recurrent", true, true},
         {"hidden_size", "int", "256", "Legacy hidden-state width", {}, "1-1048576",
          "Hidden Size", "Recurrent", true, false},
         {"num_layers", "int", "1", "Legacy stacked-layer count", {}, "1-1048576",
          "Layers", "Recurrent", true, false},
         {"bidirectional", "bool", "false", "Legacy bidirectional intent", {}, "",
          "Bidirectional", "Recurrent", true, false},
         {"return_sequences", "bool", "false", "Legacy full-sequence output intent", {}, "",
          "Return Sequences", "Output", true, false},
         {"dropout", "float", "0.0", "Legacy inter-layer dropout", {}, "0.0-1.0",
          "Dropout", "Regularization", true, false},
         {"nonlinearity", "string", "tanh", "Legacy activation intent from registry-era graphs", {}, "",
          "Nonlinearity", "Recurrent", true, false}},
        NodeImplementationStatus::Template, 0, "Blocked"});

    RegisterNode({NodeType::Bidirectional, NodeCategory::Recurrent, "Bidirectional", ICON_FA_REPEAT,
        {"bidirectional", "wrapper", "sequence"}, 0, false,
        "Blocked standalone bidirectional-wrapper compatibility node",
        "The bidirectional setting belongs to concrete LSTM/GRU nodes and must "
        "follow their own verified execution limits. No standalone wrapper binds "
        "an inner recurrent layer in GraphCompiler, ModelBuilder, or SequentialModel.", "",
        {{"Input", PinType::Tensor, true, "Legacy sequence input."}},
        {{"Output", PinType::Tensor, true, "Legacy merged directional output; unavailable while blocked."}},
        {{"merge_mode", "string", "concat", "Historical merge-mode text", {}, "",
          "Merge Mode", "Wrapper", true, false}},
        NodeImplementationStatus::Template, 0, "Blocked"});

    RegisterNode({NodeType::TimeDistributed, NodeCategory::Recurrent, "TimeDistributed Dense", ICON_FA_REPEAT,
        {"time", "distributed", "sequence", "token", "ner", "token classifier"}, 0, false,
        "Apply one shared Dense token-classifier projection independently to every timestep",
        "This is a specific bias-enabled Dense sequence head, not a generic wrapper. "
        "It reshapes [batch, sequence, features] to token rows, uses the existing "
        "trainable LinearModule, and restores the sequence shape.", "",
        {{"Input", PinType::Tensor, true, "Float32 sequence features [batch, sequence, features]."}},
        {{"Output", PinType::Tensor, true, "Float32 token outputs [batch, sequence, units]."}},
        {{"units", "int", "128", "Shared Dense output width for every timestep", {},
          "1-2147483647", "Per-timestep Output Units", "Projection", true, false}},
        NodeImplementationStatus::Implemented, 0});

    RegisterNode({NodeType::Dropout, NodeCategory::Regularization, "Dropout", ICON_FA_SHUFFLE,
        {"dropout", "regularization"}, 0, false,
        "Randomly suppress activations during training",
        "Applies inverted dropout during training and passes values through unchanged during evaluation.", "",
        {{"Input", PinType::Tensor, true, "Activations of any supported shape."}},
        {{"Output", PinType::Tensor, true, "Same shape as Input."}},
        {{"rate", "float", "0.5", "Probability of dropping each activation", {}, "0.0-0.999",
          "Drop Probability", "Regularization", true, false}},
        NodeImplementationStatus::Implemented, 0});

    RegisterNode({NodeType::BatchNorm, NodeCategory::Normalization, "BatchNorm", ICON_FA_SCALE_BALANCED,
        {"batchnorm", "normalization"}, 0, false,
        "Normalize rank-2 features with batch statistics and running estimates",
        "The Engine BatchNorm node is BatchNorm1D for Float32 [batch, features] input. "
        "Training updates running mean and variance; evaluation reuses those estimates.", "",
        {{"Input", PinType::Tensor, true, "Float32 feature activations [batch, features]."}},
        {{"Output", PinType::Tensor, true, "Normalized Float32 [batch, features]."}},
        {{"eps", "float", "1e-5", "Positive numerical stability term", {}, "0.000000001-3.402823e38",
          "Epsilon", "Normalization", true, false},
         {"momentum", "float", "0.1", "Running-statistics update momentum", {}, "0.0-1.0",
          "Momentum", "Normalization", true, false}},
        NodeImplementationStatus::Implemented, 0});

    RegisterNode({NodeType::LayerNorm, NodeCategory::Normalization, "LayerNorm", ICON_FA_SCALE_BALANCED,
        {"layernorm", "normalization", "transformer"}, 0, false,
        "Normalize the trailing feature dimensions of each sample", "", "",
        {{"Input", PinType::Tensor, true, "Feature or sequence activations to normalize."}},
        {{"Output", PinType::Tensor, true, "Normalized tensor with the same shape."}},
        {{"normalized_shape", "string", "",
          "Comma-separated trailing dimensions; empty uses the current feature width", {}, "",
          "Normalized Shape", "Normalization", false, false},
         {"eps", "float", "1e-5", "Positive numerical stability term", {}, "0.000000001-3.402823e38",
          "Epsilon", "Normalization", true, false},
         {"elementwise_affine", "bool", "true", "Learn per-element scale and bias", {}, "",
          "Elementwise Affine", "Normalization", false, true}},
        NodeImplementationStatus::Implemented, 0});

    RegisterNode({NodeType::GroupNorm, NodeCategory::Normalization, "GroupNorm", ICON_FA_SCALE_BALANCED,
        {"groupnorm", "normalization", "small batch", "channels"}, 0, false,
        "Normalize channels in groups for each sample (blocked Engine layer)",
        "The saved-graph contract matches the native GroupNormLayer constructor, but Studio has no "
        "ModelBuilder/SequentialModel owner and the backend path is not ArrayFire-first. Input must "
        "eventually be Float32 [H,W,C,N], with num_channels equal to C and divisible by num_groups.", "",
        {{"Input", PinType::Tensor, true, "Spatial activations [H,W,C,N]."}},
        {{"Output", PinType::Tensor, true,
          "Same-shape normalized activations; unavailable while the Engine layer is blocked."}},
        {{"num_groups", "int", "32", "Number of channel groups; must divide num_channels", {},
          "1-1048576", "Groups", "Normalization", true, false},
         {"num_channels", "int", "256", "Expected input channel count C", {},
          "1-1048576", "Channels", "Normalization", true, false},
         {"eps", "float", "1e-5", "Positive numerical-stability term", {},
          "0.000000001-1.0", "Epsilon", "Normalization", true, true},
         {"affine", "bool", "true", "Learn one scale and bias per channel", {}, "",
          "Affine", "Normalization", false, true}},
        NodeImplementationStatus::Template, 0, "Blocked"});

    RegisterNode({NodeType::InstanceNorm, NodeCategory::Normalization, "InstanceNorm", ICON_FA_SCALE_BALANCED,
        {"instancenorm", "instance normalization", "style transfer", "channels"}, 0, false,
        "Normalize each channel independently per sample (blocked Engine layer)",
        "The saved-graph contract matches the native InstanceNorm2DLayer constructor, but Studio has "
        "no ModelBuilder/SequentialModel owner and the backend path is not ArrayFire-first. Input must "
        "eventually be Float32 [H,W,C,N], with num_features equal to C.", "",
        {{"Input", PinType::Tensor, true, "Spatial activations [H,W,C,N]."}},
        {{"Output", PinType::Tensor, true,
          "Same-shape normalized activations; unavailable while the Engine layer is blocked."}},
        {{"num_features", "int", "64", "Expected input channel count C", {},
          "1-1048576", "Channels", "Normalization", true, false},
         {"eps", "float", "1e-5", "Positive numerical-stability term", {},
          "0.000000001-1.0", "Epsilon", "Normalization", true, true},
         {"affine", "bool", "false", "Learn one scale and bias per channel", {}, "",
          "Affine", "Normalization", false, true}},
        NodeImplementationStatus::Template, 0, "Blocked"});

    RegisterNode({NodeType::MultiHeadAttention, NodeCategory::Attention, "Multi-Head Attention", ICON_FA_BULLSEYE,
        {"attention", "transformer", "self-attention"}, 0, false,
        "Trainable self-attention over a sequence tensor",
        "Connect one sequence tensor to Query. Studio uses it as query, key, and value. "
        "Cross-attention and attention masks remain fail-closed until their "
        "multi-input runtime contract is implemented.",
        "",
        {{"Query", PinType::Tensor, true,
          "Sequence tensor [batch, sequence, features], used as query, key, and value."},
         {"Key", PinType::Tensor, false,
          "Reserved for cross-attention; connecting this pin is currently rejected."},
         {"Value", PinType::Tensor, false,
          "Reserved for cross-attention; connecting this pin is currently rejected."},
         {"Mask", PinType::Tensor, false,
          "Reserved for attention masking; connecting this pin is currently rejected."}},
        {{"Output", PinType::Tensor, true,
          "Self-attention output [batch, sequence, features]."}},
        {{"embed_dim", "int", "512", "Expected input feature width", {}, "1-65536",
          "Embedding Dimension", "Attention", true, false},
         {"num_heads", "int", "8",
          "Number of attention heads; embed_dim must divide evenly", {}, "1-4096",
          "Attention Heads", "Attention", true, false},
         {"dropout", "float", "0.0", "Attention dropout probability", {}, "0.0-0.999",
          "Dropout", "Regularization", false, false},
         {"use_bias", "bool", "true", "Enable bias in the attention projections", {}, "",
          "Use Bias", "Advanced", false, true}},
        NodeImplementationStatus::Implemented, 0});

    RegisterNode({NodeType::TransformerEncoder, NodeCategory::Attention, "Transformer Encoder", ICON_FA_BULLSEYE,
        {"transformer", "attention", "encoder"}, 0, false,
        "One trainable transformer encoder block for sequence tensors",
        "Connect Float32 [batch, sequence, d_model]. This node constructs one "
        "self-attention/feed-forward block; stack nodes to create multiple layers.", "",
        {{"Input", PinType::Tensor, true, "Float32 sequence [batch, sequence, d_model]."}},
        {{"Output", PinType::Tensor, true, "Same-shape encoded Float32 sequence."}},
        {{"d_model", "int", "512", "Required input/output feature width", {}, "1-65536",
          "Model Width", "Transformer", true, false},
         {"num_heads", "int", "8", "Attention heads; d_model must divide evenly", {}, "1-4096",
          "Attention Heads", "Transformer", true, false},
         {"dim_feedforward", "int", "2048", "Inner feed-forward feature width", {}, "1-1048576",
          "Feed-forward Width", "Transformer", true, false},
         {"dropout", "float", "0.1", "Training dropout probability", {}, "0.0-0.999",
          "Dropout", "Regularization", false, false},
         {"norm_first", "bool", "false", "Apply normalization before each sublayer", {}, "",
          "Pre-Norm", "Transformer", false, true}},
        NodeImplementationStatus::Implemented, 0});

    RegisterNode({NodeType::TransformerDecoder, NodeCategory::Attention, "Transformer Decoder", ICON_FA_BULLSEYE,
        {"transformer", "attention", "decoder", "causal"}, 0, false,
        "One decoder-only causal transformer block",
        "Connect Float32 [batch, sequence, d_model]. Stack nodes for depth. "
        "Connected Memory remains fail-closed until seq2seq cross-attention has a graph owner.", "",
        {{"Input", PinType::Tensor, true, "Target Float32 sequence [batch, sequence, d_model]."},
         {"Memory", PinType::Tensor, false, "Reserved optional encoder memory; a connected pin is rejected."}},
        {{"Output", PinType::Tensor, true, "Same-shape decoded Float32 sequence."}},
        {{"d_model", "int", "512", "Required input/output feature width", {}, "1-65536",
          "Model Width", "Transformer", true, false},
         {"num_heads", "int", "8", "Attention heads; d_model must divide evenly", {}, "1-4096",
          "Attention Heads", "Transformer", true, false},
         {"dim_feedforward", "int", "2048", "Inner feed-forward feature width", {}, "1-1048576",
          "Feed-forward Width", "Transformer", true, false},
         {"dropout", "float", "0.1", "Training dropout probability", {}, "0.0-0.999",
          "Dropout", "Regularization", false, false},
         {"norm_first", "bool", "false", "Apply normalization before each sublayer", {}, "",
          "Pre-Norm", "Transformer", false, true}},
        NodeImplementationStatus::Implemented, 0});

    RegisterNode({NodeType::PositionalEncoding, NodeCategory::Attention, "Positional Encoding", ICON_FA_BULLSEYE,
        {"transformer", "position", "encoding", "sequence"}, 0, false,
        "Deterministic sinusoidal encoding for sequence positions",
        "Adds fixed sine/cosine values to Float32 [batch, sequence, d_model]. "
        "The current module is native CPU-backed and has no trainable parameters.", "",
        {{"Input", PinType::Tensor, true, "Float32 sequence [batch, sequence, d_model]."}},
        {{"Output", PinType::Tensor, true, "Same-shape position-aware Float32 sequence."}},
        {{"d_model", "int", "512", "Required input/output feature width", {}, "1-65536",
          "Model Width", "Position", true, false},
         {"max_sequence_length", "int", "5000", "Largest accepted sequence length", {}, "1-1048576",
          "Maximum Sequence Length", "Position", true, false}},
        NodeImplementationStatus::Implemented, 0});

    RegisterNode({NodeType::SelfAttention, NodeCategory::Attention, "Self Attention", ICON_FA_BULLSEYE,
        {"attention", "self_attention", "transformer"}, 0, false,
        "Blocked self-attention compatibility node",
        "Saved graphs retain explicit Query, Key, Value, and Mask pins, but "
        "Studio has no distinct SelfAttention GraphCompiler/ModelBuilder owner. "
        "Use Multi-Head Attention for the supported unary self-attention path.", "",
        {{"Query", PinType::Tensor, true, "Legacy query tensor [batch, query length, embed_dim]."},
         {"Key", PinType::Tensor, true, "Legacy key tensor [batch, key/value length, embed_dim]."},
         {"Value", PinType::Tensor, true, "Legacy value tensor [batch, key/value length, embed_dim]."},
         {"Mask", PinType::Tensor, false, "Optional legacy attention mask."}},
        {{"Output", PinType::Tensor, true, "Legacy attention result; unavailable while blocked."},
         {"Attn Weights", PinType::Tensor, false, "Optional legacy per-head weights; unavailable while blocked."}},
        {{"embed_dim", "int", "512", "Legacy embedding width", {}, "1-1048576",
          "Embedding Dimension", "Attention", true, false},
         {"num_heads", "int", "8", "Legacy attention-head count", {}, "1-1048576",
          "Heads", "Attention", true, false},
         {"dropout", "float", "0.0", "Legacy attention-weight dropout", {}, "0.0-1.0",
          "Dropout", "Attention", true, false},
         {"batch_first", "bool", "true", "Legacy batch-first layout flag", {}, "",
          "Batch First", "Layout", true, false}},
        NodeImplementationStatus::Template, 0, "Blocked"});

    RegisterNode({NodeType::CrossAttention, NodeCategory::Attention, "Cross Attention", ICON_FA_BULLSEYE,
        {"attention", "cross_attention", "transformer"}, 0, false,
        "Blocked cross-attention compatibility node",
        "Saved graphs retain explicit Query, Key, Value, and Mask pins, but "
        "Studio has no multi-input CrossAttention GraphCompiler/ModelBuilder owner.", "",
        {{"Query", PinType::Tensor, true, "Legacy query tensor [batch, query length, embed_dim]."},
         {"Key", PinType::Tensor, true, "Legacy context key tensor [batch, key/value length, embed_dim]."},
         {"Value", PinType::Tensor, true, "Legacy context value tensor [batch, key/value length, embed_dim]."},
         {"Mask", PinType::Tensor, false, "Optional legacy cross-attention mask."}},
        {{"Output", PinType::Tensor, true, "Legacy attention result; unavailable while blocked."},
         {"Attn Weights", PinType::Tensor, false, "Optional legacy per-head weights; unavailable while blocked."}},
        {{"embed_dim", "int", "512", "Legacy embedding width", {}, "1-1048576",
          "Embedding Dimension", "Attention", true, false},
         {"num_heads", "int", "8", "Legacy attention-head count", {}, "1-1048576",
          "Heads", "Attention", true, false},
         {"dropout", "float", "0.0", "Legacy attention-weight dropout", {}, "0.0-1.0",
          "Dropout", "Attention", true, false},
         {"batch_first", "bool", "true", "Legacy batch-first layout flag", {}, "",
          "Batch First", "Layout", true, false}},
        NodeImplementationStatus::Template, 0, "Blocked"});

    RegisterNode({NodeType::LinearAttention, NodeCategory::Attention, "Linear Attention", ICON_FA_BULLSEYE,
        {"attention", "linear_attention", "performer"}, 0, false,
        "Blocked linear-attention compatibility node",
        "Saved graphs retain the historical linear-attention sketch, but no "
        "backend primitive or Studio training owner implements its advertised semantics.", "",
        {{"Query", PinType::Tensor, true, "Legacy query tensor."},
         {"Key", PinType::Tensor, true, "Legacy key tensor."},
         {"Value", PinType::Tensor, true, "Legacy value tensor."},
         {"Mask", PinType::Tensor, false, "Optional legacy attention mask."}},
        {{"Output", PinType::Tensor, true, "Legacy linear-attention result; unavailable while blocked."}},
        {{"embed_dim", "int", "512", "Legacy embedding width", {}, "1-1048576",
          "Embedding Dimension", "Attention", true, false},
         {"num_heads", "int", "8", "Legacy attention-head count", {}, "1-1048576",
          "Heads", "Attention", true, false},
         {"feature_map", "enum", "elu", "Legacy kernel feature-map sketch",
          {"elu", "relu", "favor+"}, "", "Feature Map", "Approximation", true, false},
         {"eps", "float", "1e-6", "Legacy numerical-stability epsilon", {}, "0.0-1.0",
          "Epsilon", "Approximation", true, false},
         {"causal", "bool", "false", "Legacy causal-attention flag", {}, "",
          "Causal", "Attention", true, false}},
        NodeImplementationStatus::Template, 0, "Blocked"});

    RegisterNode({NodeType::MaxPool2D, NodeCategory::Pooling, "MaxPool2D", ICON_FA_COMPRESS,
        {"maxpool", "pooling"}, 0, false,
        "Blocked 2D max-pooling layer retained for graph compatibility",
        "A backend pooling primitive exists, but GraphCompiler, ModelBuilder, "
        "and SequentialModel do not construct it for Studio training.", "",
        {{"Input", PinType::Tensor, true, "Legacy image feature-map input."}},
        {{"Output", PinType::Tensor, true,
          "Pooled feature map; unavailable at runtime while this node is blocked."}},
        {{"pool_size", "int", "2", "Legacy square pooling-window size", {}, "1-1048576",
          "Pool Size", "Pooling", true, false},
         {"stride", "int", "2", "Legacy spatial pooling stride", {}, "1-1048576",
          "Stride", "Pooling", true, false}},
        NodeImplementationStatus::Template, 0, "Blocked"});

    RegisterNode({NodeType::AvgPool2D, NodeCategory::Pooling, "AvgPool2D", ICON_FA_COMPRESS,
        {"avgpool", "average", "pooling"}, 0, false,
        "Blocked 2D average-pooling layer retained for graph compatibility",
        "A backend pooling primitive exists, but GraphCompiler, ModelBuilder, "
        "and SequentialModel do not construct it for Studio training.", "",
        {{"Input", PinType::Tensor, true, "Legacy image feature-map input."}},
        {{"Output", PinType::Tensor, true,
          "Pooled feature map; unavailable at runtime while this node is blocked."}},
        {{"pool_size", "int", "2", "Legacy square pooling-window size", {}, "1-1048576",
          "Pool Size", "Pooling", true, false},
         {"stride", "int", "2", "Legacy spatial pooling stride", {}, "1-1048576",
          "Stride", "Pooling", true, false}},
        NodeImplementationStatus::Template, 0, "Blocked"});

    RegisterNode({NodeType::GlobalMaxPool, NodeCategory::Pooling, "Global Max Pool", ICON_FA_COMPRESS,
        {"global", "max", "pooling"}, 0, false,
        "Blocked global max-pooling design node retained for graph compatibility",
        "No backend global-max layer or ModelBuilder/SequentialModel execution "
        "path currently owns this node.", "",
        {{"Input", PinType::Tensor, true, "Legacy image feature-map input."}},
        {{"Output", PinType::Tensor, true,
          "Channel summary; unavailable at runtime while this node is blocked."}},
        {}, NodeImplementationStatus::Template, 0, "Blocked"});

    RegisterNode({NodeType::GlobalAvgPool, NodeCategory::Pooling, "Global Avg Pool", ICON_FA_COMPRESS,
        {"global", "average", "pooling"}, 0, false,
        "Blocked global average-pooling layer retained for graph compatibility",
        "A native backend primitive exists, but GraphCompiler, ModelBuilder, "
        "and SequentialModel do not construct it for Studio training.", "",
        {{"Input", PinType::Tensor, true, "Legacy image feature-map input."}},
        {{"Output", PinType::Tensor, true,
          "Channel average; unavailable at runtime while this node is blocked."}},
        {}, NodeImplementationStatus::Template, 0, "Blocked"});

    RegisterNode({NodeType::AdaptiveAvgPool, NodeCategory::Pooling,
        "Adaptive Average Pool", ICON_FA_COMPRESS,
        {"pooling", "adaptive", "average"}, 0, false,
        "Blocked adaptive average-pooling design node retained for graph compatibility",
        "No backend adaptive-pooling layer, GraphCompiler extraction, "
        "ModelBuilder module, or SequentialModel execution path currently owns this node.",
        "",
        {{"Input", PinType::Tensor, true, "Legacy image feature-map input."}},
        {{"Output", PinType::Tensor, true,
          "Adaptively pooled feature map; unavailable at runtime while this node is blocked."}},
        {{"output_size", "int", "1", "Legacy square output size", {}, "1-1048576",
          "Output Size", "Pooling", true, false}},
        NodeImplementationStatus::Template, 0, "Blocked"});

    RegisterNode({NodeType::ConvTranspose2D, NodeCategory::Upsampling, "ConvTranspose2D", ICON_FA_EXPAND,
        {"convtranspose", "transposed", "convolution", "upsample"}, 0, false,
        "Blocked transposed-convolution layer retained for graph compatibility",
        "A native backend primitive exists, but GraphCompiler, ModelBuilder, "
        "and SequentialModel do not construct an executable Studio layer. "
        "ArrayFire-first forward/backward and training ownership remain unproven.",
        "",
        {{"Input", PinType::Tensor, true,
          "Legacy image feature-map input; no executable Engine layout contract exists yet."}},
        {{"Output", PinType::Tensor, true,
          "Upsampled feature map; unavailable at runtime while this node is blocked."}},
        {{"in_channels", "int", "64", "Legacy input-channel count", {}, "1-1048576",
          "Input Channels", "Transposed Convolution", true, false},
         {"out_channels", "int", "32", "Legacy output-channel count", {}, "1-1048576",
          "Output Channels", "Transposed Convolution", true, false},
         {"kernel_size", "int", "3", "Legacy square-kernel size", {}, "1-1048576",
          "Kernel Size", "Transposed Convolution", true, false},
         {"stride", "int", "2", "Legacy spatial stride", {}, "1-1048576",
          "Stride", "Transposed Convolution", true, false},
         {"padding", "int", "1", "Legacy symmetric input padding", {}, "0-1048576",
          "Padding", "Transposed Convolution", true, false},
         {"output_padding", "int", "1", "Legacy output-shape adjustment", {}, "0-1048576",
          "Output Padding", "Transposed Convolution", true, false}},
        NodeImplementationStatus::Template, 0, "Blocked"});

    RegisterNode({NodeType::Upsample, NodeCategory::Upsampling, "Upsample", ICON_FA_EXPAND,
        {"upsample", "resize", "interpolate"}, 0, false,
        "Blocked spatial upsampling layer retained for graph compatibility",
        "A native nearest/bilinear backend primitive exists, but GraphCompiler, "
        "ModelBuilder, and SequentialModel do not construct it for Studio training. "
        "The numeric mode field is retained only for saved-graph compatibility.",
        "",
        {{"Input", PinType::Tensor, true,
          "Legacy image feature-map input; no executable Engine layout contract exists yet."}},
        {{"Output", PinType::Tensor, true,
          "Upsampled feature map; unavailable at runtime while this node is blocked."}},
        {{"scale_factor", "int", "2", "Legacy positive spatial scale", {}, "1-1048576",
          "Scale Factor", "Upsampling", true, false},
         {"mode", "enum", "0", "Legacy interpolation code: 0=nearest, 1=bilinear",
          {"0", "1"}, "", "Interpolation Mode", "Upsampling", true, false}},
        NodeImplementationStatus::Template, 0, "Blocked"});

    RegisterNode({NodeType::PixelShuffle, NodeCategory::Upsampling, "Pixel Shuffle", ICON_FA_EXPAND,
        {"pixel", "shuffle", "subpixel", "upsample"}, 0, false,
        "Blocked depth-to-space layer retained for graph compatibility",
        "A native backend primitive exists, but GraphCompiler, ModelBuilder, "
        "and SequentialModel do not construct it for Studio training. Input "
        "channel divisibility and ArrayFire-first execution remain unowned.",
        "",
        {{"Input", PinType::Tensor, true,
          "Legacy image feature map whose channels must be divisible by upscale_factor squared."}},
        {{"Output", PinType::Tensor, true,
          "Depth-to-space result; unavailable at runtime while this node is blocked."}},
        {{"upscale_factor", "int", "2", "Legacy positive spatial upscale factor", {}, "1-1048576",
          "Upscale Factor", "Upsampling", true, false}},
        NodeImplementationStatus::Template, 0, "Blocked"});

    RegisterNode({NodeType::Flatten, NodeCategory::ShapeOps, "Flatten", ICON_FA_ARROWS_LEFT_RIGHT,
        {"flatten", "reshape"}, 0, false, "Collapse each sample to one feature dimension", "", "",
        {{"Input", PinType::Tensor, true, "Tensor with one or more sample dimensions"}},
        {{"Output", PinType::Tensor, true, "Tensor with sample dimensions flattened to one"}},
        {}, NodeImplementationStatus::Implemented, 0});

    RegisterNode({NodeType::Reshape, NodeCategory::ShapeOps, "Reshape", ICON_FA_ARROWS_LEFT_RIGHT,
        {"reshape", "view", "shape", "tensor"}, 0, false, "Reshape each sample without changing its element count", "", "",
        {{"Input", PinType::Tensor, true, "Tensor whose sample shape will be changed"}},
        {{"Output", PinType::Tensor, true, "Tensor with the requested sample shape"}},
        {{"shape", "string", "", "Comma-separated target sample dimensions; one dimension may be -1", {}, "", "Target Shape", "Shape", true, false}},
        NodeImplementationStatus::Implemented, 0});

    RegisterNode({NodeType::View, NodeCategory::ShapeOps, "View", ICON_FA_ARROWS_LEFT_RIGHT,
        {"view", "reshape", "shape", "tensor"}, 0, false, "View each sample with a different shape", "", "",
        {{"Input", PinType::Tensor, true, "Tensor whose sample shape will be changed"}},
        {{"Output", PinType::Tensor, true, "Tensor viewed with the requested sample shape"}},
        {{"shape", "string", "", "Comma-separated target sample dimensions; one dimension may be -1", {}, "", "Target Shape", "Shape", true, false}},
        NodeImplementationStatus::Implemented, 0});

    RegisterNode({NodeType::Permute, NodeCategory::ShapeOps, "Permute", ICON_FA_SHUFFLE,
        {"permute", "transpose", "axes", "shape", "tensor"}, 0, false, "Reorder every sample dimension", "", "",
        {{"Input", PinType::Tensor, true, "Tensor whose sample dimensions will be reordered"}},
        {{"Output", PinType::Tensor, true, "Tensor with dimensions in the requested order"}},
        {{"dims", "string", "", "Comma-separated dimension order containing every input axis once", {}, "", "Dimension Order", "Shape", true, false}},
        NodeImplementationStatus::Implemented, 0});

    RegisterNode({NodeType::Squeeze, NodeCategory::ShapeOps, "Squeeze", ICON_FA_COMPRESS,
        {"squeeze", "shape", "dimension", "tensor"}, 0, false, "Remove singleton sample dimensions", "", "",
        {{"Input", PinType::Tensor, true, "Tensor containing one or more size-one dimensions"}},
        {{"Output", PinType::Tensor, true, "Tensor with selected singleton dimensions removed"}},
        {{"dim", "int", "-1", "Dimension to remove, or -1 to remove all singleton dimensions", {}, "", "Dimension", "Shape"}},
        NodeImplementationStatus::Implemented, 0});

    RegisterNode({NodeType::Unsqueeze, NodeCategory::ShapeOps, "Unsqueeze", ICON_FA_EXPAND,
        {"unsqueeze", "shape", "dimension", "tensor"}, 0, false, "Insert a singleton sample dimension", "", "",
        {{"Input", PinType::Tensor, true, "Tensor whose sample rank will be increased"}},
        {{"Output", PinType::Tensor, true, "Tensor with a size-one dimension inserted"}},
        {{"dim", "int", "0", "Position at which to insert the singleton dimension", {}, "", "Dimension", "Shape"}},
        NodeImplementationStatus::Implemented, 0});

    RegisterNode({NodeType::Split, NodeCategory::ShapeOps, "Split", ICON_FA_CODE_BRANCH,
        {"split", "chunk", "shape", "tensor"}, 0, false, "Split tensor along a dimension", "", "",
        {{"Input", PinType::Tensor, true, "Input"}},
        {{"Output 1", PinType::Tensor, true, "First split"}, {"Output 2", PinType::Tensor, true, "Second split"}},
        {{"split_size", "int", "2", "Split size", {}, ""}, {"dim", "int", "0", "Dimension", {}, ""}},
        NodeImplementationStatus::Template, 0});

    RegisterNode({NodeType::Concatenate, NodeCategory::MergeOps, "Concatenate", ICON_FA_CODE_BRANCH,
        {"concatenate", "concat", "cat", "merge", "tensor"}, 0, false, "Concatenate tensors along a dimension", "", "",
        {{"Input 1", PinType::Tensor, true, "First input"}, {"Input 2", PinType::Tensor, true, "Second input"},
         {"Input 3+", PinType::Tensor, false, "Optional additional inputs", true, 0, -1}},
        {{"Output", PinType::Tensor, true, "Concatenated"}},
        {{"dim", "int", "1", "Dimension", {}, ""}},
        NodeImplementationStatus::Implemented, 0});

    RegisterNode({NodeType::Add, NodeCategory::MergeOps, "Add", ICON_FA_PLUS,
        {"add", "sum", "merge", "tensor"}, 0, false, "Add tensors elementwise", "", "",
        {{"Input 1", PinType::Tensor, true, "First input"}, {"Input 2", PinType::Tensor, true, "Second input"},
         {"Input 3+", PinType::Tensor, false, "Optional additional inputs", true, 0, -1}},
        {{"Output", PinType::Tensor, true, "Sum"}},
        {}, NodeImplementationStatus::Implemented, 0});

    RegisterNode({NodeType::Multiply, NodeCategory::MergeOps, "Multiply", ICON_FA_XMARK,
        {"multiply", "mul", "product", "merge", "tensor"}, 0, false, "Multiply tensors elementwise", "", "",
        {{"Input 1", PinType::Tensor, true, "First input"}, {"Input 2", PinType::Tensor, true, "Second input"},
         {"Input 3+", PinType::Tensor, false, "Optional additional inputs", true, 0, -1}},
        {{"Output", PinType::Tensor, true, "Product"}},
        {}, NodeImplementationStatus::Implemented, 0});

    RegisterNode({NodeType::Average, NodeCategory::MergeOps, "Average", ICON_FA_CALCULATOR,
        {"average", "mean", "merge", "tensor"}, 0, false, "Average tensors elementwise", "", "",
        {{"Input 1", PinType::Tensor, true, "First input"}, {"Input 2", PinType::Tensor, true, "Second input"},
         {"Input 3+", PinType::Tensor, false, "Optional additional inputs", true, 0, -1}},
        {{"Output", PinType::Tensor, true, "Average"}},
        {}, NodeImplementationStatus::Implemented, 0});

    RegisterNode({NodeType::TensorSum, NodeCategory::Analytics, "Tensor Sum", ICON_FA_CALCULATOR,
        {"tensor", "sum", "reduce", "reduction"}, 0, false, "Sum values along sample dimensions", "", "",
        {{"Input", PinType::Tensor, true, "Tensor whose sample dimensions will be reduced"}},
        {{"Output", PinType::Tensor, true, "Reduced tensor with the batch dimension preserved"}},
        {{"dim", "int", "-1", "Sample dimension to reduce, or -1 for all sample values", {}, "", "Dimension", "Reduction"},
         {"keepdim", "bool", "false", "Retain reduced sample dimensions with size one", {}, "", "Keep Dimension", "Reduction"}},
        NodeImplementationStatus::Implemented, 0});

    RegisterNode({NodeType::TensorMean, NodeCategory::Analytics, "Tensor Mean", ICON_FA_CALCULATOR,
        {"tensor", "mean", "average", "reduce", "reduction"}, 0, false, "Average values along sample dimensions", "", "",
        {{"Input", PinType::Tensor, true, "Tensor whose sample dimensions will be reduced"}},
        {{"Output", PinType::Tensor, true, "Reduced tensor with the batch dimension preserved"}},
        {{"dim", "int", "-1", "Sample dimension to reduce, or -1 for all sample values", {}, "", "Dimension", "Reduction"},
         {"keepdim", "bool", "false", "Retain reduced sample dimensions with size one", {}, "", "Keep Dimension", "Reduction"}},
        NodeImplementationStatus::Implemented, 0});

    RegisterNode({NodeType::TensorMax, NodeCategory::Analytics, "Tensor Max", ICON_FA_CALCULATOR,
        {"tensor", "max", "maximum", "reduce", "reduction"}, 0, false, "Take the maximum along sample dimensions", "", "",
        {{"Input", PinType::Tensor, true, "Tensor whose sample dimensions will be reduced"}},
        {{"Output", PinType::Tensor, true, "Reduced tensor with the batch dimension preserved"}},
        {{"dim", "int", "-1", "Sample dimension to reduce, or -1 for all sample values", {}, "", "Dimension", "Reduction"},
         {"keepdim", "bool", "false", "Retain reduced sample dimensions with size one", {}, "", "Keep Dimension", "Reduction"}},
        NodeImplementationStatus::Implemented, 0});

    RegisterNode({NodeType::TensorMin, NodeCategory::Analytics, "Tensor Min", ICON_FA_CALCULATOR,
        {"tensor", "min", "minimum", "reduce", "reduction"}, 0, false, "Take the minimum along sample dimensions", "", "",
        {{"Input", PinType::Tensor, true, "Tensor whose sample dimensions will be reduced"}},
        {{"Output", PinType::Tensor, true, "Reduced tensor with the batch dimension preserved"}},
        {{"dim", "int", "-1", "Sample dimension to reduce, or -1 for all sample values", {}, "", "Dimension", "Reduction"},
         {"keepdim", "bool", "false", "Retain reduced sample dimensions with size one", {}, "", "Keep Dimension", "Reduction"}},
        NodeImplementationStatus::Implemented, 0});

    RegisterNode({NodeType::TensorProd, NodeCategory::Analytics, "Tensor Prod", ICON_FA_CALCULATOR,
        {"tensor", "prod", "product", "reduce", "reduction"}, 0, false, "Multiply values along sample dimensions", "", "",
        {{"Input", PinType::Tensor, true, "Tensor whose sample dimensions will be reduced"}},
        {{"Output", PinType::Tensor, true, "Reduced tensor with the batch dimension preserved"}},
        {{"dim", "int", "-1", "Sample dimension to reduce, or -1 for all sample values", {}, "", "Dimension", "Reduction"},
         {"keepdim", "bool", "false", "Retain reduced sample dimensions with size one", {}, "", "Keep Dimension", "Reduction"}},
        NodeImplementationStatus::Implemented, 0});

    RegisterNode({NodeType::TensorVar, NodeCategory::Analytics, "Tensor Var", ICON_FA_CALCULATOR,
        {"tensor", "var", "variance", "reduce", "reduction"}, 0, false, "Compute population variance along sample dimensions", "", "",
        {{"Input", PinType::Tensor, true, "Tensor whose sample dimensions will be reduced"}},
        {{"Output", PinType::Tensor, true, "Population variance with the batch dimension preserved"}},
        {{"dim", "int", "-1", "Sample dimension to reduce, or -1 for all sample values", {}, "", "Dimension", "Reduction"},
         {"keepdim", "bool", "false", "Retain reduced sample dimensions with size one", {}, "", "Keep Dimension", "Reduction"}},
        NodeImplementationStatus::Implemented, 0});

    RegisterNode({NodeType::TensorStd, NodeCategory::Analytics, "Tensor Std", ICON_FA_CALCULATOR,
        {"tensor", "std", "standard deviation", "reduce", "reduction"}, 0, false, "Compute population standard deviation along sample dimensions", "", "",
        {{"Input", PinType::Tensor, true, "Tensor whose sample dimensions will be reduced"}},
        {{"Output", PinType::Tensor, true, "Population standard deviation with the batch dimension preserved"}},
        {{"dim", "int", "-1", "Sample dimension to reduce, or -1 for all sample values", {}, "", "Dimension", "Reduction"},
         {"keepdim", "bool", "false", "Retain reduced sample dimensions with size one", {}, "", "Keep Dimension", "Reduction"}},
        NodeImplementationStatus::Implemented, 0});

    RegisterNode({NodeType::TensorBroadcastTo, NodeCategory::ShapeOps, "Tensor Broadcast To", ICON_FA_EXPAND,
        {"tensor", "broadcast", "shape", "expand"}, 0, false, "Broadcast tensor to a target shape", "", "",
        {{"Input", PinType::Tensor, true, "Tensor with dimensions compatible with the target shape"}},
        {{"Output", PinType::Tensor, true, "Tensor broadcast to the target sample shape"}},
        {{"shape", "string", "", "Comma-separated positive target sample dimensions", {}, "", "Target Shape", "Shape", true, false}},
        NodeImplementationStatus::Implemented, 0});

    RegisterNode({NodeType::TensorExpand, NodeCategory::ShapeOps, "Tensor Expand", ICON_FA_EXPAND,
        {"tensor", "expand", "broadcast", "shape"}, 0, false, "Materialize tensor expanded to a target shape", "", "",
        {{"Input", PinType::Tensor, true, "Tensor with dimensions compatible with the target shape"}},
        {{"Output", PinType::Tensor, true, "Materialized tensor expanded to the target sample shape"}},
        {{"shape", "string", "", "Comma-separated positive target sample dimensions", {}, "", "Target Shape", "Shape", true, false}},
        NodeImplementationStatus::Implemented, 0});

    RegisterNode({NodeType::TensorIndexSelect, NodeCategory::ShapeOps, "Tensor Index Select", ICON_FA_LIST_CHECK,
        {"tensor", "index", "select", "gather", "slice"}, 0, false, "Select entries along one dimension by index list", "", "",
        {{"Input", PinType::Tensor, true, "Tensor from which values will be selected"}},
        {{"Output", PinType::Tensor, true, "Tensor containing the selected entries"}},
        {{"dim", "int", "0", "Dimension along which to select", {}, "", "Dimension", "Selection"},
         {"indices", "string", "", "Comma-separated indices; negative indices count from the end", {}, "", "Indices", "Selection", true, false}},
        NodeImplementationStatus::Implemented, 0});

    RegisterNode({NodeType::TensorPow, NodeCategory::Analytics, "Tensor Pow", ICON_FA_CALCULATOR,
        {"tensor", "pow", "power", "elementwise"}, 0, false, "Raise tensor values to a scalar power", "", "",
        {{"Input", PinType::Tensor, true, "Tensor values used as the power base"}},
        {{"Output", PinType::Tensor, true, "Elementwise power result with the same shape"}},
        {{"exponent", "float", "2.0", "Scalar exponent applied to every value", {}, "", "Exponent", "Operation"}},
        NodeImplementationStatus::Implemented, 0});

    RegisterNode({NodeType::TensorSqrt, NodeCategory::Analytics, "Tensor Sqrt", ICON_FA_CALCULATOR,
        {"tensor", "sqrt", "square root", "elementwise"}, 0, false, "Elementwise square root", "", "",
        {{"Input", PinType::Tensor, true, "Tensor containing non-negative values"}},
        {{"Output", PinType::Tensor, true, "Elementwise square root with the same shape"}},
        {}, NodeImplementationStatus::Implemented, 0});

    RegisterNode({NodeType::TensorExp, NodeCategory::Analytics, "Tensor Exp", ICON_FA_CALCULATOR,
        {"tensor", "exp", "exponential", "elementwise"}, 0, false, "Elementwise exponential", "", "",
        {{"Input", PinType::Tensor, true, "Input tensor"}},
        {{"Output", PinType::Tensor, true, "Elementwise exponential with the same shape"}},
        {}, NodeImplementationStatus::Implemented, 0});

    RegisterNode({NodeType::TensorLog, NodeCategory::Analytics, "Tensor Log", ICON_FA_CALCULATOR,
        {"tensor", "log", "natural log", "elementwise"}, 0, false, "Elementwise natural log", "", "",
        {{"Input", PinType::Tensor, true, "Tensor containing positive values"}},
        {{"Output", PinType::Tensor, true, "Elementwise natural logarithm with the same shape"}},
        {}, NodeImplementationStatus::Implemented, 0});

    RegisterNode({NodeType::TensorAbs, NodeCategory::Analytics, "Tensor Abs", ICON_FA_CALCULATOR,
        {"tensor", "abs", "absolute", "elementwise"}, 0, false, "Elementwise absolute value", "", "",
        {{"Input", PinType::Tensor, true, "Input tensor"}},
        {{"Output", PinType::Tensor, true, "Elementwise absolute value with the same shape"}},
        {}, NodeImplementationStatus::Implemented, 0});

    RegisterNode({NodeType::TensorSign, NodeCategory::Analytics, "Tensor Sign", ICON_FA_CALCULATOR,
        {"tensor", "sign", "elementwise"}, 0, false, "Elementwise sign", "", "",
        {{"Input", PinType::Tensor, true, "Input tensor"}},
        {{"Output", PinType::Tensor, true, "Elementwise sign values with the same shape"}},
        {}, NodeImplementationStatus::Implemented, 0});

    RegisterNode({NodeType::TensorClip, NodeCategory::Analytics, "Tensor Clip", ICON_FA_CALCULATOR,
        {"tensor", "clip", "clamp", "elementwise"}, 0, false, "Clip tensor values to a scalar range", "", "",
        {{"Input", PinType::Tensor, true, "Tensor whose values will be clamped"}},
        {{"Output", PinType::Tensor, true, "Clamped tensor with the same shape"}},
        {{"min", "float", "0.0", "Inclusive lower bound", {}, "", "Minimum", "Range"},
         {"max", "float", "1.0", "Inclusive upper bound", {}, "", "Maximum", "Range"}},
        NodeImplementationStatus::Implemented, 0});

    RegisterNode({NodeType::TensorDot, NodeCategory::Analytics, "Tensor Dot", ICON_FA_CALCULATOR,
        {"tensor", "dot", "vector", "linalg"}, 0, false, "Compute vector or row-wise batch dot product", "", "",
        {{"A", PinType::Tensor, true, "Left 1D vector or [batch, features] tensor"},
         {"B", PinType::Tensor, true, "Right tensor with the same shape and data type as A"}},
        {{"Output", PinType::Tensor, true, "Scalar dot product or one result per batch row"}},
        {}, NodeImplementationStatus::Implemented, 0});

    RegisterNode({NodeType::TensorBatchMatMul, NodeCategory::Analytics, "Tensor Batch MatMul", ICON_FA_CALCULATOR,
        {"tensor", "batch", "matmul", "matrix", "linalg"}, 0, false, "Blocked batched matrix multiplication retained for graph compatibility", "", "",
        {{"A", PinType::Tensor, true, "Left tensor [batch, rows, inner]"},
         {"B", PinType::Tensor, true, "Right tensor [batch, inner, columns] with the same data type"}},
        {{"Output", PinType::Tensor, true, "Batched matrix product [batch, rows, columns]"}},
        {}, NodeImplementationStatus::Template, 0, "Blocked"});

    RegisterNode({NodeType::TensorCompare, NodeCategory::Analytics, "Tensor Compare", ICON_FA_CALCULATOR,
        {"tensor", "compare", "greater", "less", "equal", "mask"}, 0, false, "Compare A with a scalar or a second tensor", "", "",
        {{"A", PinType::Tensor, true, "Left operand tensor"},
         {"B", PinType::Tensor, false, "Optional right operand tensor; when connected, Scalar is ignored"}},
        {{"Mask", PinType::Tensor, true, "Zero-or-one comparison mask matching the input shape"}},
        {{"op", "enum", ">", "Comparison applied to A and Scalar or A and B", {">", ">=", "<", "<=", "==", "!="}, "", "Operator", "Comparison"},
         {"scalar", "float", "0.0", "Right operand used when B is not connected", {}, "", "Scalar", "Comparison"}},
        NodeImplementationStatus::Implemented, 0});

    RegisterNode({NodeType::TensorLogicalMask, NodeCategory::Analytics, "Tensor Logical Mask", ICON_FA_CALCULATOR,
        {"tensor", "logical", "mask", "not", "and", "or"}, 0, false, "Invert one mask or combine two masks", "", "",
        {{"A", PinType::Tensor, true, "Left zero-or-nonzero mask"},
         {"B", PinType::Tensor, false, "Optional right mask required by and/or"}},
        {{"Mask", PinType::Tensor, true, "Zero-or-one logical mask matching the input shape"}},
        {{"op", "enum", "not", "Use not with A only; use and/or when B is connected", {"not", "and", "or"}, "", "Operator", "Logical"}},
        NodeImplementationStatus::Implemented, 0});

    RegisterNode(WithPropertiesEditor({NodeType::Embedding, NodeCategory::Layers, "Embedding", ICON_FA_CUBES,
        {"embedding", "lookup", "token", "vocabulary", "pretrained"}, 0, false,
        "Look up trainable dense vectors for exact integer token IDs",
        "Accepts rank-1 or rank-2 token IDs and produces one Float32 vector per "
        "token. The table participates in optimizer updates unless a loaded "
        "pretrained matrix is frozen. Padding produces zeros and receives no gradient.", "",
        {{"Indices", PinType::Tensor, true,
          "Exact integer token IDs [sequence] or [batch, sequence]; Float32, Int32, and Int64 ingress are accepted."}},
        {{"Embeddings", PinType::Tensor, true,
          "Float32 vectors [sequence, embedding_dim] or [batch, sequence, embedding_dim]."}},
        {{"num_embeddings", "int", "10000", "Number of token rows in the lookup table", {},
          "2-2147483647", "Vocabulary Size", "Shape", true, false},
         {"embedding_dim", "int", "256", "Features in each learned token vector", {},
          "1-2147483647", "Embedding Dimension", "Shape", true, false},
         {"padding_idx", "int", "-1", "Token row forced to zero and excluded from gradients; -1 disables padding", {},
          "", "Padding Index", "Lookup", false, false},
         {"max_norm", "float", "0", "Maximum L2 norm applied before lookup; 0 disables clipping", {},
          "0-3.402823e38", "Maximum Norm", "Lookup", false, true},
         {"freeze", "bool", "false", "Do not update a loaded pretrained table", {},
          "", "Freeze Loaded Weights", "Weights", false, false},
         {"weights_file", "file", "", "Optional whitespace-delimited Float32 matrix with num_embeddings rows and embedding_dim columns", {},
          "", "Pretrained Weights", "Weights", false, false},
         {"init_mode", "enum", "normal", "Starter-matrix mode used only by Build, Save, and Use in the configuration dialog", {"normal", "uniform", "one_hot"},
          "", "Starter Initialization", "Starter Matrix", false, true, ParameterConsumption::UiOnly},
         {"output_weights_file", "file", "", "Destination used only when the dialog builds a starter matrix", {},
          "", "Starter Matrix Output", "Starter Matrix", false, true, ParameterConsumption::UiOnly}},
        NodeImplementationStatus::Implemented, 0}, NodePropertiesEditor::Dialog));
}

// =============================================================================
// Activation Nodes
// =============================================================================
void NodeMetadataRegistry::InitializeActivationNodes() {
    RegisterNode({NodeType::ReLU, NodeCategory::Activation, "ReLU", ICON_FA_BOLT,
        {"relu", "activation"}, 0, false, "ReLU activation", "", "",
        {{"Input", PinType::Tensor, true, "Input"}},
        {{"Output", PinType::Tensor, true, "Activated"}},
        {}, NodeImplementationStatus::Implemented, 0});

    RegisterNode({NodeType::Sigmoid, NodeCategory::Activation, "Sigmoid", ICON_FA_BOLT,
        {"sigmoid", "logistic"}, 0, false, "Sigmoid activation", "", "",
        {{"Input", PinType::Tensor, true, "Input"}},
        {{"Output", PinType::Tensor, true, "Activated"}},
        {}, NodeImplementationStatus::Implemented, 0});

    RegisterNode({NodeType::Softmax, NodeCategory::Activation, "Softmax", ICON_FA_BOLT,
        {"softmax", "probability"}, 0, false, "Softmax activation", "", "",
        {{"Input", PinType::Tensor, true, "Input"}},
        {{"Output", PinType::Tensor, true, "Probabilities"}},
        {},
        NodeImplementationStatus::Implemented, 0});

    RegisterNode({NodeType::GELU, NodeCategory::Activation, "GELU", ICON_FA_BOLT,
        {"gelu", "gaussian"}, 0, false, "GELU activation", "", "",
        {{"Input", PinType::Tensor, true, "Input"}},
        {{"Output", PinType::Tensor, true, "Activated"}},
        {}, NodeImplementationStatus::Implemented, 0});

    RegisterNode({NodeType::Tanh, NodeCategory::Activation, "Tanh", ICON_FA_BOLT,
        {"tanh", "hyperbolic"}, 0, false, "Tanh activation", "", "",
        {{"Input", PinType::Tensor, true, "Input"}},
        {{"Output", PinType::Tensor, true, "Activated"}},
        {}, NodeImplementationStatus::Implemented, 0});

    RegisterNode({NodeType::LeakyReLU, NodeCategory::Activation, "Leaky ReLU", ICON_FA_BOLT,
        {"leaky", "relu"}, 0, false, "Leaky ReLU", "", "",
        {{"Input", PinType::Tensor, true, "Input"}},
        {{"Output", PinType::Tensor, true, "Activated"}},
        {{"negative_slope", "float", "0.01", "Non-negative slope for negative inputs", {}, "0-3.402823e38"}},
        NodeImplementationStatus::Implemented, 0});

    RegisterNode({NodeType::ELU, NodeCategory::Activation, "ELU", ICON_FA_BOLT,
        {"elu", "activation"}, 0, false, "Exponential linear unit", "", "",
        {{"Input", PinType::Tensor, true, "Input"}},
        {{"Output", PinType::Tensor, true, "Activated"}},
        {{"alpha", "float", "1.0", "Positive negative-saturation scale", {}, ">0"}},
        NodeImplementationStatus::Implemented, 0});

    RegisterNode({NodeType::Swish, NodeCategory::Activation, "Swish", ICON_FA_BOLT,
        {"swish", "activation", "silu"}, 0, false, "Swish activation", "", "",
        {{"Input", PinType::Tensor, true, "Input"}},
        {{"Output", PinType::Tensor, true, "Activated"}},
        {}, NodeImplementationStatus::Implemented, 0});

    RegisterNode({NodeType::Mish, NodeCategory::Activation, "Mish", ICON_FA_BOLT,
        {"mish", "activation"}, 0, false, "Mish activation", "", "",
        {{"Input", PinType::Tensor, true, "Input"}},
        {{"Output", PinType::Tensor, true, "Activated"}},
        {}, NodeImplementationStatus::Implemented, 0});
}

// =============================================================================
// Training Nodes
// =============================================================================
void NodeMetadataRegistry::InitializeTrainingNodes() {
    RegisterNode({NodeType::MSELoss, NodeCategory::Training, "MSE Loss", ICON_FA_CHART_LINE,
        {"mse", "loss", "regression", "criterion", "objective", "optimization"},
        0, false, "Squared-error loss for same-shaped numeric predictions and targets", "", "",
        {{"Predictions", PinType::Tensor, true, "Numeric model predictions"},
         {"Targets", PinType::Labels, true, "Same-shaped numeric target values"}},
        {{"Loss", PinType::Loss, true, "Loss value"}},
        {{"reduction", "enum", "mean", "Reduction", {"mean", "sum", "none"}, ""}},
        NodeImplementationStatus::Implemented, 0});

    RegisterNode({NodeType::CrossEntropyLoss, NodeCategory::Training, "CrossEntropy / Token CE", ICON_FA_CHART_PIE,
        {"crossentropy", "cross entropy", "ce", "classification", "multiclass",
         "token loss", "sequence", "ner", "criterion", "objective",
         "optimization", "loss"},
        0, false,
        "Class or token-level cross-entropy loss", "", "",
        {{"Logits", PinType::Tensor, true, "Class logits [N,C] or token logits [N,T,C]"},
         {"Labels", PinType::Labels, true, "Class labels [N] or token labels [N,T]"}},
        {{"Loss", PinType::Loss, true, "Loss value"}},
        {{"reduction", "enum", "mean", "Reduction", {"mean", "sum", "none"}, ""},
         {"ignore_index", "int", "-100", "Padding label ignore index", {}, ""},
         {"label_smoothing", "float", "0.0", "Label smoothing", {}, ""},
         {"class_weight", "enum", "none", "Class weight mode", {"none", "manual", "balanced"}, ""},
         {"class_weights", "string", "", "Manual per-class weights", {}, ""}},
        NodeImplementationStatus::Implemented, 0});

    RegisterNode({NodeType::FocalLoss, NodeCategory::Training, "Focal Loss", ICON_FA_CHART_PIE,
        {"focal", "focal loss", "classification", "multiclass", "imbalanced",
         "class imbalance", "criterion", "objective", "optimization", "loss"},
        0, false,
        "Focal loss for imbalanced multiclass classification logits", "", "",
        {{"Logits", PinType::Tensor, true, "Class logits [N,C]"},
         {"Labels", PinType::Labels, true, "Class labels [N]"}},
        {{"Loss", PinType::Loss, true, "Loss value"}},
        {{"reduction", "enum", "mean", "Reduction", {"mean", "sum", "none"}, ""},
         {"alpha", "float", "0.25", "Class imbalance scale", {}, ""},
        {"gamma", "float", "2.0", "Focusing parameter", {}, ""}},
        NodeImplementationStatus::Implemented, 0});

    RegisterNode({NodeType::SoftDiceLoss, NodeCategory::Training, "Soft Dice Loss", ICON_FA_CHART_PIE,
        {"soft dice", "dice", "dice loss", "segmentation", "mask",
         "class imbalance", "criterion", "objective", "optimization", "loss"},
        0, false,
        "Soft Dice loss for probability masks and same-shaped Float32 targets", "", "",
        {{"Predictions", PinType::Tensor, true, "Probability mask predictions"},
         {"Targets", PinType::Labels, true, "Same-shaped Float32 target masks"}},
        {{"Loss", PinType::Loss, true, "Loss value"}},
        {{"reduction", "enum", "mean", "Reduction", {"mean", "sum", "none"}, ""},
         {"smooth", "float", "1.0", "Smoothing constant", {}, ""}},
        NodeImplementationStatus::Implemented, 0});

    RegisterNode({NodeType::TverskyLoss, NodeCategory::Training, "Tversky Loss", ICON_FA_CHART_PIE,
        {"tversky", "tversky loss", "segmentation", "mask", "dice",
         "false positive", "false negative", "class imbalance", "criterion",
         "objective", "optimization", "loss"},
        0, false,
        "Tversky loss for imbalanced probability masks and same-shaped Float32 targets", "", "",
        {{"Predictions", PinType::Tensor, true, "Probability mask predictions"},
         {"Targets", PinType::Labels, true, "Same-shaped Float32 target masks"}},
        {{"Loss", PinType::Loss, true, "Loss value"}},
        {{"reduction", "enum", "mean", "Reduction", {"mean", "sum", "none"}, ""},
         {"alpha", "float", "0.5", "False-positive penalty", {}, ""},
         {"beta", "float", "0.5", "False-negative penalty", {}, ""},
         {"smooth", "float", "1.0", "Smoothing constant", {}, ""}},
        NodeImplementationStatus::Implemented, 0});

    RegisterNode({NodeType::JaccardLoss, NodeCategory::Training, "Jaccard / IoU Loss", ICON_FA_CHART_PIE,
        {"jaccard", "jaccard loss", "iou", "iou loss", "intersection over union",
         "segmentation", "mask", "criterion", "objective", "optimization", "loss"},
        0, false,
        "Jaccard/IoU loss for probability masks and same-shaped Float32 targets", "", "",
        {{"Predictions", PinType::Tensor, true, "Probability mask predictions"},
         {"Targets", PinType::Labels, true, "Same-shaped Float32 target masks"}},
        {{"Loss", PinType::Loss, true, "Loss value"}},
        {{"reduction", "enum", "mean", "Reduction", {"mean", "sum", "none"}, ""},
         {"smooth", "float", "1.0", "Smoothing constant", {}, ""}},
        NodeImplementationStatus::Implemented, 0});

    RegisterNode({NodeType::BCELoss, NodeCategory::Training, "BCE Loss", ICON_FA_CHART_PIE,
        {"bce", "binary", "binary cross entropy", "classification", "criterion",
         "objective", "optimization", "loss"},
        0, false, "Binary cross-entropy loss for probability predictions", "", "",
        {{"Predictions", PinType::Tensor, true, "Predicted probabilities"},
         {"Targets", PinType::Labels, true, "Same-shaped binary Float32 targets"}},
        {{"Loss", PinType::Loss, true, "Loss value"}},
        {{"reduction", "enum", "mean", "Reduction", {"mean", "sum", "none"}, ""}},
        NodeImplementationStatus::Implemented, 0});

    RegisterNode({NodeType::BCEWithLogits, NodeCategory::Training, "BCE with Logits", ICON_FA_CHART_PIE,
        {"bce", "bcewithlogits", "binary", "binary cross entropy", "logits",
         "classification", "criterion", "objective", "optimization", "loss"},
        0, false,
        "Numerically stable binary cross-entropy loss for logits", "", "",
        {{"Logits", PinType::Tensor, true, "Binary logits"},
         {"Targets", PinType::Labels, true, "Same-shaped binary Float32 targets"}},
        {{"Loss", PinType::Loss, true, "Loss value"}},
        {{"reduction", "enum", "mean", "Reduction", {"mean", "sum", "none"}, ""},
         {"pos_weight", "float", "1.0", "Positive-class weight", {}, ""}},
        NodeImplementationStatus::Implemented, 0});

    RegisterNode({NodeType::L1Loss, NodeCategory::Training, "L1 Loss", ICON_FA_CHART_LINE,
        {"l1", "mae", "mean absolute error", "absolute", "regression",
         "criterion", "objective", "optimization", "loss"},
        0, false, "Absolute-error loss for same-shaped regression values", "", "",
        {{"Predictions", PinType::Tensor, true, "Numeric model predictions"},
         {"Targets", PinType::Labels, true, "Same-shaped numeric target values"}},
        {{"Loss", PinType::Loss, true, "Loss value"}},
        {{"reduction", "enum", "mean", "Reduction", {"mean", "sum", "none"}, ""}},
        NodeImplementationStatus::Implemented, 0});

    RegisterNode({NodeType::SmoothL1Loss, NodeCategory::Training, "Smooth L1 Loss", ICON_FA_CHART_LINE,
        {"smooth l1", "smoothl1", "huber", "regression", "robust",
         "criterion", "objective", "optimization", "loss"},
        0, false, "Smooth L1 loss for robust regression", "", "",
        {{"Predictions", PinType::Tensor, true, "Numeric model predictions"},
         {"Targets", PinType::Labels, true, "Same-shaped numeric target values"}},
        {{"Loss", PinType::Loss, true, "Loss value"}},
        {{"reduction", "enum", "mean", "Reduction", {"mean", "sum", "none"}, ""},
         {"beta", "float", "1.0", "Quadratic-to-linear transition width", {}, ""}},
        NodeImplementationStatus::Implemented, 0});

    RegisterNode({NodeType::HuberLoss, NodeCategory::Training, "Huber Loss", ICON_FA_CHART_LINE,
        {"huber", "smooth l1", "smoothl1", "regression", "robust",
         "criterion", "objective", "optimization", "loss"},
        0, false, "Huber loss implemented by the equivalent Smooth L1 objective", "", "",
        {{"Predictions", PinType::Tensor, true, "Numeric model predictions"},
         {"Targets", PinType::Labels, true, "Same-shaped numeric target values"}},
        {{"Loss", PinType::Loss, true, "Loss value"}},
        {{"reduction", "enum", "mean", "Reduction", {"mean", "sum", "none"}, ""},
         {"beta", "float", "1.0", "Huber delta / Smooth L1 transition width", {}, ""}},
        NodeImplementationStatus::Implemented, 0});

    RegisterNode({NodeType::NLLLoss, NodeCategory::Training, "NLL Loss", ICON_FA_CHART_PIE,
        {"nll", "negative log likelihood", "log likelihood", "classification",
         "multiclass", "criterion", "objective", "optimization", "loss"},
        0, false, "Negative log-likelihood loss for log-probability inputs", "", "",
        {{"Log Probabilities", PinType::Tensor, true, "Log-probability predictions"},
         {"Labels", PinType::Labels, true, "Class labels"}},
        {{"Loss", PinType::Loss, true, "Loss value"}},
        {{"reduction", "enum", "mean", "Reduction", {"mean", "sum", "none"}, ""},
         {"ignore_index", "int", "-100", "Padding label ignore index", {}, ""}},
        NodeImplementationStatus::Implemented, 0});

    RegisterNode({NodeType::Adam, NodeCategory::Training, "Adam", ICON_FA_GRADUATION_CAP,
        {"adam", "optimizer", "optimization", "training"}, 0, false,
        "Adaptive moment estimation optimizer", "", "",
        {{"Loss", PinType::Loss, true, "Scalar training loss used for backpropagation"}},
        {{"State", PinType::Optimizer, false, "Optional optimizer-state handle"}},
        {{"learning_rate", "float", "0.001", "Positive parameter-update step size", {}, ""},
         {"beta1", "float", "0.9", "First-moment decay coefficient in [0, 1)", {}, ""},
         {"beta2", "float", "0.999", "Second-moment decay coefficient in [0, 1)", {}, ""},
         {"epsilon", "float", "1e-8", "Positive denominator stability constant", {}, ""}},
        NodeImplementationStatus::Implemented, 0});

    RegisterNode({NodeType::SGD, NodeCategory::Training, "SGD", ICON_FA_GRADUATION_CAP,
        {"sgd", "stochastic gradient descent", "optimizer", "optimization", "training"},
        0, false, "Stochastic gradient descent with optional momentum", "", "",
        {{"Loss", PinType::Loss, true, "Scalar training loss used for backpropagation"}},
        {{"State", PinType::Optimizer, false, "Optional optimizer-state handle"}},
        {{"learning_rate", "float", "0.01", "Positive parameter-update step size", {}, ""},
         {"momentum", "float", "0.9", "Momentum coefficient in [0, 1)", {}, ""}},
        NodeImplementationStatus::Implemented, 0});

    RegisterNode({NodeType::AdamW, NodeCategory::Training, "AdamW", ICON_FA_GRADUATION_CAP,
        {"adamw", "adam", "weight decay", "optimizer", "optimization", "training"},
        0, false, "Adam optimizer with decoupled weight decay", "", "",
        {{"Loss", PinType::Loss, true, "Scalar training loss used for backpropagation"}},
        {{"State", PinType::Optimizer, false, "Optional optimizer-state handle"}},
        {{"learning_rate", "float", "0.001", "Positive parameter-update step size", {}, ""},
         {"beta1", "float", "0.9", "First-moment decay coefficient in [0, 1)", {}, ""},
         {"beta2", "float", "0.999", "Second-moment decay coefficient in [0, 1)", {}, ""},
         {"epsilon", "float", "1e-8", "Positive denominator stability constant", {}, ""},
         {"weight_decay", "float", "0.01", "Non-negative decoupled weight-decay coefficient", {}, ""}},
        NodeImplementationStatus::Implemented, 0});

    RegisterNode({NodeType::RMSprop, NodeCategory::Training, "RMSprop", ICON_FA_GRADUATION_CAP,
        {"rmsprop", "optimizer", "adaptive", "optimization", "training"},
        0, false, "Adaptive optimizer using a moving average of squared gradients", "", "",
        {{"Loss", PinType::Loss, true, "Scalar training loss used for backpropagation"}},
        {{"State", PinType::Optimizer, false, "Optional optimizer-state handle"}},
        {{"learning_rate", "float", "0.001", "Positive parameter-update step size", {}, ""},
         {"alpha", "float", "0.99", "Squared-gradient moving-average coefficient in [0, 1)", {}, ""},
         {"epsilon", "float", "1e-8", "Positive denominator stability constant", {}, ""},
         {"momentum", "float", "0.0", "Momentum coefficient in [0, 1)", {}, ""}},
        NodeImplementationStatus::Implemented, 0});

    RegisterNode({NodeType::Adagrad, NodeCategory::Training, "Adagrad", ICON_FA_GRADUATION_CAP,
        {"adagrad", "ada grad", "optimizer", "adaptive", "optimization", "training"},
        0, false, "Per-parameter adaptive learning rates from accumulated squared gradients", "", "",
        {{"Loss", PinType::Loss, true, "Scalar training loss used for backpropagation"}},
        {{"State", PinType::Optimizer, false, "Optional optimizer-state handle"}},
        {{"learning_rate", "float", "0.01", "Positive initial parameter-update step size", {}, ""},
         {"epsilon", "float", "1e-10", "Positive denominator stability constant", {}, ""}},
        NodeImplementationStatus::Implemented, 0});

    RegisterNode({NodeType::NAdam, NodeCategory::Training, "NAdam", ICON_FA_GRADUATION_CAP,
        {"nadam", "nesterov adam", "adam", "optimizer", "optimization", "training"},
        0, false, "Nesterov-accelerated Adam optimizer", "", "",
        {{"Loss", PinType::Loss, true, "Scalar training loss used for backpropagation"}},
        {{"State", PinType::Optimizer, false, "Optional optimizer-state handle"}},
        {{"learning_rate", "float", "0.002", "Positive parameter-update step size", {}, ""},
         {"beta1", "float", "0.9", "First-moment decay coefficient in [0, 1)", {}, ""},
         {"beta2", "float", "0.999", "Second-moment decay coefficient in [0, 1)", {}, ""},
         {"epsilon", "float", "1e-8", "Positive denominator stability constant", {}, ""}},
        NodeImplementationStatus::Implemented, 0});

    RegisterNode({NodeType::StepLR, NodeCategory::Training, "Step LR", ICON_FA_GRADUATION_CAP,
        {"step", "scheduler"}, 0, false,
        "Blocked legacy step learning-rate scheduler preview",
        "Saved-graph compatibility contract only. A backend scheduler primitive exists, but the Engine graph and training lifecycle do not construct, step, restore, or checkpoint it.", "",
        {{"Optimizer", PinType::Optimizer, true, "Legacy optimizer-state input"}},
        {{"Scheduled", PinType::Optimizer, true, "Reserved scheduled-optimizer output"}},
        {{"step_size", "int", "10", "Legacy epoch interval retained for saved graphs", {}, "", "Step size", "Compatibility"},
         {"gamma", "float", "0.1", "Legacy multiplicative decay retained for saved graphs", {}, "", "Gamma", "Compatibility"}},
        NodeImplementationStatus::Template, 0, "Blocked"});

    RegisterNode({NodeType::CosineAnnealing, NodeCategory::Training, "Cosine LR", ICON_FA_WAVE_SINE,
        {"cosine", "scheduler"}, 0, false,
        "Blocked legacy cosine-annealing scheduler preview",
        "Saved-graph compatibility contract only. A backend scheduler primitive exists, but the Engine graph and training lifecycle do not construct, step, restore, or checkpoint it.", "",
        {{"Optimizer", PinType::Optimizer, true, "Legacy optimizer-state input"}},
        {{"Scheduled", PinType::Optimizer, true, "Reserved scheduled-optimizer output"}},
        {{"T_max", "int", "100", "Legacy annealing period retained for saved graphs", {}, "", "T max", "Compatibility"},
         {"eta_min", "float", "0.0", "Legacy minimum learning rate retained for saved graphs", {}, "", "Minimum learning rate", "Compatibility"}},
        NodeImplementationStatus::Template, 0, "Blocked"});

    RegisterNode({NodeType::ReduceOnPlateau, NodeCategory::Training, "Reduce LR", ICON_FA_GRADUATION_CAP,
        {"plateau", "scheduler"}, 0, false,
        "Blocked legacy reduce-on-plateau scheduler preview",
        "Saved-graph compatibility contract only. A backend scheduler primitive exists, but the Engine graph and training lifecycle do not construct, step, restore, or checkpoint it.", "",
        {{"Optimizer", PinType::Optimizer, true, "Legacy optimizer-state input"}},
        {{"Scheduled", PinType::Optimizer, true, "Reserved scheduled-optimizer output"}},
        {{"mode", "enum", "min", "Legacy monitored-metric direction retained for saved graphs", {"min", "max"}, "", "Mode", "Compatibility"},
         {"factor", "float", "0.1", "Legacy multiplicative reduction retained for saved graphs", {}, "", "Factor", "Compatibility"},
         {"patience", "int", "10", "Legacy plateau patience retained for saved graphs", {}, "", "Patience", "Compatibility"}},
        NodeImplementationStatus::Template, 0, "Blocked"});

    RegisterNode({NodeType::ExponentialLR, NodeCategory::Training, "Exponential LR", ICON_FA_GRADUATION_CAP,
        {"exponential", "scheduler"}, 0, false,
        "Blocked legacy exponential learning-rate scheduler preview",
        "Saved-graph compatibility contract only. A backend scheduler primitive exists, but the Engine graph and training lifecycle do not construct, step, restore, or checkpoint it.", "",
        {{"Optimizer", PinType::Optimizer, true, "Legacy optimizer-state input"}},
        {{"Scheduled", PinType::Optimizer, true, "Reserved scheduled-optimizer output"}},
        {{"gamma", "float", "0.95", "Legacy multiplicative decay retained for saved graphs", {}, "", "Gamma", "Compatibility"}},
        NodeImplementationStatus::Template, 0, "Blocked"});

    RegisterNode({NodeType::WarmupScheduler, NodeCategory::Training, "Warmup LR", ICON_FA_GRADUATION_CAP,
        {"warmup", "scheduler"}, 0, false,
        "Blocked legacy warmup learning-rate scheduler preview",
        "Saved-graph compatibility contract only. Backend warmup primitives exist, but the Engine graph and training lifecycle do not construct, step, restore, or checkpoint one for this node.", "",
        {{"Optimizer", PinType::Optimizer, true, "Legacy optimizer-state input"}},
        {{"Scheduled", PinType::Optimizer, true, "Reserved scheduled-optimizer output"}},
        {{"warmup_steps", "int", "1000", "Legacy warmup duration retained for saved graphs", {}, "", "Warmup steps", "Compatibility"},
         {"warmup_ratio", "float", "0.1", "Legacy starting-rate ratio retained for saved graphs", {}, "", "Warmup ratio", "Compatibility"}},
        NodeImplementationStatus::Template, 0, "Blocked"});

    RegisterNode({NodeType::L1Regularization, NodeCategory::Regularization, "L1 Regularization", ICON_FA_GRADUATION_CAP,
        {"l1", "regularization"}, 0, false,
        "Blocked legacy L1 penalty preview",
        "Saved-graph compatibility contract only. No Engine owner reads model parameters, computes a differentiable L1 penalty, and adds it to the selected training loss.", "",
        {{"Parameters", PinType::Parameters, true, "Legacy model-parameters input"}},
        {{"Penalty", PinType::Loss, true, "Reserved scalar penalty output"}},
        {{"lambda", "float", "0.01", "Legacy penalty coefficient retained for saved graphs", {}, "", "Lambda", "Compatibility"}},
        NodeImplementationStatus::Template, 0, "Blocked"});

    RegisterNode({NodeType::L2Regularization, NodeCategory::Regularization, "L2 Regularization", ICON_FA_GRADUATION_CAP,
        {"l2", "regularization"}, 0, false,
        "Blocked legacy L2 penalty preview",
        "Saved-graph compatibility contract only. No Engine owner reads model parameters, computes a differentiable L2 penalty, and adds it to the selected training loss. AdamW weight decay is a separate optimizer behavior.", "",
        {{"Parameters", PinType::Parameters, true, "Legacy model-parameters input"}},
        {{"Penalty", PinType::Loss, true, "Reserved scalar penalty output"}},
        {{"lambda", "float", "0.01", "Legacy penalty coefficient retained for saved graphs", {}, "", "Lambda", "Compatibility"}},
        NodeImplementationStatus::Template, 0, "Blocked"});

    RegisterNode({NodeType::ElasticNet, NodeCategory::Regularization, "Elastic Net", ICON_FA_GRADUATION_CAP,
        {"elasticnet", "regularization"}, 0, false,
        "Blocked legacy Elastic Net penalty preview",
        "Saved-graph compatibility contract only. No Engine owner reads model parameters, combines differentiable L1/L2 penalties, and adds the result to the selected training loss.", "",
        {{"Parameters", PinType::Parameters, true, "Legacy model-parameters input"}},
        {{"Penalty", PinType::Loss, true, "Reserved scalar penalty output"}},
        {{"lambda", "float", "0.01", "Legacy overall penalty coefficient retained for saved graphs", {}, "", "Lambda", "Compatibility"},
         {"l1_ratio", "float", "0.5", "Legacy L1 share retained for saved graphs", {}, "", "L1 ratio", "Compatibility"}},
        NodeImplementationStatus::Template, 0, "Blocked"});

    RegisterNode({NodeType::Output, NodeCategory::Training, "Output", ICON_FA_ARROW_RIGHT,
        {"output", "final", "predictions", "classes"}, 0, false,
        "Declare the model or training workflow output",
        "Acts as an identity marker: place it after the optimizer as the "
        "terminal training node, or between the final model tensor and a loss "
        "when an explicit prediction relay is useful. It does not add a layer. "
        "The class count validates classification output width; the preceding "
        "model layer still owns the actual projection.",
        "",
        {{"Input", PinType::Tensor, true,
          "Final model tensor or terminal optimizer state."}},
        {{"Predictions", PinType::Tensor, false,
          "Optional identity relay for the final model tensor."}},
        {{"num_classes", "int", "10",
          "Expected classification output width", {}, "1-1048576",
          "Classes", "Output", false, false}},
        NodeImplementationStatus::Implemented, 0});

    RegisterNode({NodeType::PairDatasetBuilder, NodeCategory::Training, "Pair Dataset Builder", ICON_FA_CODE_BRANCH,
        {"metric", "learning", "pair", "siamese"}, 0, false,
        "Blocked metric-learning pair-batch contract",
        "Preserves pair-column and label-convention settings for saved graphs. A production owner must materialize those columns as device-ready paired tensors before this node can execute.", "",
        {{"Rows", PinType::Dataset, true, "Source rows with sample A/B columns and labels"}},
        {{"Pair Batch", PinType::Dataset, true, "Typed PairBatch payload"}},
        {{"sample_a_column", "string", "", "First sample column", {}, ""},
         {"sample_b_column", "string", "", "Second sample column", {}, ""},
         {"pair_label_column", "string", "", "Pair label column", {}, ""},
         {"label_convention", "enum", "contrastive_zero_similar", "Label convention",
          {"contrastive_zero_similar", "cosine_one_similar"}, ""}},
        NodeImplementationStatus::Template, 0, "Blocked"});

    RegisterNode({NodeType::TripletDatasetBuilder, NodeCategory::Training, "Triplet Dataset Builder", ICON_FA_CODE_BRANCH,
        {"metric", "learning", "triplet", "siamese"}, 0, false,
        "Blocked metric-learning triplet-batch contract",
        "Preserves anchor, positive, and negative column settings for saved graphs. A production owner must materialize those columns as device-ready triplet tensors before this node can execute.", "",
        {{"Rows", PinType::Dataset, true, "Source rows with triplet sample columns"}},
        {{"Triplet Batch", PinType::Dataset, true, "Typed TripletBatch payload"}},
        {{"anchor_column", "string", "", "Anchor sample column", {}, ""},
         {"positive_column", "string", "", "Positive sample column", {}, ""},
         {"negative_column", "string", "", "Negative sample column", {}, ""}},
        NodeImplementationStatus::Template, 0, "Blocked"});

    RegisterNode({NodeType::SharedEncoder, NodeCategory::Training, "Shared Encoder", ICON_FA_SHARE_NODES,
        {"metric", "learning", "shared", "encoder", "siamese"}, 0, false,
        "Blocked metric-learning shared-encoder contract",
        "Declares one encoder identity for saved metric-learning graphs. Visual graph ownership, stateful branch snapshots, and device-resident gradient accumulation are not implemented.", "",
        {{"Encoder", PinType::Tensor, true, "Encoder layer chain"}},
        {{"Shared Encoder", PinType::Parameters, true, "Shared encoder reference"}},
        {{"encoder_id", "string", "shared_encoder", "Shared encoder id", {}, ""}},
        NodeImplementationStatus::Template, 0, "Blocked"});

    RegisterNode({NodeType::SiameseBranch, NodeCategory::Training, "Siamese Branch", ICON_FA_CODE_BRANCH,
        {"metric", "learning", "branch", "siamese"}, 0, false,
        "Blocked metric-learning encoder-branch contract",
        "Preserves a branch role and shared-encoder reference for saved graphs. The visual runtime does not yet route branch tensors through shared parameters.", "",
        {{"Input", PinType::Tensor, true, "Branch input"},
         {"Shared Encoder", PinType::Parameters, true, "Shared encoder reference"}},
        {{"Embedding", PinType::Tensor, true, "Branch embedding"}},
        {{"branch", "enum", "a", "Branch role", {"a", "b", "anchor", "positive", "negative"}, ""}},
        NodeImplementationStatus::Template, 0, "Blocked"});

    RegisterNode({NodeType::ContrastiveLoss, NodeCategory::Training, "Contrastive Loss", ICON_FA_SCALE_BALANCED,
        {"metric", "learning", "contrastive", "loss"}, 0, false,
        "Blocked metric-learning contrastive-loss contract",
        "Uses the saved-graph convention 0=similar and 1=dissimilar. A backend primitive exists, but the visual training path does not route paired embeddings and labels through it.", "",
        {{"Embedding A", PinType::Tensor, true, "First embedding"},
         {"Embedding B", PinType::Tensor, true, "Second embedding"},
         {"Labels", PinType::Labels, true, "Pair labels"}},
        {{"Loss", PinType::Loss, true, "Contrastive loss"}},
        {{"margin", "float", "1.0", "Distance margin", {}, ""}},
        NodeImplementationStatus::Template, 0, "Blocked"});

    RegisterNode({NodeType::CosineEmbeddingLoss, NodeCategory::Training, "Cosine Embedding Loss", ICON_FA_SCALE_BALANCED,
        {"metric", "learning", "cosine", "loss"}, 0, false,
        "Blocked metric-learning cosine-loss contract",
        "Uses the saved-graph convention 1=similar and -1=dissimilar. A backend primitive exists, but the visual training path does not route paired embeddings and labels through it.", "",
        {{"Embedding A", PinType::Tensor, true, "First embedding"},
         {"Embedding B", PinType::Tensor, true, "Second embedding"},
         {"Labels", PinType::Labels, true, "Pair labels"}},
        {{"Loss", PinType::Loss, true, "Cosine embedding loss"}},
        {{"margin", "float", "0.0", "Cosine margin", {}, ""}},
        NodeImplementationStatus::Template, 0, "Blocked"});

    RegisterNode({NodeType::TripletLoss, NodeCategory::Training, "Triplet Loss", ICON_FA_SCALE_BALANCED,
        {"metric", "learning", "triplet", "loss"}, 0, false,
        "Blocked metric-learning triplet-loss contract",
        "Preserves the margin for anchor, positive, and negative embeddings. A backend primitive exists, but the visual training path does not own the triplet/shared-encoder update.", "",
        {{"Anchor", PinType::Tensor, true, "Anchor embedding"},
         {"Positive", PinType::Tensor, true, "Positive embedding"},
         {"Negative", PinType::Tensor, true, "Negative embedding"}},
        {{"Loss", PinType::Loss, true, "Triplet loss"}},
        {{"margin", "float", "1.0", "Triplet margin", {}, ""}},
        NodeImplementationStatus::Template, 0, "Blocked"});

    RegisterNode({NodeType::PairMetrics, NodeCategory::Training, "Pair Metrics", ICON_FA_CHART_LINE,
        {"metric", "learning", "pair", "metrics"}, 0, false,
        "Blocked metric-learning pair-metrics contract",
        "Preserves the distance threshold for saved graphs. The visual runtime does not yet compute and report pair metrics from routed embedding batches.", "",
        {{"Embedding A", PinType::Tensor, true, "First embedding"},
         {"Embedding B", PinType::Tensor, true, "Second embedding"},
         {"Labels", PinType::Labels, true, "Pair labels"}},
        {{"Metrics", PinType::Dataset, true, "Pair metric rows"}},
        {{"threshold", "float", "0.5", "Distance threshold", {}, ""}},
        NodeImplementationStatus::Template, 0, "Blocked"});

    RegisterNode({NodeType::RetrievalMetrics, NodeCategory::Training, "Retrieval Metrics", ICON_FA_CHART_LINE,
        {"metric", "learning", "retrieval", "metrics"}, 0, false,
        "Blocked metric-learning retrieval-metrics contract",
        "Preserves the retrieval cutoff for recall@k, MRR, and nearest-neighbor agreement. The visual runtime does not yet own metric computation and reporting.", "",
        {{"Embeddings", PinType::Tensor, true, "Embedding matrix"},
         {"Class IDs", PinType::Labels, true, "Class ids"}},
        {{"Metrics", PinType::Dataset, true, "Retrieval metric rows"}},
        {{"k", "int", "10", "Retrieval cutoff", {}, ""}},
        NodeImplementationStatus::Template, 0, "Blocked"});

    RegisterNode({NodeType::EmbeddingOutput, NodeCategory::Training, "Embedding Output", ICON_FA_CUBE,
        {"metric", "learning", "embedding", "output"}, 0, false,
        "Blocked metric-learning embedding-output contract",
        "Preserves embedding-output metadata settings for saved graphs. Response packaging exists for inference endpoints, but visual graph/runtime routing is not implemented.", "",
        {{"Embeddings", PinType::Tensor, true, "Embedding matrix"}},
        {{"Embedding Records", PinType::Dataset, true, "Embedding output records"}},
        {{"include_metadata", "bool", "true", "Include sample/class metadata", {}, ""}},
        NodeImplementationStatus::Template, 0, "Blocked"});

    RegisterNode({NodeType::PairScoreOutput, NodeCategory::Training, "Pair Score Output", ICON_FA_CHART_LINE,
        {"metric", "learning", "pair", "score", "output"}, 0, false,
        "Blocked metric-learning pair-score output contract",
        "Preserves distance or similarity scoring mode for saved graphs. Response packaging exists for inference endpoints, but visual graph/runtime routing is not implemented.", "",
        {{"Embedding A", PinType::Tensor, true, "First embedding"},
         {"Embedding B", PinType::Tensor, true, "Second embedding"}},
        {{"Pair Scores", PinType::Dataset, true, "Pair score records"}},
        {{"score_mode", "enum", "distance", "Score mode",
          {"distance", "negative_distance", "cosine_similarity"}, ""}},
        NodeImplementationStatus::Template, 0, "Blocked"});
}

// =============================================================================
// DNN Nodes
// =============================================================================
void NodeMetadataRegistry::InitializeDNNNodes() {
    RegisterNode({NodeType::DNNModelLoad, NodeCategory::DNN, "Model Loader", ICON_FA_DOWNLOAD,
        {"dnn", "model", "load"}, 0, false, "Load pre-trained model", "", "",
        {}, {{"Model", PinType::Parameters, true, "Model"}},
        {{"model_path", "file", "", "Model file", {}, "*.onnx"}},
        NodeImplementationStatus::Template, 0, "Blocked"});

    RegisterNode({NodeType::DNNDetect, NodeCategory::DNN, "Object Detector", ICON_FA_CROSSHAIRS,
        {"detect", "yolo", "object"}, 0, false, "Object detection", "", "",
        {{"Image", PinType::Tensor, true, "Image"}, {"Model", PinType::Parameters, true, "Model"}},
        {{"Detections", PinType::Dataset, true, "Detections"}},
        {{"confidence", "float", "0.5", "Threshold", {}, ""}},
        NodeImplementationStatus::Template, 0, "Blocked"});

    RegisterNode({NodeType::PretrainedYOLO, NodeCategory::DNN, "YOLOv4", ICON_FA_CROSSHAIRS,
        {"yolo", "detection"}, 0, false, "YOLOv4 detector", "", "",
        {{"Image", PinType::Tensor, true, "Image"}},
        {{"Detections", PinType::Dataset, true, "Detections"}},
        {{"confidence", "float", "0.5", "Threshold", {}, ""}},
        NodeImplementationStatus::Template, 0, "Blocked"});
}

// =============================================================================
// Text Nodes
// =============================================================================
void NodeMetadataRegistry::InitializeTextNodes() {
    RegisterNode({NodeType::TextCleanNode, NodeCategory::TextProcessing, "Text Clean", ICON_FA_ERASER,
        {"clean", "text", "normalize", "lowercase", "html"}, 0, false,
        "Clean one text column with lowercase, HTML removal, and special-character normalization", "", "",
        {{"Text", PinType::Dataset, true, "Input text table"}},
        {{"Cleaned", PinType::Dataset, true, "Input table plus cleaned text column"}},
        {{"text_column", "string", "", "Text column", {}, ""},
         {"lowercase", "bool", "true", "Lowercase text", {}, ""},
         {"remove_html", "bool", "true", "Remove HTML tags", {}, ""},
         {"remove_special_chars", "bool", "true", "Normalize special characters", {}, ""}},
        NodeImplementationStatus::Implemented, 0});

    RegisterNode(WithPropertiesEditor({NodeType::TextTokenizer, NodeCategory::TextProcessing, "Tokenizer", ICON_FA_ALIGN_LEFT,
        {"tokenize", "text", "nlp", "vocabulary", "padding"}, 0, false,
        "Tokenize text, build/load vocabulary, and pad/truncate sequences", "", "",
        {{"Text", PinType::Dataset, true, "Text"}},
        {{"Tokens", PinType::Tensor, true, "Token indices"}},
        {{"tokenizer_type", "enum", "1", "Mode", {"0", "1", "2"}, ""},
         {"text_col", "string", "", "Text column", {}, ""},
         {"label_col", "string", "", "Label column", {}, ""},
         {"max_length", "int", "256", "Max sequence length", {}, ""},
         {"lowercase", "bool", "true", "Convert text to lowercase", {}, ""},
         {"padding", "bool", "true", "Pad short sequences", {}, ""},
         {"truncation", "bool", "true", "Truncate long sequences", {}, ""},
         {"min_word_freq", "int", "2", "Minimum token frequency", {}, ""},
         {"max_vocab_size", "int", "10000", "Vocabulary cap", {}, ""},
         {"vocab_file", "string", "", "Vocabulary file", {}, ""},
         {"pad_value", "int", "0", "Padding token id", {}, ""},
         {"vocab_build_if_missing", "bool", "false", "Build and save a missing vocabulary", {}, ""},
         {"source_csv", "file", "", "Optional corpus used by the vocabulary builder", {}, "*.csv", "", "", false, false,
          ParameterConsumption::UiOnly}},
        NodeImplementationStatus::Implemented, 0}, NodePropertiesEditor::Dialog));

    RegisterNode(WithPropertiesEditor({NodeType::TextVocabulary, NodeCategory::TextProcessing, "Vocabulary (legacy)", ICON_FA_LIST_UL,
        {"vocabulary", "vocab", "legacy"}, 0, false,
        "Legacy folded config node; use Tokenizer vocabulary settings for new graphs", "", "",
        {{"Text", PinType::Dataset, true, "Text"}},
        {{"Vocab", PinType::Parameters, true, "Vocabulary"}},
        {{"max_words", "int", "10000", "Max words", {}, ""}},
        NodeImplementationStatus::Deprecated, 0, "Folded into TextTokenizer"}, NodePropertiesEditor::Dialog));

    RegisterNode(WithPropertiesEditor({NodeType::TextPadding, NodeCategory::TextProcessing, "Padding (legacy)", ICON_FA_ARROWS_LEFT_RIGHT,
        {"pad", "sequence", "legacy"}, 0, false,
        "Legacy folded config node; use Tokenizer padding settings for new graphs", "", "",
        {{"Tokens", PinType::Tensor, true, "Tokens"}},
        {{"Padded", PinType::Tensor, true, "Padded"}},
        {{"max_length", "int", "128", "Max length", {}, ""}},
        NodeImplementationStatus::Deprecated, 0, "Folded into TextTokenizer"}, NodePropertiesEditor::Dialog));

    RegisterNode(WithPropertiesEditor({NodeType::NERSequenceBuilder, NodeCategory::TextProcessing, "NER Sequence Builder", ICON_FA_TAG,
        {"ner", "sequence", "builder", "token", "tagging"}, 0, false,
        "Build token-level sequence samples for NER", "", "",
        {{"Rows", PinType::Dataset, true, "Already-tokenized token/POS/tag rows"}},
        {{"Sequence Samples", PinType::Tensor, true, "word_ids, optional pos_ids, attention_mask, tag_ids"}},
        {{"token_column", "string", "tokens", "Token column", {}, ""},
         {"pos_column", "string", "", "Optional POS column", {}, ""},
         {"tag_column", "string", "ner_tags", "NER tag column", {}, ""},
         {"sentence_id_column", "string", "", "Optional sentence id column", {}, ""},
         {"max_sequence_length", "int", "0", "Max sequence length (0 = infer)", {}, ""},
         {"ignore_index", "int", "-100", "Padding label ignore index", {}, ""},
        {"create_attention_mask", "bool", "true", "Create attention mask", {}, ""}},
        NodeImplementationStatus::Implemented, 0}, NodePropertiesEditor::Custom));

    RegisterNode(WithPropertiesEditor({NodeType::TokenVocabulary, NodeCategory::TextProcessing, "Token Vocabulary", ICON_FA_BOOK,
        {"token", "vocabulary", "ner", "sequence"}, 0, false,
        "Build deterministic token id vocabulary for sequence tagging", "", "",
        {{"Tokens", PinType::Dataset, true, "Token sequences"}},
        {{"Token Vocabulary", PinType::Parameters, true, "value,id vocabulary table"}},
        {{"min_freq", "int", "1", "Minimum frequency", {}, ""},
         {"max_vocab_size", "int", "0", "Max vocab size (0 = unlimited)", {}, ""},
         {"lowercase", "bool", "true", "Lowercase tokens", {}, ""},
         {"pad_token", "string", "[PAD]", "Padding token", {}, ""},
         {"unk_token", "string", "[UNK]", "Unknown token", {}, ""}},
        NodeImplementationStatus::Implemented, 0}, NodePropertiesEditor::Custom));

    RegisterNode(WithPropertiesEditor({NodeType::POSVocabulary, NodeCategory::TextProcessing, "POS Vocabulary", ICON_FA_BOOK,
        {"pos", "part-of-speech", "vocabulary", "ner"}, 0, false,
        "Build deterministic POS id vocabulary for sequence tagging", "", "",
        {{"POS Tags", PinType::Dataset, true, "POS tag sequences"}},
        {{"POS Vocabulary", PinType::Parameters, true, "value,id vocabulary table"}},
        {{"min_freq", "int", "1", "Minimum frequency", {}, ""},
         {"max_vocab_size", "int", "0", "Max vocab size (0 = unlimited)", {}, ""},
         {"lowercase", "bool", "false", "Lowercase POS tags", {}, ""},
         {"pad_token", "string", "[PAD]", "Padding token", {}, ""},
         {"unk_token", "string", "[UNK]", "Unknown token", {}, ""}},
        NodeImplementationStatus::Implemented, 0}, NodePropertiesEditor::Custom));

    RegisterNode(WithPropertiesEditor({NodeType::NERTagVocabulary, NodeCategory::TextProcessing, "NER Tag Vocabulary", ICON_FA_TAG,
        {"ner", "tag", "bio", "vocabulary"}, 0, false,
        "Build deterministic BIO tag vocabulary for sequence tagging", "", "",
        {{"NER Tags", PinType::Dataset, true, "NER tag sequences"}},
        {{"NER Tag Vocabulary", PinType::Parameters, true, "value,id BIO tag vocabulary table"}},
        {{"outside_tag", "string", "O", "Outside tag label", {}, ""},
         {"bio_scheme", "enum", "BIO", "Tag scheme", {"BIO"}, ""}},
        NodeImplementationStatus::Implemented, 0}, NodePropertiesEditor::Custom));

    RegisterNode(WithPropertiesEditor({NodeType::SequenceTagOutput, NodeCategory::TextProcessing, "Sequence Tag Output", ICON_FA_TAG,
        {"sequence", "tag", "output", "ner", "decode"}, 0, false,
        "Declare token-level sequence tagger output and decode metadata", "", "",
        {{"Token Logits", PinType::Tensor, true, "Per-token logits [batch, seq_len, num_tags]"}},
        {{"Predictions", PinType::Tensor, false, "Token-level prediction tensor"}},
        {{"num_tags", "int", "0", "Number of BIO tags (0 = infer)", {}, ""},
         {"tag_vocab_file", "string", "", "Tag vocabulary file", {}, ""},
         {"decode_scheme", "enum", "BIO", "Decode scheme", {"BIO"}, ""}},
        NodeImplementationStatus::Implemented, 0}, NodePropertiesEditor::Custom));
}

// =============================================================================
// Time Series Nodes
// =============================================================================
void NodeMetadataRegistry::InitializeTimeSeriesNodes() {
    RegisterNode({NodeType::TimeSeriesSegment, NodeCategory::TimeSeries,
        "Time Integrity & Segments", ICON_FA_CLOCK,
        {"timestamp", "integrity", "gap", "segment"}, 0, false,
        "Validate ordered timestamps and assign continuous segment IDs", "", "",
        {{"Data", PinType::Dataset, true, "Input time-ordered table"}},
        {{"Segmented", PinType::Dataset, true,
          "Input table plus segment and time-delta metadata"}},
        {{"timestamp_col", "string", "", "Timestamp column", {}, ""},
         {"gap_threshold_seconds", "float", "30",
          "Start a new segment at deltas greater than or equal to this value",
          {}, ""},
         {"segment_col", "string", "__segment_id",
          "Output continuous-segment column", {}, ""},
         {"delta_col", "string", "__time_delta_seconds",
          "Output seconds-since-previous-row column", {}, ""}},
        NodeImplementationStatus::Implemented, 0});

    RegisterNode({NodeType::TimeSeriesWindow, NodeCategory::TimeSeries, "Sliding Window", ICON_FA_CHART_LINE,
        {"window", "sliding", "sequence"}, 0, false, "Create sliding windows", "", "",
        {{"Data", PinType::Dataset, true, "Input time-ordered table"}},
        {{"Windowed", PinType::Dataset, true, "Windowed table with x_* features, ordered y targets, and hidden target bounds"}},
        {{"value_col", "string", "", "Numeric source/target column", {}, ""},
         {"feature_cols", "string", "", "Extra numeric feature columns to window", {}, ""},
         {"time_col", "string", "", "Optional numeric time column", {}, ""},
         {"segment_col", "string", "", "Optional int64 segment column; windows cannot cross segment boundaries", {}, ""},
         {"input_width", "int", "12", "Lookback steps per sample", {}, ""},
         {"label_width", "int", "1", "Ordered forecast target steps", {}, ""},
         {"shift", "int", "1", "Forecast offset", {}, ""}},
        NodeImplementationStatus::Implemented, 0});

    RegisterNode({NodeType::TimeSeriesFeatures, NodeCategory::TimeSeries, "TS Features", ICON_FA_SLIDERS,
        {"features", "lag", "rolling"}, 0, false, "Extract TS features", "", "",
        {{"Data", PinType::Dataset, true, "Input time-ordered table"}},
        {{"Enriched", PinType::Dataset, true, "Input table plus lag and rolling feature columns"}},
        {{"value_col", "string", "", "Numeric column to featurize", {}, ""},
         {"lag_values", "string", "", "Lag values", {}, ""},
         {"rolling_windows", "string", "", "Rolling window sizes", {}, ""},
         {"rolling_aggregations", "string", "mean", "Rolling aggregations", {}, ""}},
        NodeImplementationStatus::Implemented, 0});

    RegisterNode({NodeType::TimeSeriesLag, NodeCategory::TimeSeries, "TS Lag", ICON_FA_CHART_LINE,
        {"lag", "lagged", "time", "series"}, 0, false, "Add lag columns for one numeric time-series column", "", "",
        {{"Data", PinType::Dataset, true, "Input time-ordered table"}},
        {{"Lagged", PinType::Dataset, true, "Input table plus lag feature columns"}},
        {{"columns", "string", "", "Numeric column to lag", {}, ""},
         {"lag_periods", "string", "1", "Comma-separated lag periods", {}, ""}},
        NodeImplementationStatus::Implemented, 0});

    RegisterNode({NodeType::TimeSeriesSplit, NodeCategory::TimeSeries, "TS Split", ICON_FA_SCISSORS,
        {"split", "train", "test"}, 0, false, "Chronological split", "", "",
        {{"Data", PinType::Dataset, true, "Input time-ordered table"}},
        {{"Partitioned", PinType::Dataset, true, "Input table plus __partition__ split column"}},
        {{"train_ratio", "float", "0.8", "Train ratio", {}, ""},
         {"val_ratio", "float", "0.1", "Validation ratio", {}, ""},
         {"test_ratio", "float", "0.1", "Test ratio", {}, ""},
         {"boundary_policy", "enum", "targets_within_partition", "Window boundary policy", {"targets_within_partition", "window_rows"}, ""},
         {"train_end_source_row", "int", "-1", "Exclusive Train source-row boundary (-1 uses ratios)", {}, ""},
         {"val_end_source_row", "int", "-1", "Exclusive Validation source-row boundary (-1 uses ratios)", {}, ""}},
        NodeImplementationStatus::Implemented, 0});

    RegisterNode({NodeType::SeasonalNaive, NodeCategory::TimeSeries, "Seasonal Naive", ICON_FA_REPEAT,
        {"seasonal", "naive", "baseline", "forecast", "rolling"}, 0, false,
        "Repeat the latest seasonal cycle as a deterministic forecast baseline", "", "",
        {{"Windowed", PinType::Dataset, true, "Sliding Window output with x_* history and ordered y targets"}},
        {{"Predictions", PinType::Dataset, true, "Long-form actual/prediction rows for filtering and Regression Metrics"}},
        {{"seasonal_period", "int", "1", "Observations in one seasonal cycle", {}, ""}},
        NodeImplementationStatus::Implemented, 0});

    RegisterNode({NodeType::LogTransform, NodeCategory::TimeSeries, "Log Transform", ICON_FA_CHART_LINE,
        {"log", "log1p", "stabilize", "time", "series"}, 0, false,
        "Apply log1p to a numeric time-series column", "", "",
        {{"Data", PinType::Dataset, true, "Input table"}},
        {{"Transformed", PinType::Dataset, true, "Input table with transformed column"}},
        {{"value_col", "string", "", "Numeric column to transform", {}, ""}},
        NodeImplementationStatus::Implemented, 0});

    RegisterNode({NodeType::Differencing, NodeCategory::TimeSeries, "Differencing", ICON_FA_MINUS,
        {"difference", "differencing", "lag", "stationary", "time", "series"}, 0, false,
        "Difference a numeric column by lag and order", "", "",
        {{"Data", PinType::Dataset, true, "Input table"}},
        {{"Differenced", PinType::Dataset, true, "Shortened table after differencing"}},
        {{"value_col", "string", "", "Numeric column to difference", {}, ""},
         {"lag", "int", "1", "Lag", {}, ""},
         {"order", "int", "1", "Differencing order", {}, ""}},
        NodeImplementationStatus::Implemented, 0});
}

// =============================================================================
// Audio Nodes
// =============================================================================
void NodeMetadataRegistry::InitializeAudioNodes() {
    RegisterNode({NodeType::AudioInput, NodeCategory::Audio, "Audio Input", ICON_FA_WAVE_SQUARE,
        {"audio", "waveform", "dataset", "source"}, 0, false,
        "Expose waveform and label batches from the loaded project audio dataset", "", "",
        {}, {{"Waveform", PinType::Tensor, true, "Batched audio waveform tensor"},
             {"Labels", PinType::Labels, false, "Optional batched class labels"}},
        {},
        NodeImplementationStatus::Implemented, 0});

    RegisterNode({NodeType::Spectrogram, NodeCategory::Audio, "Spectrogram", ICON_FA_CHART_AREA,
        {"spectrogram", "stft"}, 0, false, "Compute spectrogram", "", "",
        {{"Waveform", PinType::Tensor, true, "Waveform"}},
        {{"Spectrogram", PinType::Tensor, true, "Spectrogram"}},
        {{"n_fft", "int", "512", "FFT window size", {}, ">0"},
         {"hop_length", "int", "256", "Samples advanced between frames", {}, ">0"},
         {"log_scale", "bool", "true", "Convert magnitudes to logarithmic scale", {}, ""}},
        NodeImplementationStatus::Implemented, 0});

    RegisterNode({NodeType::MelSpectrogram, NodeCategory::Audio, "Mel Spectrogram", ICON_FA_CHART_AREA,
        {"mel", "spectrogram"}, 0, false, "Mel spectrogram", "", "",
        {{"Waveform", PinType::Tensor, true, "Waveform"}},
        {{"MelSpec", PinType::Tensor, true, "Mel spec"}},
        {{"n_fft", "int", "512", "FFT window size", {}, ">0"},
         {"hop_length", "int", "256", "Samples advanced between frames", {}, ">0"},
         {"n_mels", "int", "128", "Mel bands", {}, ">0"},
         {"fmin", "float", "0", "Minimum analysis frequency", {}, ">=0"},
         {"fmax", "float", "0", "Maximum frequency, or 0 for Nyquist", {}, ">=0"},
         {"log_scale", "bool", "true", "Convert magnitudes to logarithmic scale", {}, ""}},
        NodeImplementationStatus::Implemented, 0});

    RegisterNode({NodeType::MFCC, NodeCategory::Audio, "MFCC", ICON_FA_WAVE_SQUARE,
        {"mfcc", "cepstral"}, 0, false, "Extract MFCCs", "", "",
        {{"Waveform", PinType::Tensor, true, "Waveform"}},
        {{"MFCC", PinType::Tensor, true, "MFCCs"}},
        {{"n_mfcc", "int", "13", "Number of cepstral coefficients", {}, ">0"},
         {"n_fft", "int", "512", "FFT window size", {}, ">0"},
         {"hop_length", "int", "256", "Samples advanced between frames", {}, ">0"},
         {"n_mels", "int", "128", "Mel bands used before the DCT", {}, ">0"}},
        NodeImplementationStatus::Implemented, 0});
}

// =============================================================================
// RL Nodes
// =============================================================================
void NodeMetadataRegistry::InitializeRLNodes() {
    RegisterNode({NodeType::GymEnvironment, NodeCategory::RL, "Gym Environment", ICON_FA_ROCKET,
        {"gym", "environment", "rl"}, 0, false, "OpenAI Gym connector", "", "",
        {}, {{"Observation", PinType::Tensor, true, "Obs"}, {"Info", PinType::Parameters, true, "Info"}},
        {{"env_name", "string", "CartPole-v1", "Environment", {}, ""}},
        NodeImplementationStatus::Implemented, 0});

    RegisterNode({NodeType::ReplayBufferNode, NodeCategory::RL, "Replay Buffer", ICON_FA_DATABASE,
        {"replay", "buffer", "experience"}, 0, false, "Experience replay", "", "",
        {{"State", PinType::Tensor, true, "State"}, {"Action", PinType::Tensor, true, "Action"},
         {"Reward", PinType::Tensor, true, "Reward"}, {"NextState", PinType::Tensor, true, "Next"}},
        {{"Batch", PinType::Dataset, true, "Batch"}},
        {{"capacity", "int", "100000", "Capacity", {}, ""}},
        NodeImplementationStatus::Implemented, 0});

    RegisterNode({NodeType::PolicyNetwork, NodeCategory::RL, "Policy Network", ICON_FA_BRAIN,
        {"policy", "actor"}, 0, false, "Actor/Policy network", "", "",
        {{"Observation", PinType::Tensor, true, "Obs"}},
        {{"Action", PinType::Tensor, true, "Action"}},
        {{"hidden_sizes", "string", "256,256", "Hidden sizes", {}, ""}},
        NodeImplementationStatus::Implemented, 0});

    RegisterNode({NodeType::ValueNetwork, NodeCategory::RL, "Value Network", ICON_FA_CHART_LINE,
        {"value", "critic"}, 0, false, "Critic/Value network", "", "",
        {{"Observation", PinType::Tensor, true, "Obs"}},
        {{"Value", PinType::Tensor, true, "Value"}},
        {{"hidden_sizes", "string", "256,256", "Hidden sizes", {}, ""}},
        NodeImplementationStatus::Implemented, 0});
}

// =============================================================================
// Export Nodes
// =============================================================================
void NodeMetadataRegistry::InitializeExportNodes() {
    RegisterNode({NodeType::ExportCSV, NodeCategory::DataSources, "Export CSV", ICON_FA_FILE_EXPORT,
        {"export", "csv", "save"}, 0, false, "Export Arrow table to CSV file", "", "",
        {{"Table", PinType::Dataset, true, "Table"}}, {},
        {{"file_path", "file", "", "Output file", {}, "*.csv"}},
        NodeImplementationStatus::Implemented, 0});

    RegisterNode({NodeType::ExportParquet, NodeCategory::DataSources, "Export Parquet", ICON_FA_FILE_EXPORT,
        {"export", "parquet"}, 0, false, "Export Arrow table to Parquet file", "", "",
        {{"Table", PinType::Dataset, true, "Table"}}, {},
        {{"file_path", "file", "", "Output file", {}, "*.parquet"}},
        NodeImplementationStatus::Implemented, 0});

    RegisterNode({NodeType::ExportJSON, NodeCategory::DataSources, "Export JSON", ICON_FA_FILE_EXPORT,
        {"export", "json"}, 0, false, "Export Arrow table to JSON file", "", "",
        {{"Table", PinType::Dataset, true, "Table"}}, {},
        {{"file_path", "file", "", "Output file", {}, "*.json"}},
        NodeImplementationStatus::Implemented, 0});

    RegisterNode({NodeType::ExportSQL, NodeCategory::DataSources, "Export SQL (planned)", ICON_FA_DATABASE,
        {"export", "sql", "database"}, 0, false, "SQL database export is not implemented in PipelineExecutor", "", "",
        {{"Table", PinType::Dataset, true, "Table"}}, {},
        {{"connection", "string", "", "Connection string", {}, ""}},
        NodeImplementationStatus::Template, 0, "Blocked"});

    RegisterNode({NodeType::ExportExcel, NodeCategory::DataSources, "Export Excel (planned)", ICON_FA_FILE_EXCEL,
        {"export", "excel", "xlsx", "spreadsheet"}, 0, false, "Excel export is not implemented in PipelineExecutor", "", "",
        {{"Table", PinType::Dataset, true, "Table"}}, {},
        {{"file_path", "file", "", "Output file", {}, "*.xlsx"},
         {"sheet_name", "string", "Sheet1", "Sheet name", {}, ""}},
        NodeImplementationStatus::Template, 0, "Blocked"});
}

// =============================================================================
// KNIME-Style Table Manipulation Nodes
// =============================================================================
void NodeMetadataRegistry::InitializeKNIMENodes() {
    RegisterNode({NodeType::RowToColumnNames, NodeCategory::DataTransform, "Row to Column Names", ICON_FA_TH,
        {"header", "column names", "promote row"}, 0, false,
        "Promote a row to column headers", "", "",
        {{"Table", PinType::Dataset, true, "Table"}},
        {{"Table", PinType::Dataset, true, "Table with new headers"}},
        {{"row_index", "int", "0", "Row index to use as headers", {}, ""}},
        NodeImplementationStatus::Implemented, 0});

    RegisterNode({NodeType::TableSplitter, NodeCategory::DataTransform, "Table Splitter", ICON_FA_SCISSORS,
        {"split", "divide", "partition"}, 0, false,
        "Split table at specified row", "", "",
        {{"Table", PinType::Dataset, true, "Table"}},
        {{"Top", PinType::Dataset, true, "Rows above split"},
         {"Bottom", PinType::Dataset, true, "Rows below split"}},
        {{"split_row", "int", "0", "Row index to split at", {}, ""}},
        NodeImplementationStatus::Template, 0, "Blocked"});

    RegisterNode({NodeType::CellExtractor, NodeCategory::DataTransform, "Cell Extractor", ICON_FA_CROSSHAIRS,
        {"cell", "extract", "value"}, 0, false,
        "Extract value from specific cell", "", "",
        {{"Table", PinType::Dataset, true, "Table"}},
        {{"Value", PinType::Tensor, true, "Extracted value"},
         {"Table", PinType::Dataset, true, "Table passthrough"}},
        {{"row", "int", "0", "Row index", {}, ""},
         {"column", "string", "", "Column name", {}, ""}},
        NodeImplementationStatus::Implemented, 0});

    RegisterNode({NodeType::CellUpdater, NodeCategory::DataTransform, "Cell Updater", ICON_FA_PEN,
        {"cell", "update", "modify"}, 0, false,
        "Update value in specific cell", "", "",
        {{"Table", PinType::Dataset, true, "Table"},
         {"Value", PinType::Tensor, false, "New value (optional)"}},
        {{"Table", PinType::Dataset, true, "Updated table"}},
        {{"row", "int", "0", "Row index", {}, ""},
         {"column", "string", "", "Column name", {}, ""},
         {"value", "string", "", "New value (if not using input)", {}, ""}},
        NodeImplementationStatus::Implemented, 0});

    RegisterNode({NodeType::TableCropper, NodeCategory::DataTransform, "Table Cropper", ICON_FA_COMPRESS,
        {"crop", "slice", "subset"}, 0, false,
        "Crop table to specified dimensions", "", "",
        {{"Table", PinType::Dataset, true, "Table"}},
        {{"Table", PinType::Dataset, true, "Cropped table"}},
        {{"start_row", "int", "0", "Start row (inclusive)", {}, ""},
         {"end_row", "int", "-1", "End row (-1 for last)", {}, ""},
         {"columns", "string", "", "Columns to keep (comma-separated, empty=all)", {}, ""}},
        NodeImplementationStatus::Implemented, 0});

    RegisterNode({NodeType::ColumnAppender, NodeCategory::DataTransform, "Column Appender", ICON_FA_TABLE_COLUMNS,
        {"append", "columns", "horizontal"}, 0, false,
        "Append columns from multiple tables", "", "",
        {{"Left", PinType::Dataset, true, "Left table"},
         {"Right", PinType::Dataset, true, "Right table"}},
        {{"Table", PinType::Dataset, true, "Combined table"}},
        {{"suffix", "string", "_right", "Suffix for duplicate columns", {}, ""}},
        NodeImplementationStatus::Implemented, 0});

    RegisterNode({NodeType::RowAppender, NodeCategory::DataTransform, "Row Appender", ICON_FA_LAYER_GROUP,
        {"append", "rows", "vertical", "union"}, 0, false,
        "Append rows from multiple tables (UNION)", "", "",
        {{"Top", PinType::Dataset, true, "Top table"},
         {"Bottom", PinType::Dataset, true, "Bottom table"}},
        {{"Table", PinType::Dataset, true, "Combined table"}},
        {{"match_columns", "bool", "true", "Match columns by name", {}, ""}},
        NodeImplementationStatus::Implemented, 0});

    RegisterNode({NodeType::Unpivot, NodeCategory::DataTransform, "Unpivot", ICON_FA_ROTATE,
        {"unpivot", "melt", "wide to long"}, 0, false,
        "Unpivot wide to long format", "", "",
        {{"Table", PinType::Dataset, true, "Wide table"}},
        {{"Table", PinType::Dataset, true, "Long table"}},
        {{"id_columns", "string", "", "ID columns (comma-separated)", {}, ""},
         {"value_name", "string", "value", "Name for value column", {}, ""},
         {"variable_name", "string", "variable", "Name for variable column", {}, ""}},
        NodeImplementationStatus::Implemented, 0});

    RegisterNode({NodeType::StringManipulation, NodeCategory::DataTransform, "String Manipulation", ICON_FA_FONT,
        {"string", "text", "replace", "trim"}, 0, false,
        "String operations on columns", "", "",
        {{"Table", PinType::Dataset, true, "Table"}},
        {{"Table", PinType::Dataset, true, "Table with modified strings"}},
        {{"column", "string", "", "Column to manipulate", {}, ""},
         {"operation", "enum", "trim", "Operation", {"trim", "upper", "lower", "replace", "substring"}, ""},
         {"param1", "string", "", "Parameter 1 (find/start)", {}, ""},
         {"param2", "string", "", "Parameter 2 (replace/length)", {}, ""}},
        NodeImplementationStatus::Implemented, 0});

    RegisterNode({NodeType::MathFormula, NodeCategory::DataTransform, "Math Formula", ICON_FA_CALCULATOR,
        {"math", "formula", "calculate"}, 0, false,
        "Apply math formula to create/modify columns", "", "",
        {{"Table", PinType::Dataset, true, "Table"}},
        {{"Table", PinType::Dataset, true, "Table with new column"}},
        {{"output_column", "string", "result", "Output column name", {}, ""},
         {"formula", "string", "", "Formula (e.g., col1 + col2 * 2)", {}, ""}},
        NodeImplementationStatus::Implemented, 0});

    RegisterNode({NodeType::RuleEngine, NodeCategory::DataTransform, "Rule Engine", ICON_FA_SCALE_BALANCED,
        {"rule", "if", "condition", "case"}, 0, false,
        "Apply if-then-else rules to create/modify columns", "", "",
        {{"Table", PinType::Dataset, true, "Table"}},
        {{"Table", PinType::Dataset, true, "Table with rule result"}},
        {{"output_column", "string", "result", "Output column name", {}, ""},
         {"rules", "string", "", "Rules (one per line: condition => value)", {}, ""},
         {"default_value", "string", "NULL", "Default value if no rule matches", {}, ""}},
        NodeImplementationStatus::Implemented, 0});
}

// =============================================================================
// Utility Nodes
// =============================================================================
void NodeMetadataRegistry::InitializeUtilityNodes() {
    RegisterNode({NodeType::Lambda, NodeCategory::Utility, "Lambda", ICON_FA_CODE,
        {"lambda", "custom", "function"}, 0, false,
        "Blocked custom tensor function retained for saved-graph compatibility",
        "No expression evaluator, model layer, PipelineExecutor operator, or "
        "sandboxed execution owner is registered for this node.", "",
        {{"Input", PinType::Tensor, true, "Input tensor"}},
        {{"Output", PinType::Tensor, true, "Uncomputed tensor output"}},
        {{"function", "string", "lambda x: x",
          "Historical function text; not executed while the node is blocked", {}, ""}},
        NodeImplementationStatus::Template, 0});

    RegisterNode({NodeType::Identity, NodeCategory::Utility, "Identity", ICON_FA_EQUALS,
        {"identity", "passthrough", "table"}, 0, false,
        "Pass an Arrow table through unchanged",
        "PipelineOperatorFactory owns this Data Studio table operation. The output "
        "retains the input table's rows, columns, schema, and values.", "",
        {{"Table", PinType::Dataset, true, "Input Arrow table"}},
        {{"Table", PinType::Dataset, true, "The same Arrow table, unchanged"}},
        {}, NodeImplementationStatus::Implemented, 0});

    RegisterNode({NodeType::Constant, NodeCategory::Utility, "Constant", ICON_FA_HASHTAG,
        {"constant", "value", "simulation", "scalar"}, 0, false,
        "Emit a fixed scalar value on every simulation tick",
        "GraphExecutor owns this bounded scalar source. It is not a trainable "
        "parameter or a PipelineExecutor Tensor source.", "",
        {}, {{"Value", PinType::Tensor, true, "Scalar simulation value"}},
        {{"value", "float", "1.0", "Finite scalar output", {}, "",
          "Value", "Signal"}},
        NodeImplementationStatus::Implemented, 0});

    RegisterNode({NodeType::Parameter, NodeCategory::Utility, "Parameter", ICON_FA_CIRCLE_DOT,
        {"parameter", "constant", "trainable"}, 0, false,
        "Blocked standalone trainable parameter retained for saved-graph compatibility",
        "No model-registration, initialization, gradient, optimizer, checkpoint, or "
        "graph execution owner is registered for a standalone Parameter node.", "",
        {}, {{"Parameter", PinType::Tensor, true, "Unregistered parameter tensor"}},
        {{"shape", "string", "256", "Historical parameter shape", {}, ""},
         {"init", "enum", "xavier", "Historical initialization policy",
          {"xavier", "zeros", "ones", "normal", "uniform"}, ""},
         {"requires_grad", "bool", "true", "Historical gradient intent", {}, ""}},
        NodeImplementationStatus::Template, 0});

    RegisterNode(WithPropertiesEditor({NodeType::SignalSlider, NodeCategory::Signal, "Signal Slider", ICON_FA_SLIDERS,
        {"slider", "control", "input", "simulation", "scalar"}, 0, false,
        "Interactively emit one scalar value during simulation",
        "Properties publishes live value changes to GraphExecutor. The value "
        "must remain inside the configured finite range.", "",
        {}, {{"Value", PinType::Tensor, true, "Scalar simulation value"}},
        {{"value", "float", "0.0", "Current scalar output", {}, "",
          "Value", "Signal"},
         {"min", "float", "-1.0", "Minimum selectable value", {}, "",
          "Minimum", "Range"},
         {"max", "float", "1.0", "Maximum selectable value", {}, "",
          "Maximum", "Range"}},
        NodeImplementationStatus::Implemented, 0}, NodePropertiesEditor::Custom));

    RegisterNode(WithPropertiesEditor({NodeType::SineWave, NodeCategory::Signal, "Sine Wave", ICON_FA_WAVE_SQUARE,
        {"signal", "sine", "wave", "simulation"}, 0, false,
        "Emit amplitude * sin(2*pi*frequency*time + phase) + offset",
        "GraphExecutor evaluates this finite scalar source using simulation "
        "time in seconds and phase in radians.", "",
        {}, {{"Signal", PinType::Tensor, true, "Scalar sine-wave sample"}},
        {{"amplitude", "float", "1.0", "Wave amplitude", {}, "", "Amplitude", "Wave"},
         {"frequency", "float", "1.0", "Frequency in hertz", {}, "", "Frequency (Hz)", "Wave"},
         {"phase", "float", "0.0", "Phase in radians", {}, "", "Phase (rad)", "Wave"},
         {"offset", "float", "0.0", "Constant output offset", {}, "", "Offset", "Wave"}},
        NodeImplementationStatus::Implemented, 0}, NodePropertiesEditor::Custom));

    RegisterNode(WithPropertiesEditor({NodeType::StepSignal, NodeCategory::Signal, "Step Signal", ICON_FA_ARROW_RIGHT,
        {"signal", "step", "simulation"}, 0, false,
        "Switch from an initial scalar to a final scalar at the step time",
        "GraphExecutor evaluates the initial value before step_time seconds and "
        "the final value at and after that time.", "",
        {}, {{"Signal", PinType::Tensor, true, "Scalar step-signal sample"}},
        {{"step_time", "float", "1.0", "Transition time in seconds", {}, ">= 0", "Step time (s)", "Step"},
         {"initial_value", "float", "0.0", "Value before the transition", {}, "", "Initial value", "Step"},
         {"final_value", "float", "1.0", "Value at and after the transition", {}, "", "Final value", "Step"}},
        NodeImplementationStatus::Implemented, 0}, NodePropertiesEditor::Custom));

    RegisterNode(WithPropertiesEditor({NodeType::RampSignal, NodeCategory::Signal, "Ramp Signal", ICON_FA_ARROW_TREND_UP,
        {"signal", "ramp", "simulation"}, 0, false,
        "Linearly interpolate between two scalar values over a duration",
        "GraphExecutor starts at start_value when simulation time is zero, "
        "reaches end_value at duration seconds, and then holds it.", "",
        {}, {{"Signal", PinType::Tensor, true, "Scalar ramp-signal sample"}},
        {{"start_value", "float", "0.0", "Value at time zero", {}, "", "Start value", "Ramp"},
         {"end_value", "float", "1.0", "Value at the end of the ramp", {}, "", "End value", "Ramp"},
         {"duration", "float", "5.0", "Positive ramp duration in seconds", {}, "> 0", "Duration (s)", "Ramp"}},
        NodeImplementationStatus::Implemented, 0}, NodePropertiesEditor::Custom));

    RegisterNode(WithPropertiesEditor({NodeType::SignalScope, NodeCategory::Signal, "Signal Scope", ICON_FA_CHART_LINE,
        {"scope", "plot", "monitor", "simulation", "scalar"}, 0, false,
        "Plot one connected scalar signal from the live simulation",
        "Properties reads the connected input value and simulation timestamp "
        "from GraphExecutor. It does not synthesize preview data.", "",
        {{"Signal", PinType::Tensor, true, "Connected scalar simulation signal"}}, {},
        {{"window_size", "int", "500", "Maximum retained samples", {}, "10-100000", "Window size", "Display", false, false,
          ParameterConsumption::UiOnly},
         {"auto_scale", "bool", "true", "Automatically fit the value axis", {}, "", "Auto scale", "Display", false, false,
          ParameterConsumption::UiOnly}},
        NodeImplementationStatus::Implemented, 0}, NodePropertiesEditor::Custom));

    // ===== Signal Processing Nodes (Phase 4) =====
    RegisterNode({NodeType::FFTNode, NodeCategory::Signal, "FFT", ICON_FA_WAVE_SQUARE,
        {"fft", "fourier", "frequency"}, 0, false, "Fast Fourier Transform",
        "Computes a one-sided frequency spectrum from one numeric table column.", "",
        {{"Data", PinType::Dataset, true, "Input table containing the signal column"}},
        {{"Spectrum", PinType::Dataset, true, "Frequency, magnitude, and phase columns"}},
        {{"signal_col", "string", "", "Numeric signal column", {}, "", "", "", true, false},
         {"sample_rate", "float", "1.0", "Samples per second used for the frequency axis", {}, "> 0"}},
        NodeImplementationStatus::Implemented, 0});

    RegisterNode({NodeType::IFFTNode, NodeCategory::Signal, "IFFT", ICON_FA_WAVE_SQUARE,
        {"ifft", "inverse", "fourier"}, 0, false, "Inverse FFT",
        "Convert frequency domain back to time domain.", "",
        {{"Spectrum", PinType::Tensor, true, "Frequency spectrum"}},
        {{"Signal", PinType::Tensor, true, "Time-domain signal"}},
        {}, NodeImplementationStatus::Template, 0, "Blocked"});

    RegisterNode({NodeType::FilterDesigner, NodeCategory::Signal, "Filter Designer", ICON_FA_FILTER,
        {"filter", "fir", "iir", "lowpass"}, 0, false, "Design and apply a digital filter",
        "Designs and applies a filter to one numeric signal column.", "",
        {{"Data", PinType::Dataset, true, "Input table containing the signal column"}},
        {{"Filtered", PinType::Dataset, true, "Input table with the selected column replaced by Float32 filtered values"}},
        {{"signal_col", "string", "", "Numeric signal column", {}, "", "", "", true, false},
         {"filter_type", "dropdown", "lowpass", "Filter type", {"lowpass", "highpass", "bandpass", "bandstop"}, ""},
         {"cutoff", "float", "0.5", "Primary cutoff frequency", {}, "> 0"},
         {"cutoff_high", "float", "0", "Upper cutoff required by bandpass and bandstop", {}, "Greater than cutoff for band filters"},
         {"sample_rate", "float", "1.0", "Samples per second", {}, "> 0"},
         {"order", "int", "4", "Filter order", {}, ">= 1"}},
        NodeImplementationStatus::Implemented, 0});

    RegisterNode({NodeType::WaveletTransform, NodeCategory::Signal, "Wavelet Transform", ICON_FA_WATER,
        {"wavelet", "dwt", "cwt"}, 0, false, "Discrete Wavelet Transform",
        "Multi-resolution analysis using wavelets.", "",
        {{"Signal", PinType::Tensor, true, "Input signal"}},
        {{"Coefficients", PinType::Tensor, true, "Wavelet coefficients"}, {"Approximation", PinType::Tensor, true, "Approximation"}, {"Detail", PinType::Tensor, true, "Detail coefficients"}},
        {{"wavelet", "enum", "db4", "Wavelet family", {"db4", "haar", "sym4", "coif2"}, ""},
         {"level", "int", "3", "Decomposition level", {}, ""}},
        NodeImplementationStatus::Template, 0, "Blocked"});

    // ===== Text Analytics Nodes (Phase 4) =====
    RegisterNode({NodeType::TFIDFVectorizer, NodeCategory::TextProcessing, "TF-IDF", ICON_FA_ALIGN_LEFT,
        {"tfidf", "vectorizer", "text"}, 0, false, "TF-IDF text vectorization",
        "Convert a text column into dense TF-IDF feature columns. Use this for classic NLP baselines where term frequency and inverse document frequency should both matter. Dense memory cost scales as rows x max_features x float32, before temporary vocabulary and Arrow builder overhead.", "",
        {{"Text", PinType::Dataset, true, "Text data (column of strings)"}},
        {{"Vectors", PinType::Dataset, true, "Dense indexed tfidf_* feature columns with optional y label column; fitted state preserves the term-to-column mapping"}},
        {{"text_col", "string", "", "Text column", {}, "Required string/large_string column to vectorize.", "", "", true, false},
         {"label_col", "string", "", "Label column", {}, "Optional label column copied to output as y."},
         {"max_features", "int", "2000", "Max vocabulary size", {}, "Dense output width cap. Memory scales as rows x max_features x float32 before temporary materializer overhead."},
         {"min_df", "int", "1", "Min document frequency", {}, "Keep terms appearing in at least this many documents."},
         {"use_idf", "bool", "true", "Use IDF", {}, "Apply inverse-document-frequency weighting."},
         {"smooth_idf", "bool", "true", "Smooth IDF", {}, "Use smoothed IDF to avoid zero-division and soften rare-term weights."},
         {"norm", "dropdown", "l2", "Row normalization", {"l1", "l2", "none"}, "Normalize each output feature row."},
         {"ngram_range", "dropdown", "1,1", "N-gram range", {"1,1", "1,2", "1,3", "2,2", "2,3", "3,3"}, "Canonical minimum,maximum n-gram sizes. Larger ranges increase vocabulary planning and dense output cost."},
         {"stop_words", "dropdown", "english", "Stop words", {"english", "none"}, "Use none for sentiment tasks where negation words such as 'not' matter."},
         {"output_format", "dropdown", "dense", "Output format", {"dense"}, "Current engine support is dense Arrow feature columns. Sparse output is planned but not executable yet."},
         {"operation_mode", "enum", "fit_transform", "Fit a vocabulary and IDF weights on this input or transform with a saved training artifact", {"fit_transform", "transform_only"}, "", "Mode", "Fitted vectorizer state"},
         {"save_state", "bool", "false", "Persist the ordered vocabulary, feature mapping, and IDF weights for validation, test, or inference", {}, "", "Save fitted state", "Fitted vectorizer state"},
         {"state_path", "file", "", "Artifact path to save during Fit + Transform or load during Transform Only", {}, "*.cyxstate.json", "State artifact path", "Fitted vectorizer state"},
         {"state_overwrite", "bool", "false", "Allow replacing an existing fitted-state artifact", {}, "", "Allow state overwrite", "Fitted vectorizer state", false, true}},
        NodeImplementationStatus::Implemented, 0});

    RegisterNode({NodeType::CountVectorizer, NodeCategory::TextProcessing, "Count Vectorizer", ICON_FA_LIST_OL,
        {"count", "bow", "bag", "words"}, 0, false, "Bag-of-words vectorization",
        "Convert a text column into dense count or binary-presence feature columns. Use this when raw term frequency or term presence is preferable to TF-IDF weighting. Dense memory cost scales as rows x max_features x float32, before temporary vocabulary and Arrow builder overhead.", "",
        {{"Text", PinType::Dataset, true, "Text data"}},
        {{"Vectors", PinType::Dataset, true, "Dense indexed count_* feature columns with optional y label column; fitted state preserves the term-to-column mapping"}},
        {{"text_col", "string", "", "Text column", {}, "Required string/large_string column to vectorize.", "", "", true, false},
         {"label_col", "string", "", "Label column", {}, "Optional label column copied to output as y."},
         {"max_features", "int", "2000", "Max vocabulary size", {}, "Dense output width cap. Memory scales as rows x max_features x float32 before temporary materializer overhead."},
         {"norm", "dropdown", "l2", "Row normalization", {"l1", "l2", "none"}, "Normalize each output feature row."},
         {"ngram_range", "dropdown", "1,1", "N-gram range", {"1,1", "1,2", "1,3", "2,2", "2,3", "3,3"}, "Canonical minimum,maximum n-gram sizes. Larger ranges increase vocabulary planning and dense output cost."},
         {"stop_words", "dropdown", "english", "Stop words", {"english", "none"}, "Use none for sentiment tasks where negation words such as 'not' matter."},
         {"binary", "bool", "false", "Binary counts", {}, "When true, output term presence instead of term frequency before optional row normalization."},
         {"output_format", "dropdown", "dense", "Output format", {"dense"}, "Current engine support is dense Arrow feature columns. Sparse output is planned but not executable yet."},
         {"operation_mode", "enum", "fit_transform", "Fit an ordered vocabulary on this input or transform with a saved training artifact", {"fit_transform", "transform_only"}, "", "Mode", "Fitted vectorizer state"},
         {"save_state", "bool", "false", "Persist the ordered vocabulary and feature mapping for validation, test, or inference", {}, "", "Save fitted state", "Fitted vectorizer state"},
         {"state_path", "file", "", "Artifact path to save during Fit + Transform or load during Transform Only", {}, "*.cyxstate.json", "State artifact path", "Fitted vectorizer state"},
         {"state_overwrite", "bool", "false", "Allow replacing an existing fitted-state artifact", {}, "", "Allow state overwrite", "Fitted vectorizer state", false, true}},
        NodeImplementationStatus::Implemented, 0});

    RegisterNode({NodeType::SentimentAnalyzer, NodeCategory::TextProcessing, "Sentiment Analysis", ICON_FA_FACE_SMILE,
        {"sentiment", "opinion", "polarity"}, 0, false, "Sentiment analysis",
        "Compute lexicon-based sentiment for each text row without loading an external model.", "",
        {{"Text", PinType::Dataset, true, "Input table containing text"}},
        {{"Sentiment", PinType::Dataset, true, "Polarity, subjectivity, sentiment_label, confidence, and optional y columns"}},
        {{"text_col", "string", "", "Text column", {}, "", "", "", true, false},
         {"label_col", "string", "", "Optional label column copied to output as y", {}, ""},
         {"method", "enum", "vader", "Sentiment method", {"simple", "vader", "afinn"}, ""}},
        NodeImplementationStatus::Implemented, 0});

    // ===== Utility Tools (Phase 4) =====
    RegisterNode({NodeType::CalculatorNode, NodeCategory::Utility, "Calculator", ICON_FA_CALCULATOR,
        {"calculator", "math", "compute"}, 0, false, "Math expression calculator",
        "Evaluate mathematical expressions with variables.", "",
        {{"Variables", PinType::Dataset, false, "Input variables (optional)"}},
        {{"Result", PinType::Dataset, true, "Computed result"}},
        {{"expression", "string", "2 + 2", "Math expression", {}, ""},
         {"precision", "int", "6", "Decimal precision", {}, ""}},
        NodeImplementationStatus::Implemented, 0});

    RegisterNode({NodeType::UnitConverter, NodeCategory::Utility, "Unit Converter", ICON_FA_SCALE_BALANCED,
        {"unit", "convert", "conversion"}, 0, false, "Unit conversion utility",
        "Convert between different units of measurement.", "",
        {{"Value", PinType::Dataset, true, "Input value"}},
        {{"Converted", PinType::Dataset, true, "Converted value"}},
        {{"category", "enum", "length", "Unit category", {"length", "mass", "temperature", "time", "area", "volume"}, ""},
         {"from_unit", "string", "m", "From unit", {}, ""},
         {"to_unit", "string", "ft", "To unit", {}, ""}},
        NodeImplementationStatus::Implemented, 0});

    RegisterNode({NodeType::RegexTester, NodeCategory::Utility, "Regex Tester", ICON_FA_CODE,
        {"regex", "regular", "expression", "pattern"}, 0, false, "Regular expression tester",
        "Test and apply regex patterns to text.", "",
        {{"Text", PinType::Dataset, true, "Input text"}},
        {{"Matches", PinType::Dataset, true, "Match results"}, {"Groups", PinType::Dataset, true, "Capture groups"}},
        {{"pattern", "string", ".*", "Regex pattern", {}, ""},
         {"flags", "string", "", "Flags (i=ignorecase, m=multiline)", {}, ""}},
        NodeImplementationStatus::Implemented, 0});

    RegisterNode({NodeType::JSONPathExtractor, NodeCategory::Utility, "JSONPath", ICON_FA_CODE_BRANCH,
        {"json", "jsonpath", "extract"}, 0, false, "Extract data using JSONPath",
        "Query JSON data using JSONPath expressions.", "",
        {{"JSON", PinType::Dataset, true, "JSON data"}},
        {{"Result", PinType::Dataset, true, "Extracted values"}},
        {{"path", "string", "$.data[*].value", "JSONPath expression", {}, ""}},
        NodeImplementationStatus::Implemented, 0});
}

// =============================================================================
// Linear Algebra Nodes (Tool-to-Node Migration)
// =============================================================================
void NodeMetadataRegistry::InitializeLinearAlgebraNodes() {
    RegisterNode({NodeType::SVDNode, NodeCategory::Analytics, "SVD", ICON_FA_TABLE_CELLS,
        {"svd", "singular", "value", "decomposition", "matrix"}, 0, false,
        "Singular Value Decomposition", "", "",
        {{"Matrix", PinType::Dataset, true, "Matrix"}},
        {{"U", PinType::Dataset, false, "U"}, {"S", PinType::Dataset, false, "S"}, {"V", PinType::Dataset, false, "V"}},
        {},
        NodeImplementationStatus::Template, 0});

    RegisterNode({NodeType::QRDecomposition, NodeCategory::Analytics, "QR Decomposition", ICON_FA_TABLE_COLUMNS,
        {"qr", "decomposition", "matrix", "orthogonal"}, 0, false,
        "QR Matrix Decomposition", "", "",
        {{"Matrix", PinType::Dataset, true, "Matrix"}},
        {{"Q", PinType::Dataset, false, "Q"}, {"R", PinType::Dataset, false, "R"}},
        {},
        NodeImplementationStatus::Template, 0});

    RegisterNode({NodeType::CholeskyDecomposition, NodeCategory::Analytics, "Cholesky", ICON_FA_SQUARE_ROOT_VARIABLE,
        {"cholesky", "decomposition", "matrix", "positive", "definite"}, 0, false,
        "Cholesky Decomposition", "", "",
        {{"Matrix", PinType::Dataset, true, "Matrix"}},
        {{"L", PinType::Dataset, false, "L"}},
        {},
        NodeImplementationStatus::Template, 0});

    RegisterNode({NodeType::EigenDecomposition, NodeCategory::Analytics, "Eigenvalue", ICON_FA_CHART_LINE,
        {"eigen", "eigenvalue", "eigenvector", "decomposition"}, 0, false,
        "Eigenvalue Decomposition", "", "",
        {{"Matrix", PinType::Dataset, true, "Matrix"}},
        {{"Eigenvalues", PinType::Dataset, false, "Eigenvalues"}, {"Eigenvectors", PinType::Dataset, false, "Eigenvectors"}},
        {},
        NodeImplementationStatus::Template, 0});

    RegisterNode({NodeType::MatrixCalculator, NodeCategory::Analytics, "Matrix Calculator", ICON_FA_CALCULATOR,
        {"matrix", "calculator", "operations", "multiply", "transpose", "inverse"}, 0, false,
        "Matrix Operations", "", "",
        {{"A", PinType::Dataset, true, "A"}, {"B", PinType::Dataset, false, "B"}},
        {{"Result", PinType::Dataset, false, "Result"}},
        {{"operation", "dropdown", "multiply", "Operation", {"multiply", "add", "subtract", "transpose", "inverse", "determinant"}, ""}},
        NodeImplementationStatus::Template, 0});
}

// =============================================================================
// Time Series Analysis Nodes (Tool-to-Node Migration)
// =============================================================================
void NodeMetadataRegistry::InitializeTimeSeriesAnalysisNodes() {
    RegisterNode({NodeType::TimeSeriesDecomposition, NodeCategory::TimeSeries, "Decomposition", ICON_FA_LAYER_GROUP,
        {"decomposition", "trend", "seasonal", "residual", "stl"}, 0, false,
        "Append trend, seasonal, and residual columns to a time-series table", "", "",
        {{"Data", PinType::Dataset, true, "Input table"}},
        {{"Decomposed", PinType::Dataset, true, "Input table plus trend/seasonal/residual"}},
        {{"signal_col", "string", "", "Numeric signal column", {}, ""},
         {"period", "int", "12", "Seasonal period", {}, ""},
         {"method", "dropdown", "additive", "Composition method", {"additive", "multiplicative"}, ""},
         {"algorithm", "dropdown", "classical", "Algorithm", {"classical", "stl"}, ""}},
        NodeImplementationStatus::Implemented, 0});

    RegisterNode({NodeType::ACFNode, NodeCategory::TimeSeries, "ACF", ICON_FA_CHART_BAR,
        {"acf", "autocorrelation", "correlogram", "time", "series"}, 0, false,
        "Compute autocorrelation by lag as a table", "", "",
        {{"Data", PinType::Dataset, true, "Input table"}},
        {{"ACF", PinType::Dataset, true, "Lag table with acf and confidence bounds"}},
        {{"signal_col", "string", "", "Numeric signal column", {}, ""},
         {"max_lag", "int", "-1", "Maximum lag (-1 auto)", {}, ""}},
        NodeImplementationStatus::Implemented, 0});

    RegisterNode({NodeType::PACFNode, NodeCategory::TimeSeries, "PACF", ICON_FA_CHART_BAR,
        {"pacf", "partial", "autocorrelation", "time", "series"}, 0, false,
        "Compute partial autocorrelation by lag as a table", "", "",
        {{"Data", PinType::Dataset, true, "Input table"}},
        {{"PACF", PinType::Dataset, true, "Lag table with pacf and confidence bounds"}},
        {{"signal_col", "string", "", "Numeric signal column", {}, ""},
         {"max_lag", "int", "-1", "Maximum lag (-1 auto)", {}, ""}},
        NodeImplementationStatus::Implemented, 0});

    RegisterNode({NodeType::StationarityTest, NodeCategory::TimeSeries, "Stationarity Test", ICON_FA_SCALE_BALANCED,
        {"stationarity", "adf", "kpss", "unit", "root", "test"}, 0, false,
        "Run ADF and KPSS stationarity checks", "", "",
        {{"Data", PinType::Dataset, true, "Input table"}},
        {{"Results", PinType::Dataset, true, "One-row stationarity summary"}},
        {{"signal_col", "string", "", "Numeric signal column", {}, ""},
         {"max_lags", "int", "-1", "ADF max lags (-1 auto)", {}, ""}},
        NodeImplementationStatus::Implemented, 0});

    RegisterNode({NodeType::SeasonalityDetector, NodeCategory::TimeSeries, "Seasonality Detector", ICON_FA_CALENDAR,
        {"seasonality", "period", "detect", "frequency"}, 0, false,
        "Detect candidate seasonal periods", "", "",
        {{"Data", PinType::Dataset, true, "Input table"}},
        {{"Periods", PinType::Dataset, true, "Candidate period table"}},
        {{"signal_col", "string", "", "Numeric signal column", {}, ""},
         {"min_period", "int", "2", "Minimum period", {}, ""},
         {"max_period", "int", "-1", "Maximum period (-1 auto)", {}, ""}},
        NodeImplementationStatus::Implemented, 0});

    RegisterNode({NodeType::ARIMAForecaster, NodeCategory::TimeSeries, "ARIMA", ICON_FA_CHART_LINE,
        {"arima", "forecast", "prediction", "time", "series"}, 0, false,
        "Fit ARIMA in-sample and append fitted/residual columns", "", "",
        {{"Data", PinType::Dataset, true, "Input table"}},
        {{"Fitted", PinType::Dataset, true, "Input table plus fitted/residual"}},
        {{"signal_col", "string", "", "Numeric signal column", {}, ""},
         {"p", "int", "-1", "AR order (-1 auto)", {}, ""},
         {"d", "int", "-1", "Differencing order (-1 auto)", {}, ""},
         {"q", "int", "-1", "MA order (-1 auto)", {}, ""}},
        NodeImplementationStatus::Implemented, 0});

    RegisterNode({NodeType::ExponentialSmoothing, NodeCategory::TimeSeries, "Exp. Smoothing", ICON_FA_CHART_LINE,
        {"exponential", "smoothing", "holt", "winters", "forecast"}, 0, false,
        "Fit exponential smoothing in-sample and append fitted/residual columns", "", "",
        {{"Data", PinType::Dataset, true, "Input table"}},
        {{"Fitted", PinType::Dataset, true, "Input table plus fitted/residual"}},
        {{"signal_col", "string", "", "Numeric signal column", {}, ""},
         {"method", "dropdown", "simple", "Method", {"simple", "holt", "holt_winters"}, ""},
         {"alpha", "float", "-1", "Level smoothing (-1 auto)", {}, ""},
         {"beta", "float", "-1", "Trend smoothing (-1 auto)", {}, ""},
         {"gamma", "float", "-1", "Seasonal smoothing (-1 auto)", {}, ""},
         {"period", "int", "-1", "Seasonal period for Holt-Winters", {}, ""},
         {"damped", "bool", "false", "Use damped trend for Holt", {}, ""}},
        NodeImplementationStatus::Implemented, 0});
}

// =============================================================================
// Statistics Nodes (Tool-to-Node Migration)
// =============================================================================
void NodeMetadataRegistry::InitializeStatisticsNodes() {
    RegisterNode({NodeType::HypothesisTest, NodeCategory::Analytics, "Hypothesis Test", ICON_FA_FLASK,
        {"hypothesis", "test", "ttest", "anova", "chi", "square", "statistics"}, 0, false,
        "Statistical Hypothesis Testing", "", "",
        {{"Data", PinType::Dataset, true, "Data"}},
        {{"Results", PinType::Dataset, false, "Results"}},
        {{"test", "dropdown", "ttest_1samp", "Test Type", {"ttest_1samp", "ttest_ind", "ttest_paired", "anova", "chi_square", "mann_whitney"}, ""}},
        NodeImplementationStatus::Template, 0});

    RegisterNode({NodeType::DistributionFitter, NodeCategory::Analytics, "Distribution Fitter", ICON_FA_CHART_AREA,
        {"distribution", "fit", "normal", "exponential", "probability"}, 0, false,
        "Fit Probability Distribution", "", "",
        {{"Data", PinType::Dataset, true, "Data"}},
        {{"Parameters", PinType::Dataset, false, "Parameters"}},
        {{"distribution", "dropdown", "normal", "Distribution", {"normal", "uniform", "exponential", "lognormal", "poisson", "gamma"}, ""}},
        NodeImplementationStatus::Template, 0});
}

// =============================================================================
// Deep Learning Interpretation Nodes (Tool-to-Node Migration)
// =============================================================================
void NodeMetadataRegistry::InitializeInterpretationNodes() {
    RegisterNode({NodeType::GradCAMNode, NodeCategory::Visualization, "Grad-CAM", ICON_FA_FIRE,
        {"gradcam", "visualization", "cnn", "attention", "heatmap"}, 0, false,
        "Grad-CAM Visualization", "", "",
        {{"Model", PinType::Parameters, true, "Model"}, {"Image", PinType::Tensor, true, "Image"}},
        {{"Heatmap", PinType::Tensor, false, "Heatmap"}},
        {{"layer", "string", "", "Target layer", {}, ""}},
        NodeImplementationStatus::Template, 0});

    RegisterNode({NodeType::SaliencyMapNode, NodeCategory::Visualization, "Saliency Map", ICON_FA_EYE,
        {"saliency", "gradient", "visualization", "attention"}, 0, false,
        "Gradient Saliency Maps", "", "",
        {{"Model", PinType::Parameters, true, "Model"}, {"Image", PinType::Tensor, true, "Image"}},
        {{"Saliency", PinType::Tensor, false, "Saliency"}},
        {{"method", "dropdown", "vanilla", "Method", {"vanilla", "smoothgrad", "integrated"}, ""}},
        NodeImplementationStatus::Template, 0});
}

// =============================================================================
// Visualization Nodes — chart framework lives in
// cyxwiz-engine/src/gui/visualization/. Register metadata here so nodes
// show up in the Node Browser + search. BarChart is Cat 2 inspection
// (no output pin, rich dialog on double-click). Histogram / LinePlot /
// ScatterPlot / PieChart follow the same shape — add them as they get
// implementations.
// =============================================================================
void NodeMetadataRegistry::InitializeVisualizationNodes() {
    RegisterNode({NodeType::BarChart, NodeCategory::Visualization, "Bar Chart",
        ICON_FA_CHART_COLUMN,
        {"bar", "chart", "distribution", "histogram", "counts", "class"},
        0, false,
        "Count and plot a categorical column. Cat 2 inspection — "
        "double-click to open the rendering dialog. Pick a dataset + "
        "column, the chart counts distinct values and renders an "
        "ImPlot bar chart sorted by frequency with an imbalance flag "
        "when max/min >= 10x.",
        "", "",
        {{"Data", PinType::Tensor, true, "Tabular stream to chart"},
         {"Labels", PinType::Labels, false, "Optional label stream"}},
        {},
        {{"chart_type", "dropdown", "bar", "Orient", {"bar", "horizontal_bar"}, ""},
         {"column", "string", "", "Column", {}, ""},
         {"title", "string", "Bar Chart", "Title", {}, ""},
         {"max_bars", "int", "20", "Top-N cap", {}, ""}},
        NodeImplementationStatus::Implemented, 0});
}

// =============================================================================
// Optimization Nodes (Tool-to-Node Migration)
// =============================================================================
void NodeMetadataRegistry::InitializeOptimizationNodes() {
    RegisterNode({NodeType::GradientDescentViz, NodeCategory::Visualization, "GD Visualizer", ICON_FA_CHART_LINE,
        {"gradient", "descent", "optimizer", "visualize", "adam", "sgd"}, 0, false,
        "Gradient Descent Visualizer", "", "",
        {{"Function", PinType::Dataset, true, "Function"}},
        {{"Path", PinType::Dataset, false, "Path"}},
        {{"optimizer", "dropdown", "sgd", "Optimizer", {"sgd", "momentum", "adam", "rmsprop"}, ""}},
        NodeImplementationStatus::Template, 0});

    RegisterNode({NodeType::ConvexityAnalyzer, NodeCategory::Analytics, "Convexity Analyzer", ICON_FA_MOUNTAIN,
        {"convex", "hessian", "eigenvalue", "analysis"}, 0, false,
        "Analyze Function Convexity", "", "",
        {{"Function", PinType::Dataset, true, "Function"}},
        {{"Analysis", PinType::Dataset, false, "Analysis"}},
        {},
        NodeImplementationStatus::Template, 0});

    RegisterNode({NodeType::LPSolver, NodeCategory::Analytics, "LP Solver", ICON_FA_CHART_PIE,
        {"linear", "programming", "optimization", "simplex"}, 0, false,
        "Linear Programming Solver", "", "",
        {{"Objective", PinType::Dataset, true, "Objective"}, {"Constraints", PinType::Dataset, true, "Constraints"}},
        {{"Solution", PinType::Dataset, false, "Solution"}},
        {},
        NodeImplementationStatus::Template, 0});

    RegisterNode({NodeType::QPSolver, NodeCategory::Analytics, "QP Solver", ICON_FA_SQUARE_POLL_VERTICAL,
        {"quadratic", "programming", "optimization"}, 0, false,
        "Quadratic Programming Solver", "", "",
        {{"Objective", PinType::Dataset, true, "Objective"}, {"Constraints", PinType::Dataset, true, "Constraints"}},
        {{"Solution", PinType::Dataset, false, "Solution"}},
        {},
        NodeImplementationStatus::Template, 0});

    RegisterNode({NodeType::NumericalDifferentiation, NodeCategory::Analytics, "Differentiation", ICON_FA_SUPERSCRIPT,
        {"differentiation", "derivative", "gradient", "numerical"}, 0, false,
        "Numerical Differentiation", "", "",
        {{"Data", PinType::Dataset, true, "Data"}},
        {{"Derivative", PinType::Dataset, false, "Derivative"}},
        {{"method", "dropdown", "central", "Method", {"forward", "backward", "central"}, ""}},
        NodeImplementationStatus::Template, 0});

    RegisterNode({NodeType::NumericalIntegration, NodeCategory::Analytics, "Integration", ICON_FA_CHART_AREA,
        {"integration", "integral", "trapezoid", "simpson", "numerical"}, 0, false,
        "Numerical Integration", "", "",
        {{"Data", PinType::Dataset, true, "Data"}},
        {{"Integral", PinType::Dataset, false, "Integral"}},
        {{"method", "dropdown", "trapezoid", "Method", {"trapezoid", "simpson", "romberg", "gaussian"}, ""}},
        NodeImplementationStatus::Template, 0});
}

// =============================================================================
// Additional Text Processing Nodes (Tool-to-Node Migration)
// =============================================================================
void NodeMetadataRegistry::InitializeAdditionalTextNodes() {
    RegisterNode({NodeType::WordFrequencyNode, NodeCategory::TextProcessing, "Word Frequency", ICON_FA_CHART_BAR,
        {"word", "frequency", "count", "text", "analysis"}, 0, false,
        "Word Frequency Analysis", "", "",
        {{"Text", PinType::Dataset, true, "Text"}},
        {{"Frequencies", PinType::Dataset, false, "Frequencies"}},
        {{"top_n", "int", "100", "Top N words", {}, ""}},
        NodeImplementationStatus::Template, 0});

    RegisterNode({NodeType::TokenizerNode, NodeCategory::TextProcessing, "Tokenizer", ICON_FA_SCISSORS,
        {"tokenize", "text", "words", "split", "nlp"}, 0, false,
        "Text Tokenization", "", "",
        {{"Text", PinType::Dataset, true, "Text"}},
        {{"Tokens", PinType::Dataset, false, "Tokens"}},
        {{"method", "dropdown", "word", "Method", {"word", "sentence", "ngram", "bpe"}, ""}},
        NodeImplementationStatus::Template, 0});

    // Also register existing enum entries that weren't registered
    RegisterNode({NodeType::GMMCluster, NodeCategory::Analytics, "GMM Clustering", ICON_FA_CHART_PIE,
        {"gmm", "gaussian", "mixture", "clustering", "soft"}, 0, false,
        "Assign rows using a Gaussian mixture model",
        "Fits a Gaussian mixture to selected numeric features and appends hard Int32 cluster_id assignments; component probabilities are not emitted.", "",
        {{"Data", PinType::Dataset, true, "Input table containing numeric features"}},
        {{"Clustered", PinType::Dataset, true, "Input table plus cluster_id"}},
        {{"feature_cols", "string", "", "Comma-separated numeric features; empty auto-detects numeric columns", {}, "", "Feature columns", "Columns"},
         {"label_col", "string", "", "Optional label column excluded from automatic feature detection", {}, "", "Label column", "Columns"},
         {"n_components", "int", "3", "Number of mixture components", {}, "1-2147483647"},
         {"covariance_type", "enum", "full", "Covariance model", {"full", "tied", "diag", "spherical"}, ""},
         {"max_iter", "int", "100", "Maximum fit iterations", {}, "1-2147483647"},
         {"tol", "float", "0.001", "Convergence tolerance", {}, ""},
         {"n_init", "int", "1", "Independent initializations", {}, "1-2147483647"},
         {"seed", "int", "0", "Random seed; 0 selects nondeterministic initialization", {}, "0-2147483647"}},
        NodeImplementationStatus::Implemented, 0});

    RegisterNode({NodeType::Convolution1D, NodeCategory::Signal, "Convolution 1D", ICON_FA_WAVE_SQUARE,
        {"convolution", "1d", "signal", "filter", "kernel"}, 0, false,
        "Apply same-length 1D convolution", "Convolves one numeric column and replaces it with Float32 values while preserving row alignment and all other columns.", "",
        {{"Data", PinType::Dataset, true, "Input table containing the signal column"}},
        {{"Convolved", PinType::Dataset, true, "Input table with the selected column replaced by convolved values"}},
        {{"signal_col", "string", "", "Numeric signal column", {}, "", "", "", true, false},
         {"kernel", "string", "0.25,0.5,0.25", "Required comma-separated finite kernel taps", {}, "", "", "", true, false}},
        NodeImplementationStatus::Implemented, 0});

    RegisterNode({NodeType::WordEmbeddings, NodeCategory::TextProcessing, "Word Embeddings", ICON_FA_CUBE,
        {"embeddings", "word2vec", "glove", "vectors", "nlp"}, 0, false,
        "Word Embeddings", "", "",
        {{"Text", PinType::Dataset, true, "Text"}},
        {{"Embeddings", PinType::Dataset, false, "Embeddings"}},
        {{"method", "dropdown", "word2vec", "Method", {"word2vec", "glove", "fasttext"}, ""}},
        NodeImplementationStatus::Template, 0});

    RegisterNode({NodeType::NamedEntityRecognizer, NodeCategory::TextProcessing, "NER", ICON_FA_TAG,
        {"ner", "named", "entity", "recognition", "nlp"}, 0, false,
        "Named Entity Recognition", "", "",
        {{"Text", PinType::Dataset, true, "Text"}},
        {{"Entities", PinType::Dataset, false, "Entities"}},
        {},
        NodeImplementationStatus::Template, 0});
}


} // namespace cyxwiz
