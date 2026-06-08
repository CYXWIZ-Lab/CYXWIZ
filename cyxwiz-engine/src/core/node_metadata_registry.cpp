#include "node_metadata_registry.h"
#include "pipeline_runtime_capabilities.h"
#include "../gui/icons.h"
#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
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

void AppendHelpTextSection(NodeMetadata& metadata, const std::string& section) {
    if (section.empty() || metadata.help_text.find(section) != std::string::npos) {
        return;
    }
    if (!metadata.help_text.empty()) {
        metadata.help_text += "\n\n";
    }
    metadata.help_text += section;
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

    std::string summary = "Runtime support: mode=";
    summary += PipelineRuntimeSupportModeName(support.mode);
    summary += "; fail_mode=";
    summary += PipelineRuntimeFailModeName(support.fail_mode);
    summary += "; pipeline_executor=";
    summary += support.pipeline_executor_supported ? "supported" : "unsupported";
    summary += "; materializer=";
    summary += PipelineMaterializerStorageSupportName(
        support.materializer_storage_support);
    summary += "; owner=";
    summary += PipelineRuntimeImplementationOwnerName(
        support.implementation_owner);
    if (!reason.empty()) {
        summary += "; reason=";
        summary += reason;
    }
    AppendHelpTextSection(metadata, summary);
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

    InitializeKNIMENodes();
    InitializeUtilityNodes();
    ApplyRuntimeCapabilityStatus();
}

void NodeMetadataRegistry::ApplyRuntimeCapabilityStatus() {
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
        metadata.status = NodeImplementationStatus::Template;
        metadata.badge = "Blocked";

        const std::string reason =
            capability.reason != nullptr ? capability.reason : "";
        const auto support =
            ResolvePipelineRuntimeSupport(capability.legacy_type_name);
        ApplyRuntimeSupportAxes(metadata, support, reason);
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

            std::string summary = "Training backend support: mode=";
            summary += PipelineTrainingBackendSupportModeName(support.mode);
            summary += "; compile=";
            summary += support.compile_supported ? "supported" : "unsupported";
            summary += "; training=";
            summary += support.training_supported ? "supported" : "unsupported";
            if (support.reason != nullptr) {
                summary += "; reason=";
                summary += support.reason;
            }
            AppendHelpTextSection(metadata, summary);
        };

    for (const auto& capability :
         GetPipelineUnsupportedSequentialModelLayerCapabilities()) {
        apply_training_backend_status(capability.node_type);
    }

    for (const auto& capability :
         GetPipelineUnsupportedTrainingControlCapabilities()) {
        apply_training_backend_status(capability.node_type);
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

    // Iterate over all JSON files in the directory
    for (const auto& entry : fs::directory_iterator(directory)) {
        if (entry.path().extension() == ".json") {
            try {
                std::ifstream file(entry.path());
                if (!file.is_open()) {
                    spdlog::warn("NodeMetadataRegistry: Could not open template file: {}", entry.path().string());
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
                spdlog::error("NodeMetadataRegistry: Error parsing template file {}: {}", entry.path().string(), e.what());
            }
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
void NodeMetadataRegistry::InitializeDataSourceNodes() {
    // ===== Smart I/O Nodes (Universal - replaces individual format nodes) =====
    RegisterNode({NodeType::DataInput, NodeCategory::DataSources, "Data Input", ICON_FA_FILE_IMPORT,
        {"csv", "excel", "json", "parquet", "hdf5", "input", "load", "read", "import", "file"}, 0, false,
        "Universal data loader - auto-detects CSV, Excel, JSON, Parquet, HDF5", "", "",
        {}, {{"Data", PinType::Dataset, true, "Output dataset"}},
        {{"file_path", "file", "", "Data file", {}, "*.csv;*.xlsx;*.json;*.parquet;*.hdf5"},
         {"configured", "bool", "false", "Dialog configured", {}, ""}},
        NodeImplementationStatus::Implemented, 0});

    RegisterNode({NodeType::DataOutput, NodeCategory::DataSources, "Data Output", ICON_FA_FILE_EXPORT,
        {"csv", "excel", "json", "parquet", "hdf5", "output", "save", "write", "export", "file"}, 0, false,
        "Universal data exporter - supports CSV, Excel, JSON, Parquet, HDF5", "", "",
        {{"Data", PinType::Dataset, true, "Input dataset"}}, {},
        {{"file_path", "file", "", "Output file", {}, "*.csv;*.xlsx;*.json;*.parquet;*.hdf5"},
         {"configured", "bool", "false", "Dialog configured", {}, ""}},
        NodeImplementationStatus::Implemented, 0});

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
        NodeImplementationStatus::Implemented, 0});

    RegisterNode({NodeType::ParquetFile, NodeCategory::DataSources, "Parquet Reader", ICON_FA_DATABASE,
        {"parquet", "columnar"}, 0, false, "Read Parquet file", "", "",
        {}, {{"Table", PinType::Dataset, true, "Arrow table"}},
        {{"file_path", "file", "", "Parquet file", {}, "*.parquet"}},
        NodeImplementationStatus::Implemented, 0});

    RegisterNode({NodeType::JSONFile, NodeCategory::DataSources, "JSON Reader", ICON_FA_BRACKETS_CURLY,
        {"json", "javascript"}, 0, false, "Read JSON file", "", "",
        {}, {{"Table", PinType::Dataset, true, "Arrow table"}},
        {{"file_path", "file", "", "JSON file", {}, "*.json"}},
        NodeImplementationStatus::Implemented, 0});

    RegisterNode({NodeType::SQLQuery, NodeCategory::DataSources, "SQL Query", ICON_FA_DATABASE,
        {"sql", "query", "database"}, 0, false, "Execute SQL query", "", "",
        {{"Source", PinType::Dataset, false, "Input table"}},
        {{"Result", PinType::Dataset, true, "Query result"}},
        {{"query", "string", "SELECT * FROM data", "SQL query", {}, ""}},
        NodeImplementationStatus::Implemented, 0});

    RegisterNode({NodeType::HDF5Dataset, NodeCategory::DataSources, "HDF5 Reader", ICON_FA_HARD_DRIVE,
        {"hdf5", "h5", "scientific"}, 0, false, "Read HDF5 dataset", "", "",
        {}, {{"Table", PinType::Dataset, true, "Arrow table"}},
        {{"file_path", "file", "", "HDF5 file", {}, "*.h5;*.hdf5"}},
        NodeImplementationStatus::Implemented, 0});

    RegisterNode({NodeType::RESTAPISource, NodeCategory::DataSources, "REST API", ICON_FA_GLOBE,
        {"rest", "api", "http"}, 0, false, "Fetch from REST API", "", "",
        {}, {{"Response", PinType::Dataset, true, "API response"}},
        {{"url", "string", "", "API URL", {}, ""},
         {"method", "enum", "GET", "HTTP method", {"GET", "POST"}, ""}},
        NodeImplementationStatus::Implemented, 0});

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
        {{"condition", "string", "", "WHERE condition", {}, ""}},
        NodeImplementationStatus::Implemented, 0});

    RegisterNode({NodeType::SelectColumns, NodeCategory::DataTransform, "Column Filter", ICON_FA_TABLE_COLUMNS,
        {"select", "columns", "project"}, 0, false, "Select columns", "", "",
        {{"Table", PinType::Dataset, true, "Input"}},
        {{"Selected", PinType::Dataset, true, "Selected columns"}},
        {{"columns", "string", "", "Column names", {}, ""}},
        NodeImplementationStatus::Implemented, 0});

    RegisterNode({NodeType::JoinTables, NodeCategory::DataTransform, "Joiner", ICON_FA_CODE_BRANCH,
        {"join", "merge", "combine"}, 0, false, "Join tables", "", "",
        {{"Left", PinType::Dataset, true, "Left"}, {"Right", PinType::Dataset, true, "Right"}},
        {{"Joined", PinType::Dataset, true, "Joined"}},
        {{"join_type", "enum", "INNER", "Join type", {"INNER", "LEFT", "RIGHT", "OUTER"}, ""}},
        NodeImplementationStatus::Implemented, 0});

    RegisterNode({NodeType::GroupByAggregate, NodeCategory::DataTransform, "GroupBy", ICON_FA_OBJECT_GROUP,
        {"group", "aggregate", "sum"}, 0, false, "Group and aggregate", "", "",
        {{"Table", PinType::Dataset, true, "Input"}},
        {{"Grouped", PinType::Dataset, true, "Aggregated"}},
        {{"group_by", "string", "", "Group columns", {}, ""}},
        NodeImplementationStatus::Implemented, 0});

    RegisterNode({NodeType::SortRows, NodeCategory::DataTransform, "Sorter", ICON_FA_ARROW_DOWN_LONG,
        {"sort", "order"}, 0, false, "Sort rows", "", "",
        {{"Table", PinType::Dataset, true, "Input"}},
        {{"Sorted", PinType::Dataset, true, "Sorted"}},
        {{"sort_by", "string", "", "Sort column", {}, ""}},
        NodeImplementationStatus::Implemented, 0});

    RegisterNode({NodeType::FillMissingValues, NodeCategory::DataTransform, "Missing Value", ICON_FA_ERASER,
        {"missing", "null", "fill"}, 0, false, "Handle missing values", "", "",
        {{"Table", PinType::Dataset, true, "Input"}},
        {{"Filled", PinType::Dataset, true, "Filled"}},
        {{"strategy", "enum", "mean", "Strategy", {"mean", "median", "mode", "constant"}, ""}},
        NodeImplementationStatus::Implemented, 0});

    RegisterNode({NodeType::RemoveDuplicateRows, NodeCategory::DataTransform, "Duplicate Remover", ICON_FA_COPY,
        {"duplicate", "unique"}, 0, false, "Remove duplicates", "", "",
        {{"Table", PinType::Dataset, true, "Input"}},
        {{"Unique", PinType::Dataset, true, "Unique rows"}},
        {}, NodeImplementationStatus::Implemented, 0});

    RegisterNode({NodeType::Normalize, NodeCategory::DataTransform, "Normalizer", ICON_FA_SCALE_BALANCED,
        {"normalize", "scale"}, 0, false, "Normalize values", "", "",
        {{"Data", PinType::Tensor, true, "Input"}},
        {{"Normalized", PinType::Tensor, true, "Normalized"}},
        {{"method", "enum", "minmax", "Method", {"minmax", "zscore"}, ""}},
        NodeImplementationStatus::Implemented, 0});

    RegisterNode({NodeType::OneHotEncode, NodeCategory::DataTransform, "One-Hot Encoder", ICON_FA_TH,
        {"onehot", "encode", "categorical"}, 0, false, "One-hot encode", "", "",
        {{"Labels", PinType::Labels, true, "Labels"}},
        {{"Encoded", PinType::Tensor, true, "Encoded"}},
        {{"num_classes", "int", "0", "Classes (0=auto)", {}, ""}},
        NodeImplementationStatus::Implemented, 0});
}

// =============================================================================
// Analytics Nodes
// =============================================================================
void NodeMetadataRegistry::InitializeAnalyticsNodes() {
    RegisterNode({NodeType::DescribeStats, NodeCategory::Analytics, "Descriptive Stats", ICON_FA_CHART_SIMPLE,
        {"statistics", "describe", "summary"}, 0, false, "Statistical summary", "", "",
        {{"Table", PinType::Dataset, true, "Input"}},
        {{"Stats", PinType::Dataset, true, "Summary"}},
        {}, NodeImplementationStatus::Implemented, 0});

    RegisterNode({NodeType::CorrelationMatrix, NodeCategory::Analytics, "Correlation Matrix", ICON_FA_TABLE_CELLS,
        {"correlation", "pearson"}, 0, false, "Compute correlations", "", "",
        {{"Table", PinType::Dataset, true, "Input"}},
        {{"Matrix", PinType::Dataset, true, "Correlation matrix"}},
        {{"method", "enum", "pearson", "Method", {"pearson", "spearman"}, ""}},
        NodeImplementationStatus::Implemented, 0});

    RegisterNode({NodeType::VisualizeData, NodeCategory::Analytics, "Visualizer", ICON_FA_CHART_LINE,
        {"plot", "chart", "visualize"}, 0, false, "Create visualizations", "", "",
        {{"Table", PinType::Dataset, true, "Input"}}, {},
        {{"chart_type", "enum", "scatter", "Type", {"scatter", "bar", "line", "histogram"}, ""}},
        NodeImplementationStatus::Implemented, 0});

    RegisterNode({NodeType::SampleRows, NodeCategory::Analytics, "Row Sampler", ICON_FA_DICE,
        {"sample", "random"}, 0, false, "Sample random rows", "", "",
        {{"Table", PinType::Dataset, true, "Input"}},
        {{"Sample", PinType::Dataset, true, "Sampled"}},
        {{"count", "int", "100", "Sample size", {}, ""}},
        NodeImplementationStatus::Implemented, 0});

    RegisterNode({NodeType::ValueCounts, NodeCategory::Analytics, "Value Counts", ICON_FA_LIST_OL,
        {"count", "frequency"}, 0, false, "Count unique values", "", "",
        {{"Table", PinType::Dataset, true, "Input"}},
        {{"Counts", PinType::Dataset, true, "Counts"}},
        {{"column", "string", "", "Column", {}, ""}},
        NodeImplementationStatus::Implemented, 0});

    // ===== Machine Learning Algorithms (Phase 4) =====

    // Clustering
    RegisterNode({NodeType::KMeansCluster, NodeCategory::Analytics, "K-Means", ICON_FA_CIRCLE_NODES,
        {"kmeans", "cluster", "clustering"}, 0, false, "K-Means clustering algorithm",
        "Partitions data into k clusters by minimizing within-cluster variance. "
        "Use elbow method or silhouette score to find optimal k.", "",
        {{"Data", PinType::Dataset, true, "Input data matrix"}},
        {{"Labels", PinType::Labels, true, "Cluster assignments"}, {"Centroids", PinType::Dataset, true, "Cluster centers"}},
        {{"n_clusters", "int", "3", "Number of clusters", {}, "2-100"},
         {"max_iter", "int", "300", "Max iterations", {}, ""},
         {"init", "enum", "k-means++", "Initialization", {"k-means++", "random"}, ""}},
        NodeImplementationStatus::Implemented, 0});

    RegisterNode({NodeType::DBSCANCluster, NodeCategory::Analytics, "DBSCAN", ICON_FA_CIRCLE_NODES,
        {"dbscan", "density", "cluster"}, 0, false, "Density-based clustering",
        "Finds clusters of arbitrary shape based on density. Does not require specifying k.", "",
        {{"Data", PinType::Dataset, true, "Input data matrix"}},
        {{"Labels", PinType::Labels, true, "Cluster labels (-1 = noise)"}},
        {{"eps", "float", "0.5", "Epsilon (neighborhood radius)", {}, "0.01-10"},
         {"min_samples", "int", "5", "Min samples per cluster", {}, ""}},
        NodeImplementationStatus::Implemented, 0});

    RegisterNode({NodeType::HierarchicalCluster, NodeCategory::Analytics, "Hierarchical Clustering", ICON_FA_SITEMAP,
        {"hierarchical", "dendrogram", "agglomerative"}, 0, false, "Agglomerative hierarchical clustering",
        "Builds a hierarchy of clusters using linkage criteria.", "",
        {{"Data", PinType::Dataset, true, "Input data matrix"}},
        {{"Labels", PinType::Labels, true, "Cluster assignments"}, {"Dendrogram", PinType::Dataset, true, "Linkage matrix"}},
        {{"n_clusters", "int", "3", "Number of clusters", {}, ""},
         {"linkage", "enum", "ward", "Linkage method", {"ward", "complete", "average", "single"}, ""}},
        NodeImplementationStatus::Implemented, 0});

    // Dimensionality Reduction
    RegisterNode({NodeType::PCANode, NodeCategory::Analytics, "PCA", ICON_FA_COMPRESS,
        {"pca", "principal", "dimensionality"}, 0, false, "Principal Component Analysis",
        "Reduces dimensionality while preserving maximum variance.", "",
        {{"Data", PinType::Dataset, true, "Input data matrix"}},
        {{"Transformed", PinType::Dataset, true, "Reduced data"}, {"Components", PinType::Dataset, true, "Principal components"}},
        {{"n_components", "int", "2", "Components to keep", {}, ""},
         {"whiten", "bool", "false", "Whiten output", {}, ""}},
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
        {"decision", "tree", "classifier"}, 0, false, "Decision Tree classifier",
        "Learns decision rules from features. Interpretable and handles non-linear boundaries.", "",
        {{"X", PinType::Dataset, true, "Features"}, {"y", PinType::Labels, true, "Labels"}},
        {{"Model", PinType::Parameters, true, "Trained model"}, {"Predictions", PinType::Labels, true, "Predictions"}},
        {{"max_depth", "int", "10", "Max depth", {}, ""},
         {"min_samples_split", "int", "2", "Min samples to split", {}, ""},
         {"criterion", "enum", "gini", "Split criterion", {"gini", "entropy"}, ""}},
        NodeImplementationStatus::Template, 0, "Blocked"});

    RegisterNode({NodeType::RandomForestClassifier, NodeCategory::Analytics, "Random Forest", ICON_FA_CUBES,
        {"random", "forest", "ensemble"}, 0, false, "Random Forest ensemble",
        "Ensemble of decision trees with bagging and feature randomization.", "",
        {{"X", PinType::Dataset, true, "Features"}, {"y", PinType::Labels, true, "Labels"}},
        {{"Model", PinType::Parameters, true, "Trained model"}, {"Predictions", PinType::Labels, true, "Predictions"}},
        {{"n_estimators", "int", "100", "Number of trees", {}, ""},
         {"max_depth", "int", "10", "Max depth per tree", {}, ""},
         {"max_features", "enum", "sqrt", "Features per split", {"sqrt", "log2", "all"}, ""}},
        NodeImplementationStatus::Template, 0, "Blocked"});

    RegisterNode({NodeType::GradientBoostingClassifier, NodeCategory::Analytics, "Gradient Boosting", ICON_FA_CHART_LINE,
        {"gradient", "boosting", "classifier", "trees"}, 0, false, "Gradient Boosted Trees classifier",
        "Boosted decision-tree classifier. Not wired to a real graph executor yet.", "",
        {{"X", PinType::Dataset, true, "Features"}, {"y", PinType::Labels, true, "Labels"}},
        {{"Model", PinType::Parameters, true, "Trained model"}, {"Predictions", PinType::Labels, true, "Predictions"}},
        {{"n_estimators", "int", "100", "Number of trees", {}, ""},
         {"learning_rate", "float", "0.1", "Learning rate", {}, ""},
         {"max_depth", "int", "3", "Max tree depth", {}, ""}},
        NodeImplementationStatus::Template, 0, "Blocked"});

    RegisterNode({NodeType::SVMClassifier, NodeCategory::Analytics, "SVM Classifier", ICON_FA_BORDER_ALL,
        {"svm", "support", "vector"}, 0, false, "Support Vector Machine classifier",
        "Finds optimal hyperplane for classification using kernel trick.", "",
        {{"X", PinType::Dataset, true, "Features"}, {"y", PinType::Labels, true, "Labels"}},
        {{"Model", PinType::Parameters, true, "Trained model"}, {"Predictions", PinType::Labels, true, "Predictions"}},
        {{"kernel", "enum", "rbf", "Kernel function", {"linear", "rbf", "poly", "sigmoid"}, ""},
         {"C", "float", "1.0", "Regularization", {}, ""},
         {"gamma", "enum", "scale", "Kernel coefficient", {"scale", "auto"}, ""}},
        NodeImplementationStatus::Template, 0, "Blocked"});

    RegisterNode({NodeType::KNNClassifier, NodeCategory::Analytics, "KNN Classifier", ICON_FA_USERS,
        {"knn", "nearest", "neighbor"}, 0, false, "K-Nearest Neighbors classifier",
        "Classifies based on majority vote of k nearest neighbors.", "",
        {{"X", PinType::Dataset, true, "Features"}, {"y", PinType::Labels, true, "Labels"}},
        {{"Model", PinType::Parameters, true, "Trained model"}, {"Predictions", PinType::Labels, true, "Predictions"}},
        {{"n_neighbors", "int", "5", "Number of neighbors", {}, "1-100"},
         {"weights", "enum", "uniform", "Weight function", {"uniform", "distance"}, ""},
         {"metric", "enum", "euclidean", "Distance metric", {"euclidean", "manhattan", "cosine"}, ""}},
        NodeImplementationStatus::Template, 0, "Blocked"});

    RegisterNode({NodeType::NaiveBayesClassifier, NodeCategory::Analytics, "Naive Bayes", ICON_FA_PERCENT,
        {"naive", "bayes", "classifier"}, 0, false, "Naive Bayes classifier",
        "Probabilistic classifier. Not wired to a real graph executor yet.", "",
        {{"X", PinType::Dataset, true, "Features"}, {"y", PinType::Labels, true, "Labels"}},
        {{"Model", PinType::Parameters, true, "Trained model"}, {"Predictions", PinType::Labels, true, "Predictions"}},
        {{"variant", "enum", "gaussian", "Bayes variant", {"gaussian", "multinomial", "bernoulli"}, ""}},
        NodeImplementationStatus::Template, 0, "Blocked"});

    RegisterNode({NodeType::LogisticRegressionNode, NodeCategory::Analytics, "Logistic Regression", ICON_FA_PERCENT,
        {"logistic", "regression", "classifier"}, 0, false, "Logistic Regression classifier",
        "Linear classifier with sigmoid activation. Outputs probabilities.", "",
        {{"X", PinType::Dataset, true, "Features"}, {"y", PinType::Labels, true, "Labels"}},
        {{"Model", PinType::Parameters, true, "Trained model"}, {"Predictions", PinType::Labels, true, "Predictions"}, {"Probabilities", PinType::Dataset, true, "Class probabilities"}},
        {{"penalty", "enum", "l2", "Regularization", {"l1", "l2", "elasticnet", "none"}, ""},
         {"C", "float", "1.0", "Inverse regularization strength", {}, ""}},
        NodeImplementationStatus::Template, 0, "Blocked"});

    // Regression
    RegisterNode({NodeType::LinearRegressionNode, NodeCategory::Analytics, "Linear Regression", ICON_FA_CHART_LINE,
        {"linear", "regression"}, 0, false, "Linear Regression",
        "Fits a linear model and appends prediction/residual columns.", "",
        {{"Data", PinType::Dataset, true, "Input table"}},
        {{"Fitted", PinType::Dataset, true, "Input table plus prediction/residual"}},
        {{"feature_cols", "string", "", "Predictor columns", {}, ""},
         {"target_col", "string", "", "Target column", {}, ""},
         {"fit_intercept", "bool", "true", "Fit intercept", {}, ""}},
        NodeImplementationStatus::Implemented, 0});

    RegisterNode({NodeType::PolynomialRegressionNode, NodeCategory::Analytics, "Polynomial Regression", ICON_FA_CHART_LINE,
        {"polynomial", "regression", "curve", "fit"}, 0, false, "Polynomial Regression",
        "Fits a univariate polynomial model and appends prediction/residual columns.", "",
        {{"Data", PinType::Dataset, true, "Input table"}},
        {{"Fitted", PinType::Dataset, true, "Input table plus prediction/residual"}},
        {{"feature_col", "string", "", "Single predictor column", {}, ""},
         {"target_col", "string", "", "Target column", {}, ""},
         {"degree", "int", "2", "Polynomial degree", {}, "1-10"}},
        NodeImplementationStatus::Implemented, 0});

    // Model Evaluation
    RegisterNode({NodeType::ConfusionMatrixNode, NodeCategory::Analytics, "Confusion Matrix", ICON_FA_TABLE_CELLS,
        {"confusion", "matrix", "evaluation"}, 0, false, "Confusion Matrix visualization",
        "Visualize classification performance with true/false positives/negatives.", "",
        {{"y_true", PinType::Labels, true, "True labels"}, {"y_pred", PinType::Labels, true, "Predictions"}},
        {{"Matrix", PinType::Dataset, true, "Confusion matrix"}},
        {{"normalize", "enum", "none", "Normalization", {"none", "true", "pred", "all"}, ""}},
        NodeImplementationStatus::Template, 0, "UI-only"});

    RegisterNode({NodeType::ROCCurveNode, NodeCategory::Analytics, "ROC Curve", ICON_FA_CHART_AREA,
        {"roc", "auc", "curve"}, 0, false, "ROC curve and AUC",
        "Receiver Operating Characteristic curve for binary classification.", "",
        {{"y_true", PinType::Labels, true, "True labels"}, {"y_score", PinType::Dataset, true, "Prediction scores"}},
        {{"FPR", PinType::Dataset, true, "False positive rates"}, {"TPR", PinType::Dataset, true, "True positive rates"}, {"AUC", PinType::Dataset, true, "Area under curve"}},
        {}, NodeImplementationStatus::Template, 0, "UI-only"});

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

    // Preprocessing (Phase 4)
    RegisterNode({NodeType::StandardScaler, NodeCategory::Preprocessing, "Standard Scaler", ICON_FA_SCALE_BALANCED,
        {"standardize", "zscore", "scaler"}, 0, false, "Z-score standardization",
        "Transforms features to have mean=0 and std=1.", "",
        {{"Data", PinType::Dataset, true, "Input data"}},
        {{"Scaled", PinType::Dataset, true, "Standardized data"}},
        {{"columns", "string", "", "Columns to scale (empty = numeric auto-detect)", {}, ""},
         {"label_col", "string", "", "Label column to exclude", {}, ""},
         {"with_mean", "bool", "true", "Center data", {}, ""},
         {"with_std", "bool", "true", "Scale to unit variance", {}, ""}},
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
        {{"column", "string", "", "Column to encode", {}, ""}},
        NodeImplementationStatus::Implemented, 0});

    RegisterNode({NodeType::OrdinalEncoder, NodeCategory::Preprocessing, "Ordinal Encoder", ICON_FA_LIST_OL,
        {"ordinal", "encoder", "categorical"}, 0, false, "Encode categorical columns",
        "Replaces one or more string columns with alphabetical int32 category codes.", "",
        {{"Data", PinType::Dataset, true, "Input data"}},
        {{"Encoded", PinType::Dataset, true, "Encoded data"}},
        {{"columns", "string", "", "Columns to encode", {}, ""},
         {"categories", "string", "auto", "Category ordering (auto only in v1)", {}, ""}},
        NodeImplementationStatus::Implemented, 0});

    RegisterNode({NodeType::TargetEncoder, NodeCategory::Preprocessing, "Target Encoder", ICON_FA_PERCENT,
        {"target", "encoder", "categorical", "mean"}, 0, false, "Target mean encoding",
        "Replaces categorical columns with smoothed target means.", "",
        {{"Data", PinType::Dataset, true, "Input data"}},
        {{"Encoded", PinType::Dataset, true, "Encoded data"}},
        {{"columns", "string", "", "Categorical columns to encode", {}, ""},
         {"target_col", "string", "", "Numeric target column", {}, ""},
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
         {"threshold", "float", "1.5", "IQR multiplier or Z-score threshold", {}, ""},
         {"action", "dropdown", "flag", "Action (flag only in v1)", {"flag"}, ""}},
        NodeImplementationStatus::Implemented, 0});

    RegisterNode({NodeType::DataProfiler, NodeCategory::Analytics, "Data Profiler", ICON_FA_MAGNIFYING_GLASS_CHART,
        {"profiler", "eda", "exploration"}, 0, false, "Comprehensive data profiling",
        "Generate detailed data quality report: types, missing, distributions, correlations.", "",
        {{"Data", PinType::Dataset, true, "Input dataset"}},
        {{"Report", PinType::Dataset, true, "Profiling report"}},
        {{"minimal", "bool", "false", "Minimal mode (faster)", {}, ""}},
        NodeImplementationStatus::Template, 0, "UI-only"});
}

// =============================================================================
// ML Layer Nodes
// =============================================================================
void NodeMetadataRegistry::InitializeLayerNodes() {
    RegisterNode({NodeType::Dense, NodeCategory::Layers, "Dense", ICON_FA_LAYER_GROUP,
        {"dense", "linear", "fc"}, 0, false, "Fully connected layer", "", "",
        {{"Input", PinType::Tensor, true, "Input"}},
        {{"Output", PinType::Tensor, true, "Output"}},
        {{"units", "int", "128", "Output units", {}, ""}},
        NodeImplementationStatus::Implemented, 0});

    RegisterNode({NodeType::Conv2D, NodeCategory::Layers, "Conv2D", ICON_FA_BORDER_ALL,
        {"conv", "convolution"}, 0, false, "2D convolution", "", "",
        {{"Input", PinType::Tensor, true, "Input [N,C,H,W]"}},
        {{"Output", PinType::Tensor, true, "Output"}},
        {{"filters", "int", "32", "Filters", {}, ""}, {"kernel_size", "int", "3", "Kernel", {}, ""}},
        NodeImplementationStatus::Template, 0, "Blocked"});

    RegisterNode({NodeType::LSTM, NodeCategory::Recurrent, "LSTM", ICON_FA_REPEAT,
        {"lstm", "recurrent", "sequence"}, 0, false, "LSTM layer", "", "",
        {{"Input", PinType::Tensor, true, "Input [N,T,F]"}},
        {{"Output", PinType::Tensor, true, "Output"}, {"Hidden", PinType::Tensor, false, "Hidden"}},
        {{"input_size", "int", "256", "Input size", {}, ""},
         {"hidden_size", "int", "128", "Hidden size", {}, ""},
         {"num_layers", "int", "1", "Stacked layers", {}, ""},
         {"bidirectional", "bool", "false", "Bidirectional", {}, ""},
         {"dropout", "float", "0.0", "Dropout", {}, ""}},
        NodeImplementationStatus::Implemented, 0});

    RegisterNode({NodeType::GRU, NodeCategory::Recurrent, "GRU", ICON_FA_REPEAT,
        {"gru", "recurrent", "sequence"}, 0, false, "GRU layer", "", "",
        {{"Input", PinType::Tensor, true, "Input [N,T,F]"}},
        {{"Output", PinType::Tensor, true, "Output"}, {"Hidden", PinType::Tensor, false, "Hidden"}},
        {{"input_size", "int", "256", "Input size", {}, ""},
         {"hidden_size", "int", "128", "Hidden size", {}, ""},
         {"num_layers", "int", "1", "Stacked layers", {}, ""},
         {"bidirectional", "bool", "false", "Bidirectional", {}, ""},
         {"dropout", "float", "0.0", "Dropout", {}, ""}},
        NodeImplementationStatus::Implemented, 0});

    RegisterNode({NodeType::RNN, NodeCategory::Recurrent, "RNN", ICON_FA_REPEAT,
        {"rnn", "recurrent", "sequence"}, 0, false, "Simple RNN layer", "", "",
        {{"Input", PinType::Tensor, true, "Input [N,T,F]"}},
        {{"Output", PinType::Tensor, true, "Output"}, {"Hidden", PinType::Tensor, false, "Hidden"}},
        {{"input_size", "int", "256", "Input size", {}, ""},
         {"hidden_size", "int", "128", "Hidden size", {}, ""},
         {"num_layers", "int", "1", "Stacked layers", {}, ""},
         {"nonlinearity", "string", "tanh", "Nonlinearity", {}, ""}},
        NodeImplementationStatus::Template, 0});

    RegisterNode({NodeType::Bidirectional, NodeCategory::Recurrent, "Bidirectional", ICON_FA_REPEAT,
        {"bidirectional", "wrapper", "sequence"}, 0, false, "Bidirectional wrapper", "", "",
        {{"Input", PinType::Tensor, true, "Input"}},
        {{"Output", PinType::Tensor, true, "Output"}},
        {{"merge_mode", "string", "concat", "Merge mode", {}, ""}},
        NodeImplementationStatus::Template, 0});

    RegisterNode({NodeType::Dropout, NodeCategory::Regularization, "Dropout", ICON_FA_SHUFFLE,
        {"dropout", "regularization"}, 0, false, "Dropout layer", "", "",
        {{"Input", PinType::Tensor, true, "Input"}},
        {{"Output", PinType::Tensor, true, "Output"}},
        {{"rate", "float", "0.5", "Rate", {}, ""}},
        NodeImplementationStatus::Implemented, 0});

    RegisterNode({NodeType::BatchNorm, NodeCategory::Normalization, "BatchNorm", ICON_FA_SCALE_BALANCED,
        {"batchnorm", "normalization"}, 0, false, "Batch normalization", "", "",
        {{"Input", PinType::Tensor, true, "Input"}},
        {{"Output", PinType::Tensor, true, "Normalized"}},
        {{"momentum", "float", "0.1", "Momentum", {}, ""}},
        NodeImplementationStatus::Implemented, 0});

    RegisterNode({NodeType::MultiHeadAttention, NodeCategory::Attention, "Multi-Head Attention", ICON_FA_BULLSEYE,
        {"attention", "transformer"}, 0, false, "Multi-head attention", "", "",
        {{"Query", PinType::Tensor, true, "Query"}, {"Key", PinType::Tensor, false, "Key"}},
        {{"Output", PinType::Tensor, true, "Output"}},
        {{"embed_dim", "int", "512", "Dim", {}, ""}, {"num_heads", "int", "8", "Heads", {}, ""}},
        NodeImplementationStatus::Template, 0});

    RegisterNode({NodeType::MaxPool2D, NodeCategory::Pooling, "MaxPool2D", ICON_FA_COMPRESS,
        {"maxpool", "pooling"}, 0, false, "Max pooling", "", "",
        {{"Input", PinType::Tensor, true, "Input"}},
        {{"Output", PinType::Tensor, true, "Pooled"}},
        {{"kernel_size", "int", "2", "Kernel", {}, ""}},
        NodeImplementationStatus::Template, 0, "Blocked"});

    RegisterNode({NodeType::AvgPool2D, NodeCategory::Pooling, "AvgPool2D", ICON_FA_COMPRESS,
        {"avgpool", "average", "pooling"}, 0, false, "Average pooling", "", "",
        {{"Input", PinType::Tensor, true, "Input"}},
        {{"Output", PinType::Tensor, true, "Pooled"}},
        {{"kernel_size", "int", "2", "Kernel", {}, ""}},
        NodeImplementationStatus::Template, 0, "Blocked"});

    RegisterNode({NodeType::GlobalMaxPool, NodeCategory::Pooling, "Global Max Pool", ICON_FA_COMPRESS,
        {"global", "max", "pooling"}, 0, false, "Global max pooling", "", "",
        {{"Input", PinType::Tensor, true, "Input"}},
        {{"Output", PinType::Tensor, true, "Pooled"}},
        {}, NodeImplementationStatus::Template, 0, "Blocked"});

    RegisterNode({NodeType::GlobalAvgPool, NodeCategory::Pooling, "Global Avg Pool", ICON_FA_COMPRESS,
        {"global", "average", "pooling"}, 0, false, "Global average pooling", "", "",
        {{"Input", PinType::Tensor, true, "Input"}},
        {{"Output", PinType::Tensor, true, "Pooled"}},
        {}, NodeImplementationStatus::Template, 0, "Blocked"});

    RegisterNode({NodeType::ConvTranspose2D, NodeCategory::Upsampling, "ConvTranspose2D", ICON_FA_EXPAND,
        {"convtranspose", "transposed", "convolution", "upsample"}, 0, false, "Transposed convolution", "", "",
        {{"Input", PinType::Tensor, true, "Input"}},
        {{"Output", PinType::Tensor, true, "Upsampled"}},
        {{"filters", "int", "32", "Filters", {}, ""},
         {"kernel_size", "int", "3", "Kernel", {}, ""},
         {"stride", "int", "2", "Stride", {}, ""}},
        NodeImplementationStatus::Template, 0, "Blocked"});

    RegisterNode({NodeType::Upsample, NodeCategory::Upsampling, "Upsample", ICON_FA_EXPAND,
        {"upsample", "resize", "interpolate"}, 0, false, "Tensor upsampling", "", "",
        {{"Input", PinType::Tensor, true, "Input"}},
        {{"Output", PinType::Tensor, true, "Upsampled"}},
        {{"scale_factor", "int", "2", "Scale factor", {}, ""},
         {"mode", "enum", "nearest", "Interpolation mode", {"nearest", "bilinear"}, ""}},
        NodeImplementationStatus::Template, 0, "Blocked"});

    RegisterNode({NodeType::PixelShuffle, NodeCategory::Upsampling, "Pixel Shuffle", ICON_FA_EXPAND,
        {"pixel", "shuffle", "subpixel", "upsample"}, 0, false, "Pixel shuffle upsampling", "", "",
        {{"Input", PinType::Tensor, true, "Input"}},
        {{"Output", PinType::Tensor, true, "Upsampled"}},
        {{"scale_factor", "int", "2", "Scale factor", {}, ""}},
        NodeImplementationStatus::Template, 0, "Blocked"});

    RegisterNode({NodeType::Flatten, NodeCategory::ShapeOps, "Flatten", ICON_FA_ARROWS_LEFT_RIGHT,
        {"flatten", "reshape"}, 0, false, "Flatten tensor", "", "",
        {{"Input", PinType::Tensor, true, "Input"}},
        {{"Output", PinType::Tensor, true, "Flattened"}},
        {}, NodeImplementationStatus::Implemented, 0});

    RegisterNode({NodeType::Reshape, NodeCategory::ShapeOps, "Reshape", ICON_FA_ARROWS_LEFT_RIGHT,
        {"reshape", "view", "shape", "tensor"}, 0, false, "Reshape tensor dimensions", "", "",
        {{"Input", PinType::Tensor, true, "Input"}},
        {{"Output", PinType::Tensor, true, "Reshaped"}},
        {{"shape", "string", "-1,256", "Target shape", {}, ""}},
        NodeImplementationStatus::Implemented, 0});

    RegisterNode({NodeType::View, NodeCategory::ShapeOps, "View", ICON_FA_ARROWS_LEFT_RIGHT,
        {"view", "reshape", "shape", "tensor"}, 0, false, "View tensor with a new shape", "", "",
        {{"Input", PinType::Tensor, true, "Input"}},
        {{"Output", PinType::Tensor, true, "Viewed"}},
        {{"shape", "string", "-1,256", "Target shape", {}, ""}},
        NodeImplementationStatus::Implemented, 0});

    RegisterNode({NodeType::Permute, NodeCategory::ShapeOps, "Permute", ICON_FA_SHUFFLE,
        {"permute", "transpose", "axes", "shape", "tensor"}, 0, false, "Reorder tensor dimensions", "", "",
        {{"Input", PinType::Tensor, true, "Input"}},
        {{"Output", PinType::Tensor, true, "Permuted"}},
        {{"dims", "string", "0,2,1", "Dimension order", {}, ""}},
        NodeImplementationStatus::Implemented, 0});

    RegisterNode({NodeType::Squeeze, NodeCategory::ShapeOps, "Squeeze", ICON_FA_COMPRESS,
        {"squeeze", "shape", "dimension", "tensor"}, 0, false, "Remove singleton tensor dimensions", "", "",
        {{"Input", PinType::Tensor, true, "Input"}},
        {{"Output", PinType::Tensor, true, "Squeezed"}},
        {{"dim", "int", "0", "Dimension", {}, ""}},
        NodeImplementationStatus::Implemented, 0});

    RegisterNode({NodeType::Unsqueeze, NodeCategory::ShapeOps, "Unsqueeze", ICON_FA_EXPAND,
        {"unsqueeze", "shape", "dimension", "tensor"}, 0, false, "Insert a singleton tensor dimension", "", "",
        {{"Input", PinType::Tensor, true, "Input"}},
        {{"Output", PinType::Tensor, true, "Unsqueezed"}},
        {{"dim", "int", "0", "Dimension", {}, ""}},
        NodeImplementationStatus::Implemented, 0});

    RegisterNode({NodeType::Split, NodeCategory::ShapeOps, "Split", ICON_FA_CODE_BRANCH,
        {"split", "chunk", "shape", "tensor"}, 0, false, "Split tensor along a dimension", "", "",
        {{"Input", PinType::Tensor, true, "Input"}},
        {{"Output 1", PinType::Tensor, true, "First split"}, {"Output 2", PinType::Tensor, true, "Second split"}},
        {{"split_size", "int", "2", "Split size", {}, ""}, {"dim", "int", "0", "Dimension", {}, ""}},
        NodeImplementationStatus::Template, 0});

    RegisterNode({NodeType::Concatenate, NodeCategory::MergeOps, "Concatenate", ICON_FA_CODE_BRANCH,
        {"concatenate", "concat", "cat", "merge", "tensor"}, 0, false, "Concatenate tensors along a dimension", "", "",
        {{"Input 1", PinType::Tensor, true, "First input"}, {"Input 2", PinType::Tensor, true, "Second input"}},
        {{"Output", PinType::Tensor, true, "Concatenated"}},
        {{"dim", "int", "1", "Dimension", {}, ""}},
        NodeImplementationStatus::Implemented, 0});

    RegisterNode({NodeType::Add, NodeCategory::MergeOps, "Add", ICON_FA_PLUS,
        {"add", "sum", "merge", "tensor"}, 0, false, "Add tensors elementwise", "", "",
        {{"Input 1", PinType::Tensor, true, "First input"}, {"Input 2", PinType::Tensor, true, "Second input"}},
        {{"Output", PinType::Tensor, true, "Sum"}},
        {}, NodeImplementationStatus::Implemented, 0});

    RegisterNode({NodeType::Multiply, NodeCategory::MergeOps, "Multiply", ICON_FA_XMARK,
        {"multiply", "mul", "product", "merge", "tensor"}, 0, false, "Multiply tensors elementwise", "", "",
        {{"Input 1", PinType::Tensor, true, "First input"}, {"Input 2", PinType::Tensor, true, "Second input"}},
        {{"Output", PinType::Tensor, true, "Product"}},
        {}, NodeImplementationStatus::Implemented, 0});

    RegisterNode({NodeType::Average, NodeCategory::MergeOps, "Average", ICON_FA_CALCULATOR,
        {"average", "mean", "merge", "tensor"}, 0, false, "Average tensors elementwise", "", "",
        {{"Input 1", PinType::Tensor, true, "First input"}, {"Input 2", PinType::Tensor, true, "Second input"}},
        {{"Output", PinType::Tensor, true, "Average"}},
        {}, NodeImplementationStatus::Implemented, 0});

    RegisterNode({NodeType::TensorSum, NodeCategory::Analytics, "Tensor Sum", ICON_FA_CALCULATOR,
        {"tensor", "sum", "reduce", "reduction"}, 0, false, "Sum tensor values", "", "",
        {{"Input", PinType::Tensor, true, "Input"}},
        {{"Output", PinType::Tensor, true, "Reduced tensor"}},
        {{"dim", "int", "-1", "Dimension, or -1 for all values", {}, ""},
         {"keepdim", "bool", "false", "Keep reduced dimension", {}, ""}},
        NodeImplementationStatus::Implemented, 0});

    RegisterNode({NodeType::TensorMean, NodeCategory::Analytics, "Tensor Mean", ICON_FA_CALCULATOR,
        {"tensor", "mean", "average", "reduce", "reduction"}, 0, false, "Average tensor values", "", "",
        {{"Input", PinType::Tensor, true, "Input"}},
        {{"Output", PinType::Tensor, true, "Reduced tensor"}},
        {{"dim", "int", "-1", "Dimension, or -1 for all values", {}, ""},
         {"keepdim", "bool", "false", "Keep reduced dimension", {}, ""}},
        NodeImplementationStatus::Implemented, 0});

    RegisterNode({NodeType::TensorMax, NodeCategory::Analytics, "Tensor Max", ICON_FA_CALCULATOR,
        {"tensor", "max", "maximum", "reduce", "reduction"}, 0, false, "Maximum tensor values", "", "",
        {{"Input", PinType::Tensor, true, "Input"}},
        {{"Output", PinType::Tensor, true, "Reduced tensor"}},
        {{"dim", "int", "-1", "Dimension, or -1 for all values", {}, ""},
         {"keepdim", "bool", "false", "Keep reduced dimension", {}, ""}},
        NodeImplementationStatus::Implemented, 0});

    RegisterNode({NodeType::TensorMin, NodeCategory::Analytics, "Tensor Min", ICON_FA_CALCULATOR,
        {"tensor", "min", "minimum", "reduce", "reduction"}, 0, false, "Minimum tensor values", "", "",
        {{"Input", PinType::Tensor, true, "Input"}},
        {{"Output", PinType::Tensor, true, "Reduced tensor"}},
        {{"dim", "int", "-1", "Dimension, or -1 for all values", {}, ""},
         {"keepdim", "bool", "false", "Keep reduced dimension", {}, ""}},
        NodeImplementationStatus::Implemented, 0});

    RegisterNode({NodeType::TensorProd, NodeCategory::Analytics, "Tensor Prod", ICON_FA_CALCULATOR,
        {"tensor", "prod", "product", "reduce", "reduction"}, 0, false, "Product of tensor values", "", "",
        {{"Input", PinType::Tensor, true, "Input"}},
        {{"Output", PinType::Tensor, true, "Reduced tensor"}},
        {{"dim", "int", "-1", "Dimension, or -1 for all values", {}, ""},
         {"keepdim", "bool", "false", "Keep reduced dimension", {}, ""}},
        NodeImplementationStatus::Implemented, 0});

    RegisterNode({NodeType::TensorVar, NodeCategory::Analytics, "Tensor Var", ICON_FA_CALCULATOR,
        {"tensor", "var", "variance", "reduce", "reduction"}, 0, false, "Variance of tensor values", "", "",
        {{"Input", PinType::Tensor, true, "Input"}},
        {{"Output", PinType::Tensor, true, "Reduced tensor"}},
        {{"dim", "int", "-1", "Dimension, or -1 for all values", {}, ""},
         {"keepdim", "bool", "false", "Keep reduced dimension", {}, ""}},
        NodeImplementationStatus::Implemented, 0});

    RegisterNode({NodeType::TensorStd, NodeCategory::Analytics, "Tensor Std", ICON_FA_CALCULATOR,
        {"tensor", "std", "standard deviation", "reduce", "reduction"}, 0, false, "Standard deviation of tensor values", "", "",
        {{"Input", PinType::Tensor, true, "Input"}},
        {{"Output", PinType::Tensor, true, "Reduced tensor"}},
        {{"dim", "int", "-1", "Dimension, or -1 for all values", {}, ""},
         {"keepdim", "bool", "false", "Keep reduced dimension", {}, ""}},
        NodeImplementationStatus::Implemented, 0});

    RegisterNode({NodeType::TensorBroadcastTo, NodeCategory::ShapeOps, "Tensor Broadcast To", ICON_FA_EXPAND,
        {"tensor", "broadcast", "shape", "expand"}, 0, false, "Broadcast tensor to a target shape", "", "",
        {{"Input", PinType::Tensor, true, "Input"}},
        {{"Output", PinType::Tensor, true, "Broadcast tensor"}},
        {{"shape", "string", "", "Target shape", {}, ""}},
        NodeImplementationStatus::Implemented, 0});

    RegisterNode({NodeType::TensorExpand, NodeCategory::ShapeOps, "Tensor Expand", ICON_FA_EXPAND,
        {"tensor", "expand", "broadcast", "shape"}, 0, false, "Materialize tensor expanded to a target shape", "", "",
        {{"Input", PinType::Tensor, true, "Input"}},
        {{"Output", PinType::Tensor, true, "Expanded tensor"}},
        {{"shape", "string", "", "Target shape", {}, ""}},
        NodeImplementationStatus::Implemented, 0});

    RegisterNode({NodeType::TensorIndexSelect, NodeCategory::ShapeOps, "Tensor Index Select", ICON_FA_LIST_CHECK,
        {"tensor", "index", "select", "gather", "slice"}, 0, false, "Select entries along one dimension by index list", "", "",
        {{"Input", PinType::Tensor, true, "Input"}},
        {{"Output", PinType::Tensor, true, "Selected tensor"}},
        {{"dim", "int", "0", "Dimension", {}, ""},
         {"indices", "string", "", "Comma-separated indices", {}, ""}},
        NodeImplementationStatus::Implemented, 0});

    RegisterNode({NodeType::TensorPow, NodeCategory::Analytics, "Tensor Pow", ICON_FA_CALCULATOR,
        {"tensor", "pow", "power", "elementwise"}, 0, false, "Raise tensor values to a scalar power", "", "",
        {{"Input", PinType::Tensor, true, "Input"}},
        {{"Output", PinType::Tensor, true, "Output"}},
        {{"exponent", "float", "2.0", "Scalar exponent", {}, ""}},
        NodeImplementationStatus::Implemented, 0});

    RegisterNode({NodeType::TensorSqrt, NodeCategory::Analytics, "Tensor Sqrt", ICON_FA_CALCULATOR,
        {"tensor", "sqrt", "square root", "elementwise"}, 0, false, "Elementwise square root", "", "",
        {{"Input", PinType::Tensor, true, "Input"}},
        {{"Output", PinType::Tensor, true, "Output"}},
        {}, NodeImplementationStatus::Implemented, 0});

    RegisterNode({NodeType::TensorExp, NodeCategory::Analytics, "Tensor Exp", ICON_FA_CALCULATOR,
        {"tensor", "exp", "exponential", "elementwise"}, 0, false, "Elementwise exponential", "", "",
        {{"Input", PinType::Tensor, true, "Input"}},
        {{"Output", PinType::Tensor, true, "Output"}},
        {}, NodeImplementationStatus::Implemented, 0});

    RegisterNode({NodeType::TensorLog, NodeCategory::Analytics, "Tensor Log", ICON_FA_CALCULATOR,
        {"tensor", "log", "natural log", "elementwise"}, 0, false, "Elementwise natural log", "", "",
        {{"Input", PinType::Tensor, true, "Input"}},
        {{"Output", PinType::Tensor, true, "Output"}},
        {}, NodeImplementationStatus::Implemented, 0});

    RegisterNode({NodeType::TensorAbs, NodeCategory::Analytics, "Tensor Abs", ICON_FA_CALCULATOR,
        {"tensor", "abs", "absolute", "elementwise"}, 0, false, "Elementwise absolute value", "", "",
        {{"Input", PinType::Tensor, true, "Input"}},
        {{"Output", PinType::Tensor, true, "Output"}},
        {}, NodeImplementationStatus::Implemented, 0});

    RegisterNode({NodeType::TensorSign, NodeCategory::Analytics, "Tensor Sign", ICON_FA_CALCULATOR,
        {"tensor", "sign", "elementwise"}, 0, false, "Elementwise sign", "", "",
        {{"Input", PinType::Tensor, true, "Input"}},
        {{"Output", PinType::Tensor, true, "Output"}},
        {}, NodeImplementationStatus::Implemented, 0});

    RegisterNode({NodeType::TensorClip, NodeCategory::Analytics, "Tensor Clip", ICON_FA_CALCULATOR,
        {"tensor", "clip", "clamp", "elementwise"}, 0, false, "Clip tensor values to a scalar range", "", "",
        {{"Input", PinType::Tensor, true, "Input"}},
        {{"Output", PinType::Tensor, true, "Output"}},
        {{"min", "float", "0.0", "Minimum value", {}, ""},
         {"max", "float", "1.0", "Maximum value", {}, ""}},
        NodeImplementationStatus::Implemented, 0});

    RegisterNode({NodeType::TensorDot, NodeCategory::Analytics, "Tensor Dot", ICON_FA_CALCULATOR,
        {"tensor", "dot", "vector", "linalg"}, 0, false, "Compute vector or row-wise batch dot product", "", "",
        {{"A", PinType::Tensor, true, "Left input"}, {"B", PinType::Tensor, true, "Right input"}},
        {{"Output", PinType::Tensor, true, "Dot product"}},
        {}, NodeImplementationStatus::Implemented, 0});

    RegisterNode({NodeType::TensorBatchMatMul, NodeCategory::Analytics, "Tensor Batch MatMul", ICON_FA_CALCULATOR,
        {"tensor", "batch", "matmul", "matrix", "linalg"}, 0, false, "Compute batched matrix multiplication", "", "",
        {{"A", PinType::Tensor, true, "Left input"}, {"B", PinType::Tensor, true, "Right input"}},
        {{"Output", PinType::Tensor, true, "Batched product"}},
        {}, NodeImplementationStatus::Template, 0});

    RegisterNode({NodeType::TensorCompare, NodeCategory::Analytics, "Tensor Compare", ICON_FA_CALCULATOR,
        {"tensor", "compare", "greater", "less", "equal", "mask"}, 0, false, "Compare tensors or tensor and scalar", "", "",
        {{"A", PinType::Tensor, true, "Input tensor"},
         {"B", PinType::Tensor, false, "Optional tensor rhs; when connected, scalar is ignored"}},
        {{"Mask", PinType::Tensor, true, "Comparison mask"}},
        {{"op", "enum", ">", "Comparison operator", {">", ">=", "<", "<=", "==", "!="}, ""},
         {"scalar", "float", "0.0", "Scalar comparison value", {}, ""}},
        NodeImplementationStatus::Implemented, 0});

    RegisterNode({NodeType::TensorLogicalMask, NodeCategory::Analytics, "Tensor Logical Mask", ICON_FA_CALCULATOR,
        {"tensor", "logical", "mask", "not", "and", "or"}, 0, false, "Combine or invert tensor masks", "", "",
        {{"A", PinType::Tensor, true, "Input mask"},
         {"B", PinType::Tensor, false, "Optional rhs mask for and/or"}},
        {{"Mask", PinType::Tensor, true, "Logical mask"}},
        {{"op", "enum", "not", "Logical operator", {"not", "and", "or"}, ""}},
        NodeImplementationStatus::Implemented, 0});

    RegisterNode({NodeType::Embedding, NodeCategory::Layers, "Embedding", ICON_FA_CUBES,
        {"embedding", "lookup"}, 0, false, "Embedding layer", "", "",
        {{"Indices", PinType::Labels, true, "Token indices"}},
        {{"Embeddings", PinType::Tensor, true, "Vectors"}},
        {{"num_embeddings", "int", "10000", "Vocab size", {}, ""}, {"embedding_dim", "int", "256", "Dim", {}, ""}},
        NodeImplementationStatus::Implemented, 0});
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
        {{"dim", "int", "-1", "Dimension", {}, ""}},
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
        {{"negative_slope", "float", "0.01", "Slope", {}, ""}},
        NodeImplementationStatus::Implemented, 0});
}

// =============================================================================
// Training Nodes
// =============================================================================
void NodeMetadataRegistry::InitializeTrainingNodes() {
    RegisterNode({NodeType::MSELoss, NodeCategory::Training, "MSE Loss", ICON_FA_CHART_LINE,
        {"mse", "loss", "regression"}, 0, false, "Mean squared error", "", "",
        {{"Predictions", PinType::Tensor, true, "Predictions"}, {"Targets", PinType::Tensor, true, "Targets"}},
        {{"Loss", PinType::Loss, true, "Loss value"}},
        {}, NodeImplementationStatus::Implemented, 0});

    RegisterNode({NodeType::CrossEntropyLoss, NodeCategory::Training, "CrossEntropy", ICON_FA_CHART_PIE,
        {"crossentropy", "classification"}, 0, false, "Cross-entropy loss", "", "",
        {{"Logits", PinType::Tensor, true, "Logits"}, {"Labels", PinType::Labels, true, "Labels"}},
        {{"Loss", PinType::Loss, true, "Loss value"}},
        {}, NodeImplementationStatus::Implemented, 0});

    RegisterNode({NodeType::Adam, NodeCategory::Training, "Adam", ICON_FA_GRADUATION_CAP,
        {"adam", "optimizer"}, 0, false, "Adam optimizer", "", "",
        {{"Loss", PinType::Loss, true, "Loss"}, {"Parameters", PinType::Parameters, true, "Params"}},
        {{"Optimizer", PinType::Optimizer, true, "Optimizer"}},
        {{"lr", "float", "0.001", "Learning rate", {}, ""}},
        NodeImplementationStatus::Implemented, 0});

    RegisterNode({NodeType::SGD, NodeCategory::Training, "SGD", ICON_FA_GRADUATION_CAP,
        {"sgd", "optimizer"}, 0, false, "SGD optimizer", "", "",
        {{"Loss", PinType::Loss, true, "Loss"}, {"Parameters", PinType::Parameters, true, "Params"}},
        {{"Optimizer", PinType::Optimizer, true, "Optimizer"}},
        {{"lr", "float", "0.01", "Learning rate", {}, ""}, {"momentum", "float", "0.9", "Momentum", {}, ""}},
        NodeImplementationStatus::Implemented, 0});

    RegisterNode({NodeType::StepLR, NodeCategory::Training, "Step LR", ICON_FA_GRADUATION_CAP,
        {"step", "scheduler"}, 0, false, "Step learning-rate scheduler", "", "",
        {{"Optimizer", PinType::Optimizer, true, "Optimizer"}},
        {{"Optimizer", PinType::Optimizer, true, "Scheduled"}},
        {{"step_size", "int", "10", "Step size", {}, ""},
         {"gamma", "float", "0.1", "Decay factor", {}, ""}},
        NodeImplementationStatus::Template, 0, "Blocked"});

    RegisterNode({NodeType::CosineAnnealing, NodeCategory::Training, "Cosine LR", ICON_FA_WAVE_SINE,
        {"cosine", "scheduler"}, 0, false, "Cosine LR scheduler", "", "",
        {{"Optimizer", PinType::Optimizer, true, "Optimizer"}},
        {{"Optimizer", PinType::Optimizer, true, "Scheduled"}},
        {{"T_max", "int", "100", "Max iterations", {}, ""}},
        NodeImplementationStatus::Template, 0});

    RegisterNode({NodeType::ReduceOnPlateau, NodeCategory::Training, "Reduce LR", ICON_FA_GRADUATION_CAP,
        {"plateau", "scheduler"}, 0, false, "Reduce learning rate on plateau", "", "",
        {{"Optimizer", PinType::Optimizer, true, "Optimizer"}},
        {{"Optimizer", PinType::Optimizer, true, "Scheduled"}},
        {{"patience", "int", "10", "Patience", {}, ""},
         {"factor", "float", "0.1", "Reduction factor", {}, ""}},
        NodeImplementationStatus::Template, 0, "Blocked"});

    RegisterNode({NodeType::ExponentialLR, NodeCategory::Training, "Exponential LR", ICON_FA_GRADUATION_CAP,
        {"exponential", "scheduler"}, 0, false, "Exponential learning-rate scheduler", "", "",
        {{"Optimizer", PinType::Optimizer, true, "Optimizer"}},
        {{"Optimizer", PinType::Optimizer, true, "Scheduled"}},
        {{"gamma", "float", "0.95", "Decay factor", {}, ""}},
        NodeImplementationStatus::Template, 0, "Blocked"});

    RegisterNode({NodeType::WarmupScheduler, NodeCategory::Training, "Warmup LR", ICON_FA_GRADUATION_CAP,
        {"warmup", "scheduler"}, 0, false, "Warmup learning-rate scheduler", "", "",
        {{"Optimizer", PinType::Optimizer, true, "Optimizer"}},
        {{"Optimizer", PinType::Optimizer, true, "Scheduled"}},
        {{"warmup_steps", "int", "1000", "Warmup steps", {}, ""}},
        NodeImplementationStatus::Template, 0, "Blocked"});

    RegisterNode({NodeType::L1Regularization, NodeCategory::Regularization, "L1 Regularization", ICON_FA_GRADUATION_CAP,
        {"l1", "regularization"}, 0, false, "L1 regularization", "", "",
        {{"Loss", PinType::Loss, true, "Loss"}},
        {{"Loss", PinType::Loss, true, "Regularized loss"}},
        {{"lambda", "float", "0.0001", "Penalty strength", {}, ""}},
        NodeImplementationStatus::Template, 0, "Blocked"});

    RegisterNode({NodeType::L2Regularization, NodeCategory::Regularization, "L2 Regularization", ICON_FA_GRADUATION_CAP,
        {"l2", "regularization"}, 0, false, "L2 regularization", "", "",
        {{"Loss", PinType::Loss, true, "Loss"}},
        {{"Loss", PinType::Loss, true, "Regularized loss"}},
        {{"lambda", "float", "0.0001", "Penalty strength", {}, ""}},
        NodeImplementationStatus::Template, 0, "Blocked"});

    RegisterNode({NodeType::ElasticNet, NodeCategory::Regularization, "Elastic Net", ICON_FA_GRADUATION_CAP,
        {"elasticnet", "regularization"}, 0, false, "Elastic-net regularization", "", "",
        {{"Loss", PinType::Loss, true, "Loss"}},
        {{"Loss", PinType::Loss, true, "Regularized loss"}},
        {{"l1_lambda", "float", "0.0001", "L1 penalty", {}, ""},
         {"l2_lambda", "float", "0.0001", "L2 penalty", {}, ""}},
        NodeImplementationStatus::Template, 0, "Blocked"});

    RegisterNode({NodeType::Output, NodeCategory::Training, "Output", ICON_FA_ARROW_RIGHT,
        {"output", "final"}, 0, false, "Model output", "", "",
        {{"Input", PinType::Tensor, true, "Final output"}}, {},
        {}, NodeImplementationStatus::Implemented, 0});
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
    RegisterNode({NodeType::TextTokenizer, NodeCategory::TextProcessing, "Tokenizer", ICON_FA_ALIGN_LEFT,
        {"tokenize", "text", "nlp"}, 0, false, "Tokenize text", "", "",
        {{"Text", PinType::Dataset, true, "Text"}},
        {{"Tokens", PinType::Tensor, true, "Token indices"}},
        {{"mode", "enum", "word", "Mode", {"word", "char"}, ""}},
        NodeImplementationStatus::Implemented, 0});

    RegisterNode({NodeType::TextVocabulary, NodeCategory::TextProcessing, "Vocabulary", ICON_FA_LIST_UL,
        {"vocabulary", "vocab"}, 0, false, "Build vocabulary", "", "",
        {{"Text", PinType::Dataset, true, "Text"}},
        {{"Vocab", PinType::Parameters, true, "Vocabulary"}},
        {{"max_words", "int", "10000", "Max words", {}, ""}},
        NodeImplementationStatus::Implemented, 0});

    RegisterNode({NodeType::TextPadding, NodeCategory::TextProcessing, "Padding", ICON_FA_ARROWS_LEFT_RIGHT,
        {"pad", "sequence"}, 0, false, "Pad sequences", "", "",
        {{"Tokens", PinType::Tensor, true, "Tokens"}},
        {{"Padded", PinType::Tensor, true, "Padded"}},
        {{"max_length", "int", "128", "Max length", {}, ""}},
        NodeImplementationStatus::Implemented, 0});
}

// =============================================================================
// Time Series Nodes
// =============================================================================
void NodeMetadataRegistry::InitializeTimeSeriesNodes() {
    RegisterNode({NodeType::TimeSeriesWindow, NodeCategory::TimeSeries, "Sliding Window", ICON_FA_CHART_LINE,
        {"window", "sliding", "sequence"}, 0, false, "Create sliding windows", "", "",
        {{"Data", PinType::Dataset, true, "Input time-ordered table"}},
        {{"Windowed", PinType::Dataset, true, "Windowed table with x_* feature columns and y label"}},
        {{"value_col", "string", "", "Numeric source/target column", {}, ""},
         {"feature_cols", "string", "", "Extra numeric feature columns to window", {}, ""},
         {"time_col", "string", "", "Optional numeric time column", {}, ""},
         {"input_width", "int", "12", "Lookback steps per sample", {}, ""},
         {"label_width", "int", "1", "Forecast steps (v1 requires 1)", {}, ""},
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

    RegisterNode({NodeType::TimeSeriesSplit, NodeCategory::TimeSeries, "TS Split", ICON_FA_SCISSORS,
        {"split", "train", "test"}, 0, false, "Chronological split", "", "",
        {{"Data", PinType::Dataset, true, "Input time-ordered table"}},
        {{"Partitioned", PinType::Dataset, true, "Input table plus __partition__ split column"}},
        {{"train_ratio", "float", "0.8", "Train ratio", {}, ""},
         {"val_ratio", "float", "0.1", "Validation ratio", {}, ""},
         {"test_ratio", "float", "0.1", "Test ratio", {}, ""}},
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
        {"audio", "wav", "load"}, 0, false, "Load audio file", "", "",
        {}, {{"Waveform", PinType::Tensor, true, "Waveform"}, {"SampleRate", PinType::Parameters, true, "SR"}},
        {{"file_path", "file", "", "Audio file", {}, "*.wav"}},
        NodeImplementationStatus::Implemented, 0});

    RegisterNode({NodeType::Spectrogram, NodeCategory::Audio, "Spectrogram", ICON_FA_CHART_AREA,
        {"spectrogram", "stft"}, 0, false, "Compute spectrogram", "", "",
        {{"Waveform", PinType::Tensor, true, "Waveform"}},
        {{"Spectrogram", PinType::Tensor, true, "Spectrogram"}},
        {{"n_fft", "int", "2048", "FFT size", {}, ""}},
        NodeImplementationStatus::Implemented, 0});

    RegisterNode({NodeType::MelSpectrogram, NodeCategory::Audio, "Mel Spectrogram", ICON_FA_CHART_AREA,
        {"mel", "spectrogram"}, 0, false, "Mel spectrogram", "", "",
        {{"Waveform", PinType::Tensor, true, "Waveform"}},
        {{"MelSpec", PinType::Tensor, true, "Mel spec"}},
        {{"n_mels", "int", "128", "Mel bands", {}, ""}},
        NodeImplementationStatus::Implemented, 0});

    RegisterNode({NodeType::MFCC, NodeCategory::Audio, "MFCC", ICON_FA_WAVE_SQUARE,
        {"mfcc", "cepstral"}, 0, false, "Extract MFCCs", "", "",
        {{"Waveform", PinType::Tensor, true, "Waveform"}},
        {{"MFCC", PinType::Tensor, true, "MFCCs"}},
        {{"n_mfcc", "int", "13", "Num MFCCs", {}, ""}},
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
        {"export", "csv", "save"}, 0, false, "Export to CSV", "", "",
        {{"Table", PinType::Dataset, true, "Table"}}, {},
        {{"file_path", "file", "", "Output file", {}, "*.csv"}},
        NodeImplementationStatus::Implemented, 0});

    RegisterNode({NodeType::ExportParquet, NodeCategory::DataSources, "Export Parquet", ICON_FA_FILE_EXPORT,
        {"export", "parquet"}, 0, false, "Export to Parquet", "", "",
        {{"Table", PinType::Dataset, true, "Table"}}, {},
        {{"file_path", "file", "", "Output file", {}, "*.parquet"}},
        NodeImplementationStatus::Implemented, 0});

    RegisterNode({NodeType::ExportJSON, NodeCategory::DataSources, "Export JSON", ICON_FA_FILE_EXPORT,
        {"export", "json"}, 0, false, "Export to JSON", "", "",
        {{"Table", PinType::Dataset, true, "Table"}}, {},
        {{"file_path", "file", "", "Output file", {}, "*.json"}},
        NodeImplementationStatus::Template, 0, "Blocked"});

    RegisterNode({NodeType::ExportSQL, NodeCategory::DataSources, "Export SQL", ICON_FA_DATABASE,
        {"export", "sql", "database"}, 0, false, "Export to SQL DB", "", "",
        {{"Table", PinType::Dataset, true, "Table"}}, {},
        {{"connection", "string", "", "Connection string", {}, ""}},
        NodeImplementationStatus::Implemented, 0});

    RegisterNode({NodeType::ExportExcel, NodeCategory::DataSources, "Excel Writer", ICON_FA_FILE_EXCEL,
        {"export", "excel", "xlsx", "spreadsheet"}, 0, false, "Export to Excel (.xlsx)", "", "",
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
        NodeImplementationStatus::Template, 0, "Blocked"});

    RegisterNode({NodeType::CellUpdater, NodeCategory::DataTransform, "Cell Updater", ICON_FA_PEN,
        {"cell", "update", "modify"}, 0, false,
        "Update value in specific cell", "", "",
        {{"Table", PinType::Dataset, true, "Table"},
         {"Value", PinType::Tensor, false, "New value (optional)"}},
        {{"Table", PinType::Dataset, true, "Updated table"}},
        {{"row", "int", "0", "Row index", {}, ""},
         {"column", "string", "", "Column name", {}, ""},
         {"value", "string", "", "New value (if not using input)", {}, ""}},
        NodeImplementationStatus::Template, 0, "Blocked"});

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
        NodeImplementationStatus::Template, 0, "Blocked"});

    RegisterNode({NodeType::RowAppender, NodeCategory::DataTransform, "Row Appender", ICON_FA_LAYER_GROUP,
        {"append", "rows", "vertical", "union"}, 0, false,
        "Append rows from multiple tables (UNION)", "", "",
        {{"Top", PinType::Dataset, true, "Top table"},
         {"Bottom", PinType::Dataset, true, "Bottom table"}},
        {{"Table", PinType::Dataset, true, "Combined table"}},
        {{"match_columns", "bool", "true", "Match columns by name", {}, ""}},
        NodeImplementationStatus::Template, 0, "Blocked"});

    RegisterNode({NodeType::Unpivot, NodeCategory::DataTransform, "Unpivot", ICON_FA_ROTATE,
        {"unpivot", "melt", "wide to long"}, 0, false,
        "Unpivot wide to long format", "", "",
        {{"Table", PinType::Dataset, true, "Wide table"}},
        {{"Table", PinType::Dataset, true, "Long table"}},
        {{"id_columns", "string", "", "ID columns (comma-separated)", {}, ""},
         {"value_name", "string", "value", "Name for value column", {}, ""},
         {"variable_name", "string", "variable", "Name for variable column", {}, ""}},
        NodeImplementationStatus::Template, 0, "Blocked"});

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
        NodeImplementationStatus::Template, 0, "Blocked"});
}

// =============================================================================
// Utility Nodes
// =============================================================================
void NodeMetadataRegistry::InitializeUtilityNodes() {
    RegisterNode({NodeType::Lambda, NodeCategory::Utility, "Lambda", ICON_FA_CODE,
        {"lambda", "custom", "function"}, 0, false, "Custom function", "", "",
        {{"Input", PinType::Tensor, true, "Input"}},
        {{"Output", PinType::Tensor, true, "Output"}},
        {{"expression", "string", "x", "Python expression", {}, ""}},
        NodeImplementationStatus::Implemented, 0});

    RegisterNode({NodeType::Identity, NodeCategory::Utility, "Identity", ICON_FA_EQUALS,
        {"identity", "passthrough"}, 0, false, "Pass through unchanged", "", "",
        {{"Input", PinType::Tensor, true, "Input"}},
        {{"Output", PinType::Tensor, true, "Output"}},
        {}, NodeImplementationStatus::Implemented, 0});

    RegisterNode({NodeType::Constant, NodeCategory::Utility, "Constant", ICON_FA_HASHTAG,
        {"constant", "value"}, 0, false, "Output constant", "", "",
        {}, {{"Value", PinType::Tensor, true, "Value"}},
        {{"value", "float", "1.0", "Value", {}, ""}},
        NodeImplementationStatus::Implemented, 0});

    RegisterNode({NodeType::SignalSlider, NodeCategory::Signal, "Slider", ICON_FA_SLIDERS,
        {"slider", "control", "input"}, 0, false, "Interactive slider", "", "",
        {}, {{"Value", PinType::Tensor, true, "Value"}},
        {{"min", "float", "0.0", "Min", {}, ""}, {"max", "float", "1.0", "Max", {}, ""}},
        NodeImplementationStatus::Implemented, 0});

    RegisterNode({NodeType::SignalScope, NodeCategory::Signal, "Scope", ICON_FA_CHART_LINE,
        {"scope", "plot", "monitor"}, 0, false, "Signal visualizer", "", "",
        {{"Signal", PinType::Tensor, true, "Signal"}}, {},
        {{"buffer_size", "int", "1000", "Buffer size", {}, ""}},
        NodeImplementationStatus::Implemented, 0});

    // ===== Signal Processing Nodes (Phase 4) =====
    RegisterNode({NodeType::FFTNode, NodeCategory::Signal, "FFT", ICON_FA_WAVE_SQUARE,
        {"fft", "fourier", "frequency"}, 0, false, "Fast Fourier Transform",
        "Computes frequency, magnitude, and phase columns from one numeric signal column.", "",
        {{"Data", PinType::Dataset, true, "Input table"}},
        {{"Spectrum", PinType::Dataset, true, "Frequency-domain table"}},
        {{"signal_col", "string", "", "Numeric signal column", {}, ""},
         {"sample_rate", "float", "1.0", "Sample rate", {}, ""}},
        NodeImplementationStatus::Implemented, 0});

    RegisterNode({NodeType::IFFTNode, NodeCategory::Signal, "IFFT", ICON_FA_WAVE_SQUARE,
        {"ifft", "inverse", "fourier"}, 0, false, "Inverse FFT",
        "Convert frequency domain back to time domain.", "",
        {{"Spectrum", PinType::Tensor, true, "Frequency spectrum"}},
        {{"Signal", PinType::Tensor, true, "Time-domain signal"}},
        {}, NodeImplementationStatus::Template, 0, "Blocked"});

    RegisterNode({NodeType::FilterDesigner, NodeCategory::Signal, "Filter Designer", ICON_FA_FILTER,
        {"filter", "fir", "iir", "lowpass"}, 0, false, "Design digital filters",
        "Designs and applies a filter to one numeric signal column.", "",
        {{"Data", PinType::Dataset, true, "Input table"}},
        {{"Filtered", PinType::Dataset, true, "Input table with filtered column"}},
        {{"signal_col", "string", "", "Numeric signal column", {}, ""},
         {"filter_type", "dropdown", "lowpass", "Filter type", {"lowpass", "highpass", "bandpass", "bandstop"}, ""},
         {"cutoff", "float", "0.5", "Cutoff frequency", {}, ""},
         {"cutoff_high", "float", "0", "Upper cutoff for band filters", {}, ""},
         {"sample_rate", "float", "1.0", "Sample rate", {}, ""},
         {"order", "int", "4", "Filter order", {}, ""}},
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
        "Convert text to TF-IDF weighted feature vectors.", "",
        {{"Text", PinType::Dataset, true, "Text data (column of strings)"}},
        {{"Vectors", PinType::Dataset, true, "TF-IDF vectors"}, {"Vocabulary", PinType::Dataset, true, "Feature names"}},
        {{"max_features", "int", "10000", "Max vocabulary size", {}, ""},
         {"ngram_min", "int", "1", "Min n-gram", {}, ""},
         {"ngram_max", "int", "1", "Max n-gram", {}, ""},
         {"min_df", "int", "1", "Min document frequency", {}, ""}},
        NodeImplementationStatus::Implemented, 0});

    RegisterNode({NodeType::CountVectorizer, NodeCategory::TextProcessing, "Count Vectorizer", ICON_FA_LIST_OL,
        {"count", "bow", "bag", "words"}, 0, false, "Bag-of-words vectorization",
        "Convert text to token count vectors.", "",
        {{"Text", PinType::Dataset, true, "Text data"}},
        {{"Vectors", PinType::Dataset, true, "Count vectors"}, {"Vocabulary", PinType::Dataset, true, "Feature names"}},
        {{"max_features", "int", "10000", "Max vocabulary size", {}, ""},
         {"binary", "bool", "false", "Binary counts", {}, ""}},
        NodeImplementationStatus::Implemented, 0});

    RegisterNode({NodeType::SentimentAnalyzer, NodeCategory::TextProcessing, "Sentiment Analysis", ICON_FA_FACE_SMILE,
        {"sentiment", "opinion", "polarity"}, 0, false, "Sentiment analysis",
        "Classify text sentiment (positive, negative, neutral).", "",
        {{"Text", PinType::Dataset, true, "Text data"}},
        {{"Sentiment", PinType::Labels, true, "Sentiment labels"}, {"Scores", PinType::Dataset, true, "Sentiment scores"}},
        {{"model", "enum", "vader", "Sentiment model", {"vader", "textblob", "transformer"}, ""}},
        NodeImplementationStatus::Implemented, 0});

    // ===== Utility Tools (Phase 4) =====
    RegisterNode({NodeType::CalculatorNode, NodeCategory::Utility, "Calculator", ICON_FA_CALCULATOR,
        {"calculator", "math", "compute"}, 0, false, "Math expression calculator",
        "Evaluate mathematical expressions with variables.", "",
        {{"Variables", PinType::Dataset, false, "Input variables (optional)"}},
        {{"Result", PinType::Dataset, true, "Computed result"}},
        {{"expression", "string", "2 + 2", "Math expression", {}, ""},
         {"precision", "int", "6", "Decimal precision", {}, ""}},
        NodeImplementationStatus::Template, 0, "Blocked"});

    RegisterNode({NodeType::UnitConverter, NodeCategory::Utility, "Unit Converter", ICON_FA_SCALE_BALANCED,
        {"unit", "convert", "conversion"}, 0, false, "Unit conversion utility",
        "Convert between different units of measurement.", "",
        {{"Value", PinType::Dataset, true, "Input value"}},
        {{"Converted", PinType::Dataset, true, "Converted value"}},
        {{"category", "enum", "length", "Unit category", {"length", "mass", "temperature", "time", "area", "volume"}, ""},
         {"from_unit", "string", "m", "From unit", {}, ""},
         {"to_unit", "string", "ft", "To unit", {}, ""}},
        NodeImplementationStatus::Template, 0, "Blocked"});

    RegisterNode({NodeType::RegexTester, NodeCategory::Utility, "Regex Tester", ICON_FA_CODE,
        {"regex", "regular", "expression", "pattern"}, 0, false, "Regular expression tester",
        "Test and apply regex patterns to text.", "",
        {{"Text", PinType::Dataset, true, "Input text"}},
        {{"Matches", PinType::Dataset, true, "Match results"}, {"Groups", PinType::Dataset, true, "Capture groups"}},
        {{"pattern", "string", ".*", "Regex pattern", {}, ""},
         {"flags", "string", "", "Flags (i=ignorecase, m=multiline)", {}, ""}},
        NodeImplementationStatus::Template, 0, "Blocked"});

    RegisterNode({NodeType::JSONPathExtractor, NodeCategory::Utility, "JSONPath", ICON_FA_CODE_BRANCH,
        {"json", "jsonpath", "extract"}, 0, false, "Extract data using JSONPath",
        "Query JSON data using JSONPath expressions.", "",
        {{"JSON", PinType::Dataset, true, "JSON data"}},
        {{"Result", PinType::Dataset, true, "Extracted values"}},
        {{"path", "string", "$.data[*].value", "JSONPath expression", {}, ""}},
        NodeImplementationStatus::Template, 0, "Blocked"});
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
        {{"test", "dropdown", "ttest", "Test Type", {"ttest_1samp", "ttest_ind", "ttest_paired", "anova", "chi_square", "mann_whitney"}, ""}},
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
        "Gaussian Mixture Model clustering", "", "",
        {{"Data", PinType::Dataset, true, "Data"}},
        {{"Clustered", PinType::Dataset, true, "Input table plus cluster_id"}},
        {{"feature_cols", "string", "", "Feature columns (empty = numeric auto-detect)", {}, ""},
         {"label_col", "string", "", "Label column to exclude", {}, ""},
         {"n_components", "int", "3", "Components", {}, ""},
         {"covariance_type", "dropdown", "full", "Covariance type", {"full", "tied", "diag", "spherical"}, ""},
         {"max_iter", "int", "100", "Max iterations", {}, ""},
         {"tol", "float", "0.001", "Convergence tolerance", {}, ""},
         {"n_init", "int", "1", "Random restarts", {}, ""},
         {"seed", "int", "0", "Random seed (0 = non-deterministic)", {}, ""}},
        NodeImplementationStatus::Implemented, 0});

    RegisterNode({NodeType::Convolution1D, NodeCategory::Signal, "Convolution 1D", ICON_FA_WAVE_SQUARE,
        {"convolution", "1d", "signal", "filter", "kernel"}, 0, false,
        "1D signal convolution", "", "",
        {{"Data", PinType::Dataset, true, "Input table"}},
        {{"Convolved", PinType::Dataset, true, "Input table plus convolved signal"}},
        {{"signal_col", "string", "", "Numeric signal column", {}, ""},
         {"kernel", "string", "0.25,0.5,0.25", "Comma-separated FIR kernel taps", {}, ""},
         {"mode", "dropdown", "same", "Mode (same only in v1)", {"same"}, ""}},
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
