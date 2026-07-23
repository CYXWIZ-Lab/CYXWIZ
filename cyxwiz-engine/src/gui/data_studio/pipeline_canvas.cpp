#include "pipeline_canvas.h"
#include "../../core/pipeline_executor.h"  // Unified Canvas Phase 2: Moved to core/
#include "../../core/pipeline_execution_task.h"
#include "../../core/async_task_manager.h"
#include "gui/icons.h"
#include "core/file_dialogs.h"
#include <spdlog/spdlog.h>
#include <nlohmann/json.hpp>
#include <fstream>
#include <chrono>
#include <iomanip>

namespace cyxwiz {

namespace {

void MigrateLegacyTimeSeriesCanvasParams(
    const std::string& type,
    std::map<std::string, std::string>& parameters) {
    if (type == "TSWindow") {
        if (parameters.count("value_col") == 0) {
            auto target = parameters.find("target_column");
            if (target != parameters.end()) {
                parameters["value_col"] = target->second;
            }
        }
        if (parameters.count("input_width") == 0) {
            auto window = parameters.find("window_size");
            if (window != parameters.end()) {
                parameters["input_width"] = window->second;
            }
        }
        if (parameters.count("shift") == 0) {
            parameters["shift"] = "1";
        }

        parameters.erase("target_column");
        parameters.erase("window_size");
        parameters.erase("stride");
    } else if (type == "TSFeatures") {
        if (parameters.count("value_col") == 0) {
            auto columns = parameters.find("columns");
            if (columns != parameters.end()) {
                parameters["value_col"] = columns->second;
            }
        }
        if (parameters.count("lag_values") == 0) {
            auto lag = parameters.find("lag_features");
            if (lag != parameters.end()) {
                parameters["lag_values"] = lag->second;
            }
        }
        if (parameters.count("rolling_windows") == 0) {
            auto window = parameters.find("rolling_window");
            if (window != parameters.end()) {
                parameters["rolling_windows"] = window->second;
            }
        }
        if (parameters.count("rolling_aggregations") == 0) {
            auto rolling = parameters.find("rolling_features");
            if (rolling != parameters.end()) {
                parameters["rolling_aggregations"] = rolling->second;
            }
        }

        parameters.erase("columns");
        parameters.erase("rolling_window");
        parameters.erase("lag_features");
        parameters.erase("rolling_features");
    }
}

} // namespace

PipelineCanvas::PipelineCanvas()
    : context_(nullptr)
    , next_node_id_(1)
    , next_link_id_(1)
    , show_node_palette_(false)
    , selected_node_id_(-1)
    , deployment_requested_(false)
{
    // Create separate ImNodes context for Data Studio
    // (separate from ML Node Editor context)
    context_ = ImNodes::CreateContext();
    ImNodes::SetCurrentContext(context_);

    // Configure ImNodes style for Data Studio
    ImNodesStyle& style = ImNodes::GetStyle();
    style.GridSpacing = 32.0f;
    style.NodeCornerRounding = 4.0f;
    style.NodePadding = ImVec2(8.0f, 8.0f);
    style.NodeBorderThickness = 1.0f;
    style.LinkThickness = 3.0f;
    style.LinkLineSegmentsPerLength = 0.1f;
    style.PinCircleRadius = 4.0f;
    style.PinQuadSideLength = 7.0f;
    style.PinTriangleSideLength = 9.5f;

    // Create pipeline executor
    executor_ = std::make_shared<PipelineExecutor>();

    spdlog::info("[Data Studio] PipelineCanvas initialized");
}

PipelineCanvas::~PipelineCanvas() {
    // Skip context cleanup - ImNodes contexts are cleaned up automatically
    // when ImGui shuts down. Explicit cleanup can cause crashes during
    // application shutdown if ImGui is already destroyed.
}

bool PipelineCanvas::RenderQuickAddItem(const QuickAddNode& item) {
    if (!ImGui::Selectable(item.label)) {
        return false;
    }
    AddNode(item.type_id, ImGui::GetMousePos());
    show_node_palette_ = false;
    return true;
}

void PipelineCanvas::Render() {
    // Set Data Studio ImNodes context
    ImNodes::SetCurrentContext(context_);

    // Phase 7: Render toolbar with Save/Load buttons
    RenderToolbar();

    // Begin ImNodes editor
    ImNodes::BeginNodeEditor();

    // Render all nodes
    for (const auto& node : nodes_) {
        RenderNode(node);
    }

    // Render all links
    for (const auto& link : links_) {
        ImNodes::Link(link.id, link.start_attr, link.end_attr);
    }

    ImNodes::EndNodeEditor();

    // Handle link creation
    HandleLinkCreation();

    // Handle node deletion
    HandleNodeDeletion();

    // Render node palette popup
    if (show_node_palette_) {
        RenderNodePalette();
    }

    // Context menu
    RenderContextMenu();
}

void PipelineCanvas::RenderNode(const Node& node) {
    ImNodes::BeginNode(node.id);

    // Node title bar
    ImNodes::BeginNodeTitleBar();
    ImGui::TextUnformatted(node.name.c_str());
    ImNodes::EndNodeTitleBar();

    // Input pin (all nodes except FileInput have input)
    if (node.type != "FileInput") {
        ImNodes::BeginInputAttribute(node.id * 1000);  // Input pin ID
        ImGui::Text("In");
        ImNodes::EndInputAttribute();
    }

    // Node content (parameters)
    ImGui::Spacing();
    ImGui::Text("Type: %s", node.type.c_str());

    // Display node-specific parameters
    for (const auto& [key, value] : node.parameters) {
        ImGui::Text("%s: %s", key.c_str(), value.c_str());
    }

    // Output pin
    ImNodes::BeginOutputAttribute(node.id * 1000 + 1);  // Output pin ID
    ImGui::Indent(100);
    ImGui::Text("Out");
    ImNodes::EndOutputAttribute();

    ImNodes::EndNode();

    // Phase 7: Show tooltip when hovering over node
    int hovered_id = node.id;
    if (ImNodes::IsNodeHovered(&hovered_id) && hovered_id == node.id) {
        ImGui::BeginTooltip();
        ImGui::TextUnformatted(GetNodeTooltip(node.type).c_str());
        ImGui::EndTooltip();
    }
}

void PipelineCanvas::RenderNodePalette() {
    ImGui::OpenPopup("Add Node");

    if (ImGui::BeginPopup("Add Node")) {
        const auto& quick_add_nodes = GetQuickAddNodes();
        size_t quick_add_index = 0;

        // Phase 7: Data Input/Output category with icon
        ImGui::Text(ICON_FA_DATABASE " Data Input/Output");
        ImGui::Separator();
        for (int i = 0; i < 2; ++i) {
            RenderQuickAddItem(quick_add_nodes[quick_add_index++]);
        }

        ImGui::Spacing();
        // Phase 7: Tabular Operations category with icon
        ImGui::Text(ICON_FA_TABLE " Tabular Operations");
        ImGui::Separator();
        for (int i = 0; i < 3; ++i) {
            RenderQuickAddItem(quick_add_nodes[quick_add_index++]);
        }

        // Phase 6 Week 8-9: Text Processing with icon
        ImGui::Spacing();
        ImGui::Text(ICON_FA_FILE_LINES " Text Processing");
        ImGui::Separator();
        for (int i = 0; i < 3; ++i) {
            RenderQuickAddItem(quick_add_nodes[quick_add_index++]);
        }

        // Phase 6 Week 8-9: Time-Series with icon
        ImGui::Spacing();
        ImGui::Text(ICON_FA_CHART_LINE " Time-Series");
        ImGui::Separator();
        for (int i = 0; i < 3; ++i) {
            RenderQuickAddItem(quick_add_nodes[quick_add_index++]);
        }

        // Phase 6 Week 8-9: Feature Engineering with icon
        ImGui::Spacing();
        ImGui::Text(ICON_FA_GEARS " Feature Engineering");
        ImGui::Separator();
        while (quick_add_index < quick_add_nodes.size()) {
            RenderQuickAddItem(quick_add_nodes[quick_add_index++]);
        }
        ImGui::EndPopup();
    } else {
        show_node_palette_ = false;
    }
}

void PipelineCanvas::RenderContextMenu() {
    int hovered_node = -1;
    if (ImNodes::IsNodeHovered(&hovered_node)) {
        if (ImGui::IsMouseClicked(1)) {  // Right click
            selected_node_id_ = hovered_node;
            ImGui::OpenPopup("Node Context Menu");
        }
    }

    if (ImGui::BeginPopup("Node Context Menu")) {
        if (ImGui::Selectable("Delete Node")) {
            DeleteNode(selected_node_id_);
        }
        if (ImGui::Selectable("Configure")) {
            // TODO: Open configuration dialog
        }
        ImGui::EndPopup();
    }
}

void PipelineCanvas::HandleLinkCreation() {
    int start_attr, end_attr;
    if (ImNodes::IsLinkCreated(&start_attr, &end_attr)) {
        Link link;
        link.id = next_link_id_++;
        link.start_attr = start_attr;
        link.end_attr = end_attr;
        link.start_node_id = start_attr / 1000;
        link.end_node_id = end_attr / 1000;

        links_.push_back(link);
        spdlog::debug("[Data Studio] Created link {} -> {}", link.start_node_id, link.end_node_id);
    }
}

void PipelineCanvas::HandleNodeDeletion() {
    // Check if any node is selected
    const int num_selected = ImNodes::NumSelectedNodes();
    if (num_selected > 0 && ImGui::IsKeyPressed(ImGuiKey_Delete)) {
        std::vector<int> selected_nodes(num_selected);
        ImNodes::GetSelectedNodes(selected_nodes.data());
        for (int node_id : selected_nodes) {
            DeleteNode(node_id);
        }
    }
}

void PipelineCanvas::AddNode(const std::string& type, ImVec2 position) {
    Node node;
    node.id = next_node_id_++;
    node.type = type;
    node.name = type + " #" + std::to_string(node.id);
    node.position = position;

    // Set default parameters based on type
    if (type == "FileInput") {
        node.parameters["path"] = "";
    } else if (type == "DataInput") {
        node.parameters["source_type"] = "file";
        node.parameters["file_path"] = "";
        node.parameters["type"] = "auto";
    } else if (type == "DataOutput") {
        node.parameters["file_path"] = "";
        node.parameters["file_type"] = "csv";
    } else if (type == "FilterRows") {
        node.parameters["condition"] = "";
    } else if (type == "SelectColumns") {
        node.parameters["columns"] = "";
    }
    // Phase 6 Week 8-9: Text Processing
    else if (type == "TextCleanNode" || type == "TextClean") {
        node.parameters["text_column"] = "text";
        node.parameters["lowercase"] = "true";
        node.parameters["remove_html"] = "true";
        node.parameters["remove_special_chars"] = "true";
    } else if (type == "TextTokenizer") {
        node.parameters["text_col"] = "text";
        node.parameters["tokenizer_type"] = "1";
        node.parameters["max_length"] = "256";
        node.parameters["min_word_freq"] = "2";
        node.parameters["max_vocab_size"] = "10000";
    } else if (type == "TextTokenize") {
        node.parameters["text_column"] = "text";
        node.parameters["method"] = "word";
    } else if (type == "CountVectorizer") {
        node.parameters["text_col"] = "text";
        node.parameters["max_features"] = "2000";
        node.parameters["norm"] = "l2";
    } else if (type == "TextVectorize") {
        node.parameters["text_column"] = "text";
        node.parameters["method"] = "count";
    }
    // Phase 6 Week 8-9: Time-Series
    else if (type == "TimeSeriesWindow") {
        node.parameters["value_col"] = "value";
        node.parameters["input_width"] = "10";
        node.parameters["shift"] = "1";
    } else if (type == "TSWindow") {
        node.parameters["value_col"] = "value";
        node.parameters["input_width"] = "10";
        node.parameters["shift"] = "1";
    } else if (type == "TimeSeriesFeatures") {
        node.parameters["value_col"] = "value";
        node.parameters["lag_values"] = "1,7,30";
        node.parameters["rolling_windows"] = "7";
    } else if (type == "TSFeatures") {
        node.parameters["value_col"] = "value";
        node.parameters["lag_values"] = "1,7,30";
        node.parameters["rolling_windows"] = "7";
        node.parameters["rolling_aggregations"] = "mean,std,min,max";
    } else if (type == "TSLag") {
        node.parameters["columns"] = "value";
        node.parameters["lag_periods"] = "1,7,30";
    } else if (type == "TimeSeriesLag") {
        node.parameters["columns"] = "value";
        node.parameters["lag_periods"] = "1";
    } else if (type == "Differencing") {
        node.parameters["value_col"] = "value";
        node.parameters["lag"] = "1";
        node.parameters["order"] = "1";
    } else if (type == "TSDiff") {
        node.parameters["columns"] = "value";
        node.parameters["order"] = "1";
    }
    // Phase 6 Week 8-9: Feature Engineering
    else if (type == "PCANode") {
        node.parameters["n_components"] = "2";
        node.parameters["center"] = "true";
        node.parameters["scale"] = "false";
    } else if (type == "BinningNode") {
        node.parameters["columns"] = "value";
        node.parameters["n_bins"] = "10";
        node.parameters["method"] = "equal_width";
    } else if (type == "PolynomialFeaturesNode") {
        node.parameters["degree"] = "2";
        node.parameters["columns"] = "value";
    } else if (type == "PolynomialFeatures") {
        node.parameters["degree"] = "2";
        node.parameters["columns"] = "";
        node.parameters["interaction_only"] = "false";
    } else if (type == "Binning") {
        node.parameters["columns"] = "value";
        node.parameters["n_bins"] = "10";
        node.parameters["method"] = "equal_width";
    }

    nodes_.push_back(node);
    ImNodes::SetNodeScreenSpacePos(node.id, position);

    spdlog::info("[Data Studio] Added node: {} (id={})", node.name, node.id);
}

void PipelineCanvas::DeleteNode(int node_id) {
    // Delete associated links
    links_.erase(
        std::remove_if(links_.begin(), links_.end(),
            [node_id](const Link& link) {
                return link.start_node_id == node_id || link.end_node_id == node_id;
            }),
        links_.end()
    );

    // Delete node
    nodes_.erase(
        std::remove_if(nodes_.begin(), nodes_.end(),
            [node_id](const Node& node) { return node.id == node_id; }),
        nodes_.end()
    );

    spdlog::info("[Data Studio] Deleted node: {}", node_id);
}

void PipelineCanvas::DeleteLink(int link_id) {
    links_.erase(
        std::remove_if(links_.begin(), links_.end(),
            [link_id](const Link& link) { return link.id == link_id; }),
        links_.end()
    );
}

void PipelineCanvas::Clear() {
    nodes_.clear();
    links_.clear();
    next_node_id_ = 1;
    next_link_id_ = 1;
    spdlog::info("[Data Studio] Pipeline cleared");
}

bool PipelineCanvas::ExecutePipeline() {
    if (IsPipelineExecutionTaskActive(pipeline_task_id_)) {
        spdlog::warn("[Data Studio] Pipeline execution is already active (task ID: {})",
                     pipeline_task_id_);
        return false;
    }
    if (!ValidatePipeline()) {
        spdlog::error("[Data Studio] Pipeline validation failed");
        return false;
    }

    // Serialize pipeline to JSON
    std::string pipeline_json = SerializePipeline();

    spdlog::info("[Data Studio] Queuing pipeline execution with {} nodes", nodes_.size());
    auto submission = SubmitPipelineExecutionTask(
        "Execute Data Pipeline", std::move(pipeline_json), executor_);
    pipeline_task_id_ = submission.task_id;
    return pipeline_task_id_ != 0;
}

bool PipelineCanvas::ValidatePipeline() const {
    // Check for cycles
    if (HasCycles()) {
        spdlog::error("[Data Studio] Pipeline contains cycles");
        return false;
    }

    // Check that all nodes are connected
    // TODO: Implement connectivity check

    return true;
}

bool PipelineCanvas::HasCycles() const {
    // TODO: Implement cycle detection (DFS)
    return false;
}

std::string PipelineCanvas::SerializePipeline() const {
    nlohmann::json j;

    // Phase 7: Add metadata
    j["version"] = "1.0";
    j["name"] = pipeline_name_.empty() ? "Untitled Pipeline" : pipeline_name_;

    // Add timestamp
    auto now = std::chrono::system_clock::now();
    auto time_t_now = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    ss << std::put_time(std::localtime(&time_t_now), "%Y-%m-%dT%H:%M:%S");
    j["created"] = ss.str();

    j["nodes"] = nlohmann::json::array();
    j["links"] = nlohmann::json::array();

    for (const auto& node : nodes_) {
        nlohmann::json node_json;
        node_json["id"] = node.id;
        node_json["type"] = node.type;
        node_json["name"] = node.name;

        // Phase 7: Save node position
        ImVec2 pos = ImNodes::GetNodeEditorSpacePos(node.id);
        node_json["position"]["x"] = pos.x;
        node_json["position"]["y"] = pos.y;

        node_json["parameters"] = node.parameters;
        j["nodes"].push_back(node_json);
    }

    for (const auto& link : links_) {
        nlohmann::json link_json;
        link_json["id"] = link.id;
        link_json["start_node"] = link.start_node_id;
        link_json["end_node"] = link.end_node_id;
        link_json["start_pin"] = link.start_attr;
        link_json["end_pin"] = link.end_attr;
        j["links"].push_back(link_json);
    }

    return j.dump(2);
}

bool PipelineCanvas::LoadPipeline(const std::string& json) {
    try {
        auto j = nlohmann::json::parse(json);

        Clear();

        // Phase 7: Load metadata
        if (j.contains("version")) {
            std::string version = j["version"];
            if (version != "1.0") {
                spdlog::warn("[Data Studio] Pipeline version {} may not be fully compatible", version);
            }
        }

        if (j.contains("name")) {
            pipeline_name_ = j["name"];
        }

        // Load nodes
        for (const auto& node_json : j["nodes"]) {
            Node node;
            node.id = node_json["id"];
            node.type = node_json["type"];
            node.name = node_json["name"];
            node.parameters = node_json["parameters"].get<std::map<std::string, std::string>>();
            MigrateLegacyTimeSeriesCanvasParams(node.type, node.parameters);

            // Phase 7: Load node position if available
            if (node_json.contains("position")) {
                float x = node_json["position"]["x"];
                float y = node_json["position"]["y"];
                node.position = ImVec2(x, y);
            } else {
                node.position = ImVec2(0, 0);
            }

            nodes_.push_back(node);

            // Restore node position in ImNodes
            ImNodes::SetNodeScreenSpacePos(node.id, node.position);

            if (node.id >= next_node_id_) {
                next_node_id_ = node.id + 1;
            }
        }

        // Load links
        for (const auto& link_json : j["links"]) {
            Link link;
            link.id = link_json["id"];
            link.start_node_id = link_json["start_node"];
            link.end_node_id = link_json["end_node"];

            // Phase 7: Use saved pin IDs if available
            if (link_json.contains("start_pin") && link_json.contains("end_pin")) {
                link.start_attr = link_json["start_pin"];
                link.end_attr = link_json["end_pin"];
            } else {
                // Fallback to old calculation
                link.start_attr = link.start_node_id * 1000 + 1;
                link.end_attr = link.end_node_id * 1000;
            }

            links_.push_back(link);

            if (link.id >= next_link_id_) {
                next_link_id_ = link.id + 1;
            }
        }

        spdlog::info("[Data Studio] Loaded pipeline '{}' with {} nodes, {} links",
                     pipeline_name_, nodes_.size(), links_.size());
        return true;

    } catch (const std::exception& e) {
        spdlog::error("[Data Studio] Failed to load pipeline: {}", e.what());
        return false;
    }
}

// ============================================================================
// Phase 5 Week 7 - Node Editor Handoff Methods
// ============================================================================

bool PipelineCanvas::IsDeploymentReady() const {
    return executor_ && executor_->IsDeploymentReady();
}

std::string PipelineCanvas::GetDeploymentDataset() const {
    if (executor_ && executor_->IsDeploymentReady()) {
        return executor_->GetDeploymentDataset();
    }
    return "";
}

void PipelineCanvas::ClearDeploymentStatus() {
    if (executor_) {
        executor_->ClearDeploymentStatus();
    }
}

// ============================================================================
// Phase 7 Week 10 - Save/Load & Polish
// ============================================================================

bool PipelineCanvas::SavePipelineToFile(const std::string& filepath) {
    try {
        std::string json = SerializePipeline();
        std::ofstream file(filepath);
        if (!file.is_open()) {
            spdlog::error("[Data Studio] Failed to open file for writing: {}", filepath);
            return false;
        }
        file << json;
        file.close();

        spdlog::info("[Data Studio] Saved pipeline to: {}", filepath);
        return true;
    } catch (const std::exception& e) {
        spdlog::error("[Data Studio] Failed to save pipeline: {}", e.what());
        return false;
    }
}

bool PipelineCanvas::LoadPipelineFromFile(const std::string& filepath) {
    try {
        std::ifstream file(filepath);
        if (!file.is_open()) {
            spdlog::error("[Data Studio] Failed to open file for reading: {}", filepath);
            return false;
        }

        std::string json((std::istreambuf_iterator<char>(file)),
                         std::istreambuf_iterator<char>());
        file.close();

        bool success = LoadPipeline(json);
        if (success) {
            spdlog::info("[Data Studio] Loaded pipeline from: {}", filepath);
        }
        return success;
    } catch (const std::exception& e) {
        spdlog::error("[Data Studio] Failed to load pipeline: {}", e.what());
        return false;
    }
}

void PipelineCanvas::RenderToolbar() {
    // Save button
    if (ImGui::Button(ICON_FA_FLOPPY_DISK " Save Pipeline")) {
        auto result = FileDialogs::SaveFile(
            "Save Pipeline",
            {{"Data Pipeline", "json"}},
            nullptr,
            "pipeline.json"
        );
        if (result) {
            SavePipelineToFile(*result);
        }
    }

    ImGui::SameLine();

    // Load button
    if (ImGui::Button(ICON_FA_FOLDER_OPEN " Load Pipeline")) {
        auto result = FileDialogs::OpenFile(
            "Load Pipeline",
            {{"Data Pipeline", "json"}}
        );
        if (result) {
            LoadPipelineFromFile(*result);
        }
    }

    ImGui::SameLine();
    ImGui::Separator();
    ImGui::SameLine();

    // Add Node button
    if (ImGui::Button(ICON_FA_PLUS " Add Node")) {
        show_node_palette_ = true;
    }

    ImGui::SameLine();

    const auto execution_task = pipeline_task_id_ == 0
        ? nullptr
        : AsyncTaskManager::Instance().GetTask(pipeline_task_id_);
    const bool execution_active = execution_task &&
        (execution_task->GetState() == TaskState::Pending ||
         execution_task->GetState() == TaskState::Running);

    // Execute or Cancel button depending on task state.
    if (execution_active) {
        // Cancel button (red style)
        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.8f, 0.2f, 0.2f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.9f, 0.3f, 0.3f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.7f, 0.1f, 0.1f, 1.0f));
        if (ImGui::Button(ICON_FA_STOP " Cancel")) {
            AsyncTaskManager::Instance().Cancel(pipeline_task_id_);
        }
        ImGui::PopStyleColor(3);
    } else {
        // Execute button
        if (ImGui::Button(ICON_FA_PLAY " Execute Pipeline")) {
            ExecutePipeline();
        }
    }

    ImGui::SameLine();

    // Clear button
    if (ImGui::Button(ICON_FA_TRASH " Clear")) {
        Clear();
    }

    // Phase 5 Week 7: Deploy button (only show if deployment is ready)
    if (executor_ && executor_->IsDeploymentReady()) {
        ImGui::SameLine();
        ImGui::Spacing();
        ImGui::SameLine();
        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.2f, 0.7f, 0.3f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.3f, 0.8f, 0.4f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.1f, 0.6f, 0.2f, 1.0f));
        if (ImGui::Button(ICON_FA_ROCKET " Deploy to Node Editor")) {
            // Signal to DataStudioPanel that deployment should happen
            deployment_requested_ = true;
            spdlog::info("[Data Studio] Deploy to Node Editor requested");
        }
        ImGui::PopStyleColor(3);
    }

    // Display pipeline name if set
    if (!pipeline_name_.empty()) {
        ImGui::SameLine();
        ImGui::TextDisabled("| Pipeline: %s", pipeline_name_.c_str());
    }

    // Phase 8: Show progress bar when executing
    if (execution_active) {
        const auto info = execution_task->GetInfo();
        ImGui::Separator();

        // Progress bar
        ImGui::ProgressBar(info.progress, ImVec2(-1, 0));

        // Status message
        ImGui::TextWrapped("%s", info.status_message.c_str());
    }
}

std::string PipelineCanvas::GetNodeTooltip(const std::string& node_type) const {
    // Phase 7: Comprehensive tooltips for all node types
    if (node_type == "DataInput" || node_type == "FileInput") {
        return "Load supported tabular files through the pipeline runtime\n"
               "Output: Dataset";
    } else if (node_type == "ArrowDataset") {
        return "Internal Arrow storage type\n"
               "Not addable as a pipeline runtime node";
    } else if (node_type == "DataOutput" || node_type == "SaveDataset") {
        return "Save processed dataset to file or register for ML training\n"
               "Input: Dataset\n"
               "Formats: CSV, Parquet";
    } else if (node_type == "FilterRows") {
        return "Filter rows based on SQL WHERE condition\n"
               "Example: age > 18 AND status = 'active'\n"
               "Input: Dataset\n"
               "Output: Filtered dataset";
    } else if (node_type == "SelectColumns") {
        return "Select specific columns from dataset\n"
               "Example: id, name, price\n"
               "Input: Dataset\n"
               "Output: Dataset with selected columns";
    } else if (node_type == "RemoveDuplicateRows" || node_type == "RemoveDuplicates") {
        return "Remove duplicate rows from dataset\n"
               "Uses all columns or specified columns for comparison\n"
               "Input: Dataset\n"
               "Output: Deduplicated dataset";
    } else if (node_type == "TextCleanNode" || node_type == "TextClean") {
        return "Remove HTML tags, special characters, and normalize text\n"
               "Options: lowercase, remove HTML, remove special chars\n"
               "Input: Dataset with text column\n"
               "Output: Cleaned dataset";
    } else if (node_type == "TextTokenizer" || node_type == "TextTokenize") {
        return "Split text into tokens (words, sentences, or characters)\n"
               "Methods: word, sentence, character\n"
               "Input: Dataset with text column\n"
               "Output: Tokenized dataset";
    } else if (node_type == "CountVectorizer") {
        return "Convert text to bag-of-words count features\n"
               "Input: Dataset with text column\n"
               "Output: Count feature matrix";
    } else if (node_type == "TextVectorize") {
        return "Convert text to simple numerical features\n"
               "Features: text length, word count\n"
               "Input: Dataset with text column\n"
               "Output: Dataset with text features";
    } else if (node_type == "TimeSeriesWindow" || node_type == "TSWindow") {
        return "Create sliding windows for time-series sequences\n"
               "Useful for LSTM/GRU input preparation\n"
               "Input: Time-series dataset\n"
               "Output: Windowed dataset";
    } else if (node_type == "TimeSeriesFeatures" || node_type == "TSFeatures") {
        return "Compute rolling statistics over time windows\n"
               "Features: mean, std, min, max\n"
               "Input: Time-series dataset\n"
               "Output: Dataset with rolling features";
    } else if (node_type == "TimeSeriesLag" || node_type == "TSLag") {
        return "Create lagged features (shifted values)\n"
               "Example: lag_1, lag_7, lag_30 for daily data\n"
               "Input: Time-series dataset\n"
               "Output: Dataset with lag features";
    } else if (node_type == "Differencing" || node_type == "TSDiff") {
        return "Compute differences between consecutive values\n"
               "Useful for making time-series stationary\n"
               "Input: Time-series dataset\n"
               "Output: Differenced dataset";
    } else if (node_type == "PCANode") {
        return "Dimensionality reduction using Principal Component Analysis\n"
               "Reduces feature space while preserving variance\n"
               "Input: Numeric dataset\n"
               "Output: Transformed dataset";
    } else if (node_type == "BinningNode") {
        return "Bin one numeric column into discrete buckets\n"
               "Methods: equal width, equal frequency\n"
               "Input: Numeric dataset\n"
               "Output: Dataset with bin column";
    } else if (node_type == "PolynomialFeaturesNode") {
        return "Generate polynomial features for one numeric column\n"
               "Creates x^2, x^3, and higher-degree columns\n"
               "Input: Numeric dataset\n"
               "Output: Dataset with polynomial features";
    } else if (node_type == "PolynomialFeatures") {
        return "Generate polynomial features (x^2, x^3, etc.)\n"
               "Useful for capturing non-linear relationships\n"
               "Input: Numeric dataset\n"
               "Output: Dataset with polynomial features";
    } else if (node_type == "Binning") {
        return "Discretize continuous values into bins\n"
               "Methods: equal_width, equal_freq (alias: equal_frequency)\n"
               "Input: Numeric dataset\n"
               "Output: Binned dataset";
    } else {
        return "Unknown node type";
    }
}

} // namespace cyxwiz
