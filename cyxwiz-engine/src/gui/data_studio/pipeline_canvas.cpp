#include "pipeline_canvas.h"
#include "pipeline_executor.h"
#include <spdlog/spdlog.h>
#include <nlohmann/json.hpp>

namespace cyxwiz {

PipelineCanvas::PipelineCanvas()
    : context_(nullptr)
    , next_node_id_(1)
    , next_link_id_(1)
    , show_node_palette_(false)
    , selected_node_id_(-1)
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
    executor_ = std::make_unique<PipelineExecutor>();

    spdlog::info("[Data Studio] PipelineCanvas initialized");
}

PipelineCanvas::~PipelineCanvas() {
    // Skip context cleanup - ImNodes contexts are cleaned up automatically
    // when ImGui shuts down. Explicit cleanup can cause crashes during
    // application shutdown if ImGui is already destroyed.
}

void PipelineCanvas::Render() {
    // Set Data Studio ImNodes context
    ImNodes::SetCurrentContext(context_);

    // Begin ImNodes editor
    ImNodes::BeginNodeEditor();

    // Render node palette button
    if (ImGui::Button("Add Node")) {
        show_node_palette_ = true;
    }
    ImGui::SameLine();
    if (ImGui::Button("Execute Pipeline")) {
        ExecutePipeline();
    }
    ImGui::SameLine();
    if (ImGui::Button("Clear")) {
        Clear();
    }

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
}

void PipelineCanvas::RenderNodePalette() {
    ImGui::OpenPopup("Add Node");

    if (ImGui::BeginPopup("Add Node")) {
        ImGui::Text("Data Sources");
        ImGui::Separator();
        if (ImGui::Selectable("File Input")) {
            AddNode("FileInput", ImGui::GetMousePos());
            show_node_palette_ = false;
        }
        if (ImGui::Selectable("Arrow Dataset")) {
            AddNode("ArrowDataset", ImGui::GetMousePos());
            show_node_palette_ = false;
        }

        ImGui::Spacing();
        ImGui::Text("Transformations");
        ImGui::Separator();
        if (ImGui::Selectable("Filter Rows")) {
            AddNode("FilterRows", ImGui::GetMousePos());
            show_node_palette_ = false;
        }
        if (ImGui::Selectable("Select Columns")) {
            AddNode("SelectColumns", ImGui::GetMousePos());
            show_node_palette_ = false;
        }
        if (ImGui::Selectable("Remove Duplicates")) {
            AddNode("RemoveDuplicates", ImGui::GetMousePos());
            show_node_palette_ = false;
        }

        ImGui::Spacing();
        ImGui::Text("Output");
        ImGui::Separator();
        if (ImGui::Selectable("Save Dataset")) {
            AddNode("SaveDataset", ImGui::GetMousePos());
            show_node_palette_ = false;
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
    } else if (type == "FilterRows") {
        node.parameters["condition"] = "";
    } else if (type == "SelectColumns") {
        node.parameters["columns"] = "";
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
    if (!ValidatePipeline()) {
        spdlog::error("[Data Studio] Pipeline validation failed");
        return false;
    }

    // Serialize pipeline to JSON
    std::string pipeline_json = SerializePipeline();

    // Execute pipeline using the executor
    spdlog::info("[Data Studio] Starting pipeline execution with {} nodes", nodes_.size());
    bool success = executor_->ExecutePipeline(pipeline_json);

    if (success) {
        spdlog::info("[Data Studio] Pipeline execution completed successfully");
    } else {
        spdlog::error("[Data Studio] Pipeline execution failed: {}", executor_->GetLastError());
    }

    return success;
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
    j["nodes"] = nlohmann::json::array();
    j["links"] = nlohmann::json::array();

    for (const auto& node : nodes_) {
        nlohmann::json node_json;
        node_json["id"] = node.id;
        node_json["type"] = node.type;
        node_json["name"] = node.name;
        node_json["parameters"] = node.parameters;
        j["nodes"].push_back(node_json);
    }

    for (const auto& link : links_) {
        nlohmann::json link_json;
        link_json["id"] = link.id;
        link_json["start_node"] = link.start_node_id;
        link_json["end_node"] = link.end_node_id;
        j["links"].push_back(link_json);
    }

    return j.dump(2);
}

bool PipelineCanvas::LoadPipeline(const std::string& json) {
    try {
        auto j = nlohmann::json::parse(json);

        Clear();

        // Load nodes
        for (const auto& node_json : j["nodes"]) {
            Node node;
            node.id = node_json["id"];
            node.type = node_json["type"];
            node.name = node_json["name"];
            node.parameters = node_json["parameters"].get<std::map<std::string, std::string>>();
            nodes_.push_back(node);

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
            link.start_attr = link.start_node_id * 1000 + 1;
            link.end_attr = link.end_node_id * 1000;
            links_.push_back(link);

            if (link.id >= next_link_id_) {
                next_link_id_ = link.id + 1;
            }
        }

        spdlog::info("[Data Studio] Loaded pipeline with {} nodes, {} links",
                     nodes_.size(), links_.size());
        return true;

    } catch (const std::exception& e) {
        spdlog::error("[Data Studio] Failed to load pipeline: {}", e.what());
        return false;
    }
}

} // namespace cyxwiz
