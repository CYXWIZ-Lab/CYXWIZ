#include "node_editor.h"
#include "node_documentation.h"
#include "node_editor_shape_inference.h"
#include "panels/script_editor.h"
#include "properties.h"
#include "patterns/pattern_library.h"
#include "icons.h"
#include "../core/data_registry.h"
#include "../core/training_manager.h"
#include "../core/async_task_manager.h"
#include "../core/project_manager.h"
#include "../core/graph_executor.h"
#include "../core/rl_training_executor.h"
#include "../core/pipeline_executor.h"  // Unified Canvas Phase 2
#include "panels/training_dashboard.h"
#include "../core/rl_script_generator.h"
#include "../scripting/scripting_engine.h"
#include "../plugin/registries/plugin_node_registry.h"
#include "../core/node_metadata_registry.h"
#include <imgui.h>
#include <imgui_internal.h>
#include <imnodes.h>
#include <spdlog/spdlog.h>
#include <algorithm>
#include <memory>
#include <chrono>
#include <map>
#include <set>
#include <queue>
#include <functional>
#include <cmath>
#include <cstring>
#include <nlohmann/json.hpp>
#include <fstream>
#ifdef _WIN32
#include <windows.h>
#include <commdlg.h>
#endif

namespace gui {

// Forward declaration (used in Render before definition)
static std::string GetPinTypeName(PinType type);

void NodeEditor::SetWorkflowDescription(const std::string& desc) {
    strncpy(workflow_description_, desc.c_str(), sizeof(workflow_description_) - 1);
    workflow_description_[sizeof(workflow_description_) - 1] = '\0';
}

// ===== Pin / Node State =====

void NodeEditor::SetNodePinState(int node_id, NodePinState state) {
    node_pin_state_[node_id] = state;
}

void NodeEditor::SetAllNodesPinState(NodePinState state) {
    for (const auto& node : nodes_) {
        node_pin_state_[node.id] = state;
    }
}

void NodeEditor::ClearValidationState() {
    node_pin_state_.clear();
}

NodeEditor::NodePinState NodeEditor::GetNodePinState(int node_id) const {
    auto it = node_pin_state_.find(node_id);
    if (it == node_pin_state_.end()) return NodePinState::Default;
    return it->second;
}

void NodeEditor::AddAnnotation() {
    // Add annotation at center of visible area
    ImVec2 center_pos = ImVec2(400.0f, 300.0f);  // Default position
    AddAnnotationAt(center_pos);
}

void NodeEditor::AddAnnotationAt(const ImVec2& position) {
    CanvasAnnotation annotation;
    annotation.id = next_annotation_id_++;
    annotation.title = "Note";
    annotation.content = "Enter description here...";
    annotation.position = position;
    annotation.size = ImVec2(200.0f, 100.0f);
    annotation.color = IM_COL32(255, 255, 200, 255);  // Yellow
    annotation.is_minimized = false;
    annotations_.push_back(annotation);
}

NodeEditor::NodeEditor()
    : show_window_(true),
      next_node_id_(1),
      next_pin_id_(1),
      next_link_id_(1),
      show_context_menu_(false),
      context_menu_node_id_(-1),
      selected_node_id_(-1),
      selected_framework_(CodeFramework::PyTorch),
      execution_mode_(ExecutionMode::CodeGeneration),  // Unified Canvas Phase 2: Default to code generation
      editor_context_(nullptr),
      script_editor_(nullptr),
      properties_panel_(nullptr) {

    // Create ImNodes editor context
    editor_context_ = ImNodes::EditorContextCreate();
    ImNodes::EditorContextSet(editor_context_);

    // Configure ImNodes
    ImNodesIO& io = ImNodes::GetIO();
    io.LinkDetachWithModifierClick.Modifier = &ImGui::GetIO().KeyCtrl;
    io.MultipleSelectModifier.Modifier = &ImGui::GetIO().KeyCtrl;

    // Enable left-click drag for panning (always enabled, no modifier needed)
    static bool always_true = true;
    io.EmulateThreeButtonMouse.Modifier = &always_true;

    ImNodesStyle& style = ImNodes::GetStyle();
    style.Flags |= ImNodesStyleFlags_GridLines;

    // Initialize shape inference engine
    shape_inference_ = std::make_unique<ShapeInferenceEngine>();

    // Create ML Training Pipeline showcase
    // Demonstrates: Complete machine learning workflow with data preparation, model, loss, and optimizer
    //
    // Flow: DataInput → DataSplit → Normalize → Dense(128) → ReLU → Dense(10) → MSELoss → Adam → Output

    // ========== DATA PREPARATION ==========

    // 1. Data Input - Load dataset
    MLNode data_input = CreateNode(NodeType::DataInput, "Data Input");
    data_input.parameters["format"] = "csv";
    data_input.parameters["file_path"] = "dataset.csv";
    nodes_.push_back(data_input);
    ImNodes::SetNodeGridSpacePos(data_input.id, ImVec2(50.0f, 200.0f));

    // 2. Data Split - Split into train/val/test
    MLNode data_split = CreateNode(NodeType::DataSplit, "Train/Val/Test Split");
    data_split.parameters["train_ratio"] = "0.8";
    data_split.parameters["val_ratio"] = "0.1";
    data_split.parameters["test_ratio"] = "0.1";
    data_split.parameters["shuffle"] = "true";
    nodes_.push_back(data_split);
    ImNodes::SetNodeGridSpacePos(data_split.id, ImVec2(250.0f, 200.0f));

    // 3. Normalize - Normalize input features
    MLNode normalize = CreateNode(NodeType::Normalize, "Normalize");
    normalize.parameters["method"] = "z-score";
    normalize.parameters["mean"] = "0.0";
    normalize.parameters["std"] = "1.0";
    nodes_.push_back(normalize);
    ImNodes::SetNodeGridSpacePos(normalize.id, ImVec2(450.0f, 200.0f));

    // ========== NEURAL NETWORK ==========

    // 4. Dense Layer - First hidden layer
    MLNode dense1 = CreateNode(NodeType::Dense, "Dense (128)");
    dense1.parameters["units"] = "128";
    dense1.parameters["use_bias"] = "true";
    nodes_.push_back(dense1);
    ImNodes::SetNodeGridSpacePos(dense1.id, ImVec2(650.0f, 200.0f));

    // 5. ReLU Activation
    MLNode relu = CreateNode(NodeType::ReLU, "ReLU");
    nodes_.push_back(relu);
    ImNodes::SetNodeGridSpacePos(relu.id, ImVec2(850.0f, 200.0f));

    // 6. Dense Layer - Output layer
    MLNode dense2 = CreateNode(NodeType::Dense, "Dense (10)");
    dense2.parameters["units"] = "10";
    dense2.parameters["use_bias"] = "true";
    nodes_.push_back(dense2);
    ImNodes::SetNodeGridSpacePos(dense2.id, ImVec2(1050.0f, 200.0f));

    // ========== TRAINING CONFIGURATION ==========

    // 7. Loss Function
    MLNode loss = CreateNode(NodeType::MSELoss, "MSE Loss");
    nodes_.push_back(loss);
    ImNodes::SetNodeGridSpacePos(loss.id, ImVec2(1250.0f, 200.0f));

    // 8. Optimizer
    MLNode optimizer = CreateNode(NodeType::Adam, "Adam");
    optimizer.parameters["learning_rate"] = "0.001";
    optimizer.parameters["beta1"] = "0.9";
    optimizer.parameters["beta2"] = "0.999";
    nodes_.push_back(optimizer);
    ImNodes::SetNodeGridSpacePos(optimizer.id, ImVec2(1450.0f, 200.0f));

    // 9. Output
    MLNode output = CreateNode(NodeType::Output, "Output");
    nodes_.push_back(output);
    ImNodes::SetNodeGridSpacePos(output.id, ImVec2(1650.0f, 200.0f));

    // ========== CREATE CONNECTIONS ==========

    // Data flow: DataInput -> DataSplit -> Normalize
    CreateLink(data_input.outputs[0].id, data_split.inputs[0].id,
               data_input.id, data_split.id);
    CreateLink(data_split.outputs[0].id, normalize.inputs[0].id,
               data_split.id, normalize.id);

    // Model flow: Normalize -> Dense(128) -> ReLU -> Dense(10)
    CreateLink(normalize.outputs[0].id, dense1.inputs[0].id,
               normalize.id, dense1.id);
    CreateLink(dense1.outputs[0].id, relu.inputs[0].id,
               dense1.id, relu.id);
    CreateLink(relu.outputs[0].id, dense2.inputs[0].id,
               relu.id, dense2.id);

    // Training flow: Dense(10) -> MSELoss -> Adam -> Output
    CreateLink(dense2.outputs[0].id, loss.inputs[0].id,
               dense2.id, loss.id);
    CreateLink(loss.outputs[0].id, optimizer.inputs[0].id,
               loss.id, optimizer.id);
    CreateLink(optimizer.outputs[0].id, output.inputs[0].id,
               optimizer.id, output.id);

    spdlog::info("Created ML Training Pipeline with {} nodes and {} connections",
                 nodes_.size(), links_.size());
    spdlog::info("Pipeline: DataInput -> DataSplit -> Normalize -> Dense(128) -> ReLU -> Dense(10) -> MSELoss -> Adam -> Output");
}

NodeEditor::~NodeEditor() {
    OnStopSimulation();
    if (editor_context_) {
        ImNodes::EditorContextFree(editor_context_);
    }
}

void NodeEditor::Render() {
    if (!show_window_) return;

    // Update training animation time
    if (is_training_) {
        training_animation_time_ += ImGui::GetIO().DeltaTime;
    }

    // Set the editor context for this node editor instance
    ImNodes::EditorContextSet(editor_context_);

    // Handle full context reset (after ClearGraph)
    // This fully recreates the ImNodes editor context to clear all internal state
    // and prevent crashes from stale node references
    if (pending_context_reset_) {
        spdlog::info("Resetting ImNodes editor context");
        ImNodes::EditorContextFree(editor_context_);
        editor_context_ = ImNodes::EditorContextCreate();
        ImNodes::EditorContextSet(editor_context_);
        pending_context_reset_ = false;
    }

    if (ImGui::Begin("CyxWiz Studio", &show_window_)) {
        ShowToolbar();

        ImGui::Separator();

        // Check if mouse is over minimap (using stored bounds from previous frame)
        // This needs to be done before ImNodes::BeginNodeEditor to prevent canvas panning
        ImVec2 mouse_pos = ImGui::GetMousePos();
        bool mouse_in_minimap_bounds = show_minimap_ &&
            mouse_pos.x >= minimap_screen_min_.x && mouse_pos.x <= minimap_screen_max_.x &&
            mouse_pos.y >= minimap_screen_min_.y && mouse_pos.y <= minimap_screen_max_.y;

        // If mouse is in minimap or minimap is being navigated, temporarily consume mouse input
        // This prevents ImNodes from handling panning when we're working with the minimap
        if (mouse_in_minimap_bounds || minimap_navigating_) {
            // Mark mouse as captured so ImNodes doesn't process canvas panning
            ImGui::GetIO().WantCaptureMouse = true;
        }

        // Apply zoom to ImNodes style
        ImNodes::PushStyleVar(ImNodesStyleVar_GridSpacing, 32.0f * zoom_);
        ImNodes::PushStyleVar(ImNodesStyleVar_NodePadding, ImVec2(8.0f * zoom_, 8.0f * zoom_));
        ImNodes::PushStyleVar(ImNodesStyleVar_PinCircleRadius, 4.0f * zoom_);
        ImNodes::PushStyleVar(ImNodesStyleVar_LinkThickness, 3.0f * zoom_);
        ImNodes::PushStyleVar(ImNodesStyleVar_PinLineThickness, 1.0f * zoom_);
        ImGui::SetWindowFontScale(zoom_);

        ImNodes::BeginNodeEditor();

        // Handle deferred ImNodes clear (must be inside BeginNodeEditor scope)
        // Note: We skip ImNodes::ClearNodeSelection/ClearLinkSelection because they may
        // crash if called when ImNodes has stale internal references to nodes that no
        // longer exist. Instead, we just reset our internal state and let ImNodes
        // naturally clear selection when it finds no valid selected nodes.
        if (pending_clear_imnodes_) {
            // Reset internal selection state only - don't call ImNodes functions
            // ImNodes will auto-clear selection when nodes aren't rendered
            pending_clear_imnodes_ = false;
        }

        // Render annotations first (appear behind everything)
        RenderAnnotations();

        // Render group backgrounds before nodes so they appear behind
        RenderGroups();

        // Render frames (visual organization boxes) before nodes
        RenderFrames();

        RenderNodes();

        // Handle mouse wheel zoom (skip if mouse is over minimap)
        if (ImGui::IsWindowHovered() && !mouse_in_minimap_bounds) {
            float wheel = ImGui::GetIO().MouseWheel;
            if (wheel != 0.0f) {
                // Real zoom implementation - adjust zoom factor
                float old_zoom = zoom_;
                zoom_ = std::clamp(zoom_ + wheel * 0.1f, ZOOM_MIN, ZOOM_MAX);

                // Zoom toward mouse position by adjusting panning
                if (zoom_ != old_zoom) {
                    ImVec2 panning = ImNodes::EditorContextGetPanning();
                    ImVec2 editor_origin = ImGui::GetWindowPos();
                    editor_origin.y += ImGui::GetFrameHeight() + ImGui::GetStyle().ItemSpacing.y + 30.0f;
                    editor_origin.x += ImGui::GetStyle().WindowPadding.x;

                    ImVec2 mouse_rel = ImVec2(
                        mouse_pos.x - editor_origin.x - panning.x,
                        mouse_pos.y - editor_origin.y - panning.y
                    );

                    float zoom_factor = zoom_ / old_zoom;
                    ImNodes::EditorContextResetPanning(ImVec2(
                        panning.x - mouse_rel.x * (zoom_factor - 1.0f),
                        panning.y - mouse_rel.y * (zoom_factor - 1.0f)
                    ));
                }
            }
        }

        // Handle right-click context menu (skip if mouse is over minimap)
        bool right_click_detected = false;
        if (ImNodes::IsEditorHovered() && !mouse_in_minimap_bounds && ImGui::IsMouseClicked(ImGuiMouseButton_Right)) {
            // Store mouse position for node placement in grid space
            // The editor origin is at the window content region start
            ImVec2 editor_origin = ImGui::GetWindowPos();
            editor_origin.y += ImGui::GetFrameHeight() + ImGui::GetStyle().ItemSpacing.y + 30.0f;  // Account for title bar and toolbar
            editor_origin.x += ImGui::GetStyle().WindowPadding.x;
            ImVec2 panning = ImNodes::EditorContextGetPanning();

            // Convert screen position to grid space
            context_menu_pos_ = ImVec2(
                mouse_pos.x - editor_origin.x - panning.x,
                mouse_pos.y - editor_origin.y - panning.y
            );
            right_click_detected = true;
        }

        // Cache node positions while still inside BeginNodeEditor/EndNodeEditor scope
        // This is needed because GetNodeGridSpacePos only works inside this scope
        cached_node_positions_.clear();
        for (const auto& node : nodes_) {
            cached_node_positions_[node.id] = ImNodes::GetNodeGridSpacePos(node.id);
        }

        ImNodes::EndNodeEditor();

        // Pin hover tooltip for all nodes (after EndNodeEditor)
        int hovered_pin_id = -1;
        if (ImNodes::IsPinHovered(&hovered_pin_id)) {
            const MLNode* hovered_node = nullptr;
            const NodePin* hovered_pin = nullptr;
            bool hovered_is_input = false;

            // O(1) pin lookup using hash map
            auto it = pin_lookup_.find(hovered_pin_id);
            if (it != pin_lookup_.end()) {
                hovered_node = it->second.first;
                hovered_pin = it->second.second;
                hovered_is_input = hovered_pin->is_input;
            }

            if (hovered_node && hovered_pin) {
                ImGui::BeginTooltip();
                ImGui::Text("%s - %s", hovered_node->name.c_str(),
                            hovered_is_input ? "Input" : "Output");
                ImGui::Separator();
                ImGui::Text("Pin: %s", hovered_pin->name.c_str());
                ImGui::Text("Type: %s", GetPinTypeName(hovered_pin->type).c_str());
                int connections = GetConnectionCount(hovered_pin->id);
                if (connections > 0) {
                    ImGui::Text("Connections: %d", connections);
                } else {
                    ImGui::TextDisabled("Not connected");
                }
                if (!hovered_pin->description.empty()) {
                    ImGui::Separator();
                    ImGui::PushTextWrapPos(ImGui::GetFontSize() * 24.0f);
                    ImGui::TextUnformatted(hovered_pin->description.c_str());
                    ImGui::PopTextWrapPos();
                }
                ImGui::EndTooltip();
            }
        }

        // === Handle right-click context menu AFTER EndNodeEditor ===
        // IsNodeHovered() only works after EndNodeEditor() is called
        // Skip if right-click was on a frame (frame menu already shown)
        if (right_click_detected && !frame_right_clicked_) {
            int hovered_node_id = -1;
            bool is_node_hovered = ImNodes::IsNodeHovered(&hovered_node_id);
            if (is_node_hovered && hovered_node_id >= 0) {
                // Right-click on node - show node-specific menu
                right_clicked_node_id_ = hovered_node_id;
                ImGui::OpenPopup("SingleNodeContextMenu");
            } else {
                // Right-click on canvas - show add node menu
                ImGui::OpenPopup("NodeContextMenu");
            }
        }

        // Node-specific context menu (right-click on node)
        if (ImGui::BeginPopup("SingleNodeContextMenu")) {
            ShowSingleNodeContextMenu();
            ImGui::EndPopup();
        }

        // Canvas context menu (right-click on empty space)
        if (ImGui::BeginPopup("NodeContextMenu")) {
            ShowContextMenu();
            ImGui::EndPopup();
        }

        // Node description edit popup
        ShowNodeDescriptionEditPopup();

        // === Handle drag-drop from Node Browser ===
        // Check if a NODE_TYPE payload is being dropped on the canvas
        if (ImGui::BeginDragDropTarget()) {
            if (const ImGuiPayload* payload = ImGui::AcceptDragDropPayload("NODE_TYPE")) {
                // Get the dropped node type
                cyxwiz::NodeType dropped_type = *(const cyxwiz::NodeType*)payload->Data;

                // Get metadata to create proper node name
                auto& registry = cyxwiz::NodeMetadataRegistry::Instance();
                auto* metadata = registry.GetMetadata(dropped_type);
                std::string node_name = metadata ? metadata->name : "Node";

                // Calculate drop position in grid space (reuse mouse_pos from earlier)
                ImVec2 editor_origin = ImGui::GetWindowPos();
                ImVec2 panning = ImNodes::EditorContextGetPanning();
                ImVec2 drop_pos(
                    (mouse_pos.x - editor_origin.x - panning.x) / zoom_,
                    (mouse_pos.y - editor_origin.y - panning.y - 50) / zoom_  // Offset for toolbar
                );

                // Queue node for addition
                PendingNode pending;
                pending.type = static_cast<NodeType>(dropped_type);
                pending.name = node_name;
                pending.position = drop_pos;
                pending_nodes_.push_back(pending);

                spdlog::info("Drag-drop: Adding {} node at ({}, {})", node_name, drop_pos.x, drop_pos.y);
            }

            // Handle ANNOTATION drag-drop
            if (const ImGuiPayload* payload = ImGui::AcceptDragDropPayload("ANNOTATION")) {
                // Calculate drop position in grid space
                ImVec2 editor_origin = ImGui::GetWindowPos();
                ImVec2 panning = ImNodes::EditorContextGetPanning();
                ImVec2 drop_pos(
                    (mouse_pos.x - editor_origin.x - panning.x) / zoom_,
                    (mouse_pos.y - editor_origin.y - panning.y - 50) / zoom_
                );

                // Add annotation at drop position
                AddAnnotationAt(drop_pos);
                spdlog::info("Drag-drop: Adding annotation at ({}, {})", drop_pos.x, drop_pos.y);
            }

            // Handle STUDIO_FRAME drag-drop
            if (const ImGuiPayload* payload = ImGui::AcceptDragDropPayload("STUDIO_FRAME")) {
                // Calculate drop position in grid space
                ImVec2 editor_origin = ImGui::GetWindowPos();
                ImVec2 panning = ImNodes::EditorContextGetPanning();
                ImVec2 drop_pos(
                    (mouse_pos.x - editor_origin.x - panning.x) / zoom_,
                    (mouse_pos.y - editor_origin.y - panning.y - 50) / zoom_
                );

                // Add frame at drop position
                AddFrameAt(drop_pos);
                spdlog::info("Drag-drop: Adding frame at ({}, {})", drop_pos.x, drop_pos.y);
            }
            ImGui::EndDragDropTarget();
        }

        // Pop zoom style variables
        ImNodes::PopStyleVar(5);
        ImGui::SetWindowFontScale(1.0f);

        // Render minimap overlay in bottom-right corner
        if (show_minimap_) {
            RenderMinimap();
        }

        // Process any pending node additions (deferred to avoid modifying nodes_ during ImNodes rendering)
        if (!pending_nodes_.empty()) {
            SaveUndoState();  // Save state before adding nodes
        }
        for (const auto& pending : pending_nodes_) {
            MLNode node = CreateNode(pending.type, pending.name);

            nodes_.push_back(node);

            // Check for sentinel value indicating auto-position should be used
            ImVec2 position = pending.position;
            if (position.x < -90000.0f && position.y < -90000.0f) {
                // Use FindEmptyPosition now that we're after EndNodeEditor
                position = FindEmptyPosition();
            }

            // Set node position
            ImNodes::SetNodeGridSpacePos(node.id, position);

            // Select the newly created node so it's highlighted
            ImNodes::ClearNodeSelection();
            ImNodes::SelectNode(node.id);
            selected_node_id_ = node.id;

            spdlog::info("Added node: {} (ID: {}) at position ({}, {})",
                        pending.name, node.id, position.x, position.y);
        }
        if (!pending_nodes_.empty()) {
            RebuildPinLookup();  // Rebuild pin lookup after adding nodes
        }
        pending_nodes_.clear();

        // Handle interactions AFTER EndNodeEditor() - this is when ImNodes processes them
        HandleInteractions();

        // Handle keyboard shortcuts (Ctrl+Z, Ctrl+C, etc.)
        HandleKeyboardShortcuts();

        // Show search bar if visible (Ctrl+F to toggle)
        ShowSearchBar();

        // Show node add search (top-right search box for quick node creation)
        ShowNodeAddSearch();

        // Update properties panel with selected node
        const int num_selected = ImNodes::NumSelectedNodes();

        // Sync selected_node_ids_ with ImNodes' selection state
        if (num_selected > 0) {
            std::vector<int> imnodes_selection(num_selected);
            ImNodes::GetSelectedNodes(imnodes_selection.data());

            // Only update if selection changed
            bool selection_changed = (selected_node_ids_.size() != static_cast<size_t>(num_selected));
            if (!selection_changed) {
                for (int i = 0; i < num_selected; ++i) {
                    if (std::find(selected_node_ids_.begin(), selected_node_ids_.end(),
                                  imnodes_selection[i]) == selected_node_ids_.end()) {
                        selection_changed = true;
                        break;
                    }
                }
            }

            if (selection_changed) {
                selected_node_ids_ = std::move(imnodes_selection);
            }
        } else {
            selected_node_ids_.clear();
        }

        if (properties_panel_) {
            if (num_selected == 1 && !nodes_.empty()) {
                int selected_nodes[1];
                ImNodes::GetSelectedNodes(selected_nodes);
                int new_selected_id = selected_nodes[0];

                // Validate the node ID - skip if invalid (stale data after ClearGraph)
                if (new_selected_id <= 0) {
                    // Invalid node ID, treat as no selection
                    if (selected_node_id_ != -1) {
                        selected_node_id_ = -1;
                        properties_panel_->ClearSelection();
                    }
                } else if (new_selected_id != selected_node_id_) {
                    // Only log if selection changed
                    spdlog::info("Node selection changed to ID: {}", new_selected_id);

                    // Find the selected node and pass it to the properties panel
                    MLNode* selected = nullptr;
                    for (auto& node : nodes_) {
                        if (node.id == new_selected_id) {
                            selected = &node;
                            spdlog::info("Found selected node: id={}, type={}, name={}",
                                         node.id, static_cast<int>(node.type), node.name);
                            break;
                        }
                    }

                    if (selected) {
                        selected_node_id_ = new_selected_id;
                        spdlog::info("About to call SetSelectedNode with node id={}", selected->id);
                        properties_panel_->SetSelectedNode(selected);
                        spdlog::info("SetSelectedNode completed successfully");
                    } else {
                        // Node ID not found in our nodes - could be stale data
                        spdlog::debug("Selection ID {} not found in nodes vector, ignoring", new_selected_id);
                    }
                } else {
                    // Selection unchanged, update silently
                    MLNode* selected = nullptr;
                    for (auto& node : nodes_) {
                        if (node.id == selected_node_id_) {
                            selected = &node;
                            break;
                        }
                    }
                    if (selected) {
                        properties_panel_->SetSelectedNode(selected);
                    }
                }
            } else if (num_selected == 0 && selected_node_id_ != -1) {
                spdlog::info("Node deselected");
                selected_node_id_ = -1;
                properties_panel_->ClearSelection();
            }
        }
    }
    ImGui::End();

    // Render RL Training Dashboard (separate window)
    if (rl_dashboard_) {
        rl_dashboard_->Render();
    }

    // ===== Save as Pattern Dialog =====
    if (show_save_pattern_dialog_) {
        ImGui::OpenPopup("Save as Pattern");
    }

    if (ImGui::BeginPopupModal("Save as Pattern", &show_save_pattern_dialog_, ImGuiWindowFlags_AlwaysAutoResize)) {
        ImGui::Text("Save selected nodes as a reusable pattern");
        ImGui::Separator();
        ImGui::Spacing();

        ImGui::Text("Pattern Name:");
        ImGui::SetNextItemWidth(300);
        ImGui::InputText("##PatternName", save_pattern_name_, sizeof(save_pattern_name_));

        ImGui::Spacing();
        ImGui::Text("Description:");
        ImGui::SetNextItemWidth(300);
        ImGui::InputTextMultiline("##PatternDescription", save_pattern_description_,
                                   sizeof(save_pattern_description_), ImVec2(300, 80));

        ImGui::Spacing();
        ImGui::TextDisabled("Selected nodes: %zu", selected_node_ids_.size());

        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();

        bool name_valid = std::strlen(save_pattern_name_) > 0;

        if (!name_valid) {
            ImGui::BeginDisabled();
        }

        if (ImGui::Button("Save Pattern", ImVec2(140, 0))) {
            // Get node positions from ImNodes and save
            auto& library = patterns::PatternLibrary::Instance();
            auto& pm = cyxwiz::ProjectManager::Instance();

            // Build nodes with positions
            std::vector<MLNode> nodes_with_pos;
            for (int node_id : selected_node_ids_) {
                for (auto& node : nodes_) {
                    if (node.id == node_id) {
                        MLNode node_copy = node;
                        auto it = cached_node_positions_.find(node.id);
                        ImVec2 pos = (it != cached_node_positions_.end()) ? it->second : ImVec2(0,0);
                        node_copy.initial_pos_x = pos.x;
                        node_copy.initial_pos_y = pos.y;
                        node_copy.has_initial_position = true;
                        nodes_with_pos.push_back(node_copy);
                        break;
                    }
                }
            }

            // Build project-specific save path: <project_root>/patterns/<name>.json
            std::string save_path = pm.GetProjectRoot() + "/patterns/" + save_pattern_name_ + ".json";

            bool success = library.SavePatternFromSelection(
                nodes_with_pos,
                links_,
                selected_node_ids_,
                save_pattern_name_,
                save_pattern_description_,
                patterns::PatternCategory::Custom,
                save_path
            );

            if (success) {
                spdlog::info("Pattern '{}' saved to project: {}", save_pattern_name_, save_path);
            } else {
                spdlog::error("Failed to save pattern '{}'", save_pattern_name_);
            }

            show_save_pattern_dialog_ = false;
            ImGui::CloseCurrentPopup();
        }

        if (!name_valid) {
            ImGui::EndDisabled();
        }

        ImGui::SameLine();

        if (ImGui::Button("Cancel", ImVec2(140, 0))) {
            show_save_pattern_dialog_ = false;
            ImGui::CloseCurrentPopup();
        }

        ImGui::EndPopup();
    }

    // ===== Empty Graph Warning Popup =====
    if (show_empty_graph_warning_) {
        ImGui::OpenPopup("Empty Graph Warning");
    }

    if (ImGui::BeginPopupModal("Empty Graph Warning", &show_empty_graph_warning_, ImGuiWindowFlags_AlwaysAutoResize)) {
        ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.2f, 1.0f), ICON_FA_TRIANGLE_EXCLAMATION);
        ImGui::SameLine();
        ImGui::Text("Cannot Save Empty Graph");
        ImGui::Separator();
        ImGui::Spacing();

        ImGui::TextWrapped("The node graph is empty. Please add at least one node before saving.");

        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();

        float button_width = 120.0f;
        float window_width = ImGui::GetWindowWidth();
        ImGui::SetCursorPosX((window_width - button_width) * 0.5f);

        if (ImGui::Button("OK", ImVec2(button_width, 0))) {
            show_empty_graph_warning_ = false;
            ImGui::CloseCurrentPopup();
        }

        ImGui::EndPopup();
    }

    // Auto-Insert Flatten Dialog
    if (show_auto_insert_flatten_dialog_) {
        ImGui::OpenPopup("Shape Mismatch Detected");
    }

    if (ImGui::BeginPopupModal("Shape Mismatch Detected",
                              &show_auto_insert_flatten_dialog_,
                              ImGuiWindowFlags_AlwaysAutoResize)) {

        ImGui::TextColored(ImVec4(1.0f, 0.76f, 0.03f, 1.0f),
                         ICON_FA_TRIANGLE_EXCLAMATION " Shape Mismatch");
        ImGui::Separator();
        ImGui::Spacing();

        // Find source and target nodes for display
        MLNode* from_node = nullptr;
        MLNode* to_node = nullptr;
        for (auto& node : nodes_) {
            if (node.id == pending_flatten_from_node_) from_node = &node;
            if (node.id == pending_flatten_to_node_) to_node = &node;
        }

        if (from_node && to_node) {
            ImGui::Text("Source: %s (4D tensor output)", from_node->name.c_str());
            ImGui::Text("Target: %s (expects 2D input)", to_node->name.c_str());
        } else {
            ImGui::Text("Connection requires shape transformation");
        }

        ImGui::Spacing();
        ImGui::TextWrapped("Dense layers require flattened 2D input [batch, features]. "
                        "The source node outputs a 4D tensor [batch, height, width, channels]. "
                        "Insert a Flatten node to convert the shape?");
        ImGui::Spacing();

        // Button 1: Auto-Insert Flatten
        if (ImGui::Button(ICON_FA_WAND_MAGIC_SPARKLES " Auto-Insert Flatten", ImVec2(210, 0))) {
            if (from_node && to_node) {
                // Create Flatten node
                MLNode flatten_node = CreateNode(NodeType::Flatten, "Flatten");

                // Calculate position at midpoint between nodes
                ImVec2 from_pos = ImNodes::GetNodeGridSpacePos(from_node->id);
                ImVec2 to_pos = ImNodes::GetNodeGridSpacePos(to_node->id);
                ImVec2 flatten_pos = ImVec2(
                    (from_pos.x + to_pos.x) / 2.0f,
                    (from_pos.y + to_pos.y) / 2.0f
                );

                // Add to graph
                nodes_.push_back(flatten_node);
                pending_positions_[flatten_node.id] = flatten_pos;
                pending_positions_frames_ = 3;  // Apply position for 3 frames

                // Get flatten node's pin IDs
                int flatten_in = flatten_node.inputs[0].id;
                int flatten_out = flatten_node.outputs[0].id;

                // Create connections: from → flatten → to
                CreateLink(pending_flatten_from_pin_, flatten_in,
                         pending_flatten_from_node_, flatten_node.id);
                CreateLink(flatten_out, pending_flatten_to_pin_,
                         flatten_node.id, pending_flatten_to_node_);

                // Invalidate shape cache
                if (shape_inference_) {
                    shape_inference_->InvalidateShapes();
                }

                // Save undo state
                SaveUndoState();

                spdlog::info("Auto-inserted Flatten node between {} and {}",
                           from_node->name, to_node->name);
            }

            show_auto_insert_flatten_dialog_ = false;
            ImGui::CloseCurrentPopup();
        }

        ImGui::SameLine();

        // Button 2: Manual
        if (ImGui::Button(ICON_FA_HAND " Manual", ImVec2(95, 0))) {
            show_auto_insert_flatten_dialog_ = false;
            ImGui::CloseCurrentPopup();
        }

        ImGui::SameLine();

        // Button 3: Cancel
        if (ImGui::Button(ICON_FA_XMARK " Cancel", ImVec2(95, 0))) {
            show_auto_insert_flatten_dialog_ = false;
            ImGui::CloseCurrentPopup();
        }

        ImGui::EndPopup();
    }
}

void NodeEditor::ShowToolbar() {
    // Enhanced toolbar styling
    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(6, 4));
    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(4, 4));
    ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.22f, 0.24f, 0.28f, 1.0f));
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.30f, 0.32f, 0.38f, 1.0f));

    // Compile Graph - always at the very start so it's never clipped by narrow
    // toolbars. Blue button, always enabled when the callback is wired.
    if (compile_callback_) {
        ImGui::PopStyleColor(2);  // temporarily drop base button colors
        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.2f, 0.45f, 0.75f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.3f, 0.55f, 0.85f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.1f, 0.35f, 0.65f, 1.0f));
        if (ImGui::Button(ICON_FA_CHECK_DOUBLE " Compile")) {
            spdlog::info("NodeEditor: Compile Graph invoked from toolbar");
            compile_callback_();
        }
        ImGui::PopStyleColor(3);
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Compile the graph (F7) - validates structure and reports config without training");
        }
        ImGui::SameLine();

        // Local Debug - runs one forward + one backward pass on synthetic
        // data to catch runtime shape / NaN / dead-grad bugs BEFORE
        // training. Yellow-green distinguishes it from the blue Compile
        // button. Only rendered when the debug callback has been wired
        // (MainWindow sets this up alongside SetCompileCallback).
        if (debug_callback_) {
            ImGui::PushStyleColor(ImGuiCol_Button,
                                  ImVec4(0.55f, 0.70f, 0.20f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered,
                                  ImVec4(0.65f, 0.80f, 0.28f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_ButtonActive,
                                  ImVec4(0.45f, 0.60f, 0.15f, 1.0f));
            if (ImGui::Button(ICON_FA_BUG " Local Debug")) {
                spdlog::info("NodeEditor: Local Debug invoked from toolbar");
                debug_callback_();
            }
            ImGui::PopStyleColor(3);
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("Local Debug (F6) - run one forward + "
                                  "one backward pass on synthetic data. "
                                  "Catches shape / NaN / dead-gradient "
                                  "bugs before real training starts.");
            }
            ImGui::SameLine();
        }

        ImGui::TextColored(ImVec4(0.4f, 0.4f, 0.45f, 1.0f), "|");
        ImGui::SameLine();

        // Restore base button colors
        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.22f, 0.24f, 0.28f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.30f, 0.32f, 0.38f, 1.0f));
    }

    // File operations with icons
    if (ImGui::Button(ICON_FA_FLOPPY_DISK " Save")) {
        ShowSaveDialog();
    }
    ImGui::SameLine();

    if (ImGui::Button(ICON_FA_FOLDER_OPEN " Load")) {
        ShowLoadDialog();
    }
    ImGui::SameLine();

    ImGui::TextColored(ImVec4(0.4f, 0.4f, 0.45f, 1.0f), "|");
    ImGui::SameLine();

    // Zoom controls
    if (ImGui::Button(ICON_FA_MINUS)) {
        zoom_ = std::max(ZOOM_MIN, zoom_ - 0.1f);
    }
    ImGui::SameLine();
    ImGui::Text("%.0f%%", zoom_ * 100.0f);
    ImGui::SameLine();
    if (ImGui::Button(ICON_FA_PLUS)) {
        zoom_ = std::min(ZOOM_MAX, zoom_ + 0.1f);
    }
    ImGui::SameLine();
    if (ImGui::Button(ICON_FA_EXPAND " Fit")) {
        zoom_ = 1.0f; ImNodes::EditorContextResetPanning(ImVec2(0, 0));
    }
    ImGui::SameLine();

    ImGui::TextColored(ImVec4(0.4f, 0.4f, 0.45f, 1.0f), "|");
    ImGui::SameLine();

    // Selection tools
    if (ImGui::Button(ICON_FA_OBJECT_GROUP " Select All")) {
        SelectAll();
    }
    ImGui::SameLine();
    if (ImGui::Button(ICON_FA_TRASH " Delete")) {
        DeleteSelected();
    }
    ImGui::SameLine();
    if (ImGui::Button(ICON_FA_COPY " Duplicate")) {
        DuplicateSelection();
    }
    ImGui::SameLine();
    if (ImGui::Button(ICON_FA_ERASER " Clear All")) {
        ClearGraph();
    }
    ImGui::SameLine();

    ImGui::TextColored(ImVec4(0.4f, 0.4f, 0.45f, 1.0f), "|");
    ImGui::SameLine();

    // Custom Node Editor - opens the panel where users define their own node types
    if (open_custom_node_editor_callback_) {
        if (ImGui::Button(ICON_FA_WAND_MAGIC_SPARKLES " Node Editor")) {
            open_custom_node_editor_callback_();
        }
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Open the Custom Node Editor to create/edit custom node types (pins, parameters, code templates)");
        }
        ImGui::SameLine();

        ImGui::TextColored(ImVec4(0.4f, 0.4f, 0.45f, 1.0f), "|");
        ImGui::SameLine();
    }

    // Minimap toggle
    if (ImGui::Button(show_minimap_ ? ICON_FA_SITEMAP " Minimap" : ICON_FA_SITEMAP)) {
        show_minimap_ = !show_minimap_;
    }
    ImGui::SameLine();

    ImGui::TextColored(ImVec4(0.4f, 0.4f, 0.45f, 1.0f), "|");
    ImGui::SameLine();

    // Stats display
    ImGui::TextColored(ImVec4(0.6f, 0.7f, 0.8f, 1.0f), ICON_FA_CIRCLE_NODES " %zu", nodes_.size());
    ImGui::SameLine();
    ImGui::TextColored(ImVec4(0.6f, 0.7f, 0.8f, 1.0f), ICON_FA_LINK " %zu", links_.size());
    
    int num_selected = ImNodes::NumSelectedNodes();
    if (num_selected > 0) {
        ImGui::SameLine();
        ImGui::TextColored(ImVec4(0.5f, 0.8f, 1.0f, 1.0f), ICON_FA_SQUARE_CHECK " %d", num_selected);
    }

    ImGui::PopStyleColor(2);
    ImGui::PopStyleVar(2);

    // Code generation controls - second toolbar row
    ImGui::Separator();
    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(6, 4));
    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(4, 4));
    ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.22f, 0.24f, 0.28f, 1.0f));
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.30f, 0.32f, 0.38f, 1.0f));
    ImGui::TextColored(ImVec4(0.6f, 0.7f, 0.8f, 1.0f), ICON_FA_CODE " Code:");
    ImGui::SameLine();

    // Framework selection
    // Unified Canvas Phase 4.2: Execution mode selector
    const char* exec_modes[] = { "Code Gen", "Data Pipeline", "Local Training" };
    int current_exec_mode = static_cast<int>(execution_mode_);
    ImGui::SetNextItemWidth(130.0f);
    if (ImGui::Combo("##ExecMode", &current_exec_mode, exec_modes, 3)) {
        execution_mode_ = static_cast<ExecutionMode>(current_exec_mode);
        spdlog::info("Execution mode changed to: {}", exec_modes[current_exec_mode]);
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Execution Mode:\n"
            "Code Gen - Generate Python code\n"
            "Data Pipeline - Execute with DuckDB/Arrow\n"
            "Local Training - Train ML model locally");
    }
    ImGui::SameLine();

    // Show appropriate controls based on execution mode
    if (execution_mode_ == ExecutionMode::CodeGeneration) {
        // Framework selector for code generation
        const char* frameworks[] = { "PyTorch", "TensorFlow", "Keras", "PyCyxWiz" };
        int current_framework = static_cast<int>(selected_framework_);
        ImGui::SetNextItemWidth(120.0f);
        if (ImGui::Combo("##Framework", &current_framework, frameworks, 4)) {
            selected_framework_ = static_cast<CodeFramework>(current_framework);
            spdlog::info("Code generation framework changed to: {}", frameworks[current_framework]);
        }
        ImGui::SameLine();

        if (ImGui::Button(ICON_FA_GEARS " Generate")) {
            GeneratePythonCode();
        }
    } else if (execution_mode_ == ExecutionMode::DuckDBPipeline) {
        // Execute data pipeline button
        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.2f, 0.5f, 0.8f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.3f, 0.6f, 0.9f, 1.0f));
        if (ImGui::Button(ICON_FA_PLAY " Execute Pipeline")) {
            ExecuteDataPipeline();
        }
        ImGui::PopStyleColor(2);
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Execute data transformation pipeline using DuckDB");
        }
    }
    ImGui::SameLine();

    if (ImGui::Button(ICON_FA_FILE_EXPORT " Export")) {
        ShowExportDialog();
    }

    // Training controls
    ImGui::SameLine();
    ImGui::TextColored(ImVec4(0.4f, 0.4f, 0.45f, 1.0f), "|");
    ImGui::SameLine();

    // Check training state from TrainingManager
    auto& training_mgr = cyxwiz::TrainingManager::Instance();
    bool training_active = training_mgr.IsTrainingActive();

    if (training_active) {
        // Show training progress and stop button
        auto metrics = training_mgr.GetCurrentMetrics();
        ImGui::TextColored(ImVec4(0.4f, 1.0f, 0.4f, 1.0f), ICON_FA_SPINNER " Epoch %d/%d",
            metrics.current_epoch, metrics.total_epochs);
        ImGui::SameLine();
        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.6f, 0.2f, 0.2f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.7f, 0.3f, 0.3f, 1.0f));
        if (ImGui::Button(ICON_FA_STOP " Stop")) {
            training_mgr.StopTraining();
        }
        ImGui::PopStyleColor(2);
    } else {
        // Train button - green when valid, disabled when invalid
        // (Compile button is at the top of the toolbar — always visible regardless of window width)
        bool can_train = IsGraphValid() && train_callback_;
        if (!can_train) {
            ImGui::BeginDisabled();
        }

        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.2f, 0.6f, 0.2f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.3f, 0.7f, 0.3f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.1f, 0.5f, 0.1f, 1.0f));
        if (ImGui::Button(ICON_FA_PLAY " Train")) {
            if (train_callback_) {
                spdlog::info("NodeEditor: Starting training from graph");
                train_callback_(nodes_, links_);
            }
        }
        ImGui::PopStyleColor(3);

        if (!can_train) {
            ImGui::EndDisabled();
            if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) {
                if (!train_callback_) {
                    ImGui::SetTooltip("Training callback not set (no dataset loaded?)");
                } else {
                    ImGui::SetTooltip("Graph is not valid for training. Need: DatasetInput -> Model layers -> Loss");
                }
            }
        }
    }

    // Simulation controls (for signal/MuJoCo Plant graphs)
    if (HasSimulationNodes()) {
        ImGui::SameLine();
        ImGui::TextColored(ImVec4(0.4f, 0.4f, 0.45f, 1.0f), "|");
        ImGui::SameLine();

        if (is_simulating_) {
            ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.6f, 0.2f, 0.2f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.7f, 0.3f, 0.3f, 1.0f));
            if (ImGui::Button(ICON_FA_STOP " Stop Sim")) {
                OnStopSimulation();
            }
            ImGui::PopStyleColor(2);
            ImGui::SameLine();
            if (graph_executor_) {
                ImGui::TextColored(ImVec4(0.4f, 1.0f, 0.4f, 1.0f), "t=%.2fs",
                    graph_executor_->GetSimTime());
            }
        } else {
            ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.15f, 0.45f, 0.6f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.2f, 0.55f, 0.7f, 1.0f));
            bool rl_running = rl_executor_ && rl_executor_->IsTraining();
            if (rl_running) {
                ImGui::BeginDisabled();
                ImGui::Button(ICON_FA_PLAY " Run Sim");
                ImGui::EndDisabled();
                if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) {
                    ImGui::SetTooltip("Stop RL training first");
                }
            } else if (ImGui::Button(ICON_FA_PLAY " Run Sim")) {
                OnRunSimulation();
            }
            ImGui::PopStyleColor(2);
        }
    }

    // RL Training controls (for graphs with RL nodes)
    if (HasRLNodes()) {
        ImGui::SameLine();
        ImGui::TextColored(ImVec4(0.4f, 0.4f, 0.45f, 1.0f), "|");
        ImGui::SameLine();

        if (rl_script_running_ || (rl_executor_ && rl_executor_->IsTraining())) {
            ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.6f, 0.2f, 0.2f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.7f, 0.3f, 0.3f, 1.0f));
            if (ImGui::Button(ICON_FA_STOP " Stop RL")) {
                OnStopRLTraining();
            }
            ImGui::PopStyleColor(2);
            ImGui::SameLine();
            if (rl_script_running_) {
                ImGui::TextColored(ImVec4(0.4f, 1.0f, 0.4f, 1.0f), "Training via Python...");
            } else if (rl_executor_) {
                auto metrics = rl_executor_->GetMetrics();
                ImGui::TextColored(ImVec4(0.4f, 1.0f, 0.4f, 1.0f), "Ep %d | R:%.1f",
                    metrics.episode_count, metrics.mean_episode_reward);
            }
        } else {
            ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.15f, 0.5f, 0.3f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.2f, 0.6f, 0.4f, 1.0f));
            if (is_simulating_) {
                ImGui::BeginDisabled();
                ImGui::Button(ICON_FA_PLAY " Train RL");
                ImGui::EndDisabled();
                if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) {
                    ImGui::SetTooltip("Stop graph simulation first");
                }
            } else if (ImGui::Button(ICON_FA_PLAY " Train RL")) {
                OnStartRLTraining();
            }
            ImGui::PopStyleColor(2);
        }
    }

    // Export Policy (ONNX) button
    if (HasRLNodes()) {
        ImGui::SameLine();
        bool has_trained = rl_executor_ && !rl_executor_->IsTraining() && rl_executor_->GetMetrics().episode_count > 0;
        if (!has_trained) {
            ImGui::BeginDisabled();
            ImGui::Button(ICON_FA_FILE_EXPORT " Export ONNX");
            if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) {
                ImGui::SetTooltip("Train an RL agent first");
            }
            ImGui::EndDisabled();
        } else {
            ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.5f, 0.3f, 0.1f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.6f, 0.4f, 0.2f, 1.0f));
            if (ImGui::Button(ICON_FA_FILE_EXPORT " Export ONNX")) {
                export_onnx_dialog_open_ = true;
            }
            ImGui::PopStyleColor(2);
        }
    }

    // Sim performance metrics (Phase 4.7)
    if (is_simulating_ && last_eval_time_ms_ > 0.0f) {
        ImGui::SameLine();
        ImGui::TextColored(ImVec4(0.4f, 0.4f, 0.45f, 1.0f), "|");
        ImGui::SameLine();
        float fps = (last_eval_time_ms_ > 0.001f) ? 1000.0f / last_eval_time_ms_ : 0.0f;
        ImVec4 color = (last_eval_time_ms_ < 16.0f) ? ImVec4(0.4f, 1.0f, 0.4f, 1.0f) : ImVec4(1.0f, 0.6f, 0.2f, 1.0f);
        ImGui::TextColored(color, "%.1fms (%.0f FPS)", last_eval_time_ms_, fps);
    }

    ImGui::PopStyleColor(2);
    ImGui::PopStyleVar(2);

    // ONNX Export dialog
    if (export_onnx_dialog_open_) {
        ImGui::OpenPopup("Export Policy (ONNX)");
        export_onnx_dialog_open_ = false;
    }
    if (ImGui::BeginPopupModal("Export Policy (ONNX)", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
        ImGui::Text("Export trained RL policy to ONNX format.");
        ImGui::Separator();

        static char onnx_path[512] = "policy.onnx";
        ImGui::Text("Output path:");
        ImGui::InputText("##onnx_path", onnx_path, sizeof(onnx_path));
        ImGui::Spacing();

        if (ImGui::Button("Export", ImVec2(120, 0))) {
            ExportPolicyONNX(std::string(onnx_path));
            ImGui::CloseCurrentPopup();
        }
        ImGui::SameLine();
        if (ImGui::Button("Cancel", ImVec2(120, 0))) {
            ImGui::CloseCurrentPopup();
        }
        ImGui::EndPopup();
    }
}

void NodeEditor::RenderMinimap() {
    if (nodes_.empty()) return;

    // Get parent window position and size for calculating minimap position
    ImVec2 parent_window_pos = ImGui::GetWindowPos();
    ImVec2 parent_window_size = ImGui::GetWindowSize();

    // Get content region to properly account for toolbar/title bar
    ImVec2 content_min = ImGui::GetWindowContentRegionMin();
    ImVec2 content_max = ImGui::GetWindowContentRegionMax();
    // Additional offset for the toolbar and separator rendered before the node canvas
    float toolbar_offset = ImGui::GetFrameHeight() + ImGui::GetStyle().ItemSpacing.y + 10.0f;
    float content_top = parent_window_pos.y + content_min.y + toolbar_offset;
    float content_bottom = parent_window_pos.y + content_max.y;
    float content_left = parent_window_pos.x + content_min.x;
    float content_right = parent_window_pos.x + content_max.x;

    // Define corner positions
    const float padding = 10.0f;

    auto getCornerPos = [&](MinimapPosition pos) -> ImVec2 {
        switch (pos) {
            case MinimapPosition::TopLeft:
                return ImVec2(content_left + padding, content_top + padding);
            case MinimapPosition::TopRight:
                return ImVec2(content_right - minimap_size_.x - padding, content_top + padding);
            case MinimapPosition::BottomLeft:
                return ImVec2(content_left + padding, content_bottom - minimap_size_.y - padding);
            case MinimapPosition::BottomRight:
            default:
                return ImVec2(content_right - minimap_size_.x - padding, content_bottom - minimap_size_.y - padding);
        }
    };

    // Calculate minimap position (always use fixed corner position)
    ImVec2 minimap_pos = getCornerPos(minimap_position_);

    // Set next window position and create a floating window for the minimap
    ImGui::SetNextWindowPos(minimap_pos, ImGuiCond_Always);
    ImGui::SetNextWindowSize(minimap_size_, ImGuiCond_Always);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 6.0f);
    ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.10f, 0.10f, 0.12f, 0.92f));
    ImGui::PushStyleColor(ImGuiCol_TitleBg, ImVec4(0.18f, 0.18f, 0.22f, 1.0f));
    ImGui::PushStyleColor(ImGuiCol_TitleBgActive, ImVec4(0.22f, 0.22f, 0.28f, 1.0f));
    ImGui::PushStyleColor(ImGuiCol_Border, ImVec4(0.35f, 0.40f, 0.50f, 0.8f));

    ImGuiWindowFlags window_flags = ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoScrollbar |
                                    ImGuiWindowFlags_NoScrollWithMouse | ImGuiWindowFlags_NoCollapse |
                                    ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoDocking |
                                    ImGuiWindowFlags_NoMove;  // We handle movement manually


    if (ImGui::Begin("##MinimapWindow", &show_minimap_, window_flags)) {
        // Draw subtle grid pattern on background
        ImDrawList* bg_draw_list = ImGui::GetWindowDrawList();
        ImVec2 win_pos = ImGui::GetWindowPos();
        ImVec2 win_size = ImGui::GetWindowSize();
        const float grid_step = 20.0f;
        ImU32 grid_color = IM_COL32(60, 65, 75, 40);
        for (float x = win_pos.x; x < win_pos.x + win_size.x; x += grid_step) {
            bg_draw_list->AddLine(ImVec2(x, win_pos.y), ImVec2(x, win_pos.y + win_size.y), grid_color);
        }
        for (float y = win_pos.y; y < win_pos.y + win_size.y; y += grid_step) {
            bg_draw_list->AddLine(ImVec2(win_pos.x, y), ImVec2(win_pos.x + win_size.x, y), grid_color);
        }
        
        // Draw stats header bar at top
        const float header_height = 16.0f;
        bg_draw_list->AddRectFilled(
            win_pos, 
            ImVec2(win_pos.x + win_size.x, win_pos.y + header_height),
            IM_COL32(30, 35, 45, 220)
        );
        bg_draw_list->AddLine(
            ImVec2(win_pos.x, win_pos.y + header_height),
            ImVec2(win_pos.x + win_size.x, win_pos.y + header_height),
            IM_COL32(60, 70, 90, 200)
        );
        
        // Draw stats text
        char stats_text[64];
        snprintf(stats_text, sizeof(stats_text), "%zu nodes | %zu links", nodes_.size(), links_.size());
        ImVec2 text_size = ImGui::CalcTextSize(stats_text);
        bg_draw_list->AddText(
            ImVec2(win_pos.x + (win_size.x - text_size.x) * 0.5f, win_pos.y + 2.0f),
            IM_COL32(160, 170, 190, 220),
            stats_text
        );
        
        // Get the draw list for this window
        ImDrawList* draw_list = ImGui::GetWindowDrawList();
        ImVec2 window_pos = ImGui::GetWindowPos();
        ImVec2 window_size = ImGui::GetWindowSize();

        // Store screen-space bounds for input blocking in Render()
        minimap_screen_min_ = window_pos;
        minimap_screen_max_ = ImVec2(window_pos.x + window_size.x, window_pos.y + window_size.y);

        // Check if mouse is over minimap window (for external use)
        ImVec2 mouse_pos = ImGui::GetMousePos();
        mouse_over_minimap_ = ImGui::IsWindowHovered(ImGuiHoveredFlags_AllowWhenBlockedByActiveItem);

        // Add invisible button covering the entire window to capture all mouse input
        // This prevents mouse events from passing through to the node editor canvas
        ImGui::SetCursorPos(ImVec2(0, 0));
        ImGui::InvisibleButton("##MinimapInputCapture", window_size);

    // Calculate bounding box of all nodes in grid space
    float min_x = FLT_MAX, min_y = FLT_MAX;
    float max_x = -FLT_MAX, max_y = -FLT_MAX;

    for (const auto& node : nodes_) {
        ImVec2 node_pos = ImNodes::GetNodeGridSpacePos(node.id);
        ImVec2 node_dims = ImNodes::GetNodeDimensions(node.id);

        min_x = std::min(min_x, node_pos.x);
        min_y = std::min(min_y, node_pos.y);
        max_x = std::max(max_x, node_pos.x + node_dims.x);
        max_y = std::max(max_y, node_pos.y + node_dims.y);
    }

    // Add padding to bounds
    const float bounds_padding = 100.0f;
    min_x -= bounds_padding;
    min_y -= bounds_padding;
    max_x += bounds_padding;
    max_y += bounds_padding;

    float graph_width = max_x - min_x;
    float graph_height = max_y - min_y;

    // Calculate scale to fit graph in minimap
    float scale_x = (minimap_size_.x - 4.0f) / graph_width;
    float scale_y = (minimap_size_.y - 4.0f) / graph_height;
    float scale = std::min(scale_x, scale_y);

    // Center the graph in minimap content area
    float offset_x = (window_size.x - graph_width * scale) * 0.5f;
    float offset_y = (window_size.y - graph_height * scale) * 0.5f;

    // Lambda to convert grid space to minimap space
    auto gridToMinimap = [&](ImVec2 grid_pos) -> ImVec2 {
        return ImVec2(
            window_pos.x + offset_x + (grid_pos.x - min_x) * scale,
            window_pos.y + offset_y + (grid_pos.y - min_y) * scale
        );
    };

    // Minimap content area rect (window already provides background)
    ImVec2 minimap_content_min = window_pos;
    ImVec2 minimap_content_max = ImVec2(window_pos.x + window_size.x, window_pos.y + window_size.y);

    // Draw links first (underneath nodes)
    for (const auto& link : links_) {
        // Find source and destination nodes
        const MLNode* from_node = nullptr;
        const MLNode* to_node = nullptr;
        ImVec2 from_pos, to_pos;

        for (const auto& node : nodes_) {
            // Check output pins for source
            for (const auto& pin : node.outputs) {
                if (pin.id == link.from_pin) {
                    from_node = &node;
                    ImVec2 node_pos = ImNodes::GetNodeGridSpacePos(node.id);
                    ImVec2 node_dims = ImNodes::GetNodeDimensions(node.id);
                    from_pos = ImVec2(node_pos.x + node_dims.x, node_pos.y + node_dims.y * 0.5f);
                    break;
                }
            }
            // Check input pins for destination
            for (const auto& pin : node.inputs) {
                if (pin.id == link.to_pin) {
                    to_node = &node;
                    ImVec2 node_pos = ImNodes::GetNodeGridSpacePos(node.id);
                    ImVec2 node_dims = ImNodes::GetNodeDimensions(node.id);
                    to_pos = ImVec2(node_pos.x, node_pos.y + node_dims.y * 0.5f);
                    break;
                }
            }
            if (from_node && to_node) break;
        }

        if (from_node && to_node) {
            ImVec2 mm_from = gridToMinimap(from_pos);
            ImVec2 mm_to = gridToMinimap(to_pos);
            // Use a gradient-like effect: brighter in middle
            ImU32 link_color = is_training_ ? 
                IM_COL32(200, 220, 100, 180) :  // Amber-ish during training
                IM_COL32(130, 160, 200, 160);   // Blue-gray normally
            draw_list->AddLine(mm_from, mm_to, link_color, 1.5f);
        }
    }

    // Draw nodes
    for (const auto& node : nodes_) {
        ImVec2 node_pos = ImNodes::GetNodeGridSpacePos(node.id);
        ImVec2 node_dims = ImNodes::GetNodeDimensions(node.id);

        ImVec2 mm_pos = gridToMinimap(node_pos);
        ImVec2 mm_size = ImVec2(
            std::max(4.0f, node_dims.x * scale),
            std::max(3.0f, node_dims.y * scale)
        );

        // Get node color based on type
        unsigned int color = GetNodeColor(node.type);
        ImU32 fill_color = IM_COL32(
            (color >> 0) & 0xFF,
            (color >> 8) & 0xFF,
            (color >> 16) & 0xFF,
            200
        );

        // Check if node is selected (support multi-selection)
        bool is_selected = ImNodes::IsNodeSelected(node.id);

        // Draw node with rounded corners
        draw_list->AddRectFilled(mm_pos, ImVec2(mm_pos.x + mm_size.x, mm_pos.y + mm_size.y), fill_color, 3.0f);
        
        // Draw subtle border for all nodes
        draw_list->AddRect(mm_pos, ImVec2(mm_pos.x + mm_size.x, mm_pos.y + mm_size.y), 
            IM_COL32(255, 255, 255, 40), 3.0f, 0, 1.0f);

        if (is_selected) {
            // Blue glow effect matching main editor
            for (int i = 2; i >= 0; --i) {
                float offset = (i + 1) * 1.5f;
                int alpha = 40 * (3 - i);  // 120, 80, 40
                draw_list->AddRect(
                    ImVec2(mm_pos.x - offset, mm_pos.y - offset),
                    ImVec2(mm_pos.x + mm_size.x + offset, mm_pos.y + mm_size.y + offset),
                    IM_COL32(100, 180, 255, alpha), 3.0f, 0, 1.5f
                );
            }
            // Bright selection border
            draw_list->AddRect(mm_pos, ImVec2(mm_pos.x + mm_size.x, mm_pos.y + mm_size.y), 
                IM_COL32(100, 180, 255, 255), 3.0f, 0, 2.0f);
        }
    }

    // Draw viewport rectangle
    ImVec2 panning = ImNodes::EditorContextGetPanning();
    ImVec2 canvas_size = ImGui::GetContentRegionAvail();

    // Viewport in grid space (note: panning is negated)
    ImVec2 viewport_min_grid = ImVec2(-panning.x, -panning.y);
    ImVec2 viewport_max_grid = ImVec2(-panning.x + canvas_size.x, -panning.y + canvas_size.y);

    ImVec2 viewport_mm_min = gridToMinimap(viewport_min_grid);
    ImVec2 viewport_mm_max = gridToMinimap(viewport_max_grid);

    // Clamp viewport rect to minimap bounds
    viewport_mm_min.x = std::max(viewport_mm_min.x, window_pos.x);
    viewport_mm_min.y = std::max(viewport_mm_min.y, window_pos.y);
    viewport_mm_max.x = std::min(viewport_mm_max.x, window_pos.x + window_size.x);
    viewport_mm_max.y = std::min(viewport_mm_max.y, window_pos.y + window_size.y);

    // Draw semi-transparent viewport indicator with enhanced styling
    draw_list->AddRectFilled(viewport_mm_min, viewport_mm_max, IM_COL32(80, 140, 255, 35));
    
    // Draw viewport border with rounded corners
    draw_list->AddRect(viewport_mm_min, viewport_mm_max, IM_COL32(100, 160, 255, 220), 2.0f, 0, 2.0f);
    
    // Draw corner handles for visual emphasis
    const float handle_size = 4.0f;
    ImU32 handle_color = IM_COL32(130, 180, 255, 255);
    
    // Top-left corner
    draw_list->AddRectFilled(
        ImVec2(viewport_mm_min.x - 1, viewport_mm_min.y - 1),
        ImVec2(viewport_mm_min.x + handle_size, viewport_mm_min.y + handle_size),
        handle_color
    );
    // Top-right corner
    draw_list->AddRectFilled(
        ImVec2(viewport_mm_max.x - handle_size, viewport_mm_min.y - 1),
        ImVec2(viewport_mm_max.x + 1, viewport_mm_min.y + handle_size),
        handle_color
    );
    // Bottom-left corner
    draw_list->AddRectFilled(
        ImVec2(viewport_mm_min.x - 1, viewport_mm_max.y - handle_size),
        ImVec2(viewport_mm_min.x + handle_size, viewport_mm_max.y + 1),
        handle_color
    );
    // Bottom-right corner
    draw_list->AddRectFilled(
        ImVec2(viewport_mm_max.x - handle_size, viewport_mm_max.y - handle_size),
        ImVec2(viewport_mm_max.x + 1, viewport_mm_max.y + 1),
        handle_color
    );

    // Handle mouse interaction with minimap using the window system
    // mouse_pos already declared above, just refresh it
    mouse_pos = ImGui::GetMousePos();
    bool mouse_in_minimap = ImGui::IsWindowHovered(ImGuiHoveredFlags_AllowWhenBlockedByActiveItem);

    // Handle ongoing navigation drag
    if (minimap_navigating_) {
        if (ImGui::IsMouseDown(ImGuiMouseButton_Left)) {
            // Convert minimap position to grid position
            float rel_x = (mouse_pos.x - window_pos.x - offset_x) / scale + min_x;
            float rel_y = (mouse_pos.y - window_pos.y - offset_y) / scale + min_y;

            // Center viewport on clicked position
            ImVec2 new_panning = ImVec2(
                -(rel_x - canvas_size.x * 0.5f),
                -(rel_y - canvas_size.y * 0.5f)
            );

            ImNodes::EditorContextResetPanning(new_panning);
            ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeAll);
        } else {
            minimap_navigating_ = false;
        }
    }

    // Handle interactions when mouse is in minimap window
    if (mouse_in_minimap && !minimap_navigating_) {
        ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeAll);

        // Draw crosshair at mouse position
        const float crosshair_size = 8.0f;
        ImU32 crosshair_color = IM_COL32(255, 255, 255, 200);

        draw_list->AddLine(
            ImVec2(mouse_pos.x - crosshair_size, mouse_pos.y),
            ImVec2(mouse_pos.x + crosshair_size, mouse_pos.y),
            crosshair_color, 1.5f
        );
        draw_list->AddLine(
            ImVec2(mouse_pos.x, mouse_pos.y - crosshair_size),
            ImVec2(mouse_pos.x, mouse_pos.y + crosshair_size),
            crosshair_color, 1.5f
        );
        draw_list->AddCircleFilled(mouse_pos, 2.0f, crosshair_color);

        // Handle left-click for navigation
        if (ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
            minimap_navigating_ = true;

            // Convert minimap position to grid position
            float rel_x = (mouse_pos.x - window_pos.x - offset_x) / scale + min_x;
            float rel_y = (mouse_pos.y - window_pos.y - offset_y) / scale + min_y;

            // Center viewport on clicked position
            ImVec2 new_panning = ImVec2(
                -(rel_x - canvas_size.x * 0.5f),
                -(rel_y - canvas_size.y * 0.5f)
            );

            ImNodes::EditorContextResetPanning(new_panning);
        }

        // Show tooltip
        if (!minimap_navigating_) {
            ImGui::BeginTooltip();
            ImGui::Text("Click to navigate | Right-click: options");
            ImGui::EndTooltip();
        }

        // Right-click context menu
        if (ImGui::IsMouseClicked(ImGuiMouseButton_Right)) {
            ImGui::OpenPopup("MinimapContextMenu");
        }
    }

    // Render minimap context menu
    if (ImGui::BeginPopup("MinimapContextMenu")) {
        ImGui::Text("Minimap Position");
        ImGui::Separator();

        if (ImGui::MenuItem("Top Left", nullptr, minimap_position_ == MinimapPosition::TopLeft)) {
            minimap_position_ = MinimapPosition::TopLeft;
        }
        if (ImGui::MenuItem("Top Right", nullptr, minimap_position_ == MinimapPosition::TopRight)) {
            minimap_position_ = MinimapPosition::TopRight;
        }
        if (ImGui::MenuItem("Bottom Left", nullptr, minimap_position_ == MinimapPosition::BottomLeft)) {
            minimap_position_ = MinimapPosition::BottomLeft;
        }
        if (ImGui::MenuItem("Bottom Right", nullptr, minimap_position_ == MinimapPosition::BottomRight)) {
            minimap_position_ = MinimapPosition::BottomRight;
        }

        ImGui::Separator();
        if (ImGui::MenuItem("Hide Minimap")) {
            show_minimap_ = false;
        }

        ImGui::EndPopup();
    }

    // Draw navigation arrows when not hovered (visual hint)
    if (!mouse_in_minimap) {
        const float arrow_size = 6.0f;
        const float arrow_padding = 6.0f;
        ImU32 arrow_color = IM_COL32(120, 120, 130, 150);

        // Draw 4-way arrow icon at bottom right corner
        ImVec2 icon_pos = ImVec2(
            window_pos.x + window_size.x - arrow_padding - arrow_size,
            window_pos.y + window_size.y - arrow_padding - arrow_size
        );

        // Up arrow
        draw_list->AddTriangleFilled(
            ImVec2(icon_pos.x, icon_pos.y - arrow_size),
            ImVec2(icon_pos.x - 3.0f, icon_pos.y - 2.0f),
            ImVec2(icon_pos.x + 3.0f, icon_pos.y - 2.0f),
            arrow_color
        );
        // Down arrow
        draw_list->AddTriangleFilled(
            ImVec2(icon_pos.x, icon_pos.y + arrow_size),
            ImVec2(icon_pos.x - 3.0f, icon_pos.y + 2.0f),
            ImVec2(icon_pos.x + 3.0f, icon_pos.y + 2.0f),
            arrow_color
        );
        // Left arrow
        draw_list->AddTriangleFilled(
            ImVec2(icon_pos.x - arrow_size, icon_pos.y),
            ImVec2(icon_pos.x - 2.0f, icon_pos.y - 3.0f),
            ImVec2(icon_pos.x - 2.0f, icon_pos.y + 3.0f),
            arrow_color
        );
        // Right arrow
        draw_list->AddTriangleFilled(
            ImVec2(icon_pos.x + arrow_size, icon_pos.y),
            ImVec2(icon_pos.x + 2.0f, icon_pos.y - 3.0f),
            ImVec2(icon_pos.x + 2.0f, icon_pos.y + 3.0f),
            arrow_color
        );
    }
    }  // End ImGui::Begin("##MinimapWindow")
    ImGui::End();
    ImGui::PopStyleColor(4);
    ImGui::PopStyleVar(2);
}

// Unified Canvas Phase 6: Helper function to get pin type name as string
static std::string GetPinTypeName(PinType type) {
    if (type == PinType::Tensor) return "Tensor";
    if (type == PinType::Labels) return "Labels";
    if (type == PinType::Parameters) return "Parameters";
    if (type == PinType::Loss) return "Loss";
    if (type == PinType::Optimizer) return "Optimizer";
    if (type == PinType::Dataset) return "Dataset";
    return "Unknown";
}

void NodeEditor::RebuildPinLookup() {
    pin_lookup_.clear();
    for (auto& node : nodes_) {
        for (auto& pin : node.inputs) {
            pin_lookup_[pin.id] = std::make_pair(&node, &pin);
        }
        for (auto& pin : node.outputs) {
            pin_lookup_[pin.id] = std::make_pair(&node, &pin);
        }
    }
}

void NodeEditor::RenderNodes() {
    // Update execution pulse animation
    execution_pulse_time_ += ImGui::GetIO().DeltaTime;

    // Render all nodes
    for (const auto& node : nodes_) {
        // Unified Canvas Phase 6: Check execution state for highlighting
        auto exec_state_it = node_execution_states_.find(node.id);
        bool has_exec_state = (exec_state_it != node_execution_states_.end());
        NodeExecutionState exec_state = has_exec_state ? exec_state_it->second : NodeExecutionState::Idle;

        // Set node color based on execution state OR type
        ImU32 title_color;
        if (exec_state == NodeExecutionState::Executing) {
            // Pulsing blue for executing node
            float pulse = 0.5f + 0.5f * std::sin(execution_pulse_time_ * 4.0f);
            ImU32 base = IM_COL32(30, 100, 200, 255);
            ImU32 highlight = IM_COL32(60, 150, 255, 255);
            title_color = ImGui::ColorConvertFloat4ToU32(ImVec4(
                (base & 0xFF) / 255.0f * (1 - pulse) + (highlight & 0xFF) / 255.0f * pulse,
                ((base >> 8) & 0xFF) / 255.0f * (1 - pulse) + ((highlight >> 8) & 0xFF) / 255.0f * pulse,
                ((base >> 16) & 0xFF) / 255.0f * (1 - pulse) + ((highlight >> 16) & 0xFF) / 255.0f * pulse,
                1.0f
            ));
        } else if (exec_state == NodeExecutionState::Completed) {
            // Green for completed
            title_color = IM_COL32(50, 150, 70, 255);
        } else if (exec_state == NodeExecutionState::Error) {
            // Red for error
            title_color = IM_COL32(200, 50, 50, 255);
        } else {
            // Normal color based on type
            title_color = GetNodeColor(node.type);
        }

        // ===== KNIME-STYLE RENDERING for ALL Nodes =====
        bool is_knime_style = true;  // Apply to all node types

        if (is_knime_style) {
            // Make ImNodes node completely invisible - we draw our own icon box
            ImU32 transparent = IM_COL32(0, 0, 0, 0);
            // Push pin closer to node content (negative offset pulls it inward)
            ImNodes::PushStyleVar(ImNodesStyleVar_PinOffset, 0.0f);
            // Remove node padding so pins sit flush with the icon box
            ImNodes::PushStyleVar(ImNodesStyleVar_NodePadding, ImVec2(0.0f, 0.0f));
            ImNodes::PushColorStyle(ImNodesCol_TitleBar, transparent);
            ImNodes::PushColorStyle(ImNodesCol_TitleBarHovered, transparent);
            ImNodes::PushColorStyle(ImNodesCol_TitleBarSelected, transparent);
            ImNodes::PushColorStyle(ImNodesCol_NodeBackground, transparent);
            ImNodes::PushColorStyle(ImNodesCol_NodeBackgroundHovered, transparent);
            ImNodes::PushColorStyle(ImNodesCol_NodeBackgroundSelected, transparent);
            ImNodes::PushColorStyle(ImNodesCol_NodeOutline, transparent);
        } else {
            ImNodes::PushColorStyle(ImNodesCol_TitleBar, title_color);
            ImNodes::PushColorStyle(ImNodesCol_TitleBarHovered, title_color);
            ImNodes::PushColorStyle(ImNodesCol_TitleBarSelected, title_color);
        }

        ImNodes::BeginNode(node.id);

        if (is_knime_style) {
            // Disable default node drag for KNIME nodes; we drag via icon handle
            ImNodes::SetNodeDraggable(node.id, false);

            // Settings - match Node Browser exactly
            const float ICON_BOX_SIZE = 64.0f * zoom_;
            const float CORNER_RADIUS = 6.0f;
            const float BORDER_THICKNESS = 2.0f;

            bool show_name = true;
            auto show_name_it = node.parameters.find("show_name");
            if (show_name_it != node.parameters.end()) {
                show_name = (show_name_it->second == "true");
            }

            // Minimal invisible title bar (required by ImNodes)
            ImNodes::BeginNodeTitleBar();
            ImGui::Dummy(ImVec2(ICON_BOX_SIZE, 1));
            ImNodes::EndNodeTitleBar();

            // Standard horizontal layout: [input] [icon] [output]
            // Using BeginGroup to keep everything on one line
            ImGui::BeginGroup();

            // Input pins (left side)
            if (!node.inputs.empty()) {
                ImGui::BeginGroup();
                // Vertical centering for ALL input pins
                const float PIN_HEIGHT = 12.0f;
                float total_pin_height = node.inputs.size() * PIN_HEIGHT;
                float vert_offset = (ICON_BOX_SIZE - total_pin_height) * 0.5f;
                if (vert_offset > 0) ImGui::Dummy(ImVec2(0, vert_offset));
                for (const auto& pin : node.inputs) {
                    bool is_running = (exec_state == NodeExecutionState::Executing);
                    bool has_error = (exec_state == NodeExecutionState::Error);

                    // Pin color is driven by the node's compile/train state.
                    // Default = red hollow, CompileFailed = red solid,
                    // CompilePassed = green hollow, Trained = green solid.
                    ImU32 pin_color;
                    ImU32 pin_hover;
                    ImNodesPinShape pin_shape;
                    switch (GetNodePinState(node.id)) {
                        case NodePinState::CompileFailed:
                            pin_color = IM_COL32(220, 50, 50, 255);
                            pin_hover = IM_COL32(255, 80, 80, 255);
                            pin_shape = ImNodesPinShape_CircleFilled;
                            break;
                        case NodePinState::CompilePassed:
                            pin_color = IM_COL32(50, 200, 50, 255);
                            pin_hover = IM_COL32(80, 230, 80, 255);
                            pin_shape = ImNodesPinShape_Circle;
                            break;
                        case NodePinState::Trained:
                            pin_color = IM_COL32(50, 200, 50, 255);
                            pin_hover = IM_COL32(80, 230, 80, 255);
                            pin_shape = ImNodesPinShape_CircleFilled;
                            break;
                        case NodePinState::Default:
                        default:
                            pin_color = IM_COL32(220, 50, 50, 255);
                            pin_hover = IM_COL32(255, 80, 80, 255);
                            pin_shape = ImNodesPinShape_Circle;
                            break;
                    }

                    // Execution state takes highest priority
                    if (has_error) {
                        pin_color = IM_COL32(220, 180, 50, 255);
                        pin_hover = IM_COL32(240, 200, 80, 255);
                        pin_shape = ImNodesPinShape_CircleFilled;
                    } else if (is_running) {
                        pin_color = IM_COL32(50, 200, 50, 255);
                        pin_hover = IM_COL32(80, 230, 80, 255);
                        pin_shape = ImNodesPinShape_CircleFilled;
                    }

                    ImNodes::PushColorStyle(ImNodesCol_Pin, pin_color);
                    ImNodes::PushColorStyle(ImNodesCol_PinHovered, pin_hover);
                    ImNodes::BeginInputAttribute(pin.id, pin_shape);
                    ImGui::Dummy(ImVec2(0, 12));
                    ImNodes::EndInputAttribute();
                    ImNodes::PopColorStyle();
                    ImNodes::PopColorStyle();
                }
                ImGui::EndGroup();
                ImGui::SameLine(0, 0);  // Exactly at icon edge
            }

            // Icon box (center)
            ImVec2 icon_pos = ImGui::GetCursorScreenPos();
            ImGui::Dummy(ImVec2(ICON_BOX_SIZE, ICON_BOX_SIZE));

            // Output pins (right side) - positioned outside icon for click detection
            if (!node.outputs.empty()) {
                ImGui::SameLine(0, 0);  // No gap so pin is tied to icon edge
                ImGui::BeginGroup();
                // Vertical centering for ALL output pins
                const float PIN_HEIGHT = 10.0f;
                float total_pin_height = node.outputs.size() * PIN_HEIGHT;
                float vert_offset = (ICON_BOX_SIZE - total_pin_height) * 0.5f;
                if (vert_offset > 0) ImGui::Dummy(ImVec2(0, vert_offset));
                for (const auto& pin : node.outputs) {
                    bool is_running = (exec_state == NodeExecutionState::Executing);
                    bool has_error = (exec_state == NodeExecutionState::Error);

                    // Pin color from node compile/train state (see input pin block above).
                    ImU32 pin_color;
                    ImU32 pin_hover;
                    ImNodesPinShape pin_shape;
                    switch (GetNodePinState(node.id)) {
                        case NodePinState::CompileFailed:
                            pin_color = IM_COL32(220, 50, 50, 255);
                            pin_hover = IM_COL32(255, 80, 80, 255);
                            pin_shape = ImNodesPinShape_CircleFilled;
                            break;
                        case NodePinState::CompilePassed:
                            pin_color = IM_COL32(50, 200, 50, 255);
                            pin_hover = IM_COL32(80, 230, 80, 255);
                            pin_shape = ImNodesPinShape_Circle;
                            break;
                        case NodePinState::Trained:
                            pin_color = IM_COL32(50, 200, 50, 255);
                            pin_hover = IM_COL32(80, 230, 80, 255);
                            pin_shape = ImNodesPinShape_CircleFilled;
                            break;
                        case NodePinState::Default:
                        default:
                            pin_color = IM_COL32(220, 50, 50, 255);
                            pin_hover = IM_COL32(255, 80, 80, 255);
                            pin_shape = ImNodesPinShape_Circle;
                            break;
                    }

                    // Execution state takes highest priority
                    if (has_error) {
                        pin_color = IM_COL32(220, 180, 50, 255);
                        pin_hover = IM_COL32(240, 200, 80, 255);
                        pin_shape = ImNodesPinShape_CircleFilled;
                    } else if (is_running) {
                        pin_color = IM_COL32(50, 200, 50, 255);
                        pin_hover = IM_COL32(80, 230, 80, 255);
                        pin_shape = ImNodesPinShape_CircleFilled;
                    }

                    ImNodes::PushColorStyle(ImNodesCol_Pin, pin_color);
                    ImNodes::PushColorStyle(ImNodesCol_PinHovered, pin_hover);
                    ImNodes::BeginOutputAttribute(pin.id, pin_shape);
                    ImGui::Dummy(ImVec2(0, 10));
                    ImNodes::EndOutputAttribute();
                    ImNodes::PopColorStyle();
                    ImNodes::PopColorStyle();
                }
                ImGui::EndGroup();
            }

            ImGui::EndGroup();

            // Draw custom icon box
            ImDrawList* draw_list = ImGui::GetWindowDrawList();
            ImVec2 icon_max(icon_pos.x + ICON_BOX_SIZE, icon_pos.y + ICON_BOX_SIZE);

            // Background with subtle border
            ImU32 bg_color = title_color;
            draw_list->AddRectFilled(icon_pos, icon_max, bg_color, CORNER_RADIUS);
            // Add rounded border around the icon box
            ImU32 border_color = IM_COL32(80, 80, 90, 200);
            draw_list->AddRect(icon_pos, icon_max, border_color, CORNER_RADIUS, 0, 1.5f);

            // Draw node name ABOVE the icon
            if (show_name) {
                ImVec2 name_size = ImGui::CalcTextSize(node.name.c_str());
                float name_x = icon_pos.x + (ICON_BOX_SIZE - name_size.x) * 0.5f;
                float name_y = icon_pos.y - name_size.y - 4.0f;
                draw_list->AddText(ImVec2(name_x, name_y), IM_COL32(200, 200, 200, 255), node.name.c_str());
            }

            // Draw centered icon (55% of box height, like Node Browser)
            const char* icon = GetNodeIcon(node.type);
            ImFont* font = ImGui::GetFont();
            float scaled_icon_size = ICON_BOX_SIZE * 0.55f;
            ImVec2 icon_text_size = font->CalcTextSizeA(scaled_icon_size, FLT_MAX, 0.0f, icon);
            ImVec2 icon_text_pos(
                icon_pos.x + (ICON_BOX_SIZE - icon_text_size.x) * 0.5f,
                icon_pos.y + (ICON_BOX_SIZE - icon_text_size.y) * 0.5f
            );
            draw_list->AddText(font, scaled_icon_size, icon_text_pos, IM_COL32(255, 255, 255, 255), icon);

            // Draw description BELOW the icon (e.g., "Reading adult.csv")
            std::string display_desc = node.description;
            if (display_desc.empty()) {
                // Show default text for unconfigured nodes
                display_desc = "Double-click to configure";
            }
            float desc_y = icon_max.y + 4.0f;
            ImVec2 desc_size = ImGui::CalcTextSize(display_desc.c_str());
            float desc_x = icon_pos.x + (ICON_BOX_SIZE - desc_size.x) * 0.5f;
            // Clamp to icon left edge if description is wider
            desc_x = std::max(desc_x, icon_pos.x - 10.0f);
            ImU32 desc_color = node.description.empty() ? IM_COL32(120, 120, 120, 200) : IM_COL32(150, 180, 220, 255);
            draw_list->AddText(ImVec2(desc_x, desc_y), desc_color, display_desc.c_str());

            // Drag handle: icon box only (keeps pin drag separate from node move)
            ImVec2 cursor_backup = ImGui::GetCursorScreenPos();
            ImGui::SetCursorScreenPos(icon_pos);
            std::string drag_id = "##knime_drag_" + std::to_string(node.id);
            ImGui::InvisibleButton(drag_id.c_str(), ImVec2(ICON_BOX_SIZE, ICON_BOX_SIZE));

            // Select node when clicked (InvisibleButton consumes click before ImNodes)
            if (ImGui::IsItemClicked(0)) {
                if (!ImGui::GetIO().KeyCtrl && !ImGui::GetIO().KeyShift) {
                    // Single click without modifier - select only this node
                    ImNodes::ClearNodeSelection();
                }
                ImNodes::SelectNode(node.id);
                selected_node_id_ = node.id;
            }

            if (ImGui::IsItemActivated()) {
                dragging_knime_node_id_ = node.id;
                ImVec2 mouse_pos = ImGui::GetMousePos();
                ImVec2 node_pos = ImNodes::GetNodeScreenSpacePos(node.id);
                knime_drag_offset_ = ImVec2(mouse_pos.x - node_pos.x, mouse_pos.y - node_pos.y);
            }
            if (dragging_knime_node_id_ == node.id && ImGui::IsMouseDown(ImGuiMouseButton_Left)) {
                ImVec2 mouse_pos = ImGui::GetMousePos();
                ImVec2 new_pos(mouse_pos.x - knime_drag_offset_.x, mouse_pos.y - knime_drag_offset_.y);
                ImNodes::SetNodeScreenSpacePos(node.id, new_pos);
            }
            if (dragging_knime_node_id_ == node.id && ImGui::IsMouseReleased(ImGuiMouseButton_Left)) {
                dragging_knime_node_id_ = -1;
            }
            ImGui::SetCursorScreenPos(cursor_backup);

            // Pop styles (7 color + 2 vars for KNIME-style)
            ImNodes::PopStyleVar();    // NodePadding
            ImNodes::PopStyleVar();    // PinOffset
            ImNodes::PopColorStyle();  // NodeOutline
            ImNodes::PopColorStyle();  // NodeBackgroundSelected
            ImNodes::PopColorStyle();  // NodeBackgroundHovered
            ImNodes::PopColorStyle();  // NodeBackground
            ImNodes::PopColorStyle();  // TitleBarSelected
            ImNodes::PopColorStyle();  // TitleBarHovered
            ImNodes::PopColorStyle();  // TitleBar

        } else {
            // ===== STANDARD NODE RENDERING =====
            // Node title bar
            ImNodes::BeginNodeTitleBar();
            ImGui::TextUnformatted(node.name.c_str());
            ImNodes::EndNodeTitleBar();

            // Input pins
            for (const auto& pin : node.inputs) {
                ImU32 pin_color;
                ImU32 pin_hover;
                ImNodesPinShape pin_shape;
                switch (GetNodePinState(node.id)) {
                    case NodePinState::CompileFailed:
                        pin_color = IM_COL32(220, 50, 50, 255);
                        pin_hover = IM_COL32(255, 80, 80, 255);
                        pin_shape = ImNodesPinShape_CircleFilled;
                        break;
                    case NodePinState::CompilePassed:
                        pin_color = IM_COL32(50, 200, 50, 255);
                        pin_hover = IM_COL32(80, 230, 80, 255);
                        pin_shape = ImNodesPinShape_Circle;
                        break;
                    case NodePinState::Trained:
                        pin_color = IM_COL32(50, 200, 50, 255);
                        pin_hover = IM_COL32(80, 230, 80, 255);
                        pin_shape = ImNodesPinShape_CircleFilled;
                        break;
                    case NodePinState::Default:
                    default:
                        pin_color = IM_COL32(220, 50, 50, 255);
                        pin_hover = IM_COL32(255, 80, 80, 255);
                        pin_shape = ImNodesPinShape_Circle;
                        break;
                }

                ImNodes::PushColorStyle(ImNodesCol_Pin, pin_color);
                ImNodes::PushColorStyle(ImNodesCol_PinHovered, pin_hover);
                ImNodes::BeginInputAttribute(pin.id, pin_shape);
                ImGui::TextUnformatted(pin.name.c_str());
                ImNodes::EndInputAttribute();
                ImNodes::PopColorStyle();
                ImNodes::PopColorStyle();
        }

        // Display key parameter based on node type
        ImGui::Spacing();
        switch (node.type) {
            case NodeType::Dense: {
                auto it = node.parameters.find("units");
                if (it != node.parameters.end() && !it->second.empty()) {
                    ImGui::TextColored(ImVec4(0.7f, 0.9f, 1.0f, 1.0f), "Units: %s", it->second.c_str());
                }
                break;
            }
            case NodeType::Conv2D: {
                auto it = node.parameters.find("filters");
                if (it != node.parameters.end() && !it->second.empty()) {
                    ImGui::TextColored(ImVec4(0.7f, 0.9f, 1.0f, 1.0f), "Filters: %s", it->second.c_str());
                }
                break;
            }
            case NodeType::MaxPool2D: {
                auto it = node.parameters.find("pool_size");
                if (it != node.parameters.end() && !it->second.empty()) {
                    ImGui::TextColored(ImVec4(0.7f, 0.9f, 1.0f, 1.0f), "Pool: %s", it->second.c_str());
                }
                break;
            }
            case NodeType::Dropout: {
                auto it = node.parameters.find("rate");
                if (it != node.parameters.end() && !it->second.empty()) {
                    ImGui::TextColored(ImVec4(0.7f, 0.9f, 1.0f, 1.0f), "Rate: %s", it->second.c_str());
                }
                break;
            }
            case NodeType::BatchNorm: {
                auto it = node.parameters.find("momentum");
                if (it != node.parameters.end() && !it->second.empty()) {
                    ImGui::TextColored(ImVec4(0.7f, 0.9f, 1.0f, 1.0f), "Momentum: %s", it->second.c_str());
                }
                break;
            }
            case NodeType::Output: {
                auto it = node.parameters.find("classes");
                if (it != node.parameters.end() && !it->second.empty()) {
                    ImGui::TextColored(ImVec4(0.7f, 0.9f, 1.0f, 1.0f), "Classes: %s", it->second.c_str());
                }
                break;
            }

            // Data Pipeline Nodes
            case NodeType::DatasetInput: {
                auto it = node.parameters.find("dataset_name");
                if (it != node.parameters.end() && !it->second.empty()) {
                    ImGui::TextColored(ImVec4(0.5f, 1.0f, 1.0f, 1.0f), "Dataset: %s", it->second.c_str());
                } else {
                    ImGui::TextColored(ImVec4(0.5f, 1.0f, 1.0f, 0.7f), "Dataset: <select>");
                }
                auto split_it = node.parameters.find("split");
                if (split_it != node.parameters.end()) {
                    ImGui::TextColored(ImVec4(0.5f, 1.0f, 1.0f, 1.0f), "Split: %s", split_it->second.c_str());
                }
                break;
            }
            case NodeType::DataLoader: {
                auto it = node.parameters.find("batch_size");
                if (it != node.parameters.end()) {
                    ImGui::TextColored(ImVec4(0.5f, 1.0f, 1.0f, 1.0f), "Batch: %s", it->second.c_str());
                }
                auto shuffle_it = node.parameters.find("shuffle");
                if (shuffle_it != node.parameters.end()) {
                    ImGui::TextColored(ImVec4(0.5f, 1.0f, 1.0f, 1.0f), "Shuffle: %s", shuffle_it->second.c_str());
                }
                break;
            }
            case NodeType::Augmentation: {
                auto it = node.parameters.find("transforms");
                if (it != node.parameters.end() && !it->second.empty()) {
                    ImGui::TextColored(ImVec4(0.5f, 1.0f, 1.0f, 1.0f), "Transforms: %s", it->second.c_str());
                }
                break;
            }
            case NodeType::DataSplit: {
                auto train_it = node.parameters.find("train_ratio");
                auto val_it = node.parameters.find("val_ratio");
                auto test_it = node.parameters.find("test_ratio");
                if (train_it != node.parameters.end() && val_it != node.parameters.end() && test_it != node.parameters.end()) {
                    ImGui::TextColored(ImVec4(0.5f, 1.0f, 1.0f, 1.0f), "Split: %s/%s/%s",
                        train_it->second.c_str(), val_it->second.c_str(), test_it->second.c_str());
                }
                break;
            }
            case NodeType::TensorReshape: {
                auto it = node.parameters.find("shape");
                if (it != node.parameters.end() && !it->second.empty()) {
                    ImGui::TextColored(ImVec4(0.5f, 1.0f, 1.0f, 1.0f), "Shape: %s", it->second.c_str());
                }
                break;
            }
            case NodeType::Normalize: {
                auto mean_it = node.parameters.find("mean");
                auto std_it = node.parameters.find("std");
                if (mean_it != node.parameters.end() && std_it != node.parameters.end()) {
                    ImGui::TextColored(ImVec4(0.5f, 1.0f, 1.0f, 1.0f), "Mean: %s", mean_it->second.c_str());
                    ImGui::TextColored(ImVec4(0.5f, 1.0f, 1.0f, 1.0f), "Std: %s", std_it->second.c_str());
                }
                break;
            }
            case NodeType::OneHotEncode: {
                auto it = node.parameters.find("num_classes");
                if (it != node.parameters.end()) {
                    ImGui::TextColored(ImVec4(0.5f, 1.0f, 1.0f, 1.0f), "Classes: %s", it->second.c_str());
                }
                break;
            }

            // Text Processing
            case NodeType::TextTokenizer: {
                const char* types[] = {"Whitespace", "Word", "Character"};
                auto type_it = node.parameters.find("tokenizer_type");
                int type_idx = type_it != node.parameters.end() ? std::stoi(type_it->second) : 1;
                if (type_idx >= 0 && type_idx < 3) {
                    ImGui::TextColored(ImVec4(0.0f, 0.9f, 0.8f, 1.0f), "Type: %s", types[type_idx]);
                }
                auto len_it = node.parameters.find("max_length");
                if (len_it != node.parameters.end()) {
                    ImGui::TextColored(ImVec4(0.0f, 0.9f, 0.8f, 1.0f), "Max Len: %s", len_it->second.c_str());
                }
                break;
            }
            case NodeType::TextVocabulary: {
                auto it = node.parameters.find("max_vocab_size");
                if (it != node.parameters.end()) {
                    ImGui::TextColored(ImVec4(0.0f, 0.9f, 0.8f, 1.0f), "Max Vocab: %s", it->second.c_str());
                }
                break;
            }
            case NodeType::TextPadding: {
                auto it = node.parameters.find("max_length");
                if (it != node.parameters.end()) {
                    ImGui::TextColored(ImVec4(0.0f, 0.9f, 0.8f, 1.0f), "Pad to: %s", it->second.c_str());
                }
                break;
            }

            // Upsampling
            case NodeType::ConvTranspose2D: {
                auto out_ch = node.parameters.find("out_channels");
                auto k = node.parameters.find("kernel_size");
                auto s = node.parameters.find("stride");
                if (out_ch != node.parameters.end()) {
                    ImGui::TextColored(ImVec4(0.6f, 0.7f, 1.0f, 1.0f), "Out: %s ch", out_ch->second.c_str());
                }
                if (k != node.parameters.end() && s != node.parameters.end()) {
                    ImGui::TextColored(ImVec4(0.6f, 0.7f, 1.0f, 1.0f), "K:%s S:%s", k->second.c_str(), s->second.c_str());
                }
                break;
            }
            case NodeType::Upsample: {
                const char* modes[] = {"Nearest", "Bilinear"};
                auto sf = node.parameters.find("scale_factor");
                auto mode = node.parameters.find("mode");
                if (sf != node.parameters.end()) {
                    ImGui::TextColored(ImVec4(0.6f, 0.7f, 1.0f, 1.0f), "Scale: %sx", sf->second.c_str());
                }
                int m = mode != node.parameters.end() ? std::stoi(mode->second) : 0;
                if (m >= 0 && m < 2) {
                    ImGui::TextColored(ImVec4(0.6f, 0.7f, 1.0f, 1.0f), "Mode: %s", modes[m]);
                }
                break;
            }
            case NodeType::PixelShuffle: {
                auto it = node.parameters.find("upscale_factor");
                if (it != node.parameters.end()) {
                    ImGui::TextColored(ImVec4(0.6f, 0.7f, 1.0f, 1.0f), "Factor: %sx", it->second.c_str());
                }
                break;
            }

            // Time-Series
            case NodeType::TimeSeriesWindow: {
                auto ws = node.parameters.find("window_size");
                auto fh = node.parameters.find("forecast_horizon");
                if (ws != node.parameters.end()) {
                    ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.0f, 1.0f), "Window: %s", ws->second.c_str());
                }
                if (fh != node.parameters.end()) {
                    ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.0f, 1.0f), "Horizon: %s", fh->second.c_str());
                }
                break;
            }
            case NodeType::TimeSeriesFeatures: {
                auto lag = node.parameters.find("lag_values");
                if (lag != node.parameters.end()) {
                    ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.0f, 1.0f), "Lags: %s", lag->second.c_str());
                }
                break;
            }
            case NodeType::TimeSeriesSplit: {
                auto tr = node.parameters.find("train_ratio");
                auto vr = node.parameters.find("val_ratio");
                if (tr != node.parameters.end() && vr != node.parameters.end()) {
                    ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.0f, 1.0f), "Train: %s  Val: %s", tr->second.c_str(), vr->second.c_str());
                }
                break;
            }

            // Audio
            case NodeType::AudioInput: {
                auto sr = node.parameters.find("sample_rate");
                if (sr != node.parameters.end()) {
                    ImGui::TextColored(ImVec4(0.7f, 0.5f, 1.0f, 1.0f), "SR: %s Hz", sr->second.c_str());
                }
                break;
            }
            case NodeType::Spectrogram:
            case NodeType::MelSpectrogram: {
                auto nfft = node.parameters.find("n_fft");
                if (nfft != node.parameters.end()) {
                    ImGui::TextColored(ImVec4(0.7f, 0.5f, 1.0f, 1.0f), "FFT: %s", nfft->second.c_str());
                }
                auto mels = node.parameters.find("n_mels");
                if (mels != node.parameters.end()) {
                    ImGui::TextColored(ImVec4(0.7f, 0.5f, 1.0f, 1.0f), "Mels: %s", mels->second.c_str());
                }
                break;
            }
            case NodeType::MFCC: {
                auto n = node.parameters.find("n_mfcc");
                if (n != node.parameters.end()) {
                    ImGui::TextColored(ImVec4(0.7f, 0.5f, 1.0f, 1.0f), "MFCC: %s", n->second.c_str());
                }
                break;
            }
            case NodeType::AudioAugmentation: {
                ImGui::TextColored(ImVec4(0.7f, 0.5f, 1.0f, 1.0f), "Audio Augment");
                break;
            }

            // RL
            case NodeType::GymEnvironment: {
                auto env = node.parameters.find("env_name");
                if (env != node.parameters.end()) {
                    ImGui::TextColored(ImVec4(1.0f, 0.4f, 0.4f, 1.0f), "Env: %s", env->second.c_str());
                }
                break;
            }
            case NodeType::ReplayBufferNode: {
                auto cap = node.parameters.find("capacity");
                if (cap != node.parameters.end()) {
                    ImGui::TextColored(ImVec4(1.0f, 0.4f, 0.4f, 1.0f), "Cap: %s", cap->second.c_str());
                }
                break;
            }
            case NodeType::PolicyNetwork:
            case NodeType::ValueNetwork: {
                auto hs = node.parameters.find("hidden_size");
                if (hs != node.parameters.end()) {
                    ImGui::TextColored(ImVec4(1.0f, 0.4f, 0.4f, 1.0f), "Hidden: %s", hs->second.c_str());
                }
                break;
            }
            case NodeType::RLTraining: {
                auto algo = node.parameters.find("algorithm");
                auto eps = node.parameters.find("episodes");
                if (algo != node.parameters.end()) {
                    ImGui::TextColored(ImVec4(1.0f, 0.4f, 0.4f, 1.0f), "Algo: %s", algo->second.c_str());
                }
                if (eps != node.parameters.end()) {
                    ImGui::TextColored(ImVec4(1.0f, 0.4f, 0.4f, 1.0f), "Episodes: %s", eps->second.c_str());
                }
                break;
            }

            default:
                // For activation layers and other nodes without parameters, show nothing
                break;
        }
        ImGui::Spacing();

        // Output pins
        for (const auto& pin : node.outputs) {
            ImU32 pin_color;
            ImU32 pin_hover;
            ImNodesPinShape pin_shape;
            switch (GetNodePinState(node.id)) {
                case NodePinState::CompileFailed:
                    pin_color = IM_COL32(220, 50, 50, 255);
                    pin_hover = IM_COL32(255, 80, 80, 255);
                    pin_shape = ImNodesPinShape_CircleFilled;
                    break;
                case NodePinState::CompilePassed:
                    pin_color = IM_COL32(50, 200, 50, 255);
                    pin_hover = IM_COL32(80, 230, 80, 255);
                    pin_shape = ImNodesPinShape_Circle;
                    break;
                case NodePinState::Trained:
                    pin_color = IM_COL32(50, 200, 50, 255);
                    pin_hover = IM_COL32(80, 230, 80, 255);
                    pin_shape = ImNodesPinShape_CircleFilled;
                    break;
                case NodePinState::Default:
                default:
                    pin_color = IM_COL32(220, 50, 50, 255);
                    pin_hover = IM_COL32(255, 80, 80, 255);
                    pin_shape = ImNodesPinShape_Circle;
                    break;
            }

            ImNodes::PushColorStyle(ImNodesCol_Pin, pin_color);
            ImNodes::PushColorStyle(ImNodesCol_PinHovered, pin_hover);
            ImNodes::BeginOutputAttribute(pin.id, pin_shape);
            const float text_width = ImGui::CalcTextSize(pin.name.c_str()).x;
            ImGui::Indent(120.0f + ImGui::CalcTextSize(pin.name.c_str()).x - text_width);
            ImGui::TextUnformatted(pin.name.c_str());
            ImNodes::EndOutputAttribute();
            ImNodes::PopColorStyle();
            ImNodes::PopColorStyle();
        }
        // Pop standard node title bar styles
            ImNodes::PopColorStyle();  // TitleBarSelected
            ImNodes::PopColorStyle();  // TitleBarHovered
            ImNodes::PopColorStyle();  // TitleBar
        } // End of standard node rendering else block

        ImNodes::EndNode();

        // Check if this node is hovered for documentation tooltip
        int hovered_node_id = -1;
        if (ImNodes::IsNodeHovered(&hovered_node_id) && hovered_node_id == node.id) {
            // Unified Canvas Phase 6: Show execution state in tooltip
            if (has_exec_state && exec_state != NodeExecutionState::Idle) {
                ImGui::BeginTooltip();

                // Show execution status
                if (exec_state == NodeExecutionState::Executing) {
                    ImGui::TextColored(ImVec4(0.3f, 0.6f, 1.0f, 1.0f), ICON_FA_SPINNER " Executing...");
                } else if (exec_state == NodeExecutionState::Completed) {
                    ImGui::TextColored(ImVec4(0.3f, 0.8f, 0.4f, 1.0f), ICON_FA_CHECK " Completed");
                } else if (exec_state == NodeExecutionState::Error) {
                    ImGui::TextColored(ImVec4(1.0f, 0.3f, 0.3f, 1.0f), ICON_FA_TIMES " Error");

                    // Show error message if available
                    auto error_it = node_execution_errors_.find(node.id);
                    if (error_it != node_execution_errors_.end()) {
                        ImGui::Separator();
                        ImGui::TextWrapped("%s", error_it->second.c_str());
                    }
                } else if (exec_state == NodeExecutionState::Pending) {
                    ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f), ICON_FA_CLOCK " Pending...");
                }

                ImGui::Separator();
                ImGui::EndTooltip();
            }

            // Show node documentation
            NodeDocumentationManager::Instance().RenderTooltip(node.type);
        }

        // Draw selection glow effect for selected nodes (skip for KNIME-style)
        bool skip_glow = true;  // All nodes are KNIME-style now
        if (!skip_glow && ImNodes::IsNodeSelected(node.id)) {
            ImVec2 node_pos = ImNodes::GetNodeScreenSpacePos(node.id);
            ImVec2 node_dims = ImNodes::GetNodeDimensions(node.id);
            ImDrawList* draw_list = ImGui::GetWindowDrawList();

            // Draw multi-layered glow effect (outer to inner)
            float glow_size = 8.0f * zoom_;

            for (int i = 3; i >= 0; --i) {
                float offset = glow_size * (i + 1) * 0.4f;
                int alpha = 15 * (4 - i);  // Fade out: 60, 45, 30, 15
                ImU32 glow_color = IM_COL32(100, 180, 255, alpha);
                draw_list->AddRect(
                    ImVec2(node_pos.x - offset, node_pos.y - offset),
                    ImVec2(node_pos.x + node_dims.x + offset, node_pos.y + node_dims.y + offset),
                    glow_color,
                    8.0f * zoom_,  // Corner rounding
                    0,
                    2.0f + i * 0.5f  // Thicker outer lines
                );
            }

            // Draw bright inner border
            draw_list->AddRect(
                ImVec2(node_pos.x - 1, node_pos.y - 1),
                ImVec2(node_pos.x + node_dims.x + 1, node_pos.y + node_dims.y + 1),
                IM_COL32(100, 180, 255, 180),
                6.0f * zoom_,
                0,
                2.0f
            );
        }

        // KNIME-style: Draw node description below the node (bound to node, moves with it)
        if (!node.description.empty()) {
            ImVec2 node_pos = ImNodes::GetNodeScreenSpacePos(node.id);
            ImVec2 node_dims = ImNodes::GetNodeDimensions(node.id);
            ImDrawList* draw_list = ImGui::GetWindowDrawList();

            // Calculate description position (below node with spacing)
            float desc_y = node_pos.y + node_dims.y + 8.0f * zoom_;
            // Wider max width for better readability
            float max_width = std::max(node_dims.x * 1.5f, 220.0f * zoom_);

            // Use larger, more readable font size
            ImFont* font = ImGui::GetFont();
            float font_size = 14.0f * zoom_;
            const char* text_start = node.description.c_str();
            const char* text_end = text_start + node.description.size();

            // Word wrapping
            std::string wrapped;
            float line_width = 0.0f;
            const char* word_start = text_start;
            float space_width = font->CalcTextSizeA(font_size, FLT_MAX, 0.0f, " ").x;

            for (const char* p = text_start; p <= text_end; ++p) {
                if (*p == ' ' || *p == '\n' || p == text_end) {
                    std::string word(word_start, p);
                    if (!word.empty()) {
                        ImVec2 word_size = font->CalcTextSizeA(font_size, FLT_MAX, 0.0f, word.c_str());
                        if (line_width + word_size.x > max_width && line_width > 0) {
                            if (!wrapped.empty() && wrapped.back() == ' ') wrapped.pop_back();
                            wrapped += "\n";
                            line_width = 0.0f;
                        }
                        wrapped += word;
                        line_width += word_size.x;
                        if (*p == ' ') { wrapped += " "; line_width += space_width; }
                    }
                    if (*p == '\n') { wrapped += "\n"; line_width = 0.0f; }
                    word_start = p + 1;
                }
            }

            // Calculate text dimensions
            ImVec2 text_size = font->CalcTextSizeA(font_size, FLT_MAX, max_width, wrapped.c_str());
            float text_x = node_pos.x;
            float padding_x = 10.0f * zoom_;
            float padding_y = 8.0f * zoom_;

            // Draw background box
            ImVec2 box_min(text_x - padding_x, desc_y - padding_y);
            ImVec2 box_max(text_x + text_size.x + padding_x, desc_y + text_size.y + padding_y);

            draw_list->AddRectFilled(box_min, box_max, IM_COL32(30, 35, 50, 245), 5.0f * zoom_);
            draw_list->AddRect(box_min, box_max, IM_COL32(80, 100, 130, 220), 5.0f * zoom_, 0, 1.2f * zoom_);

            // Left accent line
            draw_list->AddRectFilled(
                ImVec2(box_min.x + 2.0f * zoom_, box_min.y + 3.0f * zoom_),
                ImVec2(box_min.x + 5.0f * zoom_, box_max.y - 3.0f * zoom_),
                IM_COL32(100, 149, 237, 255)
            );

            // Draw text
            draw_list->AddText(font, font_size, ImVec2(text_x + 4.0f * zoom_, desc_y),
                IM_COL32(230, 235, 245, 255), wrapped.c_str());
        }

        // Apply any pending position AFTER the node has been created
        // (ImNodes needs the node to exist before SetNodeGridSpacePos works)
        // Keep applying positions while pending_positions_frames_ > 0 to ensure they stick
        auto pos_it = pending_positions_.find(node.id);
        if (pos_it != pending_positions_.end()) {
            ImNodes::SetNodeGridSpacePos(node.id, pos_it->second);
            // Only erase if we're done applying (frame counter reached 0)
            if (pending_positions_frames_ <= 0) {
                pending_positions_.erase(pos_it);
            }
        }

        // Render warning icon if node has warnings
        for (const auto& warning : validation_warnings_) {
            if (warning.node_id == node.id) {
                // Get node position and dimensions
                ImVec2 node_pos = ImNodes::GetNodeScreenSpacePos(node.id);
                ImVec2 node_dims = ImNodes::GetNodeDimensions(node.id);

                // Position warning icon in top-right corner of node
                ImVec2 icon_pos = ImVec2(node_pos.x + node_dims.x - 30, node_pos.y + 5);

                ImDrawList* draw_list = ImGui::GetWindowDrawList();

                // Draw warning icon (yellow triangle with exclamation mark)
                ImU32 warning_color = IM_COL32(255, 193, 7, 255);  // Amber/yellow
                draw_list->AddText(icon_pos, warning_color, ICON_FA_TRIANGLE_EXCLAMATION);

                // Show tooltip when hovering over icon area
                ImVec2 icon_size = ImGui::CalcTextSize(ICON_FA_TRIANGLE_EXCLAMATION);
                ImVec2 mouse_pos = ImGui::GetMousePos();
                bool is_hovering = (mouse_pos.x >= icon_pos.x && mouse_pos.x <= icon_pos.x + icon_size.x &&
                                  mouse_pos.y >= icon_pos.y && mouse_pos.y <= icon_pos.y + icon_size.y);

                if (is_hovering) {
                    ImGui::BeginTooltip();
                    ImGui::PushTextWrapPos(ImGui::GetFontSize() * 35.0f);

                    // Warning header
                    ImGui::TextColored(ImVec4(1.0f, 0.76f, 0.03f, 1.0f),
                                     ICON_FA_TRIANGLE_EXCLAMATION " Shape Mismatch");
                    ImGui::Separator();

                    // Warning message
                    ImGui::TextWrapped("%s", warning.message.c_str());

                    // Suggested fix
                    if (warning.has_auto_fix && !warning.suggested_fix.empty()) {
                        ImGui::Spacing();
                        ImGui::TextColored(ImVec4(0.5f, 1.0f, 0.5f, 1.0f),
                                         ICON_FA_LIGHTBULB " Suggestion:");
                        ImGui::TextWrapped("%s", warning.suggested_fix.c_str());
                        ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f),
                                         "(Try connecting the nodes to see auto-fix dialog)");
                    }

                    ImGui::PopTextWrapPos();
                    ImGui::EndTooltip();
                }

                // Only show one warning per node
                break;
            }
        }

        // =====================================================================
        // KNIME-style traffic light status indicator below node
        // =====================================================================
        {
            ImVec2 node_pos = ImNodes::GetNodeScreenSpacePos(node.id);
            ImVec2 node_dims = ImNodes::GetNodeDimensions(node.id);
            ImDrawList* draw_list = ImGui::GetWindowDrawList();

            // Position: centered below node
            float indicator_radius = 5.0f * zoom_;
            ImVec2 indicator_pos = ImVec2(
                node_pos.x + node_dims.x / 2.0f,
                node_pos.y + node_dims.y + indicator_radius + 4.0f * zoom_
            );

            // Determine color and style based on execution state
            ImU32 indicator_color = 0;
            bool show_indicator = true;
            bool is_executing = false;

            switch (exec_state) {
                case NodeExecutionState::Completed:
                    indicator_color = IM_COL32(76, 175, 80, 255);   // Green
                    break;
                case NodeExecutionState::Pending:
                    indicator_color = IM_COL32(255, 193, 7, 255);   // Yellow/Amber
                    break;
                case NodeExecutionState::Executing:
                    is_executing = true;
                    break;
                case NodeExecutionState::Error:
                    indicator_color = IM_COL32(244, 67, 54, 255);   // Red
                    break;
                default:
                    show_indicator = false;  // Idle - no indicator
                    break;
            }

            if (show_indicator) {
                if (is_executing) {
                    // Spinning progress ring for executing state
                    float angle = execution_pulse_time_ * 3.0f;  // Rotation speed
                    float arc_length = 2.5f;  // Radians (~143 degrees)

                    // Draw background circle (faded)
                    draw_list->AddCircle(indicator_pos, indicator_radius,
                                        IM_COL32(60, 150, 255, 60), 0, 2.0f * zoom_);

                    // Draw spinning arc
                    draw_list->PathArcTo(indicator_pos, indicator_radius, angle, angle + arc_length, 12);
                    draw_list->PathStroke(IM_COL32(60, 150, 255, 255), false, 2.5f * zoom_);
                } else {
                    // Static filled circle for other states
                    draw_list->AddCircleFilled(indicator_pos, indicator_radius, indicator_color);
                    draw_list->AddCircle(indicator_pos, indicator_radius,
                                        IM_COL32(0, 0, 0, 80), 0, 1.0f * zoom_);

                    // Add X overlay for error state
                    if (exec_state == NodeExecutionState::Error) {
                        float x_size = indicator_radius * 0.5f;
                        ImU32 x_color = IM_COL32(255, 255, 255, 255);
                        draw_list->AddLine(
                            ImVec2(indicator_pos.x - x_size, indicator_pos.y - x_size),
                            ImVec2(indicator_pos.x + x_size, indicator_pos.y + x_size),
                            x_color, 1.5f * zoom_
                        );
                        draw_list->AddLine(
                            ImVec2(indicator_pos.x + x_size, indicator_pos.y - x_size),
                            ImVec2(indicator_pos.x - x_size, indicator_pos.y + x_size),
                            x_color, 1.5f * zoom_
                        );
                    }

                    // Add checkmark for completed state
                    if (exec_state == NodeExecutionState::Completed) {
                        float check_size = indicator_radius * 0.5f;
                        ImU32 check_color = IM_COL32(255, 255, 255, 255);
                        // Draw checkmark: short line down-left, then long line down-right
                        ImVec2 p1 = ImVec2(indicator_pos.x - check_size * 0.5f, indicator_pos.y);
                        ImVec2 p2 = ImVec2(indicator_pos.x - check_size * 0.1f, indicator_pos.y + check_size * 0.4f);
                        ImVec2 p3 = ImVec2(indicator_pos.x + check_size * 0.6f, indicator_pos.y - check_size * 0.4f);
                        draw_list->AddLine(p1, p2, check_color, 1.5f * zoom_);
                        draw_list->AddLine(p2, p3, check_color, 1.5f * zoom_);
                    }
                }
            }
        }
    }

    // Render all links with color based on link type (and training animation if active)
    for (const auto& link : links_) {
        ImU32 link_color, link_hovered, link_selected;

        if (is_training_) {
            // Create pulsing amber/green effect during training
            // Pulse frequency: ~2 Hz (full cycle every 0.5 seconds)
            float pulse = (std::sin(training_animation_time_ * 12.0f + link.id * 0.5f) + 1.0f) * 0.5f;

            // Interpolate between amber (255, 191, 0) and green (0, 255, 100)
            float r = 255.0f * (1.0f - pulse) + 0.0f * pulse;
            float g = 191.0f * (1.0f - pulse) + 255.0f * pulse;
            float b = 0.0f * (1.0f - pulse) + 100.0f * pulse;

            link_color = IM_COL32(static_cast<int>(r), static_cast<int>(g), static_cast<int>(b), 255);
            link_hovered = IM_COL32(static_cast<int>(r), static_cast<int>(g), static_cast<int>(b), 200);
            link_selected = IM_COL32(255, 255, 255, 255);
        } else {
            // Standard data-flow links inherit the source node's compile/train state:
            // red default, red filled if compile failed, green if passed, solid green if trained.
            // Special link types (skip connections, attention) keep their own distinctive colors.
            if (link.type == LinkType::TensorFlow) {
                NodePinState src_state = GetNodePinState(link.from_node);
                switch (src_state) {
                    case NodePinState::CompileFailed:
                        link_color   = IM_COL32(220, 50, 50, 255);
                        link_hovered = IM_COL32(255, 80, 80, 255);
                        break;
                    case NodePinState::CompilePassed:
                        link_color   = IM_COL32(80, 200, 80, 200);  // green, slightly translucent
                        link_hovered = IM_COL32(120, 230, 120, 255);
                        break;
                    case NodePinState::Trained:
                        link_color   = IM_COL32(50, 200, 50, 255);  // solid green
                        link_hovered = IM_COL32(80, 230, 80, 255);
                        break;
                    case NodePinState::Default:
                    default:
                        link_color   = IM_COL32(220, 50, 50, 200);  // red, slightly translucent
                        link_hovered = IM_COL32(255, 80, 80, 255);
                        break;
                }
            } else {
                link_color = GetLinkColor(link.type);
                link_hovered = GetLinkHoverColor(link.type);
            }
            link_selected = IM_COL32(255, 255, 255, 255);
        }

        ImNodes::PushColorStyle(ImNodesCol_Link, link_color);
        ImNodes::PushColorStyle(ImNodesCol_LinkHovered, link_hovered);
        ImNodes::PushColorStyle(ImNodesCol_LinkSelected, link_selected);

        ImNodes::Link(link.id, link.from_pin, link.to_pin);

        ImNodes::PopColorStyle();
        ImNodes::PopColorStyle();
        ImNodes::PopColorStyle();
    }

    // Decrement pending positions frame counter after all nodes processed
    if (pending_positions_frames_ > 0) {
        pending_positions_frames_--;
        // Clear positions when we're done applying them
        if (pending_positions_frames_ == 0) {
            pending_positions_.clear();
        }
    }
}

void NodeEditor::HandleInteractions() {
    // Handle new link creation
    // Use the extended version that provides both node IDs and pin IDs
    int from_node, from_pin, to_node, to_pin;
    if (ImNodes::IsLinkCreated(&from_node, &from_pin, &to_node, &to_pin)) {
        // Validate the link before creating it
        std::string error_message;
        if (ValidateLink(from_pin, to_pin, error_message)) {
            SaveUndoState();  // Save state before creating link

            NodeLink link;
            link.id = next_link_id_++;
            link.from_node = from_node;
            link.from_pin = from_pin;
            link.to_node = to_node;
            link.to_pin = to_pin;

            links_.push_back(link);
            spdlog::info("Created link {} from node {} pin {} to node {} pin {}",
                        link.id, from_node, from_pin, to_node, to_pin);
        } else {
            spdlog::warn("Link validation failed: {}", error_message);
            // Link creation blocked - dialog may have been triggered if shape mismatch
        }
    }

    // Handle link deletion
    int deleted_link_id;
    if (ImNodes::IsLinkDestroyed(&deleted_link_id)) {
        auto it = std::find_if(links_.begin(), links_.end(),
            [deleted_link_id](const NodeLink& link) {
                return link.id == deleted_link_id;
            });

        if (it != links_.end()) {
            SaveUndoState();  // Save state before deleting link
            ClearValidationState();  // Graph changed — stale compile results
            spdlog::info("Deleted link {}", deleted_link_id);
            links_.erase(it);
        }
    }

    // Handle node deletion (Delete key)
    const int num_selected_nodes = ImNodes::NumSelectedNodes();

    // Debug: Log selection count (only when it changes)
    static int last_selected_count = 0;
    if (num_selected_nodes != last_selected_count) {
        spdlog::info("Selected nodes: {}", num_selected_nodes);
        last_selected_count = num_selected_nodes;
    }

    if (num_selected_nodes > 0 && ImGui::IsKeyReleased(ImGuiKey_Delete)) {
        SaveUndoState();  // Save state before deleting nodes

        std::vector<int> selected_nodes(num_selected_nodes);
        ImNodes::GetSelectedNodes(selected_nodes.data());

        spdlog::info("Deleting {} selected nodes", num_selected_nodes);
        for (int node_id : selected_nodes) {
            DeleteNode(node_id);
        }

        // Clear selection after deletion to prevent stale node IDs
        ImNodes::ClearNodeSelection();
        ImNodes::ClearLinkSelection();
        selected_node_id_ = -1;

        // Also clear properties panel
        if (properties_panel_) {
            properties_panel_->ClearSelection();
        }
    }
}

// ===== Search Functionality =====

void NodeEditor::ShowSearchBar() {
    if (!search_state_.search_visible) return;

    // Position the search bar at the top of the node editor window
    ImVec2 window_pos = ImGui::GetWindowPos();
    ImVec2 window_size = ImGui::GetWindowSize();
    float bar_height = 40.0f;
    float bar_width = 350.0f;

    // Position at top-center of the window
    ImVec2 bar_pos = ImVec2(
        window_pos.x + (window_size.x - bar_width) * 0.5f,
        window_pos.y + ImGui::GetFrameHeight() + 60.0f  // Below toolbar
    );

    ImGui::SetNextWindowPos(bar_pos);
    ImGui::SetNextWindowSize(ImVec2(bar_width, bar_height));
    ImGui::SetNextWindowBgAlpha(0.95f);

    ImGuiWindowFlags flags = ImGuiWindowFlags_NoTitleBar |
                             ImGuiWindowFlags_NoResize |
                             ImGuiWindowFlags_NoMove |
                             ImGuiWindowFlags_NoScrollbar |
                             ImGuiWindowFlags_NoSavedSettings;

    if (ImGui::Begin("##NodeSearchBar", nullptr, flags)) {
        // Search icon
        ImGui::Text(ICON_FA_MAGNIFYING_GLASS);
        ImGui::SameLine();

        // Focus input if just opened
        static bool was_visible = false;
        if (!was_visible && search_state_.search_visible) {
            ImGui::SetKeyboardFocusHere();
        }
        was_visible = search_state_.search_visible;

        // Search input
        ImGui::SetNextItemWidth(200.0f);
        bool changed = ImGui::InputText("##SearchInput", search_state_.search_buffer,
                                        sizeof(search_state_.search_buffer),
                                        ImGuiInputTextFlags_AutoSelectAll);
        if (changed) {
            UpdateSearchResults();
        }

        // Handle Enter key to navigate to next match
        if (ImGui::IsItemFocused() && ImGui::IsKeyPressed(ImGuiKey_Enter)) {
            if (ImGui::GetIO().KeyShift) {
                NavigateToMatch(-1);
            } else {
                NavigateToMatch(1);
            }
        }

        // Match count
        ImGui::SameLine();
        if (!search_state_.matching_node_ids.empty()) {
            ImGui::Text("%d/%zu", search_state_.current_match_index + 1,
                       search_state_.matching_node_ids.size());
        } else if (strlen(search_state_.search_buffer) > 0) {
            ImGui::TextColored(ImVec4(1.0f, 0.4f, 0.4f, 1.0f), "No matches");
        }

        // Navigation buttons
        ImGui::SameLine();
        if (ImGui::SmallButton(ICON_FA_CHEVRON_UP)) {
            NavigateToMatch(-1);
        }
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Previous (Shift+F3)");
        }
        ImGui::SameLine();
        if (ImGui::SmallButton(ICON_FA_CHEVRON_DOWN)) {
            NavigateToMatch(1);
        }
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Next (F3)");
        }

        // Close button
        ImGui::SameLine();
        if (ImGui::SmallButton(ICON_FA_XMARK)) {
            search_state_.search_visible = false;
            search_state_.matching_node_ids.clear();
            search_state_.current_match_index = -1;
        }
    }
    ImGui::End();
}

void NodeEditor::UpdateSearchResults() {
    search_state_.matching_node_ids.clear();
    search_state_.current_match_index = -1;

    std::string query = search_state_.search_buffer;
    if (query.empty()) return;

    // Convert query to lowercase for case-insensitive search
    std::transform(query.begin(), query.end(), query.begin(), ::tolower);

    for (const auto& node : nodes_) {
        bool matches = false;

        // Search node name
        std::string name_lower = node.name;
        std::transform(name_lower.begin(), name_lower.end(), name_lower.begin(), ::tolower);
        if (name_lower.find(query) != std::string::npos) {
            matches = true;
        }

        // Search node type name (category)
        if (!matches) {
            const char* category = NodeDocumentationManager::GetCategoryName(node.type);
            std::string category_lower = category;
            std::transform(category_lower.begin(), category_lower.end(), category_lower.begin(), ::tolower);
            if (category_lower.find(query) != std::string::npos) {
                matches = true;
            }
        }

        // Search parameter values
        if (!matches) {
            for (const auto& [key, value] : node.parameters) {
                std::string key_lower = key;
                std::string value_lower = value;
                std::transform(key_lower.begin(), key_lower.end(), key_lower.begin(), ::tolower);
                std::transform(value_lower.begin(), value_lower.end(), value_lower.begin(), ::tolower);

                if (key_lower.find(query) != std::string::npos ||
                    value_lower.find(query) != std::string::npos) {
                    matches = true;
                    break;
                }
            }
        }

        if (matches) {
            search_state_.matching_node_ids.push_back(node.id);
        }
    }

    // Select first match if any found
    if (!search_state_.matching_node_ids.empty()) {
        search_state_.current_match_index = 0;
        NavigateToMatch(0);  // Navigate to current (first) match
    }
}

void NodeEditor::NavigateToMatch(int direction) {
    if (search_state_.matching_node_ids.empty()) return;

    // Update index
    if (direction != 0) {
        search_state_.current_match_index += direction;

        // Wrap around
        if (search_state_.current_match_index < 0) {
            search_state_.current_match_index = static_cast<int>(search_state_.matching_node_ids.size()) - 1;
        } else if (search_state_.current_match_index >= static_cast<int>(search_state_.matching_node_ids.size())) {
            search_state_.current_match_index = 0;
        }
    }

    // Get the matched node ID
    int node_id = search_state_.matching_node_ids[search_state_.current_match_index];

    // Select the node
    ImNodes::ClearNodeSelection();
    ImNodes::SelectNode(node_id);
    selected_node_id_ = node_id;

    // Center viewport on the matched node
    auto pos_it = cached_node_positions_.find(node_id);
    if (pos_it != cached_node_positions_.end()) {
        ImVec2 node_pos = pos_it->second;

        // Get the visible area size (approximately)
        ImVec2 window_size = ImGui::GetWindowSize();
        ImVec2 center_offset = ImVec2(window_size.x * 0.5f - 100.0f, window_size.y * 0.5f - 50.0f);

        // Pan to center the node
        ImNodes::EditorContextResetPanning(ImVec2(-node_pos.x + center_offset.x, -node_pos.y + center_offset.y));
    }

    spdlog::debug("Navigated to match {}/{}: node {}",
                  search_state_.current_match_index + 1,
                  search_state_.matching_node_ids.size(),
                  node_id);
}

void NodeEditor::HighlightMatchingNodes() {
    // This function can be called from RenderNodes to add visual highlighting
    // For now, selection highlighting is handled by ImNodes automatically
    // TODO: Add custom overlay drawing for matching but not selected nodes
}

// ===== Alignment and Distribution Tools =====

void NodeEditor::AlignSelectedNodes(AlignmentType type) {
    if (selected_node_ids_.size() < 2) {
        spdlog::warn("Need at least 2 selected nodes to align");
        return;
    }

    SaveUndoState();  // Save before modification

    // Calculate the reference value based on alignment type
    float reference = 0.0f;
    bool first = true;

    for (int node_id : selected_node_ids_) {
        auto pos_it = cached_node_positions_.find(node_id);
        if (pos_it == cached_node_positions_.end()) continue;

        ImVec2 pos = pos_it->second;
        // Approximate node dimensions (ImNodes doesn't expose actual size)
        float node_width = 150.0f;
        float node_height = 100.0f;

        switch (type) {
            case AlignmentType::Left:
                if (first || pos.x < reference) reference = pos.x;
                break;
            case AlignmentType::Center:
                if (first) reference = pos.x + node_width * 0.5f;
                else reference = (reference + pos.x + node_width * 0.5f) / 2.0f;
                break;
            case AlignmentType::Right:
                if (first || pos.x + node_width > reference) reference = pos.x + node_width;
                break;
            case AlignmentType::Top:
                if (first || pos.y < reference) reference = pos.y;
                break;
            case AlignmentType::Middle:
                if (first) reference = pos.y + node_height * 0.5f;
                else reference = (reference + pos.y + node_height * 0.5f) / 2.0f;
                break;
            case AlignmentType::Bottom:
                if (first || pos.y + node_height > reference) reference = pos.y + node_height;
                break;
        }
        first = false;
    }

    // Apply alignment to all selected nodes
    for (int node_id : selected_node_ids_) {
        auto pos_it = cached_node_positions_.find(node_id);
        if (pos_it == cached_node_positions_.end()) continue;

        ImVec2 pos = pos_it->second;
        float node_width = 150.0f;
        float node_height = 100.0f;

        switch (type) {
            case AlignmentType::Left:
                pos.x = reference;
                break;
            case AlignmentType::Center:
                pos.x = reference - node_width * 0.5f;
                break;
            case AlignmentType::Right:
                pos.x = reference - node_width;
                break;
            case AlignmentType::Top:
                pos.y = reference;
                break;
            case AlignmentType::Middle:
                pos.y = reference - node_height * 0.5f;
                break;
            case AlignmentType::Bottom:
                pos.y = reference - node_height;
                break;
        }

        // Apply the new position
        pending_positions_[node_id] = pos;
        cached_node_positions_[node_id] = pos;
    }

    pending_positions_frames_ = 3;  // Apply for a few frames to ensure it sticks
    spdlog::info("Aligned {} nodes", selected_node_ids_.size());
}

void NodeEditor::DistributeSelectedNodes(DistributeType type) {
    if (selected_node_ids_.size() < 3) {
        spdlog::warn("Need at least 3 selected nodes to distribute");
        return;
    }

    SaveUndoState();  // Save before modification

    // Collect node positions and IDs, then sort by position
    struct NodePos {
        int id;
        ImVec2 pos;
    };
    std::vector<NodePos> nodes;

    for (int node_id : selected_node_ids_) {
        auto pos_it = cached_node_positions_.find(node_id);
        if (pos_it != cached_node_positions_.end()) {
            nodes.push_back({node_id, pos_it->second});
        }
    }

    if (nodes.size() < 3) return;

    // Sort nodes by the appropriate axis
    if (type == DistributeType::Horizontal) {
        std::sort(nodes.begin(), nodes.end(),
                  [](const NodePos& a, const NodePos& b) { return a.pos.x < b.pos.x; });
    } else {
        std::sort(nodes.begin(), nodes.end(),
                  [](const NodePos& a, const NodePos& b) { return a.pos.y < b.pos.y; });
    }

    // Calculate the total span and spacing
    float first_pos = (type == DistributeType::Horizontal) ? nodes.front().pos.x : nodes.front().pos.y;
    float last_pos = (type == DistributeType::Horizontal) ? nodes.back().pos.x : nodes.back().pos.y;
    float total_span = last_pos - first_pos;
    float spacing = total_span / (static_cast<float>(nodes.size()) - 1.0f);

    // Apply evenly distributed positions (keep first and last in place)
    for (size_t i = 1; i < nodes.size() - 1; ++i) {
        ImVec2 new_pos = nodes[i].pos;
        if (type == DistributeType::Horizontal) {
            new_pos.x = first_pos + spacing * static_cast<float>(i);
        } else {
            new_pos.y = first_pos + spacing * static_cast<float>(i);
        }
        pending_positions_[nodes[i].id] = new_pos;
        cached_node_positions_[nodes[i].id] = new_pos;
    }

    pending_positions_frames_ = 3;
    spdlog::info("Distributed {} nodes {}", nodes.size(),
                 type == DistributeType::Horizontal ? "horizontally" : "vertically");
}

void NodeEditor::AutoLayoutSelection() {
    if (selected_node_ids_.empty()) {
        spdlog::warn("No nodes selected for auto-layout");
        return;
    }

    SaveUndoState();  // Save before modification

    // Get bounding box of selected nodes
    float min_x = std::numeric_limits<float>::max();
    float min_y = std::numeric_limits<float>::max();

    for (int node_id : selected_node_ids_) {
        auto pos_it = cached_node_positions_.find(node_id);
        if (pos_it != cached_node_positions_.end()) {
            min_x = std::min(min_x, pos_it->second.x);
            min_y = std::min(min_y, pos_it->second.y);
        }
    }

    // Calculate grid dimensions
    size_t count = selected_node_ids_.size();
    int cols = static_cast<int>(std::ceil(std::sqrt(static_cast<double>(count))));
    float spacing_x = 200.0f;
    float spacing_y = 150.0f;

    // Arrange nodes in a grid
    int col = 0, row = 0;
    for (int node_id : selected_node_ids_) {
        ImVec2 new_pos = ImVec2(
            min_x + static_cast<float>(col) * spacing_x,
            min_y + static_cast<float>(row) * spacing_y
        );

        pending_positions_[node_id] = new_pos;
        cached_node_positions_[node_id] = new_pos;

        col++;
        if (col >= cols) {
            col = 0;
            row++;
        }
    }

    pending_positions_frames_ = 3;
    spdlog::info("Auto-arranged {} nodes in grid", count);
}

// ===== Node Grouping =====

void NodeEditor::CreateGroupFromSelection(const std::string& name) {
    if (selected_node_ids_.empty()) {
        spdlog::warn("No nodes selected to create group");
        return;
    }

    SaveUndoState();

    NodeGroup group;
    group.id = next_group_id_++;
    group.name = name.empty() ? "Group " + std::to_string(group.id) : name;
    group.node_ids = selected_node_ids_;
    group.color = ImVec4(create_group_color_[0], create_group_color_[1],
                         create_group_color_[2], create_group_color_[3]);
    group.collapsed = false;
    group.padding = 20.0f;

    groups_.push_back(group);
    spdlog::info("Created group '{}' with {} nodes", group.name, group.node_ids.size());
}

void NodeEditor::DeleteGroup(int group_id) {
    auto it = std::find_if(groups_.begin(), groups_.end(),
                           [group_id](const NodeGroup& g) { return g.id == group_id; });
    if (it != groups_.end()) {
        SaveUndoState();
        spdlog::info("Deleted group '{}'", it->name);
        groups_.erase(it);
    }
}

void NodeEditor::UngroupSelection() {
    if (selected_node_ids_.empty()) return;

    SaveUndoState();

    // Find and remove groups containing any selected node
    for (int node_id : selected_node_ids_) {
        groups_.erase(
            std::remove_if(groups_.begin(), groups_.end(),
                [node_id](const NodeGroup& g) {
                    return std::find(g.node_ids.begin(), g.node_ids.end(), node_id) != g.node_ids.end();
                }),
            groups_.end()
        );
    }

    spdlog::info("Ungrouped selected nodes");
}

NodeGroup* NodeEditor::FindGroupContainingNode(int node_id) {
    for (auto& group : groups_) {
        if (std::find(group.node_ids.begin(), group.node_ids.end(), node_id) != group.node_ids.end()) {
            return &group;
        }
    }
    return nullptr;
}

void NodeEditor::RenderAnnotations() {
    if (annotations_.empty()) return;

    ImDrawList* draw_list = ImGui::GetWindowDrawList();
    ImVec2 editor_origin = ImGui::GetWindowPos();
    ImVec2 panning = ImNodes::EditorContextGetPanning();
    ImVec2 mouse_pos = ImGui::GetMousePos();

    for (auto& annotation : annotations_) {
        // Convert grid position to screen position
        ImVec2 screen_pos(
            editor_origin.x + annotation.position.x * zoom_ + panning.x,
            editor_origin.y + annotation.position.y * zoom_ + panning.y + 50.0f
        );

        float title_height = 24.0f * zoom_;
        float btn_size = 18.0f * zoom_;

        // Calculate actual size based on minimized state
        ImVec2 size;
        if (annotation.is_minimized) {
            size = ImVec2(annotation.size.x * zoom_, title_height);
        } else {
            size = ImVec2(annotation.size.x * zoom_, annotation.size.y * zoom_);
        }

        ImVec2 p_min = screen_pos;
        ImVec2 p_max(screen_pos.x + size.x, screen_pos.y + size.y);

        // Create invisible button to capture mouse input
        ImGui::SetCursorScreenPos(p_min);
        std::string btn_id = "##annotation_" + std::to_string(annotation.id);
        ImGui::InvisibleButton(btn_id.c_str(), size);

        bool is_hovered = ImGui::IsItemHovered();
        bool is_active = ImGui::IsItemActive();
        bool mouse_in_title = is_hovered && (mouse_pos.y >= p_min.y && mouse_pos.y <= p_min.y + title_height);

        // Minimize button bounds (right side of title bar)
        ImVec2 min_btn_min(p_max.x - btn_size - 4.0f * zoom_, p_min.y + 3.0f * zoom_);
        ImVec2 min_btn_max(p_max.x - 4.0f * zoom_, p_min.y + title_height - 3.0f * zoom_);
        bool mouse_on_min_btn = (mouse_pos.x >= min_btn_min.x && mouse_pos.x <= min_btn_max.x &&
                                  mouse_pos.y >= min_btn_min.y && mouse_pos.y <= min_btn_max.y);

        // Handle minimize button click
        if (mouse_on_min_btn && ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
            annotation.is_minimized = !annotation.is_minimized;
        }
        // Handle double-click to edit (not on minimize button)
        else if (is_hovered && !mouse_on_min_btn && ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)) {
            editing_annotation_ = true;
            editing_annotation_id_ = annotation.id;
            ImGui::OpenPopup("EditAnnotationPopup");
        }
        // Handle click to select and drag
        else if (is_hovered && ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
            selected_annotation_id_ = annotation.id;
            if (mouse_in_title && !mouse_on_min_btn) {
                dragging_annotation_id_ = annotation.id;
                annotation_drag_offset_ = ImVec2(mouse_pos.x - screen_pos.x, mouse_pos.y - screen_pos.y);
            }
        }

        // Handle dragging
        if (dragging_annotation_id_ == annotation.id && ImGui::IsMouseDown(ImGuiMouseButton_Left)) {
            ImVec2 new_screen_pos(mouse_pos.x - annotation_drag_offset_.x, mouse_pos.y - annotation_drag_offset_.y);
            annotation.position = ImVec2(
                (new_screen_pos.x - editor_origin.x - panning.x) / zoom_,
                (new_screen_pos.y - editor_origin.y - panning.y - 50.0f) / zoom_
            );
            screen_pos = new_screen_pos;
            p_min = screen_pos;
            p_max = ImVec2(screen_pos.x + size.x, screen_pos.y + size.y);
            min_btn_min = ImVec2(p_max.x - btn_size - 4.0f * zoom_, p_min.y + 3.0f * zoom_);
            min_btn_max = ImVec2(p_max.x - 4.0f * zoom_, p_min.y + title_height - 3.0f * zoom_);
        }

        if (dragging_annotation_id_ == annotation.id && ImGui::IsMouseReleased(ImGuiMouseButton_Left)) {
            dragging_annotation_id_ = -1;
        }

        // Colors
        ImU32 bg_color = annotation.color;
        ImU32 title_color = IM_COL32(
            static_cast<int>((bg_color & 0xFF) * 0.75f),
            static_cast<int>(((bg_color >> 8) & 0xFF) * 0.75f),
            static_cast<int>(((bg_color >> 16) & 0xFF) * 0.75f),
            255
        );

        // Draw title bar
        ImDrawFlags corners = annotation.is_minimized ? ImDrawFlags_RoundCornersAll : ImDrawFlags_RoundCornersTop;
        draw_list->AddRectFilled(p_min, ImVec2(p_max.x, p_min.y + title_height), title_color, 4.0f * zoom_, corners);

        // Draw content background (only if not minimized)
        if (!annotation.is_minimized) {
            draw_list->AddRectFilled(
                ImVec2(p_min.x, p_min.y + title_height), p_max,
                bg_color, 4.0f * zoom_, ImDrawFlags_RoundCornersBottom
            );
        }

        // Draw border
        ImU32 border_color = (is_hovered || is_active) ? IM_COL32(130, 130, 110, 255) : IM_COL32(100, 100, 80, 200);
        draw_list->AddRect(p_min, p_max, border_color, 4.0f * zoom_, 0, 1.5f * zoom_);

        // Draw minimize button
        ImU32 min_btn_color = mouse_on_min_btn ? IM_COL32(80, 70, 50, 255) : IM_COL32(100, 90, 70, 200);
        draw_list->AddRectFilled(min_btn_min, min_btn_max, min_btn_color, 3.0f * zoom_);
        // Draw - or + symbol
        float sym_y = (min_btn_min.y + min_btn_max.y) * 0.5f;
        float sym_margin = 4.0f * zoom_;
        draw_list->AddLine(
            ImVec2(min_btn_min.x + sym_margin, sym_y),
            ImVec2(min_btn_max.x - sym_margin, sym_y),
            IM_COL32(220, 210, 180, 255), 2.0f * zoom_
        );
        if (annotation.is_minimized) {
            // Draw + (add vertical line)
            float sym_x = (min_btn_min.x + min_btn_max.x) * 0.5f;
            draw_list->AddLine(
                ImVec2(sym_x, min_btn_min.y + sym_margin),
                ImVec2(sym_x, min_btn_max.y - sym_margin),
                IM_COL32(220, 210, 180, 255), 2.0f * zoom_
            );
        }

        // Draw title text
        ImFont* font = ImGui::GetFont();
        float title_font_size = 13.0f * zoom_;
        std::string display_title = annotation.title.empty() ? "Note" : annotation.title;
        ImVec2 title_pos(p_min.x + 8.0f * zoom_, p_min.y + 5.0f * zoom_);
        draw_list->AddText(font, title_font_size, title_pos, IM_COL32(50, 40, 20, 255), display_title.c_str());

        // Draw content text (only if not minimized)
        if (!annotation.is_minimized && !annotation.content.empty()) {
            float content_font_size = 12.0f * zoom_;
            float content_x = p_min.x + 8.0f * zoom_;
            float content_y = p_min.y + title_height + 6.0f * zoom_;
            float max_width = size.x - 16.0f * zoom_;
            draw_list->AddText(font, content_font_size, ImVec2(content_x, content_y),
                IM_COL32(40, 35, 20, 255), annotation.content.c_str(), nullptr, max_width);
        }

        // Placeholder text if empty and not minimized
        if (!annotation.is_minimized && annotation.content.empty()) {
            float content_font_size = 11.0f * zoom_;
            float content_x = p_min.x + 8.0f * zoom_;
            float content_y = p_min.y + title_height + 8.0f * zoom_;
            draw_list->AddText(font, content_font_size, ImVec2(content_x, content_y),
                IM_COL32(120, 110, 90, 180), "Double-click to edit...");
        }

        // Selection highlight
        if (selected_annotation_id_ == annotation.id) {
            draw_list->AddRect(
                ImVec2(p_min.x - 2, p_min.y - 2),
                ImVec2(p_max.x + 2, p_max.y + 2),
                IM_COL32(100, 150, 255, 200),
                6.0f * zoom_, 0, 2.0f * zoom_
            );
        }
    }

    // Render edit popup
    RenderAnnotationEditPopup();
}




void NodeEditor::RenderAnnotationEditPopup() {
    if (!editing_annotation_) return;

    // Find the annotation being edited
    CanvasAnnotation* ann = nullptr;
    for (auto& a : annotations_) {
        if (a.id == editing_annotation_id_) {
            ann = &a;
            break;
        }
    }

    if (!ann) {
        editing_annotation_ = false;
        editing_annotation_id_ = -1;
        return;
    }

    ImGui::SetNextWindowSize(ImVec2(400, 300), ImGuiCond_FirstUseEver);
    if (ImGui::BeginPopupModal("EditAnnotationPopup", &editing_annotation_, ImGuiWindowFlags_NoResize)) {
        ImGui::Text("Edit Annotation");
        ImGui::Separator();

        // Title input
        ImGui::Text("Title:");
        static char title_buf[256];
        if (ImGui::IsWindowAppearing()) {
            strncpy(title_buf, ann->title.c_str(), sizeof(title_buf) - 1);
            title_buf[sizeof(title_buf) - 1] = '\0';
        }
        ImGui::SetNextItemWidth(-1);
        ImGui::InputText("##ann_title", title_buf, sizeof(title_buf));

        ImGui::Spacing();

        // Content input
        ImGui::Text("Content:");
        static char content_buf[2048];
        if (ImGui::IsWindowAppearing()) {
            strncpy(content_buf, ann->content.c_str(), sizeof(content_buf) - 1);
            content_buf[sizeof(content_buf) - 1] = '\0';
        }
        ImGui::InputTextMultiline("##ann_content", content_buf, sizeof(content_buf),
            ImVec2(-1, 150), ImGuiInputTextFlags_AllowTabInput);

        ImGui::Spacing();

        // Color picker
        ImGui::Text("Color:");
        static ImVec4 color;
        if (ImGui::IsWindowAppearing()) {
            color = ImGui::ColorConvertU32ToFloat4(ann->color);
        }
        ImGui::ColorEdit4("##ann_color", &color.x, ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_AlphaBar);

        ImGui::Spacing();
        ImGui::Separator();

        // Buttons
        if (ImGui::Button("Save", ImVec2(120, 0))) {
            ann->title = title_buf;
            ann->content = content_buf;
            ann->color = ImGui::ColorConvertFloat4ToU32(color);
            editing_annotation_ = false;
            editing_annotation_id_ = -1;
            ImGui::CloseCurrentPopup();
        }
        ImGui::SameLine();
        if (ImGui::Button("Cancel", ImVec2(120, 0))) {
            editing_annotation_ = false;
            editing_annotation_id_ = -1;
            ImGui::CloseCurrentPopup();
        }
        ImGui::SameLine();
        ImGui::SetCursorPosX(ImGui::GetWindowWidth() - 130);
        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.7f, 0.2f, 0.2f, 1.0f));
        if (ImGui::Button("Delete", ImVec2(120, 0))) {
            // Remove annotation
            annotations_.erase(
                std::remove_if(annotations_.begin(), annotations_.end(),
                    [this](const CanvasAnnotation& a) { return a.id == editing_annotation_id_; }),
                annotations_.end()
            );
            editing_annotation_ = false;
            editing_annotation_id_ = -1;
            selected_annotation_id_ = -1;
            ImGui::CloseCurrentPopup();
        }
        ImGui::PopStyleColor();

        ImGui::EndPopup();
    } else {
        // Popup was closed
        editing_annotation_ = false;
        editing_annotation_id_ = -1;
    }
}

void NodeEditor::RenderGroups() {
    ImDrawList* draw_list = ImGui::GetWindowDrawList();

    for (auto& group : groups_) {
        if (group.node_ids.empty()) continue;

        // Calculate bounding box of all nodes in the group
        float min_x = std::numeric_limits<float>::max();
        float min_y = std::numeric_limits<float>::max();
        float max_x = std::numeric_limits<float>::lowest();
        float max_y = std::numeric_limits<float>::lowest();

        bool any_valid = false;
        for (int node_id : group.node_ids) {
            auto pos_it = cached_node_positions_.find(node_id);
            if (pos_it != cached_node_positions_.end()) {
                ImVec2 pos = pos_it->second;
                // Approximate node size
                float node_w = 150.0f;
                float node_h = 100.0f;

                min_x = std::min(min_x, pos.x);
                min_y = std::min(min_y, pos.y);
                max_x = std::max(max_x, pos.x + node_w);
                max_y = std::max(max_y, pos.y + node_h);
                any_valid = true;
            }
        }

        if (!any_valid) continue;

        // Add padding
        min_x -= group.padding;
        min_y -= group.padding;
        max_x += group.padding;
        max_y += group.padding;

        // Convert to screen coordinates
        ImVec2 panning = ImNodes::EditorContextGetPanning();
        ImVec2 origin = ImGui::GetCursorScreenPos();

        ImVec2 screen_min = ImVec2(origin.x + min_x + panning.x, origin.y + min_y + panning.y);
        ImVec2 screen_max = ImVec2(origin.x + max_x + panning.x, origin.y + max_y + panning.y);

        // Draw group background
        ImU32 fill_color = ImGui::ColorConvertFloat4ToU32(group.color);
        ImU32 border_color = ImGui::ColorConvertFloat4ToU32(
            ImVec4(group.color.x * 1.5f, group.color.y * 1.5f, group.color.z * 1.5f, 0.8f));

        draw_list->AddRectFilled(screen_min, screen_max, fill_color, 8.0f);
        draw_list->AddRect(screen_min, screen_max, border_color, 8.0f, 0, 2.0f);

        // Draw group label
        ImVec2 label_pos = ImVec2(screen_min.x + 8.0f, screen_min.y + 4.0f);
        draw_list->AddText(label_pos, IM_COL32(255, 255, 255, 220), group.name.c_str());
    }
}

// ===== Canvas Frames (Visual Organization Boxes) =====

void NodeEditor::RenderFrames() {
    ImDrawList* draw_list = ImGui::GetWindowDrawList();

    // Reset frame right-click flag each frame
    frame_right_clicked_ = false;

    const float HEADER_HEIGHT = 28.0f * zoom_;
    const float CORNER_RADIUS = 8.0f * zoom_;
    const float BORDER_WIDTH = 2.0f;
    const float RESIZE_HANDLE = 14.0f * zoom_;

    ImVec2 panning = ImNodes::EditorContextGetPanning();
    ImVec2 origin = ImGui::GetCursorScreenPos();

    for (auto& frame : frames_) {
        // Convert to screen coordinates
        ImVec2 screen_min = ImVec2(
            origin.x + frame.position.x * zoom_ + panning.x,
            origin.y + frame.position.y * zoom_ + panning.y);
        ImVec2 screen_max = ImVec2(
            screen_min.x + frame.size.x * zoom_,
            screen_min.y + frame.size.y * zoom_);

        // Colors
        ImU32 header_color = ImGui::ColorConvertFloat4ToU32(
            ImVec4(frame.color.x * 0.9f, frame.color.y * 0.9f, frame.color.z * 0.9f, 0.95f));
        ImU32 body_color = ImGui::ColorConvertFloat4ToU32(
            ImVec4(frame.color.x * 0.2f, frame.color.y * 0.2f, frame.color.z * 0.2f, 0.15f));
        ImU32 border_color = ImGui::ColorConvertFloat4ToU32(
            ImVec4(frame.color.x, frame.color.y, frame.color.z, frame.is_selected ? 1.0f : 0.6f));
        ImU32 text_color = IM_COL32(255, 255, 255, 230);
        ImU32 desc_color = IM_COL32(200, 200, 200, 180);

        // Draw body background
        ImVec2 body_min = ImVec2(screen_min.x, screen_min.y + HEADER_HEIGHT);
        draw_list->AddRectFilled(body_min, screen_max, body_color, CORNER_RADIUS,
            ImDrawFlags_RoundCornersBottom);

        // Draw header background
        ImVec2 header_max = ImVec2(screen_max.x, screen_min.y + HEADER_HEIGHT);
        draw_list->AddRectFilled(screen_min, header_max, header_color, CORNER_RADIUS,
            ImDrawFlags_RoundCornersTop);

        // Draw border (thicker if selected)
        float border_thick = frame.is_selected ? 3.0f : BORDER_WIDTH;
        draw_list->AddRect(screen_min, screen_max, border_color, CORNER_RADIUS, 0, border_thick);

        // Draw header line
        draw_list->AddLine(
            ImVec2(screen_min.x, screen_min.y + HEADER_HEIGHT),
            ImVec2(screen_max.x, screen_min.y + HEADER_HEIGHT),
            border_color, 1.0f);

        // Draw title
        ImVec2 title_pos = ImVec2(screen_min.x + 10.0f * zoom_, screen_min.y + 6.0f * zoom_);
        draw_list->AddText(title_pos, text_color, frame.title.c_str());

        // Draw description (if present)
        if (!frame.description.empty()) {
            ImVec2 desc_pos = ImVec2(screen_min.x + 10.0f * zoom_, screen_min.y + HEADER_HEIGHT + 8.0f * zoom_);

            // Truncate if needed
            float max_width = (screen_max.x - screen_min.x) - 20.0f * zoom_;
            std::string display_desc = frame.description;
            if (ImGui::CalcTextSize(display_desc.c_str()).x > max_width) {
                while (!display_desc.empty() && ImGui::CalcTextSize((display_desc + "...").c_str()).x > max_width) {
                    display_desc.pop_back();
                }
                display_desc += "...";
            }
            draw_list->AddText(desc_pos, desc_color, display_desc.c_str());
        }

        // Draw resize handle (bottom-right corner)
        draw_list->AddTriangleFilled(
            ImVec2(screen_max.x, screen_max.y - RESIZE_HANDLE),
            ImVec2(screen_max.x - RESIZE_HANDLE, screen_max.y),
            screen_max,
            IM_COL32(200, 200, 200, 120));

        // Handle interactions - header for drag, resize corner for resize
        ImGui::SetCursorScreenPos(screen_min);
        std::string header_id = "##frame_header_" + std::to_string(frame.id);
        ImGui::InvisibleButton(header_id.c_str(), ImVec2(screen_max.x - screen_min.x, HEADER_HEIGHT));

        bool header_hovered = ImGui::IsItemHovered();
        bool header_clicked = ImGui::IsItemClicked(0);
        bool header_right_clicked = ImGui::IsItemClicked(1);

        // Resize handle button
        ImVec2 resize_pos(screen_max.x - RESIZE_HANDLE, screen_max.y - RESIZE_HANDLE);
        ImGui::SetCursorScreenPos(resize_pos);
        std::string resize_id = "##frame_resize_" + std::to_string(frame.id);
        ImGui::InvisibleButton(resize_id.c_str(), ImVec2(RESIZE_HANDLE, RESIZE_HANDLE));

        bool resize_hovered = ImGui::IsItemHovered();
        bool resize_clicked = ImGui::IsItemClicked(0);

        // Change cursor on resize hover
        if (resize_hovered) {
            ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeNWSE);
        }

        // Check for right-click in body area
        ImVec2 mouse_pos = ImGui::GetMousePos();
        bool in_body = (mouse_pos.x >= screen_min.x && mouse_pos.x <= screen_max.x &&
                        mouse_pos.y > screen_min.y + HEADER_HEIGHT && mouse_pos.y <= screen_max.y);
        bool body_right_clicked = in_body && ImGui::IsMouseClicked(1);

        // Selection
        if (header_clicked || resize_clicked) {
            for (auto& f : frames_) f.is_selected = false;
            frame.is_selected = true;
            selected_frame_id_ = frame.id;
        }

        // Start dragging (header only)
        if (header_clicked && dragging_frame_id_ == -1 && resizing_frame_id_ == -1) {
            dragging_frame_id_ = frame.id;
            frame_drag_offset_ = ImVec2(mouse_pos.x - screen_min.x, mouse_pos.y - screen_min.y);
        }

        // Start resizing (resize handle)
        if (resize_clicked && resizing_frame_id_ == -1 && dragging_frame_id_ == -1) {
            resizing_frame_id_ = frame.id;
        }

        // Double-click header to edit
        if (header_hovered && ImGui::IsMouseDoubleClicked(0)) {
            editing_frame_ = true;
            editing_frame_id_ = frame.id;
            strncpy(frame_edit_title_, frame.title.c_str(), sizeof(frame_edit_title_) - 1);
            strncpy(frame_edit_desc_, frame.description.c_str(), sizeof(frame_edit_desc_) - 1);
        }

        // Right-click menu (header or body)
        if (header_right_clicked || body_right_clicked) {
            frame_right_clicked_ = true;  // Prevent canvas context menu
            ImGui::OpenPopup(("FrameMenu_" + std::to_string(frame.id)).c_str());
        }

        if (ImGui::BeginPopup(("FrameMenu_" + std::to_string(frame.id)).c_str())) {
            if (ImGui::MenuItem(ICON_FA_PEN " Edit Description")) {
                editing_frame_ = true;
                editing_frame_id_ = frame.id;
                strncpy(frame_edit_title_, frame.title.c_str(), sizeof(frame_edit_title_) - 1);
                strncpy(frame_edit_desc_, frame.description.c_str(), sizeof(frame_edit_desc_) - 1);
            }
            ImGui::Separator();
            if (ImGui::MenuItem(ICON_FA_COPY " Copy", "Ctrl+C")) {
                // TODO: Copy frame to clipboard
                spdlog::info("Copy frame {} (not yet implemented)", frame.id);
            }
            if (ImGui::MenuItem(ICON_FA_SCISSORS " Cut", "Ctrl+X")) {
                // TODO: Cut frame to clipboard
                spdlog::info("Cut frame {} (not yet implemented)", frame.id);
            }
            if (ImGui::MenuItem(ICON_FA_TRASH " Delete", "Del")) {
                DeleteFrame(frame.id);
                ImGui::EndPopup();
                break;  // List modified, exit loop
            }
            ImGui::Separator();
            if (ImGui::MenuItem(ICON_FA_OBJECT_GROUP " Group Nodes Inside")) {
                // Find all nodes inside this frame's bounds and group them
                std::vector<int> nodes_inside;
                for (const auto& node : nodes_) {
                    auto pos_it = cached_node_positions_.find(node.id);
                    if (pos_it != cached_node_positions_.end()) {
                        ImVec2 node_pos = pos_it->second;
                        if (node_pos.x >= frame.position.x &&
                            node_pos.x <= frame.position.x + frame.size.x &&
                            node_pos.y >= frame.position.y &&
                            node_pos.y <= frame.position.y + frame.size.y) {
                            nodes_inside.push_back(node.id);
                        }
                    }
                }
                if (nodes_inside.size() >= 2) {
                    // Select these nodes and create group
                    selected_node_ids_.clear();
                    for (int id : nodes_inside) {
                        selected_node_ids_.push_back(id);
                    }
                    CreateGroupFromSelection(frame.title);
                    spdlog::info("Grouped {} nodes inside frame '{}'", nodes_inside.size(), frame.title);
                } else {
                    spdlog::warn("Need at least 2 nodes inside frame to create a group");
                }
            }
            ImGui::EndPopup();
        }
    }

    // Handle dragging
    if (dragging_frame_id_ != -1) {
        if (ImGui::IsMouseDown(ImGuiMouseButton_Left)) {
            ImVec2 mouse_pos = ImGui::GetMousePos();
            for (auto& frame : frames_) {
                if (frame.id == dragging_frame_id_) {
                    frame.position.x = (mouse_pos.x - origin.x - panning.x - frame_drag_offset_.x) / zoom_;
                    frame.position.y = (mouse_pos.y - origin.y - panning.y - frame_drag_offset_.y) / zoom_;
                    break;
                }
            }
        } else {
            dragging_frame_id_ = -1;
        }
    }

    // Handle resizing (bottom-right corner only)
    if (resizing_frame_id_ != -1) {
        if (ImGui::IsMouseDown(ImGuiMouseButton_Left)) {
            ImVec2 mouse_pos = ImGui::GetMousePos();
            for (auto& frame : frames_) {
                if (frame.id == resizing_frame_id_) {
                    ImVec2 screen_min = ImVec2(
                        origin.x + frame.position.x * zoom_ + panning.x,
                        origin.y + frame.position.y * zoom_ + panning.y);
                    float new_w = (mouse_pos.x - screen_min.x) / zoom_;
                    float new_h = (mouse_pos.y - screen_min.y) / zoom_;
                    frame.size.x = std::max(100.0f, new_w);
                    frame.size.y = std::max(80.0f, new_h);
                    break;
                }
            }
        } else {
            resizing_frame_id_ = -1;
        }
    }

    // Frame edit popup
    if (editing_frame_) {
        ImGui::OpenPopup("Edit Frame");
        ImVec2 center = ImGui::GetMainViewport()->GetCenter();
        ImGui::SetNextWindowPos(center, ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));
    }

    if (ImGui::BeginPopupModal("Edit Frame", &editing_frame_, ImGuiWindowFlags_AlwaysAutoResize)) {
        ImGui::Text("Title:");
        ImGui::SetNextItemWidth(300);
        ImGui::InputText("##frame_title", frame_edit_title_, sizeof(frame_edit_title_));

        ImGui::Text("Description:");
        ImGui::SetNextItemWidth(300);
        ImGui::InputTextMultiline("##frame_desc", frame_edit_desc_, sizeof(frame_edit_desc_),
            ImVec2(300, 80));

        ImGui::Spacing();
        if (ImGui::Button("Save", ImVec2(100, 0))) {
            for (auto& frame : frames_) {
                if (frame.id == editing_frame_id_) {
                    frame.title = frame_edit_title_;
                    frame.description = frame_edit_desc_;
                    break;
                }
            }
            editing_frame_ = false;
            ImGui::CloseCurrentPopup();
        }
        ImGui::SameLine();
        if (ImGui::Button("Cancel", ImVec2(100, 0))) {
            editing_frame_ = false;
            ImGui::CloseCurrentPopup();
        }
        ImGui::EndPopup();
    }
}

void NodeEditor::AddFrameAt(const ImVec2& canvas_position) {
    CanvasFrame frame;
    frame.id = next_frame_id_++;
    frame.title = "Frame " + std::to_string(frame.id);
    frame.position = canvas_position;
    frame.size = ImVec2(300, 200);
    frames_.push_back(frame);
    spdlog::info("Added frame {} at ({}, {})", frame.id, canvas_position.x, canvas_position.y);
}

void NodeEditor::DeleteFrame(int frame_id) {
    frames_.erase(
        std::remove_if(frames_.begin(), frames_.end(),
            [frame_id](const CanvasFrame& f) { return f.id == frame_id; }),
        frames_.end());
    if (selected_frame_id_ == frame_id) {
        selected_frame_id_ = -1;
    }
    spdlog::info("Deleted frame {}", frame_id);
}

// ===== Subgraph Encapsulation =====

void NodeEditor::CreateSubgraphFromSelection(const std::string& name) {
    if (selected_node_ids_.size() < 2) {
        spdlog::warn("Need at least 2 selected nodes to create subgraph");
        return;
    }

    SaveUndoState();

    // Collect selected nodes and their internal links
    std::vector<MLNode> internal_nodes;
    std::vector<NodeLink> internal_links;
    std::set<int> selected_set(selected_node_ids_.begin(), selected_node_ids_.end());

    // Copy selected nodes to internal storage
    for (const auto& node : nodes_) {
        if (selected_set.count(node.id)) {
            internal_nodes.push_back(node);
        }
    }

    // Find internal links (both endpoints in selection)
    for (const auto& link : links_) {
        if (selected_set.count(link.from_node) && selected_set.count(link.to_node)) {
            internal_links.push_back(link);
        }
    }

    // Find boundary pins - inputs are pins with external sources, outputs have external destinations
    std::vector<std::pair<int, int>> input_pins;   // (node_id, pin_id) pairs
    std::vector<std::pair<int, int>> output_pins;  // (node_id, pin_id) pairs

    for (const auto& link : links_) {
        // Link from outside to inside -> input boundary
        if (!selected_set.count(link.from_node) && selected_set.count(link.to_node)) {
            input_pins.push_back({link.to_node, link.to_pin});
        }
        // Link from inside to outside -> output boundary
        if (selected_set.count(link.from_node) && !selected_set.count(link.to_node)) {
            output_pins.push_back({link.from_node, link.from_pin});
        }
    }

    // Calculate center position of selected nodes
    float center_x = 0, center_y = 0;
    int count = 0;
    for (int node_id : selected_node_ids_) {
        auto pos_it = cached_node_positions_.find(node_id);
        if (pos_it != cached_node_positions_.end()) {
            center_x += pos_it->second.x;
            center_y += pos_it->second.y;
            count++;
        }
    }
    if (count > 0) {
        center_x /= count;
        center_y /= count;
    }

    // Create the subgraph node
    MLNode subgraph_node;
    subgraph_node.id = next_node_id_++;
    subgraph_node.type = NodeType::Subgraph;
    subgraph_node.name = name.empty() ? "Subgraph" : name;
    subgraph_node.parameters["node_count"] = std::to_string(internal_nodes.size());

    // Create input pins for boundary inputs
    for (size_t i = 0; i < input_pins.size(); ++i) {
        NodePin pin;
        pin.id = next_pin_id_++;
        pin.type = PinType::Tensor;
        pin.name = "In " + std::to_string(i + 1);
        pin.is_input = true;
        subgraph_node.inputs.push_back(pin);
    }

    // Create output pins for boundary outputs
    for (size_t i = 0; i < output_pins.size(); ++i) {
        NodePin pin;
        pin.id = next_pin_id_++;
        pin.type = PinType::Tensor;
        pin.name = "Out " + std::to_string(i + 1);
        pin.is_input = false;
        subgraph_node.outputs.push_back(pin);
    }

    // Store subgraph data
    SubgraphData data;
    data.subgraph_node_id = subgraph_node.id;
    data.internal_nodes = std::move(internal_nodes);
    data.internal_links = std::move(internal_links);
    data.expanded = false;

    // Store pin mappings
    for (const auto& [node_id, pin_id] : input_pins) {
        data.input_pin_mappings.push_back(pin_id);
    }
    for (const auto& [node_id, pin_id] : output_pins) {
        data.output_pin_mappings.push_back(pin_id);
    }

    subgraphs_.push_back(std::move(data));

    // Rewire external connections to the subgraph node
    std::vector<NodeLink> links_to_add;
    std::vector<int> links_to_remove;

    for (size_t i = 0; i < links_.size(); ++i) {
        const auto& link = links_[i];

        // Link from outside to inside -> connect to subgraph input
        if (!selected_set.count(link.from_node) && selected_set.count(link.to_node)) {
            // Find which input pin this maps to
            for (size_t j = 0; j < input_pins.size(); ++j) {
                if (input_pins[j].second == link.to_pin) {
                    NodeLink new_link;
                    new_link.id = next_link_id_++;
                    new_link.from_node = link.from_node;
                    new_link.from_pin = link.from_pin;
                    new_link.to_node = subgraph_node.id;
                    new_link.to_pin = subgraph_node.inputs[j].id;
                    new_link.type = link.type;
                    links_to_add.push_back(new_link);
                    break;
                }
            }
            links_to_remove.push_back(static_cast<int>(i));
        }

        // Link from inside to outside -> connect from subgraph output
        if (selected_set.count(link.from_node) && !selected_set.count(link.to_node)) {
            // Find which output pin this maps to
            for (size_t j = 0; j < output_pins.size(); ++j) {
                if (output_pins[j].second == link.from_pin) {
                    NodeLink new_link;
                    new_link.id = next_link_id_++;
                    new_link.from_node = subgraph_node.id;
                    new_link.from_pin = subgraph_node.outputs[j].id;
                    new_link.to_node = link.to_node;
                    new_link.to_pin = link.to_pin;
                    new_link.type = link.type;
                    links_to_add.push_back(new_link);
                    break;
                }
            }
            links_to_remove.push_back(static_cast<int>(i));
        }

        // Internal links are removed from main graph
        if (selected_set.count(link.from_node) && selected_set.count(link.to_node)) {
            links_to_remove.push_back(static_cast<int>(i));
        }
    }

    // Remove old links (in reverse order to maintain indices)
    std::sort(links_to_remove.begin(), links_to_remove.end(), std::greater<int>());
    for (int idx : links_to_remove) {
        links_.erase(links_.begin() + idx);
    }

    // Add new links
    for (auto& link : links_to_add) {
        links_.push_back(link);
    }

    // Remove selected nodes from main graph
    nodes_.erase(
        std::remove_if(nodes_.begin(), nodes_.end(),
            [&selected_set](const MLNode& n) { return selected_set.count(n.id); }),
        nodes_.end()
    );

    // Add subgraph node
    nodes_.push_back(subgraph_node);

    // Position the subgraph node at the center of removed nodes
    pending_positions_[subgraph_node.id] = ImVec2(center_x, center_y);
    pending_positions_frames_ = 3;

    // Clear selection and select the new subgraph
    selected_node_ids_.clear();
    selected_node_ids_.push_back(subgraph_node.id);

    spdlog::info("Created subgraph '{}' with {} internal nodes",
                 subgraph_node.name, subgraphs_.back().internal_nodes.size());
}

void NodeEditor::ExpandSubgraph(int node_id) {
    SubgraphData* data = GetSubgraphData(node_id);
    if (!data) {
        spdlog::warn("Node {} is not a subgraph", node_id);
        return;
    }

    if (data->expanded) return;

    SaveUndoState();
    data->expanded = true;

    // Get position of subgraph node
    ImVec2 base_pos = ImVec2(0, 0);
    auto pos_it = cached_node_positions_.find(node_id);
    if (pos_it != cached_node_positions_.end()) {
        base_pos = pos_it->second;
    }

    // Add internal nodes back to the main graph
    float offset_x = 0, offset_y = 50;
    for (auto& internal_node : data->internal_nodes) {
        // Offset position relative to subgraph node
        pending_positions_[internal_node.id] = ImVec2(base_pos.x + offset_x, base_pos.y + offset_y);
        nodes_.push_back(internal_node);
        offset_x += 180;
        if (offset_x > 500) {
            offset_x = 0;
            offset_y += 120;
        }
    }

    // Add internal links back
    for (const auto& link : data->internal_links) {
        links_.push_back(link);
    }

    pending_positions_frames_ = 3;
    spdlog::info("Expanded subgraph {}", node_id);
}

void NodeEditor::CollapseSubgraph(int node_id) {
    SubgraphData* data = GetSubgraphData(node_id);
    if (!data) return;

    if (!data->expanded) return;

    SaveUndoState();
    data->expanded = false;

    // Remove internal nodes from main graph
    std::set<int> internal_ids;
    for (const auto& node : data->internal_nodes) {
        internal_ids.insert(node.id);
    }

    nodes_.erase(
        std::remove_if(nodes_.begin(), nodes_.end(),
            [&internal_ids](const MLNode& n) { return internal_ids.count(n.id); }),
        nodes_.end()
    );

    // Remove internal links
    links_.erase(
        std::remove_if(links_.begin(), links_.end(),
            [&internal_ids](const NodeLink& l) {
                return internal_ids.count(l.from_node) && internal_ids.count(l.to_node);
            }),
        links_.end()
    );

    spdlog::info("Collapsed subgraph {}", node_id);
}

void NodeEditor::ToggleSubgraphExpansion(int node_id) {
    SubgraphData* data = GetSubgraphData(node_id);
    if (!data) return;

    if (data->expanded) {
        CollapseSubgraph(node_id);
    } else {
        ExpandSubgraph(node_id);
    }
}

bool NodeEditor::IsSubgraphNode(int node_id) const {
    const MLNode* node = FindNodeById(node_id);
    return node && node->type == NodeType::Subgraph;
}

SubgraphData* NodeEditor::GetSubgraphData(int node_id) {
    for (auto& data : subgraphs_) {
        if (data.subgraph_node_id == node_id) {
            return &data;
        }
    }
    return nullptr;
}


// ========== Menu Operations Implementation ==========

void NodeEditor::AddNodeFromMenu(NodeType type, const std::string& name) {
    // Get center of the visible area for node placement
    ImVec2 panning = ImNodes::EditorContextGetPanning();
    ImVec2 visible_center(-panning.x + 400, -panning.y + 300);
    
    // Queue the node for addition (deferred to avoid modifying nodes_ during rendering)
    PendingNode pending;
    pending.type = type;
    pending.name = name;
    pending.position = visible_center;
    pending_nodes_.push_back(pending);
    
    spdlog::info("Menu: Adding {} node at center of view", name);
}

void NodeEditor::DeleteSelectedNodes() {
    // Reuse existing DeleteSelected logic
    DeleteSelected();
}

void NodeEditor::DuplicateSelectedNodes() {
    // Reuse existing DuplicateSelection logic
    DuplicateSelection();
}

void NodeEditor::GroupSelectedNodes() {
    // TODO: Implement node grouping
    spdlog::info("Group selected nodes - not yet implemented");
}

void NodeEditor::UngroupSelectedNodes() {
    // TODO: Implement node ungrouping
    spdlog::info("Ungroup selected nodes - not yet implemented");
}

// ===== Graph Simulation =====

bool NodeEditor::HasSimulationNodes() const {
    for (const auto& node : nodes_) {
        if (node.type == NodeType::SignalSlider ||
            node.type == NodeType::SineWave ||
            node.type == NodeType::StepSignal ||
            node.type == NodeType::RampSignal ||
            node.type == NodeType::SignalScope ||
            node.type == NodeType::Constant) {
            return true;
        }
        // Check for MuJoCo Plant or other simulation plugin nodes
        if (node.type == NodeType::PluginCustom) {
            auto qname = node.plugin_qualified_name;
            if (qname.find("MuJoCoPlant") != std::string::npos ||
                qname.find("MuJoCoEnv") != std::string::npos) {
                return true;
            }
        }
    }
    return false;
}

void NodeEditor::OnRunSimulation() {
    if (is_simulating_) return;

    spdlog::info("NodeEditor: Starting graph simulation");

    // Create and build executor
    graph_executor_ = std::make_unique<cyxwiz::GraphExecutor>();

    // Set plugin eval callback: routes to PluginNodeRegistry → plugin DLL
    graph_executor_->SetPluginEvalCallback(
        [](const std::string& plugin_qualified_name,
           const cyxwiz::NodeEvalContext& ctx) -> cyxwiz::NodeEvalResult {

            // Convert to plugin types
            cyxwiz::plugin::PluginNodeEvalContext pctx;
            pctx.node_type_name = ctx.node_type_name;
            pctx.parameters = ctx.parameters;
            pctx.sim_time = ctx.sim_time;
            pctx.dt = ctx.dt;
            for (const auto& [name, val] : ctx.input_values) {
                if (std::holds_alternative<float>(val)) {
                    pctx.input_values[name] = std::get<float>(val);
                } else if (std::holds_alternative<std::vector<float>>(val)) {
                    pctx.input_values[name] = std::get<std::vector<float>>(val);
                } else if (std::holds_alternative<std::string>(val)) {
                    pctx.input_values[name] = std::get<std::string>(val);
                }
            }

            // Route to plugin via registry
            auto provider = cyxwiz::plugin::PluginNodeRegistry::Instance()
                                .GetNodeProvider(plugin_qualified_name);
            if (!provider) {
                cyxwiz::NodeEvalResult r;
                r.success = false;
                r.error_message = "No plugin provider for: " + plugin_qualified_name;
                return r;
            }

            auto presult = provider->EvaluateNode(pctx);

            // Convert back
            cyxwiz::NodeEvalResult result;
            result.success = presult.success;
            result.error_message = presult.error_message;
            for (const auto& [name, val] : presult.output_values) {
                if (std::holds_alternative<float>(val)) {
                    result.output_values[name] = std::get<float>(val);
                } else if (std::holds_alternative<std::vector<float>>(val)) {
                    result.output_values[name] = std::get<std::vector<float>>(val);
                } else if (std::holds_alternative<std::string>(val)) {
                    result.output_values[name] = std::get<std::string>(val);
                }
            }
            return result;
        }
    );

    if (!graph_executor_->Build(nodes_, links_)) {
        spdlog::error("NodeEditor: Graph build failed: {}", graph_executor_->GetError());
        graph_executor_.reset();
        return;
    }

    // Launch simulation thread
    sim_stop_requested_ = false;
    is_simulating_ = true;

    sim_thread_ = std::thread([this]() {
        float dt = 1.0f / sim_rate_hz_;
        auto tick_duration = std::chrono::microseconds(static_cast<int>(1000000.0f / sim_rate_hz_));

        while (!sim_stop_requested_) {
            auto start = std::chrono::steady_clock::now();

            if (!graph_executor_->Tick(dt)) {
                spdlog::warn("NodeEditor: Simulation tick failed: {}", graph_executor_->GetError());
                break;
            }

            // Sleep to maintain target rate
            auto elapsed = std::chrono::steady_clock::now() - start;
            auto remaining = tick_duration - elapsed;
            if (remaining > std::chrono::microseconds(0)) {
                std::this_thread::sleep_for(remaining);
            }
        }

        is_simulating_ = false;
        spdlog::info("NodeEditor: Simulation stopped at t={:.2f}s", graph_executor_->GetSimTime());
    });
}

void NodeEditor::OnStopSimulation() {
    if (!is_simulating_) return;

    spdlog::info("NodeEditor: Stopping graph simulation");
    sim_stop_requested_ = true;

    if (sim_thread_.joinable()) {
        sim_thread_.join();
    }

    is_simulating_ = false;
}

// ===== RL Training =====

bool NodeEditor::HasRLNodes() const {
    for (const auto& node : nodes_) {
        if (node.type == NodeType::RLTraining ||
            node.type == NodeType::PolicyNetwork ||
            node.type == NodeType::ValueNetwork) {
            return true;
        }
    }
    return false;
}

void NodeEditor::OnStartRLTraining() {
    if (rl_script_running_ || (rl_executor_ && rl_executor_->IsTraining())) return;
    if (!scripting_engine_) {
        spdlog::error("NodeEditor: ScriptingEngine not set, cannot run Python RL training");
        return;
    }
    if (scripting_engine_->IsScriptRunning()) {
        spdlog::warn("NodeEditor: A script is already running");
        return;
    }

    spdlog::info("NodeEditor: Starting Python-based RL training");

    // Build config from RLTraining node parameters
    cyxwiz::RLTrainingConfig config;

    for (const auto& node : nodes_) {
        if (node.type == NodeType::RLTraining) {
            auto it = node.parameters.find("total_timesteps");
            if (it != node.parameters.end()) config.total_timesteps = std::stoi(it->second);
            it = node.parameters.find("max_episode_steps");
            if (it != node.parameters.end()) config.max_episode_steps = std::stoi(it->second);
            it = node.parameters.find("learning_rate");
            if (it != node.parameters.end()) config.learning_rate = std::stof(it->second);
            it = node.parameters.find("gamma");
            if (it != node.parameters.end()) config.gamma = std::stof(it->second);
            it = node.parameters.find("clip_range");
            if (it != node.parameters.end()) config.clip_range = std::stof(it->second);
            it = node.parameters.find("n_steps");
            if (it != node.parameters.end()) config.n_steps = std::stoi(it->second);
            it = node.parameters.find("batch_size");
            if (it != node.parameters.end()) config.batch_size = std::stoi(it->second);
            it = node.parameters.find("n_epochs");
            if (it != node.parameters.end()) config.n_epochs = std::stoi(it->second);
            break;
        }
    }

    // Find MuJoCo Plant node for MJCF path
    for (const auto& node : nodes_) {
        if (node.type == NodeType::PluginCustom) {
            if (node.plugin_qualified_name.find("MuJoCoPlant") != std::string::npos) {
                config.plugin_qualified_name = node.plugin_qualified_name;
                auto mp = node.parameters.find("mjcf_path");
                if (mp != node.parameters.end() && !mp->second.empty()) {
                    config.env_mjcf_path = mp->second;
                } else {
                    auto meta = node.parameters.find("_meta_loaded_path");
                    if (meta != node.parameters.end()) config.env_mjcf_path = meta->second;
                }
                break;
            }
        }
    }

    // Extract RewardFunction and ObservationFilter node params
    std::map<std::string, std::string> reward_params, obs_filter_params;
    for (const auto& node : nodes_) {
        if (node.type == NodeType::PluginCustom) {
            if (node.plugin_qualified_name.find("RewardFunction") != std::string::npos) {
                reward_params = node.parameters;
            }
            if (node.plugin_qualified_name.find("ObservationFilter") != std::string::npos) {
                obs_filter_params = node.parameters;
            }
        }
    }

    // Determine save path
    std::string save_path = "models/rl_policy";

    // Generate Python script
    std::string script = cyxwiz::RLScriptGenerator::Generate(
        config, reward_params, obs_filter_params, save_path);

    // Create/show dashboard
    if (!rl_dashboard_) {
        rl_dashboard_ = std::make_shared<cyxwiz::TrainingDashboardPanel>();
    }
    rl_dashboard_->SetRLTrainingState(true);
    rl_dashboard_->SetVisible(true);
    rl_dashboard_->ResetRLMetrics();

    // Set up pycyxwiz flags
    std::string setup_script = "import pycyxwiz\npycyxwiz.rl_set_stop(False)\npycyxwiz.rl_set_paused(False)\n";
    scripting_engine_->ExecuteCommand(setup_script);

    // Set completion callback
    auto dashboard = rl_dashboard_;
    rl_script_running_ = true;
    scripting_engine_->SetCompletionCallback([this, dashboard](const scripting::ExecutionResult& result) {
        rl_script_running_ = false;
        dashboard->SetRLTrainingState(false);
        if (!result.success) {
            spdlog::error("RL training script failed: {}", result.error_message);
        } else {
            spdlog::info("RL training script completed successfully");
        }
    });

    // Run training script async
    scripting_engine_->ExecuteScriptAsync(script);
}


// ===== ONNX Export =====

void NodeEditor::ExportPolicyONNX(const std::string& output_path) {
    if (!rl_executor_) {
        spdlog::error("NodeEditor: No RL executor for ONNX export");
        return;
    }

    auto metrics = rl_executor_->GetMetrics();
    if (metrics.episode_count == 0) {
        spdlog::error("NodeEditor: No trained policy to export");
        return;
    }

    // Request export via plugin's EvaluateNode with "export_onnx" command
    std::string plugin_qname;
    for (const auto& node : nodes_) {
        if (node.type == NodeType::PluginCustom &&
            node.plugin_qualified_name.find("MuJoCo") != std::string::npos) {
            plugin_qname = node.plugin_qualified_name;
            break;
        }
    }

    if (plugin_qname.empty()) {
        spdlog::error("NodeEditor: No MuJoCo plugin node found for ONNX export");
        return;
    }

    auto* provider = cyxwiz::plugin::PluginNodeRegistry::Instance().GetNodeProvider(plugin_qname);
    if (!provider) {
        spdlog::error("NodeEditor: Plugin provider not found");
        return;
    }

    cyxwiz::plugin::PluginNodeEvalContext ctx;
    ctx.node_type_name = "RLAgent";
    ctx.parameters["command"] = "export_onnx";
    ctx.parameters["output_path"] = output_path;

    auto result = provider->EvaluateNode(ctx);
    if (result.success) {
        spdlog::info("NodeEditor: Policy exported to {}", output_path);
    } else {
        spdlog::warn("NodeEditor: ONNX export: {}", result.error_message.empty() ? "not yet fully wired" : result.error_message);
    }
}

void NodeEditor::OnStopRLTraining() {
    spdlog::info("NodeEditor: Stopping RL training");

    // Stop Python-based training
    if (rl_script_running_ && scripting_engine_) {
        // Signal Python to stop via pycyxwiz atomic flag
        scripting_engine_->ExecuteCommand("import pycyxwiz; pycyxwiz.rl_set_stop(True)");
        scripting_engine_->StopScript();
        rl_script_running_ = false;
    }

    // Stop old C++ executor (for backward compat)
    if (rl_executor_) {
        rl_executor_->Stop();
        rl_executor_.reset();
    }

    if (rl_dashboard_) {
        rl_dashboard_->SetRLTrainingState(false);
    }
}

// ============================================================================
// Phase 5 Week 7 - Data Studio Integration
// ============================================================================

void NodeEditor::SetDatasetFromDataStudio(const std::string& dataset_name) {
    spdlog::info("[Node Editor] Receiving dataset from Data Studio: '{}'", dataset_name);

    // Find or create DatasetInput node
    MLNode* dataset_input = nullptr;
    for (auto& node : nodes_) {
        if (node.type == NodeType::DatasetInput) {
            dataset_input = &node;
            spdlog::info("[Node Editor] Found existing DatasetInput node (ID: {})", node.id);
            break;
        }
    }

    if (!dataset_input) {
        // Create new DatasetInput node at center
        spdlog::info("[Node Editor] Creating new DatasetInput node");

        // Find center of visible area
        ImVec2 center_pos = FindEmptyPosition();

        // Create the node
        MLNode new_node = CreateNode(NodeType::DatasetInput, "Dataset from Data Studio");
        new_node.parameters["dataset_name"] = dataset_name;
        new_node.parameters["split"] = "train";

        nodes_.push_back(new_node);
        dataset_input = &nodes_.back();

        // Rebuild pin lookup after adding node
        RebuildPinLookup();

        // Set position for next frame
        pending_positions_[dataset_input->id] = center_pos;
        pending_positions_frames_ = 3;  // Apply for 3 frames to ensure ImNodes registers it
    } else {
        // Update existing DatasetInput node
        dataset_input->parameters["dataset_name"] = dataset_name;
        dataset_input->name = "Dataset: " + dataset_name;
    }

    // Trigger shape inference
    if (shape_inference_) {
        shape_inference_->ComputeAllShapes(nodes_, links_);
    }

    // Save undo state
    SaveUndoState();

    // Frame the DatasetInput node (zoom to it)
    selected_node_ids_.clear();
    selected_node_ids_.push_back(dataset_input->id);

    // Frame selected will be called on next Render()
    // We set a flag to defer this until we're in the ImNodes context
    // For now, just select the node - user can press 'F' to frame it

    spdlog::info("[Node Editor] Dataset '{}' successfully set in DatasetInput node (ID: {})",
                 dataset_name, dataset_input->id);
}

// Unified Canvas Phase 4.2: Execute data pipeline using DuckDB/Arrow
bool NodeEditor::ExecuteDataPipeline() {
    spdlog::info("Executing data transformation pipeline...");

    if (nodes_.empty()) {
        spdlog::warn("No nodes in graph - cannot execute pipeline");
        return false;
    }

    // Check if we have a pipeline executor
    if (!pipeline_executor_) {
        pipeline_executor_ = std::make_unique<cyxwiz::PipelineExecutor>();
        spdlog::info("Created PipelineExecutor instance");
    }

    // Convert node graph to JSON for PipelineExecutor
    nlohmann::json pipeline_json;
    pipeline_json["nodes"] = nlohmann::json::array();

    for (const auto& node : nodes_) {
        nlohmann::json node_json;
        node_json["id"] = node.id;
        node_json["type"] = GetNodeTypeName(node.type);
        node_json["name"] = node.name;

        // Add parameters
        node_json["parameters"] = nlohmann::json::object();
        for (const auto& [key, value] : node.parameters) {
            node_json["parameters"][key] = value;
        }

        // Add inputs (connected node IDs)
        node_json["inputs"] = nlohmann::json::array();
        for (const auto& link : links_) {
            // Find if this link connects TO this node
            for (const auto& input_pin : node.inputs) {
                if (link.to_pin == input_pin.id) {
                    // Find the source node
                    for (const auto& src_node : nodes_) {
                        for (const auto& out_pin : src_node.outputs) {
                            if (link.from_pin == out_pin.id) {
                                node_json["inputs"].push_back(src_node.id);
                                break;
                            }
                        }
                    }
                }
            }
        }

        // Add outputs (connected node IDs)
        node_json["outputs"] = nlohmann::json::array();
        for (const auto& link : links_) {
            // Find if this link connects FROM this node
            for (const auto& output_pin : node.outputs) {
                if (link.from_pin == output_pin.id) {
                    // Find the destination node
                    for (const auto& dst_node : nodes_) {
                        for (const auto& in_pin : dst_node.inputs) {
                            if (link.to_pin == in_pin.id) {
                                node_json["outputs"].push_back(dst_node.id);
                                break;
                            }
                        }
                    }
                }
            }
        }

        pipeline_json["nodes"].push_back(node_json);
    }

    // Set up progress callback
    pipeline_executor_->SetProgressCallback([](float progress, const std::string& status) {
        spdlog::info("Pipeline progress: {:.1f}% - {}", progress * 100, status);
    });

    // Set up completion callback
    pipeline_executor_->SetCompletionCallback([](bool success) {
        if (success) {
            spdlog::info("Pipeline execution completed successfully!");
        } else {
            spdlog::error("Pipeline execution failed!");
        }
    });

    // Execute the pipeline
    std::string pipeline_str = pipeline_json.dump(2);
    spdlog::debug("Pipeline JSON:\n{}", pipeline_str);

    bool success = pipeline_executor_->ExecutePipeline(pipeline_str);
    if (!success) {
        spdlog::error("Pipeline execution failed: {}", pipeline_executor_->GetLastError());
    }
    return success;
}

// Helper to get node type name as string for PipelineExecutor
std::string NodeEditor::GetNodeTypeName(NodeType type) const {
    switch (type) {
        // Smart I/O nodes (universal data input/output)
        case NodeType::DataInput: return "DataInput";
        case NodeType::DataOutput: return "DataOutput";
        case NodeType::CSVFile: return "FileInput";
        case NodeType::FilterRows: return "FilterRows";
        case NodeType::SelectColumns: return "SelectColumns";
        case NodeType::JoinTables: return "Join";
        case NodeType::GroupByAggregate: return "GroupBy";
        case NodeType::SortRows: return "SortRows";
        case NodeType::FillMissingValues: return "FillMissing";
        case NodeType::RemoveDuplicateRows: return "RemoveDuplicates";
        case NodeType::RenameColumns: return "RenameColumns";
        case NodeType::SampleRows: return "SampleRows";
        case NodeType::SQLQuery: return "SQLQuery";
        case NodeType::ParquetFile: return "ParquetInput";
        case NodeType::ExportCSV: return "ExportCSV";
        case NodeType::ExportParquet: return "ExportParquet";
        case NodeType::ExportJSON: return "ExportJSON";
        case NodeType::DescribeStats: return "DescribeStats";
        // KNIME-style table manipulation nodes
        case NodeType::ExcelFile: return "ExcelInput";
        case NodeType::ExportExcel: return "ExportExcel";
        case NodeType::RowToColumnNames: return "RowToColumnNames";
        case NodeType::TableSplitter: return "TableSplitter";
        case NodeType::CellExtractor: return "CellExtractor";
        case NodeType::CellUpdater: return "CellUpdater";
        case NodeType::TableCropper: return "TableCropper";
        case NodeType::ColumnAppender: return "ColumnAppender";
        case NodeType::RowAppender: return "RowAppender";
        case NodeType::Unpivot: return "Unpivot";
        case NodeType::StringManipulation: return "StringManipulation";
        case NodeType::MathFormula: return "MathFormula";
        case NodeType::RuleEngine: return "RuleEngine";
        default: return "Unknown";
    }
}

// ===== Unified Canvas Phase 6: Execution Visualization =====

void NodeEditor::SetNodeExecutionState(int node_id, NodeExecutionState state) {
    node_execution_states_[node_id] = state;
    if (state == NodeExecutionState::Executing) {
        currently_executing_node_id_ = node_id;
    } else if (currently_executing_node_id_ == node_id) {
        currently_executing_node_id_ = -1;
    }
}

void NodeEditor::SetNodeExecutionError(int node_id, const std::string& error) {
    node_execution_errors_[node_id] = error;
    SetNodeExecutionState(node_id, NodeExecutionState::Error);
}

void NodeEditor::ClearExecutionStates() {
    node_execution_states_.clear();
    node_execution_errors_.clear();
    currently_executing_node_id_ = -1;
}

} // namespace gui
