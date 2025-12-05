#include "node_editor.h"
#include "node_documentation.h"
#include "panels/script_editor.h"
#include "properties.h"
#include "patterns/pattern_library.h"
#include "icons.h"
#include "../core/data_registry.h"
#include "../core/training_manager.h"
#include "../core/async_task_manager.h"
#include "../core/project_manager.h"
#include <imgui.h>
#include <imnodes.h>
#include <spdlog/spdlog.h>
#include <algorithm>
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

NodeEditor::NodeEditor()
    : show_window_(true),
      next_node_id_(1),
      next_pin_id_(1),
      next_link_id_(1),
      show_context_menu_(false),
      context_menu_node_id_(-1),
      selected_node_id_(-1),
      selected_framework_(CodeFramework::PyTorch),
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

    // Create a Linear Attention Transformer showcase model
    // This demonstrates the O(n) Linear Attention node for efficient sequence processing
    // Architecture: Embedding -> PositionalEncoding -> LinearAttention -> LayerNorm -> FFN -> Output

    // ========== INPUT PIPELINE (Left side) ==========

    // 1. Dataset Input - Sequence data (e.g., text tokens)
    MLNode dataset_input = CreateNode(NodeType::DatasetInput, "Sequence Data");
    dataset_input.parameters["dataset_name"] = "";
    dataset_input.parameters["split"] = "train";
    nodes_.push_back(dataset_input);
    ImNodes::SetNodeGridSpacePos(dataset_input.id, ImVec2(50.0f, 200.0f));

    // 2. Embedding - Convert token IDs to vectors
    MLNode embedding = CreateNode(NodeType::Embedding, "Token Embedding");
    embedding.parameters["num_embeddings"] = "30000";  // Vocabulary size
    embedding.parameters["embedding_dim"] = "512";
    nodes_.push_back(embedding);
    ImNodes::SetNodeGridSpacePos(embedding.id, ImVec2(250.0f, 200.0f));

    // 3. Positional Encoding - Add position information
    MLNode pos_enc = CreateNode(NodeType::PositionalEncoding, "Positional Encoding");
    pos_enc.parameters["max_seq_len"] = "512";
    pos_enc.parameters["embed_dim"] = "512";
    nodes_.push_back(pos_enc);
    ImNodes::SetNodeGridSpacePos(pos_enc.id, ImVec2(450.0f, 200.0f));

    // ========== LINEAR ATTENTION BLOCK (Center - Purple) ==========

    // 4. Linear Attention - O(n) efficient attention (MAIN SHOWCASE)
    MLNode linear_attn = CreateNode(NodeType::LinearAttention, "Linear Attention");
    linear_attn.parameters["embed_dim"] = "512";
    linear_attn.parameters["num_heads"] = "8";
    linear_attn.parameters["feature_map"] = "elu";  // ELU feature map (Performer-style)
    linear_attn.parameters["eps"] = "1e-6";
    nodes_.push_back(linear_attn);
    ImNodes::SetNodeGridSpacePos(linear_attn.id, ImVec2(700.0f, 200.0f));

    // 5. Add (Residual Connection) - Skip connection around attention
    MLNode residual1 = CreateNode(NodeType::Add, "Residual Add");
    nodes_.push_back(residual1);
    ImNodes::SetNodeGridSpacePos(residual1.id, ImVec2(950.0f, 200.0f));

    // 6. Layer Normalization after attention
    MLNode layer_norm1 = CreateNode(NodeType::LayerNorm, "LayerNorm");
    layer_norm1.parameters["normalized_shape"] = "512";
    layer_norm1.parameters["eps"] = "1e-5";
    nodes_.push_back(layer_norm1);
    ImNodes::SetNodeGridSpacePos(layer_norm1.id, ImVec2(1150.0f, 200.0f));

    // ========== FEED-FORWARD NETWORK ==========

    // 7. Dense (FFN expand) - 4x expansion
    MLNode ffn1 = CreateNode(NodeType::Dense, "FFN Expand (2048)");
    ffn1.parameters["units"] = "2048";
    nodes_.push_back(ffn1);
    ImNodes::SetNodeGridSpacePos(ffn1.id, ImVec2(1350.0f, 100.0f));

    // 8. GELU Activation
    MLNode gelu = CreateNode(NodeType::GELU, "GELU");
    nodes_.push_back(gelu);
    ImNodes::SetNodeGridSpacePos(gelu.id, ImVec2(1550.0f, 100.0f));

    // 9. Dense (FFN contract) - Back to 512
    MLNode ffn2 = CreateNode(NodeType::Dense, "FFN Contract (512)");
    ffn2.parameters["units"] = "512";
    nodes_.push_back(ffn2);
    ImNodes::SetNodeGridSpacePos(ffn2.id, ImVec2(1750.0f, 100.0f));

    // 10. Dropout
    MLNode dropout = CreateNode(NodeType::Dropout, "Dropout");
    dropout.parameters["rate"] = "0.1";
    nodes_.push_back(dropout);
    ImNodes::SetNodeGridSpacePos(dropout.id, ImVec2(1350.0f, 300.0f));

    // 11. Add (Residual Connection) - Skip connection around FFN
    MLNode residual2 = CreateNode(NodeType::Add, "Residual Add");
    nodes_.push_back(residual2);
    ImNodes::SetNodeGridSpacePos(residual2.id, ImVec2(1550.0f, 300.0f));

    // 12. Layer Normalization after FFN
    MLNode layer_norm2 = CreateNode(NodeType::LayerNorm, "LayerNorm");
    layer_norm2.parameters["normalized_shape"] = "512";
    nodes_.push_back(layer_norm2);
    ImNodes::SetNodeGridSpacePos(layer_norm2.id, ImVec2(1750.0f, 300.0f));

    // ========== OUTPUT HEAD ==========

    // 13. Output Dense - Classification head
    MLNode output_dense = CreateNode(NodeType::Dense, "Output Dense");
    output_dense.parameters["units"] = "10";  // 10 classes
    nodes_.push_back(output_dense);
    ImNodes::SetNodeGridSpacePos(output_dense.id, ImVec2(1950.0f, 200.0f));

    // 14. Softmax
    MLNode softmax = CreateNode(NodeType::Softmax, "Softmax");
    nodes_.push_back(softmax);
    ImNodes::SetNodeGridSpacePos(softmax.id, ImVec2(2150.0f, 200.0f));

    // 15. Output
    MLNode output = CreateNode(NodeType::Output, "Output");
    output.parameters["classes"] = "10";
    nodes_.push_back(output);
    ImNodes::SetNodeGridSpacePos(output.id, ImVec2(2350.0f, 200.0f));

    // ========== LOSS & OPTIMIZER ==========

    // 16. One-Hot Encode labels
    MLNode onehot = CreateNode(NodeType::OneHotEncode, "One-Hot Labels");
    onehot.parameters["num_classes"] = "10";
    nodes_.push_back(onehot);
    ImNodes::SetNodeGridSpacePos(onehot.id, ImVec2(2150.0f, 450.0f));

    // 17. Cross Entropy Loss
    MLNode loss = CreateNode(NodeType::CrossEntropyLoss, "CrossEntropy Loss");
    nodes_.push_back(loss);
    ImNodes::SetNodeGridSpacePos(loss.id, ImVec2(2350.0f, 350.0f));

    // 18. AdamW Optimizer (commonly used for transformers)
    MLNode optimizer = CreateNode(NodeType::AdamW, "AdamW Optimizer");
    optimizer.parameters["learning_rate"] = "1e-4";
    optimizer.parameters["weight_decay"] = "0.01";
    nodes_.push_back(optimizer);
    ImNodes::SetNodeGridSpacePos(optimizer.id, ImVec2(2550.0f, 350.0f));

    // ========== CREATE CONNECTIONS ==========

    // Input flow: DatasetInput -> Embedding -> PositionalEncoding
    CreateLink(dataset_input.outputs[0].id, embedding.inputs[0].id,
               dataset_input.id, embedding.id);

    CreateLink(embedding.outputs[0].id, pos_enc.inputs[0].id,
               embedding.id, pos_enc.id);

    // Linear Attention: Q, K, V all come from positional encoding output
    // (Self-attention pattern)
    CreateLink(pos_enc.outputs[0].id, linear_attn.inputs[0].id,
               pos_enc.id, linear_attn.id);  // Query
    CreateLink(pos_enc.outputs[0].id, linear_attn.inputs[1].id,
               pos_enc.id, linear_attn.id);  // Key
    CreateLink(pos_enc.outputs[0].id, linear_attn.inputs[2].id,
               pos_enc.id, linear_attn.id);  // Value

    // Residual connection: Attention output + Position encoding -> Add
    CreateLink(linear_attn.outputs[0].id, residual1.inputs[0].id,
               linear_attn.id, residual1.id);
    CreateLink(pos_enc.outputs[0].id, residual1.inputs[1].id,
               pos_enc.id, residual1.id);

    // LayerNorm after residual
    CreateLink(residual1.outputs[0].id, layer_norm1.inputs[0].id,
               residual1.id, layer_norm1.id);

    // Feed-Forward Network
    CreateLink(layer_norm1.outputs[0].id, ffn1.inputs[0].id,
               layer_norm1.id, ffn1.id);

    CreateLink(ffn1.outputs[0].id, gelu.inputs[0].id,
               ffn1.id, gelu.id);

    CreateLink(gelu.outputs[0].id, ffn2.inputs[0].id,
               gelu.id, ffn2.id);

    CreateLink(ffn2.outputs[0].id, dropout.inputs[0].id,
               ffn2.id, dropout.id);

    // Second residual: Dropout output + LayerNorm1 output -> Add
    CreateLink(dropout.outputs[0].id, residual2.inputs[0].id,
               dropout.id, residual2.id);
    CreateLink(layer_norm1.outputs[0].id, residual2.inputs[1].id,
               layer_norm1.id, residual2.id);

    // LayerNorm after second residual
    CreateLink(residual2.outputs[0].id, layer_norm2.inputs[0].id,
               residual2.id, layer_norm2.id);

    // Output head
    CreateLink(layer_norm2.outputs[0].id, output_dense.inputs[0].id,
               layer_norm2.id, output_dense.id);

    CreateLink(output_dense.outputs[0].id, softmax.inputs[0].id,
               output_dense.id, softmax.id);

    CreateLink(softmax.outputs[0].id, output.inputs[0].id,
               softmax.id, output.id);

    // Labels flow
    CreateLink(dataset_input.outputs[1].id, onehot.inputs[0].id,
               dataset_input.id, onehot.id);

    // Loss connections
    CreateLink(output.outputs[0].id, loss.inputs[0].id,
               output.id, loss.id);  // Predictions
    CreateLink(onehot.outputs[0].id, loss.inputs[1].id,
               onehot.id, loss.id);  // Targets

    // Optimizer
    CreateLink(loss.outputs[0].id, optimizer.inputs[0].id,
               loss.id, optimizer.id);

    spdlog::info("Created Linear Attention Transformer showcase with {} nodes and {} connections",
                 nodes_.size(), links_.size());
    spdlog::info("Architecture: Embedding -> PositionalEncoding -> LinearAttention(O(n)) -> LayerNorm -> FFN -> Output");
}

NodeEditor::~NodeEditor() {
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

    if (ImGui::Begin("Node Editor", &show_window_)) {
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

        // Render group backgrounds before nodes so they appear behind
        RenderGroups();

        RenderNodes();

        // Handle mouse wheel zoom (skip if mouse is over minimap)
        if (ImGui::IsWindowHovered() && !mouse_in_minimap_bounds) {
            float wheel = ImGui::GetIO().MouseWheel;
            if (wheel != 0.0f) {
                // Zoom by adjusting the panning offset
                ImVec2 panning = ImNodes::EditorContextGetPanning();
                float zoom_delta = wheel * 50.0f;  // Zoom speed

                ImNodes::EditorContextResetPanning(ImVec2(
                    panning.x + zoom_delta,
                    panning.y + zoom_delta
                ));
            }
        }

        // Handle right-click context menu (skip if mouse is over minimap)
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

            ImGui::OpenPopup("NodeContextMenu");
        }

        if (ImGui::BeginPopup("NodeContextMenu")) {
            ShowContextMenu();
            ImGui::EndPopup();
        }

        // Cache node positions while still inside BeginNodeEditor/EndNodeEditor scope
        // This is needed because GetNodeGridSpacePos only works inside this scope
        cached_node_positions_.clear();
        for (const auto& node : nodes_) {
            cached_node_positions_[node.id] = ImNodes::GetNodeGridSpacePos(node.id);
        }

        ImNodes::EndNodeEditor();

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
        pending_nodes_.clear();

        // Handle interactions AFTER EndNodeEditor() - this is when ImNodes processes them
        HandleInteractions();

        // Handle keyboard shortcuts (Ctrl+Z, Ctrl+C, etc.)
        HandleKeyboardShortcuts();

        // Show search bar if visible (Ctrl+F to toggle)
        ShowSearchBar();

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
}

void NodeEditor::ShowToolbar() {
    // File operations
    if (ImGui::Button("Save Graph")) {
        ShowSaveDialog();
    }
    ImGui::SameLine();

    if (ImGui::Button("Load Graph")) {
        ShowLoadDialog();
    }
    ImGui::SameLine();

    ImGui::Text("|");
    ImGui::SameLine();

    if (ImGui::Button("Add Dense Layer")) {
        // Use special sentinel position - will be replaced by FindEmptyPosition after EndNodeEditor
        context_menu_pos_ = ImVec2(-99999.0f, -99999.0f);
        AddNode(NodeType::Dense, "Dense Layer");
    }
    ImGui::SameLine();

    if (ImGui::Button("Add ReLU")) {
        // Use special sentinel position - will be replaced by FindEmptyPosition after EndNodeEditor
        context_menu_pos_ = ImVec2(-99999.0f, -99999.0f);
        AddNode(NodeType::ReLU, "ReLU");
    }
    ImGui::SameLine();

    // Delete selected nodes (Clear)
    if (ImGui::Button("Clear")) {
        DeleteSelected();
    }
    ImGui::SameLine();

    if (ImGui::Button("Clear All")) {
        ClearGraph();
    }
    ImGui::SameLine();

    ImGui::Text("|");
    ImGui::SameLine();

    ImGui::Text("Nodes: %zu | Links: %zu", nodes_.size(), links_.size());

    // Code generation controls on a new line for better visibility
    ImGui::Separator();
    ImGui::Text("Code Generation:");
    ImGui::SameLine();

    // Framework selection
    const char* frameworks[] = { "PyTorch", "TensorFlow", "Keras", "PyCyxWiz" };
    int current_framework = static_cast<int>(selected_framework_);
    ImGui::SetNextItemWidth(120.0f);
    if (ImGui::Combo("##Framework", &current_framework, frameworks, 4)) {
        selected_framework_ = static_cast<CodeFramework>(current_framework);
        spdlog::info("Code generation framework changed to: {}", frameworks[current_framework]);
    }
    ImGui::SameLine();

    if (ImGui::Button("Generate Code")) {
        GeneratePythonCode();
    }
    ImGui::SameLine();

    if (ImGui::Button("Export Code")) {
        ShowExportDialog();
    }

    // Training controls
    ImGui::SameLine();
    ImGui::Text("|");
    ImGui::SameLine();

    // Check training state from TrainingManager
    auto& training_mgr = cyxwiz::TrainingManager::Instance();
    bool training_active = training_mgr.IsTrainingActive();

    if (training_active) {
        // Show training progress and stop button
        auto metrics = training_mgr.GetCurrentMetrics();
        ImGui::TextColored(ImVec4(0.4f, 1.0f, 0.4f, 1.0f), "Training... Epoch %d/%d",
            metrics.current_epoch, metrics.total_epochs);
        ImGui::SameLine();
        if (ImGui::Button("Stop Training")) {
            training_mgr.StopTraining();
        }
    } else {
        // Train button - green when valid, disabled when invalid
        bool can_train = IsGraphValid() && train_callback_;
        if (!can_train) {
            ImGui::BeginDisabled();
        }

        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.2f, 0.6f, 0.2f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.3f, 0.7f, 0.3f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.1f, 0.5f, 0.1f, 1.0f));
        if (ImGui::Button("Train Model")) {
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
    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 4.0f);
    ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.12f, 0.12f, 0.14f, 0.95f));
    ImGui::PushStyleColor(ImGuiCol_TitleBg, ImVec4(0.20f, 0.20f, 0.22f, 1.0f));
    ImGui::PushStyleColor(ImGuiCol_TitleBgActive, ImVec4(0.25f, 0.25f, 0.28f, 1.0f));
    ImGui::PushStyleColor(ImGuiCol_Border, ImVec4(0.31f, 0.31f, 0.35f, 1.0f));

    ImGuiWindowFlags window_flags = ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoScrollbar |
                                    ImGuiWindowFlags_NoScrollWithMouse | ImGuiWindowFlags_NoCollapse |
                                    ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoDocking |
                                    ImGuiWindowFlags_NoMove;  // We handle movement manually

    
    if (ImGui::Begin("##MinimapWindow", &show_minimap_, window_flags)) {
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
            draw_list->AddLine(mm_from, mm_to, IM_COL32(150, 150, 150, 150), 1.0f);
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

        // Check if node is selected
        bool is_selected = (node.id == selected_node_id_);

        draw_list->AddRectFilled(mm_pos, ImVec2(mm_pos.x + mm_size.x, mm_pos.y + mm_size.y), fill_color, 2.0f);

        if (is_selected) {
            draw_list->AddRect(mm_pos, ImVec2(mm_pos.x + mm_size.x, mm_pos.y + mm_size.y), IM_COL32(255, 255, 100, 255), 2.0f, 0, 2.0f);
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

    // Draw semi-transparent viewport indicator
    draw_list->AddRectFilled(viewport_mm_min, viewport_mm_max, IM_COL32(100, 150, 255, 40));
    draw_list->AddRect(viewport_mm_min, viewport_mm_max, IM_COL32(100, 150, 255, 200), 0.0f, 0, 1.5f);

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

void NodeEditor::RenderNodes() {
    // Render all nodes
    for (const auto& node : nodes_) {
        // Set node color based on type
        ImNodes::PushColorStyle(ImNodesCol_TitleBar, GetNodeColor(node.type));
        ImNodes::PushColorStyle(ImNodesCol_TitleBarHovered, GetNodeColor(node.type));
        ImNodes::PushColorStyle(ImNodesCol_TitleBarSelected, GetNodeColor(node.type));

        ImNodes::BeginNode(node.id);

        // Node title bar
        ImNodes::BeginNodeTitleBar();
        ImGui::TextUnformatted(node.name.c_str());
        ImNodes::EndNodeTitleBar();

        // Input pins
        for (const auto& pin : node.inputs) {
            ImNodes::BeginInputAttribute(pin.id);
            ImGui::TextUnformatted(pin.name.c_str());
            ImNodes::EndInputAttribute();
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

            default:
                // For activation layers and other nodes without parameters, show nothing
                break;
        }
        ImGui::Spacing();

        // Output pins
        for (const auto& pin : node.outputs) {
            ImNodes::BeginOutputAttribute(pin.id);
            const float text_width = ImGui::CalcTextSize(pin.name.c_str()).x;
            ImGui::Indent(120.0f + ImGui::CalcTextSize(pin.name.c_str()).x - text_width);
            ImGui::TextUnformatted(pin.name.c_str());
            ImNodes::EndOutputAttribute();
        }

        ImNodes::EndNode();

        // Check if this node is hovered for documentation tooltip
        int hovered_node_id = -1;
        if (ImNodes::IsNodeHovered(&hovered_node_id) && hovered_node_id == node.id) {
            NodeDocumentationManager::Instance().RenderTooltip(node.type);
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

        // Pop color styles
        ImNodes::PopColorStyle();
        ImNodes::PopColorStyle();
        ImNodes::PopColorStyle();
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
            // Use link type colors when not training
            link_color = GetLinkColor(link.type);
            link_hovered = GetLinkHoverColor(link.type);
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

} // namespace gui
