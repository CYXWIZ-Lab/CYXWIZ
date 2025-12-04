#include "node_editor.h"
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

        // Apply any pending position AFTER the node has been created
        // (ImNodes needs the node to exist before SetNodeGridSpacePos works)
        auto pos_it = pending_positions_.find(node.id);
        if (pos_it != pending_positions_.end()) {
            ImNodes::SetNodeGridSpacePos(node.id, pos_it->second);
            pending_positions_.erase(pos_it);
        }

        // Pop color styles
        ImNodes::PopColorStyle();
        ImNodes::PopColorStyle();
        ImNodes::PopColorStyle();
    }

    // Render all links with training animation if active
    for (const auto& link : links_) {
        if (is_training_) {
            // Create pulsing amber/green effect during training
            // Pulse frequency: ~2 Hz (full cycle every 0.5 seconds)
            float pulse = (std::sin(training_animation_time_ * 12.0f + link.id * 0.5f) + 1.0f) * 0.5f;

            // Interpolate between amber (255, 191, 0) and green (0, 255, 100)
            float r = 255.0f * (1.0f - pulse) + 0.0f * pulse;
            float g = 191.0f * (1.0f - pulse) + 255.0f * pulse;
            float b = 0.0f * (1.0f - pulse) + 100.0f * pulse;

            ImU32 link_color = IM_COL32(static_cast<int>(r), static_cast<int>(g), static_cast<int>(b), 255);
            ImU32 link_hovered = IM_COL32(static_cast<int>(r), static_cast<int>(g), static_cast<int>(b), 200);
            ImU32 link_selected = IM_COL32(255, 255, 255, 255);

            ImNodes::PushColorStyle(ImNodesCol_Link, link_color);
            ImNodes::PushColorStyle(ImNodesCol_LinkHovered, link_hovered);
            ImNodes::PushColorStyle(ImNodesCol_LinkSelected, link_selected);

            ImNodes::Link(link.id, link.from_pin, link.to_pin);

            ImNodes::PopColorStyle();
            ImNodes::PopColorStyle();
            ImNodes::PopColorStyle();
        } else {
            ImNodes::Link(link.id, link.from_pin, link.to_pin);
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

void NodeEditor::ShowContextMenu() {
    ImGui::Text("Add Node:");
    ImGui::Separator();

    // ===== LAYERS =====
    if (ImGui::BeginMenu("Layers")) {
        // Dense/Linear
        if (ImGui::BeginMenu("Dense / Linear")) {
            if (ImGui::MenuItem("Dense (64 units)")) {
                AddNode(NodeType::Dense, "Dense (64)");
                ImGui::CloseCurrentPopup();
            }
            if (ImGui::MenuItem("Dense (128 units)")) {
                AddNode(NodeType::Dense, "Dense (128)");
                ImGui::CloseCurrentPopup();
            }
            if (ImGui::MenuItem("Dense (256 units)")) {
                AddNode(NodeType::Dense, "Dense (256)");
                ImGui::CloseCurrentPopup();
            }
            if (ImGui::MenuItem("Dense (512 units)")) {
                AddNode(NodeType::Dense, "Dense (512)");
                ImGui::CloseCurrentPopup();
            }
            ImGui::EndMenu();
        }

        // Convolutional
        if (ImGui::BeginMenu("Convolutional")) {
            if (ImGui::MenuItem("Conv1D")) {
                AddNode(NodeType::Conv1D, "Conv1D");
                ImGui::CloseCurrentPopup();
            }
            if (ImGui::MenuItem("Conv2D")) {
                AddNode(NodeType::Conv2D, "Conv2D");
                ImGui::CloseCurrentPopup();
            }
            if (ImGui::MenuItem("Conv3D")) {
                AddNode(NodeType::Conv3D, "Conv3D");
                ImGui::CloseCurrentPopup();
            }
            if (ImGui::MenuItem("DepthwiseConv2D")) {
                AddNode(NodeType::DepthwiseConv2D, "DepthwiseConv2D");
                ImGui::CloseCurrentPopup();
            }
            ImGui::EndMenu();
        }

        // Pooling
        if (ImGui::BeginMenu("Pooling")) {
            if (ImGui::MenuItem("MaxPool2D")) {
                AddNode(NodeType::MaxPool2D, "MaxPool2D");
                ImGui::CloseCurrentPopup();
            }
            if (ImGui::MenuItem("AvgPool2D")) {
                AddNode(NodeType::AvgPool2D, "AvgPool2D");
                ImGui::CloseCurrentPopup();
            }
            if (ImGui::MenuItem("GlobalMaxPool")) {
                AddNode(NodeType::GlobalMaxPool, "GlobalMaxPool");
                ImGui::CloseCurrentPopup();
            }
            if (ImGui::MenuItem("GlobalAvgPool")) {
                AddNode(NodeType::GlobalAvgPool, "GlobalAvgPool");
                ImGui::CloseCurrentPopup();
            }
            if (ImGui::MenuItem("AdaptiveAvgPool")) {
                AddNode(NodeType::AdaptiveAvgPool, "AdaptiveAvgPool");
                ImGui::CloseCurrentPopup();
            }
            ImGui::EndMenu();
        }

        // Normalization
        if (ImGui::BeginMenu("Normalization")) {
            if (ImGui::MenuItem("BatchNorm")) {
                AddNode(NodeType::BatchNorm, "BatchNorm");
                ImGui::CloseCurrentPopup();
            }
            if (ImGui::MenuItem("LayerNorm")) {
                AddNode(NodeType::LayerNorm, "LayerNorm");
                ImGui::CloseCurrentPopup();
            }
            if (ImGui::MenuItem("GroupNorm")) {
                AddNode(NodeType::GroupNorm, "GroupNorm");
                ImGui::CloseCurrentPopup();
            }
            if (ImGui::MenuItem("InstanceNorm")) {
                AddNode(NodeType::InstanceNorm, "InstanceNorm");
                ImGui::CloseCurrentPopup();
            }
            ImGui::EndMenu();
        }

        // Regularization
        if (ImGui::BeginMenu("Regularization")) {
            if (ImGui::MenuItem("Dropout (0.5)")) {
                AddNode(NodeType::Dropout, "Dropout (0.5)");
                ImGui::CloseCurrentPopup();
            }
            if (ImGui::MenuItem("Dropout (0.3)")) {
                AddNode(NodeType::Dropout, "Dropout (0.3)");
                ImGui::CloseCurrentPopup();
            }
            if (ImGui::MenuItem("Dropout (0.2)")) {
                AddNode(NodeType::Dropout, "Dropout (0.2)");
                ImGui::CloseCurrentPopup();
            }
            ImGui::EndMenu();
        }

        if (ImGui::MenuItem("Flatten")) {
            AddNode(NodeType::Flatten, "Flatten");
            ImGui::CloseCurrentPopup();
        }

        ImGui::EndMenu();
    }

    // ===== ACTIVATIONS =====
    if (ImGui::BeginMenu("Activations")) {
        if (ImGui::MenuItem("ReLU")) {
            AddNode(NodeType::ReLU, "ReLU");
            ImGui::CloseCurrentPopup();
        }
        if (ImGui::MenuItem("LeakyReLU")) {
            AddNode(NodeType::LeakyReLU, "LeakyReLU");
            ImGui::CloseCurrentPopup();
        }
        if (ImGui::MenuItem("PReLU")) {
            AddNode(NodeType::PReLU, "PReLU");
            ImGui::CloseCurrentPopup();
        }
        if (ImGui::MenuItem("ELU")) {
            AddNode(NodeType::ELU, "ELU");
            ImGui::CloseCurrentPopup();
        }
        if (ImGui::MenuItem("SELU")) {
            AddNode(NodeType::SELU, "SELU");
            ImGui::CloseCurrentPopup();
        }
        if (ImGui::MenuItem("GELU")) {
            AddNode(NodeType::GELU, "GELU");
            ImGui::CloseCurrentPopup();
        }
        if (ImGui::MenuItem("Swish / SiLU")) {
            AddNode(NodeType::Swish, "Swish");
            ImGui::CloseCurrentPopup();
        }
        if (ImGui::MenuItem("Mish")) {
            AddNode(NodeType::Mish, "Mish");
            ImGui::CloseCurrentPopup();
        }
        ImGui::Separator();
        if (ImGui::MenuItem("Sigmoid")) {
            AddNode(NodeType::Sigmoid, "Sigmoid");
            ImGui::CloseCurrentPopup();
        }
        if (ImGui::MenuItem("Tanh")) {
            AddNode(NodeType::Tanh, "Tanh");
            ImGui::CloseCurrentPopup();
        }
        if (ImGui::MenuItem("Softmax")) {
            AddNode(NodeType::Softmax, "Softmax");
            ImGui::CloseCurrentPopup();
        }
        ImGui::EndMenu();
    }

    // ===== RECURRENT & ATTENTION =====
    if (ImGui::BeginMenu("Recurrent & Attention")) {
        // Recurrent
        if (ImGui::BeginMenu("Recurrent")) {
            if (ImGui::MenuItem("RNN")) {
                AddNode(NodeType::RNN, "RNN");
                ImGui::CloseCurrentPopup();
            }
            if (ImGui::MenuItem("LSTM")) {
                AddNode(NodeType::LSTM, "LSTM");
                ImGui::CloseCurrentPopup();
            }
            if (ImGui::MenuItem("GRU")) {
                AddNode(NodeType::GRU, "GRU");
                ImGui::CloseCurrentPopup();
            }
            if (ImGui::MenuItem("Bidirectional")) {
                AddNode(NodeType::Bidirectional, "Bidirectional");
                ImGui::CloseCurrentPopup();
            }
            if (ImGui::MenuItem("TimeDistributed")) {
                AddNode(NodeType::TimeDistributed, "TimeDistributed");
                ImGui::CloseCurrentPopup();
            }
            ImGui::EndMenu();
        }

        if (ImGui::MenuItem("Embedding")) {
            AddNode(NodeType::Embedding, "Embedding");
            ImGui::CloseCurrentPopup();
        }

        ImGui::Separator();

        // Attention & Transformer
        if (ImGui::BeginMenu("Attention")) {
            if (ImGui::MenuItem("MultiHeadAttention")) {
                AddNode(NodeType::MultiHeadAttention, "MultiHeadAttention");
                ImGui::CloseCurrentPopup();
            }
            if (ImGui::MenuItem("SelfAttention")) {
                AddNode(NodeType::SelfAttention, "SelfAttention");
                ImGui::CloseCurrentPopup();
            }
            if (ImGui::MenuItem("CrossAttention")) {
                AddNode(NodeType::CrossAttention, "CrossAttention");
                ImGui::CloseCurrentPopup();
            }
            if (ImGui::MenuItem("LinearAttention")) {
                AddNode(NodeType::LinearAttention, "LinearAttention");
                ImGui::CloseCurrentPopup();
            }
            ImGui::EndMenu();
        }

        if (ImGui::BeginMenu("Transformer")) {
            if (ImGui::MenuItem("TransformerEncoder")) {
                AddNode(NodeType::TransformerEncoder, "TransformerEncoder");
                ImGui::CloseCurrentPopup();
            }
            if (ImGui::MenuItem("TransformerDecoder")) {
                AddNode(NodeType::TransformerDecoder, "TransformerDecoder");
                ImGui::CloseCurrentPopup();
            }
            if (ImGui::MenuItem("PositionalEncoding")) {
                AddNode(NodeType::PositionalEncoding, "PositionalEncoding");
                ImGui::CloseCurrentPopup();
            }
            ImGui::EndMenu();
        }

        ImGui::EndMenu();
    }

    // ===== SHAPE OPERATIONS =====
    if (ImGui::BeginMenu("Shape Operations")) {
        if (ImGui::MenuItem("Reshape")) {
            AddNode(NodeType::Reshape, "Reshape");
            ImGui::CloseCurrentPopup();
        }
        if (ImGui::MenuItem("Permute")) {
            AddNode(NodeType::Permute, "Permute");
            ImGui::CloseCurrentPopup();
        }
        if (ImGui::MenuItem("Squeeze")) {
            AddNode(NodeType::Squeeze, "Squeeze");
            ImGui::CloseCurrentPopup();
        }
        if (ImGui::MenuItem("Unsqueeze")) {
            AddNode(NodeType::Unsqueeze, "Unsqueeze");
            ImGui::CloseCurrentPopup();
        }
        if (ImGui::MenuItem("View")) {
            AddNode(NodeType::View, "View");
            ImGui::CloseCurrentPopup();
        }
        if (ImGui::MenuItem("Split")) {
            AddNode(NodeType::Split, "Split");
            ImGui::CloseCurrentPopup();
        }
        ImGui::EndMenu();
    }

    // ===== MERGE OPERATIONS =====
    if (ImGui::BeginMenu("Merge Operations")) {
        if (ImGui::MenuItem("Concatenate")) {
            AddNode(NodeType::Concatenate, "Concatenate");
            ImGui::CloseCurrentPopup();
        }
        if (ImGui::MenuItem("Add")) {
            AddNode(NodeType::Add, "Add");
            ImGui::CloseCurrentPopup();
        }
        if (ImGui::MenuItem("Multiply")) {
            AddNode(NodeType::Multiply, "Multiply");
            ImGui::CloseCurrentPopup();
        }
        if (ImGui::MenuItem("Average")) {
            AddNode(NodeType::Average, "Average");
            ImGui::CloseCurrentPopup();
        }
        ImGui::EndMenu();
    }

    ImGui::Separator();

    // ===== DATA PIPELINE =====
    if (ImGui::BeginMenu("Data Pipeline")) {
        if (ImGui::MenuItem("DatasetInput")) {
            AddNode(NodeType::DatasetInput, "DatasetInput");
            ImGui::CloseCurrentPopup();
        }
        if (ImGui::MenuItem("DataLoader")) {
            AddNode(NodeType::DataLoader, "DataLoader");
            ImGui::CloseCurrentPopup();
        }
        if (ImGui::MenuItem("Augmentation")) {
            AddNode(NodeType::Augmentation, "Augmentation");
            ImGui::CloseCurrentPopup();
        }
        if (ImGui::MenuItem("DataSplit")) {
            AddNode(NodeType::DataSplit, "DataSplit");
            ImGui::CloseCurrentPopup();
        }
        ImGui::Separator();
        if (ImGui::MenuItem("TensorReshape")) {
            AddNode(NodeType::TensorReshape, "TensorReshape");
            ImGui::CloseCurrentPopup();
        }
        if (ImGui::MenuItem("Normalize")) {
            AddNode(NodeType::Normalize, "Normalize");
            ImGui::CloseCurrentPopup();
        }
        if (ImGui::MenuItem("OneHotEncode")) {
            AddNode(NodeType::OneHotEncode, "OneHotEncode");
            ImGui::CloseCurrentPopup();
        }
        ImGui::EndMenu();
    }

    ImGui::Separator();

    // ===== LOSS FUNCTIONS =====
    if (ImGui::BeginMenu("Loss Functions")) {
        if (ImGui::BeginMenu("Regression")) {
            if (ImGui::MenuItem("MSE Loss")) {
                AddNode(NodeType::MSELoss, "MSE Loss");
                ImGui::CloseCurrentPopup();
            }
            if (ImGui::MenuItem("L1 Loss (MAE)")) {
                AddNode(NodeType::L1Loss, "L1 Loss");
                ImGui::CloseCurrentPopup();
            }
            if (ImGui::MenuItem("Smooth L1 Loss")) {
                AddNode(NodeType::SmoothL1Loss, "SmoothL1 Loss");
                ImGui::CloseCurrentPopup();
            }
            if (ImGui::MenuItem("Huber Loss")) {
                AddNode(NodeType::HuberLoss, "Huber Loss");
                ImGui::CloseCurrentPopup();
            }
            ImGui::EndMenu();
        }
        if (ImGui::BeginMenu("Classification")) {
            if (ImGui::MenuItem("CrossEntropy Loss")) {
                AddNode(NodeType::CrossEntropyLoss, "CrossEntropy Loss");
                ImGui::CloseCurrentPopup();
            }
            if (ImGui::MenuItem("BCE Loss")) {
                AddNode(NodeType::BCELoss, "BCE Loss");
                ImGui::CloseCurrentPopup();
            }
            if (ImGui::MenuItem("BCE with Logits")) {
                AddNode(NodeType::BCEWithLogits, "BCEWithLogits");
                ImGui::CloseCurrentPopup();
            }
            if (ImGui::MenuItem("NLL Loss")) {
                AddNode(NodeType::NLLLoss, "NLL Loss");
                ImGui::CloseCurrentPopup();
            }
            ImGui::EndMenu();
        }
        ImGui::EndMenu();
    }

    // ===== OPTIMIZERS =====
    if (ImGui::BeginMenu("Optimizers")) {
        if (ImGui::MenuItem("SGD")) {
            AddNode(NodeType::SGD, "SGD");
            ImGui::CloseCurrentPopup();
        }
        if (ImGui::MenuItem("Adam")) {
            AddNode(NodeType::Adam, "Adam");
            ImGui::CloseCurrentPopup();
        }
        if (ImGui::MenuItem("AdamW")) {
            AddNode(NodeType::AdamW, "AdamW");
            ImGui::CloseCurrentPopup();
        }
        if (ImGui::MenuItem("RMSprop")) {
            AddNode(NodeType::RMSprop, "RMSprop");
            ImGui::CloseCurrentPopup();
        }
        if (ImGui::MenuItem("Adagrad")) {
            AddNode(NodeType::Adagrad, "Adagrad");
            ImGui::CloseCurrentPopup();
        }
        if (ImGui::MenuItem("NAdam")) {
            AddNode(NodeType::NAdam, "NAdam");
            ImGui::CloseCurrentPopup();
        }
        ImGui::EndMenu();
    }

    // ===== LEARNING RATE SCHEDULERS =====
    if (ImGui::BeginMenu("LR Schedulers")) {
        if (ImGui::MenuItem("StepLR")) {
            AddNode(NodeType::StepLR, "StepLR");
            ImGui::CloseCurrentPopup();
        }
        if (ImGui::MenuItem("CosineAnnealing")) {
            AddNode(NodeType::CosineAnnealing, "CosineAnnealing");
            ImGui::CloseCurrentPopup();
        }
        if (ImGui::MenuItem("ReduceOnPlateau")) {
            AddNode(NodeType::ReduceOnPlateau, "ReduceOnPlateau");
            ImGui::CloseCurrentPopup();
        }
        if (ImGui::MenuItem("ExponentialLR")) {
            AddNode(NodeType::ExponentialLR, "ExponentialLR");
            ImGui::CloseCurrentPopup();
        }
        if (ImGui::MenuItem("WarmupScheduler")) {
            AddNode(NodeType::WarmupScheduler, "WarmupScheduler");
            ImGui::CloseCurrentPopup();
        }
        ImGui::EndMenu();
    }

    // ===== REGULARIZATION NODES =====
    if (ImGui::BeginMenu("Regularization")) {
        if (ImGui::MenuItem("L1 Regularization")) {
            AddNode(NodeType::L1Regularization, "L1 Regularization");
            ImGui::CloseCurrentPopup();
        }
        if (ImGui::MenuItem("L2 Regularization")) {
            AddNode(NodeType::L2Regularization, "L2 Regularization");
            ImGui::CloseCurrentPopup();
        }
        if (ImGui::MenuItem("ElasticNet")) {
            AddNode(NodeType::ElasticNet, "ElasticNet");
            ImGui::CloseCurrentPopup();
        }
        ImGui::EndMenu();
    }

    // ===== UTILITY NODES =====
    if (ImGui::BeginMenu("Utility")) {
        if (ImGui::MenuItem("Lambda")) {
            AddNode(NodeType::Lambda, "Lambda");
            ImGui::CloseCurrentPopup();
        }
        if (ImGui::MenuItem("Identity")) {
            AddNode(NodeType::Identity, "Identity");
            ImGui::CloseCurrentPopup();
        }
        if (ImGui::MenuItem("Constant")) {
            AddNode(NodeType::Constant, "Constant");
            ImGui::CloseCurrentPopup();
        }
        if (ImGui::MenuItem("Parameter")) {
            AddNode(NodeType::Parameter, "Parameter");
            ImGui::CloseCurrentPopup();
        }
        ImGui::EndMenu();
    }

    ImGui::Separator();

    // ===== OUTPUT =====
    if (ImGui::MenuItem("Output")) {
        AddNode(NodeType::Output, "Output");
        ImGui::CloseCurrentPopup();
    }

    // ===== SELECTION-BASED OPTIONS =====
    if (!selected_node_ids_.empty()) {
        ImGui::Separator();
        ImGui::TextDisabled("Selection (%zu nodes)", selected_node_ids_.size());

        auto& pm = cyxwiz::ProjectManager::Instance();
        bool has_project = pm.HasActiveProject();

        if (!has_project) {
            ImGui::BeginDisabled();
        }

        if (ImGui::MenuItem(ICON_FA_BOOKMARK " Save as Pattern...")) {
            // Open save pattern dialog
            show_save_pattern_dialog_ = true;
            std::memset(save_pattern_name_, 0, sizeof(save_pattern_name_));
            std::memset(save_pattern_description_, 0, sizeof(save_pattern_description_));
            ImGui::CloseCurrentPopup();
        }

        if (!has_project) {
            ImGui::EndDisabled();
            if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) {
                ImGui::SetTooltip("Create or open a project first to save patterns");
            }
        }
    }

    ImGui::Separator();

}

void NodeEditor::AddNode(NodeType type, const std::string& name) {
    // Queue the node for deferred addition (after ImNodes::EndNodeEditor())
    pending_nodes_.push_back({type, name, context_menu_pos_});
    spdlog::info("Queued node for addition: type={}, name={} at position x={} y={}",
                 static_cast<int>(type), name, context_menu_pos_.x, context_menu_pos_.y);
}

MLNode NodeEditor::CreateNode(NodeType type, const std::string& name) {
    MLNode node;
    node.id = next_node_id_++;
    node.type = type;
    node.name = name;

    // Create pins based on node type
    switch (type) {
        case NodeType::Dense: {
            // Dense layer has input and output
            NodePin input_pin;
            input_pin.id = next_pin_id_++;
            input_pin.type = PinType::Tensor;
            input_pin.name = "Input";
            input_pin.is_input = true;
            node.inputs.push_back(input_pin);

            NodePin output_pin;
            output_pin.id = next_pin_id_++;
            output_pin.type = PinType::Tensor;
            output_pin.name = "Output";
            output_pin.is_input = false;
            node.outputs.push_back(output_pin);

            // Extract units from name (e.g., "Dense (128)")
            size_t start = name.find('(');
            size_t end = name.find(')');
            if (start != std::string::npos && end != std::string::npos) {
                node.parameters["units"] = name.substr(start + 1, end - start - 1);
            } else {
                node.parameters["units"] = "128";
            }
            break;
        }

        case NodeType::ReLU:
        case NodeType::Sigmoid:
        case NodeType::Tanh:
        case NodeType::Softmax:
        case NodeType::LeakyReLU:
        case NodeType::PReLU:
        case NodeType::ELU:
        case NodeType::SELU:
        case NodeType::GELU:
        case NodeType::Swish:
        case NodeType::Mish: {
            // Activation functions
            NodePin input_pin;
            input_pin.id = next_pin_id_++;
            input_pin.type = PinType::Tensor;
            input_pin.name = "Input";
            input_pin.is_input = true;
            node.inputs.push_back(input_pin);

            NodePin output_pin;
            output_pin.id = next_pin_id_++;
            output_pin.type = PinType::Tensor;
            output_pin.name = "Output";
            output_pin.is_input = false;
            node.outputs.push_back(output_pin);

            // PReLU and LeakyReLU have a negative slope parameter
            if (node.type == NodeType::LeakyReLU) {
                node.parameters["negative_slope"] = "0.01";
            } else if (node.type == NodeType::PReLU) {
                node.parameters["num_parameters"] = "1";
                node.parameters["init"] = "0.25";
            } else if (node.type == NodeType::ELU) {
                node.parameters["alpha"] = "1.0";
            }
            break;
        }

        case NodeType::Output: {
            // Output node - final layer that produces predictions
            // Input: From previous layer
            NodePin input_pin;
            input_pin.id = next_pin_id_++;
            input_pin.type = PinType::Tensor;
            input_pin.name = "Input";
            input_pin.is_input = true;
            node.inputs.push_back(input_pin);

            // Output: Predictions (goes to Loss function)
            NodePin output_pin;
            output_pin.id = next_pin_id_++;
            output_pin.type = PinType::Tensor;
            output_pin.name = "Predictions";
            output_pin.is_input = false;
            node.outputs.push_back(output_pin);

            node.parameters["classes"] = "10";
            break;
        }

        case NodeType::Conv1D:
        case NodeType::Conv2D:
        case NodeType::Conv3D:
        case NodeType::DepthwiseConv2D: {
            // Convolutional layers
            NodePin input_pin;
            input_pin.id = next_pin_id_++;
            input_pin.type = PinType::Tensor;
            input_pin.name = "Input";
            input_pin.is_input = true;
            node.inputs.push_back(input_pin);

            NodePin output_pin;
            output_pin.id = next_pin_id_++;
            output_pin.type = PinType::Tensor;
            output_pin.name = "Output";
            output_pin.is_input = false;
            node.outputs.push_back(output_pin);

            // Initialize default parameters
            node.parameters["filters"] = "32";
            node.parameters["kernel_size"] = "3";
            node.parameters["stride"] = "1";
            node.parameters["padding"] = "same";
            node.parameters["activation"] = "relu";
            if (node.type == NodeType::DepthwiseConv2D) {
                node.parameters["depth_multiplier"] = "1";
            }
            break;
        }

        case NodeType::MaxPool2D:
        case NodeType::AvgPool2D: {
            // Pooling layers with size parameters
            NodePin input_pin;
            input_pin.id = next_pin_id_++;
            input_pin.type = PinType::Tensor;
            input_pin.name = "Input";
            input_pin.is_input = true;
            node.inputs.push_back(input_pin);

            NodePin output_pin;
            output_pin.id = next_pin_id_++;
            output_pin.type = PinType::Tensor;
            output_pin.name = "Output";
            output_pin.is_input = false;
            node.outputs.push_back(output_pin);

            // Initialize default parameters
            node.parameters["pool_size"] = "2";
            node.parameters["stride"] = "2";
            break;
        }

        case NodeType::GlobalMaxPool:
        case NodeType::GlobalAvgPool:
        case NodeType::AdaptiveAvgPool: {
            // Global pooling layers
            NodePin input_pin;
            input_pin.id = next_pin_id_++;
            input_pin.type = PinType::Tensor;
            input_pin.name = "Input";
            input_pin.is_input = true;
            node.inputs.push_back(input_pin);

            NodePin output_pin;
            output_pin.id = next_pin_id_++;
            output_pin.type = PinType::Tensor;
            output_pin.name = "Output";
            output_pin.is_input = false;
            node.outputs.push_back(output_pin);

            // AdaptiveAvgPool has output size parameter
            if (node.type == NodeType::AdaptiveAvgPool) {
                node.parameters["output_size"] = "1";
            }
            break;
        }

        case NodeType::Flatten: {
            // Flatten layer
            NodePin input_pin;
            input_pin.id = next_pin_id_++;
            input_pin.type = PinType::Tensor;
            input_pin.name = "Input";
            input_pin.is_input = true;
            node.inputs.push_back(input_pin);

            NodePin output_pin;
            output_pin.id = next_pin_id_++;
            output_pin.type = PinType::Tensor;
            output_pin.name = "Output";
            output_pin.is_input = false;
            node.outputs.push_back(output_pin);
            break;
        }

        case NodeType::Dropout: {
            // Dropout layer
            NodePin input_pin;
            input_pin.id = next_pin_id_++;
            input_pin.type = PinType::Tensor;
            input_pin.name = "Input";
            input_pin.is_input = true;
            node.inputs.push_back(input_pin);

            NodePin output_pin;
            output_pin.id = next_pin_id_++;
            output_pin.type = PinType::Tensor;
            output_pin.name = "Output";
            output_pin.is_input = false;
            node.outputs.push_back(output_pin);

            // Initialize default parameters
            node.parameters["rate"] = "0.5";
            break;
        }

        case NodeType::BatchNorm:
        case NodeType::LayerNorm:
        case NodeType::GroupNorm:
        case NodeType::InstanceNorm: {
            // Normalization layers
            NodePin input_pin;
            input_pin.id = next_pin_id_++;
            input_pin.type = PinType::Tensor;
            input_pin.name = "Input";
            input_pin.is_input = true;
            node.inputs.push_back(input_pin);

            NodePin output_pin;
            output_pin.id = next_pin_id_++;
            output_pin.type = PinType::Tensor;
            output_pin.name = "Output";
            output_pin.is_input = false;
            node.outputs.push_back(output_pin);

            // Initialize parameters based on norm type
            node.parameters["epsilon"] = "1e-5";
            if (node.type == NodeType::BatchNorm) {
                node.parameters["momentum"] = "0.1";
            } else if (node.type == NodeType::LayerNorm) {
                node.parameters["normalized_shape"] = "256";
            } else if (node.type == NodeType::GroupNorm) {
                node.parameters["num_groups"] = "32";
                node.parameters["num_channels"] = "256";
            } else if (node.type == NodeType::InstanceNorm) {
                node.parameters["num_features"] = "64";
            }
            break;
        }

        // ========== Data Pipeline Nodes ==========

        case NodeType::DatasetInput: {
            // DatasetInput node - loads from DataRegistry
            // No input pins (this is a source node)

            // Output: Data tensor
            NodePin data_pin;
            data_pin.id = next_pin_id_++;
            data_pin.type = PinType::Tensor;
            data_pin.name = "Data";
            data_pin.is_input = false;
            node.outputs.push_back(data_pin);

            // Output: Labels tensor
            NodePin labels_pin;
            labels_pin.id = next_pin_id_++;
            labels_pin.type = PinType::Labels;
            labels_pin.name = "Labels";
            labels_pin.is_input = false;
            node.outputs.push_back(labels_pin);

            // Note: Shape is metadata (displayed in properties panel), not a data flow output.
            // In ML frameworks, shape is intrinsic to tensors (accessed via tensor.shape).

            // Parameters
            node.parameters["dataset_name"] = "";  // Name in DataRegistry
            node.parameters["split"] = "train";    // train, val, test
            break;
        }

        case NodeType::DataLoader: {
            // DataLoader node - batch iterator
            // Input: Dataset reference
            NodePin dataset_pin;
            dataset_pin.id = next_pin_id_++;
            dataset_pin.type = PinType::Dataset;
            dataset_pin.name = "Dataset";
            dataset_pin.is_input = true;
            node.inputs.push_back(dataset_pin);

            // Output: Batched data
            NodePin batch_pin;
            batch_pin.id = next_pin_id_++;
            batch_pin.type = PinType::Tensor;
            batch_pin.name = "Batch";
            batch_pin.is_input = false;
            node.outputs.push_back(batch_pin);

            // Output: Batched labels
            NodePin labels_pin;
            labels_pin.id = next_pin_id_++;
            labels_pin.type = PinType::Labels;
            labels_pin.name = "Labels";
            labels_pin.is_input = false;
            node.outputs.push_back(labels_pin);

            // Parameters
            node.parameters["batch_size"] = "32";
            node.parameters["shuffle"] = "true";
            node.parameters["drop_last"] = "false";
            node.parameters["num_workers"] = "4";
            break;
        }

        case NodeType::Augmentation: {
            // Augmentation node - transform pipeline
            // Input: Data tensor
            NodePin input_pin;
            input_pin.id = next_pin_id_++;
            input_pin.type = PinType::Tensor;
            input_pin.name = "Input";
            input_pin.is_input = true;
            node.inputs.push_back(input_pin);

            // Output: Augmented data
            NodePin output_pin;
            output_pin.id = next_pin_id_++;
            output_pin.type = PinType::Tensor;
            output_pin.name = "Output";
            output_pin.is_input = false;
            node.outputs.push_back(output_pin);

            // Parameters (transform pipeline)
            node.parameters["transforms"] = "RandomFlip,Normalize";
            node.parameters["flip_prob"] = "0.5";
            node.parameters["normalize_mean"] = "0.0";
            node.parameters["normalize_std"] = "1.0";
            break;
        }

        case NodeType::DataSplit: {
            // DataSplit node - train/val/test splitter
            // Input: Data tensor
            NodePin data_in;
            data_in.id = next_pin_id_++;
            data_in.type = PinType::Tensor;
            data_in.name = "Data";
            data_in.is_input = true;
            node.inputs.push_back(data_in);

            // Input: Labels tensor
            NodePin labels_in;
            labels_in.id = next_pin_id_++;
            labels_in.type = PinType::Labels;
            labels_in.name = "Labels";
            labels_in.is_input = true;
            node.inputs.push_back(labels_in);

            // Output: Train Data
            NodePin train_data;
            train_data.id = next_pin_id_++;
            train_data.type = PinType::Tensor;
            train_data.name = "Train Data";
            train_data.is_input = false;
            node.outputs.push_back(train_data);

            // Output: Train Labels
            NodePin train_labels;
            train_labels.id = next_pin_id_++;
            train_labels.type = PinType::Labels;
            train_labels.name = "Train Labels";
            train_labels.is_input = false;
            node.outputs.push_back(train_labels);

            // Output: Val Data
            NodePin val_data;
            val_data.id = next_pin_id_++;
            val_data.type = PinType::Tensor;
            val_data.name = "Val Data";
            val_data.is_input = false;
            node.outputs.push_back(val_data);

            // Output: Val Labels
            NodePin val_labels;
            val_labels.id = next_pin_id_++;
            val_labels.type = PinType::Labels;
            val_labels.name = "Val Labels";
            val_labels.is_input = false;
            node.outputs.push_back(val_labels);

            // Output: Test Data
            NodePin test_data;
            test_data.id = next_pin_id_++;
            test_data.type = PinType::Tensor;
            test_data.name = "Test Data";
            test_data.is_input = false;
            node.outputs.push_back(test_data);

            // Output: Test Labels
            NodePin test_labels;
            test_labels.id = next_pin_id_++;
            test_labels.type = PinType::Labels;
            test_labels.name = "Test Labels";
            test_labels.is_input = false;
            node.outputs.push_back(test_labels);

            // Parameters
            node.parameters["train_ratio"] = "0.8";
            node.parameters["val_ratio"] = "0.1";
            node.parameters["test_ratio"] = "0.1";
            node.parameters["stratified"] = "true";
            node.parameters["seed"] = "42";
            break;
        }

        case NodeType::TensorReshape: {
            // TensorReshape node
            NodePin input_pin;
            input_pin.id = next_pin_id_++;
            input_pin.type = PinType::Tensor;
            input_pin.name = "Input";
            input_pin.is_input = true;
            node.inputs.push_back(input_pin);

            NodePin output_pin;
            output_pin.id = next_pin_id_++;
            output_pin.type = PinType::Tensor;
            output_pin.name = "Output";
            output_pin.is_input = false;
            node.outputs.push_back(output_pin);

            node.parameters["shape"] = "-1,28,28,1";
            break;
        }

        case NodeType::Normalize: {
            // Normalize node
            NodePin input_pin;
            input_pin.id = next_pin_id_++;
            input_pin.type = PinType::Tensor;
            input_pin.name = "Input";
            input_pin.is_input = true;
            node.inputs.push_back(input_pin);

            NodePin output_pin;
            output_pin.id = next_pin_id_++;
            output_pin.type = PinType::Tensor;
            output_pin.name = "Output";
            output_pin.is_input = false;
            node.outputs.push_back(output_pin);

            node.parameters["mean"] = "0.0";
            node.parameters["std"] = "1.0";
            break;
        }

        case NodeType::OneHotEncode: {
            // OneHotEncode node
            NodePin input_pin;
            input_pin.id = next_pin_id_++;
            input_pin.type = PinType::Labels;
            input_pin.name = "Labels";
            input_pin.is_input = true;
            node.inputs.push_back(input_pin);

            NodePin output_pin;
            output_pin.id = next_pin_id_++;
            output_pin.type = PinType::Tensor;
            output_pin.name = "OneHot";
            output_pin.is_input = false;
            node.outputs.push_back(output_pin);

            node.parameters["num_classes"] = "10";
            break;
        }

        // ========== Loss Functions ==========

        case NodeType::MSELoss:
        case NodeType::CrossEntropyLoss: {
            // Loss function: takes predictions and targets, outputs loss value
            // Input 1: Predictions (from model output)
            NodePin pred_pin;
            pred_pin.id = next_pin_id_++;
            pred_pin.type = PinType::Tensor;
            pred_pin.name = "Predictions";
            pred_pin.is_input = true;
            node.inputs.push_back(pred_pin);

            // Input 2: Targets (ground truth labels)
            NodePin target_pin;
            target_pin.id = next_pin_id_++;
            target_pin.type = PinType::Tensor;
            target_pin.name = "Targets";
            target_pin.is_input = true;
            node.inputs.push_back(target_pin);

            // Output: Loss value
            NodePin loss_pin;
            loss_pin.id = next_pin_id_++;
            loss_pin.type = PinType::Loss;
            loss_pin.name = "Loss";
            loss_pin.is_input = false;
            node.outputs.push_back(loss_pin);

            // Parameters
            if (node.type == NodeType::CrossEntropyLoss) {
                node.parameters["reduction"] = "mean";  // mean, sum, none
            }
            break;
        }

        // ========== Optimizers ==========

        case NodeType::SGD:
        case NodeType::Adam:
        case NodeType::AdamW:
        case NodeType::RMSprop:
        case NodeType::Adagrad:
        case NodeType::NAdam: {
            // Optimizer: takes loss and updates model parameters
            NodePin loss_pin;
            loss_pin.id = next_pin_id_++;
            loss_pin.type = PinType::Loss;
            loss_pin.name = "Loss";
            loss_pin.is_input = true;
            node.inputs.push_back(loss_pin);

            NodePin state_pin;
            state_pin.id = next_pin_id_++;
            state_pin.type = PinType::Optimizer;
            state_pin.name = "State";
            state_pin.is_input = false;
            node.outputs.push_back(state_pin);

            // Parameters based on optimizer type
            node.parameters["learning_rate"] = "0.001";
            if (node.type == NodeType::SGD) {
                node.parameters["learning_rate"] = "0.01";
                node.parameters["momentum"] = "0.9";
                node.parameters["weight_decay"] = "0.0";
            } else if (node.type == NodeType::Adam || node.type == NodeType::NAdam) {
                node.parameters["beta1"] = "0.9";
                node.parameters["beta2"] = "0.999";
                node.parameters["epsilon"] = "1e-8";
            } else if (node.type == NodeType::AdamW) {
                node.parameters["beta1"] = "0.9";
                node.parameters["beta2"] = "0.999";
                node.parameters["weight_decay"] = "0.01";
            } else if (node.type == NodeType::RMSprop) {
                node.parameters["alpha"] = "0.99";
                node.parameters["epsilon"] = "1e-8";
                node.parameters["momentum"] = "0.0";
            } else if (node.type == NodeType::Adagrad) {
                node.parameters["lr_decay"] = "0.0";
                node.parameters["epsilon"] = "1e-10";
            }
            break;
        }

        // ========== Recurrent Layers ==========

        case NodeType::RNN:
        case NodeType::LSTM:
        case NodeType::GRU: {
            // Recurrent layers
            NodePin input_pin;
            input_pin.id = next_pin_id_++;
            input_pin.type = PinType::Tensor;
            input_pin.name = "Input";
            input_pin.is_input = true;
            node.inputs.push_back(input_pin);

            NodePin output_pin;
            output_pin.id = next_pin_id_++;
            output_pin.type = PinType::Tensor;
            output_pin.name = "Output";
            output_pin.is_input = false;
            node.outputs.push_back(output_pin);

            NodePin hidden_pin;
            hidden_pin.id = next_pin_id_++;
            hidden_pin.type = PinType::Tensor;
            hidden_pin.name = "Hidden";
            hidden_pin.is_input = false;
            node.outputs.push_back(hidden_pin);

            node.parameters["input_size"] = "256";
            node.parameters["hidden_size"] = "256";
            node.parameters["num_layers"] = "1";
            node.parameters["bidirectional"] = "false";
            node.parameters["dropout"] = "0.0";
            break;
        }

        case NodeType::Bidirectional:
        case NodeType::TimeDistributed: {
            // Wrapper layers
            NodePin input_pin;
            input_pin.id = next_pin_id_++;
            input_pin.type = PinType::Tensor;
            input_pin.name = "Input";
            input_pin.is_input = true;
            node.inputs.push_back(input_pin);

            NodePin output_pin;
            output_pin.id = next_pin_id_++;
            output_pin.type = PinType::Tensor;
            output_pin.name = "Output";
            output_pin.is_input = false;
            node.outputs.push_back(output_pin);

            if (node.type == NodeType::Bidirectional) {
                node.parameters["merge_mode"] = "concat";
            }
            break;
        }

        case NodeType::Embedding: {
            // Embedding layer
            NodePin input_pin;
            input_pin.id = next_pin_id_++;
            input_pin.type = PinType::Tensor;
            input_pin.name = "Indices";
            input_pin.is_input = true;
            node.inputs.push_back(input_pin);

            NodePin output_pin;
            output_pin.id = next_pin_id_++;
            output_pin.type = PinType::Tensor;
            output_pin.name = "Embeddings";
            output_pin.is_input = false;
            node.outputs.push_back(output_pin);

            node.parameters["num_embeddings"] = "10000";
            node.parameters["embedding_dim"] = "256";
            node.parameters["padding_idx"] = "-1";
            break;
        }

        // ========== Attention & Transformer ==========

        case NodeType::MultiHeadAttention:
        case NodeType::SelfAttention:
        case NodeType::CrossAttention: {
            // Attention layers
            NodePin query_pin;
            query_pin.id = next_pin_id_++;
            query_pin.type = PinType::Tensor;
            query_pin.name = "Query";
            query_pin.is_input = true;
            node.inputs.push_back(query_pin);

            NodePin key_pin;
            key_pin.id = next_pin_id_++;
            key_pin.type = PinType::Tensor;
            key_pin.name = "Key";
            key_pin.is_input = true;
            node.inputs.push_back(key_pin);

            NodePin value_pin;
            value_pin.id = next_pin_id_++;
            value_pin.type = PinType::Tensor;
            value_pin.name = "Value";
            value_pin.is_input = true;
            node.inputs.push_back(value_pin);

            NodePin output_pin;
            output_pin.id = next_pin_id_++;
            output_pin.type = PinType::Tensor;
            output_pin.name = "Output";
            output_pin.is_input = false;
            node.outputs.push_back(output_pin);

            node.parameters["embed_dim"] = "512";
            node.parameters["num_heads"] = "8";
            node.parameters["dropout"] = "0.0";
            break;
        }

        case NodeType::LinearAttention: {
            // Linear attention (O(n) complexity) - Performer/Linear Transformer style
            NodePin query_pin;
            query_pin.id = next_pin_id_++;
            query_pin.type = PinType::Tensor;
            query_pin.name = "Query";
            query_pin.is_input = true;
            node.inputs.push_back(query_pin);

            NodePin key_pin;
            key_pin.id = next_pin_id_++;
            key_pin.type = PinType::Tensor;
            key_pin.name = "Key";
            key_pin.is_input = true;
            node.inputs.push_back(key_pin);

            NodePin value_pin;
            value_pin.id = next_pin_id_++;
            value_pin.type = PinType::Tensor;
            value_pin.name = "Value";
            value_pin.is_input = true;
            node.inputs.push_back(value_pin);

            NodePin output_pin;
            output_pin.id = next_pin_id_++;
            output_pin.type = PinType::Tensor;
            output_pin.name = "Output";
            output_pin.is_input = false;
            node.outputs.push_back(output_pin);

            node.parameters["embed_dim"] = "512";
            node.parameters["num_heads"] = "8";
            node.parameters["feature_map"] = "elu";  // elu, relu, favor+
            node.parameters["eps"] = "1e-6";
            break;
        }

        case NodeType::TransformerEncoder:
        case NodeType::TransformerDecoder: {
            // Transformer blocks
            NodePin input_pin;
            input_pin.id = next_pin_id_++;
            input_pin.type = PinType::Tensor;
            input_pin.name = "Input";
            input_pin.is_input = true;
            node.inputs.push_back(input_pin);

            if (node.type == NodeType::TransformerDecoder) {
                NodePin memory_pin;
                memory_pin.id = next_pin_id_++;
                memory_pin.type = PinType::Tensor;
                memory_pin.name = "Memory";
                memory_pin.is_input = true;
                node.inputs.push_back(memory_pin);
            }

            NodePin output_pin;
            output_pin.id = next_pin_id_++;
            output_pin.type = PinType::Tensor;
            output_pin.name = "Output";
            output_pin.is_input = false;
            node.outputs.push_back(output_pin);

            node.parameters["d_model"] = "512";
            node.parameters["nhead"] = "8";
            node.parameters["num_layers"] = "6";
            node.parameters["dim_feedforward"] = "2048";
            node.parameters["dropout"] = "0.1";
            break;
        }

        case NodeType::PositionalEncoding: {
            NodePin input_pin;
            input_pin.id = next_pin_id_++;
            input_pin.type = PinType::Tensor;
            input_pin.name = "Input";
            input_pin.is_input = true;
            node.inputs.push_back(input_pin);

            NodePin output_pin;
            output_pin.id = next_pin_id_++;
            output_pin.type = PinType::Tensor;
            output_pin.name = "Output";
            output_pin.is_input = false;
            node.outputs.push_back(output_pin);

            node.parameters["d_model"] = "512";
            node.parameters["max_len"] = "5000";
            node.parameters["dropout"] = "0.1";
            break;
        }

        // ========== Shape Operations ==========

        case NodeType::Reshape:
        case NodeType::View: {
            NodePin input_pin;
            input_pin.id = next_pin_id_++;
            input_pin.type = PinType::Tensor;
            input_pin.name = "Input";
            input_pin.is_input = true;
            node.inputs.push_back(input_pin);

            NodePin output_pin;
            output_pin.id = next_pin_id_++;
            output_pin.type = PinType::Tensor;
            output_pin.name = "Output";
            output_pin.is_input = false;
            node.outputs.push_back(output_pin);

            node.parameters["shape"] = "-1,256";
            break;
        }

        case NodeType::Permute: {
            NodePin input_pin;
            input_pin.id = next_pin_id_++;
            input_pin.type = PinType::Tensor;
            input_pin.name = "Input";
            input_pin.is_input = true;
            node.inputs.push_back(input_pin);

            NodePin output_pin;
            output_pin.id = next_pin_id_++;
            output_pin.type = PinType::Tensor;
            output_pin.name = "Output";
            output_pin.is_input = false;
            node.outputs.push_back(output_pin);

            node.parameters["dims"] = "0,2,1";
            break;
        }

        case NodeType::Squeeze:
        case NodeType::Unsqueeze: {
            NodePin input_pin;
            input_pin.id = next_pin_id_++;
            input_pin.type = PinType::Tensor;
            input_pin.name = "Input";
            input_pin.is_input = true;
            node.inputs.push_back(input_pin);

            NodePin output_pin;
            output_pin.id = next_pin_id_++;
            output_pin.type = PinType::Tensor;
            output_pin.name = "Output";
            output_pin.is_input = false;
            node.outputs.push_back(output_pin);

            node.parameters["dim"] = "0";
            break;
        }

        case NodeType::Split: {
            NodePin input_pin;
            input_pin.id = next_pin_id_++;
            input_pin.type = PinType::Tensor;
            input_pin.name = "Input";
            input_pin.is_input = true;
            node.inputs.push_back(input_pin);

            // Multiple outputs for split
            NodePin output1;
            output1.id = next_pin_id_++;
            output1.type = PinType::Tensor;
            output1.name = "Output 1";
            output1.is_input = false;
            node.outputs.push_back(output1);

            NodePin output2;
            output2.id = next_pin_id_++;
            output2.type = PinType::Tensor;
            output2.name = "Output 2";
            output2.is_input = false;
            node.outputs.push_back(output2);

            node.parameters["split_size"] = "2";
            node.parameters["dim"] = "0";
            break;
        }

        // ========== Merge Operations ==========

        case NodeType::Concatenate:
        case NodeType::Add:
        case NodeType::Multiply:
        case NodeType::Average: {
            // Multi-input merge operations
            NodePin input1;
            input1.id = next_pin_id_++;
            input1.type = PinType::Tensor;
            input1.name = "Input 1";
            input1.is_input = true;
            node.inputs.push_back(input1);

            NodePin input2;
            input2.id = next_pin_id_++;
            input2.type = PinType::Tensor;
            input2.name = "Input 2";
            input2.is_input = true;
            node.inputs.push_back(input2);

            NodePin output_pin;
            output_pin.id = next_pin_id_++;
            output_pin.type = PinType::Tensor;
            output_pin.name = "Output";
            output_pin.is_input = false;
            node.outputs.push_back(output_pin);

            if (node.type == NodeType::Concatenate) {
                node.parameters["dim"] = "1";
            }
            break;
        }

        // ========== Additional Loss Functions ==========

        case NodeType::BCELoss:
        case NodeType::BCEWithLogits:
        case NodeType::L1Loss:
        case NodeType::SmoothL1Loss:
        case NodeType::HuberLoss:
        case NodeType::NLLLoss: {
            NodePin pred_pin;
            pred_pin.id = next_pin_id_++;
            pred_pin.type = PinType::Tensor;
            pred_pin.name = "Predictions";
            pred_pin.is_input = true;
            node.inputs.push_back(pred_pin);

            NodePin target_pin;
            target_pin.id = next_pin_id_++;
            target_pin.type = PinType::Tensor;
            target_pin.name = "Targets";
            target_pin.is_input = true;
            node.inputs.push_back(target_pin);

            NodePin loss_pin;
            loss_pin.id = next_pin_id_++;
            loss_pin.type = PinType::Loss;
            loss_pin.name = "Loss";
            loss_pin.is_input = false;
            node.outputs.push_back(loss_pin);

            node.parameters["reduction"] = "mean";
            if (node.type == NodeType::SmoothL1Loss || node.type == NodeType::HuberLoss) {
                node.parameters["beta"] = "1.0";
            }
            break;
        }

        // ========== Learning Rate Schedulers ==========

        case NodeType::StepLR:
        case NodeType::CosineAnnealing:
        case NodeType::ReduceOnPlateau:
        case NodeType::ExponentialLR:
        case NodeType::WarmupScheduler: {
            // Schedulers connect to optimizer
            NodePin optim_pin;
            optim_pin.id = next_pin_id_++;
            optim_pin.type = PinType::Optimizer;
            optim_pin.name = "Optimizer";
            optim_pin.is_input = true;
            node.inputs.push_back(optim_pin);

            NodePin output_pin;
            output_pin.id = next_pin_id_++;
            output_pin.type = PinType::Optimizer;
            output_pin.name = "Scheduled";
            output_pin.is_input = false;
            node.outputs.push_back(output_pin);

            if (node.type == NodeType::StepLR) {
                node.parameters["step_size"] = "10";
                node.parameters["gamma"] = "0.1";
            } else if (node.type == NodeType::CosineAnnealing) {
                node.parameters["T_max"] = "100";
                node.parameters["eta_min"] = "0.0";
            } else if (node.type == NodeType::ReduceOnPlateau) {
                node.parameters["mode"] = "min";
                node.parameters["factor"] = "0.1";
                node.parameters["patience"] = "10";
            } else if (node.type == NodeType::ExponentialLR) {
                node.parameters["gamma"] = "0.95";
            } else if (node.type == NodeType::WarmupScheduler) {
                node.parameters["warmup_steps"] = "1000";
                node.parameters["warmup_ratio"] = "0.1";
            }
            break;
        }

        // ========== Regularization ==========

        case NodeType::L1Regularization:
        case NodeType::L2Regularization:
        case NodeType::ElasticNet: {
            NodePin input_pin;
            input_pin.id = next_pin_id_++;
            input_pin.type = PinType::Parameters;
            input_pin.name = "Parameters";
            input_pin.is_input = true;
            node.inputs.push_back(input_pin);

            NodePin output_pin;
            output_pin.id = next_pin_id_++;
            output_pin.type = PinType::Loss;
            output_pin.name = "Penalty";
            output_pin.is_input = false;
            node.outputs.push_back(output_pin);

            node.parameters["lambda"] = "0.01";
            if (node.type == NodeType::ElasticNet) {
                node.parameters["l1_ratio"] = "0.5";
            }
            break;
        }

        // ========== Utility Nodes ==========

        case NodeType::Lambda: {
            NodePin input_pin;
            input_pin.id = next_pin_id_++;
            input_pin.type = PinType::Tensor;
            input_pin.name = "Input";
            input_pin.is_input = true;
            node.inputs.push_back(input_pin);

            NodePin output_pin;
            output_pin.id = next_pin_id_++;
            output_pin.type = PinType::Tensor;
            output_pin.name = "Output";
            output_pin.is_input = false;
            node.outputs.push_back(output_pin);

            node.parameters["function"] = "lambda x: x";
            break;
        }

        case NodeType::Identity: {
            NodePin input_pin;
            input_pin.id = next_pin_id_++;
            input_pin.type = PinType::Tensor;
            input_pin.name = "Input";
            input_pin.is_input = true;
            node.inputs.push_back(input_pin);

            NodePin output_pin;
            output_pin.id = next_pin_id_++;
            output_pin.type = PinType::Tensor;
            output_pin.name = "Output";
            output_pin.is_input = false;
            node.outputs.push_back(output_pin);
            break;
        }

        case NodeType::Constant: {
            NodePin output_pin;
            output_pin.id = next_pin_id_++;
            output_pin.type = PinType::Tensor;
            output_pin.name = "Value";
            output_pin.is_input = false;
            node.outputs.push_back(output_pin);

            node.parameters["value"] = "1.0";
            node.parameters["shape"] = "1";
            break;
        }

        case NodeType::Parameter: {
            NodePin output_pin;
            output_pin.id = next_pin_id_++;
            output_pin.type = PinType::Tensor;
            output_pin.name = "Parameter";
            output_pin.is_input = false;
            node.outputs.push_back(output_pin);

            node.parameters["shape"] = "256";
            node.parameters["init"] = "xavier";
            node.parameters["requires_grad"] = "true";
            break;
        }

        default:
            // Default: input and output pins
            NodePin input_pin;
            input_pin.id = next_pin_id_++;
            input_pin.type = PinType::Tensor;
            input_pin.name = "Input";
            input_pin.is_input = true;
            node.inputs.push_back(input_pin);

            NodePin output_pin;
            output_pin.id = next_pin_id_++;
            output_pin.type = PinType::Tensor;
            output_pin.name = "Output";
            output_pin.is_input = false;
            node.outputs.push_back(output_pin);
            break;
    }

    return node;
}

void NodeEditor::DeleteNode(int node_id) {
    // Delete node
    auto node_it = std::find_if(nodes_.begin(), nodes_.end(),
        [node_id](const MLNode& node) {
            return node.id == node_id;
        });

    if (node_it != nodes_.end()) {
        spdlog::info("Deleting node: {} (ID: {})", node_it->name, node_id);

        // Delete all links connected to this node
        links_.erase(
            std::remove_if(links_.begin(), links_.end(),
                [node_id](const NodeLink& link) {
                    return link.from_node == node_id || link.to_node == node_id;
                }),
            links_.end());

        nodes_.erase(node_it);
    }
}

void NodeEditor::ClearGraph() {
    SaveUndoState();
    nodes_.clear();
    links_.clear();
    next_node_id_ = 1;
    next_pin_id_ = 1;
    next_link_id_ = 1;

    // Reset selection state
    selected_node_id_ = -1;
    selected_node_ids_.clear();

    // Request a full ImNodes context reset - this fully clears ImNodes' internal state
    // which prevents crashes from stale node references
    pending_context_reset_ = true;

    // Clear any pending positions
    pending_positions_.clear();

    spdlog::info("Cleared node graph");
}

void NodeEditor::InsertPattern(const std::vector<MLNode>& nodes, const std::vector<NodeLink>& links) {
    if (nodes.empty()) {
        spdlog::warn("InsertPattern called with empty nodes list");
        return;
    }

    SaveUndoState();

    // Add all nodes from the pattern
    for (const auto& node : nodes) {
        nodes_.push_back(node);

        // Queue position for deferred setting (will be applied during render)
        if (node.has_initial_position) {
            pending_positions_[node.id] = ImVec2(node.initial_pos_x, node.initial_pos_y);
        }

        // Update next IDs to avoid collisions
        if (node.id >= next_node_id_) {
            next_node_id_ = node.id + 1;
        }
        for (const auto& pin : node.inputs) {
            if (pin.id >= next_pin_id_) {
                next_pin_id_ = pin.id + 1;
            }
        }
        for (const auto& pin : node.outputs) {
            if (pin.id >= next_pin_id_) {
                next_pin_id_ = pin.id + 1;
            }
        }
    }

    // Add all links from the pattern
    for (const auto& link : links) {
        links_.push_back(link);

        // Update next link ID
        if (link.id >= next_link_id_) {
            next_link_id_ = link.id + 1;
        }
    }

    spdlog::info("Inserted pattern with {} nodes and {} links (positions queued: {})",
                 nodes.size(), links.size(), pending_positions_.size());
}

// ===== Undo/Redo System =====

void NodeEditor::SaveUndoState() {
    // Create snapshot of current state
    GraphSnapshot snapshot;
    snapshot.nodes = nodes_;
    snapshot.links = links_;
    snapshot.next_node_id = next_node_id_;
    snapshot.next_pin_id = next_pin_id_;
    snapshot.next_link_id = next_link_id_;

    // Push to undo stack
    undo_stack_.push_back(snapshot);

    // Limit stack size
    if (undo_stack_.size() > MAX_UNDO_LEVELS) {
        undo_stack_.erase(undo_stack_.begin());
    }

    // Clear redo stack when new action is performed
    redo_stack_.clear();

    spdlog::debug("Saved undo state (stack size: {})", undo_stack_.size());
}

void NodeEditor::Undo() {
    if (!CanUndo()) {
        spdlog::debug("Nothing to undo");
        return;
    }

    // Save current state to redo stack
    GraphSnapshot current;
    current.nodes = nodes_;
    current.links = links_;
    current.next_node_id = next_node_id_;
    current.next_pin_id = next_pin_id_;
    current.next_link_id = next_link_id_;
    redo_stack_.push_back(current);

    // Restore previous state from undo stack
    GraphSnapshot previous = undo_stack_.back();
    undo_stack_.pop_back();

    nodes_ = previous.nodes;
    links_ = previous.links;
    next_node_id_ = previous.next_node_id;
    next_pin_id_ = previous.next_pin_id;
    next_link_id_ = previous.next_link_id;

    // Clear selection
    ImNodes::ClearNodeSelection();
    ImNodes::ClearLinkSelection();
    selected_node_id_ = -1;

    spdlog::info("Undo performed (undo stack: {}, redo stack: {})",
                 undo_stack_.size(), redo_stack_.size());
}

void NodeEditor::Redo() {
    if (!CanRedo()) {
        spdlog::debug("Nothing to redo");
        return;
    }

    // Save current state to undo stack
    GraphSnapshot current;
    current.nodes = nodes_;
    current.links = links_;
    current.next_node_id = next_node_id_;
    current.next_pin_id = next_pin_id_;
    current.next_link_id = next_link_id_;
    undo_stack_.push_back(current);

    // Restore next state from redo stack
    GraphSnapshot next = redo_stack_.back();
    redo_stack_.pop_back();

    nodes_ = next.nodes;
    links_ = next.links;
    next_node_id_ = next.next_node_id;
    next_pin_id_ = next.next_pin_id;
    next_link_id_ = next.next_link_id;

    // Clear selection
    ImNodes::ClearNodeSelection();
    ImNodes::ClearLinkSelection();
    selected_node_id_ = -1;

    spdlog::info("Redo performed (undo stack: {}, redo stack: {})",
                 undo_stack_.size(), redo_stack_.size());
}

// ===== Clipboard Support =====

void NodeEditor::SelectAll() {
    ImNodes::ClearNodeSelection();
    selected_node_ids_.clear();

    for (const auto& node : nodes_) {
        ImNodes::SelectNode(node.id);
        selected_node_ids_.push_back(node.id);
    }

    spdlog::info("Selected all {} nodes", nodes_.size());
}

void NodeEditor::ClearSelection() {
    ImNodes::ClearNodeSelection();
    ImNodes::ClearLinkSelection();
    selected_node_ids_.clear();
    selected_node_id_ = -1;

    if (properties_panel_) {
        properties_panel_->ClearSelection();
    }

    spdlog::debug("Cleared selection");
}

void NodeEditor::DeleteSelected() {
    // Delete selected nodes without copying to clipboard
    const int num_selected = ImNodes::NumSelectedNodes();
    if (num_selected == 0) {
        spdlog::debug("No nodes selected to delete");
        return;
    }

    SaveUndoState();

    std::vector<int> selected_ids(num_selected);
    ImNodes::GetSelectedNodes(selected_ids.data());

    for (int node_id : selected_ids) {
        DeleteNode(node_id);
    }

    ClearSelection();
    spdlog::info("Deleted {} selected nodes", num_selected);
}

ImVec2 NodeEditor::FindEmptyPosition() {
    // Find a position that doesn't overlap with existing nodes
    // Start at a reasonable default position and search for empty space

    const float NODE_WIDTH = 200.0f;
    const float NODE_HEIGHT = 120.0f;
    const float SPACING = 50.0f;

    // Get current view panning to place node in visible area
    ImVec2 panning = ImNodes::EditorContextGetPanning();

    // Start position - relative to current view
    float start_x = -panning.x + 100.0f;
    float start_y = -panning.y + 100.0f;

    // If no nodes exist, return a simple position
    if (nodes_.empty()) {
        return ImVec2(start_x, start_y);
    }

    // Collect all existing node positions
    std::vector<ImVec2> node_positions;
    for (const auto& node : nodes_) {
        auto it = cached_node_positions_.find(node.id);
        ImVec2 pos = (it != cached_node_positions_.end()) ? it->second : ImVec2(0,0);
        node_positions.push_back(pos);
    }

    // Search for empty position using grid search
    for (int row = 0; row < 20; ++row) {
        for (int col = 0; col < 20; ++col) {
            float test_x = start_x + col * (NODE_WIDTH + SPACING);
            float test_y = start_y + row * (NODE_HEIGHT + SPACING);

            bool overlaps = false;
            for (const auto& pos : node_positions) {
                // Check if rectangles overlap
                if (test_x < pos.x + NODE_WIDTH + SPACING &&
                    test_x + NODE_WIDTH + SPACING > pos.x &&
                    test_y < pos.y + NODE_HEIGHT + SPACING &&
                    test_y + NODE_HEIGHT + SPACING > pos.y) {
                    overlaps = true;
                    break;
                }
            }

            if (!overlaps) {
                return ImVec2(test_x, test_y);
            }
        }
    }

    // Fallback: place below the lowest node
    float max_y = 0.0f;
    for (const auto& pos : node_positions) {
        if (pos.y > max_y) {
            max_y = pos.y;
        }
    }
    return ImVec2(start_x, max_y + NODE_HEIGHT + SPACING);
}

void NodeEditor::CopySelection() {
    // Get selected nodes from ImNodes
    const int num_selected = ImNodes::NumSelectedNodes();
    if (num_selected == 0) {
        spdlog::debug("No nodes selected to copy");
        return;
    }

    std::vector<int> selected_ids(num_selected);
    ImNodes::GetSelectedNodes(selected_ids.data());

    // Build set for quick lookup
    std::set<int> selected_set(selected_ids.begin(), selected_ids.end());

    // Copy selected nodes
    clipboard_.nodes.clear();
    clipboard_.links.clear();

    for (int node_id : selected_ids) {
        auto it = std::find_if(nodes_.begin(), nodes_.end(),
            [node_id](const MLNode& node) { return node.id == node_id; });
        if (it != nodes_.end()) {
            clipboard_.nodes.push_back(*it);
        }
    }

    // Copy internal links (links between selected nodes only)
    for (const auto& link : links_) {
        if (selected_set.count(link.from_node) && selected_set.count(link.to_node)) {
            clipboard_.links.push_back(link);
        }
    }

    clipboard_.valid = true;
    spdlog::info("Copied {} nodes and {} internal links to clipboard",
                 clipboard_.nodes.size(), clipboard_.links.size());
}

void NodeEditor::CutSelection() {
    CopySelection();

    if (!clipboard_.valid) {
        return;
    }

    // Delete the selected nodes
    const int num_selected = ImNodes::NumSelectedNodes();
    if (num_selected > 0) {
        SaveUndoState();

        std::vector<int> selected_ids(num_selected);
        ImNodes::GetSelectedNodes(selected_ids.data());

        for (int node_id : selected_ids) {
            DeleteNode(node_id);
        }

        ClearSelection();
        spdlog::info("Cut {} nodes", num_selected);
    }
}

void NodeEditor::PasteClipboard() {
    if (!clipboard_.valid || clipboard_.nodes.empty()) {
        spdlog::debug("Nothing to paste");
        return;
    }

    SaveUndoState();

    // Map from old node IDs to new node IDs
    std::map<int, int> node_id_map;
    // Map from old pin IDs to new pin IDs
    std::map<int, int> pin_id_map;

    // Clear selection before pasting
    ImNodes::ClearNodeSelection();

    // Create new nodes with new IDs
    for (const auto& old_node : clipboard_.nodes) {
        MLNode new_node = old_node;
        int old_id = new_node.id;
        new_node.id = next_node_id_++;

        // Assign new pin IDs
        for (auto& pin : new_node.inputs) {
            int old_pin_id = pin.id;
            pin.id = next_pin_id_++;
            pin_id_map[old_pin_id] = pin.id;
        }
        for (auto& pin : new_node.outputs) {
            int old_pin_id = pin.id;
            pin.id = next_pin_id_++;
            pin_id_map[old_pin_id] = pin.id;
        }

        node_id_map[old_id] = new_node.id;
        nodes_.push_back(new_node);

        // Position the new node with offset
        ImVec2 old_pos = ImNodes::GetNodeGridSpacePos(old_id);
        // If we can't get old position (node doesn't exist), use a default
        ImVec2 new_pos(old_pos.x + paste_offset_.x, old_pos.y + paste_offset_.y);

        // For pasted nodes, we need to set position after they're added
        // ImNodes requires the node to exist first, so we'll set it in the next frame
        // For now, use a simple offset from screen center
        ImVec2 canvas_origin = ImNodes::GetNodeEditorSpacePos(0);
        new_pos = ImVec2(paste_offset_.x + nodes_.size() * 10.0f,
                         paste_offset_.y + nodes_.size() * 10.0f);

        ImNodes::SetNodeGridSpacePos(new_node.id, new_pos);
        ImNodes::SelectNode(new_node.id);
    }

    // Recreate internal links with new IDs
    for (const auto& old_link : clipboard_.links) {
        NodeLink new_link;
        new_link.id = next_link_id_++;
        new_link.from_node = node_id_map[old_link.from_node];
        new_link.to_node = node_id_map[old_link.to_node];
        new_link.from_pin = pin_id_map[old_link.from_pin];
        new_link.to_pin = pin_id_map[old_link.to_pin];

        links_.push_back(new_link);
    }

    // Increase paste offset for next paste
    paste_offset_.x += 50.0f;
    paste_offset_.y += 50.0f;

    // Reset paste offset if it gets too large
    if (paste_offset_.x > 300.0f) {
        paste_offset_ = ImVec2(50.0f, 50.0f);
    }

    spdlog::info("Pasted {} nodes and {} links",
                 clipboard_.nodes.size(), clipboard_.links.size());
}

void NodeEditor::DuplicateSelection() {
    CopySelection();
    PasteClipboard();
}

// ===== Keyboard Shortcuts =====

void NodeEditor::HandleKeyboardShortcuts() {
    // Only process shortcuts when node editor window is focused
    if (!ImGui::IsWindowFocused(ImGuiFocusedFlags_RootAndChildWindows)) {
        return;
    }

    const bool ctrl = ImGui::GetIO().KeyCtrl;
    const bool shift = ImGui::GetIO().KeyShift;

    // Ctrl+Z - Undo
    if (ctrl && !shift && ImGui::IsKeyPressed(ImGuiKey_Z)) {
        Undo();
    }

    // Ctrl+Y or Ctrl+Shift+Z - Redo
    if ((ctrl && ImGui::IsKeyPressed(ImGuiKey_Y)) ||
        (ctrl && shift && ImGui::IsKeyPressed(ImGuiKey_Z))) {
        Redo();
    }

    // Ctrl+C - Copy
    if (ctrl && ImGui::IsKeyPressed(ImGuiKey_C)) {
        CopySelection();
    }

    // Ctrl+X - Cut
    if (ctrl && ImGui::IsKeyPressed(ImGuiKey_X)) {
        CutSelection();
    }

    // Ctrl+V - Paste
    if (ctrl && ImGui::IsKeyPressed(ImGuiKey_V)) {
        PasteClipboard();
    }

    // Ctrl+D - Duplicate
    if (ctrl && ImGui::IsKeyPressed(ImGuiKey_D)) {
        DuplicateSelection();
    }

    // Ctrl+A - Select All
    if (ctrl && ImGui::IsKeyPressed(ImGuiKey_A)) {
        SelectAll();
    }

    // Escape - Clear selection or close context menu
    if (ImGui::IsKeyPressed(ImGuiKey_Escape)) {
        if (show_context_menu_) {
            show_context_menu_ = false;
        } else {
            ClearSelection();
        }
    }

    // M - Toggle minimap
    if (ImGui::IsKeyPressed(ImGuiKey_M) && !ctrl) {
        show_minimap_ = !show_minimap_;
        spdlog::info("Minimap {}", show_minimap_ ? "enabled" : "disabled");
    }

    // F - Frame selected or frame all
    if (ImGui::IsKeyPressed(ImGuiKey_F) && !ctrl) {
        if (ImNodes::NumSelectedNodes() > 0) {
            FrameSelected();
        } else {
            FrameAll();
        }
    }
}

void NodeEditor::FrameSelected() {
    // Get bounding box of selected nodes and center view on them
    const int num_selected = ImNodes::NumSelectedNodes();
    if (num_selected == 0) {
        return;
    }

    std::vector<int> selected_ids(num_selected);
    ImNodes::GetSelectedNodes(selected_ids.data());

    // Calculate bounding box
    float min_x = FLT_MAX, min_y = FLT_MAX;
    float max_x = -FLT_MAX, max_y = -FLT_MAX;

    for (int node_id : selected_ids) {
        auto it = cached_node_positions_.find(node_id);
        ImVec2 pos = (it != cached_node_positions_.end()) ? it->second : ImVec2(0,0);
        ImVec2 dims = ImNodes::GetNodeDimensions(node_id);

        min_x = std::min(min_x, pos.x);
        min_y = std::min(min_y, pos.y);
        max_x = std::max(max_x, pos.x + dims.x);
        max_y = std::max(max_y, pos.y + dims.y);
    }

    // Calculate center and pan to it
    ImVec2 center((min_x + max_x) / 2.0f, (min_y + max_y) / 2.0f);
    ImNodes::EditorContextResetPanning(ImVec2(-center.x + 400.0f, -center.y + 300.0f));

    spdlog::debug("Framed {} selected nodes", num_selected);
}

void NodeEditor::FrameAll() {
    if (nodes_.empty()) {
        return;
    }

    // Calculate bounding box of all nodes
    float min_x = FLT_MAX, min_y = FLT_MAX;
    float max_x = -FLT_MAX, max_y = -FLT_MAX;

    for (const auto& node : nodes_) {
        auto it = cached_node_positions_.find(node.id);
        ImVec2 pos = (it != cached_node_positions_.end()) ? it->second : ImVec2(0,0);
        ImVec2 dims = ImNodes::GetNodeDimensions(node.id);

        min_x = std::min(min_x, pos.x);
        min_y = std::min(min_y, pos.y);
        max_x = std::max(max_x, pos.x + dims.x);
        max_y = std::max(max_y, pos.y + dims.y);
    }

    // Calculate center and pan to it
    ImVec2 center((min_x + max_x) / 2.0f, (min_y + max_y) / 2.0f);
    ImNodes::EditorContextResetPanning(ImVec2(-center.x + 400.0f, -center.y + 300.0f));

    spdlog::debug("Framed all {} nodes", nodes_.size());
}

void NodeEditor::CreateLink(int from_pin, int to_pin, int from_node, int to_node) {
    NodeLink link;
    link.id = next_link_id_++;
    link.from_pin = from_pin;
    link.to_pin = to_pin;
    link.from_node = from_node;
    link.to_node = to_node;
    links_.push_back(link);
}

void NodeEditor::GeneratePythonCode() {
    // Validate graph before generating code
    std::string error_message;
    if (!ValidateGraph(error_message)) {
        spdlog::error("Graph validation failed: {}", error_message);
        // TODO: Show error dialog to user
        return;
    }

    GenerateCodeForFramework(selected_framework_);
}

void NodeEditor::GenerateCodeForFramework(CodeFramework framework) {
    spdlog::info("Generating code from node graph (async)...");

    if (nodes_.empty()) {
        spdlog::warn("No nodes in graph - cannot generate code");
        return;
    }

    // Get topologically sorted node order (do this synchronously for validation)
    std::vector<int> sorted_ids = TopologicalSort();
    if (sorted_ids.empty()) {
        spdlog::error("Failed to perform topological sort - graph may have cycles");
        return;
    }

    // Copy graph data for thread safety
    std::vector<MLNode> nodes_copy = nodes_;
    std::vector<NodeLink> links_copy = links_;
    size_t total_nodes = sorted_ids.size();

    // Determine framework name
    std::string framework_name;
    switch (framework) {
        case CodeFramework::PyTorch: framework_name = "PyTorch"; break;
        case CodeFramework::TensorFlow: framework_name = "TensorFlow"; break;
        case CodeFramework::Keras: framework_name = "Keras"; break;
        case CodeFramework::PyCyxWiz: framework_name = "PyCyxWiz"; break;
        default: framework_name = "Unknown"; break;
    }

    // Store result for completion callback
    auto result = std::make_shared<std::string>();
    auto fw_name = std::make_shared<std::string>(framework_name);

    // Capture script_editor_ for completion callback
    auto script_editor = script_editor_;

    // Run code generation async
    cyxwiz::AsyncTaskManager::Instance().RunAsync(
        "Generate " + framework_name + " Code",
        [this, framework, sorted_ids, nodes_copy, total_nodes, result, fw_name](cyxwiz::LambdaTask& task) {
            task.ReportProgress(0.0f, "Starting code generation...");

            std::string code;

            // Generate code based on selected framework
            task.ReportProgress(0.1f, "Generating " + *fw_name + " code...");

            switch (framework) {
                case CodeFramework::PyTorch:
                    code = GeneratePyTorchCode(sorted_ids);
                    break;
                case CodeFramework::TensorFlow:
                    code = GenerateTensorFlowCode(sorted_ids);
                    break;
                case CodeFramework::Keras:
                    code = GenerateKerasCode(sorted_ids);
                    break;
                case CodeFramework::PyCyxWiz:
                    code = GeneratePyCyxWizCode(sorted_ids);
                    break;
                default:
                    task.MarkFailed("Unknown framework selected");
                    return;
            }

            if (task.ShouldStop()) {
                task.MarkFailed("Code generation cancelled");
                return;
            }

            task.ReportProgress(0.9f, "Finalizing...");

            // Store result
            *result = std::move(code);

            task.ReportProgress(1.0f, "Complete!");
            spdlog::info("Generated {} code ({} lines)", *fw_name, std::count(result->begin(), result->end(), '\n'));
        },
        // Progress callback (optional - can be used for detailed UI updates)
        nullptr,
        // Completion callback - runs on main thread
        [script_editor, result, fw_name](bool success, const std::string& error) {
            if (success && script_editor) {
                script_editor->LoadGeneratedCode(*result, fw_name->c_str());
                script_editor->SetVisible(true);
                spdlog::info("Code sent to Script Editor panel");
            } else if (!success) {
                spdlog::error("Code generation failed: {}", error);
            } else {
                spdlog::warn("Script Editor panel not available");
            }
        }
    );
}

std::string NodeEditor::GeneratePyTorchCode(const std::vector<int>& sorted_ids) {
    std::string code;

    // Header
    code += "# Auto-generated PyTorch model from CyxWiz Node Editor\n";
    code += "# Generated at: " + std::string(__DATE__) + " " + std::string(__TIME__) + "\n\n";
    code += "import torch\n";
    code += "import torch.nn as nn\n";
    code += "import torch.nn.functional as F\n";
    code += "import torch.optim as optim\n\n";

    // Model class
    code += "class GeneratedModel(nn.Module):\n";
    code += "    def __init__(self):\n";
    code += "        super(GeneratedModel, self).__init__()\n";

    // Generate layer definitions
    int layer_idx = 0;
    for (int node_id : sorted_ids) {
        const MLNode* node = FindNodeById(node_id);
        if (!node) continue;

        // Skip DatasetInput and Output nodes in __init__ (they don't have layers)
        if (node->type == NodeType::DatasetInput || node->type == NodeType::Output) {
            continue;
        }

        std::string layer_code = NodeTypeToPythonLayer(*node);
        if (!layer_code.empty()) {
            code += "        self.layer" + std::to_string(layer_idx) + " = " + layer_code + "\n";
            layer_idx++;
        }
    }

    code += "\n";

    // Forward pass
    code += "    def forward(self, x):\n";
    layer_idx = 0;
    for (int node_id : sorted_ids) {
        const MLNode* node = FindNodeById(node_id);
        if (!node) continue;

        switch (node->type) {
            case NodeType::DatasetInput:
                code += "        # Dataset input layer (x is already the input)\n";
                break;

            case NodeType::Dense:
                code += "        x = self.layer" + std::to_string(layer_idx++) + "(x)\n";
                break;

            case NodeType::ReLU:
                code += "        x = F.relu(x)\n";
                break;

            case NodeType::Sigmoid:
                code += "        x = torch.sigmoid(x)\n";
                break;

            case NodeType::Tanh:
                code += "        x = torch.tanh(x)\n";
                break;

            case NodeType::Softmax:
                code += "        x = F.softmax(x, dim=1)\n";
                break;

            case NodeType::Dropout:
                code += "        x = F.dropout(x, p=0.5, training=self.training)\n";
                break;

            case NodeType::Flatten:
                code += "        x = torch.flatten(x, 1)\n";
                break;

            case NodeType::Output:
                code += "        # Output layer\n";
                break;

            default:
                break;
        }
    }
    code += "        return x\n\n";

    // Training code
    code += "# Training setup\n";
    code += "if __name__ == '__main__':\n";
    code += "    # Create model\n";
    code += "    model = GeneratedModel()\n";
    code += "    print(model)\n\n";

    code += "    # Loss and optimizer\n";
    code += "    criterion = nn.CrossEntropyLoss()\n";
    code += "    optimizer = optim.Adam(model.parameters(), lr=0.001)\n\n";

    code += "    # TODO: Add your training data here\n";
    code += "    # Example training loop:\n";
    code += "    # for epoch in range(num_epochs):\n";
    code += "    #     for batch_idx, (data, target) in enumerate(train_loader):\n";
    code += "    #         optimizer.zero_grad()\n";
    code += "    #         output = model(data)\n";
    code += "    #         loss = criterion(output, target)\n";
    code += "    #         loss.backward()\n";
    code += "    #         optimizer.step()\n";

    return code;
}

std::string NodeEditor::GenerateTensorFlowCode(const std::vector<int>& sorted_ids) {
    std::string code;

    // Header
    code += "# Auto-generated TensorFlow model from CyxWiz Node Editor\n";
    code += "# Generated at: " + std::string(__DATE__) + " " + std::string(__TIME__) + "\n\n";
    code += "import tensorflow as tf\n";
    code += "from tensorflow.keras import layers, models, optimizers\n\n";

    // Model class using tf.keras
    code += "class GeneratedModel(tf.keras.Model):\n";
    code += "    def __init__(self):\n";
    code += "        super(GeneratedModel, self).__init__()\n";

    // Generate layer definitions
    int layer_idx = 0;
    for (int node_id : sorted_ids) {
        const MLNode* node = FindNodeById(node_id);
        if (!node) continue;

        // Skip DatasetInput and Output nodes in __init__
        if (node->type == NodeType::DatasetInput || node->type == NodeType::Output) {
            continue;
        }

        std::string layer_code = NodeTypeToTensorFlowLayer(*node, layer_idx);
        if (!layer_code.empty()) {
            code += "        self.layer" + std::to_string(layer_idx) + " = " + layer_code + "\n";
            layer_idx++;
        }
    }

    code += "\n";

    // Call method (forward pass in TensorFlow)
    code += "    def call(self, x, training=False):\n";
    layer_idx = 0;
    for (int node_id : sorted_ids) {
        const MLNode* node = FindNodeById(node_id);
        if (!node) continue;

        switch (node->type) {
            case NodeType::DatasetInput:
                code += "        # Dataset input layer (x is already the input)\n";
                break;

            case NodeType::Dense:
                code += "        x = self.layer" + std::to_string(layer_idx++) + "(x)\n";
                break;

            case NodeType::ReLU:
                code += "        x = tf.nn.relu(x)\n";
                break;

            case NodeType::Sigmoid:
                code += "        x = tf.nn.sigmoid(x)\n";
                break;

            case NodeType::Tanh:
                code += "        x = tf.nn.tanh(x)\n";
                break;

            case NodeType::Softmax:
                code += "        x = tf.nn.softmax(x)\n";
                break;

            case NodeType::Dropout:
                code += "        x = tf.keras.layers.Dropout(0.5)(x, training=training)\n";
                break;

            case NodeType::Flatten:
                code += "        x = tf.keras.layers.Flatten()(x)\n";
                break;

            case NodeType::Output:
                code += "        # Output layer\n";
                break;

            default:
                break;
        }
    }
    code += "        return x\n\n";

    // Training code
    code += "# Training setup\n";
    code += "if __name__ == '__main__':\n";
    code += "    # Create model\n";
    code += "    model = GeneratedModel()\n";
    code += "    model.build(input_shape=(None, 784))  # Adjust input shape as needed\n";
    code += "    model.summary()\n\n";

    code += "    # Compile model\n";
    code += "    model.compile(\n";
    code += "        optimizer='adam',\n";
    code += "        loss='sparse_categorical_crossentropy',\n";
    code += "        metrics=['accuracy']\n";
    code += "    )\n\n";

    code += "    # TODO: Add your training data here\n";
    code += "    # Example training:\n";
    code += "    # model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.2)\n";

    return code;
}

std::string NodeEditor::GenerateKerasCode(const std::vector<int>& sorted_ids) {
    std::string code;

    // Header
    code += "# Auto-generated Keras model from CyxWiz Node Editor\n";
    code += "# Generated at: " + std::string(__DATE__) + " " + std::string(__TIME__) + "\n\n";
    code += "from tensorflow import keras\n";
    code += "from tensorflow.keras import layers\n\n";

    // Sequential model approach
    code += "# Build model using Sequential API\n";
    code += "model = keras.Sequential([\n";

    bool first_layer = true;
    for (int node_id : sorted_ids) {
        const MLNode* node = FindNodeById(node_id);
        if (!node) continue;

        // Skip DatasetInput node
        if (node->type == NodeType::DatasetInput) {
            continue;
        }

        std::string layer_code = NodeTypeToKerasLayer(*node);
        if (!layer_code.empty()) {
            if (!first_layer) {
                code += ",\n";
            }
            code += "    " + layer_code;
            first_layer = false;
        }
    }

    code += "\n])\n\n";

    // Model summary and compilation
    code += "# Model configuration\n";
    code += "model.build(input_shape=(None, 784))  # Adjust input shape as needed\n";
    code += "model.summary()\n\n";

    code += "# Compile model\n";
    code += "model.compile(\n";
    code += "    optimizer='adam',\n";
    code += "    loss='sparse_categorical_crossentropy',\n";
    code += "    metrics=['accuracy']\n";
    code += ")\n\n";

    code += "# TODO: Add your training data here\n";
    code += "# Example training:\n";
    code += "# history = model.fit(\n";
    code += "#     x_train, y_train,\n";
    code += "#     epochs=10,\n";
    code += "#     batch_size=32,\n";
    code += "#     validation_split=0.2\n";
    code += "# )\n";

    return code;
}

std::string NodeEditor::GeneratePyCyxWizCode(const std::vector<int>& sorted_ids) {
    std::string code;

    // Header
    code += "# Auto-generated PyCyxWiz model from CyxWiz Node Editor\n";
    code += "# Generated at: " + std::string(__DATE__) + " " + std::string(__TIME__) + "\n\n";
    code += "import pycyxwiz as cx\n";
    code += "import numpy as np\n\n";

    // Model class using pycyxwiz
    code += "class GeneratedModel:\n";
    code += "    def __init__(self):\n";

    // Generate layer definitions
    int layer_idx = 0;
    for (int node_id : sorted_ids) {
        const MLNode* node = FindNodeById(node_id);
        if (!node) continue;

        // Skip DatasetInput and Output nodes in __init__
        if (node->type == NodeType::DatasetInput || node->type == NodeType::Output) {
            continue;
        }

        std::string layer_code = NodeTypeToPyCyxWizLayer(*node);
        if (!layer_code.empty()) {
            code += "        self.layer" + std::to_string(layer_idx) + " = " + layer_code + "\n";
            layer_idx++;
        }
    }

    code += "\n";

    // Forward method
    code += "    def forward(self, x):\n";
    layer_idx = 0;
    for (int node_id : sorted_ids) {
        const MLNode* node = FindNodeById(node_id);
        if (!node) continue;

        switch (node->type) {
            case NodeType::DatasetInput:
                code += "        # Dataset input layer (x is already the input tensor)\n";
                break;

            case NodeType::Dense:
                code += "        x = self.layer" + std::to_string(layer_idx++) + ".forward(x)\n";
                break;

            case NodeType::ReLU:
                code += "        x = cx.relu(x)\n";
                break;

            case NodeType::Sigmoid:
                code += "        x = cx.sigmoid(x)\n";
                break;

            case NodeType::Tanh:
                code += "        x = cx.tanh(x)\n";
                break;

            case NodeType::Softmax:
                code += "        x = cx.softmax(x)\n";
                break;

            case NodeType::Dropout:
                code += "        x = cx.dropout(x, p=0.5)\n";
                break;

            case NodeType::Flatten:
                code += "        x = cx.flatten(x)\n";
                break;

            case NodeType::Output:
                code += "        # Output layer\n";
                break;

            default:
                break;
        }
    }
    code += "        return x\n\n";

    code += "    def train(self, x_train, y_train, epochs=10, learning_rate=0.001):\n";
    code += "        \"\"\"Training loop using CyxWiz backend\"\"\"\n";
    code += "        optimizer = cx.Adam(learning_rate=learning_rate)\n";
    code += "        loss_fn = cx.CrossEntropyLoss()\n\n";
    code += "        for epoch in range(epochs):\n";
    code += "            # Forward pass\n";
    code += "            predictions = self.forward(x_train)\n";
    code += "            loss = loss_fn(predictions, y_train)\n\n";
    code += "            # Backward pass\n";
    code += "            loss.backward()\n";
    code += "            optimizer.step()\n";
    code += "            optimizer.zero_grad()\n\n";
    code += "            print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}')\n\n";

    // Training setup
    code += "# Training setup\n";
    code += "if __name__ == '__main__':\n";
    code += "    # Initialize CyxWiz backend\n";
    code += "    cx.initialize()\n\n";
    code += "    # Select device (GPU if available)\n";
    code += "    device = cx.get_device(cx.DeviceType.CUDA if cx.cuda_available() else cx.DeviceType.CPU)\n";
    code += "    cx.set_device(device)\n";
    code += "    print(f'Using device: {device.name()}')\n\n";

    code += "    # Create model\n";
    code += "    model = GeneratedModel()\n\n";

    code += "    # TODO: Load your training data here\n";
    code += "    # x_train = cx.Tensor(your_data)\n";
    code += "    # y_train = cx.Tensor(your_labels)\n";
    code += "    # model.train(x_train, y_train, epochs=10)\n";

    return code;
}

std::string NodeEditor::NodeTypeToPythonLayer(const MLNode& node) {
    std::string code;

    switch (node.type) {
        case NodeType::Dense: {
            std::string units = "128";
            auto it = node.parameters.find("units");
            if (it != node.parameters.end()) {
                units = it->second;
            }
            // Note: input size needs to be determined from graph connections
            code = "nn.Linear(in_features=AUTO, out_features=" + units + ")";
            break;
        }

        case NodeType::Conv2D:
            code = "nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3)";
            break;

        case NodeType::MaxPool2D:
            code = "nn.MaxPool2d(kernel_size=2)";
            break;

        case NodeType::BatchNorm:
            code = "nn.BatchNorm2d(num_features=AUTO)";
            break;

        case NodeType::Dropout: {
            code = "nn.Dropout(p=0.5)";
            break;
        }

        case NodeType::LinearAttention: {
            std::string embed_dim = "512";
            std::string num_heads = "8";
            std::string feature_map = "elu";
            std::string eps = "1e-6";
            auto it = node.parameters.find("embed_dim");
            if (it != node.parameters.end()) embed_dim = it->second;
            it = node.parameters.find("num_heads");
            if (it != node.parameters.end()) num_heads = it->second;
            it = node.parameters.find("feature_map");
            if (it != node.parameters.end()) feature_map = it->second;
            it = node.parameters.find("eps");
            if (it != node.parameters.end()) eps = it->second;
            // Linear attention with O(n) complexity (Performer-style)
            // Requires: pip install performer-pytorch or custom implementation
            code = "LinearAttention(dim=" + embed_dim + ", heads=" + num_heads +
                   ", dim_head=" + embed_dim + "//" + num_heads +
                   ", feature_map='" + feature_map + "', eps=" + eps + ")";
            break;
        }

        case NodeType::MultiHeadAttention: {
            std::string embed_dim = "512";
            std::string num_heads = "8";
            auto it = node.parameters.find("embed_dim");
            if (it != node.parameters.end()) embed_dim = it->second;
            it = node.parameters.find("num_heads");
            if (it != node.parameters.end()) num_heads = it->second;
            code = "nn.MultiheadAttention(embed_dim=" + embed_dim + ", num_heads=" + num_heads + ")";
            break;
        }

        case NodeType::LayerNorm: {
            std::string normalized_shape = "512";
            auto it = node.parameters.find("normalized_shape");
            if (it != node.parameters.end()) normalized_shape = it->second;
            code = "nn.LayerNorm(" + normalized_shape + ")";
            break;
        }

        case NodeType::Embedding: {
            std::string num_embeddings = "10000";
            std::string embedding_dim = "512";
            auto it = node.parameters.find("num_embeddings");
            if (it != node.parameters.end()) num_embeddings = it->second;
            it = node.parameters.find("embedding_dim");
            if (it != node.parameters.end()) embedding_dim = it->second;
            code = "nn.Embedding(num_embeddings=" + num_embeddings + ", embedding_dim=" + embedding_dim + ")";
            break;
        }

        case NodeType::GELU:
            code = "nn.GELU()";
            break;

        case NodeType::ReLU:
            code = "nn.ReLU()";
            break;

        default:
            // Activation functions and others don't need layers in __init__
            code = "";
            break;
    }

    return code;
}

std::string NodeEditor::NodeTypeToTensorFlowLayer(const MLNode& node, int /*layer_idx*/) {
    std::string code;

    switch (node.type) {
        case NodeType::Dense: {
            std::string units = "128";
            auto it = node.parameters.find("units");
            if (it != node.parameters.end()) {
                units = it->second;
            }
            code = "layers.Dense(" + units + ")";
            break;
        }

        case NodeType::Conv2D:
            code = "layers.Conv2D(32, kernel_size=3)";
            break;

        case NodeType::MaxPool2D:
            code = "layers.MaxPool2D(pool_size=2)";
            break;

        case NodeType::BatchNorm:
            code = "layers.BatchNormalization()";
            break;

        case NodeType::Dropout:
            code = "layers.Dropout(0.5)";
            break;

        case NodeType::LinearAttention: {
            std::string embed_dim = "512";
            std::string num_heads = "8";
            auto it = node.parameters.find("embed_dim");
            if (it != node.parameters.end()) embed_dim = it->second;
            it = node.parameters.find("num_heads");
            if (it != node.parameters.end()) num_heads = it->second;
            // TensorFlow doesn't have native linear attention - use MultiHeadAttention or custom layer
            // Comment indicates O(n) linear attention should be used
            code = "# LinearAttention (O(n)) - requires tensorflow-addons or custom impl\n"
                   "        layers.MultiHeadAttention(key_dim=" + embed_dim + "//" + num_heads +
                   ", num_heads=" + num_heads + ")  # Replace with linear attention";
            break;
        }

        case NodeType::MultiHeadAttention: {
            std::string embed_dim = "512";
            std::string num_heads = "8";
            auto it = node.parameters.find("embed_dim");
            if (it != node.parameters.end()) embed_dim = it->second;
            it = node.parameters.find("num_heads");
            if (it != node.parameters.end()) num_heads = it->second;
            code = "layers.MultiHeadAttention(key_dim=" + embed_dim + "//" + num_heads +
                   ", num_heads=" + num_heads + ")";
            break;
        }

        case NodeType::LayerNorm: {
            std::string normalized_shape = "512";
            auto it = node.parameters.find("normalized_shape");
            if (it != node.parameters.end()) normalized_shape = it->second;
            code = "layers.LayerNormalization()";
            break;
        }

        case NodeType::Embedding: {
            std::string num_embeddings = "10000";
            std::string embedding_dim = "512";
            auto it = node.parameters.find("num_embeddings");
            if (it != node.parameters.end()) num_embeddings = it->second;
            it = node.parameters.find("embedding_dim");
            if (it != node.parameters.end()) embedding_dim = it->second;
            code = "layers.Embedding(input_dim=" + num_embeddings + ", output_dim=" + embedding_dim + ")";
            break;
        }

        case NodeType::GELU:
            code = "layers.Activation('gelu')";
            break;

        case NodeType::ReLU:
            code = "layers.ReLU()";
            break;

        default:
            // Activation functions and others don't need layers in __init__
            code = "";
            break;
    }

    return code;
}

std::string NodeEditor::NodeTypeToKerasLayer(const MLNode& node) {
    std::string code;

    switch (node.type) {
        case NodeType::Dense: {
            std::string units = "128";
            auto it = node.parameters.find("units");
            if (it != node.parameters.end()) {
                units = it->second;
            }
            code = "layers.Dense(" + units + ")";
            break;
        }

        case NodeType::Conv2D:
            code = "layers.Conv2D(32, kernel_size=3)";
            break;

        case NodeType::MaxPool2D:
            code = "layers.MaxPool2D(pool_size=2)";
            break;

        case NodeType::Flatten:
            code = "layers.Flatten()";
            break;

        case NodeType::Dropout:
            code = "layers.Dropout(0.5)";
            break;

        case NodeType::BatchNorm:
            code = "layers.BatchNormalization()";
            break;

        case NodeType::ReLU:
            code = "layers.ReLU()";
            break;

        case NodeType::Sigmoid:
            code = "layers.Activation('sigmoid')";
            break;

        case NodeType::Tanh:
            code = "layers.Activation('tanh')";
            break;

        case NodeType::Softmax:
            code = "layers.Activation('softmax')";
            break;

        case NodeType::Output: {
            std::string units = "10";
            auto it = node.parameters.find("units");
            if (it != node.parameters.end()) {
                units = it->second;
            }
            code = "layers.Dense(" + units + ", activation='softmax')";
            break;
        }

        case NodeType::LinearAttention: {
            std::string embed_dim = "512";
            std::string num_heads = "8";
            auto it = node.parameters.find("embed_dim");
            if (it != node.parameters.end()) embed_dim = it->second;
            it = node.parameters.find("num_heads");
            if (it != node.parameters.end()) num_heads = it->second;
            // Keras uses same MultiHeadAttention as TensorFlow
            code = "# LinearAttention (O(n)) - requires custom implementation\n"
                   "        layers.MultiHeadAttention(key_dim=" + embed_dim + "//" + num_heads +
                   ", num_heads=" + num_heads + ")  # Replace with linear attention";
            break;
        }

        case NodeType::MultiHeadAttention: {
            std::string embed_dim = "512";
            std::string num_heads = "8";
            auto it = node.parameters.find("embed_dim");
            if (it != node.parameters.end()) embed_dim = it->second;
            it = node.parameters.find("num_heads");
            if (it != node.parameters.end()) num_heads = it->second;
            code = "layers.MultiHeadAttention(key_dim=" + embed_dim + "//" + num_heads +
                   ", num_heads=" + num_heads + ")";
            break;
        }

        case NodeType::LayerNorm:
            code = "layers.LayerNormalization()";
            break;

        case NodeType::Embedding: {
            std::string num_embeddings = "10000";
            std::string embedding_dim = "512";
            auto it = node.parameters.find("num_embeddings");
            if (it != node.parameters.end()) num_embeddings = it->second;
            it = node.parameters.find("embedding_dim");
            if (it != node.parameters.end()) embedding_dim = it->second;
            code = "layers.Embedding(input_dim=" + num_embeddings + ", output_dim=" + embedding_dim + ")";
            break;
        }

        case NodeType::GELU:
            code = "layers.Activation('gelu')";
            break;

        default:
            code = "";
            break;
    }

    return code;
}

std::string NodeEditor::NodeTypeToPyCyxWizLayer(const MLNode& node) {
    std::string code;

    switch (node.type) {
        case NodeType::Dense: {
            std::string units = "128";
            auto it = node.parameters.find("units");
            if (it != node.parameters.end()) {
                units = it->second;
            }
            // Note: pycyxwiz Dense layer requires input size determination from graph
            code = "cx.Dense(in_features=AUTO, out_features=" + units + ")";
            break;
        }

        case NodeType::Conv2D:
            code = "cx.Conv2D(in_channels=1, out_channels=32, kernel_size=3)";
            break;

        case NodeType::MaxPool2D:
            code = "cx.MaxPool2D(kernel_size=2)";
            break;

        case NodeType::BatchNorm:
            code = "cx.BatchNorm()";
            break;

        case NodeType::Dropout:
            code = "cx.Dropout(p=0.5)";
            break;

        // ===== Attention & Transformer Layers =====
        case NodeType::LinearAttention: {
            std::string embed_dim = "512";
            std::string num_heads = "8";
            auto it = node.parameters.find("embed_dim");
            if (it != node.parameters.end()) embed_dim = it->second;
            it = node.parameters.find("num_heads");
            if (it != node.parameters.end()) num_heads = it->second;
            code = "cx.LinearAttention(dim=" + embed_dim + ", heads=" + num_heads + ")";
            break;
        }

        case NodeType::MultiHeadAttention: {
            std::string embed_dim = "512";
            std::string num_heads = "8";
            auto it = node.parameters.find("embed_dim");
            if (it != node.parameters.end()) embed_dim = it->second;
            it = node.parameters.find("num_heads");
            if (it != node.parameters.end()) num_heads = it->second;
            code = "cx.MultiHeadAttention(embed_dim=" + embed_dim + ", num_heads=" + num_heads + ")";
            break;
        }

        case NodeType::SelfAttention: {
            std::string embed_dim = "512";
            auto it = node.parameters.find("embed_dim");
            if (it != node.parameters.end()) embed_dim = it->second;
            code = "cx.SelfAttention(embed_dim=" + embed_dim + ")";
            break;
        }

        case NodeType::CrossAttention: {
            std::string embed_dim = "512";
            auto it = node.parameters.find("embed_dim");
            if (it != node.parameters.end()) embed_dim = it->second;
            code = "cx.CrossAttention(embed_dim=" + embed_dim + ")";
            break;
        }

        case NodeType::TransformerEncoder: {
            std::string embed_dim = "512";
            std::string num_heads = "8";
            std::string ff_dim = "2048";
            auto it = node.parameters.find("embed_dim");
            if (it != node.parameters.end()) embed_dim = it->second;
            it = node.parameters.find("num_heads");
            if (it != node.parameters.end()) num_heads = it->second;
            it = node.parameters.find("ff_dim");
            if (it != node.parameters.end()) ff_dim = it->second;
            code = "cx.TransformerEncoder(d_model=" + embed_dim + ", nhead=" + num_heads + ", dim_feedforward=" + ff_dim + ")";
            break;
        }

        case NodeType::TransformerDecoder: {
            std::string embed_dim = "512";
            std::string num_heads = "8";
            std::string ff_dim = "2048";
            auto it = node.parameters.find("embed_dim");
            if (it != node.parameters.end()) embed_dim = it->second;
            it = node.parameters.find("num_heads");
            if (it != node.parameters.end()) num_heads = it->second;
            it = node.parameters.find("ff_dim");
            if (it != node.parameters.end()) ff_dim = it->second;
            code = "cx.TransformerDecoder(d_model=" + embed_dim + ", nhead=" + num_heads + ", dim_feedforward=" + ff_dim + ")";
            break;
        }

        // ===== Normalization Layers =====
        case NodeType::LayerNorm: {
            std::string normalized_shape = "512";
            auto it = node.parameters.find("normalized_shape");
            if (it != node.parameters.end()) normalized_shape = it->second;
            code = "cx.LayerNorm(normalized_shape=" + normalized_shape + ")";
            break;
        }

        case NodeType::GroupNorm: {
            std::string num_groups = "32";
            std::string num_channels = "256";
            auto it = node.parameters.find("num_groups");
            if (it != node.parameters.end()) num_groups = it->second;
            it = node.parameters.find("num_channels");
            if (it != node.parameters.end()) num_channels = it->second;
            code = "cx.GroupNorm(num_groups=" + num_groups + ", num_channels=" + num_channels + ")";
            break;
        }

        case NodeType::InstanceNorm:
            code = "cx.InstanceNorm()";
            break;

        // ===== Embedding Layer =====
        case NodeType::Embedding: {
            std::string num_embeddings = "10000";
            std::string embedding_dim = "512";
            auto it = node.parameters.find("num_embeddings");
            if (it != node.parameters.end()) num_embeddings = it->second;
            it = node.parameters.find("embedding_dim");
            if (it != node.parameters.end()) embedding_dim = it->second;
            code = "cx.Embedding(num_embeddings=" + num_embeddings + ", embedding_dim=" + embedding_dim + ")";
            break;
        }

        case NodeType::PositionalEncoding: {
            std::string max_len = "5000";
            std::string d_model = "512";
            auto it = node.parameters.find("max_len");
            if (it != node.parameters.end()) max_len = it->second;
            it = node.parameters.find("d_model");
            if (it != node.parameters.end()) d_model = it->second;
            code = "cx.PositionalEncoding(d_model=" + d_model + ", max_len=" + max_len + ")";
            break;
        }

        // ===== Activation Functions =====
        case NodeType::ReLU:
            code = "cx.ReLU()";
            break;

        case NodeType::GELU:
            code = "cx.GELU()";
            break;

        case NodeType::LeakyReLU: {
            std::string negative_slope = "0.01";
            auto it = node.parameters.find("negative_slope");
            if (it != node.parameters.end()) negative_slope = it->second;
            code = "cx.LeakyReLU(negative_slope=" + negative_slope + ")";
            break;
        }

        case NodeType::Swish:
            code = "cx.Swish()";
            break;

        case NodeType::Mish:
            code = "cx.Mish()";
            break;

        case NodeType::Sigmoid:
            code = "cx.Sigmoid()";
            break;

        case NodeType::Tanh:
            code = "cx.Tanh()";
            break;

        case NodeType::Softmax: {
            std::string dim = "-1";
            auto it = node.parameters.find("dim");
            if (it != node.parameters.end()) dim = it->second;
            code = "cx.Softmax(dim=" + dim + ")";
            break;
        }

        // ===== Recurrent Layers =====
        case NodeType::LSTM: {
            std::string input_size = "512";
            std::string hidden_size = "256";
            std::string num_layers = "1";
            auto it = node.parameters.find("input_size");
            if (it != node.parameters.end()) input_size = it->second;
            it = node.parameters.find("hidden_size");
            if (it != node.parameters.end()) hidden_size = it->second;
            it = node.parameters.find("num_layers");
            if (it != node.parameters.end()) num_layers = it->second;
            code = "cx.LSTM(input_size=" + input_size + ", hidden_size=" + hidden_size + ", num_layers=" + num_layers + ")";
            break;
        }

        case NodeType::GRU: {
            std::string input_size = "512";
            std::string hidden_size = "256";
            std::string num_layers = "1";
            auto it = node.parameters.find("input_size");
            if (it != node.parameters.end()) input_size = it->second;
            it = node.parameters.find("hidden_size");
            if (it != node.parameters.end()) hidden_size = it->second;
            it = node.parameters.find("num_layers");
            if (it != node.parameters.end()) num_layers = it->second;
            code = "cx.GRU(input_size=" + input_size + ", hidden_size=" + hidden_size + ", num_layers=" + num_layers + ")";
            break;
        }

        // ===== Shape Operations =====
        case NodeType::Flatten:
            code = "cx.Flatten()";
            break;

        case NodeType::Reshape:
            code = "cx.Reshape(shape=AUTO)";  // Shape determined from graph
            break;

        // ===== Merge Operations =====
        case NodeType::Add:
            code = "cx.Add()";
            break;

        case NodeType::Concatenate: {
            std::string dim = "1";
            auto it = node.parameters.find("dim");
            if (it != node.parameters.end()) dim = it->second;
            code = "cx.Concatenate(dim=" + dim + ")";
            break;
        }

        default:
            // Other node types handled in forward pass or not yet implemented
            code = "";
            break;
    }

    return code;
}

std::vector<int> NodeEditor::TopologicalSort() {
    std::vector<int> result;
    std::map<int, int> in_degree;
    std::map<int, std::vector<int>> adj_list;

    // Initialize in-degree for all nodes
    for (const auto& node : nodes_) {
        in_degree[node.id] = 0;
        adj_list[node.id] = {};
    }

    // Build adjacency list and calculate in-degrees
    for (const auto& link : links_) {
        adj_list[link.from_node].push_back(link.to_node);
        in_degree[link.to_node]++;
    }

    // Find all nodes with in-degree 0 (starting nodes)
    std::vector<int> queue;
    for (const auto& [node_id, degree] : in_degree) {
        if (degree == 0) {
            queue.push_back(node_id);
        }
    }

    // Process nodes
    while (!queue.empty()) {
        int current = queue.front();
        queue.erase(queue.begin());
        result.push_back(current);

        // Reduce in-degree for neighbors
        for (int neighbor : adj_list[current]) {
            in_degree[neighbor]--;
            if (in_degree[neighbor] == 0) {
                queue.push_back(neighbor);
            }
        }
    }

    // Check if all nodes were processed (no cycles)
    if (result.size() != nodes_.size()) {
        spdlog::error("Graph has cycles - cannot generate code");
        return {};
    }

    return result;
}

const MLNode* NodeEditor::FindNodeById(int node_id) const {
    for (const auto& node : nodes_) {
        if (node.id == node_id) {
            return &node;
        }
    }
    return nullptr;
}

// ========== Color-Coding Implementation ==========
unsigned int NodeEditor::GetNodeColor(NodeType type) {
    switch (type) {
        // ===== Output - Blue =====
        case NodeType::Output:
            return IM_COL32(52, 152, 219, 255);

        // ===== Core Layers - Green =====
        case NodeType::Dense:
            return IM_COL32(39, 174, 96, 255);

        // ===== Convolutional Layers - Purple =====
        case NodeType::Conv1D:
        case NodeType::Conv2D:
        case NodeType::Conv3D:
        case NodeType::DepthwiseConv2D:
            return IM_COL32(142, 68, 173, 255);

        // ===== Pooling Layers - Light Purple =====
        case NodeType::MaxPool2D:
        case NodeType::AvgPool2D:
        case NodeType::GlobalMaxPool:
        case NodeType::GlobalAvgPool:
        case NodeType::AdaptiveAvgPool:
            return IM_COL32(155, 89, 182, 255);

        // ===== Normalization Layers - Pink/Coral =====
        case NodeType::BatchNorm:
        case NodeType::LayerNorm:
        case NodeType::GroupNorm:
        case NodeType::InstanceNorm:
            return IM_COL32(236, 112, 99, 255);

        // ===== Regularization - Red =====
        case NodeType::Dropout:
            return IM_COL32(231, 76, 60, 255);

        // ===== Utility Layers - Teal =====
        case NodeType::Flatten:
            return IM_COL32(22, 160, 133, 255);

        // ===== Recurrent Layers - Indigo =====
        case NodeType::RNN:
        case NodeType::LSTM:
        case NodeType::GRU:
        case NodeType::Bidirectional:
        case NodeType::TimeDistributed:
        case NodeType::Embedding:
            return IM_COL32(63, 81, 181, 255);

        // ===== Attention & Transformer - Deep Purple =====
        case NodeType::MultiHeadAttention:
        case NodeType::SelfAttention:
        case NodeType::CrossAttention:
        case NodeType::LinearAttention:
        case NodeType::TransformerEncoder:
        case NodeType::TransformerDecoder:
        case NodeType::PositionalEncoding:
            return IM_COL32(103, 58, 183, 255);

        // ===== Activation Functions - Orange/Yellow =====
        case NodeType::ReLU:
            return IM_COL32(243, 156, 18, 255);
        case NodeType::Sigmoid:
            return IM_COL32(241, 196, 15, 255);
        case NodeType::Tanh:
            return IM_COL32(230, 126, 34, 255);
        case NodeType::Softmax:
            return IM_COL32(211, 84, 0, 255);
        case NodeType::LeakyReLU:
        case NodeType::PReLU:
        case NodeType::ELU:
        case NodeType::SELU:
        case NodeType::GELU:
        case NodeType::Swish:
        case NodeType::Mish:
            return IM_COL32(235, 152, 78, 255);

        // ===== Shape Operations - Turquoise =====
        case NodeType::Reshape:
        case NodeType::Permute:
        case NodeType::Squeeze:
        case NodeType::Unsqueeze:
        case NodeType::View:
        case NodeType::Split:
            return IM_COL32(26, 188, 156, 255);

        // ===== Merge Operations - Lime Green =====
        case NodeType::Concatenate:
        case NodeType::Add:
        case NodeType::Multiply:
        case NodeType::Average:
            return IM_COL32(139, 195, 74, 255);

        // ===== Loss Functions - Dark Red =====
        case NodeType::MSELoss:
        case NodeType::CrossEntropyLoss:
        case NodeType::BCELoss:
        case NodeType::BCEWithLogits:
        case NodeType::L1Loss:
        case NodeType::SmoothL1Loss:
        case NodeType::HuberLoss:
        case NodeType::NLLLoss:
            return IM_COL32(192, 57, 43, 255);

        // ===== Optimizers - Dark Blue Gray =====
        case NodeType::SGD:
        case NodeType::Adam:
        case NodeType::AdamW:
        case NodeType::RMSprop:
        case NodeType::Adagrad:
        case NodeType::NAdam:
            return IM_COL32(52, 73, 94, 255);

        // ===== Learning Rate Schedulers - Steel Blue =====
        case NodeType::StepLR:
        case NodeType::CosineAnnealing:
        case NodeType::ReduceOnPlateau:
        case NodeType::ExponentialLR:
        case NodeType::WarmupScheduler:
            return IM_COL32(96, 125, 139, 255);

        // ===== Regularization Nodes - Magenta/Pink =====
        case NodeType::L1Regularization:
        case NodeType::L2Regularization:
        case NodeType::ElasticNet:
            return IM_COL32(233, 30, 99, 255);

        // ===== Utility Nodes - Gray =====
        case NodeType::Lambda:
        case NodeType::Identity:
        case NodeType::Constant:
        case NodeType::Parameter:
            return IM_COL32(158, 158, 158, 255);

        // ===== Data Pipeline - Cyan =====
        case NodeType::DatasetInput:
            return IM_COL32(0, 188, 212, 255);
        case NodeType::DataLoader:
            return IM_COL32(0, 172, 193, 255);
        case NodeType::Augmentation:
            return IM_COL32(0, 151, 167, 255);
        case NodeType::DataSplit:
            return IM_COL32(38, 198, 218, 255);
        case NodeType::TensorReshape:
            return IM_COL32(77, 208, 225, 255);
        case NodeType::Normalize:
            return IM_COL32(128, 222, 234, 255);
        case NodeType::OneHotEncode:
            return IM_COL32(0, 131, 143, 255);

        default:
            return IM_COL32(127, 140, 141, 255);
    }
}

// ========== Graph Validation ==========
bool NodeEditor::ValidateGraph(std::string& error_message) {
    if (nodes_.empty()) {
        error_message = "Graph is empty. Add nodes first.";
        return false;
    }

    // Check for Input node
    if (!HasInputNode()) {
        error_message = "Graph must have at least one Input node.";
        return false;
    }

    // Check for Output node
    if (!HasOutputNode()) {
        error_message = "Graph must have at least one Output node.";
        return false;
    }

    // Check for cycles
    if (HasCycle()) {
        error_message = "Graph contains cycles. Neural networks must be acyclic (DAG).";
        return false;
    }

    // Check that all nodes are reachable from input
    if (!AllNodesReachable()) {
        error_message = "Some nodes are not connected to the network. All nodes must be reachable from input nodes.";
        return false;
    }

    return true;
}

bool NodeEditor::IsGraphValid() const {
    // Quick check for training readiness
    // Need: DatasetInput node, at least one model layer, and a loss node
    if (nodes_.empty()) return false;

    bool has_dataset_input = false;
    bool has_loss = false;
    bool has_model_layer = false;

    for (const auto& node : nodes_) {
        if (node.type == NodeType::DatasetInput) has_dataset_input = true;
        if (node.type == NodeType::CrossEntropyLoss || node.type == NodeType::MSELoss) has_loss = true;
        if (node.type == NodeType::Dense || node.type == NodeType::Conv2D) has_model_layer = true;
    }

    // For training we need: dataset input, model layers, and loss
    return has_dataset_input && has_model_layer && has_loss;
}

void NodeEditor::UpdateDatasetNodeName(const std::string& dataset_name) {
    // Find the first DatasetInput node and update its name
    for (auto& node : nodes_) {
        if (node.type == NodeType::DatasetInput) {
            // Use dataset name if provided, otherwise default to "DataInput"
            if (dataset_name.empty()) {
                node.name = "DataInput";
            } else {
                node.name = dataset_name;
            }
            node.parameters["dataset_name"] = dataset_name;
            spdlog::info("Updated DatasetInput node name to: {}", node.name);
            break;
        }
    }
}

bool NodeEditor::HasCycle() {
    // Build adjacency list
    std::map<int, std::vector<int>> adj;
    for (const auto& link : links_) {
        adj[link.from_node].push_back(link.to_node);
    }

    // Track visited nodes and recursion stack for DFS
    std::set<int> visited;
    std::set<int> rec_stack;

    // DFS function to detect cycle
    std::function<bool(int)> dfs = [&](int node_id) -> bool {
        visited.insert(node_id);
        rec_stack.insert(node_id);

        // Visit all neighbors
        if (adj.find(node_id) != adj.end()) {
            for (int neighbor : adj[node_id]) {
                if (!visited.count(neighbor)) {
                    // Recursively visit unvisited neighbors
                    if (dfs(neighbor)) {
                        return true;  // Cycle found in subtree
                    }
                } else if (rec_stack.count(neighbor)) {
                    // Found a back edge (cycle detected)
                    spdlog::warn("Cycle detected: node {} -> node {}", node_id, neighbor);
                    return true;
                }
            }
        }

        // Remove from recursion stack before returning
        rec_stack.erase(node_id);
        return false;
    };

    // Check each unvisited node (handles disconnected components)
    for (const auto& node : nodes_) {
        if (!visited.count(node.id)) {
            if (dfs(node.id)) {
                return true;  // Cycle found
            }
        }
    }

    return false;  // No cycles found
}

bool NodeEditor::AllNodesReachable() {
    if (nodes_.empty()) return true;

    // Find all DatasetInput nodes
    std::vector<int> input_nodes;
    for (const auto& node : nodes_) {
        if (node.type == NodeType::DatasetInput) {
            input_nodes.push_back(node.id);
        }
    }

    if (input_nodes.empty()) return false;

    // Build adjacency list
    std::map<int, std::vector<int>> adj;
    for (const auto& link : links_) {
        adj[link.from_node].push_back(link.to_node);
    }

    // BFS from all input nodes to find reachable nodes
    std::set<int> reachable;
    std::queue<int> queue;

    // Start from all input nodes
    for (int input_id : input_nodes) {
        queue.push(input_id);
        reachable.insert(input_id);
    }

    // Perform BFS
    while (!queue.empty()) {
        int current = queue.front();
        queue.pop();

        // Visit all neighbors
        if (adj.find(current) != adj.end()) {
            for (int neighbor : adj[current]) {
                if (!reachable.count(neighbor)) {
                    reachable.insert(neighbor);
                    queue.push(neighbor);
                }
            }
        }
    }

    // Check if all nodes are reachable
    bool all_reachable = (reachable.size() == nodes_.size());

    if (!all_reachable) {
        // Log which nodes are unreachable for debugging
        for (const auto& node : nodes_) {
            if (!reachable.count(node.id)) {
                spdlog::warn("Node {} ('{}') is not reachable from input nodes", node.id, node.name);
            }
        }
    }

    return all_reachable;
}

bool NodeEditor::HasInputNode() {
    for (const auto& node : nodes_) {
        // DatasetInput is the valid input source for the graph
        if (node.type == NodeType::DatasetInput) {
            return true;
        }
    }
    return false;
}

bool NodeEditor::HasOutputNode() {
    for (const auto& node : nodes_) {
        if (node.type == NodeType::Output) {
            return true;
        }
    }
    return false;
}

// ========== Save/Load Implementation ==========

bool NodeEditor::SaveGraph(const std::string& filepath) {
    using json = nlohmann::json;

    try {
        json j;
        j["version"] = "1.0";
        j["framework"] = static_cast<int>(selected_framework_);

        // Serialize nodes
        json nodes_array = json::array();
        for (const auto& node : nodes_) {
            json node_json;
            node_json["id"] = node.id;
            node_json["type"] = static_cast<int>(node.type);
            node_json["name"] = node.name;
            node_json["parameters"] = node.parameters;

            // Save node position
            auto it = cached_node_positions_.find(node.id);
            ImVec2 pos = (it != cached_node_positions_.end()) ? it->second : ImVec2(0,0);
            node_json["pos_x"] = pos.x;
            node_json["pos_y"] = pos.y;

            nodes_array.push_back(node_json);
        }
        j["nodes"] = nodes_array;

        // Serialize links
        json links_array = json::array();
        for (const auto& link : links_) {
            json link_json;
            link_json["id"] = link.id;
            link_json["from_node"] = link.from_node;
            link_json["from_pin"] = link.from_pin;
            link_json["to_node"] = link.to_node;
            link_json["to_pin"] = link.to_pin;
            links_array.push_back(link_json);
        }
        j["links"] = links_array;

        // Write to file
        std::ofstream file(filepath);
        if (!file.is_open()) {
            spdlog::error("Failed to open file for writing: {}", filepath);
            return false;
        }

        file << j.dump(4);  // Pretty print with 4-space indent
        current_file_path_ = filepath;
        spdlog::info("Graph saved to: {}", filepath);
        return true;

    } catch (const std::exception& e) {
        spdlog::error("Error saving graph: {}", e.what());
        return false;
    }
}

bool NodeEditor::LoadGraph(const std::string& filepath) {
    using json = nlohmann::json;

    try {
        std::ifstream file(filepath);
        if (!file.is_open()) {
            spdlog::error("Failed to open file for reading: {}", filepath);
            return false;
        }

        json j;
        file >> j;

        // Clear existing graph
        ClearGraph();

        // Update next IDs to avoid conflicts
        int max_node_id = 0;
        int max_pin_id = 0;
        int max_link_id = 0;

        // Load framework
        if (j.contains("framework")) {
            selected_framework_ = static_cast<CodeFramework>(j["framework"].get<int>());
        }

        // Load nodes
        for (const auto& node_json : j["nodes"]) {
            MLNode node;
            node.id = node_json["id"];
            node.type = static_cast<NodeType>(node_json["type"].get<int>());
            node.name = node_json["name"];

            if (node_json.contains("parameters")) {
                node.parameters = node_json["parameters"].get<std::map<std::string, std::string>>();
            }

            // Recreate pins based on node type
            MLNode template_node = CreateNode(node.type, node.name);
            node.inputs = template_node.inputs;
            node.outputs = template_node.outputs;

            // Update max IDs
            max_node_id = std::max(max_node_id, node.id);
            for (const auto& pin : node.inputs) {
                max_pin_id = std::max(max_pin_id, pin.id);
            }
            for (const auto& pin : node.outputs) {
                max_pin_id = std::max(max_pin_id, pin.id);
            }

            nodes_.push_back(node);

            // Restore node position
            if (node_json.contains("pos_x") && node_json.contains("pos_y")) {
                float pos_x = node_json["pos_x"];
                float pos_y = node_json["pos_y"];
                ImNodes::SetNodeGridSpacePos(node.id, ImVec2(pos_x, pos_y));
            }
        }

        // Load links
        for (const auto& link_json : j["links"]) {
            NodeLink link;
            link.id = link_json["id"];
            link.from_node = link_json["from_node"];
            link.from_pin = link_json["from_pin"];
            link.to_node = link_json["to_node"];
            link.to_pin = link_json["to_pin"];
            links_.push_back(link);

            max_link_id = std::max(max_link_id, link.id);
        }

        // Update next IDs
        next_node_id_ = max_node_id + 1;
        next_pin_id_ = max_pin_id + 1;
        next_link_id_ = max_link_id + 1;

        current_file_path_ = filepath;
        spdlog::info("Graph loaded from: {} ({} nodes, {} links)",
                     filepath, nodes_.size(), links_.size());
        return true;

    } catch (const std::exception& e) {
        spdlog::error("Error loading graph: {}", e.what());
        return false;
    }
}

#ifdef _WIN32
void NodeEditor::ShowSaveDialog() {
    char szFile[260] = {0};

    OPENFILENAMEA ofn;
    ZeroMemory(&ofn, sizeof(ofn));
    ofn.lStructSize = sizeof(ofn);
    ofn.hwndOwner = NULL;
    ofn.lpstrFilter = "CyxWiz Graph Files\0*.cyxgraph\0All Files\0*.*\0";
    ofn.lpstrFile = szFile;
    ofn.nMaxFile = sizeof(szFile);
    ofn.Flags = OFN_PATHMUSTEXIST | OFN_OVERWRITEPROMPT;
    ofn.lpstrDefExt = "cyxwiz";
    ofn.lpstrTitle = "Save Neural Network Graph";

    if (GetSaveFileNameA(&ofn)) {
        if (SaveGraph(szFile)) {
            spdlog::info("Graph successfully saved");
        }
    }
}

void NodeEditor::ShowLoadDialog() {
    char szFile[260] = {0};

    OPENFILENAMEA ofn;
    ZeroMemory(&ofn, sizeof(ofn));
    ofn.lStructSize = sizeof(ofn);
    ofn.hwndOwner = NULL;
    ofn.lpstrFilter = "CyxWiz Graph Files\0*.cyxgraph\0All Files\0*.*\0";
    ofn.lpstrFile = szFile;
    ofn.nMaxFile = sizeof(szFile);
    ofn.Flags = OFN_PATHMUSTEXIST | OFN_FILEMUSTEXIST;
    ofn.lpstrTitle = "Load Neural Network Graph";

    if (GetOpenFileNameA(&ofn)) {
        if (LoadGraph(szFile)) {
            spdlog::info("Graph successfully loaded");
        }
    }
}
#endif // _WIN32

// ========== Code Export Implementation ==========

void NodeEditor::ExportCodeToFile() {
    // Validate graph first
    std::string error_message;
    if (!ValidateGraph(error_message)) {
        spdlog::error("Cannot export code: {}", error_message);
        // TODO: Show error dialog to user
        return;
    }

    // Generate code
    auto sorted_ids = TopologicalSort();
    if (sorted_ids.empty()) {
        spdlog::error("Failed to sort graph for code generation");
        return;
    }

    std::string code;
    std::string extension = ".py";
    std::string framework_name;

    switch (selected_framework_) {
        case CodeFramework::PyTorch:
            code = GeneratePyTorchCode(sorted_ids);
            framework_name = "PyTorch";
            break;
        case CodeFramework::TensorFlow:
            code = GenerateTensorFlowCode(sorted_ids);
            framework_name = "TensorFlow";
            break;
        case CodeFramework::Keras:
            code = GenerateKerasCode(sorted_ids);
            framework_name = "Keras";
            break;
        case CodeFramework::PyCyxWiz:
            code = GeneratePyCyxWizCode(sorted_ids);
            framework_name = "PyCyxWiz";
            break;
    }

    // Build the code with header and footer
    std::string header = "# Neural Network Model Generated by CyxWiz\n";
    header += "# Framework: " + framework_name + "\n";
    header += "# Generated on: " + std::string(__DATE__) + " " + std::string(__TIME__) + "\n\n";

    std::string full_code = header + code;

    // Save to file - will be called from ShowExportDialog
    return;
}

#ifdef _WIN32
void NodeEditor::ShowExportDialog() {
    // Validate graph first
    std::string error_message;
    if (!ValidateGraph(error_message)) {
        spdlog::error("Cannot export code: {}", error_message);
        // TODO: Show error dialog to user
        return;
    }

    // Generate code
    auto sorted_ids = TopologicalSort();
    if (sorted_ids.empty()) {
        spdlog::error("Failed to sort graph for code generation");
        return;
    }

    std::string code;
    std::string framework_name;

    switch (selected_framework_) {
        case CodeFramework::PyTorch:
            code = GeneratePyTorchCode(sorted_ids);
            framework_name = "PyTorch";
            break;
        case CodeFramework::TensorFlow:
            code = GenerateTensorFlowCode(sorted_ids);
            framework_name = "TensorFlow";
            break;
        case CodeFramework::Keras:
            code = GenerateKerasCode(sorted_ids);
            framework_name = "Keras";
            break;
        case CodeFramework::PyCyxWiz:
            code = GeneratePyCyxWizCode(sorted_ids);
            framework_name = "PyCyxWiz";
            break;
    }

    // Build the code with header
    std::string header = "# Neural Network Model Generated by CyxWiz\n";
    header += "# Framework: " + framework_name + "\n";
    header += "# Generated on: " + std::string(__DATE__) + " " + std::string(__TIME__) + "\n\n";

    std::string full_code = header + code;

    // Show Windows file save dialog
    char szFile[260] = {0};

    // Suggest a default filename based on framework
    std::string default_name = "model_" + framework_name + ".py";
    strncpy(szFile, default_name.c_str(), sizeof(szFile) - 1);

    OPENFILENAMEA ofn;
    ZeroMemory(&ofn, sizeof(ofn));
    ofn.lStructSize = sizeof(ofn);
    ofn.hwndOwner = NULL;
    ofn.lpstrFilter = "Python Files\0*.py\0All Files\0*.*\0";
    ofn.lpstrFile = szFile;
    ofn.nMaxFile = sizeof(szFile);
    ofn.Flags = OFN_PATHMUSTEXIST | OFN_OVERWRITEPROMPT;
    ofn.lpstrDefExt = "py";
    ofn.lpstrTitle = "Export Neural Network Code";

    if (GetSaveFileNameA(&ofn)) {
        // Save code to file
        std::ofstream file(szFile);
        if (file.is_open()) {
            file << full_code;
            file.close();
            spdlog::info("Code exported successfully to: {}", szFile);
        } else {
            spdlog::error("Failed to open file for writing: {}", szFile);
        }
    }
}
#else
// For non-Windows platforms, implement later or use a cross-platform dialog library
void NodeEditor::ShowSaveDialog() {
    SaveGraph("model.cyxgraph");
}

void NodeEditor::ShowLoadDialog() {
    LoadGraph("model.cyxgraph");
}

void NodeEditor::ShowExportDialog() {
    // Export to default filename for now
    std::string filename = "model_export.py";

    // Generate and save code
    ExportCodeToFile();

    // Validate graph first
    std::string error_message;
    if (!ValidateGraph(error_message)) {
        spdlog::error("Cannot export code: {}", error_message);
        return;
    }

    auto sorted_ids = TopologicalSort();
    if (sorted_ids.empty()) {
        return;
    }

    std::string code;
    std::string framework_name;

    switch (selected_framework_) {
        case CodeFramework::PyTorch:
            code = GeneratePyTorchCode(sorted_ids);
            framework_name = "PyTorch";
            filename = "model_pytorch.py";
            break;
        case CodeFramework::TensorFlow:
            code = GenerateTensorFlowCode(sorted_ids);
            framework_name = "TensorFlow";
            filename = "model_tensorflow.py";
            break;
        case CodeFramework::Keras:
            code = GenerateKerasCode(sorted_ids);
            framework_name = "Keras";
            filename = "model_keras.py";
            break;
        case CodeFramework::PyCyxWiz:
            code = GeneratePyCyxWizCode(sorted_ids);
            framework_name = "PyCyxWiz";
            filename = "model_pycyxwiz.py";
            break;
    }

    std::string header = "# Neural Network Model Generated by CyxWiz\n";
    header += "# Framework: " + framework_name + "\n\n";
    std::string full_code = header + code;

    std::ofstream file(filename);
    if (file.is_open()) {
        file << full_code;
        file.close();
        spdlog::info("Code exported to: {}", filename);
    }
}
#endif

} // namespace gui
