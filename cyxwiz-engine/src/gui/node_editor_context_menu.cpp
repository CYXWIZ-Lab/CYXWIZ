/**
 * Node Editor - Context Menu
 * Handles the right-click context menu for adding nodes
 */

#include "node_editor.h"
#include "icons.h"
#include "../core/project_manager.h"
#include <imgui.h>
#include <cstring>

namespace gui {

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

    // ===== SKIP CONNECTIONS =====
    if (ImGui::BeginMenu("Skip Connections")) {
        if (ImGui::MenuItem("Add Residual (Add node)")) {
            // Add a new Add node for residual connection
            AddNode(NodeType::Add, "Residual Add");
            ImGui::CloseCurrentPopup();
        }
        if (ImGui::MenuItem("Add Dense Skip (Concat node)")) {
            // Add a new Concatenate node for dense-style skip
            AddNode(NodeType::Concatenate, "Dense Concat");
            ImGui::CloseCurrentPopup();
        }
        ImGui::Separator();
        // Selection-based operations (only enabled when nodes are selected)
        bool has_selection = !selected_node_ids_.empty();
        if (ImGui::MenuItem("Wrap Selection with Residual", nullptr, false, has_selection)) {
            WrapSelectionWithResidual();
            ImGui::CloseCurrentPopup();
        }
        ImGui::Separator();
        if (ImGui::MenuItem("Auto-Detect Skip Connections")) {
            DetectSkipConnections();
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

        // Arrangement/Alignment submenu
        if (ImGui::BeginMenu(ICON_FA_LAYER_GROUP " Arrange")) {
            ImGui::TextDisabled("Align");
            if (ImGui::MenuItem("Align Left", nullptr, false, selected_node_ids_.size() >= 2)) {
                AlignSelectedNodes(AlignmentType::Left);
                ImGui::CloseCurrentPopup();
            }
            if (ImGui::MenuItem("Align Center", nullptr, false, selected_node_ids_.size() >= 2)) {
                AlignSelectedNodes(AlignmentType::Center);
                ImGui::CloseCurrentPopup();
            }
            if (ImGui::MenuItem("Align Right", nullptr, false, selected_node_ids_.size() >= 2)) {
                AlignSelectedNodes(AlignmentType::Right);
                ImGui::CloseCurrentPopup();
            }
            ImGui::Separator();
            if (ImGui::MenuItem("Align Top", nullptr, false, selected_node_ids_.size() >= 2)) {
                AlignSelectedNodes(AlignmentType::Top);
                ImGui::CloseCurrentPopup();
            }
            if (ImGui::MenuItem("Align Middle", nullptr, false, selected_node_ids_.size() >= 2)) {
                AlignSelectedNodes(AlignmentType::Middle);
                ImGui::CloseCurrentPopup();
            }
            if (ImGui::MenuItem("Align Bottom", nullptr, false, selected_node_ids_.size() >= 2)) {
                AlignSelectedNodes(AlignmentType::Bottom);
                ImGui::CloseCurrentPopup();
            }

            ImGui::Separator();
            ImGui::TextDisabled("Distribute");
            if (ImGui::MenuItem("Distribute Horizontally", nullptr, false, selected_node_ids_.size() >= 3)) {
                DistributeSelectedNodes(DistributeType::Horizontal);
                ImGui::CloseCurrentPopup();
            }
            if (ImGui::MenuItem("Distribute Vertically", nullptr, false, selected_node_ids_.size() >= 3)) {
                DistributeSelectedNodes(DistributeType::Vertical);
                ImGui::CloseCurrentPopup();
            }

            ImGui::Separator();
            if (ImGui::MenuItem("Auto Layout (Grid)", nullptr, false, !selected_node_ids_.empty())) {
                AutoLayoutSelection();
                ImGui::CloseCurrentPopup();
            }

            ImGui::EndMenu();
        }

        ImGui::Separator();

        // Grouping options
        if (ImGui::MenuItem(ICON_FA_OBJECT_GROUP " Create Group", "Ctrl+G", false, selected_node_ids_.size() >= 1)) {
            CreateGroupFromSelection("");
            ImGui::CloseCurrentPopup();
        }
        if (ImGui::MenuItem(ICON_FA_OBJECT_UNGROUP " Ungroup", "Ctrl+Shift+G", false, !selected_node_ids_.empty())) {
            UngroupSelection();
            ImGui::CloseCurrentPopup();
        }

        ImGui::Separator();

        // Subgraph options
        if (ImGui::MenuItem(ICON_FA_COMPRESS " Create Subgraph", "Ctrl+Shift+S", false, selected_node_ids_.size() >= 2)) {
            CreateSubgraphFromSelection("");
            ImGui::CloseCurrentPopup();
        }

        // If a single subgraph node is selected, show expand/collapse option
        if (selected_node_ids_.size() == 1 && IsSubgraphNode(selected_node_ids_[0])) {
            SubgraphData* data = GetSubgraphData(selected_node_ids_[0]);
            if (data) {
                if (data->expanded) {
                    if (ImGui::MenuItem(ICON_FA_COMPRESS " Collapse Subgraph")) {
                        CollapseSubgraph(selected_node_ids_[0]);
                        ImGui::CloseCurrentPopup();
                    }
                } else {
                    if (ImGui::MenuItem(ICON_FA_EXPAND " Expand Subgraph")) {
                        ExpandSubgraph(selected_node_ids_[0]);
                        ImGui::CloseCurrentPopup();
                    }
                }
            }
        }
    }

    ImGui::Separator();
}

} // namespace gui
