/**
 * Node Editor - Context Menu
 * Handles the right-click context menu for adding nodes
 */

#include "node_editor.h"
#include "icons.h"
#include "properties.h"
#include "../core/project_manager.h"
#include "../plugin/registries/plugin_node_registry.h"
#include <imgui.h>
#include <algorithm>
#include <cstring>

namespace gui {

void NodeEditor::ShowSingleNodeContextMenu() {
    MLNode* node = FindNodeById(right_clicked_node_id_);
    if (!node) {
        ImGui::Text("Node not found");
        return;
    }

    ImGui::TextDisabled("%s", node->name.c_str());
    ImGui::Separator();

    // Set Description (KNIME-style)
    if (ImGui::MenuItem(ICON_FA_COMMENT " Set Description...")) {
        editing_node_description_ = true;
        editing_node_id_ = right_clicked_node_id_;
        strncpy(node_description_buffer_, node->description.c_str(), sizeof(node_description_buffer_) - 1);
        node_description_buffer_[sizeof(node_description_buffer_) - 1] = '\0';
        ImGui::CloseCurrentPopup();
    }

    // Rename
    if (ImGui::MenuItem(ICON_FA_PEN " Rename...")) {
        // TODO: Implement rename dialog
        ImGui::CloseCurrentPopup();
    }

    ImGui::Separator();

    // Duplicate
    if (ImGui::MenuItem(ICON_FA_COPY " Duplicate", "Ctrl+D")) {
        // Select only this node and duplicate
        selected_node_ids_.clear();
        selected_node_ids_.push_back(right_clicked_node_id_);
        DuplicateSelection();
        ImGui::CloseCurrentPopup();
    }

    // Delete
    if (ImGui::MenuItem(ICON_FA_TRASH " Delete", "Delete")) {
        DeleteNode(right_clicked_node_id_);
        ImGui::CloseCurrentPopup();
    }

    ImGui::Separator();

    // Configure (open properties)
    if (ImGui::MenuItem(ICON_FA_GEAR " Configure...")) {
        selected_node_id_ = right_clicked_node_id_;
        selected_node_ids_.clear();
        selected_node_ids_.push_back(right_clicked_node_id_);
        if (properties_panel_) {
            properties_panel_->SetSelectedNode(node);
        }
        ImGui::CloseCurrentPopup();
    }

    // Group operations
    NodeGroup* existing_group = FindGroupContainingNode(right_clicked_node_id_);
    if (existing_group) {
        if (ImGui::MenuItem(ICON_FA_OBJECT_UNGROUP " Remove from Group")) {
            existing_group->node_ids.erase(
                std::remove(existing_group->node_ids.begin(), existing_group->node_ids.end(), right_clicked_node_id_),
                existing_group->node_ids.end());
            ImGui::CloseCurrentPopup();
        }
    }
}

void NodeEditor::ShowNodeDescriptionEditPopup() {
    if (!editing_node_description_) return;

    ImGui::OpenPopup("Edit Node Description");

    // Center the modal
    ImVec2 center = ImGui::GetMainViewport()->GetCenter();
    ImGui::SetNextWindowPos(center, ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));
    ImGui::SetNextWindowSize(ImVec2(450, 250), ImGuiCond_Appearing);

    if (ImGui::BeginPopupModal("Edit Node Description", &editing_node_description_, ImGuiWindowFlags_AlwaysAutoResize)) {
        MLNode* node = FindNodeById(editing_node_id_);
        if (node) {
            ImGui::Text("Node: %s", node->name.c_str());
            ImGui::Separator();

            ImGui::Text("Description:");
            ImGui::PushItemWidth(-1);
            ImGui::InputTextMultiline("##description", node_description_buffer_, sizeof(node_description_buffer_),
                ImVec2(-1, 120), ImGuiInputTextFlags_AllowTabInput);
            ImGui::PopItemWidth();

            ImGui::TextDisabled("Tip: This text will appear below the node on the canvas.");

            ImGui::Separator();

            if (ImGui::Button("Save", ImVec2(120, 0))) {
                node->description = node_description_buffer_;
                editing_node_description_ = false;
                ImGui::CloseCurrentPopup();
            }
            ImGui::SameLine();
            if (ImGui::Button("Cancel", ImVec2(120, 0))) {
                editing_node_description_ = false;
                ImGui::CloseCurrentPopup();
            }
        } else {
            ImGui::Text("Node not found");
            if (ImGui::Button("Close")) {
                editing_node_description_ = false;
                ImGui::CloseCurrentPopup();
            }
        }
        ImGui::EndPopup();
    }
}

#if 0
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
        // Phase 6: Advanced Augmentation Nodes
        if (ImGui::BeginMenu("Advanced Augmentation")) {
            if (ImGui::MenuItem("Augmentation Preset")) {
                AddNode(NodeType::AugmentationPreset, "AugmentationPreset");
                ImGui::CloseCurrentPopup();
            }
            if (ImGui::MenuItem("Geometric Transform")) {
                AddNode(NodeType::GeometricTransform, "GeometricTransform");
                ImGui::CloseCurrentPopup();
            }
            if (ImGui::MenuItem("Color Transform")) {
                AddNode(NodeType::ColorTransform, "ColorTransform");
                ImGui::CloseCurrentPopup();
            }
            if (ImGui::MenuItem("Morphology Transform")) {
                AddNode(NodeType::MorphologyTransform, "MorphologyTransform");
                ImGui::CloseCurrentPopup();
            }
            if (ImGui::MenuItem("Advanced Augment")) {
                AddNode(NodeType::AdvancedAugment, "AdvancedAugment");
                ImGui::CloseCurrentPopup();
            }
            ImGui::EndMenu();
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

    // ===== MODEL EVALUATION (Phase 7: UI Consolidation) =====
    if (ImGui::BeginMenu("Model Evaluation")) {
        if (ImGui::MenuItem("Confusion Matrix")) {
            AddNode(NodeType::ConfusionMatrixNode, "ConfusionMatrix");
            ImGui::CloseCurrentPopup();
        }
        if (ImGui::MenuItem("ROC Curve")) {
            AddNode(NodeType::ROCCurveNode, "ROCCurve");
            ImGui::CloseCurrentPopup();
        }
        if (ImGui::MenuItem("PR Curve")) {
            AddNode(NodeType::PRCurveNode, "PRCurve");
            ImGui::CloseCurrentPopup();
        }
        if (ImGui::MenuItem("Learning Curves")) {
            AddNode(NodeType::LearningCurvesNode, "LearningCurves");
            ImGui::CloseCurrentPopup();
        }
        if (ImGui::MenuItem("Feature Importance")) {
            AddNode(NodeType::FeatureImportanceNode, "FeatureImportance");
            ImGui::CloseCurrentPopup();
        }
        if (ImGui::MenuItem("Cross Validation")) {
            AddNode(NodeType::CrossValidationNode, "CrossValidation");
            ImGui::CloseCurrentPopup();
        }
        if (ImGui::MenuItem("Regression Metrics")) {
            AddNode(NodeType::RegressionMetricsNode, "RegressionMetrics");
            ImGui::CloseCurrentPopup();
        }
        ImGui::EndMenu();
    }

    // ===== PREPROCESSING (Phase 8: UI Consolidation) =====
    if (ImGui::BeginMenu("Preprocessing")) {
        if (ImGui::MenuItem("Outlier Detector")) {
            AddNode(NodeType::OutlierDetector, "OutlierDetector");
            ImGui::CloseCurrentPopup();
        }
        if (ImGui::MenuItem("Image Preprocessor")) {
            AddNode(NodeType::ImagePreprocessor, "ImagePreprocessor");
            ImGui::CloseCurrentPopup();
        }
        if (ImGui::MenuItem("Quality Analyzer")) {
            AddNode(NodeType::QualityAnalyzer, "QualityAnalyzer");
            ImGui::CloseCurrentPopup();
        }
        if (ImGui::MenuItem("Data Validator")) {
            AddNode(NodeType::DataValidator, "DataValidator");
            ImGui::CloseCurrentPopup();
        }
        ImGui::EndMenu();
    }

    // ===== TEXT PROCESSING =====
    if (ImGui::BeginMenu("Text Processing")) {
        if (ImGui::MenuItem("Text Tokenizer")) {
            AddNode(NodeType::TextTokenizer, "TextTokenizer");
            ImGui::CloseCurrentPopup();
        }
        if (ImGui::MenuItem("Vocabulary")) {
            AddNode(NodeType::TextVocabulary, "TextVocabulary");
            ImGui::CloseCurrentPopup();
        }
        if (ImGui::MenuItem("Text Padding")) {
            AddNode(NodeType::TextPadding, "TextPadding");
            ImGui::CloseCurrentPopup();
        }
        ImGui::EndMenu();
    }

    // ===== UPSAMPLING =====
    if (ImGui::BeginMenu("Upsampling")) {
        if (ImGui::MenuItem("ConvTranspose2D")) {
            AddNode(NodeType::ConvTranspose2D, "ConvTranspose2D");
            ImGui::CloseCurrentPopup();
        }
        if (ImGui::MenuItem("Upsample")) {
            AddNode(NodeType::Upsample, "Upsample");
            ImGui::CloseCurrentPopup();
        }
        if (ImGui::MenuItem("PixelShuffle")) {
            AddNode(NodeType::PixelShuffle, "PixelShuffle");
            ImGui::CloseCurrentPopup();
        }
        ImGui::EndMenu();
    }

    // ===== TIME-SERIES =====
    if (ImGui::BeginMenu("Time-Series")) {
        if (ImGui::MenuItem("Sliding Window")) {
            AddNode(NodeType::TimeSeriesWindow, "TimeSeriesWindow");
            ImGui::CloseCurrentPopup();
        }
        if (ImGui::MenuItem("Feature Engineering")) {
            AddNode(NodeType::TimeSeriesFeatures, "TimeSeriesFeatures");
            ImGui::CloseCurrentPopup();
        }
        if (ImGui::MenuItem("Chronological Split")) {
            AddNode(NodeType::TimeSeriesSplit, "TimeSeriesSplit");
            ImGui::CloseCurrentPopup();
        }
        ImGui::EndMenu();
    }

    // ===== AUDIO PROCESSING =====
    if (ImGui::BeginMenu("Audio Processing")) {
        if (ImGui::MenuItem("Audio Input")) {
            AddNode(NodeType::AudioInput, "AudioInput");
            ImGui::CloseCurrentPopup();
        }
        if (ImGui::MenuItem("Spectrogram")) {
            AddNode(NodeType::Spectrogram, "Spectrogram");
            ImGui::CloseCurrentPopup();
        }
        if (ImGui::MenuItem("Mel Spectrogram")) {
            AddNode(NodeType::MelSpectrogram, "MelSpectrogram");
            ImGui::CloseCurrentPopup();
        }
        if (ImGui::MenuItem("MFCC")) {
            AddNode(NodeType::MFCC, "MFCC");
            ImGui::CloseCurrentPopup();
        }
        if (ImGui::MenuItem("Audio Augmentation")) {
            AddNode(NodeType::AudioAugmentation, "AudioAugmentation");
            ImGui::CloseCurrentPopup();
        }
        ImGui::EndMenu();
    }

    // ===== REINFORCEMENT LEARNING =====
    if (ImGui::BeginMenu("Reinforcement Learning")) {
        if (ImGui::MenuItem("Gym Environment")) {
            AddNode(NodeType::GymEnvironment, "GymEnvironment");
            ImGui::CloseCurrentPopup();
        }
        if (ImGui::MenuItem("Replay Buffer")) {
            AddNode(NodeType::ReplayBufferNode, "ReplayBuffer");
            ImGui::CloseCurrentPopup();
        }
        if (ImGui::MenuItem("Policy Network")) {
            AddNode(NodeType::PolicyNetwork, "PolicyNetwork");
            ImGui::CloseCurrentPopup();
        }
        if (ImGui::MenuItem("Value Network")) {
            AddNode(NodeType::ValueNetwork, "ValueNetwork");
            ImGui::CloseCurrentPopup();
        }
        if (ImGui::MenuItem("RL Training")) {
            AddNode(NodeType::RLTraining, "RLTraining");
            ImGui::CloseCurrentPopup();
        }

        // Plugin RL/Simulation nodes
        {
            auto plugin_nodes = cyxwiz::plugin::PluginNodeRegistry::Instance().GetAllNodeTypesWithNames();
            bool has_rl = false;
            for (const auto& [qname, info] : plugin_nodes) {
                if (info.category.find("RL") != std::string::npos ||
                    info.category.find("Simulation") != std::string::npos) {
                    if (!has_rl) { ImGui::Separator(); has_rl = true; }
                    if (ImGui::MenuItem(info.display_name.c_str())) {
                        AddNode(NodeType::PluginCustom, qname);
                        ImGui::CloseCurrentPopup();
                    }
                }
            }
        }

        ImGui::EndMenu();
    }

    // ===== ANALYTICS (Clustering, Statistics) =====
    if (ImGui::BeginMenu("Analytics")) {
        if (ImGui::BeginMenu("Clustering")) {
            if (ImGui::MenuItem("K-Means Clustering")) {
                AddNode(NodeType::KMeansCluster, "K-Means Clustering");
                ImGui::CloseCurrentPopup();
            }
            // TODO: Add more clustering algorithms as executors are implemented
            // if (ImGui::MenuItem("DBSCAN")) { AddNode(NodeType::DBSCANCluster, "DBSCAN"); ImGui::CloseCurrentPopup(); }
            // if (ImGui::MenuItem("Hierarchical")) { AddNode(NodeType::HierarchicalCluster, "Hierarchical"); ImGui::CloseCurrentPopup(); }
            ImGui::EndMenu();
        }
        if (ImGui::BeginMenu("Dimensionality Reduction")) {
            if (ImGui::MenuItem("PCA")) {
                AddNode(NodeType::PCANode, "PCA");
                ImGui::CloseCurrentPopup();
            }
            // TODO: Add more as executors are implemented
            ImGui::EndMenu();
        }
        if (ImGui::MenuItem("Correlation Matrix")) {
            AddNode(NodeType::CorrelationMatrix, "Correlation Matrix");
            ImGui::CloseCurrentPopup();
        }
        ImGui::EndMenu();
    }

    // ===== PLUGIN NODES (non-RL categories) =====
    {
        auto plugin_nodes = cyxwiz::plugin::PluginNodeRegistry::Instance().GetAllNodeTypesWithNames();
        std::map<std::string, std::vector<std::pair<std::string, cyxwiz::plugin::PluginNodeTypeInfo>>> by_category;
        for (auto& [qname, info] : plugin_nodes) {
            if (info.category.find("RL") != std::string::npos ||
                info.category.find("Simulation") != std::string::npos)
                continue;
            by_category[info.category].emplace_back(qname, std::move(info));
        }
        for (const auto& [category, nodes] : by_category) {
            if (ImGui::BeginMenu(category.c_str())) {
                for (const auto& [qname, info] : nodes) {
                    if (ImGui::MenuItem(info.display_name.c_str())) {
                        AddNode(NodeType::PluginCustom, qname);
                        ImGui::CloseCurrentPopup();
                    }
                }
                ImGui::EndMenu();
            }
        }
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

    // ===== DNN INFERENCE =====
    if (ImGui::BeginMenu("Inference")) {
        if (ImGui::BeginMenu("Pre-trained Models")) {
            if (ImGui::MenuItem("YOLO v4")) { AddNode(NodeType::PretrainedYOLO, "YOLOv4"); ImGui::CloseCurrentPopup(); }
            if (ImGui::MenuItem("MobileNet SSD")) { AddNode(NodeType::PretrainedMobileNet, "MobileNet-SSD"); ImGui::CloseCurrentPopup(); }
            if (ImGui::MenuItem("OpenPose")) { AddNode(NodeType::PretrainedOpenPose, "OpenPose"); ImGui::CloseCurrentPopup(); }
            if (ImGui::MenuItem("Face Detector")) { AddNode(NodeType::PretrainedFaceNet, "FaceDetector"); ImGui::CloseCurrentPopup(); }
            ImGui::EndMenu();
        }
        if (ImGui::BeginMenu("Model Loading")) {
            if (ImGui::MenuItem("Load DNN Model")) { AddNode(NodeType::DNNModelLoad, "Load Model"); ImGui::CloseCurrentPopup(); }
            if (ImGui::MenuItem("DNN Preprocessor")) { AddNode(NodeType::DNNPreprocess, "Preprocess"); ImGui::CloseCurrentPopup(); }
            ImGui::EndMenu();
        }
        if (ImGui::BeginMenu("Inference Operations")) {
            if (ImGui::MenuItem("Detect Objects")) { AddNode(NodeType::DNNDetect, "Detect"); ImGui::CloseCurrentPopup(); }
            if (ImGui::MenuItem("Classify Image")) { AddNode(NodeType::DNNClassify, "Classify"); ImGui::CloseCurrentPopup(); }
            if (ImGui::MenuItem("Estimate Pose")) { AddNode(NodeType::DNNPoseEstimate, "Pose"); ImGui::CloseCurrentPopup(); }
            if (ImGui::MenuItem("Detect Faces")) { AddNode(NodeType::DNNFaceDetect, "Faces"); ImGui::CloseCurrentPopup(); }
            ImGui::EndMenu();
        }
        if (ImGui::BeginMenu("Post-processing")) {
            if (ImGui::MenuItem("Non-Max Suppression")) { AddNode(NodeType::NonMaxSuppression, "NMS"); ImGui::CloseCurrentPopup(); }
            if (ImGui::MenuItem("ArgMax")) { AddNode(NodeType::ArgMax, "ArgMax"); ImGui::CloseCurrentPopup(); }
            if (ImGui::MenuItem("Top-K")) { AddNode(NodeType::TopK, "TopK (5)"); ImGui::CloseCurrentPopup(); }
            if (ImGui::MenuItem("Threshold Filter")) { AddNode(NodeType::ThresholdFilter, "Filter"); ImGui::CloseCurrentPopup(); }
            ImGui::EndMenu();
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

    // ===== SIGNAL / CONTROL =====
    if (ImGui::BeginMenu("Signal / Control")) {
        if (ImGui::MenuItem("Slider")) {
            AddNode(NodeType::SignalSlider, "Slider");
            ImGui::CloseCurrentPopup();
        }
        if (ImGui::MenuItem("Sine Wave")) {
            AddNode(NodeType::SineWave, "Sine Wave");
            ImGui::CloseCurrentPopup();
        }
        if (ImGui::MenuItem("Step")) {
            AddNode(NodeType::StepSignal, "Step");
            ImGui::CloseCurrentPopup();
        }
        if (ImGui::MenuItem("Ramp")) {
            AddNode(NodeType::RampSignal, "Ramp");
            ImGui::CloseCurrentPopup();
        }
        if (ImGui::MenuItem("Scope")) {
            AddNode(NodeType::SignalScope, "Scope");
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

        // KNIME-style: Set description for single node
        if (selected_node_ids_.size() == 1) {
            if (ImGui::MenuItem(ICON_FA_COMMENT " Set Description...")) {
                editing_node_description_ = true;
                editing_node_id_ = selected_node_ids_[0];
                // Copy current description to edit buffer
                MLNode* node = FindNodeById(editing_node_id_);
                if (node) {
                    strncpy(node_description_buffer_, node->description.c_str(), sizeof(node_description_buffer_) - 1);
                    node_description_buffer_[sizeof(node_description_buffer_) - 1] = '\0';
                } else {
                    node_description_buffer_[0] = '\0';
                }
                ImGui::CloseCurrentPopup();
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
#endif

void NodeEditor::ShowContextMenu() {
    InitializeSearchableNodes();

    ImGui::TextDisabled("Add Node");
    ImGui::SetNextItemWidth(-1);
    ImGui::InputTextWithHint("##ctx_search",
                             ICON_FA_MAGNIFYING_GLASS " Search nodes...",
                             context_menu_search_, sizeof(context_menu_search_));
    ImGui::Separator();

    std::string search_lower = context_menu_search_;
    std::transform(search_lower.begin(), search_lower.end(), search_lower.begin(), ::tolower);
    const bool is_searching = !search_lower.empty();

    // Group nodes by category label (from searchable list)
    std::map<std::string, std::vector<SearchableNode*>> grouped;
    for (auto& node : all_searchable_nodes_) {
        if (node.status == NodeImplementationStatus::Deprecated) {
            continue;
        }
        if (is_searching) {
            std::string name_lower = node.name;
            std::string cat_lower = node.category;
            std::string key_lower = node.keywords;
            std::transform(name_lower.begin(), name_lower.end(), name_lower.begin(), ::tolower);
            std::transform(cat_lower.begin(), cat_lower.end(), cat_lower.begin(), ::tolower);
            std::transform(key_lower.begin(), key_lower.end(), key_lower.begin(), ::tolower);
            if (name_lower.find(search_lower) == std::string::npos &&
                cat_lower.find(search_lower) == std::string::npos &&
                key_lower.find(search_lower) == std::string::npos) {
                continue;
            }
        }
        grouped[node.category].push_back(&node);
    }

    if (grouped.empty()) {
        ImGui::TextDisabled("No matches");
    } else {
        for (auto& [category, nodes] : grouped) {
            if (ImGui::CollapsingHeader(category.c_str(), ImGuiTreeNodeFlags_DefaultOpen)) {
                ImGui::Indent();
                for (auto* node : nodes) {
                    const bool is_template = (node->status == NodeImplementationStatus::Template);
                    if (is_template) ImGui::BeginDisabled();
                    if (ImGui::Selectable(node->name.c_str())) {
                        if (!is_template && node->type != NodeType::Unknown) {
                            AddNode(node->type, node->name);
                            context_menu_search_[0] = '\0';
                            ImGui::CloseCurrentPopup();
                        }
                    }
                    if (is_template) {
                        if (ImGui::IsItemHovered() && !node->tooltip.empty()) {
                            ImGui::BeginTooltip();
                            ImGui::TextUnformatted(node->tooltip.c_str());
                            ImGui::EndTooltip();
                        }
                        ImGui::EndDisabled();
                    }
                }
                ImGui::Unindent();
            }
        }
    }

    ImGui::Separator();
    if (ImGui::BeginMenu("Actions")) {
        const bool has_selection = !selected_node_ids_.empty();

        if (ImGui::MenuItem(ICON_FA_PASTE " Paste", "Ctrl+V", false, clipboard_.valid)) {
            PasteClipboard();
            ImGui::CloseCurrentPopup();
        }

        if (ImGui::BeginMenu("Align / Distribute", has_selection)) {
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
            if (ImGui::MenuItem("Auto Layout (Grid)", nullptr, false, has_selection)) {
                AutoLayoutSelection();
                ImGui::CloseCurrentPopup();
            }
            ImGui::EndMenu();
        }

        ImGui::Separator();
        if (ImGui::MenuItem(ICON_FA_OBJECT_GROUP " Create Group", "Ctrl+G", false, has_selection)) {
            CreateGroupFromSelection("");
            ImGui::CloseCurrentPopup();
        }
        if (ImGui::MenuItem(ICON_FA_OBJECT_UNGROUP " Ungroup", "Ctrl+Shift+G", false, has_selection)) {
            UngroupSelection();
            ImGui::CloseCurrentPopup();
        }

        if (ImGui::MenuItem(ICON_FA_COMPRESS " Create Subgraph", "Ctrl+Shift+S", false, selected_node_ids_.size() >= 2)) {
            CreateSubgraphFromSelection("");
            ImGui::CloseCurrentPopup();
        }

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

        ImGui::EndMenu();
    }
}

// ============================================================================
// Unified Canvas Phase 3: Categorized Node Palette
// ============================================================================

const char* NodeEditor::GetCategoryIcon(NodeCategory category) {
    switch (category) {
        case NodeCategory::DataSources:      return ICON_FA_DATABASE;
        case NodeCategory::DataTransform:    return ICON_FA_FILTER;
        case NodeCategory::Analytics:        return ICON_FA_CHART_LINE;
        case NodeCategory::Preprocessing:    return ICON_FA_WAND_MAGIC_SPARKLES;
        case NodeCategory::Layers:           return ICON_FA_LAYER_GROUP;
        case NodeCategory::Activation:       return ICON_FA_BOLT;
        case NodeCategory::Pooling:          return ICON_FA_COMPRESS;
        case NodeCategory::Normalization:    return ICON_FA_SCALE_BALANCED;
        case NodeCategory::Attention:        return ICON_FA_EYE;
        case NodeCategory::Recurrent:        return ICON_FA_ROTATE;
        case NodeCategory::ShapeOps:         return ICON_FA_OBJECT_GROUP;  // was ICON_FA_SHAPES
        case NodeCategory::MergeOps:         return ICON_FA_RIGHT_LEFT;    // was ICON_FA_CODE_MERGE
        case NodeCategory::Training:         return ICON_FA_GRADUATION_CAP;
        case NodeCategory::Regularization:   return ICON_FA_SCALE_BALANCED;  // was ICON_FA_SHIELD_HALVED
        case NodeCategory::Utility:          return ICON_FA_COG;            // was ICON_FA_SCREWDRIVER_WRENCH
        case NodeCategory::Signal:           return ICON_FA_ARROWS_ROTATE; // was ICON_FA_WAVE_SQUARE
        case NodeCategory::DataPipeline:     return ICON_FA_DIAGRAM_PROJECT;
        case NodeCategory::DNN:              return ICON_FA_LIGHTBULB;      // was ICON_FA_BRAIN
        case NodeCategory::TextProcessing:   return ICON_FA_FILE_LINES;
        case NodeCategory::Upsampling:       return ICON_FA_EXPAND;         // was ICON_FA_UP_RIGHT_AND_DOWN_LEFT_FROM_CENTER
        case NodeCategory::TimeSeries:       return ICON_FA_ARROW_TREND_UP; // was ICON_FA_CHART_AREA
        case NodeCategory::Audio:            return ICON_FA_STETHOSCOPE;    // was ICON_FA_VOLUME_HIGH
        case NodeCategory::RL:               return ICON_FA_GRADUATION_CAP; // was ICON_FA_ROBOT
        case NodeCategory::Plugin:           return ICON_FA_DIAGRAM_PROJECT; // was ICON_FA_PUZZLE_PIECE
        default:                             return ICON_FA_CUBE;
    }
}

const char* NodeEditor::GetCategoryName(NodeCategory category) {
    switch (category) {
        case NodeCategory::DataSources:      return "Data Sources";
        case NodeCategory::DataTransform:    return "Data Transform";
        case NodeCategory::Analytics:        return "Analytics";
        case NodeCategory::Preprocessing:    return "Preprocessing";
        case NodeCategory::Layers:           return "Layers";
        case NodeCategory::Activation:       return "Activation";
        case NodeCategory::Pooling:          return "Pooling";
        case NodeCategory::Normalization:    return "Normalization";
        case NodeCategory::Attention:        return "Attention";
        case NodeCategory::Recurrent:        return "Recurrent";
        case NodeCategory::ShapeOps:         return "Shape Operations";
        case NodeCategory::MergeOps:         return "Merge Operations";
        case NodeCategory::Training:         return "Training";
        case NodeCategory::Regularization:   return "Regularization";
        case NodeCategory::Utility:          return "Utility";
        case NodeCategory::Signal:           return "Signal / Control";
        case NodeCategory::DataPipeline:     return "Data Pipeline";
        case NodeCategory::DNN:              return "DNN / Pre-trained";
        case NodeCategory::TextProcessing:   return "Text Processing";
        case NodeCategory::Upsampling:       return "Upsampling";
        case NodeCategory::TimeSeries:       return "Time Series";
        case NodeCategory::Audio:            return "Audio";
        case NodeCategory::RL:               return "Reinforcement Learning";
        case NodeCategory::Plugin:           return "Plugin Nodes";
        default:                             return "Unknown";
    }
}

void NodeEditor::ShowCategorizedNodeMenu() {
    // Initialize nodes_by_category_ on first call
    if (!nodes_by_category_initialized_) {
        // Group all node types by category
        // Data I/O (Smart unified nodes)
        nodes_by_category_[NodeCategory::DataSources] = {
            {NodeType::DataInput, "Data Input"},
            {NodeType::DataOutput, "Data Output"}
        };

        // Data Transforms
        nodes_by_category_[NodeCategory::DataTransform] = {
            {NodeType::FilterRows, "Filter Rows"},
            {NodeType::SelectColumns, "Select Columns"},
            {NodeType::JoinTables, "Join Tables"},
            {NodeType::GroupByAggregate, "Group By"},
            {NodeType::SortRows, "Sort Rows"},
            {NodeType::FillMissingValues, "Fill Missing"},
            {NodeType::RemoveDuplicateRows, "Remove Duplicates"},
            {NodeType::PivotTable, "Pivot Table"},
            {NodeType::UnionTables, "Union Tables"},
            {NodeType::RenameColumns, "Rename Columns"}
        };

        // Analytics (Clustering, Statistics, Visualization)
        nodes_by_category_[NodeCategory::Analytics] = {
            {NodeType::KMeansCluster, "K-Means Clustering"},
            {NodeType::DescribeStats, "Describe Stats"},
            {NodeType::VisualizeData, "Visualize Data"},
            {NodeType::SampleRows, "Sample Rows"},
            {NodeType::CorrelationMatrix, "Correlation Matrix"},
            {NodeType::ValueCounts, "Value Counts"},
            {NodeType::CrossTabulation, "Cross Tabulation"}
        };

        // Data Export (consolidated into DataOutput node above)
        // Legacy export nodes removed - use DataOutput instead

        // Layers (Dense, Conv, etc.)
        nodes_by_category_[NodeCategory::Layers] = {
            {NodeType::Dense, "Dense (128)"},
            {NodeType::Conv1D, "Conv1D"},
            {NodeType::Conv2D, "Conv2D"},
            {NodeType::Conv3D, "Conv3D"},
            {NodeType::DepthwiseConv2D, "DepthwiseConv2D"}
        };

        // Activation
        nodes_by_category_[NodeCategory::Activation] = {
            {NodeType::ReLU, "ReLU"},
            {NodeType::LeakyReLU, "LeakyReLU"},
            {NodeType::GELU, "GELU"},
            {NodeType::Swish, "Swish"},
            {NodeType::Sigmoid, "Sigmoid"},
            {NodeType::Tanh, "Tanh"},
            {NodeType::Softmax, "Softmax"}
        };

        // Preprocessing (Phase 8: UI Consolidation)
        nodes_by_category_[NodeCategory::Preprocessing] = {
            {NodeType::Normalize, "Normalize"},
            {NodeType::OneHotEncode, "One-Hot Encode"},
            {NodeType::OutlierDetector, "Outlier Detector"},
            {NodeType::ImagePreprocessor, "Image Preprocessor"},
            {NodeType::QualityAnalyzer, "Quality Analyzer"},
            {NodeType::DataValidator, "Data Validator"}
        };

        // ... (Add more categories as needed)

        nodes_by_category_initialized_ = true;
    }

    // Search filter
    ImGui::SetNextItemWidth(-1);
    ImGui::InputTextWithHint("##search", ICON_FA_MAGNIFYING_GLASS " Search nodes...", context_menu_search_, sizeof(context_menu_search_));
    ImGui::Separator();

    // Filter nodes based on search
    std::string search_lower = context_menu_search_;
    std::transform(search_lower.begin(), search_lower.end(), search_lower.begin(), ::tolower);
    bool is_searching = strlen(context_menu_search_) > 0;

    // Render categories
    for (auto& [category, nodes] : nodes_by_category_) {
        // Skip empty categories
        if (nodes.empty()) continue;

        // If searching, filter nodes
        std::vector<std::pair<NodeType, std::string>> filtered_nodes;
        if (is_searching) {
            for (auto& [type, name] : nodes) {
                std::string name_lower = name;
                std::transform(name_lower.begin(), name_lower.end(), name_lower.begin(), ::tolower);
                if (name_lower.find(search_lower) != std::string::npos) {
                    filtered_nodes.push_back({type, name});
                }
            }
            // Skip category if no matches
            if (filtered_nodes.empty()) continue;
        } else {
            filtered_nodes = nodes;
        }

        RenderNodeCategory(category, GetCategoryName(category), GetCategoryIcon(category));
    }
}

void NodeEditor::RenderNodeCategory(NodeCategory category, const char* category_name, const char* icon) {
    auto it = nodes_by_category_.find(category);
    if (it == nodes_by_category_.end()) return;

    // Category header with icon
    if (ImGui::CollapsingHeader((std::string(icon) + " " + category_name).c_str(), ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::Indent();

        for (auto& [type, name] : it->second) {
            // Filter by search
            if (strlen(context_menu_search_) > 0) {
                std::string name_lower = name;
                std::string search_lower = context_menu_search_;
                std::transform(name_lower.begin(), name_lower.end(), name_lower.begin(), ::tolower);
                std::transform(search_lower.begin(), search_lower.end(), search_lower.begin(), ::tolower);
                if (name_lower.find(search_lower) == std::string::npos) {
                    continue;
                }
            }

            if (ImGui::Selectable(name.c_str())) {
                AddNode(type, name);
                context_menu_search_[0] = '\0';  // Clear search on selection
                ImGui::CloseCurrentPopup();
            }
        }

        ImGui::Unindent();
    }
}

} // namespace gui
