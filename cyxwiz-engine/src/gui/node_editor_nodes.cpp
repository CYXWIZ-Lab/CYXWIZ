#include "node_editor.h"
#include "../core/graph_node_factory.h"
#include "properties.h"
#include "../core/node_metadata_registry.h"
#include "../plugin/registries/plugin_node_registry.h"
#include "../core/data_registry.h"
#include <imgui.h>
#include <imnodes.h>
#include <spdlog/spdlog.h>
#include <algorithm>

namespace gui {

NodeCategory NodeEditor::GetCategoryForNodeType(NodeType type) {
    return GetNodeCategoryForType(type);
}

#ifndef CYXWIZ_NODE_FACTORY_ONLY
void NodeEditor::AddNode(NodeType type, const std::string& name) {
    if (!CanAddNodeToGraph(type)) {
        spdlog::warn("Blocked graph add for unsupported node '{}' (type={})",
                     name, static_cast<int>(type));
        return;
    }

    // Queue the node for deferred addition (after ImNodes::EndNodeEditor())
    pending_nodes_.push_back({type, name, context_menu_pos_});
    ClearValidationState();  // Graph changed — stale compile results
    spdlog::info("Queued node for addition: type={}, name={} at position x={} y={}",
                 static_cast<int>(type), name, context_menu_pos_.x, context_menu_pos_.y);
}

bool NodeEditor::CanAddNodeToGraph(NodeType type) const {
    auto& registry = cyxwiz::NodeMetadataRegistry::Instance();
    registry.Initialize();
    const auto* metadata = registry.GetMetadata(type);

    // Plugin nodes may be registered dynamically outside the built-in
    // metadata catalog. Preserve that extension path when no static contract
    // exists; catalogued nodes must obey central support truth.
    return metadata == nullptr || cyxwiz::CanAddNodeToGraph(*metadata);
}
#endif

MLNode NodeEditor::CreateNodeWithIds(NodeType type,
                                     const std::string& name,
                                     int& next_node_id,
                                     int& next_pin_id) {
    return CreateGraphNode(type, name, next_node_id, next_pin_id);
}

MLNode NodeEditor::CreateNode(NodeType type, const std::string& name) {
    return CreateNodeWithIds(
        type, name, next_node_id_, next_pin_id_);
}

#ifndef CYXWIZ_NODE_FACTORY_ONLY
// Helper: if this node owns a dataset (DataInput / DatasetInput with a
// non-empty dataset_name parameter), drop it from every registry so the
// next graph rebuild doesn't see a stale entry under the same name.
// Defined locally because it's only used by DeleteNode and ClearGraph.
static void UnregisterNodeDatasetIfOwned(const MLNode& node) {
    if (node.type != NodeType::DataInput &&
        node.type != NodeType::DatasetInput) {
        return;
    }
    auto it = node.parameters.find("dataset_name");
    if (it == node.parameters.end() || it->second.empty()) {
        return;
    }
    auto& reg = cyxwiz::DataRegistry::Instance();
    // Try every registry the node could have populated. Each is a no-op
    // for names that don't exist in that map, so it's safe to call them
    // all without knowing which type the node actually loaded.
    reg.UnregisterTabularDataset(it->second);
    reg.UnregisterImageDataset(it->second);
    reg.UnregisterAudioDataset(it->second);
    reg.UnregisterTextDataset(it->second);
}

void NodeEditor::DeleteNode(int node_id) {
    if (IsSubgraphMember(node_id)) {
        spdlog::warn("Cannot delete a subgraph member individually yet; delete its Subgraph container instead.");
        return;
    }
    if (auto* data = GetSubgraphData(node_id)) {
        const auto members = data->internal_nodes;
        const bool expanded = data->expanded;
        std::erase_if(subgraphs_, [node_id](const SubgraphData& entry) { return entry.subgraph_node_id == node_id; });
        for (const auto& member : members) {
            if (expanded) DeleteNode(member.id);
            else UnregisterNodeDatasetIfOwned(member);
        }
    }
    ClearValidationState();  // Graph changed — stale compile results

    // Delete node
    auto node_it = std::find_if(nodes_.begin(), nodes_.end(),
        [node_id](const MLNode& node) {
            return node.id == node_id;
        });

    if (node_it != nodes_.end()) {
        spdlog::info("Deleting node: {} (ID: {})", node_it->name, node_id);

        // Properties and configuration dialogs hold raw pointers into nodes_.
        // Erasing a vector element can invalidate pointers to this node and to
        // any elements shifted after it, so release every such reference first.
        if (properties_panel_) {
            properties_panel_->ClearNodeReferences();
        }

        // If this is a data input node, drop its registered dataset before
        // we erase the node — otherwise the registry entry leaks until app
        // exit (the next graph won't have a node referencing it).
        UnregisterNodeDatasetIfOwned(*node_it);

        // Delete all links connected to this node
        links_.erase(
            std::remove_if(links_.begin(), links_.end(),
                [node_id](const NodeLink& link) {
                    return link.from_node == node_id || link.to_node == node_id;
                }),
            links_.end());

        nodes_.erase(node_it);
        RebuildPinLookup();  // Rebuild pin lookup after deleting node
    }
}

void NodeEditor::ClearGraph() {
    if (!CanReplaceGraph("clear the graph")) return;
    SaveUndoState();
    ClearValidationState();  // Graph changed — stale compile results

    // IMPORTANT: Clear every properties/dialog node reference BEFORE clearing
    // nodes. Both the panel and active configuration dialog hold raw pointers.
    if (properties_panel_) {
        properties_panel_->ClearNodeReferences();
    }

    // Drop every data input node's registered dataset before we lose the
    // references in nodes_. Without this, "Clear All" leaves the datasets
    // orphaned in the registry.
    for (const auto& node : nodes_) {
        UnregisterNodeDatasetIfOwned(node);
    }

    for (const auto& data : subgraphs_) {
        if (!data.expanded) for (const auto& node : data.internal_nodes) UnregisterNodeDatasetIfOwned(node);
    }
    subgraphs_.clear();
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

    // CyxWiz Studio: Clear groups and annotations
    groups_.clear();
    next_group_id_ = 1;
    annotations_.clear();
    next_annotation_id_ = 1;

    // Clear pin lookup
    pin_lookup_.clear();

    spdlog::info("Cleared node graph");
}

void NodeEditor::InsertPattern(const std::vector<MLNode>& nodes, const std::vector<NodeLink>& links) {
    if (!CanReplaceGraph("insert a pattern")) return;
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

    // Set frame counter to apply positions during next few render frames
    // This is required because ImNodes needs nodes to exist before SetNodeGridSpacePos works
    if (!pending_positions_.empty()) {
        pending_positions_frames_ = 3;  // Apply for 3 frames to ensure positions stick
    }

    // Rebuild pin lookup after inserting pattern
    RebuildPinLookup();

    spdlog::info("Inserted pattern with {} nodes and {} links (positions queued: {})",
                 nodes.size(), links.size(), pending_positions_.size());
}

// ===== Undo/Redo System =====

const MLNode* NodeEditor::FindNodeById(int node_id) const {
    for (const auto& node : nodes_) {
        if (node.id == node_id) {
            return &node;
        }
    }
    return nullptr;
}

MLNode* NodeEditor::FindNodeById(int node_id) {
    for (auto& node : nodes_) {
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
        case NodeType::TensorBroadcastTo:
        case NodeType::TensorExpand:
        case NodeType::TensorIndexSelect:
            return IM_COL32(26, 188, 156, 255);

        // ===== Merge Operations - Lime Green =====
        case NodeType::Concatenate:
        case NodeType::Add:
        case NodeType::Multiply:
        case NodeType::Average:
            return IM_COL32(139, 195, 74, 255);

        // ===== Tensor Reductions - Purple =====
        case NodeType::TensorSum:
        case NodeType::TensorMean:
        case NodeType::TensorMax:
        case NodeType::TensorMin:
        case NodeType::TensorProd:
        case NodeType::TensorVar:
        case NodeType::TensorStd:
        case NodeType::TensorPow:
        case NodeType::TensorSqrt:
        case NodeType::TensorExp:
        case NodeType::TensorLog:
        case NodeType::TensorAbs:
        case NodeType::TensorSign:
        case NodeType::TensorClip:
        case NodeType::TensorDot:
        case NodeType::TensorBatchMatMul:
        case NodeType::TensorCompare:
        case NodeType::TensorLogicalMask:
            return IM_COL32(156, 39, 176, 255);

        // ===== Loss Functions - Dark Red =====
        case NodeType::MSELoss:
        case NodeType::CrossEntropyLoss:
        case NodeType::BCELoss:
        case NodeType::BCEWithLogits:
        case NodeType::L1Loss:
        case NodeType::SmoothL1Loss:
        case NodeType::HuberLoss:
        case NodeType::NLLLoss:
        case NodeType::FocalLoss:
        case NodeType::SoftDiceLoss:
        case NodeType::TverskyLoss:
        case NodeType::JaccardLoss:
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

        // ===== Signal / Control - Teal =====
        case NodeType::SignalSlider:
            return IM_COL32(0, 150, 136, 255);
        case NodeType::SineWave:
            return IM_COL32(38, 166, 154, 255);
        case NodeType::StepSignal:
            return IM_COL32(77, 182, 172, 255);
        case NodeType::RampSignal:
            return IM_COL32(77, 182, 172, 255);
        case NodeType::SignalScope:
            return IM_COL32(0, 121, 107, 255);

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

        // ===== Text Processing - Teal =====
        case NodeType::TextTokenizer:
            return IM_COL32(0, 150, 136, 255);
        case NodeType::TextVocabulary:
            return IM_COL32(38, 166, 154, 255);
        case NodeType::TextPadding:
            return IM_COL32(77, 182, 172, 255);
        case NodeType::NERSequenceBuilder:
            return IM_COL32(0, 137, 123, 255);
        case NodeType::TokenVocabulary:
        case NodeType::POSVocabulary:
        case NodeType::NERTagVocabulary:
        case NodeType::SequenceTagOutput:
            return IM_COL32(0, 150, 136, 255);

        // ===== Upsampling - Indigo =====
        case NodeType::ConvTranspose2D:
            return IM_COL32(92, 107, 192, 255);
        case NodeType::Upsample:
            return IM_COL32(121, 134, 203, 255);
        case NodeType::PixelShuffle:
            return IM_COL32(159, 168, 218, 255);

        // ===== Time-Series - Amber =====
        case NodeType::TimeSeriesSegment:
            return IM_COL32(255, 145, 0, 255);
        case NodeType::TimeSeriesWindow:
            return IM_COL32(255, 160, 0, 255);
        case NodeType::TimeSeriesFeatures:
            return IM_COL32(255, 179, 0, 255);
        case NodeType::TimeSeriesSplit:
            return IM_COL32(255, 196, 0, 255);
        case NodeType::SeasonalNaive:
            return IM_COL32(255, 183, 32, 255);
        case NodeType::LogTransform:
            return IM_COL32(255, 213, 0, 255);
        case NodeType::Differencing:
            return IM_COL32(255, 230, 0, 255);
        case NodeType::TimeSeriesDecomposition:
        case NodeType::ACFNode:
        case NodeType::PACFNode:
        case NodeType::StationarityTest:
        case NodeType::SeasonalityDetector:
        case NodeType::ARIMAForecaster:
        case NodeType::ExponentialSmoothing:
            return IM_COL32(255, 179, 64, 255);

        // ===== Audio - Deep Purple =====
        case NodeType::AudioInput:
            return IM_COL32(103, 58, 183, 255);
        case NodeType::Spectrogram:
        case NodeType::MelSpectrogram:
            return IM_COL32(126, 87, 194, 255);
        case NodeType::MFCC:
            return IM_COL32(149, 117, 205, 255);
        case NodeType::AudioAugmentation:
            return IM_COL32(179, 157, 219, 255);

        // ===== RL - Red =====
        case NodeType::GymEnvironment:
            return IM_COL32(229, 57, 53, 255);
        case NodeType::ReplayBufferNode:
            return IM_COL32(239, 83, 80, 255);
        case NodeType::PolicyNetwork:
        case NodeType::ValueNetwork:
            return IM_COL32(244, 113, 108, 255);
        case NodeType::RLTraining:
            return IM_COL32(198, 40, 40, 255);

        // ===== DNN Inference Nodes - Deep Blue =====
        case NodeType::DNNModelLoad:
        case NodeType::DNNDetect:
        case NodeType::DNNClassify:
        case NodeType::DNNPoseEstimate:
        case NodeType::DNNFaceDetect:
        case NodeType::DNNPreprocess:
        case NodeType::PretrainedYOLO:
        case NodeType::PretrainedMobileNet:
        case NodeType::PretrainedOpenPose:
        case NodeType::PretrainedFaceNet:
            return IM_COL32(41, 98, 255, 255);

        // ===== Post-processing Nodes - Steel Blue =====
        case NodeType::NonMaxSuppression:
        case NodeType::ArgMax:
        case NodeType::TopK:
        case NodeType::ThresholdFilter:
            return IM_COL32(70, 130, 180, 255);

        // ===== Smart I/O Nodes - Bright Blue =====
        case NodeType::DataInput:
        case NodeType::DataOutput:
        case NodeType::DataConvert:
            return IM_COL32(30, 136, 229, 255);  // Material Blue 600

        // ===== Legacy Data Source Nodes - Light Blue =====
        case NodeType::CSVFile:
        case NodeType::SQLQuery:
        case NodeType::HDF5Dataset:
        case NodeType::ParquetFile:
        case NodeType::JSONFile:
        case NodeType::ExcelFile:
        case NodeType::RESTAPISource:
            return IM_COL32(100, 181, 246, 255);

        // ===== Data Transform Nodes - Teal =====
        case NodeType::FilterRows:
        case NodeType::SelectColumns:
        case NodeType::JoinTables:
        case NodeType::GroupByAggregate:
        case NodeType::SortRows:
        case NodeType::FillMissingValues:
        case NodeType::RemoveDuplicateRows:
        case NodeType::PivotTable:
        case NodeType::UnionTables:
        case NodeType::RenameColumns:
            return IM_COL32(38, 166, 154, 255);

        // ===== Analytics Nodes - Purple =====
        case NodeType::DescribeStats:
        case NodeType::VisualizeData:
        case NodeType::SampleRows:
        case NodeType::CorrelationMatrix:
        case NodeType::ValueCounts:
        case NodeType::CrossTabulation:
            return IM_COL32(156, 39, 176, 255);

        // ===== Data Export Nodes - Green =====
        case NodeType::ExportCSV:
        case NodeType::ExportParquet:
        case NodeType::ExportSQL:
        case NodeType::ExportJSON:
        case NodeType::ExportExcel:
            return IM_COL32(76, 175, 80, 255);

        // ===== KNIME-Style Table Nodes - Orange =====
        case NodeType::RowToColumnNames:
        case NodeType::TableSplitter:
        case NodeType::CellExtractor:
        case NodeType::CellUpdater:
        case NodeType::TableCropper:
        case NodeType::ColumnAppender:
        case NodeType::RowAppender:
        case NodeType::Unpivot:
        case NodeType::StringManipulation:
        case NodeType::MathFormula:
        case NodeType::RuleEngine:
            return IM_COL32(255, 152, 0, 255);

        // ===== ML Clustering Nodes - Indigo =====
        case NodeType::KMeansCluster:
        case NodeType::DBSCANCluster:
        case NodeType::HierarchicalCluster:
        case NodeType::GMMCluster:
            return IM_COL32(63, 81, 181, 255);

        // ===== Dimensionality Reduction - Blue =====
        case NodeType::PCANode:
        case NodeType::TSNENode:
        case NodeType::UMAPNode:
            return IM_COL32(33, 150, 243, 255);

        // ===== ML Classification Nodes - Blue Green =====
        case NodeType::DecisionTreeClassifier:
        case NodeType::RandomForestClassifier:
        case NodeType::GradientBoostingClassifier:
        case NodeType::TreeModelPredictor:
        case NodeType::SVMClassifier:
        case NodeType::KNNClassifier:
        case NodeType::NaiveBayesClassifier:
        case NodeType::LogisticRegressionNode:
            return IM_COL32(0, 137, 123, 255);

        // ===== ML Regression Nodes - Light Teal =====
        case NodeType::LinearRegressionNode:
        case NodeType::PolynomialRegressionNode:
        case NodeType::RegressionModelPredictor:
        case NodeType::SVMRegressor:
            return IM_COL32(77, 182, 172, 255);

        // ===== Model Evaluation Nodes - Pink =====
        case NodeType::ConfusionMatrixNode:
        case NodeType::ROCCurveNode:
        case NodeType::PRCurveNode:
        case NodeType::LearningCurvesNode:
        case NodeType::FeatureImportanceNode:
        case NodeType::CrossValidationNode:
        case NodeType::RegressionMetricsNode:
        case NodeType::ClassificationMetricsNode:
            return IM_COL32(233, 30, 99, 255);

        // ===== Data Preprocessing Nodes - Light Green =====
        case NodeType::StandardScaler:
        case NodeType::MinMaxScaler:
        case NodeType::RobustScaler:
        case NodeType::LabelEncoder:
        case NodeType::OrdinalEncoder:
        case NodeType::TargetEncoder:
            return IM_COL32(129, 199, 132, 255);

        // ===== Advanced Preprocessing Nodes - Amber =====
        case NodeType::OutlierDetector:
        case NodeType::ImagePreprocessor:
        case NodeType::QualityAnalyzer:
        case NodeType::DataValidator:
            return IM_COL32(255, 193, 7, 255);

        // ===== Dataset Source Nodes - Cyan =====
        case NodeType::ImageFolderDataset:
        case NodeType::MNISTDataset:
        case NodeType::CIFAR10Dataset:
        case NodeType::HuggingFaceDataset:
        case NodeType::KaggleDataset:
            return IM_COL32(0, 188, 212, 255);

        // ===== Advanced Augmentation Nodes - Light Purple =====
        case NodeType::AugmentationPreset:
        case NodeType::GeometricTransform:
        case NodeType::ColorTransform:
        case NodeType::MorphologyTransform:
        case NodeType::AdvancedAugment:
            return IM_COL32(179, 136, 255, 255);

        // ===== Signal Processing Nodes - Deep Blue =====
        case NodeType::FFTNode:
        case NodeType::IFFTNode:
        case NodeType::FilterDesigner:
        case NodeType::Convolution1D:
        case NodeType::WaveletTransform:
            return IM_COL32(48, 63, 159, 255);

        // ===== Text Analytics Nodes - Dark Teal =====
        case NodeType::TFIDFVectorizer:
        case NodeType::CountVectorizer:
        case NodeType::WordEmbeddings:
        case NodeType::SentimentAnalyzer:
        case NodeType::NamedEntityRecognizer:
            return IM_COL32(0, 121, 107, 255);

        // ===== Utility Nodes - Blue Gray =====
        case NodeType::CalculatorNode:
        case NodeType::UnitConverter:
        case NodeType::RegexTester:
        case NodeType::JSONPathExtractor:
        case NodeType::DataProfiler:
            return IM_COL32(96, 125, 139, 255);

        // ===== Composite Nodes - Dark Cyan =====
        case NodeType::Subgraph:
            return IM_COL32(0, 131, 143, 255);

        case NodeType::PluginCustom:
            return IM_COL32(68, 136, 170, 255);

        default:
            return IM_COL32(127, 140, 141, 255);
    }
}
#endif

} // namespace gui
