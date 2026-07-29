// File I/O Module for Node Editor
// This module contains all file operations for the visual node editor:
// - Graph save/load functionality (JSON serialization)
// - Cross-platform native file dialogs
// - Code export functionality (PyTorch, TensorFlow, Keras, PyCyxWiz)

#include "node_editor.h"
#include "node_import_guardrails.h"
#include "../core/file_dialogs.h"
#include "../core/project_manager.h"
#include <nlohmann/json.hpp>
#include <fstream>
#include <filesystem>
#include <unordered_map>
#include <spdlog/spdlog.h>

namespace gui {

// ========== Pattern to Graph Conversion ==========

// Helper: Convert string type name to NodeType enum
static NodeType StringToNodeType(const std::string& type_str) {
    static const std::unordered_map<std::string, NodeType> type_map = {
        // Data pipeline - Smart I/O nodes
        {"DataInput", NodeType::DataInput},           // Universal source with one Dataset output
        {"DataOutput", NodeType::DataOutput},         // Smart universal data output
        {"DeployToNodeEditorNode", NodeType::DeployToNodeEditorNode},
        // Legacy data input nodes
        {"Input", NodeType::DatasetInput},            // Pattern uses "Input" for legacy data input
        {"DatasetInput", NodeType::DatasetInput},     // Legacy single-output data input
        {"Output", NodeType::Output},

        // Core layers
        {"Dense", NodeType::Dense},
        {"Conv1D", NodeType::Conv1D},
        {"Conv2D", NodeType::Conv2D},
        {"Conv3D", NodeType::Conv3D},
        {"DepthwiseConv2D", NodeType::DepthwiseConv2D},

        // Pooling
        {"MaxPool2D", NodeType::MaxPool2D},
        {"AvgPool2D", NodeType::AvgPool2D},
        {"GlobalMaxPool", NodeType::GlobalMaxPool},
        {"GlobalAvgPool", NodeType::GlobalAvgPool},
        {"GlobalAvgPool2D", NodeType::GlobalAvgPool},  // Alias
        {"AdaptiveAvgPool", NodeType::AdaptiveAvgPool},

        // Normalization
        {"BatchNorm", NodeType::BatchNorm},
        {"BatchNorm2D", NodeType::BatchNorm},  // Alias
        {"LayerNorm", NodeType::LayerNorm},
        {"GroupNorm", NodeType::GroupNorm},
        {"InstanceNorm", NodeType::InstanceNorm},

        // Regularization
        {"Dropout", NodeType::Dropout},
        {"Flatten", NodeType::Flatten},

        // Recurrent
        {"RNN", NodeType::RNN},
        {"LSTM", NodeType::LSTM},
        {"GRU", NodeType::GRU},
        {"Bidirectional", NodeType::Bidirectional},
        {"Embedding", NodeType::Embedding},

        // Attention & Transformer
        {"MultiHeadAttention", NodeType::MultiHeadAttention},
        {"SelfAttention", NodeType::SelfAttention},
        {"CrossAttention", NodeType::CrossAttention},
        {"LinearAttention", NodeType::LinearAttention},
        {"TransformerEncoder", NodeType::TransformerEncoder},
        {"TransformerDecoder", NodeType::TransformerDecoder},
        {"PositionalEncoding", NodeType::PositionalEncoding},

        // Activations
        {"ReLU", NodeType::ReLU},
        {"LeakyReLU", NodeType::LeakyReLU},
        {"PReLU", NodeType::PReLU},
        {"ELU", NodeType::ELU},
        {"SELU", NodeType::SELU},
        {"GELU", NodeType::GELU},
        {"Swish", NodeType::Swish},
        {"Mish", NodeType::Mish},
        {"Sigmoid", NodeType::Sigmoid},
        {"Tanh", NodeType::Tanh},
        {"Softmax", NodeType::Softmax},

        // Shape operations
        {"Reshape", NodeType::Reshape},
        {"Permute", NodeType::Permute},
        {"Squeeze", NodeType::Squeeze},
        {"Unsqueeze", NodeType::Unsqueeze},
        {"View", NodeType::View},
        {"Split", NodeType::Split},

        // Merge operations
        {"Concatenate", NodeType::Concatenate},
        {"Concat", NodeType::Concatenate},  // Alias
        {"Add", NodeType::Add},
        {"Multiply", NodeType::Multiply},
        {"Average", NodeType::Average},

        // Tensor reductions
        {"TensorSum", NodeType::TensorSum},
        {"TensorMean", NodeType::TensorMean},
        {"TensorMax", NodeType::TensorMax},
        {"TensorMin", NodeType::TensorMin},
        {"TensorProd", NodeType::TensorProd},
        {"TensorVar", NodeType::TensorVar},
        {"TensorStd", NodeType::TensorStd},
        {"TensorBroadcastTo", NodeType::TensorBroadcastTo},
        {"TensorExpand", NodeType::TensorExpand},
        {"TensorPow", NodeType::TensorPow},
        {"TensorSqrt", NodeType::TensorSqrt},
        {"TensorExp", NodeType::TensorExp},
        {"TensorLog", NodeType::TensorLog},
        {"TensorAbs", NodeType::TensorAbs},
        {"TensorSign", NodeType::TensorSign},
        {"TensorClip", NodeType::TensorClip},
        {"TensorDot", NodeType::TensorDot},
        {"TensorBatchMatMul", NodeType::TensorBatchMatMul},
        {"TensorCompare", NodeType::TensorCompare},
        {"TensorLogicalMask", NodeType::TensorLogicalMask},
        {"TensorIndexSelect", NodeType::TensorIndexSelect},

        // Loss functions
        {"MSELoss", NodeType::MSELoss},
        {"CrossEntropyLoss", NodeType::CrossEntropyLoss},
        {"CrossEntropy", NodeType::CrossEntropyLoss},  // Alias
        {"BCELoss", NodeType::BCELoss},
        {"BCEWithLogits", NodeType::BCEWithLogits},
        {"L1Loss", NodeType::L1Loss},
        {"SmoothL1Loss", NodeType::SmoothL1Loss},
        {"HuberLoss", NodeType::HuberLoss},
        {"NLLLoss", NodeType::NLLLoss},
        {"FocalLoss", NodeType::FocalLoss},
        {"Focal", NodeType::FocalLoss},
        {"SoftDiceLoss", NodeType::SoftDiceLoss},
        {"SoftDice", NodeType::SoftDiceLoss},
        {"DiceLoss", NodeType::SoftDiceLoss},
        {"TverskyLoss", NodeType::TverskyLoss},
        {"Tversky", NodeType::TverskyLoss},
        {"JaccardLoss", NodeType::JaccardLoss},
        {"Jaccard", NodeType::JaccardLoss},
        {"IoULoss", NodeType::JaccardLoss},
        {"IoU", NodeType::JaccardLoss},

        // Optimizers
        {"SGD", NodeType::SGD},
        {"Adam", NodeType::Adam},
        {"AdamW", NodeType::AdamW},
        {"RMSprop", NodeType::RMSprop},
        {"Adagrad", NodeType::Adagrad},
        {"NAdam", NodeType::NAdam},

        // Data Pipeline / Preprocessing
        {"Normalize", NodeType::Normalize},
        {"OneHotEncode", NodeType::OneHotEncode},
        {"DataLoader", NodeType::DataLoader},
        {"DataSplit", NodeType::DataSplit},
        {"Augmentation", NodeType::Augmentation},
        {"TensorReshape", NodeType::TensorReshape},
        {"StandardScaler", NodeType::StandardScaler},
        {"MinMaxScaler", NodeType::MinMaxScaler},
        {"RobustScaler", NodeType::RobustScaler},
        {"LabelEncoder", NodeType::LabelEncoder},
        {"OrdinalEncoder", NodeType::OrdinalEncoder},
        {"TargetEncoder", NodeType::TargetEncoder},
        {"BinningNode", NodeType::BinningNode},
        {"PolynomialFeaturesNode", NodeType::PolynomialFeaturesNode},
        {"OutlierDetector", NodeType::OutlierDetector},
        {"PCANode", NodeType::PCANode},
        {"KMeansCluster", NodeType::KMeansCluster},
        {"DBSCANCluster", NodeType::DBSCANCluster},
        {"HierarchicalCluster", NodeType::HierarchicalCluster},
        {"GMMCluster", NodeType::GMMCluster},
        {"LinearRegressionNode", NodeType::LinearRegressionNode},
        {"PolynomialRegressionNode", NodeType::PolynomialRegressionNode},
        {"DecisionTreeClassifier", NodeType::DecisionTreeClassifier},
        {"RandomForestClassifier", NodeType::RandomForestClassifier},
        {"GradientBoostingClassifier", NodeType::GradientBoostingClassifier},
        {"TreeModelPredictor", NodeType::TreeModelPredictor},

        // Text Preprocessing (Phase 3) — the GraphCompiler treats these
        // as config-only nodes: ExtractTextTokenizer / ExtractTextVocabulary
        // / ExtractTextPadding read the node params into
        // TrainingConfiguration.text_preprocessing. They do NOT contribute
        // a layer to the model chain. Missing here previously → loader
        // fell through to NodeType::Dense, compile crashed on a missing
        // "units" param with "invalid stoi argument".
        {"TextCleanNode", NodeType::TextCleanNode},
        {"TextTokenizer", NodeType::TextTokenizer},
        {"TextVocabulary", NodeType::TextVocabulary},
        {"TextPadding", NodeType::TextPadding},
        {"NERSequenceBuilder", NodeType::NERSequenceBuilder},
        {"TokenVocabulary", NodeType::TokenVocabulary},
        {"POSVocabulary", NodeType::POSVocabulary},
        {"NERTagVocabulary", NodeType::NERTagVocabulary},
        {"SequenceTagOutput", NodeType::SequenceTagOutput},
        {"PairDatasetBuilder", NodeType::PairDatasetBuilder},
        {"TripletDatasetBuilder", NodeType::TripletDatasetBuilder},
        {"SharedEncoder", NodeType::SharedEncoder},
        {"SiameseBranch", NodeType::SiameseBranch},
        {"ContrastiveLoss", NodeType::ContrastiveLoss},
        {"CosineEmbeddingLoss", NodeType::CosineEmbeddingLoss},
        {"TripletLoss", NodeType::TripletLoss},
        {"PairMetrics", NodeType::PairMetrics},
        {"RetrievalMetrics", NodeType::RetrievalMetrics},
        {"EmbeddingOutput", NodeType::EmbeddingOutput},
        {"PairScoreOutput", NodeType::PairScoreOutput},
        {"TFIDFVectorizer", NodeType::TFIDFVectorizer},
        {"CountVectorizer", NodeType::CountVectorizer},
        {"SentimentAnalyzer", NodeType::SentimentAnalyzer},

        // Time Series Preprocessing (Phase 4) — missing from the loader
        // map previously caused cyxgraph loads to fall through to
        // NodeType::Dense and crash at compile time on missing `units`.
        // TimeSeriesWindow and TimeSeriesSplit are real Cat-1 pipeline
        // operators (see node_executors/time_series_*_operator.{h,cpp})
        // so dropping them in a graph triggers the materializer pass
        // before training.
        {"TimeSeriesWindow", NodeType::TimeSeriesWindow},
        {"TimeSeriesFeatures", NodeType::TimeSeriesFeatures},
        {"TimeSeriesLag", NodeType::TimeSeriesLag},
        {"TimeSeriesSplit", NodeType::TimeSeriesSplit},
        {"TimeSeriesCSV", NodeType::TimeSeriesCSV},
        {"LogTransform", NodeType::LogTransform},
        {"Differencing", NodeType::Differencing},
        // Phase 5 time-series analysis (in-sample fit only).
        {"TimeSeriesDecomposition", NodeType::TimeSeriesDecomposition},
        {"ACFNode", NodeType::ACFNode},
        {"PACFNode", NodeType::PACFNode},
        {"StationarityTest", NodeType::StationarityTest},
        {"SeasonalityDetector", NodeType::SeasonalityDetector},
        {"SeasonalNaive", NodeType::SeasonalNaive},
        {"ARIMAForecaster", NodeType::ARIMAForecaster},
        {"ExponentialSmoothing", NodeType::ExponentialSmoothing},

        // Image Transforms (Phase 1)
        {"Resize", NodeType::Resize},
        {"CenterCrop", NodeType::CenterCrop},
        {"RandomCrop", NodeType::RandomCrop},
        {"HorizontalFlip", NodeType::HorizontalFlip},
        {"VerticalFlip", NodeType::VerticalFlip},
        {"ImageRotate", NodeType::ImageRotate},
        {"ColorJitter", NodeType::ColorJitter},
        {"ImageGaussianBlur", NodeType::ImageGaussianBlur},
        {"Grayscale", NodeType::Grayscale},

        {"FFTNode", NodeType::FFTNode},
        {"FilterDesigner", NodeType::FilterDesigner},
        {"Convolution1D", NodeType::Convolution1D}
    };

    auto it = type_map.find(type_str);
    if (it != type_map.end()) {
        return it->second;
    }
    spdlog::warn("Unknown node type '{}'", type_str);
    return NodeType::Unknown;
}

static bool TryReadSerializedNodeType(const nlohmann::json& node_json,
                                      NodeType& node_type) {
    if (!node_json.contains("type") || !node_json["type"].is_number_integer()) {
        spdlog::error("Serialized node '{}' is missing an integer node type",
                      node_json.value("name", "<unnamed>"));
        return false;
    }

    const int type_value = node_json["type"].get<int>();
    if (type_value < 0 || type_value >= static_cast<int>(NodeType::Unknown)) {
        spdlog::error("Serialized node '{}' has unsupported node type id {}",
                      node_json.value("name", "<unnamed>"),
                      type_value);
        return false;
    }

    node_type = static_cast<NodeType>(type_value);
    return true;
}

static bool RejectDenseEncodedSequencePlaceholder(const MLNode& node,
                                                  const std::string& source) {
    std::string matched_marker;
    if (!detail::IsDenseEncodedSequencePlaceholder(node, matched_marker)) {
        return false;
    }

    spdlog::error("{} node '{}' is encoded as Dense but matches sequence/NER "
                  "placeholder marker '{}'; import requires a first-class "
                  "supported node type instead of erasing the original identity",
                  source,
                  node.name,
                  matched_marker);
    return true;
}

static bool HasParamValue(const std::map<std::string, std::string>& params,
                          const std::string& key) {
    auto it = params.find(key);
    return it != params.end() && !it->second.empty();
}

static void CopyLegacyParamIfMissing(std::map<std::string, std::string>& params,
                                     const std::string& canonical_key,
                                     const std::string& legacy_key,
                                     bool prefer_legacy = false) {
    if ((!prefer_legacy && HasParamValue(params, canonical_key)) ||
        !HasParamValue(params, legacy_key)) {
        return;
    }
    params[canonical_key] = params[legacy_key];
}

static std::string FirstCsvToken(const std::string& value) {
    const size_t comma = value.find(',');
    std::string token = value.substr(0, comma);
    const size_t start = token.find_first_not_of(" \t");
    if (start == std::string::npos) {
        return "";
    }
    const size_t end = token.find_last_not_of(" \t");
    return token.substr(start, end - start + 1);
}

static void CopyLegacyColumnIfMissing(std::map<std::string, std::string>& params,
                                      const std::string& canonical_key,
                                      const std::string& legacy_key,
                                      bool prefer_legacy = false) {
    if ((!prefer_legacy && HasParamValue(params, canonical_key)) ||
        !HasParamValue(params, legacy_key)) {
        return;
    }
    const std::string token = FirstCsvToken(params[legacy_key]);
    if (!token.empty()) {
        params[canonical_key] = token;
    }
}

static void MigrateLegacyNodeParameters(NodeType type,
                                        std::map<std::string, std::string>& params,
                                        bool prefer_legacy = false) {
    switch (type) {
        case NodeType::TimeSeriesWindow:
            CopyLegacyColumnIfMissing(params, "value_col", "target_column", prefer_legacy);
            CopyLegacyColumnIfMissing(params, "value_col", "column", prefer_legacy);
            CopyLegacyColumnIfMissing(params, "value_col", "columns", prefer_legacy);
            CopyLegacyParamIfMissing(params, "input_width", "window_size", prefer_legacy);
            CopyLegacyParamIfMissing(params, "shift", "forecast_horizon", prefer_legacy);
            break;

        case NodeType::TimeSeriesFeatures:
            CopyLegacyColumnIfMissing(params, "value_col", "columns", prefer_legacy);
            CopyLegacyParamIfMissing(params, "lag_values", "lag_features", prefer_legacy);
            CopyLegacyParamIfMissing(params, "lag_values", "lag_periods", prefer_legacy);
            CopyLegacyParamIfMissing(params, "rolling_windows", "rolling_window", prefer_legacy);
            CopyLegacyParamIfMissing(params, "rolling_aggregations", "rolling_features", prefer_legacy);
            break;

        case NodeType::LogTransform:
        case NodeType::Differencing:
            CopyLegacyColumnIfMissing(params, "value_col", "column", prefer_legacy);
            CopyLegacyColumnIfMissing(params, "value_col", "columns", prefer_legacy);
            CopyLegacyColumnIfMissing(params, "value_col", "target_column", prefer_legacy);
            break;

        default:
            break;
    }
}

bool NodeEditor::LoadPatternAsGraph(const nlohmann::json& j) {
    using json = nlohmann::json;

    // Clear existing graph
    ClearGraph();

    const auto& tmpl = j["template"];

    // Map string IDs to integer IDs
    std::unordered_map<std::string, int> id_map;
    int next_id = 1;

    // Load nodes from template
    if (tmpl.contains("nodes") && tmpl["nodes"].is_array()) {
        for (const auto& node_json : tmpl["nodes"]) {
            std::string str_id = node_json.value("id", "");
            std::string type_str = node_json.value("type", "Dense");
            std::string name = node_json.value("name", type_str);

            // Convert string type to NodeType enum
            NodeType node_type = StringToNodeType(type_str);
            if (node_type == NodeType::Unknown) {
                spdlog::error("Cannot load pattern node '{}' with unknown type '{}'",
                              name,
                              type_str);
                ClearGraph();
                return false;
            }
            MLNode early_node;
            early_node.type = node_type;
            early_node.name = name;
            if (RejectDenseEncodedSequencePlaceholder(early_node, "Pattern")) {
                ClearGraph();
                return false;
            }

            // Create node with proper pins
            MLNode node = CreateNode(node_type, name);
            node.id = next_id;
            id_map[str_id] = next_id;
            next_id++;

            // Parse position (two formats supported: pos_x/pos_y or pos: [x, y])
            float pos_x = node_json.value("pos_x", 0.0f);
            float pos_y = node_json.value("pos_y", 0.0f);

            if (node_json.contains("pos") && node_json["pos"].is_array() && node_json["pos"].size() >= 2) {
                pos_x = node_json["pos"][0].get<float>();
                pos_y = node_json["pos"][1].get<float>();
            }

            // Apply any node parameters (substitute pattern parameters with defaults)
            if (node_json.contains("params") && node_json["params"].is_object()) {
                for (auto& [key, value] : node_json["params"].items()) {
                    std::string param_value;
                    if (value.is_string()) {
                        param_value = value.get<std::string>();
                        // Handle pattern parameter references like "$hidden1_size"
                        if (!param_value.empty() && param_value[0] == '$') {
                            // Find the default value for this parameter
                            std::string param_name = param_value.substr(1);
                            if (j.contains("parameters") && j["parameters"].is_array()) {
                                for (const auto& p : j["parameters"]) {
                                    if (p.value("name", "") == param_name) {
                                        param_value = p.value("default_value", param_value);
                                        break;
                                    }
                                }
                            }
                        }
                    } else if (value.is_number_integer()) {
                        param_value = std::to_string(value.get<int>());
                    } else if (value.is_number_float()) {
                        param_value = std::to_string(value.get<float>());
                    } else if (value.is_boolean()) {
                        param_value = value.get<bool>() ? "true" : "false";
                    }
                    node.parameters[key] = param_value;
                }
            }
            MigrateLegacyNodeParameters(node.type, node.parameters, true);
            if (RejectDenseEncodedSequencePlaceholder(node, "Pattern")) {
                ClearGraph();
                return false;
            }

            nodes_.push_back(node);

            // Queue position
            pending_positions_[node.id] = ImVec2(pos_x, pos_y);
        }
    }

    pending_positions_frames_ = 3;

    // Load links from template
    int link_id = 1;
    if (tmpl.contains("links") && tmpl["links"].is_array()) {
        for (const auto& link_json : tmpl["links"]) {
            std::string from_str = link_json.value("from", "");
            std::string to_str = link_json.value("to", "");

            auto from_it = id_map.find(from_str);
            auto to_it = id_map.find(to_str);

            if (from_it == id_map.end() || to_it == id_map.end()) {
                spdlog::warn("Link references unknown node: {} -> {}", from_str, to_str);
                continue;
            }

            int from_node_id = from_it->second;
            int to_node_id = to_it->second;

            const MLNode* from_node = FindNodeById(from_node_id);
            const MLNode* to_node = FindNodeById(to_node_id);

            if (!from_node || !to_node) {
                spdlog::warn("Could not find nodes for link: {} -> {}", from_str, to_str);
                continue;
            }

            // Get pin indices (default to first pin)
            int from_pin_idx = link_json.value("from_pin", 0);
            int to_pin_idx = link_json.value("to_pin", 0);

            if (from_pin_idx < 0 ||
                from_pin_idx >= static_cast<int>(from_node->outputs.size())) {
                spdlog::warn(
                    "Skipping pattern link {} -> {}: source pin index {} is out of range for node '{}' ({} outputs)",
                    from_str,
                    to_str,
                    from_pin_idx,
                    from_node->name,
                    from_node->outputs.size());
                continue;
            }
            if (to_pin_idx < 0 ||
                to_pin_idx >= static_cast<int>(to_node->inputs.size())) {
                spdlog::warn(
                    "Skipping pattern link {} -> {}: target pin index {} is out of range for node '{}' ({} inputs)",
                    from_str,
                    to_str,
                    to_pin_idx,
                    to_node->name,
                    to_node->inputs.size());
                continue;
            }

            // Create link using actual pin IDs
            NodeLink link;
            link.id = link_id++;
            link.from_pin = from_node->outputs[from_pin_idx].id;
            link.to_pin = to_node->inputs[to_pin_idx].id;
            link.from_node = from_node_id;
            link.to_node = to_node_id;
            link.type = LinkType::TensorFlow;

            links_.push_back(link);
        }
    }

    // Update next IDs
    next_node_id_ = next_id;
    next_link_id_ = link_id;

    std::string name = j.value("name", "Imported Pattern");
    spdlog::info("Loaded pattern '{}' as graph ({} nodes, {} links)",
                 name, nodes_.size(), links_.size());

    return true;
}

// ========== Graph Save/Load Implementation ==========

void NodeEditor::RebuildDataBoundaryPins(
    MLNode& node,
    bool legacy_contract) {
    if (node.type != NodeType::DataInput &&
        node.type != NodeType::DataSplit &&
        node.type != NodeType::DataLoader) {
        return;
    }

    node.inputs.clear();
    node.outputs.clear();
    const auto add_pin = [&](std::vector<NodePin>& pins,
                             PinType type,
                             const char* name,
                             bool is_input,
                             bool required,
                             const char* description) {
        NodePin pin;
        pin.id = next_pin_id_++;
        pin.type = type;
        pin.name = name;
        pin.is_input = is_input;
        pin.is_required = required;
        pin.description = description;
        pins.push_back(std::move(pin));
    };

    if (legacy_contract) {
        node.parameters["data_boundary_pin_contract"] = "legacy.v1";
        node.parameters["data_boundary_migration_required"] = "true";
        if (node.type == NodeType::DataInput) {
            add_pin(node.outputs, PinType::Tensor, "Data", false, true,
                    "Preserved legacy feature output. Migrate the graph data boundary to use Dataset pins.");
            add_pin(node.outputs, PinType::Labels, "Labels", false, false,
                    "Preserved legacy label output. Migrate explicitly before saving as the Dataset contract.");
        } else if (node.type == NodeType::DataSplit) {
            add_pin(node.inputs, PinType::Tensor, "Data", true, true,
                    "Preserved legacy feature input.");
            add_pin(node.inputs, PinType::Labels, "Labels", true, true,
                    "Preserved legacy label input.");
            add_pin(node.outputs, PinType::Tensor, "Train Data", false, true,
                    "Preserved legacy training feature output.");
            add_pin(node.outputs, PinType::Labels, "Train Labels", false, false,
                    "Preserved legacy training label output.");
            add_pin(node.outputs, PinType::Tensor, "Val Data", false, false,
                    "Preserved legacy validation feature output.");
            add_pin(node.outputs, PinType::Labels, "Val Labels", false, false,
                    "Preserved legacy validation label output.");
            add_pin(node.outputs, PinType::Tensor, "Test Data", false, false,
                    "Preserved legacy held-out feature output.");
            add_pin(node.outputs, PinType::Labels, "Test Labels", false, false,
                    "Preserved legacy held-out label output.");
        } else {
            add_pin(node.inputs, PinType::Tensor, "Data", true, true,
                    "Preserved legacy unbatched feature input.");
            add_pin(node.inputs, PinType::Labels, "Labels", true, true,
                    "Preserved legacy unbatched label input.");
            add_pin(node.outputs, PinType::Tensor, "Data", false, true,
                    "Batched feature tensor.");
            add_pin(node.outputs, PinType::Labels, "Labels", false, false,
                    "Batched labels for the loss target.");
        }
        return;
    }

    node.parameters["data_boundary_pin_contract"] = "dataset.v2";
    node.parameters.erase("data_boundary_migration_required");
    if (node.type == NodeType::DataInput) {
        add_pin(node.outputs, PinType::Dataset, "Dataset", false, true,
                "Loaded Dataset asset. Connect it to a named Data Split role input.");
    } else if (node.type == NodeType::DataSplit) {
        add_pin(node.inputs, PinType::Dataset, "Training Dataset", true, true,
                "Required Training source. Missing roles are derived only from this Dataset.");
        add_pin(node.inputs, PinType::Dataset, "Validation Dataset", true, false,
                "Optional external Validation/Dev Dataset preserved in full.");
        add_pin(node.inputs, PinType::Dataset, "Test Dataset", true, false,
                "Optional external held-out Test Dataset preserved in full.");
        add_pin(node.outputs, PinType::Dataset, "Partitions", false, true,
                "Resolved Train/Validation/Test partitions and manifest.");
    } else {
        add_pin(node.inputs, PinType::Dataset, "Partitions", true, true,
                "Resolved partition contract from Data Split.");
        add_pin(node.outputs, PinType::Tensor, "Data", false, true,
                "Batched feature tensor for the model.");
        add_pin(node.outputs, PinType::Labels, "Labels", false, false,
                "Batched labels for supervised loss targets.");
    }
}

bool NodeEditor::HasLegacyDataBoundary() const {
    for (const auto& node : nodes_) {
        auto it = node.parameters.find("data_boundary_pin_contract");
        if (it != node.parameters.end() && it->second == "legacy.v1") {
            return true;
        }
    }
    return false;
}

DataBoundaryMigrationResult NodeEditor::MigrateLegacyDataBoundary() {
    DataBoundaryMigrationResult result;
    if (!HasLegacyDataBoundary()) {
        result.message = "The graph already uses the Dataset v2 boundary.";
        return result;
    }

    const auto pin_index = [](const std::vector<NodePin>& pins, int pin_id) {
        for (size_t i = 0; i < pins.size(); ++i) {
            if (pins[i].id == pin_id) return static_cast<int>(i);
        }
        return -1;
    };
    const auto is_legacy = [](const MLNode* node) {
        if (!node) return false;
        auto it = node->parameters.find("data_boundary_pin_contract");
        return it != node->parameters.end() && it->second == "legacy.v1";
    };

    std::unordered_map<int, int> split_to_loader;
    for (const auto& link : links_) {
        const auto* from = FindNodeById(link.from_node);
        const auto* to = FindNodeById(link.to_node);
        if (!from || !to || from->type != NodeType::DataSplit ||
            to->type != NodeType::DataLoader || !is_legacy(from) ||
            !is_legacy(to)) {
            continue;
        }
        if (pin_index(from->outputs, link.from_pin) == 0 &&
            pin_index(to->inputs, link.to_pin) == 0) {
            if (split_to_loader.count(from->id) > 0 &&
                split_to_loader[from->id] != to->id) {
                result.message = "Migration is blocked: legacy Data Split '" +
                    from->name + "' feeds more than one Data Loader.";
                return result;
            }
            split_to_loader[from->id] = to->id;
        }
    }

    std::unordered_map<int, int> input_to_loader;
    for (const auto& link : links_) {
        const auto* from = FindNodeById(link.from_node);
        const auto* to = FindNodeById(link.to_node);
        if (!from || !to || from->type != NodeType::DataInput ||
            to->type != NodeType::DataSplit || !is_legacy(from) ||
            !is_legacy(to)) {
            continue;
        }
        if (pin_index(from->outputs, link.from_pin) == 0 &&
            pin_index(to->inputs, link.to_pin) == 0 &&
            split_to_loader.count(to->id) > 0) {
            input_to_loader[from->id] = split_to_loader[to->id];
        }
    }

    enum class LinkAction { Keep, RemoveDuplicate, RerouteToLoaderLabels };
    struct PlannedLink {
        NodeLink link;
        int from_index = -1;
        int to_index = -1;
        LinkAction action = LinkAction::Keep;
        int reroute_loader_id = -1;
    };
    std::vector<PlannedLink> plan;
    plan.reserve(links_.size());

    for (const auto& link : links_) {
        const auto* from = FindNodeById(link.from_node);
        const auto* to = FindNodeById(link.to_node);
        if (!from || !to) {
            result.message = "Migration is blocked: a graph link references a missing node.";
            return result;
        }
        PlannedLink item;
        item.link = link;
        item.from_index = pin_index(from->outputs, link.from_pin);
        item.to_index = pin_index(to->inputs, link.to_pin);
        if (item.from_index < 0 || item.to_index < 0) {
            result.message = "Migration is blocked: link " +
                std::to_string(link.id) + " references an unknown pin.";
            return result;
        }

        if (is_legacy(from) && from->type == NodeType::DataInput &&
            item.from_index == 0 &&
            !(is_legacy(to) && to->type == NodeType::DataSplit &&
              item.to_index == 0)) {
            result.message = "Migration is blocked: legacy Data Input features from '" +
                from->name + "' bypass Data Split. Insert/preserve an explicit Split + Loader boundary before migration.";
            return result;
        }
        if (is_legacy(from) && from->type == NodeType::DataSplit &&
            item.from_index == 0 &&
            !(is_legacy(to) && to->type == NodeType::DataLoader &&
              item.to_index == 0)) {
            result.message = "Migration is blocked: legacy Training Data from Data Split '" +
                from->name + "' bypasses the Data Loader.";
            return result;
        }
        if (is_legacy(to) && to->type == NodeType::DataSplit &&
            item.to_index == 0 &&
            !(is_legacy(from) && from->type == NodeType::DataInput &&
              item.from_index == 0)) {
            result.message = "Migration is blocked: legacy Data Split '" +
                to->name + "' has a non-standard Data source.";
            return result;
        }
        if (is_legacy(to) && to->type == NodeType::DataLoader &&
            item.to_index == 0 &&
            !(is_legacy(from) && from->type == NodeType::DataSplit &&
              item.from_index == 0)) {
            result.message = "Migration is blocked: legacy Data Loader '" +
                to->name + "' does not receive Training Data from a legacy Data Split.";
            return result;
        }

        if (is_legacy(from) && from->type == NodeType::DataSplit &&
            item.from_index >= 2) {
            result.message = "Migration is blocked: legacy Data Split '" +
                from->name + "' has a connected Val/Test canvas branch. "
                "Disconnect or preserve that graph until the branch is redesigned around runtime partitions.";
            return result;
        }

        if (is_legacy(from) && from->type == NodeType::DataInput &&
            item.from_index == 1) {
            if (is_legacy(to) && to->type == NodeType::DataSplit &&
                item.to_index == 1) {
                item.action = LinkAction::RemoveDuplicate;
            } else if (input_to_loader.count(from->id) > 0) {
                item.action = LinkAction::RerouteToLoaderLabels;
                item.reroute_loader_id = input_to_loader[from->id];
            } else {
                result.message = "Migration is blocked: legacy label output from Data Input '" +
                    from->name + "' cannot be mapped to a unique Data Loader Labels output.";
                return result;
            }
        }

        if (is_legacy(from) && from->type == NodeType::DataSplit &&
            item.from_index == 1) {
            auto loader_it = split_to_loader.find(from->id);
            if (loader_it == split_to_loader.end()) {
                result.message = "Migration is blocked: legacy Data Split '" +
                    from->name + "' has no unique downstream Data Loader.";
                return result;
            }
            if (is_legacy(to) && to->type == NodeType::DataLoader &&
                to->id == loader_it->second && item.to_index == 1) {
                item.action = LinkAction::RemoveDuplicate;
            } else {
                item.action = LinkAction::RerouteToLoaderLabels;
                item.reroute_loader_id = loader_it->second;
            }
        }

        if (is_legacy(to) && to->type == NodeType::DataSplit &&
            item.to_index == 1 && item.action != LinkAction::RemoveDuplicate) {
            result.message = "Migration is blocked: legacy Data Split Labels input has a non-standard source.";
            return result;
        }
        if (is_legacy(to) && to->type == NodeType::DataLoader &&
            item.to_index == 1 && item.action != LinkAction::RemoveDuplicate) {
            result.message = "Migration is blocked: legacy Data Loader Labels input has a non-standard source.";
            return result;
        }
        plan.push_back(std::move(item));
    }

    SaveUndoState();
    for (auto& node : nodes_) {
        if (is_legacy(&node)) {
            RebuildDataBoundaryPins(node, false);
            ++result.nodes_migrated;
        }
    }

    std::vector<NodeLink> migrated_links;
    migrated_links.reserve(plan.size());
    for (auto& item : plan) {
        if (item.action == LinkAction::RemoveDuplicate) {
            ++result.links_removed;
            continue;
        }
        auto* from = FindNodeById(item.link.from_node);
        auto* to = FindNodeById(item.link.to_node);
        if (item.action == LinkAction::RerouteToLoaderLabels) {
            from = FindNodeById(item.reroute_loader_id);
            item.link.from_node = item.reroute_loader_id;
            item.from_index = 1;
            ++result.links_rerouted;
        }
        if (!from || !to || item.from_index < 0 || item.to_index < 0 ||
            item.from_index >= static_cast<int>(from->outputs.size()) ||
            item.to_index >= static_cast<int>(to->inputs.size())) {
            result.message = "Migration failed while applying the validated link plan.";
            Undo();
            return result;
        }
        item.link.from_pin = from->outputs[item.from_index].id;
        item.link.to_pin = to->inputs[item.to_index].id;
        migrated_links.push_back(item.link);
    }
    links_ = std::move(migrated_links);
    RebuildPinLookup();
    ClearValidationState();

    result.success = true;
    result.message = "Migrated " + std::to_string(result.nodes_migrated) +
        " data-boundary nodes; removed " +
        std::to_string(result.links_removed) +
        " duplicate legacy label links and rerouted " +
        std::to_string(result.links_rerouted) + " label targets.";
    spdlog::info("Track70 data-boundary migration: {}", result.message);
    return result;
}

bool NodeEditor::SaveGraph(const std::string& filepath) {
    using json = nlohmann::json;

    try {
        json j;
        // CyxWiz Studio: Update to v2.1 format with annotations
        j["version"] = "2.1";
        j["data_boundary_version"] = HasLegacyDataBoundary()
            ? detail::kLegacyDataBoundaryVersion
            : detail::kCurrentDataBoundaryVersion;
        j["framework"] = static_cast<int>(selected_framework_);
        j["execution_mode"] = static_cast<int>(execution_mode_);  // Save execution mode

        // CyxWiz Studio: Save workflow description
        j["workflow_description"] = std::string(workflow_description_);

        // CyxWiz Studio: Save canvas annotations
        json annotations_array = json::array();
        for (const auto& annotation : annotations_) {
            json ann_json;
            ann_json["id"] = annotation.id;
            ann_json["title"] = annotation.title;
            ann_json["content"] = annotation.content;
            ann_json["pos_x"] = annotation.position.x;
            ann_json["pos_y"] = annotation.position.y;
            ann_json["width"] = annotation.size.x;
            ann_json["height"] = annotation.size.y;
            ann_json["color"] = annotation.color;
            ann_json["minimized"] = annotation.is_minimized;
            annotations_array.push_back(ann_json);
        }
        j["annotations"] = annotations_array;

        // CyxWiz Studio: Save node groups
        json groups_array = json::array();
        for (const auto& group : groups_) {
            json group_json;
            group_json["id"] = group.id;
            group_json["name"] = group.name;
            group_json["description"] = group.description;
            group_json["node_ids"] = group.node_ids;
            group_json["color_r"] = group.color.x;
            group_json["color_g"] = group.color.y;
            group_json["color_b"] = group.color.z;
            group_json["color_a"] = group.color.w;
            group_json["collapsed"] = group.collapsed;
            group_json["padding"] = group.padding;
            groups_array.push_back(group_json);
        }
        j["groups"] = groups_array;

        // Serialize nodes
        json nodes_array = json::array();
        for (const auto& node : nodes_) {
            json node_json;
            node_json["id"] = node.id;
            node_json["type"] = static_cast<int>(node.type);
            node_json["name"] = node.name;
            node_json["description"] = node.description;
            node_json["parameters"] = node.parameters;

            // Unified Canvas Phase 7: Save node category for better organization
            NodeCategory category = GetCategoryForNodeType(node.type);
            node_json["category"] = static_cast<int>(category);

            // Save node position
            auto it = cached_node_positions_.find(node.id);
            ImVec2 pos = (it != cached_node_positions_.end()) ? it->second : ImVec2(0,0);
            node_json["pos_x"] = pos.x;
            node_json["pos_y"] = pos.y;

            nodes_array.push_back(node_json);
        }
        j["nodes"] = nodes_array;

        // Serialize links with pin indices for multi-pin support
        json links_array = json::array();
        for (const auto& link : links_) {
            json link_json;
            link_json["id"] = link.id;
            link_json["from_node"] = link.from_node;
            link_json["from_pin"] = link.from_pin;
            link_json["to_node"] = link.to_node;
            link_json["to_pin"] = link.to_pin;

            // Save pin indices for proper multi-pin node support
            const MLNode* from_node = FindNodeById(link.from_node);
            const MLNode* to_node = FindNodeById(link.to_node);

            int from_pin_index = 0;
            if (from_node) {
                for (size_t i = 0; i < from_node->outputs.size(); ++i) {
                    if (from_node->outputs[i].id == link.from_pin) {
                        from_pin_index = static_cast<int>(i);
                        break;
                    }
                }
            }

            int to_pin_index = 0;
            if (to_node) {
                for (size_t i = 0; i < to_node->inputs.size(); ++i) {
                    if (to_node->inputs[i].id == link.to_pin) {
                        to_pin_index = static_cast<int>(i);
                        break;
                    }
                }
            }

            link_json["from_pin_index"] = from_pin_index;
            link_json["to_pin_index"] = to_pin_index;

            // Save link type for skip connection visualization
            link_json["link_type"] = static_cast<int>(link.type);

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

static bool ResolveSavedGraphLinkPins(const nlohmann::json& link_json,
                                      const MLNode* from_node,
                                      const MLNode* to_node,
                                      NodeLink& link) {
    if (!from_node || !to_node) {
        spdlog::warn(
            "Skipping saved graph link {} ({} -> {}): referenced node is missing",
            link.id,
            link.from_node,
            link.to_node);
        return false;
    }

    int from_pin_index = 0;
    if (!detail::ResolveSerializedPinIndex(
            link_json, "from_pin_index", from_node->outputs.size(), from_pin_index)) {
        const std::string saved_index = link_json.contains("from_pin_index")
            ? link_json["from_pin_index"].dump()
            : "legacy default 0";
        spdlog::warn(
            "Skipping saved graph link {} ({} -> {}): source pin index {} is invalid "
            "for node '{}' ({} outputs)",
            link.id,
            link.from_node,
            link.to_node,
            saved_index,
            from_node->name,
            from_node->outputs.size());
        return false;
    }

    int to_pin_index = 0;
    if (!detail::ResolveSerializedPinIndex(
            link_json, "to_pin_index", to_node->inputs.size(), to_pin_index)) {
        const std::string saved_index = link_json.contains("to_pin_index")
            ? link_json["to_pin_index"].dump()
            : "legacy default 0";
        spdlog::warn(
            "Skipping saved graph link {} ({} -> {}): target pin index {} is invalid "
            "for node '{}' ({} inputs)",
            link.id,
            link.from_node,
            link.to_node,
            saved_index,
            to_node->name,
            to_node->inputs.size());
        return false;
    }

    link.from_pin = from_node->outputs[from_pin_index].id;
    link.to_pin = to_node->inputs[to_pin_index].id;
    return true;
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

        // Check if this is a pattern template format (has "template" key with nodes inside)
        if (j.contains("template") && j["template"].is_object() &&
            j["template"].contains("nodes")) {
            spdlog::info("Detected pattern template format, converting to graph format");
            return LoadPatternAsGraph(j);
        }

        // Clear existing graph
        ClearGraph();

        const bool preserve_legacy_data_boundary =
            detail::PreserveLegacyDataBoundaryPins(j);
        if (preserve_legacy_data_boundary) {
            spdlog::warn(
                "Loading an unversioned/legacy data boundary without changing its pins or links. Use the Data Split migration action to adopt Dataset v2 explicitly.");
        }

        // Unified Canvas Phase 7: Check version and load execution_mode
        std::string version = "1.0";
        if (j.contains("version")) {
            version = j["version"].get<std::string>();
        }

        // Update next IDs to avoid conflicts
        int max_node_id = 0;
        int max_link_id = 0;

        // Load framework
        if (j.contains("framework")) {
            selected_framework_ = static_cast<CodeFramework>(j["framework"].get<int>());
        }

        // Unified Canvas Phase 7: Load execution_mode (v2.0+)
        if (j.contains("execution_mode")) {
            execution_mode_ = static_cast<ExecutionMode>(j["execution_mode"].get<int>());
            spdlog::info("Loaded execution mode: {}", static_cast<int>(execution_mode_));
        } else {
            // Backward compatibility: v1.0 files default to CodeGeneration
            execution_mode_ = ExecutionMode::CodeGeneration;
            spdlog::info("Legacy v1.0 format - defaulting to CodeGeneration mode");
        }

        // CyxWiz Studio: Load workflow description (v2.1+)
        if (j.contains("workflow_description")) {
            std::string desc = j["workflow_description"].get<std::string>();
            SetWorkflowDescription(desc);
            spdlog::info("Loaded workflow description ({} chars)", desc.length());
        } else {
            SetWorkflowDescription("");
        }

        // CyxWiz Studio: Load canvas annotations (v2.1+)
        annotations_.clear();
        next_annotation_id_ = 1;
        if (j.contains("annotations") && j["annotations"].is_array()) {
            for (const auto& ann_json : j["annotations"]) {
                CanvasAnnotation annotation;
                annotation.id = ann_json.value("id", next_annotation_id_);
                annotation.title = ann_json.value("title", "");
                annotation.content = ann_json.value("content", "");
                annotation.position.x = ann_json.value("pos_x", 0.0f);
                annotation.position.y = ann_json.value("pos_y", 0.0f);
                annotation.size.x = ann_json.value("width", 200.0f);
                annotation.size.y = ann_json.value("height", 100.0f);
                annotation.color = ann_json.value("color", (ImU32)IM_COL32(255, 255, 200, 255));
                annotation.is_minimized = ann_json.value("minimized", false);

                annotations_.push_back(annotation);
                next_annotation_id_ = std::max(next_annotation_id_, annotation.id + 1);
            }
            spdlog::info("Loaded {} canvas annotations", annotations_.size());
        }

        // CyxWiz Studio: Load node groups (v2.1+)
        groups_.clear();
        next_group_id_ = 1;
        if (j.contains("groups") && j["groups"].is_array()) {
            for (const auto& group_json : j["groups"]) {
                NodeGroup group;
                group.id = group_json.value("id", next_group_id_);
                group.name = group_json.value("name", "Group");
                group.description = group_json.value("description", "");
                if (group_json.contains("node_ids") && group_json["node_ids"].is_array()) {
                    group.node_ids = group_json["node_ids"].get<std::vector<int>>();
                }
                group.color.x = group_json.value("color_r", 0.2f);
                group.color.y = group_json.value("color_g", 0.3f);
                group.color.z = group_json.value("color_b", 0.4f);
                group.color.w = group_json.value("color_a", 0.3f);
                group.collapsed = group_json.value("collapsed", false);
                group.padding = group_json.value("padding", 20.0f);

                groups_.push_back(group);
                next_group_id_ = std::max(next_group_id_, group.id + 1);
            }
            spdlog::info("Loaded {} node groups", groups_.size());
        }

        // Load nodes
        for (const auto& node_json : j["nodes"]) {
            MLNode node;
            node.id = node_json["id"];
            node.name = node_json["name"];
            if (!TryReadSerializedNodeType(node_json, node.type)) {
                ClearGraph();
                return false;
            }
            if (node_json.contains("description")) {
                node.description = node_json["description"];
            }

            if (node_json.contains("parameters")) {
                node.parameters = node_json["parameters"].get<std::map<std::string, std::string>>();
            }
            MigrateLegacyNodeParameters(node.type, node.parameters);
            if (RejectDenseEncodedSequencePlaceholder(node, "Saved graph")) {
                ClearGraph();
                return false;
            }

            // Recreate pins based on node type using fresh pin IDs
            // Create node with fresh pin IDs
            MLNode template_node = CreateNode(node.type, node.name);
            node.inputs = template_node.inputs;
            node.outputs = template_node.outputs;
            if (node.type == NodeType::DataInput ||
                node.type == NodeType::DataSplit ||
                node.type == NodeType::DataLoader) {
                RebuildDataBoundaryPins(node, preserve_legacy_data_boundary);
            }

            // Update max IDs
            max_node_id = std::max(max_node_id, node.id);

            nodes_.push_back(node);

            // Queue position restore for next render frame (must be inside ImNodes scope)
            if (node_json.contains("pos_x") && node_json.contains("pos_y")) {
                float pos_x = node_json["pos_x"];
                float pos_y = node_json["pos_y"];
                pending_positions_[node.id] = ImVec2(pos_x, pos_y);
            }
        }

        // Need to apply positions for multiple frames because ImNodes needs the node
        // to exist before SetNodeGridSpacePos takes effect
        pending_positions_frames_ = 3;  // Apply for 3 frames to ensure positions stick

        // Load links with pin index support for multi-pin nodes
        for (const auto& link_json : j["links"]) {
            NodeLink link;
            link.id = link_json["id"];
            link.from_node = link_json["from_node"];
            link.to_node = link_json["to_node"];

            // Find actual pin IDs from loaded nodes using pin indices
            const MLNode* from_node = FindNodeById(link.from_node);
            const MLNode* to_node = FindNodeById(link.to_node);

            if (!ResolveSavedGraphLinkPins(link_json, from_node, to_node, link)) {
                continue;
            }

            // Load link type for skip connection visualization
            if (link_json.contains("link_type")) {
                link.type = static_cast<LinkType>(link_json["link_type"].get<int>());
            } else {
                link.type = LinkType::TensorFlow;  // Default for legacy files
            }

            links_.push_back(link);
            max_link_id = std::max(max_link_id, link.id);
        }

        // Update next IDs
        next_node_id_ = max_node_id + 1;
        next_link_id_ = max_link_id + 1;

        current_file_path_ = filepath;

        // Rebuild pin lookup after loading graph
        RebuildPinLookup();

        spdlog::info("Graph loaded from: {} ({} nodes, {} links)",
                     filepath, nodes_.size(), links_.size());
        return true;

    } catch (const std::exception& e) {
        spdlog::error("Error loading graph: {}", e.what());
        return false;
    }
}

std::string NodeEditor::GetGraphJson() const {
    using json = nlohmann::json;

    try {
        json j;
        j["version"] = "1.0";
        j["data_boundary_version"] = HasLegacyDataBoundary()
            ? detail::kLegacyDataBoundaryVersion
            : detail::kCurrentDataBoundaryVersion;
        j["framework"] = static_cast<int>(selected_framework_);

        // Serialize nodes
        json nodes_array = json::array();
        for (const auto& node : nodes_) {
            json node_json;
            node_json["id"] = node.id;
            node_json["type"] = static_cast<int>(node.type);
            node_json["name"] = node.name;
            node_json["description"] = node.description;
            node_json["parameters"] = node.parameters;

            // Save node position
            auto it = cached_node_positions_.find(node.id);
            ImVec2 pos = (it != cached_node_positions_.end()) ? it->second : ImVec2(0,0);
            node_json["pos_x"] = pos.x;
            node_json["pos_y"] = pos.y;

            nodes_array.push_back(node_json);
        }
        j["nodes"] = nodes_array;

        // Serialize links with pin indices for multi-pin support
        json links_array = json::array();
        for (const auto& link : links_) {
            json link_json;
            link_json["id"] = link.id;
            link_json["from_node"] = link.from_node;
            link_json["from_pin"] = link.from_pin;
            link_json["to_node"] = link.to_node;
            link_json["to_pin"] = link.to_pin;

            // Save pin indices for proper multi-pin node support
            const MLNode* from_node = FindNodeById(link.from_node);
            const MLNode* to_node = FindNodeById(link.to_node);

            int from_pin_index = 0;
            if (from_node) {
                for (size_t i = 0; i < from_node->outputs.size(); ++i) {
                    if (from_node->outputs[i].id == link.from_pin) {
                        from_pin_index = static_cast<int>(i);
                        break;
                    }
                }
            }

            int to_pin_index = 0;
            if (to_node) {
                for (size_t i = 0; i < to_node->inputs.size(); ++i) {
                    if (to_node->inputs[i].id == link.to_pin) {
                        to_pin_index = static_cast<int>(i);
                        break;
                    }
                }
            }

            link_json["from_pin_index"] = from_pin_index;
            link_json["to_pin_index"] = to_pin_index;

            // Save link type for skip connection visualization
            link_json["link_type"] = static_cast<int>(link.type);

            links_array.push_back(link_json);
        }
        j["links"] = links_array;

        return j.dump(4);  // Pretty print with 4-space indent

    } catch (const std::exception& e) {
        spdlog::error("Error serializing graph: {}", e.what());
        return "";
    }
}

bool NodeEditor::LoadGraphFromString(const std::string& json_string) {
    using json = nlohmann::json;

    if (json_string.empty()) {
        spdlog::error("Cannot load graph from empty JSON string");
        return false;
    }

    try {
        json j = json::parse(json_string);

        // Clear existing graph
        ClearGraph();

        const bool preserve_legacy_data_boundary =
            detail::PreserveLegacyDataBoundaryPins(j);
        if (preserve_legacy_data_boundary) {
            spdlog::warn(
                "Loading JSON with an unversioned/legacy data boundary without changing its pins or links. Explicit migration is required for Dataset v2.");
        }

        // Update next IDs to avoid conflicts
        int max_node_id = 0;
        int max_link_id = 0;

        // Load framework
        if (j.contains("framework")) {
            selected_framework_ = static_cast<CodeFramework>(j["framework"].get<int>());
        }

        // Load nodes
        for (const auto& node_json : j["nodes"]) {
            MLNode node;
            node.id = node_json["id"];
            node.name = node_json["name"];
            if (!TryReadSerializedNodeType(node_json, node.type)) {
                ClearGraph();
                return false;
            }
            if (node_json.contains("description")) {
                node.description = node_json["description"];
            }

            if (node_json.contains("parameters")) {
                node.parameters = node_json["parameters"].get<std::map<std::string, std::string>>();
            }
            MigrateLegacyNodeParameters(node.type, node.parameters);
            if (RejectDenseEncodedSequencePlaceholder(node, "Saved graph")) {
                ClearGraph();
                return false;
            }

            // Recreate pins based on node type using fresh pin IDs
            MLNode template_node = CreateNode(node.type, node.name);
            node.inputs = template_node.inputs;
            node.outputs = template_node.outputs;
            if (node.type == NodeType::DataInput ||
                node.type == NodeType::DataSplit ||
                node.type == NodeType::DataLoader) {
                RebuildDataBoundaryPins(node, preserve_legacy_data_boundary);
            }

            // Update max IDs
            max_node_id = std::max(max_node_id, node.id);

            nodes_.push_back(node);

            // Queue position restore for next render frame
            if (node_json.contains("pos_x") && node_json.contains("pos_y")) {
                float pos_x = node_json["pos_x"];
                float pos_y = node_json["pos_y"];
                pending_positions_[node.id] = ImVec2(pos_x, pos_y);
            }
        }

        // Need to apply positions for multiple frames
        pending_positions_frames_ = 3;

        // Load links with pin index support
        for (const auto& link_json : j["links"]) {
            NodeLink link;
            link.id = link_json["id"];
            link.from_node = link_json["from_node"];
            link.to_node = link_json["to_node"];

            // Find actual pin IDs from loaded nodes using pin indices
            const MLNode* from_node = FindNodeById(link.from_node);
            const MLNode* to_node = FindNodeById(link.to_node);

            if (!ResolveSavedGraphLinkPins(link_json, from_node, to_node, link)) {
                continue;
            }

            // Load link type
            if (link_json.contains("link_type")) {
                link.type = static_cast<LinkType>(link_json["link_type"].get<int>());
            } else {
                link.type = LinkType::TensorFlow;
            }

            links_.push_back(link);
            max_link_id = std::max(max_link_id, link.id);
        }

        // Update next IDs
        next_node_id_ = max_node_id + 1;
        next_link_id_ = max_link_id + 1;

        // Rebuild pin lookup after loading graph
        RebuildPinLookup();

        spdlog::info("Graph loaded from JSON string ({} nodes, {} links)",
                     nodes_.size(), links_.size());
        return true;

    } catch (const std::exception& e) {
        spdlog::error("Error loading graph from JSON string: {}", e.what());
        return false;
    }
}

// ========== Cross-Platform File Dialogs ==========

void NodeEditor::ShowSaveDialog() {
    auto& project = cyxwiz::ProjectManager::Instance();
    const std::string default_path =
        project.HasActiveProject() ? project.GetCyxGraphsPath() : std::string();
    auto result = cyxwiz::FileDialogs::SaveGraph(
        default_path.empty() ? nullptr : default_path.c_str());
    if (result) {
        if (SaveGraph(*result)) {
            spdlog::info("Graph successfully saved");
        }
    }
}

void NodeEditor::ShowLoadDialog() {
    auto& project = cyxwiz::ProjectManager::Instance();
    const std::string default_path =
        project.HasActiveProject() ? project.GetCyxGraphsPath() : std::string();
    auto result = cyxwiz::FileDialogs::OpenGraph(
        default_path.empty() ? nullptr : default_path.c_str());
    if (result) {
        if (LoadGraph(*result)) {
            spdlog::info("Graph successfully loaded");
        }
    }
}

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

void NodeEditor::ShowExportDialog() {
    // Validate graph first
    std::string error_message;
    if (!ValidateGraph(error_message)) {
        spdlog::error("Cannot export code: {}", error_message);
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

    // Default filename based on framework
    std::string default_name = "model_" + framework_name + ".py";

    // Show cross-platform save dialog
    auto result = cyxwiz::FileDialogs::SaveScript();
    if (result) {
        std::ofstream file(*result);
        if (file.is_open()) {
            file << full_code;
            file.close();
            spdlog::info("Code exported successfully to: {}", *result);
        } else {
            spdlog::error("Failed to open file for writing: {}", *result);
        }
    }
}

} // namespace gui
