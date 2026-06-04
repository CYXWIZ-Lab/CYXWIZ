#include "model_builder.h"
#include "graph_executable_model.h"
#include <spdlog/spdlog.h>
#include <algorithm>
#include <sstream>

namespace cyxwiz {

namespace {

size_t ParseSizeParam(const CompiledLayer& layer,
                      const std::string& key,
                      size_t fallback) {
    auto it = layer.parameters.find(key);
    if (it == layer.parameters.end()) {
        return fallback;
    }
    try {
        return static_cast<size_t>(std::stoul(it->second));
    } catch (...) {
        return fallback;
    }
}

float ParseFloatParam(const CompiledLayer& layer,
                      const std::string& key,
                      float fallback) {
    auto it = layer.parameters.find(key);
    if (it == layer.parameters.end()) {
        return fallback;
    }
    try {
        return std::stof(it->second);
    } catch (...) {
        return fallback;
    }
}

bool ParseBoolParam(const CompiledLayer& layer,
                    const std::string& key,
                    bool fallback) {
    auto it = layer.parameters.find(key);
    if (it == layer.parameters.end()) {
        return fallback;
    }
    return it->second == "true" || it->second == "1";
}

std::vector<int> ParseIntListParam(const CompiledLayer& layer,
                                   const std::string& key) {
    auto it = layer.parameters.find(key);
    if (it == layer.parameters.end()) {
        return {};
    }

    std::string value = it->second;
    value.erase(std::remove(value.begin(), value.end(), '['), value.end());
    value.erase(std::remove(value.begin(), value.end(), ']'), value.end());
    value.erase(std::remove(value.begin(), value.end(), ' '), value.end());

    std::vector<int> out;
    std::stringstream ss(value);
    std::string token;
    while (std::getline(ss, token, ',')) {
        if (token.empty()) {
            continue;
        }
        try {
            out.push_back(std::stoi(token));
        } catch (...) {
        }
    }
    return out;
}

TensorMaskOp ParseTensorMaskOp(const CompiledLayer& layer) {
    auto it = layer.parameters.find("op");
    const std::string op = it != layer.parameters.end() ? it->second : ">";

    if (layer.type == gui::NodeType::TensorLogicalMask) {
        return TensorMaskOp::LogicalNot;
    }
    if (op == ">=") return TensorMaskOp::CompareGreaterEqual;
    if (op == "<") return TensorMaskOp::CompareLess;
    if (op == "<=") return TensorMaskOp::CompareLessEqual;
    if (op == "==") return TensorMaskOp::CompareEqual;
    if (op == "!=") return TensorMaskOp::CompareNotEqual;
    return TensorMaskOp::CompareGreater;
}

// Populate `model` with modules per config.layers. Returns false if
// config produced zero layers. Logging is identical to the pre-extraction
// path so existing training regression tests still match on log output.
bool BuildSequential(SequentialModel& model, const TrainingConfiguration& config) {
    spdlog::info("TrainingExecutor: Building model from {} layer configs", config.layers.size());

    // Track input size for each layer
    size_t current_input_size = config.input_size;
    size_t current_sequence_length = config.input_size > 0 ? config.input_size : 1;

    for (size_t i = 0; i < config.layers.size(); ++i) {
        const auto& layer_cfg = config.layers[i];

        switch (layer_cfg.type) {
            case gui::NodeType::Dense: {
                size_t out_features = layer_cfg.units > 0 ? layer_cfg.units : 64;
                model.Add<LinearModule>(current_input_size, out_features, true);
                spdlog::info("  [{}] Linear({} -> {})", i, current_input_size, out_features);
                current_input_size = out_features;
                break;
            }

            case gui::NodeType::Embedding: {
                // Read num_embeddings (vocab size) and embedding_dim from
                // the generic parameters map. Defaults cover the case
                // where the dialog-created node still has its factory
                // defaults (10000 / 256). Both params come from the
                // node editor's Properties panel.
                size_t num_embeddings = 10000;
                size_t embedding_dim = 256;
                auto ne_it = layer_cfg.parameters.find("num_embeddings");
                if (ne_it != layer_cfg.parameters.end()) {
                    try { num_embeddings = static_cast<size_t>(std::stoi(ne_it->second)); }
                    catch (...) {}
                }
                auto ed_it = layer_cfg.parameters.find("embedding_dim");
                if (ed_it != layer_cfg.parameters.end()) {
                    try { embedding_dim = static_cast<size_t>(std::stoi(ed_it->second)); }
                    catch (...) {}
                }

                if (num_embeddings < 2) num_embeddings = 2;
                if (embedding_dim < 1) embedding_dim = 1;

                model.Add<EmbeddingModule>(num_embeddings, embedding_dim);

                // Shape tracking: input is [batch, seq_len] with
                // current_input_size = seq_len. Embedding output is
                // [batch, seq_len, embedding_dim].
                //
                // Lookahead: if the next layer is a recurrent layer
                // (LSTM / GRU / RNN) or TransformerEncoder, keep
                // current_input_size = embedding_dim because the sequence
                // layer's
                // `input_size` is the per-timestep feature count, not
                // the flattened sequence length. Otherwise collapse
                // to seq_len * embedding_dim so the downstream
                // Flatten/Dense head gets the right feature count
                // even if the user didn't drop a Flatten node.
                const size_t seq_len = current_input_size;
                current_sequence_length = seq_len > 0 ? seq_len : 1;
                bool next_is_recurrent = false;
                if (i + 1 < config.layers.size()) {
                    const auto nt = config.layers[i + 1].type;
                    if (nt == gui::NodeType::LSTM ||
                        nt == gui::NodeType::GRU  ||
                        nt == gui::NodeType::RNN  ||
                        nt == gui::NodeType::TransformerEncoder) {
                        next_is_recurrent = true;
                    }
                }
                if (next_is_recurrent) {
                    spdlog::info("  [{}] Embedding({} x {}) — shape "
                                 "[seq_len={}] -> [seq_len={}, embed={}], "
                                 "next layer is recurrent: input_size={} "
                                 "(per-timestep features)",
                                 i, num_embeddings, embedding_dim,
                                 seq_len, seq_len, embedding_dim,
                                 embedding_dim);
                    current_input_size = embedding_dim;
                } else {
                    const size_t new_size = seq_len * embedding_dim;
                    spdlog::info("  [{}] Embedding({} x {}) — shape "
                                 "[seq_len={}] -> [seq_len={}, embed={}], "
                                 "next Flatten/Dense sees {} features",
                                 i, num_embeddings, embedding_dim,
                                 seq_len, seq_len, embedding_dim, new_size);
                    current_input_size = new_size;
                }
                break;
            }

            case gui::NodeType::LSTM: {
                size_t hidden_size = 128;
                size_t num_layers  = 1;
                bool bidirectional = false;
                bool return_sequences = false;

                auto hs_it = layer_cfg.parameters.find("hidden_size");
                if (hs_it != layer_cfg.parameters.end()) {
                    try { hidden_size = static_cast<size_t>(std::stoi(hs_it->second)); }
                    catch (...) {}
                }
                auto nl_it = layer_cfg.parameters.find("num_layers");
                if (nl_it != layer_cfg.parameters.end()) {
                    try { num_layers = static_cast<size_t>(std::stoi(nl_it->second)); }
                    catch (...) {}
                }
                auto bi_it = layer_cfg.parameters.find("bidirectional");
                if (bi_it != layer_cfg.parameters.end()) {
                    bidirectional = (bi_it->second == "true" ||
                                     bi_it->second == "1");
                }
                auto rs_it = layer_cfg.parameters.find("return_sequences");
                if (rs_it != layer_cfg.parameters.end()) {
                    return_sequences = (rs_it->second == "true" ||
                                        rs_it->second == "1");
                }
                if (hidden_size < 1) hidden_size = 1;
                if (num_layers < 1) num_layers = 1;

                model.Add<LSTMModule>(current_input_size, hidden_size,
                                      num_layers, bidirectional,
                                      return_sequences);

                const size_t output_features = hidden_size *
                                               (bidirectional ? 2 : 1);
                spdlog::info("  [{}] LSTM(in={}, hidden={}, layers={}, "
                             "bidir={}, return_seq={}) — output "
                             "[batch, {}] ({} features)",
                             i, current_input_size, hidden_size,
                             num_layers, bidirectional, return_sequences,
                             output_features, output_features);
                current_input_size = output_features;
                break;
            }

            case gui::NodeType::GRU: {
                size_t hidden_size = 128;
                size_t num_layers  = 1;
                bool bidirectional = false;
                bool return_sequences = false;

                auto hs_it = layer_cfg.parameters.find("hidden_size");
                if (hs_it != layer_cfg.parameters.end()) {
                    try { hidden_size = static_cast<size_t>(std::stoi(hs_it->second)); }
                    catch (...) {}
                }
                auto nl_it = layer_cfg.parameters.find("num_layers");
                if (nl_it != layer_cfg.parameters.end()) {
                    try { num_layers = static_cast<size_t>(std::stoi(nl_it->second)); }
                    catch (...) {}
                }
                auto bi_it = layer_cfg.parameters.find("bidirectional");
                if (bi_it != layer_cfg.parameters.end()) {
                    bidirectional = (bi_it->second == "true" ||
                                     bi_it->second == "1");
                }
                auto rs_it = layer_cfg.parameters.find("return_sequences");
                if (rs_it != layer_cfg.parameters.end()) {
                    return_sequences = (rs_it->second == "true" ||
                                        rs_it->second == "1");
                }
                if (hidden_size < 1) hidden_size = 1;
                if (num_layers < 1) num_layers = 1;

                model.Add<GRUModule>(current_input_size, hidden_size,
                                     num_layers, bidirectional,
                                     return_sequences);

                const size_t output_features = hidden_size *
                                               (bidirectional ? 2 : 1);
                spdlog::info("  [{}] GRU(in={}, hidden={}, layers={}, "
                             "bidir={}, return_seq={}) — output "
                             "[batch, {}] ({} features)",
                             i, current_input_size, hidden_size,
                             num_layers, bidirectional, return_sequences,
                             output_features, output_features);
                current_input_size = output_features;
                break;
            }

            case gui::NodeType::TransformerEncoder: {
                const size_t d_model = current_input_size > 0
                    ? current_input_size
                    : ParseSizeParam(layer_cfg, "d_model", 64);
                size_t num_heads = ParseSizeParam(layer_cfg, "num_heads",
                                                  ParseSizeParam(layer_cfg, "nhead", 1));
                size_t dim_feedforward = ParseSizeParam(
                    layer_cfg, "dim_feedforward",
                    ParseSizeParam(layer_cfg, "ff_dim", d_model * 4));
                float dropout = ParseFloatParam(layer_cfg, "dropout",
                                                ParseFloatParam(layer_cfg, "dropout_rate", 0.1f));
                bool norm_first = ParseBoolParam(layer_cfg, "norm_first", false);

                const size_t requested_d_model =
                    ParseSizeParam(layer_cfg, "d_model", d_model);
                if (requested_d_model != d_model) {
                    spdlog::warn("  [{}] TransformerEncoder d_model={} does "
                                 "not match incoming feature size {}; using {}",
                                 i, requested_d_model, d_model, d_model);
                }

                model.Add<TransformerEncoderModule>(d_model, num_heads,
                                                    dim_feedforward, dropout,
                                                    norm_first);

                bool next_is_transformer = false;
                if (i + 1 < config.layers.size()) {
                    next_is_transformer =
                        config.layers[i + 1].type == gui::NodeType::TransformerEncoder;
                }

                const size_t downstream_features = current_sequence_length * d_model;
                spdlog::info("  [{}] TransformerEncoder(d_model={}, heads={}, "
                             "ff={}, dropout={}) - output [batch, {}, {}]",
                             i, d_model, num_heads, dim_feedforward, dropout,
                             current_sequence_length, d_model);
                current_input_size = next_is_transformer
                    ? d_model
                    : downstream_features;
                break;
            }

            case gui::NodeType::ReLU: {
                model.Add<ReLUModule>();
                spdlog::info("  [{}] ReLU", i);
                break;
            }

            case gui::NodeType::Sigmoid: {
                model.Add<SigmoidModule>();
                spdlog::info("  [{}] Sigmoid", i);
                break;
            }

            case gui::NodeType::Tanh: {
                model.Add<TanhModule>();
                spdlog::info("  [{}] Tanh", i);
                break;
            }

            case gui::NodeType::LeakyReLU: {
                float slope = layer_cfg.negative_slope > 0 ? layer_cfg.negative_slope : 0.01f;
                model.Add<LeakyReLUModule>(slope);
                spdlog::info("  [{}] LeakyReLU(slope={})", i, slope);
                break;
            }

            case gui::NodeType::ELU: {
                float alpha = layer_cfg.alpha > 0 ? layer_cfg.alpha : 1.0f;
                model.Add<ELUModule>(alpha);
                spdlog::info("  [{}] ELU(alpha={})", i, alpha);
                break;
            }

            case gui::NodeType::GELU: {
                model.Add<GELUModule>();
                spdlog::info("  [{}] GELU", i);
                break;
            }

            case gui::NodeType::Swish: {
                model.Add<SwishModule>();
                spdlog::info("  [{}] Swish", i);
                break;
            }

            case gui::NodeType::Mish: {
                model.Add<MishModule>();
                spdlog::info("  [{}] Mish", i);
                break;
            }

            case gui::NodeType::Softmax: {
                model.Add<SoftmaxModule>();
                spdlog::info("  [{}] Softmax", i);
                break;
            }

            case gui::NodeType::Dropout: {
                float p = layer_cfg.dropout_rate > 0 ? layer_cfg.dropout_rate : 0.5f;
                model.Add<DropoutModule>(p);
                spdlog::info("  [{}] Dropout(p={})", i, p);
                break;
            }

            case gui::NodeType::Flatten: {
                model.Add<FlattenModule>(1);
                spdlog::info("  [{}] Flatten", i);
                break;
            }

            case gui::NodeType::Reshape:
            case gui::NodeType::View:
            case gui::NodeType::Squeeze:
            case gui::NodeType::Unsqueeze: {
                if (layer_cfg.output_shape.empty()) {
                    spdlog::error("  [{}] shape op missing resolved output_shape", i);
                    break;
                }
                model.Add<ReshapeModule>(layer_cfg.output_shape);
                spdlog::info("  [{}] ShapeOp({} dims)", i, layer_cfg.output_shape.size());
                break;
            }

            case gui::NodeType::Permute: {
                if (layer_cfg.dims.empty()) {
                    spdlog::error("  [{}] Permute missing normalized dims", i);
                    break;
                }
                model.Add<PermuteModule>(layer_cfg.dims);
                spdlog::info("  [{}] Permute({} dims)", i, layer_cfg.dims.size());
                break;
            }

            case gui::NodeType::TensorBroadcastTo: {
                if (layer_cfg.output_shape.empty()) {
                    spdlog::error("  [{}] TensorBroadcastTo missing resolved output_shape", i);
                    break;
                }
                model.Add<TensorShapeModule>(TensorShapeOp::BroadcastTo, layer_cfg.output_shape);
                spdlog::info("  [{}] TensorBroadcastTo({} dims)", i, layer_cfg.output_shape.size());
                break;
            }

            case gui::NodeType::TensorExpand: {
                if (layer_cfg.output_shape.empty()) {
                    spdlog::error("  [{}] TensorExpand missing resolved output_shape", i);
                    break;
                }
                model.Add<TensorShapeModule>(TensorShapeOp::Expand, layer_cfg.output_shape);
                spdlog::info("  [{}] TensorExpand({} dims)", i, layer_cfg.output_shape.size());
                break;
            }

            case gui::NodeType::TensorIndexSelect: {
                const int dim = static_cast<int>(ParseFloatParam(layer_cfg, "dim", 0.0f));
                const std::vector<int> indices = ParseIntListParam(layer_cfg, "indices");
                model.Add<TensorShapeModule>(TensorShapeOp::IndexSelect,
                                             std::vector<size_t>{},
                                             dim,
                                             indices);
                spdlog::info("  [{}] TensorIndexSelect(dim={}, indices={})",
                             i, dim, indices.size());
                break;
            }

            case gui::NodeType::TensorAbs: {
                model.Add<TensorUnaryModule>(TensorUnaryOp::Abs);
                spdlog::info("  [{}] TensorAbs", i);
                break;
            }

            case gui::NodeType::TensorExp: {
                model.Add<TensorUnaryModule>(TensorUnaryOp::Exp);
                spdlog::info("  [{}] TensorExp", i);
                break;
            }

            case gui::NodeType::TensorLog: {
                model.Add<TensorUnaryModule>(TensorUnaryOp::Log);
                spdlog::info("  [{}] TensorLog", i);
                break;
            }

            case gui::NodeType::TensorSqrt: {
                model.Add<TensorUnaryModule>(TensorUnaryOp::Sqrt);
                spdlog::info("  [{}] TensorSqrt", i);
                break;
            }

            case gui::NodeType::TensorSign: {
                model.Add<TensorUnaryModule>(TensorUnaryOp::Sign);
                spdlog::info("  [{}] TensorSign", i);
                break;
            }

            case gui::NodeType::TensorPow: {
                const float exponent = ParseFloatParam(layer_cfg, "exponent", 2.0f);
                model.Add<TensorUnaryModule>(TensorUnaryOp::Pow, exponent);
                spdlog::info("  [{}] TensorPow(exponent={})", i, exponent);
                break;
            }

            case gui::NodeType::TensorClip: {
                const float min_val = ParseFloatParam(layer_cfg, "min", 0.0f);
                const float max_val = ParseFloatParam(layer_cfg, "max", 1.0f);
                model.Add<TensorUnaryModule>(TensorUnaryOp::Clip, min_val, max_val);
                spdlog::info("  [{}] TensorClip(min={}, max={})", i, min_val, max_val);
                break;
            }

            case gui::NodeType::TensorCompare: {
                const TensorMaskOp op = ParseTensorMaskOp(layer_cfg);
                const float scalar = ParseFloatParam(layer_cfg, "scalar", 0.0f);
                model.Add<TensorMaskModule>(op, scalar);
                spdlog::info("  [{}] TensorCompare(scalar={})", i, scalar);
                break;
            }

            case gui::NodeType::TensorLogicalMask: {
                model.Add<TensorMaskModule>(TensorMaskOp::LogicalNot);
                spdlog::info("  [{}] TensorLogicalMask(op=not)", i);
                break;
            }

            case gui::NodeType::TensorSum: {
                const int dim = static_cast<int>(ParseFloatParam(layer_cfg, "dim", -1.0f));
                const bool keepdim = ParseBoolParam(layer_cfg, "keepdim", false);
                model.Add<TensorReductionModule>(TensorReductionOp::Sum, dim, keepdim);
                spdlog::info("  [{}] TensorSum(dim={}, keepdim={})", i, dim, keepdim);
                break;
            }

            case gui::NodeType::TensorMean: {
                const int dim = static_cast<int>(ParseFloatParam(layer_cfg, "dim", -1.0f));
                const bool keepdim = ParseBoolParam(layer_cfg, "keepdim", false);
                model.Add<TensorReductionModule>(TensorReductionOp::Mean, dim, keepdim);
                spdlog::info("  [{}] TensorMean(dim={}, keepdim={})", i, dim, keepdim);
                break;
            }

            case gui::NodeType::TensorMax: {
                const int dim = static_cast<int>(ParseFloatParam(layer_cfg, "dim", -1.0f));
                const bool keepdim = ParseBoolParam(layer_cfg, "keepdim", false);
                model.Add<TensorReductionModule>(TensorReductionOp::Max, dim, keepdim);
                spdlog::info("  [{}] TensorMax(dim={}, keepdim={})", i, dim, keepdim);
                break;
            }

            case gui::NodeType::TensorMin: {
                const int dim = static_cast<int>(ParseFloatParam(layer_cfg, "dim", -1.0f));
                const bool keepdim = ParseBoolParam(layer_cfg, "keepdim", false);
                model.Add<TensorReductionModule>(TensorReductionOp::Min, dim, keepdim);
                spdlog::info("  [{}] TensorMin(dim={}, keepdim={})", i, dim, keepdim);
                break;
            }

            case gui::NodeType::TensorProd: {
                const int dim = static_cast<int>(ParseFloatParam(layer_cfg, "dim", -1.0f));
                const bool keepdim = ParseBoolParam(layer_cfg, "keepdim", false);
                model.Add<TensorReductionModule>(TensorReductionOp::Prod, dim, keepdim);
                spdlog::info("  [{}] TensorProd(dim={}, keepdim={})", i, dim, keepdim);
                break;
            }

            case gui::NodeType::TensorVar: {
                const int dim = static_cast<int>(ParseFloatParam(layer_cfg, "dim", -1.0f));
                const bool keepdim = ParseBoolParam(layer_cfg, "keepdim", false);
                model.Add<TensorReductionModule>(TensorReductionOp::Var, dim, keepdim);
                spdlog::info("  [{}] TensorVar(dim={}, keepdim={})", i, dim, keepdim);
                break;
            }

            case gui::NodeType::TensorStd: {
                const int dim = static_cast<int>(ParseFloatParam(layer_cfg, "dim", -1.0f));
                const bool keepdim = ParseBoolParam(layer_cfg, "keepdim", false);
                model.Add<TensorReductionModule>(TensorReductionOp::Std, dim, keepdim);
                spdlog::info("  [{}] TensorStd(dim={}, keepdim={})", i, dim, keepdim);
                break;
            }

            case gui::NodeType::BatchNorm: {
                // BatchNorm uses current feature size (output of previous Dense layer)
                float eps = layer_cfg.eps > 0 ? layer_cfg.eps : 1e-5f;
                float momentum = layer_cfg.momentum > 0 ? layer_cfg.momentum : 0.1f;
                model.Add<BatchNormModule>(current_input_size, eps, momentum);
                spdlog::info("  [{}] BatchNorm({})", i, current_input_size);
                break;
            }

            case gui::NodeType::Output: {
                // Output node is just a marker, not an actual layer
                // The actual output transformation is done by the preceding Dense layer
                spdlog::info("  [{}] Output (marker, no layer added)", i);
                break;
            }

            // Skip non-layer nodes (preprocessing, loss functions, optimizers)
            case gui::NodeType::DatasetInput:
            case gui::NodeType::DataLoader:
            case gui::NodeType::Augmentation:
            case gui::NodeType::DataSplit:
            case gui::NodeType::TensorReshape:
            case gui::NodeType::Normalize:
            case gui::NodeType::OneHotEncode:
            // Loss functions
            case gui::NodeType::MSELoss:
            case gui::NodeType::CrossEntropyLoss:
            case gui::NodeType::BCELoss:
            case gui::NodeType::BCEWithLogits:
            case gui::NodeType::L1Loss:
            case gui::NodeType::SmoothL1Loss:
            case gui::NodeType::HuberLoss:
            case gui::NodeType::NLLLoss:
            // Optimizers
            case gui::NodeType::SGD:
            case gui::NodeType::Adam:
            case gui::NodeType::AdamW:
                // These are not layers in the sequential model
                break;

            // CNN layers (not yet supported in SequentialModel, need CNN module wrappers)
            case gui::NodeType::Conv2D:
            case gui::NodeType::MaxPool2D:
            case gui::NodeType::AvgPool2D:
            case gui::NodeType::GlobalMaxPool:
            case gui::NodeType::GlobalAvgPool:
                spdlog::warn("  [{}] CNN layer {} not yet supported in SequentialModel",
                             i, static_cast<int>(layer_cfg.type));
                break;

            default:
                spdlog::warn("  [{}] Unknown layer type: {}", i, static_cast<int>(layer_cfg.type));
                break;
        }
    }

    if (model.Size() == 0) {
        spdlog::error("TrainingExecutor: No layers were added to the model!");
        return false;
    }

    // Print model summary
    model.Summary();

    return true;
}

std::unique_ptr<Loss> BuildLossFromConfig(const TrainingConfiguration& config) {
    switch (config.loss_type) {
        case gui::NodeType::CrossEntropyLoss:
            spdlog::info("TrainingExecutor: Using CrossEntropy loss");
            return CreateLoss(LossType::CrossEntropy);
        case gui::NodeType::MSELoss:
            spdlog::info("TrainingExecutor: Using MSE loss");
            return CreateLoss(LossType::MSE);
        case gui::NodeType::BCELoss:
            spdlog::info("TrainingExecutor: Using BCE loss");
            return CreateLoss(LossType::BinaryCrossEntropy);
        case gui::NodeType::BCEWithLogits:
            spdlog::info("TrainingExecutor: Using BCEWithLogits loss");
            return CreateLoss(LossType::BCEWithLogits);
        case gui::NodeType::L1Loss:
            spdlog::info("TrainingExecutor: Using L1 loss");
            return CreateLoss(LossType::L1);
        case gui::NodeType::SmoothL1Loss:
        case gui::NodeType::HuberLoss:
            spdlog::info("TrainingExecutor: Using SmoothL1/Huber loss");
            return CreateLoss(LossType::SmoothL1);
        case gui::NodeType::NLLLoss:
            spdlog::info("TrainingExecutor: Using NLL loss");
            return CreateLoss(LossType::NLLLoss);
        default:
            spdlog::info("TrainingExecutor: Defaulting to CrossEntropy loss");
            return CreateLoss(LossType::CrossEntropy);
    }
}

} // namespace

BuiltModel BuildSequentialFromConfig(const TrainingConfiguration& config) {
    BuiltModel out;

    out.model = std::make_unique<SequentialModel>();
    if (!BuildSequential(*out.model, config)) {
        out.model.reset();
        return out;
    }

    out.loss = BuildLossFromConfig(config);

    out.optimizer = CreateOptimizer(config.GetOptimizerType(), config.learning_rate);
    spdlog::info("TrainingExecutor: Using {} optimizer with lr={}",
                 config.GetOptimizerName(), config.learning_rate);

    return out;
}

BuiltExecutableModel BuildExecutableFromConfig(const TrainingConfiguration& config) {
    BuiltExecutableModel out;
    BuiltModel sequential = BuildSequentialFromConfig(config);
    if (!sequential.ok()) {
        return out;
    }

    out.model = std::make_unique<SequentialExecutableModel>(
        std::move(sequential.model));
    out.loss = std::move(sequential.loss);
    out.optimizer = std::move(sequential.optimizer);
    return out;
}

BuiltExecutableModel BuildGraphExecutableFromConfig(const TrainingConfiguration& config) {
    BuiltExecutableModel out;

    std::vector<int> layer_node_ids;
    layer_node_ids.reserve(config.layers.size());
    for (const auto& layer : config.layers) {
        layer_node_ids.push_back(layer.node_id);
    }

    const bool has_graph_ops = !config.graph_op_node_ids.empty();

    if (!has_graph_ops) {
        std::string reason;
        if (!GraphExecutableModel::CanRunLinearPlan(config.graph_plan,
                                                   layer_node_ids,
                                                   &reason)) {
            spdlog::warn("GraphExecutableModel: cannot build graph executable: {}",
                         reason);
            return out;
        }
    }

    BuiltModel sequential;
    if (!config.layers.empty()) {
        sequential = BuildSequentialFromConfig(config);
        if (!sequential.ok()) {
            return out;
        }
    } else {
        sequential.model = std::make_unique<SequentialModel>();
        sequential.loss = BuildLossFromConfig(config);
        sequential.optimizer = CreateOptimizer(config.GetOptimizerType(),
                                               config.learning_rate);
        spdlog::info("TrainingExecutor: Using {} optimizer with lr={}",
                     config.GetOptimizerName(), config.learning_rate);
    }

    out.model = std::make_unique<GraphExecutableModel>(
        std::move(sequential.model),
        config.graph_plan,
        std::move(layer_node_ids),
        config.graph_op_node_ids);
    out.loss = std::move(sequential.loss);
    out.optimizer = std::move(sequential.optimizer);
    return out;
}

} // namespace cyxwiz
