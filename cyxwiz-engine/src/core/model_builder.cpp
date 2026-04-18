#include "model_builder.h"
#include <spdlog/spdlog.h>

namespace cyxwiz {

namespace {

// Populate `model` with modules per config.layers. Returns false if
// config produced zero layers. Logging is identical to the pre-extraction
// path so existing training regression tests still match on log output.
bool BuildSequential(SequentialModel& model, const TrainingConfiguration& config) {
    spdlog::info("TrainingExecutor: Building model from {} layer configs", config.layers.size());

    // Track input size for each layer
    size_t current_input_size = config.input_size;

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
                // (LSTM / GRU / RNN), keep current_input_size =
                // embedding_dim because the recurrent layer's
                // `input_size` is the per-timestep feature count, not
                // the flattened sequence length. Otherwise collapse
                // to seq_len * embedding_dim so the downstream
                // Flatten/Dense head gets the right feature count
                // even if the user didn't drop a Flatten node.
                const size_t seq_len = current_input_size;
                bool next_is_recurrent = false;
                if (i + 1 < config.layers.size()) {
                    const auto nt = config.layers[i + 1].type;
                    if (nt == gui::NodeType::LSTM ||
                        nt == gui::NodeType::GRU  ||
                        nt == gui::NodeType::RNN) {
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

} // namespace cyxwiz
