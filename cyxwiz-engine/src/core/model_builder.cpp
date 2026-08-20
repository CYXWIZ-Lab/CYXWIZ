#include "model_builder.h"
#include "error_codes.h"
#include "graph_executable_model.h"
#include <spdlog/spdlog.h>
#include <algorithm>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <vector>

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

std::string ParseStringParam(const CompiledLayer& layer,
                             const std::string& key,
                             const std::string& fallback = "");

Tensor LoadEmbeddingWeightsTextFile(const std::string& path,
                                    size_t expected_rows,
                                    size_t expected_cols);

std::string TrimAscii(std::string value) {
    auto is_space = [](unsigned char c) {
        return std::isspace(c) != 0;
    };
    value.erase(value.begin(),
                std::find_if(value.begin(), value.end(),
                             [&](char c) { return !is_space(static_cast<unsigned char>(c)); }));
    value.erase(std::find_if(value.rbegin(), value.rend(),
                             [&](char c) { return !is_space(static_cast<unsigned char>(c)); }).base(),
                value.end());
    return value;
}

std::string ToLowerAscii(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char c) {
                       return static_cast<char>(std::tolower(c));
                   });
    return value;
}

const std::string* FindLossParam(const TrainingConfiguration& config,
                                 std::initializer_list<const char*> keys) {
    for (const char* key : keys) {
        auto it = config.loss_params.find(key);
        if (it != config.loss_params.end()) {
            return &it->second;
        }
    }
    return nullptr;
}

bool IsNeutralLossValue(const std::string& value) {
    const std::string lower = ToLowerAscii(TrimAscii(value));
    return lower.empty() || lower == "none" || lower == "false" ||
           lower == "0" || lower == "off";
}

std::vector<float> ParseFloatVectorLiteral(const std::string& raw,
                                           const char* context) {
    std::string value = raw;
    for (char& c : value) {
        if (c == '[' || c == ']' || c == '(' || c == ')' ||
            c == ',' || c == ';') {
            c = ' ';
        }
    }

    std::vector<float> values;
    std::istringstream in(value);
    std::string token;
    while (in >> token) {
        try {
            size_t parsed = 0;
            const float weight = std::stof(token, &parsed);
            if (parsed != token.size() || !std::isfinite(weight) ||
                weight < 0.0f) {
                throw std::runtime_error("invalid weight");
            }
            values.push_back(weight);
        } catch (...) {
            throw std::runtime_error(std::string(context) +
                                     " contains invalid float weight '" +
                                     token + "'");
        }
    }
    return values;
}

std::vector<float> ResolveCrossEntropyClassWeights(
    const TrainingConfiguration& config) {
    const std::string* class_weight =
        FindLossParam(config, {"class_weight"});
    const std::string* explicit_weights =
        FindLossParam(config, {"class_weights", "weight", "weights"});

    if (class_weight && IsNeutralLossValue(*class_weight)) {
        return {};
    }

    if (class_weight) {
        const std::string mode = ToLowerAscii(TrimAscii(*class_weight));
        if (mode == "balanced") {
            spdlog::warn(
                "TrainingExecutor: class_weight=balanced was not resolved "
                "before loss construction; "
                "using unweighted CrossEntropy loss");
            return {};
        }
        if (mode == "manual") {
            if (!explicit_weights || IsNeutralLossValue(*explicit_weights)) {
                throw std::runtime_error(
                    "CrossEntropy class_weight=manual requires class_weights");
            }
        } else if (!explicit_weights) {
            explicit_weights = class_weight;
        }
    }

    if (!explicit_weights || IsNeutralLossValue(*explicit_weights)) {
        return {};
    }

    std::vector<float> weights =
        ParseFloatVectorLiteral(*explicit_weights,
                                "CrossEntropy class_weights");
    const size_t expected_classes =
        config.output_size > 0
            ? config.output_size
            : config.preprocessing.num_classes;
    if (expected_classes > 0 && weights.size() != expected_classes) {
        throw std::runtime_error(
            "CrossEntropy class_weights size (" +
            std::to_string(weights.size()) +
            ") does not match class/output count (" +
            std::to_string(expected_classes) + ")");
    }
    return weights;
}

float ResolveBCEWithLogitsPosWeight(const TrainingConfiguration& config) {
    const std::string* value = FindLossParam(config, {"pos_weight"});
    if (!value || IsNeutralLossValue(*value)) {
        return 1.0f;
    }
    const std::string text = TrimAscii(*value);
    try {
        size_t parsed = 0;
        const float pos_weight = std::stof(text, &parsed);
        if (parsed != text.size() || !std::isfinite(pos_weight) ||
            pos_weight <= 0.0f) {
            throw std::runtime_error("invalid pos_weight");
        }
        return pos_weight;
    } catch (...) {
        throw std::runtime_error(
            "BCEWithLogits pos_weight must be a positive finite float");
    }
}

float ResolveCrossEntropyLabelSmoothing(const TrainingConfiguration& config) {
    const std::string* value = FindLossParam(config, {"label_smoothing"});
    if (!value || IsNeutralLossValue(*value)) {
        return 0.0f;
    }
    const std::string text = TrimAscii(*value);
    try {
        size_t parsed = 0;
        const float smoothing = std::stof(text, &parsed);
        if (parsed != text.size() || !std::isfinite(smoothing) ||
            smoothing < 0.0f || smoothing >= 1.0f) {
            throw std::runtime_error("invalid label_smoothing");
        }
        return smoothing;
    } catch (...) {
        throw std::runtime_error(
            "CrossEntropy label_smoothing must be a finite float in [0, 1)");
    }
}

float ResolveLossFloatParam(const TrainingConfiguration& config,
                            const char* key,
                            float fallback,
                            float min_value,
                            const char* context) {
    const std::string* value = FindLossParam(config, {key});
    if (!value || IsNeutralLossValue(*value)) {
        return fallback;
    }
    const std::string text = TrimAscii(*value);
    try {
        size_t parsed = 0;
        const float parsed_value = std::stof(text, &parsed);
        if (parsed != text.size() || !std::isfinite(parsed_value) ||
            parsed_value < min_value) {
            throw std::runtime_error("invalid float");
        }
        return parsed_value;
    } catch (...) {
        throw std::runtime_error(std::string(context) +
                                 " must be a finite float >= " +
                                 std::to_string(min_value));
    }
}

Reduction ResolveLossReduction(const TrainingConfiguration& config) {
    const std::string* value = FindLossParam(config, {"reduction"});
    if (!value) {
        return Reduction::Mean;
    }
    const std::string mode = ToLowerAscii(TrimAscii(*value));
    if (mode.empty() || mode == "false" || mode == "0" || mode == "off") {
        return Reduction::Mean;
    }
    if (mode == "mean") {
        return Reduction::Mean;
    }
    if (mode == "sum") {
        return Reduction::Sum;
    }
    if (mode == "none") {
        return Reduction::None;
    }
    throw std::runtime_error(
        "Loss reduction must be one of: mean, sum, none");
}

const char* ReductionName(Reduction reduction) {
    switch (reduction) {
        case Reduction::Mean: return "mean";
        case Reduction::Sum: return "sum";
        case Reduction::None: return "none";
    }
    return "mean";
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
bool BuildSequential(
    SequentialModel& model,
    const TrainingConfiguration& config,
    std::vector<BuiltModuleProvenance>* provenance) {
    spdlog::info("TrainingExecutor: Building model from {} layer configs", config.layers.size());

    if (provenance) {
        provenance->clear();
        provenance->reserve(config.layers.size());
    }

    // Track input size for each layer
    size_t current_input_size = config.input_size;
    size_t current_sequence_length = config.input_size > 0 ? config.input_size : 1;

    for (size_t i = 0; i < config.layers.size(); ++i) {
        const auto& layer_cfg = config.layers[i];
        const size_t module_count_before = model.Size();

        switch (layer_cfg.type) {
            case gui::NodeType::Dense: {
                size_t out_features = layer_cfg.units > 0 ? layer_cfg.units : 64;
                model.Add<LinearModule>(current_input_size, out_features, true);
                spdlog::info("  [{}] Linear({} -> {})", i, current_input_size, out_features);
                current_input_size = out_features;
                break;
            }

            case gui::NodeType::Concatenate: {
                if (!ParseBoolParam(layer_cfg, "sequence_feature_fusion", false)) {
                    spdlog::warn(
                        "  [{}] Concatenate is not supported in SequentialModel",
                        i);
                    break;
                }
                if (i != 0) {
                    throw std::runtime_error(
                        "sequence feature fusion must be the first model layer");
                }

                size_t word_num_embeddings =
                    ParseSizeParam(layer_cfg, "word_num_embeddings",
                                   ParseSizeParam(layer_cfg, "num_embeddings", 10000));
                size_t word_embedding_dim =
                    ParseSizeParam(layer_cfg, "word_embedding_dim",
                                   ParseSizeParam(layer_cfg, "embedding_dim", 256));
                size_t pos_num_embeddings =
                    ParseSizeParam(layer_cfg, "pos_num_embeddings", 128);
                size_t pos_embedding_dim =
                    ParseSizeParam(layer_cfg, "pos_embedding_dim", 32);

                if (word_num_embeddings < 2) word_num_embeddings = 2;
                if (pos_num_embeddings < 2) pos_num_embeddings = 2;
                if (word_embedding_dim < 1) word_embedding_dim = 1;
                if (pos_embedding_dim < 1) pos_embedding_dim = 1;

                int word_padding_idx = -1;
                if (auto it = layer_cfg.parameters.find("word_padding_idx");
                    it != layer_cfg.parameters.end()) {
                    try { word_padding_idx = std::stoi(it->second); }
                    catch (...) {}
                }
                int pos_padding_idx = -1;
                if (auto it = layer_cfg.parameters.find("pos_padding_idx");
                    it != layer_cfg.parameters.end()) {
                    try { pos_padding_idx = std::stoi(it->second); }
                    catch (...) {}
                }

                model.Add<SequenceFeatureFusionModule>(
                    word_num_embeddings,
                    word_embedding_dim,
                    pos_num_embeddings,
                    pos_embedding_dim,
                    word_padding_idx,
                    pos_padding_idx);
                current_sequence_length =
                    config.input_size > 0 ? config.input_size : 1;
                current_input_size = word_embedding_dim + pos_embedding_dim;
                spdlog::info(
                    "  [{}] SequenceFeatureFusion(word={}x{}, pos={}x{}) - "
                    "output [batch, seq_len={}, features={}]",
                    i,
                    word_num_embeddings,
                    word_embedding_dim,
                    pos_num_embeddings,
                    pos_embedding_dim,
                    current_sequence_length,
                    current_input_size);
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

                int padding_idx = -1;
                auto pad_it = layer_cfg.parameters.find("padding_idx");
                if (pad_it != layer_cfg.parameters.end()) {
                    try { padding_idx = std::stoi(pad_it->second); }
                    catch (...) {}
                }

                const std::string weights_file =
                    ParseStringParam(layer_cfg, "weights_file",
                        ParseStringParam(layer_cfg, "embedding_weights_file"));
                const bool freeze_embedding =
                    ParseBoolParam(layer_cfg, "freeze", false);
                const float max_norm =
                    ParseFloatParam(layer_cfg, "max_norm", 0.0f);

                if (!weights_file.empty()) {
                    auto embedding = std::make_unique<EmbeddingModule>(
                        num_embeddings, embedding_dim, padding_idx, max_norm);
                    try {
                        Tensor weights = LoadEmbeddingWeightsTextFile(
                            weights_file, num_embeddings, embedding_dim);
                        embedding->LoadPretrainedWeights(weights, freeze_embedding);
                        spdlog::info("  [{}] Loaded embedding weights from '{}'{}",
                                     i, weights_file,
                                     freeze_embedding ? " (frozen)" : "");
                    } catch (const std::exception& e) {
                        throw std::runtime_error(
                            "Embedding layer failed to load weights_file '" +
                            weights_file + "': " + e.what());
                    }
                    model.AddModule(std::move(embedding));
                } else {
                    model.Add<EmbeddingModule>(num_embeddings, embedding_dim,
                                               padding_idx, max_norm);
                }

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
                        nt == gui::NodeType::LayerNorm ||
                        nt == gui::NodeType::PositionalEncoding ||
                        nt == gui::NodeType::MultiHeadAttention ||
                        nt == gui::NodeType::TransformerEncoder ||
                        nt == gui::NodeType::TransformerDecoder ||
                        nt == gui::NodeType::TimeDistributed) {
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
                        config.layers[i + 1].type == gui::NodeType::TransformerEncoder ||
                        config.layers[i + 1].type == gui::NodeType::TransformerDecoder ||
                        config.layers[i + 1].type == gui::NodeType::MultiHeadAttention ||
                        config.layers[i + 1].type == gui::NodeType::LayerNorm ||
                        config.layers[i + 1].type == gui::NodeType::TimeDistributed;
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

            case gui::NodeType::MultiHeadAttention: {
                const size_t embed_dim = current_input_size > 0
                    ? current_input_size
                    : ParseSizeParam(layer_cfg, "embed_dim",
                                     ParseSizeParam(layer_cfg, "d_model", 64));
                size_t num_heads = ParseSizeParam(layer_cfg, "num_heads",
                                                  ParseSizeParam(layer_cfg, "heads", 1));
                float dropout = ParseFloatParam(layer_cfg, "dropout",
                                                ParseFloatParam(layer_cfg, "dropout_rate", 0.0f));
                const bool use_bias = ParseBoolParam(layer_cfg, "use_bias", true);

                const size_t requested_embed_dim =
                    ParseSizeParam(layer_cfg, "embed_dim",
                                   ParseSizeParam(layer_cfg, "d_model", embed_dim));
                if (requested_embed_dim != embed_dim) {
                    spdlog::warn("  [{}] MultiHeadAttention embed_dim={} does "
                                 "not match incoming feature size {}; using {}",
                                 i, requested_embed_dim, embed_dim, embed_dim);
                }

                model.Add<MultiHeadAttentionModule>(embed_dim, num_heads,
                                                    dropout, use_bias);

                bool next_is_sequence_layer = false;
                if (i + 1 < config.layers.size()) {
                    next_is_sequence_layer =
                        config.layers[i + 1].type == gui::NodeType::TransformerEncoder ||
                        config.layers[i + 1].type == gui::NodeType::TransformerDecoder ||
                        config.layers[i + 1].type == gui::NodeType::MultiHeadAttention ||
                        config.layers[i + 1].type == gui::NodeType::LayerNorm ||
                        config.layers[i + 1].type == gui::NodeType::TimeDistributed;
                }

                const size_t downstream_features =
                    current_sequence_length * embed_dim;
                spdlog::info("  [{}] MultiHeadAttention(embed_dim={}, heads={}, "
                             "dropout={}) - self-attention output [batch, {}, {}]",
                             i, embed_dim, num_heads, dropout,
                             current_sequence_length, embed_dim);
                current_input_size = next_is_sequence_layer
                    ? embed_dim
                    : downstream_features;
                break;
            }

            case gui::NodeType::PositionalEncoding: {
                const size_t d_model = current_input_size > 0
                    ? current_input_size
                    : ParseSizeParam(layer_cfg, "d_model", 64);
                const size_t max_sequence_length = ParseSizeParam(
                    layer_cfg, "max_sequence_length",
                    ParseSizeParam(layer_cfg, "max_length", current_sequence_length));

                model.Add<PositionalEncodingModule>(d_model, max_sequence_length);
                spdlog::info("  [{}] PositionalEncoding(d_model={}, max_len={}) - output "
                             "[batch, {}, {}]",
                             i, d_model, max_sequence_length,
                             current_sequence_length, d_model);
                current_input_size = d_model;
                break;
            }

            case gui::NodeType::TransformerDecoder: {
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
                    spdlog::warn("  [{}] TransformerDecoder d_model={} does "
                                 "not match incoming feature size {}; using {}",
                                 i, requested_d_model, d_model, d_model);
                }

                model.Add<TransformerDecoderModule>(d_model, num_heads,
                                                    dim_feedforward, dropout,
                                                    norm_first);

                bool next_is_transformer = false;
                if (i + 1 < config.layers.size()) {
                    next_is_transformer =
                        config.layers[i + 1].type == gui::NodeType::TransformerEncoder ||
                        config.layers[i + 1].type == gui::NodeType::TransformerDecoder ||
                        config.layers[i + 1].type == gui::NodeType::MultiHeadAttention ||
                        config.layers[i + 1].type == gui::NodeType::LayerNorm ||
                        config.layers[i + 1].type == gui::NodeType::TimeDistributed;
                }

                const size_t downstream_features = current_sequence_length * d_model;
                spdlog::info("  [{}] TransformerDecoder(d_model={}, heads={}, "
                             "ff={}, dropout={}) - output [batch, {}, {}]",
                             i, d_model, num_heads, dim_feedforward, dropout,
                             current_sequence_length, d_model);
                current_input_size = next_is_transformer
                    ? d_model
                    : downstream_features;
                break;
            }

            case gui::NodeType::TimeDistributed: {
                size_t out_features = layer_cfg.units > 0
                    ? layer_cfg.units
                    : ParseSizeParam(layer_cfg, "units",
                                     ParseSizeParam(layer_cfg, "out_features",
                                                    config.output_size > 0 ? config.output_size : 64));
                if (out_features < 1) {
                    out_features = 1;
                }
                model.Add<TimeDistributedDenseModule>(current_input_size,
                                                      out_features,
                                                      true);
                spdlog::info("  [{}] TimeDistributedDense({} -> {}) - output "
                             "[batch, seq_len, {}]",
                             i, current_input_size, out_features,
                             out_features);
                current_input_size = out_features;
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

            case gui::NodeType::LayerNorm: {
                std::vector<int> normalized_shape =
                    ParseIntListParam(layer_cfg, "normalized_shape");
                if (normalized_shape.empty()) {
                    normalized_shape.push_back(
                        static_cast<int>(current_input_size > 0
                                             ? current_input_size
                                             : 1));
                }
                for (int& dim : normalized_shape) {
                    if (dim <= 0) {
                        dim = static_cast<int>(current_input_size > 0
                                                   ? current_input_size
                                                   : 1);
                    }
                }
                const float eps = ParseFloatParam(
                    layer_cfg, "epsilon", ParseFloatParam(layer_cfg, "eps", 1e-5f));
                const bool elementwise_affine =
                    ParseBoolParam(layer_cfg, "elementwise_affine", true);
                model.Add<LayerNormModule>(
                    normalized_shape, eps, elementwise_affine);
                spdlog::info("  [{}] LayerNorm(normalized_shape={}, eps={}, affine={})",
                             i, normalized_shape.front(), eps,
                             elementwise_affine);
                bool next_is_sequence_layer = false;
                if (i + 1 < config.layers.size()) {
                    next_is_sequence_layer =
                        config.layers[i + 1].type == gui::NodeType::TransformerEncoder ||
                        config.layers[i + 1].type == gui::NodeType::TransformerDecoder ||
                        config.layers[i + 1].type == gui::NodeType::MultiHeadAttention ||
                        config.layers[i + 1].type == gui::NodeType::LayerNorm ||
                        config.layers[i + 1].type == gui::NodeType::TimeDistributed;
                }
                if (config.preprocessing_domain == PreprocessingDomain::Text &&
                    !next_is_sequence_layer) {
                    current_input_size = current_sequence_length * current_input_size;
                }
                break;
            }

            case gui::NodeType::Output:
            case gui::NodeType::SequenceTagOutput: {
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
            case gui::NodeType::FocalLoss:
            case gui::NodeType::BCELoss:
            case gui::NodeType::BCEWithLogits:
            case gui::NodeType::L1Loss:
            case gui::NodeType::SmoothL1Loss:
            case gui::NodeType::HuberLoss:
            case gui::NodeType::NLLLoss:
            case gui::NodeType::SoftDiceLoss:
            case gui::NodeType::TverskyLoss:
            case gui::NodeType::JaccardLoss:
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

        if (provenance) {
            const size_t module_count_after = model.Size();
            if (module_count_after == module_count_before) {
                provenance->push_back({
                    i,
                    std::nullopt,
                    layer_cfg.node_id,
                    layer_cfg.name,
                    layer_cfg.type,
                    {},
                    layer_cfg.input_shape,
                    layer_cfg.output_shape,
                    layer_cfg.parameters,
                });
            } else {
                for (size_t module_index = module_count_before;
                     module_index < module_count_after;
                     ++module_index) {
                    Module* module = model.GetModule(module_index);
                    provenance->push_back({
                        i,
                        module_index,
                        layer_cfg.node_id,
                        layer_cfg.name,
                        layer_cfg.type,
                        module ? module->GetName() : std::string{},
                        layer_cfg.input_shape,
                        layer_cfg.output_shape,
                        layer_cfg.parameters,
                    });
                }
            }
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

int ResolveCrossEntropyIgnoreIndex(const TrainingConfiguration& config) {
    auto it = config.loss_params.find("ignore_index");
    if (it != config.loss_params.end() && !it->second.empty()) {
        try {
            return std::stoi(it->second);
        } catch (...) {
            spdlog::warn("TrainingExecutor: ignoring invalid CrossEntropy "
                         "ignore_index='{}'", it->second);
        }
    }
    if (config.sequence_batch.enabled &&
        config.sequence_batch.create_causal_lm_targets) {
        return config.sequence_batch.target_ignore_index;
    }
    if (config.sequence_batch.enabled) {
        return config.sequence_batch.ignore_index;
    }
    return -100;
}

ResolvedLossConfiguration ResolveLossConfigurationImpl(
    const TrainingConfiguration& config) {
    ResolvedLossConfiguration out;
    out.loss_type = config.loss_type;
    out.loss_name = config.GetLossName();
    out.reduction = ResolveLossReduction(config);

    switch (config.loss_type) {
        case gui::NodeType::CrossEntropyLoss:
            out.ignore_index_applicable = true;
            out.ignore_index = ResolveCrossEntropyIgnoreIndex(config);
            out.class_weights_applicable = true;
            out.class_weights = ResolveCrossEntropyClassWeights(config);
            out.label_smoothing_applicable = true;
            out.label_smoothing = ResolveCrossEntropyLabelSmoothing(config);
            break;
        case gui::NodeType::FocalLoss:
            out.alpha = ResolveLossFloatParam(
                config, "alpha", 0.25f, 0.0f, "FocalLoss alpha");
            out.gamma = ResolveLossFloatParam(
                config, "gamma", 2.0f, 0.0f, "FocalLoss gamma");
            break;
        case gui::NodeType::BCEWithLogits:
            out.pos_weight = ResolveBCEWithLogitsPosWeight(config);
            break;
        case gui::NodeType::SmoothL1Loss:
        case gui::NodeType::HuberLoss:
            out.beta = ResolveLossFloatParam(
                config, "beta", 1.0f, 0.000001f,
                "SmoothL1/Huber beta");
            break;
        case gui::NodeType::NLLLoss:
            out.ignore_index_applicable = true;
            out.ignore_index = ResolveCrossEntropyIgnoreIndex(config);
            break;
        case gui::NodeType::SoftDiceLoss:
            out.smooth = ResolveLossFloatParam(
                config, "smooth", 1.0f, 0.0f, "SoftDice smooth");
            break;
        case gui::NodeType::TverskyLoss:
            out.alpha = ResolveLossFloatParam(
                config, "alpha", 0.5f, 0.0f, "Tversky alpha");
            out.beta = ResolveLossFloatParam(
                config, "beta", 0.5f, 0.0f, "Tversky beta");
            out.smooth = ResolveLossFloatParam(
                config, "smooth", 1.0f, 0.0f, "Tversky smooth");
            break;
        case gui::NodeType::JaccardLoss:
            out.smooth = ResolveLossFloatParam(
                config, "smooth", 1.0f, 0.0f, "Jaccard smooth");
            break;
        default:
            break;
    }
    return out;
}

std::unique_ptr<Loss> BuildLossFromConfigImpl(const TrainingConfiguration& config) {
    const ResolvedLossConfiguration resolved =
        ResolveLossConfigurationImpl(config);
    const Reduction reduction = resolved.reduction;
    switch (config.loss_type) {
        case gui::NodeType::CrossEntropyLoss: {
            spdlog::info("TrainingExecutor: Using CrossEntropy loss "
                         "(reduction={}, ignore_index={}, class_weights={}, "
                         "label_smoothing={})",
                         ReductionName(reduction), resolved.ignore_index,
                         resolved.class_weights.size(),
                         resolved.label_smoothing);
            return std::make_unique<CrossEntropyLoss>(
                reduction, resolved.ignore_index, resolved.class_weights,
                resolved.label_smoothing);
        }
        case gui::NodeType::FocalLoss: {
            const float alpha = resolved.alpha.value();
            const float gamma = resolved.gamma.value();
            spdlog::info("TrainingExecutor: Using Focal loss "
                         "(reduction={}, alpha={}, gamma={})",
                         ReductionName(reduction), alpha, gamma);
            return std::make_unique<FocalLoss>(
                alpha, gamma, reduction);
        }
        case gui::NodeType::MSELoss:
            spdlog::info("TrainingExecutor: Using MSE loss (reduction={})",
                         ReductionName(reduction));
            return CreateLoss(LossType::MSE, reduction);
        case gui::NodeType::BCELoss:
            spdlog::info("TrainingExecutor: Using BCE loss (reduction={})",
                         ReductionName(reduction));
            return CreateLoss(LossType::BinaryCrossEntropy, reduction);
        case gui::NodeType::BCEWithLogits: {
            const float pos_weight = resolved.pos_weight.value();
            spdlog::info("TrainingExecutor: Using BCEWithLogits loss "
                         "(reduction={}, pos_weight={})",
                         ReductionName(reduction), pos_weight);
            return std::make_unique<BCEWithLogitsLoss>(
                reduction, pos_weight);
        }
        case gui::NodeType::L1Loss:
            spdlog::info("TrainingExecutor: Using L1 loss (reduction={})",
                         ReductionName(reduction));
            return CreateLoss(LossType::L1, reduction);
        case gui::NodeType::SmoothL1Loss:
        case gui::NodeType::HuberLoss: {
            const float beta = resolved.beta.value();
            spdlog::info("TrainingExecutor: Using SmoothL1/Huber loss "
                         "(reduction={}, beta={})",
                         ReductionName(reduction), beta);
            return CreateLoss(LossType::SmoothL1, reduction, beta);
        }
        case gui::NodeType::NLLLoss: {
            spdlog::info("TrainingExecutor: Using NLL loss "
                         "(reduction={}, ignore_index={})",
                         ReductionName(reduction), resolved.ignore_index);
            return std::make_unique<NLLLoss>(
                reduction, resolved.ignore_index);
        }
        case gui::NodeType::SoftDiceLoss: {
            const float smooth = resolved.smooth.value();
            spdlog::info("TrainingExecutor: Using SoftDice loss "
                         "(reduction={}, smooth={})",
                         ReductionName(reduction), smooth);
            return std::make_unique<SoftDiceLoss>(reduction, smooth);
        }
        case gui::NodeType::TverskyLoss: {
            const float alpha = resolved.alpha.value();
            const float beta = resolved.beta.value();
            const float smooth = resolved.smooth.value();
            spdlog::info("TrainingExecutor: Using Tversky loss "
                         "(reduction={}, alpha={}, beta={}, smooth={})",
                         ReductionName(reduction), alpha, beta, smooth);
            return std::make_unique<TverskyLoss>(
                reduction, alpha, beta, smooth);
        }
        case gui::NodeType::JaccardLoss: {
            const float smooth = resolved.smooth.value();
            spdlog::info("TrainingExecutor: Using Jaccard loss "
                         "(reduction={}, smooth={})",
                         ReductionName(reduction), smooth);
            return std::make_unique<JaccardLoss>(reduction, smooth);
        }
        default:
            spdlog::info("TrainingExecutor: Defaulting to CrossEntropy loss "
                         "(reduction={})", ReductionName(reduction));
            return CreateLoss(LossType::CrossEntropy, reduction);
    }
}

std::string ParseStringParam(const CompiledLayer& layer,
                             const std::string& key,
                             const std::string& fallback) {
    auto it = layer.parameters.find(key);
    if (it == layer.parameters.end()) {
        return fallback;
    }
    return it->second;
}

Tensor LoadEmbeddingWeightsTextFile(const std::string& path,
                                    size_t expected_rows,
                                    size_t expected_cols) {
    std::ifstream in(path);
    if (!in.is_open()) {
        throw std::runtime_error("could not open embedding weights file: " + path);
    }

    std::vector<float> values;
    size_t rows = 0;
    size_t cols = 0;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::replace(line.begin(), line.end(), ',', ' ');
        std::istringstream iss(line);
        std::vector<float> row;
        float value = 0.0f;
        while (iss >> value) {
            row.push_back(value);
        }
        if (row.empty()) continue;
        if (cols == 0) {
            cols = row.size();
        } else if (row.size() != cols) {
            throw std::runtime_error("embedding weights file has inconsistent row widths");
        }
        values.insert(values.end(), row.begin(), row.end());
        ++rows;
    }

    if (rows != expected_rows || cols != expected_cols) {
        std::ostringstream msg;
        msg << "embedding weights shape mismatch: expected "
            << expected_rows << " x " << expected_cols
            << ", got " << rows << " x " << cols;
        throw std::runtime_error(msg.str());
    }

    return Tensor({rows, cols}, values.data(), DataType::Float32);
}

} // namespace

std::unique_ptr<Loss> BuildLossFromConfig(
    const TrainingConfiguration& config) {
    return BuildLossFromConfigImpl(config);
}

ResolvedLossConfiguration ResolveLossConfiguration(
    const TrainingConfiguration& config) {
    return ResolveLossConfigurationImpl(config);
}

BuiltModel BuildSequentialFromConfig(const TrainingConfiguration& config) {
    BuiltModel out;

    try {
        out.model = std::make_unique<SequentialModel>();
        if (!BuildSequential(*out.model, config, &out.module_provenance)) {
            out.error_message = errors::FormatError(
                errors::Compiler::UnsupportedTrainingNode,
                "ModelBuilder could not add any supported trainable layers");
            out.model.reset();
            return out;
        }

        out.loss = BuildLossFromConfig(config);
        out.optimizer =
            CreateOptimizer(config.GetOptimizerType(), config.learning_rate);
        if (!out.loss) {
            out.error_message = errors::FormatError(
                errors::Training::LossSetupFailed,
                "ModelBuilder failed to create loss");
            out.model.reset();
            return out;
        }
        if (!out.optimizer) {
            out.error_message = errors::FormatError(
                errors::Training::OptimizerSetupFailed,
                "ModelBuilder failed to create optimizer");
            out.model.reset();
            out.loss.reset();
            return out;
        }

        spdlog::info("TrainingExecutor: Using {} optimizer with lr={}",
                     config.GetOptimizerName(), config.learning_rate);
    } catch (const std::exception& e) {
        out.error_message = errors::FormatError(
            errors::Training::ModelBuildFailed,
            "ModelBuilder failed to build model/loss/optimizer",
            e.what());
        out.model.reset();
        out.loss.reset();
        out.optimizer.reset();
    }

    return out;
}

BuiltExecutableModel BuildExecutableFromConfig(const TrainingConfiguration& config) {
    if (!config.graph_op_node_ids.empty()) {
        return BuildGraphExecutableFromConfig(config);
    }

    BuiltExecutableModel out;
    BuiltModel sequential = BuildSequentialFromConfig(config);
    if (!sequential.ok()) {
        out.error_message = sequential.error_message;
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
            out.error_message = errors::FormatError(
                errors::Training::ModelBuildFailed,
                "GraphExecutableModel cannot build graph executable",
                reason);
            spdlog::warn("{}", out.error_message);
            return out;
        }
    }

    BuiltModel sequential;
    if (!config.layers.empty()) {
        sequential = BuildSequentialFromConfig(config);
        if (!sequential.ok()) {
            out.error_message = sequential.error_message;
            return out;
        }
    } else {
        try {
            sequential.model = std::make_unique<SequentialModel>();
            sequential.loss = BuildLossFromConfig(config);
            sequential.optimizer = CreateOptimizer(config.GetOptimizerType(),
                                                   config.learning_rate);
        } catch (const std::exception& e) {
            out.error_message = errors::FormatError(
                errors::Training::ModelBuildFailed,
                "GraphExecutableModel failed to build loss/optimizer",
                e.what());
            return out;
        }
        if (!sequential.loss || !sequential.optimizer) {
            out.error_message = errors::FormatError(
                errors::Training::ModelBuildFailed,
                "GraphExecutableModel failed to create loss or optimizer");
            return out;
        }
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
