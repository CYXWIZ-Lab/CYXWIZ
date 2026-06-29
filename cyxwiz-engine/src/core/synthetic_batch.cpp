#include "synthetic_batch.h"
#include <spdlog/spdlog.h>
#include <algorithm>
#include <cstdint>
#include <random>
#include <vector>

namespace cyxwiz {

namespace {

// Grab the first Embedding node's num_embeddings (vocab size) from the
// compiled layers. Used to clamp synthetic text token IDs so they don't
// index past the embedding table. Defaults to 10000 (the dialog factory
// default) when no Embedding node is present.
size_t InferNumEmbeddings(const TrainingConfiguration& config) {
    for (const auto& layer : config.layers) {
        if (layer.type == gui::NodeType::Embedding) {
            auto it = layer.parameters.find("num_embeddings");
            if (it != layer.parameters.end()) {
                try { return static_cast<size_t>(std::stoi(it->second)); }
                catch (...) {}
            }
            return 10000;
        }
    }
    return 10000;
}

Tensor MakeFloatRandom(const std::vector<size_t>& shape, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    size_t n = 1;
    for (size_t d : shape) n *= d;
    std::vector<float> buf(n);
    for (size_t i = 0; i < n; ++i) buf[i] = dist(rng);
    return Tensor(shape, buf.data(), DataType::Float32);
}

Tensor MakeInt64Random(const std::vector<size_t>& shape,
                       int64_t lo, int64_t hi_exclusive,
                       uint32_t seed) {
    std::mt19937 rng(seed);
    if (hi_exclusive <= lo) hi_exclusive = lo + 1;
    std::uniform_int_distribution<int64_t> dist(lo, hi_exclusive - 1);
    size_t n = 1;
    for (size_t d : shape) n *= d;
    std::vector<int64_t> buf(n);
    for (size_t i = 0; i < n; ++i) buf[i] = dist(rng);
    return Tensor(shape, buf.data(), DataType::Int64);
}

Tensor MakeBinaryFloat(const std::vector<size_t>& shape, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> dist(0, 1);
    size_t n = 1;
    for (size_t d : shape) n *= d;
    std::vector<float> buf(n);
    for (size_t i = 0; i < n; ++i) buf[i] = static_cast<float>(dist(rng));
    return Tensor(shape, buf.data(), DataType::Float32);
}

Tensor MakeFeatures(const TrainingConfiguration& config, uint32_t seed) {
    // input_size is the flattened feature count the first layer expects
    // (or seq_len for text). Fall back to 1 to avoid zero-size tensors.
    const size_t input_size = config.input_size > 0 ? config.input_size : 1;

    switch (config.preprocessing_domain) {
        case PreprocessingDomain::Text: {
            const size_t vocab = InferNumEmbeddings(config);
            return MakeInt64Random({1, input_size}, 0,
                                   static_cast<int64_t>(vocab), seed);
        }

        // Image / TimeSeries / Audio: v1 tabular fallback. The plan
        // (commit 2+) will upgrade these to their true shapes once
        // DebugExecutor needs per-domain forward traces.
        case PreprocessingDomain::Tabular:
        case PreprocessingDomain::Image:
        case PreprocessingDomain::TimeSeries:
        case PreprocessingDomain::Audio:
        case PreprocessingDomain::General:
        default:
            return MakeFloatRandom({1, input_size}, seed);
    }
}

Tensor MakeLabels(const TrainingConfiguration& config, uint32_t seed) {
    const size_t num_classes = config.output_size > 0 ? config.output_size : 1;

    switch (config.loss_type) {
        case gui::NodeType::CrossEntropyLoss:
        case gui::NodeType::FocalLoss:
        case gui::NodeType::NLLLoss:
            return MakeInt64Random({1}, 0,
                                   static_cast<int64_t>(num_classes),
                                   seed + 1);

        case gui::NodeType::BCELoss:
        case gui::NodeType::BCEWithLogits:
        case gui::NodeType::SoftDiceLoss:
        case gui::NodeType::TverskyLoss:
        case gui::NodeType::JaccardLoss:
            return MakeBinaryFloat({1, num_classes}, seed + 1);

        case gui::NodeType::MSELoss:
        case gui::NodeType::L1Loss:
        case gui::NodeType::SmoothL1Loss:
        case gui::NodeType::HuberLoss:
            return MakeFloatRandom({1, num_classes}, seed + 1);

        default:
            return MakeInt64Random({1}, 0,
                                   static_cast<int64_t>(num_classes),
                                   seed + 1);
    }
}

} // namespace

SyntheticBatch MakeSyntheticBatch(const TrainingConfiguration& config,
                                  uint32_t seed) {
    SyntheticBatch out;
    out.features = MakeFeatures(config, seed);
    out.labels   = MakeLabels(config, seed);

    auto shape_str = [](const std::vector<size_t>& s) {
        std::string r;
        for (size_t i = 0; i < s.size(); ++i) {
            if (i) r += ",";
            r += std::to_string(s[i]);
        }
        return r;
    };
    spdlog::debug("SyntheticBatch: features shape=[{}], labels shape=[{}], "
                  "domain={}, loss={}, seed={}",
                  shape_str(out.features.Shape()),
                  shape_str(out.labels.Shape()),
                  static_cast<int>(config.preprocessing_domain),
                  static_cast<int>(config.loss_type),
                  seed);
    return out;
}

} // namespace cyxwiz
