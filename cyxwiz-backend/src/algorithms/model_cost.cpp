#include "cyxwiz/model_cost.h"

#include "cyxwiz/layers/recurrent.h"
#include "cyxwiz/layers/transformer.h"
#include "cyxwiz/sequential.h"

#include <algorithm>
#include <utility>

namespace cyxwiz {

namespace {

long long Elements(const std::vector<size_t>& shape) {
    if (shape.empty()) return 0;
    long long count = 1;
    for (size_t d : shape) count *= static_cast<long long>(d);
    return count;
}

// "194,688"
std::string Grouped(long long value) {
    const bool negative = value < 0;
    const std::string digits = std::to_string(negative ? -value : value);
    std::string out;
    const size_t lead = digits.size() % 3;
    for (size_t i = 0; i < digits.size(); ++i) {
        if (i != 0 && i >= lead && (i - lead) % 3 == 0) out += ',';
        out += digits[i];
    }
    return negative ? "-" + out : out;
}

// "26·26·32" from a shape.
std::string Product(const std::vector<size_t>& shape) {
    std::string text;
    for (size_t i = 0; i < shape.size(); ++i) text += (i ? "·" : "") + std::to_string(shape[i]);
    return text.empty() ? "1" : text;
}

LayerCost Counted(long long value, std::string formula) { return {true, value, std::move(formula) + " = " + Grouped(value)}; }

LayerCost NotCounted(std::string why) { return {false, 0, std::move(why)}; }

// A sample of a sequence layer: [length, width]; a flat [width] is one step.
std::pair<long long, long long> LengthAndWidth(const std::vector<size_t>& shape) {
    if (shape.size() >= 2) return {static_cast<long long>(shape[shape.size() - 2]), static_cast<long long>(shape.back())};
    if (shape.size() == 1) return {1, static_cast<long long>(shape[0])};
    return {0, 0};
}

LayerCost Recurrent(const LayerCostInput& layer, long long gates, const char* gate_words) {
    const auto [length, width] = LengthAndWidth(layer.input);
    if (length <= 0 || width <= 0) return NotCounted("its input has no sequence shape yet");
    const long long hidden = std::max<long long>(1, layer.hidden);
    const long long stacked = std::max<long long>(1, layer.stacked);
    const long long directions = layer.bidirectional ? 2 : 1;
    // Each step and direction: gates x hidden x (input + hidden); a stacked
    // layer reads the one below (hidden x directions).
    const long long first = gates * hidden * (width + hidden);
    const long long upper = gates * hidden * (hidden * directions + hidden);
    const long long value = length * directions * (first + (stacked - 1) * upper);
    std::string formula = std::to_string(length) + " steps · " + (directions == 2 ? "2 directions · " : "") +
                          gate_words + " · " + std::to_string(hidden) + " · (" + std::to_string(width) + " + " +
                          std::to_string(hidden) + ")";
    if (stacked > 1)
        formula += " + " + std::to_string(stacked - 1) + " stacked layer" + (stacked > 2 ? "s" : "") + " on " +
                   std::to_string(hidden * directions);
    return Counted(value, formula);
}

// Self-attention projections and scores: Q and the output E x E, K and V
// E x (E · kv heads / heads), scores and weighted sum 2 · L² · E; a
// transformer block adds its feed-forward (2 or, gated, 3 matrices).
LayerCost Attention(const LayerCostInput& layer, bool feedforward) {
    const auto [length, width] = LengthAndWidth(layer.input);
    if (length <= 0 || width <= 0) return NotCounted("its input has no sequence shape yet");
    const long long heads = std::max<long long>(1, layer.heads);
    const long long kv_heads = layer.kv_heads > 0 ? layer.kv_heads : heads;
    const long long kv_width = width * kv_heads / heads;
    const long long projections = length * (2 * width * width + 2 * width * kv_width);
    const long long scores = 2 * length * length * width;
    long long value = projections + scores;
    std::string formula = std::to_string(length) + " · (2·" + std::to_string(width) + "² + 2·" + std::to_string(width) +
                          "·" + std::to_string(kv_width) + ") projections + 2·" + std::to_string(length) + "²·" +
                          std::to_string(width) + " attention";
    if (feedforward) {
        const long long factor = layer.gated ? 3 : 2;
        value += factor * length * width * layer.feedforward;
        formula += " + " + std::to_string(factor) + "·" + std::to_string(length) + "·" + std::to_string(width) + "·" +
                   std::to_string(layer.feedforward) + (layer.gated ? " gated feed-forward" : " feed-forward");
    }
    return Counted(value, formula);
}

}  // namespace

LayerCost CountMultiplyAdds(const LayerCostInput& layer) {
    const auto& in = layer.input;
    const auto& out = layer.output;
    switch (layer.kind) {
        case LayerCostKind::None:
            return {true, 0, ""};
        case LayerCostKind::Dense: {
            // Each output value is a dot product over the input features of
            // its position: a sequence keeps its positions.
            const long long outputs = Elements(out);
            const long long units = out.empty() ? 0 : static_cast<long long>(out.back());
            if (outputs <= 0 || units <= 0 || in.empty()) return NotCounted("its shapes are not known yet");
            const long long positions = outputs / units;
            const long long features = Elements(in) / std::max<long long>(1, positions);
            const std::string formula = (positions > 1 ? std::to_string(positions) + " positions · " : "") +
                                        std::to_string(features) + " · " + std::to_string(units);
            return Counted(outputs * features, formula);
        }
        case LayerCostKind::Conv2D:
        case LayerCostKind::DepthwiseConv2D: {
            if (in.size() != 3 || out.size() != 3) return NotCounted("its input has no image shape yet");
            const bool depthwise = layer.kind == LayerCostKind::DepthwiseConv2D;
            const long long per_output = layer.kernel * layer.kernel * (depthwise ? 1 : static_cast<long long>(in[2]));
            const std::string window = std::to_string(layer.kernel) + "·" + std::to_string(layer.kernel) +
                                       (depthwise ? "" : "·" + std::to_string(in[2]));
            return Counted(Elements(out) * per_output, Product(out) + " · " + window);
        }
        case LayerCostKind::Conv1D:
            if (in.size() != 2 || out.size() != 2) return NotCounted("its input has no sequence shape yet");
            return Counted(Elements(out) * layer.kernel * static_cast<long long>(in[1]),
                           Product(out) + " · " + std::to_string(layer.kernel) + "·" + std::to_string(in[1]));
        case LayerCostKind::Conv3D: {
            if (in.size() != 4 || out.size() != 4) return NotCounted("its input has no volume shape yet");
            const long long k = layer.kernel;
            return Counted(Elements(out) * k * k * k * static_cast<long long>(in[3]),
                           Product(out) + " · " + std::to_string(k) + "·" + std::to_string(k) + "·" + std::to_string(k) +
                               "·" + std::to_string(in[3]));
        }
        case LayerCostKind::ConvTranspose2D: {
            // Every input value is spread over a k x k window of each output channel.
            if (in.size() != 3 || out.size() != 3) return NotCounted("its input has no image shape yet");
            const long long per_input = layer.kernel * layer.kernel * static_cast<long long>(out[2]);
            return Counted(Elements(in) * per_input, Product(in) + " · " + std::to_string(layer.kernel) + "·" +
                                                         std::to_string(layer.kernel) + "·" + std::to_string(out[2]));
        }
        case LayerCostKind::LSTM:
            return Recurrent(layer, 4, "4 gates");
        case LayerCostKind::GRU:
            return Recurrent(layer, 3, "3 gates");
        case LayerCostKind::RNN:
            return Recurrent(layer, 1, "1 gate");
        case LayerCostKind::Attention:
            return Attention(layer, false);
        case LayerCostKind::TransformerBlock:
            return Attention(layer, true);
        case LayerCostKind::LinearAttention: {
            // Projections 4·L·E², then per head phi(K)ᵀV and phi(Q)·(KᵀV): 2·L·E·d.
            const auto [length, width] = LengthAndWidth(in);
            if (length <= 0 || width <= 0) return NotCounted("its input has no sequence shape yet");
            const long long heads = std::max<long long>(1, layer.heads);
            const long long head_width = width / heads;
            const long long value = 4 * length * width * width + 2 * length * width * head_width;
            return Counted(value, "4·" + std::to_string(length) + "·" + std::to_string(width) + "² projections + 2·" +
                                      std::to_string(length) + "·" + std::to_string(width) + "·" +
                                      std::to_string(head_width) + " kernel attention");
        }
    }
    return {true, 0, ""};
}

// ---- The modules' sizes ---------------------------------------------------------

namespace {

LayerCostInput Shaped(LayerCostKind kind, const std::vector<size_t>& input, const std::vector<size_t>& output) {
    LayerCostInput cost;
    cost.kind = kind;
    cost.input = input;
    cost.output = output;
    return cost;
}

}  // namespace

LayerCostInput Conv1DModule::CostInput(const std::vector<size_t>& input, const std::vector<size_t>& output) const {
    auto cost = Shaped(LayerCostKind::Conv1D, input, output);
    cost.kernel = kernel_size_;
    return cost;
}

LayerCostInput Conv2DModule::CostInput(const std::vector<size_t>& input, const std::vector<size_t>& output) const {
    auto cost = Shaped(LayerCostKind::Conv2D, input, output);
    cost.kernel = kernel_size_;
    return cost;
}

LayerCostInput DepthwiseConv2DModule::CostInput(const std::vector<size_t>& input, const std::vector<size_t>& output) const {
    auto cost = Shaped(LayerCostKind::DepthwiseConv2D, input, output);
    cost.kernel = kernel_size_;
    return cost;
}

LayerCostInput ConvTranspose2DModule::CostInput(const std::vector<size_t>& input, const std::vector<size_t>& output) const {
    auto cost = Shaped(LayerCostKind::ConvTranspose2D, input, output);
    cost.kernel = kernel_size_;
    return cost;
}

LayerCostInput Conv3DModule::CostInput(const std::vector<size_t>& input, const std::vector<size_t>& output) const {
    auto cost = Shaped(LayerCostKind::Conv3D, input, output);
    cost.kernel = kernel_size_;
    return cost;
}

LayerCostInput LinearModule::CostInput(const std::vector<size_t>& input, const std::vector<size_t>& output) const {
    auto cost = Shaped(LayerCostKind::Dense, input, output);
    return cost;
}

LayerCostInput TimeDistributedDenseModule::CostInput(const std::vector<size_t>& input, const std::vector<size_t>& output) const {
    auto cost = Shaped(LayerCostKind::Dense, input, output);
    return cost;
}

LayerCostInput LSTMModule::CostInput(const std::vector<size_t>& input, const std::vector<size_t>& output) const {
    auto cost = Shaped(LayerCostKind::LSTM, input, output);
    cost.hidden = static_cast<long long>(hidden_size_);
    cost.stacked = layer_->GetNumLayers();
    cost.bidirectional = bidirectional_;
    return cost;
}

LayerCostInput GRUModule::CostInput(const std::vector<size_t>& input, const std::vector<size_t>& output) const {
    auto cost = Shaped(LayerCostKind::GRU, input, output);
    cost.hidden = static_cast<long long>(hidden_size_);
    cost.stacked = layer_->GetNumLayers();
    cost.bidirectional = bidirectional_;
    return cost;
}

LayerCostInput RNNModule::CostInput(const std::vector<size_t>& input, const std::vector<size_t>& output) const {
    auto cost = Shaped(LayerCostKind::RNN, input, output);
    cost.hidden = static_cast<long long>(hidden_size_);
    cost.stacked = layer_->GetNumLayers();
    cost.bidirectional = bidirectional_;
    return cost;
}

LayerCostInput TransformerEncoderModule::CostInput(const std::vector<size_t>& input, const std::vector<size_t>& output) const {
    auto cost = Shaped(LayerCostKind::TransformerBlock, input, output);
    cost.heads = static_cast<long long>(num_heads_);
    cost.feedforward = static_cast<long long>(dim_feedforward_);
    return cost;
}

LayerCostInput TransformerDecoderModule::CostInput(const std::vector<size_t>& input, const std::vector<size_t>& output) const {
    auto cost = Shaped(LayerCostKind::TransformerBlock, input, output);
    const auto& options = layer_->GetBlockOptions();
    cost.heads = static_cast<long long>(num_heads_);
    cost.kv_heads = options.num_kv_heads;
    cost.feedforward = static_cast<long long>(dim_feedforward_);
    cost.gated = options.ffn_type == TransformerFeedForwardType::Gated;
    return cost;
}

LayerCostInput MultiHeadAttentionModule::CostInput(const std::vector<size_t>& input, const std::vector<size_t>& output) const {
    auto cost = Shaped(LayerCostKind::Attention, input, output);
    cost.heads = static_cast<long long>(num_heads_);
    cost.kv_heads = options_.num_kv_heads;
    return cost;
}

LayerCostInput LinearAttentionModule::CostInput(const std::vector<size_t>& input, const std::vector<size_t>& output) const {
    auto cost = Shaped(LayerCostKind::LinearAttention, input, output);
    cost.heads = static_cast<long long>(num_heads_);
    return cost;
}

LayerCostInput ModuleCostInput(const Module& module, const std::vector<size_t>& input,
                               const std::vector<size_t>& output) {
    if (const auto* typed = dynamic_cast<const Conv1DModule*>(&module)) return typed->CostInput(input, output);
    if (const auto* typed = dynamic_cast<const Conv2DModule*>(&module)) return typed->CostInput(input, output);
    if (const auto* typed = dynamic_cast<const DepthwiseConv2DModule*>(&module)) return typed->CostInput(input, output);
    if (const auto* typed = dynamic_cast<const ConvTranspose2DModule*>(&module)) return typed->CostInput(input, output);
    if (const auto* typed = dynamic_cast<const Conv3DModule*>(&module)) return typed->CostInput(input, output);
    if (const auto* typed = dynamic_cast<const LinearModule*>(&module)) return typed->CostInput(input, output);
    if (const auto* typed = dynamic_cast<const TimeDistributedDenseModule*>(&module)) return typed->CostInput(input, output);
    if (const auto* typed = dynamic_cast<const LSTMModule*>(&module)) return typed->CostInput(input, output);
    if (const auto* typed = dynamic_cast<const GRUModule*>(&module)) return typed->CostInput(input, output);
    if (const auto* typed = dynamic_cast<const RNNModule*>(&module)) return typed->CostInput(input, output);
    if (const auto* typed = dynamic_cast<const TransformerEncoderModule*>(&module)) return typed->CostInput(input, output);
    if (const auto* typed = dynamic_cast<const TransformerDecoderModule*>(&module)) return typed->CostInput(input, output);
    if (const auto* typed = dynamic_cast<const MultiHeadAttentionModule*>(&module)) return typed->CostInput(input, output);
    if (const auto* typed = dynamic_cast<const LinearAttentionModule*>(&module)) return typed->CostInput(input, output);
    return Shaped(LayerCostKind::None, input, output);
}

}  // namespace cyxwiz
