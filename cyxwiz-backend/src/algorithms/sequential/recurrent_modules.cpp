#include <cyxwiz/sequential.h>
#include <spdlog/spdlog.h>
#include <algorithm>
#include <stdexcept>
#include <utility>

namespace cyxwiz {
// ============================================================================
// LSTMModule Implementation
// ============================================================================
//
// Wraps cyxwiz::LSTMLayer with two classification-friendly behaviors:
//   1. Keras-style `return_sequences=false` reduction — slices out the
//      last timestep of the full LSTM output so a Dense head can sit
//      directly after the LSTM without an intervening Flatten.
//   2. Symmetric last-step gradient re-expansion in Backward, zeroing
//      all non-terminal steps.
//
// When `return_sequences=true`, the wrapper is a pure passthrough to
// LSTMLayer and output retains the `[batch, seq_len, hidden*dirs]`
// shape — needed for stacked LSTMs and seq-to-seq heads.

LSTMModule::LSTMModule(size_t input_size, size_t hidden_size,
                       size_t num_layers, bool bidirectional,
                       bool return_sequences)
    : input_size_(input_size)
    , hidden_size_(hidden_size)
    , num_layers_(num_layers)
    , bidirectional_(bidirectional)
    , return_sequences_(return_sequences)
{
    layer_ = std::make_unique<LSTMLayer>(
        static_cast<int>(input_size),
        static_cast<int>(hidden_size),
        static_cast<int>(num_layers),
        /*batch_first=*/true,
        bidirectional,
        /*dropout=*/0.0f);
}

Tensor LSTMModule::Forward(const Tensor& input) {
    input_cache_ = input.Clone();

    // LSTMLayer returns the full sequence output:
    //   [batch, seq_len, hidden_size * num_directions]
    Tensor full_output = layer_->Forward(input);
    last_full_output_shape_ = full_output.Shape();

    if (return_sequences_) {
        return full_output;
    }

    // Defensive: if the output isn't 3D for some reason, bail out and
    // pass through. Should never happen under normal use, but we'd
    // rather forward a weird tensor than crash on the slice.
    if (last_full_output_shape_.size() != 3) {
        spdlog::warn("LSTMModule: expected 3D output [batch, seq, hidden_dirs] "
                     "but got {}D — passing full output through",
                     last_full_output_shape_.size());
        return full_output;
    }

    const size_t batch = last_full_output_shape_[0];
    const size_t seq_len = last_full_output_shape_[1];
    const size_t hd = last_full_output_shape_[2];

    // Slice out the last timestep: out[:, seq_len-1, :] ? [batch, hd].
    // Row-major layout means sample b's last step is at offset
    //   b * seq_len * hd + (seq_len - 1) * hd
    Tensor last({batch, hd}, DataType::Float32);
    const float* src = full_output.Data<float>();
    float* dst = static_cast<float*>(last.Data());
    for (size_t b = 0; b < batch; ++b) {
        const float* src_step = src + b * seq_len * hd + (seq_len - 1) * hd;
        std::memcpy(dst + b * hd, src_step, hd * sizeof(float));
    }
    return last;
}

Tensor LSTMModule::Backward(const Tensor& grad_output) {
    if (return_sequences_) {
        // Full-sequence mode — grad_output already has shape
        // [batch, seq_len, hidden*dirs]. Pass straight through.
        return layer_->Backward(grad_output);
    }

    // Last-step mode: re-expand [batch, hidden] gradient to the full
    // [batch, seq_len, hidden] shape with zeros everywhere except the
    // terminal step. LSTMLayer::Backward expects the gradient of the
    // whole sequence output; since only the last step fed into the
    // loss, all earlier timesteps have zero contribution.
    if (last_full_output_shape_.size() != 3) {
        spdlog::warn("LSTMModule::Backward called without a 3D shape cache "
                     "— falling back to direct grad passthrough");
        return layer_->Backward(grad_output);
    }

    const size_t batch = last_full_output_shape_[0];
    const size_t seq_len = last_full_output_shape_[1];
    const size_t hd = last_full_output_shape_[2];

    Tensor expanded = Tensor::Zeros({batch, seq_len, hd});
    const float* src = grad_output.Data<float>();
    float* dst = static_cast<float*>(expanded.Data());
    for (size_t b = 0; b < batch; ++b) {
        float* dst_step = dst + b * seq_len * hd + (seq_len - 1) * hd;
        std::memcpy(dst_step, src + b * hd, hd * sizeof(float));
    }
    return layer_->Backward(expanded);
}

std::map<std::string, Tensor> LSTMModule::GetParameters() {
    return layer_->GetParameters();
}

void LSTMModule::SetParameters(const std::map<std::string, Tensor>& params) {
    layer_->SetParameters(params);
}

std::map<std::string, Tensor> LSTMModule::GetGradients() {
    // LSTMLayer doesn't expose GetGradients() yet — it writes gradients
    // directly into the parameters map keyed as "grad_W_ih", "grad_W_hh",
    // etc. (the same "grad_X"-named entries the legacy optimizer path
    // looked up). The SequentialModel's training step uses GetParameters
    // for both weights AND grads through this naming, so we can forward
    // GetParameters() here. When LSTMLayer grows a dedicated
    // GetGradients() (matching LinearLayer / EmbeddingLayer), this
    // passthrough can become layer_->GetGradients().
    return layer_->GetParameters();
}

std::string LSTMModule::GetName() const {
    const int dirs = bidirectional_ ? 2 : 1;
    return "LSTM(" + std::to_string(input_size_) + " -> " +
           std::to_string(hidden_size_ * dirs) +
           (return_sequences_ ? ", seq" : ", last") + ")";
}

// ============================================================================
// GRUModule Implementation — direct mirror of LSTMModule. The slice and
// re-expand logic for return_sequences=false is identical because
// GRULayer matches LSTMLayer's [batch, seq, hidden*dirs] full-output
// contract.
// ============================================================================

static Tensor ReverseSequenceTensor(const Tensor& input, bool batch_first) {
    const auto& shape = input.Shape();
    if (shape.size() != 3) {
        return input.Clone();
    }

    const size_t batch = batch_first ? shape[0] : shape[1];
    const size_t seq_len = batch_first ? shape[1] : shape[0];
    const size_t features = shape[2];

    Tensor output = Tensor::Zeros(shape, input.GetDataType());
    const float* src = input.Data<float>();
    float* dst = output.Data<float>();

    for (size_t b = 0; b < batch; ++b) {
        for (size_t t = 0; t < seq_len; ++t) {
            const size_t src_t = seq_len - 1 - t;
            const float* src_step = batch_first
                ? src + b * seq_len * features + src_t * features
                : src + src_t * batch * features + b * features;
            float* dst_step = batch_first
                ? dst + b * seq_len * features + t * features
                : dst + t * batch * features + b * features;
            std::memcpy(dst_step, src_step, features * sizeof(float));
        }
    }

    return output;
}

static Tensor ConcatFeatureTensor(const Tensor& left, const Tensor& right) {
    const auto& lshape = left.Shape();
    const auto& rshape = right.Shape();
    if (lshape.size() != 3 || rshape.size() != 3) {
        return left.Clone();
    }

    const size_t batch = lshape[0];
    const size_t seq_len = lshape[1];
    const size_t left_features = lshape[2];
    const size_t right_features = rshape[2];

    Tensor output = Tensor::Zeros({batch, seq_len, left_features + right_features},
                                  left.GetDataType());
    const float* lsrc = left.Data<float>();
    const float* rsrc = right.Data<float>();
    float* dst = output.Data<float>();

    for (size_t b = 0; b < batch; ++b) {
        for (size_t t = 0; t < seq_len; ++t) {
            float* dst_step = dst + b * seq_len * (left_features + right_features)
                             + t * (left_features + right_features);
            const float* lstep = lsrc + b * seq_len * left_features + t * left_features;
            const float* rstep = rsrc + b * seq_len * right_features + t * right_features;
            std::memcpy(dst_step, lstep, left_features * sizeof(float));
            std::memcpy(dst_step + left_features, rstep, right_features * sizeof(float));
        }
    }

    return output;
}

static Tensor SliceFeatureTensor(const Tensor& input, size_t offset, size_t width) {
    const auto& shape = input.Shape();
    if (shape.size() != 3) {
        return input.Clone();
    }

    const size_t batch = shape[0];
    const size_t seq_len = shape[1];
    const size_t features = shape[2];
    if (offset + width > features) {
        return input.Clone();
    }

    Tensor output = Tensor::Zeros({batch, seq_len, width}, input.GetDataType());
    const float* src = input.Data<float>();
    float* dst = output.Data<float>();

    for (size_t b = 0; b < batch; ++b) {
        for (size_t t = 0; t < seq_len; ++t) {
            const float* src_step = src + b * seq_len * features + t * features + offset;
            float* dst_step = dst + b * seq_len * width + t * width;
            std::memcpy(dst_step, src_step, width * sizeof(float));
        }
    }

    return output;
}

static std::string NormalizeGRULayerKey(const std::string& key) {
    std::string normalized = key;
    if (normalized.rfind("layer0_", 0) == 0) {
        normalized.erase(0, 7);
    }
    if (normalized.rfind("grad_", 0) == 0) {
        normalized.erase(0, 5);
    }
    return normalized;
}

static std::string MakeGRUBranchKey(size_t layer_idx, const std::string& branch,
                                    const std::string& normalized_key) {
    return "layer" + std::to_string(layer_idx) + "." + branch + "." + normalized_key;
}

GRUModule::GRUModule(size_t input_size, size_t hidden_size,
                     size_t num_layers, bool bidirectional,
                     bool return_sequences)
    : input_size_(input_size)
    , hidden_size_(hidden_size)
    , num_layers_(num_layers)
    , bidirectional_(bidirectional)
    , return_sequences_(return_sequences)
{
    if (bidirectional_) {
        split_bidirectional_path_ = true;
        forward_layers_.reserve(num_layers_);
        reverse_layers_.reserve(num_layers_);

        for (size_t layer = 0; layer < num_layers_; ++layer) {
            const int layer_input_size = (layer == 0)
                ? static_cast<int>(input_size)
                : static_cast<int>(hidden_size * 2);
            forward_layers_.push_back(std::make_unique<GRULayer>(
                layer_input_size,
                static_cast<int>(hidden_size),
                /*num_layers=*/1,
                /*batch_first=*/true,
                /*bidirectional=*/false,
                /*dropout=*/0.0f));
            reverse_layers_.push_back(std::make_unique<GRULayer>(
                layer_input_size,
                static_cast<int>(hidden_size),
                /*num_layers=*/1,
                /*batch_first=*/true,
                /*bidirectional=*/false,
                /*dropout=*/0.0f));
        }

        spdlog::info("[GRUModule] Using split bidirectional GRU path "
                     "({} layer pairs). GPU placement for this split path "
                     "remains disabled until the single-direction ArrayFire "
                     "GRU path has dedicated correctness and timeout coverage.",
                     num_layers_);
    } else {
        layer_ = std::make_unique<GRULayer>(
            static_cast<int>(input_size),
            static_cast<int>(hidden_size),
            static_cast<int>(num_layers),
            /*batch_first=*/true,
            bidirectional,
            /*dropout=*/0.0f);
    }
}

Tensor GRUModule::Forward(const Tensor& input) {
    input_cache_ = input.Clone();

    Tensor full_output;
    if (split_bidirectional_path_) {
        Tensor layer_input = input;
        for (size_t layer = 0; layer < num_layers_; ++layer) {
            Tensor forward_output = forward_layers_[layer]->Forward(layer_input);
            Tensor reverse_input = ReverseSequenceTensor(layer_input, /*batch_first=*/true);
            Tensor reverse_output = reverse_layers_[layer]->Forward(reverse_input);
            reverse_output = ReverseSequenceTensor(reverse_output, /*batch_first=*/true);
            layer_input = ConcatFeatureTensor(forward_output, reverse_output);
        }
        full_output = layer_input;
    } else {
        full_output = layer_->Forward(input);
    }
    last_full_output_shape_ = full_output.Shape();

    if (return_sequences_) {
        return full_output;
    }

    if (last_full_output_shape_.size() != 3) {
        spdlog::warn("GRUModule: expected 3D output [batch, seq, hidden_dirs] "
                     "but got {}D — passing full output through",
                     last_full_output_shape_.size());
        return full_output;
    }

    const size_t batch = last_full_output_shape_[0];
    const size_t seq_len = last_full_output_shape_[1];
    const size_t hd = last_full_output_shape_[2];

    Tensor last({batch, hd}, DataType::Float32);
    const float* src = full_output.Data<float>();
    float* dst = static_cast<float*>(last.Data());
    for (size_t b = 0; b < batch; ++b) {
        const float* src_step = src + b * seq_len * hd + (seq_len - 1) * hd;
        std::memcpy(dst + b * hd, src_step, hd * sizeof(float));
    }
    return last;
}

Tensor GRUModule::Backward(const Tensor& grad_output) {
    Tensor upstream = grad_output;
    if (!return_sequences_) {
        if (last_full_output_shape_.size() != 3) {
            spdlog::warn("GRUModule::Backward called without a 3D shape cache "
                         "- falling back to direct grad passthrough");
            return split_bidirectional_path_
                ? Tensor::Zeros(input_cache_.Shape())
                : layer_->Backward(grad_output);
        }

        const size_t batch = last_full_output_shape_[0];
        const size_t seq_len = last_full_output_shape_[1];
        const size_t hd = last_full_output_shape_[2];

        Tensor expanded = Tensor::Zeros({batch, seq_len, hd});
        const float* src = grad_output.Data<float>();
        float* dst = static_cast<float*>(expanded.Data());
        for (size_t b = 0; b < batch; ++b) {
            float* dst_step = dst + b * seq_len * hd + (seq_len - 1) * hd;
            std::memcpy(dst_step, src + b * hd, hd * sizeof(float));
        }
        upstream = expanded;
    }

    if (split_bidirectional_path_) {
        if (upstream.Shape().size() != 3) {
            spdlog::warn("GRUModule::Backward expected 3D upstream gradient "
                         "for split bidirectional path");
            return Tensor::Zeros(input_cache_.Shape());
        }

        Tensor layer_grad = upstream;
        for (int layer = static_cast<int>(num_layers_) - 1; layer >= 0; --layer) {
            const size_t total_features = layer_grad.Shape()[2];
            const size_t half_features = total_features / 2;
            Tensor forward_grad = SliceFeatureTensor(layer_grad, 0, half_features);
            Tensor reverse_grad = SliceFeatureTensor(layer_grad, half_features, half_features);

            Tensor dx_forward = forward_layers_[static_cast<size_t>(layer)]->Backward(forward_grad);
            Tensor dx_reverse = reverse_layers_[static_cast<size_t>(layer)]->Backward(
                ReverseSequenceTensor(reverse_grad, /*batch_first=*/true));
            dx_reverse = ReverseSequenceTensor(dx_reverse, /*batch_first=*/true);

            layer_grad = dx_forward + dx_reverse;
        }

        return layer_grad;
    }

    return layer_->Backward(upstream);
}

std::map<std::string, Tensor> GRUModule::GetParameters() {
    if (split_bidirectional_path_) {
        std::map<std::string, Tensor> params;
        for (size_t layer = 0; layer < num_layers_; ++layer) {
            auto forward_params = forward_layers_[layer]->GetParameters();
            auto reverse_params = reverse_layers_[layer]->GetParameters();
            for (const auto& [key, tensor] : forward_params) {
                if (key.find("grad_") != std::string::npos) continue;
                params[MakeGRUBranchKey(layer, "forward", NormalizeGRULayerKey(key))] = tensor;
            }
            for (const auto& [key, tensor] : reverse_params) {
                if (key.find("grad_") != std::string::npos) continue;
                params[MakeGRUBranchKey(layer, "reverse", NormalizeGRULayerKey(key))] = tensor;
            }
        }
        return params;
    }
    return layer_->GetParameters();
}

void GRUModule::SetParameters(const std::map<std::string, Tensor>& params) {
    if (split_bidirectional_path_) {
        std::vector<std::map<std::string, Tensor>> forward_params(num_layers_);
        std::vector<std::map<std::string, Tensor>> reverse_params(num_layers_);
        for (const auto& [key, tensor] : params) {
            if (key.rfind("layer", 0) != 0) {
                continue;
            }
            const size_t dot1 = key.find('.');
            const size_t dot2 = key.find('.', dot1 == std::string::npos ? 0 : dot1 + 1);
            if (dot1 == std::string::npos || dot2 == std::string::npos) {
                continue;
            }

            const size_t layer_idx = static_cast<size_t>(std::stoul(key.substr(5, dot1 - 5)));
            if (layer_idx >= num_layers_) {
                continue;
            }

            const std::string branch = key.substr(dot1 + 1, dot2 - dot1 - 1);
            const std::string base_key = key.substr(dot2 + 1);
            if (base_key.empty()) {
                continue;
            }
            const std::string child_key = "layer0_" + base_key;
            if (branch == "forward") {
                forward_params[layer_idx][child_key] = tensor;
            } else if (branch == "reverse") {
                reverse_params[layer_idx][child_key] = tensor;
            }
        }
        for (size_t layer = 0; layer < num_layers_; ++layer) {
            forward_layers_[layer]->SetParameters(forward_params[layer]);
            reverse_layers_[layer]->SetParameters(reverse_params[layer]);
        }
        return;
    }
    layer_->SetParameters(params);
}

std::map<std::string, Tensor> GRUModule::GetGradients() {
    auto build_gradient_map = [](const std::map<std::string, Tensor>& params,
                                 const std::string& prefix) {
        std::map<std::string, Tensor> grads;
        for (const auto& [key, value] : params) {
            if (key.find("grad_") == std::string::npos) continue;
            const std::string base = NormalizeGRULayerKey(key);
            grads[prefix + base] = value;
        }
        return grads;
    };

    if (split_bidirectional_path_) {
        std::map<std::string, Tensor> grads;
        for (size_t layer = 0; layer < num_layers_; ++layer) {
            auto forward_grads = build_gradient_map(forward_layers_[layer]->GetParameters(),
                                                     MakeGRUBranchKey(layer, "forward", ""));
            auto reverse_grads = build_gradient_map(reverse_layers_[layer]->GetParameters(),
                                                     MakeGRUBranchKey(layer, "reverse", ""));
            grads.insert(forward_grads.begin(), forward_grads.end());
            grads.insert(reverse_grads.begin(), reverse_grads.end());
        }
        return grads;
    }
    // Same convention as LSTMModule - GRULayer writes "grad_*" keys
    // into its parameters map and the SequentialModel optimizer step
    // reads them via GetParameters().
    return layer_->GetParameters();
}

std::string GRUModule::GetName() const {
    const int dirs = bidirectional_ ? 2 : 1;
    const std::string prefix = split_bidirectional_path_ ? "Bi" : "";
    return prefix + std::string("GRU(") + std::to_string(input_size_) + " -> " +
           std::to_string(hidden_size_ * dirs) +
           (return_sequences_ ? ", seq" : ", last") + ")";
}

void GRUModule::SetTraining(bool training) {
    Module::SetTraining(training);
    if (layer_) layer_->SetTraining(training);
    for (auto& layer : forward_layers_) {
        layer->SetTraining(training);
    }
    for (auto& layer : reverse_layers_) {
        layer->SetTraining(training);
    }
}

} // namespace cyxwiz

