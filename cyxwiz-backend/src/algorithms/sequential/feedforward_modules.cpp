#include <cyxwiz/sequential.h>
#include "../arrayfire_backend_utils.h"
#ifdef CYXWIZ_HAS_ARRAYFIRE
#include <arrayfire.h>
#endif
#include <cmath>
#include <random>
#include <limits>
#include <stdexcept>
#include <utility>

#ifdef max
#undef max
#endif

namespace cyxwiz {
// ============================================================================
// LinearModule Implementation
// ============================================================================

LinearModule::LinearModule(size_t in_features, size_t out_features, bool use_bias)
    : in_features_(in_features)
    , out_features_(out_features)
{
    layer_ = std::make_unique<LinearLayer>(in_features, out_features, use_bias);
}

Tensor LinearModule::Forward(const Tensor& input) {
    input_cache_ = input.Clone();
    return layer_->Forward(input);
}

Tensor LinearModule::Backward(const Tensor& grad_output) {
    return layer_->Backward(grad_output);
}

Tensor LinearModule::ForwardSequence(const Tensor& input) {
    input_cache_ = input;
    return layer_->ForwardSequence(input);
}

Tensor LinearModule::BackwardSequence(const Tensor& grad_output) {
    return layer_->BackwardSequence(grad_output);
}

Tensor LinearModule::ForwardSparseCsr(
    const LinearSparseCsrBatchView& input) {
    return layer_->ForwardSparseCsr(input);
}

void LinearModule::BackwardSparseCsr(
    const LinearSparseCsrBatchView& input,
    const Tensor& grad_output) {
    layer_->BackwardSparseCsr(input, grad_output);
}

std::map<std::string, Tensor> LinearModule::GetParameters() {
    return layer_->GetParameters();
}

void LinearModule::SetParameters(const std::map<std::string, Tensor>& params) {
    layer_->SetParameters(params);
}

std::map<std::string, Tensor> LinearModule::GetGradients() {
    return layer_->GetGradients();
}

std::string LinearModule::GetName() const {
    return "Linear(" + std::to_string(in_features_) + " -> " + std::to_string(out_features_) + ")";
}

// ============================================================================
// TimeDistributedDenseModule Implementation
// ============================================================================

TimeDistributedDenseModule::TimeDistributedDenseModule(size_t in_features,
                                                       size_t out_features,
                                                       bool use_bias)
    : linear_(in_features, out_features, use_bias)
    , in_features_(in_features)
    , out_features_(out_features) {}

Tensor TimeDistributedDenseModule::Forward(const Tensor& input) {
    input_cache_ = input.Clone();
    input_shape_ = input.Shape();
    if (input_shape_.size() != 3) {
        throw std::runtime_error(
            "TimeDistributedDenseModule: input must be [batch, seq_len, features]");
    }
    if (input_shape_[2] != in_features_) {
        throw std::runtime_error(
            "TimeDistributedDenseModule: Input features mismatch. Expected " +
            std::to_string(in_features_) + ", got " +
            std::to_string(input_shape_[2]));
    }

    return linear_.ForwardSequence(input);
}

Tensor TimeDistributedDenseModule::Backward(const Tensor& grad_output) {
    if (input_shape_.size() != 3) {
        throw std::runtime_error(
            "TimeDistributedDenseModule: Backward called before Forward");
    }
    const auto& grad_shape = grad_output.Shape();
    if (grad_shape.size() != 3 ||
        grad_shape[0] != input_shape_[0] ||
        grad_shape[1] != input_shape_[1] ||
        grad_shape[2] != out_features_) {
        throw std::runtime_error(
            "TimeDistributedDenseModule: grad_output must be [batch, seq_len, out_features]");
    }

    return linear_.BackwardSequence(grad_output);
}

std::map<std::string, Tensor> TimeDistributedDenseModule::GetParameters() {
    return linear_.GetParameters();
}

void TimeDistributedDenseModule::SetParameters(
    const std::map<std::string, Tensor>& params) {
    linear_.SetParameters(params);
}

std::map<std::string, Tensor> TimeDistributedDenseModule::GetGradients() {
    return linear_.GetGradients();
}

std::string TimeDistributedDenseModule::GetName() const {
    return "TimeDistributedDense(" + std::to_string(in_features_) +
           " -> " + std::to_string(out_features_) + ")";
}

// ============================================================================
// EmbeddingModule Implementation
// ============================================================================
//
// The backend's EmbeddingLayer reads indices as int32, but CyxWiz's
// IBatcher / training pipeline ships every tensor as float32 (for
// uniformity with Dense/Conv/etc.). We bridge the type gap here by
// building an int32 Tensor on every forward pass whose values are the
// rounded floats from the upstream batch. This is the cost of dropping
// Embedding into an otherwise float-only graph — worth it, since the
// alternative (dual-typing the whole batching layer) is much bigger.
//
// Shape contract:
//   Forward input:  [batch, seq_len]                float, integer values
//   Forward output: [batch, seq_len, embedding_dim] float
//
// A Flatten module after this produces [batch, seq_len * embedding_dim]
// which is what a Dense layer head expects.

EmbeddingModule::EmbeddingModule(size_t num_embeddings, size_t embedding_dim,
                                 int padding_idx, float max_norm)
    : num_embeddings_(num_embeddings)
    , embedding_dim_(embedding_dim)
    , padding_idx_(padding_idx)
    , max_norm_(max_norm)
{
    if (num_embeddings_ < 2 ||
        num_embeddings_ > static_cast<size_t>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument(
            "EmbeddingModule: num_embeddings must be in [2, INT_MAX]");
    }
    if (embedding_dim_ < 1 ||
        embedding_dim_ > static_cast<size_t>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument(
            "EmbeddingModule: embedding_dim must be in [1, INT_MAX]");
    }
    if (padding_idx_ < -1 ||
        (padding_idx_ >= 0 &&
         static_cast<size_t>(padding_idx_) >= num_embeddings_)) {
        throw std::invalid_argument(
            "EmbeddingModule: padding_idx must be -1 or a valid token id");
    }
    if (!std::isfinite(max_norm_) || max_norm_ < 0.0f) {
        throw std::invalid_argument(
            "EmbeddingModule: max_norm must be finite and >= 0");
    }
    layer_ = std::make_unique<EmbeddingLayer>(
        static_cast<int>(num_embeddings),
        static_cast<int>(embedding_dim),
        padding_idx,
        max_norm);
}

Tensor EmbeddingModule::Forward(const Tensor& input) {
    input_cache_ = input.Clone();

    // Convert [batch, seq_len] token IDs to int32. Most production batchers
    // still carry token IDs as float32, while sequence/synthetic paths use
    // integer tensors. Normalize both forms here so the model-facing contract
    // stays narrow.
    const auto& shape = input.Shape();
    if (shape.size() != 1 && shape.size() != 2) {
        throw std::runtime_error(
            "EmbeddingModule: input must be [sequence] or [batch, sequence]");
    }
    size_t total = 1;
    for (size_t d : shape) {
        if (d == 0 || total > std::numeric_limits<size_t>::max() / d) {
            throw std::runtime_error(
                "EmbeddingModule: input dimensions must be nonzero and bounded");
        }
        total *= d;
    }

    Tensor int_input(shape, DataType::Int32);
    int32_t* dst = int_input.MutableData<int32_t>();
    const int32_t vocab_max = static_cast<int32_t>(num_embeddings_) - 1;
    const float* float_ids = input.GetDataType() == DataType::Float32
        ? input.ReadData<float>() : nullptr;
    const int32_t* int32_ids = input.GetDataType() == DataType::Int32
        ? input.ReadData<int32_t>() : nullptr;
    const int64_t* int64_ids = input.GetDataType() == DataType::Int64
        ? input.ReadData<int64_t>() : nullptr;
    for (size_t i = 0; i < total; ++i) {
        int32_t idx = 0;
        switch (input.GetDataType()) {
            case DataType::Float32: {
                const float value = float_ids[i];
                if (!std::isfinite(value) || std::trunc(value) != value ||
                    value < 0.0f || value > static_cast<float>(vocab_max)) {
                    throw std::runtime_error(
                        "EmbeddingModule: Float32 token ids must be finite exact integers in vocabulary range");
                }
                idx = static_cast<int32_t>(value);
                break;
            }
            case DataType::Int32:
                idx = int32_ids[i];
                break;
            case DataType::Int64: {
                const int64_t value = int64_ids[i];
                if (value < 0 || value > static_cast<int64_t>(vocab_max)) {
                    throw std::runtime_error(
                        "EmbeddingModule: Int64 token id " +
                        std::to_string(value) +
                        " is outside vocabulary range [0, " +
                        std::to_string(num_embeddings_) + ") at position " +
                        std::to_string(i));
                }
                idx = static_cast<int32_t>(value);
                break;
            }
            default:
                throw std::runtime_error(
                    "EmbeddingModule: input token ids must be Float32, Int32, or Int64");
        }
        if (idx < 0 || idx > vocab_max) {
            throw std::runtime_error(
                "EmbeddingModule: token id is outside vocabulary range");
        }
        dst[i] = idx;
    }

    return layer_->Forward(int_input);
}

Tensor EmbeddingModule::Backward(const Tensor& grad_output) {
    // EmbeddingLayer::Backward returns an empty tensor because you
    // can't differentiate w.r.t. integer indices — but it DOES update
    // its own weight gradients internally. Propagating "nothing"
    // upstream is correct: Embedding is almost always the first
    // trainable layer, so there's no upstream tensor gradient anyway.
    return layer_->Backward(grad_output);
}

std::map<std::string, Tensor> EmbeddingModule::GetParameters() {
    return layer_->GetParameters();
}

void EmbeddingModule::SetParameters(const std::map<std::string, Tensor>& params) {
    layer_->SetParameters(params);
}

std::map<std::string, Tensor> EmbeddingModule::GetGradients() {
    return layer_->GetGradients();
}

void EmbeddingModule::LoadPretrainedWeights(const Tensor& weights, bool freeze) {
    layer_->LoadPretrainedWeights(weights, freeze);
    SetTrainable(!freeze);
}

std::string EmbeddingModule::GetName() const {
    return "Embedding(" + std::to_string(num_embeddings_) + " x " +
           std::to_string(embedding_dim_) + ")";
}

// ============================================================================
// SequenceFeatureFusionModule Implementation
// ============================================================================

namespace {

int64_t ReadIdAt(const Tensor& tensor, size_t index) {
    switch (tensor.GetDataType()) {
        case DataType::Float32:
            return static_cast<int64_t>(tensor.Data<float>()[index]);
        case DataType::Int32:
            return static_cast<int64_t>(tensor.Data<int32_t>()[index]);
        case DataType::Int64:
            return tensor.Data<int64_t>()[index];
        default:
            throw std::runtime_error(
                "SequenceFeatureFusionModule: ids must be Float32, Int32, or Int64");
    }
}

std::map<std::string, Tensor> FilterPrefixedParams(
    const std::map<std::string, Tensor>& params,
    const std::string& prefix) {
    std::map<std::string, Tensor> out;
    for (const auto& [key, tensor] : params) {
        if (key.rfind(prefix, 0) == 0) {
            out[key.substr(prefix.size())] = tensor;
        }
    }
    return out;
}

void AppendPrefixedParams(std::map<std::string, Tensor>& out,
                          std::map<std::string, Tensor> params,
                          const std::string& prefix) {
    for (auto& [key, tensor] : params) {
        out[prefix + key] = std::move(tensor);
    }
}

} // namespace

SequenceFeatureFusionModule::SequenceFeatureFusionModule(
    size_t word_num_embeddings,
    size_t word_embedding_dim,
    size_t pos_num_embeddings,
    size_t pos_embedding_dim,
    int word_padding_idx,
    int pos_padding_idx)
    : word_embedding_(word_num_embeddings, word_embedding_dim, word_padding_idx)
    , pos_embedding_(pos_num_embeddings, pos_embedding_dim, pos_padding_idx)
    , word_embedding_dim_(word_embedding_dim)
    , pos_embedding_dim_(pos_embedding_dim)
    , fused_embedding_dim_(word_embedding_dim + pos_embedding_dim) {
    if (word_num_embeddings < 2 || pos_num_embeddings < 2) {
        throw std::runtime_error(
            "SequenceFeatureFusionModule: vocabularies must have at least two entries");
    }
    if (word_embedding_dim_ < 1 || pos_embedding_dim_ < 1) {
        throw std::runtime_error(
            "SequenceFeatureFusionModule: embedding dimensions must be positive");
    }
}

Tensor SequenceFeatureFusionModule::Forward(const Tensor& input) {
    input_cache_ = input.Clone();
    input_shape_ = input.Shape();
    if (input_shape_.size() != 3 || input_shape_[2] != 2) {
        throw std::runtime_error(
            "SequenceFeatureFusionModule: input must be [batch, seq_len, 2]");
    }

    const size_t batch = input_shape_[0];
    const size_t seq_len = input_shape_[1];
    const size_t token_count = batch * seq_len;
    std::vector<int64_t> word_ids(token_count, 0);
    std::vector<int64_t> pos_ids(token_count, 0);

    for (size_t i = 0; i < token_count; ++i) {
        word_ids[i] = ReadIdAt(input, i * 2);
        pos_ids[i] = ReadIdAt(input, i * 2 + 1);
    }

    Tensor word_tensor({batch, seq_len}, word_ids.data(), DataType::Int64);
    Tensor pos_tensor({batch, seq_len}, pos_ids.data(), DataType::Int64);
    Tensor word_emb = word_embedding_.Forward(word_tensor);
    Tensor pos_emb = pos_embedding_.Forward(pos_tensor);

    const auto& word_shape = word_emb.Shape();
    const auto& pos_shape = pos_emb.Shape();
    if (word_shape.size() != 3 || pos_shape.size() != 3 ||
        word_shape[0] != batch || pos_shape[0] != batch ||
        word_shape[1] != seq_len || pos_shape[1] != seq_len ||
        word_shape[2] != word_embedding_dim_ ||
        pos_shape[2] != pos_embedding_dim_) {
        throw std::runtime_error(
            "SequenceFeatureFusionModule: embedding output shape mismatch");
    }

    Tensor output({batch, seq_len, fused_embedding_dim_}, DataType::Float32);
    const float* word_data = word_emb.Data<float>();
    const float* pos_data = pos_emb.Data<float>();
    float* dst = output.Data<float>();
    for (size_t token = 0; token < token_count; ++token) {
        const size_t out_base = token * fused_embedding_dim_;
        const size_t word_base = token * word_embedding_dim_;
        const size_t pos_base = token * pos_embedding_dim_;
        for (size_t dim = 0; dim < word_embedding_dim_; ++dim) {
            dst[out_base + dim] = word_data[word_base + dim];
        }
        for (size_t dim = 0; dim < pos_embedding_dim_; ++dim) {
            dst[out_base + word_embedding_dim_ + dim] =
                pos_data[pos_base + dim];
        }
    }
    return output;
}

Tensor SequenceFeatureFusionModule::Backward(const Tensor& grad_output) {
    if (input_shape_.size() != 3) {
        throw std::runtime_error(
            "SequenceFeatureFusionModule: Backward called before Forward");
    }
    const auto& grad_shape = grad_output.Shape();
    if (grad_shape.size() != 3 ||
        grad_shape[0] != input_shape_[0] ||
        grad_shape[1] != input_shape_[1] ||
        grad_shape[2] != fused_embedding_dim_) {
        throw std::runtime_error(
            "SequenceFeatureFusionModule: grad_output shape mismatch");
    }

    const size_t batch = input_shape_[0];
    const size_t seq_len = input_shape_[1];
    const size_t token_count = batch * seq_len;
    Tensor word_grad({batch, seq_len, word_embedding_dim_}, DataType::Float32);
    Tensor pos_grad({batch, seq_len, pos_embedding_dim_}, DataType::Float32);

    const float* src = grad_output.Data<float>();
    float* word_dst = word_grad.Data<float>();
    float* pos_dst = pos_grad.Data<float>();
    for (size_t token = 0; token < token_count; ++token) {
        const size_t in_base = token * fused_embedding_dim_;
        const size_t word_base = token * word_embedding_dim_;
        const size_t pos_base = token * pos_embedding_dim_;
        for (size_t dim = 0; dim < word_embedding_dim_; ++dim) {
            word_dst[word_base + dim] = src[in_base + dim];
        }
        for (size_t dim = 0; dim < pos_embedding_dim_; ++dim) {
            pos_dst[pos_base + dim] =
                src[in_base + word_embedding_dim_ + dim];
        }
    }

    word_embedding_.Backward(word_grad);
    pos_embedding_.Backward(pos_grad);
    return Tensor();
}

std::map<std::string, Tensor> SequenceFeatureFusionModule::GetParameters() {
    std::map<std::string, Tensor> out;
    AppendPrefixedParams(out, word_embedding_.GetParameters(), "word.");
    AppendPrefixedParams(out, pos_embedding_.GetParameters(), "pos.");
    return out;
}

void SequenceFeatureFusionModule::SetParameters(
    const std::map<std::string, Tensor>& params) {
    word_embedding_.SetParameters(FilterPrefixedParams(params, "word."));
    pos_embedding_.SetParameters(FilterPrefixedParams(params, "pos."));
}

std::map<std::string, Tensor> SequenceFeatureFusionModule::GetGradients() {
    std::map<std::string, Tensor> out;
    AppendPrefixedParams(out, word_embedding_.GetGradients(), "word.");
    AppendPrefixedParams(out, pos_embedding_.GetGradients(), "pos.");
    return out;
}

std::string SequenceFeatureFusionModule::GetName() const {
    return "SequenceFeatureFusion(word=" +
           std::to_string(word_embedding_dim_) + ", pos=" +
           std::to_string(pos_embedding_dim_) + ")";
}

// ============================================================================
// PositionalEncodingModule Implementation
// ============================================================================

PositionalEncodingModule::PositionalEncodingModule(
    size_t d_model,
    size_t max_sequence_length)
    : d_model_(d_model)
    , max_sequence_length_(max_sequence_length)
{
    if (d_model_ < 1) d_model_ = 1;
    if (max_sequence_length_ < 1) max_sequence_length_ = 1;
}

Tensor PositionalEncodingModule::Forward(const Tensor& input) {
    return ForwardAt(input, 0);
}

Tensor PositionalEncodingModule::ForwardIncremental(const Tensor& input, size_t position_offset) {
    return ForwardAt(input, position_offset);
}

Tensor PositionalEncodingModule::ForwardAt(const Tensor& input, size_t position_offset) {
    input_cache_ = input.Clone();
    const auto& shape = input.Shape();
    if (input.GetDataType() != DataType::Float32 ||
        shape.size() != 3 ||
        shape[0] == 0 || shape[1] == 0 || shape[2] != d_model_) {
        throw std::runtime_error(
            "PositionalEncodingModule: input must be Float32 [batch, seq_len, d_model]");
    }
    if (shape[1] + position_offset > max_sequence_length_) {
        throw std::runtime_error(
            "PositionalEncodingModule: sequence length exceeds max_sequence_length");
    }

#ifdef CYXWIZ_HAS_ARRAYFIRE
    try {
        // Compute one sinusoidal table for this sequence on the selected
        // backend. Preserve the native formula's double intermediate and
        // Float32 encoding before adding it to the model activations.
        const af::dim4 dims(1, static_cast<dim_t>(shape[1]), static_cast<dim_t>(d_model_));
        const af::array position = af::range(dims, 1, f64) + static_cast<double>(position_offset);
        const af::array dimension = af::range(dims, 2, f64);
        const af::array angle = position / af::pow(10000.0, 2.0 * af::floor(dimension / 2.0) /
                                                           static_cast<double>(d_model_));
        const af::array encoding = af::select(af::mod(dimension, 2.0) == 0.0,
                                               af::sin(angle), af::cos(angle)).as(f32);
        const af::array output = input.GetSemanticArray() +
            af::tile(encoding, af::dim4(static_cast<dim_t>(shape[0])));
        output.eval(); // Finish the lazy table/addition inside its fallback boundary.
        return Tensor::FromSemanticArray(output, shape);
    } catch (const af::exception& e) {
        ThrowIfArrayFireNativeCpuFallbackForbidden(
            "PositionalEncodingModule::Forward", ClassifyArrayFireBackendFallbackReason(e.what()), e.what(),
            BuildArrayFireBackendFallbackContext(BuildTensorShapeContext("input", shape)));
    }
#else
    ThrowIfArrayFireNativeCpuFallbackForbidden(
        "PositionalEncodingModule::Forward", BackendFallbackReason::UnsupportedOperation,
        "ArrayFire is not compiled into this backend", BuildTensorShapeContext("input", shape));
#endif
    const ScopedArrayFireHostSyncAttribution attribution(
        ArrayFireHostSyncCategory::LayerCpuPath, "PositionalEncodingModule::Forward");

    Tensor output(shape, DataType::Float32);
    const float* src = input.ReadData<float>();
    float* dst = output.MutableData<float>();
    const size_t batch = shape[0];
    const size_t seq_len = shape[1];

    for (size_t b = 0; b < batch; ++b) {
        for (size_t pos = 0; pos < seq_len; ++pos) {
            for (size_t dim = 0; dim < d_model_; ++dim) {
                const size_t offset = (b * seq_len + pos) * d_model_ + dim;
                const double angle = static_cast<double>(pos + position_offset) /
                    std::pow(10000.0, static_cast<double>(2 * (dim / 2)) /
                                      static_cast<double>(d_model_));
                const double encoded = (dim % 2 == 0)
                    ? std::sin(angle)
                    : std::cos(angle);
                dst[offset] = src[offset] + static_cast<float>(encoded);
            }
        }
    }

    return output;
}

Tensor PositionalEncodingModule::Backward(const Tensor& grad_output) {
    return grad_output.Clone();
}

std::string PositionalEncodingModule::GetName() const {
    return "PositionalEncoding(d_model=" + std::to_string(d_model_) + ")";
}

// ============================================================================
// TiedOutputProjectionModule Implementation
// ============================================================================

TiedOutputProjectionModule::TiedOutputProjectionModule(EmbeddingModule& embedding, bool use_bias)
    : embedding_(embedding), use_bias_(use_bias) {
    if (use_bias_) {
        bias_ = Tensor::Zeros({embedding_.NumEmbeddings()});
        grad_bias_ = Tensor::Zeros({embedding_.NumEmbeddings()});
    }
}

Tensor TiedOutputProjectionModule::Forward(const Tensor& input) {
    const auto& shape = input.Shape();
    const size_t width = embedding_.EmbeddingDim();
    const size_t vocab = embedding_.NumEmbeddings();
    if (input.GetDataType() != DataType::Float32 || shape.size() != 3 || shape[2] != width) {
        throw std::runtime_error("TiedOutputProjection: input must be Float32 [batch, seq_len, embedding_dim]");
    }
    cached_input_ = input;
#ifdef CYXWIZ_HAS_ARRAYFIRE
    try {
        const af::array x = input.GetSemanticArray();
        const dim_t rows = static_cast<dim_t>(shape[0] * shape[1]);
        af::array logits = af::matmul(af::moddims(x, rows, static_cast<dim_t>(width)),
                                      embedding_.GetLayer().GetWeight().GetSemanticArray(), AF_MAT_NONE, AF_MAT_TRANS);
        if (use_bias_) {
            logits = logits + af::tile(af::moddims(bias_.GetSemanticArray(), 1, static_cast<dim_t>(vocab)),
                                       af::dim4(rows));
        }
        logits = af::moddims(logits, af::dim4(x.dims(0), x.dims(1), static_cast<dim_t>(vocab)));
        logits.eval();
        return Tensor::FromSemanticArray(logits, {shape[0], shape[1], vocab});
    } catch (const af::exception& e) {
        ThrowIfArrayFireNativeCpuFallbackForbidden(
            "TiedOutputProjectionModule::Forward", ClassifyArrayFireBackendFallbackReason(e.what()), e.what(),
            BuildArrayFireBackendFallbackContext(BuildTensorShapeContext("input", shape)));
    }
#endif
    throw std::runtime_error("TiedOutputProjection needs the ArrayFire path");
}

Tensor TiedOutputProjectionModule::Backward(const Tensor& grad_output) {
    const auto& shape = cached_input_.Shape();
    const size_t width = embedding_.EmbeddingDim();
    const size_t vocab = embedding_.NumEmbeddings();
    if (grad_output.Shape() != std::vector<size_t>{shape[0], shape[1], vocab}) {
        throw std::runtime_error("TiedOutputProjection: backward shape does not match forward");
    }
#ifdef CYXWIZ_HAS_ARRAYFIRE
    try {
        const dim_t rows = static_cast<dim_t>(shape[0] * shape[1]);
        const af::array dy = af::moddims(grad_output.GetSemanticArray(), rows, static_cast<dim_t>(vocab));
        const af::array x = af::moddims(cached_input_.GetSemanticArray(), rows, static_cast<dim_t>(width));
        const af::array weight = embedding_.GetLayer().GetWeight().GetSemanticArray();
        af::array dweight = af::matmul(dy, x, AF_MAT_TRANS);  // [vocab, width]
        af::array dx = af::moddims(af::matmul(dy, weight), cached_input_.GetSemanticArray().dims());
        dweight.eval();
        dx.eval();
        if (use_bias_) {
            af::array db = af::flat(af::sum(dy, 0));
            db.eval();
            grad_bias_ = Tensor::FromSemanticArray(db, {vocab});
        }
        embedding_.GetLayer().AddPendingWeightGradient(Tensor::FromSemanticArray(dweight, {vocab, width}));
        return Tensor::FromSemanticArray(dx, shape);
    } catch (const af::exception& e) {
        ThrowIfArrayFireNativeCpuFallbackForbidden(
            "TiedOutputProjectionModule::Backward", ClassifyArrayFireBackendFallbackReason(e.what()), e.what(),
            BuildArrayFireBackendFallbackContext(BuildTensorShapeContext("grad_output", grad_output.Shape())));
    }
#endif
    throw std::runtime_error("TiedOutputProjection needs the ArrayFire path");
}

std::map<std::string, Tensor> TiedOutputProjectionModule::GetParameters() {
    if (!use_bias_) return {};
    return {{"bias", bias_}};
}

void TiedOutputProjectionModule::SetParameters(const std::map<std::string, Tensor>& params) {
    const auto it = params.find("bias");
    if (use_bias_ && it != params.end()) {
        if (it->second.Shape() != std::vector<size_t>{embedding_.NumEmbeddings()}) {
            throw std::invalid_argument("TiedOutputProjection bias must be [num_embeddings]");
        }
        bias_ = it->second;
    }
}

std::map<std::string, Tensor> TiedOutputProjectionModule::GetGradients() {
    if (!use_bias_) return {};
    return {{"bias", grad_bias_}};
}

std::string TiedOutputProjectionModule::GetName() const {
    return "TiedOutputProjection(" + std::to_string(embedding_.EmbeddingDim()) + " -> " +
           std::to_string(embedding_.NumEmbeddings()) + (use_bias_ ? ", bias" : "") + ")";
}

// ============================================================================
// LearnedPositionalEmbeddingModule Implementation
// ============================================================================

LearnedPositionalEmbeddingModule::LearnedPositionalEmbeddingModule(size_t d_model,
                                                                   size_t max_sequence_length)
    : d_model_(d_model), max_sequence_length_(max_sequence_length) {
    if (d_model_ < 1 || max_sequence_length_ < 1) {
        throw std::invalid_argument("LearnedPositionalEmbedding needs positive d_model and max_sequence_length");
    }
    const std::vector<size_t> shape{max_sequence_length_, d_model_};
#ifdef CYXWIZ_HAS_ARRAYFIRE
    try {
        af::array w = af::randn(af::dim4(static_cast<dim_t>(max_sequence_length_), static_cast<dim_t>(d_model_))) * 0.02f;
        w.eval();
        weight_ = Tensor::FromSemanticArray(w, shape);
        grad_weight_ = Tensor::Zeros(shape);
        return;
    } catch (const af::exception& e) {
        ThrowIfArrayFireNativeCpuFallbackForbidden(
            "LearnedPositionalEmbeddingModule::LearnedPositionalEmbeddingModule",
            ClassifyArrayFireBackendFallbackReason(e.what()), e.what(),
            BuildArrayFireBackendFallbackContext("max_sequence_length=" + std::to_string(max_sequence_length_) +
                                                 "; d_model=" + std::to_string(d_model_)));
    }
#endif
    std::mt19937 gen(std::random_device{}());
    std::normal_distribution<float> normal(0.0f, 0.02f);
    weight_ = Tensor(shape, DataType::Float32);
    float* w = weight_.MutableData<float>();
    for (size_t i = 0; i < max_sequence_length_ * d_model_; ++i) w[i] = normal(gen);
    grad_weight_ = Tensor::Zeros(shape);
}

Tensor LearnedPositionalEmbeddingModule::Forward(const Tensor& input) {
    return ForwardAt(input, 0);
}

Tensor LearnedPositionalEmbeddingModule::ForwardIncremental(const Tensor& input, size_t position_offset) {
    return ForwardAt(input, position_offset);
}

Tensor LearnedPositionalEmbeddingModule::ForwardAt(const Tensor& input, size_t position_offset) {
    const auto& shape = input.Shape();
    if (input.GetDataType() != DataType::Float32 || shape.size() != 3 ||
        shape[0] == 0 || shape[1] == 0 || shape[2] != d_model_) {
        throw std::runtime_error("LearnedPositionalEmbedding: input must be Float32 [batch, seq_len, d_model]");
    }
    if (shape[1] + position_offset > max_sequence_length_) {
        throw std::runtime_error("LearnedPositionalEmbedding: sequence length exceeds max_sequence_length");
    }
    cached_seq_len_ = shape[1];
#ifdef CYXWIZ_HAS_ARRAYFIRE
    try {
        const dim_t seq = static_cast<dim_t>(shape[1]);
        const double first = static_cast<double>(position_offset);
        const af::array table = weight_.GetSemanticArray()(af::seq(first, first + static_cast<double>(seq - 1)), af::span);
        const af::array output = input.GetSemanticArray() +
            af::tile(af::moddims(table, af::dim4(1, seq, static_cast<dim_t>(d_model_))),
                     af::dim4(static_cast<dim_t>(shape[0])));
        output.eval();
        return Tensor::FromSemanticArray(output, shape);
    } catch (const af::exception& e) {
        ThrowIfArrayFireNativeCpuFallbackForbidden(
            "LearnedPositionalEmbeddingModule::Forward", ClassifyArrayFireBackendFallbackReason(e.what()), e.what(),
            BuildArrayFireBackendFallbackContext(BuildTensorShapeContext("input", shape)));
    }
#endif
    const ScopedArrayFireHostSyncAttribution attribution(
        ArrayFireHostSyncCategory::LayerCpuPath, "LearnedPositionalEmbeddingModule::Forward");
    Tensor output(shape, DataType::Float32);
    const float* src = input.ReadData<float>();
    const float* w = weight_.ReadData<float>();
    float* dst = output.MutableData<float>();
    for (size_t b = 0; b < shape[0]; ++b)
        for (size_t p = 0; p < shape[1]; ++p)
            for (size_t d = 0; d < d_model_; ++d) {
                const size_t offset = (b * shape[1] + p) * d_model_ + d;
                dst[offset] = src[offset] + w[(p + position_offset) * d_model_ + d];
            }
    return output;
}

Tensor LearnedPositionalEmbeddingModule::Backward(const Tensor& grad_output) {
    const auto& shape = grad_output.Shape();
    if (shape.size() != 3 || shape[1] != cached_seq_len_ || shape[2] != d_model_) {
        throw std::runtime_error("LearnedPositionalEmbedding: backward shape does not match forward");
    }
    const std::vector<size_t> weight_shape{max_sequence_length_, d_model_};
#ifdef CYXWIZ_HAS_ARRAYFIRE
    try {
        const dim_t seq = static_cast<dim_t>(shape[1]);
        const af::array used = af::moddims(af::sum(grad_output.GetSemanticArray(), 0),
                                           af::dim4(seq, static_cast<dim_t>(d_model_)));
        af::array grad = af::constant(0.0f, af::dim4(static_cast<dim_t>(max_sequence_length_),
                                                    static_cast<dim_t>(d_model_)));
        grad(af::seq(0, static_cast<double>(seq - 1)), af::span) = used;
        grad.eval();
        grad_weight_ = Tensor::FromSemanticArray(grad, weight_shape);
        return grad_output;
    } catch (const af::exception& e) {
        ThrowIfArrayFireNativeCpuFallbackForbidden(
            "LearnedPositionalEmbeddingModule::Backward", ClassifyArrayFireBackendFallbackReason(e.what()), e.what(),
            BuildArrayFireBackendFallbackContext(BuildTensorShapeContext("grad_output", shape)));
    }
#endif
    const ScopedArrayFireHostSyncAttribution attribution(
        ArrayFireHostSyncCategory::LayerCpuPath, "LearnedPositionalEmbeddingModule::Backward");
    grad_weight_ = Tensor::Zeros(weight_shape);
    float* g = grad_weight_.MutableData<float>();
    const float* dy = grad_output.ReadData<float>();
    for (size_t b = 0; b < shape[0]; ++b)
        for (size_t p = 0; p < shape[1]; ++p)
            for (size_t d = 0; d < d_model_; ++d)
                g[p * d_model_ + d] += dy[(b * shape[1] + p) * d_model_ + d];
    return grad_output;
}

std::map<std::string, Tensor> LearnedPositionalEmbeddingModule::GetParameters() {
    return {{"weight", weight_}};
}

void LearnedPositionalEmbeddingModule::SetParameters(const std::map<std::string, Tensor>& params) {
    const auto it = params.find("weight");
    if (it == params.end()) return;
    if (it->second.GetDataType() != DataType::Float32 ||
        it->second.Shape() != std::vector<size_t>{max_sequence_length_, d_model_}) {
        throw std::invalid_argument("LearnedPositionalEmbedding weight must be Float32 [max_sequence_length, d_model]");
    }
    weight_ = it->second;
}

std::map<std::string, Tensor> LearnedPositionalEmbeddingModule::GetGradients() {
    return {{"weight", grad_weight_}};
}

std::string LearnedPositionalEmbeddingModule::GetName() const {
    return "LearnedPositionalEmbedding(max_len=" + std::to_string(max_sequence_length_) +
           ", d_model=" + std::to_string(d_model_) + ")";
}

} // namespace cyxwiz
