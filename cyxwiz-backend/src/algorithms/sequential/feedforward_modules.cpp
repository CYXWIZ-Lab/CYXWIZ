#include <cyxwiz/sequential.h>
#include "../arrayfire_backend_utils.h"
#include <arrayfire.h>
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

    const DataType dtype = input.GetDataType();
    if (dtype != DataType::Float32 && dtype != DataType::Int32 && dtype != DataType::Int64) {
        throw std::runtime_error(
            "EmbeddingModule: input token ids must be Float32, Int32, or Int64");
    }
    Tensor int_input;
    try {
        // Ids are checked on the device: finite exact integers within the
        // vocabulary. One scalar comes back; the offending id is looked up
        // only when the check fails.
        const af::array ids = input.GetRowMajorFlatArray();
        const double vocab_max = static_cast<double>(num_embeddings_) - 1.0;
        af::array invalid = (ids < 0.0) || (ids > vocab_max);
        if (dtype == DataType::Float32) {
            invalid = invalid || af::isNaN(ids) || af::isInf(ids) || (af::trunc(ids) != ids);
        }
        const af::array first_bad = af::where(af::flat(invalid));
        if (first_bad.elements() > 0) {
            const Tensor position = Tensor::FromRowMajorFlatArray(first_bad(0), {1});
            const unsigned at = position.ReadData<unsigned>()[0];
            const Tensor value = Tensor::FromRowMajorFlatArray(ids(at).as(f64), {1});
            throw std::runtime_error(
                "EmbeddingModule: token id " + std::to_string(value.ReadData<double>()[0]) +
                " at position " + std::to_string(at) + " is not an integer in the vocabulary range [0, " +
                std::to_string(num_embeddings_) + ")");
        }
        int_input = Tensor::FromRowMajorFlatArray(ids.as(s32), shape);
    } catch (const af::exception& e) {
        throw std::runtime_error(std::string("EmbeddingModule::Forward failed on the ArrayFire device: ") + e.what());
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
    // Column 0 holds word ids, column 1 POS ids; EmbeddingModule checks them.
    const Tensor word_tensor = input.Slice(2, 0, 1).Reshape({batch, seq_len});
    const Tensor pos_tensor = input.Slice(2, 1, 2).Reshape({batch, seq_len});
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

    return Tensor::Cat({word_emb, pos_emb}, 2);
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

    const int word = static_cast<int>(word_embedding_dim_);
    const int fused = static_cast<int>(fused_embedding_dim_);
    const Tensor word_grad = grad_output.Slice(2, 0, word);
    const Tensor pos_grad = grad_output.Slice(2, word, fused);

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
        throw std::runtime_error(std::string("PositionalEncodingModule::Forward failed on the ArrayFire device: ") + e.what());
    }
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
        throw std::runtime_error(std::string("TiedOutputProjectionModule::Forward failed on the ArrayFire device: ") + e.what());
    }
}

Tensor TiedOutputProjectionModule::Backward(const Tensor& grad_output) {
    const auto& shape = cached_input_.Shape();
    const size_t width = embedding_.EmbeddingDim();
    const size_t vocab = embedding_.NumEmbeddings();
    if (grad_output.Shape() != std::vector<size_t>{shape[0], shape[1], vocab}) {
        throw std::runtime_error("TiedOutputProjection: backward shape does not match forward");
    }
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
        throw std::runtime_error(std::string("TiedOutputProjectionModule::Backward failed on the ArrayFire device: ") + e.what());
    }
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
    try {
        af::array w = af::randn(af::dim4(static_cast<dim_t>(max_sequence_length_), static_cast<dim_t>(d_model_))) * 0.02f;
        w.eval();
        weight_ = Tensor::FromSemanticArray(w, shape);
        grad_weight_ = Tensor::Zeros(shape);
        return;
    } catch (const af::exception& e) {
        throw std::runtime_error(std::string("LearnedPositionalEmbeddingModule::LearnedPositionalEmbeddingModule failed on the ArrayFire device: ") + e.what());
    }
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
        throw std::runtime_error(std::string("LearnedPositionalEmbeddingModule::Forward failed on the ArrayFire device: ") + e.what());
    }
}

Tensor LearnedPositionalEmbeddingModule::Backward(const Tensor& grad_output) {
    const auto& shape = grad_output.Shape();
    if (shape.size() != 3 || shape[1] != cached_seq_len_ || shape[2] != d_model_) {
        throw std::runtime_error("LearnedPositionalEmbedding: backward shape does not match forward");
    }
    const std::vector<size_t> weight_shape{max_sequence_length_, d_model_};
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
        throw std::runtime_error(std::string("LearnedPositionalEmbeddingModule::Backward failed on the ArrayFire device: ") + e.what());
    }
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
