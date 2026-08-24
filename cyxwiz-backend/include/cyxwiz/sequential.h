#pragma once

#include "api_export.h"
#include "tensor.h"
#include "layer.h"
#include "activation.h"
#include "optimizer.h"
#include "layers/linear.h"
#include "layers/normalization.h"
#include "layers/attention.h"
#include "activations/relu.h"
#include "activations/sigmoid.h"
#include "activations/tanh.h"
#include <vector>
#include <memory>
#include <string>
#include <variant>
#include <functional>

namespace cyxwiz {

/**
 * @brief Module types that can be added to a SequentialModel
 */
enum class ModuleType {
    Linear,
    ReLU,
    Sigmoid,
    Tanh,
    Softmax,
    Dropout,
    BatchNorm,
    Flatten,
    LeakyReLU,
    ELU,
    GELU,
    Swish,
    Mish,
    Embedding,
    PositionalEncoding,
    TransformerEncoder,
    TransformerDecoder,
    LayerNorm,
    MultiHeadAttention
};

enum class TensorUnaryOp {
    Abs,
    Exp,
    Log,
    Sqrt,
    Sign,
    Pow,
    Clip
};

enum class TensorReductionOp {
    Sum,
    Mean,
    Max,
    Min,
    Prod,
    Var,
    Std
};

enum class TensorShapeOp {
    BroadcastTo,
    Expand,
    IndexSelect
};

enum class TensorMaskOp {
    CompareGreater,
    CompareGreaterEqual,
    CompareLess,
    CompareLessEqual,
    CompareEqual,
    CompareNotEqual,
    LogicalNot
};

/**
 * @brief Canonical model-facing training unit.
 *
 * Modules are the objects owned by SequentialModel. They provide the stable
 * runtime contract for forward/backward execution, parameter and gradient
 * collection, freezing, training/eval mode, and serialization metadata. A
 * module may wrap a lower-level Layer primitive, but model builders and graph
 * training paths should target Module/SequentialModel rather than direct Layer
 * ownership.
 */
class CYXWIZ_API Module {
public:
    virtual ~Module() = default;

    /**
     * @brief Forward pass
     * @param input Input tensor
     * @return Output tensor
     */
    virtual Tensor Forward(const Tensor& input) = 0;

    /**
     * @brief Backward pass
     * @param grad_output Gradient from next layer
     * @return Gradient w.r.t input
     */
    virtual Tensor Backward(const Tensor& grad_output) = 0;

    /**
     * @brief Get trainable parameters
     * @return Map of parameter name -> tensor (empty if no parameters)
     */
    virtual std::map<std::string, Tensor> GetParameters() { return {}; }

    /**
     * @brief Set trainable parameters
     * @param params Map of parameter name -> tensor
     */
    virtual void SetParameters(const std::map<std::string, Tensor>& /*params*/) {}

    /**
     * @brief Get parameter gradients
     * @return Map of parameter name -> gradient tensor (empty if no parameters)
     */
    virtual std::map<std::string, Tensor> GetGradients() { return {}; }

    /**
     * @brief Check if module has trainable parameters
     */
    virtual bool HasParameters() const { return false; }

    /**
     * @brief Get module name for debugging
     */
    virtual std::string GetName() const = 0;

    /**
     * @brief Set training mode (affects Dropout, BatchNorm)
     */
    virtual void SetTraining(bool training) { is_training_ = training; }

    bool IsTraining() const { return is_training_; }

    /**
     * @brief Set trainable state for transfer learning
     * @param trainable If false, parameters won't be updated during training
     */
    void SetTrainable(bool trainable) { trainable_ = trainable; }

    /**
     * @brief Check if module is trainable
     */
    bool IsTrainable() const { return trainable_; }

    /**
     * @brief Freeze the module (disable parameter updates)
     */
    void Freeze() { trainable_ = false; }

    /**
     * @brief Unfreeze the module (enable parameter updates)
     */
    void Unfreeze() { trainable_ = true; }

protected:
    bool is_training_ = true;
    bool trainable_ = true;  // For transfer learning - frozen layers won't update
    Tensor input_cache_;  // Cached input for backward pass
};

/**
 * @brief Wrapper for LinearLayer
 */
class CYXWIZ_API LinearModule : public Module {
public:
    LinearModule(size_t in_features, size_t out_features, bool use_bias = true);

    Tensor Forward(const Tensor& input) override;
    Tensor Backward(const Tensor& grad_output) override;
    std::map<std::string, Tensor> GetParameters() override;
    void SetParameters(const std::map<std::string, Tensor>& params) override;
    std::map<std::string, Tensor> GetGradients() override;
    bool HasParameters() const override { return true; }
    std::string GetName() const override;

private:
    std::unique_ptr<LinearLayer> layer_;
    size_t in_features_;
    size_t out_features_;
};

/**
 * @brief Wrapper for EmbeddingLayer — token-ID lookup table.
 *
 * Translates [batch, seq_len] integer indices into
 * [batch, seq_len, embedding_dim] dense float vectors. The underlying
 * `cyxwiz::EmbeddingLayer` expects an int32 Tensor, but CyxWiz's training
 * pipeline carries all tensors as float32 (IBatcher contract). This
 * wrapper casts float → int32 on each forward pass so text pipelines can
 * drop an Embedding node between DataLoader and the MLP head without
 * changing the batcher type.
 */
// Applies a Dense projection independently to each timestep:
// [batch, seq_len, in_features] -> [batch, seq_len, out_features].
class CYXWIZ_API TimeDistributedDenseModule : public Module {
public:
    TimeDistributedDenseModule(size_t in_features,
                               size_t out_features,
                               bool use_bias = true);

    Tensor Forward(const Tensor& input) override;
    Tensor Backward(const Tensor& grad_output) override;
    std::map<std::string, Tensor> GetParameters() override;
    void SetParameters(const std::map<std::string, Tensor>& params) override;
    std::map<std::string, Tensor> GetGradients() override;
    bool HasParameters() const override { return true; }
    std::string GetName() const override;

private:
    LinearModule linear_;
    size_t in_features_;
    size_t out_features_;
    std::vector<size_t> input_shape_;
};

// Token embedding lookup: [batch, seq_len] -> [batch, seq_len, embedding_dim].
class CYXWIZ_API EmbeddingModule : public Module {
public:
    EmbeddingModule(size_t num_embeddings, size_t embedding_dim,
                    int padding_idx = -1, float max_norm = 0.0f);

    Tensor Forward(const Tensor& input) override;
    Tensor Backward(const Tensor& grad_output) override;
    std::map<std::string, Tensor> GetParameters() override;
    void SetParameters(const std::map<std::string, Tensor>& params) override;
    std::map<std::string, Tensor> GetGradients() override;
    void LoadPretrainedWeights(const Tensor& weights, bool freeze = false);
    bool HasParameters() const override { return true; }
    std::string GetName() const override;

private:
    std::unique_ptr<EmbeddingLayer> layer_;
    size_t num_embeddings_;
    size_t embedding_dim_;
    int padding_idx_;
    float max_norm_;
};

// Focused token-feature fusion for sequence taggers.
// Input is [batch, seq_len, 2] ids: channel 0 = word, channel 1 = POS.
// Output is [batch, seq_len, word_embedding_dim + pos_embedding_dim].
class CYXWIZ_API SequenceFeatureFusionModule : public Module {
public:
    SequenceFeatureFusionModule(size_t word_num_embeddings,
                                size_t word_embedding_dim,
                                size_t pos_num_embeddings,
                                size_t pos_embedding_dim,
                                int word_padding_idx = -1,
                                int pos_padding_idx = -1);

    Tensor Forward(const Tensor& input) override;
    Tensor Backward(const Tensor& grad_output) override;
    std::map<std::string, Tensor> GetParameters() override;
    void SetParameters(const std::map<std::string, Tensor>& params) override;
    std::map<std::string, Tensor> GetGradients() override;
    bool HasParameters() const override { return true; }
    std::string GetName() const override;

private:
    EmbeddingModule word_embedding_;
    EmbeddingModule pos_embedding_;
    size_t word_embedding_dim_;
    size_t pos_embedding_dim_;
    size_t fused_embedding_dim_;
    std::vector<size_t> input_shape_;
};

/**
 * @brief Parameter-free sinusoidal positional encoding.
 *
 * Consumes and returns `[batch, seq_len, d_model]` tensors. Gradients pass
 * through unchanged because the encoding is constant.
 */
class CYXWIZ_API PositionalEncodingModule : public Module {
public:
    explicit PositionalEncodingModule(size_t d_model,
                                      size_t max_sequence_length = 512);

    Tensor Forward(const Tensor& input) override;
    Tensor Backward(const Tensor& grad_output) override;
    std::string GetName() const override;

private:
    size_t d_model_;
    size_t max_sequence_length_;
};

/**
 * @brief Wrapper for LSTMLayer — recurrent sequence processor.
 *
 * Consumes `[batch, seq_len, input_size]` float tensors and produces
 * either:
 *   - `[batch, hidden_size * num_directions]` when `return_sequences`
 *     is false (default — the "last timestep" reduction every
 *     classification head expects), OR
 *   - `[batch, seq_len, hidden_size * num_directions]` when
 *     `return_sequences` is true (Keras / PyTorch convention; needed
 *     for stacked LSTM / sequence-to-sequence heads).
 *
 * The underlying `cyxwiz::LSTMLayer` always returns the full sequence
 * output; this wrapper slices out the last timestep per-sample when
 * `return_sequences=false` so a Dense classification head can sit
 * directly after the LSTM without an intervening Flatten.
 *
 * For `Backward` with `return_sequences=false`, the incoming
 * `[batch, hidden]` gradient is re-expanded to
 * `[batch, seq_len, hidden]` with zeros everywhere except the last
 * timestep — which is the correct dL/d(full_output) because only the
 * last timestep affected the downstream loss.
 */
class CYXWIZ_API LSTMModule : public Module {
public:
    LSTMModule(size_t input_size, size_t hidden_size,
               size_t num_layers = 1,
               bool bidirectional = false,
               bool return_sequences = false);

    Tensor Forward(const Tensor& input) override;
    Tensor Backward(const Tensor& grad_output) override;
    std::map<std::string, Tensor> GetParameters() override;
    void SetParameters(const std::map<std::string, Tensor>& params) override;
    std::map<std::string, Tensor> GetGradients() override;
    bool HasParameters() const override { return true; }
    std::string GetName() const override;

private:
    std::unique_ptr<LSTMLayer> layer_;
    size_t input_size_;
    size_t hidden_size_;
    size_t num_layers_;
    bool bidirectional_;
    bool return_sequences_;
    // Cached full LSTM output shape [batch, seq_len, hidden*dirs] so
    // Backward can re-expand the last-step gradient when
    // return_sequences_ is false.
    std::vector<size_t> last_full_output_shape_;
};

/**
 * @brief Module wrapper around GRULayer.
 *
 * Mirrors LSTMModule: the underlying GRULayer always returns the full
 * sequence output [batch, seq_len, hidden_size * num_directions]. When
 * `return_sequences == false`, this wrapper slices out the final
 * timestep so the Module yields [batch, hidden_size * num_directions]
 * — the shape a downstream Dense classifier expects without an
 * intervening Flatten. Backward re-expands the [batch, hidden] gradient
 * back to the full [batch, seq_len, hidden] shape with zeros at every
 * non-terminal step before delegating to GRULayer::Backward.
 */
class CYXWIZ_API GRUModule : public Module {
public:
    GRUModule(size_t input_size, size_t hidden_size,
              size_t num_layers = 1,
              bool bidirectional = false,
              bool return_sequences = false);

    Tensor Forward(const Tensor& input) override;
    Tensor Backward(const Tensor& grad_output) override;
    void SetTraining(bool training) override;
    std::map<std::string, Tensor> GetParameters() override;
    void SetParameters(const std::map<std::string, Tensor>& params) override;
    std::map<std::string, Tensor> GetGradients() override;
    bool HasParameters() const override { return true; }
    std::string GetName() const override;

private:
    std::unique_ptr<GRULayer> layer_;
    std::vector<std::unique_ptr<GRULayer>> forward_layers_;
    std::vector<std::unique_ptr<GRULayer>> reverse_layers_;
    size_t input_size_;
    size_t hidden_size_;
    size_t num_layers_;
    bool bidirectional_;
    bool return_sequences_;
    bool split_bidirectional_path_ = false;
    std::vector<size_t> last_full_output_shape_;
};

/**
 * @brief Wrapper around TransformerEncoderLayer.
 *
 * Consumes and returns `[batch, seq_len, d_model]` tensors. Use a
 * Flatten or pooling module after this wrapper before a Dense
 * classification head.
 */
class CYXWIZ_API TransformerEncoderModule : public Module {
public:
    TransformerEncoderModule(size_t d_model, size_t num_heads,
                             size_t dim_feedforward = 2048,
                             float dropout = 0.1f,
                             bool norm_first = false);

    Tensor Forward(const Tensor& input) override;
    Tensor Backward(const Tensor& grad_output) override;
    void SetTraining(bool training) override;
    std::map<std::string, Tensor> GetParameters() override;
    void SetParameters(const std::map<std::string, Tensor>& params) override;
    std::map<std::string, Tensor> GetGradients() override;
    bool HasParameters() const override { return true; }
    std::string GetName() const override;

private:
    std::unique_ptr<TransformerEncoderLayer> layer_;
    size_t d_model_;
    size_t num_heads_;
    size_t dim_feedforward_;
    float dropout_;
    bool norm_first_;
};

/**
 * @brief Wrapper around TransformerDecoderLayer.
 *
 * Consumes and returns `[batch, seq_len, d_model]` tensors. The wrapped
 * decoder layer owns causal self-attention masking for single-input forward.
 */
class CYXWIZ_API TransformerDecoderModule : public Module {
public:
    TransformerDecoderModule(size_t d_model, size_t num_heads,
                             size_t dim_feedforward = 2048,
                             float dropout = 0.1f,
                             bool norm_first = false);

    Tensor Forward(const Tensor& input) override;
    Tensor Backward(const Tensor& grad_output) override;
    void SetTraining(bool training) override;
    std::map<std::string, Tensor> GetParameters() override;
    void SetParameters(const std::map<std::string, Tensor>& params) override;
    std::map<std::string, Tensor> GetGradients() override;
    bool HasParameters() const override { return true; }
    std::string GetName() const override;

private:
    std::unique_ptr<TransformerDecoderLayer> layer_;
    size_t d_model_;
    size_t num_heads_;
    size_t dim_feedforward_;
    float dropout_;
    bool norm_first_;
};

/**
 * @brief Wrapper around MultiHeadAttentionLayer for self-attention.
 *
 * Consumes and returns `[batch, seq_len, embed_dim]` tensors. Multi-input
 * cross-attention is intentionally not exposed through SequentialModel yet;
 * graph/compiler support must define that contract first.
 */
class CYXWIZ_API MultiHeadAttentionModule : public Module {
public:
    MultiHeadAttentionModule(size_t embed_dim, size_t num_heads,
                             float dropout = 0.0f,
                             bool use_bias = true);

    Tensor Forward(const Tensor& input) override;
    Tensor Backward(const Tensor& grad_output) override;
    void SetTraining(bool training) override;
    std::map<std::string, Tensor> GetParameters() override;
    void SetParameters(const std::map<std::string, Tensor>& params) override;
    std::map<std::string, Tensor> GetGradients() override;
    bool HasParameters() const override { return true; }
    std::string GetName() const override;

private:
    std::unique_ptr<MultiHeadAttentionLayer> layer_;
    size_t embed_dim_;
    size_t num_heads_;
    float dropout_;
    bool use_bias_;
};

/**
 * @brief Wrapper for ReLU activation
 */
class CYXWIZ_API ReLUModule : public Module {
public:
    ReLUModule();

    Tensor Forward(const Tensor& input) override;
    Tensor Backward(const Tensor& grad_output) override;
    std::string GetName() const override { return "ReLU"; }

private:
    std::unique_ptr<ReLU> activation_;
};

/**
 * @brief Wrapper for Sigmoid activation
 */
class CYXWIZ_API SigmoidModule : public Module {
public:
    SigmoidModule();

    Tensor Forward(const Tensor& input) override;
    Tensor Backward(const Tensor& grad_output) override;
    std::string GetName() const override { return "Sigmoid"; }

private:
    std::unique_ptr<Sigmoid> activation_;
};

/**
 * @brief Wrapper for Tanh activation
 */
class CYXWIZ_API TanhModule : public Module {
public:
    TanhModule();

    Tensor Forward(const Tensor& input) override;
    Tensor Backward(const Tensor& grad_output) override;
    std::string GetName() const override { return "Tanh"; }

private:
    std::unique_ptr<Tanh> activation_;
};

/**
 * @brief Softmax activation module
 */
class CYXWIZ_API SoftmaxModule : public Module {
public:
    SoftmaxModule(int dim = -1);

    Tensor Forward(const Tensor& input) override;
    Tensor Backward(const Tensor& grad_output) override;
    std::string GetName() const override { return "Softmax"; }

private:
    int dim_;
    Tensor output_cache_;  // Cache softmax output for backward
};

/**
 * @brief Dropout module for regularization
 */
class CYXWIZ_API DropoutModule : public Module {
public:
    DropoutModule(float p = 0.5f);

    Tensor Forward(const Tensor& input) override;
    Tensor Backward(const Tensor& grad_output) override;
    std::string GetName() const override;

private:
    float p_;  // Dropout probability
    Tensor mask_;  // Dropout mask for backward
};

/**
 * @brief Flatten module - reshapes input to [batch, features]
 */
class CYXWIZ_API FlattenModule : public Module {
public:
    FlattenModule(int start_dim = 1);

    Tensor Forward(const Tensor& input) override;
    Tensor Backward(const Tensor& grad_output) override;
    std::string GetName() const override { return "Flatten"; }

private:
    int start_dim_;
    std::vector<size_t> original_shape_;  // For backward reshape
    std::vector<size_t> output_shape_;    // Exact gradient contract
    DataType output_dtype_ = DataType::Float32;
};

/**
 * @brief Batch-preserving reshape module.
 *
 * The target shape describes one sample. Forward preserves the leading
 * batch dimension and reshapes [batch, ...] to [batch, target...].
 */
class CYXWIZ_API ReshapeModule : public Module {
public:
    explicit ReshapeModule(std::vector<size_t> target_sample_shape);

    Tensor Forward(const Tensor& input) override;
    Tensor Backward(const Tensor& grad_output) override;
    std::string GetName() const override { return "Reshape"; }

private:
    std::vector<size_t> target_sample_shape_;
    std::vector<size_t> original_shape_;  // For backward reshape
};

/**
 * @brief Batch-preserving permute module.
 *
 * The dims vector describes one sample. Forward preserves the leading
 * batch dimension and permutes only sample dimensions.
 */
class CYXWIZ_API PermuteModule : public Module {
public:
    explicit PermuteModule(std::vector<int> sample_dims);

    Tensor Forward(const Tensor& input) override;
    Tensor Backward(const Tensor& grad_output) override;
    std::string GetName() const override { return "Permute"; }

private:
    std::vector<int> sample_dims_;
    std::vector<int> inverse_sample_dims_;
};

/**
 * @brief Parameter-free elementwise tensor math module.
 */
class CYXWIZ_API TensorUnaryModule : public Module {
public:
    explicit TensorUnaryModule(TensorUnaryOp op,
                               float scalar = 0.0f,
                               float scalar2 = 0.0f);

    Tensor Forward(const Tensor& input) override;
    Tensor Backward(const Tensor& grad_output) override;
    std::string GetName() const override;

private:
    TensorUnaryOp op_;
    float scalar_;
    float scalar2_;
    Tensor output_cache_;
};

/**
 * @brief Batch-preserving tensor reduction module.
 *
 * The dim parameter addresses sample dimensions. dim=-1 reduces all
 * sample dimensions while preserving the leading batch dimension.
 */
class CYXWIZ_API TensorReductionModule : public Module {
public:
    TensorReductionModule(TensorReductionOp op, int dim = -1, bool keepdim = false);

    Tensor Forward(const Tensor& input) override;
    Tensor Backward(const Tensor& grad_output) override;
    std::string GetName() const override;

private:
    TensorReductionOp op_;
    int dim_;
    bool keepdim_;
    std::vector<size_t> original_shape_;
    std::vector<size_t> output_shape_;
    size_t reduced_count_ = 1;
    Tensor output_cache_;
};

/**
 * @brief Batch-preserving tensor shape/index module.
 *
 * The target shape and dim/indices parameters address sample dimensions.
 * The leading batch dimension is preserved automatically.
 */
class CYXWIZ_API TensorShapeModule : public Module {
public:
    explicit TensorShapeModule(TensorShapeOp op,
                               std::vector<size_t> target_shape = {},
                               int dim = 0,
                               std::vector<int> indices = {});

    Tensor Forward(const Tensor& input) override;
    Tensor Backward(const Tensor& grad_output) override;
    std::string GetName() const override;

private:
    TensorShapeOp op_;
    std::vector<size_t> target_shape_;
    int dim_;
    int normalized_dim_ = 0;
    std::vector<int> indices_;
    std::vector<int> normalized_indices_;
    std::vector<size_t> original_shape_;
    std::vector<size_t> padded_input_shape_;
    std::vector<size_t> output_shape_;
    size_t sample_pad_ = 0;
};

/**
 * @brief Non-differentiable scalar compare/logical tensor mask module.
 *
 * This is intentionally single-input only. Multi-input mask/merge/linalg nodes
 * need graph-runtime tensor fan-in before they can be executable layers.
 */
class CYXWIZ_API TensorMaskModule : public Module {
public:
    explicit TensorMaskModule(TensorMaskOp op, float scalar = 0.0f);

    Tensor Forward(const Tensor& input) override;
    Tensor Backward(const Tensor& grad_output) override;
    std::string GetName() const override;

private:
    TensorMaskOp op_;
    float scalar_;
};

/**
 * @brief Wrapper for LeakyReLU activation
 */
class CYXWIZ_API LeakyReLUModule : public Module {
public:
    LeakyReLUModule(float negative_slope = 0.01f);

    Tensor Forward(const Tensor& input) override;
    Tensor Backward(const Tensor& grad_output) override;
    std::string GetName() const override;

private:
    std::unique_ptr<LeakyReLUActivation> activation_;
    float negative_slope_;
};

/**
 * @brief Wrapper for ELU activation
 */
class CYXWIZ_API ELUModule : public Module {
public:
    ELUModule(float alpha = 1.0f);

    Tensor Forward(const Tensor& input) override;
    Tensor Backward(const Tensor& grad_output) override;
    std::string GetName() const override;

private:
    std::unique_ptr<ELUActivation> activation_;
    float alpha_;
};

/**
 * @brief Wrapper for GELU activation
 */
class CYXWIZ_API GELUModule : public Module {
public:
    GELUModule();

    Tensor Forward(const Tensor& input) override;
    Tensor Backward(const Tensor& grad_output) override;
    std::string GetName() const override { return "GELU"; }

private:
    std::unique_ptr<GELUActivation> activation_;
};

/**
 * @brief Wrapper for Swish activation (SiLU)
 */
class CYXWIZ_API SwishModule : public Module {
public:
    SwishModule();

    Tensor Forward(const Tensor& input) override;
    Tensor Backward(const Tensor& grad_output) override;
    std::string GetName() const override { return "Swish"; }

private:
    std::unique_ptr<SwishActivation> activation_;
};

/**
 * @brief Wrapper for Mish activation
 */
class CYXWIZ_API MishModule : public Module {
public:
    MishModule();

    Tensor Forward(const Tensor& input) override;
    Tensor Backward(const Tensor& grad_output) override;
    std::string GetName() const override { return "Mish"; }

private:
    std::unique_ptr<MishActivation> activation_;
};

/**
 * @brief BatchNorm1D module for normalizing activations in MLPs
 * Normalizes across the batch dimension for [batch, features] input
 */
class CYXWIZ_API BatchNormModule : public Module {
public:
    BatchNormModule(size_t num_features, float eps = 1e-5f, float momentum = 0.1f);

    Tensor Forward(const Tensor& input) override;
    Tensor Backward(const Tensor& grad_output) override;
    std::map<std::string, Tensor> GetParameters() override;
    void SetParameters(const std::map<std::string, Tensor>& params) override;
    std::map<std::string, Tensor> GetGradients() override;
    bool HasParameters() const override { return true; }
    std::string GetName() const override;

private:
    size_t num_features_;
    float eps_;
    float momentum_;

    // Learnable parameters
    Tensor gamma_;   // Scale [num_features]
    Tensor beta_;    // Shift [num_features]

    // Running statistics for inference
    Tensor running_mean_;
    Tensor running_var_;

    // Cached for backward pass
    Tensor normalized_;
    Tensor std_inv_;
    Tensor batch_mean_;

    // Gradients
    Tensor grad_gamma_;
    Tensor grad_beta_;
};

/**
 * @brief LayerNorm module for transformer and sequence activations.
 * Normalizes over the last normalized_shape dimensions for each sample.
 */
class CYXWIZ_API LayerNormModule : public Module {
public:
    LayerNormModule(const std::vector<int>& normalized_shape,
                    float eps = 1e-5f,
                    bool elementwise_affine = true);

    Tensor Forward(const Tensor& input) override;
    Tensor Backward(const Tensor& grad_output) override;
    std::map<std::string, Tensor> GetParameters() override;
    void SetParameters(const std::map<std::string, Tensor>& params) override;
    std::map<std::string, Tensor> GetGradients() override;
    bool HasParameters() const override { return elementwise_affine_; }
    std::string GetName() const override;

private:
    std::unique_ptr<LayerNormLayer> layer_;
    std::vector<int> normalized_shape_;
    float eps_;
    bool elementwise_affine_;
};

/**
 * @brief Canonical ordered model container for backend training/inference.
 *
 * SequentialModel owns Module instances, not raw Layer instances. Use it for
 * model-facing training, inference, serialization, distributed wrappers, and
 * Python bindings. Lower-level Layer classes remain available as primitives
 * and implementation details behind modules.
 *
 * Example:
 *   SequentialModel model;
 *   model.Add<LinearModule>(784, 128);
 *   model.Add<ReLUModule>();
 *   model.Add<LinearModule>(128, 10);
 */
class CYXWIZ_API SequentialModel {
public:
    SequentialModel() = default;
    ~SequentialModel() = default;

    // Non-copyable
    SequentialModel(const SequentialModel&) = delete;
    SequentialModel& operator=(const SequentialModel&) = delete;

    // Movable
    SequentialModel(SequentialModel&&) = default;
    SequentialModel& operator=(SequentialModel&&) = default;

    /**
     * @brief Add a module to the sequence
     * @tparam T Module type (LinearModule, ReLUModule, etc.)
     * @tparam Args Constructor arguments for the module
     */
    template<typename T, typename... Args>
    void Add(Args&&... args) {
        modules_.push_back(std::make_unique<T>(std::forward<Args>(args)...));
    }

    /**
     * @brief Add a pre-created module
     */
    void AddModule(std::unique_ptr<Module> module) {
        modules_.push_back(std::move(module));
    }

    /**
     * @brief Forward pass through all layers
     * @param input Input tensor
     * @return Output tensor
     */
    Tensor Forward(const Tensor& input);

    /**
     * @brief Backward pass through all layers (reverse order)
     * @param grad_output Gradient from loss function
     * @return Gradient w.r.t input (usually not needed)
     */
    Tensor Backward(const Tensor& grad_output);

    /**
     * @brief Get all trainable parameters
     * @return Map of "layer_idx.param_name" -> tensor
     */
    std::map<std::string, Tensor> GetParameters();

    /**
     * @brief Set all trainable parameters
     * @param params Map of "layer_idx.param_name" -> tensor
     */
    void SetParameters(const std::map<std::string, Tensor>& params);

    /**
     * @brief Get all parameter gradients
     * @return Map of "layer_idx.param_name" -> gradient tensor
     */
    std::map<std::string, Tensor> GetGradients();

    /**
     * @brief Apply optimizer to all parameters
     * @param optimizer Optimizer to use
     */
    void UpdateParameters(Optimizer* optimizer);

    /**
     * @brief Set training mode for all modules
     */
    void SetTraining(bool training);

    /**
     * @brief Get number of modules
     */
    size_t Size() const { return modules_.size(); }

    /**
     * @brief Get module at index
     */
    Module* GetModule(size_t index) {
        return index < modules_.size() ? modules_[index].get() : nullptr;
    }

    /**
     * @brief Get module at index (const version)
     */
    const Module* GetModule(size_t index) const {
        return index < modules_.size() ? modules_[index].get() : nullptr;
    }

    /**
     * @brief Print model summary
     */
    void Summary() const;

    // ==================== Serialization ====================

    /**
     * @brief Save model to file
     * @param path Base path (will create .json and .bin files)
     * @return true if successful
     */
    bool Save(const std::string& path) const;

    /**
     * @brief Load model weights from file
     * @param path Base path (expects .json and .bin files)
     * @return true if successful
     * @note Model architecture must already be set up before loading
     */
    bool Load(const std::string& path);

    /**
     * @brief Set model name (for metadata)
     */
    void SetName(const std::string& name) { model_name_ = name; }

    /**
     * @brief Get model name
     */
    const std::string& GetName() const { return model_name_; }

    /**
     * @brief Set model description (for metadata)
     */
    void SetDescription(const std::string& desc) { model_description_ = desc; }

    /**
     * @brief Get model description
     */
    const std::string& GetDescription() const { return model_description_; }

    // ==================== Transfer Learning ====================

    /**
     * @brief Freeze a specific layer by index
     * @param layer_idx Index of the layer to freeze
     */
    void FreezeLayer(size_t layer_idx);

    /**
     * @brief Freeze all layers up to (but not including) the given index
     * @param layer_idx First layer that remains trainable
     */
    void FreezeUpTo(size_t layer_idx);

    /**
     * @brief Freeze all layers except the last N layers
     * @param n Number of layers to keep trainable at the end
     */
    void FreezeExceptLast(size_t n);

    /**
     * @brief Unfreeze all layers
     */
    void UnfreezeAll();

    /**
     * @brief Check if a layer is trainable
     * @param layer_idx Index of the layer
     * @return true if the layer is trainable, false if frozen
     */
    bool IsLayerTrainable(size_t layer_idx) const;

private:
    std::vector<std::unique_ptr<Module>> modules_;
    std::vector<Tensor> intermediate_outputs_;  // Cached for backward pass
    std::string model_name_;
    std::string model_description_;
};

/**
 * @brief Factory function to create a module from type enum
 */
CYXWIZ_API std::unique_ptr<Module> CreateModule(
    ModuleType type,
    const std::map<std::string, std::string>& params = {}
);

} // namespace cyxwiz
