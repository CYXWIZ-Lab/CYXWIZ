#pragma once

#include "api_export.h"
#include "tensor.h"
#include <string>
#include <map>
#include <memory>
#include <vector>

namespace cyxwiz {

// ============================================================================
// Base Layer Class
// ============================================================================

class CYXWIZ_API Layer {
public:
    virtual ~Layer() = default;
    virtual Tensor Forward(const Tensor& input) = 0;
    virtual Tensor Backward(const Tensor& grad_output) = 0;
    virtual std::map<std::string, Tensor> GetParameters() = 0;
    virtual void SetParameters(const std::map<std::string, Tensor>& params) = 0;

    // Training mode (affects BatchNorm, Dropout, etc.)
    virtual void SetTraining(bool training) { training_ = training; }
    bool IsTraining() const { return training_; }

    // Layer name for debugging/serialization
    virtual std::string GetName() const { return "Layer"; }

protected:
    bool training_ = true;
    Tensor cached_input_;  // For backward pass
};

// ============================================================================
// Dense (Fully Connected) Layer
// ============================================================================

class CYXWIZ_API DenseLayer : public Layer {
public:
    DenseLayer(int in_features, int out_features, bool use_bias = true);

    Tensor Forward(const Tensor& input) override;
    Tensor Backward(const Tensor& grad_output) override;
    std::map<std::string, Tensor> GetParameters() override;
    void SetParameters(const std::map<std::string, Tensor>& params) override;
    std::string GetName() const override { return "Dense"; }

private:
    int in_features_;
    int out_features_;
    bool use_bias_;

    Tensor weights_;      // [out_features, in_features]
    Tensor bias_;         // [out_features]
    Tensor grad_weights_; // Gradient accumulator
    Tensor grad_bias_;    // Gradient accumulator
};

// ============================================================================
// Conv2D Layer - 2D Convolution using ArrayFire
// ============================================================================

class CYXWIZ_API Conv2DLayer : public Layer {
public:
    /**
     * Create a 2D convolutional layer
     * @param in_channels Number of input channels
     * @param out_channels Number of output channels (filters)
     * @param kernel_size Size of the convolution kernel (assumes square)
     * @param stride Stride of the convolution (default: 1)
     * @param padding Padding added to input (default: 0)
     * @param use_bias Whether to include bias (default: true)
     */
    Conv2DLayer(int in_channels, int out_channels, int kernel_size,
                int stride = 1, int padding = 0, bool use_bias = true);

    Tensor Forward(const Tensor& input) override;
    Tensor Backward(const Tensor& grad_output) override;
    std::map<std::string, Tensor> GetParameters() override;
    void SetParameters(const std::map<std::string, Tensor>& params) override;
    std::string GetName() const override { return "Conv2D"; }

    // Accessors
    int GetInChannels() const { return in_channels_; }
    int GetOutChannels() const { return out_channels_; }
    int GetKernelSize() const { return kernel_size_; }
    int GetStride() const { return stride_; }
    int GetPadding() const { return padding_; }

private:
    int in_channels_;
    int out_channels_;
    int kernel_size_;
    int stride_;
    int padding_;
    bool use_bias_;

    Tensor weights_;      // [out_channels, in_channels, kernel_size, kernel_size]
    Tensor bias_;         // [out_channels]
    Tensor grad_weights_;
    Tensor grad_bias_;

    // Helper for im2col/col2im operations
    Tensor Im2Col(const Tensor& input, int kernel_h, int kernel_w,
                  int stride_h, int stride_w, int pad_h, int pad_w);
    Tensor Col2Im(const Tensor& col, int height, int width, int channels,
                  int kernel_h, int kernel_w, int stride_h, int stride_w,
                  int pad_h, int pad_w);
};

// ============================================================================
// MaxPool2D Layer - 2D Max Pooling using ArrayFire
// ============================================================================

class CYXWIZ_API MaxPool2DLayer : public Layer {
public:
    /**
     * Create a 2D max pooling layer
     * @param pool_size Size of the pooling window (assumes square)
     * @param stride Stride of the pooling (default: same as pool_size)
     * @param padding Padding added to input (default: 0)
     */
    MaxPool2DLayer(int pool_size, int stride = -1, int padding = 0);

    Tensor Forward(const Tensor& input) override;
    Tensor Backward(const Tensor& grad_output) override;
    std::map<std::string, Tensor> GetParameters() override { return {}; }
    void SetParameters(const std::map<std::string, Tensor>&) override {}
    std::string GetName() const override { return "MaxPool2D"; }

private:
    int pool_size_;
    int stride_;
    int padding_;

    Tensor max_indices_;  // Store indices for backward pass
};

// ============================================================================
// AvgPool2D Layer - 2D Average Pooling using ArrayFire
// ============================================================================

class CYXWIZ_API AvgPool2DLayer : public Layer {
public:
    /**
     * Create a 2D average pooling layer
     * @param pool_size Size of the pooling window (assumes square)
     * @param stride Stride of the pooling (default: same as pool_size)
     * @param padding Padding added to input (default: 0)
     */
    AvgPool2DLayer(int pool_size, int stride = -1, int padding = 0);

    Tensor Forward(const Tensor& input) override;
    Tensor Backward(const Tensor& grad_output) override;
    std::map<std::string, Tensor> GetParameters() override { return {}; }
    void SetParameters(const std::map<std::string, Tensor>&) override {}
    std::string GetName() const override { return "AvgPool2D"; }

private:
    int pool_size_;
    int stride_;
    int padding_;
};

// ============================================================================
// GlobalAvgPool2D Layer - Global Average Pooling
// ============================================================================

class CYXWIZ_API GlobalAvgPool2DLayer : public Layer {
public:
    GlobalAvgPool2DLayer() = default;

    Tensor Forward(const Tensor& input) override;
    Tensor Backward(const Tensor& grad_output) override;
    std::map<std::string, Tensor> GetParameters() override { return {}; }
    void SetParameters(const std::map<std::string, Tensor>&) override {}
    std::string GetName() const override { return "GlobalAvgPool2D"; }
};

// ============================================================================
// BatchNorm2D Layer - Batch Normalization using ArrayFire
// ============================================================================

class CYXWIZ_API BatchNorm2DLayer : public Layer {
public:
    /**
     * Create a 2D batch normalization layer
     * @param num_features Number of features/channels
     * @param eps Small value for numerical stability (default: 1e-5)
     * @param momentum Momentum for running statistics (default: 0.1)
     */
    BatchNorm2DLayer(int num_features, float eps = 1e-5f, float momentum = 0.1f);

    Tensor Forward(const Tensor& input) override;
    Tensor Backward(const Tensor& grad_output) override;
    std::map<std::string, Tensor> GetParameters() override;
    void SetParameters(const std::map<std::string, Tensor>& params) override;
    std::string GetName() const override { return "BatchNorm2D"; }

private:
    int num_features_;
    float eps_;
    float momentum_;

    // Learnable parameters
    Tensor gamma_;    // Scale [num_features]
    Tensor beta_;     // Shift [num_features]

    // Running statistics (for inference)
    Tensor running_mean_;
    Tensor running_var_;

    // Cached for backward pass
    Tensor normalized_;
    Tensor std_inv_;

    // Gradients
    Tensor grad_gamma_;
    Tensor grad_beta_;
};

// ============================================================================
// Flatten Layer - Flatten spatial dimensions
// ============================================================================

class CYXWIZ_API FlattenLayer : public Layer {
public:
    FlattenLayer() = default;

    Tensor Forward(const Tensor& input) override;
    Tensor Backward(const Tensor& grad_output) override;
    std::map<std::string, Tensor> GetParameters() override { return {}; }
    void SetParameters(const std::map<std::string, Tensor>&) override {}
    std::string GetName() const override { return "Flatten"; }

private:
    std::vector<size_t> input_shape_;  // Original shape for backward
};

// ============================================================================
// Dropout Layer - Regularization
// ============================================================================

class CYXWIZ_API DropoutLayer : public Layer {
public:
    /**
     * Create a dropout layer
     * @param p Probability of dropping (default: 0.5)
     */
    explicit DropoutLayer(float p = 0.5f);

    Tensor Forward(const Tensor& input) override;
    Tensor Backward(const Tensor& grad_output) override;
    std::map<std::string, Tensor> GetParameters() override { return {}; }
    void SetParameters(const std::map<std::string, Tensor>&) override {}
    std::string GetName() const override { return "Dropout"; }

private:
    float p_;
    Tensor mask_;  // Dropout mask for backward pass
};

// ============================================================================
// LSTM Layer - Long Short-Term Memory using ArrayFire
// ============================================================================

class CYXWIZ_API LSTMLayer : public Layer {
public:
    /**
     * Create an LSTM layer
     * @param input_size Number of input features
     * @param hidden_size Number of hidden units
     * @param num_layers Number of stacked LSTM layers (default: 1)
     * @param batch_first If true, input shape is [batch, seq, features] (default: true)
     * @param bidirectional If true, use bidirectional LSTM (default: false)
     * @param dropout Dropout probability between layers (default: 0.0)
     */
    LSTMLayer(int input_size, int hidden_size, int num_layers = 1,
              bool batch_first = true, bool bidirectional = false,
              float dropout = 0.0f);

    /**
     * Forward pass
     * @param input Input tensor [batch, seq_len, input_size] if batch_first
     *              or [seq_len, batch, input_size] otherwise
     * @return Output tensor [batch, seq_len, hidden_size * num_directions]
     *
     * Also updates internal hidden and cell states
     */
    Tensor Forward(const Tensor& input) override;

    /**
     * Backward pass (Backpropagation Through Time)
     * @param grad_output Gradient from next layer
     * @return Gradient w.r.t input
     */
    Tensor Backward(const Tensor& grad_output) override;

    std::map<std::string, Tensor> GetParameters() override;
    void SetParameters(const std::map<std::string, Tensor>& params) override;
    std::string GetName() const override { return "LSTM"; }

    /**
     * Reset hidden and cell states to zeros
     * Call this at the start of each sequence/batch
     */
    void ResetState();

    /**
     * Set initial hidden state
     * @param h0 Initial hidden state [num_layers * num_directions, batch, hidden_size]
     */
    void SetHiddenState(const Tensor& h0);

    /**
     * Set initial cell state
     * @param c0 Initial cell state [num_layers * num_directions, batch, hidden_size]
     */
    void SetCellState(const Tensor& c0);

    /**
     * Get current hidden state
     */
    Tensor GetHiddenState() const { return h_n_; }

    /**
     * Get current cell state
     */
    Tensor GetCellState() const { return c_n_; }

    // Accessors
    int GetInputSize() const { return input_size_; }
    int GetHiddenSize() const { return hidden_size_; }
    int GetNumLayers() const { return num_layers_; }
    bool IsBatchFirst() const { return batch_first_; }
    bool IsBidirectional() const { return bidirectional_; }
    int GetNumDirections() const { return bidirectional_ ? 2 : 1; }

private:
    int input_size_;
    int hidden_size_;
    int num_layers_;
    bool batch_first_;
    bool bidirectional_;
    float dropout_;

    // Combined weight matrices for efficiency
    // W_ih: [4 * hidden_size, input_size] - input to hidden weights
    // W_hh: [4 * hidden_size, hidden_size] - hidden to hidden weights
    // b_ih: [4 * hidden_size] - input to hidden bias
    // b_hh: [4 * hidden_size] - hidden to hidden bias
    // Order: [input_gate, forget_gate, cell_gate, output_gate]
    std::vector<Tensor> W_ih_;  // Per layer
    std::vector<Tensor> W_hh_;  // Per layer
    std::vector<Tensor> b_ih_;  // Per layer
    std::vector<Tensor> b_hh_;  // Per layer

    // Reverse direction weights (for bidirectional)
    std::vector<Tensor> W_ih_reverse_;
    std::vector<Tensor> W_hh_reverse_;
    std::vector<Tensor> b_ih_reverse_;
    std::vector<Tensor> b_hh_reverse_;

    // Gradient accumulators
    std::vector<Tensor> grad_W_ih_;
    std::vector<Tensor> grad_W_hh_;
    std::vector<Tensor> grad_b_ih_;
    std::vector<Tensor> grad_b_hh_;
    std::vector<Tensor> grad_W_ih_reverse_;
    std::vector<Tensor> grad_W_hh_reverse_;
    std::vector<Tensor> grad_b_ih_reverse_;
    std::vector<Tensor> grad_b_hh_reverse_;

    // Hidden and cell states
    Tensor h_n_;  // Final hidden state
    Tensor c_n_;  // Final cell state

    // Cached values for backward pass
    std::vector<Tensor> cached_inputs_;      // Input at each layer
    std::vector<Tensor> cached_gates_;       // Gate activations [i, f, g, o]
    std::vector<Tensor> cached_cell_states_; // Cell states over time
    std::vector<Tensor> cached_hidden_states_; // Hidden states over time

    // Initialize weights using Xavier initialization
    void InitializeWeights();
};

// ============================================================================
// Embedding Layer - Lookup table for token embeddings
// ============================================================================

class CYXWIZ_API EmbeddingLayer : public Layer {
public:
    /**
     * Create an embedding layer (lookup table)
     * @param num_embeddings Size of the vocabulary (number of unique tokens)
     * @param embedding_dim Dimension of each embedding vector
     * @param padding_idx If specified, embeddings at this index are always zero (default: -1 = none)
     * @param max_norm If > 0, embeddings are normalized to this max norm (default: 0 = disabled)
     */
    EmbeddingLayer(int num_embeddings, int embedding_dim,
                   int padding_idx = -1, float max_norm = 0.0f);

    /**
     * Forward pass - lookup embeddings for input indices
     * @param input Integer tensor of indices [batch, seq_len] or [seq_len]
     * @return Embedding tensor [batch, seq_len, embedding_dim] or [seq_len, embedding_dim]
     */
    Tensor Forward(const Tensor& input) override;

    /**
     * Backward pass - accumulate gradients for used embeddings
     * @param grad_output Gradient from next layer [batch, seq_len, embedding_dim]
     * @return Empty tensor (no gradient w.r.t. integer indices)
     */
    Tensor Backward(const Tensor& grad_output) override;

    std::map<std::string, Tensor> GetParameters() override;
    void SetParameters(const std::map<std::string, Tensor>& params) override;
    std::string GetName() const override { return "Embedding"; }

    /**
     * Get embedding for a specific index
     * @param index Token index
     * @return Embedding vector [embedding_dim]
     */
    Tensor GetEmbedding(int index) const;

    /**
     * Set embedding for a specific index
     * @param index Token index
     * @param embedding Embedding vector [embedding_dim]
     */
    void SetEmbedding(int index, const Tensor& embedding);

    /**
     * Initialize embeddings from pretrained weights
     * @param weights Weight matrix [num_embeddings, embedding_dim]
     * @param freeze If true, embeddings won't be updated during training
     */
    void LoadPretrainedWeights(const Tensor& weights, bool freeze = false);

    // Accessors
    int GetNumEmbeddings() const { return num_embeddings_; }
    int GetEmbeddingDim() const { return embedding_dim_; }
    int GetPaddingIdx() const { return padding_idx_; }
    bool IsFrozen() const { return frozen_; }
    void SetFrozen(bool frozen) { frozen_ = frozen; }

private:
    int num_embeddings_;
    int embedding_dim_;
    int padding_idx_;
    float max_norm_;
    bool frozen_ = false;

    Tensor weight_;       // [num_embeddings, embedding_dim]
    Tensor grad_weight_;  // Gradient accumulator

    // Cached input indices for backward pass
    Tensor cached_indices_;

    void InitializeWeights();
    void NormalizeEmbeddings();
};


// ============================================================================
// LayerNorm Layer - Layer Normalization
// ============================================================================

class CYXWIZ_API LayerNormLayer : public Layer {
public:
    /**
     * Create a layer normalization layer
     * @param normalized_shape Shape of the normalized dimensions (last N dims)
     * @param eps Small value for numerical stability (default: 1e-5)
     * @param elementwise_affine Whether to use learnable affine parameters (default: true)
     */
    LayerNormLayer(const std::vector<int>& normalized_shape,
                   float eps = 1e-5f, bool elementwise_affine = true);

    Tensor Forward(const Tensor& input) override;
    Tensor Backward(const Tensor& grad_output) override;
    std::map<std::string, Tensor> GetParameters() override;
    void SetParameters(const std::map<std::string, Tensor>& params) override;
    std::string GetName() const override { return "LayerNorm"; }

private:
    std::vector<int> normalized_shape_;
    float eps_;
    bool elementwise_affine_;

    Tensor gamma_;      // Scale [normalized_shape]
    Tensor beta_;       // Shift [normalized_shape]
    Tensor grad_gamma_;
    Tensor grad_beta_;

    // Cached for backward
    Tensor normalized_;
    Tensor std_inv_;
};

// ============================================================================
// InstanceNorm2D Layer - Instance Normalization for CNNs
// ============================================================================

class CYXWIZ_API InstanceNorm2DLayer : public Layer {
public:
    /**
     * Create a 2D instance normalization layer
     * @param num_features Number of features/channels
     * @param eps Small value for numerical stability (default: 1e-5)
     * @param affine Whether to use learnable affine parameters (default: false)
     */
    InstanceNorm2DLayer(int num_features, float eps = 1e-5f, bool affine = false);

    Tensor Forward(const Tensor& input) override;
    Tensor Backward(const Tensor& grad_output) override;
    std::map<std::string, Tensor> GetParameters() override;
    void SetParameters(const std::map<std::string, Tensor>& params) override;
    std::string GetName() const override { return "InstanceNorm2D"; }

private:
    int num_features_;
    float eps_;
    bool affine_;

    Tensor gamma_;
    Tensor beta_;
    Tensor grad_gamma_;
    Tensor grad_beta_;

    // Cached for backward
    Tensor normalized_;
    Tensor std_inv_;
};

// ============================================================================
// GroupNorm Layer - Group Normalization
// ============================================================================

class CYXWIZ_API GroupNormLayer : public Layer {
public:
    /**
     * Create a group normalization layer
     * @param num_groups Number of groups to divide channels into
     * @param num_channels Number of channels
     * @param eps Small value for numerical stability (default: 1e-5)
     * @param affine Whether to use learnable affine parameters (default: true)
     */
    GroupNormLayer(int num_groups, int num_channels,
                   float eps = 1e-5f, bool affine = true);

    Tensor Forward(const Tensor& input) override;
    Tensor Backward(const Tensor& grad_output) override;
    std::map<std::string, Tensor> GetParameters() override;
    void SetParameters(const std::map<std::string, Tensor>& params) override;
    std::string GetName() const override { return "GroupNorm"; }

private:
    int num_groups_;
    int num_channels_;
    float eps_;
    bool affine_;

    Tensor gamma_;      // [num_channels]
    Tensor beta_;       // [num_channels]
    Tensor grad_gamma_;
    Tensor grad_beta_;

    // Cached for backward
    Tensor normalized_;
    Tensor std_inv_;
};

// ============================================================================
// Conv1D Layer - 1D Convolution using ArrayFire
// ============================================================================

class CYXWIZ_API Conv1DLayer : public Layer {
public:
    /**
     * Create a 1D convolutional layer
     * @param in_channels Number of input channels
     * @param out_channels Number of output channels (filters)
     * @param kernel_size Size of the convolution kernel
     * @param stride Stride of the convolution (default: 1)
     * @param padding Padding added to input (default: 0)
     * @param dilation Dilation of the kernel (default: 1)
     * @param use_bias Whether to include bias (default: true)
     */
    Conv1DLayer(int in_channels, int out_channels, int kernel_size,
                int stride = 1, int padding = 0, int dilation = 1,
                bool use_bias = true);

    Tensor Forward(const Tensor& input) override;
    Tensor Backward(const Tensor& grad_output) override;
    std::map<std::string, Tensor> GetParameters() override;
    void SetParameters(const std::map<std::string, Tensor>& params) override;
    std::string GetName() const override { return "Conv1D"; }

    // Accessors
    int GetInChannels() const { return in_channels_; }
    int GetOutChannels() const { return out_channels_; }
    int GetKernelSize() const { return kernel_size_; }
    int GetStride() const { return stride_; }
    int GetPadding() const { return padding_; }
    int GetDilation() const { return dilation_; }

private:
    int in_channels_;
    int out_channels_;
    int kernel_size_;
    int stride_;
    int padding_;
    int dilation_;
    bool use_bias_;

    Tensor weights_;      // [out_channels, in_channels, kernel_size]
    Tensor bias_;         // [out_channels]
    Tensor grad_weights_;
    Tensor grad_bias_;
};

// ============================================================================
// GRU Layer - Gated Recurrent Unit using ArrayFire
// ============================================================================

class CYXWIZ_API GRULayer : public Layer {
public:
    /**
     * Create a GRU layer
     * @param input_size Number of input features
     * @param hidden_size Number of hidden units
     * @param num_layers Number of stacked GRU layers (default: 1)
     * @param batch_first If true, input shape is [batch, seq, features] (default: true)
     * @param bidirectional If true, use bidirectional GRU (default: false)
     * @param dropout Dropout probability between layers (default: 0.0)
     */
    GRULayer(int input_size, int hidden_size, int num_layers = 1,
             bool batch_first = true, bool bidirectional = false,
             float dropout = 0.0f);

    Tensor Forward(const Tensor& input) override;
    Tensor Backward(const Tensor& grad_output) override;
    std::map<std::string, Tensor> GetParameters() override;
    void SetParameters(const std::map<std::string, Tensor>& params) override;
    std::string GetName() const override { return "GRU"; }

    void ResetState();
    void SetHiddenState(const Tensor& h0);
    Tensor GetHiddenState() const { return h_n_; }

    int GetInputSize() const { return input_size_; }
    int GetHiddenSize() const { return hidden_size_; }
    int GetNumLayers() const { return num_layers_; }
    bool IsBatchFirst() const { return batch_first_; }
    bool IsBidirectional() const { return bidirectional_; }

private:
    int input_size_;
    int hidden_size_;
    int num_layers_;
    bool batch_first_;
    bool bidirectional_;
    float dropout_;

    // Combined weight matrices [reset_gate, update_gate, new_gate]
    std::vector<Tensor> W_ih_;  // [3 * hidden_size, input_size]
    std::vector<Tensor> W_hh_;  // [3 * hidden_size, hidden_size]
    std::vector<Tensor> b_ih_;  // [3 * hidden_size]
    std::vector<Tensor> b_hh_;  // [3 * hidden_size]

    std::vector<Tensor> W_ih_reverse_;
    std::vector<Tensor> W_hh_reverse_;
    std::vector<Tensor> b_ih_reverse_;
    std::vector<Tensor> b_hh_reverse_;

    // Gradients
    std::vector<Tensor> grad_W_ih_;
    std::vector<Tensor> grad_W_hh_;
    std::vector<Tensor> grad_b_ih_;
    std::vector<Tensor> grad_b_hh_;

    Tensor h_n_;

    // Cached for backward
    std::vector<Tensor> cached_inputs_;
    std::vector<Tensor> cached_gates_;
    std::vector<Tensor> cached_hidden_states_;

    void InitializeWeights();
};

// ============================================================================
// MultiHeadAttention Layer - Multi-Head Self/Cross Attention
// ============================================================================

class CYXWIZ_API MultiHeadAttentionLayer : public Layer {
public:
    /**
     * Create a multi-head attention layer
     * @param embed_dim Total embedding dimension (must be divisible by num_heads)
     * @param num_heads Number of attention heads
     * @param dropout Dropout probability on attention weights (default: 0.0)
     * @param use_bias Whether to use bias in projections (default: true)
     */
    MultiHeadAttentionLayer(int embed_dim, int num_heads,
                            float dropout = 0.0f, bool use_bias = true);

    /**
     * Self-attention forward pass
     * @param input Input tensor [batch, seq_len, embed_dim]
     * @return Output tensor [batch, seq_len, embed_dim]
     */
    Tensor Forward(const Tensor& input) override;

    /**
     * Full attention forward pass with separate Q, K, V
     * @param query Query tensor [batch, seq_len_q, embed_dim]
     * @param key Key tensor [batch, seq_len_kv, embed_dim]
     * @param value Value tensor [batch, seq_len_kv, embed_dim]
     * @param attn_mask Attention mask [seq_len_q, seq_len_kv] or nullptr (optional)
     * @return Output tensor [batch, seq_len_q, embed_dim]
     */
    Tensor Forward(const Tensor& query, const Tensor& key, const Tensor& value,
                   const Tensor* attn_mask = nullptr);

    Tensor Backward(const Tensor& grad_output) override;
    std::map<std::string, Tensor> GetParameters() override;
    void SetParameters(const std::map<std::string, Tensor>& params) override;
    std::string GetName() const override { return "MultiHeadAttention"; }

    // Get last computed attention weights (for visualization)
    Tensor GetAttentionWeights() const { return cached_attn_weights_; }

    // Accessors
    int GetEmbedDim() const { return embed_dim_; }
    int GetNumHeads() const { return num_heads_; }
    int GetHeadDim() const { return head_dim_; }

private:
    int embed_dim_;
    int num_heads_;
    int head_dim_;
    float dropout_;
    bool use_bias_;
    float scale_;  // 1/sqrt(head_dim)

    // Projection weights [embed_dim, embed_dim]
    Tensor W_q_, W_k_, W_v_, W_o_;
    Tensor b_q_, b_k_, b_v_, b_o_;

    // Gradients
    Tensor grad_W_q_, grad_W_k_, grad_W_v_, grad_W_o_;
    Tensor grad_b_q_, grad_b_k_, grad_b_v_, grad_b_o_;

    // Cached for backward
    Tensor cached_query_, cached_key_, cached_value_;
    Tensor cached_Q_, cached_K_, cached_V_;  // After projection
    Tensor cached_attn_weights_;  // After softmax
    Tensor cached_context_;  // Before output projection
    Tensor dropout_mask_;  // Dropout mask for attention weights

    void InitializeWeights();
};

// ============================================================================
// TransformerEncoderLayer - Single Transformer Encoder Block
// ============================================================================

class CYXWIZ_API TransformerEncoderLayer : public Layer {
public:
    /**
     * Create a transformer encoder layer
     * @param d_model Model dimension (embedding size)
     * @param nhead Number of attention heads
     * @param dim_feedforward Feedforward network dimension (default: 2048)
     * @param dropout Dropout probability (default: 0.1)
     * @param norm_first Apply LayerNorm before attention/FFN (Pre-LN, default: false)
     */
    TransformerEncoderLayer(int d_model, int nhead, int dim_feedforward = 2048,
                            float dropout = 0.1f, bool norm_first = false);

    /**
     * Forward pass
     * @param input Input tensor [batch, seq_len, d_model]
     * @param src_mask Attention mask [seq_len, seq_len] or nullptr
     * @return Output tensor [batch, seq_len, d_model]
     */
    Tensor Forward(const Tensor& input) override;
    Tensor Forward(const Tensor& input, const Tensor* src_mask);

    Tensor Backward(const Tensor& grad_output) override;
    std::map<std::string, Tensor> GetParameters() override;
    void SetParameters(const std::map<std::string, Tensor>& params) override;
    std::string GetName() const override { return "TransformerEncoderLayer"; }

    void SetTraining(bool training) override;

private:
    int d_model_;
    int nhead_;
    int dim_feedforward_;
    float dropout_;
    bool norm_first_;

    std::unique_ptr<MultiHeadAttentionLayer> self_attn_;
    std::unique_ptr<LayerNormLayer> norm1_;
    std::unique_ptr<LayerNormLayer> norm2_;
    std::unique_ptr<DenseLayer> linear1_;
    std::unique_ptr<DenseLayer> linear2_;
    std::unique_ptr<DropoutLayer> dropout1_;
    std::unique_ptr<DropoutLayer> dropout2_;

    // Cached for backward
    Tensor cached_attn_output_;
    Tensor cached_ffn_mid_;
    Tensor cached_residual1_;
    Tensor cached_residual2_;
};

// ============================================================================
// TransformerDecoderLayer - Single Transformer Decoder Block
// ============================================================================

class CYXWIZ_API TransformerDecoderLayer : public Layer {
public:
    /**
     * Create a transformer decoder layer
     * @param d_model Model dimension (embedding size)
     * @param nhead Number of attention heads
     * @param dim_feedforward Feedforward network dimension (default: 2048)
     * @param dropout Dropout probability (default: 0.1)
     * @param norm_first Apply LayerNorm before attention/FFN (Pre-LN, default: false)
     */
    TransformerDecoderLayer(int d_model, int nhead, int dim_feedforward = 2048,
                            float dropout = 0.1f, bool norm_first = false);

    /**
     * Forward pass (self-attention only, no encoder memory)
     * @param input Target tensor [batch, tgt_len, d_model]
     * @return Output tensor [batch, tgt_len, d_model]
     */
    Tensor Forward(const Tensor& input) override;

    /**
     * Full forward pass with encoder memory
     * @param tgt Target tensor [batch, tgt_len, d_model]
     * @param memory Encoder output [batch, src_len, d_model]
     * @param tgt_mask Target attention mask [tgt_len, tgt_len] or nullptr
     * @param memory_mask Memory attention mask [tgt_len, src_len] or nullptr
     * @return Output tensor [batch, tgt_len, d_model]
     */
    Tensor Forward(const Tensor& tgt, const Tensor& memory,
                   const Tensor* tgt_mask = nullptr,
                   const Tensor* memory_mask = nullptr);

    Tensor Backward(const Tensor& grad_output) override;
    std::map<std::string, Tensor> GetParameters() override;
    void SetParameters(const std::map<std::string, Tensor>& params) override;
    std::string GetName() const override { return "TransformerDecoderLayer"; }

    void SetTraining(bool training) override;

    /**
     * Generate causal mask for autoregressive decoding
     * @param size Sequence length
     * @return Upper triangular mask [size, size] with -inf above diagonal
     */
    static Tensor GenerateCausalMask(int size);

private:
    int d_model_;
    int nhead_;
    int dim_feedforward_;
    float dropout_;
    bool norm_first_;

    std::unique_ptr<MultiHeadAttentionLayer> self_attn_;
    std::unique_ptr<MultiHeadAttentionLayer> cross_attn_;
    std::unique_ptr<LayerNormLayer> norm1_;
    std::unique_ptr<LayerNormLayer> norm2_;
    std::unique_ptr<LayerNormLayer> norm3_;
    std::unique_ptr<DenseLayer> linear1_;
    std::unique_ptr<DenseLayer> linear2_;
    std::unique_ptr<DropoutLayer> dropout1_;
    std::unique_ptr<DropoutLayer> dropout2_;
    std::unique_ptr<DropoutLayer> dropout3_;

    // Cached for backward
    Tensor cached_self_attn_output_;
    Tensor cached_cross_attn_output_;
    Tensor cached_ffn_mid_;
    Tensor cached_memory_;
    Tensor cached_residual1_;
    Tensor cached_residual2_;
    Tensor cached_residual3_;
};

} // namespace cyxwiz
