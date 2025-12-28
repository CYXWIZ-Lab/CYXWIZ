# Layer API Reference

Neural network layer implementations in cyxwiz-backend, providing GPU-accelerated building blocks for deep learning models.

## Base Layer Class

```cpp
namespace cyxwiz {

class CYXWIZ_API Layer {
public:
    Layer();
    virtual ~Layer() = default;

    // Forward pass
    virtual Tensor Forward(const Tensor& input) = 0;

    // Get/Set parameters
    virtual std::vector<Tensor*> Parameters();
    virtual std::vector<Tensor*> Gradients();

    // Training mode
    void Train(bool mode = true);
    void Eval();
    bool IsTraining() const;

    // Layer name
    void SetName(const std::string& name);
    std::string Name() const;

    // Shape inference
    virtual std::vector<int> OutputShape(const std::vector<int>& input_shape) const;

    // Parameter count
    virtual int64_t NumParameters() const;

protected:
    bool training_ = true;
    std::string name_;
};

} // namespace cyxwiz
```

## Dense (Fully Connected) Layer

```cpp
class CYXWIZ_API Dense : public Layer {
public:
    Dense(int units,
          ActivationType activation = ActivationType::None,
          bool use_bias = true,
          InitializerType kernel_initializer = InitializerType::GlorotUniform,
          InitializerType bias_initializer = InitializerType::Zeros);

    Tensor Forward(const Tensor& input) override;

    // Accessors
    int Units() const;
    Tensor& Weights();
    Tensor& Bias();

private:
    int units_;
    ActivationType activation_;
    bool use_bias_;
    Tensor weights_;
    Tensor bias_;
};
```

### Usage

```cpp
#include <cyxwiz/layer.h>

using namespace cyxwiz;

// Create dense layer
Dense dense1(128, ActivationType::ReLU);
Dense dense2(64, ActivationType::ReLU, true, InitializerType::HeNormal);
Dense output(10, ActivationType::Softmax);

// Forward pass
Tensor x = Randn({32, 784});  // Batch of 32, input size 784
Tensor h1 = dense1.Forward(x);    // Shape: (32, 128)
Tensor h2 = dense2.Forward(h1);   // Shape: (32, 64)
Tensor out = output.Forward(h2);  // Shape: (32, 10)
```

## Convolutional Layers

### Conv2D

```cpp
class CYXWIZ_API Conv2D : public Layer {
public:
    Conv2D(int filters,
           std::pair<int, int> kernel_size,
           std::pair<int, int> stride = {1, 1},
           PaddingType padding = PaddingType::Valid,
           ActivationType activation = ActivationType::None,
           bool use_bias = true);

    Tensor Forward(const Tensor& input) override;

    // Input shape: (batch, channels, height, width)
    // Output shape: (batch, filters, new_height, new_width)

private:
    int filters_;
    std::pair<int, int> kernel_size_;
    std::pair<int, int> stride_;
    PaddingType padding_;
    Tensor kernel_;
    Tensor bias_;
};
```

### Conv1D

```cpp
class CYXWIZ_API Conv1D : public Layer {
public:
    Conv1D(int filters,
           int kernel_size,
           int stride = 1,
           PaddingType padding = PaddingType::Valid,
           ActivationType activation = ActivationType::None,
           bool use_bias = true);

    Tensor Forward(const Tensor& input) override;

    // Input shape: (batch, channels, length)
    // Output shape: (batch, filters, new_length)
};
```

### Usage

```cpp
// Create convolutional layers
Conv2D conv1(32, {3, 3}, {1, 1}, PaddingType::Same, ActivationType::ReLU);
Conv2D conv2(64, {3, 3}, {1, 1}, PaddingType::Same, ActivationType::ReLU);

// Forward pass
Tensor x = Randn({32, 3, 224, 224});  // Batch of 32 RGB images
Tensor h1 = conv1.Forward(x);          // Shape: (32, 32, 224, 224)
Tensor h2 = conv2.Forward(h1);         // Shape: (32, 64, 224, 224)
```

## Pooling Layers

### MaxPool2D

```cpp
class CYXWIZ_API MaxPool2D : public Layer {
public:
    MaxPool2D(std::pair<int, int> pool_size,
              std::pair<int, int> stride = {0, 0},  // 0 = same as pool_size
              PaddingType padding = PaddingType::Valid);

    Tensor Forward(const Tensor& input) override;
};
```

### AvgPool2D

```cpp
class CYXWIZ_API AvgPool2D : public Layer {
public:
    AvgPool2D(std::pair<int, int> pool_size,
              std::pair<int, int> stride = {0, 0},
              PaddingType padding = PaddingType::Valid);

    Tensor Forward(const Tensor& input) override;
};
```

### GlobalAveragePooling2D

```cpp
class CYXWIZ_API GlobalAveragePooling2D : public Layer {
public:
    GlobalAveragePooling2D();

    Tensor Forward(const Tensor& input) override;
    // Input: (batch, channels, height, width)
    // Output: (batch, channels)
};
```

### Usage

```cpp
MaxPool2D pool1({2, 2});
GlobalAveragePooling2D gap;

Tensor x = Randn({32, 64, 112, 112});
Tensor h1 = pool1.Forward(x);  // Shape: (32, 64, 56, 56)
Tensor h2 = gap.Forward(h1);   // Shape: (32, 64)
```

## Normalization Layers

### BatchNorm

```cpp
class CYXWIZ_API BatchNorm : public Layer {
public:
    BatchNorm(int num_features,
              float epsilon = 1e-5f,
              float momentum = 0.1f);

    Tensor Forward(const Tensor& input) override;

    // Learnable parameters
    Tensor& Gamma();  // Scale
    Tensor& Beta();   // Shift

    // Running statistics
    Tensor& RunningMean();
    Tensor& RunningVar();

private:
    int num_features_;
    float epsilon_;
    float momentum_;
    Tensor gamma_;
    Tensor beta_;
    Tensor running_mean_;
    Tensor running_var_;
};
```

### LayerNorm

```cpp
class CYXWIZ_API LayerNorm : public Layer {
public:
    LayerNorm(const std::vector<int>& normalized_shape,
              float epsilon = 1e-5f);

    Tensor Forward(const Tensor& input) override;

private:
    std::vector<int> normalized_shape_;
    float epsilon_;
    Tensor gamma_;
    Tensor beta_;
};
```

### Usage

```cpp
// After conv layer
BatchNorm bn1(64);  // 64 channels

// In transformer
LayerNorm ln({512});  // Hidden size 512

Tensor x = Randn({32, 64, 56, 56});
Tensor normalized = bn1.Forward(x);
```

## Recurrent Layers

### LSTM

```cpp
class CYXWIZ_API LSTM : public Layer {
public:
    LSTM(int hidden_size,
         int num_layers = 1,
         bool bidirectional = false,
         float dropout = 0.0f,
         bool batch_first = true);

    // Returns (output, (h_n, c_n))
    std::tuple<Tensor, Tensor, Tensor> ForwardWithState(
        const Tensor& input,
        const Tensor& h_0 = Tensor(),
        const Tensor& c_0 = Tensor());

    Tensor Forward(const Tensor& input) override;

private:
    int hidden_size_;
    int num_layers_;
    bool bidirectional_;
    float dropout_;
    bool batch_first_;
};
```

### GRU

```cpp
class CYXWIZ_API GRU : public Layer {
public:
    GRU(int hidden_size,
        int num_layers = 1,
        bool bidirectional = false,
        float dropout = 0.0f,
        bool batch_first = true);

    std::pair<Tensor, Tensor> ForwardWithState(
        const Tensor& input,
        const Tensor& h_0 = Tensor());

    Tensor Forward(const Tensor& input) override;
};
```

### Usage

```cpp
LSTM lstm(256, 2, true, 0.2f);  // Bidirectional 2-layer LSTM

Tensor x = Randn({32, 100, 128});  // (batch, seq_len, input_size)
auto [output, h_n, c_n] = lstm.ForwardWithState(x);
// output: (32, 100, 512)  // 256*2 for bidirectional
// h_n: (4, 32, 256)       // 2 layers * 2 directions
// c_n: (4, 32, 256)
```

## Attention Layers

### MultiHeadAttention

```cpp
class CYXWIZ_API MultiHeadAttention : public Layer {
public:
    MultiHeadAttention(int embed_dim,
                       int num_heads,
                       float dropout = 0.0f,
                       bool bias = true);

    // Self-attention (query = key = value)
    Tensor Forward(const Tensor& input) override;

    // Cross-attention
    Tensor Forward(const Tensor& query,
                   const Tensor& key,
                   const Tensor& value,
                   const Tensor& attn_mask = Tensor());

    // Returns (output, attention_weights)
    std::pair<Tensor, Tensor> ForwardWithWeights(
        const Tensor& query,
        const Tensor& key,
        const Tensor& value,
        const Tensor& attn_mask = Tensor());

private:
    int embed_dim_;
    int num_heads_;
    int head_dim_;
    float dropout_;
    Dense q_proj_, k_proj_, v_proj_, out_proj_;
};
```

### Usage

```cpp
MultiHeadAttention mha(512, 8, 0.1f);

Tensor x = Randn({32, 100, 512});  // (batch, seq_len, embed_dim)
Tensor output = mha.Forward(x);    // Self-attention
// output: (32, 100, 512)

// Cross-attention
Tensor query = Randn({32, 50, 512});
Tensor key = Randn({32, 100, 512});
Tensor value = Randn({32, 100, 512});
Tensor cross_output = mha.Forward(query, key, value);
```

## Regularization Layers

### Dropout

```cpp
class CYXWIZ_API Dropout : public Layer {
public:
    Dropout(float rate = 0.5f);

    Tensor Forward(const Tensor& input) override;
    // Only applies dropout in training mode

private:
    float rate_;
};
```

### Dropout2D

```cpp
class CYXWIZ_API Dropout2D : public Layer {
public:
    Dropout2D(float rate = 0.5f);
    // Drops entire channels
};
```

### Usage

```cpp
Dropout drop(0.3f);

model.Train();
Tensor train_out = drop.Forward(x);  // Dropout applied

model.Eval();
Tensor eval_out = drop.Forward(x);   // No dropout
```

## Activation Layers

### Standalone Activation Classes

```cpp
class CYXWIZ_API ReLU : public Layer {
public:
    Tensor Forward(const Tensor& input) override;
};

class CYXWIZ_API LeakyReLU : public Layer {
public:
    LeakyReLU(float negative_slope = 0.01f);
    Tensor Forward(const Tensor& input) override;
};

class CYXWIZ_API GELU : public Layer {
public:
    Tensor Forward(const Tensor& input) override;
};

class CYXWIZ_API Sigmoid : public Layer {
public:
    Tensor Forward(const Tensor& input) override;
};

class CYXWIZ_API Tanh : public Layer {
public:
    Tensor Forward(const Tensor& input) override;
};

class CYXWIZ_API Softmax : public Layer {
public:
    Softmax(int dim = -1);
    Tensor Forward(const Tensor& input) override;
};
```

### Activation Type Enum

```cpp
enum class ActivationType {
    None,
    ReLU,
    LeakyReLU,
    GELU,
    Sigmoid,
    Tanh,
    Softmax,
    Swish,
    Mish
};
```

## Embedding Layer

```cpp
class CYXWIZ_API Embedding : public Layer {
public:
    Embedding(int num_embeddings, int embedding_dim,
              int padding_idx = -1);

    Tensor Forward(const Tensor& indices) override;
    // indices: (batch, seq_len) integer tensor
    // output: (batch, seq_len, embedding_dim)

    Tensor& Weights();

private:
    int num_embeddings_;
    int embedding_dim_;
    int padding_idx_;
    Tensor weights_;
};
```

### Usage

```cpp
Embedding embed(10000, 512);  // Vocab size 10000, embedding dim 512

Tensor tokens = ...;  // Integer tensor (32, 100)
Tensor embeddings = embed.Forward(tokens);  // (32, 100, 512)
```

## Flatten Layer

```cpp
class CYXWIZ_API Flatten : public Layer {
public:
    Flatten(int start_dim = 1, int end_dim = -1);

    Tensor Forward(const Tensor& input) override;
    // Default: flatten all dims except batch
};
```

## Layer Factory

```cpp
namespace cyxwiz {

std::unique_ptr<Layer> CreateLayer(LayerType type, const LayerConfig& config);

enum class LayerType {
    Dense,
    Conv2D,
    Conv1D,
    MaxPool2D,
    AvgPool2D,
    GlobalAvgPool2D,
    BatchNorm,
    LayerNorm,
    LSTM,
    GRU,
    MultiHeadAttention,
    Dropout,
    Embedding,
    Flatten,
    // ... more
};

}
```

## Python Bindings

```python
import pycyxwiz as cyx

# Dense layer
dense = cyx.layers.Dense(128, activation='relu')

# Conv2D
conv = cyx.layers.Conv2D(32, kernel_size=(3, 3), padding='same')

# BatchNorm
bn = cyx.layers.BatchNorm(64)

# LSTM
lstm = cyx.layers.LSTM(256, num_layers=2, bidirectional=True)

# Attention
mha = cyx.layers.MultiHeadAttention(512, num_heads=8)

# Sequential model
model = cyx.Sequential([
    cyx.layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
    cyx.layers.MaxPool2D((2, 2)),
    cyx.layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
    cyx.layers.GlobalAveragePooling2D(),
    cyx.layers.Dense(10, activation='softmax')
])

# Forward pass
output = model(input_tensor)
```

---

**Next**: [Optimizer API](optimizers.md) | [Loss API](loss.md)
