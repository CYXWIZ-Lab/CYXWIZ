#include "node_documentation.h"
#include <imgui.h>

namespace gui {

NodeDocumentationManager& NodeDocumentationManager::Instance() {
    static NodeDocumentationManager instance;
    return instance;
}

NodeDocumentationManager::NodeDocumentationManager() {
    InitializeDocumentation();
}

const NodeDocumentation* NodeDocumentationManager::GetDocumentation(NodeType type) const {
    auto it = docs_.find(type);
    if (it != docs_.end()) {
        return &it->second;
    }
    return nullptr;
}

void NodeDocumentationManager::RenderTooltip(NodeType type) {
    const NodeDocumentation* doc = GetDocumentation(type);
    if (!doc) return;

    ImGui::BeginTooltip();

    // Title with category
    ImGui::TextColored(ImVec4(0.4f, 0.8f, 1.0f, 1.0f), "%s", doc->title.c_str());
    ImGui::SameLine();
    ImGui::TextDisabled("[%s]", doc->category.c_str());

    ImGui::Separator();

    // Description
    ImGui::PushTextWrapPos(400.0f);
    ImGui::TextWrapped("%s", doc->description.c_str());
    ImGui::PopTextWrapPos();

    // Parameters
    if (!doc->parameters.empty()) {
        ImGui::Spacing();
        ImGui::TextColored(ImVec4(0.8f, 0.8f, 0.4f, 1.0f), "Parameters:");
        for (const auto& param : doc->parameters) {
            ImGui::BulletText("%s", param.first.c_str());
            ImGui::SameLine();
            ImGui::TextDisabled("- %s", param.second.c_str());
        }
    }

    // Tips
    if (!doc->tips.empty()) {
        ImGui::Spacing();
        ImGui::TextColored(ImVec4(0.4f, 1.0f, 0.6f, 1.0f), "Tips:");
        for (const auto& tip : doc->tips) {
            ImGui::BulletText("%s", tip.c_str());
        }
    }

    // Usage example
    if (!doc->usage.empty()) {
        ImGui::Spacing();
        ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.8f, 1.0f), "Usage:");
        ImGui::TextDisabled("%s", doc->usage.c_str());
    }

    ImGui::EndTooltip();
}

bool NodeDocumentationManager::RenderHelpMarker(NodeType type) {
    ImGui::TextDisabled("(?)");
    if (ImGui::IsItemHovered(ImGuiHoveredFlags_DelayNormal)) {
        RenderTooltip(type);
        return true;
    }
    return false;
}

void NodeDocumentationManager::RenderCompactTooltip(NodeType type) {
    const NodeDocumentation* doc = GetDocumentation(type);
    if (!doc) return;

    ImGui::BeginTooltip();
    ImGui::Text("%s", doc->title.c_str());
    ImGui::PushTextWrapPos(300.0f);
    ImGui::TextDisabled("%s", doc->description.c_str());
    ImGui::PopTextWrapPos();
    ImGui::EndTooltip();
}

const char* NodeDocumentationManager::GetCategoryName(NodeType type) {
    switch (type) {
        // Core Layers
        case NodeType::Dense:
            return "Core Layers";

        // Convolutional
        case NodeType::Conv1D:
        case NodeType::Conv2D:
        case NodeType::Conv3D:
        case NodeType::DepthwiseConv2D:
            return "Convolutional";

        // Pooling
        case NodeType::MaxPool2D:
        case NodeType::AvgPool2D:
        case NodeType::GlobalMaxPool:
        case NodeType::GlobalAvgPool:
        case NodeType::AdaptiveAvgPool:
            return "Pooling";

        // Normalization
        case NodeType::BatchNorm:
        case NodeType::LayerNorm:
        case NodeType::GroupNorm:
        case NodeType::InstanceNorm:
            return "Normalization";

        // Regularization
        case NodeType::Dropout:
        case NodeType::Flatten:
            return "Regularization";

        // Recurrent
        case NodeType::RNN:
        case NodeType::LSTM:
        case NodeType::GRU:
        case NodeType::Bidirectional:
        case NodeType::TimeDistributed:
        case NodeType::Embedding:
            return "Recurrent";

        // Attention
        case NodeType::MultiHeadAttention:
        case NodeType::SelfAttention:
        case NodeType::CrossAttention:
        case NodeType::LinearAttention:
        case NodeType::TransformerEncoder:
        case NodeType::TransformerDecoder:
        case NodeType::PositionalEncoding:
            return "Attention";

        // Activations
        case NodeType::ReLU:
        case NodeType::LeakyReLU:
        case NodeType::PReLU:
        case NodeType::ELU:
        case NodeType::SELU:
        case NodeType::GELU:
        case NodeType::Swish:
        case NodeType::Mish:
        case NodeType::Sigmoid:
        case NodeType::Tanh:
        case NodeType::Softmax:
            return "Activation";

        // Shape Operations
        case NodeType::Reshape:
        case NodeType::Permute:
        case NodeType::Squeeze:
        case NodeType::Unsqueeze:
        case NodeType::View:
        case NodeType::Split:
            return "Shape Operations";

        // Merge Operations
        case NodeType::Concatenate:
        case NodeType::Add:
        case NodeType::Multiply:
        case NodeType::Average:
            return "Merge Operations";

        // Output
        case NodeType::Output:
            return "Output";

        // Loss Functions
        case NodeType::MSELoss:
        case NodeType::CrossEntropyLoss:
        case NodeType::BCELoss:
        case NodeType::BCEWithLogits:
        case NodeType::L1Loss:
        case NodeType::SmoothL1Loss:
        case NodeType::HuberLoss:
        case NodeType::NLLLoss:
            return "Loss Functions";

        // Optimizers
        case NodeType::SGD:
        case NodeType::Adam:
        case NodeType::AdamW:
        case NodeType::RMSprop:
        case NodeType::Adagrad:
        case NodeType::NAdam:
            return "Optimizers";

        // LR Schedulers
        case NodeType::StepLR:
        case NodeType::CosineAnnealing:
        case NodeType::ReduceOnPlateau:
        case NodeType::ExponentialLR:
        case NodeType::WarmupScheduler:
            return "LR Schedulers";

        // Regularization Nodes
        case NodeType::L1Regularization:
        case NodeType::L2Regularization:
        case NodeType::ElasticNet:
            return "Regularization";

        // Utility
        case NodeType::Lambda:
        case NodeType::Identity:
        case NodeType::Constant:
        case NodeType::Parameter:
            return "Utility";

        // Data Pipeline
        case NodeType::DatasetInput:
        case NodeType::DataLoader:
        case NodeType::Augmentation:
        case NodeType::DataSplit:
        case NodeType::TensorReshape:
        case NodeType::Normalize:
        case NodeType::OneHotEncode:
            return "Data Pipeline";

        default:
            return "Other";
    }
}

unsigned int NodeDocumentationManager::GetCategoryColor(NodeType type) {
    const char* category = GetCategoryName(type);

    // Return ImU32 colors for each category
    if (strcmp(category, "Core Layers") == 0) return IM_COL32(100, 150, 200, 255);
    if (strcmp(category, "Convolutional") == 0) return IM_COL32(150, 100, 200, 255);
    if (strcmp(category, "Pooling") == 0) return IM_COL32(100, 200, 150, 255);
    if (strcmp(category, "Normalization") == 0) return IM_COL32(200, 150, 100, 255);
    if (strcmp(category, "Regularization") == 0) return IM_COL32(200, 100, 150, 255);
    if (strcmp(category, "Recurrent") == 0) return IM_COL32(150, 200, 100, 255);
    if (strcmp(category, "Attention") == 0) return IM_COL32(200, 200, 100, 255);
    if (strcmp(category, "Activation") == 0) return IM_COL32(100, 200, 200, 255);
    if (strcmp(category, "Shape Operations") == 0) return IM_COL32(180, 180, 180, 255);
    if (strcmp(category, "Merge Operations") == 0) return IM_COL32(200, 150, 200, 255);
    if (strcmp(category, "Output") == 0) return IM_COL32(100, 200, 100, 255);
    if (strcmp(category, "Loss Functions") == 0) return IM_COL32(200, 100, 100, 255);
    if (strcmp(category, "Optimizers") == 0) return IM_COL32(100, 100, 200, 255);
    if (strcmp(category, "LR Schedulers") == 0) return IM_COL32(150, 150, 200, 255);
    if (strcmp(category, "Utility") == 0) return IM_COL32(150, 150, 150, 255);
    if (strcmp(category, "Data Pipeline") == 0) return IM_COL32(100, 180, 180, 255);

    return IM_COL32(128, 128, 128, 255);  // Default gray
}

void NodeDocumentationManager::InitializeDocumentation() {
    // ===== Core Layers =====
    docs_[NodeType::Dense] = {
        "Dense (Fully Connected)",
        "A fully connected neural network layer where every input is connected to every output. "
        "Also known as Linear layer in PyTorch or Dense in Keras.",
        "Connect after any layer that outputs a 1D tensor, or after Flatten for 2D+ inputs.",
        {
            {"units", "Number of output neurons"},
            {"activation", "Optional activation function to apply"},
            {"use_bias", "Whether to add a learnable bias term"}
        },
        {
            "For image classification, use after Flatten layer",
            "Last Dense should match number of classes for classification"
        },
        "Core Layers"
    };

    // ===== Convolutional Layers =====
    docs_[NodeType::Conv1D] = {
        "Conv1D",
        "1D convolution layer for sequence data. Slides a kernel across the input sequence "
        "to extract local features. Commonly used for text and time series.",
        "Input shape: (batch, channels, length). Output: (batch, out_channels, new_length).",
        {
            {"filters", "Number of output channels/filters"},
            {"kernel_size", "Size of the sliding window"},
            {"stride", "Step size between kernel applications"},
            {"padding", "Zero-padding added to both sides"}
        },
        {
            "Use padding='same' to preserve sequence length",
            "Good for NLP when combined with embeddings"
        },
        "Convolutional"
    };

    docs_[NodeType::Conv2D] = {
        "Conv2D",
        "2D convolution layer for image data. Slides a 2D kernel across the input image "
        "to extract spatial features. The foundation of CNNs for computer vision.",
        "Input shape: (batch, channels, height, width). Output: (batch, out_channels, new_h, new_w).",
        {
            {"filters", "Number of output channels/filters"},
            {"kernel_size", "Size of the 2D kernel (e.g., 3 for 3x3)"},
            {"stride", "Step size (e.g., 2 for downsampling)"},
            {"padding", "Zero-padding around the input"}
        },
        {
            "3x3 kernels are most common and efficient",
            "Use stride=2 instead of pooling for modern architectures",
            "Increase filters as you go deeper in the network"
        },
        "Convolutional"
    };

    docs_[NodeType::Conv3D] = {
        "Conv3D",
        "3D convolution layer for volumetric data. Used for video processing (time as 3rd dimension) "
        "or 3D medical imaging like CT/MRI scans.",
        "Input shape: (batch, channels, depth, height, width).",
        {
            {"filters", "Number of output channels"},
            {"kernel_size", "3D kernel size (d, h, w)"},
            {"stride", "Step size in each dimension"},
            {"padding", "Zero-padding for each dimension"}
        },
        {
            "Very memory intensive - start with small batch sizes",
            "Consider (1, 3, 3) kernels to reduce computation"
        },
        "Convolutional"
    };

    docs_[NodeType::DepthwiseConv2D] = {
        "Depthwise Conv2D",
        "Depthwise separable convolution that applies a single filter per input channel. "
        "Much more efficient than standard Conv2D, used in MobileNet and EfficientNet.",
        "Each input channel is convolved independently.",
        {
            {"kernel_size", "Size of the depthwise kernel"},
            {"stride", "Step size"},
            {"padding", "Zero-padding"}
        },
        {
            "Follow with 1x1 Conv2D (pointwise) for full depthwise separable conv",
            "Reduces parameters by ~8-9x compared to standard conv"
        },
        "Convolutional"
    };

    // ===== Pooling Layers =====
    docs_[NodeType::MaxPool2D] = {
        "MaxPool2D",
        "Downsamples by taking the maximum value in each pooling window. "
        "Provides translation invariance and reduces spatial dimensions.",
        "Commonly used with 2x2 kernel and stride 2 to halve dimensions.",
        {
            {"pool_size", "Size of the pooling window (e.g., 2 for 2x2)"},
            {"stride", "Step between windows (default: same as pool_size)"},
            {"padding", "Padding mode"}
        },
        {
            "Preserves strong activations, good for detecting features",
            "Consider stride in Conv2D as alternative in modern architectures"
        },
        "Pooling"
    };

    docs_[NodeType::AvgPool2D] = {
        "AvgPool2D",
        "Downsamples by taking the average value in each pooling window. "
        "Smoother than max pooling, preserves more spatial information.",
        "Similar to MaxPool2D but averages instead of taking maximum.",
        {
            {"pool_size", "Size of the pooling window"},
            {"stride", "Step between windows"},
            {"padding", "Padding mode"}
        },
        {
            "Better for regression tasks where all values matter",
            "Less aggressive than max pooling"
        },
        "Pooling"
    };

    docs_[NodeType::GlobalMaxPool] = {
        "Global Max Pooling",
        "Takes the maximum value across all spatial dimensions. "
        "Reduces each feature map to a single value.",
        "Output shape: (batch, channels). Often used before final Dense layer.",
        {},
        {
            "Alternative to Flatten that's input-size independent",
            "Good for variable-size inputs"
        },
        "Pooling"
    };

    docs_[NodeType::GlobalAvgPool] = {
        "Global Average Pooling",
        "Takes the average across all spatial dimensions. "
        "Popular in modern architectures like ResNet and EfficientNet.",
        "Reduces (batch, C, H, W) to (batch, C).",
        {},
        {
            "Reduces overfitting compared to large Dense layers",
            "Standard in most modern CNN architectures"
        },
        "Pooling"
    };

    docs_[NodeType::AdaptiveAvgPool] = {
        "Adaptive Average Pooling",
        "Pools to a fixed output size regardless of input size. "
        "Automatically calculates kernel and stride.",
        "Set output_size to (1,1) for global average pooling.",
        {
            {"output_size", "Target output dimensions (H, W)"}
        },
        {
            "Enables networks to accept variable input sizes",
            "Use (1,1) as GAP, (7,7) for certain transfer learning"
        },
        "Pooling"
    };

    // ===== Normalization Layers =====
    docs_[NodeType::BatchNorm] = {
        "Batch Normalization",
        "Normalizes activations by mini-batch statistics. Stabilizes training, "
        "allows higher learning rates, and provides slight regularization.",
        "Place after convolution/dense, before or after activation (debated).",
        {
            {"num_features", "Number of features/channels to normalize"},
            {"momentum", "Running statistics momentum (default: 0.1)"},
            {"eps", "Small constant for numerical stability"}
        },
        {
            "Essential for deep networks - enables training very deep models",
            "Has different behavior in training vs inference mode"
        },
        "Normalization"
    };

    docs_[NodeType::LayerNorm] = {
        "Layer Normalization",
        "Normalizes across the feature dimension rather than batch. "
        "Works with any batch size, essential for Transformers.",
        "Normalizes each sample independently.",
        {
            {"normalized_shape", "Shape of the normalized dimensions"},
            {"eps", "Numerical stability constant"}
        },
        {
            "Use in Transformers and RNNs",
            "Works with batch size 1 unlike BatchNorm"
        },
        "Normalization"
    };

    docs_[NodeType::GroupNorm] = {
        "Group Normalization",
        "Divides channels into groups and normalizes within each group. "
        "Combines benefits of LayerNorm and BatchNorm.",
        "Independent of batch size, good for small batch training.",
        {
            {"num_groups", "Number of groups to divide channels into"},
            {"num_channels", "Total number of channels"}
        },
        {
            "Use when batch size is too small for BatchNorm",
            "32 groups is a common default"
        },
        "Normalization"
    };

    docs_[NodeType::InstanceNorm] = {
        "Instance Normalization",
        "Normalizes each sample and channel independently. "
        "Popular in style transfer and image generation.",
        "Equivalent to GroupNorm with num_groups = num_channels.",
        {
            {"num_features", "Number of channels"}
        },
        {
            "Standard for style transfer networks",
            "Removes instance-specific contrast"
        },
        "Normalization"
    };

    // ===== Regularization =====
    docs_[NodeType::Dropout] = {
        "Dropout",
        "Randomly sets a fraction of inputs to zero during training. "
        "Prevents overfitting by forcing redundant representations.",
        "Disabled during inference (model.eval()).",
        {
            {"rate", "Fraction of inputs to drop (0.0-1.0)"}
        },
        {
            "0.2-0.5 is typical for Dense layers",
            "Use after Dense or before output layer"
        },
        "Regularization"
    };

    docs_[NodeType::Flatten] = {
        "Flatten",
        "Reshapes multi-dimensional input to 1D. Required between Conv layers "
        "and Dense layers to convert spatial features to a vector.",
        "Preserves batch dimension: (N, C, H, W) -> (N, C*H*W).",
        {},
        {
            "Place between last Conv/Pool and first Dense",
            "Consider GlobalAvgPool as alternative"
        },
        "Regularization"
    };

    // ===== Recurrent Layers =====
    docs_[NodeType::LSTM] = {
        "LSTM",
        "Long Short-Term Memory network for sequence modeling. Uses gates to "
        "control information flow, solving the vanishing gradient problem.",
        "Input: (batch, sequence_length, features). Can return all timesteps or just last.",
        {
            {"units", "Number of hidden units"},
            {"num_layers", "Number of stacked LSTM layers"},
            {"bidirectional", "Process sequence in both directions"},
            {"dropout", "Dropout between LSTM layers"}
        },
        {
            "Standard choice for sequence tasks",
            "Use bidirectional for classification, unidirectional for generation"
        },
        "Recurrent"
    };

    docs_[NodeType::GRU] = {
        "GRU",
        "Gated Recurrent Unit - simplified LSTM with fewer parameters. "
        "Often performs similarly with less computation.",
        "Faster than LSTM, good when data is limited.",
        {
            {"units", "Number of hidden units"},
            {"num_layers", "Number of stacked GRU layers"},
            {"bidirectional", "Process both directions"}
        },
        {
            "Try GRU first - simpler and often sufficient",
            "Switch to LSTM if GRU underperforms"
        },
        "Recurrent"
    };

    docs_[NodeType::RNN] = {
        "Simple RNN",
        "Basic recurrent layer with simple hidden state update. "
        "Suffers from vanishing gradients for long sequences.",
        "Use LSTM/GRU for better long-term memory.",
        {
            {"units", "Number of hidden units"},
            {"activation", "Activation function (default: tanh)"}
        },
        {
            "Only for very short sequences or educational purposes",
            "LSTM/GRU are almost always better"
        },
        "Recurrent"
    };

    docs_[NodeType::Bidirectional] = {
        "Bidirectional Wrapper",
        "Wraps an RNN layer to process the sequence in both directions. "
        "Doubles the output features (forward + backward).",
        "Use for tasks where future context helps (classification).",
        {},
        {
            "Output is concatenated forward and backward states",
            "Not suitable for autoregressive generation"
        },
        "Recurrent"
    };

    docs_[NodeType::Embedding] = {
        "Embedding",
        "Learns dense vector representations for discrete tokens (words, IDs). "
        "Maps indices to trainable vectors.",
        "Input: integer indices. Output: (batch, sequence, embedding_dim).",
        {
            {"num_embeddings", "Vocabulary size (number of unique tokens)"},
            {"embedding_dim", "Dimension of the dense vectors"}
        },
        {
            "Use pre-trained embeddings (GloVe, Word2Vec) for better results",
            "embedding_dim 128-512 is typical"
        },
        "Recurrent"
    };

    docs_[NodeType::TimeDistributed] = {
        "TimeDistributed",
        "Applies a layer to every temporal slice of an input. "
        "Useful for applying Dense/Conv to sequence outputs.",
        "Wraps a layer and applies it at each timestep.",
        {},
        {
            "Use to apply Dense to each LSTM output",
            "Preserves sequence structure"
        },
        "Recurrent"
    };

    // ===== Attention Layers =====
    docs_[NodeType::MultiHeadAttention] = {
        "Multi-Head Attention",
        "Core building block of Transformers. Computes attention with multiple parallel heads, "
        "allowing the model to attend to different representation subspaces.",
        "Takes Query, Key, Value inputs. Key and Value are often the same.",
        {
            {"embed_dim", "Total dimension of the model"},
            {"num_heads", "Number of parallel attention heads"},
            {"dropout", "Dropout on attention weights"}
        },
        {
            "embed_dim must be divisible by num_heads",
            "8 heads is typical, 12-16 for larger models"
        },
        "Attention"
    };

    docs_[NodeType::SelfAttention] = {
        "Self-Attention",
        "Attention where Query, Key, and Value all come from the same sequence. "
        "Each position attends to all positions in the input.",
        "Building block for encoders. Q=K=V=input.",
        {
            {"embed_dim", "Embedding dimension"},
            {"num_heads", "Number of attention heads"}
        },
        {
            "Add positional encoding for position awareness",
            "Use causal mask for autoregressive models"
        },
        "Attention"
    };

    docs_[NodeType::CrossAttention] = {
        "Cross-Attention",
        "Attention where Query comes from one sequence and Key/Value from another. "
        "Used in encoder-decoder architectures.",
        "Query from decoder, Key/Value from encoder output.",
        {
            {"embed_dim", "Embedding dimension"},
            {"num_heads", "Number of attention heads"}
        },
        {
            "Essential for seq2seq with attention",
            "Used in decoder layers of Transformers"
        },
        "Attention"
    };

    docs_[NodeType::LinearAttention] = {
        "Linear Attention",
        "O(n) complexity attention using kernel approximations. "
        "Scales to very long sequences unlike quadratic attention.",
        "Trade-off: faster but may lose some attention precision.",
        {
            {"embed_dim", "Embedding dimension"},
            {"num_heads", "Number of attention heads"}
        },
        {
            "Use for sequences longer than 2048 tokens",
            "Good for efficient inference"
        },
        "Attention"
    };

    docs_[NodeType::TransformerEncoder] = {
        "Transformer Encoder",
        "Stack of encoder layers with self-attention and feedforward networks. "
        "Used for encoding sequences (BERT-style models).",
        "Each layer: Self-Attention -> Add&Norm -> FFN -> Add&Norm.",
        {
            {"num_layers", "Number of encoder layers"},
            {"d_model", "Model dimension"},
            {"num_heads", "Number of attention heads"},
            {"d_ff", "Feedforward hidden dimension"}
        },
        {
            "6 layers is standard, 12 for BERT-base",
            "d_ff is typically 4x d_model"
        },
        "Attention"
    };

    docs_[NodeType::TransformerDecoder] = {
        "Transformer Decoder",
        "Stack of decoder layers with causal self-attention, cross-attention, "
        "and feedforward networks. Used for generation (GPT-style).",
        "Uses causal mask to prevent attending to future tokens.",
        {
            {"num_layers", "Number of decoder layers"},
            {"d_model", "Model dimension"},
            {"num_heads", "Number of attention heads"},
            {"d_ff", "Feedforward hidden dimension"}
        },
        {
            "For decoder-only (GPT): no cross-attention",
            "For encoder-decoder: add cross-attention"
        },
        "Attention"
    };

    docs_[NodeType::PositionalEncoding] = {
        "Positional Encoding",
        "Adds position information to embeddings using sinusoidal functions. "
        "Essential because attention is position-invariant.",
        "Add to embeddings before first attention layer.",
        {
            {"max_len", "Maximum sequence length"},
            {"d_model", "Model dimension"}
        },
        {
            "Learnable positional embeddings are an alternative",
            "Critical for Transformer performance"
        },
        "Attention"
    };

    // ===== Activation Functions =====
    docs_[NodeType::ReLU] = {
        "ReLU",
        "Rectified Linear Unit: f(x) = max(0, x). Simple, fast, and effective. "
        "The default choice for hidden layers.",
        "Can cause 'dying ReLU' if many neurons get stuck at 0.",
        {},
        {
            "Use as default activation",
            "Consider LeakyReLU if neurons are dying"
        },
        "Activation"
    };

    docs_[NodeType::LeakyReLU] = {
        "Leaky ReLU",
        "f(x) = x if x > 0, else alpha*x. Allows small gradients for negative values, "
        "preventing dying neurons.",
        "Typically alpha = 0.01.",
        {
            {"alpha", "Slope for negative values (default: 0.01)"}
        },
        {
            "Good default if ReLU causes dead neurons",
            "Often used in GANs"
        },
        "Activation"
    };

    docs_[NodeType::PReLU] = {
        "PReLU",
        "Parametric ReLU: LeakyReLU with learnable slope. "
        "The network learns the optimal negative slope.",
        "One parameter per channel or shared across all.",
        {},
        {
            "Slightly more parameters than LeakyReLU",
            "Can overfit on small datasets"
        },
        "Activation"
    };

    docs_[NodeType::ELU] = {
        "ELU",
        "Exponential Linear Unit: smooth version of ReLU for negative values. "
        "f(x) = x if x > 0, else alpha*(exp(x) - 1).",
        "Mean activations closer to zero, faster learning.",
        {
            {"alpha", "Scale for negative values (default: 1.0)"}
        },
        {
            "Better than ReLU for deep networks",
            "Slightly slower due to exponential"
        },
        "Activation"
    };

    docs_[NodeType::SELU] = {
        "SELU",
        "Scaled ELU with self-normalizing properties. "
        "Automatically keeps activations normalized through the network.",
        "Use with AlphaDropout, not standard Dropout.",
        {},
        {
            "Requires lecun_normal initialization",
            "Works best with fully-connected networks"
        },
        "Activation"
    };

    docs_[NodeType::GELU] = {
        "GELU",
        "Gaussian Error Linear Unit: smooth approximation of ReLU. "
        "The default activation in BERT, GPT, and modern Transformers.",
        "f(x) = x * Phi(x) where Phi is Gaussian CDF.",
        {},
        {
            "Standard for Transformer architectures",
            "Slightly better than ReLU for NLP"
        },
        "Activation"
    };

    docs_[NodeType::Swish] = {
        "Swish",
        "Self-gated activation: f(x) = x * sigmoid(x). "
        "Discovered by neural architecture search, often outperforms ReLU.",
        "Smooth, non-monotonic function.",
        {},
        {
            "Used in EfficientNet and modern CNNs",
            "Slightly more expensive than ReLU"
        },
        "Activation"
    };

    docs_[NodeType::Mish] = {
        "Mish",
        "f(x) = x * tanh(softplus(x)). Similar to Swish but smoother. "
        "Often slightly better than Swish in practice.",
        "More computationally expensive.",
        {},
        {
            "Try if Swish works well for your task",
            "Used in YOLOv4 and other vision models"
        },
        "Activation"
    };

    docs_[NodeType::Sigmoid] = {
        "Sigmoid",
        "f(x) = 1 / (1 + exp(-x)). Squashes input to (0, 1). "
        "Use for binary classification output or gates.",
        "Suffers from vanishing gradients in deep networks.",
        {},
        {
            "Use only for output layer (binary)",
            "ReLU/GELU better for hidden layers"
        },
        "Activation"
    };

    docs_[NodeType::Tanh] = {
        "Tanh",
        "f(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x)). "
        "Squashes input to (-1, 1). Zero-centered unlike Sigmoid.",
        "Common in LSTM gates and some normalization contexts.",
        {},
        {
            "Better than Sigmoid for hidden layers",
            "Still has vanishing gradient issues"
        },
        "Activation"
    };

    docs_[NodeType::Softmax] = {
        "Softmax",
        "Converts logits to probability distribution that sums to 1. "
        "Standard output for multi-class classification.",
        "softmax(x_i) = exp(x_i) / sum(exp(x_j)).",
        {
            {"dim", "Dimension to apply softmax (default: -1)"}
        },
        {
            "Use with CrossEntropyLoss (which includes Softmax)",
            "Don't use Softmax + NLLLoss (redundant)"
        },
        "Activation"
    };

    // ===== Shape Operations =====
    docs_[NodeType::Reshape] = {
        "Reshape",
        "Changes tensor dimensions without changing data. "
        "Total number of elements must remain the same.",
        "Use -1 for one dimension to infer automatically.",
        {
            {"shape", "Target shape tuple"}
        },
        {
            "Use to prepare data for different layer types",
            "-1 is useful when batch size varies"
        },
        "Shape Operations"
    };

    docs_[NodeType::Permute] = {
        "Permute",
        "Reorders tensor dimensions. Similar to numpy transpose but more general.",
        "Specify new order of dimensions.",
        {
            {"dims", "New dimension order (e.g., (0, 2, 1))"}
        },
        {
            "Common: (N,C,H,W) -> (N,H,W,C) for channels_last",
            "No data copy, just changes strides"
        },
        "Shape Operations"
    };

    docs_[NodeType::Squeeze] = {
        "Squeeze",
        "Removes dimensions of size 1. Cleans up extra dimensions.",
        "squeeze(dim) removes specific dim, squeeze() removes all size-1 dims.",
        {
            {"dim", "Dimension to squeeze (optional)"}
        },
        {
            "Useful after operations that add singleton dims",
            "Common after pooling to remove spatial dims"
        },
        "Shape Operations"
    };

    docs_[NodeType::Unsqueeze] = {
        "Unsqueeze",
        "Adds a dimension of size 1 at the specified position. "
        "Opposite of Squeeze.",
        "unsqueeze(0) adds batch dim, unsqueeze(-1) adds at end.",
        {
            {"dim", "Position to insert new dimension"}
        },
        {
            "Add batch dim: unsqueeze(0)",
            "Add channel dim: unsqueeze(1)"
        },
        "Shape Operations"
    };

    docs_[NodeType::View] = {
        "View",
        "Returns a new tensor with different shape but same data (PyTorch). "
        "Requires contiguous memory layout.",
        "Similar to Reshape but stricter about memory layout.",
        {
            {"shape", "Target shape"}
        },
        {
            "Call .contiguous() first if needed",
            "Slightly faster than Reshape when applicable"
        },
        "Shape Operations"
    };

    docs_[NodeType::Split] = {
        "Split",
        "Divides tensor into multiple parts along a dimension.",
        "Can split into equal parts or specify sizes.",
        {
            {"split_size", "Size of each split or list of sizes"},
            {"dim", "Dimension to split"}
        },
        {
            "Useful for multi-head attention",
            "Inverse of Concatenate"
        },
        "Shape Operations"
    };

    // ===== Merge Operations =====
    docs_[NodeType::Concatenate] = {
        "Concatenate",
        "Joins multiple tensors along a dimension. "
        "All tensors must have same shape except in the concat dimension.",
        "Common for skip connections (DenseNet) and merging branches.",
        {
            {"dim", "Dimension to concatenate along"}
        },
        {
            "Use for DenseNet-style skip connections",
            "Increases channel count"
        },
        "Merge Operations"
    };

    docs_[NodeType::Add] = {
        "Add",
        "Element-wise addition of multiple tensors. "
        "All tensors must have the same shape (or broadcastable).",
        "Standard for residual/skip connections (ResNet).",
        {},
        {
            "Use for ResNet-style skip connections",
            "Doesn't increase channel count"
        },
        "Merge Operations"
    };

    docs_[NodeType::Multiply] = {
        "Multiply",
        "Element-wise multiplication of tensors. "
        "Used for attention weights and gating mechanisms.",
        "All inputs must be broadcastable.",
        {},
        {
            "Common in attention mechanisms",
            "Use for feature modulation"
        },
        "Merge Operations"
    };

    docs_[NodeType::Average] = {
        "Average",
        "Element-wise average of multiple tensors. "
        "Smoother than Add, useful for ensemble-like behavior.",
        "All inputs must have same shape.",
        {},
        {
            "Use for model ensembling",
            "Less common than Add or Concatenate"
        },
        "Merge Operations"
    };

    // ===== Output =====
    docs_[NodeType::Output] = {
        "Output",
        "Marks the final output of the network. Connect the last layer here. "
        "Used for graph validation and code generation.",
        "Every valid graph must have at least one Output node.",
        {},
        {
            "Connect model output AND loss output",
            "Multiple outputs are supported"
        },
        "Output"
    };

    // ===== Loss Functions =====
    docs_[NodeType::MSELoss] = {
        "Mean Squared Error Loss",
        "Average of squared differences: mean((y_pred - y_true)^2). "
        "Standard loss for regression tasks.",
        "Heavily penalizes large errors (outliers).",
        {},
        {
            "Use for regression (continuous outputs)",
            "Consider SmoothL1Loss if outliers are common"
        },
        "Loss Functions"
    };

    docs_[NodeType::CrossEntropyLoss] = {
        "Cross Entropy Loss",
        "Combines LogSoftmax and NLLLoss. Standard for multi-class classification. "
        "Input: raw logits (not softmaxed). Target: class indices.",
        "Numerically stable implementation.",
        {
            {"weight", "Class weights for imbalanced data"},
            {"label_smoothing", "Smooth labels to prevent overconfidence"}
        },
        {
            "Don't apply Softmax before this loss",
            "Use class weights for imbalanced datasets"
        },
        "Loss Functions"
    };

    docs_[NodeType::BCELoss] = {
        "Binary Cross Entropy Loss",
        "Loss for binary classification. Input must be probabilities (after Sigmoid). "
        "-[y*log(p) + (1-y)*log(1-p)].",
        "Use BCEWithLogits for better numerical stability.",
        {},
        {
            "Input must be in (0,1) - apply Sigmoid first",
            "Prefer BCEWithLogits in practice"
        },
        "Loss Functions"
    };

    docs_[NodeType::BCEWithLogits] = {
        "BCE with Logits Loss",
        "Binary cross entropy with built-in Sigmoid. More numerically stable than "
        "separate Sigmoid + BCELoss. The preferred choice for binary classification.",
        "Input: raw logits. Sigmoid is applied internally.",
        {
            {"pos_weight", "Weight for positive class (for imbalanced data)"}
        },
        {
            "Standard choice for binary classification",
            "Works well for multi-label classification too"
        },
        "Loss Functions"
    };

    docs_[NodeType::L1Loss] = {
        "L1 Loss (MAE)",
        "Mean Absolute Error: mean(|y_pred - y_true|). "
        "More robust to outliers than MSE.",
        "Linear penalty regardless of error magnitude.",
        {},
        {
            "Use when outliers are common",
            "Gradients don't diminish for large errors"
        },
        "Loss Functions"
    };

    docs_[NodeType::SmoothL1Loss] = {
        "Smooth L1 Loss (Huber)",
        "Combines L1 and L2: L2 for small errors, L1 for large errors. "
        "Best of both worlds for regression.",
        "Threshold 'beta' controls the switch point.",
        {
            {"beta", "Threshold for L1/L2 switch (default: 1.0)"}
        },
        {
            "Standard for bounding box regression (object detection)",
            "Good default for robust regression"
        },
        "Loss Functions"
    };

    docs_[NodeType::HuberLoss] = {
        "Huber Loss",
        "Same as Smooth L1 Loss. Quadratic for small errors, linear for large errors.",
        "Configurable threshold (delta).",
        {
            {"delta", "Threshold for quadratic/linear switch"}
        },
        {
            "Use when you want to limit influence of outliers",
            "Smooth transition between L2 and L1"
        },
        "Loss Functions"
    };

    docs_[NodeType::NLLLoss] = {
        "Negative Log Likelihood Loss",
        "Use with LogSoftmax output. For multi-class classification. "
        "Usually prefer CrossEntropyLoss which combines both.",
        "Target: class indices (not one-hot).",
        {
            {"weight", "Class weights"}
        },
        {
            "Requires LogSoftmax activation before",
            "CrossEntropyLoss is usually more convenient"
        },
        "Loss Functions"
    };

    // ===== Optimizers =====
    docs_[NodeType::SGD] = {
        "Stochastic Gradient Descent",
        "Classic optimizer. Simple but requires careful learning rate tuning. "
        "Add momentum for faster convergence.",
        "weight = weight - lr * gradient.",
        {
            {"lr", "Learning rate (e.g., 0.01)"},
            {"momentum", "Momentum factor (0.9 typical)"},
            {"weight_decay", "L2 regularization"}
        },
        {
            "Use momentum >= 0.9 for better convergence",
            "May generalize better than Adam for some tasks"
        },
        "Optimizers"
    };

    docs_[NodeType::Adam] = {
        "Adam Optimizer",
        "Adaptive learning rates per parameter. Combines momentum and RMSprop. "
        "Good default choice that works well for most tasks.",
        "Tracks running mean and variance of gradients.",
        {
            {"lr", "Learning rate (e.g., 0.001)"},
            {"betas", "Coefficients for running averages (0.9, 0.999)"},
            {"eps", "Numerical stability (1e-8)"}
        },
        {
            "Default choice for most deep learning",
            "lr=3e-4 is a common starting point"
        },
        "Optimizers"
    };

    docs_[NodeType::AdamW] = {
        "AdamW Optimizer",
        "Adam with decoupled weight decay. Fixes L2 regularization in Adam. "
        "Better generalization than standard Adam.",
        "Weight decay applied directly to weights, not to gradient.",
        {
            {"lr", "Learning rate"},
            {"betas", "Running average coefficients"},
            {"weight_decay", "Weight decay coefficient (0.01 typical)"}
        },
        {
            "Preferred over Adam when using weight decay",
            "Standard for Transformer fine-tuning"
        },
        "Optimizers"
    };

    docs_[NodeType::RMSprop] = {
        "RMSprop Optimizer",
        "Adaptive learning rate based on recent gradient magnitudes. "
        "Good for non-stationary objectives and RNNs.",
        "Divides learning rate by running average of recent gradient magnitudes.",
        {
            {"lr", "Learning rate (0.01 typical)"},
            {"alpha", "Smoothing constant (0.99)"},
            {"eps", "Numerical stability"}
        },
        {
            "Often works well for RNNs",
            "Predecessor to Adam"
        },
        "Optimizers"
    };

    docs_[NodeType::Adagrad] = {
        "Adagrad Optimizer",
        "Adapts learning rate per parameter based on historical gradients. "
        "Good for sparse data but learning rate decays aggressively.",
        "Parameters with large gradients get smaller updates.",
        {
            {"lr", "Initial learning rate"},
            {"eps", "Numerical stability"}
        },
        {
            "Good for sparse features (NLP, recommender systems)",
            "Learning rate may decay too fast for deep learning"
        },
        "Optimizers"
    };

    docs_[NodeType::NAdam] = {
        "NAdam Optimizer",
        "Adam with Nesterov momentum. Looks ahead in gradient direction. "
        "Often slightly faster convergence than Adam.",
        "Combines Adam's adaptivity with Nesterov's look-ahead.",
        {
            {"lr", "Learning rate"},
            {"betas", "Running average coefficients"}
        },
        {
            "Try if Adam is working well",
            "May converge faster in some cases"
        },
        "Optimizers"
    };

    // ===== LR Schedulers =====
    docs_[NodeType::StepLR] = {
        "Step LR Scheduler",
        "Decays learning rate by gamma every step_size epochs. "
        "Simple and predictable schedule.",
        "new_lr = lr * gamma^(epoch // step_size).",
        {
            {"step_size", "Epochs between LR decay"},
            {"gamma", "Multiplicative factor (e.g., 0.1)"}
        },
        {
            "Common: decay by 0.1 every 30 epochs",
            "Good for fixed training schedules"
        },
        "LR Schedulers"
    };

    docs_[NodeType::CosineAnnealing] = {
        "Cosine Annealing LR",
        "Smoothly decreases LR following a cosine curve to eta_min. "
        "Warm restarts version resets periodically.",
        "LR follows cosine from initial to minimum over T_max epochs.",
        {
            {"T_max", "Maximum number of iterations"},
            {"eta_min", "Minimum learning rate"}
        },
        {
            "Good for training to convergence",
            "Often combined with warm restarts"
        },
        "LR Schedulers"
    };

    docs_[NodeType::ReduceOnPlateau] = {
        "Reduce LR on Plateau",
        "Reduces LR when a metric stops improving. Adaptive schedule based on validation loss.",
        "Monitors a metric and reduces LR after 'patience' epochs of no improvement.",
        {
            {"mode", "min or max (track loss or accuracy)"},
            {"factor", "Factor to reduce LR by"},
            {"patience", "Epochs to wait before reducing"}
        },
        {
            "Requires passing validation loss each epoch",
            "Good when optimal schedule is unknown"
        },
        "LR Schedulers"
    };

    docs_[NodeType::ExponentialLR] = {
        "Exponential LR Scheduler",
        "Decays LR by gamma every epoch. Continuous decay.",
        "new_lr = lr * gamma^epoch.",
        {
            {"gamma", "Decay rate per epoch (e.g., 0.95)"}
        },
        {
            "Smooth continuous decay",
            "May decay too fast - use gamma close to 1"
        },
        "LR Schedulers"
    };

    docs_[NodeType::WarmupScheduler] = {
        "Warmup Scheduler",
        "Linearly increases LR during initial epochs before main schedule. "
        "Helps with training stability for large batches.",
        "LR starts at 0 and increases to target over warmup epochs.",
        {
            {"warmup_epochs", "Number of warmup epochs"},
            {"initial_lr", "Target learning rate after warmup"}
        },
        {
            "Essential for Transformer training",
            "5-10% of total epochs is typical warmup"
        },
        "LR Schedulers"
    };

    // ===== Regularization Nodes =====
    docs_[NodeType::L1Regularization] = {
        "L1 Regularization (Lasso)",
        "Adds sum of absolute weights to loss. Encourages sparsity - "
        "some weights become exactly zero.",
        "loss = original_loss + lambda * sum(|weights|).",
        {
            {"lambda", "Regularization strength"}
        },
        {
            "Use for feature selection (zeroes unimportant weights)",
            "Usually applied via optimizer weight_decay"
        },
        "Regularization"
    };

    docs_[NodeType::L2Regularization] = {
        "L2 Regularization (Ridge)",
        "Adds sum of squared weights to loss. Encourages small weights, "
        "prevents any weight from being too large.",
        "loss = original_loss + lambda * sum(weights^2).",
        {
            {"lambda", "Regularization strength"}
        },
        {
            "Standard regularization, usually via weight_decay",
            "Doesn't produce sparse weights like L1"
        },
        "Regularization"
    };

    docs_[NodeType::ElasticNet] = {
        "Elastic Net Regularization",
        "Combines L1 and L2 regularization. Balance between sparsity (L1) "
        "and weight shrinkage (L2).",
        "loss = original_loss + alpha*L1 + beta*L2.",
        {
            {"l1_ratio", "Balance between L1 and L2 (0-1)"},
            {"alpha", "Overall regularization strength"}
        },
        {
            "Good when you want some sparsity with stability",
            "l1_ratio=0.5 for equal L1/L2"
        },
        "Regularization"
    };

    // ===== Utility Nodes =====
    docs_[NodeType::Lambda] = {
        "Lambda Layer",
        "Wraps an arbitrary function as a layer. For custom operations "
        "not covered by standard layers.",
        "Define any tensor transformation.",
        {
            {"function", "Python lambda or function name"}
        },
        {
            "Use for simple custom ops",
            "Consider custom Layer class for complex logic"
        },
        "Utility"
    };

    docs_[NodeType::Identity] = {
        "Identity",
        "Passes input through unchanged. Useful as a placeholder "
        "or for conditional bypass.",
        "Output equals input exactly.",
        {},
        {
            "Use in conditional architectures",
            "Helpful during model development"
        },
        "Utility"
    };

    docs_[NodeType::Constant] = {
        "Constant",
        "Outputs a fixed constant value. Used for fixed biases "
        "or reference values.",
        "Not trainable.",
        {
            {"value", "The constant value or tensor"}
        },
        {
            "Use for fixed parameters",
            "Consider Parameter node for trainable version"
        },
        "Utility"
    };

    docs_[NodeType::Parameter] = {
        "Parameter",
        "A trainable parameter tensor. Can be initialized and updated "
        "during training like layer weights.",
        "Registered as model parameter.",
        {
            {"shape", "Shape of the parameter tensor"},
            {"init", "Initialization method"}
        },
        {
            "Use for learnable embeddings or scaling factors",
            "Will be updated by optimizer"
        },
        "Utility"
    };

    // ===== Data Pipeline Nodes =====
    docs_[NodeType::DatasetInput] = {
        "Dataset Input",
        "Loads a dataset from the Data Registry. This is the entry point "
        "for training data into your model.",
        "Select a loaded dataset to use for training.",
        {
            {"dataset", "Name of dataset in Data Registry"}
        },
        {
            "Load dataset in Dataset Panel first",
            "Returns (features, labels) tuple"
        },
        "Data Pipeline"
    };

    docs_[NodeType::DataLoader] = {
        "Data Loader",
        "Creates batched iterator with shuffling and parallel loading. "
        "Wraps dataset for efficient training.",
        "Outputs batches of (features, labels).",
        {
            {"batch_size", "Samples per batch"},
            {"shuffle", "Randomize order each epoch"},
            {"drop_last", "Drop incomplete final batch"}
        },
        {
            "shuffle=True for training, False for validation",
            "Larger batch_size = faster but more memory"
        },
        "Data Pipeline"
    };

    docs_[NodeType::Augmentation] = {
        "Augmentation",
        "Applies data augmentation transforms. Increases effective "
        "dataset size through variations.",
        "Applied during training, disabled during inference.",
        {
            {"transforms", "List of transforms to apply"}
        },
        {
            "Use for images: flip, rotate, color jitter",
            "Reduces overfitting significantly"
        },
        "Data Pipeline"
    };

    docs_[NodeType::DataSplit] = {
        "Data Split",
        "Splits dataset into training, validation, and optionally test sets. "
        "Essential for proper model evaluation.",
        "Outputs multiple dataset handles.",
        {
            {"train_ratio", "Fraction for training (e.g., 0.8)"},
            {"val_ratio", "Fraction for validation (e.g., 0.1)"},
            {"shuffle", "Shuffle before splitting"}
        },
        {
            "80/10/10 is a common split",
            "Always shuffle for non-time-series data"
        },
        "Data Pipeline"
    };

    docs_[NodeType::TensorReshape] = {
        "Tensor Reshape (Legacy)",
        "Reshapes tensor dimensions. Legacy node - prefer the Reshape node instead.",
        "Same as Reshape but with older interface.",
        {
            {"shape", "Target shape"}
        },
        {
            "Use Reshape node instead",
            "Kept for backward compatibility"
        },
        "Data Pipeline"
    };

    docs_[NodeType::Normalize] = {
        "Normalize",
        "Normalizes tensor values using mean and standard deviation. "
        "Common preprocessing step for neural networks.",
        "output = (input - mean) / std.",
        {
            {"mean", "Mean values per channel"},
            {"std", "Standard deviation per channel"}
        },
        {
            "ImageNet mean/std for pre-trained models",
            "Compute on training set for custom data"
        },
        "Data Pipeline"
    };

    docs_[NodeType::OneHotEncode] = {
        "One-Hot Encode",
        "Converts class indices to one-hot vectors. "
        "[0,1,2] with 3 classes -> [[1,0,0], [0,1,0], [0,0,1]].",
        "Used when loss function expects one-hot labels.",
        {
            {"num_classes", "Total number of classes"}
        },
        {
            "CrossEntropyLoss doesn't need one-hot",
            "Required for some custom loss functions"
        },
        "Data Pipeline"
    };
}

} // namespace gui
