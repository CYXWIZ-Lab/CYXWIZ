#include "node_documentation.h"
#include "../core/node_metadata_registry.h"
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
    auto& registry = cyxwiz::NodeMetadataRegistry::Instance();
    registry.Initialize();
    const auto* metadata = registry.GetMetadata(type);
    const NodeDocumentation* doc = GetDocumentation(type);
    if (!metadata && !doc) return;

    const std::string title = metadata ? metadata->name : doc->title;
    const std::string category = metadata
        ? cyxwiz::GetCategoryDisplayName(metadata->category)
        : doc->category;
    const std::string brief = metadata
        ? metadata->brief_description
        : doc->description;
    const std::string details = metadata
        ? metadata->help_text
        : std::string{};
    const std::string usage = metadata
        ? metadata->example_usage
        : doc->usage;

    ImGui::BeginTooltip();

    // Title with category
    ImGui::TextColored(ImVec4(0.4f, 0.8f, 1.0f, 1.0f), "%s", title.c_str());
    ImGui::SameLine();
    ImGui::TextDisabled("[%s]", category.c_str());

    ImGui::Separator();

    // Description
    ImGui::PushTextWrapPos(400.0f);
    if (!brief.empty()) {
        ImGui::TextWrapped("%s", brief.c_str());
    }
    if (!details.empty() && details != brief) {
        ImGui::Spacing();
        ImGui::TextWrapped("%s", details.c_str());
    }
    ImGui::PopTextWrapPos();

    // Parameters
    if (metadata && !metadata->parameters.empty()) {
        ImGui::Spacing();
        ImGui::TextColored(ImVec4(0.8f, 0.8f, 0.4f, 1.0f), "Parameters:");
        for (const auto& param : metadata->parameters) {
            const std::string& label = param.display_name.empty()
                ? param.name
                : param.display_name;
            ImGui::BulletText("%s%s", label.c_str(),
                              param.required ? " *" : "");
            ImGui::SameLine();
            ImGui::TextDisabled("- %s", param.description.c_str());
        }
    } else if (doc && !doc->parameters.empty()) {
        ImGui::Spacing();
        ImGui::TextColored(ImVec4(0.8f, 0.8f, 0.4f, 1.0f), "Parameters:");
        for (const auto& param : doc->parameters) {
            ImGui::BulletText("%s", param.first.c_str());
            ImGui::SameLine();
            ImGui::TextDisabled("- %s", param.second.c_str());
        }
    }

    // Legacy documentation remains a fallback for types not represented in
    // the authoritative metadata registry.
    if (!metadata && doc && !doc->tips.empty()) {
        ImGui::Spacing();
        ImGui::TextColored(ImVec4(0.4f, 1.0f, 0.6f, 1.0f), "Tips:");
        for (const auto& tip : doc->tips) {
            ImGui::BulletText("%s", tip.c_str());
        }
    }

    // Usage example
    if (!usage.empty()) {
        ImGui::Spacing();
        ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.8f, 1.0f), "Usage:");
        ImGui::TextDisabled("%s", usage.c_str());
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
        case NodeType::TimeDistributed:
        case NodeType::Embedding:
            return "Recurrent";

        // Attention
        case NodeType::MultiHeadAttention:
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

        case NodeType::TensorSum:
        case NodeType::TensorMean:
        case NodeType::TensorMax:
        case NodeType::TensorMin:
        case NodeType::TensorProd:
        case NodeType::TensorVar:
        case NodeType::TensorStd:
        case NodeType::TensorBroadcastTo:
        case NodeType::TensorExpand:
        case NodeType::TensorIndexSelect:
        case NodeType::TensorPow:
        case NodeType::TensorSqrt:
        case NodeType::TensorExp:
        case NodeType::TensorLog:
        case NodeType::TensorAbs:
        case NodeType::TensorSign:
        case NodeType::TensorClip:
        case NodeType::TensorDot:
        case NodeType::TensorBatchMatMul:
        case NodeType::TensorCompare:
        case NodeType::TensorLogicalMask:
            return "Tensor Operations";

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
        case NodeType::SoftDiceLoss:
        case NodeType::TverskyLoss:
        case NodeType::JaccardLoss:
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
        "1D convolution over a sequence: slides a kernel along the length to extract local "
        "patterns. Put it first for time-series windows, audio features or table rows (channel-major "
        "rows), or after an Embedding for text. End with Flatten or Global Avg Pool before Dense.",
        "[L, C] -> [floor((L + 2p - k) / s) + 1, filters], as torch.nn.Conv1d on (batch, channels, "
        "length).",
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
        "Convolves each input channel with its own depth_multiplier kernels. Far fewer weights than "
        "Conv2D; used in MobileNet and EfficientNet.",
        "[H,W,C] -> [H',W',C*M], as torch.nn.Conv2d(C, C*M, k, groups=C): output channel c*M + m "
        "sees only input channel c. H' = floor((H + 2p - k) / s) + 1.",
        {
            {"kernel_size", "Square kernel size k"},
            {"stride", "Step size s"},
            {"padding", "same ((k-1)/2, odd k) or valid (0)"},
            {"depth_multiplier", "Output channels per input channel M"}
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
        "Takes each channel's maximum over its height and width (or over the sequence length after "
        "Conv1D). Ends the convolution section the way Flatten does, with one value per channel.",
        "[H,W,C] -> [C] or [L,C] -> [C], as torch adaptive_max_pool2d / adaptive_max_pool1d(x, 1)"
        ".flatten(1); the gradient goes to the first position holding the maximum.",
        {},
        {
            "Max-over-time pooling for text CNNs: Embedding -> Conv1D -> ReLU -> Global Max Pool",
            "Far fewer Dense weights than Flatten: C inputs"
        },
        "Pooling"
    };

    docs_[NodeType::GlobalAvgPool] = {
        "Global Average Pooling",
        "Averages each channel over its height and width. Ends the convolution section of a model "
        "the way Flatten does, with one value per channel. Used by ResNet and EfficientNet.",
        "[H,W,C] -> [C], as torch adaptive_avg_pool2d(x, 1).flatten(1); the gradient spreads each "
        "channel's gradient evenly over its H x W positions.",
        {},
        {
            "Place it after the last convolution block, before Dense",
            "Far fewer Dense weights than Flatten: C inputs instead of H x W x C"
        },
        "Pooling"
    };

    docs_[NodeType::AdaptiveAvgPool] = {
        "Adaptive Average Pooling",
        "Average-pools a feature map to a fixed output_size x output_size, whatever its input size.",
        "[H,W,C] -> [s,s,C], as torch.nn.AdaptiveAvgPool2d(s): cell i averages rows floor(i*H/s) to "
        "ceil((i+1)*H/s) - 1 and the same columns; bins may overlap.",
        {
            {"output_size", "Output height and width s"}
        },
        {
            "output_size 1 is a global average pool that keeps [1,1,C]; Flatten then gives C",
            "Before Flatten it fixes the Dense input size: s x s x C"
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
            {"num_channels", "Total number of input channels"},
            {"eps", "Positive numerical stability constant"},
            {"affine", "Learn one scale and bias per channel"}
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
            {"num_features", "Number of input channels"},
            {"eps", "Positive numerical stability constant"},
            {"affine", "Learn one scale and bias per channel"}
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
        "control information flow. Engine training currently supports the "
        "unidirectional configuration with dropout=0.0.",
        "Input: [batch, sequence_length, features]. Output can retain all "
        "timesteps or select the final timestep.",
        {
            {"input_size", "Read-only input feature size derived from upstream"},
            {"hidden_size", "Number of hidden units"},
            {"num_layers", "Number of stacked LSTM layers"},
            {"bidirectional", "Must remain false for Engine training; reverse backward is incomplete"},
            {"return_sequences", "Return all timesteps instead of only the final timestep"},
            {"dropout", "Must remain 0.0; use a separate Dropout node"}
        },
        {
            "Use the Output pin; the legacy Hidden pin is not routed separately",
            "Compile fails closed for bidirectional training or nonzero recurrent dropout"
        },
        "Recurrent"
    };

    docs_[NodeType::GRU] = {
        "GRU",
        "Gated Recurrent Unit - simplified LSTM with fewer parameters. "
        "Engine training supports unidirectional and split-path bidirectional "
        "execution with dropout=0.0.",
        "Input: [batch, sequence_length, features]. Output can retain all "
        "timesteps or select the final timestep.",
        {
            {"input_size", "Read-only input feature size derived from upstream"},
            {"hidden_size", "Number of hidden units"},
            {"num_layers", "Number of stacked GRU layers"},
            {"bidirectional", "Run explicit forward and reverse GRU branches"},
            {"return_sequences", "Return all timesteps instead of only the final timestep"},
            {"dropout", "Must remain 0.0; use a separate Dropout node"}
        },
        {
            "Bidirectional GRU currently uses the native CPU recurrent path",
            "Use the Output pin; the legacy Hidden pin is not routed separately"
        },
        "Recurrent"
    };

    docs_[NodeType::RNN] = {
        "Simple RNN",
        "Trainable simple (Elman) recurrent layer: h_t = act(W_ih x_t + b_ih + W_hh h_{t-1} + b_hh).",
        "Runs on the native CPU simple-RNN reference layer with tanh or relu "
        "nonlinearity, or on the native neural provider (CUDA/OpenCL) when one "
        "serves the run's device. dropout=0.0 only; bidirectional runs as "
        "split forward/reverse branches.",
        {
            {"input_size", "Input feature size per timestep (auto-derived)"},
            {"hidden_size", "Number of hidden units"},
            {"num_layers", "Number of stacked layers"},
            {"bidirectional", "Run explicit forward and reverse RNN branches"},
            {"return_sequences", "Return every timestep instead of the last"},
            {"dropout", "Must remain 0.0; use an explicit Dropout node"},
            {"nonlinearity", "Cell activation: tanh or relu"}
        },
        {
            "Trains on the CPU reference path unless a native provider serves the run's device",
            "The placement audit shows which path was selected for the run"
        },
        "Recurrent"
    };

    docs_[NodeType::Embedding] = {
        "Embedding",
        "Looks up trainable dense vectors for exact integer token IDs.",
        "Input is [sequence] or [batch, sequence]. Output appends embedding_dim. "
        "A configured padding row produces zeros and receives no gradient.",
        {
            {"num_embeddings", "Vocabulary size (number of unique tokens)"},
            {"embedding_dim", "Dimension of each dense token vector"},
            {"padding_idx", "Padding token id, or -1 to disable padding"},
            {"max_norm", "Optional per-row L2 norm cap; 0 disables clipping"},
            {"weights_file", "Optional pretrained Float32 text matrix"},
            {"freeze", "Keep loaded pretrained weights fixed during training"}
        },
        {
            "Vocabulary size must match the tokenizer's emitted token-id range",
            "Use the configuration dialog to inspect or build a matrix file"
        },
        "Recurrent"
    };

    docs_[NodeType::TimeDistributed] = {
        "TimeDistributed Dense",
        "Applies one shared bias-enabled Dense projection to every timestep.",
        "Input [batch, sequence, features] becomes [batch, sequence, units]. "
        "This node is a specific Dense sequence head, not a generic layer wrapper.",
        {{"units", "Per-timestep output features/classes"},
         {"tie_embedding", "Language-model head: reuse the upstream Embedding table (logits = h E^T, no bias)"}},
        {
            "Use after a sequence-producing recurrent or transformer layer",
            "All timesteps share the same weights and bias",
            "tie_embedding needs input width = embedding_dim and units = vocabulary size; PyTorch export does not cover this node yet"
        },
        "Recurrent"
    };

    // ===== Attention Layers =====
    docs_[NodeType::MultiHeadAttention] = {
        "Multi-Head Attention",
        "One CPU-backed unary self-attention block with multiple parallel heads.",
        "Current Studio execution uses Input as Query, Key, and Value. The legacy "
        "Key, Value, and Mask pins are reserved and fail closed when connected.",
        {
            {"embed_dim", "Total dimension of the model"},
            {"num_heads", "Number of parallel attention heads"},
            {"dropout", "Dropout on attention weights"},
            {"use_bias", "Whether attention projections use bias"}
        },
        {
            "embed_dim must be divisible by num_heads",
            "8 heads is typical, 12-16 for larger models"
        },
        "Attention"
    };

    docs_[NodeType::CrossAttention] = {
        "Cross-Attention",
        "Trainable attention of a Query sequence over a Key / Value sequence.",
        "Each Query position attends over the Key / Value sequence (softmax(Q K^T / sqrt(d)) V per head) "
        "with learned projections, as torch.nn.MultiheadAttention(query, key, value) with batch_first. "
        "Query, Key and Value usually come from different graph branches; the output has the Query's shape.",
        {
            {"embed_dim", "Feature width of Query, Key and Value"},
            {"num_heads", "Number of attention heads (embed_dim must divide evenly)"},
            {"dropout", "Dropout on the attention weights while training"},
            {"use_bias", "Bias in the attention projections"}
        },
        {
            "Link the same layer to Key and Value for the usual encoder memory input",
            "Query and Key may have different lengths; Key and Value must match",
            "Code export does not support Cross Attention yet"
        },
        "Attention"
    };

    docs_[NodeType::LinearAttention] = {
        "Linear Attention",
        "Blocked compatibility node for a historical kernel-attention design.",
        "No backend primitive or Studio compiler/model owner currently implements "
        "the advertised linear-attention semantics.",
        {
            {"embed_dim", "Embedding dimension"},
            {"num_heads", "Number of attention heads"},
            {"feature_map", "Historical kernel feature-map selection"},
            {"eps", "Historical numerical-stability epsilon"},
            {"causal", "Historical causal-attention intent"}
        },
        {
            "This node can be inspected in saved graphs but cannot compile or train",
            "Code export must not replace it with ordinary quadratic attention"
        },
        "Attention"
    };

    docs_[NodeType::TransformerEncoder] = {
        "Transformer Encoder",
        "One CPU-backed encoder block with self-attention and a feedforward network.",
        "Stack multiple nodes to create a deeper encoder. One node does not expand "
        "a num_layers setting internally.",
        {
            {"d_model", "Model dimension"},
            {"num_heads", "Number of attention heads"},
            {"dim_feedforward", "Feedforward hidden dimension"},
            {"dropout", "Dropout probability"},
            {"ffn_dropout", "Hidden FFN dropout after activation; default 0 preserves legacy behavior"},
            {"norm_first", "Apply normalization before attention and feedforward"}
        },
        {
            "d_model must divide evenly by num_heads",
            "dim_feedforward is commonly 4x d_model"
        },
        "Attention"
    };

    docs_[NodeType::TransformerDecoder] = {
        "Transformer Decoder",
        "One decoder-only Transformer block with masked self-attention, residual "
        "paths, normalization, and an internal feed-forward network. The feed-forward "
        "path is Dense(d_model -> dim_feedforward) -> activation -> Dense(dim_feedforward -> d_model), "
        "or a gated GLU-family network. Defaults are the original 2017 block; architecture_preset=llama_style "
        "sets norm_first=true, norm_type=rms_norm, ffn_type=gated, ffn_activation=silu (SwiGLU), ffn_bias=false, "
        "position_encoding=rope, attention_bias=false. qk_norm adds per-head query/key RMSNorm (OLMo 2, Gemma 3, Qwen3). "
        "Also: ALiBi positions, partial RoPE, parallel (GPT-J/PaLM) and sandwich (Gemma) layouts, grouped-query "
        "attention, sliding window, logit soft-cap and GPT-2 residual init scaling.",
        "Stack multiple nodes for depth. Internal attention projections and Dense/FC "
        "layers are owned by this composite node; edit dim_feedforward to change the "
        "hidden Dense width. Add another TransformerDecoder node to add a block. The Memory pin is reserved and fails "
        "closed when connected because cross-attention is not yet owned by the "
        "Studio runtime.",
        {
            {"d_model", "Model dimension"},
            {"num_heads", "Number of attention heads"},
            {"dim_feedforward", "Feedforward hidden dimension"},
            {"dropout", "Dropout probability"},
            {"ffn_dropout", "Hidden FFN dropout after activation; default 0 preserves legacy behavior"},
            {"norm_first", "Apply normalization before attention and feedforward (pre-norm)"},
            {"norm_type", "layer_norm or rms_norm (RMSNorm: scale only, no mean centring; LLaMA)"},
            {"norm_eps", "Epsilon inside the normalization square root (default 1e-5)"},
            {"ffn_type", "mlp, or gated: act(gate(x)) * up(x) then down; adds a third weight matrix"},
            {"architecture_preset", "custom, classic or llama_style; fills the block fields, editing one switches to custom"},
            {"ffn_activation", "relu, gelu (tanh approximation), gelu_exact (erf), silu/Swish, mish, elu, selu, leaky_relu, sigmoid, tanh, hardswish, squared_relu; gated+silu = SwiGLU, gated+gelu = GEGLU"},
            {"ffn_bias", "Learn biases in the feed-forward Dense layers (LLaMA turns this off)"},
            {"position_encoding", "external (Positional Encoding node) or rope (rotary embedding inside self-attention)"},
            {"rope_base", "Rotary frequency base, default 10000"},
            {"attention_bias", "Learn biases in the attention Q/K/V/output projections (LLaMA turns this off)"},
            {"qk_norm", "Per-head RMSNorm of queries and keys before the scores; stabilises training at higher learning rates"},
            {"rope_fraction", "Partial RoPE: fraction of head features rotated (GPT-NeoX 0.25)"},
            {"block_layout", "sequential, or parallel (GPT-J/PaLM: x + attn(n(x)) + ffn(n(x)); pre-norm only)"},
            {"sandwich_norm", "Gemma-style extra norm on each sub-layer output before the residual add (pre-norm only)"},
            {"num_kv_heads", "Grouped-query attention key/value heads; 0 = num_heads, 1 = multi-query"},
            {"sliding_window", "Mistral-style local causal window; 0 = full"},
            {"attn_logit_softcap", "Gemma 2 soft-cap on attention scores; 0 = off"},
            {"residual_init_scale", "Scales initial attention-output and FFN-down weights (GPT-2: 1/sqrt(2N))"}
        },
        {
            "d_model must divide evenly by num_heads",
            "Autoregressive generation loops remain a separate future contract",
            "position_encoding=rope/alibi, qk_norm, grouped-query attention and soft-capping run on the ArrayFire attention path only; the native CPU fallback refuses them with a clear error"
        },
        "Attention"
    };

    docs_[NodeType::PositionalEncoding] = {
        "Positional Encoding",
        "Adds deterministic sinusoidal position information to rank-3 embeddings.",
        "This node is CPU-backed, has no trainable parameters, and preserves the "
        "input shape. Add it before the first attention block.",
        {
            {"d_model", "Model dimension"},
            {"max_sequence_length", "Maximum supported sequence length"}
        },
        {
            "The input feature width must match d_model",
            "This implementation does not apply dropout"
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
        "Divides a tensor in two along a sample dimension.",
        "Output 1 takes the first split_size entries along dim and Output 2 the rest, "
        "like torch.split(x, [split_size, n - split_size], dim). Each output feeds its own "
        "branch; merge the branches again with Concatenate, Add, Multiply or Average.",
        {
            {"split_size", "Entries in Output 1 (at least 1, less than the dimension's size)"},
            {"dim", "Batched dimension to split: 1 = features, -1 = last; 0 (the batch) is refused"}
        },
        {
            "An unused output passes a zero gradient back",
            "Inverse of Concatenate on the same dim",
            "In a CNN, split after Flatten"
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

    docs_[NodeType::SoftDiceLoss] = {
        "Soft Dice Loss",
        "Dice overlap loss for segmentation-style probability masks. "
        "Predictions and targets must be same-shaped Float32 tensors.",
        "Computes 1 - Dice coefficient with a smoothing constant.",
        {
            {"smooth", "Smoothing constant to avoid division by zero"}
        },
        {
            "Use with probability masks, not raw logits",
            "Targets must match prediction shape"
        },
        "Loss Functions"
    };

    docs_[NodeType::TverskyLoss] = {
        "Tversky Loss",
        "Tversky overlap loss for imbalanced segmentation-style probability masks. "
        "Predictions and targets must be same-shaped Float32 tensors.",
        "Computes 1 - (TP + smooth) / (TP + alpha*FP + beta*FN + smooth).",
        {
            {"alpha", "False-positive penalty"},
            {"beta", "False-negative penalty"},
            {"smooth", "Smoothing constant to avoid division by zero"}
        },
        {
            "Use with probability masks, not raw logits",
            "Targets must match prediction shape",
            "Increase beta when false negatives are more costly"
        },
        "Loss Functions"
    };

    docs_[NodeType::JaccardLoss] = {
        "Jaccard / IoU Loss",
        "Intersection-over-union loss for segmentation-style probability masks. "
        "Predictions and targets must be same-shaped Float32 tensors.",
        "Computes 1 - (intersection + smooth) / (union + smooth).",
        {
            {"smooth", "Smoothing constant to avoid division by zero"}
        },
        {
            "Use with probability masks, not raw logits",
            "Targets must match prediction shape",
            "Equivalent to optimizing soft IoU overlap"
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
            {"learning_rate", "Positive update step size (0.01 default)"},
            {"momentum", "Momentum coefficient in [0, 1)"}
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
            {"learning_rate", "Positive update step size (0.001 default)"},
            {"beta1", "First-moment decay coefficient"},
            {"beta2", "Second-moment decay coefficient"},
            {"epsilon", "Positive numerical-stability constant"}
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
            {"learning_rate", "Positive update step size"},
            {"beta1", "First-moment decay coefficient"},
            {"beta2", "Second-moment decay coefficient"},
            {"epsilon", "Positive numerical-stability constant"},
            {"weight_decay", "Weight decay coefficient (0.01 typical)"}
        },
        {
            "Preferred over Adam when using weight decay",
            "Common in Transformer training stacks"
        },
        "Optimizers"
    };

    docs_[NodeType::RMSprop] = {
        "RMSprop Optimizer",
        "Adaptive learning rate based on recent gradient magnitudes. "
        "Good for non-stationary objectives and RNNs.",
        "Divides learning rate by running average of recent gradient magnitudes.",
        {
            {"learning_rate", "Positive update step size (0.001 default)"},
            {"alpha", "Smoothing constant (0.99)"},
            {"epsilon", "Positive numerical-stability constant"},
            {"momentum", "Momentum coefficient in [0, 1)"}
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
            {"learning_rate", "Positive initial update step size"},
            {"epsilon", "Positive numerical-stability constant"}
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
            {"learning_rate", "Positive update step size"},
            {"beta1", "First-moment decay coefficient"},
            {"beta2", "Second-moment decay coefficient"},
            {"epsilon", "Positive numerical-stability constant"}
        },
        {
            "Try if Adam is working well",
            "May converge faster in some cases"
        },
        "Optimizers"
    };

    // ===== LR Schedulers =====
    docs_[NodeType::StepLR] = {
        "Step LR",
        "Multiplies the learning rate by gamma every step_size epochs. Connect the optimizer's "
        "State output to its Optimizer input.",
        "lr = learning_rate * gamma^floor(epoch / step_size), set after each completed epoch "
        "(torch.optim.lr_scheduler.StepLR).",
        {
            {"step_size", "Epochs between decays"},
            {"gamma", "Factor applied at each decay (0.1 is common)"}
        },
        {
            "Common: decay by 0.1 every 30 epochs",
            "One scheduler per optimizer; not together with the optimizer's lr_schedule",
            "Resuming a run continues the schedule from the checkpoint"
        },
        "LR Schedulers"
    };

    docs_[NodeType::CosineAnnealing] = {
        "Cosine LR",
        "Anneals the learning rate along a cosine from learning_rate down to eta_min over T_max epochs.",
        "lr = eta_min + (learning_rate - eta_min) * (1 + cos(pi * epoch / T_max)) / 2, set after each "
        "completed epoch (torch.optim.lr_scheduler.CosineAnnealingLR).",
        {
            {"T_max", "Epochs from learning_rate down to eta_min (usually the epoch count)"},
            {"eta_min", "Lowest learning rate"}
        },
        {
            "Set T_max to the run's epochs to end at eta_min",
            "Past T_max the cosine rises again, as in PyTorch"
        },
        "LR Schedulers"
    };

    docs_[NodeType::ReduceOnPlateau] = {
        "Reduce LR",
        "Cuts the learning rate by factor when the validation loss has not improved for patience "
        "validated epochs. Needs validation data (Data Split val_ratio above 0, or a Dev dataset).",
        "After each epoch that ran validation: if the loss is not below best - threshold for more than "
        "patience epochs, lr = max(lr * factor, min_lr) (torch ReduceLROnPlateau, mode='min', "
        "threshold_mode='abs', cooldown=0).",
        {
            {"factor", "Factor applied when the loss plateaus"},
            {"patience", "Validated epochs without improvement before a cut"},
            {"threshold", "Smallest loss decrease that counts as improvement (absolute)"},
            {"min_lr", "Learning rate floor"}
        },
        {
            "With validation_freq above 1 it steps only on validated epochs",
            "Good when the right schedule is unknown"
        },
        "LR Schedulers"
    };

    docs_[NodeType::ExponentialLR] = {
        "Exponential LR",
        "Multiplies the learning rate by gamma after every epoch.",
        "lr = learning_rate * gamma^epoch, set after each completed epoch "
        "(torch.optim.lr_scheduler.ExponentialLR).",
        {
            {"gamma", "Factor applied after each epoch (e.g. 0.95)"}
        },
        {
            "Decays fast: keep gamma close to 1"
        },
        "LR Schedulers"
    };

    docs_[NodeType::WarmupScheduler] = {
        "Warmup LR",
        "Ramps the learning rate up linearly from start_factor * learning_rate to learning_rate over "
        "warmup_epochs, then holds it.",
        "lr = learning_rate * (start_factor + (1 - start_factor) * min(epoch / warmup_epochs, 1)), set "
        "after each completed epoch (torch LinearLR, end_factor=1.0, total_iters=warmup_epochs).",
        {
            {"warmup_epochs", "Epochs to reach learning_rate"},
            {"start_factor", "First epoch's rate as a fraction of learning_rate (above 0, at most 1)"}
        },
        {
            "Per-update warmup with a decay after it: use the optimizer's lr_schedule instead"
        },
        "LR Schedulers"
    };

    // ===== Regularization Nodes =====
    docs_[NodeType::L1Regularization] = {
        "L1 Regularization (Lasso)",
        "Adds lambda x the sum of absolute parameter values to the training loss. Pushes weights "
        "towards exactly zero. Wire it between the loss and the optimizer: Loss -> L1 -> Optimizer.",
        "loss = loss + lambda * sum(|w|) over every trainable parameter (weights and biases); the "
        "gradient lambda * sign(w) joins every optimizer step before gradient clipping, as PyTorch's "
        "loss + lam * sum(p.abs().sum() for p in model.parameters()).",
        {
            {"lambda", "Penalty strength (0 or above)"}
        },
        {
            "One regularization node per loss; Elastic Net combines L1 and L2",
            "The reported training loss is the data loss without the penalty"
        },
        "Regularization"
    };

    docs_[NodeType::L2Regularization] = {
        "L2 Regularization (Ridge)",
        "Adds lambda x the sum of squared parameter values to the training loss. Keeps weights small. "
        "Wire it between the loss and the optimizer: Loss -> L2 -> Optimizer.",
        "loss = loss + lambda * sum(w^2) over every trainable parameter (weights and biases); the "
        "gradient 2 * lambda * w joins every optimizer step before gradient clipping.",
        {
            {"lambda", "Penalty strength (0 or above)"}
        },
        {
            "With SGD this equals weight_decay = 2 * lambda; with Adam it is not the same, and AdamW's "
            "weight_decay is decoupled from the loss",
            "The reported training loss is the data loss without the penalty"
        },
        "Regularization"
    };

    docs_[NodeType::ElasticNet] = {
        "Elastic Net Regularization",
        "Adds a mix of the L1 and L2 penalties to the training loss. Wire it between the loss and the "
        "optimizer: Loss -> Elastic Net -> Optimizer.",
        "loss = loss + lambda * (l1_ratio * sum(|w|) + (1 - l1_ratio) * sum(w^2)) over every trainable "
        "parameter. l1_ratio 1 is L1 Regularization, 0 is L2 Regularization.",
        {
            {"lambda", "Penalty strength (0 or above)"},
            {"l1_ratio", "Share of the L1 term, 0 to 1"}
        },
        {
            "l1_ratio 0.5 weighs both terms equally",
            "The reported training loss is the data loss without the penalty"
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
        "Creates runtime batchers from the resolved Dataset partitions. "
        "Train updates weights; Validation/Test batchers are owned by runtime evaluation.",
        "Consumes one Partitions Dataset contract and emits model-facing batched Data/Labels.",
        {
            {"batch_size", "Samples per batch"},
            {"shuffle", "Randomize training order each epoch"},
            {"drop_last", "Drop incomplete final training batch"}
        },
        {
            "Training shuffle/balancing applies to Train only",
            "Validation and Test are resolved from the partition manifest, not separate canvas branches"
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
        "Resolves train, validation, and test partition policy for a dataset. "
        "Runtime batchers consume the compiler-resolved partitions.",
        "New nodes expose Dataset role inputs and one partition output. Older "
        "saved graphs may still show Train/Val/Test tensor pins as legacy "
        "compatibility pins, not separate model-branch execution paths.",
        {
            {"train_ratio", "Fraction for training when a role is derived (e.g., 0.8)"},
            {"val_ratio", "Fraction for validation when derived from Train (e.g., 0.1)"},
            {"test_ratio", "Fraction for test when derived from Train (e.g., 0.1)"},
            {"stratified", "Preserve class distribution when supported"},
            {"seed", "Deterministic split seed"}
        },
        {
            "External Dev/Test sources are preserved by role resolution",
            "Do not build separate Val/Test model branches from legacy pins"
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
