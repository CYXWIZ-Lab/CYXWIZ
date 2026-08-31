#include "node_editor.h"
#include "properties.h"
#include "../core/node_metadata_registry.h"
#include "../plugin/registries/plugin_node_registry.h"
#include "../core/data_registry.h"
#include <imgui.h>
#include <imnodes.h>
#include <spdlog/spdlog.h>
#include <algorithm>

namespace gui {

namespace {

void PopulateStaticNodeContractFromMetadata(MLNode& node, int& next_pin_id) {
    auto& registry = cyxwiz::NodeMetadataRegistry::Instance();
    registry.Initialize();

    const auto* metadata = registry.GetMetadata(node.type);
    if (metadata == nullptr) {
        spdlog::error("No node metadata registered for type {}",
                      static_cast<int>(node.type));
        return;
    }

    cyxwiz::ApplyStaticNodeMetadataContract(*metadata, node, next_pin_id);
}

} // namespace

// Unified Canvas Phase 1: Map NodeType to NodeCategory for UI organization
NodeCategory NodeEditor::GetCategoryForNodeType(NodeType type) {
    switch (type) {
        // Smart I/O Nodes (Universal)
        case NodeType::DataInput:
        case NodeType::DataOutput:
        case NodeType::DataConvert:
        // Legacy Data Sources
        case NodeType::CSVFile:
        case NodeType::SQLQuery:
        case NodeType::HDF5Dataset:
        case NodeType::ParquetFile:
        case NodeType::JSONFile:
        case NodeType::ExcelFile:
        case NodeType::RESTAPISource:
        // Phase 4: Dataset Source Nodes (UI Consolidation)
        case NodeType::ImageFolderDataset:
        case NodeType::MNISTDataset:
        case NodeType::CIFAR10Dataset:
        case NodeType::HuggingFaceDataset:
        case NodeType::KaggleDataset:
            return NodeCategory::DataSources;

        // Data Transforms
        case NodeType::FilterRows:
        case NodeType::SelectColumns:
        case NodeType::JoinTables:
        case NodeType::GroupByAggregate:
        case NodeType::SortRows:
        case NodeType::FillMissingValues:
        case NodeType::RemoveDuplicateRows:
        case NodeType::PivotTable:
        case NodeType::UnionTables:
        case NodeType::RenameColumns:
            return NodeCategory::DataTransform;

        // Analytics
        case NodeType::DescribeStats:
        case NodeType::VisualizeData:
        case NodeType::SampleRows:
        case NodeType::CorrelationMatrix:
        case NodeType::ValueCounts:
        case NodeType::CrossTabulation:
            return NodeCategory::Analytics;

        // Data Export
        case NodeType::ExportCSV:
        case NodeType::ExportParquet:
        case NodeType::ExportSQL:
        case NodeType::ExportJSON:
            return NodeCategory::DataSources;

        // Preprocessing
        case NodeType::Normalize:
        case NodeType::OneHotEncode:
        case NodeType::OutlierDetector:
        case NodeType::ImagePreprocessor:
        case NodeType::QualityAnalyzer:
        case NodeType::DataValidator:
            return NodeCategory::Preprocessing;

        // Dense/Linear layers
        case NodeType::Dense:
            return NodeCategory::Layers;

        // Convolutional layers
        case NodeType::Conv1D:
        case NodeType::Conv2D:
        case NodeType::Conv3D:
        case NodeType::DepthwiseConv2D:
            return NodeCategory::Layers;

        // Pooling layers
        case NodeType::MaxPool2D:
        case NodeType::AvgPool2D:
        case NodeType::GlobalMaxPool:
        case NodeType::GlobalAvgPool:
        case NodeType::AdaptiveAvgPool:
            return NodeCategory::Pooling;

        // Normalization layers
        case NodeType::BatchNorm:
        case NodeType::LayerNorm:
        case NodeType::GroupNorm:
        case NodeType::InstanceNorm:
            return NodeCategory::Normalization;

        // Regularization
        case NodeType::Dropout:
        case NodeType::L1Regularization:
        case NodeType::L2Regularization:
        case NodeType::ElasticNet:
            return NodeCategory::Regularization;

        // Activation functions
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
            return NodeCategory::Activation;

        // Recurrent layers
        case NodeType::RNN:
        case NodeType::LSTM:
        case NodeType::GRU:
        case NodeType::Bidirectional:
        case NodeType::TimeDistributed:
        case NodeType::Embedding:
            return NodeCategory::Recurrent;

        // Attention & Transformer
        case NodeType::MultiHeadAttention:
        case NodeType::SelfAttention:
        case NodeType::CrossAttention:
        case NodeType::LinearAttention:
        case NodeType::TransformerEncoder:
        case NodeType::TransformerDecoder:
        case NodeType::PositionalEncoding:
            return NodeCategory::Attention;

        // Shape operations
        case NodeType::Reshape:
        case NodeType::Permute:
        case NodeType::Squeeze:
        case NodeType::Unsqueeze:
        case NodeType::View:
        case NodeType::Split:
        case NodeType::Flatten:
            return NodeCategory::ShapeOps;

        // Merge operations
        case NodeType::Concatenate:
        case NodeType::Add:
        case NodeType::Multiply:
        case NodeType::Average:
            return NodeCategory::MergeOps;

        // Tensor reductions
        case NodeType::TensorSum:
        case NodeType::TensorMean:
        case NodeType::TensorMax:
        case NodeType::TensorMin:
        case NodeType::TensorProd:
        case NodeType::TensorVar:
        case NodeType::TensorStd:
            return NodeCategory::Analytics;

        case NodeType::TensorBroadcastTo:
        case NodeType::TensorExpand:
        case NodeType::TensorIndexSelect:
            return NodeCategory::ShapeOps;

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
            return NodeCategory::Analytics;

        // Training
        case NodeType::SGD:
        case NodeType::Adam:
        case NodeType::AdamW:
        case NodeType::RMSprop:
        case NodeType::Adagrad:
        case NodeType::NAdam:
        case NodeType::StepLR:
        case NodeType::CosineAnnealing:
        case NodeType::ReduceOnPlateau:
        case NodeType::ExponentialLR:
        case NodeType::WarmupScheduler:
        case NodeType::MSELoss:
        case NodeType::CrossEntropyLoss:
        case NodeType::BCELoss:
        case NodeType::BCEWithLogits:
        case NodeType::L1Loss:
        case NodeType::SmoothL1Loss:
        case NodeType::HuberLoss:
        case NodeType::NLLLoss:
        case NodeType::FocalLoss:
        case NodeType::SoftDiceLoss:
        case NodeType::TverskyLoss:
        case NodeType::JaccardLoss:
        case NodeType::Output:
            return NodeCategory::Training;

        // Utility
        case NodeType::Lambda:
        case NodeType::Identity:
        case NodeType::Constant:
        case NodeType::Parameter:
            return NodeCategory::Utility;

        // Signal/Control
        case NodeType::SignalSlider:
        case NodeType::SineWave:
        case NodeType::StepSignal:
        case NodeType::RampSignal:
        case NodeType::SignalScope:
            return NodeCategory::Signal;

        // Data Pipeline
        case NodeType::DatasetInput:
        case NodeType::DataLoader:
        case NodeType::Augmentation:
        case NodeType::DataSplit:
        case NodeType::TensorReshape:
        // Image Transform Nodes (Phase 1)
        case NodeType::Resize:
        case NodeType::CenterCrop:
        case NodeType::RandomCrop:
        case NodeType::HorizontalFlip:
        case NodeType::VerticalFlip:
        case NodeType::ImageRotate:
        case NodeType::ColorJitter:
        case NodeType::ImageGaussianBlur:
        case NodeType::Grayscale:
        // Phase 6: Advanced Augmentation Nodes (UI Consolidation)
        case NodeType::AugmentationPreset:
        case NodeType::GeometricTransform:
        case NodeType::ColorTransform:
        case NodeType::MorphologyTransform:
        case NodeType::AdvancedAugment:
            return NodeCategory::DataPipeline;

        // DNN Inference
        case NodeType::DNNModelLoad:
        case NodeType::DNNDetect:
        case NodeType::DNNClassify:
        case NodeType::DNNPoseEstimate:
        case NodeType::DNNFaceDetect:
        case NodeType::DNNPreprocess:
        case NodeType::PretrainedYOLO:
        case NodeType::PretrainedMobileNet:
        case NodeType::PretrainedOpenPose:
        case NodeType::PretrainedFaceNet:
        case NodeType::NonMaxSuppression:
        case NodeType::ArgMax:
        case NodeType::TopK:
        case NodeType::ThresholdFilter:
            return NodeCategory::DNN;

        // Text Processing
        case NodeType::TextTokenizer:
        case NodeType::TextVocabulary:
        case NodeType::TextPadding:
        case NodeType::NERSequenceBuilder:
        case NodeType::TokenVocabulary:
        case NodeType::POSVocabulary:
        case NodeType::NERTagVocabulary:
        case NodeType::SequenceTagOutput:
            return NodeCategory::TextProcessing;

        // Upsampling
        case NodeType::ConvTranspose2D:
        case NodeType::Upsample:
        case NodeType::PixelShuffle:
            return NodeCategory::Upsampling;

        // Time Series
        case NodeType::TimeSeriesSegment:
        case NodeType::TimeSeriesWindow:
        case NodeType::TimeSeriesFeatures:
        case NodeType::TimeSeriesSplit:
        case NodeType::SeasonalNaive:
        case NodeType::LogTransform:
        case NodeType::Differencing:
        case NodeType::TimeSeriesDecomposition:
        case NodeType::ACFNode:
        case NodeType::PACFNode:
        case NodeType::StationarityTest:
        case NodeType::SeasonalityDetector:
        case NodeType::ARIMAForecaster:
        case NodeType::ExponentialSmoothing:
            return NodeCategory::TimeSeries;

        // Audio
        case NodeType::AudioInput:
        case NodeType::Spectrogram:
        case NodeType::MelSpectrogram:
        case NodeType::MFCC:
        case NodeType::AudioAugmentation:
            return NodeCategory::Audio;

        // RL
        case NodeType::GymEnvironment:
        case NodeType::ReplayBufferNode:
        case NodeType::PolicyNetwork:
        case NodeType::ValueNetwork:
        case NodeType::RLTraining:
            return NodeCategory::RL;

        // Plugin
        case NodeType::PluginCustom:
            return NodeCategory::Plugin;

        default:
            return NodeCategory::Unknown;
    }
}

#ifndef CYXWIZ_NODE_FACTORY_ONLY
void NodeEditor::AddNode(NodeType type, const std::string& name) {
    if (!CanAddNodeToGraph(type)) {
        spdlog::warn("Blocked graph add for unsupported node '{}' (type={})",
                     name, static_cast<int>(type));
        return;
    }

    // Queue the node for deferred addition (after ImNodes::EndNodeEditor())
    pending_nodes_.push_back({type, name, context_menu_pos_});
    ClearValidationState();  // Graph changed — stale compile results
    spdlog::info("Queued node for addition: type={}, name={} at position x={} y={}",
                 static_cast<int>(type), name, context_menu_pos_.x, context_menu_pos_.y);
}

bool NodeEditor::CanAddNodeToGraph(NodeType type) const {
    auto& registry = cyxwiz::NodeMetadataRegistry::Instance();
    registry.Initialize();
    const auto* metadata = registry.GetMetadata(type);

    // Plugin nodes may be registered dynamically outside the built-in
    // metadata catalog. Preserve that extension path when no static contract
    // exists; catalogued nodes must obey central support truth.
    return metadata == nullptr || cyxwiz::CanAddNodeToGraph(*metadata);
}
#endif

MLNode NodeEditor::CreateNodeWithIds(NodeType type,
                                     const std::string& name,
                                     int& next_node_id,
                                     int& next_pin_id) {
    // Preserve the established switch body while making cursor ownership
    // explicit for deterministic, GUI-free contract validation.
    int& next_node_id_ = next_node_id;
    int& next_pin_id_ = next_pin_id;

    MLNode node;
    node.id = next_node_id_++;
    node.type = type;
    node.category = GetCategoryForNodeType(type);  // Unified Canvas Phase 1
    node.name = name;

    // Create pins based on node type
    switch (type) {
        case NodeType::Dense: {
            PopulateStaticNodeContractFromMetadata(node, next_pin_id_);

            // Toolbar presets such as "Dense (128)" intentionally override
            // the metadata default; ordinary Dense creation stays schema-led.
            if (IsGeneratedDenseName(name)) {
                const size_t start = name.find('(') + 1;
                node.parameters["units"] =
                    name.substr(start, name.size() - start - 1);
            }
            break;
        }

        case NodeType::ReLU:
        case NodeType::Sigmoid:
        case NodeType::Tanh:
        case NodeType::Softmax:
        case NodeType::LeakyReLU:
        case NodeType::ELU:
        case NodeType::GELU:
        case NodeType::Swish:
        case NodeType::Mish: {
            PopulateStaticNodeContractFromMetadata(node, next_pin_id_);
            break;
        }

        case NodeType::PReLU:
        case NodeType::SELU: {
            // Catalog-preview activations remain constructible only for
            // compatibility with saved graphs.
            NodePin input_pin;
            input_pin.id = next_pin_id_++;
            input_pin.type = PinType::Tensor;
            input_pin.name = "Input";
            input_pin.is_input = true;
            node.inputs.push_back(input_pin);

            NodePin output_pin;
            output_pin.id = next_pin_id_++;
            output_pin.type = PinType::Tensor;
            output_pin.name = "Output";
            output_pin.is_input = false;
            node.outputs.push_back(output_pin);

            if (node.type == NodeType::PReLU) {
                node.parameters["num_parameters"] = "1";
                node.parameters["init"] = "0.25";
            }
            break;
        }

        case NodeType::Output: {
            PopulateStaticNodeContractFromMetadata(node, next_pin_id_);
            break;
        }

        case NodeType::Conv2D:
        case NodeType::Conv1D:
        case NodeType::Conv3D:
        case NodeType::DepthwiseConv2D: {
            PopulateStaticNodeContractFromMetadata(node, next_pin_id_);
            break;
        }

        case NodeType::MaxPool2D:
        case NodeType::AvgPool2D: {
            PopulateStaticNodeContractFromMetadata(node, next_pin_id_);
            break;
        }

        case NodeType::GlobalMaxPool:
        case NodeType::GlobalAvgPool:
        case NodeType::AdaptiveAvgPool: {
            PopulateStaticNodeContractFromMetadata(node, next_pin_id_);
            break;
        }

        case NodeType::Flatten: {
            PopulateStaticNodeContractFromMetadata(node, next_pin_id_);
            break;
        }

        case NodeType::Dropout: {
            PopulateStaticNodeContractFromMetadata(node, next_pin_id_);
            break;
        }

        case NodeType::BatchNorm:
        case NodeType::LayerNorm:
        case NodeType::GroupNorm:
        case NodeType::InstanceNorm: {
            PopulateStaticNodeContractFromMetadata(node, next_pin_id_);
            break;
        }

        // ========== Data Pipeline Nodes ==========

        case NodeType::DatasetInput: {
            // DatasetInput node - loads from DataRegistry
            // No input pins (this is a source node)

            // Output: Data tensor
            NodePin data_pin;
            data_pin.id = next_pin_id_++;
            data_pin.type = PinType::Tensor;
            data_pin.name = "Data";
            data_pin.is_input = false;
            node.outputs.push_back(data_pin);

            // Output: Labels tensor
            NodePin labels_pin;
            labels_pin.id = next_pin_id_++;
            labels_pin.type = PinType::Labels;
            labels_pin.name = "Labels";
            labels_pin.is_input = false;
            node.outputs.push_back(labels_pin);

            // Note: Shape is metadata (displayed in properties panel), not a data flow output.
            // In ML frameworks, shape is intrinsic to tensors (accessed via tensor.shape).

            // Parameters
            node.parameters["dataset_name"] = "";  // Name in DataRegistry
            node.parameters["split"] = "train";    // train, val, test
            break;
        }

        case NodeType::DataLoader: {
            PopulateStaticNodeContractFromMetadata(node, next_pin_id_);
            break;
        }

        case NodeType::Augmentation: {
            // Augmentation node - transform pipeline
            // Input: Data tensor
            NodePin input_pin;
            input_pin.id = next_pin_id_++;
            input_pin.type = PinType::Tensor;
            input_pin.name = "Input";
            input_pin.is_input = true;
            input_pin.description =
                "Image (or tabular) tensor stream to apply augmentations "
                "to. Augmentation runs train-only — eval/test passes are "
                "pass-through, so the model sees the unperturbed data "
                "for fair metrics.";
            node.inputs.push_back(input_pin);

            // Output: Augmented data
            NodePin output_pin;
            output_pin.id = next_pin_id_++;
            output_pin.type = PinType::Tensor;
            output_pin.name = "Output";
            output_pin.is_input = false;
            output_pin.description =
                "Same shape as Input. The transforms parameter is a "
                "csv pipeline (RandomFlip,Normalize,...) applied left "
                "to right.";
            node.outputs.push_back(output_pin);

            // Parameters (transform pipeline)
            node.parameters["transforms"] = "RandomFlip,Normalize";
            node.parameters["flip_prob"] = "0.5";
            node.parameters["normalize_mean"] = "0.0";
            node.parameters["normalize_std"] = "1.0";
            break;
        }

        // ===== Image Transform Nodes (Phase 1) =====
        // All share the same pin layout: one Tensor in, one Tensor out.
        // Parameters vary per transform type. Random* / *Flip / Rotate /
        // Jitter / Blur are train-only (eval/test pass-through); Resize /
        // CenterCrop / Grayscale always run.
        case NodeType::Resize: {
            NodePin in; in.id = next_pin_id_++; in.type = PinType::Tensor;
            in.name = "Input"; in.is_input = true;
            in.description = "Image tensor [batch, channels, H, W] (any size).";
            node.inputs.push_back(in);
            NodePin out; out.id = next_pin_id_++; out.type = PinType::Tensor;
            out.name = "Output"; out.is_input = false;
            out.description = "Resized to [batch, channels, height, width]. "
                              "Mode 'exact' stretches; other modes preserve aspect ratio.";
            node.outputs.push_back(out);
            node.parameters["width"] = "224";
            node.parameters["height"] = "224";
            node.parameters["mode"] = "exact";
            break;
        }
        case NodeType::CenterCrop: {
            NodePin in; in.id = next_pin_id_++; in.type = PinType::Tensor;
            in.name = "Input"; in.is_input = true;
            in.description = "Image tensor [batch, channels, H, W]. Must be at "
                             "least width × height in spatial size.";
            node.inputs.push_back(in);
            NodePin out; out.id = next_pin_id_++; out.type = PinType::Tensor;
            out.name = "Output"; out.is_input = false;
            out.description = "Center crop of shape [batch, channels, height, width].";
            node.outputs.push_back(out);
            node.parameters["width"] = "224";
            node.parameters["height"] = "224";
            break;
        }
        case NodeType::RandomCrop: {
            NodePin in; in.id = next_pin_id_++; in.type = PinType::Tensor;
            in.name = "Input"; in.is_input = true;
            in.description = "Image tensor [batch, channels, H, W]. Train-only — "
                             "eval/test pass through unchanged.";
            node.inputs.push_back(in);
            NodePin out; out.id = next_pin_id_++; out.type = PinType::Tensor;
            out.name = "Output"; out.is_input = false;
            out.description = "Random crop of shape [batch, channels, height, width]. "
                              "Position is resampled per batch.";
            node.outputs.push_back(out);
            node.parameters["width"] = "224";
            node.parameters["height"] = "224";
            break;
        }
        case NodeType::HorizontalFlip: {
            NodePin in; in.id = next_pin_id_++; in.type = PinType::Tensor;
            in.name = "Input"; in.is_input = true;
            in.description = "Image tensor [batch, channels, H, W]. Train-only.";
            node.inputs.push_back(in);
            NodePin out; out.id = next_pin_id_++; out.type = PinType::Tensor;
            out.name = "Output"; out.is_input = false;
            out.description = "Each sample independently flipped left↔right with "
                              "the configured probability.";
            node.outputs.push_back(out);
            node.parameters["probability"] = "0.5";
            break;
        }
        case NodeType::VerticalFlip: {
            NodePin in; in.id = next_pin_id_++; in.type = PinType::Tensor;
            in.name = "Input"; in.is_input = true;
            in.description = "Image tensor [batch, channels, H, W]. Train-only.";
            node.inputs.push_back(in);
            NodePin out; out.id = next_pin_id_++; out.type = PinType::Tensor;
            out.name = "Output"; out.is_input = false;
            out.description = "Each sample independently flipped top↔bottom with "
                              "the configured probability. Avoid for natural "
                              "imagery where 'up' is meaningful.";
            node.outputs.push_back(out);
            node.parameters["probability"] = "0.5";
            break;
        }
        case NodeType::ImageRotate: {
            NodePin in; in.id = next_pin_id_++; in.type = PinType::Tensor;
            in.name = "Input"; in.is_input = true;
            in.description = "Image tensor [batch, channels, H, W]. Train-only.";
            node.inputs.push_back(in);
            NodePin out; out.id = next_pin_id_++; out.type = PinType::Tensor;
            out.name = "Output"; out.is_input = false;
            out.description = "Each sample rotated by an angle uniformly sampled "
                              "in [-max_angle, +max_angle] degrees.";
            node.outputs.push_back(out);
            node.parameters["max_angle"] = "15.0";
            node.parameters["probability"] = "0.5";
            break;
        }
        case NodeType::ColorJitter: {
            NodePin in; in.id = next_pin_id_++; in.type = PinType::Tensor;
            in.name = "Input"; in.is_input = true;
            in.description = "Color image tensor [batch, 3, H, W]. Train-only.";
            node.inputs.push_back(in);
            NodePin out; out.id = next_pin_id_++; out.type = PinType::Tensor;
            out.name = "Output"; out.is_input = false;
            out.description = "Per-sample random adjustments to brightness, "
                              "contrast, saturation, and hue within the "
                              "configured ranges.";
            node.outputs.push_back(out);
            node.parameters["brightness"] = "0.2";
            node.parameters["contrast"] = "0.2";
            node.parameters["saturation"] = "0.2";
            node.parameters["hue"] = "0.1";
            break;
        }
        case NodeType::ImageGaussianBlur: {
            NodePin in; in.id = next_pin_id_++; in.type = PinType::Tensor;
            in.name = "Input"; in.is_input = true;
            in.description = "Image tensor [batch, channels, H, W]. Train-only.";
            node.inputs.push_back(in);
            NodePin out; out.id = next_pin_id_++; out.type = PinType::Tensor;
            out.name = "Output"; out.is_input = false;
            out.description = "Gaussian-blurred image with the configured "
                              "kernel_size and sigma. Same shape as Input.";
            node.outputs.push_back(out);
            node.parameters["kernel_size"] = "5";
            node.parameters["sigma"] = "1.0";
            break;
        }
        case NodeType::Grayscale: {
            NodePin in; in.id = next_pin_id_++; in.type = PinType::Tensor;
            in.name = "Input"; in.is_input = true;
            in.description = "Color image tensor [batch, 3, H, W].";
            node.inputs.push_back(in);
            NodePin out; out.id = next_pin_id_++; out.type = PinType::Tensor;
            out.name = "Output"; out.is_input = false;
            out.description = "Single-channel grayscale [batch, 1, H, W] using "
                              "the standard luminosity weighting.";
            node.outputs.push_back(out);
            break;
        }

        case NodeType::DataSplit: {
            PopulateStaticNodeContractFromMetadata(node, next_pin_id_);
            break;
        }

        case NodeType::TensorReshape: {
            // TensorReshape node
            NodePin input_pin;
            input_pin.id = next_pin_id_++;
            input_pin.type = PinType::Tensor;
            input_pin.name = "Input";
            input_pin.is_input = true;
            input_pin.description =
                "Any tensor — total element count must match the target "
                "shape (one -1 entry is auto-computed).";
            node.inputs.push_back(input_pin);

            NodePin output_pin;
            output_pin.id = next_pin_id_++;
            output_pin.type = PinType::Tensor;
            output_pin.name = "Output";
            output_pin.is_input = false;
            output_pin.description =
                "Reshaped to the comma-separated `shape` parameter — "
                "e.g. '-1,28,28,1' to recover MNIST-shaped images from "
                "a flattened tensor.";
            node.outputs.push_back(output_pin);

            node.parameters["shape"] = "-1,28,28,1";
            break;
        }

        case NodeType::Normalize: {
            PopulateStaticNodeContractFromMetadata(node, next_pin_id_);
            break;
        }

        case NodeType::OneHotEncode: {
            PopulateStaticNodeContractFromMetadata(node, next_pin_id_);
            break;
        }

        // ========== Loss Functions ==========

        case NodeType::MSELoss:
        case NodeType::CrossEntropyLoss:
        case NodeType::FocalLoss:
        case NodeType::BCELoss:
        case NodeType::BCEWithLogits:
        case NodeType::L1Loss:
        case NodeType::SmoothL1Loss:
        case NodeType::HuberLoss:
        case NodeType::NLLLoss:
        case NodeType::SoftDiceLoss:
        case NodeType::TverskyLoss:
        case NodeType::JaccardLoss: {
            PopulateStaticNodeContractFromMetadata(node, next_pin_id_);
            break;
        }

        // ========== Optimizers ==========

        case NodeType::SGD:
        case NodeType::Adam:
        case NodeType::AdamW:
        case NodeType::RMSprop:
        case NodeType::Adagrad:
        case NodeType::NAdam: {
            PopulateStaticNodeContractFromMetadata(node, next_pin_id_);
            break;
        }

        // ========== Recurrent Layers ==========

        case NodeType::LSTM:
        case NodeType::GRU: {
            PopulateStaticNodeContractFromMetadata(node, next_pin_id_);
            break;
        }

        case NodeType::RNN:
        case NodeType::Bidirectional: {
            PopulateStaticNodeContractFromMetadata(node, next_pin_id_);
            break;
        }

        case NodeType::TimeDistributed: {
            PopulateStaticNodeContractFromMetadata(node, next_pin_id_);
            break;
        }

        case NodeType::Embedding: {
            PopulateStaticNodeContractFromMetadata(node, next_pin_id_);
            break;
        }

        // ========== Attention & Transformer ==========

        case NodeType::MultiHeadAttention: {
            PopulateStaticNodeContractFromMetadata(node, next_pin_id_);
            break;
        }

        case NodeType::SelfAttention:
        case NodeType::CrossAttention:
        case NodeType::LinearAttention: {
            PopulateStaticNodeContractFromMetadata(node, next_pin_id_);
            break;
        }

        case NodeType::TransformerEncoder:
        case NodeType::TransformerDecoder:
        case NodeType::PositionalEncoding: {
            PopulateStaticNodeContractFromMetadata(node, next_pin_id_);
            break;
        }

        // ========== Shape Operations ==========

        case NodeType::Reshape:
        case NodeType::View:
        case NodeType::Permute:
        case NodeType::Squeeze:
        case NodeType::Unsqueeze:
        case NodeType::TensorBroadcastTo:
        case NodeType::TensorExpand:
        case NodeType::TensorIndexSelect: {
            PopulateStaticNodeContractFromMetadata(node, next_pin_id_);
            break;
        }

        case NodeType::Split: {
            NodePin input_pin;
            input_pin.id = next_pin_id_++;
            input_pin.type = PinType::Tensor;
            input_pin.name = "Input";
            input_pin.is_input = true;
            node.inputs.push_back(input_pin);

            // Multiple outputs for split
            NodePin output1;
            output1.id = next_pin_id_++;
            output1.type = PinType::Tensor;
            output1.name = "Output 1";
            output1.is_input = false;
            node.outputs.push_back(output1);

            NodePin output2;
            output2.id = next_pin_id_++;
            output2.type = PinType::Tensor;
            output2.name = "Output 2";
            output2.is_input = false;
            node.outputs.push_back(output2);

            node.parameters["split_size"] = "2";
            node.parameters["dim"] = "0";
            break;
        }

        // ========== Merge Operations ==========

        case NodeType::Concatenate:
        case NodeType::Add:
        case NodeType::Multiply:
        case NodeType::Average: {
            PopulateStaticNodeContractFromMetadata(node, next_pin_id_);
            break;
        }

        // ========== Tensor Reductions ==========

        case NodeType::TensorSum:
        case NodeType::TensorMean:
        case NodeType::TensorMax:
        case NodeType::TensorMin:
        case NodeType::TensorProd:
        case NodeType::TensorVar:
        case NodeType::TensorStd: {
            PopulateStaticNodeContractFromMetadata(node, next_pin_id_);
            break;
        }

        case NodeType::TensorPow:
        case NodeType::TensorSqrt:
        case NodeType::TensorExp:
        case NodeType::TensorLog:
        case NodeType::TensorAbs:
        case NodeType::TensorSign:
        case NodeType::TensorClip: {
            PopulateStaticNodeContractFromMetadata(node, next_pin_id_);
            break;
        }

        case NodeType::TensorDot:
        case NodeType::TensorBatchMatMul:
        case NodeType::TensorCompare:
        case NodeType::TensorLogicalMask: {
            PopulateStaticNodeContractFromMetadata(node, next_pin_id_);
            break;
        }

        // ========== Learning Rate Schedulers ==========

        case NodeType::StepLR:
        case NodeType::CosineAnnealing:
        case NodeType::ReduceOnPlateau:
        case NodeType::ExponentialLR:
        case NodeType::WarmupScheduler: {
            PopulateStaticNodeContractFromMetadata(node, next_pin_id_);
            break;
        }

        // ========== Regularization ==========

        case NodeType::L1Regularization:
        case NodeType::L2Regularization:
        case NodeType::ElasticNet: {
            PopulateStaticNodeContractFromMetadata(node, next_pin_id_);
            break;
        }

        // ========== Utility Nodes ==========

        case NodeType::Lambda:
        case NodeType::Identity:
        case NodeType::Parameter:
        case NodeType::Constant:
        case NodeType::SignalSlider:
        case NodeType::SineWave:
        case NodeType::StepSignal:
        case NodeType::RampSignal:
        case NodeType::SignalScope: {
            PopulateStaticNodeContractFromMetadata(node, next_pin_id_);
            break;
        }

        // ===== DNN Inference Nodes =====
        case NodeType::DNNModelLoad: {
            NodePin model_out;
            model_out.id = next_pin_id_++;
            model_out.type = PinType::Parameters;
            model_out.name = "Model";
            model_out.is_input = false;
            node.outputs.push_back(model_out);
            node.parameters["model_path"] = "";
            node.parameters["config_path"] = "";
            node.parameters["model_type"] = "yolov4";
            node.parameters["backend"] = "cuda";
            break;
        }

        case NodeType::DNNDetect: {
            NodePin image_in;
            image_in.id = next_pin_id_++;
            image_in.type = PinType::Tensor;
            image_in.name = "Image";
            image_in.is_input = true;
            node.inputs.push_back(image_in);
            NodePin model_in;
            model_in.id = next_pin_id_++;
            model_in.type = PinType::Parameters;
            model_in.name = "Model";
            model_in.is_input = true;
            node.inputs.push_back(model_in);
            NodePin boxes_out;
            boxes_out.id = next_pin_id_++;
            boxes_out.type = PinType::Tensor;
            boxes_out.name = "Detections";
            boxes_out.is_input = false;
            node.outputs.push_back(boxes_out);
            node.parameters["confidence"] = "0.5";
            node.parameters["nms_threshold"] = "0.4";
            break;
        }

        case NodeType::DNNClassify:
        case NodeType::DNNPoseEstimate:
        case NodeType::DNNFaceDetect:
        case NodeType::DNNPreprocess: {
            NodePin image_in;
            image_in.id = next_pin_id_++;
            image_in.type = PinType::Tensor;
            image_in.name = "Image";
            image_in.is_input = true;
            node.inputs.push_back(image_in);
            NodePin out;
            out.id = next_pin_id_++;
            out.type = PinType::Tensor;
            out.name = "Output";
            out.is_input = false;
            node.outputs.push_back(out);
            node.parameters["threshold"] = "0.5";
            break;
        }

        case NodeType::PretrainedYOLO:
        case NodeType::PretrainedMobileNet:
        case NodeType::PretrainedOpenPose:
        case NodeType::PretrainedFaceNet: {
            NodePin image_in;
            image_in.id = next_pin_id_++;
            image_in.type = PinType::Tensor;
            image_in.name = "Image";
            image_in.is_input = true;
            node.inputs.push_back(image_in);
            NodePin out;
            out.id = next_pin_id_++;
            out.type = PinType::Tensor;
            out.name = "Output";
            out.is_input = false;
            node.outputs.push_back(out);
            node.parameters["confidence"] = "0.5";
            node.parameters["backend"] = "cuda";
            break;
        }

        case NodeType::NonMaxSuppression:
        case NodeType::ArgMax:
        case NodeType::TopK:
        case NodeType::ThresholdFilter: {
            NodePin in;
            in.id = next_pin_id_++;
            in.type = PinType::Tensor;
            in.name = "Input";
            in.is_input = true;
            node.inputs.push_back(in);
            NodePin out;
            out.id = next_pin_id_++;
            out.type = PinType::Tensor;
            out.name = "Output";
            out.is_input = false;
            node.outputs.push_back(out);
            node.parameters["threshold"] = "0.5";
            break;
        }

        // ========== Text Processing Nodes ==========

        case NodeType::TextTokenizer: {
            PopulateStaticNodeContractFromMetadata(node, next_pin_id_);
            break;
        }

        case NodeType::TextVocabulary: {
            NodePin in;
            in.id = next_pin_id_++;
            in.type = PinType::Tensor;
            in.name = "Text Data";
            in.is_input = true;
            node.inputs.push_back(in);
            NodePin out;
            out.id = next_pin_id_++;
            out.type = PinType::Tensor;
            out.name = "Vocabulary";
            out.is_input = false;
            node.outputs.push_back(out);
            node.parameters["source_csv"] = "";
            node.parameters["text_col"] = "text";
            node.parameters["tokenizer_type"] = "1";
            node.parameters["method"] = "word";
            node.parameters["lowercase"] = "true";
            node.parameters["min_freq"] = "1";
            node.parameters["min_word_freq"] = "1";
            node.parameters["max_vocab_size"] = "-1";
            node.parameters["vocab_file"] = "";
            break;
        }

        case NodeType::TextPadding: {
            NodePin in;
            in.id = next_pin_id_++;
            in.type = PinType::Tensor;
            in.name = "Sequences";
            in.is_input = true;
            node.inputs.push_back(in);
            NodePin out;
            out.id = next_pin_id_++;
            out.type = PinType::Tensor;
            out.name = "Padded";
            out.is_input = false;
            node.outputs.push_back(out);
            node.parameters["max_length"] = "512";
            node.parameters["pad_value"] = "0";
            break;
        }

        case NodeType::NERSequenceBuilder: {
            NodePin data_in;
            data_in.id = next_pin_id_++;
            data_in.type = PinType::Dataset;
            data_in.name = "Rows";
            data_in.is_input = true;
            node.inputs.push_back(data_in);
            NodePin sequences_out;
            sequences_out.id = next_pin_id_++;
            sequences_out.type = PinType::Tensor;
            sequences_out.name = "Sequence Samples";
            sequences_out.is_input = false;
            node.outputs.push_back(sequences_out);
            node.parameters["token_column"] = "tokens";
            node.parameters["pos_column"] = "";
            node.parameters["tag_column"] = "ner_tags";
            node.parameters["sentence_id_column"] = "";
            node.parameters["max_sequence_length"] = "0";
            node.parameters["ignore_index"] = "-100";
            node.parameters["create_attention_mask"] = "true";
            break;
        }

        case NodeType::TokenVocabulary: {
            NodePin in;
            in.id = next_pin_id_++;
            in.type = PinType::Dataset;
            in.name = "Tokens";
            in.is_input = true;
            node.inputs.push_back(in);
            NodePin out;
            out.id = next_pin_id_++;
            out.type = PinType::Parameters;
            out.name = "Token Vocabulary";
            out.is_input = false;
            node.outputs.push_back(out);
            node.parameters["min_freq"] = "1";
            node.parameters["max_vocab_size"] = "0";
            node.parameters["lowercase"] = "true";
            node.parameters["pad_token"] = "[PAD]";
            node.parameters["unk_token"] = "[UNK]";
            break;
        }

        case NodeType::POSVocabulary: {
            NodePin in;
            in.id = next_pin_id_++;
            in.type = PinType::Dataset;
            in.name = "POS Tags";
            in.is_input = true;
            node.inputs.push_back(in);
            NodePin out;
            out.id = next_pin_id_++;
            out.type = PinType::Parameters;
            out.name = "POS Vocabulary";
            out.is_input = false;
            node.outputs.push_back(out);
            node.parameters["min_freq"] = "1";
            node.parameters["max_vocab_size"] = "0";
            node.parameters["lowercase"] = "false";
            node.parameters["pad_token"] = "[PAD]";
            node.parameters["unk_token"] = "[UNK]";
            break;
        }

        case NodeType::NERTagVocabulary: {
            NodePin in;
            in.id = next_pin_id_++;
            in.type = PinType::Dataset;
            in.name = "NER Tags";
            in.is_input = true;
            node.inputs.push_back(in);
            NodePin out;
            out.id = next_pin_id_++;
            out.type = PinType::Parameters;
            out.name = "NER Tag Vocabulary";
            out.is_input = false;
            node.outputs.push_back(out);
            node.parameters["outside_tag"] = "O";
            node.parameters["bio_scheme"] = "BIO";
            break;
        }

        case NodeType::SequenceTagOutput: {
            NodePin in;
            in.id = next_pin_id_++;
            in.type = PinType::Tensor;
            in.name = "Token Logits";
            in.is_input = true;
            node.inputs.push_back(in);
            NodePin out;
            out.id = next_pin_id_++;
            out.type = PinType::Tensor;
            out.name = "Predictions";
            out.is_input = false;
            out.is_required = false;
            node.outputs.push_back(out);
            node.parameters["num_tags"] = "0";
            node.parameters["tag_vocab_file"] = "";
            node.parameters["decode_scheme"] = "BIO";
            break;
        }

        case NodeType::PairDatasetBuilder:
        case NodeType::TripletDatasetBuilder:
        case NodeType::SharedEncoder:
        case NodeType::SiameseBranch:
        case NodeType::ContrastiveLoss:
        case NodeType::CosineEmbeddingLoss:
        case NodeType::TripletLoss:
        case NodeType::PairMetrics:
        case NodeType::RetrievalMetrics:
        case NodeType::EmbeddingOutput:
        case NodeType::PairScoreOutput: {
            PopulateStaticNodeContractFromMetadata(node, next_pin_id_);
            break;
        }

        // ========== Upsampling Layers ==========

        case NodeType::ConvTranspose2D:
        case NodeType::Upsample:
        case NodeType::PixelShuffle: {
            PopulateStaticNodeContractFromMetadata(node, next_pin_id_);
            break;
        }

        // ========== Time-Series Nodes ==========

        case NodeType::TimeSeriesSegment: {
            NodePin in;
            in.id = next_pin_id_++;
            in.type = PinType::Dataset;
            in.name = "Data";
            in.is_input = true;
            in.description =
                "Time-ordered table containing a timestamp column.";
            node.inputs.push_back(in);
            NodePin out;
            out.id = next_pin_id_++;
            out.type = PinType::Dataset;
            out.name = "Segmented";
            out.is_input = false;
            out.description =
                "Original table plus continuous segment and time-delta "
                "metadata. Duplicate, backward, null, or unparseable "
                "timestamps fail closed.";
            node.outputs.push_back(out);
            node.parameters["timestamp_col"] = "";
            node.parameters["gap_threshold_seconds"] = "30";
            node.parameters["segment_col"] = "__segment_id";
            node.parameters["delta_col"] = "__time_delta_seconds";
            break;
        }

        case NodeType::TimeSeriesWindow: {
            NodePin in;
            in.id = next_pin_id_++;
            in.type = PinType::Dataset;
            in.name = "Data";
            in.is_input = true;
            in.description =
                "Time-ordered tabular stream from DataInput or "
                "TimeSeriesFeatures. Must contain value_col plus any "
                "feature_cols you want windowed.";
            node.inputs.push_back(in);
            NodePin out;
            out.id = next_pin_id_++;
            out.type = PinType::Dataset;
            out.name = "Windowed";
            out.is_input = false;
            out.description =
                "Windowed Arrow table with x_* feature columns and a "
                "y label column plus ordered y_1.. targets when "
                "label_width > 1. Hidden target-bound metadata supports "
                "leakage-safe chronological splitting. Optional "
                "__window_start_time metadata is emitted when time_col is set.";
            node.outputs.push_back(out);
            // Phase 4 canonical params read by TimeSeriesWindowOperator.
            // value_col is required and has no reasonable default; the
            // user must fill it in via Properties before training. Leaving
            // it empty here trips the operator's Configure error which
            // the compile gate surfaces before training launch.
            // feature_cols is optional multivariate extension: comma-sep
            // names of extra columns to window alongside value_col.
            // Typically populated by upstream TimeSeriesFeatures lag /
            // rolling columns.
            node.parameters["value_col"] = "";
            node.parameters["feature_cols"] = "";
            // Optional numeric time column (unix epoch / day index).
            // When set, TimeSeriesWindow emits a __window_start_time
            // metadata column for forecast-plotting alignment. __-prefix
            // hides it from feature auto-detect. Empty = off.
            node.parameters["time_col"] = "";
            node.parameters["segment_col"] = "";
            node.parameters["input_width"] = "12";
            node.parameters["label_width"] = "1";
            node.parameters["shift"] = "1";
            break;
        }

        case NodeType::TimeSeriesFeatures: {
            NodePin in;
            in.id = next_pin_id_++;
            in.type = PinType::Dataset;
            in.name = "Data";
            in.is_input = true;
            in.description =
                "Time-ordered tabular stream containing value_col. "
                "Augments the table with lag and rolling-aggregation "
                "columns derived from value_col.";
            node.inputs.push_back(in);
            NodePin out;
            out.id = next_pin_id_++;
            out.type = PinType::Dataset;
            out.name = "Enriched";
            out.is_input = false;
            out.description =
                "Same rows as Data plus lag_<n> and rolling_<agg>_<w> "
                "columns. Feed into TimeSeriesWindow so feature_cols "
                "can pick up the new columns alongside value_col.";
            node.outputs.push_back(out);
            // Phase 4 canonical params read by TimeSeriesFeaturesOperator.
            // value_col is required and has no default; user must fill
            // via Properties before training. Operator's Configure errors
            // if empty; compile gate surfaces it.
            node.parameters["value_col"] = "";
            node.parameters["lag_values"] = "";      // e.g. "1,12"
            node.parameters["rolling_windows"] = ""; // e.g. "7"
            // Rolling aggregations (csv of mean/std/min/max/median).
            // Empty defaults to "mean" so existing graphs don't change
            // their column schema.
            node.parameters["rolling_aggregations"] = "mean";
            break;
        }

        case NodeType::TimeSeriesSplit: {
            NodePin in;
            in.id = next_pin_id_++;
            in.type = PinType::Dataset;
            in.name = "Data";
            in.is_input = true;
            in.description =
                "Time-ordered tabular stream. Unlike DataSplit, rows "
                "are not shuffled; the split is chronological so the "
                "model never sees future data during training.";
            node.inputs.push_back(in);
            NodePin partitioned_out;
            partitioned_out.id = next_pin_id_++;
            partitioned_out.type = PinType::Dataset;
            partitioned_out.name = "Partitioned";
            partitioned_out.is_input = false;
            partitioned_out.description =
                "Input table with an appended __partition__ column: "
                "0=train, 1=validation, 2=test.";
            node.outputs.push_back(partitioned_out);
            // Chronological 80/10/10 split. The operator validates that
            // ratios sum to 1.0 +/- 0.01.
            node.parameters["train_ratio"] = "0.8";
            node.parameters["val_ratio"] = "0.1";
            node.parameters["test_ratio"] = "0.1";
            node.parameters["boundary_policy"] = "targets_within_partition";
            // Optional exact source-row cutoffs. -1 uses ratios against the
            // original source timeline reconstructed from window metadata.
            node.parameters["train_end_source_row"] = "-1";
            node.parameters["val_end_source_row"] = "-1";
            break;
        }

        case NodeType::LogTransform: {
            NodePin in;
            in.id = next_pin_id_++;
            in.type = PinType::Dataset;
            in.name = "Data";
            in.is_input = true;
            node.inputs.push_back(in);
            NodePin out;
            out.id = next_pin_id_++;
            out.type = PinType::Dataset;
            out.name = "Transformed";
            out.is_input = false;
            node.outputs.push_back(out);
            node.parameters["value_col"] = "";
            break;
        }

        case NodeType::Differencing: {
            NodePin in;
            in.id = next_pin_id_++;
            in.type = PinType::Dataset;
            in.name = "Data";
            in.is_input = true;
            node.inputs.push_back(in);
            NodePin out;
            out.id = next_pin_id_++;
            out.type = PinType::Dataset;
            out.name = "Differenced";
            out.is_input = false;
            node.outputs.push_back(out);
            node.parameters["value_col"] = "";
            node.parameters["lag"] = "1";
            node.parameters["order"] = "1";
            break;
        }

        // Phase 5 time-series analysis: real Cat-1 operators that
        // append analysis columns to the input. Forecast horizon is
        // fixed to 0 — these are in-sample fit only, which preserves
        // row count and downstream alignment.
        case NodeType::TimeSeriesDecomposition: {
            NodePin in;
            in.id = next_pin_id_++;
            in.type = PinType::Dataset;
            in.name = "Data";
            in.is_input = true;
            node.inputs.push_back(in);
            NodePin out;
            out.id = next_pin_id_++;
            out.type = PinType::Dataset;
            out.name = "Decomposed";
            out.is_input = false;
            node.outputs.push_back(out);
            // Canonical params read by TimeSeriesDecompositionOperator.
            node.parameters["signal_col"] = "";
            node.parameters["period"] = "12";
            node.parameters["method"] = "additive";     // additive / multiplicative
            node.parameters["algorithm"] = "classical"; // classical / stl
            break;
        }

        case NodeType::ACFNode: {
            NodePin in;
            in.id = next_pin_id_++;
            in.type = PinType::Dataset;
            in.name = "Data";
            in.is_input = true;
            node.inputs.push_back(in);
            NodePin out;
            out.id = next_pin_id_++;
            out.type = PinType::Dataset;
            out.name = "ACF";
            out.is_input = false;
            node.outputs.push_back(out);
            node.parameters["signal_col"] = "";
            node.parameters["max_lag"] = "-1";
            break;
        }

        case NodeType::PACFNode: {
            NodePin in;
            in.id = next_pin_id_++;
            in.type = PinType::Dataset;
            in.name = "Data";
            in.is_input = true;
            node.inputs.push_back(in);
            NodePin out;
            out.id = next_pin_id_++;
            out.type = PinType::Dataset;
            out.name = "PACF";
            out.is_input = false;
            node.outputs.push_back(out);
            node.parameters["signal_col"] = "";
            node.parameters["max_lag"] = "-1";
            break;
        }

        case NodeType::StationarityTest: {
            NodePin in;
            in.id = next_pin_id_++;
            in.type = PinType::Dataset;
            in.name = "Data";
            in.is_input = true;
            node.inputs.push_back(in);
            NodePin out;
            out.id = next_pin_id_++;
            out.type = PinType::Dataset;
            out.name = "Results";
            out.is_input = false;
            node.outputs.push_back(out);
            node.parameters["signal_col"] = "";
            node.parameters["max_lags"] = "-1";
            break;
        }

        case NodeType::SeasonalityDetector: {
            NodePin in;
            in.id = next_pin_id_++;
            in.type = PinType::Dataset;
            in.name = "Data";
            in.is_input = true;
            node.inputs.push_back(in);
            NodePin out;
            out.id = next_pin_id_++;
            out.type = PinType::Dataset;
            out.name = "Periods";
            out.is_input = false;
            node.outputs.push_back(out);
            node.parameters["signal_col"] = "";
            node.parameters["min_period"] = "2";
            node.parameters["max_period"] = "-1";
            break;
        }

        case NodeType::ARIMAForecaster: {
            NodePin in;
            in.id = next_pin_id_++;
            in.type = PinType::Dataset;
            in.name = "Data";
            in.is_input = true;
            node.inputs.push_back(in);
            NodePin out;
            out.id = next_pin_id_++;
            out.type = PinType::Dataset;
            out.name = "Fitted";
            out.is_input = false;
            node.outputs.push_back(out);
            // Canonical params read by ARIMAOperator. p/d/q = -1 lets
            // the backend auto-select. Horizon is fixed to 0 — this is
            // in-sample fit; true forecasting needs a row-count-changing
            // operator deferred in tofix.
            node.parameters["signal_col"] = "";
            node.parameters["p"] = "-1";
            node.parameters["d"] = "-1";
            node.parameters["q"] = "-1";
            break;
        }

        case NodeType::ExponentialSmoothing: {
            NodePin in;
            in.id = next_pin_id_++;
            in.type = PinType::Dataset;
            in.name = "Data";
            in.is_input = true;
            node.inputs.push_back(in);
            NodePin out;
            out.id = next_pin_id_++;
            out.type = PinType::Dataset;
            out.name = "Fitted";
            out.is_input = false;
            node.outputs.push_back(out);
            // Canonical params read by ExponentialSmoothingOperator.
            // method picks simple / holt / holt_winters. alpha/beta/gamma=-1
            // lets the backend auto-tune. period is holt_winters only.
            node.parameters["signal_col"] = "";
            node.parameters["method"] = "simple";  // simple / holt / holt_winters
            node.parameters["alpha"] = "-1";
            node.parameters["beta"] = "-1";
            node.parameters["gamma"] = "-1";
            node.parameters["period"] = "-1";
            node.parameters["damped"] = "false";
            break;
        }

        // ========== Audio Processing Nodes ==========

        case NodeType::AudioInput: {
            PopulateStaticNodeContractFromMetadata(node, next_pin_id_);
            break;
        }

        case NodeType::Spectrogram:
        case NodeType::MelSpectrogram:
        case NodeType::MFCC: {
            PopulateStaticNodeContractFromMetadata(node, next_pin_id_);
            break;
        }

        case NodeType::AudioAugmentation: {
            NodePin in;
            in.id = next_pin_id_++;
            in.type = PinType::Tensor;
            in.name = "Audio";
            in.is_input = true;
            node.inputs.push_back(in);
            NodePin out;
            out.id = next_pin_id_++;
            out.type = PinType::Tensor;
            out.name = "Augmented";
            out.is_input = false;
            node.outputs.push_back(out);
            node.parameters["noise_level"] = "0.01";
            node.parameters["time_stretch"] = "false";
            node.parameters["pitch_shift"] = "false";
            break;
        }

        // ========== Reinforcement Learning Nodes ==========

        case NodeType::GymEnvironment: {
            NodePin obs_out;
            obs_out.id = next_pin_id_++;
            obs_out.type = PinType::Tensor;
            obs_out.name = "Observation";
            obs_out.is_input = false;
            node.outputs.push_back(obs_out);
            NodePin reward_out;
            reward_out.id = next_pin_id_++;
            reward_out.type = PinType::Tensor;
            reward_out.name = "Reward";
            reward_out.is_input = false;
            node.outputs.push_back(reward_out);
            NodePin done_out;
            done_out.id = next_pin_id_++;
            done_out.type = PinType::Tensor;
            done_out.name = "Done";
            done_out.is_input = false;
            node.outputs.push_back(done_out);
            NodePin info_out;
            info_out.id = next_pin_id_++;
            info_out.type = PinType::Tensor;
            info_out.name = "Info";
            info_out.is_input = false;
            node.outputs.push_back(info_out);
            // Environment parameters
            node.parameters["env_name"] = "CartPole-v1";
            node.parameters["env_type"] = "classic_control";  // classic_control, box2d, mujoco, atari
            node.parameters["render"] = "false";
            node.parameters["render_mode"] = "rgb_array";  // human, rgb_array, ansi
            node.parameters["max_episode_steps"] = "500";
            node.parameters["n_envs"] = "4";
            node.parameters["seed"] = "42";
            node.parameters["normalize_obs"] = "false";
            node.parameters["normalize_reward"] = "false";
            node.parameters["frame_stack"] = "1";  // For Atari/image-based envs
            break;
        }

        case NodeType::ReplayBufferNode: {
            NodePin transition_in;
            transition_in.id = next_pin_id_++;
            transition_in.type = PinType::Tensor;
            transition_in.name = "Transition";
            transition_in.is_input = true;
            node.inputs.push_back(transition_in);
            NodePin batch_out;
            batch_out.id = next_pin_id_++;
            batch_out.type = PinType::Tensor;
            batch_out.name = "Batch";
            batch_out.is_input = false;
            node.outputs.push_back(batch_out);
            NodePin priority_out;
            priority_out.id = next_pin_id_++;
            priority_out.type = PinType::Tensor;
            priority_out.name = "Priorities";
            priority_out.is_input = false;
            node.outputs.push_back(priority_out);
            // Buffer configuration
            node.parameters["capacity"] = "100000";
            node.parameters["batch_size"] = "256";
            // Prioritized experience replay
            node.parameters["prioritized"] = "false";
            node.parameters["alpha"] = "0.6";  // PER priority exponent
            node.parameters["beta_start"] = "0.4";  // PER importance sampling initial
            node.parameters["beta_frames"] = "100000";  // Frames to anneal beta to 1.0
            // N-step returns
            node.parameters["n_step"] = "1";
            node.parameters["gamma"] = "0.99";  // Discount for n-step
            // HER (Hindsight Experience Replay)
            node.parameters["her_strategy"] = "none";  // none, future, episode, random
            node.parameters["her_k"] = "4";  // Number of HER goals per transition
            break;
        }

        case NodeType::PolicyNetwork: {
            NodePin state_in;
            state_in.id = next_pin_id_++;
            state_in.type = PinType::Tensor;
            state_in.name = "State";
            state_in.is_input = true;
            node.inputs.push_back(state_in);
            NodePin action_out;
            action_out.id = next_pin_id_++;
            action_out.type = PinType::Tensor;
            action_out.name = "Action";
            action_out.is_input = false;
            node.outputs.push_back(action_out);
            NodePin log_prob_out;
            log_prob_out.id = next_pin_id_++;
            log_prob_out.type = PinType::Tensor;
            log_prob_out.name = "LogProb";
            log_prob_out.is_input = false;
            node.outputs.push_back(log_prob_out);
            // Network architecture
            node.parameters["hidden_sizes"] = "128,128";  // Comma-separated layer sizes
            node.parameters["activation"] = "ReLU";  // ReLU, Tanh, GELU, Swish
            node.parameters["output_activation"] = "None";  // None, Tanh (for continuous)
            // Distribution type
            node.parameters["action_space"] = "discrete";  // discrete, continuous, multi_discrete
            node.parameters["log_std_init"] = "0.0";  // For continuous actions
            // Orthogonal initialization
            node.parameters["ortho_init"] = "true";
            break;
        }

        case NodeType::ValueNetwork: {
            NodePin state_in;
            state_in.id = next_pin_id_++;
            state_in.type = PinType::Tensor;
            state_in.name = "State";
            state_in.is_input = true;
            node.inputs.push_back(state_in);
            NodePin value_out;
            value_out.id = next_pin_id_++;
            value_out.type = PinType::Tensor;
            value_out.name = "Value";
            value_out.is_input = false;
            node.outputs.push_back(value_out);
            // Network architecture
            node.parameters["hidden_sizes"] = "128,128";  // Comma-separated layer sizes
            node.parameters["activation"] = "ReLU";  // ReLU, Tanh, GELU, Swish
            // Orthogonal initialization
            node.parameters["ortho_init"] = "true";
            break;
        }

        case NodeType::RLTraining: {
            NodePin policy_in;
            policy_in.id = next_pin_id_++;
            policy_in.type = PinType::Tensor;
            policy_in.name = "Policy";
            policy_in.is_input = true;
            node.inputs.push_back(policy_in);
            NodePin env_in;
            env_in.id = next_pin_id_++;
            env_in.type = PinType::Tensor;
            env_in.name = "Environment";
            env_in.is_input = true;
            node.inputs.push_back(env_in);
            NodePin value_in;
            value_in.id = next_pin_id_++;
            value_in.type = PinType::Tensor;
            value_in.name = "Value";
            value_in.is_input = true;
            node.inputs.push_back(value_in);
            NodePin buffer_in;
            buffer_in.id = next_pin_id_++;
            buffer_in.type = PinType::Tensor;
            buffer_in.name = "ReplayBuffer";
            buffer_in.is_input = true;
            node.inputs.push_back(buffer_in);
            NodePin metrics_out;
            metrics_out.id = next_pin_id_++;
            metrics_out.type = PinType::Tensor;
            metrics_out.name = "Metrics";
            metrics_out.is_input = false;
            node.outputs.push_back(metrics_out);
            NodePin model_out;
            model_out.id = next_pin_id_++;
            model_out.type = PinType::Tensor;
            model_out.name = "TrainedModel";
            model_out.is_input = false;
            node.outputs.push_back(model_out);
            // Algorithm selection
            node.parameters["algorithm"] = "PPO";  // PPO, A2C, SAC, TD3, DQN, DDPG, TRPO
            // Training hyperparameters
            node.parameters["total_timesteps"] = "100000";
            node.parameters["learning_rate"] = "3e-4";
            node.parameters["batch_size"] = "64";
            node.parameters["n_epochs"] = "10";  // PPO epochs per update
            // Discount and advantage
            node.parameters["gamma"] = "0.99";
            node.parameters["gae_lambda"] = "0.95";  // GAE lambda for advantage estimation
            // PPO-specific
            node.parameters["clip_range"] = "0.2";
            node.parameters["ent_coef"] = "0.0";  // Entropy coefficient
            node.parameters["vf_coef"] = "0.5";  // Value function coefficient
            node.parameters["max_grad_norm"] = "0.5";
            // SAC/TD3-specific
            node.parameters["tau"] = "0.005";  // Soft update coefficient
            node.parameters["train_freq"] = "1";
            node.parameters["gradient_steps"] = "1";
            // Logging and checkpointing
            node.parameters["log_interval"] = "1";
            node.parameters["eval_freq"] = "5000";
            node.parameters["save_freq"] = "10000";
            node.parameters["tensorboard"] = "true";
            break;
        }

        case NodeType::PluginCustom: {
            // Copy node info to avoid use-after-free if plugin unloads
            auto info_opt = cyxwiz::plugin::PluginNodeRegistry::Instance().GetNodeTypeInfoCopy(name);
            node.parameters["plugin_qualified_name"] = name;
            node.plugin_qualified_name = name;
            if (info_opt.has_value()) {
                const auto& info = info_opt.value();
                node.name = info.display_name;
                for (const auto& pin : info.pins) {
                    NodePin p;
                    p.id = next_pin_id_++;
                    p.type = PinType::Tensor;  // Default; plugins use Tensor type
                    p.name = pin.name;
                    p.is_input = pin.is_input;
                    if (pin.is_input) node.inputs.push_back(p);
                    else node.outputs.push_back(p);
                }
                for (const auto& [key, val] : info.default_parameters) {
                    node.parameters[key] = val;
                }
                // Dynamic pin support
                if (info.supports_dynamic_pins) {
                    node.has_dynamic_pins = true;
                    node.dynamic_pin_trigger = info.dynamic_pin_trigger;
                }
            } else {
                // Fallback: plugin not loaded, create generic node
                node.name = "Plugin Node (Missing)";
                NodePin in; in.id = next_pin_id_++; in.type = PinType::Tensor;
                in.name = "Input"; in.is_input = true; node.inputs.push_back(in);
                NodePin out; out.id = next_pin_id_++; out.type = PinType::Tensor;
                out.name = "Output"; out.is_input = false; node.outputs.push_back(out);
            }
            break;
        }

        // ========== Smart I/O Nodes (Universal Data Input/Output) ==========

        case NodeType::DataInput: {
            // The dialog owns dynamic source, parser, schema, and loading
            // fields. Metadata owns only the new node's static contract.
            PopulateStaticNodeContractFromMetadata(node, next_pin_id_);
            break;
        }

        case NodeType::DataOutput: {
            PopulateStaticNodeContractFromMetadata(node, next_pin_id_);
            break;
        }

        case NodeType::SeasonalNaive: {
            NodePin in;
            in.id = next_pin_id_++;
            in.type = PinType::Dataset;
            in.name = "Windowed";
            in.is_input = true;
            in.description =
                "Canonical Sliding Window table containing x_* history and "
                "ordered y targets. Time-series split metadata is optional.";
            node.inputs.push_back(in);
            NodePin out;
            out.id = next_pin_id_++;
            out.type = PinType::Dataset;
            out.name = "Predictions";
            out.is_input = false;
            out.description =
                "Long-form actual and seasonal-naive prediction rows. "
                "Filter __partition__ = 2 for held-out testing, then connect "
                "to Regression Metrics.";
            node.outputs.push_back(out);
            node.parameters["seasonal_period"] = "1";
            break;
        }

        case NodeType::DataConvert: {
            PopulateStaticNodeContractFromMetadata(node, next_pin_id_);
            break;
        }

        // ========== Legacy Data Source Nodes (kept for compatibility) ==========

        case NodeType::CSVFile: {
            // CSV File data source node - loads CSV into dataset
            NodePin output_pin;
            output_pin.id = next_pin_id_++;
            output_pin.type = PinType::Dataset;
            output_pin.name = "Data";
            output_pin.is_input = false;
            node.outputs.push_back(output_pin);

            node.parameters["file_path"] = "";
            node.parameters["delimiter"] = ",";
            node.parameters["header"] = "true";
            break;
        }

        case NodeType::FilterRows:
        case NodeType::SelectColumns: {
            PopulateStaticNodeContractFromMetadata(node, next_pin_id_);
            break;
        }

        case NodeType::DescribeStats: {
            PopulateStaticNodeContractFromMetadata(node, next_pin_id_);
            break;
        }

        case NodeType::JoinTables: {
            PopulateStaticNodeContractFromMetadata(node, next_pin_id_);
            break;
        }

        case NodeType::SortRows:
        case NodeType::GroupByAggregate:
        case NodeType::FillMissingValues:
        case NodeType::RemoveDuplicateRows:
        case NodeType::RenameColumns: {
            PopulateStaticNodeContractFromMetadata(node, next_pin_id_);
            break;
        }

        case NodeType::SampleRows:
        case NodeType::ValueCounts:
        case NodeType::CorrelationMatrix: {
            PopulateStaticNodeContractFromMetadata(node, next_pin_id_);
            break;
        }

        case NodeType::SQLQuery: {
            // SQL Query data source node - execute custom SQL
            NodePin output_pin;
            output_pin.id = next_pin_id_++;
            output_pin.type = PinType::Dataset;
            output_pin.name = "Data";
            output_pin.is_input = false;
            node.outputs.push_back(output_pin);

            node.parameters["connection_string"] = "sqlite:///data.db";
            node.parameters["query"] = "SELECT * FROM table";
            break;
        }

        case NodeType::ParquetFile: {
            // Parquet File data source node
            NodePin output_pin;
            output_pin.id = next_pin_id_++;
            output_pin.type = PinType::Dataset;
            output_pin.name = "Data";
            output_pin.is_input = false;
            node.outputs.push_back(output_pin);

            node.parameters["file_path"] = "";
            break;
        }

        case NodeType::ExportCSV:
        case NodeType::ExportParquet:
        case NodeType::ExportJSON:
        case NodeType::ExportExcel: {
            PopulateStaticNodeContractFromMetadata(node, next_pin_id_);
            break;
        }

        // ========== KNIME-Style Table Manipulation Nodes ==========

        case NodeType::TableSplitter: {
            PopulateStaticNodeContractFromMetadata(node, next_pin_id_);
            break;
        }

        case NodeType::CellExtractor: {
            PopulateStaticNodeContractFromMetadata(node, next_pin_id_);
            break;
        }

        case NodeType::CellUpdater: {
            PopulateStaticNodeContractFromMetadata(node, next_pin_id_);
            break;
        }

        case NodeType::RowAppender: {
            PopulateStaticNodeContractFromMetadata(node, next_pin_id_);
            break;
        }

        case NodeType::ColumnAppender: {
            PopulateStaticNodeContractFromMetadata(node, next_pin_id_);
            break;
        }

        case NodeType::RowToColumnNames:
        case NodeType::TableCropper:
        case NodeType::Unpivot:
        case NodeType::StringManipulation:
        case NodeType::MathFormula:
        case NodeType::RuleEngine: {
            PopulateStaticNodeContractFromMetadata(node, next_pin_id_);
            break;
        }

        
        // ===== Phase 4: Machine Learning Algorithm Nodes =====

        // Clustering nodes
        case NodeType::KMeansCluster:
        case NodeType::DBSCANCluster:
        case NodeType::HierarchicalCluster:
        case NodeType::GMMCluster: {
            PopulateStaticNodeContractFromMetadata(node, next_pin_id_);
            break;
        }

        // Dimensionality Reduction nodes
        case NodeType::PCANode: {
            PopulateStaticNodeContractFromMetadata(node, next_pin_id_);
            break;
        }

        case NodeType::TSNENode: {
            NodePin input_pin;
            input_pin.id = next_pin_id_++;
            input_pin.type = PinType::Dataset;
            input_pin.name = "Data";
            input_pin.is_input = true;
            node.inputs.push_back(input_pin);

            NodePin output_pin;
            output_pin.id = next_pin_id_++;
            output_pin.type = PinType::Dataset;
            output_pin.name = "Embedded";
            output_pin.is_input = false;
            node.outputs.push_back(output_pin);

            node.parameters["n_components"] = "2";
            node.parameters["perplexity"] = "30";
            node.parameters["learning_rate"] = "200";
            break;
        }

        case NodeType::UMAPNode: {
            NodePin input_pin;
            input_pin.id = next_pin_id_++;
            input_pin.type = PinType::Dataset;
            input_pin.name = "Data";
            input_pin.is_input = true;
            node.inputs.push_back(input_pin);

            NodePin output_pin;
            output_pin.id = next_pin_id_++;
            output_pin.type = PinType::Dataset;
            output_pin.name = "Embedded";
            output_pin.is_input = false;
            node.outputs.push_back(output_pin);

            node.parameters["n_components"] = "2";
            node.parameters["n_neighbors"] = "15";
            node.parameters["min_dist"] = "0.1";
            break;
        }

        // Classification nodes
        case NodeType::DecisionTreeClassifier:
        case NodeType::RandomForestClassifier:
        case NodeType::GradientBoostingClassifier:
        case NodeType::TreeModelPredictor: {
            PopulateStaticNodeContractFromMetadata(node, next_pin_id_);
            break;
        }

        case NodeType::RegressionModelPredictor: {
            PopulateStaticNodeContractFromMetadata(node, next_pin_id_);
            break;
        }

        case NodeType::SVMClassifier:
        case NodeType::KNNClassifier:
        case NodeType::NaiveBayesClassifier:
        case NodeType::LogisticRegressionNode: {
            // Saved graphs retain their historical static preview contract,
            // but runtime capability truth keeps these nodes blocked until a
            // real fit/artifact/predict owner exists.
            PopulateStaticNodeContractFromMetadata(node, next_pin_id_);
            break;
        }

        // Regression nodes
        case NodeType::LinearRegressionNode:
        case NodeType::PolynomialRegressionNode: {
            PopulateStaticNodeContractFromMetadata(node, next_pin_id_);
            break;
        }

        case NodeType::SVMRegressor: {
            NodePin data_in;
            data_in.id = next_pin_id_++;
            data_in.type = PinType::Dataset;
            data_in.name = "Train Data";
            data_in.is_input = true;
            node.inputs.push_back(data_in);

            NodePin target_in;
            target_in.id = next_pin_id_++;
            target_in.type = PinType::Tensor;
            target_in.name = "Target";
            target_in.is_input = true;
            node.inputs.push_back(target_in);

            NodePin model_out;
            model_out.id = next_pin_id_++;
            model_out.type = PinType::Parameters;
            model_out.name = "Model";
            model_out.is_input = false;
            node.outputs.push_back(model_out);

            NodePin pred_out;
            pred_out.id = next_pin_id_++;
            pred_out.type = PinType::Tensor;
            pred_out.name = "Predictions";
            pred_out.is_input = false;
            node.outputs.push_back(pred_out);

            node.parameters["kernel"] = "rbf";
            node.parameters["C"] = "1.0";
            node.parameters["epsilon"] = "0.1";
            break;
        }

        // ===== Phase 4: Model Evaluation Nodes =====
        case NodeType::ConfusionMatrixNode:
        case NodeType::ROCCurveNode:
        case NodeType::PRCurveNode: {
            PopulateStaticNodeContractFromMetadata(node, next_pin_id_);
            break;
        }

        case NodeType::LearningCurvesNode: {
            NodePin model_in;
            model_in.id = next_pin_id_++;
            model_in.type = PinType::Parameters;
            model_in.name = "Model";
            model_in.is_input = true;
            node.inputs.push_back(model_in);

            NodePin data_in;
            data_in.id = next_pin_id_++;
            data_in.type = PinType::Dataset;
            data_in.name = "Data";
            data_in.is_input = true;
            node.inputs.push_back(data_in);

            NodePin labels_in;
            labels_in.id = next_pin_id_++;
            labels_in.type = PinType::Labels;
            labels_in.name = "Labels";
            labels_in.is_input = true;
            node.inputs.push_back(labels_in);

            NodePin curves_out;
            curves_out.id = next_pin_id_++;
            curves_out.type = PinType::Dataset;
            curves_out.name = "Curves";
            curves_out.is_input = false;
            node.outputs.push_back(curves_out);

            node.parameters["cv"] = "5";
            node.parameters["train_sizes"] = "0.1,0.3,0.5,0.7,0.9";
            break;
        }

        case NodeType::FeatureImportanceNode: {
            NodePin model_in;
            model_in.id = next_pin_id_++;
            model_in.type = PinType::Parameters;
            model_in.name = "Model";
            model_in.is_input = true;
            node.inputs.push_back(model_in);

            NodePin importance_out;
            importance_out.id = next_pin_id_++;
            importance_out.type = PinType::Dataset;
            importance_out.name = "Importance";
            importance_out.is_input = false;
            node.outputs.push_back(importance_out);

            node.parameters["method"] = "builtin";
            break;
        }

        case NodeType::CrossValidationNode: {
            NodePin model_in;
            model_in.id = next_pin_id_++;
            model_in.type = PinType::Parameters;
            model_in.name = "Model";
            model_in.is_input = true;
            node.inputs.push_back(model_in);

            NodePin data_in;
            data_in.id = next_pin_id_++;
            data_in.type = PinType::Dataset;
            data_in.name = "Data";
            data_in.is_input = true;
            node.inputs.push_back(data_in);

            NodePin labels_in;
            labels_in.id = next_pin_id_++;
            labels_in.type = PinType::Labels;
            labels_in.name = "Labels";
            labels_in.is_input = true;
            node.inputs.push_back(labels_in);

            NodePin scores_out;
            scores_out.id = next_pin_id_++;
            scores_out.type = PinType::Dataset;
            scores_out.name = "Scores";
            scores_out.is_input = false;
            node.outputs.push_back(scores_out);

            node.parameters["cv"] = "5";
            node.parameters["scoring"] = "accuracy";
            break;
        }

        case NodeType::RegressionMetricsNode:
        case NodeType::ClassificationMetricsNode: {
            PopulateStaticNodeContractFromMetadata(node, next_pin_id_);
            break;
        }

        // ===== Phase 4: Data Preprocessing Nodes =====
        case NodeType::StandardScaler: {
            PopulateStaticNodeContractFromMetadata(node, next_pin_id_);
            break;
        }

        case NodeType::MinMaxScaler:
        case NodeType::RobustScaler:
        case NodeType::LabelEncoder:
        case NodeType::OrdinalEncoder:
        case NodeType::TargetEncoder: {
            PopulateStaticNodeContractFromMetadata(node, next_pin_id_);
            break;
        }

        // (TrainTestSplit case removed — use NodeType::DataSplit instead,
        //  which supports 3-way train/val/test split and is the single source
        //  of truth for dataset partitioning.)

        // ===== Phase 8: Advanced Preprocessing Nodes (UI Consolidation) =====
        case NodeType::OutlierDetector: {
            PopulateStaticNodeContractFromMetadata(node, next_pin_id_);
            break;
        }

        case NodeType::ImagePreprocessor: {
            NodePin images_in;
            images_in.id = next_pin_id_++;
            images_in.type = PinType::Tensor;
            images_in.name = "Images";
            images_in.is_input = true;
            node.inputs.push_back(images_in);

            NodePin processed_out;
            processed_out.id = next_pin_id_++;
            processed_out.type = PinType::Tensor;
            processed_out.name = "Processed";
            processed_out.is_input = false;
            node.outputs.push_back(processed_out);

            node.parameters["resize_mode"] = "aspect_fit";  // exact, aspect_fit, aspect_fill, center
            node.parameters["target_width"] = "224";
            node.parameters["target_height"] = "224";
            node.parameters["normalize"] = "true";
            node.parameters["mean"] = "0.485,0.456,0.406";  // ImageNet mean
            node.parameters["std"] = "0.229,0.224,0.225";   // ImageNet std
            node.parameters["interpolation"] = "bilinear";  // nearest, bilinear, bicubic
            node.parameters["padding_mode"] = "reflect";    // constant, reflect, replicate
            break;
        }

        case NodeType::QualityAnalyzer: {
            NodePin images_in;
            images_in.id = next_pin_id_++;
            images_in.type = PinType::Dataset;
            images_in.name = "Images";
            images_in.is_input = true;
            node.inputs.push_back(images_in);

            NodePin passed_out;
            passed_out.id = next_pin_id_++;
            passed_out.type = PinType::Dataset;
            passed_out.name = "Passed";
            passed_out.is_input = false;
            node.outputs.push_back(passed_out);

            NodePin rejected_out;
            rejected_out.id = next_pin_id_++;
            rejected_out.type = PinType::Dataset;
            rejected_out.name = "Rejected";
            rejected_out.is_input = false;
            node.outputs.push_back(rejected_out);

            NodePin report_out;
            report_out.id = next_pin_id_++;
            report_out.type = PinType::Dataset;
            report_out.name = "Report";
            report_out.is_input = false;
            node.outputs.push_back(report_out);

            node.parameters["blur_threshold"] = "100.0";      // Laplacian variance threshold
            node.parameters["brightness_min"] = "30";         // Min acceptable brightness
            node.parameters["brightness_max"] = "220";        // Max acceptable brightness
            node.parameters["contrast_threshold"] = "0.2";    // Min contrast (std dev)
            node.parameters["noise_threshold"] = "50.0";      // Max noise level
            node.parameters["duplicate_check"] = "true";      // Check for duplicates
            node.parameters["aspect_ratio_tolerance"] = "0.1"; // Aspect ratio consistency
            break;
        }

        case NodeType::DataValidator: {
            PopulateStaticNodeContractFromMetadata(node, next_pin_id_);
            break;
        }

        // ===== Phase 4: Dataset Source Nodes (UI Consolidation) =====
        case NodeType::ImageFolderDataset: {
            // ImageFolderDataset - Load images from folder with class labels
            NodePin images_out;
            images_out.id = next_pin_id_++;
            images_out.type = PinType::Dataset;
            images_out.name = "Images";
            images_out.is_input = false;
            node.outputs.push_back(images_out);

            NodePin labels_out;
            labels_out.id = next_pin_id_++;
            labels_out.type = PinType::Tensor;
            labels_out.name = "Labels";
            labels_out.is_input = false;
            node.outputs.push_back(labels_out);

            NodePin metadata_out;
            metadata_out.id = next_pin_id_++;
            metadata_out.type = PinType::Dataset;
            metadata_out.name = "Metadata";
            metadata_out.is_input = false;
            node.outputs.push_back(metadata_out);

            node.parameters["path"] = "";                      // Folder path
            node.parameters["extensions"] = ".jpg,.png,.bmp";  // File extensions to include
            node.parameters["class_mode"] = "folder";          // folder, filename, none
            node.parameters["recursive"] = "true";             // Scan subdirectories
            break;
        }

        case NodeType::MNISTDataset: {
            // MNISTDataset - Load MNIST handwritten digits
            NodePin images_out;
            images_out.id = next_pin_id_++;
            images_out.type = PinType::Tensor;
            images_out.name = "Images";
            images_out.is_input = false;
            node.outputs.push_back(images_out);

            NodePin labels_out;
            labels_out.id = next_pin_id_++;
            labels_out.type = PinType::Tensor;
            labels_out.name = "Labels";
            labels_out.is_input = false;
            node.outputs.push_back(labels_out);

            node.parameters["split"] = "train";                // train, test
            node.parameters["path"] = "";                      // Cache directory (default: ~/.cyxwiz/datasets)
            node.parameters["download"] = "true";              // Auto-download if missing
            node.parameters["flatten"] = "false";              // Flatten 28x28 to 784
            break;
        }

        case NodeType::CIFAR10Dataset: {
            // CIFAR10Dataset - Load CIFAR-10 image classification dataset
            NodePin images_out;
            images_out.id = next_pin_id_++;
            images_out.type = PinType::Tensor;
            images_out.name = "Images";
            images_out.is_input = false;
            node.outputs.push_back(images_out);

            NodePin labels_out;
            labels_out.id = next_pin_id_++;
            labels_out.type = PinType::Tensor;
            labels_out.name = "Labels";
            labels_out.is_input = false;
            node.outputs.push_back(labels_out);

            node.parameters["split"] = "train";                // train, test
            node.parameters["path"] = "";                      // Cache directory (default: ~/.cyxwiz/datasets)
            node.parameters["download"] = "true";              // Auto-download if missing
            break;
        }

        case NodeType::HuggingFaceDataset: {
            // HuggingFaceDataset - Load dataset from HuggingFace Hub
            NodePin dataset_out;
            dataset_out.id = next_pin_id_++;
            dataset_out.type = PinType::Dataset;
            dataset_out.name = "Dataset";
            dataset_out.is_input = false;
            node.outputs.push_back(dataset_out);

            node.parameters["dataset_id"] = "";                // HuggingFace dataset ID (e.g., "imdb", "squad")
            node.parameters["split"] = "train";                // train, test, validation
            node.parameters["subset"] = "";                    // Dataset subset/config (optional)
            node.parameters["streaming"] = "false";            // Enable streaming for large datasets
            break;
        }

        case NodeType::KaggleDataset: {
            // KaggleDataset - Load dataset from Kaggle
            NodePin dataset_out;
            dataset_out.id = next_pin_id_++;
            dataset_out.type = PinType::Dataset;
            dataset_out.name = "Dataset";
            dataset_out.is_input = false;
            node.outputs.push_back(dataset_out);

            node.parameters["dataset_id"] = "";                // Kaggle dataset ID (e.g., "username/dataset-name")
            node.parameters["path"] = "";                      // Download path
            node.parameters["unzip"] = "true";                 // Auto-unzip downloaded files
            break;
        }

        // ===== Phase 6: Advanced Augmentation Nodes (UI Consolidation) =====
        case NodeType::AugmentationPreset: {
            // AugmentationPreset - Predefined augmentation pipelines
            NodePin input_pin;
            input_pin.id = next_pin_id_++;
            input_pin.type = PinType::Tensor;
            input_pin.name = "Input";
            input_pin.is_input = true;
            node.inputs.push_back(input_pin);

            NodePin output_pin;
            output_pin.id = next_pin_id_++;
            output_pin.type = PinType::Tensor;
            output_pin.name = "Output";
            output_pin.is_input = false;
            node.outputs.push_back(output_pin);

            node.parameters["preset"] = "ImageNet";            // ImageNet, CIFAR, Medical, SelfSupervised, Custom
            node.parameters["normalize"] = "true";             // Apply normalization
            node.parameters["resize"] = "224,224";             // Target size (preset-specific default)
            break;
        }

        case NodeType::GeometricTransform: {
            // GeometricTransform - Geometric transforms (rotate, flip, crop, perspective)
            NodePin input_pin;
            input_pin.id = next_pin_id_++;
            input_pin.type = PinType::Tensor;
            input_pin.name = "Input";
            input_pin.is_input = true;
            node.inputs.push_back(input_pin);

            NodePin output_pin;
            output_pin.id = next_pin_id_++;
            output_pin.type = PinType::Tensor;
            output_pin.name = "Output";
            output_pin.is_input = false;
            node.outputs.push_back(output_pin);

            node.parameters["transform"] = "rotate";           // rotate, flip_h, flip_v, crop, perspective, affine
            node.parameters["angle_range"] = "-30,30";         // Rotation angle range (degrees)
            node.parameters["flip_prob"] = "0.5";              // Probability for flip transforms
            node.parameters["crop_scale"] = "0.8,1.0";         // Random crop scale range
            node.parameters["perspective_distortion"] = "0.2"; // Perspective distortion scale
            break;
        }

        case NodeType::ColorTransform: {
            // ColorTransform - Color space transforms
            NodePin input_pin;
            input_pin.id = next_pin_id_++;
            input_pin.type = PinType::Tensor;
            input_pin.name = "Input";
            input_pin.is_input = true;
            node.inputs.push_back(input_pin);

            NodePin output_pin;
            output_pin.id = next_pin_id_++;
            output_pin.type = PinType::Tensor;
            output_pin.name = "Output";
            output_pin.is_input = false;
            node.outputs.push_back(output_pin);

            node.parameters["brightness_range"] = "0.8,1.2";   // Brightness multiplier range
            node.parameters["contrast_range"] = "0.8,1.2";     // Contrast multiplier range
            node.parameters["saturation_range"] = "0.8,1.2";   // Saturation multiplier range
            node.parameters["hue_range"] = "-0.1,0.1";         // Hue shift range
            node.parameters["gamma_range"] = "0.8,1.2";        // Gamma correction range
            break;
        }

        case NodeType::MorphologyTransform: {
            // MorphologyTransform - Morphological operations
            NodePin input_pin;
            input_pin.id = next_pin_id_++;
            input_pin.type = PinType::Tensor;
            input_pin.name = "Input";
            input_pin.is_input = true;
            node.inputs.push_back(input_pin);

            NodePin output_pin;
            output_pin.id = next_pin_id_++;
            output_pin.type = PinType::Tensor;
            output_pin.name = "Output";
            output_pin.is_input = false;
            node.outputs.push_back(output_pin);

            node.parameters["operation"] = "blur";             // blur, sharpen, dilate, erode, edge
            node.parameters["kernel_size"] = "3";              // Kernel size for morphological ops
            node.parameters["sigma"] = "1.0";                  // Sigma for Gaussian blur
            node.parameters["strength"] = "1.0";               // Effect strength
            break;
        }

        case NodeType::AdvancedAugment: {
            // AdvancedAugment - Advanced augmentation techniques
            NodePin input_pin;
            input_pin.id = next_pin_id_++;
            input_pin.type = PinType::Tensor;
            input_pin.name = "Input";
            input_pin.is_input = true;
            node.inputs.push_back(input_pin);

            NodePin output_pin;
            output_pin.id = next_pin_id_++;
            output_pin.type = PinType::Tensor;
            output_pin.name = "Output";
            output_pin.is_input = false;
            node.outputs.push_back(output_pin);

            node.parameters["method"] = "Cutout";              // Cutout, MixUp, CutMix, RandAugment, AutoAugment
            node.parameters["cutout_size"] = "16";             // Size of cutout region (Cutout)
            node.parameters["mixup_alpha"] = "0.2";            // MixUp/CutMix alpha parameter
            node.parameters["num_ops"] = "2";                  // Number of operations (RandAugment)
            node.parameters["magnitude"] = "9";                // Magnitude level (RandAugment/AutoAugment)
            break;
        }

        // ===== Phase 4: Signal Processing Nodes =====
        case NodeType::FFTNode:
        case NodeType::FilterDesigner:
        case NodeType::Convolution1D: {
            PopulateStaticNodeContractFromMetadata(node, next_pin_id_);
            break;
        }

        case NodeType::IFFTNode: {
            NodePin input_pin;
            input_pin.id = next_pin_id_++;
            input_pin.type = PinType::Tensor;
            input_pin.name = "Spectrum";
            input_pin.is_input = true;
            node.inputs.push_back(input_pin);

            NodePin output_pin;
            output_pin.id = next_pin_id_++;
            output_pin.type = PinType::Tensor;
            output_pin.name = "Signal";
            output_pin.is_input = false;
            node.outputs.push_back(output_pin);
            break;
        }

        case NodeType::WaveletTransform: {
            NodePin input_pin;
            input_pin.id = next_pin_id_++;
            input_pin.type = PinType::Tensor;
            input_pin.name = "Signal";
            input_pin.is_input = true;
            node.inputs.push_back(input_pin);

            NodePin approx_out;
            approx_out.id = next_pin_id_++;
            approx_out.type = PinType::Tensor;
            approx_out.name = "Approximation";
            approx_out.is_input = false;
            node.outputs.push_back(approx_out);

            NodePin detail_out;
            detail_out.id = next_pin_id_++;
            detail_out.type = PinType::Tensor;
            detail_out.name = "Detail";
            detail_out.is_input = false;
            node.outputs.push_back(detail_out);

            node.parameters["wavelet"] = "db4";
            node.parameters["level"] = "1";
            break;
        }

        // ===== Phase 4: Text Analytics Nodes =====
        case NodeType::TFIDFVectorizer:
        case NodeType::CountVectorizer:
        case NodeType::SentimentAnalyzer: {
            PopulateStaticNodeContractFromMetadata(node, next_pin_id_);
            break;
        }

        case NodeType::WordEmbeddings: {
            NodePin input_pin;
            input_pin.id = next_pin_id_++;
            input_pin.type = PinType::Dataset;
            input_pin.name = "Text";
            input_pin.is_input = true;
            node.inputs.push_back(input_pin);

            NodePin output_pin;
            output_pin.id = next_pin_id_++;
            output_pin.type = PinType::Tensor;
            output_pin.name = "Embeddings";
            output_pin.is_input = false;
            node.outputs.push_back(output_pin);

            node.parameters["model"] = "word2vec";
            node.parameters["dim"] = "100";
            node.parameters["window"] = "5";
            break;
        }

        case NodeType::NamedEntityRecognizer: {
            NodePin input_pin;
            input_pin.id = next_pin_id_++;
            input_pin.type = PinType::Dataset;
            input_pin.name = "Text";
            input_pin.is_input = true;
            node.inputs.push_back(input_pin);

            NodePin output_pin;
            output_pin.id = next_pin_id_++;
            output_pin.type = PinType::Dataset;
            output_pin.name = "Entities";
            output_pin.is_input = false;
            node.outputs.push_back(output_pin);

            node.parameters["model"] = "spacy";
            break;
        }

        // ===== Phase 4: Utility Nodes =====
        case NodeType::CalculatorNode:
        case NodeType::UnitConverter:
        case NodeType::RegexTester:
        case NodeType::JSONPathExtractor:
        case NodeType::DataProfiler: {
            PopulateStaticNodeContractFromMetadata(node, next_pin_id_);
            break;
        }

        case NodeType::BarChart: {
            PopulateStaticNodeContractFromMetadata(node, next_pin_id_);
            break;
        }

        default: {
            auto& registry = cyxwiz::NodeMetadataRegistry::Instance();
            registry.Initialize();
            if (registry.GetMetadata(type) != nullptr) {
                // Registered built-ins use the central contract even before a
                // specialized creation case is added. Unknown/dynamic plugin
                // nodes retain the compatibility Tensor fallback below.
                PopulateStaticNodeContractFromMetadata(node, next_pin_id_);
                break;
            }

            NodePin input_pin{};
            input_pin.id = next_pin_id_++;
            input_pin.type = PinType::Tensor;
            input_pin.name = "Input";
            input_pin.is_input = true;
            node.inputs.push_back(input_pin);

            NodePin output_pin{};
            output_pin.id = next_pin_id_++;
            output_pin.type = PinType::Tensor;
            output_pin.name = "Output";
            output_pin.is_input = false;
            node.outputs.push_back(output_pin);
            break;
        }
    }

    return node;
}

MLNode NodeEditor::CreateNode(NodeType type, const std::string& name) {
    return CreateNodeWithIds(
        type, name, next_node_id_, next_pin_id_);
}

#ifndef CYXWIZ_NODE_FACTORY_ONLY
// Helper: if this node owns a dataset (DataInput / DatasetInput with a
// non-empty dataset_name parameter), drop it from every registry so the
// next graph rebuild doesn't see a stale entry under the same name.
// Defined locally because it's only used by DeleteNode and ClearGraph.
static void UnregisterNodeDatasetIfOwned(const MLNode& node) {
    if (node.type != NodeType::DataInput &&
        node.type != NodeType::DatasetInput) {
        return;
    }
    auto it = node.parameters.find("dataset_name");
    if (it == node.parameters.end() || it->second.empty()) {
        return;
    }
    auto& reg = cyxwiz::DataRegistry::Instance();
    // Try every registry the node could have populated. Each is a no-op
    // for names that don't exist in that map, so it's safe to call them
    // all without knowing which type the node actually loaded.
    reg.UnregisterTabularDataset(it->second);
    reg.UnregisterImageDataset(it->second);
    reg.UnregisterAudioDataset(it->second);
    reg.UnregisterTextDataset(it->second);
}

void NodeEditor::DeleteNode(int node_id) {
    ClearValidationState();  // Graph changed — stale compile results

    // Delete node
    auto node_it = std::find_if(nodes_.begin(), nodes_.end(),
        [node_id](const MLNode& node) {
            return node.id == node_id;
        });

    if (node_it != nodes_.end()) {
        spdlog::info("Deleting node: {} (ID: {})", node_it->name, node_id);

        // Properties and configuration dialogs hold raw pointers into nodes_.
        // Erasing a vector element can invalidate pointers to this node and to
        // any elements shifted after it, so release every such reference first.
        if (properties_panel_) {
            properties_panel_->ClearNodeReferences();
        }

        // If this is a data input node, drop its registered dataset before
        // we erase the node — otherwise the registry entry leaks until app
        // exit (the next graph won't have a node referencing it).
        UnregisterNodeDatasetIfOwned(*node_it);

        // Delete all links connected to this node
        links_.erase(
            std::remove_if(links_.begin(), links_.end(),
                [node_id](const NodeLink& link) {
                    return link.from_node == node_id || link.to_node == node_id;
                }),
            links_.end());

        nodes_.erase(node_it);
        RebuildPinLookup();  // Rebuild pin lookup after deleting node
    }
}

void NodeEditor::ClearGraph() {
    SaveUndoState();
    ClearValidationState();  // Graph changed — stale compile results

    // IMPORTANT: Clear every properties/dialog node reference BEFORE clearing
    // nodes. Both the panel and active configuration dialog hold raw pointers.
    if (properties_panel_) {
        properties_panel_->ClearNodeReferences();
    }

    // Drop every data input node's registered dataset before we lose the
    // references in nodes_. Without this, "Clear All" leaves the datasets
    // orphaned in the registry.
    for (const auto& node : nodes_) {
        UnregisterNodeDatasetIfOwned(node);
    }

    nodes_.clear();
    links_.clear();
    next_node_id_ = 1;
    next_pin_id_ = 1;
    next_link_id_ = 1;

    // Reset selection state
    selected_node_id_ = -1;
    selected_node_ids_.clear();

    // Request a full ImNodes context reset - this fully clears ImNodes' internal state
    // which prevents crashes from stale node references
    pending_context_reset_ = true;

    // Clear any pending positions
    pending_positions_.clear();

    // CyxWiz Studio: Clear groups and annotations
    groups_.clear();
    next_group_id_ = 1;
    annotations_.clear();
    next_annotation_id_ = 1;

    // Clear pin lookup
    pin_lookup_.clear();

    spdlog::info("Cleared node graph");
}

void NodeEditor::InsertPattern(const std::vector<MLNode>& nodes, const std::vector<NodeLink>& links) {
    if (nodes.empty()) {
        spdlog::warn("InsertPattern called with empty nodes list");
        return;
    }

    SaveUndoState();

    // Add all nodes from the pattern
    for (const auto& node : nodes) {
        nodes_.push_back(node);

        // Queue position for deferred setting (will be applied during render)
        if (node.has_initial_position) {
            pending_positions_[node.id] = ImVec2(node.initial_pos_x, node.initial_pos_y);
        }

        // Update next IDs to avoid collisions
        if (node.id >= next_node_id_) {
            next_node_id_ = node.id + 1;
        }
        for (const auto& pin : node.inputs) {
            if (pin.id >= next_pin_id_) {
                next_pin_id_ = pin.id + 1;
            }
        }
        for (const auto& pin : node.outputs) {
            if (pin.id >= next_pin_id_) {
                next_pin_id_ = pin.id + 1;
            }
        }
    }

    // Add all links from the pattern
    for (const auto& link : links) {
        links_.push_back(link);

        // Update next link ID
        if (link.id >= next_link_id_) {
            next_link_id_ = link.id + 1;
        }
    }

    // Set frame counter to apply positions during next few render frames
    // This is required because ImNodes needs nodes to exist before SetNodeGridSpacePos works
    if (!pending_positions_.empty()) {
        pending_positions_frames_ = 3;  // Apply for 3 frames to ensure positions stick
    }

    // Rebuild pin lookup after inserting pattern
    RebuildPinLookup();

    spdlog::info("Inserted pattern with {} nodes and {} links (positions queued: {})",
                 nodes.size(), links.size(), pending_positions_.size());
}

// ===== Undo/Redo System =====

const MLNode* NodeEditor::FindNodeById(int node_id) const {
    for (const auto& node : nodes_) {
        if (node.id == node_id) {
            return &node;
        }
    }
    return nullptr;
}

MLNode* NodeEditor::FindNodeById(int node_id) {
    for (auto& node : nodes_) {
        if (node.id == node_id) {
            return &node;
        }
    }
    return nullptr;
}

// ========== Color-Coding Implementation ==========
unsigned int NodeEditor::GetNodeColor(NodeType type) {
    switch (type) {
        // ===== Output - Blue =====
        case NodeType::Output:
            return IM_COL32(52, 152, 219, 255);

        // ===== Core Layers - Green =====
        case NodeType::Dense:
            return IM_COL32(39, 174, 96, 255);

        // ===== Convolutional Layers - Purple =====
        case NodeType::Conv1D:
        case NodeType::Conv2D:
        case NodeType::Conv3D:
        case NodeType::DepthwiseConv2D:
            return IM_COL32(142, 68, 173, 255);

        // ===== Pooling Layers - Light Purple =====
        case NodeType::MaxPool2D:
        case NodeType::AvgPool2D:
        case NodeType::GlobalMaxPool:
        case NodeType::GlobalAvgPool:
        case NodeType::AdaptiveAvgPool:
            return IM_COL32(155, 89, 182, 255);

        // ===== Normalization Layers - Pink/Coral =====
        case NodeType::BatchNorm:
        case NodeType::LayerNorm:
        case NodeType::GroupNorm:
        case NodeType::InstanceNorm:
            return IM_COL32(236, 112, 99, 255);

        // ===== Regularization - Red =====
        case NodeType::Dropout:
            return IM_COL32(231, 76, 60, 255);

        // ===== Utility Layers - Teal =====
        case NodeType::Flatten:
            return IM_COL32(22, 160, 133, 255);

        // ===== Recurrent Layers - Indigo =====
        case NodeType::RNN:
        case NodeType::LSTM:
        case NodeType::GRU:
        case NodeType::Bidirectional:
        case NodeType::TimeDistributed:
        case NodeType::Embedding:
            return IM_COL32(63, 81, 181, 255);

        // ===== Attention & Transformer - Deep Purple =====
        case NodeType::MultiHeadAttention:
        case NodeType::SelfAttention:
        case NodeType::CrossAttention:
        case NodeType::LinearAttention:
        case NodeType::TransformerEncoder:
        case NodeType::TransformerDecoder:
        case NodeType::PositionalEncoding:
            return IM_COL32(103, 58, 183, 255);

        // ===== Activation Functions - Orange/Yellow =====
        case NodeType::ReLU:
            return IM_COL32(243, 156, 18, 255);
        case NodeType::Sigmoid:
            return IM_COL32(241, 196, 15, 255);
        case NodeType::Tanh:
            return IM_COL32(230, 126, 34, 255);
        case NodeType::Softmax:
            return IM_COL32(211, 84, 0, 255);
        case NodeType::LeakyReLU:
        case NodeType::PReLU:
        case NodeType::ELU:
        case NodeType::SELU:
        case NodeType::GELU:
        case NodeType::Swish:
        case NodeType::Mish:
            return IM_COL32(235, 152, 78, 255);

        // ===== Shape Operations - Turquoise =====
        case NodeType::Reshape:
        case NodeType::Permute:
        case NodeType::Squeeze:
        case NodeType::Unsqueeze:
        case NodeType::View:
        case NodeType::Split:
        case NodeType::TensorBroadcastTo:
        case NodeType::TensorExpand:
        case NodeType::TensorIndexSelect:
            return IM_COL32(26, 188, 156, 255);

        // ===== Merge Operations - Lime Green =====
        case NodeType::Concatenate:
        case NodeType::Add:
        case NodeType::Multiply:
        case NodeType::Average:
            return IM_COL32(139, 195, 74, 255);

        // ===== Tensor Reductions - Purple =====
        case NodeType::TensorSum:
        case NodeType::TensorMean:
        case NodeType::TensorMax:
        case NodeType::TensorMin:
        case NodeType::TensorProd:
        case NodeType::TensorVar:
        case NodeType::TensorStd:
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
            return IM_COL32(156, 39, 176, 255);

        // ===== Loss Functions - Dark Red =====
        case NodeType::MSELoss:
        case NodeType::CrossEntropyLoss:
        case NodeType::BCELoss:
        case NodeType::BCEWithLogits:
        case NodeType::L1Loss:
        case NodeType::SmoothL1Loss:
        case NodeType::HuberLoss:
        case NodeType::NLLLoss:
        case NodeType::FocalLoss:
        case NodeType::SoftDiceLoss:
        case NodeType::TverskyLoss:
        case NodeType::JaccardLoss:
            return IM_COL32(192, 57, 43, 255);

        // ===== Optimizers - Dark Blue Gray =====
        case NodeType::SGD:
        case NodeType::Adam:
        case NodeType::AdamW:
        case NodeType::RMSprop:
        case NodeType::Adagrad:
        case NodeType::NAdam:
            return IM_COL32(52, 73, 94, 255);

        // ===== Learning Rate Schedulers - Steel Blue =====
        case NodeType::StepLR:
        case NodeType::CosineAnnealing:
        case NodeType::ReduceOnPlateau:
        case NodeType::ExponentialLR:
        case NodeType::WarmupScheduler:
            return IM_COL32(96, 125, 139, 255);

        // ===== Regularization Nodes - Magenta/Pink =====
        case NodeType::L1Regularization:
        case NodeType::L2Regularization:
        case NodeType::ElasticNet:
            return IM_COL32(233, 30, 99, 255);

        // ===== Utility Nodes - Gray =====
        case NodeType::Lambda:
        case NodeType::Identity:
        case NodeType::Constant:
        case NodeType::Parameter:
            return IM_COL32(158, 158, 158, 255);

        // ===== Signal / Control - Teal =====
        case NodeType::SignalSlider:
            return IM_COL32(0, 150, 136, 255);
        case NodeType::SineWave:
            return IM_COL32(38, 166, 154, 255);
        case NodeType::StepSignal:
            return IM_COL32(77, 182, 172, 255);
        case NodeType::RampSignal:
            return IM_COL32(77, 182, 172, 255);
        case NodeType::SignalScope:
            return IM_COL32(0, 121, 107, 255);

        // ===== Data Pipeline - Cyan =====
        case NodeType::DatasetInput:
            return IM_COL32(0, 188, 212, 255);
        case NodeType::DataLoader:
            return IM_COL32(0, 172, 193, 255);
        case NodeType::Augmentation:
            return IM_COL32(0, 151, 167, 255);
        case NodeType::DataSplit:
            return IM_COL32(38, 198, 218, 255);
        case NodeType::TensorReshape:
            return IM_COL32(77, 208, 225, 255);
        case NodeType::Normalize:
            return IM_COL32(128, 222, 234, 255);
        case NodeType::OneHotEncode:
            return IM_COL32(0, 131, 143, 255);

        // ===== Text Processing - Teal =====
        case NodeType::TextTokenizer:
            return IM_COL32(0, 150, 136, 255);
        case NodeType::TextVocabulary:
            return IM_COL32(38, 166, 154, 255);
        case NodeType::TextPadding:
            return IM_COL32(77, 182, 172, 255);
        case NodeType::NERSequenceBuilder:
            return IM_COL32(0, 137, 123, 255);
        case NodeType::TokenVocabulary:
        case NodeType::POSVocabulary:
        case NodeType::NERTagVocabulary:
        case NodeType::SequenceTagOutput:
            return IM_COL32(0, 150, 136, 255);

        // ===== Upsampling - Indigo =====
        case NodeType::ConvTranspose2D:
            return IM_COL32(92, 107, 192, 255);
        case NodeType::Upsample:
            return IM_COL32(121, 134, 203, 255);
        case NodeType::PixelShuffle:
            return IM_COL32(159, 168, 218, 255);

        // ===== Time-Series - Amber =====
        case NodeType::TimeSeriesSegment:
            return IM_COL32(255, 145, 0, 255);
        case NodeType::TimeSeriesWindow:
            return IM_COL32(255, 160, 0, 255);
        case NodeType::TimeSeriesFeatures:
            return IM_COL32(255, 179, 0, 255);
        case NodeType::TimeSeriesSplit:
            return IM_COL32(255, 196, 0, 255);
        case NodeType::SeasonalNaive:
            return IM_COL32(255, 183, 32, 255);
        case NodeType::LogTransform:
            return IM_COL32(255, 213, 0, 255);
        case NodeType::Differencing:
            return IM_COL32(255, 230, 0, 255);
        case NodeType::TimeSeriesDecomposition:
        case NodeType::ACFNode:
        case NodeType::PACFNode:
        case NodeType::StationarityTest:
        case NodeType::SeasonalityDetector:
        case NodeType::ARIMAForecaster:
        case NodeType::ExponentialSmoothing:
            return IM_COL32(255, 179, 64, 255);

        // ===== Audio - Deep Purple =====
        case NodeType::AudioInput:
            return IM_COL32(103, 58, 183, 255);
        case NodeType::Spectrogram:
        case NodeType::MelSpectrogram:
            return IM_COL32(126, 87, 194, 255);
        case NodeType::MFCC:
            return IM_COL32(149, 117, 205, 255);
        case NodeType::AudioAugmentation:
            return IM_COL32(179, 157, 219, 255);

        // ===== RL - Red =====
        case NodeType::GymEnvironment:
            return IM_COL32(229, 57, 53, 255);
        case NodeType::ReplayBufferNode:
            return IM_COL32(239, 83, 80, 255);
        case NodeType::PolicyNetwork:
        case NodeType::ValueNetwork:
            return IM_COL32(244, 113, 108, 255);
        case NodeType::RLTraining:
            return IM_COL32(198, 40, 40, 255);

        // ===== DNN Inference Nodes - Deep Blue =====
        case NodeType::DNNModelLoad:
        case NodeType::DNNDetect:
        case NodeType::DNNClassify:
        case NodeType::DNNPoseEstimate:
        case NodeType::DNNFaceDetect:
        case NodeType::DNNPreprocess:
        case NodeType::PretrainedYOLO:
        case NodeType::PretrainedMobileNet:
        case NodeType::PretrainedOpenPose:
        case NodeType::PretrainedFaceNet:
            return IM_COL32(41, 98, 255, 255);

        // ===== Post-processing Nodes - Steel Blue =====
        case NodeType::NonMaxSuppression:
        case NodeType::ArgMax:
        case NodeType::TopK:
        case NodeType::ThresholdFilter:
            return IM_COL32(70, 130, 180, 255);

        // ===== Smart I/O Nodes - Bright Blue =====
        case NodeType::DataInput:
        case NodeType::DataOutput:
        case NodeType::DataConvert:
            return IM_COL32(30, 136, 229, 255);  // Material Blue 600

        // ===== Legacy Data Source Nodes - Light Blue =====
        case NodeType::CSVFile:
        case NodeType::SQLQuery:
        case NodeType::HDF5Dataset:
        case NodeType::ParquetFile:
        case NodeType::JSONFile:
        case NodeType::ExcelFile:
        case NodeType::RESTAPISource:
            return IM_COL32(100, 181, 246, 255);

        // ===== Data Transform Nodes - Teal =====
        case NodeType::FilterRows:
        case NodeType::SelectColumns:
        case NodeType::JoinTables:
        case NodeType::GroupByAggregate:
        case NodeType::SortRows:
        case NodeType::FillMissingValues:
        case NodeType::RemoveDuplicateRows:
        case NodeType::PivotTable:
        case NodeType::UnionTables:
        case NodeType::RenameColumns:
            return IM_COL32(38, 166, 154, 255);

        // ===== Analytics Nodes - Purple =====
        case NodeType::DescribeStats:
        case NodeType::VisualizeData:
        case NodeType::SampleRows:
        case NodeType::CorrelationMatrix:
        case NodeType::ValueCounts:
        case NodeType::CrossTabulation:
            return IM_COL32(156, 39, 176, 255);

        // ===== Data Export Nodes - Green =====
        case NodeType::ExportCSV:
        case NodeType::ExportParquet:
        case NodeType::ExportSQL:
        case NodeType::ExportJSON:
        case NodeType::ExportExcel:
            return IM_COL32(76, 175, 80, 255);

        // ===== KNIME-Style Table Nodes - Orange =====
        case NodeType::RowToColumnNames:
        case NodeType::TableSplitter:
        case NodeType::CellExtractor:
        case NodeType::CellUpdater:
        case NodeType::TableCropper:
        case NodeType::ColumnAppender:
        case NodeType::RowAppender:
        case NodeType::Unpivot:
        case NodeType::StringManipulation:
        case NodeType::MathFormula:
        case NodeType::RuleEngine:
            return IM_COL32(255, 152, 0, 255);

        // ===== ML Clustering Nodes - Indigo =====
        case NodeType::KMeansCluster:
        case NodeType::DBSCANCluster:
        case NodeType::HierarchicalCluster:
        case NodeType::GMMCluster:
            return IM_COL32(63, 81, 181, 255);

        // ===== Dimensionality Reduction - Blue =====
        case NodeType::PCANode:
        case NodeType::TSNENode:
        case NodeType::UMAPNode:
            return IM_COL32(33, 150, 243, 255);

        // ===== ML Classification Nodes - Blue Green =====
        case NodeType::DecisionTreeClassifier:
        case NodeType::RandomForestClassifier:
        case NodeType::GradientBoostingClassifier:
        case NodeType::TreeModelPredictor:
        case NodeType::SVMClassifier:
        case NodeType::KNNClassifier:
        case NodeType::NaiveBayesClassifier:
        case NodeType::LogisticRegressionNode:
            return IM_COL32(0, 137, 123, 255);

        // ===== ML Regression Nodes - Light Teal =====
        case NodeType::LinearRegressionNode:
        case NodeType::PolynomialRegressionNode:
        case NodeType::RegressionModelPredictor:
        case NodeType::SVMRegressor:
            return IM_COL32(77, 182, 172, 255);

        // ===== Model Evaluation Nodes - Pink =====
        case NodeType::ConfusionMatrixNode:
        case NodeType::ROCCurveNode:
        case NodeType::PRCurveNode:
        case NodeType::LearningCurvesNode:
        case NodeType::FeatureImportanceNode:
        case NodeType::CrossValidationNode:
        case NodeType::RegressionMetricsNode:
        case NodeType::ClassificationMetricsNode:
            return IM_COL32(233, 30, 99, 255);

        // ===== Data Preprocessing Nodes - Light Green =====
        case NodeType::StandardScaler:
        case NodeType::MinMaxScaler:
        case NodeType::RobustScaler:
        case NodeType::LabelEncoder:
        case NodeType::OrdinalEncoder:
        case NodeType::TargetEncoder:
            return IM_COL32(129, 199, 132, 255);

        // ===== Advanced Preprocessing Nodes - Amber =====
        case NodeType::OutlierDetector:
        case NodeType::ImagePreprocessor:
        case NodeType::QualityAnalyzer:
        case NodeType::DataValidator:
            return IM_COL32(255, 193, 7, 255);

        // ===== Dataset Source Nodes - Cyan =====
        case NodeType::ImageFolderDataset:
        case NodeType::MNISTDataset:
        case NodeType::CIFAR10Dataset:
        case NodeType::HuggingFaceDataset:
        case NodeType::KaggleDataset:
            return IM_COL32(0, 188, 212, 255);

        // ===== Advanced Augmentation Nodes - Light Purple =====
        case NodeType::AugmentationPreset:
        case NodeType::GeometricTransform:
        case NodeType::ColorTransform:
        case NodeType::MorphologyTransform:
        case NodeType::AdvancedAugment:
            return IM_COL32(179, 136, 255, 255);

        // ===== Signal Processing Nodes - Deep Blue =====
        case NodeType::FFTNode:
        case NodeType::IFFTNode:
        case NodeType::FilterDesigner:
        case NodeType::Convolution1D:
        case NodeType::WaveletTransform:
            return IM_COL32(48, 63, 159, 255);

        // ===== Text Analytics Nodes - Dark Teal =====
        case NodeType::TFIDFVectorizer:
        case NodeType::CountVectorizer:
        case NodeType::WordEmbeddings:
        case NodeType::SentimentAnalyzer:
        case NodeType::NamedEntityRecognizer:
            return IM_COL32(0, 121, 107, 255);

        // ===== Utility Nodes - Blue Gray =====
        case NodeType::CalculatorNode:
        case NodeType::UnitConverter:
        case NodeType::RegexTester:
        case NodeType::JSONPathExtractor:
        case NodeType::DataProfiler:
            return IM_COL32(96, 125, 139, 255);

        // ===== Composite Nodes - Dark Cyan =====
        case NodeType::Subgraph:
            return IM_COL32(0, 131, 143, 255);

        case NodeType::PluginCustom:
            return IM_COL32(68, 136, 170, 255);

        default:
            return IM_COL32(127, 140, 141, 255);
    }
}
#endif

} // namespace gui
