#include "node_editor.h"
#include "properties.h"
#include "visualization/visualization_nodes.h"
#include "../core/worker_defaults.h"
#include "../plugin/registries/plugin_node_registry.h"
#include "../core/data_registry.h"
#include <imgui.h>
#include <imnodes.h>
#include <spdlog/spdlog.h>
#include <algorithm>

namespace gui {

// Unified Canvas Phase 1: Map NodeType to NodeCategory for UI organization
NodeCategory NodeEditor::GetCategoryForNodeType(NodeType type) {
    switch (type) {
        // Smart I/O Nodes (Universal)
        case NodeType::DataInput:
        case NodeType::DataOutput:
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
            return NodeCategory::TextProcessing;

        // Upsampling
        case NodeType::ConvTranspose2D:
        case NodeType::Upsample:
        case NodeType::PixelShuffle:
            return NodeCategory::Upsampling;

        // Time Series
        case NodeType::TimeSeriesWindow:
        case NodeType::TimeSeriesFeatures:
        case NodeType::TimeSeriesSplit:
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

void NodeEditor::AddNode(NodeType type, const std::string& name) {
    // Queue the node for deferred addition (after ImNodes::EndNodeEditor())
    pending_nodes_.push_back({type, name, context_menu_pos_});
    ClearValidationState();  // Graph changed — stale compile results
    spdlog::info("Queued node for addition: type={}, name={} at position x={} y={}",
                 static_cast<int>(type), name, context_menu_pos_.x, context_menu_pos_.y);
}

MLNode NodeEditor::CreateNode(NodeType type, const std::string& name) {
    MLNode node;
    node.id = next_node_id_++;
    node.type = type;
    node.category = GetCategoryForNodeType(type);  // Unified Canvas Phase 1
    node.name = name;

    // Create pins based on node type
    switch (type) {
        case NodeType::Dense: {
            // Dense layer has input and output
            NodePin input_pin;
            input_pin.id = next_pin_id_++;
            input_pin.type = PinType::Tensor;
            input_pin.name = "Input";
            input_pin.is_input = true;
            input_pin.description =
                "Input features. Expects shape [batch, in_features] — "
                "if upstream produces a higher-rank tensor, drop a "
                "Flatten before this node.";
            node.inputs.push_back(input_pin);

            NodePin output_pin;
            output_pin.id = next_pin_id_++;
            output_pin.type = PinType::Tensor;
            output_pin.name = "Output";
            output_pin.is_input = false;
            output_pin.description =
                "Linear projection of shape [batch, units]. Feeds an "
                "activation, the next Dense layer, or the Output node.";
            node.outputs.push_back(output_pin);

            // Extract units from name (e.g., "Dense (128)")
            size_t start = name.find('(');
            size_t end = name.find(')');
            if (start != std::string::npos && end != std::string::npos) {
                node.parameters["units"] = name.substr(start + 1, end - start - 1);
            } else {
                node.parameters["units"] = "128";
            }
            break;
        }

        case NodeType::ReLU:
        case NodeType::Sigmoid:
        case NodeType::Tanh:
        case NodeType::Softmax:
        case NodeType::LeakyReLU:
        case NodeType::PReLU:
        case NodeType::ELU:
        case NodeType::SELU:
        case NodeType::GELU:
        case NodeType::Swish:
        case NodeType::Mish: {
            // Activation functions
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

            // PReLU and LeakyReLU have a negative slope parameter
            if (node.type == NodeType::LeakyReLU) {
                node.parameters["negative_slope"] = "0.01";
            } else if (node.type == NodeType::PReLU) {
                node.parameters["num_parameters"] = "1";
                node.parameters["init"] = "0.25";
            } else if (node.type == NodeType::ELU) {
                node.parameters["alpha"] = "1.0";
            }
            break;
        }

        case NodeType::Output: {
            // Output node - final layer that produces predictions
            // Input: From previous layer
            NodePin input_pin;
            input_pin.id = next_pin_id_++;
            input_pin.type = PinType::Tensor;
            input_pin.name = "Input";
            input_pin.is_input = true;
            input_pin.description =
                "Logits / final activations from the model head — usually "
                "the last Dense layer's output of shape [batch, classes] "
                "(or [batch, 1] for regression).";
            node.inputs.push_back(input_pin);

            // Output: Predictions (goes to Loss function)
            NodePin output_pin;
            output_pin.id = next_pin_id_++;
            output_pin.type = PinType::Tensor;
            output_pin.name = "Predictions";
            output_pin.is_input = false;
            output_pin.is_required = false;  // Output is typically terminal; Predictions rarely chained further.
            output_pin.description =
                "Model predictions for the current batch. Connect to the "
                "loss node's Predictions pin so the loss can compare "
                "against DataLoader.Labels.";
            node.outputs.push_back(output_pin);

            node.parameters["classes"] = "10";
            break;
        }

        case NodeType::Conv1D:
        case NodeType::Conv2D:
        case NodeType::Conv3D:
        case NodeType::DepthwiseConv2D: {
            // Convolutional layers
            NodePin input_pin;
            input_pin.id = next_pin_id_++;
            input_pin.type = PinType::Tensor;
            input_pin.name = "Input";
            input_pin.is_input = true;
            input_pin.description =
                "Input feature map. Conv2D expects [batch, channels, "
                "height, width]; Conv1D expects [batch, channels, "
                "length]; Conv3D expects [batch, channels, depth, "
                "height, width].";
            node.inputs.push_back(input_pin);

            NodePin output_pin;
            output_pin.id = next_pin_id_++;
            output_pin.type = PinType::Tensor;
            output_pin.name = "Output";
            output_pin.is_input = false;
            output_pin.description =
                "Convolved feature map with `filters` channels. Spatial "
                "dimensions depend on kernel_size, stride, and padding "
                "('same' preserves them, 'valid' shrinks).";
            node.outputs.push_back(output_pin);

            // Initialize default parameters
            node.parameters["filters"] = "32";
            node.parameters["kernel_size"] = "3";
            node.parameters["stride"] = "1";
            node.parameters["padding"] = "same";
            node.parameters["activation"] = "relu";
            if (node.type == NodeType::DepthwiseConv2D) {
                node.parameters["depth_multiplier"] = "1";
            }
            break;
        }

        case NodeType::MaxPool2D:
        case NodeType::AvgPool2D: {
            // Pooling layers with size parameters
            NodePin input_pin;
            input_pin.id = next_pin_id_++;
            input_pin.type = PinType::Tensor;
            input_pin.name = "Input";
            input_pin.is_input = true;
            input_pin.description =
                "Feature map of shape [batch, channels, H, W]. Pooling "
                "reduces the spatial dimensions while leaving channel "
                "count unchanged.";
            node.inputs.push_back(input_pin);

            NodePin output_pin;
            output_pin.id = next_pin_id_++;
            output_pin.type = PinType::Tensor;
            output_pin.name = "Output";
            output_pin.is_input = false;
            output_pin.description =
                "Spatially down-sampled feature map: each pool_size × "
                "pool_size window collapses to one value (max or mean).";
            node.outputs.push_back(output_pin);

            // Initialize default parameters
            node.parameters["pool_size"] = "2";
            node.parameters["stride"] = "2";
            break;
        }

        case NodeType::GlobalMaxPool:
        case NodeType::GlobalAvgPool:
        case NodeType::AdaptiveAvgPool: {
            // Global pooling layers
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

            // AdaptiveAvgPool has output size parameter
            if (node.type == NodeType::AdaptiveAvgPool) {
                node.parameters["output_size"] = "1";
            }
            break;
        }

        case NodeType::Flatten: {
            // Flatten layer
            NodePin input_pin;
            input_pin.id = next_pin_id_++;
            input_pin.type = PinType::Tensor;
            input_pin.name = "Input";
            input_pin.is_input = true;
            input_pin.description =
                "Higher-rank tensor [batch, ...]. Typically the output "
                "of the last Conv/Pool block before the Dense head.";
            node.inputs.push_back(input_pin);

            NodePin output_pin;
            output_pin.id = next_pin_id_++;
            output_pin.type = PinType::Tensor;
            output_pin.name = "Output";
            output_pin.is_input = false;
            output_pin.description =
                "2D tensor of shape [batch, prod(rest)] — all non-batch "
                "axes collapsed into one. Feed straight into a Dense "
                "layer.";
            node.outputs.push_back(output_pin);
            break;
        }

        case NodeType::Dropout: {
            // Dropout layer
            NodePin input_pin;
            input_pin.id = next_pin_id_++;
            input_pin.type = PinType::Tensor;
            input_pin.name = "Input";
            input_pin.is_input = true;
            input_pin.description =
                "Any tensor. Dropout randomly zeros activations during "
                "training only — eval/test passes are pass-through.";
            node.inputs.push_back(input_pin);

            NodePin output_pin;
            output_pin.id = next_pin_id_++;
            output_pin.type = PinType::Tensor;
            output_pin.name = "Output";
            output_pin.is_input = false;
            output_pin.description =
                "Same shape as Input. During training, surviving "
                "activations are scaled by 1/(1-rate) so the expected "
                "magnitude matches eval-time behavior.";
            node.outputs.push_back(output_pin);

            // Initialize default parameters
            node.parameters["rate"] = "0.5";
            break;
        }

        case NodeType::BatchNorm:
        case NodeType::LayerNorm:
        case NodeType::GroupNorm:
        case NodeType::InstanceNorm: {
            // Normalization layers
            NodePin input_pin;
            input_pin.id = next_pin_id_++;
            input_pin.type = PinType::Tensor;
            input_pin.name = "Input";
            input_pin.is_input = true;
            input_pin.description =
                "Activations to normalize. BatchNorm normalizes across "
                "the batch dimension (uses train-time running stats at "
                "eval); LayerNorm/InstanceNorm/GroupNorm normalize "
                "within each sample so train and eval behave the same.";
            node.inputs.push_back(input_pin);

            NodePin output_pin;
            output_pin.id = next_pin_id_++;
            output_pin.type = PinType::Tensor;
            output_pin.name = "Output";
            output_pin.is_input = false;
            output_pin.description =
                "Same shape as Input, normalized to (approximately) "
                "zero mean and unit variance, then affine-transformed "
                "by learnable gamma/beta.";
            node.outputs.push_back(output_pin);

            // Initialize parameters based on norm type
            node.parameters["epsilon"] = "1e-5";
            if (node.type == NodeType::BatchNorm) {
                node.parameters["momentum"] = "0.1";
            } else if (node.type == NodeType::LayerNorm) {
                node.parameters["normalized_shape"] = "256";
            } else if (node.type == NodeType::GroupNorm) {
                node.parameters["num_groups"] = "32";
                node.parameters["num_channels"] = "256";
            } else if (node.type == NodeType::InstanceNorm) {
                node.parameters["num_features"] = "64";
            }
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
            // DataLoader node — training-loop hyperparameters AND batching.
            // Sits between the data pipeline and the model: takes the raw
            // (data, labels) pair from upstream (DataSplit or DataInput),
            // batches them, and emits the batched (data, labels) pair to
            // the model and the loss function respectively.

            // Input 1: Data tensor stream (from DataSplit.TrainData or DataInput.Data)
            NodePin data_in;
            data_in.id = next_pin_id_++;
            data_in.type = PinType::Tensor;
            data_in.name = "Data";
            data_in.is_input = true;
            data_in.description =
                "Unbatched feature stream. Usually wired from "
                "DataSplit.Train Data, or directly from DataInput.Data "
                "if you skip the split.";
            node.inputs.push_back(data_in);

            // Input 2: Labels stream (from DataSplit.TrainLabels or DataInput.Labels)
            NodePin labels_in;
            labels_in.id = next_pin_id_++;
            labels_in.type = PinType::Labels;
            labels_in.name = "Labels";
            labels_in.is_input = true;
            labels_in.description =
                "Unbatched label stream, row-aligned with Data. Usually "
                "wired from DataSplit.Train Labels.";
            node.inputs.push_back(labels_in);

            // Output 1: Batched data tensor → first model layer
            NodePin data_out;
            data_out.id = next_pin_id_++;
            data_out.type = PinType::Tensor;
            data_out.name = "Data";
            data_out.is_input = false;
            data_out.description =
                "Batched feature tensor of shape [batch_size, ...]. "
                "Connect to the model's first layer (Embedding, Conv2D, "
                "Dense, ...). Reshuffled each epoch when shuffle=true.";
            node.outputs.push_back(data_out);

            // Output 2: Batched labels → loss function's Targets pin
            // Marked optional for the same reason as DataSplit.Train Labels:
            // every example graph bypasses this pin and wires labels
            // direct from DataInput to Loss. Tighten once the runtime
            // walks pins.
            NodePin labels_out;
            labels_out.id = next_pin_id_++;
            labels_out.type = PinType::Labels;
            labels_out.name = "Labels";
            labels_out.is_input = false;
            labels_out.is_required = false;
            labels_out.description =
                "Batched label tensor of shape [batch_size, ...], "
                "row-aligned with Data. Connect to the loss node's "
                "Targets pin.";
            node.outputs.push_back(labels_out);

            // Training-loop hyperparameters. DataLoader owns ALL the
            // "how do I iterate training data" knobs (epochs included).
            // Optimizer node owns gradient-update knobs (lr, momentum, betas).
            node.parameters["epochs"] = "10";
            node.parameters["batch_size"] = "32";
            node.parameters["grad_accum_steps"] = "1";  // simulate larger effective batch
            node.parameters["shuffle"] = "true";
            node.parameters["drop_last"] = "false";
            node.parameters["seed"] = "42";             // reproducibility
            node.parameters["num_workers"] = std::to_string(cyxwiz::GetDefaultNumWorkers());
            node.parameters["prefetch_factor"] = "2";
            node.parameters["pin_memory"] = "false";    // CUDA host→device speedup
            node.parameters["log_interval"] = "10";     // log every N batches
            node.parameters["validation_freq"] = "1";   // validate every N epochs
            node.parameters["save_best_checkpoint"] = "true";
            node.parameters["early_stopping_patience"] = "5";
            node.parameters["checkpoint_dir"] = "";
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
            // DataSplit node - train/val/test splitter
            // Input: Data tensor
            NodePin data_in;
            data_in.id = next_pin_id_++;
            data_in.type = PinType::Tensor;
            data_in.name = "Data";
            data_in.is_input = true;
            data_in.description =
                "Incoming feature stream (X) — usually wired from "
                "DataInput.Data or a Normalize/Preprocess node. Will be "
                "partitioned row-wise by the train/val/test ratios.";
            node.inputs.push_back(data_in);

            // Input: Labels tensor
            NodePin labels_in;
            labels_in.id = next_pin_id_++;
            labels_in.type = PinType::Labels;
            labels_in.name = "Labels";
            labels_in.is_input = true;
            labels_in.description =
                "Incoming label stream (y) — usually wired from "
                "DataInput.Labels. Partitioned with the same row indices "
                "as Data so each split's (X, y) pairs stay aligned.";
            node.inputs.push_back(labels_in);

            // Output: Train Data
            NodePin train_data;
            train_data.id = next_pin_id_++;
            train_data.type = PinType::Tensor;
            train_data.name = "Train Data";
            train_data.is_input = false;
            train_data.description =
                "Feature subset for training — train_ratio of the input "
                "rows. Connect to a DataLoader (or directly to the model "
                "for tiny in-memory datasets).";
            node.outputs.push_back(train_data);

            // Output: Train Labels
            // Marked optional because the existing convention (used by
            // every example graph) routes labels directly DataInput →
            // Loss.Targets, bypassing DataSplit/DataLoader. The runtime
            // pin-walking fix (tofix.md) will make this path
            // canonical; tighten is_required to true once labels are
            // required to flow through the split/loader for shuffling
            // alignment.
            NodePin train_labels;
            train_labels.id = next_pin_id_++;
            train_labels.type = PinType::Labels;
            train_labels.name = "Train Labels";
            train_labels.is_input = false;
            train_labels.is_required = false;
            train_labels.description =
                "Label subset for training, row-aligned with Train Data. "
                "Pair with the matching DataLoader or wire straight to "
                "the loss node's Targets pin.";
            node.outputs.push_back(train_labels);

            // Output: Val Data
            NodePin val_data;
            val_data.id = next_pin_id_++;
            val_data.type = PinType::Tensor;
            val_data.name = "Val Data";
            val_data.is_input = false;
            val_data.is_required = false;  // Optional — common to skip validation during early dev.
            val_data.description =
                "Feature subset for validation — val_ratio of the input "
                "rows. Used by the eval pass between training epochs (no "
                "weight updates).";
            node.outputs.push_back(val_data);

            // Output: Val Labels
            NodePin val_labels;
            val_labels.id = next_pin_id_++;
            val_labels.type = PinType::Labels;
            val_labels.name = "Val Labels";
            val_labels.is_input = false;
            val_labels.is_required = false;
            val_labels.description =
                "Label subset for validation, row-aligned with Val Data.";
            node.outputs.push_back(val_labels);

            // Output: Test Data
            NodePin test_data;
            test_data.id = next_pin_id_++;
            test_data.type = PinType::Tensor;
            test_data.name = "Test Data";
            test_data.is_input = false;
            test_data.is_required = false;  // Optional — held-out test is often skipped.
            test_data.description =
                "Feature subset for the held-out test pass — test_ratio "
                "of the input rows. Only touched after training completes.";
            node.outputs.push_back(test_data);

            // Output: Test Labels
            NodePin test_labels;
            test_labels.id = next_pin_id_++;
            test_labels.type = PinType::Labels;
            test_labels.name = "Test Labels";
            test_labels.is_input = false;
            test_labels.is_required = false;
            test_labels.description =
                "Label subset for the held-out test pass, row-aligned "
                "with Test Data.";
            node.outputs.push_back(test_labels);

            // Parameters
            node.parameters["train_ratio"] = "0.8";
            node.parameters["val_ratio"] = "0.1";
            node.parameters["test_ratio"] = "0.1";
            node.parameters["stratified"] = "true";
            node.parameters["seed"] = "42";
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
            // Normalize node
            NodePin input_pin;
            input_pin.id = next_pin_id_++;
            input_pin.type = PinType::Tensor;
            input_pin.name = "Input";
            input_pin.is_input = true;
            input_pin.description =
                "Feature stream to normalize. The mean/std parameters "
                "are applied as (x - mean) / std elementwise — pre-"
                "computed stats, NOT learned. Place between DataInput "
                "and DataSplit.";
            node.inputs.push_back(input_pin);

            NodePin output_pin;
            output_pin.id = next_pin_id_++;
            output_pin.type = PinType::Tensor;
            output_pin.name = "Output";
            output_pin.is_input = false;
            output_pin.description =
                "Same shape as Input, with each feature centered and "
                "scaled by the configured mean/std.";
            node.outputs.push_back(output_pin);

            node.parameters["mean"] = "0.0";
            node.parameters["std"] = "1.0";
            break;
        }

        case NodeType::OneHotEncode: {
            // OneHotEncode node
            NodePin input_pin;
            input_pin.id = next_pin_id_++;
            input_pin.type = PinType::Labels;
            input_pin.name = "Labels";
            input_pin.is_input = true;
            input_pin.description =
                "Integer class indices [batch] in the range "
                "[0, num_classes). Wire from DataInput.Labels (or "
                "DataLoader.Labels) when your loss expects one-hot "
                "instead of class indices.";
            node.inputs.push_back(input_pin);

            NodePin output_pin;
            output_pin.id = next_pin_id_++;
            output_pin.type = PinType::Tensor;
            output_pin.name = "OneHot";
            output_pin.is_input = false;
            output_pin.description =
                "Float tensor [batch, num_classes] with a single 1 per "
                "row at the index column. Note the type changes from "
                "Labels (orange) to Tensor (blue) — feed into MSE/BCE "
                "rather than CrossEntropy.";
            node.outputs.push_back(output_pin);

            node.parameters["num_classes"] = "10";
            break;
        }

        // ========== Loss Functions ==========

        case NodeType::MSELoss:
        case NodeType::CrossEntropyLoss: {
            // Loss function: takes predictions and targets, outputs loss value
            // Input 1: Predictions (from model output)
            NodePin pred_pin;
            pred_pin.id = next_pin_id_++;
            pred_pin.type = PinType::Tensor;
            pred_pin.name = "Predictions";
            pred_pin.is_input = true;
            pred_pin.description =
                "Model output for the current batch. Shape depends on the "
                "model head — e.g. [batch, num_classes] for a classifier "
                "or [batch, 1] for a regressor.";
            node.inputs.push_back(pred_pin);

            // Input 2: Targets (ground truth labels). Typed as Labels so the
            // green label-stream visibly terminates here. ValidateLink also
            // accepts Tensor → Labels via the "Tensor is universal" rule, so
            // older graphs wired with raw tensors still connect.
            NodePin target_pin;
            target_pin.id = next_pin_id_++;
            target_pin.type = PinType::Labels;
            target_pin.name = "Targets";
            target_pin.is_input = true;
            target_pin.description =
                "Ground-truth labels (y) for the current batch. Connect "
                "from DataLoader.Labels — the label stream that traveled "
                "the data path from DataInput. Row-aligned with "
                "Predictions.";
            node.inputs.push_back(target_pin);

            // Output: Loss value
            NodePin loss_pin;
            loss_pin.id = next_pin_id_++;
            loss_pin.type = PinType::Loss;
            loss_pin.name = "Loss";
            loss_pin.is_input = false;
            loss_pin.description =
                "Scalar loss value for the batch. Connect to an Optimizer "
                "node — backprop runs from here.";
            node.outputs.push_back(loss_pin);

            // Parameters
            if (node.type == NodeType::CrossEntropyLoss) {
                node.parameters["reduction"] = "mean";  // mean, sum, none
            }
            break;
        }

        // ========== Optimizers ==========

        case NodeType::SGD:
        case NodeType::Adam:
        case NodeType::AdamW:
        case NodeType::RMSprop:
        case NodeType::Adagrad:
        case NodeType::NAdam: {
            // Optimizer: takes loss and updates model parameters
            NodePin loss_pin;
            loss_pin.id = next_pin_id_++;
            loss_pin.type = PinType::Loss;
            loss_pin.name = "Loss";
            loss_pin.is_input = true;
            loss_pin.description =
                "Scalar loss tensor from a Loss node. Backprop runs "
                "from this value through the model, then this optimizer "
                "applies the update rule (SGD / Adam / AdamW / ...) to "
                "every learnable parameter.";
            node.inputs.push_back(loss_pin);

            NodePin state_pin;
            state_pin.id = next_pin_id_++;
            state_pin.type = PinType::Optimizer;
            state_pin.name = "State";
            state_pin.is_input = false;
            state_pin.is_required = false;  // Optional — graphs without an Output node are fine.
            state_pin.description =
                "Optimizer-state handle. Connect to the Output / "
                "training-control node to close the training loop.";
            node.outputs.push_back(state_pin);

            // Parameters based on optimizer type
            node.parameters["learning_rate"] = "0.001";
            if (node.type == NodeType::SGD) {
                node.parameters["learning_rate"] = "0.01";
                node.parameters["momentum"] = "0.9";
                node.parameters["weight_decay"] = "0.0";
            } else if (node.type == NodeType::Adam || node.type == NodeType::NAdam) {
                node.parameters["beta1"] = "0.9";
                node.parameters["beta2"] = "0.999";
                node.parameters["epsilon"] = "1e-8";
            } else if (node.type == NodeType::AdamW) {
                node.parameters["beta1"] = "0.9";
                node.parameters["beta2"] = "0.999";
                node.parameters["weight_decay"] = "0.01";
            } else if (node.type == NodeType::RMSprop) {
                node.parameters["alpha"] = "0.99";
                node.parameters["epsilon"] = "1e-8";
                node.parameters["momentum"] = "0.0";
            } else if (node.type == NodeType::Adagrad) {
                node.parameters["lr_decay"] = "0.0";
                node.parameters["epsilon"] = "1e-10";
            }
            break;
        }

        // ========== Recurrent Layers ==========

        case NodeType::RNN:
        case NodeType::LSTM:
        case NodeType::GRU: {
            // Recurrent layers
            NodePin input_pin;
            input_pin.id = next_pin_id_++;
            input_pin.type = PinType::Tensor;
            input_pin.name = "Input";
            input_pin.is_input = true;
            input_pin.description =
                "3D sequence tensor [batch, seq_len, features]. For "
                "text models this is usually the Embedding output; "
                "features must equal the recurrent layer's input_size "
                "(per-timestep, NOT seq_len * embed_dim).";
            node.inputs.push_back(input_pin);

            NodePin output_pin;
            output_pin.id = next_pin_id_++;
            output_pin.type = PinType::Tensor;
            output_pin.name = "Output";
            output_pin.is_input = false;
            output_pin.description =
                "If return_sequences=true: full sequence "
                "[batch, seq_len, hidden * num_directions]. "
                "Otherwise: last timestep [batch, hidden * "
                "num_directions] — the common 'feed into a Dense "
                "classifier' shape.";
            node.outputs.push_back(output_pin);

            NodePin hidden_pin;
            hidden_pin.id = next_pin_id_++;
            hidden_pin.type = PinType::Tensor;
            hidden_pin.name = "Hidden";
            hidden_pin.is_input = false;
            hidden_pin.is_required = false;  // Optional — most users want Output only.
            hidden_pin.description =
                "Final hidden state h_n of shape "
                "[num_layers * num_directions, batch, hidden]. Useful "
                "for stacking recurrent blocks or for seq2seq models; "
                "leave disconnected if you only want the Output.";
            node.outputs.push_back(hidden_pin);

            node.parameters["input_size"] = "256";
            node.parameters["hidden_size"] = "256";
            node.parameters["num_layers"] = "1";
            node.parameters["bidirectional"] = "false";
            node.parameters["dropout"] = "0.0";
            break;
        }

        case NodeType::Bidirectional:
        case NodeType::TimeDistributed: {
            // Wrapper layers
            NodePin input_pin;
            input_pin.id = next_pin_id_++;
            input_pin.type = PinType::Tensor;
            input_pin.name = "Input";
            input_pin.is_input = true;
            input_pin.description =
                "Sequence tensor [batch, seq_len, features]. "
                "Bidirectional wraps a recurrent layer to run "
                "forward+backward; TimeDistributed applies a layer "
                "independently at each timestep.";
            node.inputs.push_back(input_pin);

            NodePin output_pin;
            output_pin.id = next_pin_id_++;
            output_pin.type = PinType::Tensor;
            output_pin.name = "Output";
            output_pin.is_input = false;
            output_pin.description =
                "Bidirectional output is concatenated (or summed, per "
                "merge_mode) across directions — feature dim doubles "
                "with concat. TimeDistributed preserves [batch, "
                "seq_len, ...] but the inner layer transforms the "
                "feature dim.";
            node.outputs.push_back(output_pin);

            if (node.type == NodeType::Bidirectional) {
                node.parameters["merge_mode"] = "concat";
            }
            break;
        }

        case NodeType::Embedding: {
            // Embedding layer
            NodePin input_pin;
            input_pin.id = next_pin_id_++;
            input_pin.type = PinType::Tensor;
            input_pin.name = "Indices";
            input_pin.is_input = true;
            input_pin.description =
                "Integer token IDs of shape [batch, seq_len]. Each ID "
                "must be < num_embeddings; padding_idx is treated as "
                "a no-grad zero embedding.";
            node.inputs.push_back(input_pin);

            NodePin output_pin;
            output_pin.id = next_pin_id_++;
            output_pin.type = PinType::Tensor;
            output_pin.name = "Embeddings";
            output_pin.is_input = false;
            output_pin.description =
                "Dense vectors of shape [batch, seq_len, embedding_dim]. "
                "Feed into a recurrent layer (LSTM/GRU) for sequence "
                "models, or Flatten + Dense for bag-of-embeddings.";
            node.outputs.push_back(output_pin);

            node.parameters["num_embeddings"] = "10000";
            node.parameters["embedding_dim"] = "256";
            node.parameters["padding_idx"] = "-1";
            break;
        }

        // ========== Attention & Transformer ==========

        case NodeType::MultiHeadAttention:
        case NodeType::SelfAttention:
        case NodeType::CrossAttention: {
            // Attention layers with Q, K, V and optional Mask
            NodePin query_pin;
            query_pin.id = next_pin_id_++;
            query_pin.type = PinType::Tensor;
            query_pin.name = "Query";
            query_pin.is_input = true;
            query_pin.is_required = true;
            query_pin.description =
                "Query tensor [batch, q_len, embed_dim]. For "
                "SelfAttention wire the same upstream tensor into "
                "Query, Key, and Value. For CrossAttention this is "
                "the decoder's current state.";
            node.inputs.push_back(query_pin);

            NodePin key_pin;
            key_pin.id = next_pin_id_++;
            key_pin.type = PinType::Tensor;
            key_pin.name = "Key";
            key_pin.is_input = true;
            key_pin.is_required = true;
            key_pin.description =
                "Key tensor [batch, kv_len, embed_dim]. Determines "
                "what positions Query attends over.";
            node.inputs.push_back(key_pin);

            NodePin value_pin;
            value_pin.id = next_pin_id_++;
            value_pin.type = PinType::Tensor;
            value_pin.name = "Value";
            value_pin.is_input = true;
            value_pin.is_required = true;
            value_pin.description =
                "Value tensor [batch, kv_len, embed_dim] — what gets "
                "weighted-summed by the attention scores. Same kv_len "
                "as Key.";
            node.inputs.push_back(value_pin);

            // Optional attention mask (for padding/causal masks)
            NodePin mask_pin;
            mask_pin.id = next_pin_id_++;
            mask_pin.type = PinType::Tensor;
            mask_pin.name = "Mask";
            mask_pin.is_input = true;
            mask_pin.is_required = false;  // Optional
            mask_pin.is_variadic = false;
            mask_pin.description =
                "Optional. Boolean / additive mask blocking certain "
                "positions — common uses: causal mask for autoregressive "
                "decoding, padding mask to ignore PAD tokens.";
            node.inputs.push_back(mask_pin);

            // Output: Attended values
            NodePin output_pin;
            output_pin.id = next_pin_id_++;
            output_pin.type = PinType::Tensor;
            output_pin.name = "Output";
            output_pin.is_input = false;
            output_pin.description =
                "Attention output [batch, q_len, embed_dim]. Drop into "
                "the next layer of a Transformer block.";
            node.outputs.push_back(output_pin);

            // Optional output: Attention weights for visualization/debugging
            NodePin attn_weights_pin;
            attn_weights_pin.id = next_pin_id_++;
            attn_weights_pin.type = PinType::Tensor;
            attn_weights_pin.name = "Attn Weights";
            attn_weights_pin.is_input = false;
            attn_weights_pin.is_required = false;  // Optional — visualization only.
            attn_weights_pin.description =
                "Per-head attention weights [batch, num_heads, q_len, "
                "kv_len]. Useful for visualization / interpretability "
                "panels; leave disconnected if you don't need them.";
            node.outputs.push_back(attn_weights_pin);

            node.parameters["embed_dim"] = "512";
            node.parameters["num_heads"] = "8";
            node.parameters["dropout"] = "0.0";
            node.parameters["batch_first"] = "true";
            break;
        }

        case NodeType::LinearAttention: {
            // Linear attention (O(n) complexity) - Performer/Linear Transformer style
            NodePin query_pin;
            query_pin.id = next_pin_id_++;
            query_pin.type = PinType::Tensor;
            query_pin.name = "Query";
            query_pin.is_input = true;
            query_pin.is_required = true;
            node.inputs.push_back(query_pin);

            NodePin key_pin;
            key_pin.id = next_pin_id_++;
            key_pin.type = PinType::Tensor;
            key_pin.name = "Key";
            key_pin.is_input = true;
            key_pin.is_required = true;
            node.inputs.push_back(key_pin);

            NodePin value_pin;
            value_pin.id = next_pin_id_++;
            value_pin.type = PinType::Tensor;
            value_pin.name = "Value";
            value_pin.is_input = true;
            value_pin.is_required = true;
            node.inputs.push_back(value_pin);

            // Optional causal mask (for autoregressive)
            NodePin mask_pin;
            mask_pin.id = next_pin_id_++;
            mask_pin.type = PinType::Tensor;
            mask_pin.name = "Mask";
            mask_pin.is_input = true;
            mask_pin.is_required = false;
            node.inputs.push_back(mask_pin);

            NodePin output_pin;
            output_pin.id = next_pin_id_++;
            output_pin.type = PinType::Tensor;
            output_pin.name = "Output";
            output_pin.is_input = false;
            node.outputs.push_back(output_pin);

            node.parameters["embed_dim"] = "512";
            node.parameters["num_heads"] = "8";
            node.parameters["feature_map"] = "elu";  // elu, relu, favor+
            node.parameters["eps"] = "1e-6";
            node.parameters["causal"] = "false";
            break;
        }

        case NodeType::TransformerEncoder:
        case NodeType::TransformerDecoder: {
            // Transformer blocks
            NodePin input_pin;
            input_pin.id = next_pin_id_++;
            input_pin.type = PinType::Tensor;
            input_pin.name = "Input";
            input_pin.is_input = true;
            input_pin.description =
                "Sequence input [batch, seq_len, d_model]. Usually "
                "the output of an Embedding + PositionalEncoding pair. "
                "All `num_layers` encoder/decoder blocks run in series.";
            node.inputs.push_back(input_pin);

            if (node.type == NodeType::TransformerDecoder) {
                NodePin memory_pin;
                memory_pin.id = next_pin_id_++;
                memory_pin.type = PinType::Tensor;
                memory_pin.name = "Memory";
                memory_pin.is_input = true;
                memory_pin.description =
                    "Encoder output [batch, src_len, d_model] used as "
                    "the cross-attention key/value source. Required "
                    "for seq2seq models.";
                node.inputs.push_back(memory_pin);
            }

            NodePin output_pin;
            output_pin.id = next_pin_id_++;
            output_pin.type = PinType::Tensor;
            output_pin.name = "Output";
            output_pin.is_input = false;
            output_pin.description =
                "Same shape as Input. For classifier heads use only "
                "the [CLS] token; for seq2seq feed into the decoder's "
                "Memory pin (encoder) or into the projection head "
                "(decoder).";
            node.outputs.push_back(output_pin);

            node.parameters["d_model"] = "512";
            node.parameters["nhead"] = "8";
            node.parameters["num_layers"] = "6";
            node.parameters["dim_feedforward"] = "2048";
            node.parameters["dropout"] = "0.1";
            break;
        }

        case NodeType::PositionalEncoding: {
            NodePin input_pin;
            input_pin.id = next_pin_id_++;
            input_pin.type = PinType::Tensor;
            input_pin.name = "Input";
            input_pin.is_input = true;
            input_pin.description =
                "Embedded sequence [batch, seq_len, d_model] from an "
                "Embedding layer. PositionalEncoding adds sinusoidal "
                "position vectors so the Transformer can distinguish "
                "token order.";
            node.inputs.push_back(input_pin);

            NodePin output_pin;
            output_pin.id = next_pin_id_++;
            output_pin.type = PinType::Tensor;
            output_pin.name = "Output";
            output_pin.is_input = false;
            output_pin.description =
                "Same shape as Input, with positional vectors added in "
                "and a small dropout applied. Feed straight into a "
                "TransformerEncoder/Decoder.";
            node.outputs.push_back(output_pin);

            node.parameters["d_model"] = "512";
            node.parameters["max_len"] = "5000";
            node.parameters["dropout"] = "0.1";
            break;
        }

        // ========== Shape Operations ==========

        case NodeType::Reshape:
        case NodeType::View: {
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

            node.parameters["shape"] = "-1,256";
            break;
        }

        case NodeType::Permute: {
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

            node.parameters["dims"] = "0,2,1";
            break;
        }

        case NodeType::Squeeze:
        case NodeType::Unsqueeze: {
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

            node.parameters["dim"] = "0";
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
            // Multi-input merge operations with variadic support
            // Two inputs by default, but can accept more

            NodePin input1;
            input1.id = next_pin_id_++;
            input1.type = PinType::Tensor;
            input1.name = "Input 1";
            input1.is_input = true;
            input1.is_variadic = false;  // First input is always required
            input1.is_required = true;
            node.inputs.push_back(input1);

            NodePin input2;
            input2.id = next_pin_id_++;
            input2.type = PinType::Tensor;
            input2.name = "Input 2";
            input2.is_input = true;
            input2.is_variadic = false;  // Second input required for merge
            input2.is_required = true;
            node.inputs.push_back(input2);

            // Third input is optional/variadic for N-way merges
            NodePin input3;
            input3.id = next_pin_id_++;
            input3.type = PinType::Tensor;
            input3.name = "Input 3+";
            input3.is_input = true;
            input3.is_variadic = true;
            input3.is_required = false;
            input3.min_connections = 0;
            input3.max_connections = PIN_UNLIMITED;  // Accept any number
            node.inputs.push_back(input3);

            NodePin output_pin;
            output_pin.id = next_pin_id_++;
            output_pin.type = PinType::Tensor;
            output_pin.name = "Output";
            output_pin.is_input = false;
            node.outputs.push_back(output_pin);

            if (node.type == NodeType::Concatenate) {
                node.parameters["dim"] = "1";
            }
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

            node.parameters["dim"] = "-1";
            node.parameters["keepdim"] = "false";
            break;
        }

        case NodeType::TensorBroadcastTo:
        case NodeType::TensorExpand: {
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

            node.parameters["shape"] = "";
            break;
        }

        case NodeType::TensorIndexSelect: {
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

            node.parameters["dim"] = "0";
            node.parameters["indices"] = "";
            break;
        }

        case NodeType::TensorPow:
        case NodeType::TensorSqrt:
        case NodeType::TensorExp:
        case NodeType::TensorLog:
        case NodeType::TensorAbs:
        case NodeType::TensorSign:
        case NodeType::TensorClip: {
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

            if (node.type == NodeType::TensorPow) {
                node.parameters["exponent"] = "2.0";
            } else if (node.type == NodeType::TensorClip) {
                node.parameters["min"] = "0.0";
                node.parameters["max"] = "1.0";
            }
            break;
        }

        case NodeType::TensorDot:
        case NodeType::TensorBatchMatMul: {
            NodePin input_a;
            input_a.id = next_pin_id_++;
            input_a.type = PinType::Tensor;
            input_a.name = "A";
            input_a.is_input = true;
            node.inputs.push_back(input_a);

            NodePin input_b;
            input_b.id = next_pin_id_++;
            input_b.type = PinType::Tensor;
            input_b.name = "B";
            input_b.is_input = true;
            node.inputs.push_back(input_b);

            NodePin output_pin;
            output_pin.id = next_pin_id_++;
            output_pin.type = PinType::Tensor;
            output_pin.name = "Output";
            output_pin.is_input = false;
            node.outputs.push_back(output_pin);
            break;
        }

        case NodeType::TensorCompare:
        case NodeType::TensorLogicalMask: {
            NodePin input_a;
            input_a.id = next_pin_id_++;
            input_a.type = PinType::Tensor;
            input_a.name = "A";
            input_a.is_input = true;
            node.inputs.push_back(input_a);

            NodePin input_b;
            input_b.id = next_pin_id_++;
            input_b.type = PinType::Tensor;
            input_b.name = "B";
            input_b.is_input = true;
            input_b.is_required = false;
            node.inputs.push_back(input_b);

            NodePin output_pin;
            output_pin.id = next_pin_id_++;
            output_pin.type = PinType::Tensor;
            output_pin.name = "Mask";
            output_pin.is_input = false;
            node.outputs.push_back(output_pin);

            if (node.type == NodeType::TensorCompare) {
                node.parameters["op"] = ">";
                node.parameters["scalar"] = "0.0";
            } else {
                node.parameters["op"] = "not";
            }
            break;
        }

        // ========== Additional Loss Functions ==========

        case NodeType::BCELoss:
        case NodeType::BCEWithLogits:
        case NodeType::L1Loss:
        case NodeType::SmoothL1Loss:
        case NodeType::HuberLoss:
        case NodeType::NLLLoss: {
            NodePin pred_pin;
            pred_pin.id = next_pin_id_++;
            pred_pin.type = PinType::Tensor;
            pred_pin.name = "Predictions";
            pred_pin.is_input = true;
            pred_pin.description =
                "Model output. BCE/BCEWithLogits expect a single sigmoid "
                "logit/prob per sample; L1/SmoothL1/Huber expect a "
                "regression value matching Targets shape; NLL expects "
                "log-probabilities of shape [batch, classes].";
            node.inputs.push_back(pred_pin);

            NodePin target_pin;
            target_pin.id = next_pin_id_++;
            target_pin.type = PinType::Tensor;
            target_pin.name = "Targets";
            target_pin.is_input = true;
            target_pin.description =
                "Ground-truth values. BCE wants 0/1 floats; L1/Huber "
                "want continuous floats matching Predictions shape; NLL "
                "wants integer class indices.";
            node.inputs.push_back(target_pin);

            NodePin loss_pin;
            loss_pin.id = next_pin_id_++;
            loss_pin.type = PinType::Loss;
            loss_pin.name = "Loss";
            loss_pin.is_input = false;
            loss_pin.description =
                "Scalar loss for the batch (reduced via mean/sum/none "
                "per the reduction parameter). Connect to an Optimizer "
                "node to drive backprop.";
            node.outputs.push_back(loss_pin);

            node.parameters["reduction"] = "mean";
            if (node.type == NodeType::SmoothL1Loss || node.type == NodeType::HuberLoss) {
                node.parameters["beta"] = "1.0";
            }
            break;
        }

        // ========== Learning Rate Schedulers ==========

        case NodeType::StepLR:
        case NodeType::CosineAnnealing:
        case NodeType::ReduceOnPlateau:
        case NodeType::ExponentialLR:
        case NodeType::WarmupScheduler: {
            // Schedulers connect to optimizer
            NodePin optim_pin;
            optim_pin.id = next_pin_id_++;
            optim_pin.type = PinType::Optimizer;
            optim_pin.name = "Optimizer";
            optim_pin.is_input = true;
            node.inputs.push_back(optim_pin);

            NodePin output_pin;
            output_pin.id = next_pin_id_++;
            output_pin.type = PinType::Optimizer;
            output_pin.name = "Scheduled";
            output_pin.is_input = false;
            node.outputs.push_back(output_pin);

            if (node.type == NodeType::StepLR) {
                node.parameters["step_size"] = "10";
                node.parameters["gamma"] = "0.1";
            } else if (node.type == NodeType::CosineAnnealing) {
                node.parameters["T_max"] = "100";
                node.parameters["eta_min"] = "0.0";
            } else if (node.type == NodeType::ReduceOnPlateau) {
                node.parameters["mode"] = "min";
                node.parameters["factor"] = "0.1";
                node.parameters["patience"] = "10";
            } else if (node.type == NodeType::ExponentialLR) {
                node.parameters["gamma"] = "0.95";
            } else if (node.type == NodeType::WarmupScheduler) {
                node.parameters["warmup_steps"] = "1000";
                node.parameters["warmup_ratio"] = "0.1";
            }
            break;
        }

        // ========== Regularization ==========

        case NodeType::L1Regularization:
        case NodeType::L2Regularization:
        case NodeType::ElasticNet: {
            NodePin input_pin;
            input_pin.id = next_pin_id_++;
            input_pin.type = PinType::Parameters;
            input_pin.name = "Parameters";
            input_pin.is_input = true;
            node.inputs.push_back(input_pin);

            NodePin output_pin;
            output_pin.id = next_pin_id_++;
            output_pin.type = PinType::Loss;
            output_pin.name = "Penalty";
            output_pin.is_input = false;
            node.outputs.push_back(output_pin);

            node.parameters["lambda"] = "0.01";
            if (node.type == NodeType::ElasticNet) {
                node.parameters["l1_ratio"] = "0.5";
            }
            break;
        }

        // ========== Utility Nodes ==========

        case NodeType::Lambda: {
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

            node.parameters["function"] = "lambda x: x";
            break;
        }

        case NodeType::Identity: {
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
            break;
        }

        case NodeType::Constant: {
            NodePin output_pin;
            output_pin.id = next_pin_id_++;
            output_pin.type = PinType::Tensor;
            output_pin.name = "Value";
            output_pin.is_input = false;
            node.outputs.push_back(output_pin);

            node.parameters["value"] = "1.0";
            node.parameters["shape"] = "1";
            break;
        }

        case NodeType::SignalSlider: {
            NodePin out;
            out.id = next_pin_id_++;
            out.type = PinType::Tensor;
            out.name = "Value";
            out.is_input = false;
            node.outputs.push_back(out);

            node.parameters["min"] = "-1.0";
            node.parameters["max"] = "1.0";
            node.parameters["value"] = "0.0";
            node.parameters["label"] = name;
            break;
        }

        case NodeType::SineWave: {
            NodePin out;
            out.id = next_pin_id_++;
            out.type = PinType::Tensor;
            out.name = "Signal";
            out.is_input = false;
            node.outputs.push_back(out);

            node.parameters["amplitude"] = "1.0";
            node.parameters["frequency"] = "1.0";
            node.parameters["phase"] = "0.0";
            node.parameters["offset"] = "0.0";
            break;
        }

        case NodeType::StepSignal: {
            NodePin out;
            out.id = next_pin_id_++;
            out.type = PinType::Tensor;
            out.name = "Signal";
            out.is_input = false;
            node.outputs.push_back(out);

            node.parameters["step_time"] = "1.0";
            node.parameters["initial_value"] = "0.0";
            node.parameters["final_value"] = "1.0";
            break;
        }

        case NodeType::RampSignal: {
            NodePin out;
            out.id = next_pin_id_++;
            out.type = PinType::Tensor;
            out.name = "Signal";
            out.is_input = false;
            node.outputs.push_back(out);

            node.parameters["start_value"] = "0.0";
            node.parameters["end_value"] = "1.0";
            node.parameters["duration"] = "5.0";
            break;
        }

        case NodeType::SignalScope: {
            NodePin in;
            in.id = next_pin_id_++;
            in.type = PinType::Tensor;
            in.name = "Signal";
            in.is_input = true;
            in.is_variadic = true;
            in.max_connections = PIN_UNLIMITED;
            node.inputs.push_back(in);

            node.parameters["window_size"] = "500";
            node.parameters["auto_scale"] = "true";
            break;
        }

        case NodeType::Parameter: {
            NodePin output_pin;
            output_pin.id = next_pin_id_++;
            output_pin.type = PinType::Tensor;
            output_pin.name = "Parameter";
            output_pin.is_input = false;
            node.outputs.push_back(output_pin);

            node.parameters["shape"] = "256";
            node.parameters["init"] = "xavier";
            node.parameters["requires_grad"] = "true";
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
            NodePin in;
            in.id = next_pin_id_++;
            in.type = PinType::Tensor;
            in.name = "Text Data";
            in.is_input = true;
            node.inputs.push_back(in);
            NodePin out;
            out.id = next_pin_id_++;
            out.type = PinType::Tensor;
            out.name = "Token IDs";
            out.is_input = false;
            node.outputs.push_back(out);
            // Fix B canonical params read by TextTokenizerOperator.
            // text_col + label_col are required when this node runs as
            // a real Cat-1 operator on an Arrow source. The legacy
            // GraphCompiler config-extractor path reads
            // tokenizer_type/max_length/lowercase/min_word_freq/
            // max_vocab_size; both views coexist (extractor ignores
            // text_col/label_col, operator ignores padding/truncation
            // since it always pads + truncates).
            node.parameters["text_col"] = "";
            node.parameters["label_col"] = "";
            node.parameters["tokenizer_type"] = "1"; // 0=Whitespace, 1=Word, 2=Character
            node.parameters["max_length"] = "256";
            node.parameters["lowercase"] = "true";
            node.parameters["min_word_freq"] = "2";
            node.parameters["max_vocab_size"] = "10000";
            // Legacy fields for back-compat with the existing extractor.
            // Operator implicitly sets both to true regardless.
            node.parameters["padding"] = "true";
            node.parameters["truncation"] = "true";
            // Legacy alias — extractor reads `min_freq`, operator reads
            // `min_word_freq`; keep both pointing at the same value
            // until the extractor is deleted in a future commit.
            node.parameters["min_freq"] = "2";
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
            node.parameters["min_freq"] = "1";
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

        // ========== Upsampling Layers ==========

        case NodeType::ConvTranspose2D: {
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
            node.parameters["in_channels"] = "64";
            node.parameters["out_channels"] = "32";
            node.parameters["kernel_size"] = "3";
            node.parameters["stride"] = "2";
            node.parameters["padding"] = "1";
            node.parameters["output_padding"] = "1";
            break;
        }

        case NodeType::Upsample: {
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
            node.parameters["scale_factor"] = "2";
            node.parameters["mode"] = "0"; // 0=Nearest, 1=Bilinear
            break;
        }

        case NodeType::PixelShuffle: {
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
            node.parameters["upscale_factor"] = "2";
            break;
        }

        // ========== Time-Series Nodes ==========

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
                "y label column. Optional __window_start_time metadata "
                "is emitted when time_col is set.";
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
            NodePin out;
            out.id = next_pin_id_++;
            out.type = PinType::Tensor;
            out.name = "Waveform";
            out.is_input = false;
            node.outputs.push_back(out);
            NodePin labels_out;
            labels_out.id = next_pin_id_++;
            labels_out.type = PinType::Labels;
            labels_out.name = "Labels";
            labels_out.is_input = false;
            node.outputs.push_back(labels_out);
            node.parameters["sample_rate"] = "16000";
            node.parameters["duration_ms"] = "3000";
            node.parameters["dataset_path"] = "";
            break;
        }

        case NodeType::Spectrogram:
        case NodeType::MelSpectrogram: {
            NodePin in;
            in.id = next_pin_id_++;
            in.type = PinType::Tensor;
            in.name = "Waveform";
            in.is_input = true;
            node.inputs.push_back(in);
            NodePin out;
            out.id = next_pin_id_++;
            out.type = PinType::Tensor;
            out.name = "Spectrogram";
            out.is_input = false;
            node.outputs.push_back(out);
            node.parameters["n_fft"] = "512";
            node.parameters["hop_length"] = "256";
            if (type == NodeType::MelSpectrogram) {
                node.parameters["n_mels"] = "128";
            }
            node.parameters["log_scale"] = "true";
            break;
        }

        case NodeType::MFCC: {
            NodePin in;
            in.id = next_pin_id_++;
            in.type = PinType::Tensor;
            in.name = "Waveform";
            in.is_input = true;
            node.inputs.push_back(in);
            NodePin out;
            out.id = next_pin_id_++;
            out.type = PinType::Tensor;
            out.name = "MFCC";
            out.is_input = false;
            node.outputs.push_back(out);
            node.parameters["n_mfcc"] = "13";
            node.parameters["n_fft"] = "512";
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
            // Universal Data Input node - smart dialog auto-detects format
            // Supports: CSV, TSV, JSON, Parquet, Excel, HDF5, and more
            // TWO outputs: Data (X) and Labels (y) — same naming the rest
            // of the chain (DataSplit / DataLoader) uses, so the canvas
            // reads as a single Data + Label flow end-to-end.

            // Output 1: Data (the X tensor that feeds the model)
            NodePin data_pin;
            data_pin.id = next_pin_id_++;
            data_pin.type = PinType::Tensor;
            data_pin.name = "Data";
            data_pin.is_input = false;
            data_pin.description =
                "Feature stream (X). Carries every column from the loaded "
                "dataset that is NOT marked as the label column. Connect to "
                "the next node in the data pipeline (Normalize, DataSplit, "
                "DataLoader, ...) — eventually feeds the model's first layer.";
            node.outputs.push_back(data_pin);

            // Output 2: Labels (targets for loss function)
            NodePin labels_pin;
            labels_pin.id = next_pin_id_++;
            labels_pin.type = PinType::Labels;
            labels_pin.name = "Labels";
            labels_pin.is_input = false;
            labels_pin.description =
                "Label stream (y). Carries the column selected as 'Label "
                "Column' in the DataInput dialog. Travels alongside Data "
                "through DataSplit and DataLoader and terminates at the "
                "loss function's Targets pin.";
            node.outputs.push_back(labels_pin);

            // Core parameters (set by DataInputDialog)
            node.parameters["file_path"] = "";
            node.parameters["file_type"] = "auto";  // auto, csv, tsv, json, parquet, excel, hdf5
            node.parameters["configured"] = "false";  // Triggers dialog on first use

            // Format options (dynamically shown based on file_type)
            node.parameters["delimiter"] = ",";
            node.parameters["header"] = "true";
            node.parameters["sheet_name"] = "";
            node.parameters["hdf5_key"] = "";

            // Column selection
            node.parameters["columns"] = "*";  // * = all, or comma-separated list
            node.parameters["label_column"] = "";  // Target column name (e.g., "label", "class", "target")
            node.parameters["feature_columns"] = "*";  // Feature columns (* = all except label)

            // Row filtering
            node.parameters["skip_rows"] = "0";
            node.parameters["max_rows"] = "";  // empty = all
            node.parameters["where_clause"] = "";

            // Streaming settings (legacy defaults; streaming path is being removed)
            node.parameters["chunk_size"] = "10000";
            node.parameters["enable_streaming"] = "false";
            break;
        }

        case NodeType::DataOutput: {
            // Universal Data Output node - smart dialog for export
            // Supports: CSV, Parquet
            NodePin input_pin;
            input_pin.id = next_pin_id_++;
            input_pin.type = PinType::Dataset;
            input_pin.name = "Data";
            input_pin.is_input = true;
            node.inputs.push_back(input_pin);

            // Core parameters (set by DataOutputDialog)
            node.parameters["file_path"] = "";
            node.parameters["file_type"] = "csv";  // csv, parquet
            node.parameters["configured"] = "false";  // Triggers dialog on first use

            // Export options
            node.parameters["overwrite"] = "true";
            node.parameters["include_header"] = "true";
            node.parameters["compression"] = "none";  // none, gzip, snappy, zstd
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

        case NodeType::FilterRows: {
            // Filter Rows transformation node - SQL WHERE clause
            NodePin input_pin;
            input_pin.id = next_pin_id_++;
            input_pin.type = PinType::Dataset;
            input_pin.name = "Input";
            input_pin.is_input = true;
            node.inputs.push_back(input_pin);

            NodePin output_pin;
            output_pin.id = next_pin_id_++;
            output_pin.type = PinType::Dataset;
            output_pin.name = "Output";
            output_pin.is_input = false;
            node.outputs.push_back(output_pin);

            node.parameters["condition"] = "column > 0";
            break;
        }

        case NodeType::SelectColumns: {
            // Select Columns transformation node - choose columns to keep
            NodePin input_pin;
            input_pin.id = next_pin_id_++;
            input_pin.type = PinType::Dataset;
            input_pin.name = "Input";
            input_pin.is_input = true;
            node.inputs.push_back(input_pin);

            NodePin output_pin;
            output_pin.id = next_pin_id_++;
            output_pin.type = PinType::Dataset;
            output_pin.name = "Output";
            output_pin.is_input = false;
            node.outputs.push_back(output_pin);

            node.parameters["columns"] = "col1, col2, col3";
            break;
        }

        case NodeType::DescribeStats: {
            // Describe Statistics analytics node - computes summary stats
            NodePin input_pin;
            input_pin.id = next_pin_id_++;
            input_pin.type = PinType::Dataset;
            input_pin.name = "Input";
            input_pin.is_input = true;
            node.inputs.push_back(input_pin);

            // No output pin - analytics node displays results in properties panel
            node.parameters["show_percentiles"] = "true";
            break;
        }

        case NodeType::JoinTables: {
            // Join Tables transformation node - SQL JOIN
            NodePin left_input_pin;
            left_input_pin.id = next_pin_id_++;
            left_input_pin.type = PinType::Dataset;
            left_input_pin.name = "Left";
            left_input_pin.is_input = true;
            node.inputs.push_back(left_input_pin);

            NodePin right_input_pin;
            right_input_pin.id = next_pin_id_++;
            right_input_pin.type = PinType::Dataset;
            right_input_pin.name = "Right";
            right_input_pin.is_input = true;
            node.inputs.push_back(right_input_pin);

            NodePin output_pin;
            output_pin.id = next_pin_id_++;
            output_pin.type = PinType::Dataset;
            output_pin.name = "Output";
            output_pin.is_input = false;
            node.outputs.push_back(output_pin);

            node.parameters["join_type"] = "inner";
            node.parameters["left_on"] = "id";
            node.parameters["right_on"] = "id";
            break;
        }

        case NodeType::SortRows: {
            // Sort Rows transformation node - ORDER BY
            NodePin input_pin;
            input_pin.id = next_pin_id_++;
            input_pin.type = PinType::Dataset;
            input_pin.name = "Input";
            input_pin.is_input = true;
            node.inputs.push_back(input_pin);

            NodePin output_pin;
            output_pin.id = next_pin_id_++;
            output_pin.type = PinType::Dataset;
            output_pin.name = "Output";
            output_pin.is_input = false;
            node.outputs.push_back(output_pin);

            node.parameters["columns"] = "column1";
            node.parameters["ascending"] = "true";
            break;
        }

        case NodeType::GroupByAggregate: {
            // Group By Aggregate transformation node - SQL GROUP BY
            NodePin input_pin;
            input_pin.id = next_pin_id_++;
            input_pin.type = PinType::Dataset;
            input_pin.name = "Input";
            input_pin.is_input = true;
            node.inputs.push_back(input_pin);

            NodePin output_pin;
            output_pin.id = next_pin_id_++;
            output_pin.type = PinType::Dataset;
            output_pin.name = "Output";
            output_pin.is_input = false;
            node.outputs.push_back(output_pin);

            node.parameters["group_by"] = "column1";
            node.parameters["aggregations"] = "COUNT(*) as count";
            break;
        }

        case NodeType::FillMissingValues: {
            // Fill Missing Values transformation node - handle NULLs
            NodePin input_pin;
            input_pin.id = next_pin_id_++;
            input_pin.type = PinType::Dataset;
            input_pin.name = "Input";
            input_pin.is_input = true;
            node.inputs.push_back(input_pin);

            NodePin output_pin;
            output_pin.id = next_pin_id_++;
            output_pin.type = PinType::Dataset;
            output_pin.name = "Output";
            output_pin.is_input = false;
            node.outputs.push_back(output_pin);

            node.parameters["strategy"] = "mean";  // mean, median, mode, constant
            node.parameters["fill_value"] = "0";
            break;
        }

        case NodeType::RemoveDuplicateRows: {
            // Remove Duplicate Rows transformation node - SQL DISTINCT
            NodePin input_pin;
            input_pin.id = next_pin_id_++;
            input_pin.type = PinType::Dataset;
            input_pin.name = "Input";
            input_pin.is_input = true;
            node.inputs.push_back(input_pin);

            NodePin output_pin;
            output_pin.id = next_pin_id_++;
            output_pin.type = PinType::Dataset;
            output_pin.name = "Output";
            output_pin.is_input = false;
            node.outputs.push_back(output_pin);

            node.parameters["subset"] = "";  // Empty = all columns
            node.parameters["keep"] = "first";  // first, last, none
            break;
        }

        case NodeType::RenameColumns: {
            // Rename Columns transformation node
            NodePin input_pin;
            input_pin.id = next_pin_id_++;
            input_pin.type = PinType::Dataset;
            input_pin.name = "Input";
            input_pin.is_input = true;
            node.inputs.push_back(input_pin);

            NodePin output_pin;
            output_pin.id = next_pin_id_++;
            output_pin.type = PinType::Dataset;
            output_pin.name = "Output";
            output_pin.is_input = false;
            node.outputs.push_back(output_pin);

            node.parameters["mapping"] = "old_name:new_name";
            break;
        }

        case NodeType::SampleRows: {
            // Sample Rows analytics/transform node - random sampling
            NodePin input_pin;
            input_pin.id = next_pin_id_++;
            input_pin.type = PinType::Dataset;
            input_pin.name = "Input";
            input_pin.is_input = true;
            node.inputs.push_back(input_pin);

            NodePin output_pin;
            output_pin.id = next_pin_id_++;
            output_pin.type = PinType::Dataset;
            output_pin.name = "Output";
            output_pin.is_input = false;
            node.outputs.push_back(output_pin);

            node.parameters["n"] = "100";
            node.parameters["random_state"] = "42";
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
            // Export nodes - save dataset to file
            NodePin input_pin;
            input_pin.id = next_pin_id_++;
            input_pin.type = PinType::Dataset;
            input_pin.name = "Input";
            input_pin.is_input = true;
            node.inputs.push_back(input_pin);

            node.parameters["file_path"] = "";
            if (node.type == NodeType::ExportCSV) {
                node.parameters["delimiter"] = ",";
                node.parameters["header"] = "true";
            }
            break;
        }

        // ========== KNIME-Style Table Manipulation Nodes ==========

        case NodeType::TableSplitter: {
            // Table Splitter - 1 input, 2 outputs (Top, Bottom)
            NodePin input_pin;
            input_pin.id = next_pin_id_++;
            input_pin.type = PinType::Dataset;
            input_pin.name = "Table";
            input_pin.is_input = true;
            node.inputs.push_back(input_pin);

            NodePin top_output;
            top_output.id = next_pin_id_++;
            top_output.type = PinType::Dataset;
            top_output.name = "Top";
            top_output.is_input = false;
            node.outputs.push_back(top_output);

            NodePin bottom_output;
            bottom_output.id = next_pin_id_++;
            bottom_output.type = PinType::Dataset;
            bottom_output.name = "Bottom";
            bottom_output.is_input = false;
            node.outputs.push_back(bottom_output);

            node.parameters["split_row"] = "0";
            break;
        }

        case NodeType::CellExtractor: {
            // Cell Extractor - 1 input, 2 outputs (Value, Table passthrough)
            NodePin input_pin;
            input_pin.id = next_pin_id_++;
            input_pin.type = PinType::Dataset;
            input_pin.name = "Table";
            input_pin.is_input = true;
            node.inputs.push_back(input_pin);

            NodePin value_output;
            value_output.id = next_pin_id_++;
            value_output.type = PinType::Tensor;
            value_output.name = "Value";
            value_output.is_input = false;
            node.outputs.push_back(value_output);

            NodePin table_output;
            table_output.id = next_pin_id_++;
            table_output.type = PinType::Dataset;
            table_output.name = "Table";
            table_output.is_input = false;
            node.outputs.push_back(table_output);

            node.parameters["row"] = "0";
            node.parameters["column"] = "";
            break;
        }

        case NodeType::CellUpdater: {
            // Cell Updater - 2 inputs (Table, Value), 1 output
            NodePin table_input;
            table_input.id = next_pin_id_++;
            table_input.type = PinType::Dataset;
            table_input.name = "Table";
            table_input.is_input = true;
            node.inputs.push_back(table_input);

            NodePin value_input;
            value_input.id = next_pin_id_++;
            value_input.type = PinType::Tensor;
            value_input.name = "Value";
            value_input.is_input = true;
            node.inputs.push_back(value_input);

            NodePin output_pin;
            output_pin.id = next_pin_id_++;
            output_pin.type = PinType::Dataset;
            output_pin.name = "Table";
            output_pin.is_input = false;
            node.outputs.push_back(output_pin);

            node.parameters["row"] = "0";
            node.parameters["column"] = "";
            break;
        }

        case NodeType::RowAppender: {
            // Row Appender (Concatenate) - 2 inputs (Top, Bottom), 1 output
            NodePin top_input;
            top_input.id = next_pin_id_++;
            top_input.type = PinType::Dataset;
            top_input.name = "Top";
            top_input.is_input = true;
            node.inputs.push_back(top_input);

            NodePin bottom_input;
            bottom_input.id = next_pin_id_++;
            bottom_input.type = PinType::Dataset;
            bottom_input.name = "Bottom";
            bottom_input.is_input = true;
            node.inputs.push_back(bottom_input);

            NodePin output_pin;
            output_pin.id = next_pin_id_++;
            output_pin.type = PinType::Dataset;
            output_pin.name = "Table";
            output_pin.is_input = false;
            node.outputs.push_back(output_pin);

            node.parameters["match_columns"] = "true";
            break;
        }

        case NodeType::ColumnAppender: {
            // Column Appender - 2 inputs (Left, Right), 1 output
            NodePin left_input;
            left_input.id = next_pin_id_++;
            left_input.type = PinType::Dataset;
            left_input.name = "Left";
            left_input.is_input = true;
            node.inputs.push_back(left_input);

            NodePin right_input;
            right_input.id = next_pin_id_++;
            right_input.type = PinType::Dataset;
            right_input.name = "Right";
            right_input.is_input = true;
            node.inputs.push_back(right_input);

            NodePin output_pin;
            output_pin.id = next_pin_id_++;
            output_pin.type = PinType::Dataset;
            output_pin.name = "Table";
            output_pin.is_input = false;
            node.outputs.push_back(output_pin);

            node.parameters["suffix"] = "_right";
            break;
        }

        case NodeType::RowToColumnNames:
        case NodeType::TableCropper:
        case NodeType::Unpivot:
        case NodeType::StringManipulation:
        case NodeType::MathFormula:
        case NodeType::RuleEngine: {
            // Single input Dataset, single output Dataset
            NodePin input_pin;
            input_pin.id = next_pin_id_++;
            input_pin.type = PinType::Dataset;
            input_pin.name = "Table";
            input_pin.is_input = true;
            node.inputs.push_back(input_pin);

            NodePin output_pin;
            output_pin.id = next_pin_id_++;
            output_pin.type = PinType::Dataset;
            output_pin.name = "Table";
            output_pin.is_input = false;
            node.outputs.push_back(output_pin);
            break;
        }

        
        // ===== Phase 4: Machine Learning Algorithm Nodes =====

        // Clustering nodes
        case NodeType::KMeansCluster: {
            NodePin input_pin;
            input_pin.id = next_pin_id_++;
            input_pin.type = PinType::Dataset;
            input_pin.name = "Data";
            input_pin.is_input = true;
            node.inputs.push_back(input_pin);

            NodePin output_pin;
            output_pin.id = next_pin_id_++;
            output_pin.type = PinType::Dataset;
            output_pin.name = "Clustered";
            output_pin.is_input = false;
            node.outputs.push_back(output_pin);

            // Tool-to-Node canonical params read by KMeansOperator.
            // feature_cols empty = auto-detect numeric columns (drops label
            // and __-prefixed metadata). label_col is excluded from
            // auto-detect but otherwise passed through untouched.
            node.parameters["feature_cols"] = "";
            node.parameters["label_col"] = "";
            node.parameters["n_clusters"] = "8";
            node.parameters["max_iter"] = "300";
            node.parameters["init"] = "kmeans++";
            node.parameters["n_init"] = "10";
            node.parameters["tol"] = "0.0001";
            node.parameters["seed"] = "0";
            // Legacy float-panel param (operator ignores).
            node.parameters["random_state"] = "42";
            break;
        }

        case NodeType::DBSCANCluster: {
            NodePin input_pin;
            input_pin.id = next_pin_id_++;
            input_pin.type = PinType::Dataset;
            input_pin.name = "Data";
            input_pin.is_input = true;
            node.inputs.push_back(input_pin);

            NodePin output_pin;
            output_pin.id = next_pin_id_++;
            output_pin.type = PinType::Dataset;
            output_pin.name = "Clustered";
            output_pin.is_input = false;
            node.outputs.push_back(output_pin);

            // Tool-to-Node canonical params read by DBSCANOperator.
            node.parameters["feature_cols"] = "";
            node.parameters["label_col"] = "";
            node.parameters["eps"] = "0.5";
            node.parameters["min_samples"] = "5";
            node.parameters["metric"] = "euclidean";
            break;
        }

        case NodeType::HierarchicalCluster: {
            NodePin input_pin;
            input_pin.id = next_pin_id_++;
            input_pin.type = PinType::Dataset;
            input_pin.name = "Data";
            input_pin.is_input = true;
            node.inputs.push_back(input_pin);

            NodePin output_pin;
            output_pin.id = next_pin_id_++;
            output_pin.type = PinType::Dataset;
            output_pin.name = "Clustered";
            output_pin.is_input = false;
            node.outputs.push_back(output_pin);

            // Tool-to-Node canonical params read by HierarchicalOperator.
            // linkage="ward" only works with metric="euclidean" (enforced
            // by the operator).
            node.parameters["feature_cols"] = "";
            node.parameters["label_col"] = "";
            node.parameters["n_clusters"] = "3";
            node.parameters["linkage"] = "ward";
            node.parameters["metric"] = "euclidean";
            break;
        }

        case NodeType::GMMCluster: {
            NodePin input_pin;
            input_pin.id = next_pin_id_++;
            input_pin.type = PinType::Dataset;
            input_pin.name = "Data";
            input_pin.is_input = true;
            node.inputs.push_back(input_pin);

            NodePin output_pin;
            output_pin.id = next_pin_id_++;
            output_pin.type = PinType::Dataset;
            output_pin.name = "Clustered";
            output_pin.is_input = false;
            node.outputs.push_back(output_pin);

            // Tool-to-Node canonical params read by GMMOperator.
            node.parameters["feature_cols"] = "";
            node.parameters["label_col"] = "";
            node.parameters["n_components"] = "3";
            node.parameters["covariance_type"] = "full";
            node.parameters["max_iter"] = "100";
            node.parameters["tol"] = "0.001";
            node.parameters["n_init"] = "1";
            node.parameters["seed"] = "0";
            break;
        }

        // Dimensionality Reduction nodes
        case NodeType::PCANode: {
            NodePin input_pin;
            input_pin.id = next_pin_id_++;
            input_pin.type = PinType::Dataset;
            input_pin.name = "Data";
            input_pin.is_input = true;
            node.inputs.push_back(input_pin);

            NodePin output_pin;
            output_pin.id = next_pin_id_++;
            output_pin.type = PinType::Dataset;
            output_pin.name = "Transformed";
            output_pin.is_input = false;
            node.outputs.push_back(output_pin);

            // Tool-to-Node canonical params read by PCAOperator.
            // feature_cols empty = auto-detect numeric columns (drop label
            // and __-prefixed metadata). label_col is passed through to
            // output as `y` int32. center/scale match sklearn's PCA defaults.
            node.parameters["feature_cols"] = "";
            node.parameters["label_col"] = "";
            node.parameters["n_components"] = "2";
            node.parameters["center"] = "true";
            node.parameters["scale"] = "false";
            // Legacy floating-panel param (operator ignores in v1).
            node.parameters["whiten"] = "false";
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
        case NodeType::DecisionTreeClassifier: {
            NodePin data_in;
            data_in.id = next_pin_id_++;
            data_in.type = PinType::Dataset;
            data_in.name = "Train Data";
            data_in.is_input = true;
            node.inputs.push_back(data_in);

            NodePin labels_in;
            labels_in.id = next_pin_id_++;
            labels_in.type = PinType::Labels;
            labels_in.name = "Labels";
            labels_in.is_input = true;
            node.inputs.push_back(labels_in);

            NodePin model_out;
            model_out.id = next_pin_id_++;
            model_out.type = PinType::Parameters;
            model_out.name = "Model";
            model_out.is_input = false;
            node.outputs.push_back(model_out);

            NodePin pred_out;
            pred_out.id = next_pin_id_++;
            pred_out.type = PinType::Labels;
            pred_out.name = "Predictions";
            pred_out.is_input = false;
            node.outputs.push_back(pred_out);

            node.parameters["max_depth"] = "10";
            node.parameters["min_samples_split"] = "2";
            node.parameters["criterion"] = "gini";
            break;
        }

        case NodeType::RandomForestClassifier: {
            NodePin data_in;
            data_in.id = next_pin_id_++;
            data_in.type = PinType::Dataset;
            data_in.name = "Train Data";
            data_in.is_input = true;
            node.inputs.push_back(data_in);

            NodePin labels_in;
            labels_in.id = next_pin_id_++;
            labels_in.type = PinType::Labels;
            labels_in.name = "Labels";
            labels_in.is_input = true;
            node.inputs.push_back(labels_in);

            NodePin model_out;
            model_out.id = next_pin_id_++;
            model_out.type = PinType::Parameters;
            model_out.name = "Model";
            model_out.is_input = false;
            node.outputs.push_back(model_out);

            NodePin pred_out;
            pred_out.id = next_pin_id_++;
            pred_out.type = PinType::Labels;
            pred_out.name = "Predictions";
            pred_out.is_input = false;
            node.outputs.push_back(pred_out);

            node.parameters["n_estimators"] = "100";
            node.parameters["max_depth"] = "10";
            node.parameters["min_samples_split"] = "2";
            break;
        }

        case NodeType::GradientBoostingClassifier: {
            NodePin data_in;
            data_in.id = next_pin_id_++;
            data_in.type = PinType::Dataset;
            data_in.name = "Train Data";
            data_in.is_input = true;
            node.inputs.push_back(data_in);

            NodePin labels_in;
            labels_in.id = next_pin_id_++;
            labels_in.type = PinType::Labels;
            labels_in.name = "Labels";
            labels_in.is_input = true;
            node.inputs.push_back(labels_in);

            NodePin model_out;
            model_out.id = next_pin_id_++;
            model_out.type = PinType::Parameters;
            model_out.name = "Model";
            model_out.is_input = false;
            node.outputs.push_back(model_out);

            NodePin pred_out;
            pred_out.id = next_pin_id_++;
            pred_out.type = PinType::Labels;
            pred_out.name = "Predictions";
            pred_out.is_input = false;
            node.outputs.push_back(pred_out);

            node.parameters["n_estimators"] = "100";
            node.parameters["learning_rate"] = "0.1";
            node.parameters["max_depth"] = "3";
            break;
        }

        case NodeType::SVMClassifier: {
            NodePin data_in;
            data_in.id = next_pin_id_++;
            data_in.type = PinType::Dataset;
            data_in.name = "Train Data";
            data_in.is_input = true;
            node.inputs.push_back(data_in);

            NodePin labels_in;
            labels_in.id = next_pin_id_++;
            labels_in.type = PinType::Labels;
            labels_in.name = "Labels";
            labels_in.is_input = true;
            node.inputs.push_back(labels_in);

            NodePin model_out;
            model_out.id = next_pin_id_++;
            model_out.type = PinType::Parameters;
            model_out.name = "Model";
            model_out.is_input = false;
            node.outputs.push_back(model_out);

            NodePin pred_out;
            pred_out.id = next_pin_id_++;
            pred_out.type = PinType::Labels;
            pred_out.name = "Predictions";
            pred_out.is_input = false;
            node.outputs.push_back(pred_out);

            node.parameters["kernel"] = "rbf";
            node.parameters["C"] = "1.0";
            node.parameters["gamma"] = "scale";
            break;
        }

        case NodeType::KNNClassifier: {
            NodePin data_in;
            data_in.id = next_pin_id_++;
            data_in.type = PinType::Dataset;
            data_in.name = "Train Data";
            data_in.is_input = true;
            node.inputs.push_back(data_in);

            NodePin labels_in;
            labels_in.id = next_pin_id_++;
            labels_in.type = PinType::Labels;
            labels_in.name = "Labels";
            labels_in.is_input = true;
            node.inputs.push_back(labels_in);

            NodePin model_out;
            model_out.id = next_pin_id_++;
            model_out.type = PinType::Parameters;
            model_out.name = "Model";
            model_out.is_input = false;
            node.outputs.push_back(model_out);

            NodePin pred_out;
            pred_out.id = next_pin_id_++;
            pred_out.type = PinType::Labels;
            pred_out.name = "Predictions";
            pred_out.is_input = false;
            node.outputs.push_back(pred_out);

            node.parameters["n_neighbors"] = "5";
            node.parameters["weights"] = "uniform";
            node.parameters["metric"] = "euclidean";
            break;
        }

        case NodeType::NaiveBayesClassifier: {
            NodePin data_in;
            data_in.id = next_pin_id_++;
            data_in.type = PinType::Dataset;
            data_in.name = "Train Data";
            data_in.is_input = true;
            node.inputs.push_back(data_in);

            NodePin labels_in;
            labels_in.id = next_pin_id_++;
            labels_in.type = PinType::Labels;
            labels_in.name = "Labels";
            labels_in.is_input = true;
            node.inputs.push_back(labels_in);

            NodePin model_out;
            model_out.id = next_pin_id_++;
            model_out.type = PinType::Parameters;
            model_out.name = "Model";
            model_out.is_input = false;
            node.outputs.push_back(model_out);

            NodePin pred_out;
            pred_out.id = next_pin_id_++;
            pred_out.type = PinType::Labels;
            pred_out.name = "Predictions";
            pred_out.is_input = false;
            node.outputs.push_back(pred_out);

            node.parameters["var_smoothing"] = "1e-9";
            break;
        }

        case NodeType::LogisticRegressionNode: {
            NodePin data_in;
            data_in.id = next_pin_id_++;
            data_in.type = PinType::Dataset;
            data_in.name = "Train Data";
            data_in.is_input = true;
            node.inputs.push_back(data_in);

            NodePin labels_in;
            labels_in.id = next_pin_id_++;
            labels_in.type = PinType::Labels;
            labels_in.name = "Labels";
            labels_in.is_input = true;
            node.inputs.push_back(labels_in);

            NodePin model_out;
            model_out.id = next_pin_id_++;
            model_out.type = PinType::Parameters;
            model_out.name = "Model";
            model_out.is_input = false;
            node.outputs.push_back(model_out);

            NodePin pred_out;
            pred_out.id = next_pin_id_++;
            pred_out.type = PinType::Labels;
            pred_out.name = "Predictions";
            pred_out.is_input = false;
            node.outputs.push_back(pred_out);

            node.parameters["C"] = "1.0";
            node.parameters["solver"] = "lbfgs";
            node.parameters["max_iter"] = "100";
            break;
        }

        // Regression nodes
        case NodeType::LinearRegressionNode: {
            NodePin data_in;
            data_in.id = next_pin_id_++;
            data_in.type = PinType::Dataset;
            data_in.name = "Data";
            data_in.is_input = true;
            node.inputs.push_back(data_in);

            NodePin fitted_out;
            fitted_out.id = next_pin_id_++;
            fitted_out.type = PinType::Dataset;
            fitted_out.name = "Fitted";
            fitted_out.is_input = false;
            node.outputs.push_back(fitted_out);

            // Tool-to-Node canonical params read by LinearRegressionOperator.
            // feature_cols is comma-sep predictor columns; target_col is the
            // response. Output appends prediction + residual columns to the
            // input table.
            node.parameters["feature_cols"] = "";
            node.parameters["target_col"] = "";
            node.parameters["fit_intercept"] = "true";
            break;
        }

        case NodeType::PolynomialRegressionNode: {
            NodePin data_in;
            data_in.id = next_pin_id_++;
            data_in.type = PinType::Dataset;
            data_in.name = "Data";
            data_in.is_input = true;
            node.inputs.push_back(data_in);

            NodePin fitted_out;
            fitted_out.id = next_pin_id_++;
            fitted_out.type = PinType::Dataset;
            fitted_out.name = "Fitted";
            fitted_out.is_input = false;
            node.outputs.push_back(fitted_out);

            // Tool-to-Node canonical params read by PolynomialRegressionOperator.
            // Single predictor only (backend PolynomialRegression takes one x).
            node.parameters["feature_col"] = "";
            node.parameters["target_col"] = "";
            node.parameters["degree"] = "2";
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
        case NodeType::ConfusionMatrixNode: {
            NodePin pred_in;
            pred_in.id = next_pin_id_++;
            pred_in.type = PinType::Labels;
            pred_in.name = "Predictions";
            pred_in.is_input = true;
            node.inputs.push_back(pred_in);

            NodePin labels_in;
            labels_in.id = next_pin_id_++;
            labels_in.type = PinType::Labels;
            labels_in.name = "True Labels";
            labels_in.is_input = true;
            node.inputs.push_back(labels_in);

            NodePin matrix_out;
            matrix_out.id = next_pin_id_++;
            matrix_out.type = PinType::Dataset;
            matrix_out.name = "Matrix";
            matrix_out.is_input = false;
            node.outputs.push_back(matrix_out);

            node.parameters["normalize"] = "false";
            break;
        }

        case NodeType::ROCCurveNode: {
            NodePin proba_in;
            proba_in.id = next_pin_id_++;
            proba_in.type = PinType::Tensor;
            proba_in.name = "Probabilities";
            proba_in.is_input = true;
            node.inputs.push_back(proba_in);

            NodePin labels_in;
            labels_in.id = next_pin_id_++;
            labels_in.type = PinType::Labels;
            labels_in.name = "True Labels";
            labels_in.is_input = true;
            node.inputs.push_back(labels_in);

            NodePin curve_out;
            curve_out.id = next_pin_id_++;
            curve_out.type = PinType::Dataset;
            curve_out.name = "ROC Data";
            curve_out.is_input = false;
            node.outputs.push_back(curve_out);

            NodePin auc_out;
            auc_out.id = next_pin_id_++;
            auc_out.type = PinType::Tensor;
            auc_out.name = "AUC";
            auc_out.is_input = false;
            node.outputs.push_back(auc_out);
            break;
        }

        case NodeType::PRCurveNode: {
            NodePin proba_in;
            proba_in.id = next_pin_id_++;
            proba_in.type = PinType::Tensor;
            proba_in.name = "Probabilities";
            proba_in.is_input = true;
            node.inputs.push_back(proba_in);

            NodePin labels_in;
            labels_in.id = next_pin_id_++;
            labels_in.type = PinType::Labels;
            labels_in.name = "True Labels";
            labels_in.is_input = true;
            node.inputs.push_back(labels_in);

            NodePin curve_out;
            curve_out.id = next_pin_id_++;
            curve_out.type = PinType::Dataset;
            curve_out.name = "PR Data";
            curve_out.is_input = false;
            node.outputs.push_back(curve_out);
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

        case NodeType::RegressionMetricsNode: {
            // RegressionMetricsNode - Compute regression metrics
            NodePin pred_in;
            pred_in.id = next_pin_id_++;
            pred_in.type = PinType::Tensor;
            pred_in.name = "Predictions";
            pred_in.is_input = true;
            node.inputs.push_back(pred_in);

            NodePin truth_in;
            truth_in.id = next_pin_id_++;
            truth_in.type = PinType::Tensor;
            truth_in.name = "Ground Truth";
            truth_in.is_input = true;
            node.inputs.push_back(truth_in);

            NodePin metrics_out;
            metrics_out.id = next_pin_id_++;
            metrics_out.type = PinType::Dataset;
            metrics_out.name = "Metrics";
            metrics_out.is_input = false;
            node.outputs.push_back(metrics_out);

            node.parameters["metrics"] = "mse,rmse,mae,r2";  // Comma-separated metrics to compute
            break;
        }

        // ===== Phase 4: Data Preprocessing Nodes =====
        case NodeType::StandardScaler: {
            NodePin input_pin;
            input_pin.id = next_pin_id_++;
            input_pin.type = PinType::Dataset;
            input_pin.name = "Data";
            input_pin.is_input = true;
            node.inputs.push_back(input_pin);

            NodePin output_pin;
            output_pin.id = next_pin_id_++;
            output_pin.type = PinType::Dataset;
            output_pin.name = "Scaled";
            output_pin.is_input = false;
            node.outputs.push_back(output_pin);

            // Tool-to-Node canonical params read by StandardScalerOperator.
            // columns empty = auto-detect numeric; label_col excluded from auto-detect.
            node.parameters["columns"] = "";
            node.parameters["label_col"] = "";
            node.parameters["with_mean"] = "true";
            node.parameters["with_std"] = "true";
            break;
        }

        case NodeType::MinMaxScaler: {
            NodePin input_pin;
            input_pin.id = next_pin_id_++;
            input_pin.type = PinType::Dataset;
            input_pin.name = "Data";
            input_pin.is_input = true;
            node.inputs.push_back(input_pin);

            NodePin output_pin;
            output_pin.id = next_pin_id_++;
            output_pin.type = PinType::Dataset;
            output_pin.name = "Scaled";
            output_pin.is_input = false;
            node.outputs.push_back(output_pin);

            // Tool-to-Node canonical params read by MinMaxScalerOperator.
            node.parameters["columns"] = "";
            node.parameters["label_col"] = "";
            node.parameters["min"] = "0";
            node.parameters["max"] = "1";
            break;
        }

        case NodeType::RobustScaler: {
            NodePin input_pin;
            input_pin.id = next_pin_id_++;
            input_pin.type = PinType::Dataset;
            input_pin.name = "Data";
            input_pin.is_input = true;
            node.inputs.push_back(input_pin);

            NodePin output_pin;
            output_pin.id = next_pin_id_++;
            output_pin.type = PinType::Dataset;
            output_pin.name = "Scaled";
            output_pin.is_input = false;
            node.outputs.push_back(output_pin);

            // Tool-to-Node canonical params read by RobustScalerOperator.
            node.parameters["columns"] = "";
            node.parameters["label_col"] = "";
            node.parameters["with_centering"] = "true";
            node.parameters["with_scaling"] = "true";
            node.parameters["quantile_min"] = "25";
            node.parameters["quantile_max"] = "75";
            break;
        }

        case NodeType::LabelEncoder: {
            NodePin input_pin;
            input_pin.id = next_pin_id_++;
            input_pin.type = PinType::Dataset;
            input_pin.name = "Data";
            input_pin.is_input = true;
            node.inputs.push_back(input_pin);

            NodePin output_pin;
            output_pin.id = next_pin_id_++;
            output_pin.type = PinType::Dataset;
            output_pin.name = "Encoded";
            output_pin.is_input = false;
            node.outputs.push_back(output_pin);

            node.parameters["column"] = "";
            break;
        }

        case NodeType::OrdinalEncoder: {
            NodePin input_pin;
            input_pin.id = next_pin_id_++;
            input_pin.type = PinType::Dataset;
            input_pin.name = "Data";
            input_pin.is_input = true;
            node.inputs.push_back(input_pin);

            NodePin output_pin;
            output_pin.id = next_pin_id_++;
            output_pin.type = PinType::Dataset;
            output_pin.name = "Encoded";
            output_pin.is_input = false;
            node.outputs.push_back(output_pin);

            node.parameters["columns"] = "";
            node.parameters["categories"] = "auto";
            break;
        }

        case NodeType::TargetEncoder: {
            NodePin data_in;
            data_in.id = next_pin_id_++;
            data_in.type = PinType::Dataset;
            data_in.name = "Data";
            data_in.is_input = true;
            node.inputs.push_back(data_in);

            NodePin output_pin;
            output_pin.id = next_pin_id_++;
            output_pin.type = PinType::Dataset;
            output_pin.name = "Encoded";
            output_pin.is_input = false;
            node.outputs.push_back(output_pin);

            // Tool-to-Node canonical params read by TargetEncoderOperator.
            // columns = categorical cols to encode; target_col = numeric target.
            node.parameters["columns"] = "";
            node.parameters["target_col"] = "";
            node.parameters["smoothing"] = "1.0";
            break;
        }

        // (TrainTestSplit case removed — use NodeType::DataSplit instead,
        //  which supports 3-way train/val/test split and is the single source
        //  of truth for dataset partitioning.)

        // ===== Phase 8: Advanced Preprocessing Nodes (UI Consolidation) =====
        case NodeType::OutlierDetector: {
            NodePin data_in;
            data_in.id = next_pin_id_++;
            data_in.type = PinType::Dataset;
            data_in.name = "Data";
            data_in.is_input = true;
            node.inputs.push_back(data_in);

            NodePin flagged_out;
            flagged_out.id = next_pin_id_++;
            flagged_out.type = PinType::Dataset;
            flagged_out.name = "Flagged";
            flagged_out.is_input = false;
            node.outputs.push_back(flagged_out);

            // Tool-to-Node canonical params read by OutlierDetectorOperator.
            // columns="all" or empty = auto-detect numeric. method: iqr|zscore.
            // Only "flag" action is wired in v1 (adds is_outlier column);
            // remove/clip/isolation_forest/lof deferred to tofix.
            node.parameters["method"] = "iqr";           // iqr, zscore (isolation_forest/lof deferred)
            node.parameters["threshold"] = "1.5";        // IQR multiplier or Z-score threshold
            node.parameters["columns"] = "all";          // "all" or csv list
            node.parameters["label_col"] = "";
            node.parameters["action"] = "flag";          // only "flag" is live; remove/clip deferred
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
            NodePin data_in;
            data_in.id = next_pin_id_++;
            data_in.type = PinType::Dataset;
            data_in.name = "Data";
            data_in.is_input = true;
            node.inputs.push_back(data_in);

            NodePin valid_out;
            valid_out.id = next_pin_id_++;
            valid_out.type = PinType::Dataset;
            valid_out.name = "Valid";
            valid_out.is_input = false;
            node.outputs.push_back(valid_out);

            NodePin invalid_out;
            invalid_out.id = next_pin_id_++;
            invalid_out.type = PinType::Dataset;
            invalid_out.name = "Invalid";
            invalid_out.is_input = false;
            node.outputs.push_back(invalid_out);

            NodePin issues_out;
            issues_out.id = next_pin_id_++;
            issues_out.type = PinType::Dataset;
            issues_out.name = "Issues";
            issues_out.is_input = false;
            node.outputs.push_back(issues_out);

            node.parameters["required_columns"] = "";       // Comma-separated required column names
            node.parameters["unique_columns"] = "";         // Columns that must have unique values
            node.parameters["column_types"] = "";           // JSON: {"col": "type"}
            node.parameters["value_ranges"] = "";           // JSON: {"col": [min, max]}
            node.parameters["not_null_columns"] = "";       // Columns that cannot have nulls
            node.parameters["regex_patterns"] = "";         // JSON: {"col": "pattern"}
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
        case NodeType::FFTNode: {
            NodePin input_pin;
            input_pin.id = next_pin_id_++;
            input_pin.type = PinType::Dataset;
            input_pin.name = "Data";
            input_pin.is_input = true;
            node.inputs.push_back(input_pin);

            NodePin output_pin;
            output_pin.id = next_pin_id_++;
            output_pin.type = PinType::Dataset;
            output_pin.name = "Spectrum";
            output_pin.is_input = false;
            node.outputs.push_back(output_pin);

            // Tool-to-Node canonical params read by FFTOperator.
            // signal_col is required. sample_rate controls frequency-axis
            // scaling (Hz). Output is a frequency-domain table (row count
            // changes) with frequency/magnitude/phase columns.
            node.parameters["signal_col"] = "";
            node.parameters["sample_rate"] = "1.0";
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

        case NodeType::FilterDesigner: {
            NodePin input_pin;
            input_pin.id = next_pin_id_++;
            input_pin.type = PinType::Dataset;
            input_pin.name = "Data";
            input_pin.is_input = true;
            node.inputs.push_back(input_pin);

            NodePin output_pin;
            output_pin.id = next_pin_id_++;
            output_pin.type = PinType::Dataset;
            output_pin.name = "Filtered";
            output_pin.is_input = false;
            node.outputs.push_back(output_pin);

            // Tool-to-Node canonical params read by FilterDesignerOperator.
            // The operator combines filter design + application: the signal
            // column is filtered in-place (column retyped to float32).
            // bandpass/bandstop require cutoff_high > cutoff.
            node.parameters["signal_col"] = "";
            node.parameters["filter_type"] = "lowpass";
            node.parameters["cutoff"] = "0.5";
            node.parameters["cutoff_high"] = "0";
            node.parameters["sample_rate"] = "1.0";
            node.parameters["order"] = "4";
            break;
        }

        case NodeType::Convolution1D: {
            NodePin data_in;
            data_in.id = next_pin_id_++;
            data_in.type = PinType::Dataset;
            data_in.name = "Data";
            data_in.is_input = true;
            node.inputs.push_back(data_in);

            NodePin output_pin;
            output_pin.id = next_pin_id_++;
            output_pin.type = PinType::Dataset;
            output_pin.name = "Convolved";
            output_pin.is_input = false;
            node.outputs.push_back(output_pin);

            // Tool-to-Node canonical params read by Convolve1DOperator.
            // kernel is a comma-separated list of taps. The operator
            // preserves row alignment with same-length output.
            node.parameters["signal_col"] = "";
            node.parameters["kernel"] = "0.25,0.5,0.25";
            node.parameters["mode"] = "same";
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
        case NodeType::TFIDFVectorizer: {
            NodePin input_pin;
            input_pin.id = next_pin_id_++;
            input_pin.type = PinType::Dataset;
            input_pin.name = "Text";
            input_pin.is_input = true;
            node.inputs.push_back(input_pin);

            NodePin output_pin;
            output_pin.id = next_pin_id_++;
            output_pin.type = PinType::Dataset;
            output_pin.name = "Vectors";
            output_pin.is_input = false;
            node.outputs.push_back(output_pin);

            // Tool-to-Node canonical params read by TFIDFVectorizerOperator.
            // text_col + label_col are required when this node runs as a real
            // Cat-1 operator on an Arrow source. Legacy ngram_range / min_df
            // params kept for back-compat with the floating panel; operator
            // ignores them in v1.
            node.parameters["text_col"] = "";
            node.parameters["label_col"] = "";
            node.parameters["max_features"] = "2000";
            node.parameters["use_idf"] = "true";
            node.parameters["smooth_idf"] = "true";
            node.parameters["norm"] = "l2";
            // Legacy floating-panel params (operator ignores).
            node.parameters["ngram_range"] = "1,1";
            node.parameters["min_df"] = "1";
            break;
        }

        case NodeType::CountVectorizer: {
            NodePin input_pin;
            input_pin.id = next_pin_id_++;
            input_pin.type = PinType::Dataset;
            input_pin.name = "Text";
            input_pin.is_input = true;
            node.inputs.push_back(input_pin);

            NodePin output_pin;
            output_pin.id = next_pin_id_++;
            output_pin.type = PinType::Dataset;
            output_pin.name = "Vectors";
            output_pin.is_input = false;
            node.outputs.push_back(output_pin);

            // Tool-to-Node canonical params read by CountVectorizerOperator.
            // text_col + label_col are required when this node runs as a real
            // Cat-1 operator on an Arrow source. Legacy ngram_range param
            // kept for back-compat with the floating panel; operator ignores
            // it in v1.
            node.parameters["text_col"] = "";
            node.parameters["label_col"] = "";
            node.parameters["max_features"] = "2000";
            node.parameters["norm"] = "l2";
            // Legacy floating-panel param (operator ignores).
            node.parameters["ngram_range"] = "1,1";
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

        case NodeType::SentimentAnalyzer: {
            NodePin input_pin;
            input_pin.id = next_pin_id_++;
            input_pin.type = PinType::Dataset;
            input_pin.name = "Text";
            input_pin.is_input = true;
            node.inputs.push_back(input_pin);

            NodePin output_pin;
            output_pin.id = next_pin_id_++;
            output_pin.type = PinType::Dataset;
            output_pin.name = "Sentiment";
            output_pin.is_input = false;
            node.outputs.push_back(output_pin);

            // Tool-to-Node canonical params read by SentimentAnalyzerOperator.
            // text_col is required (no auto-detect — operator errors cleanly).
            // label_col empty = no passthrough. method picks the built-in
            // lexicon ("simple" / "vader" / "afinn").
            node.parameters["text_col"] = "text";
            node.parameters["label_col"] = "";
            node.parameters["method"] = "vader";
            // Legacy param kept for cyxgraph JSON back-compat (operator ignores).
            node.parameters["model"] = "vader";
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
        case NodeType::CalculatorNode: {
            NodePin input_a;
            input_a.id = next_pin_id_++;
            input_a.type = PinType::Tensor;
            input_a.name = "A";
            input_a.is_input = true;
            node.inputs.push_back(input_a);

            NodePin input_b;
            input_b.id = next_pin_id_++;
            input_b.type = PinType::Tensor;
            input_b.name = "B";
            input_b.is_input = true;
            input_b.is_required = false;
            node.inputs.push_back(input_b);

            NodePin output_pin;
            output_pin.id = next_pin_id_++;
            output_pin.type = PinType::Tensor;
            output_pin.name = "Result";
            output_pin.is_input = false;
            node.outputs.push_back(output_pin);

            node.parameters["expression"] = "A + B";
            break;
        }

        case NodeType::UnitConverter: {
            NodePin input_pin;
            input_pin.id = next_pin_id_++;
            input_pin.type = PinType::Tensor;
            input_pin.name = "Value";
            input_pin.is_input = true;
            node.inputs.push_back(input_pin);

            NodePin output_pin;
            output_pin.id = next_pin_id_++;
            output_pin.type = PinType::Tensor;
            output_pin.name = "Converted";
            output_pin.is_input = false;
            node.outputs.push_back(output_pin);

            node.parameters["from_unit"] = "meters";
            node.parameters["to_unit"] = "feet";
            break;
        }

        case NodeType::RegexTester: {
            NodePin input_pin;
            input_pin.id = next_pin_id_++;
            input_pin.type = PinType::Dataset;
            input_pin.name = "Text";
            input_pin.is_input = true;
            node.inputs.push_back(input_pin);

            NodePin matches_out;
            matches_out.id = next_pin_id_++;
            matches_out.type = PinType::Dataset;
            matches_out.name = "Matches";
            matches_out.is_input = false;
            node.outputs.push_back(matches_out);

            node.parameters["pattern"] = "";
            node.parameters["flags"] = "";
            break;
        }

        case NodeType::JSONPathExtractor: {
            NodePin input_pin;
            input_pin.id = next_pin_id_++;
            input_pin.type = PinType::Dataset;
            input_pin.name = "JSON";
            input_pin.is_input = true;
            node.inputs.push_back(input_pin);

            NodePin output_pin;
            output_pin.id = next_pin_id_++;
            output_pin.type = PinType::Dataset;
            output_pin.name = "Extracted";
            output_pin.is_input = false;
            node.outputs.push_back(output_pin);

            node.parameters["path"] = "$.data";
            break;
        }

        case NodeType::DataProfiler: {
            NodePin input_pin;
            input_pin.id = next_pin_id_++;
            input_pin.type = PinType::Dataset;
            input_pin.name = "Data";
            input_pin.is_input = true;
            node.inputs.push_back(input_pin);

            NodePin report_out;
            report_out.id = next_pin_id_++;
            report_out.type = PinType::Dataset;
            report_out.name = "Profile";
            report_out.is_input = false;
            node.outputs.push_back(report_out);

            node.parameters["include_correlations"] = "true";
            node.parameters["include_histograms"] = "true";
            break;
        }

        // Visualization nodes are defined in their own translation unit
        // (cyxwiz-engine/src/gui/visualization/) to keep this already-huge
        // factory file from owning the chart framework too. New chart
        // types register themselves by adding a case here that delegates
        // to the corresponding Populate* function.
        case NodeType::BarChart: {
            visualization::PopulateBarChartNode(node, next_pin_id_);
            break;
        }

        default:
            // Default: input and output pins
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
            break;
    }

    return node;
}

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

    // IMPORTANT: Clear properties panel selection BEFORE clearing nodes
    // to prevent dangling pointer access (the properties panel holds a raw pointer
    // to the selected node which becomes invalid after nodes_.clear())
    if (properties_panel_) {
        properties_panel_->ClearSelection();
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

        // ===== Upsampling - Indigo =====
        case NodeType::ConvTranspose2D:
            return IM_COL32(92, 107, 192, 255);
        case NodeType::Upsample:
            return IM_COL32(121, 134, 203, 255);
        case NodeType::PixelShuffle:
            return IM_COL32(159, 168, 218, 255);

        // ===== Time-Series - Amber =====
        case NodeType::TimeSeriesWindow:
            return IM_COL32(255, 160, 0, 255);
        case NodeType::TimeSeriesFeatures:
            return IM_COL32(255, 179, 0, 255);
        case NodeType::TimeSeriesSplit:
            return IM_COL32(255, 196, 0, 255);
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
        case NodeType::SVMClassifier:
        case NodeType::KNNClassifier:
        case NodeType::NaiveBayesClassifier:
        case NodeType::LogisticRegressionNode:
            return IM_COL32(0, 137, 123, 255);

        // ===== ML Regression Nodes - Light Teal =====
        case NodeType::LinearRegressionNode:
        case NodeType::PolynomialRegressionNode:
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

} // namespace gui
