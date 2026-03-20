#include "node_editor.h"
#include "properties.h"
#include "icons.h"
#include "../plugin/registries/plugin_node_registry.h"
#include <imgui.h>
#include <imnodes.h>
#include <spdlog/spdlog.h>
#include <algorithm>

namespace gui {

// Unified Canvas Phase 1: Map NodeType to NodeCategory for UI organization
NodeCategory NodeEditor::GetCategoryForNodeType(NodeType type) {
    switch (type) {
        // Data Sources
        case NodeType::CSVFile:
        case NodeType::SQLQuery:
        case NodeType::HDF5Dataset:
        case NodeType::ParquetFile:
        case NodeType::JSONFile:
        case NodeType::ExcelFile:
        case NodeType::RESTAPISource:
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
            return NodeCategory::DataExport;

        // Preprocessing
        case NodeType::Normalize:
        case NodeType::OneHotEncode:
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
            node.inputs.push_back(input_pin);

            NodePin output_pin;
            output_pin.id = next_pin_id_++;
            output_pin.type = PinType::Tensor;
            output_pin.name = "Output";
            output_pin.is_input = false;
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
            node.inputs.push_back(input_pin);

            // Output: Predictions (goes to Loss function)
            NodePin output_pin;
            output_pin.id = next_pin_id_++;
            output_pin.type = PinType::Tensor;
            output_pin.name = "Predictions";
            output_pin.is_input = false;
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
            node.inputs.push_back(input_pin);

            NodePin output_pin;
            output_pin.id = next_pin_id_++;
            output_pin.type = PinType::Tensor;
            output_pin.name = "Output";
            output_pin.is_input = false;
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
            node.inputs.push_back(input_pin);

            NodePin output_pin;
            output_pin.id = next_pin_id_++;
            output_pin.type = PinType::Tensor;
            output_pin.name = "Output";
            output_pin.is_input = false;
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
            node.inputs.push_back(input_pin);

            NodePin output_pin;
            output_pin.id = next_pin_id_++;
            output_pin.type = PinType::Tensor;
            output_pin.name = "Output";
            output_pin.is_input = false;
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
            node.inputs.push_back(input_pin);

            NodePin output_pin;
            output_pin.id = next_pin_id_++;
            output_pin.type = PinType::Tensor;
            output_pin.name = "Output";
            output_pin.is_input = false;
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
            node.inputs.push_back(input_pin);

            NodePin output_pin;
            output_pin.id = next_pin_id_++;
            output_pin.type = PinType::Tensor;
            output_pin.name = "Output";
            output_pin.is_input = false;
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
            // DataLoader node - batch iterator
            // Input: Dataset reference
            NodePin dataset_pin;
            dataset_pin.id = next_pin_id_++;
            dataset_pin.type = PinType::Dataset;
            dataset_pin.name = "Dataset";
            dataset_pin.is_input = true;
            node.inputs.push_back(dataset_pin);

            // Output: Batched data
            NodePin batch_pin;
            batch_pin.id = next_pin_id_++;
            batch_pin.type = PinType::Tensor;
            batch_pin.name = "Batch";
            batch_pin.is_input = false;
            node.outputs.push_back(batch_pin);

            // Output: Batched labels
            NodePin labels_pin;
            labels_pin.id = next_pin_id_++;
            labels_pin.type = PinType::Labels;
            labels_pin.name = "Labels";
            labels_pin.is_input = false;
            node.outputs.push_back(labels_pin);

            // Parameters
            node.parameters["batch_size"] = "32";
            node.parameters["shuffle"] = "true";
            node.parameters["drop_last"] = "false";
            node.parameters["num_workers"] = "4";
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
            node.inputs.push_back(input_pin);

            // Output: Augmented data
            NodePin output_pin;
            output_pin.id = next_pin_id_++;
            output_pin.type = PinType::Tensor;
            output_pin.name = "Output";
            output_pin.is_input = false;
            node.outputs.push_back(output_pin);

            // Parameters (transform pipeline)
            node.parameters["transforms"] = "RandomFlip,Normalize";
            node.parameters["flip_prob"] = "0.5";
            node.parameters["normalize_mean"] = "0.0";
            node.parameters["normalize_std"] = "1.0";
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
            node.inputs.push_back(data_in);

            // Input: Labels tensor
            NodePin labels_in;
            labels_in.id = next_pin_id_++;
            labels_in.type = PinType::Labels;
            labels_in.name = "Labels";
            labels_in.is_input = true;
            node.inputs.push_back(labels_in);

            // Output: Train Data
            NodePin train_data;
            train_data.id = next_pin_id_++;
            train_data.type = PinType::Tensor;
            train_data.name = "Train Data";
            train_data.is_input = false;
            node.outputs.push_back(train_data);

            // Output: Train Labels
            NodePin train_labels;
            train_labels.id = next_pin_id_++;
            train_labels.type = PinType::Labels;
            train_labels.name = "Train Labels";
            train_labels.is_input = false;
            node.outputs.push_back(train_labels);

            // Output: Val Data
            NodePin val_data;
            val_data.id = next_pin_id_++;
            val_data.type = PinType::Tensor;
            val_data.name = "Val Data";
            val_data.is_input = false;
            node.outputs.push_back(val_data);

            // Output: Val Labels
            NodePin val_labels;
            val_labels.id = next_pin_id_++;
            val_labels.type = PinType::Labels;
            val_labels.name = "Val Labels";
            val_labels.is_input = false;
            node.outputs.push_back(val_labels);

            // Output: Test Data
            NodePin test_data;
            test_data.id = next_pin_id_++;
            test_data.type = PinType::Tensor;
            test_data.name = "Test Data";
            test_data.is_input = false;
            node.outputs.push_back(test_data);

            // Output: Test Labels
            NodePin test_labels;
            test_labels.id = next_pin_id_++;
            test_labels.type = PinType::Labels;
            test_labels.name = "Test Labels";
            test_labels.is_input = false;
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
            node.inputs.push_back(input_pin);

            NodePin output_pin;
            output_pin.id = next_pin_id_++;
            output_pin.type = PinType::Tensor;
            output_pin.name = "Output";
            output_pin.is_input = false;
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
            node.inputs.push_back(input_pin);

            NodePin output_pin;
            output_pin.id = next_pin_id_++;
            output_pin.type = PinType::Tensor;
            output_pin.name = "Output";
            output_pin.is_input = false;
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
            node.inputs.push_back(input_pin);

            NodePin output_pin;
            output_pin.id = next_pin_id_++;
            output_pin.type = PinType::Tensor;
            output_pin.name = "OneHot";
            output_pin.is_input = false;
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
            node.inputs.push_back(pred_pin);

            // Input 2: Targets (ground truth labels)
            NodePin target_pin;
            target_pin.id = next_pin_id_++;
            target_pin.type = PinType::Tensor;
            target_pin.name = "Targets";
            target_pin.is_input = true;
            node.inputs.push_back(target_pin);

            // Output: Loss value
            NodePin loss_pin;
            loss_pin.id = next_pin_id_++;
            loss_pin.type = PinType::Loss;
            loss_pin.name = "Loss";
            loss_pin.is_input = false;
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
            node.inputs.push_back(loss_pin);

            NodePin state_pin;
            state_pin.id = next_pin_id_++;
            state_pin.type = PinType::Optimizer;
            state_pin.name = "State";
            state_pin.is_input = false;
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
            node.inputs.push_back(input_pin);

            NodePin output_pin;
            output_pin.id = next_pin_id_++;
            output_pin.type = PinType::Tensor;
            output_pin.name = "Output";
            output_pin.is_input = false;
            node.outputs.push_back(output_pin);

            NodePin hidden_pin;
            hidden_pin.id = next_pin_id_++;
            hidden_pin.type = PinType::Tensor;
            hidden_pin.name = "Hidden";
            hidden_pin.is_input = false;
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
            node.inputs.push_back(input_pin);

            NodePin output_pin;
            output_pin.id = next_pin_id_++;
            output_pin.type = PinType::Tensor;
            output_pin.name = "Output";
            output_pin.is_input = false;
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
            node.inputs.push_back(input_pin);

            NodePin output_pin;
            output_pin.id = next_pin_id_++;
            output_pin.type = PinType::Tensor;
            output_pin.name = "Embeddings";
            output_pin.is_input = false;
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

            // Optional attention mask (for padding/causal masks)
            NodePin mask_pin;
            mask_pin.id = next_pin_id_++;
            mask_pin.type = PinType::Tensor;
            mask_pin.name = "Mask";
            mask_pin.is_input = true;
            mask_pin.is_required = false;  // Optional
            mask_pin.is_variadic = false;
            node.inputs.push_back(mask_pin);

            // Output: Attended values
            NodePin output_pin;
            output_pin.id = next_pin_id_++;
            output_pin.type = PinType::Tensor;
            output_pin.name = "Output";
            output_pin.is_input = false;
            node.outputs.push_back(output_pin);

            // Optional output: Attention weights for visualization/debugging
            NodePin attn_weights_pin;
            attn_weights_pin.id = next_pin_id_++;
            attn_weights_pin.type = PinType::Tensor;
            attn_weights_pin.name = "Attn Weights";
            attn_weights_pin.is_input = false;
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
            node.inputs.push_back(input_pin);

            if (node.type == NodeType::TransformerDecoder) {
                NodePin memory_pin;
                memory_pin.id = next_pin_id_++;
                memory_pin.type = PinType::Tensor;
                memory_pin.name = "Memory";
                memory_pin.is_input = true;
                node.inputs.push_back(memory_pin);
            }

            NodePin output_pin;
            output_pin.id = next_pin_id_++;
            output_pin.type = PinType::Tensor;
            output_pin.name = "Output";
            output_pin.is_input = false;
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
            node.inputs.push_back(input_pin);

            NodePin output_pin;
            output_pin.id = next_pin_id_++;
            output_pin.type = PinType::Tensor;
            output_pin.name = "Output";
            output_pin.is_input = false;
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
            node.inputs.push_back(pred_pin);

            NodePin target_pin;
            target_pin.id = next_pin_id_++;
            target_pin.type = PinType::Tensor;
            target_pin.name = "Targets";
            target_pin.is_input = true;
            node.inputs.push_back(target_pin);

            NodePin loss_pin;
            loss_pin.id = next_pin_id_++;
            loss_pin.type = PinType::Loss;
            loss_pin.name = "Loss";
            loss_pin.is_input = false;
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
            node.parameters["tokenizer_type"] = "1"; // 0=Whitespace, 1=Word, 2=Character
            node.parameters["max_length"] = "512";
            node.parameters["lowercase"] = "true";
            node.parameters["padding"] = "true";
            node.parameters["truncation"] = "true";
            node.parameters["min_freq"] = "1";
            node.parameters["max_vocab_size"] = "-1";
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
            in.type = PinType::Tensor;
            in.name = "Sequential Data";
            in.is_input = true;
            node.inputs.push_back(in);
            NodePin out_x;
            out_x.id = next_pin_id_++;
            out_x.type = PinType::Tensor;
            out_x.name = "Windows";
            out_x.is_input = false;
            node.outputs.push_back(out_x);
            NodePin out_y;
            out_y.id = next_pin_id_++;
            out_y.type = PinType::Labels;
            out_y.name = "Targets";
            out_y.is_input = false;
            node.outputs.push_back(out_y);
            node.parameters["window_size"] = "10";
            node.parameters["forecast_horizon"] = "1";
            node.parameters["stride"] = "1";
            break;
        }

        case NodeType::TimeSeriesFeatures: {
            NodePin in;
            in.id = next_pin_id_++;
            in.type = PinType::Tensor;
            in.name = "Input";
            in.is_input = true;
            node.inputs.push_back(in);
            NodePin out;
            out.id = next_pin_id_++;
            out.type = PinType::Tensor;
            out.name = "Enriched";
            out.is_input = false;
            node.outputs.push_back(out);
            node.parameters["lag_values"] = "1,7,30";
            node.parameters["rolling_windows"] = "7,30";
            node.parameters["add_diff"] = "false";
            break;
        }

        case NodeType::TimeSeriesSplit: {
            NodePin in;
            in.id = next_pin_id_++;
            in.type = PinType::Tensor;
            in.name = "Data";
            in.is_input = true;
            node.inputs.push_back(in);
            NodePin train_out;
            train_out.id = next_pin_id_++;
            train_out.type = PinType::Tensor;
            train_out.name = "Train";
            train_out.is_input = false;
            node.outputs.push_back(train_out);
            NodePin val_out;
            val_out.id = next_pin_id_++;
            val_out.type = PinType::Tensor;
            val_out.name = "Validation";
            val_out.is_input = false;
            node.outputs.push_back(val_out);
            NodePin test_out;
            test_out.id = next_pin_id_++;
            test_out.type = PinType::Tensor;
            test_out.name = "Test";
            test_out.is_input = false;
            node.outputs.push_back(test_out);
            node.parameters["train_ratio"] = "0.7";
            node.parameters["val_ratio"] = "0.15";
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

        // ========== Data Transformation Nodes (Unified Canvas Phase 4) ==========

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
        case NodeType::ExportJSON: {
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

void NodeEditor::DeleteNode(int node_id) {
    // Delete node
    auto node_it = std::find_if(nodes_.begin(), nodes_.end(),
        [node_id](const MLNode& node) {
            return node.id == node_id;
        });

    if (node_it != nodes_.end()) {
        spdlog::info("Deleting node: {} (ID: {})", node_it->name, node_id);

        // Delete all links connected to this node
        links_.erase(
            std::remove_if(links_.begin(), links_.end(),
                [node_id](const NodeLink& link) {
                    return link.from_node == node_id || link.to_node == node_id;
                }),
            links_.end());

        nodes_.erase(node_it);
    }
}

void NodeEditor::ClearGraph() {
    SaveUndoState();

    // IMPORTANT: Clear properties panel selection BEFORE clearing nodes
    // to prevent dangling pointer access (the properties panel holds a raw pointer
    // to the selected node which becomes invalid after nodes_.clear())
    if (properties_panel_) {
        properties_panel_->ClearSelection();
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
            return IM_COL32(26, 188, 156, 255);

        // ===== Merge Operations - Lime Green =====
        case NodeType::Concatenate:
        case NodeType::Add:
        case NodeType::Multiply:
        case NodeType::Average:
            return IM_COL32(139, 195, 74, 255);

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

        case NodeType::PluginCustom:
            return IM_COL32(68, 136, 170, 255);

        default:
            return IM_COL32(127, 140, 141, 255);
    }
}

// ========== Icon Implementation ==========
const char* NodeEditor::GetNodeIcon(NodeType type) {
    switch (type) {
        // Data Pipeline
        case NodeType::DatasetInput:
            return ICON_FA_DATABASE;
        case NodeType::DataLoader:
            return ICON_FA_SPINNER;
        case NodeType::Augmentation:
            return ICON_FA_WAND_MAGIC_SPARKLES;
        case NodeType::DataSplit:
            return ICON_FA_SITEMAP;
        case NodeType::Normalize:
            return ICON_FA_CHART_BAR;
        case NodeType::OneHotEncode:
            return ICON_FA_BORDER_ALL;

        // Core Layers
        case NodeType::Dense:
            return ICON_FA_BARS;
        case NodeType::Flatten:
            return ICON_FA_LAYER_GROUP;

        // Convolutional Layers
        case NodeType::Conv1D:
        case NodeType::Conv2D:
        case NodeType::Conv3D:
        case NodeType::DepthwiseConv2D:
            return ICON_FA_TABLE;

        // Pooling Layers
        case NodeType::MaxPool2D:
        case NodeType::AvgPool2D:
        case NodeType::GlobalMaxPool:
        case NodeType::GlobalAvgPool:
        case NodeType::AdaptiveAvgPool:
            return ICON_FA_COMPRESS;

        // Normalization Layers
        case NodeType::BatchNorm:
        case NodeType::LayerNorm:
        case NodeType::GroupNorm:
        case NodeType::InstanceNorm:
            return ICON_FA_CHART_BAR;

        // Regularization
        case NodeType::Dropout:
            return ICON_FA_DICE;

        // Recurrent Layers
        case NodeType::RNN:
        case NodeType::LSTM:
        case NodeType::GRU:
        case NodeType::Bidirectional:
        case NodeType::TimeDistributed:
            return ICON_FA_ROTATE;
        case NodeType::Embedding:
            return ICON_FA_FONT;

        // Attention & Transformer
        case NodeType::MultiHeadAttention:
        case NodeType::SelfAttention:
        case NodeType::CrossAttention:
        case NodeType::LinearAttention:
            return ICON_FA_BULLSEYE;
        case NodeType::TransformerEncoder:
        case NodeType::TransformerDecoder:
            return ICON_FA_MICROCHIP;
        case NodeType::PositionalEncoding:
            return ICON_FA_HASHTAG;

        // Activation Functions
        case NodeType::ReLU:
        case NodeType::LeakyReLU:
        case NodeType::PReLU:
        case NodeType::ELU:
        case NodeType::SELU:
        case NodeType::GELU:
        case NodeType::Swish:
        case NodeType::Mish:
            return ICON_FA_BOLT;
        case NodeType::Sigmoid:
            return ICON_FA_SIGNAL;
        case NodeType::Tanh:
            return ICON_FA_WAVE_SQUARE;
        case NodeType::Softmax:
            return ICON_FA_CHART_PIE;

        // Shape Operations
        case NodeType::Reshape:
        case NodeType::TensorReshape:
        case NodeType::View:
            return ICON_FA_EXPAND;
        case NodeType::Permute:
            return ICON_FA_SHUFFLE;
        case NodeType::Squeeze:
        case NodeType::Unsqueeze:
            return ICON_FA_COMPRESS;
        case NodeType::Split:
            return ICON_FA_SITEMAP;

        // Merge Operations
        case NodeType::Concatenate:
            return ICON_FA_LINK;
        case NodeType::Add:
            return ICON_FA_PLUS;
        case NodeType::Multiply:
            return ICON_FA_XMARK;
        case NodeType::Average:
            return ICON_FA_CHART_LINE;

        // Output
        case NodeType::Output:
            return ICON_FA_CIRCLE_CHECK;

        // Loss Functions
        case NodeType::MSELoss:
        case NodeType::L1Loss:
        case NodeType::SmoothL1Loss:
        case NodeType::HuberLoss:
            return ICON_FA_CALCULATOR;
        case NodeType::CrossEntropyLoss:
        case NodeType::BCELoss:
        case NodeType::BCEWithLogits:
        case NodeType::NLLLoss:
            return ICON_FA_SCALE_BALANCED;

        // Optimizers
        case NodeType::SGD:
        case NodeType::Adam:
        case NodeType::AdamW:
        case NodeType::RMSprop:
        case NodeType::Adagrad:
        case NodeType::NAdam:
            return ICON_FA_SLIDERS;

        // Learning Rate Schedulers
        case NodeType::StepLR:
        case NodeType::CosineAnnealing:
        case NodeType::ReduceOnPlateau:
        case NodeType::ExponentialLR:
        case NodeType::WarmupScheduler:
            return ICON_FA_CHART_LINE;

        // Regularization Nodes
        case NodeType::L1Regularization:
        case NodeType::L2Regularization:
        case NodeType::ElasticNet:
            return ICON_FA_SHIELD;

        // Utility Nodes
        case NodeType::Lambda:
            return ICON_FA_CODE;
        case NodeType::Identity:
            return ICON_FA_EQUALS;
        case NodeType::Constant:
            return ICON_FA_CIRCLE;
        case NodeType::Parameter:
            return ICON_FA_GEAR;

        // Signal / Control
        case NodeType::SignalSlider:
            return ICON_FA_SLIDERS;
        case NodeType::SineWave:
            return ICON_FA_WAVE_SQUARE;
        case NodeType::StepSignal:
            return ICON_FA_WAVE_SQUARE;
        case NodeType::RampSignal:
            return ICON_FA_ARROW_TREND_UP;
        case NodeType::SignalScope:
            return ICON_FA_CHART_LINE;

        // Subgraph
        case NodeType::Subgraph:
            return ICON_FA_OBJECT_GROUP;

        // DNN Inference Nodes
        case NodeType::DNNModelLoad:
            return ICON_FA_BRAIN;
        case NodeType::DNNDetect:
        case NodeType::DNNFaceDetect:
            return ICON_FA_CROSSHAIRS;
        case NodeType::DNNClassify:
            return ICON_FA_TAGS;
        case NodeType::DNNPoseEstimate:
            return ICON_FA_USER;
        case NodeType::DNNPreprocess:
            return ICON_FA_WAND_MAGIC_SPARKLES;
        case NodeType::PretrainedYOLO:
        case NodeType::PretrainedMobileNet:
        case NodeType::PretrainedOpenPose:
        case NodeType::PretrainedFaceNet:
            return ICON_FA_BRAIN;
        case NodeType::NonMaxSuppression:
        case NodeType::ThresholdFilter:
            return ICON_FA_FILTER;
        case NodeType::ArgMax:
        case NodeType::TopK:
            return ICON_FA_ARROW_UP;

        // Text Processing
        case NodeType::TextTokenizer:
            return ICON_FA_FONT;
        case NodeType::TextVocabulary:
            return ICON_FA_BOOK;
        case NodeType::TextPadding:
            return ICON_FA_ALIGN_LEFT;

        // Upsampling
        case NodeType::ConvTranspose2D:
        case NodeType::Upsample:
        case NodeType::PixelShuffle:
            return ICON_FA_EXPAND;

        // Time-Series
        case NodeType::TimeSeriesWindow:
        case NodeType::TimeSeriesFeatures:
        case NodeType::TimeSeriesSplit:
            return ICON_FA_CHART_LINE;

        // Audio
        case NodeType::AudioInput:
        case NodeType::AudioAugmentation:
            return ICON_FA_MUSIC;
        case NodeType::Spectrogram:
        case NodeType::MelSpectrogram:
        case NodeType::MFCC:
            return ICON_FA_WAVE_SQUARE;

        // RL
        case NodeType::GymEnvironment:
            return ICON_FA_BULLSEYE;
        case NodeType::ReplayBufferNode:
            return ICON_FA_DATABASE;
        case NodeType::PolicyNetwork:
        case NodeType::ValueNetwork:
            return ICON_FA_BRAIN;
        case NodeType::RLTraining:
            return ICON_FA_CROSSHAIRS;

        case NodeType::PluginCustom:
            return ICON_FA_PLUG;

        default:
            return ICON_FA_CIRCLE_NODES;
    }
}

} // namespace gui
