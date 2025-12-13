#include "training_executor.h"
#include <spdlog/spdlog.h>
#include <spdlog/fmt/fmt.h>
#include <cmath>
#include <algorithm>

namespace cyxwiz {

// ============================================================================
// TrainingExecutor Implementation
// ============================================================================

TrainingExecutor::TrainingExecutor(TrainingConfiguration config, DatasetHandle dataset)
    : config_(std::move(config))
    , dataset_(dataset)
{
    spdlog::info("TrainingExecutor: Created with {} layers, input_size={}, output_size={}",
                 config_.layers.size(), config_.input_size, config_.output_size);
}

TrainingExecutor::~TrainingExecutor() {
    Stop();
}

bool TrainingExecutor::BuildModelFromConfig() {
    model_ = std::make_unique<SequentialModel>();

    spdlog::info("TrainingExecutor: Building model from {} layer configs", config_.layers.size());

    // Track input size for each layer
    size_t current_input_size = config_.input_size;

    for (size_t i = 0; i < config_.layers.size(); ++i) {
        const auto& layer_cfg = config_.layers[i];

        switch (layer_cfg.type) {
            case gui::NodeType::Dense: {
                size_t out_features = layer_cfg.units > 0 ? layer_cfg.units : 64;
                model_->Add<LinearModule>(current_input_size, out_features, true);
                spdlog::info("  [{}] Linear({} -> {})", i, current_input_size, out_features);
                current_input_size = out_features;
                break;
            }

            case gui::NodeType::ReLU: {
                model_->Add<ReLUModule>();
                spdlog::info("  [{}] ReLU", i);
                break;
            }

            case gui::NodeType::Sigmoid: {
                model_->Add<SigmoidModule>();
                spdlog::info("  [{}] Sigmoid", i);
                break;
            }

            case gui::NodeType::Tanh: {
                model_->Add<TanhModule>();
                spdlog::info("  [{}] Tanh", i);
                break;
            }

            case gui::NodeType::LeakyReLU: {
                float slope = layer_cfg.negative_slope > 0 ? layer_cfg.negative_slope : 0.01f;
                model_->Add<LeakyReLUModule>(slope);
                spdlog::info("  [{}] LeakyReLU(slope={})", i, slope);
                break;
            }

            case gui::NodeType::ELU: {
                float alpha = layer_cfg.alpha > 0 ? layer_cfg.alpha : 1.0f;
                model_->Add<ELUModule>(alpha);
                spdlog::info("  [{}] ELU(alpha={})", i, alpha);
                break;
            }

            case gui::NodeType::GELU: {
                model_->Add<GELUModule>();
                spdlog::info("  [{}] GELU", i);
                break;
            }

            case gui::NodeType::Swish: {
                model_->Add<SwishModule>();
                spdlog::info("  [{}] Swish", i);
                break;
            }

            case gui::NodeType::Mish: {
                model_->Add<MishModule>();
                spdlog::info("  [{}] Mish", i);
                break;
            }

            case gui::NodeType::Softmax: {
                model_->Add<SoftmaxModule>();
                spdlog::info("  [{}] Softmax", i);
                break;
            }

            case gui::NodeType::Dropout: {
                float p = layer_cfg.dropout_rate > 0 ? layer_cfg.dropout_rate : 0.5f;
                model_->Add<DropoutModule>(p);
                spdlog::info("  [{}] Dropout(p={})", i, p);
                break;
            }

            case gui::NodeType::Flatten: {
                model_->Add<FlattenModule>(1);
                spdlog::info("  [{}] Flatten", i);
                break;
            }

            case gui::NodeType::Output: {
                // Output node is just a marker, not an actual layer
                // The actual output transformation is done by the preceding Dense layer
                spdlog::info("  [{}] Output (marker, no layer added)", i);
                break;
            }

            // Skip non-layer nodes (preprocessing, loss functions, optimizers)
            case gui::NodeType::DatasetInput:
            case gui::NodeType::DataLoader:
            case gui::NodeType::Augmentation:
            case gui::NodeType::DataSplit:
            case gui::NodeType::TensorReshape:
            case gui::NodeType::Normalize:
            case gui::NodeType::OneHotEncode:
            // Loss functions
            case gui::NodeType::MSELoss:
            case gui::NodeType::CrossEntropyLoss:
            case gui::NodeType::BCELoss:
            case gui::NodeType::BCEWithLogits:
            case gui::NodeType::L1Loss:
            case gui::NodeType::SmoothL1Loss:
            case gui::NodeType::HuberLoss:
            case gui::NodeType::NLLLoss:
            // Optimizers
            case gui::NodeType::SGD:
            case gui::NodeType::Adam:
            case gui::NodeType::AdamW:
                // These are not layers in the sequential model
                break;

            // CNN layers (not yet supported in SequentialModel, need CNN module wrappers)
            case gui::NodeType::Conv2D:
            case gui::NodeType::MaxPool2D:
            case gui::NodeType::AvgPool2D:
            case gui::NodeType::GlobalMaxPool:
            case gui::NodeType::GlobalAvgPool:
            case gui::NodeType::BatchNorm:
                spdlog::warn("  [{}] CNN layer {} not yet supported in SequentialModel",
                             i, static_cast<int>(layer_cfg.type));
                break;

            default:
                spdlog::warn("  [{}] Unknown layer type: {}", i, static_cast<int>(layer_cfg.type));
                break;
        }
    }

    if (model_->Size() == 0) {
        spdlog::error("TrainingExecutor: No layers were added to the model!");
        return false;
    }

    // Print model summary
    model_->Summary();

    return true;
}

bool TrainingExecutor::Initialize(int batch_size) {
    // Build model from configuration
    if (!BuildModelFromConfig()) {
        spdlog::error("TrainingExecutor: Failed to build model from config");
        return false;
    }

    // Create loss function based on config
    switch (config_.loss_type) {
        case gui::NodeType::CrossEntropyLoss:
            loss_ = CreateLoss(LossType::CrossEntropy);
            spdlog::info("TrainingExecutor: Using CrossEntropy loss");
            break;
        case gui::NodeType::MSELoss:
            loss_ = CreateLoss(LossType::MSE);
            spdlog::info("TrainingExecutor: Using MSE loss");
            break;
        case gui::NodeType::BCELoss:
            loss_ = CreateLoss(LossType::BinaryCrossEntropy);
            spdlog::info("TrainingExecutor: Using BCE loss");
            break;
        case gui::NodeType::BCEWithLogits:
            loss_ = CreateLoss(LossType::BCEWithLogits);
            spdlog::info("TrainingExecutor: Using BCEWithLogits loss");
            break;
        case gui::NodeType::L1Loss:
            loss_ = CreateLoss(LossType::L1);
            spdlog::info("TrainingExecutor: Using L1 loss");
            break;
        case gui::NodeType::SmoothL1Loss:
        case gui::NodeType::HuberLoss:
            loss_ = CreateLoss(LossType::SmoothL1);
            spdlog::info("TrainingExecutor: Using SmoothL1/Huber loss");
            break;
        case gui::NodeType::NLLLoss:
            loss_ = CreateLoss(LossType::NLLLoss);
            spdlog::info("TrainingExecutor: Using NLL loss");
            break;
        default:
            loss_ = CreateLoss(LossType::CrossEntropy);
            spdlog::info("TrainingExecutor: Defaulting to CrossEntropy loss");
            break;
    }

    // Create optimizer from backend
    optimizer_ = CreateOptimizer(config_.GetOptimizerType(), config_.learning_rate);

    spdlog::info("TrainingExecutor: Using {} optimizer with lr={}",
                 config_.GetOptimizerName(), config_.learning_rate);

    return true;
}

void TrainingExecutor::Train(
    int epochs,
    int batch_size,
    BatchCallback batch_cb,
    EpochCallback epoch_cb,
    TrainingCompleteCallback complete_cb)
{
    if (is_training_.load()) {
        spdlog::warn("TrainingExecutor: Already training");
        return;
    }

    is_training_.store(true);
    stop_requested_.store(false);
    is_paused_.store(false);

    // Initialize
    if (!Initialize(batch_size)) {
        spdlog::error("TrainingExecutor: Failed to initialize");
        is_training_.store(false);
        return;
    }

    // Setup metrics
    UpdateMetrics([epochs](TrainingMetrics& m) {
        m.total_epochs = epochs;
        m.current_epoch = 0;
        m.is_training = true;
        m.is_complete = false;
        m.status_message = "Starting training...";
        m.loss_history.clear();
        m.accuracy_history.clear();
        m.val_loss_history.clear();
        m.val_accuracy_history.clear();
    });

    // Create batchers
    DatasetBatcher train_batcher(dataset_, batch_size, DatasetSplit::Train, true, false);
    DatasetBatcher val_batcher(dataset_, batch_size, DatasetSplit::Validation, false, false);

    // Apply preprocessing settings
    if (config_.preprocessing.has_normalization) {
        spdlog::info("TrainingExecutor: Applying normalization (mean={}, std={})",
                     config_.preprocessing.norm_mean, config_.preprocessing.norm_std);
        train_batcher.SetNormalization(config_.preprocessing.norm_mean,
                                        config_.preprocessing.norm_std);
        val_batcher.SetNormalization(config_.preprocessing.norm_mean,
                                      config_.preprocessing.norm_std);
    } else {
        spdlog::info("TrainingExecutor: No normalization configured");
    }

    if (config_.preprocessing.has_onehot) {
        train_batcher.SetOneHotEncoding(config_.preprocessing.num_classes);
        val_batcher.SetOneHotEncoding(config_.preprocessing.num_classes);
    }

    // Flatten input for MLP
    train_batcher.SetFlatten(true);
    val_batcher.SetFlatten(true);

    spdlog::info("TrainingExecutor: Starting training for {} epochs, batch_size={}",
                 epochs, batch_size);

    // Set model to training mode
    model_->SetTraining(true);

    // Training loop
    for (int epoch = 1; epoch <= epochs; ++epoch) {
        if (ShouldStop()) break;
        WaitWhilePaused();

        auto epoch_start = std::chrono::steady_clock::now();

        UpdateMetrics([epoch](TrainingMetrics& m) {
            m.current_epoch = epoch;
            m.status_message = "Training epoch " + std::to_string(epoch) + "...";
        });

        // Run training epoch
        RunTrainingEpoch(train_batcher, epoch, batch_cb);

        if (ShouldStop()) break;

        // Run validation (eval mode)
        model_->SetTraining(false);
        RunValidation(val_batcher);
        model_->SetTraining(true);

        auto epoch_end = std::chrono::steady_clock::now();
        float epoch_time = std::chrono::duration<float>(epoch_end - epoch_start).count();

        // Get current metrics for callback
        TrainingMetrics current = GetMetrics();

        // Compute samples per second
        float samples_per_sec = train_batcher.GetNumSamples() / epoch_time;

        // Update history
        UpdateMetrics([&](TrainingMetrics& m) {
            m.epoch_time_seconds = epoch_time;
            m.samples_per_second = samples_per_sec;
            m.loss_history.push_back(m.train_loss);
            m.accuracy_history.push_back(m.train_accuracy);
            m.val_loss_history.push_back(m.val_loss);
            m.val_accuracy_history.push_back(m.val_accuracy);
        });

        // Epoch callback
        if (epoch_cb) {
            epoch_cb(epoch, current.train_loss, current.train_accuracy,
                     current.val_loss, current.val_accuracy, epoch_time);
        }

        spdlog::info("Epoch {}/{}: loss={:.4f}, acc={:.2f}%, val_loss={:.4f}, val_acc={:.2f}% ({:.1f}s, {:.0f} samples/sec)",
                     epoch, epochs, current.train_loss, current.train_accuracy * 100,
                     current.val_loss, current.val_accuracy * 100, epoch_time, samples_per_sec);

        // Reset batchers for next epoch
        train_batcher.Reset();
        val_batcher.Reset();
    }

    // Mark complete
    UpdateMetrics([](TrainingMetrics& m) {
        m.is_training = false;
        m.is_complete = true;
        m.status_message = "Training complete";
    });

    is_training_.store(false);

    // Complete callback
    if (complete_cb) {
        complete_cb(GetMetrics());
    }

    spdlog::info("TrainingExecutor: Training complete");
}

void TrainingExecutor::RunTrainingEpoch(
    DatasetBatcher& batcher,
    int epoch,
    BatchCallback batch_cb)
{
    float epoch_loss = 0.0f;
    int correct = 0;
    int total = 0;
    int batch_num = 0;

    size_t total_batches = batcher.GetNumBatches();

    UpdateMetrics([total_batches](TrainingMetrics& m) {
        m.total_batches = static_cast<int>(total_batches);
        m.current_batch = 0;
    });

    while (!batcher.IsEpochComplete()) {
        if (ShouldStop()) break;
        WaitWhilePaused();

        Batch batch = batcher.GetNextBatch();
        if (!batch.IsValid()) break;

        batch_num++;

        // Forward pass through model
        Tensor predictions = Forward(batch.data);

        // DEBUG: Log sample values for first batch of first epoch
        if (epoch == 1 && batch_num == 1) {
            const float* input_data = batch.data.Data<float>();
            const float* pred_data_debug = predictions.Data<float>();
            const float* target_data_debug = batch.labels.Data<float>();

            // Log input data range
            float min_input = input_data[0], max_input = input_data[0];
            size_t input_size = batch.data.Shape()[0] * batch.data.Shape()[1];
            for (size_t i = 1; i < std::min(input_size, size_t(1000)); ++i) {
                min_input = std::min(min_input, input_data[i]);
                max_input = std::max(max_input, input_data[i]);
            }
            spdlog::info("DEBUG: Input data range: [{:.4f}, {:.4f}]", min_input, max_input);

            // Log first sample prediction
            spdlog::info("DEBUG: First sample predictions:");
            std::string pred_str = "  [";
            for (size_t c = 0; c < config_.output_size; ++c) {
                pred_str += fmt::format("{:.4f}", pred_data_debug[c]);
                if (c < config_.output_size - 1) pred_str += ", ";
            }
            pred_str += "]";
            spdlog::info("{}", pred_str);

            // Log first sample target
            spdlog::info("DEBUG: First sample target:");
            std::string target_str = "  [";
            for (size_t c = 0; c < config_.output_size; ++c) {
                target_str += fmt::format("{:.1f}", target_data_debug[c]);
                if (c < config_.output_size - 1) target_str += ", ";
            }
            target_str += "]";
            spdlog::info("{}", target_str);
        }

        // Compute loss
        float batch_loss = ComputeLoss(predictions, batch.labels);
        epoch_loss += batch_loss;

        // Compute accuracy
        const float* pred_data = predictions.Data<float>();
        const float* target_data = batch.labels.Data<float>();

        for (size_t b = 0; b < batch.size; ++b) {
            int pred_class = 0, true_class = 0;
            float max_pred = pred_data[b * config_.output_size];
            float max_target = target_data[b * config_.output_size];

            for (size_t c = 1; c < config_.output_size; ++c) {
                if (pred_data[b * config_.output_size + c] > max_pred) {
                    max_pred = pred_data[b * config_.output_size + c];
                    pred_class = static_cast<int>(c);
                }
                if (target_data[b * config_.output_size + c] > max_target) {
                    max_target = target_data[b * config_.output_size + c];
                    true_class = static_cast<int>(c);
                }
            }
            if (pred_class == true_class) correct++;
            total++;
        }

        // Backward pass
        Backward(predictions, batch.labels);

        // Update weights using optimizer
        model_->UpdateParameters(optimizer_.get());

        // Update metrics
        float current_loss = epoch_loss / batch_num;
        float current_acc = static_cast<float>(correct) / total;

        UpdateMetrics([batch_num, current_loss, current_acc](TrainingMetrics& m) {
            m.current_batch = batch_num;
            m.train_loss = current_loss;
            m.train_accuracy = current_acc;
        });

        // Batch callback
        if (batch_cb) {
            batch_cb(epoch, batch_num, batch_loss, current_acc);
        }
    }

    // Final epoch metrics
    float final_loss = batch_num > 0 ? epoch_loss / batch_num : 0.0f;
    float final_acc = total > 0 ? static_cast<float>(correct) / total : 0.0f;

    UpdateMetrics([final_loss, final_acc](TrainingMetrics& m) {
        m.train_loss = final_loss;
        m.train_accuracy = final_acc;
    });
}

void TrainingExecutor::RunValidation(DatasetBatcher& batcher) {
    float val_loss = 0.0f;
    int correct = 0;
    int total = 0;
    int batch_num = 0;

    batcher.Reset();

    while (!batcher.IsEpochComplete()) {
        if (ShouldStop()) break;

        Batch batch = batcher.GetNextBatch();
        if (!batch.IsValid()) break;

        batch_num++;

        // Forward pass only (no backprop)
        Tensor predictions = Forward(batch.data);

        // Compute loss
        float batch_loss = ComputeLoss(predictions, batch.labels);
        val_loss += batch_loss;

        // Compute accuracy
        const float* pred_data = predictions.Data<float>();
        const float* target_data = batch.labels.Data<float>();

        for (size_t b = 0; b < batch.size; ++b) {
            int pred_class = 0, true_class = 0;
            float max_pred = pred_data[b * config_.output_size];
            float max_target = target_data[b * config_.output_size];

            for (size_t c = 1; c < config_.output_size; ++c) {
                if (pred_data[b * config_.output_size + c] > max_pred) {
                    max_pred = pred_data[b * config_.output_size + c];
                    pred_class = static_cast<int>(c);
                }
                if (target_data[b * config_.output_size + c] > max_target) {
                    max_target = target_data[b * config_.output_size + c];
                    true_class = static_cast<int>(c);
                }
            }
            if (pred_class == true_class) correct++;
            total++;
        }
    }

    float final_loss = batch_num > 0 ? val_loss / batch_num : 0.0f;
    float final_acc = total > 0 ? static_cast<float>(correct) / total : 0.0f;

    UpdateMetrics([final_loss, final_acc](TrainingMetrics& m) {
        m.val_loss = final_loss;
        m.val_accuracy = final_acc;
    });
}

Tensor TrainingExecutor::Forward(const Tensor& input) {
    if (!model_) {
        spdlog::error("TrainingExecutor::Forward: Model not initialized");
        return Tensor();
    }

    last_predictions_ = model_->Forward(input);
    return last_predictions_;
}

float TrainingExecutor::ComputeLoss(const Tensor& predictions, const Tensor& targets) {
    if (!loss_) {
        spdlog::error("TrainingExecutor::ComputeLoss: No loss function");
        return 0.0f;
    }

    Tensor loss_tensor = loss_->Forward(predictions, targets);
    const float* loss_data = loss_tensor.Data<float>();
    return loss_data[0];
}

float TrainingExecutor::ComputeAccuracy(const Tensor& predictions, const Tensor& targets) {
    const auto& shape = predictions.Shape();
    if (shape.size() != 2) return 0.0f;

    size_t batch_size = shape[0];
    size_t num_classes = shape[1];

    const float* pred_data = predictions.Data<float>();
    const float* target_data = targets.Data<float>();

    int correct = 0;
    for (size_t b = 0; b < batch_size; ++b) {
        int pred_class = 0, true_class = 0;
        float max_pred = pred_data[b * num_classes];
        float max_target = target_data[b * num_classes];

        for (size_t c = 1; c < num_classes; ++c) {
            if (pred_data[b * num_classes + c] > max_pred) {
                max_pred = pred_data[b * num_classes + c];
                pred_class = static_cast<int>(c);
            }
            if (target_data[b * num_classes + c] > max_target) {
                max_target = target_data[b * num_classes + c];
                true_class = static_cast<int>(c);
            }
        }
        if (pred_class == true_class) correct++;
    }

    return static_cast<float>(correct) / batch_size;
}

void TrainingExecutor::Backward(const Tensor& predictions, const Tensor& targets) {
    if (!model_) {
        spdlog::error("TrainingExecutor::Backward: Model not initialized");
        return;
    }

    if (!loss_) {
        spdlog::error("TrainingExecutor::Backward: No loss function");
        return;
    }

    // Compute loss gradient
    Tensor grad = loss_->Backward(predictions, targets);

    // Backward through model
    model_->Backward(grad);
}

void TrainingExecutor::Stop() {
    stop_requested_.store(true);
    is_paused_.store(false);  // Unpause so thread can exit
}

void TrainingExecutor::Pause() {
    is_paused_.store(true);
    UpdateMetrics([](TrainingMetrics& m) {
        m.is_paused = true;
        m.status_message = "Training paused";
    });
}

void TrainingExecutor::Resume() {
    is_paused_.store(false);
    UpdateMetrics([](TrainingMetrics& m) {
        m.is_paused = false;
        m.status_message = "Training resumed";
    });
}

TrainingMetrics TrainingExecutor::GetMetrics() const {
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    return metrics_;
}

void TrainingExecutor::UpdateMetrics(const std::function<void(TrainingMetrics&)>& updater) {
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    updater(metrics_);
}

void TrainingExecutor::WaitWhilePaused() {
    while (is_paused_.load() && !stop_requested_.load()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}

void TrainingExecutor::PreprocessBatch(Batch& batch) {
    // Preprocessing is handled by DatasetBatcher
}

} // namespace cyxwiz
