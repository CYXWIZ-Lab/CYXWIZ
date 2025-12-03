#include "training_executor.h"
#include <spdlog/spdlog.h>
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
                // Output layer is a Dense layer to num_classes
                size_t out_features = config_.output_size;
                model_->Add<LinearModule>(current_input_size, out_features, true);
                spdlog::info("  [{}] Output Linear({} -> {})", i, current_input_size, out_features);
                current_input_size = out_features;
                break;
            }

            // Skip non-layer nodes
            case gui::NodeType::DatasetInput:
            case gui::NodeType::DataLoader:
            case gui::NodeType::Augmentation:
            case gui::NodeType::DataSplit:
            case gui::NodeType::TensorReshape:
            case gui::NodeType::Normalize:
            case gui::NodeType::OneHotEncode:
            case gui::NodeType::MSELoss:
            case gui::NodeType::CrossEntropyLoss:
            case gui::NodeType::SGD:
            case gui::NodeType::Adam:
            case gui::NodeType::AdamW:
                // These are not layers in the sequential model
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
            cross_entropy_loss_ = std::make_unique<CrossEntropyLoss>();
            spdlog::info("TrainingExecutor: Using CrossEntropy loss");
            break;
        case gui::NodeType::MSELoss:
            mse_loss_ = std::make_unique<MSELoss>();
            spdlog::info("TrainingExecutor: Using MSE loss");
            break;
        default:
            cross_entropy_loss_ = std::make_unique<CrossEntropyLoss>();
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
        train_batcher.SetNormalization(config_.preprocessing.norm_mean,
                                        config_.preprocessing.norm_std);
        val_batcher.SetNormalization(config_.preprocessing.norm_mean,
                                      config_.preprocessing.norm_std);
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
    Tensor loss_tensor;

    if (cross_entropy_loss_) {
        loss_tensor = cross_entropy_loss_->Forward(predictions, targets);
    } else if (mse_loss_) {
        loss_tensor = mse_loss_->Forward(predictions, targets);
    } else {
        spdlog::error("TrainingExecutor::ComputeLoss: No loss function");
        return 0.0f;
    }

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

    // Compute loss gradient
    Tensor grad;
    if (cross_entropy_loss_) {
        grad = cross_entropy_loss_->Backward(predictions, targets);
    } else if (mse_loss_) {
        grad = mse_loss_->Backward(predictions, targets);
    } else {
        spdlog::error("TrainingExecutor::Backward: No loss function");
        return;
    }

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
