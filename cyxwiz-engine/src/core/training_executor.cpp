#include "training_executor.h"
#include <spdlog/spdlog.h>
#include <cmath>
#include <algorithm>

namespace cyxwiz {

// ============================================================================
// BackendModel Implementation - Using cyxwiz-backend layers
// ============================================================================

void TrainingExecutor::BackendModel::Initialize(size_t input, size_t hidden, size_t output) {
    input_size = input;
    hidden_size = hidden;
    output_size = output;

    // Create layers using cyxwiz-backend
    fc1 = std::make_unique<LinearLayer>(input_size, hidden_size, true);
    relu1 = std::make_unique<ReLU>();
    fc2 = std::make_unique<LinearLayer>(hidden_size, output_size, true);

    // Create loss function
    loss_fn = std::make_unique<CrossEntropyLoss>();

    spdlog::info("BackendModel: Initialized MLP {} -> {} -> {} using cyxwiz-backend layers",
                 input_size, hidden_size, output_size);
}

Tensor TrainingExecutor::BackendModel::Forward(const Tensor& input) {
    // Cache input for backward pass
    input_cache = input.Clone();

    // Forward through layers: input -> fc1 -> relu -> fc2
    fc1_output = fc1->Forward(input);
    relu_output = relu1->Forward(fc1_output);
    fc2_output = fc2->Forward(relu_output);

    return fc2_output;
}

float TrainingExecutor::BackendModel::ComputeLoss(const Tensor& predictions, const Tensor& targets) {
    // Use CrossEntropyLoss from backend (includes softmax)
    Tensor loss_tensor = loss_fn->Forward(predictions, targets);
    const float* loss_data = loss_tensor.Data<float>();
    return loss_data[0];
}

void TrainingExecutor::BackendModel::Backward(const Tensor& predictions, const Tensor& targets) {
    // Compute loss gradient (softmax + cross-entropy combined)
    Tensor grad = loss_fn->Backward(predictions, targets);

    // Backprop through fc2
    Tensor grad_fc2 = fc2->Backward(grad);

    // Backprop through relu
    Tensor grad_relu = relu1->Backward(grad_fc2, fc1_output);

    // Backprop through fc1
    Tensor grad_fc1 = fc1->Backward(grad_relu);
}

void TrainingExecutor::BackendModel::UpdateWeights(float learning_rate) {
    // Get gradients from layers and update weights
    auto fc1_grads = fc1->GetGradients();
    auto fc2_grads = fc2->GetGradients();

    auto fc1_params = fc1->GetParameters();
    auto fc2_params = fc2->GetParameters();

    // Update fc1 weights
    if (fc1_grads.count("weight") && fc1_params.count("weight")) {
        Tensor& weight = fc1_params["weight"];
        const Tensor& grad = fc1_grads["weight"];
        float* w_data = weight.Data<float>();
        const float* g_data = grad.Data<float>();
        size_t n = weight.NumElements();
        for (size_t i = 0; i < n; ++i) {
            w_data[i] -= learning_rate * g_data[i];
        }
    }
    if (fc1_grads.count("bias") && fc1_params.count("bias")) {
        Tensor& bias = fc1_params["bias"];
        const Tensor& grad = fc1_grads["bias"];
        float* b_data = bias.Data<float>();
        const float* g_data = grad.Data<float>();
        size_t n = bias.NumElements();
        for (size_t i = 0; i < n; ++i) {
            b_data[i] -= learning_rate * g_data[i];
        }
    }

    // Update fc2 weights
    if (fc2_grads.count("weight") && fc2_params.count("weight")) {
        Tensor& weight = fc2_params["weight"];
        const Tensor& grad = fc2_grads["weight"];
        float* w_data = weight.Data<float>();
        const float* g_data = grad.Data<float>();
        size_t n = weight.NumElements();
        for (size_t i = 0; i < n; ++i) {
            w_data[i] -= learning_rate * g_data[i];
        }
    }
    if (fc2_grads.count("bias") && fc2_params.count("bias")) {
        Tensor& bias = fc2_params["bias"];
        const Tensor& grad = fc2_grads["bias"];
        float* b_data = bias.Data<float>();
        const float* g_data = grad.Data<float>();
        size_t n = bias.NumElements();
        for (size_t i = 0; i < n; ++i) {
            b_data[i] -= learning_rate * g_data[i];
        }
    }

    // Set updated parameters back
    fc1->SetParameters(fc1_params);
    fc2->SetParameters(fc2_params);
}

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

bool TrainingExecutor::Initialize(int batch_size) {
    // Create backend model
    model_ = std::make_unique<BackendModel>();

    // Determine hidden size from config (look for Dense layers)
    size_t hidden_size = 128;  // default
    for (const auto& layer : config_.layers) {
        if (layer.type == gui::NodeType::Dense && layer.units > 0) {
            // Use first Dense layer as hidden size (before output)
            if (static_cast<size_t>(layer.units) != config_.output_size) {
                hidden_size = layer.units;
                break;
            }
        }
    }

    model_->Initialize(config_.input_size, hidden_size, config_.output_size);

    // Create optimizer (for future use with backend optimizer)
    optimizer_ = CreateOptimizer(config_.GetOptimizerType(), config_.learning_rate);

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

    spdlog::info("TrainingExecutor: Starting training with cyxwiz-backend for {} epochs, batch_size={}",
                 epochs, batch_size);

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

        // Run validation
        RunValidation(val_batcher);

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

        // Forward pass using backend model
        Tensor predictions = model_->Forward(batch.data);

        // Compute loss
        float batch_loss = model_->ComputeLoss(predictions, batch.labels);
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
        model_->Backward(predictions, batch.labels);

        // Update weights
        model_->UpdateWeights(config_.learning_rate);

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
        Tensor predictions = model_->Forward(batch.data);

        // Compute loss
        float batch_loss = model_->ComputeLoss(predictions, batch.labels);
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

Tensor TrainingExecutor::Forward(const Tensor& input) {
    if (model_) {
        return model_->Forward(input);
    }
    return Tensor();
}

float TrainingExecutor::ComputeLoss(const Tensor& predictions, const Tensor& targets) {
    if (model_) {
        return model_->ComputeLoss(predictions, targets);
    }
    return 0.0f;
}

float TrainingExecutor::ComputeAccuracy(const Tensor& predictions, const Tensor& targets) {
    // Compute accuracy from predictions and targets
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
    if (model_) {
        model_->Backward(predictions, targets);
    }
}

void TrainingExecutor::PreprocessBatch(Batch& batch) {
    // Preprocessing is handled by DatasetBatcher
}

} // namespace cyxwiz
