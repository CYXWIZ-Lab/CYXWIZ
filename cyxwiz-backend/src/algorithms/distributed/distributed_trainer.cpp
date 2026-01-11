#include "cyxwiz/distributed/distributed_trainer.h"
#include <spdlog/spdlog.h>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <fstream>

namespace cyxwiz {

// ========== Constructor / Destructor ==========

DistributedTrainer::DistributedTrainer(SequentialModel* model, Loss* loss,
                                       Optimizer* optimizer, ProcessGroup* pg)
    : model_(model)
    , loss_(loss)
    , optimizer_(optimizer)
    , process_group_(pg) {

    if (!model_ || !loss_ || !optimizer_) {
        spdlog::error("DistributedTrainer: model, loss, and optimizer must not be null");
        return;
    }

    // Use default process group if none specified
    if (!process_group_) {
        process_group_ = GetDefaultProcessGroup();
    }

    // Create DDP wrapper
    DDPConfig ddp_config;
    ddp_config.process_group = process_group_;
    ddp_config.broadcast_parameters = true;

    ddp_ = std::make_unique<DistributedDataParallel>(model_, ddp_config);

    spdlog::info("DistributedTrainer: initialized rank {}/{}",
                 GetRank(), GetWorldSize());
}

DistributedTrainer::~DistributedTrainer() = default;

// ========== Training ==========

DistributedTrainingHistory DistributedTrainer::Fit(const Tensor& X_train,
                                                   const Tensor& y_train,
                                                   const DistributedTrainingConfig& config) {
    return Fit(X_train, y_train, Tensor(), Tensor(), config);
}

DistributedTrainingHistory DistributedTrainer::Fit(const Tensor& X_train,
                                                   const Tensor& y_train,
                                                   const Tensor& X_val,
                                                   const Tensor& y_val,
                                                   const DistributedTrainingConfig& config) {
    DistributedTrainingHistory history;
    history.world_size = GetWorldSize();
    history.effective_batch_size = config.batch_size * GetWorldSize();

    if (!ddp_ || !model_ || !loss_ || !optimizer_) {
        spdlog::error("DistributedTrainer::Fit: not properly initialized");
        return history;
    }

    // Get dataset size from first dimension
    const auto& shape = X_train.Shape();
    if (shape.empty()) {
        spdlog::error("DistributedTrainer::Fit: X_train is empty");
        return history;
    }
    size_t num_samples = shape[0];

    bool has_validation = X_val.NumElements() > 0 && y_val.NumElements() > 0;

    // Create distributed sampler
    DistributedSampler sampler(num_samples, config.shuffle, config.seed);

    // Set model to training mode
    model_->SetTraining(true);

    auto start_time = std::chrono::high_resolution_clock::now();
    size_t total_samples_processed = 0;

    if (config.verbose && IsMaster()) {
        spdlog::info("Starting distributed training:");
        spdlog::info("  Epochs: {}", config.epochs);
        spdlog::info("  Batch size per rank: {}", config.batch_size);
        spdlog::info("  Effective batch size: {}", history.effective_batch_size);
        spdlog::info("  Dataset size: {}", num_samples);
        spdlog::info("  Samples per rank: {}", sampler.LocalSize());
        spdlog::info("  World size: {}", GetWorldSize());
    }

    for (int epoch = 0; epoch < config.epochs; ++epoch) {
        // Train one epoch
        auto [train_loss, train_acc] = TrainEpoch(X_train, y_train, sampler,
                                                   config, epoch);

        history.train_losses.push_back(train_loss);
        history.train_accuracies.push_back(train_acc);
        total_samples_processed += num_samples;

        // Validation
        float val_loss = 0.0f, val_acc = 0.0f;
        if (has_validation) {
            model_->SetTraining(false);
            std::tie(val_loss, val_acc) = ValidateEpoch(X_val, y_val, config);
            model_->SetTraining(true);

            history.val_losses.push_back(val_loss);
            history.val_accuracies.push_back(val_acc);
        }

        // Logging
        if (config.verbose && IsMaster()) {
            if (has_validation) {
                spdlog::info("Epoch {}/{}: train_loss={:.4f}, train_acc={:.4f}, "
                             "val_loss={:.4f}, val_acc={:.4f}",
                             epoch + 1, config.epochs, train_loss, train_acc,
                             val_loss, val_acc);
            } else {
                spdlog::info("Epoch {}/{}: train_loss={:.4f}, train_acc={:.4f}",
                             epoch + 1, config.epochs, train_loss, train_acc);
            }
        }

        // Epoch callback
        if (epoch_callback_) {
            epoch_callback_(epoch, train_loss, train_acc);
        }

        // Checkpointing
        if (config.checkpoint_every_n_epochs > 0 &&
            (epoch + 1) % config.checkpoint_every_n_epochs == 0) {
            std::string path = config.checkpoint_dir + "/checkpoint_epoch_" +
                               std::to_string(epoch + 1) + ".bin";
            SaveCheckpoint(path);
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;

    history.total_time_seconds = elapsed.count();
    history.samples_per_second = static_cast<double>(total_samples_processed) /
                                 elapsed.count();

    if (config.verbose && IsMaster()) {
        spdlog::info("Training complete:");
        spdlog::info("  Total time: {:.2f} seconds", history.total_time_seconds);
        spdlog::info("  Throughput: {:.2f} samples/sec", history.samples_per_second);
    }

    return history;
}

std::pair<float, float> DistributedTrainer::TrainEpoch(
    const Tensor& X_train, const Tensor& y_train,
    DistributedSampler& sampler,
    const DistributedTrainingConfig& config,
    int epoch) {

    sampler.SetEpoch(epoch);

    DistributedBatchIterator batch_iter(sampler, config.batch_size);

    float epoch_loss = 0.0f;
    float epoch_correct = 0.0f;
    size_t epoch_samples = 0;
    size_t batch_idx = 0;
    size_t total_batches = batch_iter.NumBatches();

    while (batch_iter.HasNext()) {
        auto indices = batch_iter.Next();
        if (indices.empty()) continue;

        // Extract batch
        auto [X_batch, y_batch] = ExtractBatch(X_train, y_train, indices);

        // Forward pass
        Tensor predictions = ddp_->Forward(X_batch);

        // Compute loss
        Tensor loss_tensor = loss_->Forward(predictions, y_batch);
        float batch_loss = 0.0f;
        if (loss_tensor.NumElements() > 0) {
            batch_loss = loss_tensor.Data<float>()[0];
        }

        // Backward pass
        Tensor loss_grad = loss_->Backward(predictions, y_batch);
        ddp_->Backward(loss_grad);

        // Update parameters (includes gradient sync)
        ddp_->UpdateParameters(optimizer_);

        // Compute accuracy
        float batch_correct = ComputeAccuracy(predictions, y_batch) * indices.size();

        epoch_loss += batch_loss * indices.size();
        epoch_correct += batch_correct;
        epoch_samples += indices.size();
        ++batch_idx;

        // Batch callback
        if (batch_callback_) {
            batch_callback_(static_cast<int>(batch_idx),
                            static_cast<int>(total_batches), batch_loss);
        }

        // Batch logging
        if (config.log_every_n_batches > 0 &&
            batch_idx % config.log_every_n_batches == 0 && IsMaster()) {
            spdlog::info("  Batch {}/{}: loss={:.4f}",
                         batch_idx, total_batches, batch_loss);
        }
    }

    // Aggregate metrics across ranks
    float total_loss = AggregateMetric(epoch_loss);
    float total_correct = AggregateMetric(epoch_correct);
    size_t total_samples = AggregateSampleCount(epoch_samples);

    float avg_loss = total_samples > 0 ? total_loss / total_samples : 0.0f;
    float avg_acc = total_samples > 0 ? total_correct / total_samples : 0.0f;

    return {avg_loss, avg_acc};
}

std::pair<float, float> DistributedTrainer::ValidateEpoch(
    const Tensor& X_val, const Tensor& y_val,
    const DistributedTrainingConfig& config) {

    const auto& shape = X_val.Shape();
    if (shape.empty()) {
        return {0.0f, 0.0f};
    }
    size_t num_samples = shape[0];

    // Create sampler for validation (no shuffle)
    DistributedSampler sampler(num_samples, false, 0);
    DistributedBatchIterator batch_iter(sampler, config.batch_size);

    float epoch_loss = 0.0f;
    float epoch_correct = 0.0f;
    size_t epoch_samples = 0;

    while (batch_iter.HasNext()) {
        auto indices = batch_iter.Next();
        if (indices.empty()) continue;

        auto [X_batch, y_batch] = ExtractBatch(X_val, y_val, indices);

        // Forward pass only
        Tensor predictions = model_->Forward(X_batch);

        // Compute loss
        Tensor loss_tensor = loss_->Forward(predictions, y_batch);
        float batch_loss = 0.0f;
        if (loss_tensor.NumElements() > 0) {
            batch_loss = loss_tensor.Data<float>()[0];
        }

        float batch_correct = ComputeAccuracy(predictions, y_batch) * indices.size();

        epoch_loss += batch_loss * indices.size();
        epoch_correct += batch_correct;
        epoch_samples += indices.size();
    }

    // Aggregate across ranks
    float total_loss = AggregateMetric(epoch_loss);
    float total_correct = AggregateMetric(epoch_correct);
    size_t total_samples = AggregateSampleCount(epoch_samples);

    float avg_loss = total_samples > 0 ? total_loss / total_samples : 0.0f;
    float avg_acc = total_samples > 0 ? total_correct / total_samples : 0.0f;

    return {avg_loss, avg_acc};
}

std::pair<float, float> DistributedTrainer::Evaluate(const Tensor& X_test,
                                                     const Tensor& y_test) {
    model_->SetTraining(false);
    DistributedTrainingConfig config;
    config.batch_size = 64;
    auto result = ValidateEpoch(X_test, y_test, config);
    model_->SetTraining(true);
    return result;
}

// ========== Helpers ==========

float DistributedTrainer::AggregateMetric(float local_value) {
    if (!process_group_ || GetWorldSize() <= 1) {
        return local_value;
    }

    // Create tensor with single value
    std::vector<size_t> shape = {1};
    Tensor value_tensor(shape, &local_value);

    // AllReduce with SUM
    process_group_->AllReduce(value_tensor, ReduceOp::SUM);

    return value_tensor.Data<float>()[0];
}

size_t DistributedTrainer::AggregateSampleCount(size_t local_count) {
    if (!process_group_ || GetWorldSize() <= 1) {
        return local_count;
    }

    float count_f = static_cast<float>(local_count);
    std::vector<size_t> shape = {1};
    Tensor count_tensor(shape, &count_f);

    process_group_->AllReduce(count_tensor, ReduceOp::SUM);

    return static_cast<size_t>(count_tensor.Data<float>()[0]);
}

std::pair<Tensor, Tensor> DistributedTrainer::ExtractBatch(
    const Tensor& X, const Tensor& y,
    const std::vector<size_t>& indices) {

    if (indices.empty()) {
        return {Tensor(), Tensor()};
    }

    const auto& X_shape = X.Shape();
    const auto& y_shape = y.Shape();

    if (X_shape.empty() || y_shape.empty()) {
        return {Tensor(), Tensor()};
    }

    size_t num_samples = X_shape[0];
    size_t sample_size = X.NumElements() / num_samples;
    size_t label_size = y.NumElements() / num_samples;

    size_t batch_size = indices.size();

    // Create batch tensors
    std::vector<size_t> batch_X_shape = X_shape;
    batch_X_shape[0] = batch_size;

    std::vector<size_t> batch_y_shape = y_shape;
    batch_y_shape[0] = batch_size;

    Tensor X_batch(batch_X_shape);
    Tensor y_batch(batch_y_shape);

    const float* X_data = X.Data<float>();
    const float* y_data = y.Data<float>();
    float* X_batch_data = X_batch.Data<float>();
    float* y_batch_data = y_batch.Data<float>();

    // Copy data for each sample in batch
    for (size_t i = 0; i < batch_size; ++i) {
        size_t idx = indices[i];
        if (idx >= num_samples) {
            idx = idx % num_samples;  // Handle padding
        }

        std::copy(X_data + idx * sample_size,
                  X_data + (idx + 1) * sample_size,
                  X_batch_data + i * sample_size);

        std::copy(y_data + idx * label_size,
                  y_data + (idx + 1) * label_size,
                  y_batch_data + i * label_size);
    }

    return {std::move(X_batch), std::move(y_batch)};
}

float DistributedTrainer::ComputeAccuracy(const Tensor& predictions,
                                          const Tensor& targets) {
    // Simple accuracy: argmax(predictions) == targets
    // Works for classification tasks

    const auto& pred_shape = predictions.Shape();
    if (pred_shape.empty()) {
        return 0.0f;
    }

    size_t batch_size = pred_shape[0];
    size_t num_classes = pred_shape.size() > 1 ? pred_shape[1] : 1;

    const float* pred_data = predictions.Data<float>();
    const float* target_data = targets.Data<float>();

    size_t correct = 0;

    for (size_t i = 0; i < batch_size; ++i) {
        // Find argmax for this sample
        size_t pred_class = 0;
        float max_val = pred_data[i * num_classes];

        for (size_t c = 1; c < num_classes; ++c) {
            float val = pred_data[i * num_classes + c];
            if (val > max_val) {
                max_val = val;
                pred_class = c;
            }
        }

        // Get target class (assume integer or one-hot)
        size_t target_class;
        if (targets.Shape().size() > 1 && targets.Shape()[1] > 1) {
            // One-hot encoded
            target_class = 0;
            float max_target = target_data[i * num_classes];
            for (size_t c = 1; c < num_classes; ++c) {
                if (target_data[i * num_classes + c] > max_target) {
                    max_target = target_data[i * num_classes + c];
                    target_class = c;
                }
            }
        } else {
            // Integer label
            target_class = static_cast<size_t>(target_data[i]);
        }

        if (pred_class == target_class) {
            ++correct;
        }
    }

    return static_cast<float>(correct) / static_cast<float>(batch_size);
}

// ========== Checkpointing ==========

void DistributedTrainer::SaveCheckpoint(const std::string& path) {
    // Only master saves by default
    if (!IsMaster()) {
        return;
    }

    spdlog::info("Saving checkpoint to: {}", path);

    // Get model parameters
    auto params = model_->GetParameters();

    // Simple binary format: num_params, then for each: name_len, name, data_size, data
    std::ofstream file(path, std::ios::binary);
    if (!file) {
        spdlog::error("Failed to open checkpoint file: {}", path);
        return;
    }

    size_t num_params = params.size();
    file.write(reinterpret_cast<const char*>(&num_params), sizeof(num_params));

    for (const auto& [name, tensor] : params) {
        size_t name_len = name.size();
        file.write(reinterpret_cast<const char*>(&name_len), sizeof(name_len));
        file.write(name.c_str(), name_len);

        size_t data_size = tensor.NumElements();
        file.write(reinterpret_cast<const char*>(&data_size), sizeof(data_size));
        file.write(reinterpret_cast<const char*>(tensor.Data<float>()),
                   data_size * sizeof(float));
    }

    spdlog::info("Checkpoint saved: {} parameters", num_params);
}

void DistributedTrainer::LoadCheckpoint(const std::string& path) {
    spdlog::info("Loading checkpoint from: {}", path);

    std::ifstream file(path, std::ios::binary);
    if (!file) {
        spdlog::error("Failed to open checkpoint file: {}", path);
        return;
    }

    size_t num_params;
    file.read(reinterpret_cast<char*>(&num_params), sizeof(num_params));

    std::map<std::string, Tensor> params;

    for (size_t i = 0; i < num_params; ++i) {
        size_t name_len;
        file.read(reinterpret_cast<char*>(&name_len), sizeof(name_len));

        std::string name(name_len, '\0');
        file.read(&name[0], name_len);

        size_t data_size;
        file.read(reinterpret_cast<char*>(&data_size), sizeof(data_size));

        std::vector<size_t> shape = {data_size};
        Tensor tensor(shape);
        file.read(reinterpret_cast<char*>(tensor.Data<float>()),
                  data_size * sizeof(float));

        params[name] = std::move(tensor);
    }

    model_->SetParameters(params);

    // Sync parameters across all ranks
    if (ddp_) {
        ddp_->BroadcastParameters(0);
    }

    spdlog::info("Checkpoint loaded: {} parameters", num_params);
}

// ========== Accessors ==========

bool DistributedTrainer::IsMaster() const {
    return GetRank() == 0;
}

int DistributedTrainer::GetRank() const {
    if (ddp_) {
        return ddp_->GetRank();
    }
    return 0;
}

int DistributedTrainer::GetWorldSize() const {
    if (ddp_) {
        return ddp_->GetWorldSize();
    }
    return 1;
}

} // namespace cyxwiz
