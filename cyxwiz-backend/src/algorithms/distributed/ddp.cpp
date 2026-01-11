#include "cyxwiz/distributed/ddp.h"
#include <spdlog/spdlog.h>
#include <algorithm>
#include <numeric>

namespace cyxwiz {

// ========== Constructor / Destructor ==========

DistributedDataParallel::DistributedDataParallel(SequentialModel* model, DDPConfig config)
    : model_(model)
    , config_(config)
    , process_group_(config.process_group)
    , buckets_initialized_(false) {

    if (!model_) {
        spdlog::error("DistributedDataParallel: model cannot be null");
        return;
    }

    // Use default process group if none specified
    if (!process_group_) {
        process_group_ = GetDefaultProcessGroup();
    }

    if (!process_group_ || !process_group_->IsInitialized()) {
        spdlog::error("DistributedDataParallel: process group not initialized");
        return;
    }

    spdlog::info("DistributedDataParallel: rank {}/{}, bucket_size={}MB",
                 GetRank(), GetWorldSize(), config_.bucket_size_mb);

    // Broadcast parameters from rank 0 to ensure all ranks start with same weights
    if (config_.broadcast_parameters) {
        BroadcastParameters(0);
    }

    // Setup gradient buckets
    SetupBuckets();
}

DistributedDataParallel::~DistributedDataParallel() {
    // Nothing to clean up - we don't own the model or process group
}

// ========== Forward/Backward ==========

Tensor DistributedDataParallel::Forward(const Tensor& input) {
    if (!model_) {
        spdlog::error("DistributedDataParallel::Forward: model is null");
        return Tensor();
    }
    return model_->Forward(input);
}

Tensor DistributedDataParallel::Backward(const Tensor& grad_output) {
    if (!model_) {
        spdlog::error("DistributedDataParallel::Backward: model is null");
        return Tensor();
    }
    return model_->Backward(grad_output);
}

// ========== Gradient Synchronization ==========

void DistributedDataParallel::SetupBuckets() {
    if (!model_ || !process_group_) {
        return;
    }

    // Get all gradients to understand the structure
    auto grads = model_->GetGradients();
    if (grads.empty()) {
        spdlog::debug("DDP: No gradients found, skipping bucket setup");
        return;
    }

    // Calculate total gradient size and create buckets
    size_t bucket_size_bytes = config_.bucket_size_mb * 1024 * 1024;
    size_t bucket_size_elements = bucket_size_bytes / sizeof(float);

    buckets_.clear();
    GradientBucket current_bucket;

    // Collect parameters in reverse order (like PyTorch DDP)
    // This allows overlapping communication with backward computation
    std::vector<std::pair<std::string, size_t>> param_sizes;
    for (const auto& [name, tensor] : grads) {
        param_sizes.emplace_back(name, tensor.NumElements());
    }

    // Reverse order for better overlap with backward pass
    std::reverse(param_sizes.begin(), param_sizes.end());

    for (const auto& [name, size] : param_sizes) {
        // Check if adding this gradient would exceed bucket size
        if (current_bucket.total_elements > 0 &&
            current_bucket.total_elements + size > bucket_size_elements) {
            // Finalize current bucket
            current_bucket.buffer.resize(current_bucket.total_elements);
            buckets_.push_back(std::move(current_bucket));
            current_bucket = GradientBucket();
        }

        // Add to current bucket
        current_bucket.offsets.push_back(current_bucket.total_elements);
        current_bucket.sizes.push_back(size);
        current_bucket.param_names.push_back(name);
        current_bucket.total_elements += size;
    }

    // Don't forget the last bucket
    if (current_bucket.total_elements > 0) {
        current_bucket.buffer.resize(current_bucket.total_elements);
        buckets_.push_back(std::move(current_bucket));
    }

    buckets_initialized_ = true;

    spdlog::info("DDP: Created {} gradient buckets for {} parameters",
                 buckets_.size(), grads.size());

    for (size_t i = 0; i < buckets_.size(); ++i) {
        spdlog::debug("  Bucket {}: {} params, {} elements ({:.2f} MB)",
                      i, buckets_[i].param_names.size(),
                      buckets_[i].total_elements,
                      buckets_[i].total_elements * sizeof(float) / (1024.0 * 1024.0));
    }
}

void DistributedDataParallel::FuseGradientsToBuckets() {
    auto grads = model_->GetGradients();

    for (auto& bucket : buckets_) {
        for (size_t i = 0; i < bucket.param_names.size(); ++i) {
            const std::string& name = bucket.param_names[i];
            size_t offset = bucket.offsets[i];
            size_t size = bucket.sizes[i];

            auto it = grads.find(name);
            if (it == grads.end()) {
                spdlog::warn("DDP: Gradient '{}' not found", name);
                // Fill with zeros
                std::fill(bucket.buffer.begin() + offset,
                          bucket.buffer.begin() + offset + size, 0.0f);
                continue;
            }

            const Tensor& grad = it->second;
            if (grad.NumElements() != size) {
                spdlog::error("DDP: Gradient '{}' size mismatch: expected {}, got {}",
                              name, size, grad.NumElements());
                continue;
            }

            // Copy gradient data into bucket buffer
            const float* src = grad.Data<float>();
            std::copy(src, src + size, bucket.buffer.begin() + offset);
        }
    }
}

void DistributedDataParallel::UnfuseGradientsFromBuckets() {
    // This method extracts synced gradients from buckets
    // The gradients are returned via SyncGradients() return value
}

std::map<std::string, Tensor> DistributedDataParallel::SyncGradients() {
    if (!model_ || !process_group_) {
        spdlog::error("DDP::SyncGradients: not properly initialized");
        return {};
    }

    // Ensure buckets are set up
    if (!buckets_initialized_) {
        SetupBuckets();
    }

    // Get current gradients
    auto grads = model_->GetGradients();
    if (grads.empty()) {
        return grads;
    }

    // If world_size is 1, no synchronization needed
    if (GetWorldSize() <= 1) {
        return grads;
    }

    // Fuse gradients into buckets
    FuseGradientsToBuckets();

    // AllReduce each bucket
    for (auto& bucket : buckets_) {
        if (bucket.total_elements == 0) {
            continue;
        }

        // Create tensor wrapper for bucket buffer
        std::vector<size_t> shape = {bucket.total_elements};
        Tensor bucket_tensor(shape, bucket.buffer.data());

        // AllReduce with AVERAGE (SUM then divide by world_size)
        process_group_->AllReduce(bucket_tensor, ReduceOp::AVERAGE);

        // Copy back to bucket buffer (in case tensor made a copy)
        const float* reduced_data = bucket_tensor.Data<float>();
        std::copy(reduced_data, reduced_data + bucket.total_elements,
                  bucket.buffer.begin());
    }

    // Extract synced gradients from buckets back into map
    std::map<std::string, Tensor> synced_grads;

    for (const auto& bucket : buckets_) {
        for (size_t i = 0; i < bucket.param_names.size(); ++i) {
            const std::string& name = bucket.param_names[i];
            size_t offset = bucket.offsets[i];
            size_t size = bucket.sizes[i];

            // Create tensor from bucket buffer slice
            std::vector<size_t> shape = {size};
            Tensor grad_tensor(shape, bucket.buffer.data() + offset);
            synced_grads[name] = std::move(grad_tensor);
        }
    }

    return synced_grads;
}

void DistributedDataParallel::UpdateParameters(Optimizer* optimizer) {
    if (!model_) {
        spdlog::error("DDP::UpdateParameters: model is null");
        return;
    }

    if (!optimizer) {
        spdlog::error("DDP::UpdateParameters: optimizer is null");
        return;
    }

    // Get parameters
    auto params = model_->GetParameters();

    // Sync gradients across all ranks
    auto synced_grads = SyncGradients();

    // Apply optimizer step
    optimizer->Step(params, synced_grads);

    // Update model parameters
    model_->SetParameters(params);
}

void DistributedDataParallel::BroadcastParameters(int src_rank) {
    if (!model_ || !process_group_) {
        spdlog::error("DDP::BroadcastParameters: not properly initialized");
        return;
    }

    spdlog::debug("DDP: Broadcasting parameters from rank {}", src_rank);

    auto params = model_->GetParameters();

    for (auto& [name, tensor] : params) {
        process_group_->Broadcast(tensor, src_rank);
    }

    // Set the broadcasted parameters back to the model
    model_->SetParameters(params);

    // Barrier to ensure all ranks have the same parameters before continuing
    process_group_->Barrier();

    spdlog::debug("DDP: Parameter broadcast complete");
}

// ========== Accessors ==========

bool DistributedDataParallel::IsMaster() const {
    return GetRank() == 0;
}

int DistributedDataParallel::GetRank() const {
    if (process_group_) {
        return process_group_->GetRank();
    }
    return 0;
}

int DistributedDataParallel::GetWorldSize() const {
    if (process_group_) {
        return process_group_->GetWorldSize();
    }
    return 1;
}

} // namespace cyxwiz
