#pragma once

#include "../api_export.h"
#include "process_group.h"
#include "../sequential.h"
#include "../optimizer.h"
#include <memory>
#include <vector>
#include <string>
#include <map>

namespace cyxwiz {

/**
 * Configuration for DistributedDataParallel
 */
struct CYXWIZ_API DDPConfig {
    /// Broadcast parameters from rank 0 to all ranks at initialization
    bool broadcast_parameters = true;

    /// Size of gradient buckets in MB for fused AllReduce (default: 25MB like PyTorch)
    size_t bucket_size_mb = 25;

    /// Warn about parameters that don't receive gradients
    bool find_unused_parameters = false;

    /// Process group to use (nullptr = use default global process group)
    ProcessGroup* process_group = nullptr;
};

/**
 * DistributedDataParallel - Model wrapper for data parallel training
 *
 * DDP wraps a SequentialModel to synchronize gradients across multiple processes
 * during distributed training. Each process has a complete copy of the model and
 * processes different data batches. After backward pass, gradients are averaged
 * across all processes using AllReduce.
 *
 * Usage:
 *   // Initialize distributed
 *   init_distributed();
 *
 *   // Create model
 *   SequentialModel model;
 *   model.Add<LinearModule>(784, 256);
 *   model.Add<ReLUModule>();
 *   model.Add<LinearModule>(256, 10);
 *
 *   // Wrap in DDP
 *   DDPConfig config;
 *   DistributedDataParallel ddp(&model, config);
 *
 *   // Training loop
 *   for (epoch...) {
 *       Tensor output = ddp.Forward(input);
 *       Tensor loss = loss_fn.Forward(output, target);
 *       Tensor grad = loss_fn.Backward(output, target);
 *       ddp.Backward(grad);
 *       ddp.SyncGradients();  // AllReduce gradients
 *       model.UpdateParameters(optimizer);
 *   }
 *
 *   finalize_distributed();
 *
 * Key concepts:
 *   - Forward/Backward just delegate to the wrapped model
 *   - SyncGradients() performs AllReduce on all gradients
 *   - Gradient bucketing: small gradients are fused into larger buffers for efficiency
 *   - BroadcastParameters() ensures all ranks start with same parameters
 */
class CYXWIZ_API DistributedDataParallel {
public:
    /**
     * Wrap a SequentialModel for distributed training
     *
     * @param model The model to wrap (not owned, must outlive DDP)
     * @param config DDP configuration
     *
     * If config.broadcast_parameters is true (default), parameters are broadcast
     * from rank 0 to all other ranks during construction.
     */
    DistributedDataParallel(SequentialModel* model, DDPConfig config = {});

    ~DistributedDataParallel();

    // Non-copyable
    DistributedDataParallel(const DistributedDataParallel&) = delete;
    DistributedDataParallel& operator=(const DistributedDataParallel&) = delete;

    // ========== Forward/Backward ==========

    /**
     * Forward pass - delegates to wrapped model
     */
    Tensor Forward(const Tensor& input);

    /**
     * Backward pass - delegates to wrapped model
     * Gradients are stored in each layer after this call.
     * Call SyncGradients() before optimizer step.
     */
    Tensor Backward(const Tensor& grad_output);

    // ========== Gradient Synchronization ==========

    /**
     * Synchronize gradients across all ranks using AllReduce
     *
     * Call this after Backward() and before UpdateParameters().
     * Uses gradient bucketing for efficiency (fuses small gradients into larger buffers).
     *
     * After this call, all ranks have the averaged gradients.
     *
     * @return Map of synchronized gradients (averaged across all ranks)
     */
    std::map<std::string, Tensor> SyncGradients();

    /**
     * Synchronize gradients and update parameters in one call
     *
     * This is the recommended way to update parameters in distributed training.
     * It combines SyncGradients() + optimizer step + SetParameters().
     *
     * @param optimizer The optimizer to use for parameter updates
     */
    void UpdateParameters(Optimizer* optimizer);

    /**
     * Broadcast parameters from source rank to all other ranks
     *
     * Called automatically during construction if config.broadcast_parameters is true.
     * Can be called manually to re-sync parameters (e.g., after loading checkpoint).
     *
     * @param src_rank Rank to broadcast from (default: 0)
     */
    void BroadcastParameters(int src_rank = 0);

    // ========== Accessors ==========

    /**
     * Get the wrapped model
     */
    SequentialModel* GetModel() { return model_; }
    const SequentialModel* GetModel() const { return model_; }

    /**
     * Check if this is the master rank (rank 0)
     */
    bool IsMaster() const;

    /**
     * Get rank of this process
     */
    int GetRank() const;

    /**
     * Get total number of processes
     */
    int GetWorldSize() const;

    /**
     * Get process group being used
     */
    ProcessGroup* GetProcessGroup() { return process_group_; }

private:
    SequentialModel* model_;
    DDPConfig config_;
    ProcessGroup* process_group_;

    // ========== Gradient Bucketing ==========

    /**
     * A bucket holds multiple gradient tensors fused into a single buffer
     * for efficient AllReduce communication.
     */
    struct GradientBucket {
        std::vector<std::string> param_names;  // Names of parameters in this bucket
        std::vector<size_t> offsets;           // Offset of each gradient in buffer
        std::vector<size_t> sizes;             // Size of each gradient
        std::vector<float> buffer;             // Fused gradient buffer
        size_t total_elements = 0;
    };

    std::vector<GradientBucket> buckets_;
    bool buckets_initialized_ = false;

    /**
     * Setup gradient buckets based on model parameters
     * Groups parameters into buckets of approximately bucket_size_mb
     */
    void SetupBuckets();

    /**
     * Copy gradients from model into bucket buffers
     */
    void FuseGradientsToBuckets();

    /**
     * Copy reduced gradients from bucket buffers back to model
     */
    void UnfuseGradientsFromBuckets();

    /**
     * Update model gradients with the values from buckets
     * Uses a map-based approach to handle the gradient update
     */
    void UpdateModelGradients(const std::map<std::string, Tensor>& synced_gradients);
};

} // namespace cyxwiz
