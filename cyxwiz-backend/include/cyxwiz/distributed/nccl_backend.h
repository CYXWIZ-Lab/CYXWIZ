#pragma once

// NCCL backend is only available when CYXWIZ_HAS_NCCL is defined
#ifdef CYXWIZ_HAS_NCCL

#include "process_group.h"
#include <nccl.h>
#include <cuda_runtime.h>
#include <vector>
#include <memory>
#include <atomic>
#include <mutex>

namespace cyxwiz {

/**
 * NCCLProcessGroup - High-performance GPU communication using NVIDIA NCCL
 *
 * NCCL (NVIDIA Collective Communications Library) provides optimized
 * collective communication primitives for multi-GPU and multi-node training.
 *
 * Key features:
 *   - GPU-direct communication (no CPU involvement)
 *   - NVLink and InfiniBand support
 *   - Asynchronous operations with CUDA streams
 *   - Tree and ring algorithms for optimal performance
 *
 * Requirements:
 *   - NVIDIA GPU with CUDA support
 *   - NCCL library installed
 *   - CUDA toolkit
 *
 * Usage:
 *   DistributedConfig config;
 *   config.backend = BackendType::NCCL;
 *   config.rank = 0;
 *   config.world_size = 4;
 *
 *   NCCLProcessGroup pg;
 *   pg.Initialize(config);
 *
 *   Tensor tensor = ...;  // Must be on GPU
 *   pg.AllReduce(tensor, ReduceOp::SUM);
 *
 *   pg.Finalize();
 *
 * Bootstrap process:
 *   1. Rank 0 creates ncclUniqueId
 *   2. Rank 0 broadcasts ID to all ranks via TCP
 *   3. All ranks call ncclCommInitRank with the ID
 *   4. NCCL establishes GPU-direct communication channels
 */
class CYXWIZ_API NCCLProcessGroup : public ProcessGroup {
public:
    NCCLProcessGroup();
    ~NCCLProcessGroup() override;

    // Non-copyable
    NCCLProcessGroup(const NCCLProcessGroup&) = delete;
    NCCLProcessGroup& operator=(const NCCLProcessGroup&) = delete;

    // ========== Lifecycle ==========

    /**
     * Initialize NCCL process group
     *
     * Bootstrap sequence:
     *   1. Rank 0: Generate ncclUniqueId
     *   2. All ranks: Exchange ID via TCP (master_addr:master_port)
     *   3. All ranks: ncclCommInitRank()
     *   4. Set CUDA device based on local_rank
     *
     * @param config Must have valid rank, world_size, master_addr, master_port
     * @return true on success
     */
    bool Initialize(const DistributedConfig& config) override;

    void Finalize() override;
    bool IsInitialized() const override { return initialized_; }

    // ========== Rank Information ==========

    int GetRank() const override { return rank_; }
    int GetWorldSize() const override { return world_size_; }
    int GetLocalRank() const override { return local_rank_; }

    // ========== Collective Operations ==========

    /**
     * GPU AllReduce using NCCL
     * Tensor must be on GPU (CUDA memory)
     */
    void AllReduce(Tensor& tensor, ReduceOp op = ReduceOp::SUM) override;

    /**
     * Batched AllReduce - more efficient for multiple small tensors
     */
    void AllReduceMultiple(std::vector<Tensor*>& tensors, ReduceOp op = ReduceOp::SUM) override;

    /**
     * GPU Broadcast using NCCL
     */
    void Broadcast(Tensor& tensor, int src_rank = 0) override;

    /**
     * Barrier using NCCL AllReduce (guaranteed to sync all GPUs)
     */
    void Barrier() override;

    /**
     * AllGather using NCCL
     */
    std::vector<Tensor> AllGather(const Tensor& input) override;

    /**
     * ReduceScatter using NCCL
     */
    void ReduceScatter(Tensor& tensor, ReduceOp op = ReduceOp::SUM) override;

    // ========== Async Operations ==========

    /**
     * Asynchronous AllReduce - returns immediately, use WaitAll() to sync
     */
    void AllReduceAsync(Tensor& tensor, ReduceOp op = ReduceOp::SUM) override;

    /**
     * Wait for all async operations to complete
     */
    void WaitAll() override;

    // ========== Utilities ==========

    BackendType GetBackendType() const override { return BackendType::NCCL; }
    std::string GetBackendName() const override { return "NCCL"; }

    /**
     * Get NCCL communicator (for advanced use)
     */
    ncclComm_t GetNCCLComm() const { return nccl_comm_; }

    /**
     * Get CUDA stream used for NCCL operations
     */
    cudaStream_t GetStream() const { return stream_; }

    /**
     * Get current CUDA device
     */
    int GetDevice() const { return device_; }

private:
    // NCCL resources
    ncclComm_t nccl_comm_ = nullptr;
    cudaStream_t stream_ = nullptr;
    int device_ = 0;

    // Rank info
    int rank_ = -1;
    int world_size_ = 0;
    int local_rank_ = 0;

    // State
    std::atomic<bool> initialized_{false};
    std::mutex comm_mutex_;  // Protect NCCL comm from concurrent access

    // ========== Helper Methods ==========

    /**
     * Bootstrap: Exchange ncclUniqueId via TCP
     * Rank 0 generates ID and broadcasts to all others
     */
    bool ExchangeNCCLId(ncclUniqueId& id, const DistributedConfig& config);

    /**
     * Convert ReduceOp to ncclRedOp_t
     */
    static ncclRedOp_t ToNCCLReduceOp(ReduceOp op);

    /**
     * Get ncclDataType_t for tensor element type
     */
    static ncclDataType_t GetNCCLDataType(const Tensor& tensor);

    /**
     * Check NCCL result and throw on error
     */
    static void CheckNCCL(ncclResult_t result, const char* operation);

    /**
     * Check CUDA result and throw on error
     */
    static void CheckCUDA(cudaError_t result, const char* operation);

    /**
     * Ensure tensor is on the correct GPU
     */
    void EnsureOnDevice(const Tensor& tensor) const;
};

} // namespace cyxwiz

#endif // CYXWIZ_HAS_NCCL
