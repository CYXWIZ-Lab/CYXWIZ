#pragma once

#include "../api_export.h"
#include "../tensor.h"
#include <string>
#include <memory>
#include <vector>
#include <functional>

namespace cyxwiz {

/**
 * Reduce operations for collective communication
 */
enum class ReduceOp {
    SUM,        // Element-wise sum
    PRODUCT,    // Element-wise product
    MIN,        // Element-wise minimum
    MAX,        // Element-wise maximum
    AVERAGE     // Element-wise average (SUM / world_size)
};

/**
 * Backend types for distributed communication
 */
enum class BackendType {
    CPU,    // TCP socket-based (fallback)
    NCCL    // NVIDIA NCCL (GPU, high performance)
};

/**
 * Configuration for distributed training
 *
 * Values can be explicitly set or read from environment variables.
 * Environment variables take precedence if the corresponding field is -1.
 *
 * Environment variables:
 *   RANK        - Global rank of this process
 *   WORLD_SIZE  - Total number of processes
 *   LOCAL_RANK  - Rank within the local machine (for multi-GPU)
 *   MASTER_ADDR - IP address of rank 0 (default: 127.0.0.1)
 *   MASTER_PORT - Port for rank 0 (default: 29500)
 */
struct CYXWIZ_API DistributedConfig {
    BackendType backend = BackendType::CPU;  // Default to CPU for compatibility
    int rank = -1;              // -1 = read from RANK env
    int world_size = -1;        // -1 = read from WORLD_SIZE env
    int local_rank = -1;        // -1 = read from LOCAL_RANK env
    std::string master_addr = "127.0.0.1";
    int master_port = 29500;
    int timeout_ms = 30000;     // Connection timeout in milliseconds

    /**
     * Create config from environment variables
     * Reads RANK, WORLD_SIZE, LOCAL_RANK, MASTER_ADDR, MASTER_PORT
     * Returns config with values set from environment (or defaults if not set)
     */
    static DistributedConfig FromEnvironment();

    /**
     * Validate configuration
     * @return true if config is valid, false otherwise
     */
    bool IsValid() const;

    /**
     * Get string representation for debugging
     */
    std::string ToString() const;
};

/**
 * Abstract base class for process group communication
 *
 * ProcessGroup provides the collective communication primitives needed for
 * distributed training: AllReduce, Broadcast, Barrier, AllGather.
 *
 * Implementations:
 *   - CPUProcessGroup: TCP socket-based, works everywhere
 *   - NCCLProcessGroup: NCCL-based, high-performance GPU communication
 *
 * Usage:
 *   auto config = DistributedConfig::FromEnvironment();
 *   init_distributed(config);  // Creates and initializes default process group
 *
 *   // In training loop:
 *   ddp.SyncGradients();  // Uses GetDefaultProcessGroup() internally
 *
 *   finalize_distributed();
 */
class CYXWIZ_API ProcessGroup {
public:
    virtual ~ProcessGroup() = default;

    // ========== Lifecycle ==========

    /**
     * Initialize the process group
     * Must be called before any communication operations
     * @param config Distributed configuration
     * @return true on success, false on failure
     */
    virtual bool Initialize(const DistributedConfig& config) = 0;

    /**
     * Finalize the process group
     * Releases all resources. Must be called before program exit.
     */
    virtual void Finalize() = 0;

    /**
     * Check if process group is initialized
     */
    virtual bool IsInitialized() const = 0;

    // ========== Rank Information ==========

    /**
     * Get global rank of this process
     * @return Rank in [0, world_size)
     */
    virtual int GetRank() const = 0;

    /**
     * Get total number of processes
     */
    virtual int GetWorldSize() const = 0;

    /**
     * Get local rank (within this machine)
     * Useful for selecting GPU device
     */
    virtual int GetLocalRank() const = 0;

    // ========== Collective Operations ==========

    /**
     * All-reduce operation: reduce tensor across all ranks, result available on all ranks
     *
     * @param tensor Input/output tensor (modified in-place)
     * @param op Reduction operation (default: SUM)
     *
     * Example with SUM, 4 ranks:
     *   Rank 0: [1, 2, 3]    Rank 1: [4, 5, 6]    Rank 2: [7, 8, 9]    Rank 3: [10, 11, 12]
     *   After AllReduce:
     *   All ranks: [22, 26, 30]
     */
    virtual void AllReduce(Tensor& tensor, ReduceOp op = ReduceOp::SUM) = 0;

    /**
     * All-reduce multiple tensors (may be more efficient than calling AllReduce repeatedly)
     * @param tensors Vector of tensor pointers to reduce
     * @param op Reduction operation
     */
    virtual void AllReduceMultiple(std::vector<Tensor*>& tensors, ReduceOp op = ReduceOp::SUM);

    /**
     * Broadcast tensor from source rank to all other ranks
     *
     * @param tensor Input on src_rank, output on all other ranks
     * @param src_rank Rank to broadcast from (default: 0)
     *
     * Example with src_rank=0, 4 ranks:
     *   Before: Rank 0: [1, 2, 3]    Other ranks: [?, ?, ?]
     *   After:  All ranks: [1, 2, 3]
     */
    virtual void Broadcast(Tensor& tensor, int src_rank = 0) = 0;

    /**
     * Synchronization barrier
     * All processes must call this before any can proceed
     */
    virtual void Barrier() = 0;

    /**
     * All-gather: gather tensors from all ranks
     *
     * @param input Local tensor to contribute
     * @return Vector of tensors, one from each rank (ordered by rank)
     *
     * Example with 4 ranks:
     *   Rank 0: [1]    Rank 1: [2]    Rank 2: [3]    Rank 3: [4]
     *   After AllGather:
     *   All ranks: [[1], [2], [3], [4]]
     */
    virtual std::vector<Tensor> AllGather(const Tensor& input) = 0;

    /**
     * Reduce-scatter: reduce then scatter
     * Each rank gets a portion of the reduced result
     *
     * @param tensor Input tensor (must be divisible by world_size)
     *               Output: this rank's portion of the reduced result
     * @param op Reduction operation
     */
    virtual void ReduceScatter(Tensor& tensor, ReduceOp op = ReduceOp::SUM);

    // ========== Async Operations (Optional) ==========

    /**
     * Asynchronous all-reduce (for overlapping communication with computation)
     * Default implementation: synchronous
     */
    virtual void AllReduceAsync(Tensor& tensor, ReduceOp op = ReduceOp::SUM) {
        AllReduce(tensor, op);
    }

    /**
     * Wait for all async operations to complete
     */
    virtual void WaitAll() {}

    // ========== Utilities ==========

    /**
     * Get backend type
     */
    virtual BackendType GetBackendType() const = 0;

    /**
     * Get backend name for logging
     */
    virtual std::string GetBackendName() const = 0;

protected:
    DistributedConfig config_;
    bool initialized_ = false;
};

// ========== Global Process Group Management ==========

/**
 * Get the default (global) process group
 * @return Pointer to default process group, or nullptr if not initialized
 */
CYXWIZ_API ProcessGroup* GetDefaultProcessGroup();

/**
 * Set the default process group
 * Takes ownership of the process group
 */
CYXWIZ_API void SetDefaultProcessGroup(std::unique_ptr<ProcessGroup> pg);

// ========== Convenience Functions ==========

/**
 * Initialize distributed training
 * Creates and initializes the default process group
 *
 * @param config Configuration (default: read from environment)
 * @return true on success
 */
CYXWIZ_API bool init_distributed(const DistributedConfig& config = DistributedConfig::FromEnvironment());

/**
 * Finalize distributed training
 * Cleans up the default process group
 */
CYXWIZ_API void finalize_distributed();

/**
 * Get rank of current process
 * @return Rank, or -1 if not initialized
 */
CYXWIZ_API int get_rank();

/**
 * Get world size (total number of processes)
 * @return World size, or 1 if not initialized
 */
CYXWIZ_API int get_world_size();

/**
 * Get local rank
 * @return Local rank, or 0 if not initialized
 */
CYXWIZ_API int get_local_rank();

/**
 * Check if distributed training is active
 */
CYXWIZ_API bool is_distributed();

/**
 * Check if this is the master (rank 0) process
 */
CYXWIZ_API bool is_master();

// ========== Factory ==========

/**
 * Create a process group of the specified type
 * @param backend Backend type (CPU or NCCL)
 * @return Unique pointer to process group (uninitialized)
 */
CYXWIZ_API std::unique_ptr<ProcessGroup> CreateProcessGroup(BackendType backend);

} // namespace cyxwiz
