#include "cyxwiz/distributed/process_group.h"
#include "cyxwiz/distributed/cpu_backend.h"

#ifdef CYXWIZ_HAS_NCCL
#include "cyxwiz/distributed/nccl_backend.h"
#endif

#include <spdlog/spdlog.h>
#include <cstdlib>
#include <sstream>
#include <mutex>

namespace cyxwiz {

// ========== Global Process Group Singleton ==========

namespace {
    std::unique_ptr<ProcessGroup> g_default_process_group;
    std::mutex g_pg_mutex;
}

ProcessGroup* GetDefaultProcessGroup() {
    std::lock_guard<std::mutex> lock(g_pg_mutex);
    return g_default_process_group.get();
}

void SetDefaultProcessGroup(std::unique_ptr<ProcessGroup> pg) {
    std::lock_guard<std::mutex> lock(g_pg_mutex);
    g_default_process_group = std::move(pg);
}

// ========== DistributedConfig Implementation ==========

namespace {
    // Helper to read environment variable as int
    int GetEnvInt(const char* name, int default_value) {
        const char* value = std::getenv(name);
        if (value == nullptr) {
            return default_value;
        }
        try {
            return std::stoi(value);
        } catch (...) {
            spdlog::warn("Invalid value for {}: '{}', using default {}", name, value, default_value);
            return default_value;
        }
    }

    // Helper to read environment variable as string
    std::string GetEnvString(const char* name, const std::string& default_value) {
        const char* value = std::getenv(name);
        return value ? value : default_value;
    }
}

DistributedConfig DistributedConfig::FromEnvironment() {
    DistributedConfig config;

    // Read rank information
    config.rank = GetEnvInt("RANK", -1);
    config.world_size = GetEnvInt("WORLD_SIZE", -1);
    config.local_rank = GetEnvInt("LOCAL_RANK", -1);

    // Read master address/port
    config.master_addr = GetEnvString("MASTER_ADDR", "127.0.0.1");
    config.master_port = GetEnvInt("MASTER_PORT", 29500);

    // Timeout (optional)
    config.timeout_ms = GetEnvInt("DISTRIBUTED_TIMEOUT_MS", 30000);

    // Backend selection
    std::string backend_str = GetEnvString("DISTRIBUTED_BACKEND", "cpu");
    if (backend_str == "nccl" || backend_str == "NCCL") {
#ifdef CYXWIZ_HAS_NCCL
        config.backend = BackendType::NCCL;
#else
        spdlog::warn("NCCL backend requested but not available, falling back to CPU");
        config.backend = BackendType::CPU;
#endif
    } else {
        config.backend = BackendType::CPU;
    }

    // If LOCAL_RANK not set, try to infer from RANK for single-machine case
    if (config.local_rank < 0 && config.rank >= 0) {
        // For single-machine, local_rank == rank
        // For multi-machine, user must set LOCAL_RANK explicitly
        config.local_rank = config.rank;
    }

    spdlog::debug("DistributedConfig from environment: {}", config.ToString());
    return config;
}

bool DistributedConfig::IsValid() const {
    // Rank must be set
    if (rank < 0) {
        spdlog::error("Invalid config: rank not set (set RANK environment variable)");
        return false;
    }

    // World size must be set and > 0
    if (world_size <= 0) {
        spdlog::error("Invalid config: world_size not set or invalid (set WORLD_SIZE environment variable)");
        return false;
    }

    // Rank must be < world_size
    if (rank >= world_size) {
        spdlog::error("Invalid config: rank ({}) >= world_size ({})", rank, world_size);
        return false;
    }

    // Local rank should be valid if set
    if (local_rank >= 0 && local_rank >= world_size) {
        spdlog::warn("local_rank ({}) >= world_size ({}), this may cause issues", local_rank, world_size);
    }

    // Port should be valid
    if (master_port <= 0 || master_port > 65535) {
        spdlog::error("Invalid config: master_port ({}) out of range", master_port);
        return false;
    }

    return true;
}

std::string DistributedConfig::ToString() const {
    std::ostringstream oss;
    oss << "DistributedConfig{"
        << "backend=" << (backend == BackendType::NCCL ? "NCCL" : "CPU")
        << ", rank=" << rank
        << ", world_size=" << world_size
        << ", local_rank=" << local_rank
        << ", master=" << master_addr << ":" << master_port
        << ", timeout_ms=" << timeout_ms
        << "}";
    return oss.str();
}

// ========== ProcessGroup Base Implementation ==========

void ProcessGroup::AllReduceMultiple(std::vector<Tensor*>& tensors, ReduceOp op) {
    // Default implementation: reduce each tensor sequentially
    for (auto* tensor : tensors) {
        if (tensor) {
            AllReduce(*tensor, op);
        }
    }
}

void ProcessGroup::ReduceScatter(Tensor& tensor, ReduceOp op) {
    // Default implementation using AllReduce + local scatter
    // Not optimal but works as fallback

    AllReduce(tensor, op);

    // Extract this rank's portion
    const auto& shape = tensor.Shape();
    if (shape.empty()) {
        return;
    }

    size_t total_size = tensor.NumElements();
    size_t chunk_size = total_size / GetWorldSize();

    if (chunk_size * GetWorldSize() != total_size) {
        spdlog::error("ReduceScatter: tensor size ({}) not divisible by world_size ({})",
                      total_size, GetWorldSize());
        return;
    }

    // Create output tensor with this rank's portion
    size_t offset = GetRank() * chunk_size;
    std::vector<float> my_chunk(chunk_size);

    const float* src = tensor.Data<float>();
    std::copy(src + offset, src + offset + chunk_size, my_chunk.begin());

    // Resize tensor to chunk size
    std::vector<size_t> new_shape = {chunk_size};
    tensor = Tensor(new_shape, my_chunk.data());
}

// ========== Convenience Functions ==========

bool init_distributed(const DistributedConfig& config) {
    // Validate config
    if (!config.IsValid()) {
        spdlog::error("init_distributed: invalid configuration");
        return false;
    }

    // Check if already initialized
    if (GetDefaultProcessGroup() != nullptr) {
        spdlog::warn("init_distributed: already initialized, finalizing first");
        finalize_distributed();
    }

    // Create process group
    auto pg = CreateProcessGroup(config.backend);
    if (!pg) {
        spdlog::error("init_distributed: failed to create process group");
        return false;
    }

    // Initialize
    if (!pg->Initialize(config)) {
        spdlog::error("init_distributed: failed to initialize process group");
        return false;
    }

    spdlog::info("Distributed training initialized: rank {}/{} using {} backend",
                 pg->GetRank(), pg->GetWorldSize(), pg->GetBackendName());

    SetDefaultProcessGroup(std::move(pg));
    return true;
}

void finalize_distributed() {
    std::lock_guard<std::mutex> lock(g_pg_mutex);

    if (g_default_process_group) {
        spdlog::info("Finalizing distributed training (rank {})",
                     g_default_process_group->GetRank());
        g_default_process_group->Finalize();
        g_default_process_group.reset();
    }
}

int get_rank() {
    auto* pg = GetDefaultProcessGroup();
    return pg ? pg->GetRank() : -1;
}

int get_world_size() {
    auto* pg = GetDefaultProcessGroup();
    return pg ? pg->GetWorldSize() : 1;
}

int get_local_rank() {
    auto* pg = GetDefaultProcessGroup();
    return pg ? pg->GetLocalRank() : 0;
}

bool is_distributed() {
    auto* pg = GetDefaultProcessGroup();
    return pg && pg->IsInitialized() && pg->GetWorldSize() > 1;
}

bool is_master() {
    return get_rank() == 0;
}

// ========== Factory ==========

std::unique_ptr<ProcessGroup> CreateProcessGroup(BackendType backend) {
    switch (backend) {
        case BackendType::CPU:
            return std::make_unique<CPUProcessGroup>();

#ifdef CYXWIZ_HAS_NCCL
        case BackendType::NCCL:
            return std::make_unique<NCCLProcessGroup>();
#endif

        default:
            spdlog::error("Unknown or unsupported backend type: {}",
                          static_cast<int>(backend));

            // Fall back to CPU
            spdlog::warn("Falling back to CPU backend");
            return std::make_unique<CPUProcessGroup>();
    }
}

} // namespace cyxwiz
