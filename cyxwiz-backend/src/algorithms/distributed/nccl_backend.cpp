#include "cyxwiz/distributed/nccl_backend.h"

#ifdef CYXWIZ_HAS_NCCL

#include <spdlog/spdlog.h>
#include <stdexcept>
#include <cstring>

// Platform-specific socket headers for bootstrap
#ifdef _WIN32
    #ifndef NOMINMAX
        #define NOMINMAX
    #endif
    #define WIN32_LEAN_AND_MEAN
    #include <winsock2.h>
    #include <ws2tcpip.h>
    #pragma comment(lib, "ws2_32.lib")

    #define SOCKET_TYPE SOCKET
    #define SOCKET_ERROR_CODE WSAGetLastError()
    #define INVALID_SOCKET_VALUE INVALID_SOCKET
    #define CLOSE_SOCKET closesocket

    static bool InitWinsock() {
        static bool initialized = false;
        static std::mutex init_mutex;
        std::lock_guard<std::mutex> lock(init_mutex);
        if (!initialized) {
            WSADATA wsa_data;
            if (WSAStartup(MAKEWORD(2, 2), &wsa_data) != 0) {
                return false;
            }
            initialized = true;
        }
        return true;
    }
#else
    #include <sys/socket.h>
    #include <netinet/in.h>
    #include <netinet/tcp.h>
    #include <arpa/inet.h>
    #include <unistd.h>
    #include <errno.h>

    #define SOCKET_TYPE int
    #define SOCKET_ERROR_CODE errno
    #define INVALID_SOCKET_VALUE (-1)
    #define CLOSE_SOCKET close

    static bool InitWinsock() { return true; }  // No-op on Linux
#endif

namespace cyxwiz {

// ========== Constructor / Destructor ==========

NCCLProcessGroup::NCCLProcessGroup()
    : nccl_comm_(nullptr)
    , stream_(nullptr)
    , device_(0)
    , rank_(-1)
    , world_size_(0)
    , local_rank_(0)
    , initialized_(false) {
}

NCCLProcessGroup::~NCCLProcessGroup() {
    if (initialized_) {
        Finalize();
    }
}

// ========== Lifecycle ==========

bool NCCLProcessGroup::Initialize(const DistributedConfig& config) {
    if (initialized_) {
        spdlog::warn("NCCLProcessGroup::Initialize: already initialized");
        return true;
    }

    config_ = config;
    rank_ = config.rank;
    world_size_ = config.world_size;
    local_rank_ = config.local_rank >= 0 ? config.local_rank : config.rank;

    spdlog::info("NCCLProcessGroup: Initializing rank {}/{} (local_rank={})",
                 rank_, world_size_, local_rank_);

    // Set CUDA device based on local rank
    device_ = local_rank_;
    cudaError_t cuda_err = cudaSetDevice(device_);
    if (cuda_err != cudaSuccess) {
        spdlog::error("NCCLProcessGroup: cudaSetDevice({}) failed: {}",
                      device_, cudaGetErrorString(cuda_err));
        return false;
    }

    // Create CUDA stream for NCCL operations
    cuda_err = cudaStreamCreate(&stream_);
    if (cuda_err != cudaSuccess) {
        spdlog::error("NCCLProcessGroup: cudaStreamCreate failed: {}",
                      cudaGetErrorString(cuda_err));
        return false;
    }

    // Exchange NCCL unique ID via TCP bootstrap
    ncclUniqueId nccl_id;
    if (!ExchangeNCCLId(nccl_id, config)) {
        spdlog::error("NCCLProcessGroup: Failed to exchange NCCL ID");
        cudaStreamDestroy(stream_);
        stream_ = nullptr;
        return false;
    }

    // Initialize NCCL communicator
    ncclResult_t nccl_result = ncclCommInitRank(&nccl_comm_, world_size_, nccl_id, rank_);
    if (nccl_result != ncclSuccess) {
        spdlog::error("NCCLProcessGroup: ncclCommInitRank failed: {}",
                      ncclGetErrorString(nccl_result));
        cudaStreamDestroy(stream_);
        stream_ = nullptr;
        return false;
    }

    initialized_ = true;
    spdlog::info("NCCLProcessGroup: Rank {} initialized successfully on GPU {}",
                 rank_, device_);

    return true;
}

void NCCLProcessGroup::Finalize() {
    if (!initialized_) {
        return;
    }

    spdlog::debug("NCCLProcessGroup: Finalizing rank {}", rank_);

    // Synchronize stream before destroying
    if (stream_) {
        cudaStreamSynchronize(stream_);
        cudaStreamDestroy(stream_);
        stream_ = nullptr;
    }

    // Destroy NCCL communicator
    if (nccl_comm_) {
        ncclCommDestroy(nccl_comm_);
        nccl_comm_ = nullptr;
    }

    initialized_ = false;
    spdlog::debug("NCCLProcessGroup: Rank {} finalized", rank_);
}

// ========== Bootstrap: Exchange NCCL ID via TCP ==========

bool NCCLProcessGroup::ExchangeNCCLId(ncclUniqueId& id, const DistributedConfig& config) {
    // Initialize socket library (Windows)
    if (!InitWinsock()) {
        spdlog::error("NCCLProcessGroup: Failed to initialize sockets");
        return false;
    }

    if (rank_ == 0) {
        // Rank 0: Generate NCCL ID and send to all other ranks
        ncclResult_t result = ncclGetUniqueId(&id);
        if (result != ncclSuccess) {
            spdlog::error("NCCLProcessGroup: ncclGetUniqueId failed: {}",
                          ncclGetErrorString(result));
            return false;
        }

        spdlog::debug("NCCLProcessGroup: Rank 0 generated NCCL ID, broadcasting to {} peers",
                      world_size_ - 1);

        // Create server socket
        SOCKET_TYPE server_socket = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
        if (server_socket == INVALID_SOCKET_VALUE) {
            spdlog::error("NCCLProcessGroup: Failed to create server socket");
            return false;
        }

        // Allow address reuse
        int opt = 1;
        setsockopt(server_socket, SOL_SOCKET, SO_REUSEADDR,
                   reinterpret_cast<const char*>(&opt), sizeof(opt));

        // Bind to master port
        struct sockaddr_in addr;
        std::memset(&addr, 0, sizeof(addr));
        addr.sin_family = AF_INET;
        addr.sin_addr.s_addr = INADDR_ANY;
        addr.sin_port = htons(static_cast<uint16_t>(config.master_port));

        if (bind(server_socket, reinterpret_cast<struct sockaddr*>(&addr), sizeof(addr)) < 0) {
            spdlog::error("NCCLProcessGroup: Failed to bind to port {}", config.master_port);
            CLOSE_SOCKET(server_socket);
            return false;
        }

        if (listen(server_socket, world_size_) < 0) {
            spdlog::error("NCCLProcessGroup: Failed to listen");
            CLOSE_SOCKET(server_socket);
            return false;
        }

        // Accept connections from all other ranks and send ID
        for (int i = 1; i < world_size_; ++i) {
            SOCKET_TYPE client_socket = accept(server_socket, nullptr, nullptr);
            if (client_socket == INVALID_SOCKET_VALUE) {
                spdlog::error("NCCLProcessGroup: Failed to accept connection");
                CLOSE_SOCKET(server_socket);
                return false;
            }

            // Send NCCL ID
            ssize_t sent = send(client_socket, reinterpret_cast<const char*>(&id),
                                sizeof(ncclUniqueId), 0);
            if (sent != sizeof(ncclUniqueId)) {
                spdlog::error("NCCLProcessGroup: Failed to send NCCL ID");
                CLOSE_SOCKET(client_socket);
                CLOSE_SOCKET(server_socket);
                return false;
            }

            CLOSE_SOCKET(client_socket);
            spdlog::debug("NCCLProcessGroup: Sent NCCL ID to peer {}", i);
        }

        CLOSE_SOCKET(server_socket);

    } else {
        // Other ranks: Connect to rank 0 and receive NCCL ID
        spdlog::debug("NCCLProcessGroup: Rank {} connecting to {}:{} for NCCL ID",
                      rank_, config.master_addr, config.master_port);

        SOCKET_TYPE client_socket = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
        if (client_socket == INVALID_SOCKET_VALUE) {
            spdlog::error("NCCLProcessGroup: Failed to create client socket");
            return false;
        }

        struct sockaddr_in addr;
        std::memset(&addr, 0, sizeof(addr));
        addr.sin_family = AF_INET;
        addr.sin_port = htons(static_cast<uint16_t>(config.master_port));

        if (inet_pton(AF_INET, config.master_addr.c_str(), &addr.sin_addr) <= 0) {
            spdlog::error("NCCLProcessGroup: Invalid master address: {}", config.master_addr);
            CLOSE_SOCKET(client_socket);
            return false;
        }

        // Retry connection with timeout
        int retries = 0;
        const int max_retries = config.timeout_ms / 1000;
        while (retries < max_retries) {
            if (connect(client_socket, reinterpret_cast<struct sockaddr*>(&addr),
                        sizeof(addr)) == 0) {
                break;
            }
            retries++;
            spdlog::debug("NCCLProcessGroup: Connection attempt {} failed, retrying...", retries);
#ifdef _WIN32
            Sleep(1000);
#else
            sleep(1);
#endif
        }

        if (retries >= max_retries) {
            spdlog::error("NCCLProcessGroup: Failed to connect to master after {} retries",
                          max_retries);
            CLOSE_SOCKET(client_socket);
            return false;
        }

        // Receive NCCL ID
        ssize_t received = recv(client_socket, reinterpret_cast<char*>(&id),
                                sizeof(ncclUniqueId), MSG_WAITALL);
        if (received != sizeof(ncclUniqueId)) {
            spdlog::error("NCCLProcessGroup: Failed to receive NCCL ID");
            CLOSE_SOCKET(client_socket);
            return false;
        }

        CLOSE_SOCKET(client_socket);
        spdlog::debug("NCCLProcessGroup: Rank {} received NCCL ID from master", rank_);
    }

    return true;
}

// ========== Collective Operations ==========

void NCCLProcessGroup::AllReduce(Tensor& tensor, ReduceOp op) {
    if (!initialized_) {
        throw std::runtime_error("NCCLProcessGroup: Not initialized");
    }

    std::lock_guard<std::mutex> lock(comm_mutex_);
    EnsureOnDevice(tensor);

    ncclDataType_t dtype = GetNCCLDataType(tensor);
    ncclRedOp_t nccl_op = ToNCCLReduceOp(op);
    size_t count = tensor.NumElements();
    void* data = const_cast<float*>(tensor.Data<float>());

    ncclResult_t result = ncclAllReduce(data, data, count, dtype, nccl_op,
                                         nccl_comm_, stream_);
    CheckNCCL(result, "AllReduce");

    // Synchronize to ensure completion
    CheckCUDA(cudaStreamSynchronize(stream_), "cudaStreamSynchronize");

    // For AVERAGE, divide by world_size (NCCL doesn't have native AVERAGE)
    if (op == ReduceOp::AVERAGE) {
        // TODO: Implement GPU-based division by world_size
        // For now, this is handled by the caller or use SUM and divide
    }
}

void NCCLProcessGroup::AllReduceMultiple(std::vector<Tensor*>& tensors, ReduceOp op) {
    if (!initialized_) {
        throw std::runtime_error("NCCLProcessGroup: Not initialized");
    }

    if (tensors.empty()) return;

    std::lock_guard<std::mutex> lock(comm_mutex_);
    ncclRedOp_t nccl_op = ToNCCLReduceOp(op);

    // Use NCCL group for batched operations
    ncclResult_t result = ncclGroupStart();
    CheckNCCL(result, "ncclGroupStart");

    for (Tensor* tensor : tensors) {
        if (!tensor) continue;
        EnsureOnDevice(*tensor);

        ncclDataType_t dtype = GetNCCLDataType(*tensor);
        size_t count = tensor->NumElements();
        void* data = const_cast<float*>(tensor->Data<float>());

        result = ncclAllReduce(data, data, count, dtype, nccl_op, nccl_comm_, stream_);
        if (result != ncclSuccess) {
            ncclGroupEnd();
            CheckNCCL(result, "AllReduceMultiple");
        }
    }

    result = ncclGroupEnd();
    CheckNCCL(result, "ncclGroupEnd");

    CheckCUDA(cudaStreamSynchronize(stream_), "cudaStreamSynchronize");
}

void NCCLProcessGroup::Broadcast(Tensor& tensor, int src_rank) {
    if (!initialized_) {
        throw std::runtime_error("NCCLProcessGroup: Not initialized");
    }

    std::lock_guard<std::mutex> lock(comm_mutex_);
    EnsureOnDevice(tensor);

    ncclDataType_t dtype = GetNCCLDataType(tensor);
    size_t count = tensor.NumElements();
    void* data = const_cast<float*>(tensor.Data<float>());

    ncclResult_t result = ncclBroadcast(data, data, count, dtype, src_rank,
                                         nccl_comm_, stream_);
    CheckNCCL(result, "Broadcast");

    CheckCUDA(cudaStreamSynchronize(stream_), "cudaStreamSynchronize");
}

void NCCLProcessGroup::Barrier() {
    if (!initialized_) {
        throw std::runtime_error("NCCLProcessGroup: Not initialized");
    }

    std::lock_guard<std::mutex> lock(comm_mutex_);

    // Use AllReduce of a single element as barrier
    // This is the recommended way in NCCL
    float dummy = 0.0f;
    float* d_dummy;
    CheckCUDA(cudaMalloc(&d_dummy, sizeof(float)), "cudaMalloc");
    CheckCUDA(cudaMemcpy(d_dummy, &dummy, sizeof(float), cudaMemcpyHostToDevice), "cudaMemcpy");

    ncclResult_t result = ncclAllReduce(d_dummy, d_dummy, 1, ncclFloat, ncclSum,
                                         nccl_comm_, stream_);
    CheckNCCL(result, "Barrier");

    CheckCUDA(cudaStreamSynchronize(stream_), "cudaStreamSynchronize");
    CheckCUDA(cudaFree(d_dummy), "cudaFree");
}

std::vector<Tensor> NCCLProcessGroup::AllGather(const Tensor& input) {
    if (!initialized_) {
        throw std::runtime_error("NCCLProcessGroup: Not initialized");
    }

    std::lock_guard<std::mutex> lock(comm_mutex_);
    EnsureOnDevice(input);

    ncclDataType_t dtype = GetNCCLDataType(input);
    size_t count = input.NumElements();

    // Allocate output buffer (world_size * input_size)
    size_t total_count = count * world_size_;
    float* d_output;
    CheckCUDA(cudaMalloc(&d_output, total_count * sizeof(float)), "cudaMalloc");

    const void* send_data = input.Data<float>();

    ncclResult_t result = ncclAllGather(send_data, d_output, count, dtype,
                                         nccl_comm_, stream_);
    CheckNCCL(result, "AllGather");

    CheckCUDA(cudaStreamSynchronize(stream_), "cudaStreamSynchronize");

    // Copy results to host and create tensors
    std::vector<float> h_output(total_count);
    CheckCUDA(cudaMemcpy(h_output.data(), d_output, total_count * sizeof(float),
                          cudaMemcpyDeviceToHost), "cudaMemcpy");
    CheckCUDA(cudaFree(d_output), "cudaFree");

    // Create output tensors
    std::vector<Tensor> result_tensors;
    result_tensors.reserve(world_size_);

    std::vector<int64_t> shape = input.Shape();

    for (int i = 0; i < world_size_; ++i) {
        std::vector<float> tensor_data(h_output.begin() + i * count,
                                        h_output.begin() + (i + 1) * count);
        result_tensors.push_back(Tensor(tensor_data, shape));
    }

    return result_tensors;
}

void NCCLProcessGroup::ReduceScatter(Tensor& tensor, ReduceOp op) {
    if (!initialized_) {
        throw std::runtime_error("NCCLProcessGroup: Not initialized");
    }

    std::lock_guard<std::mutex> lock(comm_mutex_);
    EnsureOnDevice(tensor);

    ncclDataType_t dtype = GetNCCLDataType(tensor);
    ncclRedOp_t nccl_op = ToNCCLReduceOp(op);
    size_t total_count = tensor.NumElements();
    size_t recv_count = total_count / world_size_;

    if (total_count % world_size_ != 0) {
        throw std::runtime_error("NCCLProcessGroup::ReduceScatter: tensor size must be divisible by world_size");
    }

    void* data = const_cast<float*>(tensor.Data<float>());

    // Output goes to first recv_count elements
    ncclResult_t result = ncclReduceScatter(data, data, recv_count, dtype, nccl_op,
                                             nccl_comm_, stream_);
    CheckNCCL(result, "ReduceScatter");

    CheckCUDA(cudaStreamSynchronize(stream_), "cudaStreamSynchronize");
}

// ========== Async Operations ==========

void NCCLProcessGroup::AllReduceAsync(Tensor& tensor, ReduceOp op) {
    if (!initialized_) {
        throw std::runtime_error("NCCLProcessGroup: Not initialized");
    }

    std::lock_guard<std::mutex> lock(comm_mutex_);
    EnsureOnDevice(tensor);

    ncclDataType_t dtype = GetNCCLDataType(tensor);
    ncclRedOp_t nccl_op = ToNCCLReduceOp(op);
    size_t count = tensor.NumElements();
    void* data = const_cast<float*>(tensor.Data<float>());

    ncclResult_t result = ncclAllReduce(data, data, count, dtype, nccl_op,
                                         nccl_comm_, stream_);
    CheckNCCL(result, "AllReduceAsync");
    // Don't synchronize - that's the async part
}

void NCCLProcessGroup::WaitAll() {
    if (!initialized_) return;

    CheckCUDA(cudaStreamSynchronize(stream_), "WaitAll");
}

// ========== Helper Methods ==========

ncclRedOp_t NCCLProcessGroup::ToNCCLReduceOp(ReduceOp op) {
    switch (op) {
        case ReduceOp::SUM:
        case ReduceOp::AVERAGE:  // AVERAGE = SUM, then divide
            return ncclSum;
        case ReduceOp::PRODUCT:
            return ncclProd;
        case ReduceOp::MIN:
            return ncclMin;
        case ReduceOp::MAX:
            return ncclMax;
        default:
            return ncclSum;
    }
}

ncclDataType_t NCCLProcessGroup::GetNCCLDataType(const Tensor& tensor) {
    // TODO: Support more data types based on tensor dtype
    // For now, assume float32
    return ncclFloat;
}

void NCCLProcessGroup::CheckNCCL(ncclResult_t result, const char* operation) {
    if (result != ncclSuccess) {
        throw std::runtime_error(std::string("NCCL error in ") + operation +
                                  ": " + ncclGetErrorString(result));
    }
}

void NCCLProcessGroup::CheckCUDA(cudaError_t result, const char* operation) {
    if (result != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA error in ") + operation +
                                  ": " + cudaGetErrorString(result));
    }
}

void NCCLProcessGroup::EnsureOnDevice(const Tensor& tensor) const {
    // TODO: Implement proper device check
    // For now, assume tensors are already on the correct device
    // In a full implementation, we would check tensor.Device() and
    // potentially copy to the correct GPU
}

} // namespace cyxwiz

#endif // CYXWIZ_HAS_NCCL
