#pragma once

#include "process_group.h"
#include <vector>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <queue>

namespace cyxwiz {

/**
 * CPU-based ProcessGroup implementation using TCP sockets
 *
 * Uses ring all-reduce algorithm for efficient bandwidth utilization.
 * Cross-platform: uses Winsock on Windows, POSIX sockets on Linux/macOS.
 *
 * Algorithm overview (Ring All-Reduce):
 *   1. Divide tensor into N chunks (N = world_size)
 *   2. Scatter-Reduce: Each rank sends chunk[i] to next rank, receives from prev rank,
 *      and reduces locally. Repeat N-1 times.
 *   3. All-Gather: Each rank sends reduced chunk to next rank. Repeat N-1 times.
 *
 * This achieves O(2 * (N-1) / N * data_size) communication, nearly optimal.
 */
class CYXWIZ_API CPUProcessGroup : public ProcessGroup {
public:
    CPUProcessGroup();
    ~CPUProcessGroup() override;

    // Lifecycle
    bool Initialize(const DistributedConfig& config) override;
    void Finalize() override;
    bool IsInitialized() const override { return initialized_; }

    // Rank info
    int GetRank() const override { return rank_; }
    int GetWorldSize() const override { return world_size_; }
    int GetLocalRank() const override { return local_rank_; }

    // Collective operations
    void AllReduce(Tensor& tensor, ReduceOp op = ReduceOp::SUM) override;
    void AllReduceMultiple(std::vector<Tensor*>& tensors, ReduceOp op = ReduceOp::SUM) override;
    void Broadcast(Tensor& tensor, int src_rank = 0) override;
    void Barrier() override;
    std::vector<Tensor> AllGather(const Tensor& input) override;
    void ReduceScatter(Tensor& tensor, ReduceOp op = ReduceOp::SUM) override;

    // Backend info
    BackendType GetBackendType() const override { return BackendType::CPU; }
    std::string GetBackendName() const override { return "CPU (TCP)"; }

private:
    int rank_;
    int world_size_;
    int local_rank_;

    // Socket infrastructure
#ifdef _WIN32
    using SocketType = unsigned long long;  // SOCKET on Windows
#else
    using SocketType = int;
#endif

    SocketType server_socket_;              // For accepting connections
    std::vector<SocketType> peer_sockets_;  // peer_sockets_[i] = connection to rank i
    std::mutex socket_mutex_;

    // Receive thread for handling incoming connections
    std::thread accept_thread_;
    std::atomic<bool> should_stop_;

    // Connection establishment
    bool SetupServerSocket();
    bool ConnectToPeers();
    bool AcceptPeerConnections();
    void CloseConnections();

    // Low-level socket operations
    bool InitializeSockets();  // Platform-specific init (WSAStartup on Windows)
    void CleanupSockets();     // Platform-specific cleanup
    SocketType CreateSocket();
    void CloseSocket(SocketType socket);
    bool SetSocketOptions(SocketType socket);
    bool BindSocket(SocketType socket, int port);
    bool ListenSocket(SocketType socket, int backlog);
    SocketType AcceptConnection(SocketType server_socket);
    bool ConnectSocket(SocketType socket, const std::string& addr, int port);

    // Data transfer
    bool SendAll(SocketType socket, const void* data, size_t size);
    bool RecvAll(SocketType socket, void* data, size_t size);
    bool SendTensor(int peer_rank, const Tensor& tensor);
    bool RecvTensor(int peer_rank, Tensor& tensor);

    // Ring all-reduce implementation
    void RingAllReduce(float* data, size_t count, ReduceOp op);
    void RingBroadcast(float* data, size_t count, int src_rank);
    void RingAllGather(const float* input, size_t input_count,
                       std::vector<float>& output);

    // Reduction operations
    void ApplyReduceOp(float* dst, const float* src, size_t count, ReduceOp op);
    void ApplyReduceOpFinalize(float* data, size_t count, ReduceOp op);

    // Barrier synchronization
    void RingBarrier();
};

} // namespace cyxwiz
