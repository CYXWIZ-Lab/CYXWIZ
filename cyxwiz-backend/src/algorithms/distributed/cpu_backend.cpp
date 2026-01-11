#include "cyxwiz/distributed/cpu_backend.h"
#include <spdlog/spdlog.h>
#include <cstring>
#include <algorithm>
#include <chrono>
#include <cmath>

// Platform-specific socket headers
#ifdef _WIN32
    #ifndef NOMINMAX
        #define NOMINMAX
    #endif
    #define WIN32_LEAN_AND_MEAN
    #include <winsock2.h>
    #include <ws2tcpip.h>
    #pragma comment(lib, "ws2_32.lib")

    #define SOCKET_ERROR_CODE WSAGetLastError()
    #define INVALID_SOCKET_VALUE INVALID_SOCKET
    #define CLOSE_SOCKET closesocket
#else
    #include <sys/socket.h>
    #include <netinet/in.h>
    #include <netinet/tcp.h>
    #include <arpa/inet.h>
    #include <unistd.h>
    #include <fcntl.h>
    #include <errno.h>

    #define SOCKET_ERROR_CODE errno
    #define INVALID_SOCKET_VALUE (-1)
    #define CLOSE_SOCKET close
#endif

namespace cyxwiz {

// ========== Constructor / Destructor ==========

CPUProcessGroup::CPUProcessGroup()
    : rank_(-1)
    , world_size_(-1)
    , local_rank_(-1)
    , server_socket_(INVALID_SOCKET_VALUE)
    , should_stop_(false) {
}

CPUProcessGroup::~CPUProcessGroup() {
    if (initialized_) {
        Finalize();
    }
}

// ========== Lifecycle ==========

bool CPUProcessGroup::Initialize(const DistributedConfig& config) {
    if (initialized_) {
        spdlog::warn("CPUProcessGroup::Initialize: already initialized");
        return true;
    }

    config_ = config;
    rank_ = config.rank;
    world_size_ = config.world_size;
    local_rank_ = config.local_rank >= 0 ? config.local_rank : config.rank;

    spdlog::info("CPUProcessGroup: Initializing rank {}/{}", rank_, world_size_);

    // Initialize socket library (Windows only)
    if (!InitializeSockets()) {
        spdlog::error("CPUProcessGroup: Failed to initialize socket library");
        return false;
    }

    // Resize peer sockets vector
    peer_sockets_.resize(world_size_, INVALID_SOCKET_VALUE);

    // Setup server socket to accept connections
    if (!SetupServerSocket()) {
        spdlog::error("CPUProcessGroup: Failed to setup server socket");
        CleanupSockets();
        return false;
    }

    // Connect to peers (lower ranks) and accept from peers (higher ranks)
    if (!ConnectToPeers()) {
        spdlog::error("CPUProcessGroup: Failed to connect to peers");
        CloseConnections();
        CleanupSockets();
        return false;
    }

    initialized_ = true;
    spdlog::info("CPUProcessGroup: Rank {} initialized successfully", rank_);

    return true;
}

void CPUProcessGroup::Finalize() {
    if (!initialized_) {
        return;
    }

    spdlog::debug("CPUProcessGroup: Finalizing rank {}", rank_);

    should_stop_ = true;

    // Wait for accept thread if running
    if (accept_thread_.joinable()) {
        accept_thread_.join();
    }

    // Close all connections
    CloseConnections();

    // Cleanup socket library
    CleanupSockets();

    initialized_ = false;
    spdlog::debug("CPUProcessGroup: Rank {} finalized", rank_);
}

// ========== Socket Setup ==========

bool CPUProcessGroup::InitializeSockets() {
#ifdef _WIN32
    WSADATA wsa_data;
    int result = WSAStartup(MAKEWORD(2, 2), &wsa_data);
    if (result != 0) {
        spdlog::error("WSAStartup failed with error: {}", result);
        return false;
    }
#endif
    return true;
}

void CPUProcessGroup::CleanupSockets() {
#ifdef _WIN32
    WSACleanup();
#endif
}

CPUProcessGroup::SocketType CPUProcessGroup::CreateSocket() {
    return socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
}

void CPUProcessGroup::CloseSocket(SocketType socket) {
    if (socket != INVALID_SOCKET_VALUE) {
        CLOSE_SOCKET(socket);
    }
}

bool CPUProcessGroup::SetSocketOptions(SocketType socket) {
    // Enable TCP_NODELAY for low latency
    int flag = 1;
    if (setsockopt(socket, IPPROTO_TCP, TCP_NODELAY,
                   reinterpret_cast<const char*>(&flag), sizeof(flag)) < 0) {
        spdlog::warn("Failed to set TCP_NODELAY: {}", SOCKET_ERROR_CODE);
    }

    // Enable SO_REUSEADDR
    if (setsockopt(socket, SOL_SOCKET, SO_REUSEADDR,
                   reinterpret_cast<const char*>(&flag), sizeof(flag)) < 0) {
        spdlog::warn("Failed to set SO_REUSEADDR: {}", SOCKET_ERROR_CODE);
    }

    // Set receive/send buffer sizes for better throughput
    int buffer_size = 4 * 1024 * 1024;  // 4MB
    setsockopt(socket, SOL_SOCKET, SO_RCVBUF,
               reinterpret_cast<const char*>(&buffer_size), sizeof(buffer_size));
    setsockopt(socket, SOL_SOCKET, SO_SNDBUF,
               reinterpret_cast<const char*>(&buffer_size), sizeof(buffer_size));

    return true;
}

bool CPUProcessGroup::BindSocket(SocketType socket, int port) {
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons(static_cast<uint16_t>(port));

    if (bind(socket, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
        spdlog::error("bind failed on port {}: {}", port, SOCKET_ERROR_CODE);
        return false;
    }
    return true;
}

bool CPUProcessGroup::ListenSocket(SocketType socket, int backlog) {
    if (listen(socket, backlog) < 0) {
        spdlog::error("listen failed: {}", SOCKET_ERROR_CODE);
        return false;
    }
    return true;
}

CPUProcessGroup::SocketType CPUProcessGroup::AcceptConnection(SocketType server_socket) {
    sockaddr_in client_addr{};
    socklen_t client_len = sizeof(client_addr);
    return accept(server_socket, reinterpret_cast<sockaddr*>(&client_addr), &client_len);
}

bool CPUProcessGroup::ConnectSocket(SocketType socket, const std::string& addr, int port) {
    sockaddr_in server_addr{};
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(static_cast<uint16_t>(port));

    if (inet_pton(AF_INET, addr.c_str(), &server_addr.sin_addr) <= 0) {
        spdlog::error("Invalid address: {}", addr);
        return false;
    }

    // Try to connect with retries
    int max_retries = 60;  // 60 * 500ms = 30 seconds
    for (int retry = 0; retry < max_retries; ++retry) {
        if (connect(socket, reinterpret_cast<sockaddr*>(&server_addr),
                    sizeof(server_addr)) == 0) {
            return true;
        }

        if (retry < max_retries - 1) {
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
        }
    }

    spdlog::error("Failed to connect to {}:{} after {} retries: {}",
                  addr, port, max_retries, SOCKET_ERROR_CODE);
    return false;
}

bool CPUProcessGroup::SetupServerSocket() {
    server_socket_ = CreateSocket();
    if (server_socket_ == INVALID_SOCKET_VALUE) {
        spdlog::error("Failed to create server socket: {}", SOCKET_ERROR_CODE);
        return false;
    }

    SetSocketOptions(server_socket_);

    // Each rank listens on master_port + rank
    int listen_port = config_.master_port + rank_;
    if (!BindSocket(server_socket_, listen_port)) {
        CloseSocket(server_socket_);
        server_socket_ = INVALID_SOCKET_VALUE;
        return false;
    }

    if (!ListenSocket(server_socket_, world_size_)) {
        CloseSocket(server_socket_);
        server_socket_ = INVALID_SOCKET_VALUE;
        return false;
    }

    spdlog::debug("Rank {} listening on port {}", rank_, listen_port);
    return true;
}

bool CPUProcessGroup::ConnectToPeers() {
    // Connection strategy:
    // - Each rank connects to all lower-ranked peers
    // - Each rank accepts connections from all higher-ranked peers
    // This ensures no deadlock

    // First, connect to all lower-ranked peers
    for (int peer = 0; peer < rank_; ++peer) {
        SocketType sock = CreateSocket();
        if (sock == INVALID_SOCKET_VALUE) {
            spdlog::error("Failed to create socket for peer {}", peer);
            return false;
        }

        SetSocketOptions(sock);

        int peer_port = config_.master_port + peer;
        spdlog::debug("Rank {} connecting to peer {} at {}:{}",
                      rank_, peer, config_.master_addr, peer_port);

        if (!ConnectSocket(sock, config_.master_addr, peer_port)) {
            CloseSocket(sock);
            return false;
        }

        // Send our rank so peer knows who connected
        int32_t our_rank = static_cast<int32_t>(rank_);
        if (!SendAll(sock, &our_rank, sizeof(our_rank))) {
            spdlog::error("Failed to send rank to peer {}", peer);
            CloseSocket(sock);
            return false;
        }

        peer_sockets_[peer] = sock;
        spdlog::debug("Rank {} connected to peer {}", rank_, peer);
    }

    // Now accept connections from all higher-ranked peers
    for (int peer = rank_ + 1; peer < world_size_; ++peer) {
        SocketType client_sock = AcceptConnection(server_socket_);
        if (client_sock == INVALID_SOCKET_VALUE) {
            spdlog::error("Failed to accept connection: {}", SOCKET_ERROR_CODE);
            return false;
        }

        SetSocketOptions(client_sock);

        // Receive connecting rank
        int32_t peer_rank;
        if (!RecvAll(client_sock, &peer_rank, sizeof(peer_rank))) {
            spdlog::error("Failed to receive peer rank");
            CloseSocket(client_sock);
            return false;
        }

        if (peer_rank < 0 || peer_rank >= world_size_) {
            spdlog::error("Invalid peer rank received: {}", peer_rank);
            CloseSocket(client_sock);
            return false;
        }

        peer_sockets_[peer_rank] = client_sock;
        spdlog::debug("Rank {} accepted connection from peer {}", rank_, peer_rank);
    }

    // Self-connection (rank to itself) is not needed
    peer_sockets_[rank_] = INVALID_SOCKET_VALUE;

    return true;
}

void CPUProcessGroup::CloseConnections() {
    // Close server socket
    if (server_socket_ != INVALID_SOCKET_VALUE) {
        CloseSocket(server_socket_);
        server_socket_ = INVALID_SOCKET_VALUE;
    }

    // Close all peer connections
    for (auto& sock : peer_sockets_) {
        if (sock != INVALID_SOCKET_VALUE) {
            CloseSocket(sock);
            sock = INVALID_SOCKET_VALUE;
        }
    }
}

// ========== Data Transfer ==========

bool CPUProcessGroup::SendAll(SocketType socket, const void* data, size_t size) {
    const char* ptr = static_cast<const char*>(data);
    size_t remaining = size;

    while (remaining > 0) {
        int sent = send(socket, ptr, static_cast<int>(remaining), 0);
        if (sent <= 0) {
            spdlog::error("send failed: {}", SOCKET_ERROR_CODE);
            return false;
        }
        ptr += sent;
        remaining -= sent;
    }

    return true;
}

bool CPUProcessGroup::RecvAll(SocketType socket, void* data, size_t size) {
    char* ptr = static_cast<char*>(data);
    size_t remaining = size;

    while (remaining > 0) {
        int received = recv(socket, ptr, static_cast<int>(remaining), 0);
        if (received <= 0) {
            spdlog::error("recv failed: {}", SOCKET_ERROR_CODE);
            return false;
        }
        ptr += received;
        remaining -= received;
    }

    return true;
}

bool CPUProcessGroup::SendTensor(int peer_rank, const Tensor& tensor) {
    if (peer_rank < 0 || peer_rank >= world_size_ || peer_rank == rank_) {
        return false;
    }

    SocketType sock = peer_sockets_[peer_rank];
    if (sock == INVALID_SOCKET_VALUE) {
        return false;
    }

    // Send size first
    size_t size = tensor.NumElements();
    if (!SendAll(sock, &size, sizeof(size))) {
        return false;
    }

    // Send data
    return SendAll(sock, tensor.Data<float>(), size * sizeof(float));
}

bool CPUProcessGroup::RecvTensor(int peer_rank, Tensor& tensor) {
    if (peer_rank < 0 || peer_rank >= world_size_ || peer_rank == rank_) {
        return false;
    }

    SocketType sock = peer_sockets_[peer_rank];
    if (sock == INVALID_SOCKET_VALUE) {
        return false;
    }

    // Receive size
    size_t size;
    if (!RecvAll(sock, &size, sizeof(size))) {
        return false;
    }

    // Allocate and receive data
    std::vector<size_t> shape = {size};
    tensor = Tensor(shape);
    return RecvAll(sock, tensor.Data<float>(), size * sizeof(float));
}

// ========== Reduction Operations ==========

void CPUProcessGroup::ApplyReduceOp(float* dst, const float* src, size_t count, ReduceOp op) {
    switch (op) {
        case ReduceOp::SUM:
        case ReduceOp::AVERAGE:  // Average is handled in finalize
            for (size_t i = 0; i < count; ++i) {
                dst[i] += src[i];
            }
            break;

        case ReduceOp::PRODUCT:
            for (size_t i = 0; i < count; ++i) {
                dst[i] *= src[i];
            }
            break;

        case ReduceOp::MIN:
            for (size_t i = 0; i < count; ++i) {
                dst[i] = std::min(dst[i], src[i]);
            }
            break;

        case ReduceOp::MAX:
            for (size_t i = 0; i < count; ++i) {
                dst[i] = std::max(dst[i], src[i]);
            }
            break;
    }
}

void CPUProcessGroup::ApplyReduceOpFinalize(float* data, size_t count, ReduceOp op) {
    if (op == ReduceOp::AVERAGE) {
        float scale = 1.0f / static_cast<float>(world_size_);
        for (size_t i = 0; i < count; ++i) {
            data[i] *= scale;
        }
    }
}

// ========== Ring All-Reduce ==========

void CPUProcessGroup::RingAllReduce(float* data, size_t count, ReduceOp op) {
    if (world_size_ == 1) {
        ApplyReduceOpFinalize(data, count, op);
        return;
    }

    // Calculate chunk sizes
    size_t chunk_size = (count + world_size_ - 1) / world_size_;
    std::vector<size_t> chunk_offsets(world_size_ + 1);
    for (int i = 0; i <= world_size_; ++i) {
        chunk_offsets[i] = std::min(static_cast<size_t>(i) * chunk_size, count);
    }

    // Buffer for receiving data
    std::vector<float> recv_buffer(chunk_size);

    int prev_rank = (rank_ - 1 + world_size_) % world_size_;
    int next_rank = (rank_ + 1) % world_size_;

    SocketType prev_sock = peer_sockets_[prev_rank];
    SocketType next_sock = peer_sockets_[next_rank];

    // ========== Phase 1: Scatter-Reduce ==========
    // After N-1 steps, each rank has the fully reduced result for one chunk

    for (int step = 0; step < world_size_ - 1; ++step) {
        // Determine which chunk to send and receive
        int send_chunk = (rank_ - step + world_size_) % world_size_;
        int recv_chunk = (rank_ - step - 1 + world_size_) % world_size_;

        size_t send_offset = chunk_offsets[send_chunk];
        size_t send_size = chunk_offsets[send_chunk + 1] - send_offset;

        size_t recv_offset = chunk_offsets[recv_chunk];
        size_t recv_size = chunk_offsets[recv_chunk + 1] - recv_offset;

        // Send to next rank
        if (!SendAll(next_sock, data + send_offset, send_size * sizeof(float))) {
            spdlog::error("RingAllReduce: send failed at step {}", step);
            return;
        }

        // Receive from previous rank
        if (!RecvAll(prev_sock, recv_buffer.data(), recv_size * sizeof(float))) {
            spdlog::error("RingAllReduce: recv failed at step {}", step);
            return;
        }

        // Reduce received data into local buffer
        ApplyReduceOp(data + recv_offset, recv_buffer.data(), recv_size, op);
    }

    // ========== Phase 2: All-Gather ==========
    // Distribute the fully reduced chunks to all ranks

    for (int step = 0; step < world_size_ - 1; ++step) {
        // Determine which chunk to send and receive
        int send_chunk = (rank_ - step + 1 + world_size_) % world_size_;
        int recv_chunk = (rank_ - step + world_size_) % world_size_;

        size_t send_offset = chunk_offsets[send_chunk];
        size_t send_size = chunk_offsets[send_chunk + 1] - send_offset;

        size_t recv_offset = chunk_offsets[recv_chunk];
        size_t recv_size = chunk_offsets[recv_chunk + 1] - recv_offset;

        // Send to next rank
        if (!SendAll(next_sock, data + send_offset, send_size * sizeof(float))) {
            spdlog::error("RingAllReduce (gather): send failed at step {}", step);
            return;
        }

        // Receive from previous rank (directly into final position)
        if (!RecvAll(prev_sock, data + recv_offset, recv_size * sizeof(float))) {
            spdlog::error("RingAllReduce (gather): recv failed at step {}", step);
            return;
        }
    }

    // Finalize (e.g., divide by world_size for AVERAGE)
    ApplyReduceOpFinalize(data, count, op);
}

// ========== Collective Operations ==========

void CPUProcessGroup::AllReduce(Tensor& tensor, ReduceOp op) {
    std::lock_guard<std::mutex> lock(socket_mutex_);

    if (!initialized_) {
        spdlog::error("CPUProcessGroup::AllReduce: not initialized");
        return;
    }

    float* data = tensor.Data<float>();
    size_t count = tensor.NumElements();

    RingAllReduce(data, count, op);
}

void CPUProcessGroup::AllReduceMultiple(std::vector<Tensor*>& tensors, ReduceOp op) {
    // For efficiency, we could fuse tensors into a single buffer
    // For now, just reduce each tensor sequentially
    for (auto* tensor : tensors) {
        if (tensor) {
            AllReduce(*tensor, op);
        }
    }
}

void CPUProcessGroup::Broadcast(Tensor& tensor, int src_rank) {
    std::lock_guard<std::mutex> lock(socket_mutex_);

    if (!initialized_) {
        spdlog::error("CPUProcessGroup::Broadcast: not initialized");
        return;
    }

    float* data = tensor.Data<float>();
    size_t count = tensor.NumElements();

    RingBroadcast(data, count, src_rank);
}

void CPUProcessGroup::RingBroadcast(float* data, size_t count, int src_rank) {
    if (world_size_ == 1) {
        return;
    }

    // Simple tree broadcast from src_rank
    // For simplicity, we use a linear chain: src -> (src+1) -> (src+2) -> ...

    if (rank_ == src_rank) {
        // Source rank: send to all others
        for (int peer = 0; peer < world_size_; ++peer) {
            if (peer != rank_) {
                SendAll(peer_sockets_[peer], data, count * sizeof(float));
            }
        }
    } else {
        // Receiver: receive from source
        RecvAll(peer_sockets_[src_rank], data, count * sizeof(float));
    }
}

void CPUProcessGroup::Barrier() {
    std::lock_guard<std::mutex> lock(socket_mutex_);

    if (!initialized_) {
        return;
    }

    RingBarrier();
}

void CPUProcessGroup::RingBarrier() {
    if (world_size_ == 1) {
        return;
    }

    // Simple barrier using ring communication
    // Send a byte around the ring twice to ensure synchronization

    int prev_rank = (rank_ - 1 + world_size_) % world_size_;
    int next_rank = (rank_ + 1) % world_size_;

    SocketType prev_sock = peer_sockets_[prev_rank];
    SocketType next_sock = peer_sockets_[next_rank];

    char sync_byte = 1;
    char recv_byte;

    // Round 1: Forward pass
    if (rank_ == 0) {
        // Rank 0 initiates
        SendAll(next_sock, &sync_byte, 1);
        RecvAll(prev_sock, &recv_byte, 1);
    } else {
        // Other ranks wait for previous, then send to next
        RecvAll(prev_sock, &recv_byte, 1);
        SendAll(next_sock, &sync_byte, 1);
    }

    // Round 2: Backward pass (ensures all ranks have passed barrier)
    if (rank_ == 0) {
        SendAll(next_sock, &sync_byte, 1);
        RecvAll(prev_sock, &recv_byte, 1);
    } else {
        RecvAll(prev_sock, &recv_byte, 1);
        SendAll(next_sock, &sync_byte, 1);
    }
}

std::vector<Tensor> CPUProcessGroup::AllGather(const Tensor& input) {
    std::lock_guard<std::mutex> lock(socket_mutex_);

    if (!initialized_) {
        spdlog::error("CPUProcessGroup::AllGather: not initialized");
        return {};
    }

    size_t input_count = input.NumElements();
    std::vector<float> gathered_data(input_count * world_size_);

    RingAllGather(input.Data<float>(), input_count, gathered_data);

    // Split into separate tensors
    std::vector<Tensor> result;
    result.reserve(world_size_);

    for (int r = 0; r < world_size_; ++r) {
        std::vector<size_t> shape = {input_count};
        Tensor t(shape, gathered_data.data() + r * input_count);
        result.push_back(std::move(t));
    }

    return result;
}

void CPUProcessGroup::RingAllGather(const float* input, size_t input_count,
                                     std::vector<float>& output) {
    // Copy local data to output
    std::copy(input, input + input_count, output.begin() + rank_ * input_count);

    if (world_size_ == 1) {
        return;
    }

    int prev_rank = (rank_ - 1 + world_size_) % world_size_;
    int next_rank = (rank_ + 1) % world_size_;

    SocketType prev_sock = peer_sockets_[prev_rank];
    SocketType next_sock = peer_sockets_[next_rank];

    // Ring all-gather: pass data around the ring N-1 times
    for (int step = 0; step < world_size_ - 1; ++step) {
        int send_rank = (rank_ - step + world_size_) % world_size_;
        int recv_rank = (rank_ - step - 1 + world_size_) % world_size_;

        // Send this rank's chunk of data
        SendAll(next_sock, output.data() + send_rank * input_count,
                input_count * sizeof(float));

        // Receive previous rank's chunk
        RecvAll(prev_sock, output.data() + recv_rank * input_count,
                input_count * sizeof(float));
    }
}

void CPUProcessGroup::ReduceScatter(Tensor& tensor, ReduceOp op) {
    std::lock_guard<std::mutex> lock(socket_mutex_);

    if (!initialized_) {
        spdlog::error("CPUProcessGroup::ReduceScatter: not initialized");
        return;
    }

    // Use base class implementation
    ProcessGroup::ReduceScatter(tensor, op);
}

} // namespace cyxwiz
