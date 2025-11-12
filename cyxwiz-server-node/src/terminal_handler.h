#pragma once

#include <memory>
#include <string>
#include <atomic>
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <grpcpp/grpcpp.h>
#include "deployment.pb.h"
#include "deployment.grpc.pb.h"

namespace cyxwiz {
namespace servernode {

// Forward declarations
class PtySession;
class TerminalServiceImpl;

// PTY session manager for a single terminal
class PtySession {
public:
    PtySession(const std::string& session_id, int rows, int cols);
    ~PtySession();

    // Start PTY and shell
    bool Start();

    // Stop PTY and shell
    void Stop();

    // Write data to PTY (from user)
    bool Write(const std::string& data);

    // Read data from PTY (to user) - non-blocking
    bool Read(std::string& data);

    // Resize terminal
    bool Resize(int rows, int cols);

    // Check if session is active
    bool IsActive() const { return active_; }

    // Get session ID
    const std::string& GetSessionId() const { return session_id_; }

private:
    void ReadLoop();  // Background thread for reading from PTY

private:
    std::string session_id_;
    int rows_;
    int cols_;
    std::atomic<bool> active_{false};
    std::atomic<bool> should_stop_{false};

#ifdef _WIN32
    // Windows ConPTY handles
    void* hPC_;           // HPCON
    void* hPipeIn_;       // Input pipe
    void* hPipeOut_;      // Output pipe
    void* hProcess_;      // Shell process
#else
    // Unix/Linux PTY file descriptor
    int master_fd_;
    int slave_fd_;
    pid_t child_pid_;
#endif

    // Output buffer
    std::queue<std::string> output_queue_;
    std::mutex output_mutex_;
    std::thread read_thread_;
};

// gRPC service implementation for terminal access
class TerminalServiceImpl final : public protocol::TerminalService::Service {
public:
    TerminalServiceImpl();
    ~TerminalServiceImpl() override;

    // Create a new terminal session
    grpc::Status CreateSession(
        grpc::ServerContext* context,
        const protocol::CreateTerminalSessionRequest* request,
        protocol::CreateTerminalSessionResponse* response) override;

    // Bidirectional streaming for terminal I/O
    grpc::Status StreamTerminal(
        grpc::ServerContext* context,
        grpc::ServerReaderWriter<protocol::TerminalData, protocol::TerminalData>* stream) override;

    // Resize terminal
    grpc::Status ResizeTerminal(
        grpc::ServerContext* context,
        const protocol::TerminalResize* request,
        protocol::SimpleResponse* response) override;

    // Close terminal session
    grpc::Status CloseSession(
        grpc::ServerContext* context,
        const protocol::CloseTerminalSessionRequest* request,
        protocol::CloseTerminalSessionResponse* response) override;

private:
    std::unordered_map<std::string, std::unique_ptr<PtySession>> sessions_;
    mutable std::mutex sessions_mutex_;
};

// Main handler class that manages terminal service lifecycle
class TerminalHandler {
public:
    TerminalHandler(const std::string& listen_address);
    ~TerminalHandler();

    // Start the gRPC server (blocking call)
    bool Start();

    // Stop the gRPC server gracefully
    void Stop();

    // Check if server is running
    bool IsRunning() const { return running_; }

private:
    std::string listen_address_;
    std::unique_ptr<grpc::Server> server_;
    std::unique_ptr<TerminalServiceImpl> service_;
    bool running_;
};

} // namespace servernode
} // namespace cyxwiz
