#include "terminal_handler.h"
#include <spdlog/spdlog.h>
#include <chrono>

#ifdef _WIN32
#include <windows.h>
// TODO: Include ConPTY headers when implementing
#elif defined(__APPLE__)
#include <unistd.h>
#include <fcntl.h>
#include <termios.h>
#include <sys/ioctl.h>
#include <sys/wait.h>
#include <util.h>  // macOS uses util.h for openpty()
#else
#include <unistd.h>
#include <fcntl.h>
#include <termios.h>
#include <sys/ioctl.h>
#include <sys/wait.h>
#include <pty.h>   // Linux uses pty.h for openpty()
#endif

namespace cyxwiz {
namespace servernode {

// ============================================================================
// PtySession Implementation
// ============================================================================

PtySession::PtySession(const std::string& session_id, int rows, int cols)
    : session_id_(session_id)
    , rows_(rows)
    , cols_(cols)
#ifdef _WIN32
    , hPC_(nullptr)
    , hPipeIn_(nullptr)
    , hPipeOut_(nullptr)
    , hProcess_(nullptr)
#else
    , master_fd_(-1)
    , slave_fd_(-1)
    , child_pid_(-1)
#endif
{
    spdlog::debug("PtySession created: {} ({}x{})", session_id, cols, rows);
}

PtySession::~PtySession() {
    Stop();
}

bool PtySession::Start() {
    if (active_) {
        spdlog::warn("PtySession already active: {}", session_id_);
        return true;
    }

    spdlog::info("Starting PTY session: {}", session_id_);

#ifdef _WIN32
    // TODO: Implement Windows ConPTY
    // This requires Windows 10 1809+ and conpty.h
    spdlog::warn("Windows ConPTY not yet implemented - using stub");
    active_ = true;
    return true;

#else
    // Unix/Linux: Use openpty() and fork()
    struct winsize ws;
    ws.ws_row = rows_;
    ws.ws_col = cols_;
    ws.ws_xpixel = 0;
    ws.ws_ypixel = 0;

    // Create PTY
    if (openpty(&master_fd_, &slave_fd_, nullptr, nullptr, &ws) == -1) {
        spdlog::error("Failed to create PTY: {}", strerror(errno));
        return false;
    }

    // Fork child process
    child_pid_ = fork();
    if (child_pid_ == -1) {
        spdlog::error("Failed to fork: {}", strerror(errno));
        close(master_fd_);
        close(slave_fd_);
        return false;
    }

    if (child_pid_ == 0) {
        // Child process: Set up terminal and exec shell
        close(master_fd_);

        // Create new session
        if (setsid() == -1) {
            _exit(1);
        }

        // Set controlling terminal
        if (ioctl(slave_fd_, TIOCSCTTY, 0) == -1) {
            _exit(1);
        }

        // Redirect stdio
        dup2(slave_fd_, STDIN_FILENO);
        dup2(slave_fd_, STDOUT_FILENO);
        dup2(slave_fd_, STDERR_FILENO);
        close(slave_fd_);

        // Set environment
        setenv("TERM", "xterm-256color", 1);

        // Execute shell
        execl("/bin/bash", "bash", nullptr);
        _exit(1);  // Should never reach here
    }

    // Parent process
    close(slave_fd_);
    slave_fd_ = -1;

    // Set master fd to non-blocking
    int flags = fcntl(master_fd_, F_GETFL, 0);
    fcntl(master_fd_, F_SETFL, flags | O_NONBLOCK);

    active_ = true;

    // Start read thread
    read_thread_ = std::thread([this]() { ReadLoop(); });

    spdlog::info("PTY session started: {} (pid={})", session_id_, child_pid_);
    return true;
#endif
}

void PtySession::Stop() {
    if (!active_) {
        return;
    }

    spdlog::info("Stopping PTY session: {}", session_id_);
    should_stop_ = true;

#ifdef _WIN32
    // TODO: Cleanup Windows ConPTY resources
    if (hProcess_) {
        TerminateProcess(hProcess_, 0);
        CloseHandle(hProcess_);
        hProcess_ = nullptr;
    }
    if (hPipeIn_) CloseHandle(hPipeIn_);
    if (hPipeOut_) CloseHandle(hPipeOut_);
    if (hPC_) {
        // ClosePseudoConsole(hPC_);
    }
#else
    if (master_fd_ >= 0) {
        close(master_fd_);
        master_fd_ = -1;
    }

    if (child_pid_ > 0) {
        kill(child_pid_, SIGTERM);
        waitpid(child_pid_, nullptr, 0);
        child_pid_ = -1;
    }
#endif

    if (read_thread_.joinable()) {
        read_thread_.join();
    }

    active_ = false;
    spdlog::info("PTY session stopped: {}", session_id_);
}

bool PtySession::Write(const std::string& data) {
    if (!active_) {
        return false;
    }

#ifdef _WIN32
    // TODO: Write to ConPTY
    spdlog::debug("Write to PTY (stub): {} bytes", data.size());
    return true;
#else
    if (master_fd_ < 0) {
        return false;
    }

    ssize_t written = write(master_fd_, data.c_str(), data.size());
    if (written < 0) {
        spdlog::error("Failed to write to PTY: {}", strerror(errno));
        return false;
    }

    spdlog::debug("Wrote {} bytes to PTY: {}", written, session_id_);
    return true;
#endif
}

bool PtySession::Read(std::string& data) {
    std::lock_guard<std::mutex> lock(output_mutex_);

    if (output_queue_.empty()) {
        return false;
    }

    data = std::move(output_queue_.front());
    output_queue_.pop();
    return true;
}

bool PtySession::Resize(int rows, int cols) {
    if (!active_) {
        return false;
    }

    spdlog::debug("Resizing PTY {} to {}x{}", session_id_, cols, rows);
    rows_ = rows;
    cols_ = cols;

#ifdef _WIN32
    // TODO: Resize ConPTY
    return true;
#else
    if (master_fd_ < 0) {
        return false;
    }

    struct winsize ws;
    ws.ws_row = rows;
    ws.ws_col = cols;
    ws.ws_xpixel = 0;
    ws.ws_ypixel = 0;

    if (ioctl(master_fd_, TIOCSWINSZ, &ws) == -1) {
        spdlog::error("Failed to resize PTY: {}", strerror(errno));
        return false;
    }

    return true;
#endif
}

void PtySession::ReadLoop() {
    spdlog::debug("PTY read loop started: {}", session_id_);

#ifndef _WIN32
    char buffer[4096];

    while (!should_stop_ && active_) {
        ssize_t n = read(master_fd_, buffer, sizeof(buffer));

        if (n > 0) {
            std::lock_guard<std::mutex> lock(output_mutex_);
            output_queue_.emplace(buffer, n);
        } else if (n == 0) {
            // EOF
            spdlog::info("PTY EOF: {}", session_id_);
            break;
        } else if (errno != EAGAIN && errno != EWOULDBLOCK) {
            spdlog::error("PTY read error: {}", strerror(errno));
            break;
        }

        // Small sleep to avoid busy-waiting
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
#endif

    spdlog::debug("PTY read loop stopped: {}", session_id_);
}

// ============================================================================
// TerminalServiceImpl Implementation
// ============================================================================

TerminalServiceImpl::TerminalServiceImpl() {
    spdlog::debug("TerminalServiceImpl created");
}

TerminalServiceImpl::~TerminalServiceImpl() {
    std::lock_guard<std::mutex> lock(sessions_mutex_);
    sessions_.clear();
}

grpc::Status TerminalServiceImpl::CreateSession(
    grpc::ServerContext* context,
    const protocol::CreateTerminalSessionRequest* request,
    protocol::CreateTerminalSessionResponse* response) {

    spdlog::info("Creating terminal session for deployment: {}", request->deployment_id());

    try {
        // Generate session ID
        std::string session_id = "term_" + request->deployment_id() + "_" +
                                std::to_string(std::time(nullptr));

        // Create PTY session
        auto session = std::make_unique<PtySession>(
            session_id,
            request->rows(),
            request->cols()
        );

        if (!session->Start()) {
            return grpc::Status(grpc::StatusCode::INTERNAL, "Failed to start PTY session");
        }

        // Store session
        {
            std::lock_guard<std::mutex> lock(sessions_mutex_);
            sessions_[session_id] = std::move(session);
        }

        // Build response
        auto* session_info = response->mutable_session();
        session_info->set_session_id(session_id);
        session_info->set_deployment_id(request->deployment_id());
        session_info->set_status(protocol::TERMINAL_SESSION_ACTIVE);
        session_info->set_rows(request->rows());
        session_info->set_cols(request->cols());

        response->set_status(protocol::STATUS_SUCCESS);

        spdlog::info("Terminal session created: {}", session_id);
        return grpc::Status::OK;

    } catch (const std::exception& e) {
        spdlog::error("Failed to create terminal session: {}", e.what());
        return grpc::Status(grpc::StatusCode::INTERNAL, e.what());
    }
}

grpc::Status TerminalServiceImpl::StreamTerminal(
    grpc::ServerContext* context,
    grpc::ServerReaderWriter<protocol::TerminalData, protocol::TerminalData>* stream) {

    spdlog::debug("Terminal streaming started");

    protocol::TerminalData data;
    PtySession* session = nullptr;
    std::string session_id;

    // Read first message to get session ID
    if (!stream->Read(&data)) {
        return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, "No session ID provided");
    }

    session_id = data.session_id();

    // Find session
    {
        std::lock_guard<std::mutex> lock(sessions_mutex_);
        auto it = sessions_.find(session_id);
        if (it == sessions_.end()) {
            return grpc::Status(grpc::StatusCode::NOT_FOUND, "Session not found");
        }
        session = it->second.get();
    }

    spdlog::info("Terminal streaming for session: {}", session_id);

    // Start bidirectional streaming
    std::thread read_thread([&]() {
        while (stream->Read(&data)) {
            // Write user input to PTY
            if (!data.data().empty()) {
                session->Write(data.data());
            }
        }
    });

    // Read from PTY and send to client
    std::string output;
    while (session->IsActive() && !context->IsCancelled()) {
        if (session->Read(output)) {
            protocol::TerminalData response;
            response.set_session_id(session_id);
            response.set_data(output);

            if (!stream->Write(response)) {
                break;
            }
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    read_thread.join();

    spdlog::info("Terminal streaming ended for session: {}", session_id);
    return grpc::Status::OK;
}

grpc::Status TerminalServiceImpl::ResizeTerminal(
    grpc::ServerContext* context,
    const protocol::TerminalResize* request,
    protocol::SimpleResponse* response) {

    spdlog::debug("Resize terminal: {} ({}x{})",
                 request->session_id(), request->cols(), request->rows());

    std::lock_guard<std::mutex> lock(sessions_mutex_);
    auto it = sessions_.find(request->session_id());
    if (it == sessions_.end()) {
        response->set_status(protocol::STATUS_ERROR);
        response->mutable_error()->set_message("Session not found");
        return grpc::Status(grpc::StatusCode::NOT_FOUND, "Session not found");
    }

    if (it->second->Resize(request->rows(), request->cols())) {
        response->set_status(protocol::STATUS_SUCCESS);
        return grpc::Status::OK;
    } else {
        response->set_status(protocol::STATUS_ERROR);
        response->mutable_error()->set_message("Failed to resize terminal");
        return grpc::Status(grpc::StatusCode::INTERNAL, "Failed to resize terminal");
    }
}

grpc::Status TerminalServiceImpl::CloseSession(
    grpc::ServerContext* context,
    const protocol::CloseTerminalSessionRequest* request,
    protocol::CloseTerminalSessionResponse* response) {

    spdlog::info("Closing terminal session: {}", request->session_id());

    std::lock_guard<std::mutex> lock(sessions_mutex_);
    auto it = sessions_.find(request->session_id());
    if (it == sessions_.end()) {
        response->set_status(protocol::STATUS_ERROR);
        response->mutable_error()->set_message("Session not found");
        return grpc::Status(grpc::StatusCode::NOT_FOUND, "Session not found");
    }

    sessions_.erase(it);
    response->set_status(protocol::STATUS_SUCCESS);

    spdlog::info("Terminal session closed: {}", request->session_id());
    return grpc::Status::OK;
}

// ============================================================================
// TerminalHandler Implementation
// ============================================================================

TerminalHandler::TerminalHandler(const std::string& listen_address)
    : listen_address_(listen_address)
    , running_(false) {
    spdlog::info("TerminalHandler created for address: {}", listen_address);
}

TerminalHandler::~TerminalHandler() {
    Stop();
}

bool TerminalHandler::Start() {
    if (running_) {
        spdlog::warn("TerminalHandler already running");
        return true;
    }

    try {
        service_ = std::make_unique<TerminalServiceImpl>();

        grpc::ServerBuilder builder;
        builder.AddListeningPort(listen_address_, grpc::InsecureServerCredentials());
        builder.RegisterService(service_.get());

        server_ = builder.BuildAndStart();
        if (!server_) {
            spdlog::error("Failed to start gRPC server on {}", listen_address_);
            return false;
        }

        running_ = true;
        spdlog::info("TerminalHandler started successfully on {}", listen_address_);
        return true;

    } catch (const std::exception& e) {
        spdlog::error("Exception starting TerminalHandler: {}", e.what());
        return false;
    }
}

void TerminalHandler::Stop() {
    if (!running_) {
        return;
    }

    spdlog::info("Stopping TerminalHandler...");

    if (server_) {
        server_->Shutdown(std::chrono::system_clock::now() + std::chrono::seconds(5));
        server_->Wait();
        server_.reset();
    }

    service_.reset();
    running_ = false;

    spdlog::info("TerminalHandler stopped");
}

} // namespace servernode
} // namespace cyxwiz
