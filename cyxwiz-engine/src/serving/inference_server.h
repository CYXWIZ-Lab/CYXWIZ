#pragma once

#include <string>
#include <vector>
#include <atomic>
#include <mutex>
#include <memory>
#include <thread>
#include <chrono>
#include <deque>
#include <functional>

namespace cyxwiz {

// Forward declarations
class SequentialModel;

// Request log entry
struct RequestLogEntry {
    std::chrono::system_clock::time_point timestamp;
    std::string endpoint;
    std::string method;
    int status_code = 200;
    float latency_ms = 0.0f;
    std::string error_message;
};

// Server metrics (thread-safe)
struct ServerMetrics {
    std::atomic<uint64_t> total_requests{0};
    std::atomic<uint64_t> successful_requests{0};
    std::atomic<uint64_t> failed_requests{0};
    std::atomic<float> avg_latency_ms{0.0f};
    std::atomic<float> min_latency_ms{std::numeric_limits<float>::max()};
    std::atomic<float> max_latency_ms{0.0f};
    std::atomic<float> requests_per_second{0.0f};
    std::chrono::steady_clock::time_point start_time;

    void Reset() {
        total_requests = 0;
        successful_requests = 0;
        failed_requests = 0;
        avg_latency_ms = 0.0f;
        min_latency_ms = std::numeric_limits<float>::max();
        max_latency_ms = 0.0f;
        requests_per_second = 0.0f;
        start_time = std::chrono::steady_clock::now();
    }
};

// Inference server for serving ML models via REST API
class InferenceServer {
public:
    InferenceServer();
    ~InferenceServer();

    // Server lifecycle
    bool Start(int port = 8080);
    void Stop();
    bool IsRunning() const { return running_; }

    // Model management
    void SetModel(SequentialModel* model);
    SequentialModel* GetModel() const { return model_; }
    void SetModelName(const std::string& name) { model_name_ = name; }
    std::string GetModelName() const { return model_name_; }

    // Server info
    std::string GetServerUrl() const;
    int GetPort() const { return port_; }

    // Metrics
    const ServerMetrics& GetMetrics() const { return metrics_; }
    void ResetMetrics() { metrics_.Reset(); }

    // Request log
    std::vector<RequestLogEntry> GetRecentRequests(size_t count = 100) const;
    void ClearRequestLog();

    // Callbacks
    using StatusCallback = std::function<void(bool running, const std::string& message)>;
    void SetStatusCallback(StatusCallback callback) { status_callback_ = callback; }

private:
    void ServerThread();

    // HTTP handlers
    void HandleHealth();
    void HandleInfo();
    void HandlePredict();
    void HandleMetrics();

    // Request logging
    void LogRequest(const RequestLogEntry& entry);
    void UpdateMetrics(float latency_ms, bool success);

    // Model
    SequentialModel* model_ = nullptr;
    std::string model_name_ = "Unnamed Model";

    // Server state
    std::atomic<bool> running_{false};
    std::atomic<bool> should_stop_{false};
    int port_ = 8080;

    // Threading
    std::unique_ptr<std::thread> server_thread_;

    // Metrics
    mutable std::mutex metrics_mutex_;
    ServerMetrics metrics_;

    // Request log (ring buffer)
    mutable std::mutex log_mutex_;
    std::deque<RequestLogEntry> request_log_;
    static constexpr size_t MAX_LOG_ENTRIES = 1000;

    // Latency history for RPS calculation
    mutable std::mutex latency_mutex_;
    std::deque<std::pair<std::chrono::steady_clock::time_point, float>> latency_history_;
    static constexpr size_t MAX_LATENCY_HISTORY = 100;

    // Callbacks
    StatusCallback status_callback_;

    // Server implementation (pimpl pattern)
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace cyxwiz
