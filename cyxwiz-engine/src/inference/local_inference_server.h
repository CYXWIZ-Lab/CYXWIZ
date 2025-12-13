// local_inference_server.h - Embedded HTTP inference server for the Engine
#pragma once

#include <string>
#include <memory>
#include <thread>
#include <atomic>
#include <vector>
#include <mutex>

// Forward declare httplib to avoid including in header
namespace httplib {
class Server;
class Request;
class Response;
}

namespace cyxwiz {

// Forward declare
class SequentialModel;

/**
 * LocalInferenceServer - Embedded HTTP server for model inference
 *
 * Provides REST API endpoints for running inference on loaded models
 * directly within the Engine process, without requiring a Server Node.
 *
 * Endpoints:
 *   GET  /health       - Health check
 *   GET  /v1/model     - Model info (architecture, layers)
 *   POST /v1/predict   - Run inference
 */
class LocalInferenceServer {
public:
    LocalInferenceServer();
    ~LocalInferenceServer();

    // Non-copyable
    LocalInferenceServer(const LocalInferenceServer&) = delete;
    LocalInferenceServer& operator=(const LocalInferenceServer&) = delete;

    /**
     * Load a model from file for inference
     * @param model_path Path to .cyxmodel file
     * @return true if loaded successfully
     */
    bool LoadModel(const std::string& model_path);

    /**
     * Unload the current model
     */
    void UnloadModel();

    /**
     * Check if a model is loaded
     */
    bool HasModel() const;

    /**
     * Start serving on the specified port (non-blocking)
     * @param port Port number (e.g., 8080)
     * @return true if started successfully
     */
    bool Start(int port);

    /**
     * Stop the server
     */
    void Stop();

    /**
     * Check if server is running
     */
    bool IsRunning() const { return running_.load(); }

    /**
     * Get the current port
     */
    int GetPort() const { return port_; }

    /**
     * Get the loaded model path
     */
    const std::string& GetModelPath() const { return model_path_; }

    /**
     * Get the model name (filename without path)
     */
    std::string GetModelName() const;

    /**
     * Get total request count
     */
    uint64_t GetRequestCount() const { return request_count_.load(); }

    /**
     * Get last error message
     */
    const std::string& GetLastError() const { return last_error_; }

    /**
     * Get endpoint URL
     */
    std::string GetEndpointUrl() const;

private:
    // HTTP handlers
    void HandleHealth(const httplib::Request& req, httplib::Response& res);
    void HandleModelInfo(const httplib::Request& req, httplib::Response& res);
    void HandlePredict(const httplib::Request& req, httplib::Response& res);

    // Server thread function
    void ServerThread();

    // Register routes
    void RegisterRoutes();

private:
    std::unique_ptr<httplib::Server> server_;
    std::unique_ptr<std::thread> server_thread_;
    std::unique_ptr<SequentialModel> model_;

    std::string model_path_;
    int port_ = 0;
    std::atomic<bool> running_{false};
    std::atomic<uint64_t> request_count_{0};
    std::string last_error_;
    mutable std::mutex model_mutex_;
};

} // namespace cyxwiz
