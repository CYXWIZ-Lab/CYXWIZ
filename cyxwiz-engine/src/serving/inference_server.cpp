// Windows header order fix - must come before httplib.h
#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <winsock2.h>
#include <ws2tcpip.h>
#endif

#include "inference_server.h"
#include <cyxwiz/sequential.h>
#include <cyxwiz/tensor.h>
#include <httplib.h>
#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>
#include <algorithm>
#include <numeric>

namespace cyxwiz {

using json = nlohmann::json;

// Implementation using cpp-httplib
class InferenceServer::Impl {
public:
    httplib::Server server;
};

InferenceServer::InferenceServer()
    : impl_(std::make_unique<Impl>()) {
    metrics_.Reset();
}

InferenceServer::~InferenceServer() {
    Stop();
}

bool InferenceServer::Start(int port) {
    if (running_) {
        spdlog::warn("InferenceServer: Already running");
        return false;
    }

    port_ = port;
    should_stop_ = false;

    // Set up routes
    impl_->server.Get("/health", [this](const httplib::Request& req, httplib::Response& res) {
        auto start = std::chrono::high_resolution_clock::now();

        json response;
        response["status"] = "ok";
        response["server"] = "CyxWiz Inference Server";
        response["model_loaded"] = (model_ != nullptr);

        res.set_content(response.dump(), "application/json");

        auto end = std::chrono::high_resolution_clock::now();
        float latency = std::chrono::duration<float, std::milli>(end - start).count();

        RequestLogEntry entry;
        entry.timestamp = std::chrono::system_clock::now();
        entry.endpoint = "/health";
        entry.method = "GET";
        entry.status_code = 200;
        entry.latency_ms = latency;
        LogRequest(entry);
        UpdateMetrics(latency, true);
    });

    impl_->server.Get("/info", [this](const httplib::Request& req, httplib::Response& res) {
        auto start = std::chrono::high_resolution_clock::now();

        json response;
        response["model_name"] = model_name_;
        response["model_loaded"] = (model_ != nullptr);

        if (model_) {
            // Get basic model info
            response["num_layers"] = model_->Size();
            response["status"] = "ready";
        } else {
            response["status"] = "no_model";
        }

        res.set_content(response.dump(2), "application/json");

        auto end = std::chrono::high_resolution_clock::now();
        float latency = std::chrono::duration<float, std::milli>(end - start).count();

        RequestLogEntry entry;
        entry.timestamp = std::chrono::system_clock::now();
        entry.endpoint = "/info";
        entry.method = "GET";
        entry.status_code = 200;
        entry.latency_ms = latency;
        LogRequest(entry);
        UpdateMetrics(latency, true);
    });

    impl_->server.Post("/predict", [this](const httplib::Request& req, httplib::Response& res) {
        auto start = std::chrono::high_resolution_clock::now();
        int status_code = 200;
        bool success = true;
        std::string error_msg;

        try {
            if (!model_) {
                throw std::runtime_error("No model loaded");
            }

            // Parse JSON body
            auto body = json::parse(req.body);

            if (!body.contains("input")) {
                throw std::runtime_error("Missing 'input' field in request body");
            }

            // Convert input to tensor
            std::vector<float> input_data;
            if (body["input"].is_array()) {
                // Handle nested arrays (batch input)
                std::function<void(const json&)> flatten = [&](const json& arr) {
                    for (const auto& item : arr) {
                        if (item.is_array()) {
                            flatten(item);
                        } else if (item.is_number()) {
                            input_data.push_back(item.get<float>());
                        }
                    }
                };
                flatten(body["input"]);
            } else {
                throw std::runtime_error("'input' must be an array");
            }

            if (input_data.empty()) {
                throw std::runtime_error("Empty input data");
            }

            // Create input tensor
            // Infer shape from first layer or use provided shape
            std::vector<size_t> input_shape;
            if (body.contains("shape") && body["shape"].is_array()) {
                for (const auto& dim : body["shape"]) {
                    input_shape.push_back(static_cast<size_t>(dim.get<int64_t>()));
                }
            } else {
                // Assume 1D batch with size = input_data.size()
                input_shape = {1, input_data.size()};
            }

            Tensor input_tensor(input_shape, input_data.data(), DataType::Float32);

            // Run inference
            auto output_tensor = model_->Forward(input_tensor);

            // Get output data as vector
            const float* output_ptr = output_tensor.Data<float>();
            std::vector<float> output_data(output_ptr, output_ptr + output_tensor.NumElements());

            // Build response
            json response;
            response["output"] = output_data;
            response["shape"] = output_tensor.Shape();

            auto end = std::chrono::high_resolution_clock::now();
            float latency = std::chrono::duration<float, std::milli>(end - start).count();
            response["latency_ms"] = latency;

            res.set_content(response.dump(), "application/json");

            RequestLogEntry entry;
            entry.timestamp = std::chrono::system_clock::now();
            entry.endpoint = "/predict";
            entry.method = "POST";
            entry.status_code = 200;
            entry.latency_ms = latency;
            LogRequest(entry);
            UpdateMetrics(latency, true);

        } catch (const std::exception& e) {
            success = false;
            status_code = 400;
            error_msg = e.what();

            json response;
            response["error"] = e.what();
            res.status = 400;
            res.set_content(response.dump(), "application/json");

            auto end = std::chrono::high_resolution_clock::now();
            float latency = std::chrono::duration<float, std::milli>(end - start).count();

            RequestLogEntry entry;
            entry.timestamp = std::chrono::system_clock::now();
            entry.endpoint = "/predict";
            entry.method = "POST";
            entry.status_code = 400;
            entry.latency_ms = latency;
            entry.error_message = e.what();
            LogRequest(entry);
            UpdateMetrics(latency, false);
        }
    });

    impl_->server.Get("/metrics", [this](const httplib::Request& req, httplib::Response& res) {
        auto start = std::chrono::high_resolution_clock::now();

        json response;
        response["total_requests"] = metrics_.total_requests.load();
        response["successful_requests"] = metrics_.successful_requests.load();
        response["failed_requests"] = metrics_.failed_requests.load();
        response["avg_latency_ms"] = metrics_.avg_latency_ms.load();
        response["min_latency_ms"] = metrics_.min_latency_ms.load();
        response["max_latency_ms"] = metrics_.max_latency_ms.load();
        response["requests_per_second"] = metrics_.requests_per_second.load();

        auto uptime = std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::steady_clock::now() - metrics_.start_time).count();
        response["uptime_seconds"] = uptime;

        res.set_content(response.dump(2), "application/json");

        auto end = std::chrono::high_resolution_clock::now();
        float latency = std::chrono::duration<float, std::milli>(end - start).count();

        RequestLogEntry entry;
        entry.timestamp = std::chrono::system_clock::now();
        entry.endpoint = "/metrics";
        entry.method = "GET";
        entry.status_code = 200;
        entry.latency_ms = latency;
        LogRequest(entry);
    });

    // Enable CORS for development
    impl_->server.set_default_headers({
        {"Access-Control-Allow-Origin", "*"},
        {"Access-Control-Allow-Methods", "GET, POST, OPTIONS"},
        {"Access-Control-Allow-Headers", "Content-Type"}
    });

    // Handle OPTIONS for CORS preflight
    impl_->server.Options(".*", [](const httplib::Request& req, httplib::Response& res) {
        res.status = 204;
    });

    // Start server thread
    server_thread_ = std::make_unique<std::thread>([this]() {
        running_ = true;
        spdlog::info("InferenceServer: Starting on port {}", port_);

        if (status_callback_) {
            status_callback_(true, "Server started on port " + std::to_string(port_));
        }

        if (!impl_->server.listen("0.0.0.0", port_)) {
            if (!should_stop_) {
                spdlog::error("InferenceServer: Failed to start on port {}", port_);
                if (status_callback_) {
                    status_callback_(false, "Failed to start on port " + std::to_string(port_));
                }
            }
        }

        running_ = false;
        spdlog::info("InferenceServer: Stopped");
    });

    // Wait briefly to check if server started
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    return running_;
}

void InferenceServer::Stop() {
    if (!running_ && !server_thread_) {
        return;
    }

    should_stop_ = true;
    impl_->server.stop();

    if (server_thread_ && server_thread_->joinable()) {
        server_thread_->join();
    }
    server_thread_.reset();

    running_ = false;

    if (status_callback_) {
        status_callback_(false, "Server stopped");
    }

    spdlog::info("InferenceServer: Shutdown complete");
}

void InferenceServer::SetModel(SequentialModel* model) {
    model_ = model;
}

std::string InferenceServer::GetServerUrl() const {
    return "http://localhost:" + std::to_string(port_);
}

std::vector<RequestLogEntry> InferenceServer::GetRecentRequests(size_t count) const {
    std::lock_guard<std::mutex> lock(log_mutex_);
    size_t actual_count = std::min(count, request_log_.size());
    std::vector<RequestLogEntry> result;
    result.reserve(actual_count);

    // Return most recent first
    auto it = request_log_.rbegin();
    for (size_t i = 0; i < actual_count && it != request_log_.rend(); ++i, ++it) {
        result.push_back(*it);
    }
    return result;
}

void InferenceServer::ClearRequestLog() {
    std::lock_guard<std::mutex> lock(log_mutex_);
    request_log_.clear();
}

void InferenceServer::LogRequest(const RequestLogEntry& entry) {
    std::lock_guard<std::mutex> lock(log_mutex_);
    request_log_.push_back(entry);

    // Trim to max size
    while (request_log_.size() > MAX_LOG_ENTRIES) {
        request_log_.pop_front();
    }
}

void InferenceServer::UpdateMetrics(float latency_ms, bool success) {
    // Update counters
    metrics_.total_requests++;
    if (success) {
        metrics_.successful_requests++;
    } else {
        metrics_.failed_requests++;
    }

    // Update latency stats
    {
        std::lock_guard<std::mutex> lock(latency_mutex_);

        // Add to history
        auto now = std::chrono::steady_clock::now();
        latency_history_.push_back({now, latency_ms});

        // Trim old entries
        while (latency_history_.size() > MAX_LATENCY_HISTORY) {
            latency_history_.pop_front();
        }

        // Calculate stats
        if (!latency_history_.empty()) {
            float sum = 0;
            float min_lat = std::numeric_limits<float>::max();
            float max_lat = 0;

            for (const auto& entry : latency_history_) {
                sum += entry.second;
                min_lat = std::min(min_lat, entry.second);
                max_lat = std::max(max_lat, entry.second);
            }

            metrics_.avg_latency_ms = sum / latency_history_.size();
            metrics_.min_latency_ms = min_lat;
            metrics_.max_latency_ms = max_lat;

            // Calculate RPS over last second
            auto one_second_ago = now - std::chrono::seconds(1);
            int recent_count = 0;
            for (const auto& entry : latency_history_) {
                if (entry.first >= one_second_ago) {
                    recent_count++;
                }
            }
            metrics_.requests_per_second = static_cast<float>(recent_count);
        }
    }
}

} // namespace cyxwiz
