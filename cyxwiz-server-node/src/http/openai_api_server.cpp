// openai_api_server.cpp - HTTP REST API server implementation
#include "openai_api_server.h"
#include "../deployment_manager.h"
#include <httplib.h>
#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>
#include <chrono>
#include <cyxwiz/tensor.h>

namespace cyxwiz::servernode {

using json = nlohmann::json;

OpenAIAPIServer::OpenAIAPIServer(int port)
    : port_(port)
    , deployment_manager_(nullptr)
    , server_(std::make_unique<httplib::Server>()) {
}

OpenAIAPIServer::OpenAIAPIServer(int port, DeploymentManager* deployment_manager)
    : port_(port)
    , deployment_manager_(deployment_manager)
    , server_(std::make_unique<httplib::Server>()) {
}

OpenAIAPIServer::~OpenAIAPIServer() {
    Stop();
}

void OpenAIAPIServer::SetDeploymentManager(DeploymentManager* manager) {
    deployment_manager_ = manager;
}

bool OpenAIAPIServer::Start() {
    if (running_) {
        spdlog::warn("HTTP server already running on port {}", port_);
        return true;
    }

    if (!server_) {
        spdlog::error("HTTP server not initialized");
        return false;
    }

    RegisterRoutes();

    // Start server in a separate thread
    server_thread_ = std::thread([this]() {
        spdlog::info("Starting HTTP REST API server on port {}", port_);
        running_ = true;

        if (!server_->listen("0.0.0.0", port_)) {
            spdlog::error("Failed to start HTTP server on port {}", port_);
            running_ = false;
        }
    });

    // Give server a moment to start
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    if (running_) {
        spdlog::info("HTTP REST API server started successfully on port {}", port_);
    }

    return running_;
}

void OpenAIAPIServer::Stop() {
    if (!running_) {
        return;
    }

    spdlog::info("Stopping HTTP server...");
    running_ = false;

    if (server_) {
        server_->stop();
    }

    if (server_thread_.joinable()) {
        server_thread_.join();
    }

    spdlog::info("HTTP server stopped");
}

bool OpenAIAPIServer::IsRunning() const {
    return running_;
}

void OpenAIAPIServer::RegisterRoutes() {
    // CORS middleware
    server_->set_default_headers({
        {"Access-Control-Allow-Origin", "*"},
        {"Access-Control-Allow-Methods", "GET, POST, OPTIONS"},
        {"Access-Control-Allow-Headers", "Content-Type, Authorization"}
    });

    // Handle OPTIONS preflight requests
    server_->Options(".*", [](const httplib::Request&, httplib::Response& res) {
        res.status = 204;  // No Content
    });

    // Health check
    server_->Get("/health", [this](const httplib::Request&, httplib::Response& res) {
        json response = {
            {"status", "healthy"},
            {"version", "1.0.0"},
            {"server_type", "cyxwiz-server-node"}
        };

        if (deployment_manager_) {
            response["active_deployments"] = deployment_manager_->GetActiveDeploymentCount();
        }

        res.set_content(response.dump(), "application/json");
    });

    // List models/deployments (OpenAI-compatible format)
    server_->Get("/v1/models", [this](const httplib::Request&, httplib::Response& res) {
        if (!deployment_manager_) {
            json error = {
                {"error", {
                    {"message", "Deployment manager not available"},
                    {"type", "server_error"},
                    {"code", "service_unavailable"}
                }}
            };
            res.status = 503;
            res.set_content(error.dump(), "application/json");
            return;
        }

        auto deployments = deployment_manager_->GetAllDeployments();
        json data = json::array();
        for (const auto& dep : deployments) {
            data.push_back({
                {"id", dep.id},
                {"object", "model"},
                {"created", 0},
                {"owned_by", "cyxwiz"}
            });
        }

        json response = {
            {"object", "list"},
            {"data", data}
        };

        res.set_content(response.dump(), "application/json");
    });

    // List all deployments with details
    server_->Get("/v1/deployments", [this](const httplib::Request&, httplib::Response& res) {
        if (!deployment_manager_) {
            json error = {
                {"error", {
                    {"message", "Deployment manager not available"},
                    {"type", "server_error"},
                    {"code", "service_unavailable"}
                }}
            };
            res.status = 503;
            res.set_content(error.dump(), "application/json");
            return;
        }

        auto deployments = deployment_manager_->GetAllDeployments();
        json data = json::array();
        for (const auto& dep : deployments) {
            data.push_back({
                {"deployment_id", dep.id},
                {"model_id", dep.model_id},
                {"type", static_cast<int>(dep.type)},
                {"status", static_cast<int>(dep.status)}
            });
        }

        json response = {
            {"deployments", data},
            {"count", deployments.size()}
        };

        res.set_content(response.dump(), "application/json");
    });

    // Predict endpoint
    server_->Post("/v1/predict", [this](const httplib::Request& req, httplib::Response& res) {
        auto start_time = std::chrono::high_resolution_clock::now();

        // Check deployment manager
        if (!deployment_manager_) {
            json error = {
                {"error", {
                    {"message", "Deployment manager not available"},
                    {"type", "server_error"},
                    {"code", "service_unavailable"}
                }}
            };
            res.status = 503;
            res.set_content(error.dump(), "application/json");
            return;
        }

        // Parse request body
        json request_body;
        try {
            request_body = json::parse(req.body);
        } catch (const json::exception& e) {
            json error = {
                {"error", {
                    {"message", std::string("Invalid JSON: ") + e.what()},
                    {"type", "invalid_request_error"},
                    {"code", "parse_error"}
                }}
            };
            res.status = 400;
            res.set_content(error.dump(), "application/json");
            return;
        }

        // Validate required fields
        if (!request_body.contains("deployment_id")) {
            json error = {
                {"error", {
                    {"message", "Missing required field: deployment_id"},
                    {"type", "invalid_request_error"},
                    {"code", "missing_field"}
                }}
            };
            res.status = 400;
            res.set_content(error.dump(), "application/json");
            return;
        }

        if (!request_body.contains("input")) {
            json error = {
                {"error", {
                    {"message", "Missing required field: input"},
                    {"type", "invalid_request_error"},
                    {"code", "missing_field"}
                }}
            };
            res.status = 400;
            res.set_content(error.dump(), "application/json");
            return;
        }

        std::string deployment_id = request_body["deployment_id"];

        // Check if deployment exists
        if (!deployment_manager_->HasDeployment(deployment_id)) {
            json error = {
                {"error", {
                    {"message", "Deployment not found: " + deployment_id},
                    {"type", "invalid_request_error"},
                    {"code", "deployment_not_found"}
                }}
            };
            res.status = 404;
            res.set_content(error.dump(), "application/json");
            return;
        }

        // Parse input tensor
        std::unordered_map<std::string, cyxwiz::Tensor> inputs;
        try {
            // Expect input as 2D array: [[val, val, ...]] for batch of 1
            const auto& input_json = request_body["input"];

            if (!input_json.is_array()) {
                throw std::runtime_error("input must be an array");
            }

            // Flatten input into vector
            std::vector<float> input_data;
            std::vector<size_t> shape;

            if (input_json.empty()) {
                throw std::runtime_error("input array cannot be empty");
            }

            // Handle 1D array: [val, val, ...]
            if (!input_json[0].is_array()) {
                for (const auto& val : input_json) {
                    input_data.push_back(val.get<float>());
                }
                shape = {1, input_data.size()};  // Batch of 1
            }
            // Handle 2D array: [[val, val, ...], ...]
            else {
                size_t batch_size = input_json.size();
                size_t feature_size = 0;

                for (const auto& row : input_json) {
                    if (!row.is_array()) {
                        throw std::runtime_error("Each batch element must be an array");
                    }
                    if (feature_size == 0) {
                        feature_size = row.size();
                    } else if (row.size() != feature_size) {
                        throw std::runtime_error("Inconsistent feature dimensions");
                    }

                    for (const auto& val : row) {
                        input_data.push_back(val.get<float>());
                    }
                }
                shape = {batch_size, feature_size};
            }

            // Create tensor
            inputs["input"] = cyxwiz::Tensor(shape, input_data.data());

        } catch (const std::exception& e) {
            json error = {
                {"error", {
                    {"message", std::string("Failed to parse input: ") + e.what()},
                    {"type", "invalid_request_error"},
                    {"code", "invalid_input"}
                }}
            };
            res.status = 400;
            res.set_content(error.dump(), "application/json");
            return;
        }

        // Run inference
        std::unordered_map<std::string, cyxwiz::Tensor> outputs;
        bool success = deployment_manager_->RunInference(deployment_id, inputs, outputs);

        auto end_time = std::chrono::high_resolution_clock::now();
        double latency_ms = std::chrono::duration<double, std::milli>(
            end_time - start_time
        ).count();

        if (!success) {
            json error = {
                {"error", {
                    {"message", "Inference failed"},
                    {"type", "server_error"},
                    {"code", "inference_error"}
                }},
                {"latency_ms", latency_ms}
            };
            res.status = 500;
            res.set_content(error.dump(), "application/json");
            return;
        }

        // Format response
        json response;
        response["deployment_id"] = deployment_id;
        response["latency_ms"] = latency_ms;

        // Convert output tensors to JSON
        for (const auto& [name, tensor] : outputs) {
            const auto& output_shape = tensor.Shape();
            size_t total_size = 1;
            for (size_t dim : output_shape) {
                total_size *= dim;
            }

            // Get output data - copy from raw pointer
            std::vector<float> output_data(total_size);
            const float* data_ptr = tensor.Data<float>();
            if (data_ptr) {
                std::copy(data_ptr, data_ptr + total_size, output_data.begin());
            }

            // Add to response
            response["output"] = output_data;
            response["shape"] = output_shape;

            // Only return first output for now
            break;
        }

        res.set_content(response.dump(), "application/json");
    });

    // Get deployment info
    server_->Get("/v1/deployments/:id", [this](const httplib::Request& req, httplib::Response& res) {
        if (!deployment_manager_) {
            json error = {
                {"error", {
                    {"message", "Deployment manager not available"},
                    {"type", "server_error"},
                    {"code", "service_unavailable"}
                }}
            };
            res.status = 503;
            res.set_content(error.dump(), "application/json");
            return;
        }

        std::string deployment_id = req.path_params.at("id");

        if (!deployment_manager_->HasDeployment(deployment_id)) {
            json error = {
                {"error", {
                    {"message", "Deployment not found: " + deployment_id},
                    {"type", "invalid_request_error"},
                    {"code", "deployment_not_found"}
                }}
            };
            res.status = 404;
            res.set_content(error.dump(), "application/json");
            return;
        }

        // Get deployment status and metrics
        auto status = deployment_manager_->GetDeploymentStatus(deployment_id);
        auto metrics = deployment_manager_->GetDeploymentMetrics(deployment_id);

        json response = {
            {"id", deployment_id},
            {"status", static_cast<int>(status)},
            {"input_specs", json::array()},
            {"output_specs", json::array()}
        };

        // Add input/output specs
        auto input_specs = deployment_manager_->GetInputSpecs(deployment_id);
        for (const auto& spec : input_specs) {
            response["input_specs"].push_back({
                {"name", spec.name},
                {"shape", spec.shape},
                {"dtype", spec.dtype}
            });
        }

        auto output_specs = deployment_manager_->GetOutputSpecs(deployment_id);
        for (const auto& spec : output_specs) {
            response["output_specs"].push_back({
                {"name", spec.name},
                {"shape", spec.shape},
                {"dtype", spec.dtype}
            });
        }

        // Add metrics
        if (!metrics.empty()) {
            response["metrics"] = {
                {"request_count", metrics[0].request_count},
                {"avg_latency_ms", metrics[0].avg_latency_ms}
            };
        }

        res.set_content(response.dump(), "application/json");
    });

    spdlog::info("Registered HTTP routes: /health, /v1/models, /v1/deployments, /v1/predict, /v1/deployments/:id");
}

void OpenAIAPIServer::HandleHealth() {
    // Implemented inline in RegisterRoutes
}

void OpenAIAPIServer::HandleModels() {
    // Implemented inline in RegisterRoutes
}

void OpenAIAPIServer::HandlePredict() {
    // Implemented inline in RegisterRoutes
}

} // namespace cyxwiz::servernode
