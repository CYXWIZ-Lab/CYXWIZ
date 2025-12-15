// openai_api_server.cpp - HTTP REST API server implementation
#include "openai_api_server.h"
#include "../deployment_manager.h"
#include <httplib.h>
#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>
#include <chrono>
#include <functional>
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

        // Get the model's input specs to determine the correct input name
        auto input_specs = deployment_manager_->GetInputSpecs(deployment_id);
        std::string input_name = "input";  // Default fallback
        if (!input_specs.empty()) {
            input_name = input_specs[0].name;  // Use the model's actual input name
            spdlog::debug("Using model input name: {}", input_name);
        }

        // Parse input tensor
        std::unordered_map<std::string, cyxwiz::Tensor> inputs;
        try {
            const auto& input_json = request_body["input"];

            if (!input_json.is_array()) {
                throw std::runtime_error("input must be an array");
            }

            if (input_json.empty()) {
                throw std::runtime_error("input array cannot be empty");
            }

            // Recursively determine shape and flatten data
            std::vector<float> input_data;
            std::vector<size_t> shape;

            // Helper lambda to recursively process JSON array
            std::function<void(const json&, size_t)> flatten_array;
            flatten_array = [&](const json& arr, size_t depth) {
                if (depth >= shape.size()) {
                    shape.push_back(arr.size());
                } else if (shape[depth] != arr.size()) {
                    throw std::runtime_error("Inconsistent array dimensions at depth " + std::to_string(depth));
                }

                for (const auto& elem : arr) {
                    if (elem.is_array()) {
                        flatten_array(elem, depth + 1);
                    } else if (elem.is_number()) {
                        input_data.push_back(elem.get<float>());
                    } else {
                        throw std::runtime_error("Array elements must be numbers or arrays");
                    }
                }
            };

            flatten_array(input_json, 0);

            // Create tensor with detected shape using the model's actual input name
            inputs[input_name] = cyxwiz::Tensor(shape, input_data.data());

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

    // POST /v1/completions - Text completions for LLM models (GGUF)
    server_->Post("/v1/completions", [this](const httplib::Request& req, httplib::Response& res) {
        auto start_time = std::chrono::high_resolution_clock::now();

        json request_body;
        try {
            request_body = json::parse(req.body);
        } catch (const json::exception& e) {
            json error = {
                {"error", {
                    {"message", std::string("Invalid JSON: ") + e.what()},
                    {"type", "invalid_request_error"},
                    {"code", "invalid_json"}
                }}
            };
            res.status = 400;
            res.set_content(error.dump(), "application/json");
            return;
        }

        // Get deployment_id (required)
        if (!request_body.contains("deployment_id") && !request_body.contains("model")) {
            json error = {
                {"error", {
                    {"message", "Missing required field: deployment_id or model"},
                    {"type", "invalid_request_error"},
                    {"code", "missing_field"}
                }}
            };
            res.status = 400;
            res.set_content(error.dump(), "application/json");
            return;
        }

        std::string deployment_id = request_body.contains("deployment_id")
            ? request_body["deployment_id"].get<std::string>()
            : request_body["model"].get<std::string>();

        // Get prompt (required)
        if (!request_body.contains("prompt")) {
            json error = {
                {"error", {
                    {"message", "Missing required field: prompt"},
                    {"type", "invalid_request_error"},
                    {"code", "missing_field"}
                }}
            };
            res.status = 400;
            res.set_content(error.dump(), "application/json");
            return;
        }

        std::string prompt = request_body["prompt"].get<std::string>();

        // Get optional parameters
        int max_tokens = request_body.value("max_tokens", 256);
        float temperature = request_body.value("temperature", 0.8f);

        // Check deployment exists
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

        // Create text input tensor
        std::unordered_map<std::string, cyxwiz::Tensor> inputs;
        std::vector<size_t> shape = {prompt.length()};
        cyxwiz::Tensor prompt_tensor(shape, cyxwiz::DataType::UInt8);
        std::memcpy(prompt_tensor.Data(), prompt.data(), prompt.length());
        inputs["prompt"] = std::move(prompt_tensor);

        // Run inference
        std::unordered_map<std::string, cyxwiz::Tensor> outputs;
        bool success = deployment_manager_->RunInference(deployment_id, inputs, outputs);

        auto end_time = std::chrono::high_resolution_clock::now();
        double latency_ms = std::chrono::duration<double, std::milli>(
            end_time - start_time).count();

        if (!success) {
            json error = {
                {"error", {
                    {"message", "Inference failed"},
                    {"type", "server_error"},
                    {"code", "inference_error"}
                }}
            };
            res.status = 500;
            res.set_content(error.dump(), "application/json");
            return;
        }

        // Extract completion text from output
        std::string completion_text;
        if (outputs.count("completion")) {
            const auto& tensor = outputs["completion"];
            completion_text = std::string(
                static_cast<const char*>(tensor.Data()),
                tensor.NumElements()
            );
        } else if (outputs.count("output")) {
            const auto& tensor = outputs["output"];
            completion_text = std::string(
                static_cast<const char*>(tensor.Data()),
                tensor.NumElements()
            );
        }

        // Build OpenAI-compatible response
        json response = {
            {"id", "cmpl-" + deployment_id.substr(4, 8)},
            {"object", "text_completion"},
            {"created", std::time(nullptr)},
            {"model", deployment_id},
            {"choices", json::array({
                {
                    {"text", completion_text},
                    {"index", 0},
                    {"logprobs", nullptr},
                    {"finish_reason", "stop"}
                }
            })},
            {"usage", {
                {"prompt_tokens", -1},  // TODO: get actual token counts
                {"completion_tokens", -1},
                {"total_tokens", -1}
            }}
        };

        res.set_content(response.dump(), "application/json");
        spdlog::debug("Completions request completed in {:.2f}ms", latency_ms);
    });

    // POST /v1/chat/completions - Chat completions for LLMs (OpenAI-compatible)
    server_->Post("/v1/chat/completions", [this](const httplib::Request& req, httplib::Response& res) {
        auto start_time = std::chrono::high_resolution_clock::now();

        if (!deployment_manager_) {
            res.status = 503;
            res.set_content(R"({"error":{"message":"Deployment manager not available"}})", "application/json");
            return;
        }

        json request_body;
        try {
            request_body = json::parse(req.body);
        } catch (const json::exception& e) {
            res.status = 400;
            res.set_content(R"({"error":{"message":"Invalid JSON"}})", "application/json");
            return;
        }

        // Get deployment_id or model
        std::string deployment_id = request_body.value("deployment_id",
            request_body.value("model", ""));
        if (deployment_id.empty()) {
            res.status = 400;
            res.set_content(R"({"error":{"message":"Missing deployment_id or model"}})", "application/json");
            return;
        }

        // Check deployment exists
        if (!deployment_manager_->HasDeployment(deployment_id)) {
            res.status = 404;
            res.set_content(R"({"error":{"message":"Deployment not found"}})", "application/json");
            return;
        }

        // Get messages array
        if (!request_body.contains("messages") || !request_body["messages"].is_array()) {
            res.status = 400;
            res.set_content(R"({"error":{"message":"Missing messages array"}})", "application/json");
            return;
        }

        // Build prompt from messages (ChatML format)
        std::string prompt;
        for (const auto& msg : request_body["messages"]) {
            std::string role = msg.value("role", "user");
            std::string content = msg.value("content", "");

            if (role == "system") {
                prompt += "<|im_start|>system\n" + content + "<|im_end|>\n";
            } else if (role == "user") {
                prompt += "<|im_start|>user\n" + content + "<|im_end|>\n";
            } else if (role == "assistant") {
                prompt += "<|im_start|>assistant\n" + content + "<|im_end|>\n";
            }
        }
        prompt += "<|im_start|>assistant\n";  // Start assistant response

        // Create input tensor
        std::unordered_map<std::string, cyxwiz::Tensor> inputs;
        std::vector<size_t> shape = {prompt.length()};
        cyxwiz::Tensor prompt_tensor(shape, cyxwiz::DataType::UInt8);
        std::memcpy(prompt_tensor.Data(), prompt.data(), prompt.length());
        inputs["prompt"] = std::move(prompt_tensor);

        // Run inference
        std::unordered_map<std::string, cyxwiz::Tensor> outputs;
        bool success = deployment_manager_->RunInference(deployment_id, inputs, outputs);

        auto end_time = std::chrono::high_resolution_clock::now();
        double latency_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();

        if (!success) {
            res.status = 500;
            res.set_content(R"({"error":{"message":"Inference failed"}})", "application/json");
            return;
        }

        // Extract completion text
        std::string completion_text;
        if (outputs.count("completion")) {
            const auto& tensor = outputs["completion"];
            completion_text = std::string(static_cast<const char*>(tensor.Data()), tensor.NumElements());
        } else if (outputs.count("output")) {
            const auto& tensor = outputs["output"];
            completion_text = std::string(static_cast<const char*>(tensor.Data()), tensor.NumElements());
        }

        // Build OpenAI-compatible response
        json response = {
            {"id", "chatcmpl-" + deployment_id.substr(4, 8)},
            {"object", "chat.completion"},
            {"created", std::time(nullptr)},
            {"model", deployment_id},
            {"choices", json::array({
                {
                    {"index", 0},
                    {"message", {
                        {"role", "assistant"},
                        {"content", completion_text}
                    }},
                    {"finish_reason", "stop"}
                }
            })},
            {"usage", {
                {"prompt_tokens", -1},
                {"completion_tokens", -1},
                {"total_tokens", -1}
            }}
        };

        res.set_content(response.dump(), "application/json");
        spdlog::debug("Chat completions request completed in {:.2f}ms", latency_ms);
    });

    // POST /v1/embeddings - Generate embeddings (OpenAI-compatible)
    server_->Post("/v1/embeddings", [this](const httplib::Request& req, httplib::Response& res) {
        if (!deployment_manager_) {
            res.status = 503;
            res.set_content(R"({"error":{"message":"Deployment manager not available"}})", "application/json");
            return;
        }

        json request_body;
        try {
            request_body = json::parse(req.body);
        } catch (const json::exception& e) {
            res.status = 400;
            res.set_content(R"({"error":{"message":"Invalid JSON"}})", "application/json");
            return;
        }

        std::string deployment_id = request_body.value("deployment_id",
            request_body.value("model", ""));
        if (deployment_id.empty()) {
            res.status = 400;
            res.set_content(R"({"error":{"message":"Missing deployment_id or model"}})", "application/json");
            return;
        }

        if (!deployment_manager_->HasDeployment(deployment_id)) {
            res.status = 404;
            res.set_content(R"({"error":{"message":"Deployment not found"}})", "application/json");
            return;
        }

        // Get input text(s)
        if (!request_body.contains("input")) {
            res.status = 400;
            res.set_content(R"({"error":{"message":"Missing input field"}})", "application/json");
            return;
        }

        std::vector<std::string> texts;
        if (request_body["input"].is_string()) {
            texts.push_back(request_body["input"].get<std::string>());
        } else if (request_body["input"].is_array()) {
            for (const auto& item : request_body["input"]) {
                if (item.is_string()) {
                    texts.push_back(item.get<std::string>());
                }
            }
        }

        if (texts.empty()) {
            res.status = 400;
            res.set_content(R"({"error":{"message":"No valid input texts provided"}})", "application/json");
            return;
        }

        json embeddings_array = json::array();

        for (size_t i = 0; i < texts.size(); ++i) {
            const auto& text = texts[i];

            // Create input with "text" key (triggers embedding mode in GGUFLoader)
            std::unordered_map<std::string, cyxwiz::Tensor> inputs;
            std::vector<size_t> shape = {text.length()};
            cyxwiz::Tensor text_tensor(shape, cyxwiz::DataType::UInt8);
            std::memcpy(text_tensor.Data(), text.data(), text.length());
            inputs["text"] = std::move(text_tensor);

            std::unordered_map<std::string, cyxwiz::Tensor> outputs;
            if (!deployment_manager_->RunInference(deployment_id, inputs, outputs)) {
                res.status = 500;
                res.set_content(R"({"error":{"message":"Embedding inference failed"}})", "application/json");
                return;
            }

            // Extract embedding vector
            std::vector<float> embedding;
            if (outputs.count("embedding")) {
                const auto& tensor = outputs["embedding"];
                const float* data = tensor.Data<float>();
                size_t count = tensor.NumElements();
                embedding.assign(data, data + count);
            } else if (outputs.count("output")) {
                // Fallback to generic output
                const auto& tensor = outputs["output"];
                const float* data = tensor.Data<float>();
                size_t count = tensor.NumElements();
                embedding.assign(data, data + count);
            }

            embeddings_array.push_back({
                {"object", "embedding"},
                {"index", i},
                {"embedding", embedding}
            });
        }

        json response = {
            {"object", "list"},
            {"data", embeddings_array},
            {"model", deployment_id},
            {"usage", {
                {"prompt_tokens", -1},
                {"total_tokens", -1}
            }}
        };

        res.set_content(response.dump(), "application/json");
        spdlog::debug("Embeddings request completed for {} texts", texts.size());
    });

    spdlog::info("Registered HTTP routes: /health, /v1/models, /v1/deployments, /v1/predict, /v1/completions, /v1/chat/completions, /v1/embeddings, /v1/deployments/:id");
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
