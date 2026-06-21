// local_inference_server.cpp - Embedded HTTP inference server implementation
#include "local_inference_server.h"
#include "text_inference_input.h"
#include "../core/language_model_training.h"
#include "../core/model_importer.h"
#include <cyxwiz/sequential.h>
#include <cyxwiz/tensor.h>
#include <cyxwiz/tokenizer.h>
#include <httplib.h>
#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>
#include <filesystem>
#include <chrono>
#include <cstddef>

namespace cyxwiz {

using json = nlohmann::json;
namespace fs = std::filesystem;

LocalInferenceServer::LocalInferenceServer()
    : server_(std::make_unique<httplib::Server>()) {
}

LocalInferenceServer::~LocalInferenceServer() {
    Stop();
    UnloadModel();
}

bool LocalInferenceServer::LoadModel(const std::string& model_path) {
    std::lock_guard<std::mutex> lock(model_mutex_);

    if (!fs::exists(model_path)) {
        last_error_ = "Model file not found: " + model_path;
        spdlog::error("{}", last_error_);
        return false;
    }

    try {
        ModelImporter importer;
        auto new_model = std::make_unique<SequentialModel>();

        ImportOptions options;
        // Default options will load weights during import

        auto result = importer.ImportCyxModel(model_path, *new_model, options);

        if (!result.success) {
            last_error_ = "Failed to import model: " + result.error_message;
            spdlog::error("{}", last_error_);
            return false;
        }

        // Set model to evaluation mode (disable dropout, etc.)
        new_model->SetTraining(false);

        auto new_tokenizer = std::unique_ptr<Tokenizer>();
        bool new_has_text_vocabulary = false;

        formats::CyxModelFormat cyxmodel_format;
        std::string tokenizer_config_json;
        std::string tokenizer_vocab_text;
        if (cyxmodel_format.ExtractTextTokenizerAssets(
                model_path, tokenizer_config_json, tokenizer_vocab_text)) {
            TextTokenizerPackage tokenizer_package;
            std::string tokenizer_error;
            if (!LoadTextTokenizerPackage(tokenizer_config_json,
                                          tokenizer_vocab_text,
                                          tokenizer_package,
                                          tokenizer_error)) {
                last_error_ = "Failed to load tokenizer assets: " +
                              tokenizer_error;
                spdlog::error("{}", last_error_);
                return false;
            }
            new_tokenizer = std::move(tokenizer_package.tokenizer);
            new_has_text_vocabulary = tokenizer_package.has_vocabulary;
        }

        model_ = std::move(new_model);
        text_tokenizer_ = std::move(new_tokenizer);
        has_text_vocabulary_ = new_has_text_vocabulary;
        model_path_ = model_path;

        spdlog::info("Loaded model: {} ({} layers)", GetModelName(), model_->Size());
        return true;

    } catch (const std::exception& e) {
        last_error_ = std::string("Exception loading model: ") + e.what();
        spdlog::error("{}", last_error_);
        return false;
    }
}

void LocalInferenceServer::UnloadModel() {
    std::lock_guard<std::mutex> lock(model_mutex_);
    model_.reset();
    text_tokenizer_.reset();
    has_text_vocabulary_ = false;
    model_path_.clear();
}

bool LocalInferenceServer::HasModel() const {
    std::lock_guard<std::mutex> lock(model_mutex_);
    return model_ != nullptr;
}

std::string LocalInferenceServer::GetModelName() const {
    if (model_path_.empty()) return "";
    return fs::path(model_path_).filename().string();
}

std::string LocalInferenceServer::GetEndpointUrl() const {
    if (!running_ || port_ == 0) return "";
    return "http://localhost:" + std::to_string(port_) + "/v1/predict";
}

bool LocalInferenceServer::Start(int port) {
    if (running_) {
        spdlog::warn("LocalInferenceServer already running on port {}", port_);
        return true;
    }

    if (!HasModel()) {
        last_error_ = "No model loaded";
        spdlog::error("{}", last_error_);
        return false;
    }

    port_ = port;
    RegisterRoutes();

    // Start server in background thread
    server_thread_ = std::make_unique<std::thread>([this]() {
        ServerThread();
    });

    // Wait a bit for server to start
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    if (running_) {
        spdlog::info("LocalInferenceServer started on port {}", port_);
    }

    return running_;
}

void LocalInferenceServer::Stop() {
    if (!running_) return;

    spdlog::info("Stopping LocalInferenceServer...");
    running_ = false;

    if (server_) {
        server_->stop();
    }

    if (server_thread_ && server_thread_->joinable()) {
        server_thread_->join();
    }
    server_thread_.reset();

    spdlog::info("LocalInferenceServer stopped");
}

void LocalInferenceServer::ServerThread() {
    spdlog::info("Starting embedded inference server on port {}", port_);
    running_ = true;

    if (!server_->listen("0.0.0.0", port_)) {
        spdlog::error("Failed to start embedded server on port {}", port_);
        running_ = false;
    }
}

void LocalInferenceServer::RegisterRoutes() {
    // CORS headers
    server_->set_default_headers({
        {"Access-Control-Allow-Origin", "*"},
        {"Access-Control-Allow-Methods", "GET, POST, OPTIONS"},
        {"Access-Control-Allow-Headers", "Content-Type"}
    });

    // Handle preflight
    server_->Options(".*", [](const httplib::Request&, httplib::Response& res) {
        res.status = 204;
    });

    // Health check
    server_->Get("/health", [this](const httplib::Request& req, httplib::Response& res) {
        HandleHealth(req, res);
    });

    // Model info
    server_->Get("/v1/model", [this](const httplib::Request& req, httplib::Response& res) {
        HandleModelInfo(req, res);
    });

    // Predict
    server_->Post("/v1/predict", [this](const httplib::Request& req, httplib::Response& res) {
        HandlePredict(req, res);
    });

    // Greedy text generation
    server_->Post("/v1/generate", [this](const httplib::Request& req, httplib::Response& res) {
        HandleGenerate(req, res);
    });

    spdlog::info("Registered routes: /health, /v1/model, /v1/predict, /v1/generate");
}

void LocalInferenceServer::HandleHealth(const httplib::Request&, httplib::Response& res) {
    json response = {
        {"status", "healthy"},
        {"server_type", "cyxwiz-engine-embedded"},
        {"model_loaded", HasModel()},
        {"request_count", request_count_.load()}
    };

    if (HasModel()) {
        response["model_name"] = GetModelName();
    }

    res.set_content(response.dump(), "application/json");
}

void LocalInferenceServer::HandleModelInfo(const httplib::Request&, httplib::Response& res) {
    std::lock_guard<std::mutex> lock(model_mutex_);

    if (!model_) {
        json error = {{"error", "No model loaded"}};
        res.status = 404;
        res.set_content(error.dump(), "application/json");
        return;
    }

    json response = {
        {"model_name", GetModelName()},
        {"model_path", model_path_},
        {"num_layers", model_->Size()},
        {"has_text_tokenizer", text_tokenizer_ != nullptr},
        {"has_text_vocabulary", has_text_vocabulary_},
        {"supports_greedy_generation", text_tokenizer_ != nullptr && has_text_vocabulary_},
        {"layers", json::array()}
    };

    // Add layer info
    for (size_t i = 0; i < model_->Size(); ++i) {
        const auto* module = model_->GetModule(i);
        response["layers"].push_back({
            {"index", i},
            {"name", module->GetName()},
            {"has_parameters", module->HasParameters()}
        });
    }

    res.set_content(response.dump(), "application/json");
}

void LocalInferenceServer::HandlePredict(const httplib::Request& req, httplib::Response& res) {
    auto start_time = std::chrono::high_resolution_clock::now();

    // Check model loaded
    if (!HasModel()) {
        json error = {
            {"error", {
                {"message", "No model loaded"},
                {"type", "server_error"},
                {"code", "no_model"}
            }}
        };
        res.status = 503;
        res.set_content(error.dump(), "application/json");
        return;
    }

    // Parse request
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

    // Validate input field
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

    // Parse input tensor
    Tensor input_tensor;
    try {
        const auto& input_json = request_body["input"];

        std::vector<float> input_data;
        std::vector<size_t> shape;

        if (input_json.is_string()) {
            std::lock_guard<std::mutex> lock(model_mutex_);
            if (!text_tokenizer_) {
                throw std::runtime_error(
                    "raw text input requires packaged tokenizer metadata");
            }
            if (!has_text_vocabulary_) {
                throw std::runtime_error(
                    "raw text input requires packaged vocabulary");
            }

            input_data = EncodeTextForInference(*text_tokenizer_,
                                                input_json.get<std::string>());
            shape = {1, input_data.size()};
        } else if (!input_json.is_array()) {
            throw std::runtime_error("input must be an array or string");
        } else if (input_json.empty()) {
            throw std::runtime_error("input array cannot be empty");
        } else if (!input_json[0].is_array()) {
            // Handle 1D array: [val, val, ...]
            for (const auto& val : input_json) {
                input_data.push_back(val.get<float>());
            }
            shape = {1, input_data.size()};  // Batch of 1
        } else {
            // Handle 2D array: [[val, val, ...], ...]
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

        input_tensor = Tensor(shape, input_data.data());

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
    Tensor output_tensor;
    try {
        std::lock_guard<std::mutex> lock(model_mutex_);
        if (!model_) {
            throw std::runtime_error("Model unloaded during request");
        }
        output_tensor = model_->Forward(input_tensor);
        request_count_++;

    } catch (const std::exception& e) {
        json error = {
            {"error", {
                {"message", std::string("Inference failed: ") + e.what()},
                {"type", "server_error"},
                {"code", "inference_error"}
            }}
        };
        res.status = 500;
        res.set_content(error.dump(), "application/json");
        return;
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    double latency_ms = std::chrono::duration<double, std::milli>(
        end_time - start_time).count();

    // Format response
    const auto& output_shape = output_tensor.Shape();
    size_t total_size = 1;
    for (size_t dim : output_shape) {
        total_size *= dim;
    }

    std::vector<float> output_data(total_size);
    const float* data_ptr = output_tensor.Data<float>();
    if (data_ptr) {
        std::copy(data_ptr, data_ptr + total_size, output_data.begin());
    }

    json response = {
        {"output", output_data},
        {"shape", output_shape},
        {"latency_ms", latency_ms}
    };

    res.set_content(response.dump(), "application/json");
}

void LocalInferenceServer::HandleGenerate(const httplib::Request& req, httplib::Response& res) {
    auto start_time = std::chrono::high_resolution_clock::now();

    if (!HasModel()) {
        json error = {
            {"error", {
                {"message", "No model loaded"},
                {"type", "server_error"},
                {"code", "no_model"}
            }}
        };
        res.status = 503;
        res.set_content(error.dump(), "application/json");
        return;
    }

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

    if (!request_body.contains("input") || !request_body["input"].is_string()) {
        json error = {
            {"error", {
                {"message", "Missing required string field: input"},
                {"type", "invalid_request_error"},
                {"code", "missing_field"}
            }}
        };
        res.status = 400;
        res.set_content(error.dump(), "application/json");
        return;
    }

    size_t max_new_tokens = 16;
    if (request_body.contains("max_new_tokens")) {
        if (!request_body["max_new_tokens"].is_number_unsigned()) {
            json error = {
                {"error", {
                    {"message", "max_new_tokens must be an unsigned integer"},
                    {"type", "invalid_request_error"},
                    {"code", "invalid_parameter"}
                }}
            };
            res.status = 400;
            res.set_content(error.dump(), "application/json");
            return;
        }
        max_new_tokens = request_body["max_new_tokens"].get<size_t>();
        if (max_new_tokens > 256) {
            max_new_tokens = 256;
        }
    }

    try {
        std::lock_guard<std::mutex> lock(model_mutex_);
        if (!model_) {
            throw std::runtime_error("Model unloaded during request");
        }
        if (!text_tokenizer_) {
            throw std::runtime_error(
                "text generation requires packaged tokenizer metadata");
        }
        if (!has_text_vocabulary_) {
            throw std::runtime_error(
                "text generation requires packaged vocabulary");
        }

        const std::vector<int64_t> prompt_ids =
            EncodeTextTokenIdsForGeneration(
                *text_tokenizer_,
                request_body["input"].get<std::string>());

        int64_t eos_token_id = text_tokenizer_->GetVocabulary().EosIndex();
        if (request_body.contains("eos_token_id")) {
            if (!request_body["eos_token_id"].is_number_integer()) {
                throw std::runtime_error("eos_token_id must be an integer");
            }
            eos_token_id = request_body["eos_token_id"].get<int64_t>();
        }

        const std::vector<int64_t> generated_ids =
            GenerateGreedyTokenIds(*model_,
                                   prompt_ids,
                                   max_new_tokens,
                                   eos_token_id);
        const std::vector<int64_t> new_token_ids(
            generated_ids.begin() + static_cast<std::ptrdiff_t>(prompt_ids.size()),
            generated_ids.end());

        request_count_++;

        auto end_time = std::chrono::high_resolution_clock::now();
        double latency_ms = std::chrono::duration<double, std::milli>(
            end_time - start_time).count();

        json response = {
            {"text", DecodeGeneratedTokenIds(*text_tokenizer_, generated_ids)},
            {"prompt_token_ids", prompt_ids},
            {"generated_token_ids", generated_ids},
            {"new_token_ids", new_token_ids},
            {"latency_ms", latency_ms}
        };

        res.set_content(response.dump(), "application/json");
    } catch (const std::exception& e) {
        json error = {
            {"error", {
                {"message", std::string("Generation failed: ") + e.what()},
                {"type", "server_error"},
                {"code", "generation_error"}
            }}
        };
        res.status = 500;
        res.set_content(error.dump(), "application/json");
        return;
    }
}

} // namespace cyxwiz
