// local_inference_server.cpp - Embedded HTTP inference server implementation
#include "local_inference_server.h"
#include "text_inference_input.h"
#include "../core/language_model_training.h"
#include "../core/model_importer.h"
#include "../core/formats/cyxmodel_format.h"
#include "../core/sequence_tag_metrics.h"
#include <cyxwiz/sequential.h>
#include <cyxwiz/tensor.h>
#include <cyxwiz/tokenizer.h>
#include <httplib.h>
#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>
#include <filesystem>
#include <chrono>
#include <cstddef>
#include <algorithm>
#include <limits>

namespace cyxwiz {

using json = nlohmann::json;
namespace fs = std::filesystem;

namespace {

bool ReadInt64Value(const json& value, int64_t& out) {
    if (value.is_number_integer()) {
        out = value.get<int64_t>();
        return true;
    }
    if (value.is_number_unsigned()) {
        out = static_cast<int64_t>(value.get<uint64_t>());
        return true;
    }
    if (value.is_number_float()) {
        out = static_cast<int64_t>(value.get<double>());
        return true;
    }
    if (value.is_string()) {
        try {
            out = std::stoll(value.get<std::string>());
            return true;
        } catch (...) {
            return false;
        }
    }
    return false;
}

bool ParseIntIdRows(const json& input,
                    const char* field_name,
                    std::vector<int64_t>& data,
                    std::vector<size_t>& shape) {
    if (!input.is_array()) {
        throw std::runtime_error(std::string("`") + field_name +
                                 "` must be an array");
    }

    data.clear();
    shape.clear();

    if (input.empty()) {
        shape = {1, 0};
        return true;
    }

    const bool nested = input[0].is_array();
    if (nested) {
        const size_t rows = input.size();
        size_t cols = 0;
        for (const auto& row_json : input) {
            if (!row_json.is_array()) {
                throw std::runtime_error(std::string("`") + field_name +
                                         "` must be 2D int array or 1D int array");
            }
            if (cols == 0) {
                cols = row_json.size();
            } else if (row_json.size() != cols) {
                throw std::runtime_error(std::string("`") + field_name +
                                         "` rows must have matching lengths");
            }
            for (const auto& value : row_json) {
                int64_t parsed = 0;
                if (!ReadInt64Value(value, parsed)) {
                    throw std::runtime_error(std::string("`") + field_name +
                                             "` contains non-integer values");
                }
                data.push_back(parsed);
            }
        }
        shape = {rows, cols};
    } else {
        shape = {1, input.size()};
        for (const auto& value : input) {
            int64_t parsed = 0;
            if (!ReadInt64Value(value, parsed)) {
                throw std::runtime_error(std::string("`") + field_name +
                                         "` contains non-integer values");
            }
            data.push_back(parsed);
        }
    }

    return true;
}

bool ParseIntIdVector(const json& input,
                      const char* field_name,
                      std::vector<int64_t>& data) {
    if (!input.is_array()) {
        throw std::runtime_error(std::string("`") + field_name +
                                 "` must be an integer array");
    }

    data.clear();
    for (const auto& value : input) {
        int64_t parsed = 0;
        if (!ReadInt64Value(value, parsed)) {
            throw std::runtime_error(std::string("`") + field_name +
                                     "` contains non-integer values");
        }
        data.push_back(parsed);
    }
    return true;
}

void AppendTensorValues(const Tensor& tensor, std::vector<float>& output) {
    const size_t total_size = tensor.NumElements();
    output.resize(total_size, 0.0f);

    if (tensor.GetDataType() == DataType::Float32) {
        const auto* data_ptr = tensor.Data<float>();
        if (data_ptr) {
            std::copy(data_ptr, data_ptr + total_size, output.begin());
        }
        return;
    }

    if (tensor.GetDataType() == DataType::Float64) {
        const auto* data_ptr = tensor.Data<double>();
        if (data_ptr) {
            for (size_t i = 0; i < total_size; ++i) {
                output[i] = static_cast<float>(data_ptr[i]);
            }
        }
        return;
    }

    if (tensor.GetDataType() == DataType::Int64) {
        const auto* data_ptr = tensor.Data<int64_t>();
        if (data_ptr) {
            for (size_t i = 0; i < total_size; ++i) {
                output[i] = static_cast<float>(data_ptr[i]);
            }
        }
        return;
    }

    if (tensor.GetDataType() == DataType::Int32) {
        const auto* data_ptr = tensor.Data<int32_t>();
        if (data_ptr) {
            for (size_t i = 0; i < total_size; ++i) {
                output[i] = static_cast<float>(data_ptr[i]);
            }
        }
        return;
    }

    throw std::runtime_error("Unsupported tensor type for inference response");
}

json Int64TensorToNestedRows(const Tensor& tensor, bool transpose_output) {
    const auto& shape = tensor.Shape();
    if (shape.size() != 2) {
        throw std::runtime_error("sequence tag tensor must be 2D");
    }

    const size_t rows = shape[0];
    const size_t cols = shape[1];
    json rows_json = json::array();

    for (size_t row = 0; row < rows; ++row) {
        json row_json = json::array();
        for (size_t col = 0; col < cols; ++col) {
            size_t index = transpose_output ? (col * rows + row) : (row * cols + col);
            if (tensor.GetDataType() == DataType::Int64) {
                row_json.push_back(tensor.Data<int64_t>()[index]);
            } else if (tensor.GetDataType() == DataType::Int32) {
                row_json.push_back(static_cast<int64_t>(tensor.Data<int32_t>()[index]));
            } else {
                throw std::runtime_error("sequence tag IDs must be Int64 or Int32");
            }
        }
        rows_json.push_back(std::move(row_json));
    }
    return rows_json;
}

json DecodeTagIdsToLabels(const Tensor& predicted_tag_ids,
                         const std::vector<std::string>& label_vocab,
                         bool transpose_output) {
    const auto& shape = predicted_tag_ids.Shape();
    if (shape.size() != 2) {
        throw std::runtime_error("predicted sequence tag ids must be 2D");
    }
    const size_t rows = shape[0];
    const size_t cols = shape[1];
    const auto* data_ptr64 = predicted_tag_ids.GetDataType() == DataType::Int64
                                 ? predicted_tag_ids.Data<int64_t>()
                                 : nullptr;
    const auto* data_ptr32 = predicted_tag_ids.GetDataType() == DataType::Int32
                                 ? predicted_tag_ids.Data<int32_t>()
                                 : nullptr;

    if (!data_ptr64 && !data_ptr32) {
        throw std::runtime_error("predicted sequence tag IDs must be Int64 or Int32");
    }

    json rows_json = json::array();
    for (size_t row = 0; row < rows; ++row) {
        json row_json = json::array();
        for (size_t col = 0; col < cols; ++col) {
            size_t index = transpose_output ? (col * rows + row) : (row * cols + col);
            const int64_t value = data_ptr64 ? data_ptr64[index]
                                             : static_cast<int64_t>(data_ptr32[index]);
            if (value >= 0 &&
                static_cast<size_t>(value) < label_vocab.size()) {
                row_json.push_back(label_vocab[static_cast<size_t>(value)]);
            } else {
                row_json.push_back(std::string{});
            }
        }
        rows_json.push_back(std::move(row_json));
    }
    return rows_json;
}

Tensor TransposeSequenceLogits(const Tensor& logits) {
    const auto& shape = logits.Shape();
    if (shape.size() != 3) {
        throw std::runtime_error("sequence logits must be 3D");
    }
    const size_t time_steps = shape[0];
    const size_t batch = shape[1];
    const size_t tags = shape[2];
    if (time_steps == 0 || batch == 0 || tags == 0) {
        throw std::runtime_error("sequence logits shape must be non-zero");
    }

    std::vector<float> transposed_data(logits.NumElements());
    const float* input_ptr = logits.Data<float>();
    if (!input_ptr) {
        throw std::runtime_error("sequence logits must be Float32 for argmax");
    }

    for (size_t t = 0; t < time_steps; ++t) {
        for (size_t b = 0; b < batch; ++b) {
            for (size_t c = 0; c < tags; ++c) {
                const size_t in_offset = t * batch * tags + b * tags + c;
                const size_t out_offset = b * time_steps * tags + t * tags + c;
                transposed_data[out_offset] = input_ptr[in_offset];
            }
        }
    }

    return Tensor({batch, time_steps, tags}, transposed_data.data(), DataType::Float32);
}

}  // namespace

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
        bool new_has_sequence_model = false;
        bool new_has_sequence_token_vocabulary = false;
        bool new_has_sequence_pos_vocabulary = false;
        bool new_has_sequence_tag_vocabulary = false;
        bool new_sequence_batch_first = true;
        bool new_sequence_create_attention_mask = true;
        bool new_sequence_create_causal_lm_targets = false;
        size_t new_sequence_max_sequence_length = 0;
        int64_t new_sequence_word_pad_id = 0;
        int64_t new_sequence_pos_pad_id = 0;
        int64_t new_sequence_tag_ignore_index = -100;
        int64_t new_sequence_target_ignore_index = -100;
        std::vector<std::string> new_sequence_token_vocabulary;
        std::vector<std::string> new_sequence_pos_vocabulary;
        std::vector<std::string> new_sequence_tag_vocabulary;

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

        const auto probe = cyxmodel_format.Probe(model_path);
        new_has_sequence_model = probe.has_sequence;
        new_sequence_batch_first = probe.sequence_batch_first;
        new_sequence_create_attention_mask = probe.sequence_create_attention_mask;
        new_sequence_create_causal_lm_targets = probe.sequence_create_causal_lm_targets;
        new_sequence_max_sequence_length = probe.sequence_max_sequence_length;
        new_sequence_word_pad_id = probe.sequence_word_pad_id;
        new_sequence_pos_pad_id = probe.sequence_pos_pad_id;
        new_sequence_tag_ignore_index = probe.sequence_tag_ignore_index;
        new_sequence_target_ignore_index = probe.sequence_target_ignore_index;

        if (probe.has_sequence_token_vocabulary ||
            probe.has_sequence_pos_vocabulary ||
            probe.has_sequence_tag_vocabulary) {
            std::string token_vocab_text;
            std::string pos_vocab_text;
            std::string tag_vocab_text;
            if (cyxmodel_format.ExtractSequenceVocabularyAssets(
                    model_path,
                    token_vocab_text,
                    pos_vocab_text,
                    tag_vocab_text)) {
                if (!token_vocab_text.empty()) {
                    new_sequence_token_vocabulary = ParseVocabularyWords(token_vocab_text);
                    new_has_sequence_token_vocabulary =
                        !new_sequence_token_vocabulary.empty();
                }
                if (!pos_vocab_text.empty()) {
                    new_sequence_pos_vocabulary = ParseVocabularyWords(pos_vocab_text);
                    new_has_sequence_pos_vocabulary =
                        !new_sequence_pos_vocabulary.empty();
                }
                if (!tag_vocab_text.empty()) {
                    new_sequence_tag_vocabulary = ParseVocabularyWords(tag_vocab_text);
                    new_has_sequence_tag_vocabulary =
                        !new_sequence_tag_vocabulary.empty();
                }
            }
        }

        model_ = std::move(new_model);
        text_tokenizer_ = std::move(new_tokenizer);
        has_text_vocabulary_ = new_has_text_vocabulary;
        has_sequence_model_ = new_has_sequence_model;
        has_sequence_token_vocabulary_ = new_has_sequence_token_vocabulary;
        has_sequence_pos_vocabulary_ = new_has_sequence_pos_vocabulary;
        has_sequence_tag_vocabulary_ = new_has_sequence_tag_vocabulary;
        sequence_batch_first_ = new_sequence_batch_first;
        sequence_create_attention_mask_ = new_sequence_create_attention_mask;
        sequence_create_causal_lm_targets_ = new_sequence_create_causal_lm_targets;
        sequence_max_sequence_length_ = new_sequence_max_sequence_length;
        sequence_word_pad_id_ = new_sequence_word_pad_id;
        sequence_pos_pad_id_ = new_sequence_pos_pad_id;
        sequence_tag_ignore_index_ = new_sequence_tag_ignore_index;
        sequence_target_ignore_index_ = new_sequence_target_ignore_index;
        sequence_token_vocabulary_ = std::move(new_sequence_token_vocabulary);
        sequence_pos_vocabulary_ = std::move(new_sequence_pos_vocabulary);
        sequence_tag_vocabulary_ = std::move(new_sequence_tag_vocabulary);
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
    has_sequence_model_ = false;
    has_sequence_token_vocabulary_ = false;
    has_sequence_pos_vocabulary_ = false;
    has_sequence_tag_vocabulary_ = false;
    sequence_batch_first_ = true;
    sequence_create_attention_mask_ = true;
    sequence_create_causal_lm_targets_ = false;
    sequence_max_sequence_length_ = 0;
    sequence_word_pad_id_ = 0;
    sequence_pos_pad_id_ = 0;
    sequence_tag_ignore_index_ = -100;
    sequence_target_ignore_index_ = -100;
    sequence_token_vocabulary_.clear();
    sequence_pos_vocabulary_.clear();
    sequence_tag_vocabulary_.clear();
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
        {"has_sequence_model", has_sequence_model_},
        {"has_sequence_token_vocabulary", has_sequence_token_vocabulary_},
        {"has_sequence_pos_vocabulary", has_sequence_pos_vocabulary_},
        {"has_sequence_tag_vocabulary", has_sequence_tag_vocabulary_},
        {"sequence_batch_first", sequence_batch_first_},
        {"sequence_create_attention_mask", sequence_create_attention_mask_},
        {"sequence_create_causal_lm_targets", sequence_create_causal_lm_targets_},
        {"sequence_max_sequence_length", sequence_max_sequence_length_},
        {"sequence_word_pad_id", sequence_word_pad_id_},
        {"sequence_pos_pad_id", sequence_pos_pad_id_},
        {"sequence_tag_ignore_index", sequence_tag_ignore_index_},
        {"sequence_target_ignore_index", sequence_target_ignore_index_},
        {"supports_sequence_decoding", has_sequence_tag_vocabulary_},
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
    bool is_sequence_input = false;
    std::vector<int64_t> sequence_lengths;
    try {
        const auto& input_json = request_body["input"];

        std::vector<float> input_data;
        std::vector<size_t> shape;
        std::vector<size_t> word_shape;
        std::vector<int64_t> word_data;
        std::vector<int64_t> optional_data;

        if (input_json.is_object() && input_json.contains("word_ids")) {
            is_sequence_input = true;
            ParseIntIdRows(input_json["word_ids"], "input.word_ids",
                           word_data, word_shape);
            if (word_shape.size() != 2) {
                throw std::runtime_error("input.word_ids must be 2D [batch, seq]");
            }
            if (word_shape[0] == 0 || word_shape[1] == 0) {
                throw std::runtime_error("input.word_ids must not be empty");
            }

            if (input_json.contains("pos_ids")) {
                ParseIntIdRows(input_json["pos_ids"], "input.pos_ids",
                               optional_data, shape);
                if (shape != word_shape) {
                    throw std::runtime_error(
                        "input.pos_ids shape must match input.word_ids");
                }
            }
            if (input_json.contains("attention_mask")) {
                ParseIntIdRows(input_json["attention_mask"],
                               "input.attention_mask",
                               optional_data, shape);
                if (shape != word_shape) {
                    throw std::runtime_error(
                        "input.attention_mask shape must match input.word_ids");
                }
            }
            if (input_json.contains("sequence_lengths")) {
                ParseIntIdVector(input_json["sequence_lengths"],
                                 "input.sequence_lengths",
                                 sequence_lengths);
                const size_t expected =
                    sequence_batch_first_ ? word_shape[0] : word_shape[1];
                if (sequence_lengths.size() != expected) {
                    throw std::runtime_error(
                        "input.sequence_lengths length does not match batch dimension");
                }
            }
            input_tensor = Tensor(word_shape, word_data.data(), DataType::Int64);
        } else if (input_json.is_string()) {
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

        if (!is_sequence_input) {
            input_tensor = Tensor(shape, input_data.data());
        }

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
    std::vector<float> output_data;
    try {
        AppendTensorValues(output_tensor, output_data);
    } catch (const std::exception& e) {
        json error = {
            {"error", {
                {"message", std::string("Failed to format output: ") + e.what()},
                {"type", "server_error"},
                {"code", "output_format_error"}
            }}
        };
        res.status = 500;
        res.set_content(error.dump(), "application/json");
        return;
    }

    json response = {
        {"output", output_data},
        {"shape", output_shape},
        {"latency_ms", latency_ms}
    };

    if (!sequence_lengths.empty()) {
        response["sequence_lengths"] = sequence_lengths;
    }

    if (is_sequence_input && has_sequence_tag_vocabulary_ &&
        !sequence_tag_vocabulary_.empty() &&
        output_tensor.GetDataType() == DataType::Float32 &&
        output_shape.size() == 3) {
        try {
            Tensor logits_tensor = output_tensor;
            if (!sequence_batch_first_) {
                logits_tensor = TransposeSequenceLogits(output_tensor);
            }
            const Tensor predicted_ids = ArgmaxSequenceTagLogits(logits_tensor);
            response["sequence"] = {
                {"tag_ids", Int64TensorToNestedRows(predicted_ids, false)},
                {"tag_labels", DecodeTagIdsToLabels(
                                   predicted_ids, sequence_tag_vocabulary_, false)},
                {"tag_vocab", sequence_tag_vocabulary_},
                {"batch_first", sequence_batch_first_},
                {"ignore_index", sequence_tag_ignore_index_}
            };
        } catch (const std::exception& e) {
            json error = {
                {"error", {
                    {"message", std::string("Failed to decode sequence tags: ") +
                                    e.what()},
                    {"type", "server_error"},
                    {"code", "sequence_decode_error"}
                }}
            };
            res.status = 500;
            res.set_content(error.dump(), "application/json");
            return;
        }
    }

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
