// deployment_client.cpp - gRPC client for Server Node deployment
#include "deployment_client.h"
#include <spdlog/spdlog.h>
#include <filesystem>
#include <thread>
#include <chrono>

namespace network {

namespace fs = std::filesystem;

DeploymentClient::DeploymentClient() = default;

void DeploymentClient::AddAuthMetadata(grpc::ClientContext& context) {
    if (!auth_token_.empty()) {
        // Add Bearer token to authorization header
        context.AddMetadata("authorization", "Bearer " + auth_token_);
        spdlog::debug("Added auth token to deployment request");
    }
}

DeploymentClient::~DeploymentClient() {
    Disconnect();
}

bool DeploymentClient::Connect(const std::string& server_address) {
    spdlog::info("Connecting to Server Node: {}", server_address);

    try {
        // Create channel
        channel_ = grpc::CreateChannel(server_address, grpc::InsecureChannelCredentials());

        // Create stub
        stub_ = cyxwiz::protocol::DeploymentService::NewStub(channel_);

        // Actually wait for connection with timeout (3 seconds)
        // gRPC channels are lazy - they don't connect until an RPC is made
        // GetState(true) triggers a connection attempt
        channel_->GetState(true);

        // Wait for the channel to be connected or fail
        auto deadline = std::chrono::system_clock::now() + std::chrono::seconds(3);
        bool connected = channel_->WaitForConnected(deadline);

        if (!connected) {
            auto state = channel_->GetState(false);
            if (state == GRPC_CHANNEL_TRANSIENT_FAILURE) {
                last_error_ = "Connection refused - Server Node not running";
            } else if (state == GRPC_CHANNEL_CONNECTING) {
                last_error_ = "Connection timed out - Server Node unreachable";
            } else {
                last_error_ = "Failed to connect: Channel state " + std::to_string(state);
            }
            spdlog::error("{}", last_error_);
            stub_.reset();
            channel_.reset();
            return false;
        }

        server_address_ = server_address;
        connected_ = true;
        spdlog::info("Connected to Server Node: {}", server_address);
        return true;

    } catch (const std::exception& e) {
        last_error_ = std::string("Connection error: ") + e.what();
        spdlog::error("{}", last_error_);
        connected_ = false;
        stub_.reset();
        channel_.reset();
        return false;
    }
}

void DeploymentClient::Disconnect() {
    if (connected_) {
        spdlog::info("Disconnecting from Server Node");
        stub_.reset();
        channel_.reset();
        connected_ = false;
        server_address_.clear();
    }
}

DeploymentResult DeploymentClient::Deploy(const DeploymentConfig& config) {
    DeploymentResult result;

    if (!connected_ || !stub_) {
        result.error_message = "Not connected to Server Node";
        spdlog::error("{}", result.error_message);
        return result;
    }

    // Validate model path exists
    if (!fs::exists(config.model_path)) {
        result.error_message = "Model file not found: " + config.model_path;
        spdlog::error("{}", result.error_message);
        return result;
    }

    spdlog::info("Deploying model: {} on port {}", config.model_path, config.port);

    try {
        // Build request
        cyxwiz::protocol::CreateDeploymentRequest request;
        auto* deploy_config = request.mutable_config();

        // Set deployment type to local server node
        deploy_config->set_type(cyxwiz::protocol::DEPLOYMENT_TYPE_LOCAL_NODE);

        // Set model info
        auto* model_info = deploy_config->mutable_model();
        model_info->set_local_path(config.model_path);

        // Determine model name
        std::string name = config.model_name;
        if (name.empty()) {
            name = fs::path(config.model_path).stem().string();
        }
        model_info->set_name(name);

        // Detect format from extension
        std::string ext = fs::path(config.model_path).extension().string();
        if (ext == ".cyxmodel") {
            model_info->set_format(cyxwiz::protocol::MODEL_FORMAT_CYXMODEL);
        } else if (ext == ".onnx") {
            model_info->set_format(cyxwiz::protocol::MODEL_FORMAT_ONNX);
        } else if (ext == ".gguf") {
            model_info->set_format(cyxwiz::protocol::MODEL_FORMAT_GGUF);
        } else if (ext == ".safetensors") {
            model_info->set_format(cyxwiz::protocol::MODEL_FORMAT_SAFETENSORS);
        } else {
            model_info->set_format(cyxwiz::protocol::MODEL_FORMAT_UNKNOWN);
        }

        model_info->set_source(cyxwiz::protocol::MODEL_SOURCE_LOCAL);
        model_info->set_size_bytes(static_cast<int64_t>(fs::file_size(config.model_path)));

        // Set runtime config
        deploy_config->set_port(config.port);
        deploy_config->set_enable_terminal(config.enable_terminal);

        // Add runtime params
        (*deploy_config->mutable_runtime_params())["gpu_layers"] = std::to_string(config.gpu_layers);
        (*deploy_config->mutable_runtime_params())["context_size"] = std::to_string(config.context_size);

        // Make RPC call
        grpc::ClientContext context;
        context.set_deadline(std::chrono::system_clock::now() + std::chrono::seconds(60));
        AddAuthMetadata(context);

        cyxwiz::protocol::CreateDeploymentResponse response;
        grpc::Status status = stub_->CreateDeployment(&context, request, &response);

        if (!status.ok()) {
            result.error_message = "gRPC error: " + status.error_message();
            spdlog::error("{}", result.error_message);
            return result;
        }

        if (response.status() != cyxwiz::protocol::STATUS_SUCCESS) {
            result.error_message = response.has_error() ?
                response.error().message() : "Deployment failed";
            spdlog::error("{}", result.error_message);
            return result;
        }

        // Success
        const auto& deployment = response.deployment();
        result.success = true;
        result.deployment_id = deployment.deployment_id();
        result.endpoint_url = deployment.endpoint_url();

        if (result.endpoint_url.empty()) {
            // Construct endpoint URL if not provided
            result.endpoint_url = "http://localhost:" + std::to_string(config.port) + "/v1/predict";
        }

        spdlog::info("Deployment created: id={}, endpoint={}", result.deployment_id, result.endpoint_url);
        return result;

    } catch (const std::exception& e) {
        result.error_message = std::string("Exception: ") + e.what();
        spdlog::error("{}", result.error_message);
        return result;
    }
}

void DeploymentClient::DeployAsync(const DeploymentConfig& config,
                                    std::function<void(const DeploymentResult&)> callback) {
    std::thread([this, config, callback]() {
        auto result = Deploy(config);
        if (callback) {
            callback(result);
        }
    }).detach();
}

bool DeploymentClient::ListDeployments(std::vector<DeploymentSummary>& deployments) {
    deployments.clear();

    if (!connected_ || !stub_) {
        last_error_ = "Not connected to Server Node";
        spdlog::error("{}", last_error_);
        return false;
    }

    try {
        cyxwiz::protocol::ListDeploymentsRequest request;
        request.set_page_size(100);

        grpc::ClientContext context;
        context.set_deadline(std::chrono::system_clock::now() + std::chrono::seconds(10));
        AddAuthMetadata(context);

        cyxwiz::protocol::ListDeploymentsResponse response;
        grpc::Status status = stub_->ListDeployments(&context, request, &response);

        if (!status.ok()) {
            last_error_ = "gRPC error: " + status.error_message();
            spdlog::error("{}", last_error_);
            return false;
        }

        for (const auto& deployment : response.deployments()) {
            DeploymentSummary summary;
            summary.id = deployment.deployment_id();
            summary.model_name = deployment.config().model().name();
            summary.model_path = deployment.config().model().local_path();
            summary.status = static_cast<int>(deployment.status());
            summary.port = deployment.config().port();
            summary.request_count = deployment.total_requests();
            summary.avg_latency_ms = deployment.avg_latency_ms();

            // Get gpu_layers from runtime params
            auto it = deployment.config().runtime_params().find("gpu_layers");
            if (it != deployment.config().runtime_params().end()) {
                try {
                    summary.gpu_layers = std::stoi(it->second);
                } catch (...) {
                    summary.gpu_layers = 0;
                }
            }

            deployments.push_back(summary);
        }

        spdlog::debug("Listed {} deployments", deployments.size());
        return true;

    } catch (const std::exception& e) {
        last_error_ = std::string("Exception: ") + e.what();
        spdlog::error("{}", last_error_);
        return false;
    }
}

bool DeploymentClient::GetDeployment(const std::string& deployment_id, DeploymentSummary& summary) {
    if (!connected_ || !stub_) {
        last_error_ = "Not connected to Server Node";
        return false;
    }

    try {
        cyxwiz::protocol::GetDeploymentRequest request;
        request.set_deployment_id(deployment_id);

        grpc::ClientContext context;
        context.set_deadline(std::chrono::system_clock::now() + std::chrono::seconds(10));
        AddAuthMetadata(context);

        cyxwiz::protocol::GetDeploymentResponse response;
        grpc::Status status = stub_->GetDeployment(&context, request, &response);

        if (!status.ok()) {
            last_error_ = "gRPC error: " + status.error_message();
            return false;
        }

        const auto& deployment = response.deployment();
        summary.id = deployment.deployment_id();
        summary.model_name = deployment.config().model().name();
        summary.model_path = deployment.config().model().local_path();
        summary.status = static_cast<int>(deployment.status());
        summary.port = deployment.config().port();
        summary.request_count = deployment.total_requests();
        summary.avg_latency_ms = deployment.avg_latency_ms();

        return true;

    } catch (const std::exception& e) {
        last_error_ = std::string("Exception: ") + e.what();
        return false;
    }
}

bool DeploymentClient::StopDeployment(const std::string& deployment_id) {
    if (!connected_ || !stub_) {
        last_error_ = "Not connected to Server Node";
        spdlog::error("{}", last_error_);
        return false;
    }

    spdlog::info("Stopping deployment: {}", deployment_id);

    try {
        cyxwiz::protocol::StopDeploymentRequest request;
        request.set_deployment_id(deployment_id);

        grpc::ClientContext context;
        context.set_deadline(std::chrono::system_clock::now() + std::chrono::seconds(30));
        AddAuthMetadata(context);

        cyxwiz::protocol::StopDeploymentResponse response;
        grpc::Status status = stub_->StopDeployment(&context, request, &response);

        if (!status.ok()) {
            last_error_ = "gRPC error: " + status.error_message();
            spdlog::error("{}", last_error_);
            return false;
        }

        if (response.status() != cyxwiz::protocol::STATUS_SUCCESS) {
            last_error_ = response.has_error() ?
                response.error().message() : "Stop failed";
            spdlog::error("{}", last_error_);
            return false;
        }

        spdlog::info("Deployment stopped: {}", deployment_id);
        return true;

    } catch (const std::exception& e) {
        last_error_ = std::string("Exception: ") + e.what();
        spdlog::error("{}", last_error_);
        return false;
    }
}

bool DeploymentClient::DeleteDeployment(const std::string& deployment_id) {
    if (!connected_ || !stub_) {
        last_error_ = "Not connected to Server Node";
        spdlog::error("{}", last_error_);
        return false;
    }

    spdlog::info("Deleting deployment: {}", deployment_id);

    try {
        cyxwiz::protocol::DeleteDeploymentRequest request;
        request.set_deployment_id(deployment_id);

        grpc::ClientContext context;
        context.set_deadline(std::chrono::system_clock::now() + std::chrono::seconds(10));
        AddAuthMetadata(context);

        cyxwiz::protocol::DeleteDeploymentResponse response;
        grpc::Status status = stub_->DeleteDeployment(&context, request, &response);

        if (!status.ok()) {
            last_error_ = "gRPC error: " + status.error_message();
            spdlog::error("{}", last_error_);
            return false;
        }

        if (response.status() != cyxwiz::protocol::STATUS_SUCCESS) {
            last_error_ = response.has_error() ?
                response.error().message() : "Delete failed";
            spdlog::error("{}", last_error_);
            return false;
        }

        spdlog::info("Deployment deleted: {}", deployment_id);
        return true;

    } catch (const std::exception& e) {
        last_error_ = std::string("Exception: ") + e.what();
        spdlog::error("{}", last_error_);
        return false;
    }
}

} // namespace network
