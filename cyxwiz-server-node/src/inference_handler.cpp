// inference_handler.cpp - gRPC InferenceService implementation
#include "inference_handler.h"
#include "deployment_manager.h"
#include <spdlog/spdlog.h>
#include <cyxwiz/tensor.h>
#include <chrono>

namespace cyxwiz::servernode {

// Helper: Convert proto TensorData to cyxwiz::Tensor
static cyxwiz::Tensor TensorFromProto(const protocol::TensorData& proto) {
    // Extract shape
    std::vector<size_t> shape;
    shape.reserve(proto.shape_size());
    for (int i = 0; i < proto.shape_size(); ++i) {
        shape.push_back(static_cast<size_t>(proto.shape(i)));
    }

    // Extract data (interpret bytes as float32)
    const std::string& data_bytes = proto.data();
    const float* data_ptr = reinterpret_cast<const float*>(data_bytes.data());

    // Create tensor
    return cyxwiz::Tensor(shape, data_ptr);
}

// Helper: Convert cyxwiz::Tensor to proto TensorData
static void TensorToProto(const cyxwiz::Tensor& tensor, const std::string& name,
                          protocol::TensorData* proto) {
    proto->set_name(name);

    // Set shape
    for (size_t dim : tensor.Shape()) {
        proto->add_shape(static_cast<int64_t>(dim));
    }

    // Set dtype (assuming float32)
    proto->set_dtype(protocol::DATA_TYPE_FLOAT32);

    // Set data bytes
    size_t num_elements = 1;
    for (size_t dim : tensor.Shape()) {
        num_elements *= dim;
    }
    size_t num_bytes = num_elements * sizeof(float);

    const float* data_ptr = tensor.Data<float>();
    if (data_ptr) {
        proto->set_data(data_ptr, num_bytes);
    }
}

// ============================================================================
// InferenceServiceImpl
// ============================================================================

InferenceServiceImpl::InferenceServiceImpl(DeploymentManager* manager)
    : manager_(manager) {
    spdlog::info("InferenceServiceImpl created");
}

grpc::Status InferenceServiceImpl::Infer(
    grpc::ServerContext* context,
    const protocol::InferRequest* request,
    protocol::InferResponse* response) {

    auto start_time = std::chrono::high_resolution_clock::now();

    spdlog::debug("Infer request for deployment: {}", request->deployment_id());

    // Check manager
    if (!manager_) {
        response->set_status(protocol::STATUS_ERROR);
        response->mutable_error()->set_message("Deployment manager not available");
        return grpc::Status::OK;
    }

    // Check deployment exists
    if (!manager_->HasDeployment(request->deployment_id())) {
        response->set_status(protocol::STATUS_FAILED);
        response->mutable_error()->set_message("Deployment not found: " + request->deployment_id());
        return grpc::Status::OK;
    }

    try {
        // Convert proto inputs to cyxwiz::Tensor map
        std::unordered_map<std::string, cyxwiz::Tensor> inputs;
        for (int i = 0; i < request->inputs_size(); ++i) {
            const auto& tensor_data = request->inputs(i);
            std::string tensor_name = tensor_data.name();
            if (tensor_name.empty()) {
                tensor_name = "input";  // Default name
            }
            inputs[tensor_name] = TensorFromProto(tensor_data);

            spdlog::debug("Input tensor '{}': {} elements",
                tensor_name, inputs[tensor_name].NumElements());
        }

        // Run inference
        std::unordered_map<std::string, cyxwiz::Tensor> outputs;
        bool success = manager_->RunInference(request->deployment_id(), inputs, outputs);

        auto end_time = std::chrono::high_resolution_clock::now();
        double latency_ms = std::chrono::duration<double, std::milli>(
            end_time - start_time).count();
        response->set_latency_ms(latency_ms);

        if (success) {
            // Convert outputs to proto
            for (const auto& [name, tensor] : outputs) {
                auto* output = response->add_outputs();
                TensorToProto(tensor, name, output);
            }
            response->set_status(protocol::STATUS_SUCCESS);

            spdlog::debug("Inference completed in {:.2f}ms with {} outputs",
                latency_ms, outputs.size());
        } else {
            response->set_status(protocol::STATUS_FAILED);
            response->mutable_error()->set_message("Inference execution failed");
            spdlog::error("Inference failed for deployment: {}", request->deployment_id());
        }

    } catch (const std::exception& e) {
        spdlog::error("Infer exception: {}", e.what());
        response->set_status(protocol::STATUS_ERROR);
        response->mutable_error()->set_message(e.what());
    }

    return grpc::Status::OK;
}

grpc::Status InferenceServiceImpl::GetModelInfo(
    grpc::ServerContext* context,
    const protocol::GetModelInfoRequest* request,
    protocol::GetModelInfoResponse* response) {

    spdlog::debug("GetModelInfo request for deployment: {}", request->deployment_id());

    if (!manager_) {
        response->set_status(protocol::STATUS_ERROR);
        response->mutable_error()->set_message("Deployment manager not available");
        return grpc::Status::OK;
    }

    if (!manager_->HasDeployment(request->deployment_id())) {
        response->set_status(protocol::STATUS_FAILED);
        response->mutable_error()->set_message("Deployment not found: " + request->deployment_id());
        return grpc::Status::OK;
    }

    try {
        // Get input specs
        auto input_specs = manager_->GetInputSpecs(request->deployment_id());
        for (const auto& spec : input_specs) {
            auto* tensor_info = response->add_input_specs();
            tensor_info->set_name(spec.name);
            for (size_t dim : spec.shape) {
                tensor_info->add_shape(static_cast<int64_t>(dim));
            }
            // Map dtype string to enum
            if (spec.dtype == "float32") {
                tensor_info->set_dtype(protocol::DATA_TYPE_FLOAT32);
            } else if (spec.dtype == "float64") {
                tensor_info->set_dtype(protocol::DATA_TYPE_FLOAT64);
            } else if (spec.dtype == "int32") {
                tensor_info->set_dtype(protocol::DATA_TYPE_INT32);
            } else if (spec.dtype == "int64") {
                tensor_info->set_dtype(protocol::DATA_TYPE_INT64);
            } else {
                tensor_info->set_dtype(protocol::DATA_TYPE_UNKNOWN);
            }
        }

        // Get output specs
        auto output_specs = manager_->GetOutputSpecs(request->deployment_id());
        for (const auto& spec : output_specs) {
            auto* tensor_info = response->add_output_specs();
            tensor_info->set_name(spec.name);
            for (size_t dim : spec.shape) {
                tensor_info->add_shape(static_cast<int64_t>(dim));
            }
            // Map dtype string to enum
            if (spec.dtype == "float32") {
                tensor_info->set_dtype(protocol::DATA_TYPE_FLOAT32);
            } else if (spec.dtype == "float64") {
                tensor_info->set_dtype(protocol::DATA_TYPE_FLOAT64);
            } else if (spec.dtype == "int32") {
                tensor_info->set_dtype(protocol::DATA_TYPE_INT32);
            } else if (spec.dtype == "int64") {
                tensor_info->set_dtype(protocol::DATA_TYPE_INT64);
            } else {
                tensor_info->set_dtype(protocol::DATA_TYPE_UNKNOWN);
            }
        }

        response->set_status(protocol::STATUS_SUCCESS);
        spdlog::debug("GetModelInfo: {} inputs, {} outputs",
            input_specs.size(), output_specs.size());

    } catch (const std::exception& e) {
        spdlog::error("GetModelInfo exception: {}", e.what());
        response->set_status(protocol::STATUS_ERROR);
        response->mutable_error()->set_message(e.what());
    }

    return grpc::Status::OK;
}

// ============================================================================
// InferenceServer
// ============================================================================

InferenceServer::InferenceServer(const std::string& address, DeploymentManager* manager)
    : address_(address)
    , manager_(manager) {
}

InferenceServer::~InferenceServer() {
    Stop();
}

bool InferenceServer::Start() {
    if (running_) {
        spdlog::warn("InferenceServer already running on {}", address_);
        return true;
    }

    try {
        // Create service
        service_ = std::make_unique<InferenceServiceImpl>(manager_);

        // Build server
        grpc::ServerBuilder builder;
        builder.AddListeningPort(address_, grpc::InsecureServerCredentials());
        builder.RegisterService(service_.get());

        server_ = builder.BuildAndStart();

        if (!server_) {
            spdlog::error("Failed to start InferenceServer on {}", address_);
            return false;
        }

        running_ = true;
        spdlog::info("InferenceServer started on {}", address_);
        return true;

    } catch (const std::exception& e) {
        spdlog::error("InferenceServer Start exception: {}", e.what());
        return false;
    }
}

void InferenceServer::Stop() {
    if (!running_) {
        return;
    }

    spdlog::info("Stopping InferenceServer...");

    if (server_) {
        server_->Shutdown();
        server_.reset();
    }

    service_.reset();
    running_ = false;

    spdlog::info("InferenceServer stopped");
}

} // namespace cyxwiz::servernode
