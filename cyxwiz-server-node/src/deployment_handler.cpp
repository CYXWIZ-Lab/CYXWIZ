#include "deployment_handler.h"
#include <spdlog/spdlog.h>
#include <grpcpp/ext/proto_server_reflection_plugin.h>
#include <grpcpp/health_check_service_interface.h>

namespace cyxwiz {
namespace servernode {

// ============================================================================
// DeploymentServiceImpl Implementation
// ============================================================================

DeploymentServiceImpl::DeploymentServiceImpl(std::shared_ptr<DeploymentManager> manager)
    : manager_(manager) {
    spdlog::debug("DeploymentServiceImpl created");
}

grpc::Status DeploymentServiceImpl::AssignDeployment(
    grpc::ServerContext* context,
    const protocol::CreateDeploymentRequest* request,
    protocol::CreateDeploymentResponse* response) {

    spdlog::info("Received deployment assignment: model={}, type={}",
                 request->model_id(),
                 protocol::DeploymentType_Name(request->type()));

    try {
        // Accept the deployment
        std::string deployment_id = manager_->AcceptDeployment(
            request->model_id(),
            request->type(),
            request->config()
        );

        // Build response
        auto* deployment = response->mutable_deployment();
        deployment->set_id(deployment_id);
        deployment->set_model_id(request->model_id());
        deployment->set_type(request->type());
        deployment->set_status(protocol::DEPLOYMENT_STATUS_LOADING);
        deployment->set_assigned_node_id(manager_->GetNodeId());
        *deployment->mutable_config() = request->config();

        response->mutable_status()->set_code(protocol::STATUS_SUCCESS);
        spdlog::info("Deployment {} accepted successfully", deployment_id);
        return grpc::Status::OK;

    } catch (const std::exception& e) {
        spdlog::error("Failed to accept deployment: {}", e.what());
        response->mutable_status()->set_code(protocol::STATUS_INTERNAL_ERROR);
        response->mutable_status()->mutable_error()->set_message(e.what());
        return grpc::Status(grpc::StatusCode::INTERNAL, e.what());
    }
}

grpc::Status DeploymentServiceImpl::StopDeployment(
    grpc::ServerContext* context,
    const protocol::StopDeploymentRequest* request,
    protocol::StopDeploymentResponse* response) {

    spdlog::info("Received stop request for deployment: {}", request->deployment_id());

    try {
        manager_->StopDeployment(request->deployment_id());

        response->mutable_status()->set_code(protocol::STATUS_SUCCESS);
        spdlog::info("Deployment {} stopped successfully", request->deployment_id());
        return grpc::Status::OK;

    } catch (const std::exception& e) {
        spdlog::error("Failed to stop deployment {}: {}", request->deployment_id(), e.what());
        response->mutable_status()->set_code(protocol::STATUS_NOT_FOUND);
        response->mutable_status()->mutable_error()->set_message(e.what());
        return grpc::Status(grpc::StatusCode::NOT_FOUND, e.what());
    }
}

grpc::Status DeploymentServiceImpl::GetDeployment(
    grpc::ServerContext* context,
    const protocol::GetDeploymentRequest* request,
    protocol::GetDeploymentResponse* response) {

    spdlog::debug("Received get deployment request: {}", request->deployment_id());

    try {
        auto status = manager_->GetDeploymentStatus(request->deployment_id());

        auto* deployment = response->mutable_deployment();
        deployment->set_id(request->deployment_id());
        deployment->set_status(status);
        deployment->set_assigned_node_id(manager_->GetNodeId());

        response->mutable_status()->set_code(protocol::STATUS_SUCCESS);
        return grpc::Status::OK;

    } catch (const std::exception& e) {
        spdlog::error("Failed to get deployment {}: {}", request->deployment_id(), e.what());
        response->mutable_status()->set_code(protocol::STATUS_NOT_FOUND);
        response->mutable_status()->mutable_error()->set_message(e.what());
        return grpc::Status(grpc::StatusCode::NOT_FOUND, e.what());
    }
}

grpc::Status DeploymentServiceImpl::GetDeploymentMetrics(
    grpc::ServerContext* context,
    const protocol::GetDeploymentMetricsRequest* request,
    protocol::GetDeploymentMetricsResponse* response) {

    spdlog::debug("Received metrics request for deployment: {}", request->deployment_id());

    try {
        auto metrics = manager_->GetDeploymentMetrics(request->deployment_id());

        // Copy metrics to response
        for (const auto& metric : metrics) {
            auto* m = response->add_metrics();
            m->set_deployment_id(request->deployment_id());
            m->set_cpu_usage_percent(metric.cpu_usage);
            m->set_gpu_usage_percent(metric.gpu_usage);
            m->set_memory_usage_bytes(metric.memory_usage);
            m->set_request_count(metric.request_count);
            m->set_avg_latency_ms(metric.avg_latency_ms);
            m->set_timestamp(metric.timestamp);
        }

        response->mutable_status()->set_code(protocol::STATUS_SUCCESS);
        return grpc::Status::OK;

    } catch (const std::exception& e) {
        spdlog::error("Failed to get metrics for deployment {}: {}",
                     request->deployment_id(), e.what());
        response->mutable_status()->set_code(protocol::STATUS_NOT_FOUND);
        response->mutable_status()->mutable_error()->set_message(e.what());
        return grpc::Status(grpc::StatusCode::NOT_FOUND, e.what());
    }
}

// ============================================================================
// DeploymentHandler Implementation
// ============================================================================

DeploymentHandler::DeploymentHandler(const std::string& listen_address,
                                   std::shared_ptr<DeploymentManager> manager)
    : listen_address_(listen_address)
    , manager_(manager)
    , running_(false) {
    spdlog::info("DeploymentHandler created for address: {}", listen_address);
}

DeploymentHandler::~DeploymentHandler() {
    Stop();
}

bool DeploymentHandler::Start() {
    if (running_) {
        spdlog::warn("DeploymentHandler already running");
        return true;
    }

    try {
        // Create service implementation
        service_ = std::make_unique<DeploymentServiceImpl>(manager_);

        // Enable health check and reflection
        grpc::EnableDefaultHealthCheckService(true);
        grpc::reflection::InitProtoReflectionServerBuilderPlugin();

        // Build server
        grpc::ServerBuilder builder;
        builder.AddListeningPort(listen_address_, grpc::InsecureServerCredentials());
        builder.RegisterService(service_.get());

        // Set channel arguments for better performance
        builder.AddChannelArgument(GRPC_ARG_KEEPALIVE_TIME_MS, 10000);
        builder.AddChannelArgument(GRPC_ARG_KEEPALIVE_TIMEOUT_MS, 5000);
        builder.AddChannelArgument(GRPC_ARG_KEEPALIVE_PERMIT_WITHOUT_CALLS, 1);
        builder.AddChannelArgument(GRPC_ARG_HTTP2_MAX_PINGS_WITHOUT_DATA, 0);

        // Build and start server
        server_ = builder.BuildAndStart();
        if (!server_) {
            spdlog::error("Failed to start gRPC server on {}", listen_address_);
            return false;
        }

        running_ = true;
        spdlog::info("DeploymentHandler started successfully on {}", listen_address_);
        return true;

    } catch (const std::exception& e) {
        spdlog::error("Exception starting DeploymentHandler: {}", e.what());
        return false;
    }
}

void DeploymentHandler::Stop() {
    if (!running_) {
        return;
    }

    spdlog::info("Stopping DeploymentHandler...");

    if (server_) {
        // Graceful shutdown with 5 second timeout
        server_->Shutdown(std::chrono::system_clock::now() + std::chrono::seconds(5));
        server_->Wait();
        server_.reset();
    }

    service_.reset();
    running_ = false;

    spdlog::info("DeploymentHandler stopped");
}

} // namespace servernode
} // namespace cyxwiz
