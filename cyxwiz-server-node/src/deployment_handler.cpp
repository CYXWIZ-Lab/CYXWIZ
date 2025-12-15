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

grpc::Status DeploymentServiceImpl::CreateDeployment(
    grpc::ServerContext* context,
    const protocol::CreateDeploymentRequest* request,
    protocol::CreateDeploymentResponse* response) {

    const auto& config = request->config();
    spdlog::info("Received deployment assignment: model={}, type={}",
                 config.model().name(),
                 protocol::DeploymentType_Name(config.type()));

    try {
        // Accept the deployment - use name if model_id is empty
        std::string model_id = config.model().model_id();
        if (model_id.empty()) {
            model_id = config.model().name();
        }
        std::string deployment_id = manager_->AcceptDeployment(
            model_id,
            config.type(),
            config
        );

        // Build response
        auto* deployment = response->mutable_deployment();
        deployment->set_deployment_id(deployment_id);
        *deployment->mutable_config() = config;
        deployment->set_status(protocol::DEPLOYMENT_STATUS_LOADING);

        response->set_status(protocol::STATUS_SUCCESS);
        spdlog::info("Deployment {} accepted successfully", deployment_id);
        return grpc::Status::OK;

    } catch (const std::exception& e) {
        spdlog::error("Failed to accept deployment: {}", e.what());
        response->set_status(protocol::STATUS_ERROR);
        response->mutable_error()->set_message(e.what());
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

        response->set_status(protocol::STATUS_SUCCESS);
        spdlog::info("Deployment {} stopped successfully", request->deployment_id());
        return grpc::Status::OK;

    } catch (const std::exception& e) {
        spdlog::error("Failed to stop deployment {}: {}", request->deployment_id(), e.what());
        response->set_status(protocol::STATUS_ERROR);
        response->mutable_error()->set_message(e.what());
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
        deployment->set_deployment_id(request->deployment_id());
        deployment->set_status(status);

        return grpc::Status::OK;

    } catch (const std::exception& e) {
        spdlog::error("Failed to get deployment {}: {}", request->deployment_id(), e.what());
        response->mutable_error()->set_message(e.what());
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
            m->set_timestamp(metric.timestamp_ms);
            m->set_cpu_usage(metric.cpu_usage);
            m->set_gpu_usage(metric.gpu_usage);
            m->set_memory_usage(metric.memory_usage);
            m->set_request_count(metric.request_count);
            m->set_avg_latency_ms(metric.avg_latency_ms);
        }

        return grpc::Status::OK;

    } catch (const std::exception& e) {
        spdlog::error("Failed to get metrics for deployment {}: {}",
                     request->deployment_id(), e.what());
        response->mutable_error()->set_message(e.what());
        return grpc::Status(grpc::StatusCode::NOT_FOUND, e.what());
    }
}

grpc::Status DeploymentServiceImpl::ListDeployments(
    grpc::ServerContext* context,
    const protocol::ListDeploymentsRequest* request,
    protocol::ListDeploymentsResponse* response) {

    spdlog::debug("Received list deployments request");

    try {
        auto deployments = manager_->GetAllDeployments();

        for (const auto& info : deployments) {
            auto* deployment = response->add_deployments();
            deployment->set_deployment_id(info.id);
            deployment->mutable_config()->mutable_model()->set_name(info.model_id);
            deployment->mutable_config()->set_type(info.type);
            deployment->set_status(info.status);
        }

        response->set_total_count(static_cast<int32_t>(deployments.size()));
        spdlog::debug("Returned {} deployments", deployments.size());
        return grpc::Status::OK;

    } catch (const std::exception& e) {
        spdlog::error("Failed to list deployments: {}", e.what());
        response->mutable_error()->set_message(e.what());
        return grpc::Status(grpc::StatusCode::INTERNAL, e.what());
    }
}

grpc::Status DeploymentServiceImpl::DeleteDeployment(
    grpc::ServerContext* context,
    const protocol::DeleteDeploymentRequest* request,
    protocol::DeleteDeploymentResponse* response) {

    const auto& deployment_id = request->deployment_id();
    spdlog::info("Received delete deployment request: {}", deployment_id);

    try {
        if (!manager_->HasDeployment(deployment_id)) {
            response->set_status(protocol::STATUS_ERROR);
            response->mutable_error()->set_message("Deployment not found: " + deployment_id);
            return grpc::Status::OK;
        }

        if (manager_->RemoveDeployment(deployment_id)) {
            response->set_status(protocol::STATUS_SUCCESS);
            spdlog::info("Deployment deleted: {}", deployment_id);
        } else {
            response->set_status(protocol::STATUS_FAILED);
            response->mutable_error()->set_message("Cannot delete active deployment. Stop it first.");
        }

        return grpc::Status::OK;

    } catch (const std::exception& e) {
        spdlog::error("Failed to delete deployment {}: {}", deployment_id, e.what());
        response->set_status(protocol::STATUS_ERROR);
        response->mutable_error()->set_message(e.what());
        return grpc::Status(grpc::StatusCode::INTERNAL, e.what());
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
