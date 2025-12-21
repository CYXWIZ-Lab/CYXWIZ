#include "grpc_client.h"
#include <spdlog/spdlog.h>

namespace network {

GRPCClient::GRPCClient() : connected_(false) {
}

void GRPCClient::AddAuthMetadata(grpc::ClientContext& context) {
    if (!auth_token_.empty()) {
        // Add Bearer token to authorization header
        context.AddMetadata("authorization", "Bearer " + auth_token_);
        spdlog::debug("Added auth token to gRPC request");
    }
}

GRPCClient::~GRPCClient() {
    Disconnect();
}

bool GRPCClient::Connect(const std::string& server_address) {
    spdlog::info("Connecting to Central Server: {}", server_address);

    try {
        // Create gRPC channel
        channel_ = grpc::CreateChannel(server_address, grpc::InsecureChannelCredentials());

        // Create stubs for services
        job_stub_ = cyxwiz::protocol::JobService::NewStub(channel_);
        node_discovery_stub_ = cyxwiz::protocol::NodeDiscoveryService::NewStub(channel_);

        // Actually wait for connection with a timeout (5 seconds)
        // GetState(true) initiates connection, then WaitForConnected waits for it
        channel_->GetState(true);  // Trigger connection attempt

        auto deadline = std::chrono::system_clock::now() + std::chrono::seconds(5);
        bool connected = channel_->WaitForConnected(deadline);

        if (!connected) {
            auto state = channel_->GetState(false);
            last_error_ = "Failed to connect to server: Connection timeout";
            if (state == GRPC_CHANNEL_TRANSIENT_FAILURE) {
                last_error_ = "Failed to connect to server: Connection refused";
            }
            spdlog::error(last_error_);

            // Clean up
            job_stub_.reset();
            node_discovery_stub_.reset();
            channel_.reset();
            return false;
        }

        server_address_ = server_address;
        connected_ = true;
        spdlog::info("Successfully connected to Central Server");
        return true;

    } catch (const std::exception& e) {
        last_error_ = std::string("Connection error: ") + e.what();
        spdlog::error(last_error_);
        connected_ = false;
        return false;
    }
}

void GRPCClient::Disconnect() {
    if (connected_) {
        spdlog::info("Disconnecting from Central Server");
        job_stub_.reset();
        node_discovery_stub_.reset();
        channel_.reset();
        connected_ = false;
        server_address_.clear();
    }
}

bool GRPCClient::SubmitJob(const cyxwiz::protocol::SubmitJobRequest& request,
                            cyxwiz::protocol::SubmitJobResponse& response) {
    if (!connected_ || !job_stub_) {
        last_error_ = "Not connected to server";
        spdlog::error(last_error_);
        return false;
    }

    spdlog::info("Submitting job to Central Server...");

    try {
        grpc::ClientContext context;

        // Set timeout for the RPC call (30 seconds)
        std::chrono::system_clock::time_point deadline =
            std::chrono::system_clock::now() + std::chrono::seconds(30);
        context.set_deadline(deadline);

        // Add authentication header
        AddAuthMetadata(context);

        // Make the RPC call
        grpc::Status status = job_stub_->SubmitJob(&context, request, &response);

        if (status.ok()) {
            spdlog::info("Job submitted successfully. Job ID: {}", response.job_id());
            return true;
        } else {
            last_error_ = "Job submission failed: " + status.error_message();
            spdlog::error("{} (code: {})", last_error_, static_cast<int>(status.error_code()));
            return false;
        }

    } catch (const std::exception& e) {
        last_error_ = std::string("Job submission error: ") + e.what();
        spdlog::error(last_error_);
        return false;
    }
}

bool GRPCClient::GetJobStatus(const std::string& job_id,
                                cyxwiz::protocol::GetJobStatusResponse& response) {
    if (!connected_ || !job_stub_) {
        last_error_ = "Not connected to server";
        spdlog::error(last_error_);
        return false;
    }

    try {
        cyxwiz::protocol::GetJobStatusRequest request;
        request.set_job_id(job_id);

        grpc::ClientContext context;
        context.set_deadline(std::chrono::system_clock::now() + std::chrono::seconds(10));
        AddAuthMetadata(context);

        grpc::Status status = job_stub_->GetJobStatus(&context, request, &response);

        if (status.ok()) {
            return true;
        } else {
            last_error_ = "Get job status failed: " + status.error_message();
            spdlog::error(last_error_);
            return false;
        }

    } catch (const std::exception& e) {
        last_error_ = std::string("Get job status error: ") + e.what();
        spdlog::error(last_error_);
        return false;
    }
}

bool GRPCClient::CancelJob(const std::string& job_id,
                             cyxwiz::protocol::CancelJobResponse& response) {
    if (!connected_ || !job_stub_) {
        last_error_ = "Not connected to server";
        spdlog::error(last_error_);
        return false;
    }

    try {
        cyxwiz::protocol::CancelJobRequest request;
        request.set_job_id(job_id);

        grpc::ClientContext context;
        context.set_deadline(std::chrono::system_clock::now() + std::chrono::seconds(10));
        AddAuthMetadata(context);

        grpc::Status status = job_stub_->CancelJob(&context, request, &response);

        if (status.ok()) {
            spdlog::info("Job cancelled successfully: {}", job_id);
            return true;
        } else {
            last_error_ = "Cancel job failed: " + status.error_message();
            spdlog::error(last_error_);
            return false;
        }

    } catch (const std::exception& e) {
        last_error_ = std::string("Cancel job error: ") + e.what();
        spdlog::error(last_error_);
        return false;
    }
}

bool GRPCClient::DeleteJob(const std::string& job_id,
                             cyxwiz::protocol::DeleteJobResponse& response) {
    if (!connected_ || !job_stub_) {
        last_error_ = "Not connected to server";
        spdlog::error(last_error_);
        return false;
    }

    try {
        cyxwiz::protocol::DeleteJobRequest request;
        request.set_job_id(job_id);

        grpc::ClientContext context;
        context.set_deadline(std::chrono::system_clock::now() + std::chrono::seconds(10));
        AddAuthMetadata(context);

        grpc::Status status = job_stub_->DeleteJob(&context, request, &response);

        if (status.ok()) {
            spdlog::info("Job deleted successfully: {}", job_id);
            return true;
        } else {
            last_error_ = "Delete job failed: " + status.error_message();
            spdlog::error(last_error_);
            return false;
        }

    } catch (const std::exception& e) {
        last_error_ = std::string("Delete job error: ") + e.what();
        spdlog::error(last_error_);
        return false;
    }
}

bool GRPCClient::ListJobs(cyxwiz::protocol::ListJobsResponse& response,
                           const std::string& user_id,
                           int page_size) {
    if (!connected_ || !job_stub_) {
        last_error_ = "Not connected to server";
        spdlog::error(last_error_);
        return false;
    }

    try {
        cyxwiz::protocol::ListJobsRequest request;
        if (!user_id.empty()) {
            request.set_user_id(user_id);
        }
        request.set_page_size(page_size);

        grpc::ClientContext context;
        context.set_deadline(std::chrono::system_clock::now() + std::chrono::seconds(10));
        AddAuthMetadata(context);

        grpc::Status status = job_stub_->ListJobs(&context, request, &response);

        if (status.ok()) {
            spdlog::debug("ListJobs returned {} jobs", response.jobs_size());
            return true;
        } else {
            last_error_ = "List jobs failed: " + status.error_message();
            spdlog::error(last_error_);
            return false;
        }

    } catch (const std::exception& e) {
        last_error_ = std::string("List jobs error: ") + e.what();
        spdlog::error(last_error_);
        return false;
    }
}

// ============================================================================
// Node Discovery Operations
// ============================================================================

NodeDisplayInfo GRPCClient::ConvertNodeInfo(const cyxwiz::protocol::NodeInfo& proto) {
    NodeDisplayInfo info;
    info.node_id = proto.node_id();
    info.name = proto.name();
    info.region = proto.region();
    info.is_online = proto.is_online();

    // Network endpoint
    info.ip_address = proto.ip_address();
    info.port = proto.port() > 0 ? proto.port() : 50052;  // Default to 50052 if not set

    // Debug: Log device info
    spdlog::info("ConvertNodeInfo: node={}, endpoint={}:{}, devices_count={}",
                 proto.name(), info.ip_address, info.port, proto.devices_size());

    // Hardware - get first device type
    if (proto.devices_size() > 0) {
        const auto& device = proto.devices(0);
        spdlog::info("  Device[0]: type={}, name={}, vram_total={}, memory_total={}",
                     static_cast<int>(device.device_type()), device.device_name(),
                     device.vram_total(), device.memory_total());

        switch (device.device_type()) {
            case cyxwiz::protocol::DEVICE_CUDA:
                info.device_type = "CUDA";
                break;
            case cyxwiz::protocol::DEVICE_OPENCL:
                info.device_type = "OpenCL";
                break;
            case cyxwiz::protocol::DEVICE_CPU:
            default:
                info.device_type = "CPU";
                break;
        }
        info.vram_bytes = device.vram_total();
    } else {
        spdlog::warn("  No devices found for node {}", proto.name());
    }
    info.cpu_cores = proto.cpu_cores();

    // Performance
    info.compute_score = proto.compute_score();
    info.reputation_score = proto.reputation_score();
    info.total_jobs_completed = proto.total_jobs_completed();

    // Pricing
    if (proto.has_pricing()) {
        const auto& pricing = proto.pricing();
        info.price_per_hour = pricing.price_per_hour();
        info.price_usd_equivalent = pricing.usd_equivalent();
        info.free_tier_available = pricing.free_tier_available();

        switch (pricing.billing_model()) {
            case cyxwiz::protocol::NodePricing::BILLING_HOURLY:
                info.billing_model = "Hourly";
                break;
            case cyxwiz::protocol::NodePricing::BILLING_PER_EPOCH:
                info.billing_model = "Per Epoch";
                break;
            case cyxwiz::protocol::NodePricing::BILLING_PER_JOB:
                info.billing_model = "Per Job";
                break;
            case cyxwiz::protocol::NodePricing::BILLING_PER_INFERENCE:
                info.billing_model = "Per Inference";
                break;
            default:
                info.billing_model = "Hourly";
                break;
        }
    }

    // Staking
    info.staked_amount = proto.staked_amount();

    return info;
}

bool GRPCClient::ListNodes(std::vector<NodeDisplayInfo>& out_nodes,
                            bool online_only,
                            int page_size) {
    if (!connected_ || !node_discovery_stub_) {
        last_error_ = "Not connected to server";
        spdlog::error(last_error_);
        return false;
    }

    try {
        cyxwiz::protocol::ListNodesRequest request;
        request.set_online_only(online_only);
        request.set_page_size(page_size);

        cyxwiz::protocol::ListNodesResponse response;
        grpc::ClientContext context;
        context.set_deadline(std::chrono::system_clock::now() + std::chrono::seconds(15));
        AddAuthMetadata(context);

        grpc::Status status = node_discovery_stub_->ListNodes(&context, request, &response);

        if (status.ok()) {
            out_nodes.clear();
            out_nodes.reserve(response.nodes_size());

            for (const auto& node : response.nodes()) {
                out_nodes.push_back(ConvertNodeInfo(node));
            }

            spdlog::debug("ListNodes returned {} nodes (online: {})",
                          out_nodes.size(), response.online_count());
            return true;
        } else {
            last_error_ = "List nodes failed: " + status.error_message();
            spdlog::error(last_error_);
            return false;
        }

    } catch (const std::exception& e) {
        last_error_ = std::string("List nodes error: ") + e.what();
        spdlog::error(last_error_);
        return false;
    }
}

bool GRPCClient::FindNodes(const NodeSearchCriteria& criteria,
                            std::vector<NodeDisplayInfo>& out_nodes) {
    if (!connected_ || !node_discovery_stub_) {
        last_error_ = "Not connected to server";
        spdlog::error(last_error_);
        return false;
    }

    try {
        cyxwiz::protocol::FindNodesRequest request;

        // Hardware requirements
        if (criteria.required_device == "CUDA") {
            request.set_required_device(cyxwiz::protocol::DEVICE_CUDA);
        } else if (criteria.required_device == "OpenCL") {
            request.set_required_device(cyxwiz::protocol::DEVICE_OPENCL);
        } else if (criteria.required_device == "CPU") {
            request.set_required_device(cyxwiz::protocol::DEVICE_CPU);
        }

        if (criteria.min_vram > 0) {
            request.set_min_vram(criteria.min_vram);
        }
        if (criteria.min_compute_score > 0) {
            request.set_min_compute_score(criteria.min_compute_score);
        }

        // Trust requirements
        if (criteria.min_reputation > 0) {
            request.set_min_reputation(criteria.min_reputation);
        }
        if (criteria.min_stake > 0) {
            request.set_min_stake(criteria.min_stake);
        }

        // Pricing
        if (criteria.max_price_per_hour > 0) {
            request.set_max_price_per_hour(criteria.max_price_per_hour);
        }
        request.set_require_free_tier(criteria.require_free_tier);

        // Location
        if (!criteria.preferred_region.empty()) {
            request.set_preferred_region(criteria.preferred_region);
        }

        // Results
        request.set_max_results(criteria.max_results);

        // Sorting
        switch (criteria.sort_by) {
            case 0:
                request.set_sort_by(cyxwiz::protocol::FindNodesRequest::SORT_BY_PRICE);
                break;
            case 1:
                request.set_sort_by(cyxwiz::protocol::FindNodesRequest::SORT_BY_PERFORMANCE);
                break;
            case 2:
                request.set_sort_by(cyxwiz::protocol::FindNodesRequest::SORT_BY_REPUTATION);
                break;
            case 3:
                request.set_sort_by(cyxwiz::protocol::FindNodesRequest::SORT_BY_AVAILABILITY);
                break;
            default:
                request.set_sort_by(cyxwiz::protocol::FindNodesRequest::SORT_BY_PRICE);
                break;
        }

        cyxwiz::protocol::FindNodesResponse response;
        grpc::ClientContext context;
        context.set_deadline(std::chrono::system_clock::now() + std::chrono::seconds(15));
        AddAuthMetadata(context);

        grpc::Status status = node_discovery_stub_->FindNodes(&context, request, &response);

        if (status.ok()) {
            out_nodes.clear();
            out_nodes.reserve(response.nodes_size());

            for (const auto& node : response.nodes()) {
                out_nodes.push_back(ConvertNodeInfo(node));
            }

            spdlog::debug("FindNodes returned {} nodes (total matching: {})",
                          out_nodes.size(), response.total_matching());
            return true;
        } else {
            last_error_ = "Find nodes failed: " + status.error_message();
            spdlog::error(last_error_);
            return false;
        }

    } catch (const std::exception& e) {
        last_error_ = std::string("Find nodes error: ") + e.what();
        spdlog::error(last_error_);
        return false;
    }
}

bool GRPCClient::GetNodeInfo(const std::string& node_id,
                              NodeDisplayInfo& out_node) {
    if (!connected_ || !node_discovery_stub_) {
        last_error_ = "Not connected to server";
        spdlog::error(last_error_);
        return false;
    }

    try {
        cyxwiz::protocol::GetNodeInfoRequest request;
        request.set_node_id(node_id);

        cyxwiz::protocol::GetNodeInfoResponse response;
        grpc::ClientContext context;
        context.set_deadline(std::chrono::system_clock::now() + std::chrono::seconds(10));
        AddAuthMetadata(context);

        grpc::Status status = node_discovery_stub_->GetNodeInfo(&context, request, &response);

        if (status.ok() && response.has_info()) {
            out_node = ConvertNodeInfo(response.info());
            return true;
        } else {
            last_error_ = "Get node info failed: " + status.error_message();
            spdlog::error(last_error_);
            return false;
        }

    } catch (const std::exception& e) {
        last_error_ = std::string("Get node info error: ") + e.what();
        spdlog::error(last_error_);
        return false;
    }
}

} // namespace network
