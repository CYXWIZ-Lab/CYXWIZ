#include "job_execution_service.h"
#include <spdlog/spdlog.h>
#include <grpcpp/create_channel.h>
#include <chrono>
#include <thread>
#include <fstream>
#include <filesystem>
#include "node.grpc.pb.h"

namespace cyxwiz {
namespace server_node {

namespace fs = std::filesystem;

JobExecutionServiceImpl::JobExecutionServiceImpl() {
    // Initialize node capabilities
    capabilities_.add_supported_devices(cyxwiz::protocol::DEVICE_CPU);
#ifdef CYXWIZ_ENABLE_CUDA
    capabilities_.add_supported_devices(cyxwiz::protocol::DEVICE_CUDA);
#endif
#ifdef CYXWIZ_ENABLE_OPENCL
    capabilities_.add_supported_devices(cyxwiz::protocol::DEVICE_OPENCL);
#endif

    capabilities_.set_max_memory(8LL * 1024 * 1024 * 1024); // 8GB default
    capabilities_.set_max_batch_size(512);
    capabilities_.add_supported_optimizers("SGD");
    capabilities_.add_supported_optimizers("Adam");
    capabilities_.add_supported_optimizers("AdamW");
    capabilities_.add_supported_optimizers("RMSprop");
    capabilities_.set_supports_checkpointing(true);
    capabilities_.set_supports_distributed(false);

    // Generate node ID (in production, load from config)
    node_id_ = "node_" + std::to_string(std::chrono::system_clock::now().time_since_epoch().count());
}

JobExecutionServiceImpl::~JobExecutionServiceImpl() {
    StopServer();
}

void JobExecutionServiceImpl::Initialize(
    std::shared_ptr<JobExecutor> executor,
    const std::string& central_server_address) {
    job_executor_ = executor;
    central_server_address_ = central_server_address;
}

bool JobExecutionServiceImpl::StartServer(const std::string& listen_address) {
    if (is_running_) {
        spdlog::warn("JobExecutionService already running");
        return false;
    }

    try {
        grpc::ServerBuilder builder;
        builder.AddListeningPort(listen_address, grpc::InsecureServerCredentials());
        builder.RegisterService(this);

        // Set max message size for large model transfers
        builder.SetMaxReceiveMessageSize(100 * 1024 * 1024); // 100MB
        builder.SetMaxSendMessageSize(100 * 1024 * 1024);    // 100MB

        server_ = builder.BuildAndStart();
        if (!server_) {
            spdlog::error("Failed to start JobExecutionService on {}", listen_address);
            return false;
        }

        is_running_ = true;
        spdlog::info("JobExecutionService listening on {}", listen_address);
        return true;
    } catch (const std::exception& e) {
        spdlog::error("Failed to start JobExecutionService: {}", e.what());
        return false;
    }
}

void JobExecutionServiceImpl::StopServer() {
    if (!is_running_ || !server_) {
        return;
    }

    spdlog::info("Stopping JobExecutionService...");

    // Cancel all active jobs
    {
        std::lock_guard<std::mutex> lock(jobs_mutex_);
        for (auto& [job_id, session] : active_jobs_) {
            session->should_stop = true;
        }
    }

    server_->Shutdown();
    is_running_ = false;
    spdlog::info("JobExecutionService stopped");
}

// ========== gRPC Service Methods ==========

grpc::Status JobExecutionServiceImpl::ConnectToNode(
    grpc::ServerContext* context,
    const cyxwiz::protocol::ConnectRequest* request,
    cyxwiz::protocol::ConnectResponse* response) {

    spdlog::info("Engine connecting with job_id={}, version={}",
                request->job_id(), request->engine_version());

    // Verify auth token
    if (!VerifyAuthToken(request->auth_token(), request->job_id())) {
        response->set_status(cyxwiz::protocol::STATUS_ERROR);
        response->mutable_error()->set_code(401);
        response->mutable_error()->set_message("Invalid or expired auth token");
        return grpc::Status::OK;
    }

    // Store connection info
    ConnectionInfo conn_info;
    conn_info.job_id = request->job_id();
    conn_info.auth_token = request->auth_token();
    conn_info.engine_address = context->peer();
    conn_info.connected_at = std::chrono::system_clock::now().time_since_epoch().count();
    conn_info.is_authenticated = true;

    {
        std::lock_guard<std::mutex> lock(connections_mutex_);
        connections_[context->peer()] = conn_info;
    }

    // Build response
    response->set_status(cyxwiz::protocol::STATUS_SUCCESS);
    response->set_node_id(node_id_);
    *response->mutable_capabilities() = capabilities_;

    spdlog::info("Engine connected successfully from {}", context->peer());
    return grpc::Status::OK;
}

grpc::Status JobExecutionServiceImpl::SendJob(
    grpc::ServerContext* context,
    const cyxwiz::protocol::SendJobRequest* request,
    cyxwiz::protocol::SendJobResponse* response) {

    spdlog::info("Received job {} from Engine", request->job_id());

    // Verify connection is authenticated
    {
        std::lock_guard<std::mutex> lock(connections_mutex_);
        auto it = connections_.find(context->peer());
        if (it == connections_.end() || !it->second.is_authenticated) {
            response->set_status(cyxwiz::protocol::STATUS_ERROR);
            response->mutable_error()->set_code(401);
            response->mutable_error()->set_message("Not authenticated");
            return grpc::Status::OK;
        }
    }

    // Check if we have too many active jobs (simple capacity check)
    {
        std::lock_guard<std::mutex> lock(jobs_mutex_);
        if (active_jobs_.size() >= 10) {  // Max 10 concurrent jobs
            response->set_status(cyxwiz::protocol::STATUS_ERROR);
            response->set_accepted(false);
            response->set_rejection_reason("Node at capacity");
            return grpc::Status::OK;
        }
    }

    // Create job session
    auto session = std::make_unique<JobSession>();
    session->job_id = request->job_id();
    session->engine_address = context->peer();
    session->job_config = request->config();
    session->is_running = false;
    session->is_paused = false;
    session->should_stop = false;

    // Save dataset if provided inline
    if (!request->initial_dataset().empty()) {
        std::string dataset_path = SaveDatasetToFile(request->job_id(),
                                                     request->initial_dataset());
        if (dataset_path.empty()) {
            response->set_status(cyxwiz::protocol::STATUS_ERROR);
            response->set_accepted(false);
            response->set_rejection_reason("Failed to save dataset");
            return grpc::Status::OK;
        }
        session->job_config.set_dataset_uri("file://" + dataset_path);
    } else if (!request->dataset_uri().empty()) {
        session->job_config.set_dataset_uri(request->dataset_uri());
    }

    // Store job session
    {
        std::lock_guard<std::mutex> lock(jobs_mutex_);
        active_jobs_[request->job_id()] = std::move(session);
    }

    // Notify Central Server about job acceptance
    if (!NotifyCentralServer(request->job_id(), node_id_)) {
        spdlog::warn("Failed to notify Central Server about job acceptance");
        // Continue anyway - job can still run
    }

    // Build response
    response->set_status(cyxwiz::protocol::STATUS_SUCCESS);
    response->set_accepted(true);
    response->set_estimated_start_time(
        std::chrono::system_clock::now().time_since_epoch().count() + 5000); // 5 seconds

    spdlog::info("Job {} accepted for execution", request->job_id());
    return grpc::Status::OK;
}

grpc::Status JobExecutionServiceImpl::StreamTrainingMetrics(
    grpc::ServerContext* context,
    grpc::ServerReaderWriter<cyxwiz::protocol::TrainingUpdate,
                            cyxwiz::protocol::TrainingCommand>* stream) {

    // Find the job session for this connection
    std::string job_id;
    {
        std::lock_guard<std::mutex> lock(connections_mutex_);
        auto it = connections_.find(context->peer());
        if (it == connections_.end()) {
            return grpc::Status(grpc::StatusCode::UNAUTHENTICATED, "Not connected");
        }
        job_id = it->second.job_id;
    }

    JobSession* session = nullptr;
    {
        std::lock_guard<std::mutex> lock(jobs_mutex_);
        auto it = active_jobs_.find(job_id);
        if (it == active_jobs_.end()) {
            return grpc::Status(grpc::StatusCode::NOT_FOUND, "Job not found");
        }
        session = it->second.get();
    }

    spdlog::info("Starting training metrics stream for job {}", job_id);

    // Start the training in a separate thread
    session->is_running = true;
    std::thread training_thread([this, session, stream, context, job_id]() {
        // Simulate training with periodic updates
        const int total_epochs = session->job_config.epochs();
        const int batch_size = session->job_config.batch_size();

        for (int epoch = 1; epoch <= total_epochs && !session->should_stop; ++epoch) {
            // Simulate epoch training
            for (int batch = 0; batch < 100 && !session->should_stop; ++batch) {
                // Check if paused
                while (session->is_paused && !session->should_stop) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(100));
                }

                // Prepare training update
                cyxwiz::protocol::TrainingUpdate update;
                update.set_job_id(job_id);
                update.set_timestamp(std::chrono::system_clock::now().time_since_epoch().count());

                auto* progress = update.mutable_progress();
                progress->set_current_epoch(epoch);
                progress->set_total_epochs(total_epochs);
                progress->set_current_batch(batch);
                progress->set_total_batches(100);
                progress->set_progress_percentage(
                    static_cast<double>(epoch - 1) / total_epochs +
                    static_cast<double>(batch) / (100.0 * total_epochs));

                // Add simulated metrics
                double loss = 2.0 / (epoch + batch * 0.01);
                double accuracy = std::min(0.99, 0.5 + epoch * 0.05 + batch * 0.001);
                (*progress->mutable_metrics())["loss"] = loss;
                (*progress->mutable_metrics())["accuracy"] = accuracy;

                // Resource usage (simulated)
                progress->set_gpu_usage(0.75 + (rand() % 20) / 100.0);
                progress->set_cpu_usage(0.30 + (rand() % 20) / 100.0);
                progress->set_memory_usage(0.60 + (rand() % 10) / 100.0);

                // Time estimates
                progress->set_elapsed_time((epoch - 1) * 60 + batch);
                progress->set_estimated_time_remaining((total_epochs - epoch) * 60);
                progress->set_samples_per_second(batch_size * 10.0);

                // Send update to Engine
                if (!stream->Write(update)) {
                    spdlog::warn("Failed to write training update, client may have disconnected");
                    session->should_stop = true;
                    break;
                }

                // Store latest progress
                {
                    std::lock_guard<std::mutex> lock(session->metrics_mutex);
                    session->latest_progress = *progress;
                }

                // Simulate computation time
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }

            // Send checkpoint at end of epoch
            if (!session->should_stop && epoch % 5 == 0) {
                cyxwiz::protocol::TrainingUpdate checkpoint_update;
                checkpoint_update.set_job_id(job_id);
                checkpoint_update.set_timestamp(
                    std::chrono::system_clock::now().time_since_epoch().count());

                auto* checkpoint = checkpoint_update.mutable_checkpoint();
                checkpoint->set_epoch(epoch);
                checkpoint->set_weights_size(10 * 1024 * 1024); // 10MB simulated
                checkpoint->set_compression_type("gzip");
                checkpoint->set_checkpoint_hash("abc123def456");
                (*checkpoint->mutable_metrics_at_checkpoint())["loss"] =
                    2.0 / (epoch + 1);
                (*checkpoint->mutable_metrics_at_checkpoint())["accuracy"] =
                    std::min(0.99, 0.5 + epoch * 0.05);

                stream->Write(checkpoint_update);
            }
        }

        // Training complete
        if (!session->should_stop) {
            cyxwiz::protocol::TrainingUpdate complete_update;
            complete_update.set_job_id(job_id);
            complete_update.set_timestamp(
                std::chrono::system_clock::now().time_since_epoch().count());

            auto* complete = complete_update.mutable_complete();
            complete->set_success(true);
            complete->set_result_hash("final_model_hash_xyz");
            (*complete->mutable_final_metrics())["loss"] = 0.15;
            (*complete->mutable_final_metrics())["accuracy"] = 0.98;
            complete->set_total_training_time(total_epochs * 60);
            complete->set_total_epochs_completed(total_epochs);
            complete->set_weights_location("/models/" + job_id + ".pt");
            complete->set_final_weights_size(50 * 1024 * 1024); // 50MB
            complete->set_proof_of_compute("proof_hash_123");

            stream->Write(complete_update);

            // Save final weights path
            session->final_weights_path = "/tmp/models/" + job_id + ".pt";
        }

        session->is_running = false;
    });

    // Read commands from Engine
    cyxwiz::protocol::TrainingCommand command;
    while (stream->Read(&command)) {
        if (command.has_pause()) {
            session->is_paused = command.pause();
            spdlog::info("Job {} {}", job_id, command.pause() ? "paused" : "resumed");
        } else if (command.has_stop()) {
            session->should_stop = true;
            spdlog::info("Job {} stop requested", job_id);
            break;
        } else if (command.has_request_checkpoint()) {
            // TODO: Trigger checkpoint save
            spdlog::info("Job {} checkpoint requested", job_id);
        } else if (command.has_update_params()) {
            // TODO: Update hyperparameters
            spdlog::info("Job {} hyperparameter update requested", job_id);
        }
    }

    // Wait for training thread to complete
    if (training_thread.joinable()) {
        training_thread.join();
    }

    spdlog::info("Training metrics stream ended for job {}", job_id);
    return grpc::Status::OK;
}

grpc::Status JobExecutionServiceImpl::DownloadWeights(
    grpc::ServerContext* context,
    const cyxwiz::protocol::DownloadRequest* request,
    grpc::ServerWriter<cyxwiz::protocol::WeightsChunk>* writer) {

    spdlog::info("Weights download requested for job {}", request->job_id());

    // Find job session
    JobSession* session = nullptr;
    {
        std::lock_guard<std::mutex> lock(jobs_mutex_);
        auto it = active_jobs_.find(request->job_id());
        if (it == active_jobs_.end()) {
            return grpc::Status(grpc::StatusCode::NOT_FOUND, "Job not found");
        }
        session = it->second.get();
    }

    // Simulate sending model weights in chunks
    const size_t total_size = 50 * 1024 * 1024; // 50MB simulated
    const size_t chunk_size = request->chunk_size() > 0 ?
                             request->chunk_size() : 1024 * 1024; // 1MB default

    std::vector<char> dummy_data(chunk_size, 'W'); // Simulated weight data
    size_t offset = request->offset();

    while (offset < total_size) {
        size_t current_chunk = std::min(chunk_size, total_size - offset);

        cyxwiz::protocol::WeightsChunk chunk;
        chunk.set_data(dummy_data.data(), current_chunk);
        chunk.set_offset(offset);
        chunk.set_total_size(total_size);
        chunk.set_is_last_chunk(offset + current_chunk >= total_size);
        chunk.set_checksum("chunk_checksum_" + std::to_string(offset));

        if (!writer->Write(chunk)) {
            spdlog::warn("Failed to write weights chunk, client may have disconnected");
            break;
        }

        offset += current_chunk;
    }

    spdlog::info("Weights download completed for job {}, {} bytes sent",
                request->job_id(), offset);
    return grpc::Status::OK;
}

// ========== Helper Methods ==========

bool JobExecutionServiceImpl::VerifyAuthToken(const std::string& token,
                                              const std::string& job_id) {
    // TODO: Implement proper JWT verification
    // For now, accept any non-empty token
    if (token.empty()) {
        spdlog::warn("Empty auth token for job {}", job_id);
        return false;
    }

    // In production, verify with Central Server or decode JWT
    spdlog::debug("Auth token verified for job {}", job_id);
    return true;
}

bool JobExecutionServiceImpl::NotifyCentralServer(const std::string& job_id,
                                                  const std::string& node_id) {
    try {
        // Create gRPC channel to Central Server
        auto channel = grpc::CreateChannel(central_server_address_,
                                          grpc::InsecureChannelCredentials());
        auto stub = cyxwiz::protocol::NodeService::NewStub(channel);

        // Prepare request
        cyxwiz::protocol::JobAcceptedRequest request;
        request.set_node_id(node_id);
        request.set_job_id(job_id);
        request.set_engine_address("direct_p2p");
        request.set_accepted_at(
            std::chrono::system_clock::now().time_since_epoch().count());
        request.set_node_endpoint("0.0.0.0:50052"); // Our P2P endpoint

        // Send notification
        cyxwiz::protocol::JobAcceptedResponse response;
        grpc::ClientContext context;

        auto deadline = std::chrono::system_clock::now() + std::chrono::seconds(5);
        context.set_deadline(deadline);

        grpc::Status status = stub->NotifyJobAccepted(&context, request, &response);

        if (status.ok() && response.status() == cyxwiz::protocol::STATUS_SUCCESS) {
            spdlog::info("Central Server notified about job {} acceptance", job_id);
            return true;
        } else {
            spdlog::error("Failed to notify Central Server: {}",
                         status.error_message());
            return false;
        }
    } catch (const std::exception& e) {
        spdlog::error("Exception notifying Central Server: {}", e.what());
        return false;
    }
}

std::string JobExecutionServiceImpl::SaveDatasetToFile(const std::string& job_id,
                                                       const std::string& dataset_data) {
    try {
        fs::path dataset_dir = fs::temp_directory_path() / "cyxwiz_datasets";
        fs::create_directories(dataset_dir);

        fs::path dataset_path = dataset_dir / (job_id + ".data");

        std::ofstream file(dataset_path, std::ios::binary);
        if (!file) {
            spdlog::error("Failed to create dataset file for job {}", job_id);
            return "";
        }

        file.write(dataset_data.data(), dataset_data.size());
        file.close();

        spdlog::info("Dataset saved to {} ({} bytes)",
                    dataset_path.string(), dataset_data.size());
        return dataset_path.string();
    } catch (const std::exception& e) {
        spdlog::error("Failed to save dataset for job {}: {}", job_id, e.what());
        return "";
    }
}

} // namespace server_node
} // namespace cyxwiz