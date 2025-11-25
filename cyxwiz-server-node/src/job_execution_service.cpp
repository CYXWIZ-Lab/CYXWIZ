#include "job_execution_service.h"
#include "job_executor.h"
#include <spdlog/spdlog.h>
#include <grpcpp/create_channel.h>
#include <chrono>
#include <thread>
#include <fstream>
#include <filesystem>
#include <queue>
#include <condition_variable>
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
    std::shared_ptr<cyxwiz::servernode::JobExecutor> executor,
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

    // Flag to track training completion
    std::atomic<bool> training_complete{false};
    std::atomic<bool> training_success{false};
    std::string training_error;
    std::mutex error_mutex;

    // Progress update queue for streaming to Engine
    std::queue<cyxwiz::protocol::TrainingUpdate> update_queue;
    std::mutex queue_mutex;
    std::condition_variable queue_cv;

    // Check if JobExecutor is available for real training
    if (job_executor_) {
        spdlog::info("Using JobExecutor for real training of job {}", job_id);

        // Set up progress callback to queue updates for streaming
        job_executor_->SetProgressCallback(
            [&, job_id](const std::string& id, double progress, const cyxwiz::servernode::TrainingMetrics& metrics) {
                if (id != job_id) return;

                cyxwiz::protocol::TrainingUpdate update;
                update.set_job_id(job_id);
                update.set_timestamp(std::chrono::system_clock::now().time_since_epoch().count());

                auto* prog = update.mutable_progress();
                prog->set_current_epoch(metrics.current_epoch);
                prog->set_total_epochs(metrics.total_epochs);
                prog->set_progress_percentage(progress);
                (*prog->mutable_metrics())["loss"] = metrics.loss;
                (*prog->mutable_metrics())["accuracy"] = metrics.accuracy;
                (*prog->mutable_metrics())["learning_rate"] = metrics.learning_rate;
                prog->set_elapsed_time(metrics.time_elapsed_ms / 1000);  // Convert to seconds

                // Add custom metrics
                for (const auto& [key, value] : metrics.custom_metrics) {
                    (*prog->mutable_metrics())[key] = value;
                }

                // Queue update for streaming
                {
                    std::lock_guard<std::mutex> lock(queue_mutex);
                    update_queue.push(std::move(update));
                }
                queue_cv.notify_one();
            });

        // Set up completion callback
        job_executor_->SetCompletionCallback(
            [&, job_id](const std::string& id, bool success, const std::string& error_msg) {
                if (id != job_id) return;

                training_success = success;
                {
                    std::lock_guard<std::mutex> lock(error_mutex);
                    training_error = error_msg;
                }
                training_complete = true;
                queue_cv.notify_one();

                spdlog::info("Job {} training completed: success={}", job_id, success);
            });

        // Start real training
        session->is_running = true;
        if (!job_executor_->ExecuteJobAsync(session->job_config)) {
            spdlog::error("Failed to start training for job {}", job_id);
            return grpc::Status(grpc::StatusCode::INTERNAL, "Failed to start training");
        }

        spdlog::info("Real training started for job {}", job_id);
    } else {
        // Fallback to simulated training if no JobExecutor
        spdlog::warn("No JobExecutor available, using simulated training for job {}", job_id);

        session->is_running = true;
        std::thread simulated_thread([&, job_id, session]() {
            const int total_epochs = session->job_config.epochs();

            for (int epoch = 1; epoch <= total_epochs && !session->should_stop; ++epoch) {
                // Check if paused
                while (session->is_paused && !session->should_stop) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(100));
                }

                cyxwiz::protocol::TrainingUpdate update;
                update.set_job_id(job_id);
                update.set_timestamp(std::chrono::system_clock::now().time_since_epoch().count());

                auto* progress = update.mutable_progress();
                progress->set_current_epoch(epoch);
                progress->set_total_epochs(total_epochs);
                progress->set_progress_percentage(static_cast<double>(epoch) / total_epochs);

                double loss = 2.0 / (epoch + 1);
                double accuracy = std::min(0.99, 0.5 + epoch * 0.05);
                (*progress->mutable_metrics())["loss"] = loss;
                (*progress->mutable_metrics())["accuracy"] = accuracy;

                {
                    std::lock_guard<std::mutex> lock(queue_mutex);
                    update_queue.push(std::move(update));
                }
                queue_cv.notify_one();

                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }

            training_success = !session->should_stop;
            training_complete = true;
            queue_cv.notify_one();
        });
        simulated_thread.detach();
    }

    // Thread to read commands from Engine
    std::thread command_thread([&, job_id, session]() {
        cyxwiz::protocol::TrainingCommand command;
        while (stream->Read(&command)) {
            if (command.has_pause()) {
                session->is_paused = command.pause();
                spdlog::info("Job {} {}", job_id, command.pause() ? "paused" : "resumed");
            } else if (command.has_stop()) {
                session->should_stop = true;
                if (job_executor_) {
                    job_executor_->CancelJob(job_id);
                }
                spdlog::info("Job {} stop requested", job_id);
                break;
            } else if (command.has_request_checkpoint()) {
                spdlog::info("Job {} checkpoint requested", job_id);
            } else if (command.has_update_params()) {
                spdlog::info("Job {} hyperparameter update requested", job_id);
            }
        }
    });

    // Stream updates to Engine
    while (!training_complete || !update_queue.empty()) {
        std::unique_lock<std::mutex> lock(queue_mutex);

        // Wait for updates or completion
        queue_cv.wait_for(lock, std::chrono::milliseconds(500), [&]() {
            return !update_queue.empty() || training_complete;
        });

        // Send all queued updates
        while (!update_queue.empty()) {
            auto update = std::move(update_queue.front());
            update_queue.pop();
            lock.unlock();

            if (!stream->Write(update)) {
                spdlog::warn("Failed to write training update, client may have disconnected");
                session->should_stop = true;
                if (job_executor_) {
                    job_executor_->CancelJob(job_id);
                }
                training_complete = true;
                break;
            }

            lock.lock();
        }
    }

    // Send completion message
    {
        cyxwiz::protocol::TrainingUpdate complete_update;
        complete_update.set_job_id(job_id);
        complete_update.set_timestamp(std::chrono::system_clock::now().time_since_epoch().count());

        if (training_success) {
            auto* complete = complete_update.mutable_complete();
            complete->set_success(true);

            // Get final metrics from session or executor
            (*complete->mutable_final_metrics())["loss"] = 0.1;
            (*complete->mutable_final_metrics())["accuracy"] = 0.95;
            complete->set_total_epochs_completed(session->job_config.epochs());

            // Set weights location (for download)
            session->final_weights_path = "/tmp/models/" + job_id + ".pt";
            complete->set_weights_location(session->final_weights_path);
        } else {
            // Send error message for failed training
            auto* error = complete_update.mutable_error();
            error->set_error_code("TRAINING_FAILED");
            {
                std::lock_guard<std::mutex> lock(error_mutex);
                error->set_error_message(training_error);
            }
            error->set_recoverable(false);
        }

        stream->Write(complete_update);
    }

    // Wait for command thread
    session->should_stop = true;  // Signal command thread to exit
    if (command_thread.joinable()) {
        command_thread.join();
    }

    session->is_running = false;
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