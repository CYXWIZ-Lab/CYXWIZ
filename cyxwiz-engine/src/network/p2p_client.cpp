#include "p2p_client.h"
#include <spdlog/spdlog.h>
#include <fstream>
#include <filesystem>

namespace network {

P2PClient::P2PClient()
    : connected_(false)
    , streaming_(false)
    , node_address_("")
    , node_id_("")
    , current_job_id_("")
    , last_error_("")
    , auth_token_("")
{
}

void P2PClient::AddAuthMetadata(grpc::ClientContext& context) {
    if (!auth_token_.empty()) {
        // Add Bearer token to authorization header
        context.AddMetadata("authorization", "Bearer " + auth_token_);
        spdlog::debug("Added auth token to P2P request");
    }
}

P2PClient::~P2PClient() {
    Disconnect();
}

bool P2PClient::ConnectToNode(const std::string& node_address,
                              const std::string& job_id,
                              const std::string& auth_token,
                              const std::string& engine_version) {
    if (connected_) {
        last_error_ = "Already connected to a node";
        spdlog::warn("P2PClient: Already connected to {}", node_address_);
        return false;
    }

    spdlog::debug("P2PClient: Connecting to node at {} for job {}", node_address, job_id);

    // Create gRPC channel
    channel_ = grpc::CreateChannel(node_address, grpc::InsecureChannelCredentials());
    stub_ = cyxwiz::protocol::JobExecutionService::NewStub(channel_);

    // Store auth token for subsequent calls
    auth_token_ = auth_token;

    // Prepare connection request
    cyxwiz::protocol::ConnectRequest request;
    request.set_job_id(job_id);
    request.set_auth_token(auth_token);
    request.set_engine_version(engine_version);

    cyxwiz::protocol::ConnectResponse response;
    grpc::ClientContext context;
    AddAuthMetadata(context);

    // Call ConnectToNode RPC
    grpc::Status status = stub_->ConnectToNode(&context, request, &response);

    if (!status.ok()) {
        last_error_ = "gRPC connection failed: " + status.error_message();
        spdlog::error("P2PClient: {}", last_error_);
        return false;
    }

    if (response.status() != cyxwiz::protocol::STATUS_SUCCESS) {
        last_error_ = "Node rejected connection: " + response.error().message();
        spdlog::error("P2PClient: {}", last_error_);
        return false;
    }

    // Store connection info
    node_address_ = node_address;
    node_id_ = response.node_id();
    current_job_id_ = job_id;
    connected_ = true;

    // Extract node capabilities
    const auto& caps = response.capabilities();
    capabilities_.max_memory = caps.max_memory();
    capabilities_.max_batch_size = caps.max_batch_size();
    capabilities_.supports_checkpointing = caps.supports_checkpointing();
    capabilities_.supports_distributed = caps.supports_distributed();

    for (int i = 0; i < caps.supported_devices_size(); ++i) {
        capabilities_.supported_devices.push_back(caps.supported_devices(i));
    }

    spdlog::debug("P2PClient: Connected to node {} (max_memory={}MB, max_batch={})",
                 node_id_,
                 capabilities_.max_memory / (1024 * 1024),
                 capabilities_.max_batch_size);

    return true;
}

void P2PClient::Disconnect() {
    if (!connected_) {
        return;
    }

    spdlog::debug("P2PClient: Disconnecting from node {}", node_id_);

    // Stop any active streaming
    StopTrainingStream();

    // Reset connection state
    connected_ = false;
    node_address_.clear();
    node_id_.clear();
    current_job_id_.clear();
    stub_.reset();
    channel_.reset();

    spdlog::debug("P2PClient: Disconnected");
}

bool P2PClient::NotifyDisconnect(const std::string& reason) {
    if (!connected_ || !stub_) {
        spdlog::debug("P2PClient: Not connected, skipping disconnect notification");
        return false;
    }

    spdlog::info("P2PClient: Notifying server of disconnect (reason: {})",
                 reason.empty() ? "user_release" : reason);

    grpc::ClientContext context;
    AddAuthMetadata(context);
    context.set_deadline(std::chrono::system_clock::now() + std::chrono::seconds(5));

    cyxwiz::protocol::DisconnectRequest request;
    request.set_reason(reason.empty() ? "user_release" : reason);

    cyxwiz::protocol::DisconnectResponse response;
    grpc::Status status = stub_->DisconnectFromNode(&context, request, &response);

    if (!status.ok()) {
        spdlog::warn("P2PClient: Failed to notify server of disconnect: {}",
                     status.error_message());
        // Don't fail - we're disconnecting anyway
        return false;
    }

    spdlog::info("P2PClient: Server acknowledged disconnect: {}", response.message());
    return true;
}

bool P2PClient::SendJob(const cyxwiz::protocol::JobConfig& config,
                       const std::string& initial_dataset) {
    if (!connected_) {
        last_error_ = "Not connected to any node";
        spdlog::error("P2PClient: {}", last_error_);
        return false;
    }

    spdlog::debug("P2PClient: Sending job {} to node {}", config.job_id(), node_id_);

    cyxwiz::protocol::SendJobRequest request;
    request.set_job_id(config.job_id());
    *request.mutable_config() = config;

    if (!initial_dataset.empty()) {
        request.set_initial_dataset(initial_dataset);
        spdlog::debug("P2PClient: Sending {} bytes of inline dataset", initial_dataset.size());
    }

    cyxwiz::protocol::SendJobResponse response;
    grpc::ClientContext context;
    AddAuthMetadata(context);

    grpc::Status status = stub_->SendJob(&context, request, &response);

    if (!status.ok()) {
        last_error_ = "SendJob RPC failed: " + status.error_message();
        spdlog::error("P2PClient: {}", last_error_);
        return false;
    }

    if (response.status() != cyxwiz::protocol::STATUS_SUCCESS || !response.accepted()) {
        last_error_ = "Job rejected: " + response.rejection_reason();
        spdlog::error("P2PClient: {}", last_error_);
        return false;
    }

    spdlog::debug("P2PClient: Job {} accepted, estimated start time: {}",
                 config.job_id(),
                 response.estimated_start_time());

    return true;
}

bool P2PClient::SendJobWithDatasetURI(const cyxwiz::protocol::JobConfig& config,
                                     const std::string& dataset_uri) {
    if (!connected_) {
        last_error_ = "Not connected to any node";
        spdlog::error("P2PClient: {}", last_error_);
        return false;
    }

    spdlog::debug("P2PClient: Sending job {} with dataset URI: {}", config.job_id(), dataset_uri);

    cyxwiz::protocol::SendJobRequest request;
    request.set_job_id(config.job_id());
    *request.mutable_config() = config;
    request.set_dataset_uri(dataset_uri);

    cyxwiz::protocol::SendJobResponse response;
    grpc::ClientContext context;
    AddAuthMetadata(context);

    grpc::Status status = stub_->SendJob(&context, request, &response);

    if (!status.ok()) {
        last_error_ = "SendJob RPC failed: " + status.error_message();
        spdlog::error("P2PClient: {}", last_error_);
        return false;
    }

    if (response.status() != cyxwiz::protocol::STATUS_SUCCESS || !response.accepted()) {
        last_error_ = "Job rejected: " + response.rejection_reason();
        spdlog::error("P2PClient: {}", last_error_);
        return false;
    }

    spdlog::debug("P2PClient: Job {} accepted", config.job_id());
    return true;
}

bool P2PClient::StartTrainingStream(const std::string& job_id) {
    if (!connected_) {
        last_error_ = "Not connected to any node";
        spdlog::error("P2PClient: {}", last_error_);
        return false;
    }

    if (streaming_) {
        last_error_ = "Already streaming";
        spdlog::warn("P2PClient: {}", last_error_);
        return false;
    }

    spdlog::debug("P2PClient: Starting training stream for job {}", job_id);

    // IMPORTANT: Ensure any previous streaming thread has finished before starting a new one.
    // This handles the race condition where:
    // 1. User stops training (sends stop command)
    // 2. StreamingThreadFunc sets streaming_ = false
    // 3. User immediately starts new training
    // 4. Old thread is still in stream_->Finish() cleanup
    if (streaming_thread_.joinable()) {
        spdlog::debug("P2PClient: Waiting for previous streaming thread to finish...");
        streaming_thread_.join();
        spdlog::debug("P2PClient: Previous streaming thread finished");
    }

    // Reset stream state before starting new stream
    stream_.reset();
    stream_context_.reset();

    streaming_ = true;
    streaming_thread_done_ = false;  // Will be set to true when thread exits
    current_job_id_ = job_id;
    waiting_for_new_job_ = false;

    // Start streaming thread
    streaming_thread_ = std::thread(&P2PClient::StreamingThreadFunc, this, job_id);

    return true;
}

void P2PClient::StopTrainingStream() {
    if (!streaming_) {
        return;
    }

    spdlog::debug("P2PClient: Stopping training stream");

    // Set streaming to false first to stop any pending operations
    streaming_ = false;

    // Close the write side gracefully with WritesDone()
    {
        std::lock_guard<std::mutex> lock(stream_mutex_);
        if (stream_) {
            stream_->WritesDone();
        }
    }

    // Wait for streaming thread to finish
    if (streaming_thread_.joinable()) {
        spdlog::debug("P2PClient: Waiting for streaming thread (max 1 second)...");

        // Wait up to 1 second for graceful close
        auto start = std::chrono::steady_clock::now();
        const auto timeout = std::chrono::seconds(1);

        while (!streaming_thread_done_) {
            auto elapsed = std::chrono::steady_clock::now() - start;
            if (elapsed >= timeout) {
                spdlog::debug("P2PClient: Timeout waiting for stream, using TryCancel...");
                // Use TryCancel to unblock the Read()
                {
                    std::lock_guard<std::mutex> lock(stream_mutex_);
                    if (stream_context_) {
                        stream_context_->TryCancel();
                    }
                }
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }

        // Now join should complete
        if (streaming_thread_.joinable()) {
            streaming_thread_.join();
        }
        spdlog::debug("P2PClient: Streaming thread finished");
    }

    // Reset stream state
    {
        std::lock_guard<std::mutex> lock(stream_mutex_);
        stream_.reset();
        stream_context_.reset();
    }

    spdlog::debug("P2PClient: Training stream stopped");
}

void P2PClient::StreamingThreadFunc(const std::string& job_id) {
    spdlog::debug("P2PClient: Streaming thread started for job {}", job_id);

    // Create new context for this stream
    stream_context_ = std::make_unique<grpc::ClientContext>();
    AddAuthMetadata(*stream_context_);

    // Add job_id to metadata so Server Node can identify the job
    stream_context_->AddMetadata("x-job-id", job_id);

    // Start bidirectional stream
    stream_ = stub_->StreamTrainingMetrics(stream_context_.get());

    if (!stream_) {
        spdlog::error("P2PClient: Failed to create training stream");
        streaming_ = false;
        streaming_thread_done_ = true;
        if (error_callback_) {
            error_callback_("Failed to create training stream", true);
        }
        return;
    }

    // Read training updates from server
    spdlog::info("P2PClient: Starting to read training updates from server...");
    cyxwiz::protocol::TrainingUpdate update;
    while (streaming_ && stream_->Read(&update)) {
        if (update.has_progress()) {
            const auto& prog = update.progress();
            TrainingProgress progress;
            progress.current_epoch = prog.current_epoch();
            progress.total_epochs = prog.total_epochs();
            progress.current_batch = prog.current_batch();
            progress.total_batches = prog.total_batches();
            progress.progress_percentage = prog.progress_percentage();
            progress.gpu_usage = prog.gpu_usage();
            progress.memory_usage = prog.memory_usage();

            // Copy metrics map
            for (const auto& [key, value] : prog.metrics()) {
                progress.metrics[key] = value;
            }

            if (progress_callback_) {
                progress_callback_(progress);
            }
        }
        else if (update.has_checkpoint()) {
            const auto& ckpt = update.checkpoint();
            CheckpointInfo checkpoint;
            checkpoint.epoch = ckpt.epoch();
            checkpoint.checkpoint_hash = ckpt.checkpoint_hash();
            checkpoint.storage_uri = "";  // Not in proto, use empty string
            checkpoint.size_bytes = ckpt.weights_size();

            if (checkpoint_callback_) {
                checkpoint_callback_(checkpoint);
            }
        }
        else if (update.has_complete()) {
            const auto& comp = update.complete();
            TrainingComplete complete;
            complete.success = comp.success();
            complete.total_training_time = comp.total_training_time();
            complete.result_hash = comp.result_hash();
            complete.model_uri = comp.weights_location();  // Use weights_location instead

            // Copy final metrics
            for (const auto& [key, value] : comp.final_metrics()) {
                complete.final_metrics[key] = static_cast<float>(value);
            }

            if (completion_callback_) {
                completion_callback_(complete);
            }

            // Job complete, but reservation may still be active
            // Set waiting flag - UI can submit new job or wait for timer
            spdlog::debug("[RESERVATION] Job complete. Waiting for new job or reservation end.");
            waiting_for_new_job_ = true;

            // DON'T stop streaming - keep connection open for new jobs
            // The stream will be closed when:
            // 1. SendReservationEnd() is called (timer expires)
            // 2. Server Node closes the stream
            // 3. User explicitly disconnects

            // Continue reading from stream (Server Node may send updates)
        }
        else if (update.has_error()) {
            const auto& err = update.error();
            spdlog::error("P2PClient: Training error: {}", err.error_message());

            bool is_fatal = !err.recoverable();  // If not recoverable, it's fatal

            if (error_callback_) {
                error_callback_(err.error_message(), is_fatal);
            }

            if (is_fatal) {
                streaming_ = false;
                break;
            }
        }
        else if (update.has_log()) {
            const auto& log = update.log();
            spdlog::debug("P2PClient: [{}] {}", log.source(), log.message());

            if (log_callback_) {
                log_callback_(log.source(), log.message());
            }
        }
        // Handle dataset streaming requests from Server Node
        else if (update.has_dataset_info_request() || update.has_batch_request()) {
            HandleDatasetRequest(update);
        }
    }

    // Stream reading loop exited
    spdlog::debug("P2PClient: Stream reading loop exited, streaming_={}", streaming_.load());

    // Finish the stream (get final status)
    // Note: stream_ should still be valid here since StopTrainingStream() waits for this thread
    if (!stream_) {
        spdlog::warn("P2PClient: Stream was null when trying to finish");
        streaming_ = false;
        streaming_thread_done_ = true;
        return;
    }
    grpc::Status status = stream_->Finish();
    if (!status.ok()) {
        // CANCELLED is expected when we intentionally stop the stream
        if (status.error_code() == grpc::StatusCode::CANCELLED) {
            spdlog::debug("P2PClient: Stream was cancelled (expected during disconnect)");
        } else {
            spdlog::error("P2PClient: Stream ended with error: {}", status.error_message());
            // Only report error if streaming wasn't intentionally stopped
            if (streaming_ && error_callback_) {
                error_callback_("Stream error: " + status.error_message(), true);
            }
        }
    } else {
        spdlog::info("P2PClient: Stream finished successfully");
    }

    streaming_ = false;
    streaming_thread_done_ = true;  // Signal that thread has exited
    spdlog::debug("P2PClient: Streaming thread finished");
}

bool P2PClient::SendTrainingCommand(const cyxwiz::protocol::TrainingCommand& command) {
    // Lock mutex to ensure stream is valid during the entire write operation
    std::lock_guard<std::mutex> lock(stream_mutex_);

    if (!streaming_ || !stream_) {
        last_error_ = "Not currently streaming";
        spdlog::error("P2PClient: {}", last_error_);
        return false;
    }

    if (!stream_->Write(command)) {
        last_error_ = "Failed to send training command";
        spdlog::error("P2PClient: {}", last_error_);
        return false;
    }

    return true;
}

bool P2PClient::PauseTraining() {
    spdlog::info("P2PClient: Sending pause command");
    cyxwiz::protocol::TrainingCommand cmd;
    cmd.set_pause(true);
    return SendTrainingCommand(cmd);
}

bool P2PClient::ResumeTraining() {
    spdlog::info("P2PClient: Sending resume command");
    cyxwiz::protocol::TrainingCommand cmd;
    cmd.set_pause(false);
    return SendTrainingCommand(cmd);
}

bool P2PClient::StopTraining() {
    spdlog::info("P2PClient: Sending stop command");
    cyxwiz::protocol::TrainingCommand cmd;
    cmd.set_stop(true);
    return SendTrainingCommand(cmd);
}

bool P2PClient::RequestCheckpoint() {
    spdlog::info("P2PClient: Requesting checkpoint");
    cyxwiz::protocol::TrainingCommand cmd;
    cmd.set_request_checkpoint(true);
    return SendTrainingCommand(cmd);
}

bool P2PClient::SendNewJobConfig(const cyxwiz::protocol::JobConfig& config) {
    spdlog::debug("P2PClient: Sending new job config - epochs={}, batch_size={}",
                  config.epochs(), config.batch_size());

    cyxwiz::protocol::TrainingCommand cmd;
    *cmd.mutable_new_job_config() = config;

    bool success = SendTrainingCommand(cmd);
    if (success) {
        waiting_for_new_job_ = false;
        spdlog::debug("P2PClient: New job config sent successfully");
    } else {
        spdlog::error("P2PClient: Failed to send new job config");
    }
    return success;
}

bool P2PClient::SendReservationEnd() {
    spdlog::debug("P2PClient: Sending reservation end signal");

    cyxwiz::protocol::TrainingCommand cmd;
    cmd.set_reservation_end(true);

    bool success = SendTrainingCommand(cmd);
    if (success) {
        waiting_for_new_job_ = false;
        streaming_ = false;
        spdlog::debug("P2PClient: Reservation end sent successfully");
    } else {
        spdlog::error("P2PClient: Failed to send reservation end");
    }
    return success;
}

void P2PClient::RegisterDatasetForJob(const std::string& job_id, cyxwiz::DatasetHandle dataset) {
    dataset_provider_.RegisterDataset(job_id, dataset);
    spdlog::debug("P2PClient: Registered dataset for job {}", job_id);
}

void P2PClient::UnregisterDatasetForJob(const std::string& job_id) {
    dataset_provider_.UnregisterDataset(job_id);
    spdlog::debug("P2PClient: Unregistered dataset for job {}", job_id);
}

void P2PClient::HandleDatasetRequest(const cyxwiz::protocol::TrainingUpdate& update) {
    // Lock mutex to ensure stream is valid during write operations
    std::lock_guard<std::mutex> lock(stream_mutex_);

    if (!stream_) {
        spdlog::error("P2PClient: Cannot handle dataset request - stream not active");
        return;
    }

    cyxwiz::protocol::TrainingCommand response_cmd;

    if (update.has_dataset_info_request()) {
        const auto& request = update.dataset_info_request();
        spdlog::info("P2PClient: Received DatasetInfoRequest for job {}", request.job_id());

        auto info_response = dataset_provider_.HandleDatasetInfoRequest(request);
        *response_cmd.mutable_dataset_info_response() = info_response;

        if (!stream_->Write(response_cmd)) {
            spdlog::error("P2PClient: Failed to send DatasetInfoResponse");
        } else {
            spdlog::debug("P2PClient: Sent DatasetInfoResponse");
        }
    }
    else if (update.has_batch_request()) {
        const auto& request = update.batch_request();
        spdlog::debug("P2PClient: Received BatchRequest for job {}, {} indices",
                      request.job_id(), request.sample_indices_size());

        auto batch_response = dataset_provider_.HandleBatchRequest(request);
        *response_cmd.mutable_batch_response() = batch_response;

        if (!stream_->Write(response_cmd)) {
            spdlog::error("P2PClient: Failed to send BatchResponse");
        } else {
            spdlog::debug("P2PClient: Sent BatchResponse ({} bytes)",
                          batch_response.images().size());
        }
    }
}

bool P2PClient::DownloadWeights(const std::string& job_id,
                                const std::string& output_path,
                                size_t chunk_size) {
    return DownloadWeightsWithOffset(job_id, output_path, 0, chunk_size);
}

bool P2PClient::DownloadWeightsWithOffset(const std::string& job_id,
                                         const std::string& output_path,
                                         size_t offset,
                                         size_t chunk_size) {
    if (!connected_) {
        last_error_ = "Not connected to any node";
        spdlog::error("P2PClient: {}", last_error_);
        return false;
    }

    spdlog::info("P2PClient: Downloading weights for job {} to {}", job_id, output_path);

    // Prepare download request
    cyxwiz::protocol::DownloadRequest request;
    request.set_job_id(job_id);
    request.set_offset(offset);
    request.set_chunk_size(chunk_size);

    grpc::ClientContext context;
    AddAuthMetadata(context);
    auto reader = stub_->DownloadWeights(&context, request);

    // Open output file
    std::ofstream output_file(output_path, std::ios::binary | std::ios::trunc);
    if (!output_file.is_open()) {
        last_error_ = "Failed to open output file: " + output_path;
        spdlog::error("P2PClient: {}", last_error_);
        return false;
    }

    // Read chunks and write to file
    size_t total_bytes = 0;
    int chunks_received = 0;
    cyxwiz::protocol::WeightsChunk chunk;

    while (reader->Read(&chunk)) {
        chunks_received++;
        total_bytes += chunk.data().size();

        // Write chunk data to file
        output_file.write(chunk.data().data(), chunk.data().size());

        if (chunk.total_size() > 0) {
            double progress = (double)(chunk.offset() + chunk.data().size()) / chunk.total_size() * 100.0;
            spdlog::debug("P2PClient: Downloaded chunk {} ({:.1f}%)", chunks_received, progress);
        }

        if (chunk.is_last_chunk()) {
            spdlog::info("P2PClient: Received final chunk");
            break;
        }
    }

    output_file.close();

    // Check stream status
    grpc::Status status = reader->Finish();
    if (!status.ok()) {
        last_error_ = "Download failed: " + status.error_message();
        spdlog::error("P2PClient: {}", last_error_);
        return false;
    }

    spdlog::info("P2PClient: Download complete - {} MB in {} chunks",
                 total_bytes / (1024 * 1024),
                 chunks_received);

    return true;
}

} // namespace network
