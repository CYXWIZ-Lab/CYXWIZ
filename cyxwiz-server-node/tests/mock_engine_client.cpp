/**
 * Mock Engine Client - Simulates Engine connecting to Server Node for P2P testing
 *
 * Usage:
 *   mock_engine_client <node_address> <job_id>
 *   Example: mock_engine_client localhost:50052 test_job_001
 */

#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <chrono>
#include <grpcpp/grpcpp.h>

#include "execution.grpc.pb.h"

using namespace cyxwiz::protocol;

class MockEngineClient {
public:
    MockEngineClient(const std::string& node_address)
        : node_address_(node_address) {
        channel_ = grpc::CreateChannel(node_address,
                                       grpc::InsecureChannelCredentials());
        stub_ = JobExecutionService::NewStub(channel_);
    }

    bool ConnectToNode(const std::string& job_id,
                      const std::string& auth_token = "mock_token_123") {
        ConnectRequest request;
        request.set_job_id(job_id);
        request.set_auth_token(auth_token);
        request.set_engine_version("MockEngine/1.0.0");

        ConnectResponse response;
        grpc::ClientContext context;

        std::cout << "[>>] Connecting to node at " << node_address_ << "..." << std::endl;

        grpc::Status status = stub_->ConnectToNode(&context, request, &response);

        if (!status.ok()) {
            std::cerr << "[ERROR] Connection failed: " << status.error_message() << std::endl;
            return false;
        }

        if (response.status() != STATUS_SUCCESS) {
            std::cerr << "[ERROR] Connection rejected: " << response.error().message() << std::endl;
            return false;
        }

        std::cout << "[OK] Connected to node: " << response.node_id() << std::endl;
        std::cout << "   Capabilities:" << std::endl;
        std::cout << "   - Max Memory: " << response.capabilities().max_memory() / (1024*1024) << " MB" << std::endl;
        std::cout << "   - Max Batch Size: " << response.capabilities().max_batch_size() << std::endl;
        std::cout << "   - Devices: " << response.capabilities().supported_devices_size() << std::endl;
        std::cout << "   - Checkpointing: " << (response.capabilities().supports_checkpointing() ? "Yes" : "No") << std::endl;

        return true;
    }

    bool SendJob(const std::string& job_id, int epochs = 10, int batch_size = 32) {
        SendJobRequest request;
        request.set_job_id(job_id);

        auto* config = request.mutable_config();
        config->set_job_id(job_id);
        config->set_job_type(JOB_TYPE_TRAINING);
        config->set_priority(PRIORITY_NORMAL);
        config->set_epochs(epochs);
        config->set_batch_size(batch_size);
        config->set_model_definition(
            R"({"architecture": "MLP", "layers": [784, 256, 128, 10]})"
        );

        // Add a small mock dataset
        std::string mock_dataset(10240, 'D');  // 10KB mock data
        request.set_initial_dataset(mock_dataset);

        SendJobResponse response;
        grpc::ClientContext context;

        std::cout << "[>>] Sending job " << job_id << " (epochs=" << epochs
                  << ", batch_size=" << batch_size << ")..." << std::endl;

        grpc::Status status = stub_->SendJob(&context, request, &response);

        if (!status.ok()) {
            std::cerr << "[ERROR] SendJob failed: " << status.error_message() << std::endl;
            return false;
        }

        if (response.status() != STATUS_SUCCESS || !response.accepted()) {
            std::cerr << "[ERROR] Job rejected: " << response.rejection_reason() << std::endl;
            return false;
        }

        std::cout << "[OK] Job accepted! Estimated start: "
                  << response.estimated_start_time() << std::endl;

        return true;
    }

    void StreamTraining(const std::string& job_id, bool interactive = true) {
        grpc::ClientContext context;
        auto stream = stub_->StreamTrainingMetrics(&context);

        std::cout << "\n[**] Starting training stream for " << job_id << "..." << std::endl;
        std::cout << "   Commands: [p] pause, [r] resume, [s] stop, [c] checkpoint\n" << std::endl;

        // Thread to read updates from server
        std::thread reader_thread([&stream, &job_id]() {
            TrainingUpdate update;
            int update_count = 0;

            while (stream->Read(&update)) {
                update_count++;

                if (update.has_progress()) {
                    auto& prog = update.progress();
                    std::cout << "  [" << update_count << "] "
                              << "Epoch " << prog.current_epoch() << "/" << prog.total_epochs()
                              << " | Batch " << prog.current_batch() << "/" << prog.total_batches()
                              << " | Progress: " << (prog.progress_percentage() * 100.0) << "%"
                              << std::endl;
                    std::cout << "      Loss: " << prog.metrics().at("loss")
                              << " | Accuracy: " << prog.metrics().at("accuracy")
                              << " | GPU: " << (prog.gpu_usage() * 100.0) << "%"
                              << std::endl;
                }
                else if (update.has_checkpoint()) {
                    auto& ckpt = update.checkpoint();
                    std::cout << "  [SAVE] Checkpoint at epoch " << ckpt.epoch()
                              << " | Hash: " << ckpt.checkpoint_hash().substr(0, 8) << "..."
                              << std::endl;
                }
                else if (update.has_complete()) {
                    auto& complete = update.complete();
                    std::cout << "\n  [DONE] Training Complete!" << std::endl;
                    std::cout << "     Success: " << (complete.success() ? "Yes" : "No") << std::endl;
                    std::cout << "     Final Loss: " << complete.final_metrics().at("loss") << std::endl;
                    std::cout << "     Final Accuracy: " << complete.final_metrics().at("accuracy") << std::endl;
                    std::cout << "     Total Time: " << complete.total_training_time() << "s" << std::endl;
                    std::cout << "     Result Hash: " << complete.result_hash() << std::endl;
                    break;
                }
                else if (update.has_error()) {
                    auto& error = update.error();
                    std::cerr << "  [ERROR] Training Error: " << error.error_message() << std::endl;
                    break;
                }
                else if (update.has_log()) {
                    auto& log = update.log();
                    std::cout << "  [LOG] [" << log.source() << "] " << log.message() << std::endl;
                }
            }
        });

        // Interactive command loop (if enabled)
        if (interactive) {
            std::string command;
            while (std::getline(std::cin, command)) {
                if (command == "p") {
                    TrainingCommand cmd;
                    cmd.set_pause(true);
                    stream->Write(cmd);
                    std::cout << "[||] Pause command sent" << std::endl;
                }
                else if (command == "r") {
                    TrainingCommand cmd;
                    cmd.set_pause(false);
                    stream->Write(cmd);
                    std::cout << "[>] Resume command sent" << std::endl;
                }
                else if (command == "s") {
                    TrainingCommand cmd;
                    cmd.set_stop(true);
                    stream->Write(cmd);
                    std::cout << "[X] Stop command sent" << std::endl;
                    break;
                }
                else if (command == "c") {
                    TrainingCommand cmd;
                    cmd.set_request_checkpoint(true);
                    stream->Write(cmd);
                    std::cout << "[SAVE] Checkpoint request sent" << std::endl;
                }
                else if (command == "q") {
                    break;
                }
            }
        }

        stream->WritesDone();
        grpc::Status status = stream->Finish();

        if (reader_thread.joinable()) {
            reader_thread.join();
        }

        if (!status.ok()) {
            std::cerr << "[ERROR] Stream ended with error: " << status.error_message() << std::endl;
        }
    }

    void DownloadWeights(const std::string& job_id, const std::string& output_path) {
        DownloadRequest request;
        request.set_job_id(job_id);
        request.set_offset(0);
        request.set_chunk_size(1024 * 1024);  // 1MB chunks

        grpc::ClientContext context;
        auto reader = stub_->DownloadWeights(&context, request);

        std::cout << "\n[<<] Downloading weights for " << job_id << "..." << std::endl;

        size_t total_bytes = 0;
        int chunks_received = 0;

        WeightsChunk chunk;
        while (reader->Read(&chunk)) {
            chunks_received++;
            total_bytes += chunk.data().size();

            double progress = chunk.total_size() > 0 ?
                            (double)(chunk.offset() + chunk.data().size()) / chunk.total_size() * 100.0 :
                            0.0;

            std::cout << "  Chunk " << chunks_received
                      << " | " << (total_bytes / (1024*1024)) << " MB"
                      << " | " << progress << "%"
                      << (chunk.is_last_chunk() ? " [FINAL]" : "")
                      << std::endl;

            if (chunk.is_last_chunk()) {
                break;
            }
        }

        grpc::Status status = reader->Finish();

        if (status.ok()) {
            std::cout << "[OK] Download complete! Total: " << (total_bytes / (1024*1024))
                      << " MB in " << chunks_received << " chunks" << std::endl;
        } else {
            std::cerr << "[ERROR] Download failed: " << status.error_message() << std::endl;
        }
    }

private:
    std::string node_address_;
    std::shared_ptr<grpc::Channel> channel_;
    std::unique_ptr<JobExecutionService::Stub> stub_;
};

int main(int argc, char* argv[]) {
    std::cout << "╔═══════════════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║       Mock Engine Client - P2P Testing Tool              ║" << std::endl;
    std::cout << "╚═══════════════════════════════════════════════════════════╝" << std::endl;
    std::cout << std::endl;

    std::string node_address = argc > 1 ? argv[1] : "localhost:50052";
    std::string job_id = argc > 2 ? argv[2] : "test_job_001";

    std::cout << "Configuration:" << std::endl;
    std::cout << "  Node Address: " << node_address << std::endl;
    std::cout << "  Job ID: " << job_id << std::endl;
    std::cout << std::endl;

    try {
        MockEngineClient client(node_address);

        // Test 1: Connect
        if (!client.ConnectToNode(job_id)) {
            return 1;
        }

        std::cout << std::endl;
        std::this_thread::sleep_for(std::chrono::seconds(1));

        // Test 2: Send Job
        if (!client.SendJob(job_id, 10, 32)) {
            return 1;
        }

        std::cout << std::endl;
        std::this_thread::sleep_for(std::chrono::seconds(1));

        // Test 3: Stream Training
        client.StreamTraining(job_id, false);  // Non-interactive for automated testing

        std::cout << std::endl;
        std::this_thread::sleep_for(std::chrono::seconds(1));

        // Test 4: Download Weights
        client.DownloadWeights(job_id, "/tmp/model.pt");

        std::cout << "\n[OK] All tests completed successfully!" << std::endl;
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Exception: " << e.what() << std::endl;
        return 1;
    }
}