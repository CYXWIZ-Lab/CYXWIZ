#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_session.hpp>
#include <grpcpp/grpcpp.h>
#include <thread>
#include <chrono>

#include "../src/job_execution_service.h"
#include "execution.grpc.pb.h"

using namespace cyxwiz::server_node;
using namespace cyxwiz::protocol;

// Test fixture for JobExecutionService
class JobExecutionServiceTest {
public:
    JobExecutionServiceTest() {
        // Create service instance
        service = std::make_unique<JobExecutionServiceImpl>();

        // Initialize with mock dependencies
        service->Initialize(nullptr, "localhost:50051");

        // Start server
        REQUIRE(service->StartServer("127.0.0.1:50053"));  // Use different port for tests

        // Create client channel
        channel = grpc::CreateChannel("127.0.0.1:50053",
                                     grpc::InsecureChannelCredentials());
        stub = JobExecutionService::NewStub(channel);

        // Wait for server to be ready
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    ~JobExecutionServiceTest() {
        service->StopServer();
    }

    std::unique_ptr<JobExecutionServiceImpl> service;
    std::shared_ptr<grpc::Channel> channel;
    std::unique_ptr<JobExecutionService::Stub> stub;
};

// ========== Test Cases ==========

TEST_CASE("JobExecutionService - Server Startup", "[p2p][service]") {
    auto test_service = std::make_unique<JobExecutionServiceImpl>();
    test_service->Initialize(nullptr, "localhost:50051");

    SECTION("Server starts successfully") {
        REQUIRE(test_service->StartServer("127.0.0.1:50054"));
        test_service->StopServer();
    }

    SECTION("Server rejects duplicate start") {
        REQUIRE(test_service->StartServer("127.0.0.1:50055"));
        REQUIRE_FALSE(test_service->StartServer("127.0.0.1:50055"));
        test_service->StopServer();
    }
}

TEST_CASE("JobExecutionService - ConnectToNode", "[p2p][connect]") {
    JobExecutionServiceTest test;

    SECTION("Connect with valid auth token") {
        ConnectRequest request;
        request.set_job_id("test_job_001");
        request.set_auth_token("valid_test_token_123");
        request.set_engine_version("1.0.0");

        ConnectResponse response;
        grpc::ClientContext context;

        grpc::Status status = test.stub->ConnectToNode(&context, request, &response);

        REQUIRE(status.ok());
        REQUIRE(response.status() == STATUS_SUCCESS);
        REQUIRE_FALSE(response.node_id().empty());
        REQUIRE(response.has_capabilities());
        REQUIRE(response.capabilities().supported_devices_size() > 0);
    }

    SECTION("Connect with empty auth token fails") {
        ConnectRequest request;
        request.set_job_id("test_job_002");
        request.set_auth_token("");  // Empty token
        request.set_engine_version("1.0.0");

        ConnectResponse response;
        grpc::ClientContext context;

        grpc::Status status = test.stub->ConnectToNode(&context, request, &response);

        REQUIRE(status.ok());  // gRPC call succeeds
        REQUIRE(response.status() == STATUS_ERROR);  // But auth fails
        REQUIRE(response.has_error());
    }

    SECTION("Node capabilities are populated") {
        ConnectRequest request;
        request.set_job_id("test_job_003");
        request.set_auth_token("test_token");
        request.set_engine_version("1.0.0");

        ConnectResponse response;
        grpc::ClientContext context;

        test.stub->ConnectToNode(&context, request, &response);

        auto& caps = response.capabilities();
        REQUIRE(caps.max_memory() > 0);
        REQUIRE(caps.max_batch_size() > 0);
        REQUIRE(caps.supported_optimizers_size() > 0);
        REQUIRE(caps.supports_checkpointing());
    }
}

TEST_CASE("JobExecutionService - SendJob", "[p2p][job]") {
    JobExecutionServiceTest test;

    // First connect
    ConnectRequest conn_req;
    conn_req.set_job_id("test_job_004");
    conn_req.set_auth_token("test_token");
    conn_req.set_engine_version("1.0.0");

    ConnectResponse conn_resp;
    grpc::ClientContext conn_ctx;
    test.stub->ConnectToNode(&conn_ctx, conn_req, &conn_resp);

    SECTION("Send job with inline dataset") {
        SendJobRequest request;
        request.set_job_id("test_job_004");

        auto* config = request.mutable_config();
        config->set_job_id("test_job_004");
        config->set_job_type(JOB_TYPE_TRAINING);
        config->set_epochs(10);
        config->set_batch_size(32);
        config->set_model_definition("{\"layers\": [64, 32, 10]}");

        // Add inline dataset
        std::string dataset_data(1024, 'D');  // 1KB test data
        request.set_initial_dataset(dataset_data);

        SendJobResponse response;
        grpc::ClientContext context;

        grpc::Status status = test.stub->SendJob(&context, request, &response);

        REQUIRE(status.ok());
        REQUIRE(response.status() == STATUS_SUCCESS);
        REQUIRE(response.accepted());
        REQUIRE(response.estimated_start_time() > 0);
    }

    SECTION("Send job with dataset URI") {
        SendJobRequest request;
        request.set_job_id("test_job_005");

        auto* config = request.mutable_config();
        config->set_job_id("test_job_005");
        config->set_job_type(JOB_TYPE_TRAINING);
        config->set_epochs(5);
        config->set_batch_size(64);

        request.set_dataset_uri("ipfs://QmTest123");

        SendJobResponse response;
        grpc::ClientContext context;

        grpc::Status status = test.stub->SendJob(&context, request, &response);

        REQUIRE(status.ok());
        REQUIRE(response.status() == STATUS_SUCCESS);
        REQUIRE(response.accepted());
    }
}

TEST_CASE("JobExecutionService - StreamTrainingMetrics", "[p2p][streaming]") {
    JobExecutionServiceTest test;

    // Connect first
    ConnectRequest conn_req;
    conn_req.set_job_id("test_job_stream");
    conn_req.set_auth_token("test_token");
    conn_req.set_engine_version("1.0.0");

    ConnectResponse conn_resp;
    grpc::ClientContext conn_ctx;
    test.stub->ConnectToNode(&conn_ctx, conn_req, &conn_resp);

    // Send job
    SendJobRequest job_req;
    job_req.set_job_id("test_job_stream");
    auto* config = job_req.mutable_config();
    config->set_job_id("test_job_stream");
    config->set_job_type(JOB_TYPE_TRAINING);
    config->set_epochs(3);  // Short test
    config->set_batch_size(32);

    SendJobResponse job_resp;
    grpc::ClientContext job_ctx;
    test.stub->SendJob(&job_ctx, job_req, &job_resp);

    SECTION("Receive training progress updates") {
        grpc::ClientContext stream_ctx;
        auto stream = test.stub->StreamTrainingMetrics(&stream_ctx);

        int progress_updates = 0;
        int checkpoint_updates = 0;
        bool got_completion = false;

        // Read updates
        TrainingUpdate update;
        while (stream->Read(&update)) {
            REQUIRE(update.job_id() == "test_job_stream");
            REQUIRE(update.timestamp() > 0);

            if (update.has_progress()) {
                progress_updates++;
                auto& prog = update.progress();

                REQUIRE(prog.current_epoch() > 0);
                REQUIRE(prog.total_epochs() == 3);
                REQUIRE(prog.progress_percentage() >= 0.0);
                REQUIRE(prog.progress_percentage() <= 1.0);
                REQUIRE(prog.metrics().count("loss") > 0);
                REQUIRE(prog.metrics().count("accuracy") > 0);
                REQUIRE(prog.gpu_usage() >= 0.0);
                REQUIRE(prog.gpu_usage() <= 1.0);
            }
            else if (update.has_checkpoint()) {
                checkpoint_updates++;
                auto& ckpt = update.checkpoint();
                REQUIRE(ckpt.epoch() > 0);
                REQUIRE_FALSE(ckpt.checkpoint_hash().empty());
            }
            else if (update.has_complete()) {
                got_completion = true;
                auto& complete = update.complete();
                REQUIRE(complete.success());
                REQUIRE_FALSE(complete.result_hash().empty());
                REQUIRE(complete.total_epochs_completed() == 3);
                break;  // Training done
            }
        }

        stream->WritesDone();
        grpc::Status status = stream->Finish();

        REQUIRE(status.ok());
        REQUIRE(progress_updates > 0);
        REQUIRE(got_completion);
    }

    SECTION("Send pause command") {
        grpc::ClientContext stream_ctx;
        auto stream = test.stub->StreamTrainingMetrics(&stream_ctx);

        // Send pause command
        TrainingCommand pause_cmd;
        pause_cmd.set_pause(true);
        REQUIRE(stream->Write(pause_cmd));

        // Wait a bit
        std::this_thread::sleep_for(std::chrono::milliseconds(200));

        // Send resume command
        TrainingCommand resume_cmd;
        resume_cmd.set_pause(false);
        REQUIRE(stream->Write(resume_cmd));

        // Read some updates
        TrainingUpdate update;
        int updates_received = 0;
        while (stream->Read(&update) && updates_received < 10) {
            updates_received++;
        }

        REQUIRE(updates_received > 0);

        stream->WritesDone();
        stream->Finish();
    }

    SECTION("Send stop command") {
        grpc::ClientContext stream_ctx;
        auto stream = test.stub->StreamTrainingMetrics(&stream_ctx);

        // Read a few updates
        TrainingUpdate update;
        for (int i = 0; i < 5; i++) {
            stream->Read(&update);
        }

        // Send stop command
        TrainingCommand stop_cmd;
        stop_cmd.set_stop(true);
        REQUIRE(stream->Write(stop_cmd));

        // Training should end
        bool stream_ended = false;
        while (stream->Read(&update)) {
            // May receive a few more updates before stopping
            if (update.has_complete()) {
                stream_ended = true;
                break;
            }
        }

        stream->WritesDone();
        auto status = stream->Finish();

        // Stream should close cleanly
        REQUIRE(status.ok());
    }
}

TEST_CASE("JobExecutionService - DownloadWeights", "[p2p][download]") {
    JobExecutionServiceTest test;

    // Setup: Connect and complete a job first
    ConnectRequest conn_req;
    conn_req.set_job_id("test_job_weights");
    conn_req.set_auth_token("test_token");
    conn_req.set_engine_version("1.0.0");

    ConnectResponse conn_resp;
    grpc::ClientContext conn_ctx;
    test.stub->ConnectToNode(&conn_ctx, conn_req, &conn_resp);

    SendJobRequest job_req;
    job_req.set_job_id("test_job_weights");
    auto* config = job_req.mutable_config();
    config->set_job_id("test_job_weights");
    config->set_job_type(JOB_TYPE_TRAINING);
    config->set_epochs(1);

    SendJobResponse job_resp;
    grpc::ClientContext job_ctx;
    test.stub->SendJob(&job_ctx, job_req, &job_resp);

    SECTION("Download weights in chunks") {
        DownloadRequest request;
        request.set_job_id("test_job_weights");
        request.set_offset(0);
        request.set_chunk_size(1024 * 1024);  // 1MB chunks

        grpc::ClientContext context;
        auto reader = test.stub->DownloadWeights(&context, request);

        size_t total_bytes = 0;
        size_t chunks_received = 0;
        bool got_last_chunk = false;

        WeightsChunk chunk;
        while (reader->Read(&chunk)) {
            chunks_received++;
            total_bytes += chunk.data().size();

            REQUIRE(chunk.offset() >= 0);
            REQUIRE(chunk.total_size() > 0);
            REQUIRE(chunk.data().size() > 0);
            REQUIRE_FALSE(chunk.checksum().empty());

            if (chunk.is_last_chunk()) {
                got_last_chunk = true;
                REQUIRE(chunk.offset() + chunk.data().size() == chunk.total_size());
            }
        }

        grpc::Status status = reader->Finish();

        REQUIRE(status.ok());
        REQUIRE(chunks_received > 0);
        REQUIRE(total_bytes > 0);
        REQUIRE(got_last_chunk);
    }

    SECTION("Resume download from offset") {
        // Download first chunk
        DownloadRequest request1;
        request1.set_job_id("test_job_weights");
        request1.set_offset(0);
        request1.set_chunk_size(1024 * 1024);

        grpc::ClientContext context1;
        auto reader1 = test.stub->DownloadWeights(&context1, request1);

        WeightsChunk first_chunk;
        REQUIRE(reader1->Read(&first_chunk));
        size_t first_chunk_size = first_chunk.data().size();
        reader1->Finish();

        // Resume from offset
        DownloadRequest request2;
        request2.set_job_id("test_job_weights");
        request2.set_offset(first_chunk_size);
        request2.set_chunk_size(1024 * 1024);

        grpc::ClientContext context2;
        auto reader2 = test.stub->DownloadWeights(&context2, request2);

        WeightsChunk second_chunk;
        REQUIRE(reader2->Read(&second_chunk));
        REQUIRE(second_chunk.offset() == first_chunk_size);

        reader2->Finish();
    }
}

TEST_CASE("JobExecutionService - Multiple Concurrent Jobs", "[p2p][concurrent]") {
    JobExecutionServiceTest test;

    const int num_jobs = 3;
    std::vector<std::thread> job_threads;
    std::atomic<int> completed_jobs{0};

    for (int i = 0; i < num_jobs; i++) {
        job_threads.emplace_back([&test, i, &completed_jobs]() {
            std::string job_id = "concurrent_job_" + std::to_string(i);

            // Connect
            ConnectRequest conn_req;
            conn_req.set_job_id(job_id);
            conn_req.set_auth_token("test_token_" + std::to_string(i));
            conn_req.set_engine_version("1.0.0");

            ConnectResponse conn_resp;
            grpc::ClientContext conn_ctx;
            test.stub->ConnectToNode(&conn_ctx, conn_req, &conn_resp);

            // Send job
            SendJobRequest job_req;
            job_req.set_job_id(job_id);
            auto* config = job_req.mutable_config();
            config->set_job_id(job_id);
            config->set_job_type(JOB_TYPE_TRAINING);
            config->set_epochs(2);

            SendJobResponse job_resp;
            grpc::ClientContext job_ctx;
            auto status = test.stub->SendJob(&job_ctx, job_req, &job_resp);

            if (status.ok() && job_resp.accepted()) {
                completed_jobs++;
            }
        });
    }

    // Wait for all jobs
    for (auto& thread : job_threads) {
        thread.join();
    }

    REQUIRE(completed_jobs == num_jobs);
}

// Main function to run tests
int main(int argc, char* argv[]) {
    return Catch::Session().run(argc, argv);
}