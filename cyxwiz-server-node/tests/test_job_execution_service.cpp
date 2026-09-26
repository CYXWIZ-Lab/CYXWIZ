#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_session.hpp>
#include <grpcpp/grpcpp.h>
#include <thread>
#include <chrono>
#include <jwt-cpp/jwt.h>
#include <jwt-cpp/traits/nlohmann-json/traits.h>

#include "../src/job_execution_service.h"
#include "../src/job_executor.h"
#include "../src/remote_dataset_fetcher.h"
#include "network/dataset_file_server.h"
#include "core/arrow_dataset.h"
#include "core/graph_compiler_dataset_hooks.h"
#include "../../cyxwiz-engine/tests/causal_lm_token_window_fixture.h"
#include "../../cyxwiz-engine/tests/route_qualification_test_fixture.h"
#include "../src/node_client.h"
#include "../src/node_doctor.h"
#include "../src/node_job_timing.h"
#include "core/compute_runtime_paths.h"
#include "core/training_benchmark.h"
#include "execution.grpc.pb.h"
#include "job.grpc.pb.h"
#include "reservation.grpc.pb.h"
#include "node.grpc.pb.h"
#include <catch2/catch_approx.hpp>

using namespace cyxwiz::server_node;
using namespace cyxwiz::protocol;

// Test constants
static const std::string TEST_SECRET = "test_p2p_secret";
static const std::string TEST_NODE_ID = "test_node";
static constexpr int TEST_RPC_DEADLINE_SECONDS = 12;

void SetRpcDeadline(grpc::ClientContext& context,
                    int seconds = TEST_RPC_DEADLINE_SECONDS) {
    context.set_deadline(
        std::chrono::system_clock::now() + std::chrono::seconds(seconds));
}

void SetStreamJobId(grpc::ClientContext& context,
                    const std::string& job_id) {
    context.AddMetadata("x-job-id", job_id);
}

bool SendReservationEnd(
    grpc::ClientReaderWriter<TrainingCommand, TrainingUpdate>* stream) {
    TrainingCommand end_cmd;
    end_cmd.set_reservation_end(true);
    const bool wrote = stream->Write(end_cmd);
    stream->WritesDone();
    return wrote;
}

// Helper function to generate valid JWT tokens for testing
std::string GenerateTestJwt(const std::string& job_id,
                            const std::string& node_id = TEST_NODE_ID,
                            int expires_in_seconds = 3600) {
    using jwt_traits = jwt::traits::nlohmann_json;

    auto now = std::chrono::system_clock::now();
    auto exp = now + std::chrono::seconds(expires_in_seconds);

    auto token = jwt::create<jwt_traits>()
        .set_issuer("CyxWiz-Central-Server")
        .set_subject("test_user")
        .set_issued_at(now)
        .set_expires_at(exp)
        .set_payload_claim("job_id", job_id)
        .set_payload_claim("node_id", node_id)
        .sign(jwt::algorithm::hs256{TEST_SECRET});

    return token;
}

// Test fixture for JobExecutionService
class JobExecutionServiceTest {
public:
    JobExecutionServiceTest() {
        // Create service instance
        service = std::make_unique<JobExecutionServiceImpl>();
        executor = std::make_shared<cyxwiz::servernode::JobExecutor>("test_node");

        // Jobs train with the shared core; Central Server is not running.
        service->Initialize(executor, "localhost:50051", "test_node", "test_p2p_secret");

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

    std::shared_ptr<cyxwiz::servernode::JobExecutor> executor;
    std::unique_ptr<JobExecutionServiceImpl> service;
    std::shared_ptr<grpc::Channel> channel;
    std::unique_ptr<JobExecutionService::Stub> stub;
};

// A remote:// job whose graph and dataset live on the "Engine" (this test):
// the causal-LM token-window graph, its Parquet registered in the Engine's
// dataset catalog and file server.
struct RemoteGraphJob {
    std::filesystem::path work;
    std::shared_ptr<cyxwiz::ArrowDataset> dataset;
    network::DatasetFileServer files;
};

void PrepareRemoteGraphJob(RemoteGraphJob& job, JobConfig& config, int epochs) {
    namespace fs = std::filesystem;
    const fs::path root = CYXWIZ_SOURCE_ROOT;
    job.work = fs::temp_directory_path() / ("cyxwiz_p2p_" + config.job_id());
    std::error_code ec;
    fs::remove_all(job.work, ec);
    fs::create_directories(job.work);
    const fs::path parquet = job.work / "tokens.parquet";
    REQUIRE(cyxwiz::test::WriteTokenWindows(root, parquet, 0, 8));
    cyxwiz::test::InstallQualifiedRouteSnapshot();
    job.dataset = cyxwiz::ArrowDataset::FromParquet(parquet.string(), "tiny_causal_lm_tokens");
    cyxwiz::GraphDatasetCatalog catalog;
    catalog.arrow_dataset = [dataset = job.dataset](const std::string& name) -> std::shared_ptr<cyxwiz::ArrowDataset> {
        return name == "tiny_causal_lm_tokens" ? dataset : nullptr;
    };
    cyxwiz::SetGraphDatasetCatalog(catalog);

    auto graph = cyxwiz::test::LoadTokenWindowGraph(root);
    for (auto& node : graph["nodes"]) {
        if (node.value("type", -1) == static_cast<int>(gui::NodeType::DataLoader)) {
            node["parameters"]["checkpoint_dir"] = (job.work / "checkpoints").string();
        }
    }
    config.set_model_definition(graph.dump());
    config.set_dataset_uri("remote://engine");
    config.set_epochs(epochs);
    std::string error;
    REQUIRE(job.files.RegisterJob(config.job_id(), config.model_definition(), job.work / "engine_cache", error));
}

// Plays the Engine on the training stream: serves the node's dataset file
// requests, ends the reservation once the job has finished, and returns every
// update the node sent.
std::vector<TrainingUpdate> RunAsEngine(JobExecutionServiceTest& test, const std::string& job_id,
                                        RemoteGraphJob& job, grpc::Status& status) {
    grpc::ClientContext stream_ctx;
    SetRpcDeadline(stream_ctx, 60);
    SetStreamJobId(stream_ctx, job_id);
    auto stream = test.stub->StreamTrainingMetrics(&stream_ctx);
    std::vector<TrainingUpdate> updates;
    bool ended = false;
    TrainingUpdate update;
    while (stream->Read(&update)) {
        updates.push_back(update);
        if (update.has_dataset_file_request()) {
            job.files.HandleRequest(update.dataset_file_request(), [&stream](const DatasetFileChunk& chunk) {
                TrainingCommand command;
                *command.mutable_dataset_file_chunk() = chunk;
                return stream->Write(command);
            });
        }
        if (!ended && (update.has_complete() || update.has_error())) ended = SendReservationEnd(stream.get());
    }
    status = stream->Finish();
    return updates;
}

// ========== Test Cases ==========

TEST_CASE("JobExecutionService - Server Startup", "[p2p][service]") {
    auto test_service = std::make_unique<JobExecutionServiceImpl>();
    test_service->Initialize(nullptr, "localhost:50051", "test_node_startup", "test_p2p_secret");

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
        request.set_auth_token(GenerateTestJwt("test_job_001"));
        request.set_engine_version("1.0.0");

        ConnectResponse response;
        grpc::ClientContext context;
        SetRpcDeadline(context);

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
        SetRpcDeadline(context);

        grpc::Status status = test.stub->ConnectToNode(&context, request, &response);

        REQUIRE(status.ok());  // gRPC call succeeds
        REQUIRE(response.status() == STATUS_ERROR);  // But auth fails
        REQUIRE(response.has_error());
    }

    SECTION("Node capabilities are populated") {
        ConnectRequest request;
        request.set_job_id("test_job_003");
        request.set_auth_token(GenerateTestJwt("test_job_003"));
        request.set_engine_version("1.0.0");

        ConnectResponse response;
        grpc::ClientContext context;
        SetRpcDeadline(context);

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
    conn_req.set_auth_token(GenerateTestJwt("test_job_004"));
    conn_req.set_engine_version("1.0.0");

    ConnectResponse conn_resp;
    grpc::ClientContext conn_ctx;
    SetRpcDeadline(conn_ctx);
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
        SetRpcDeadline(context);

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
        SetRpcDeadline(context);

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
    conn_req.set_auth_token(GenerateTestJwt("test_job_stream"));
    conn_req.set_engine_version("1.0.0");

    ConnectResponse conn_resp;
    grpc::ClientContext conn_ctx;
    SetRpcDeadline(conn_ctx);
    test.stub->ConnectToNode(&conn_ctx, conn_req, &conn_resp);

    // Send job
    SendJobRequest job_req;
    job_req.set_job_id("test_job_stream");
    auto* config = job_req.mutable_config();
    config->set_job_id("test_job_stream");
    config->set_job_type(JOB_TYPE_TRAINING);
    config->set_batch_size(32);
    RemoteGraphJob remote;
    PrepareRemoteGraphJob(remote, *config, 3);  // short run

    SendJobResponse job_resp;
    grpc::ClientContext job_ctx;
    SetRpcDeadline(job_ctx);
    grpc::Status job_status = test.stub->SendJob(&job_ctx, job_req, &job_resp);

    REQUIRE(job_status.ok());
    REQUIRE(job_resp.accepted());

    SECTION("Receive training progress updates") {
        grpc::Status status;
        const auto updates = RunAsEngine(test, "test_job_stream", remote, status);

        int progress_updates = 0;
        int checkpoint_updates = 0;
        bool got_completion = false;
        bool updates_are_valid = true;
        bool fetched_dataset = false;

        for (const auto& update : updates) {
            fetched_dataset = fetched_dataset || update.has_dataset_file_request();
            updates_are_valid =
                updates_are_valid &&
                update.job_id() == "test_job_stream" &&
                update.timestamp() > 0;

            if (update.has_progress()) {
                progress_updates++;
                auto& prog = update.progress();

                updates_are_valid =
                    updates_are_valid &&
                    prog.current_epoch() > 0 &&
                    prog.total_epochs() == 3 &&
                    prog.progress_percentage() >= 0.0 &&
                    prog.progress_percentage() <= 1.0 &&
                    prog.metrics().count("loss") > 0 &&
                    prog.metrics().count("accuracy") > 0 &&
                    prog.gpu_usage() >= 0.0 &&
                    prog.gpu_usage() <= 1.0;
            }
            else if (update.has_checkpoint()) {
                checkpoint_updates++;
                auto& ckpt = update.checkpoint();
                updates_are_valid =
                    updates_are_valid &&
                    ckpt.epoch() > 0 &&
                    !ckpt.checkpoint_hash().empty();
            }
            else if (update.has_complete()) {
                got_completion = true;
                auto& complete = update.complete();
                updates_are_valid =
                    updates_are_valid &&
                    complete.success() &&
                    !complete.weights_location().empty() &&
                    complete.total_epochs_completed() == 3;
                // Where the time went and what ran it (TOFIX118 P3 S4).
                const auto& timing = complete.timing();
                INFO("transfer " << timing.transfer_seconds() << " prepare " << timing.prepare_seconds()
                                 << " train " << timing.train_seconds() << " wall " << timing.wall_seconds());
                CHECK(timing.transfer_seconds() > 0.0);
                CHECK(timing.prepare_seconds() > 0.0);
                CHECK(timing.train_seconds() > 0.0);
                CHECK(timing.wall_seconds() >= timing.transfer_seconds() + timing.prepare_seconds() +
                                                   timing.train_seconds() + timing.save_seconds() - 1e-9);
                CHECK(timing.goodput() > 0.0);
                CHECK(timing.goodput() <= 1.0);
                CHECK(timing.samples_trained() > 0);
                CHECK(timing.tokens_trained() > 0);
                CHECK(timing.tokens_per_second() > 0.0);
                CHECK(complete.environment().fingerprint().size() == 64);
            }
        }

        REQUIRE(status.ok());
        REQUIRE(fetched_dataset);
        REQUIRE(updates_are_valid);
        REQUIRE(progress_updates > 0);
        REQUIRE(got_completion);
    }

    SECTION("Send pause command") {
        grpc::ClientContext stream_ctx;
        SetRpcDeadline(stream_ctx);
        SetStreamJobId(stream_ctx, "test_job_stream");
        auto stream = test.stub->StreamTrainingMetrics(&stream_ctx);

        // Send pause command
        TrainingCommand pause_cmd;
        pause_cmd.set_pause(true);
        bool pause_written = stream->Write(pause_cmd);

        // Wait a bit
        std::this_thread::sleep_for(std::chrono::milliseconds(200));

        // Send resume command
        TrainingCommand resume_cmd;
        resume_cmd.set_pause(false);
        bool resume_written = stream->Write(resume_cmd);
        bool reservation_end_written = SendReservationEnd(stream.get());

        // Read some updates
        TrainingUpdate update;
        int updates_received = 0;
        while (stream->Read(&update)) {
            updates_received++;
        }

        grpc::Status status = stream->Finish();

        REQUIRE(pause_written);
        REQUIRE(resume_written);
        REQUIRE(reservation_end_written);
        REQUIRE(status.ok());
        REQUIRE(updates_received > 0);
    }

    SECTION("Send stop command") {
        grpc::ClientContext stream_ctx;
        SetRpcDeadline(stream_ctx);
        SetStreamJobId(stream_ctx, "test_job_stream");
        auto stream = test.stub->StreamTrainingMetrics(&stream_ctx);

        // Send stop command
        TrainingCommand stop_cmd;
        stop_cmd.set_stop(true);
        bool stop_written = stream->Write(stop_cmd);
        bool reservation_end_written = SendReservationEnd(stream.get());

        // Training should end
        TrainingUpdate update;
        bool stream_ended = false;
        while (stream->Read(&update)) {
            // May receive a few more updates before stopping
            if (update.has_complete()) {
                stream_ended = true;
            }
        }

        auto status = stream->Finish();

        // Stream should close cleanly
        REQUIRE(stop_written);
        REQUIRE(reservation_end_written);
        REQUIRE(status.ok());
        REQUIRE(stream_ended);
    }
}

TEST_CASE("JobExecutionService - DownloadWeights", "[p2p][download]") {
    JobExecutionServiceTest test;

    // Setup: Connect and complete a job first
    ConnectRequest conn_req;
    conn_req.set_job_id("test_job_weights");
    conn_req.set_auth_token(GenerateTestJwt("test_job_weights"));
    conn_req.set_engine_version("1.0.0");

    ConnectResponse conn_resp;
    grpc::ClientContext conn_ctx;
    SetRpcDeadline(conn_ctx);
    test.stub->ConnectToNode(&conn_ctx, conn_req, &conn_resp);

    SendJobRequest job_req;
    job_req.set_job_id("test_job_weights");
    auto* config = job_req.mutable_config();
    config->set_job_id("test_job_weights");
    config->set_job_type(JOB_TYPE_TRAINING);
    RemoteGraphJob remote;
    PrepareRemoteGraphJob(remote, *config, 1);

    SendJobResponse job_resp;
    grpc::ClientContext job_ctx;
    SetRpcDeadline(job_ctx);
    grpc::Status job_status = test.stub->SendJob(&job_ctx, job_req, &job_resp);

    REQUIRE(job_status.ok());
    REQUIRE(job_resp.accepted());
    grpc::Status stream_status;
    const auto updates = RunAsEngine(test, "test_job_weights", remote, stream_status);
    REQUIRE(stream_status.ok());
    bool trained = false;
    for (const auto& update : updates) {
        trained = trained || (update.has_complete() && update.complete().success() &&
                              !update.complete().weights_location().empty());
    }
    REQUIRE(trained);

    SECTION("Download weights in chunks") {
        DownloadRequest request;
        request.set_job_id("test_job_weights");
        request.set_offset(0);
        request.set_chunk_size(1024 * 1024);  // 1MB chunks

        grpc::ClientContext context;
        SetRpcDeadline(context);
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
        request1.set_chunk_size(1024);  // small: the test model is a few KB

        grpc::ClientContext context1;
        SetRpcDeadline(context1);
        auto reader1 = test.stub->DownloadWeights(&context1, request1);

        WeightsChunk first_chunk;
        REQUIRE(reader1->Read(&first_chunk));
        size_t first_chunk_size = first_chunk.data().size();

        WeightsChunk drain_chunk;
        while (reader1->Read(&drain_chunk)) {
        }
        REQUIRE(reader1->Finish().ok());

        // Resume from offset
        DownloadRequest request2;
        request2.set_job_id("test_job_weights");
        request2.set_offset(first_chunk_size);
        request2.set_chunk_size(1024);  // small: the test model is a few KB

        grpc::ClientContext context2;
        SetRpcDeadline(context2);
        auto reader2 = test.stub->DownloadWeights(&context2, request2);

        WeightsChunk second_chunk;
        REQUIRE(reader2->Read(&second_chunk));
        REQUIRE(second_chunk.offset() == first_chunk_size);

        while (reader2->Read(&drain_chunk)) {
        }
        REQUIRE(reader2->Finish().ok());
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
            conn_req.set_auth_token(GenerateTestJwt(job_id));
            conn_req.set_engine_version("1.0.0");

            ConnectResponse conn_resp;
            grpc::ClientContext conn_ctx;
            SetRpcDeadline(conn_ctx);
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
            SetRpcDeadline(job_ctx);
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

// ===== JobExecutor: local jobs train through the shared core (TOFIX118 P2) =====

namespace {

struct JobOutcome {
    std::mutex mutex;
    std::condition_variable done;
    bool finished = false;
    bool success = false;
    std::string error;
    int epochs_reported = 0;
};

// Runs one job on a fresh executor and waits for its completion callback.
void RunLocalJob(const JobConfig& config, JobOutcome& outcome) {
    cyxwiz::servernode::JobExecutor executor("test_node");
    executor.SetProgressCallback([&outcome](const std::string&, double, const cyxwiz::servernode::TrainingMetrics&) {
        std::lock_guard<std::mutex> lock(outcome.mutex);
        ++outcome.epochs_reported;
    });
    executor.SetCompletionCallback([&outcome](const std::string&, bool success, const std::string& error) {
        std::lock_guard<std::mutex> lock(outcome.mutex);
        outcome.finished = true;
        outcome.success = success;
        outcome.error = error;
        outcome.done.notify_all();
    });
    REQUIRE(executor.ExecuteJobAsync(config));
    {
        std::unique_lock<std::mutex> lock(outcome.mutex);
        REQUIRE(outcome.done.wait_for(lock, std::chrono::minutes(3), [&outcome] { return outcome.finished; }));
    }
    // The worker drops its job state after the callback; wait for that before
    // the executor goes away.
    for (int i = 0; i < 500 && executor.GetActiveJobCount() > 0; ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    REQUIRE(executor.GetActiveJobCount() == 0);
}

}  // namespace

TEST_CASE("JobExecutor - trains a graph job through the shared core", "[job_executor][training]") {
    namespace fs = std::filesystem;
    const fs::path root = CYXWIZ_SOURCE_ROOT;
    const fs::path work = fs::temp_directory_path() / "cyxwiz_node_job_executor_test";
    std::error_code ec;
    fs::remove_all(work, ec);
    fs::create_directories(work);
    const fs::path parquet = work / "tokens.parquet";
    REQUIRE(cyxwiz::test::WriteTokenWindows(root, parquet, 0, 8));
    cyxwiz::test::InstallQualifiedRouteSnapshot();

    JobConfig config;
    config.set_job_id("node_graph_job");
    config.set_job_type(JOB_TYPE_TRAINING);
    auto graph = cyxwiz::test::LoadTokenWindowGraph(root);
    for (auto& node : graph["nodes"]) {
        if (node.value("type", -1) == static_cast<int>(gui::NodeType::DataLoader)) {
            node["parameters"]["checkpoint_dir"] = (work / "checkpoints").string();
        }
    }
    config.set_model_definition(graph.dump());
    config.set_dataset_uri("file://" + parquet.generic_string());
    config.set_epochs(2);

    JobOutcome outcome;
    RunLocalJob(config, outcome);
    INFO(outcome.error);
    CHECK(outcome.success);
    CHECK(outcome.epochs_reported == 2);
    fs::remove_all(work, ec);
}

TEST_CASE("JobExecutor - refuses jobs the shared core cannot train", "[job_executor][refusal]") {
    const std::string graph =
        cyxwiz::test::LoadTokenWindowGraph(std::filesystem::path(CYXWIZ_SOURCE_ROOT)).dump();
    struct Case {
        const char* name;
        std::string model_definition;
        std::string dataset_uri;
        const char* reason;
    };
    const std::vector<Case> cases = {
        {"no graph", "", "", "no model definition"},
        {"retired MNIST loader", graph, "file://mnist/./data/mnist", "retired loader"},
        {"mock data", graph, "mock://random", "mock datasets are not trained"},
        {"unknown scheme", graph, "ipfs://QmTest123", "unsupported dataset_uri scheme"},
    };
    for (const auto& c : cases) {
        SECTION(c.name) {
            JobConfig config;
            config.set_job_id(std::string("refused_") + c.name);
            config.set_job_type(JOB_TYPE_TRAINING);
            config.set_model_definition(c.model_definition);
            config.set_dataset_uri(c.dataset_uri);
            JobOutcome outcome;
            RunLocalJob(config, outcome);
            INFO(outcome.error);
            CHECK_FALSE(outcome.success);
            CHECK(outcome.error.find(c.reason) != std::string::npos);
            CHECK(outcome.epochs_reported == 0);
        }
    }
}

TEST_CASE("Remote jobs fetch the Engine's dataset files, then train", "[remote_dataset]") {
    namespace fs = std::filesystem;
    const fs::path root = CYXWIZ_SOURCE_ROOT;
    const fs::path work = fs::temp_directory_path() / "cyxwiz_node_remote_dataset_test";
    std::error_code ec;
    fs::remove_all(work, ec);
    fs::create_directories(work);
    const fs::path parquet = work / "tokens.parquet";
    REQUIRE(cyxwiz::test::WriteTokenWindows(root, parquet, 0, 8));
    cyxwiz::test::InstallQualifiedRouteSnapshot();

    // The Engine: its registry (as the graph dataset catalog) and file server.
    const std::string dataset = "tiny_causal_lm_tokens";
    auto engine_dataset = cyxwiz::ArrowDataset::FromParquet(parquet.string(), dataset);
    cyxwiz::GraphDatasetCatalog catalog;
    catalog.arrow_dataset = [&](const std::string& name) -> std::shared_ptr<cyxwiz::ArrowDataset> {
        return name == dataset ? engine_dataset : nullptr;
    };
    cyxwiz::SetGraphDatasetCatalog(catalog);
    auto graph = cyxwiz::test::LoadTokenWindowGraph(root);
    for (auto& node : graph["nodes"]) {
        if (node.value("type", -1) == static_cast<int>(gui::NodeType::DataLoader)) {
            node["parameters"]["checkpoint_dir"] = (work / "checkpoints").string();
        }
    }
    network::DatasetFileServer engine;
    std::string error;
    REQUIRE(engine.RegisterJob("remote_job", graph.dump(), work / "engine_cache", error));

    // The stream: node requests reach the Engine, its chunks come back. The
    // first request stalls after one chunk, so the fetcher must resume.
    std::shared_ptr<RemoteDatasetFetcher> fetcher;
    std::vector<std::thread> engine_replies;
    std::atomic<int> requests{0};
    std::atomic<int64_t> resumed_from{-1};
    auto write = [&](const TrainingUpdate& update) {
        if (!update.has_dataset_file_request()) return true;
        const auto request = update.dataset_file_request();
        const int number = ++requests;
        if (number > 1 && resumed_from < 0) resumed_from = request.offset();
        engine_replies.emplace_back([&, request, number] {
            int sent = 0;
            engine.HandleRequest(
                request,
                [&](const DatasetFileChunk& chunk) {
                    if (number == 1 && sent++ > 0) return true;  // stalled connection
                    fetcher->OnChunk(chunk);
                    return true;
                },
                256);
        });
        return true;
    };
    fetcher = std::make_shared<RemoteDatasetFetcher>(write, "remote_job", work / "node_cache",
                                                     /*chunk_timeout_ms=*/300, /*max_resumes=*/3);

    JobConfig config;
    config.set_job_id("remote_job");
    config.set_job_type(JOB_TYPE_TRAINING);
    config.set_model_definition(graph.dump());
    config.set_dataset_uri("remote://engine");
    config.set_epochs(1);
    const bool fetched = FetchRemoteJobDatasets(*fetcher, config, error);
    INFO(error);
    REQUIRE(fetched);
    CHECK(requests >= 2);
    CHECK(resumed_from == 256);
    CHECK(config.dataset_uri().empty());
    CHECK(config.model_definition().find("node_cache") != std::string::npos);

    // An Engine refusal reaches the node with its reason.
    {
        fs::path unused;
        std::string refusal;
        CHECK_FALSE(fetcher->Fetch("not_in_the_graph", unused, refusal));
        CHECK(refusal.find("not an input") != std::string::npos);
    }

    for (auto& reply : engine_replies) reply.join();
    engine_replies.clear();
    cyxwiz::SetGraphDatasetCatalog({});  // the node has no Engine registry

    JobOutcome outcome;
    RunLocalJob(config, outcome);
    INFO(outcome.error);
    CHECK(outcome.success);
    CHECK(outcome.epochs_reported == 1);
    fs::remove_all(work, ec);
}

TEST_CASE("Registration reports the measured training capability", "[capability]") {
    namespace fs = std::filesystem;
    const fs::path root = fs::temp_directory_path() / "cyxwiz_node_capability_test";
    std::error_code ec;
    fs::remove_all(root, ec);
    cyxwiz::ScopedComputeRuntimeRootOverrideForTesting runtime_root(root);

    cyxwiz::RouteQualificationSnapshot snapshot;
    snapshot.matrix_id = "node-capability-test";
    cyxwiz::RouteQualificationRecord gpu;
    gpu.type = cyxwiz::DeviceType::OPENCL;
    gpu.device_id = 0;
    gpu.physical_fingerprint = "uuid:test-gpu";
    gpu.display_name = "Test GPU";
    gpu.driver_version = "32.0";
    gpu.operation_count = gpu.pass_count = 23;
    gpu.certified = true;
    cyxwiz::RouteQualificationRecord cpu = gpu;
    cpu.type = cyxwiz::DeviceType::CPU;
    cpu.physical_fingerprint.clear();
    cpu.display_name = "Test CPU";
    snapshot.routes = {gpu, cpu};
    cyxwiz::InstallRouteQualificationSnapshot(snapshot);

    cyxwiz::TrainingBenchmarkResult measured;
    measured.ok = true;
    measured.backend = "arrayfire_opencl";
    measured.device_id = 0;
    measured.physical_fingerprint = "uuid:test-gpu";
    measured.build = cyxwiz::GetVersionString();
    measured.tokens_per_second = 6585.0;
    measured.step_ms_median = 311.0;
    std::string error;
    REQUIRE(cyxwiz::SaveTrainingBenchmarkResults(cyxwiz::GetTrainingBenchmarkCachePath(), {measured}, error));

    const auto info = cyxwiz::servernode::HardwareDetector::DetectHardwareInfo("node-capability-test");
    // What the central server receives: serialize and parse back.
    std::string wire;
    REQUIRE(info.SerializeToString(&wire));
    cyxwiz::protocol::NodeInfo received;
    REQUIRE(received.ParseFromString(wire));

    CHECK(received.compute_score() == 6585.0);
    REQUIRE(received.routes_size() == 2);
    CHECK(received.routes(0).backend() == "arrayfire_opencl");
    CHECK(received.routes(0).device_name() == "Test GPU");
    CHECK(received.routes(0).driver_version() == "32.0");
    CHECK(received.routes(0).certified());
    CHECK(received.routes(0).operations_passed() == 23);
    REQUIRE(received.routes(0).has_benchmark());
    CHECK(received.routes(0).benchmark().ok());
    CHECK(received.routes(0).benchmark().current());
    CHECK(received.routes(0).benchmark().tokens_per_second() == 6585.0);
    CHECK_FALSE(received.routes(1).has_benchmark());
    CHECK(received.environment().fingerprint().size() == 64);
    CHECK(received.environment().route_matrix_id() == "node-capability-test");
    CHECK(received.environment().cyxwiz_build() == cyxwiz::GetVersionString());

    cyxwiz::ClearRouteQualificationSnapshot();
    fs::remove_all(root, ec);
}

TEST_CASE("Doctor judges whether the node can take training jobs", "[doctor]") {
    using cyxwiz::servernode::DoctorStatus;
    const auto status_of = [](const std::vector<cyxwiz::servernode::DoctorCheck>& checks, const std::string& name) {
        for (const auto& check : checks) {
            if (check.name == name) return check.status;
        }
        FAIL("no check named " << name);
        return DoctorStatus::Fail;
    };

    cyxwiz::RouteQualificationSnapshot snapshot;
    cyxwiz::RouteQualificationRecord gpu;
    gpu.type = cyxwiz::DeviceType::OPENCL;
    gpu.physical_fingerprint = "uuid:gpu";
    gpu.certified = true;
    snapshot.routes = {gpu};
    cyxwiz::TrainingBenchmarkResult measured;
    measured.ok = true;
    measured.backend = "arrayfire_opencl";
    measured.physical_fingerprint = "uuid:gpu";
    measured.build = "1.0.0";
    measured.tokens_per_second = 6000.0;

    cyxwiz::servernode::DoctorFacts ready;
    ready.build = "1.0.0";
    ready.route_evidence_loaded = true;
    ready.capability = cyxwiz::BuildMachineCapability(snapshot, {measured}, "1.0.0");
    ready.preference_file_loaded = true;
    ready.preferred_route = std::make_pair(std::string("arrayfire_opencl"), 0);
    ready.data_dir = "D:/data";
    ready.data_dir_writable = true;
    ready.data_dir_free_bytes = 100ull << 30;
    ready.central_server = "localhost:50051";
    ready.central_server_reachable = true;

    auto checks = cyxwiz::servernode::EvaluateNodeReadiness(ready);
    CHECK(cyxwiz::servernode::NodeIsReady(checks));
    for (const auto& check : checks) {
        INFO(check.name << ": " << check.detail);
        CHECK(check.status == DoctorStatus::Ok);
    }

    SECTION("no route evidence refuses jobs") {
        auto facts = ready;
        facts.route_evidence_loaded = false;
        facts.capability = cyxwiz::BuildMachineCapability({}, {}, "1.0.0");
        const auto result = cyxwiz::servernode::EvaluateNodeReadiness(facts);
        CHECK(status_of(result, "Verified routes") == DoctorStatus::Fail);
        CHECK_FALSE(cyxwiz::servernode::NodeIsReady(result));
    }
    SECTION("a benchmark from another build is a warning, not a failure") {
        auto facts = ready;
        facts.capability = cyxwiz::BuildMachineCapability(snapshot, {measured}, "1.0.1");
        const auto result = cyxwiz::servernode::EvaluateNodeReadiness(facts);
        CHECK(status_of(result, "Training benchmark") == DoctorStatus::Warn);
        CHECK(cyxwiz::servernode::NodeIsReady(result));
    }
    SECTION("a preferred route that is not verified fails") {
        auto facts = ready;
        facts.preferred_route = std::make_pair(std::string("arrayfire_cuda"), 0);
        CHECK(status_of(cyxwiz::servernode::EvaluateNodeReadiness(facts), "Compute preference") == DoctorStatus::Fail);
    }
    SECTION("an unwritable data folder fails; a nearly full one warns") {
        auto facts = ready;
        facts.data_dir_writable = false;
        CHECK(status_of(cyxwiz::servernode::EvaluateNodeReadiness(facts), "Job data folder") == DoctorStatus::Fail);
        facts.data_dir_writable = true;
        facts.data_dir_free_bytes = 1ull << 30;
        CHECK(status_of(cyxwiz::servernode::EvaluateNodeReadiness(facts), "Job data folder") == DoctorStatus::Warn);
    }
    SECTION("an unreachable central server warns (P2P jobs still work)") {
        auto facts = ready;
        facts.central_server_reachable = false;
        const auto result = cyxwiz::servernode::EvaluateNodeReadiness(facts);
        CHECK(status_of(result, "Central server") == DoctorStatus::Warn);
        CHECK(cyxwiz::servernode::NodeIsReady(result));
    }
    SECTION("TLS without certificate files fails unless auto-generated") {
        auto facts = ready;
        facts.tls_enabled = true;
        CHECK(status_of(cyxwiz::servernode::EvaluateNodeReadiness(facts), "TLS") == DoctorStatus::Fail);
        facts.tls_auto = true;
        CHECK(status_of(cyxwiz::servernode::EvaluateNodeReadiness(facts), "TLS") == DoctorStatus::Ok);
    }
}

TEST_CASE("Job timing splits a job's wall time", "[timing]") {
    cyxwiz::servernode::NodeJobPhases phases;
    phases.wall_seconds = 10.0;
    phases.transfer_seconds = 1.0;
    phases.save_seconds = 0.5;
    phases.run.prepare_seconds = 1.5;
    phases.run.train_seconds = 5.0;
    phases.run.samples_trained = 400;
    phases.run.tokens_per_sample = 256;
    auto timing = cyxwiz::servernode::MakeJobTiming(phases);
    CHECK(timing.queue_seconds() == 2.0);  // the unaccounted remainder
    CHECK(timing.goodput() == 0.5);
    CHECK(timing.tokens_trained() == 400 * 256);
    CHECK(timing.samples_per_second() == 80.0);
    CHECK(timing.tokens_per_second() == 80.0 * 256);

    // Phases measured on different clocks never make the queue negative.
    phases.wall_seconds = 7.0;
    timing = cyxwiz::servernode::MakeJobTiming(phases);
    CHECK(timing.queue_seconds() == 0.0);
    CHECK(timing.wall_seconds() == 8.0);

    // A tabular job: no tokens; a job that never trained: no rates.
    phases.run.tokens_per_sample = 0;
    CHECK(cyxwiz::servernode::MakeJobTiming(phases).tokens_trained() == 0);
    phases.run = {};
    timing = cyxwiz::servernode::MakeJobTiming(phases);
    CHECK(timing.samples_per_second() == 0.0);
    CHECK(timing.goodput() == 0.0);
}

// Against a running central server (CYXWIZ_TEST_CENTRAL_SERVER=host:port,
// CYXWIZ_TEST_CENTRAL_JWT_SECRET = its [jwt] secret):
// registration stores the measured capability, discovery reports useful
// throughput, and a reported job's time split moves it (TOFIX118 P3 S5).
TEST_CASE("Central server ranks a registered node by measured throughput", "[.][central_live]") {
    namespace fs = std::filesystem;
    const char* address = std::getenv("CYXWIZ_TEST_CENTRAL_SERVER");
    if (address == nullptr || *address == '\0') SKIP("set CYXWIZ_TEST_CENTRAL_SERVER");

    const fs::path root = fs::temp_directory_path() / "cyxwiz_central_live_test";
    std::error_code ec;
    fs::remove_all(root, ec);
    cyxwiz::ScopedComputeRuntimeRootOverrideForTesting runtime_root(root);
    cyxwiz::RouteQualificationSnapshot snapshot;
    snapshot.matrix_id = "central-live-test";
    cyxwiz::RouteQualificationRecord gpu;
    gpu.type = cyxwiz::DeviceType::OPENCL;
    gpu.physical_fingerprint = "uuid:live-gpu";
    gpu.display_name = "Live Test GPU";
    gpu.operation_count = gpu.pass_count = 23;
    gpu.certified = true;
    snapshot.routes = {gpu};
    cyxwiz::InstallRouteQualificationSnapshot(snapshot);
    cyxwiz::TrainingBenchmarkResult measured;
    measured.ok = true;
    measured.backend = "arrayfire_opencl";
    measured.physical_fingerprint = "uuid:live-gpu";
    measured.build = cyxwiz::GetVersionString();
    measured.tokens_per_second = 6000.0;
    std::string error;
    REQUIRE(cyxwiz::SaveTrainingBenchmarkResults(cyxwiz::GetTrainingBenchmarkCachePath(), {measured}, error));

    using jwt_traits = jwt::traits::nlohmann_json;
    const char* secret = std::getenv("CYXWIZ_TEST_CENTRAL_JWT_SECRET");
    REQUIRE(secret != nullptr);
    const auto now = std::chrono::system_clock::now();
    const std::string token = jwt::create<jwt_traits>()
                                  .set_subject("central-live-test-user")
                                  .set_issued_at(now)
                                  .set_expires_at(now + std::chrono::hours(1))
                                  .sign(jwt::algorithm::hs256{secret});
    const auto authorize = [&](grpc::ClientContext& context) {
        context.AddMetadata("authorization", "Bearer " + token);
    };

    cyxwiz::servernode::NodeClient client(address, "central-live-test-node");
    client.SetAuthToken(token);
    REQUIRE(client.Register());
    const std::string node_id = client.GetNodeId();
    REQUIRE_FALSE(node_id.empty());

    auto channel = grpc::CreateChannel(address, grpc::InsecureChannelCredentials());
    auto discovery = cyxwiz::protocol::NodeDiscoveryService::NewStub(channel);
    const auto read_node = [&] {
        cyxwiz::protocol::GetNodeInfoRequest request;
        request.set_node_id(node_id);
        cyxwiz::protocol::GetNodeInfoResponse response;
        grpc::ClientContext context;
        authorize(context);
        const auto status = discovery->GetNodeInfo(&context, request, &response);
        INFO(status.error_message());
        REQUIRE(status.ok());
        return response.info();
    };

    // No job history yet: the benchmark times the goodput prior (0.5).
    auto info = read_node();
    CHECK(info.compute_score() == Catch::Approx(3000.0));
    REQUIRE(info.routes_size() == 1);
    CHECK(info.routes(0).device_name() == "Live Test GPU");
    CHECK(info.routes(0).benchmark().tokens_per_second() == Catch::Approx(6000.0));
    CHECK(info.environment().fingerprint().size() == 64);

    // Reserve the node (the P2P flow), then report a job whose training ran
    // at 5000 tokens/s for 80% of its wall time, as the node does at the end
    // of a job.
    auto reservations = cyxwiz::protocol::JobReservationService::NewStub(channel);
    cyxwiz::protocol::ReserveNodeRequest reserve;
    reserve.set_node_id(node_id);
    reserve.set_user_wallet("central-live-test-wallet");
    reserve.set_duration_minutes(10);
    cyxwiz::protocol::ReserveNodeResponse reserved;
    {
        grpc::ClientContext context;
        authorize(context);
        const auto status = reservations->ReserveNode(&context, reserve, &reserved);
        INFO(status.error_message() << " / " << reserved.error().message());
        REQUIRE(status.ok());
    }
    REQUIRE_FALSE(reserved.reservation_id().empty());
    cyxwiz::protocol::JobTiming timing;
    timing.set_train_seconds(8.0);
    timing.set_wall_seconds(10.0);
    timing.set_goodput(0.8);
    timing.set_tokens_trained(40000);
    timing.set_tokens_per_second(5000.0);
    cyxwiz::protocol::EnvironmentFingerprint environment;
    environment.set_cyxwiz_build(cyxwiz::GetVersionString());
    environment.set_fingerprint(std::string(64, 'e'));
    REQUIRE(client.ReportJobCompleteFromNode(reserved.reservation_id(), reserved.job_id(), true, "", {{"loss", 2.9}},
                                             10, 1, &timing, &environment));

    // Job history replaces the benchmark: 5000 x 0.8.
    info = read_node();
    CHECK(info.compute_score() == Catch::Approx(4000.0));

    client.Disconnect();
    cyxwiz::ClearRouteQualificationSnapshot();
    fs::remove_all(root, ec);
}

// Main function to run tests
int main(int argc, char* argv[]) {
    return Catch::Session().run(argc, argv);
}
