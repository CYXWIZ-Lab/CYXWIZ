// A group worker against a coordinator that follows a script: the worker's
// side is real (the model is built and trained); the coordinator only hands
// out rounds and checks what comes back.

// gRPC first: it brings in the Windows socket headers, which need macros
// that the headers of the fixture below remove.
#include <grpcpp/grpcpp.h>
#include <training_group.grpc.pb.h>

#include "group_worker.h"

#include "core/execution_device_preferences.h"
#include "core/sha256_digest.h"
#include "../../cyxwiz-engine/tests/causal_lm_token_window_fixture.h"
#include "../../cyxwiz-engine/tests/route_qualification_test_fixture.h"

#include <catch2/catch_test_macros.hpp>
#include <cyxwiz/device.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#ifndef CYXWIZ_SOURCE_ROOT
#error "CYXWIZ_SOURCE_ROOT must name the repository root"
#endif

namespace {

namespace fs = std::filesystem;
namespace wire = cyxwiz::protocol;

std::string Sha256Of(const std::string& bytes) {
    cyxwiz::Sha256Hasher hasher;
    std::string digest, error;
    hasher.Update(bytes, error);
    hasher.Finish(digest, error);
    return digest;
}

void SelectArrayFireCpu() {
    const auto devices = cyxwiz::Device::GetAvailableDevices();
    const auto cpu = std::find_if(devices.begin(), devices.end(), [](const cyxwiz::DeviceInfo& device) {
        return device.type == cyxwiz::DeviceType::CPU;
    });
    REQUIRE(cpu != devices.end());
    cyxwiz::test::InstallQualifiedRouteSnapshot(devices);
    cyxwiz::ClearPendingExecutionDeviceSelection();
    cyxwiz::SetPendingExecutionDeviceSelection(cyxwiz::DeviceType::CPU, cpu->device_id);
    cyxwiz::ClearNextRunExecutionPolicy();
    cyxwiz::SetNextRunExecutionPolicy(cyxwiz::ExecutionPolicy::Strict);
}

struct ReceivedUpdate {
    wire::UpdateHeader header;
    std::string bytes;
};

// Hands out `rounds` rounds, each leasing the same rows, then ends the job.
class ScriptedCoordinator final : public wire::TrainingGroupService::Service {
public:
    ScriptedCoordinator(std::string graph, std::string part, std::string second_part, int rounds)
        : graph_(std::move(graph)), part_(std::move(part)), second_part_(std::move(second_part)), rounds_(rounds) {}

    grpc::Status Join(grpc::ServerContext*, const wire::JoinRequest* request, wire::JoinResponse* response) override {
        std::lock_guard<std::mutex> lock(mutex);
        ++joins;
        if (request->ticket() != "the-ticket-of-this-node") {
            response->set_refusal(wire::GROUP_REFUSAL_TICKET);
            response->set_refusal_text("this ticket is not one of the group's");
            return grpc::Status::OK;
        }
        protocol_version = request->protocol_version();
        response->set_worker_id("worker-" + std::to_string(joins));
        response->set_worker_number(static_cast<uint32_t>(joins - 1));
        response->set_session_token("session-" + std::to_string(joins));
        response->set_uploads_initial_state(initial_state.empty());
        auto* job = response->mutable_job();
        job->set_group_id("scripted-group");
        job->set_model_definition(graph_);
        job->mutable_sample()->set_local_path(part_);
        job->mutable_sample()->set_rows(8);
        auto* settings = response->mutable_settings();
        settings->set_update_encoding(wire::UPDATE_ENCODING_INT8);
        settings->set_piece_bytes(64 * 1024);
        settings->set_progress_seconds(1);
        settings->set_warm_up_steps(1);
        return grpc::Status::OK;
    }

    grpc::Status DescribeModel(grpc::ServerContext*, const wire::ModelDescription* request, wire::ModelAccepted*) override {
        std::lock_guard<std::mutex> lock(mutex);
        descriptions.push_back(*request);
        return grpc::Status::OK;
    }

    grpc::Status UploadInitialState(grpc::ServerContext*, grpc::ServerReader<wire::StatePiece>* reader, wire::StateReceipt*) override {
        wire::StatePiece piece;
        wire::StateHeader header;
        std::string bytes;
        while (reader->Read(&piece)) {
            if (piece.has_header()) header = piece.header();
            else bytes += piece.data();
        }
        std::lock_guard<std::mutex> lock(mutex);
        initial_header = header;
        initial_state = bytes;
        return grpc::Status::OK;
    }

    grpc::Status DownloadInitialState(grpc::ServerContext*, const wire::InitialStateRequest*,
                                      grpc::ServerWriter<wire::StatePiece>* writer) override {
        std::lock_guard<std::mutex> lock(mutex);
        ++downloads;
        wire::StatePiece first;
        first.mutable_header()->set_layout_hash(initial_header.layout_hash());
        first.mutable_header()->set_bytes(initial_state.size());
        first.mutable_header()->set_sha256(Sha256Of(initial_state));
        writer->Write(first);
        for (size_t offset = 0; offset < initial_state.size(); offset += 50'000) {
            wire::StatePiece piece;
            piece.set_data(initial_state.substr(offset, 50'000));
            writer->Write(piece);
        }
        return grpc::Status::OK;
    }

    grpc::Status NextRound(grpc::ServerContext*, const wire::NextRoundRequest* request,
                           grpc::ServerWriter<wire::RoundPiece>* writer) override {
        std::lock_guard<std::mutex> lock(mutex);
        wire::RoundPiece first;
        auto* header = first.mutable_header();
        if (remove_worker == request->worker_id()) {
            header->set_removed(true);
            header->set_removal(wire::GROUP_REFUSAL_SPEED);
            header->set_removal_text("slower than the group allows");
            writer->Write(first);
            return grpc::Status::OK;
        }
        if (static_cast<int>(updates.size()) >= rounds_) {
            header->set_job_ending(true);
            header->set_removal_text("completed: the script is over");
            writer->Write(first);
            return grpc::Status::OK;
        }
        // The exchanged state: here every value is exchanged, so it is the starting state.
        const std::string& state = initial_state;
        const uint64_t round = updates.size() + 1;
        header->set_round(round);
        header->set_attempt("attempt-" + std::to_string(round));
        header->set_global_epoch(round - 1);
        header->set_train_seconds(20.0);
        header->set_report_within_seconds(60.0);
        header->set_layout_hash(initial_header.layout_hash());
        header->set_state_bytes(state.size());
        header->set_state_sha256(Sha256Of(state));
        // Round 1: the first part whole. Round 2: rows of two parts.
        if (round == 1) {
            auto* lease = header->add_leases();
            lease->mutable_chunk()->set_local_path(part_);
            lease->set_rows(8);
        } else {
            auto* lease = header->add_leases();
            lease->mutable_chunk()->set_local_path(part_);
            lease->set_row_offset(4);
            lease->set_rows(4);
            lease = header->add_leases();
            lease->mutable_chunk()->set_chunk_id(1);
            lease->mutable_chunk()->set_local_path(second_part_);
            lease->set_rows(2);
        }
        writer->Write(first);
        for (size_t offset = 0; offset < state.size(); offset += 50'000) {
            wire::RoundPiece piece;
            piece.set_data(state.substr(offset, 50'000));
            writer->Write(piece);
        }
        return grpc::Status::OK;
    }

    grpc::Status ReportProgress(grpc::ServerContext*, const wire::RoundProgress*, wire::RoundDirective*) override {
        return grpc::Status::OK;
    }

    grpc::Status SubmitUpdate(grpc::ServerContext*, grpc::ServerReader<wire::UpdatePiece>* reader, wire::UpdateReceipt* receipt) override {
        ReceivedUpdate update;
        wire::UpdatePiece piece;
        while (reader->Read(&piece)) {
            if (piece.has_header()) update.header = piece.header();
            else update.bytes += piece.data();
        }
        std::lock_guard<std::mutex> lock(mutex);
        updates.push_back(std::move(update));
        receipt->set_accepted(true);
        return grpc::Status::OK;
    }

    grpc::Status Leave(grpc::ServerContext*, const wire::LeaveRequest* request, wire::LeaveResponse*) override {
        std::lock_guard<std::mutex> lock(mutex);
        leaves.push_back(*request);
        return grpc::Status::OK;
    }

    std::mutex mutex;
    int joins = 0;
    int downloads = 0;
    uint32_t protocol_version = 0;
    std::vector<wire::ModelDescription> descriptions;
    wire::StateHeader initial_header;
    std::string initial_state;
    std::vector<ReceivedUpdate> updates;
    std::vector<wire::LeaveRequest> leaves;
    std::string remove_worker;

private:
    std::string graph_;
    std::string part_;
    std::string second_part_;
    int rounds_;
};

cyxwiz::servernode::GroupWorkerOptions Options(int port, const std::string& ticket) {
    cyxwiz::servernode::GroupWorkerOptions options;
    options.coordinator_address = "127.0.0.1:" + std::to_string(port);
    options.ticket = ticket;
    options.node_version = "test";
    options.select_device = SelectArrayFireCpu;
    options.retry_seconds_max = 1.0;
    return options;
}

}  // namespace

TEST_CASE("A group worker joins, trains its rounds and leaves when the job ends", "[group][worker]") {
    const fs::path root = CYXWIZ_SOURCE_ROOT;
    const fs::path work = fs::temp_directory_path() / "cyxwiz_group_worker_test";
    std::error_code ec;
    fs::remove_all(work, ec);
    fs::create_directories(work);
    const fs::path first = work / "windows_a.parquet";
    const fs::path second = work / "windows_b.parquet";
    REQUIRE(cyxwiz::test::WriteTokenWindows(root, first, 0, 8));
    REQUIRE(cyxwiz::test::WriteTokenWindows(root, second, 8, 2));

    ScriptedCoordinator coordinator(cyxwiz::test::LoadTokenWindowGraph(root).dump(), first.string(), second.string(), 2);
    int port = 0;
    grpc::ServerBuilder builder;
    builder.AddListeningPort("127.0.0.1:0", grpc::InsecureServerCredentials(), &port);
    builder.RegisterService(&coordinator);
    auto server = builder.BuildAndStart();
    REQUIRE(server);
    REQUIRE(port > 0);

    SECTION("the first worker builds the model the group starts from") {
        cyxwiz::servernode::GroupWorker worker(Options(port, "the-ticket-of-this-node"));
        worker.Run();
        const auto status = worker.Status();
        INFO(status.text);
        REQUIRE(status.state == cyxwiz::servernode::GroupWorkerState::Finished);
        CHECK(status.text == "completed: the script is over");
        CHECK(status.group_id == "scripted-group");
        CHECK(status.rounds_joined == 2);
        CHECK(status.updates_refused == 0);

        std::lock_guard<std::mutex> lock(coordinator.mutex);
        CHECK(coordinator.protocol_version == 1);
        REQUIRE(coordinator.descriptions.size() == 1);
        const wire::ModelDescription& description = coordinator.descriptions[0];
        CHECK(description.layout_hash().size() == 64);
        CHECK(description.layout_size() > 0);
        CHECK(description.warm_up_steps() == 1);
        CHECK(description.warm_up_samples() > 0);
        CHECK(description.warm_up_seconds() > 0.0);
        uint64_t values = 0;
        for (const auto& segment : description.layout()) {
            uint64_t count = 1;
            for (const uint64_t dimension : segment.shape()) count *= dimension;
            values += count;
            CHECK(segment.kind() != wire::STATE_KIND_UNSPECIFIED);
        }
        CHECK(status.state_values == values);

        // The starting state: every value, with the hash its header gave.
        CHECK(coordinator.initial_state.size() == values * sizeof(float));
        CHECK(coordinator.initial_header.sha256() == Sha256Of(coordinator.initial_state));
        CHECK(coordinator.initial_header.layout_hash() == description.layout_hash());

        REQUIRE(coordinator.updates.size() == 2);
        for (size_t index = 0; index < 2; ++index) {
            const ReceivedUpdate& update = coordinator.updates[index];
            INFO("round " << index + 1);
            CHECK(update.header.round() == index + 1);
            CHECK(update.header.attempt() == "attempt-" + std::to_string(index + 1));
            CHECK(update.header.layout_hash() == description.layout_hash());
            CHECK(update.header.base_state_sha256() == Sha256Of(coordinator.initial_state));
            CHECK(update.header.encoding() == wire::UPDATE_ENCODING_INT8);
            CHECK(update.header.bytes() == update.bytes.size());
            CHECK(update.header.sha256() == Sha256Of(update.bytes));
            CHECK(update.header.steps() > 0);
            CHECK(update.header.learning_rate() > 0.0);
            // Smaller than float32: the large tensors travel as one byte a value plus a table.
            CHECK(update.bytes.size() < values * sizeof(float));
            CHECK(update.bytes.size() > values);
            CHECK(update.header.tables() > 0);
        }
        CHECK(coordinator.updates[0].header.samples() == 8);
        CHECK(coordinator.updates[0].header.rows_done_size() == 1);
        CHECK(coordinator.updates[1].header.samples() == 6);
        REQUIRE(coordinator.updates[1].header.rows_done_size() == 2);
        CHECK(coordinator.updates[1].header.rows_done(0) == 4);
        CHECK(coordinator.updates[1].header.rows_done(1) == 2);
        CHECK(coordinator.leaves.empty());
    }

    SECTION("a later worker fetches the starting state, and is told when it is taken out") {
        {
            cyxwiz::servernode::GroupWorker first_worker(Options(port, "the-ticket-of-this-node"));
            first_worker.Run();
            REQUIRE(first_worker.Status().state == cyxwiz::servernode::GroupWorkerState::Finished);
        }
        {
            std::lock_guard<std::mutex> lock(coordinator.mutex);
            coordinator.remove_worker = "worker-2";
        }
        cyxwiz::servernode::GroupWorker worker(Options(port, "the-ticket-of-this-node"));
        worker.Run();
        const auto status = worker.Status();
        CHECK(status.state == cyxwiz::servernode::GroupWorkerState::Removed);
        CHECK(status.text == "slower than the group allows");
        std::lock_guard<std::mutex> lock(coordinator.mutex);
        CHECK(coordinator.downloads == 1);
        REQUIRE(coordinator.descriptions.size() == 2);
        CHECK(coordinator.descriptions[1].layout_hash() == coordinator.descriptions[0].layout_hash());
    }

    SECTION("a ticket the group does not know ends the worker with the reason") {
        cyxwiz::servernode::GroupWorker worker(Options(port, "another-ticket"));
        worker.Run();
        const auto status = worker.Status();
        CHECK(status.state == cyxwiz::servernode::GroupWorkerState::Failed);
        CHECK(status.text == "The group did not take this node: this ticket is not one of the group's");
    }

    SECTION("Stop ends a worker that waits for a coordinator that is not there") {
        server->Shutdown();
        cyxwiz::servernode::GroupWorker worker(Options(port, "the-ticket-of-this-node"));
        worker.Start();
        std::this_thread::sleep_for(std::chrono::milliseconds(600));
        CHECK(worker.Status().state == cyxwiz::servernode::GroupWorkerState::Joining);
        worker.Stop();
        CHECK(worker.Status().state == cyxwiz::servernode::GroupWorkerState::Finished);
        CHECK(worker.Status().text == "Stopped on this node");
    }

    server->Shutdown();
    fs::remove_all(work, ec);
}
