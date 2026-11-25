// gRPC first: it brings in the Windows socket headers, which need macros that
// headers included further down remove.
#include <grpcpp/grpcpp.h>
#include <training_group.grpc.pb.h>

#include "group_worker.h"

#include "common/server_address.h"
#include "core/execution_device_preferences.h"
#include "core/graph_document.h"
#include "core/round_training_session.h"
#include "core/round_update_codec.h"
#include "core/sha256_digest.h"
#include "node_group_state.h"
#include "node_job_route.h"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <optional>
#include <vector>

namespace cyxwiz::servernode {

namespace {

namespace pb = cyxwiz::protocol;
using Clock = std::chrono::steady_clock;

constexpr uint32_t kProtocolVersion = 1;
constexpr int kCallSeconds = 30;          // a short call (join, describe, report, leave)
constexpr int kTransferSeconds = 15 * 60; // a state or an update on a slow link

std::string Sha256Of(const void* data, size_t bytes) {
    Sha256Hasher hasher;
    std::string digest;
    std::string error;
    if (!hasher.Update(std::string_view(static_cast<const char*>(data), bytes), error) || !hasher.Finish(digest, error)) {
        return {};
    }
    return digest;
}

std::string FloatBytes(const std::vector<float>& values) {
    std::string bytes(values.size() * sizeof(float), '\0');
    if (!values.empty()) std::memcpy(bytes.data(), values.data(), bytes.size());
    return bytes;
}

bool FloatsFrom(const std::string& bytes, std::vector<float>& values) {
    if (bytes.size() % sizeof(float) != 0) return false;
    values.resize(bytes.size() / sizeof(float));
    if (!values.empty()) std::memcpy(values.data(), bytes.data(), bytes.size());
    return true;
}

pb::StateKind KindOf(ModelStateRole role) {
    switch (role) {
        case ModelStateRole::Trainable: return pb::STATE_KIND_TRAINABLE;
        case ModelStateRole::Statistic: return pb::STATE_KIND_STATISTIC;
        case ModelStateRole::Frozen: return pb::STATE_KIND_FROZEN;
    }
    return pb::STATE_KIND_UNSPECIFIED;
}

pb::LeaveRequest::Reason ReasonOf(TrainingFailureKind kind) {
    switch (kind) {
        case TrainingFailureKind::OutOfMemory: return pb::LeaveRequest::REASON_OUT_OF_MEMORY;
        case TrainingFailureKind::DeviceError: return pb::LeaveRequest::REASON_DEVICE_ERROR;
        case TrainingFailureKind::DataError: return pb::LeaveRequest::REASON_DATA_ERROR;
        case TrainingFailureKind::Cancelled: return pb::LeaveRequest::REASON_OWNER_STOP;
        default: return pb::LeaveRequest::REASON_TRAINING_ERROR;
    }
}

// The graph's one Data Input: the name the job's data parts stand for.
bool TrainingInputName(const std::string& graph_json, std::string& name, std::string& error) {
    GraphDocument graph;
    if (!ParseGraphDocument(graph_json, graph, error)) return false;
    std::vector<std::string> names;
    for (const auto& node : graph.nodes) {
        if (node.type != gui::NodeType::DataInput) continue;
        const auto found = node.parameters.find("dataset_name");
        if (found != node.parameters.end() && !found->second.empty()) names.push_back(found->second);
    }
    if (names.size() != 1) {
        error = "a group job's graph has exactly one Data Input (the coordinator owns validation); this one has " +
                std::to_string(names.size());
        return false;
    }
    name = names.front();
    return true;
}

bool LocalPath(const pb::ChunkGrant& chunk, std::string& path, std::string& error) {
    if (!chunk.has_local_path() || chunk.local_path().empty()) {
        error = "data part " + std::to_string(chunk.chunk_id()) + " is not a path this node can read; parts from CyxCloud "
                "are not fetched by a group worker yet";
        return false;
    }
    path = chunk.local_path();
    return true;
}

}  // namespace

const char* GroupWorkerStateName(GroupWorkerState state) {
    switch (state) {
        case GroupWorkerState::NotStarted: return "Not in a group";
        case GroupWorkerState::Joining: return "Joining";
        case GroupWorkerState::BuildingModel: return "Building the model";
        case GroupWorkerState::MeasuringLink: return "Measuring the link";
        case GroupWorkerState::GettingModel: return "Getting the model";
        case GroupWorkerState::WaitingForRound: return "Waiting for the round";
        case GroupWorkerState::ReceivingState: return "Getting the model";
        case GroupWorkerState::Training: return "Training";
        case GroupWorkerState::SendingUpdate: return "Sending its update";
        case GroupWorkerState::Finished: return "Group finished";
        case GroupWorkerState::Removed: return "Removed from the group";
        case GroupWorkerState::Failed: return "Could not go on";
    }
    return "";
}

// One stay in a group: the connection, this node's place and its model.
class GroupWorker::Session {
public:
    explicit Session(GroupWorker& worker) : worker_(worker) {
        grpc::ChannelArguments arguments;
        arguments.SetInt(GRPC_ARG_KEEPALIVE_TIME_MS, 30'000);
        stub_ = pb::TrainingGroupService::NewStub(protocol::CreateServerChannel(worker.options_.coordinator_address, arguments));
    }

    enum class Step { Ok, Retry, Rejoin, End };

    // A call's context, cancelled by Stop while it is open.
    struct Call {
        Call(GroupWorker& worker, int seconds) : worker(worker) {
            if (seconds > 0) context.set_deadline(std::chrono::system_clock::now() + std::chrono::seconds(seconds));
            worker.Track(&context);
        }
        ~Call() { worker.Track(nullptr); }
        GroupWorker& worker;
        grpc::ClientContext context;
    };

    Step FromStatus(const grpc::Status& status, const char* what) {
        if (status.ok()) return Step::Ok;
        if (worker_.stop_.load()) return Step::End;
        if (status.error_code() == grpc::StatusCode::UNAUTHENTICATED) {
            spdlog::warn("Group worker: {}: the coordinator does not know this session ({}); joining again", what, status.error_message());
            return Step::Rejoin;
        }
        spdlog::warn("Group worker: {}: {} ({})", what, status.error_message(), static_cast<int>(status.error_code()));
        return Step::Retry;
    }

    Step Join() {
        worker_.Set(GroupWorkerState::Joining);
        pb::JoinRequest request;
        request.set_ticket(worker_.options_.ticket);
        request.set_node_version(worker_.options_.node_version);
        request.set_protocol_version(kProtocolVersion);
        pb::JoinResponse response;
        Call call(worker_, kCallSeconds);
        if (const Step step = FromStatus(stub_->Join(&call.context, request, &response), "join"); step != Step::Ok) {
            return step == Step::Rejoin ? Step::Retry : step;
        }
        if (response.refusal() != pb::GROUP_REFUSAL_NONE) {
            worker_.Set(GroupWorkerState::Failed, "The group did not take this node: " + response.refusal_text());
            return Step::End;
        }
        worker_id_ = response.worker_id();
        token_ = response.session_token();
        uploads_ = response.uploads_initial_state();
        if (!joined_once_) {
            job_ = response.job();
            settings_ = response.settings();
            worker_number_ = response.worker_number();
            joined_once_ = true;
        } else if (response.job().group_id() != job_.group_id()) {
            worker_.Set(GroupWorkerState::Failed, "The coordinator now runs another group");
            return Step::End;
        }
        worker_.Update([&](GroupWorkerStatus& status) { status.group_id = job_.group_id(); });
        spdlog::info("Group worker: joined group {} as worker {}", job_.group_id(), worker_number_);
        return Step::Ok;
    }

    // Builds the model against the job's sample part and times a few steps.
    Step BuildModel() {
        if (session_) return Step::Ok;
        worker_.Set(GroupWorkerState::BuildingModel);
        std::string error;
        std::string sample_path;
        if (!TrainingInputName(job_.model_definition(), dataset_name_, error) || !LocalPath(job_.sample(), sample_path, error)) {
            return Fail(pb::LeaveRequest::REASON_DATA_ERROR, error);
        }
        GraphTrainingJobRequest request;
        request.graph_json = job_.model_definition();
        request.dataset_files[dataset_name_] = sample_path;
        request.batch_size_override = static_cast<int>(job_.batch_size());
        RoundSessionOptions options;
        options.worker = worker_number_;
        options.schedule_total_steps = job_.schedule_total_steps();
        RoundFailure failure;
        SelectDevice();
        session_ = RoundTrainingSession::Open(request, options, failure);
        if (!session_) return Fail(ReasonOf(failure.kind), failure.text);
        const ModelStateLayout& layout = session_->Layout();
        worker_.Update([&](GroupWorkerStatus& status) { status.state_values = layout.exchanged_count; });

        // The warm-up: the coordinator divides samples by seconds itself.
        std::vector<float> theta(layout.exchanged_count);
        const std::vector<float>& built = session_->InitialState();
        for (const StateSegment& segment : layout.segments) {
            if (segment.role == ModelStateRole::Frozen) continue;
            std::copy_n(built.begin() + static_cast<std::ptrdiff_t>(segment.offset), segment.count,
                        theta.begin() + static_cast<std::ptrdiff_t>(segment.exchanged_offset));
        }
        RoundData data;
        data.leases.push_back({sample_path, 0, 0});
        RoundBudget budget;
        budget.max_steps = std::max<uint32_t>(1, settings_.warm_up_steps());
        RoundOutcome outcome;
        SelectDevice();
        if (!session_->TrainRound({0, 0, 0}, theta, data, budget, StopCallbacks(), outcome, failure)) {
            session_.reset();
            return Fail(ReasonOf(failure.kind), failure.text);
        }
        warm_up_steps_ = outcome.steps;
        warm_up_samples_ = outcome.samples;
        warm_up_seconds_ = outcome.train_seconds;
        spdlog::info("Group worker: model built ({} values exchanged, layout {}); warm-up {} samples in {:.2f} s",
                     layout.exchanged_count, layout.hash, warm_up_samples_, warm_up_seconds_);
        return Step::Ok;
    }

    Step Describe() {
        const ModelStateLayout& layout = session_->Layout();
        pb::ModelDescription request;
        request.set_worker_id(worker_id_);
        request.set_session_token(token_);
        request.set_layout_hash(layout.hash);
        for (const StateSegment& segment : layout.segments) {
            pb::StateSegment* sent = request.add_layout();
            sent->set_name(segment.name);
            for (const size_t dimension : segment.shape) sent->add_shape(dimension);
            sent->set_kind(KindOf(segment.role));
        }
        request.set_warm_up_steps(warm_up_steps_);
        request.set_warm_up_samples(warm_up_samples_);
        request.set_warm_up_seconds(warm_up_seconds_);
        // The group's first worker defines the layout; the others wait for it.
        for (;;) {
            pb::ModelAccepted response;
            Call call(worker_, kCallSeconds);
            if (const Step step = FromStatus(stub_->DescribeModel(&call.context, request, &response), "describe the model");
                step != Step::Ok) {
                return step;
            }
            if (response.refusal() == pb::GROUP_REFUSAL_NONE) return Step::Ok;
            if (response.refusal() != pb::GROUP_REFUSAL_STALE) {
                worker_.Set(GroupWorkerState::Failed, "The group did not take this node: " + response.refusal_text());
                return Step::End;
            }
            if (!worker_.Wait(0.5)) return Step::End;
        }
    }

    // The coordinator times everything; this side answers and sends.
    Step ProbeLink() {
        const uint64_t bytes = settings_.probe_bytes();
        if (bytes == 0) return Step::Ok;
        worker_.Set(GroupWorkerState::MeasuringLink);
        Call call(worker_, kTransferSeconds);
        auto stream = stub_->ProbeLink(&call.context);
        pb::ProbeMessage start;
        start.mutable_start()->set_worker_id(worker_id_);
        start.mutable_start()->set_session_token(token_);
        stream->Write(start);
        pb::ProbeMessage message;
        uint64_t received = 0;
        bool measured = false;
        pb::LinkMeasurement link;
        while (!measured && stream->Read(&message)) {
            switch (message.kind_case()) {
                case pb::ProbeMessage::kPing: {
                    pb::ProbeMessage pong;
                    pong.set_pong(message.ping());
                    stream->Write(pong);
                    break;
                }
                case pb::ProbeMessage::kData:
                    received += message.data().size();
                    break;
                case pb::ProbeMessage::kDone: {
                    pb::ProbeMessage got;
                    got.set_received(received);
                    bool open = stream->Write(got);
                    const std::string piece(PieceBytes(), '\x5a');
                    for (uint64_t sent = 0; open && sent < bytes;) {
                        const size_t size = static_cast<size_t>(std::min<uint64_t>(bytes - sent, piece.size()));
                        pb::ProbeMessage data;
                        data.set_data(piece.substr(0, size));
                        open = stream->Write(data);
                        sent += size;
                    }
                    pb::ProbeMessage done;
                    done.set_done(bytes);
                    if (open) stream->Write(done);
                    break;
                }
                case pb::ProbeMessage::kResult:
                    link = message.result();
                    measured = true;
                    break;
                default:
                    break;
            }
        }
        stream->WritesDone();
        if (const Step step = FromStatus(stream->Finish(), "measure the link"); step != Step::Ok) return step;
        if (!measured) return Step::Retry;
        worker_.Update([&](GroupWorkerStatus& status) {
            status.up_mbit = link.up_mbit();
            status.down_mbit = link.down_mbit();
            status.latency_ms = link.latency_ms();
        });
        spdlog::info("Group worker: link to the coordinator: up {:.1f} Mbit/s, down {:.1f} Mbit/s, round trip {:.1f} ms",
                     link.up_mbit(), link.down_mbit(), link.latency_ms());
        return Step::Ok;
    }

    Step ExchangeInitialState() {
        worker_.Set(GroupWorkerState::GettingModel);
        const ModelStateLayout& layout = session_->Layout();
        if (uploads_) {
            const std::string bytes = FloatBytes(session_->InitialState());
            pb::StateReceipt receipt;
            Call call(worker_, kTransferSeconds);
            auto writer = stub_->UploadInitialState(&call.context, &receipt);
            pb::StatePiece piece;
            pb::StateHeader* header = piece.mutable_header();
            header->set_worker_id(worker_id_);
            header->set_session_token(token_);
            header->set_layout_hash(layout.hash);
            header->set_bytes(bytes.size());
            header->set_sha256(Sha256Of(bytes.data(), bytes.size()));
            bool open = writer->Write(piece);
            for (size_t offset = 0; open && offset < bytes.size(); offset += PieceBytes()) {
                pb::StatePiece data;
                data.set_data(bytes.substr(offset, PieceBytes()));
                open = writer->Write(data);
            }
            writer->WritesDone();
            if (const Step step = FromStatus(writer->Finish(), "send the starting state"); step != Step::Ok) return step;
            if (receipt.refusal() != pb::GROUP_REFUSAL_NONE) {
                return Fail(pb::LeaveRequest::REASON_TRAINING_ERROR, "the starting state was refused: " + receipt.refusal_text());
            }
            return Step::Ok;
        }
        // The first worker may still be sending it.
        for (;;) {
            pb::InitialStateRequest request;
            request.set_worker_id(worker_id_);
            request.set_session_token(token_);
            Call call(worker_, kTransferSeconds);
            auto reader = stub_->DownloadInitialState(&call.context, request);
            pb::StatePiece piece;
            pb::StateHeader header;
            std::string bytes;
            while (reader->Read(&piece)) {
                if (piece.has_header()) header = piece.header();
                else bytes += piece.data();
            }
            const grpc::Status status = reader->Finish();
            if (status.error_code() == grpc::StatusCode::FAILED_PRECONDITION) {
                if (!worker_.Wait(0.5)) return Step::End;
                continue;
            }
            if (const Step step = FromStatus(status, "get the starting state"); step != Step::Ok) return step;
            std::vector<float> state;
            std::string error;
            if (header.layout_hash() != layout.hash || bytes.size() != header.bytes() ||
                Sha256Of(bytes.data(), bytes.size()) != header.sha256() || !FloatsFrom(bytes, state)) {
                spdlog::warn("Group worker: the starting state arrived damaged; asking again");
                return Step::Retry;
            }
            if (!session_->LoadInitialState(state, error)) return Fail(pb::LeaveRequest::REASON_TRAINING_ERROR, error);
            return Step::Ok;
        }
    }

    // One round: wait for it, train, send. Step::Ok means "ask for the next".
    Step Round() {
        worker_.Set(GroupWorkerState::WaitingForRound);
        pb::RoundHeader header;
        std::string state_bytes;
        Clock::time_point header_at;
        {
            pb::NextRoundRequest request;
            request.set_worker_id(worker_id_);
            request.set_session_token(token_);
            request.set_last_round_seen(last_round_);
            Call call(worker_, 0);  // waits as long as the round takes to open
            auto reader = stub_->NextRound(&call.context, request);
            pb::RoundPiece piece;
            bool got_header = false;
            while (reader->Read(&piece)) {
                if (piece.has_header()) {
                    header = piece.header();
                    got_header = true;
                    header_at = Clock::now();
                    state_bytes.reserve(header.state_bytes());
                    if (!header.removed() && !header.job_ending()) worker_.Set(GroupWorkerState::ReceivingState);
                } else {
                    state_bytes += piece.data();
                }
            }
            if (const Step step = FromStatus(reader->Finish(), "wait for the round"); step != Step::Ok) return step;
            if (!got_header) return Step::Retry;
        }
        if (header.removed()) {
            worker_.Set(GroupWorkerState::Removed, header.removal_text());
            return Step::End;
        }
        if (header.job_ending()) {
            worker_.Set(GroupWorkerState::Finished, header.removal_text());
            return Step::End;
        }
        const ModelStateLayout& layout = session_->Layout();
        std::vector<float> theta;
        if (header.layout_hash() != layout.hash || state_bytes.size() != header.state_bytes() ||
            Sha256Of(state_bytes.data(), state_bytes.size()) != header.state_sha256() || !FloatsFrom(state_bytes, theta)) {
            spdlog::warn("Group worker: round {}: the state arrived damaged; asking again", header.round());
            return Step::Retry;
        }
        state_bytes.clear();
        state_bytes.shrink_to_fit();

        RoundData data;
        std::string error;
        for (const pb::RowLease& lease : header.leases()) {
            std::string path;
            if (!LocalPath(lease.chunk(), path, error)) return Fail(pb::LeaveRequest::REASON_DATA_ERROR, error);
            data.leases.push_back({path, lease.row_offset(), lease.rows()});
        }
        // Stop training in time for the update to arrive: the last upload's
        // duration, doubled, is kept back from what the coordinator allows.
        const double received_seconds = std::chrono::duration<double>(Clock::now() - header_at).count();
        RoundBudget budget;
        budget.max_steps = header.max_steps();
        budget.train_seconds = header.train_seconds();
        if (header.report_within_seconds() > 0.0) {
            const double room = header.report_within_seconds() - received_seconds - 2.0 * last_send_seconds_ - 1.0;
            budget.train_seconds = std::clamp(room, 0.5, std::max(0.5, header.train_seconds()));
        }

        worker_.Set(GroupWorkerState::Training);
        worker_.Update([&](GroupWorkerStatus& status) {
            status.round = header.round();
            status.steps_this_round = 0;
            status.samples_this_round = 0;
        });
        bool abandon = false;
        RoundCallbacks callbacks = StopCallbacks();
        Clock::time_point next_report = Clock::now() + std::chrono::seconds(std::max<uint32_t>(1, settings_.progress_seconds()));
        callbacks.should_stop = [&] { return worker_.stop_.load() || abandon; };
        callbacks.on_progress = [&](uint64_t steps, uint64_t samples, double loss) {
            worker_.Update([&](GroupWorkerStatus& status) {
                status.steps_this_round = steps;
                status.samples_this_round = samples;
                status.last_loss = loss;
            });
            if (Clock::now() < next_report) return;
            next_report = Clock::now() + std::chrono::seconds(std::max<uint32_t>(1, settings_.progress_seconds()));
            pb::RoundProgress progress;
            progress.set_worker_id(worker_id_);
            progress.set_session_token(token_);
            progress.set_round(header.round());
            progress.set_attempt(header.attempt());
            progress.set_stage(pb::ROUND_STAGE_TRAINING);
            progress.set_steps(steps);
            progress.set_samples(samples);
            progress.set_last_loss(loss);
            pb::RoundDirective directive;
            grpc::ClientContext context;
            context.set_deadline(std::chrono::system_clock::now() + std::chrono::seconds(5));
            if (stub_->ReportProgress(&context, progress, &directive).ok() &&
                directive.action() != pb::RoundDirective::ACTION_CONTINUE &&
                directive.action() != pb::RoundDirective::ACTION_STOP_AND_REPORT) {
                abandon = true;  // the round is over, or this node is out: NextRound says which
            }
        };
        RoundOutcome outcome;
        RoundFailure failure;
        SelectDevice();
        const bool trained = session_->TrainRound({header.round(), header.global_epoch(), header.global_step()}, theta, data,
                                                  budget, callbacks, outcome, failure);
        if (!trained) {
            if (abandon && !worker_.stop_.load()) return Step::Ok;
            if (worker_.stop_.load()) return Step::End;
            return Fail(ReasonOf(failure.kind), failure.text);
        }
        last_round_ = header.round();
        const double speed = outcome.train_seconds > 0.0 ? static_cast<double>(outcome.samples) / outcome.train_seconds : 0.0;
        worker_.Update([&](GroupWorkerStatus& status) {
            status.steps_this_round = outcome.steps;
            status.samples_this_round = outcome.samples;
            status.samples_per_second = speed;
            status.last_loss = outcome.last_loss;
        });

        // The update on the wire: the decode tables, then the tensors.
        worker_.Set(GroupWorkerState::SendingUpdate);
        const UpdateWire wire = settings_.update_encoding() == pb::UPDATE_ENCODING_INT8 ? UpdateWire::Int8 : UpdateWire::Float32;
        EncodedUpdate encoded;
        if (!EncodeUpdate(outcome.delta, layout, wire, encoded, error)) return Fail(pb::LeaveRequest::REASON_TRAINING_ERROR, error);
        std::string bytes(encoded.tables.size() * sizeof(Int8Table) + encoded.bytes.size(), '\0');
        if (!encoded.tables.empty()) std::memcpy(bytes.data(), encoded.tables.data(), encoded.tables.size() * sizeof(Int8Table));
        if (!encoded.bytes.empty()) {
            std::memcpy(bytes.data() + encoded.tables.size() * sizeof(Int8Table), encoded.bytes.data(), encoded.bytes.size());
        }
        const auto send_start = Clock::now();
        pb::UpdateReceipt receipt;
        Call call(worker_, kTransferSeconds);
        auto writer = stub_->SubmitUpdate(&call.context, &receipt);
        pb::UpdatePiece first;
        pb::UpdateHeader* update = first.mutable_header();
        update->set_worker_id(worker_id_);
        update->set_session_token(token_);
        update->set_round(header.round());
        update->set_attempt(header.attempt());
        update->set_layout_hash(layout.hash);
        update->set_base_state_sha256(header.state_sha256());
        update->set_encoding(wire == UpdateWire::Int8 ? pb::UPDATE_ENCODING_INT8 : pb::UPDATE_ENCODING_FLOAT32);
        update->set_tables(static_cast<uint32_t>(encoded.tables.size()));
        update->set_bytes(bytes.size());
        update->set_sha256(Sha256Of(bytes.data(), bytes.size()));
        update->set_samples(outcome.samples);
        update->set_steps(outcome.steps);
        update->set_last_loss(outcome.last_loss);
        update->set_learning_rate(outcome.learning_rate);
        for (const uint64_t rows : outcome.rows_done) update->add_rows_done(rows);
        update->mutable_timing()->set_receive_seconds(received_seconds);
        update->mutable_timing()->set_train_seconds(outcome.train_seconds);
        bool open = writer->Write(first);
        for (size_t offset = 0; open && offset < bytes.size(); offset += PieceBytes()) {
            pb::UpdatePiece piece;
            piece.set_data(bytes.substr(offset, PieceBytes()));
            open = writer->Write(piece);
        }
        writer->WritesDone();
        const Step sent = FromStatus(writer->Finish(), "send the update");
        last_send_seconds_ = std::chrono::duration<double>(Clock::now() - send_start).count();
        if (sent != Step::Ok) return sent;
        if (receipt.accepted()) {
            worker_.Update([](GroupWorkerStatus& status) { ++status.rounds_joined; });
            spdlog::info("Group worker: round {}: {} steps, {} samples, loss {:.4f}, update sent in {:.2f} s", header.round(),
                         outcome.steps, outcome.samples, outcome.last_loss, last_send_seconds_);
        } else {
            worker_.Update([](GroupWorkerStatus& status) { ++status.updates_refused; });
            spdlog::warn("Group worker: round {}: the update was not taken: {}", header.round(), receipt.refusal_text());
        }
        return Step::Ok;
    }

    // An error nothing below caught (out of memory, a device or data library throwing): the node
    // leaves the group and reports it, instead of the exception ending the whole daemon.
    void Abort(const std::string& what) {
        const std::string text = "The group worker stopped on an error: " + what;
        spdlog::error("Group worker: {}", text);
        try {
            Leave(pb::LeaveRequest::REASON_TRAINING_ERROR, text);
        } catch (...) {
            // Leaving is a courtesy; the coordinator also drops a worker that stops answering.
        }
        worker_.Set(GroupWorkerState::Failed, text);
    }

    void Leave(pb::LeaveRequest::Reason reason, const std::string& text) {
        if (worker_id_.empty()) return;
        pb::LeaveRequest request;
        request.set_worker_id(worker_id_);
        request.set_session_token(token_);
        request.set_reason(reason);
        request.set_text(text);
        pb::LeaveResponse response;
        grpc::ClientContext context;
        context.set_deadline(std::chrono::system_clock::now() + std::chrono::seconds(5));
        stub_->Leave(&context, request, &response);
    }

private:
    Step Fail(pb::LeaveRequest::Reason reason, const std::string& text) {
        spdlog::error("Group worker: {}", text);
        Leave(reason, text);
        worker_.Set(GroupWorkerState::Failed, text);
        return Step::End;
    }

    void SelectDevice() {
        if (worker_.options_.select_device) {
            worker_.options_.select_device();
        } else if (const auto route = NodeJobRoute()) {
            // Every training run takes the pending selection with it.
            SetPendingExecutionDeviceSelection(route->type, route->device_id);
        }
    }

    RoundCallbacks StopCallbacks() {
        RoundCallbacks callbacks;
        callbacks.should_stop = [this] { return worker_.stop_.load(); };
        return callbacks;
    }

    size_t PieceBytes() const { return std::clamp<size_t>(settings_.piece_bytes(), 64 * 1024, 2 * 1024 * 1024); }

    GroupWorker& worker_;
    std::unique_ptr<pb::TrainingGroupService::Stub> stub_;
    std::unique_ptr<RoundTrainingSession> session_;
    pb::GroupJobSpec job_;
    pb::GroupSettings settings_;
    bool joined_once_ = false;
    bool uploads_ = false;
    uint32_t worker_number_ = 0;
    std::string worker_id_;
    std::string token_;
    std::string dataset_name_;
    uint64_t warm_up_steps_ = 0;
    uint64_t warm_up_samples_ = 0;
    double warm_up_seconds_ = 0.0;
    uint64_t last_round_ = 0;
    double last_send_seconds_ = 0.0;
};

GroupWorker::GroupWorker(GroupWorkerOptions options) : options_(std::move(options)) {}

GroupWorker::~GroupWorker() { Stop(); }

void GroupWorker::Start() {
    if (thread_.joinable()) return;
    stop_.store(false);
    thread_ = std::thread([this] { Run(); });
}

void GroupWorker::Stop() {
    stop_.store(true);
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (active_call_) active_call_->TryCancel();
    }
    if (thread_.joinable()) thread_.join();
}

void GroupWorker::Run() {
    spdlog::info("Group worker: joining the group at {}", options_.coordinator_address);
    // The device is the group's from here to the end of Run.
    struct InGroup {
        InGroup() { NodeInGroup().store(true); }
        ~InGroup() { NodeInGroup().store(false); }
    } in_group_guard;
    Session session(*this);
    using Step = Session::Step;
    double retry_seconds = 1.0;
    // After a step that asks to retry: wait, a little longer each time.
    const auto pause = [&] {
        const bool go_on = Wait(retry_seconds);
        retry_seconds = std::min(options_.retry_seconds_max, retry_seconds * 2.0);
        return go_on;
    };
    bool in_group = false;
    try {
        while (!stop_.load()) {
            Step step = Step::Ok;
            if (!in_group) {
                // Join, build (once), describe, then send or fetch the starting state.
                step = session.Join();
                if (step == Step::Ok) step = session.BuildModel();
                if (step == Step::Ok) step = session.Describe();
                if (step == Step::Ok) step = session.ProbeLink();
                if (step == Step::Ok) step = session.ExchangeInitialState();
                in_group = step == Step::Ok;
            } else {
                step = session.Round();
            }
            if (step == Step::End) break;
            if (step == Step::Rejoin) {
                in_group = false;
                continue;
            }
            if (step == Step::Retry) {
                if (!pause()) break;
                continue;
            }
            retry_seconds = 1.0;
        }
        if (stop_.load() && !Done()) {
            session.Leave(protocol::LeaveRequest::REASON_OWNER_STOP, "the node was stopped");
            Set(GroupWorkerState::Finished, "Stopped on this node");
        }
    } catch (const std::exception& error) {
        session.Abort(error.what());
    } catch (...) {
        session.Abort("an unknown error");
    }
    const GroupWorkerStatus status = Status();
    spdlog::info("Group worker: {}{}{}", GroupWorkerStateName(status.state), status.text.empty() ? "" : ": ", status.text);
}

GroupWorkerStatus GroupWorker::Status() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return status_;
}

bool GroupWorker::Done() const {
    const GroupWorkerState state = Status().state;
    return state == GroupWorkerState::Finished || state == GroupWorkerState::Removed || state == GroupWorkerState::Failed;
}

void GroupWorker::Set(GroupWorkerState state, const std::string& text) {
    std::lock_guard<std::mutex> lock(mutex_);
    status_.state = state;
    status_.text = text;
}

void GroupWorker::Update(const std::function<void(GroupWorkerStatus&)>& change) {
    std::lock_guard<std::mutex> lock(mutex_);
    change(status_);
}

bool GroupWorker::Wait(double seconds) {
    const auto until = Clock::now() + std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(seconds));
    while (Clock::now() < until) {
        if (stop_.load()) return false;
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    return !stop_.load();
}

void GroupWorker::Track(grpc::ClientContext* context) {
    std::lock_guard<std::mutex> lock(mutex_);
    active_call_ = context;
    // Stop() may have run after the worker's last stop check but before this call was tracked:
    // cancel it now (gRPC applies a cancel made before the call starts), so Stop() never waits
    // on a NextRound that has no deadline.
    if (context && stop_.load()) context->TryCancel();
}

}  // namespace cyxwiz::servernode
