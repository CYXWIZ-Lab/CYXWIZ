#pragma once

// This node's part in a training group.
//
// Several nodes train one model in rounds
// (docs/architecture/distributed-training.md). A group worker joins its
// group's coordinator with a ticket, builds the job's model once, and then,
// round after round, receives the group's state, trains on the rows it is
// leased until the round's time is up and sends back the change it made.
//
// Every call is opened by this node. The worker runs on a thread of its own,
// which is the only thread that touches the model.

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

namespace grpc {
class ClientContext;
}

namespace cyxwiz::servernode {

struct GroupWorkerOptions {
    std::string coordinator_address;  // host:port, or https://host:port
    std::string ticket;               // never logged
    std::string node_version;
    // Selects this node's compute device for the training run that follows.
    // Called on the worker's thread before the model is built and before
    // every round. Empty: the node's own route (NodeJobRoute).
    std::function<void()> select_device;
    // The longest wait between two attempts to reach the coordinator.
    double retry_seconds_max = 30.0;
};

enum class GroupWorkerState {
    NotStarted,
    Joining,
    BuildingModel,   // building the model and timing a few steps
    MeasuringLink,   // the coordinator measures this node's link
    GettingModel,    // sending or fetching the group's starting state
    WaitingForRound,
    ReceivingState,
    Training,
    SendingUpdate,
    Finished,        // the job ended, or this node was stopped
    Removed,         // the group's coordinator took this node out
    Failed,          // this node could not go on
};

const char* GroupWorkerStateName(GroupWorkerState state);

struct GroupWorkerStatus {
    GroupWorkerState state = GroupWorkerState::NotStarted;
    std::string text;            // the reason, for Finished / Removed / Failed
    std::string group_id;
    uint64_t round = 0;
    uint64_t steps_this_round = 0;
    uint64_t samples_this_round = 0;
    double samples_per_second = 0.0;
    double last_loss = 0.0;
    uint64_t rounds_joined = 0;
    uint64_t updates_refused = 0;
    uint64_t state_values = 0;   // values exchanged each round
    // The link as the coordinator measured it (0 until measured).
    double up_mbit = 0.0;
    double down_mbit = 0.0;
    double latency_ms = 0.0;
};

class GroupWorker {
public:
    explicit GroupWorker(GroupWorkerOptions options);
    ~GroupWorker();
    GroupWorker(const GroupWorker&) = delete;
    GroupWorker& operator=(const GroupWorker&) = delete;

    // Runs the worker on a thread of its own.
    void Start();
    // Ends the round in progress, tells the coordinator and returns when the
    // worker's thread is done.
    void Stop();
    // The worker itself: returns when the job ends, this node is removed or
    // stopped, or it cannot go on. For a caller that brings its own thread.
    void Run();

    GroupWorkerStatus Status() const;
    bool Done() const;

private:
    class Session;
    void Set(GroupWorkerState state, const std::string& text = {});
    void Update(const std::function<void(GroupWorkerStatus&)>& change);
    // False when Stop was called while waiting.
    bool Wait(double seconds);
    void Track(grpc::ClientContext* context);

    GroupWorkerOptions options_;
    std::thread thread_;
    std::atomic<bool> stop_{false};
    mutable std::mutex mutex_;
    GroupWorkerStatus status_;
    grpc::ClientContext* active_call_ = nullptr;  // cancelled by Stop
    friend class Session;
};

}  // namespace cyxwiz::servernode
