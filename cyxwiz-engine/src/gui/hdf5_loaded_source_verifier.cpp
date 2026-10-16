#include "hdf5_loaded_source_verifier.h"
#include "../core/async_task_manager.h"

#include <atomic>
#include <utility>

namespace gui {
struct Hdf5LoadedSourceVerifier::State {
    cyxwiz::Hdf5RegisteredSourceRequest request;
    std::atomic<bool> done{false};
    cyxwiz::Hdf5RegisteredSourceResult result;
};

Hdf5LoadedSourceVerifier::~Hdf5LoadedSourceVerifier() { Cancel(); }

void Hdf5LoadedSourceVerifier::Cancel() {
    if (task_) {
        const auto state = task_->GetState();
        if (state != cyxwiz::TaskState::Completed && state != cyxwiz::TaskState::Failed &&
            state != cyxwiz::TaskState::Cancelled) task_->RequestCancel();
    }
    task_.reset();
    state_.reset();
    request_ = {};
    dataset_.reset();
    owner_.reset();
    owner_bound_ = false;
    result_ = {};
}

void Hdf5LoadedSourceVerifier::Start(cyxwiz::Hdf5RegisteredSourceRequest request,
                                   std::weak_ptr<const void> owner) {
    Cancel();
    owner_ = std::move(owner);
    const std::weak_ptr<const void> empty;
    owner_bound_ = owner_.owner_before(empty) || empty.owner_before(owner_);
    if (!OwnerAlive()) {
        result_.status = cyxwiz::Hdf5TableStatus::Cancelled;
        result_.error = "HDF5 project session is no longer active";
        return;
    }
    request_ = request;
    dataset_ = request_.dataset;
    request_.dataset.reset();
    try {
        state_ = std::make_shared<State>();
        state_->request = std::move(request);
        task_ = std::make_shared<cyxwiz::LambdaTask>("Verify HDF5 loaded source",
            [storage = std::weak_ptr<State>(state_)](cyxwiz::LambdaTask& task) {
                auto state = storage.lock();
                if (!state || state->done.load(std::memory_order_acquire)) return;
                // A queued cancellation may skip Execute entirely. Only the
                // dialog/state or a running worker may keep this table alive.
                auto request = std::move(state->request);
                state->result = cyxwiz::VerifyHdf5RegisteredSource(request, [&] { return task.ShouldStop(); });
                if (task.ShouldStop()) {
                    state->result = {};
                    state->result.status = cyxwiz::Hdf5TableStatus::Cancelled;
                    state->result.error = "HDF5 loaded-state verification cancelled";
                }
                if (state->result.status == cyxwiz::Hdf5TableStatus::Ok) task.MarkCompleted("HDF5 source identity verified");
                else if (state->result.status == cyxwiz::Hdf5TableStatus::Cancelled) task.MarkCancelled(state->result.error);
                else task.MarkFailed(state->result.error);
                state->done.store(true, std::memory_order_release);
            }, true);
        if (owner_bound_) task_->BindOwner(owner_);
        if (cyxwiz::AsyncTaskManager::Instance().Submit(task_) == 0) {
            Cancel();
            result_.error = "Cannot schedule HDF5 loaded-state verification";
        }
    } catch (const std::exception& error) {
        Cancel();
        result_.error = std::string("Cannot start HDF5 loaded-state verification: ") + error.what();
    }
}

bool Hdf5LoadedSourceVerifier::Matches(const cyxwiz::Hdf5RegisteredSourceRequest& current) const {
    return current.dataset && dataset_.lock() == current.dataset &&
        request_.dataset_name == current.dataset_name && request_.resolved_path == current.resolved_path &&
        request_.registered_source_path == current.registered_source_path &&
        request_.settings.selection.data_path == current.settings.selection.data_path &&
        request_.settings.selection.label_path == current.settings.selection.label_path &&
        request_.settings.numeric_policy == current.settings.numeric_policy &&
        request_.settings.max_materialized_bytes == current.settings.max_materialized_bytes;
}

bool Hdf5LoadedSourceVerifier::Poll(const cyxwiz::Hdf5RegisteredSourceRequest& current) {
    if (!state_) return false;
    if (!OwnerAlive() || !Matches(current)) {
        Cancel();
        result_.error = "HDF5 loaded-state verification discarded: source, selection or registration changed";
        return true;
    }
    if (!state_->done.load(std::memory_order_acquire)) {
        const auto state = task_->GetState();
        if (state != cyxwiz::TaskState::Cancelled && state != cyxwiz::TaskState::Failed) return false;
        Cancel();
        result_.status = state == cyxwiz::TaskState::Cancelled
            ? cyxwiz::Hdf5TableStatus::Cancelled : cyxwiz::Hdf5TableStatus::ReadFailed;
        // Task terminal state can precede its message writes. Detailed worker
        // diagnostics are read only through the release/acquire result above.
        result_.error = state == cyxwiz::TaskState::Cancelled
            ? "HDF5 loaded-state verification cancelled before result delivery"
            : "HDF5 loaded-state verification failed before result delivery";
        return true;
    }
    auto result = std::move(state_->result);
    Cancel();
    result_ = std::move(result);
    return true;
}
} // namespace gui
