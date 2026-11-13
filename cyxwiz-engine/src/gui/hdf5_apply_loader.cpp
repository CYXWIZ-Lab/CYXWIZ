#include "hdf5_apply_loader.h"
#include "../core/async_task_manager.h"

#include <utility>

namespace gui {

Hdf5ApplyLoader::~Hdf5ApplyLoader() { Cancel(); }

void Hdf5ApplyLoader::Cancel() {
    if (task_) {
        const auto state = task_->GetState();
        if (state != cyxwiz::TaskState::Completed && state != cyxwiz::TaskState::Failed &&
            state != cyxwiz::TaskState::Cancelled) task_->RequestCancel();
    }
    task_.reset();
    state_.reset();
    owner_.reset();
    owner_bound_ = false;
    result_ = {};
}

void Hdf5ApplyLoader::Start(cyxwiz::Hdf5ApplyPreparationRequest request, std::weak_ptr<const void> owner) {
    Cancel();
    owner_ = std::move(owner);
    const std::weak_ptr<const void> empty;
    owner_bound_ = owner_.owner_before(empty) || empty.owner_before(owner_);
    if (!OwnerAlive()) {
        result_.status = cyxwiz::Hdf5TableStatus::Cancelled;
        result_.error = "HDF5 project session is no longer active";
        return;
    }
    try {
        state_ = std::make_shared<cyxwiz::Hdf5ApplyPreparationState>();
        task_ = cyxwiz::MakeHdf5ApplyPreparationTask(std::move(request), state_);
        if (owner_bound_) task_->BindOwner(owner_);
        if (cyxwiz::AsyncTaskManager::Instance().Submit(task_) == 0) {
            Cancel();
            result_.error = "Cannot schedule the HDF5 load";
        }
    } catch (const std::exception& error) {
        Cancel();
        result_.error = std::string("Cannot start the HDF5 load: ") + error.what();
    }
}

bool Hdf5ApplyLoader::Poll() {
    if (!state_ || !state_->done.load(std::memory_order_acquire)) return false;
    result_ = std::move(state_->result);
    task_.reset();
    state_.reset();
    return true;
}
} // namespace gui
