#pragma once

#include "../core/hdf5_apply_preparation.h"

namespace cyxwiz { class AsyncTask; }

namespace gui {

// Dialog-owned HDF5 Apply (TOFIX189): runs MakeHdf5ApplyPreparationTask on a
// worker and hands its staged publication back. Never touches a node or the
// registry; the dialog commits. Start/Poll/Cancel are UI-thread only.
class Hdf5ApplyLoader {
public:
    Hdf5ApplyLoader() = default;
    ~Hdf5ApplyLoader();
    Hdf5ApplyLoader(const Hdf5ApplyLoader&) = delete;
    Hdf5ApplyLoader& operator=(const Hdf5ApplyLoader&) = delete;
    void Start(cyxwiz::Hdf5ApplyPreparationRequest request, std::weak_ptr<const void> owner = {});
    // True once, when the worker has finished; Result() then holds its outcome.
    bool Poll();
    void Cancel();
    bool Busy() const { return task_ != nullptr; }
    bool OwnerAlive() const { return !owner_bound_ || !owner_.expired(); }
    cyxwiz::Hdf5ApplyPreparationResult& Result() { return result_; }

private:
    std::weak_ptr<const void> owner_;
    bool owner_bound_ = false;
    std::shared_ptr<cyxwiz::AsyncTask> task_;
    std::shared_ptr<cyxwiz::Hdf5ApplyPreparationState> state_;
    cyxwiz::Hdf5ApplyPreparationResult result_;
};
} // namespace gui
