#pragma once

#include "../core/hdf5_registered_source.h"

namespace cyxwiz { class AsyncTask; }

namespace gui {

// Dialog-owned verification; never captures a node or mutates the registry.
// Start/Poll/Cancel are UI-thread only. Completed-task history retains no table.
class Hdf5LoadedSourceVerifier {
public:
    Hdf5LoadedSourceVerifier() = default;
    ~Hdf5LoadedSourceVerifier();
    Hdf5LoadedSourceVerifier(const Hdf5LoadedSourceVerifier&) = delete;
    Hdf5LoadedSourceVerifier& operator=(const Hdf5LoadedSourceVerifier&) = delete;
    void Start(cyxwiz::Hdf5RegisteredSourceRequest request, std::weak_ptr<const void> owner = {});
    bool Poll(const cyxwiz::Hdf5RegisteredSourceRequest& current);
    void Cancel();
    bool Busy() const { return task_ != nullptr; }
    bool OwnerAlive() const { return !owner_bound_ || !owner_.expired(); }
    const cyxwiz::Hdf5RegisteredSourceResult& Result() const { return result_; }

private:
    bool Matches(const cyxwiz::Hdf5RegisteredSourceRequest& current) const;
    struct State;
    cyxwiz::Hdf5RegisteredSourceRequest request_; // dataset is held weakly below
    std::weak_ptr<cyxwiz::ArrowDataset> dataset_;
    std::weak_ptr<const void> owner_;
    bool owner_bound_ = false;
    std::shared_ptr<cyxwiz::AsyncTask> task_;
    std::shared_ptr<State> state_;
    cyxwiz::Hdf5RegisteredSourceResult result_;
};
} // namespace gui
