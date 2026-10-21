#pragma once

#include <mutex>
#include <optional>
#include <shared_mutex>

namespace cyxwiz::runtime {

inline std::shared_mutex& RuntimeMutationMutex() {
    static std::shared_mutex mutex;
    return mutex;
}

inline int& RuntimeMutationDepth() {
    thread_local int depth = 0;
    return depth;
}

class RuntimeExecutionLease {
public:
    RuntimeExecutionLease() : lock_(RuntimeMutationMutex()) {}

    RuntimeExecutionLease(const RuntimeExecutionLease&) = delete;
    RuntimeExecutionLease& operator=(const RuntimeExecutionLease&) = delete;

private:
    std::shared_lock<std::shared_mutex> lock_;
};

class RuntimeMutationLease {
public:
    RuntimeMutationLease() {
        if (RuntimeMutationDepth() > 0) {
            ++RuntimeMutationDepth();
            owns_ = true;
            return;
        }
        lock_.emplace(RuntimeMutationMutex(), std::try_to_lock);
        if (lock_->owns_lock()) {
            RuntimeMutationDepth() = 1;
            owns_ = true;
        }
    }

    RuntimeMutationLease(const RuntimeMutationLease&) = delete;
    RuntimeMutationLease& operator=(const RuntimeMutationLease&) = delete;

    ~RuntimeMutationLease() {
        if (!owns_) return;
        --RuntimeMutationDepth();
        if (RuntimeMutationDepth() == 0) lock_.reset();
    }

    bool OwnsMutation() const { return owns_; }

private:
    bool owns_ = false;
    std::optional<std::unique_lock<std::shared_mutex>> lock_;
};

}  // namespace cyxwiz::runtime
