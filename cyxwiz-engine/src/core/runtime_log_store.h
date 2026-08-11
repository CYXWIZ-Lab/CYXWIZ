#pragma once

#include "runtime_log_event.h"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <mutex>
#include <vector>

namespace cyxwiz {

struct RuntimeLogStoreStats {
    size_t capacity = 0;
    size_t size = 0;
    uint64_t oldest_sequence = 0;
    uint64_t newest_sequence = 0;
    uint64_t evicted_count = 0;
    uint64_t dropped_count = 0;
    uint64_t rejected_count = 0;
    uint64_t suppressed_count = 0;
};

struct RuntimeLogSnapshotRequest {
    uint64_t after_sequence = 0;
    uint64_t through_sequence = std::numeric_limits<uint64_t>::max();
    size_t limit = 1000;
};

struct RuntimeLogSnapshot {
    std::vector<RuntimeLogEvent> events;
    RuntimeLogStoreStats stats;
    bool truncated = false;
};

class RuntimeLogStore {
public:
    static constexpr size_t kDefaultCapacity = 4096;
    static constexpr size_t kMaxMessageBytes = 16 * 1024;
    static constexpr size_t kMaxEventTextBytes = 32 * 1024;
    static constexpr size_t kMaxDetails = 32;
    static constexpr size_t kMaxIssueCodes = 32;

    explicit RuntimeLogStore(size_t capacity = kDefaultCapacity);

    bool Append(RuntimeLogEvent event) noexcept;
    RuntimeLogSnapshot Snapshot(
        const RuntimeLogSnapshotRequest& request = {}) const;
    RuntimeLogStoreStats GetStats() const;

    void RecordDropped(uint64_t count = 1) noexcept;
    void RecordSuppressed(uint64_t count = 1) noexcept;

    static RuntimeLogStore& Instance();

private:
    bool Validate(const RuntimeLogEvent& event) const noexcept;
    RuntimeLogStoreStats GetStatsLocked() const;

    const size_t capacity_;
    mutable std::mutex mutex_;
    std::vector<RuntimeLogEvent> slots_;
    size_t head_ = 0;
    size_t size_ = 0;
    uint64_t next_sequence_ = 1;
    uint64_t evicted_count_ = 0;
    std::atomic<uint64_t> dropped_count_{0};
    std::atomic<uint64_t> rejected_count_{0};
    std::atomic<uint64_t> suppressed_count_{0};
};

} // namespace cyxwiz
