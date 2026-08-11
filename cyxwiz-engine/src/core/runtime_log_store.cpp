#include "runtime_log_store.h"

#include <algorithm>
#include <cctype>
#include <stdexcept>

namespace cyxwiz {
namespace {

bool Fits(std::string_view value, size_t maximum) {
    return value.size() <= maximum;
}

} // namespace

std::string_view RuntimeLogLevelName(RuntimeLogLevel level) {
    switch (level) {
        case RuntimeLogLevel::Trace: return "trace";
        case RuntimeLogLevel::Debug: return "debug";
        case RuntimeLogLevel::Info: return "info";
        case RuntimeLogLevel::Warning: return "warn";
        case RuntimeLogLevel::Error: return "error";
        case RuntimeLogLevel::Critical: return "critical";
    }
    return "unknown";
}

bool IsCanonicalRuntimeLogCategory(std::string_view category) {
    if (category.empty() || category.size() > 64) {
        return false;
    }
    return std::all_of(category.begin(), category.end(), [](unsigned char c) {
        return (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') ||
               c == '_';
    });
}

bool IsCanonicalDiagnosticCode(std::string_view code) {
    if (code.size() != 9 || code[0] != 'C' || code[1] != 'W' ||
        code[2] != '-' || code[4] != '-') {
        return false;
    }
    if (code[3] < 'A' || code[3] > 'Z') {
        return false;
    }
    return std::all_of(code.begin() + 5, code.end(), [](unsigned char c) {
        return std::isdigit(c) != 0;
    });
}

bool IsCanonicalDiagnosticFamily(std::string_view family) {
    return family.size() == 6 && family[0] == 'C' && family[1] == 'W' &&
           family[2] == '-' && family[3] >= 'A' && family[3] <= 'Z' &&
           family[4] == '-' && family[5] == '*';
}

std::optional<std::string> NormalizeDiagnosticCode(std::string_view code) {
    std::string normalized(code);
    std::transform(normalized.begin(), normalized.end(), normalized.begin(),
                   [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
    if (!IsCanonicalDiagnosticCode(normalized)) {
        return std::nullopt;
    }
    return normalized;
}

std::optional<std::string> NormalizeDiagnosticFamily(std::string_view family) {
    std::string normalized(family);
    std::transform(normalized.begin(), normalized.end(), normalized.begin(),
                   [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
    if (!IsCanonicalDiagnosticFamily(normalized)) {
        return std::nullopt;
    }
    return normalized;
}

std::optional<std::string> ExtractLegacyDiagnosticCode(
    std::string_view message) {
    if (message.size() < 11 || message.front() != '[' || message[10] != ']') {
        return std::nullopt;
    }

    const auto code = message.substr(1, 9);
    if (!IsCanonicalDiagnosticCode(code)) {
        return std::nullopt;
    }
    return std::string(code);
}

RuntimeLogStore::RuntimeLogStore(size_t capacity)
    : capacity_(capacity), slots_(capacity) {
    if (capacity == 0) {
        throw std::invalid_argument("RuntimeLogStore capacity must be positive");
    }
}

bool RuntimeLogStore::Append(RuntimeLogEvent event) noexcept {
    try {
        if (event.category.empty()) {
            event.category = "system";
        }
        if (event.source.empty()) {
            event.source = "unknown";
        }
        if (event.timestamp_utc.time_since_epoch().count() == 0) {
            event.timestamp_utc = std::chrono::system_clock::now();
        }

        if (!Validate(event)) {
            rejected_count_.fetch_add(1, std::memory_order_relaxed);
            return false;
        }

        std::lock_guard<std::mutex> lock(mutex_);
        event.sequence = next_sequence_++;

        if (size_ < capacity_) {
            const size_t index = (head_ + size_) % capacity_;
            slots_[index] = std::move(event);
            ++size_;
        } else {
            slots_[head_] = std::move(event);
            head_ = (head_ + 1) % capacity_;
            ++evicted_count_;
        }
        return true;
    } catch (...) {
        dropped_count_.fetch_add(1, std::memory_order_relaxed);
        return false;
    }
}

RuntimeLogSnapshot RuntimeLogStore::Snapshot(
    const RuntimeLogSnapshotRequest& request) const {
    RuntimeLogSnapshot snapshot;
    std::lock_guard<std::mutex> lock(mutex_);
    snapshot.stats = GetStatsLocked();
    snapshot.events.reserve(std::min(request.limit, size_));

    for (size_t offset = 0; offset < size_; ++offset) {
        const auto& event = slots_[(head_ + offset) % capacity_];
        if (event.sequence <= request.after_sequence ||
            event.sequence > request.through_sequence) {
            continue;
        }
        if (snapshot.events.size() == request.limit) {
            snapshot.truncated = true;
            break;
        }
        snapshot.events.push_back(event);
    }
    return snapshot;
}

RuntimeLogStoreStats RuntimeLogStore::GetStats() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return GetStatsLocked();
}

void RuntimeLogStore::RecordDropped(uint64_t count) noexcept {
    dropped_count_.fetch_add(count, std::memory_order_relaxed);
}

void RuntimeLogStore::RecordSuppressed(uint64_t count) noexcept {
    suppressed_count_.fetch_add(count, std::memory_order_relaxed);
}

RuntimeLogStore& RuntimeLogStore::Instance() {
    static RuntimeLogStore store;
    return store;
}

bool RuntimeLogStore::Validate(const RuntimeLogEvent& event) const noexcept {
    if (event.level < RuntimeLogLevel::Trace ||
        event.level > RuntimeLogLevel::Critical) {
        return false;
    }
    if (!IsCanonicalRuntimeLogCategory(event.category) ||
        !Fits(event.source, 256) ||
        !Fits(event.event_name, 128) ||
        !Fits(event.run_id, 256) ||
        !Fits(event.thread_id, 64) ||
        !Fits(event.backend, 128) ||
        !Fits(event.device_name, 256) ||
        !Fits(event.dataset_name, 1024) ||
        !Fits(event.diagnostic_phase, 128) ||
        !Fits(event.component, 256) ||
        event.message.size() > kMaxMessageBytes ||
        event.issue_codes.size() > kMaxIssueCodes ||
        event.details.size() > kMaxDetails) {
        return false;
    }

    if (!event.primary_error_code.empty() &&
        !IsCanonicalDiagnosticCode(event.primary_error_code)) {
        return false;
    }
    for (const auto& code : event.issue_codes) {
        if (!IsCanonicalDiagnosticCode(code)) {
            return false;
        }
    }
    size_t text_bytes = event.category.size() + event.source.size() +
                        event.event_name.size() + event.run_id.size() +
                        event.thread_id.size() + event.backend.size() +
                        event.device_name.size() + event.dataset_name.size() +
                        event.primary_error_code.size() +
                        event.diagnostic_phase.size() + event.component.size() +
                        event.message.size();
    for (const auto& code : event.issue_codes) {
        text_bytes += code.size();
    }
    for (const auto& [key, value] : event.details) {
        if (!Fits(key, 128) || !Fits(value, 4096)) {
            return false;
        }
        text_bytes += key.size() + value.size();
    }
    return text_bytes <= kMaxEventTextBytes;
}

RuntimeLogStoreStats RuntimeLogStore::GetStatsLocked() const {
    RuntimeLogStoreStats stats;
    stats.capacity = capacity_;
    stats.size = size_;
    if (size_ > 0) {
        stats.oldest_sequence = slots_[head_].sequence;
        stats.newest_sequence =
            slots_[(head_ + size_ - 1) % capacity_].sequence;
    }
    stats.evicted_count = evicted_count_;
    stats.dropped_count = dropped_count_.load(std::memory_order_relaxed);
    stats.rejected_count = rejected_count_.load(std::memory_order_relaxed);
    stats.suppressed_count = suppressed_count_.load(std::memory_order_relaxed);
    return stats;
}

} // namespace cyxwiz
