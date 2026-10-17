#pragma once

// The data budget (TOFIX154): one number for how much memory data may use.
// It decides when a table is opened from disk instead of loaded, when a step
// may read a disk table whole, and DuckDB's memory_limit / temp_directory.
// Default: half of physical RAM (owner decision D2). Preferences, tests and
// the benchmark set it with SetDataBudgetBytes; 0 restores the default.

#include "materialization_memory_guard.h"

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <mutex>
#include <string>

namespace cyxwiz {

namespace data_budget_detail {
inline std::atomic<uint64_t>& Override() {
    static std::atomic<uint64_t> bytes{0};
    return bytes;
}
inline std::mutex& SpillMutex() {
    static std::mutex mutex;
    return mutex;
}
inline std::string& SpillDirectory() {
    static std::string directory;
    return directory;
}
inline std::atomic<uint32_t>& Sharers() {
    static std::atomic<uint32_t> steps{0};
    return steps;
}
}  // namespace data_budget_detail

inline uint64_t PhysicalMemoryBytes() {
    static const uint64_t total = [] {
        const auto snapshot = DetectMaterializationMemorySnapshot();
        return snapshot.total_bytes > 0 ? snapshot.total_bytes : 8ULL * 1024 * 1024 * 1024;
    }();
    return total;
}

inline uint64_t DefaultDataBudgetBytes() { return PhysicalMemoryBytes() / 2; }

// The compute device's start-up memory (TOFIX157 A1): starting ArrayFire
// takes about 0.5-0.9 GB on any backend (measured in TOFIX154, done154
// section 10), apart from the data. The budget never reaches into it, so a
// budget near the machine's memory still leaves the device room to start.
inline constexpr uint64_t kDeviceStartupReserveBytes = 1ULL << 30;

// What the user set (0: the default), before the device reserve.
inline uint64_t RequestedDataBudgetBytes() {
    const uint64_t set = data_budget_detail::Override().load();
    return set > 0 ? set : DefaultDataBudgetBytes();
}

// The most data may use: this PC's memory less the device reserve.
inline uint64_t MaxDataBudgetBytes() {
    const uint64_t total = PhysicalMemoryBytes();
    return total > kDeviceStartupReserveBytes * 2 ? total - kDeviceStartupReserveBytes : total / 2;
}

// What data may use: the requested budget, kept out of the device reserve.
inline uint64_t DataBudgetBytes() {
    const uint64_t requested = RequestedDataBudgetBytes();
    const uint64_t most = MaxDataBudgetBytes();
    return requested < most ? requested : most;
}

inline void SetDataBudgetBytes(uint64_t bytes) { data_budget_detail::Override().store(bytes); }

inline bool DataBudgetIsDefault() { return data_budget_detail::Override().load() == 0; }

// Steps running at once share the budget (TOFIX157 A2): a pipeline node
// holds a DataBudgetShare while it runs, and what one step may use (DuckDB's
// memory and threads, reading a disk table whole) is the budget divided
// between the steps running. One step alone has the whole budget.
class DataBudgetShare {
public:
    DataBudgetShare() { ++data_budget_detail::Sharers(); }
    ~DataBudgetShare() { --data_budget_detail::Sharers(); }
    DataBudgetShare(const DataBudgetShare&) = delete;
    DataBudgetShare& operator=(const DataBudgetShare&) = delete;
};

inline uint64_t DataBudgetShareBytes() {
    const uint32_t steps = data_budget_detail::Sharers().load();
    return DataBudgetBytes() / (steps > 1 ? steps : 1);
}

// A table or result larger than this is kept on disk, not in memory (D1).
inline uint64_t DataResultThresholdBytes() { return DataBudgetBytes() / 4; }

// Where DuckDB spills sorts, joins and group-bys. The project sets
// <project>/cache/duckdb_tmp; outside a project, <temp>/cyxwiz/duckdb_tmp.
inline std::string DataSpillDirectory() {
    std::lock_guard<std::mutex> lock(data_budget_detail::SpillMutex());
    if (!data_budget_detail::SpillDirectory().empty()) return data_budget_detail::SpillDirectory();
    std::error_code error;
    const auto temp = std::filesystem::temp_directory_path(error);
    return error ? std::string() : (temp / "cyxwiz" / "duckdb_tmp").string();
}

inline void SetDataSpillDirectory(std::string directory) {
    std::lock_guard<std::mutex> lock(data_budget_detail::SpillMutex());
    data_budget_detail::SpillDirectory() = std::move(directory);
}

// "1.5 GB", "512 MB": the unit people see in messages and Preferences.
inline std::string FormatDataBytes(uint64_t bytes) {
    char text[32];
    const double mb = static_cast<double>(bytes) / (1024.0 * 1024.0);
    if (mb >= 1024.0) std::snprintf(text, sizeof(text), "%.1f GB", mb / 1024.0);
    else std::snprintf(text, sizeof(text), "%.0f MB", mb);
    return text;
}

}  // namespace cyxwiz
