#pragma once

#include "materialization_memory_types.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#ifdef LoadImage
#undef LoadImage
#endif
#elif defined(__linux__)
#include <sys/sysinfo.h>
#elif defined(__APPLE__)
#include <mach/mach.h>
#include <mach/vm_statistics.h>
#include <sys/sysctl.h>
#endif

namespace cyxwiz {

inline const char* MaterializationMemoryRiskName(
    MaterializationMemoryRisk risk) {
    switch (risk) {
    case MaterializationMemoryRisk::Safe:
        return "safe";
    case MaterializationMemoryRisk::Warning:
        return "warning";
    case MaterializationMemoryRisk::Risky:
        return "risky";
    case MaterializationMemoryRisk::Blocked:
        return "blocked";
    }
    return "unknown";
}

inline const char* MaterializationMemoryRiskToProgressStatus(
    MaterializationMemoryRisk risk) {
    switch (risk) {
    case MaterializationMemoryRisk::Safe:
        return "running";
    case MaterializationMemoryRisk::Warning:
        return "warning";
    case MaterializationMemoryRisk::Risky:
        return "risky";
    case MaterializationMemoryRisk::Blocked:
        return "blocked";
    }
    return "running";
}

inline bool CheckedMulU64(uint64_t a, uint64_t b, uint64_t& out) {
    if (a == 0 || b == 0) {
        out = 0;
        return true;
    }
    if (a > std::numeric_limits<uint64_t>::max() / b) {
        out = std::numeric_limits<uint64_t>::max();
        return false;
    }
    out = a * b;
    return true;
}

inline bool CheckedAddU64(uint64_t a, uint64_t b, uint64_t& out) {
    if (a > std::numeric_limits<uint64_t>::max() - b) {
        out = std::numeric_limits<uint64_t>::max();
        return false;
    }
    out = a + b;
    return true;
}

inline uint64_t SaturatingScaleBytes(uint64_t bytes, double multiplier) {
    if (bytes == 0 || multiplier <= 0.0) {
        return 0;
    }
    const long double scaled =
        static_cast<long double>(bytes) * static_cast<long double>(multiplier);
    const long double limit =
        static_cast<long double>(std::numeric_limits<uint64_t>::max());
    if (scaled >= limit) {
        return std::numeric_limits<uint64_t>::max();
    }
    return static_cast<uint64_t>(std::ceil(scaled));
}

inline MaterializationMemoryEstimate EstimateDenseMaterializationMemory(
    uint64_t rows,
    uint64_t output_features,
    uint64_t bytes_per_value) {
    MaterializationMemoryEstimate estimate;
    estimate.rows = rows;
    estimate.output_features = output_features;
    estimate.bytes_per_value = bytes_per_value;

    uint64_t cells = 0;
    estimate.overflow = !CheckedMulU64(rows, output_features, cells);
    if (!CheckedMulU64(cells, bytes_per_value, estimate.raw_output_bytes)) {
        estimate.overflow = true;
    }

    estimate.temporary_bytes =
        SaturatingScaleBytes(estimate.raw_output_bytes, 2.0);
    estimate.arrow_overhead_bytes =
        SaturatingScaleBytes(estimate.raw_output_bytes, 0.125);

    uint64_t peak = 0;
    if (!CheckedAddU64(
            estimate.raw_output_bytes, estimate.temporary_bytes, peak)) {
        estimate.overflow = true;
    }
    if (!CheckedAddU64(peak, estimate.arrow_overhead_bytes, peak)) {
        estimate.overflow = true;
    }
    estimate.estimated_peak_bytes = estimate.overflow
        ? std::numeric_limits<uint64_t>::max()
        : peak;
    return estimate;
}

inline MaterializationMemorySnapshot DetectMaterializationMemorySnapshot() {
    MaterializationMemorySnapshot snapshot;
#ifdef _WIN32
    MEMORYSTATUSEX mem_info;
    mem_info.dwLength = sizeof(MEMORYSTATUSEX);
    if (GlobalMemoryStatusEx(&mem_info)) {
        snapshot.total_bytes = static_cast<uint64_t>(mem_info.ullTotalPhys);
        snapshot.available_bytes = static_cast<uint64_t>(mem_info.ullAvailPhys);
        snapshot.detected = snapshot.available_bytes > 0;
    }
#elif defined(__linux__)
    struct sysinfo si;
    if (sysinfo(&si) == 0) {
        snapshot.total_bytes =
            static_cast<uint64_t>(si.totalram) *
            static_cast<uint64_t>(si.mem_unit);
        snapshot.available_bytes =
            static_cast<uint64_t>(si.freeram) *
            static_cast<uint64_t>(si.mem_unit);
        snapshot.detected = snapshot.available_bytes > 0;
    }
#elif defined(__APPLE__)
    int64_t total = 0;
    size_t total_size = sizeof(total);
    if (sysctlbyname("hw.memsize", &total, &total_size, nullptr, 0) == 0 &&
        total > 0) {
        snapshot.total_bytes = static_cast<uint64_t>(total);
    }

    vm_size_t page_size = 0;
    host_page_size(mach_host_self(), &page_size);
    vm_statistics64_data_t vmstat;
    mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
    if (host_statistics64(mach_host_self(), HOST_VM_INFO64,
                          reinterpret_cast<host_info64_t>(&vmstat),
                          &count) == KERN_SUCCESS) {
        snapshot.available_bytes =
            static_cast<uint64_t>(vmstat.free_count + vmstat.inactive_count) *
            static_cast<uint64_t>(page_size);
        snapshot.detected = snapshot.available_bytes > 0;
    }
#endif
    return snapshot;
}

inline MaterializationMemoryDecision EvaluateMaterializationMemory(
    const MaterializationMemoryEstimate& estimate,
    MaterializationMemorySnapshot snapshot,
    const MaterializationMemoryPolicy& policy = {}) {
    MaterializationMemoryDecision decision;
    if (!snapshot.detected || snapshot.available_bytes == 0) {
        snapshot.available_bytes = policy.fallback_available_bytes;
    }

    decision.available_bytes = snapshot.available_bytes;
    decision.safe_budget_bytes =
        SaturatingScaleBytes(snapshot.available_bytes, policy.blocked_fraction);

    if (estimate.overflow) {
        decision.risk = MaterializationMemoryRisk::Blocked;
        decision.blocked = true;
        decision.reason = "estimated dense output size overflowed 64-bit bytes";
    } else if (policy.hard_limit_bytes > 0 &&
               estimate.estimated_peak_bytes > policy.hard_limit_bytes) {
        decision.risk = MaterializationMemoryRisk::Blocked;
        decision.blocked = true;
        decision.reason = "estimated peak exceeds configured hard limit";
    } else if (snapshot.available_bytes == 0) {
        decision.risk = MaterializationMemoryRisk::Blocked;
        decision.blocked = true;
        decision.reason = "available system memory is unknown";
    } else {
        decision.available_fraction =
            static_cast<double>(estimate.estimated_peak_bytes) /
            static_cast<double>(snapshot.available_bytes);
        if (decision.available_fraction >= policy.blocked_fraction) {
            decision.risk = MaterializationMemoryRisk::Blocked;
            decision.blocked = true;
            decision.reason = "estimated peak exceeds safe available RAM budget";
        } else if (decision.available_fraction >= policy.risky_fraction) {
            decision.risk = MaterializationMemoryRisk::Risky;
            decision.reason = "estimated peak is close to available RAM";
        } else if (decision.available_fraction >= policy.warning_fraction) {
            decision.risk = MaterializationMemoryRisk::Warning;
            decision.reason = "estimated peak is a material share of available RAM";
        } else {
            decision.risk = MaterializationMemoryRisk::Safe;
            decision.reason = "estimated peak is within available RAM budget";
        }
    }

    decision.suggestion =
        "Reduce output dimensions or input rows, or use a sparse/chunked "
        "materialization path when supported.";
    return decision;
}

inline MaterializationMemoryDecision EvaluateMaterializationMemory(
    const MaterializationMemoryEstimate& estimate,
    const MaterializationMemoryContext& context) {
    const auto snapshot = context.snapshot_override.has_value()
        ? *context.snapshot_override
        : DetectMaterializationMemorySnapshot();
    return EvaluateMaterializationMemory(estimate, snapshot, context.policy);
}

inline std::string FormatMaterializationBytes(uint64_t bytes) {
    const char* units[] = {"B", "KB", "MB", "GB", "TB"};
    double value = static_cast<double>(bytes);
    size_t unit = 0;
    while (value >= 1024.0 && unit + 1 < (sizeof(units) / sizeof(units[0]))) {
        value /= 1024.0;
        ++unit;
    }

    std::ostringstream out;
    if (unit == 0) {
        out << bytes << " " << units[unit];
    } else {
        out << std::fixed << std::setprecision(value >= 10.0 ? 1 : 2)
            << value << " " << units[unit];
    }
    return out.str();
}

inline std::string BuildMaterializationMemoryPreflightMessage(
    const std::string& operation,
    const std::string& dimension_name,
    const MaterializationMemoryEstimate& estimate,
    const MaterializationMemoryDecision& decision,
    const std::string& suggestion = {}) {
    std::ostringstream out;
    out << operation << " memory preflight: risk="
        << MaterializationMemoryRiskName(decision.risk)
        << ", rows=" << estimate.rows
        << ", " << dimension_name << "=" << estimate.output_features
        << ", raw=" << FormatMaterializationBytes(estimate.raw_output_bytes)
        << ", estimated_peak="
        << FormatMaterializationBytes(estimate.estimated_peak_bytes)
        << ", available="
        << FormatMaterializationBytes(decision.available_bytes)
        << ", safe_budget="
        << FormatMaterializationBytes(decision.safe_budget_bytes)
        << ". " << decision.reason
        << ". Suggestion: "
        << (suggestion.empty() ? decision.suggestion : suggestion);
    return out.str();
}

} // namespace cyxwiz
