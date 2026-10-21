#pragma once

#include <cstdint>
#include <optional>
#include <string>

namespace cyxwiz {

enum class MaterializationMemoryRisk {
    Safe,
    Warning,
    Risky,
    Blocked,
};

struct MaterializationMemorySnapshot {
    uint64_t total_bytes = 0;
    uint64_t available_bytes = 0;
    bool detected = false;
};

struct MaterializationMemoryEstimate {
    uint64_t rows = 0;
    uint64_t output_features = 0;
    uint64_t bytes_per_value = 0;
    uint64_t raw_output_bytes = 0;
    uint64_t temporary_bytes = 0;
    uint64_t arrow_overhead_bytes = 0;
    uint64_t estimated_peak_bytes = 0;
    bool overflow = false;
    std::string confidence = "medium";
};

struct MaterializationMemoryPolicy {
    double warning_fraction = 0.40;
    double risky_fraction = 0.70;
    double blocked_fraction = 0.90;
    uint64_t hard_limit_bytes = 0;
    uint64_t fallback_available_bytes = 2ULL * 1024ULL * 1024ULL * 1024ULL;
};

struct MaterializationMemoryDecision {
    MaterializationMemoryRisk risk = MaterializationMemoryRisk::Safe;
    bool blocked = false;
    uint64_t available_bytes = 0;
    uint64_t safe_budget_bytes = 0;
    double available_fraction = 0.0;
    std::string reason;
    std::string suggestion;
};

// Production leaves snapshot_override empty so each preflight observes current
// system memory. Tests and future preflight UI may supply a fixed snapshot and
// policy without changing process-global state.
struct MaterializationMemoryContext {
    MaterializationMemoryPolicy policy;
    std::optional<MaterializationMemorySnapshot> snapshot_override;
};

} // namespace cyxwiz
