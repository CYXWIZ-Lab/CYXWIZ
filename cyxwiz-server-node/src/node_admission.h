#pragma once

// Admission before a job is accepted (TOFIX118 P4b): refuse a job this node
// cannot run - its compute route is not verified, or the job's measured
// memory (JobConfig.estimated_memory, from the Engine) clearly exceeds the
// device - instead of accepting it and failing later.

#include "core/training_failure.h"

#include <spdlog/fmt/fmt.h>

#include <cstdint>
#include <string>

namespace cyxwiz::servernode {

struct AdmissionFacts {
    std::string route;              // e.g. arrayfire_cuda:0
    bool route_verified = false;    // certified in this machine's route evidence
    std::uint64_t device_memory = 0;  // total bytes; 0 = unknown
    std::uint64_t job_memory = 0;     // the job's measured need; 0 = not given
};

struct AdmissionDecision {
    bool accepted = true;
    TrainingFailureKind failure = TrainingFailureKind::None;
    std::string reason;
};

// The measurement counts the job's ArrayFire allocations; the device also
// holds the runtime (CUDA context, BLAS workspace, compiled kernels). Berean
// T0 on a GTX 1050 Ti: 564 MB measured, 858 MB used by the process (2026-09-26).
inline constexpr std::uint64_t kAdmissionRuntimeReserveBytes = 512ull * 1024 * 1024;
// Share of device memory usable at all (display, driver, fragmentation).
inline constexpr double kAdmissionMemoryShare = 0.95;

inline AdmissionDecision EvaluateJobAdmission(const AdmissionFacts& facts) {
    const auto gigabytes = [](std::uint64_t bytes) {
        return fmt::format("{:.1f} GB", static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0));
    };
    if (!facts.route_verified) {
        return {false, TrainingFailureKind::DeviceError,
                "this node's compute route " + facts.route +
                    " is not verified (the operator must verify its devices)"};
    }
    const std::uint64_t need = facts.job_memory + kAdmissionRuntimeReserveBytes;
    if (facts.job_memory > 0 && facts.device_memory > 0 &&
        static_cast<double>(need) > kAdmissionMemoryShare * static_cast<double>(facts.device_memory)) {
        return {false, TrainingFailureKind::OutOfMemory,
                "the job needs about " + gigabytes(need) + " of device memory (training " +
                    gigabytes(facts.job_memory) + " + runtime); " + facts.route + " has " +
                    gigabytes(facts.device_memory)};
    }
    return {};
}

}  // namespace cyxwiz::servernode
