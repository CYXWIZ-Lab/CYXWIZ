#pragma once

// The run execution policy (Preferences > Devices, TOFIX140 F9). All compute
// runs on ArrayFire; the policy only says how far a run may stray from the
// selected device:
//   Compatibility - an unqualified GPU may recover on ArrayFire's CPU backend,
//                   stages the placement plan does not put on the device still
//                   run, and the first batch's debug samples are dumped.
//   Strict        - the run stays on the selected device: no recovery, the
//                   placement plan must be fully on the device, no dumps.

#include <optional>
#include <string>

namespace cyxwiz {

enum class ExecutionPolicy {
    Compatibility,
    Strict,
};

// Saved name ("compatibility" / "strict").
inline const char* ExecutionPolicyName(ExecutionPolicy policy) {
    return policy == ExecutionPolicy::Strict ? "strict" : "compatibility";
}

inline const char* ExecutionPolicyDisplayName(ExecutionPolicy policy) {
    return policy == ExecutionPolicy::Strict
        ? "Strict (stays on the selected device)"
        : "Compatibility (may finish on ArrayFire CPU)";
}

inline std::optional<ExecutionPolicy> ParseExecutionPolicy(const std::string& name) {
    if (name == "compatibility") return ExecutionPolicy::Compatibility;
    if (name == "strict") return ExecutionPolicy::Strict;
    return std::nullopt;
}

// The display name of a recorded policy name; "Not recorded" when empty or
// unknown.
inline const char* RecordedExecutionPolicyDisplayName(const std::string& name) {
    const auto policy = ParseExecutionPolicy(name);
    return policy ? ExecutionPolicyDisplayName(*policy) : "Not recorded";
}

}  // namespace cyxwiz
