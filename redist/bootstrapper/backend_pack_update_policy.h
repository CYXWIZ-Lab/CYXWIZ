#pragma once

#include "backend_pack_metadata_verifier.h"
#include <optional>
#include <string_view>

namespace cyxwiz::runtime {

enum class BackendPackUpdateDisposition {
  SamePackage,
  Upgrade,
  Downgrade,
  Ambiguous,
  Unknown
};
struct BackendPackUpdateDecision {
  BackendPackUpdateDisposition disposition =
      BackendPackUpdateDisposition::Unknown;
  std::string message;
};

// Bounded dotted numeric versions; missing trailing components are zero.
// Labels, wildcards, leading zeroes and overflow are deliberately not guessed.
std::optional<int> CompareBackendPackVersions(std::string_view left,
                                              std::string_view right);
BackendPackUpdateDecision
EvaluateBackendPackUpdate(const VerifiedBackendPackManifest &installed,
                          const VerifiedBackendPackManifest &candidate);

} // namespace cyxwiz::runtime
