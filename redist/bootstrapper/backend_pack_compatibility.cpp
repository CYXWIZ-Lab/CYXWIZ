#include "backend_pack_compatibility.h"

#include <algorithm>
#include <cctype>
#include <iterator>
#include <limits>
#include <optional>
#include <string_view>
#include <vector>

namespace cyxwiz::runtime {
namespace {

int ConfidenceRank(BackendPackIdentityConfidence value) {
  switch (value) {
  case BackendPackIdentityConfidence::Unknown:
    return 0;
  case BackendPackIdentityConfidence::BackendLocal:
    return 1;
  case BackendPackIdentityConfidence::ProviderReported:
    return 2;
  case BackendPackIdentityConfidence::StableHardware:
    return 3;
  }
  return 0;
}

std::optional<BackendPackIdentityConfidence>
ParseConfidence(std::string_view value) {
  if (value == "unknown")
    return BackendPackIdentityConfidence::Unknown;
  if (value == "backend_local")
    return BackendPackIdentityConfidence::BackendLocal;
  if (value == "provider_reported")
    return BackendPackIdentityConfidence::ProviderReported;
  if (value == "stable_hardware")
    return BackendPackIdentityConfidence::StableHardware;
  return std::nullopt;
}

bool Contains(const std::vector<std::string> &values,
              const std::string &value) {
  return std::find(values.begin(), values.end(), value) != values.end();
}

bool ContainsAll(const std::vector<std::string> &values,
                 const std::vector<std::string> &required) {
  return std::all_of(required.begin(), required.end(), [&](const auto &value) {
    return Contains(values, value);
  });
}

bool ContainsAny(const std::vector<std::string> &values,
                 const std::vector<std::string> &allowed) {
  return allowed.empty() ||
         std::any_of(allowed.begin(), allowed.end(), [&](const auto &value) {
           return Contains(values, value);
         });
}

bool ParseNumericVersion(std::string_view value,
                         std::vector<unsigned int> &parts) {
  parts.clear();
  std::size_t begin = 0;
  while (begin < value.size()) {
    const auto end = value.find('.', begin);
    const auto part = value.substr(
        begin, (end == std::string_view::npos ? value.size() : end) - begin);
    if (part.empty() ||
        !std::all_of(part.begin(), part.end(), [](unsigned char character) {
          return std::isdigit(character);
        })) {
      return false;
    }
    unsigned long parsed = 0;
    try {
      parsed = std::stoul(std::string(part));
    } catch (...) {
      return false;
    }
    if (parsed > std::numeric_limits<unsigned int>::max())
      return false;
    parts.push_back(static_cast<unsigned int>(parsed));
    begin = end == std::string_view::npos ? value.size() : end + 1;
  }
  return !parts.empty();
}

std::optional<int> CompareNumericVersions(std::string_view left,
                                          std::string_view right) {
  std::vector<unsigned int> left_parts;
  std::vector<unsigned int> right_parts;
  if (!ParseNumericVersion(left, left_parts) ||
      !ParseNumericVersion(right, right_parts)) {
    return std::nullopt;
  }
  const auto size = std::max(left_parts.size(), right_parts.size());
  for (std::size_t index = 0; index < size; ++index) {
    const auto left_part = index < left_parts.size() ? left_parts[index] : 0U;
    const auto right_part =
        index < right_parts.size() ? right_parts[index] : 0U;
    if (left_part < right_part)
      return -1;
    if (left_part > right_part)
      return 1;
  }
  return 0;
}

std::string_view Trim(std::string_view value) {
  while (!value.empty() &&
         std::isspace(static_cast<unsigned char>(value.front()))) {
    value.remove_prefix(1);
  }
  while (!value.empty() &&
         std::isspace(static_cast<unsigned char>(value.back()))) {
    value.remove_suffix(1);
  }
  return value;
}

std::optional<bool> DriverInTestedRange(std::string_view driver,
                                        std::string_view expression) {
  std::size_t begin = 0;
  bool matched_clause = false;
  while (begin <= expression.size()) {
    const auto end = expression.find(',', begin);
    auto clause = Trim(expression.substr(
        begin,
        (end == std::string_view::npos ? expression.size() : end) - begin));
    if (clause.empty())
      return std::nullopt;
    std::string_view operation = "=";
    if (clause.starts_with(">=") || clause.starts_with("<=") ||
        clause.starts_with("==")) {
      operation = clause.substr(0, 2);
      clause.remove_prefix(2);
    } else if (clause.starts_with('>') || clause.starts_with('<') ||
               clause.starts_with('=')) {
      operation = clause.substr(0, 1);
      clause.remove_prefix(1);
    }
    clause = Trim(clause);
    const auto comparison = CompareNumericVersions(driver, clause);
    if (!comparison)
      return std::nullopt;
    const bool clause_matches = operation == ">="   ? *comparison >= 0
                                : operation == ">"  ? *comparison > 0
                                : operation == "<=" ? *comparison <= 0
                                : operation == "<"  ? *comparison < 0
                                                    : *comparison == 0;
    if (!clause_matches)
      return false;
    matched_clause = true;
    if (end == std::string_view::npos)
      break;
    begin = end + 1;
  }
  return matched_clause;
}

BackendPackInstallRecommendation
Availability(BackendPackSupportStatus support,
             BackendPackEligibility eligibility, bool base_pack = false) {
  if (eligibility == BackendPackEligibility::Incompatible)
    return BackendPackInstallRecommendation::NotOffered;
  if (support == BackendPackSupportStatus::Diagnostic)
    return BackendPackInstallRecommendation::DiagnosticOnly;
  if (support != BackendPackSupportStatus::Supported)
    return BackendPackInstallRecommendation::NotOffered;
  return base_pack && eligibility == BackendPackEligibility::Compatible
             ? BackendPackInstallRecommendation::Available
             : BackendPackInstallRecommendation::AvailableAfterVerification;
}

void SetUnknown(BackendPackCompatibilityDecision &decision,
                BackendPackCompatibilityRule rule,
                BackendPackRemediation remediation) {
  decision.eligibility = BackendPackEligibility::Unknown;
  decision.rule = rule;
  decision.remediation = remediation;
  decision.install_recommendation =
      Availability(decision.catalog_support, decision.eligibility);
}

void SetIncompatible(BackendPackCompatibilityDecision &decision,
                     BackendPackCompatibilityRule rule,
                     BackendPackRemediation remediation) {
  decision.eligibility = BackendPackEligibility::Incompatible;
  decision.rule = rule;
  decision.remediation = remediation;
  decision.install_recommendation =
      BackendPackInstallRecommendation::NotOffered;
}

template <typename Predicate>
std::vector<const BackendPackMatchedDevice *>
Filter(const std::vector<const BackendPackMatchedDevice *> &devices,
       Predicate predicate) {
  std::vector<const BackendPackMatchedDevice *> result;
  std::copy_if(devices.begin(), devices.end(), std::back_inserter(result),
               predicate);
  return result;
}

} // namespace

BackendPackCompatibilityDecision EvaluateBackendPackCompatibility(
    const VerifiedBackendPackManifest &manifest,
    const BackendPackCompatibilityContext &context) {
  BackendPackCompatibilityDecision decision;
  decision.backend = manifest.backend;
  decision.pack_id = manifest.pack_id;
  decision.package_version = manifest.package_version;
  decision.runtime_set_id = manifest.runtime_set_id;
  decision.companion_base_id = manifest.companion_base_id;
  decision.platform = manifest.platform;
  decision.architecture = manifest.architecture;
  decision.arrayfire_abi = manifest.arrayfire_abi;
  decision.catalog_support = manifest.compatibility.support_status;
  decision.recommendation_target_eligible =
      Contains(manifest.compatibility.recommendation_targets,
               manifest.backend);

  if (decision.catalog_support == BackendPackSupportStatus::Blocked ||
      decision.catalog_support == BackendPackSupportStatus::Revoked) {
    SetIncompatible(decision, BackendPackCompatibilityRule::CatalogSupport,
                    BackendPackRemediation::SelectSupportedPack);
    return decision;
  }
  if (context.platform.empty() || context.architecture.empty()) {
    SetUnknown(decision, BackendPackCompatibilityRule::InsufficientFacts,
               BackendPackRemediation::VerifyRoute);
    return decision;
  }
  if (manifest.platform != context.platform) {
    SetIncompatible(decision, BackendPackCompatibilityRule::Platform,
                    BackendPackRemediation::SelectSupportedPack);
    return decision;
  }
  if (manifest.architecture != context.architecture) {
    SetIncompatible(decision, BackendPackCompatibilityRule::Architecture,
                    BackendPackRemediation::SelectSupportedPack);
    return decision;
  }

  const bool base_pack = manifest.kind == BackendPackManifestKind::Base;
  if (base_pack) {
    decision.eligibility = BackendPackEligibility::Compatible;
    decision.rule = BackendPackCompatibilityRule::CompleteMatch;
    decision.remediation = BackendPackRemediation::VerifyRoute;
    decision.install_recommendation =
        Availability(decision.catalog_support, decision.eligibility, true);
    return decision;
  }

  if (context.runtime_set_id.empty() || context.base_pack_id.empty() ||
      context.arrayfire_abi.empty()) {
    SetUnknown(decision, BackendPackCompatibilityRule::InsufficientFacts,
               BackendPackRemediation::InstallMatchingBase);
    return decision;
  }
  if (manifest.runtime_set_id != context.runtime_set_id) {
    SetIncompatible(decision, BackendPackCompatibilityRule::RuntimeSet,
                    BackendPackRemediation::InstallMatchingBase);
    return decision;
  }
  if (manifest.companion_base_id != context.base_pack_id) {
    SetIncompatible(decision, BackendPackCompatibilityRule::CompanionBase,
                    BackendPackRemediation::InstallMatchingBase);
    return decision;
  }
  if (manifest.arrayfire_abi != context.arrayfire_abi) {
    SetIncompatible(decision, BackendPackCompatibilityRule::ArrayFireAbi,
                    BackendPackRemediation::InstallMatchingBase);
    return decision;
  }
  if (context.devices.empty()) {
    SetUnknown(decision, BackendPackCompatibilityRule::InsufficientFacts,
               BackendPackRemediation::VerifyRoute);
    return decision;
  }

  std::vector<const BackendPackMatchedDevice *> candidates;
  candidates.reserve(context.devices.size());
  for (const auto &device : context.devices)
    candidates.push_back(&device);

  const bool unknown_kind =
      std::any_of(candidates.begin(), candidates.end(), [](const auto *device) {
        return device->device_kind == BackendPackDeviceKind::Unknown;
      });
  candidates = Filter(candidates, [&](const auto *device) {
    return Contains(manifest.compatibility.device_kinds,
                    BackendPackDeviceKindName(device->device_kind));
  });
  if (candidates.empty()) {
    if (unknown_kind) {
      SetUnknown(decision, BackendPackCompatibilityRule::DeviceKind,
                 BackendPackRemediation::ImproveDeviceIdentity);
    } else {
      SetIncompatible(decision, BackendPackCompatibilityRule::DeviceKind,
                      BackendPackRemediation::SelectAlternativeBackend);
    }
    return decision;
  }

  if (!manifest.compatibility.provider_types.empty()) {
    const bool missing_provider_facts = std::any_of(
        candidates.begin(), candidates.end(),
        [](const auto *device) { return device->provider_types.empty(); });
    candidates = Filter(candidates, [&](const auto *device) {
      return ContainsAny(device->provider_types,
                         manifest.compatibility.provider_types);
    });
    if (candidates.empty()) {
      if (missing_provider_facts) {
        SetUnknown(decision, BackendPackCompatibilityRule::Provider,
                   BackendPackRemediation::VerifyRoute);
      } else {
        SetIncompatible(decision, BackendPackCompatibilityRule::Provider,
                        BackendPackRemediation::SelectAlternativeBackend);
      }
      return decision;
    }
  }

  if (!manifest.compatibility.cpu_features.empty()) {
    const bool missing_feature_facts = std::any_of(
        candidates.begin(), candidates.end(),
        [](const auto *device) { return device->cpu_features.empty(); });
    candidates = Filter(candidates, [&](const auto *device) {
      return ContainsAll(device->cpu_features,
                         manifest.compatibility.cpu_features);
    });
    if (candidates.empty()) {
      if (missing_feature_facts) {
        SetUnknown(decision, BackendPackCompatibilityRule::CpuFeatures,
                   BackendPackRemediation::VerifyRoute);
      } else {
        SetIncompatible(decision, BackendPackCompatibilityRule::CpuFeatures,
                        BackendPackRemediation::SelectAlternativeBackend);
      }
      return decision;
    }
  }

  const auto minimum_confidence =
      ParseConfidence(manifest.compatibility.minimum_identity_confidence);
  if (!minimum_confidence) {
    SetUnknown(decision, BackendPackCompatibilityRule::InsufficientFacts,
               BackendPackRemediation::ImproveDeviceIdentity);
    return decision;
  }
  candidates = Filter(candidates, [&](const auto *device) {
    return ConfidenceRank(device->identity_confidence) >=
           ConfidenceRank(*minimum_confidence);
  });
  if (candidates.empty()) {
    SetUnknown(decision, BackendPackCompatibilityRule::IdentityConfidence,
               BackendPackRemediation::ImproveDeviceIdentity);
    return decision;
  }

  bool missing_driver_fact = false;
  bool outdated_driver = false;
  const auto driver_candidates = Filter(candidates, [&](const auto *device) {
    const auto minimum =
        manifest.compatibility.minimum_driver_versions.find(device->provider);
    if (minimum == manifest.compatibility.minimum_driver_versions.end())
      return true;
    if (device->driver_version.empty()) {
      missing_driver_fact = true;
      return false;
    }
    const auto comparison =
        CompareNumericVersions(device->driver_version, minimum->second);
    if (!comparison) {
      missing_driver_fact = true;
      return false;
    }
    if (*comparison < 0) {
      outdated_driver = true;
      return false;
    }
    return true;
  });
  candidates = driver_candidates;
  if (candidates.empty()) {
    if (missing_driver_fact) {
      SetUnknown(decision, BackendPackCompatibilityRule::MinimumDriver,
                 BackendPackRemediation::UpdateDriver);
    } else if (outdated_driver) {
      SetIncompatible(decision, BackendPackCompatibilityRule::MinimumDriver,
                      BackendPackRemediation::UpdateDriver);
    } else {
      SetUnknown(decision, BackendPackCompatibilityRule::MinimumDriver,
                 BackendPackRemediation::VerifyRoute);
    }
    return decision;
  }

  const auto best_confidence =
      (*std::max_element(candidates.begin(), candidates.end(),
                         [](const auto *left, const auto *right) {
                           return ConfidenceRank(left->identity_confidence) <
                                  ConfidenceRank(right->identity_confidence);
                         }))
          ->identity_confidence;
  candidates = Filter(candidates, [&](const auto *device) {
    return device->identity_confidence == best_confidence;
  });
  if (candidates.size() != 1) {
    SetUnknown(decision, BackendPackCompatibilityRule::CandidateTie,
               BackendPackRemediation::ImproveDeviceIdentity);
    return decision;
  }

  decision.matched_device = *candidates.front();
  decision.eligibility = BackendPackEligibility::Compatible;
  decision.rule = BackendPackCompatibilityRule::CompleteMatch;
  decision.remediation = BackendPackRemediation::VerifyRoute;
  decision.install_recommendation =
      Availability(decision.catalog_support, decision.eligibility);

  const auto tested = manifest.compatibility.tested_driver_ranges.find(
      decision.matched_device.provider);
  if (tested != manifest.compatibility.tested_driver_ranges.end()) {
    const auto in_range = DriverInTestedRange(
        decision.matched_device.driver_version, tested->second);
    if (!in_range || !*in_range) {
      decision.rule = BackendPackCompatibilityRule::TestedDriverRange;
      decision.remediation = BackendPackRemediation::VerifyRoute;
    }
  }
  return decision;
}

} // namespace cyxwiz::runtime
