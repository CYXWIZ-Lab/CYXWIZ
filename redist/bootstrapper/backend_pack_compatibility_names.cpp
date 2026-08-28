#include "backend_pack_compatibility.h"

namespace cyxwiz::runtime {

const char *BackendPackEligibilityName(BackendPackEligibility value) {
  switch (value) {
  case BackendPackEligibility::Unknown:
    return "unknown";
  case BackendPackEligibility::Compatible:
    return "compatible";
  case BackendPackEligibility::Incompatible:
    return "incompatible";
  }
  return "unknown";
}

const char *
BackendPackInstallRecommendationName(BackendPackInstallRecommendation value) {
  switch (value) {
  case BackendPackInstallRecommendation::NotOffered:
    return "not_offered";
  case BackendPackInstallRecommendation::Available:
    return "available";
  case BackendPackInstallRecommendation::AvailableAfterVerification:
    return "available_after_verification";
  case BackendPackInstallRecommendation::Recommended:
    return "recommended";
  case BackendPackInstallRecommendation::DiagnosticOnly:
    return "diagnostic_only";
  }
  return "not_offered";
}

const char *
BackendPackLocalPackageStateName(BackendPackLocalPackageState value) {
  switch (value) {
  case BackendPackLocalPackageState::NotInstalled:
    return "not_installed";
  case BackendPackLocalPackageState::Staged:
    return "staged";
  case BackendPackLocalPackageState::InstalledInactive:
    return "installed_inactive";
  case BackendPackLocalPackageState::Active:
    return "active";
  case BackendPackLocalPackageState::IntegrityRejected:
    return "integrity_rejected";
  }
  return "not_installed";
}

const char *BackendPackRouteVerificationStatusName(
    BackendPackRouteVerificationStatus value) {
  switch (value) {
  case BackendPackRouteVerificationStatus::NotRun:
    return "not_run";
  case BackendPackRouteVerificationStatus::Passed:
    return "passed";
  case BackendPackRouteVerificationStatus::Failed:
    return "failed";
  case BackendPackRouteVerificationStatus::TimedOut:
    return "timed_out";
  case BackendPackRouteVerificationStatus::Crashed:
    return "crashed";
  case BackendPackRouteVerificationStatus::Stale:
    return "stale";
  }
  return "not_run";
}

const char *BackendPackTrainingAuthorizationStatusName(
    BackendPackTrainingAuthorizationStatus value) {
  switch (value) {
  case BackendPackTrainingAuthorizationStatus::NotEvaluated:
    return "not_evaluated";
  case BackendPackTrainingAuthorizationStatus::Rejected:
    return "rejected";
  case BackendPackTrainingAuthorizationStatus::Authorized:
    return "authorized";
  case BackendPackTrainingAuthorizationStatus::DiagnosticOverride:
    return "diagnostic_override";
  }
  return "not_evaluated";
}

const char *
BackendPackPerformanceStatusName(BackendPackPerformanceStatus value) {
  switch (value) {
  case BackendPackPerformanceStatus::NotMeasured:
    return "not_measured";
  case BackendPackPerformanceStatus::Measured:
    return "measured";
  case BackendPackPerformanceStatus::PreferredMeasured:
    return "preferred_measured";
  }
  return "not_measured";
}

const char *
BackendPackIdentityConfidenceName(BackendPackIdentityConfidence value) {
  switch (value) {
  case BackendPackIdentityConfidence::Unknown:
    return "unknown";
  case BackendPackIdentityConfidence::BackendLocal:
    return "backend_local";
  case BackendPackIdentityConfidence::ProviderReported:
    return "provider_reported";
  case BackendPackIdentityConfidence::StableHardware:
    return "stable_hardware";
  }
  return "unknown";
}

const char *BackendPackDeviceKindName(BackendPackDeviceKind value) {
  switch (value) {
  case BackendPackDeviceKind::Unknown:
    return "unknown";
  case BackendPackDeviceKind::Cpu:
    return "cpu";
  case BackendPackDeviceKind::Gpu:
    return "gpu";
  case BackendPackDeviceKind::Accelerator:
    return "accelerator";
  }
  return "unknown";
}

const char *
BackendPackCompatibilityRuleName(BackendPackCompatibilityRule value) {
  switch (value) {
  case BackendPackCompatibilityRule::None:
    return "none";
  case BackendPackCompatibilityRule::CatalogSupport:
    return "catalog_support";
  case BackendPackCompatibilityRule::Platform:
    return "platform";
  case BackendPackCompatibilityRule::Architecture:
    return "architecture";
  case BackendPackCompatibilityRule::RuntimeSet:
    return "runtime_set";
  case BackendPackCompatibilityRule::CompanionBase:
    return "companion_base";
  case BackendPackCompatibilityRule::ArrayFireAbi:
    return "arrayfire_abi";
  case BackendPackCompatibilityRule::Provider:
    return "provider";
  case BackendPackCompatibilityRule::DeviceKind:
    return "device_kind";
  case BackendPackCompatibilityRule::CpuFeatures:
    return "cpu_features";
  case BackendPackCompatibilityRule::IdentityConfidence:
    return "identity_confidence";
  case BackendPackCompatibilityRule::MinimumDriver:
    return "minimum_driver";
  case BackendPackCompatibilityRule::TestedDriverRange:
    return "tested_driver_range";
  case BackendPackCompatibilityRule::CompleteMatch:
    return "complete_match";
  case BackendPackCompatibilityRule::InsufficientFacts:
    return "insufficient_facts";
  case BackendPackCompatibilityRule::CandidateTie:
    return "candidate_tie";
  }
  return "none";
}

const char *BackendPackRemediationName(BackendPackRemediation value) {
  switch (value) {
  case BackendPackRemediation::None:
    return "none";
  case BackendPackRemediation::VerifyRoute:
    return "verify_route";
  case BackendPackRemediation::UpdateDriver:
    return "update_driver";
  case BackendPackRemediation::ImproveDeviceIdentity:
    return "improve_device_identity";
  case BackendPackRemediation::InstallMatchingBase:
    return "install_matching_base";
  case BackendPackRemediation::SelectSupportedPack:
    return "select_supported_pack";
  case BackendPackRemediation::SelectAlternativeBackend:
    return "select_alternative_backend";
  case BackendPackRemediation::DiagnosticOnly:
    return "diagnostic_only";
  }
  return "none";
}

const char *BackendPackVerificationRequirementName(
    BackendPackVerificationRequirement value) {
  switch (value) {
  case BackendPackVerificationRequirement::NotRequired:
    return "not_required";
  case BackendPackVerificationRequirement::Required:
    return "required";
  }
  return "required";
}

} // namespace cyxwiz::runtime
