#pragma once

#include "backend_pack_metadata_verifier.h"

#include <string>

namespace cyxwiz::runtime {

enum class BackendPackEligibility { Unknown, Compatible, Incompatible };

enum class BackendPackInstallRecommendation {
  NotOffered,
  Available,
  AvailableAfterVerification,
  Recommended,
  DiagnosticOnly
};

enum class BackendPackLocalPackageState {
  NotInstalled,
  Staged,
  InstalledInactive,
  Active,
  IntegrityRejected
};

enum class BackendPackRouteVerificationStatus {
  NotRun,
  Passed,
  Failed,
  TimedOut,
  Crashed,
  Stale
};

enum class BackendPackTrainingAuthorizationStatus {
  NotEvaluated,
  Rejected,
  Authorized,
  DiagnosticOverride
};

enum class BackendPackPerformanceStatus {
  NotMeasured,
  Measured,
  PreferredMeasured
};

enum class BackendPackIdentityConfidence {
  Unknown,
  BackendLocal,
  ProviderReported,
  StableHardware
};

enum class BackendPackDeviceKind { Unknown, Cpu, Gpu, Accelerator };

enum class BackendPackCompatibilityRule {
  None,
  CatalogSupport,
  Platform,
  Architecture,
  RuntimeSet,
  CompanionBase,
  ArrayFireAbi,
  Provider,
  DeviceKind,
  IdentityConfidence,
  MinimumDriver,
  TestedDriverRange,
  CompleteMatch,
  InsufficientFacts,
  CandidateTie
};

enum class BackendPackRemediation {
  None,
  VerifyRoute,
  UpdateDriver,
  ImproveDeviceIdentity,
  InstallMatchingBase,
  SelectSupportedPack,
  SelectAlternativeBackend,
  DiagnosticOnly
};

enum class BackendPackVerificationRequirement { NotRequired, Required };

struct BackendPackMatchedDevice {
  // Internal identity used for exact-route reconciliation and invalidation.
  // Presentation and support serializers must not expose this value.
  std::string physical_fingerprint;
  std::string provider;
  std::string driver_version;
  BackendPackDeviceKind device_kind = BackendPackDeviceKind::Unknown;
  BackendPackIdentityConfidence identity_confidence =
      BackendPackIdentityConfidence::Unknown;
};

struct BackendPackCompatibilityDecision {
  std::string backend;
  std::string pack_id;
  std::string package_version;
  std::string runtime_set_id;
  std::string companion_base_id;
  std::string platform;
  std::string architecture;
  std::string arrayfire_abi;
  BackendPackSupportStatus catalog_support = BackendPackSupportStatus::Blocked;
  BackendPackEligibility eligibility = BackendPackEligibility::Unknown;
  BackendPackCompatibilityRule rule =
      BackendPackCompatibilityRule::InsufficientFacts;
  BackendPackMatchedDevice matched_device;
  BackendPackInstallRecommendation install_recommendation =
      BackendPackInstallRecommendation::NotOffered;
  BackendPackVerificationRequirement verification_requirement =
      BackendPackVerificationRequirement::Required;
  BackendPackLocalPackageState local_package_state =
      BackendPackLocalPackageState::NotInstalled;
  BackendPackRouteVerificationStatus verification_status =
      BackendPackRouteVerificationStatus::NotRun;
  BackendPackTrainingAuthorizationStatus training_authorization =
      BackendPackTrainingAuthorizationStatus::NotEvaluated;
  BackendPackPerformanceStatus performance_status =
      BackendPackPerformanceStatus::NotMeasured;
  BackendPackRemediation remediation = BackendPackRemediation::VerifyRoute;
};

const char *BackendPackEligibilityName(BackendPackEligibility value);
const char *
BackendPackInstallRecommendationName(BackendPackInstallRecommendation value);
const char *
BackendPackLocalPackageStateName(BackendPackLocalPackageState value);
const char *BackendPackRouteVerificationStatusName(
    BackendPackRouteVerificationStatus value);
const char *BackendPackTrainingAuthorizationStatusName(
    BackendPackTrainingAuthorizationStatus value);
const char *
BackendPackPerformanceStatusName(BackendPackPerformanceStatus value);
const char *
BackendPackIdentityConfidenceName(BackendPackIdentityConfidence value);
const char *BackendPackDeviceKindName(BackendPackDeviceKind value);
const char *
BackendPackCompatibilityRuleName(BackendPackCompatibilityRule value);
const char *BackendPackRemediationName(BackendPackRemediation value);
const char *BackendPackVerificationRequirementName(
    BackendPackVerificationRequirement value);

} // namespace cyxwiz::runtime
