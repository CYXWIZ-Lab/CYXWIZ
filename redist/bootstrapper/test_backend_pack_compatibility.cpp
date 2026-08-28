#include "backend_pack_compatibility.h"

#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

namespace {

void Check(bool condition, const std::string &message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << "\n";
    std::exit(1);
  }
}

template <typename Value, typename Name>
void CheckNames(const std::vector<Value> &values, Name name,
                const std::string &label) {
  for (const auto value : values) {
    const std::string text(name(value));
    Check(!text.empty(), label + " must have a stable bounded name");
    Check(text.find("ticket") == std::string::npos,
          label + " must not expose internal ticket text");
  }
}

void TestStableNames() {
  using namespace cyxwiz::runtime;
  CheckNames(std::vector{BackendPackEligibility::Unknown,
                         BackendPackEligibility::Compatible,
                         BackendPackEligibility::Incompatible},
             BackendPackEligibilityName, "eligibility");
  CheckNames(
      std::vector{BackendPackInstallRecommendation::NotOffered,
                  BackendPackInstallRecommendation::Available,
                  BackendPackInstallRecommendation::AvailableAfterVerification,
                  BackendPackInstallRecommendation::Recommended,
                  BackendPackInstallRecommendation::DiagnosticOnly},
      BackendPackInstallRecommendationName, "install recommendation");
  CheckNames(std::vector{BackendPackLocalPackageState::NotInstalled,
                         BackendPackLocalPackageState::Staged,
                         BackendPackLocalPackageState::InstalledInactive,
                         BackendPackLocalPackageState::Active,
                         BackendPackLocalPackageState::IntegrityRejected},
             BackendPackLocalPackageStateName, "local package state");
  CheckNames(std::vector{BackendPackRouteVerificationStatus::NotRun,
                         BackendPackRouteVerificationStatus::Passed,
                         BackendPackRouteVerificationStatus::Failed,
                         BackendPackRouteVerificationStatus::TimedOut,
                         BackendPackRouteVerificationStatus::Crashed,
                         BackendPackRouteVerificationStatus::Stale},
             BackendPackRouteVerificationStatusName, "verification status");
  CheckNames(
      std::vector{BackendPackTrainingAuthorizationStatus::NotEvaluated,
                  BackendPackTrainingAuthorizationStatus::Rejected,
                  BackendPackTrainingAuthorizationStatus::Authorized,
                  BackendPackTrainingAuthorizationStatus::DiagnosticOverride},
      BackendPackTrainingAuthorizationStatusName, "training authorization");
  CheckNames(std::vector{BackendPackPerformanceStatus::NotMeasured,
                         BackendPackPerformanceStatus::Measured,
                         BackendPackPerformanceStatus::PreferredMeasured},
             BackendPackPerformanceStatusName, "performance status");
  CheckNames(std::vector{BackendPackIdentityConfidence::Unknown,
                         BackendPackIdentityConfidence::BackendLocal,
                         BackendPackIdentityConfidence::ProviderReported,
                         BackendPackIdentityConfidence::StableHardware},
             BackendPackIdentityConfidenceName, "identity confidence");
  CheckNames(std::vector{BackendPackDeviceKind::Unknown,
                         BackendPackDeviceKind::Cpu, BackendPackDeviceKind::Gpu,
                         BackendPackDeviceKind::Accelerator},
             BackendPackDeviceKindName, "device kind");
  CheckNames(std::vector{BackendPackCompatibilityRule::None,
                         BackendPackCompatibilityRule::CatalogSupport,
                         BackendPackCompatibilityRule::Platform,
                         BackendPackCompatibilityRule::Architecture,
                         BackendPackCompatibilityRule::RuntimeSet,
                         BackendPackCompatibilityRule::CompanionBase,
                         BackendPackCompatibilityRule::ArrayFireAbi,
                         BackendPackCompatibilityRule::Provider,
                         BackendPackCompatibilityRule::DeviceKind,
                         BackendPackCompatibilityRule::CpuFeatures,
                         BackendPackCompatibilityRule::IdentityConfidence,
                         BackendPackCompatibilityRule::MinimumDriver,
                         BackendPackCompatibilityRule::TestedDriverRange,
                         BackendPackCompatibilityRule::CompleteMatch,
                         BackendPackCompatibilityRule::InsufficientFacts,
                         BackendPackCompatibilityRule::CandidateTie},
             BackendPackCompatibilityRuleName, "compatibility rule");
  CheckNames(std::vector{BackendPackRemediation::None,
                         BackendPackRemediation::VerifyRoute,
                         BackendPackRemediation::UpdateDriver,
                         BackendPackRemediation::ImproveDeviceIdentity,
                         BackendPackRemediation::InstallMatchingBase,
                         BackendPackRemediation::SelectSupportedPack,
                         BackendPackRemediation::SelectAlternativeBackend,
                         BackendPackRemediation::DiagnosticOnly},
             BackendPackRemediationName, "remediation");
  CheckNames(std::vector{BackendPackVerificationRequirement::NotRequired,
                         BackendPackVerificationRequirement::Required},
             BackendPackVerificationRequirementName,
             "verification requirement");
}

cyxwiz::runtime::VerifiedBackendPackManifest OneApiManifest() {
  using namespace cyxwiz::runtime;
  VerifiedBackendPackManifest manifest;
  manifest.kind = BackendPackManifestKind::BackendPack;
  manifest.pack_id = "oneapi-v1";
  manifest.backend = "oneapi";
  manifest.package_version = "1.0.0";
  manifest.platform = "win64";
  manifest.architecture = "x86_64";
  manifest.runtime_set_id = "runtime-v1";
  manifest.companion_base_id = "base-v1";
  manifest.arrayfire_abi = "arrayfire-3.9";
  manifest.compatibility.device_kinds = {"cpu", "gpu", "accelerator"};
  manifest.compatibility.provider_types = {"sycl-unified-runtime"};
  manifest.compatibility.minimum_driver_versions = {{"intel", "31.0.101.2115"}};
  manifest.compatibility.tested_driver_ranges = {
      {"intel", ">=31.0.101.2115,<32.0"}};
  manifest.compatibility.minimum_identity_confidence = "stable_hardware";
  manifest.compatibility.support_status = BackendPackSupportStatus::Supported;
  return manifest;
}

cyxwiz::runtime::BackendPackCompatibilityContext CompatibleMachine() {
  using namespace cyxwiz::runtime;
  BackendPackCompatibilityContext context;
  context.platform = "win64";
  context.architecture = "x86_64";
  context.runtime_set_id = "runtime-v1";
  context.base_pack_id = "base-v1";
  context.arrayfire_abi = "arrayfire-3.9";
  BackendPackMatchedDevice device;
  device.physical_fingerprint = "private-intel-device";
  device.provider = "intel";
  device.driver_version = "31.0.101.5522";
  device.provider_types = {"sycl-unified-runtime"};
  device.device_kind = BackendPackDeviceKind::Gpu;
  device.identity_confidence = BackendPackIdentityConfidence::StableHardware;
  context.devices.push_back(std::move(device));
  return context;
}

void TestCompatibilityEvaluation() {
  using namespace cyxwiz::runtime;
  auto manifest = OneApiManifest();
  auto context = CompatibleMachine();

  const auto compatible = EvaluateBackendPackCompatibility(manifest, context);
  Check(compatible.eligibility == BackendPackEligibility::Compatible &&
            compatible.rule == BackendPackCompatibilityRule::CompleteMatch &&
            compatible.install_recommendation ==
                BackendPackInstallRecommendation::AvailableAfterVerification &&
            compatible.verification_status ==
                BackendPackRouteVerificationStatus::NotRun &&
            compatible.training_authorization ==
                BackendPackTrainingAuthorizationStatus::NotEvaluated,
        "catalog compatibility must offer verification without authorizing "
        "training");

  manifest.compatibility.support_status = BackendPackSupportStatus::Blocked;
  const auto blocked = EvaluateBackendPackCompatibility(manifest, context);
  Check(blocked.eligibility == BackendPackEligibility::Incompatible &&
            blocked.rule == BackendPackCompatibilityRule::CatalogSupport &&
            blocked.install_recommendation ==
                BackendPackInstallRecommendation::NotOffered,
        "signed catalog policy must fail closed before machine matching");

  manifest = OneApiManifest();
  context.platform = "linux64";
  const auto wrong_platform =
      EvaluateBackendPackCompatibility(manifest, context);
  Check(wrong_platform.eligibility == BackendPackEligibility::Incompatible &&
            wrong_platform.rule == BackendPackCompatibilityRule::Platform,
        "a package for another platform must be incompatible");

  context = CompatibleMachine();
  context.base_pack_id.clear();
  const auto missing_base = EvaluateBackendPackCompatibility(manifest, context);
  Check(missing_base.eligibility == BackendPackEligibility::Unknown &&
            missing_base.remediation ==
                BackendPackRemediation::InstallMatchingBase,
        "missing base identity must remain unknown rather than inventing a "
        "match");

  context = CompatibleMachine();
  context.devices.front().driver_version = "30.0.100.1";
  const auto old_driver = EvaluateBackendPackCompatibility(manifest, context);
  Check(old_driver.eligibility == BackendPackEligibility::Incompatible &&
            old_driver.rule == BackendPackCompatibilityRule::MinimumDriver &&
            old_driver.remediation == BackendPackRemediation::UpdateDriver,
        "a proven minimum-driver failure must return bounded remediation");

  context = CompatibleMachine();
  context.devices.front().driver_version = "32.0.0";
  const auto outside_tested_range =
      EvaluateBackendPackCompatibility(manifest, context);
  Check(outside_tested_range.eligibility ==
                BackendPackEligibility::Compatible &&
            outside_tested_range.rule ==
                BackendPackCompatibilityRule::TestedDriverRange &&
            outside_tested_range.remediation ==
                BackendPackRemediation::VerifyRoute,
        "an untested but sufficiently new driver must require verification, "
        "not become a global block");

  context.devices.front().driver_version.clear();
  const auto unknown_driver =
      EvaluateBackendPackCompatibility(manifest, context);
  Check(unknown_driver.eligibility == BackendPackEligibility::Unknown &&
            unknown_driver.rule == BackendPackCompatibilityRule::MinimumDriver,
        "missing driver evidence must remain unknown");

  context = CompatibleMachine();
  context.devices.front().provider.clear();
  const auto unknown_provider =
      EvaluateBackendPackCompatibility(manifest, context);
  Check(unknown_provider.eligibility == BackendPackEligibility::Unknown &&
            unknown_provider.rule == BackendPackCompatibilityRule::Provider &&
            unknown_provider.remediation == BackendPackRemediation::VerifyRoute,
        "missing provider identity must not bypass provider-specific driver "
        "constraints");

  context = CompatibleMachine();
  context.devices.front().provider_types = {"opencl-icd"};
  const auto wrong_provider =
      EvaluateBackendPackCompatibility(manifest, context);
  Check(wrong_provider.eligibility == BackendPackEligibility::Incompatible &&
            wrong_provider.rule == BackendPackCompatibilityRule::Provider,
        "known provider mismatch must not be offered as compatible");

  context = CompatibleMachine();
  context.devices.push_back(context.devices.front());
  context.devices.back().physical_fingerprint = "private-second-device";
  const auto tied = EvaluateBackendPackCompatibility(manifest, context);
  Check(tied.eligibility == BackendPackEligibility::Unknown &&
            tied.rule == BackendPackCompatibilityRule::CandidateTie,
        "ambiguous physical candidates must not be silently associated");

  manifest.compatibility.support_status = BackendPackSupportStatus::Diagnostic;
  context = CompatibleMachine();
  const auto diagnostic = EvaluateBackendPackCompatibility(manifest, context);
  Check(diagnostic.eligibility == BackendPackEligibility::Compatible &&
            diagnostic.install_recommendation ==
                BackendPackInstallRecommendation::DiagnosticOnly,
        "diagnostic catalog policy may be inspected explicitly but must not "
        "become recommended");
}

void TestBaseAndLocalCrashRemainIndependent() {
  using namespace cyxwiz::runtime;
  auto base = OneApiManifest();
  base.kind = BackendPackManifestKind::Base;
  base.backend = "cpu";
  base.pack_id = "base-v1";
  base.companion_base_id.clear();
  base.compatibility.device_kinds = {"cpu"};
  base.compatibility.provider_types = {"arrayfire-cpu"};
  BackendPackCompatibilityContext base_context;
  base_context.platform = "win64";
  base_context.architecture = "x86_64";
  const auto base_decision =
      EvaluateBackendPackCompatibility(base, base_context);
  Check(base_decision.eligibility == BackendPackEligibility::Compatible &&
            base_decision.install_recommendation ==
                BackendPackInstallRecommendation::Available,
        "the required CPU base must not require preinstalled route facts");

  auto route_decision =
      EvaluateBackendPackCompatibility(OneApiManifest(), CompatibleMachine());
  route_decision.verification_status =
      BackendPackRouteVerificationStatus::Crashed;
  route_decision.training_authorization =
      BackendPackTrainingAuthorizationStatus::Rejected;
  Check(route_decision.catalog_support == BackendPackSupportStatus::Supported &&
            route_decision.eligibility == BackendPackEligibility::Compatible &&
            route_decision.verification_status ==
                BackendPackRouteVerificationStatus::Crashed &&
            route_decision.training_authorization ==
                BackendPackTrainingAuthorizationStatus::Rejected,
        "a local route crash must reject local training without rewriting "
        "signed oneAPI support");
}

void TestIndependentDefaultStates() {
  using namespace cyxwiz::runtime;
  const BackendPackCompatibilityDecision decision;
  Check(decision.catalog_support == BackendPackSupportStatus::Blocked &&
            decision.rule == BackendPackCompatibilityRule::InsufficientFacts,
        "an empty decision must fail closed on catalog support and facts");
  Check(decision.eligibility == BackendPackEligibility::Unknown,
        "missing machine facts must default to unknown eligibility");
  Check(decision.install_recommendation ==
            BackendPackInstallRecommendation::NotOffered,
        "unknown eligibility must not default to a recommendation");
  Check(decision.verification_requirement ==
                BackendPackVerificationRequirement::Required &&
            decision.verification_status ==
                BackendPackRouteVerificationStatus::NotRun,
        "verification requirement and result must remain distinct");
  Check(decision.training_authorization ==
            BackendPackTrainingAuthorizationStatus::NotEvaluated,
        "compatibility must not imply training authorization");
  Check(decision.performance_status ==
            BackendPackPerformanceStatus::NotMeasured,
        "compatibility must not imply a performance recommendation");
}

} // namespace

int main() {
  TestStableNames();
  TestIndependentDefaultStates();
  TestCompatibilityEvaluation();
  TestBaseAndLocalCrashRemainIndependent();
  std::cout << "Backend-pack compatibility contract tests passed\n";
  return 0;
}
