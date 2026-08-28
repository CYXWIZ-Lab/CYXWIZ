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
  std::cout << "Backend-pack compatibility contract tests passed\n";
  return 0;
}
