#include "backend_pack_decision_reconciliation.h"

#include <algorithm>

namespace cyxwiz {
namespace {

runtime::BackendPackLocalPackageState LocalPackageState(
    const BackendPackManagerRecord& record) {
    if (record.active) return runtime::BackendPackLocalPackageState::Active;
    if (record.installed) {
        return runtime::BackendPackLocalPackageState::InstalledInactive;
    }
    return runtime::BackendPackLocalPackageState::NotInstalled;
}

runtime::BackendPackInstallRecommendation BaselineRecommendation(
    const BackendPackManagerRecord& record,
    const runtime::BackendPackCompatibilityDecision& decision) {
    if (decision.eligibility == runtime::BackendPackEligibility::Incompatible ||
        decision.catalog_support == runtime::BackendPackSupportStatus::Blocked ||
        decision.catalog_support == runtime::BackendPackSupportStatus::Revoked) {
        return runtime::BackendPackInstallRecommendation::NotOffered;
    }
    if (decision.catalog_support ==
        runtime::BackendPackSupportStatus::Diagnostic) {
        return runtime::BackendPackInstallRecommendation::DiagnosticOnly;
    }
    if (record.backend == "cpu" &&
        decision.eligibility == runtime::BackendPackEligibility::Compatible) {
        return runtime::BackendPackInstallRecommendation::Available;
    }
    return runtime::BackendPackInstallRecommendation::AvailableAfterVerification;
}

int FailureRank(InstallerRouteVerificationStatus status) {
    switch (status) {
        case InstallerRouteVerificationStatus::Crashed: return 4;
        case InstallerRouteVerificationStatus::TimedOut: return 3;
        case InstallerRouteVerificationStatus::Failed: return 2;
        case InstallerRouteVerificationStatus::NotVerified: return 1;
        case InstallerRouteVerificationStatus::Passed: return 0;
    }
    return 0;
}

runtime::BackendPackRouteVerificationStatus MapVerificationStatus(
    InstallerRouteVerificationStatus status) {
    switch (status) {
        case InstallerRouteVerificationStatus::Passed:
            return runtime::BackendPackRouteVerificationStatus::Passed;
        case InstallerRouteVerificationStatus::Failed:
            return runtime::BackendPackRouteVerificationStatus::Failed;
        case InstallerRouteVerificationStatus::TimedOut:
            return runtime::BackendPackRouteVerificationStatus::TimedOut;
        case InstallerRouteVerificationStatus::Crashed:
            return runtime::BackendPackRouteVerificationStatus::Crashed;
        case InstallerRouteVerificationStatus::NotVerified:
            return runtime::BackendPackRouteVerificationStatus::NotRun;
    }
    return runtime::BackendPackRouteVerificationStatus::NotRun;
}

}  // namespace

void ReconcileBackendPackDecisionEvidence(
    std::vector<BackendPackManagerRecord>& records,
    const InstallerVerificationSummary& verification) {
    for (auto& record : records) {
        if (!record.compatibility) continue;
        auto& decision = *record.compatibility;
        decision.local_package_state = LocalPackageState(record);
        decision.install_recommendation =
            BaselineRecommendation(record, decision);
        decision.verification_requirement =
            runtime::BackendPackVerificationRequirement::Required;
        decision.verification_status =
            runtime::BackendPackRouteVerificationStatus::NotRun;
        decision.training_authorization =
            runtime::BackendPackTrainingAuthorizationStatus::NotEvaluated;
        decision.performance_status =
            runtime::BackendPackPerformanceStatus::NotMeasured;
        record.qualification_evidence_available = false;
        record.training_authorized = false;

        if (!verification.evidence_available) continue;
        if (!verification.evidence_matches_runtime) {
            if (record.active) {
                decision.verification_status =
                    runtime::BackendPackRouteVerificationStatus::Stale;
            }
            continue;
        }

        std::vector<const InstallerRouteVerificationResult*> routes;
        for (const auto& route : verification.routes) {
            if (route.active && route.pack_id == record.pack_id) {
                routes.push_back(&route);
            }
        }
        if (routes.empty()) continue;

        record.qualification_evidence_available = true;
        const auto passed = std::find_if(
            routes.begin(), routes.end(), [](const auto* route) {
                return route->status == InstallerRouteVerificationStatus::Passed;
            });
        const InstallerRouteVerificationResult* representative = nullptr;
        if (passed != routes.end()) {
            representative = *passed;
        } else {
            representative = *std::max_element(
                routes.begin(), routes.end(), [](const auto* left,
                                                 const auto* right) {
                    return FailureRank(left->status) < FailureRank(right->status);
                });
        }
        decision.verification_status =
            MapVerificationStatus(representative->status);

        if (representative->status ==
            InstallerRouteVerificationStatus::Passed) {
            decision.verification_requirement =
                runtime::BackendPackVerificationRequirement::NotRequired;
            if (decision.catalog_support ==
                runtime::BackendPackSupportStatus::Supported) {
                decision.training_authorization =
                    runtime::BackendPackTrainingAuthorizationStatus::Authorized;
                record.training_authorized = true;
            } else {
                decision.training_authorization =
                    runtime::BackendPackTrainingAuthorizationStatus::Rejected;
            }
        } else if (representative->status !=
                   InstallerRouteVerificationStatus::NotVerified) {
            decision.training_authorization =
                runtime::BackendPackTrainingAuthorizationStatus::Rejected;
        }

        const bool measured = std::any_of(
            routes.begin(), routes.end(),
            [](const auto* route) { return route->benchmark_available; });
        const bool preferred = std::any_of(
            routes.begin(), routes.end(),
            [](const auto* route) { return route->best_measured; });
        if (preferred) {
            decision.performance_status =
                runtime::BackendPackPerformanceStatus::PreferredMeasured;
        } else if (measured) {
            decision.performance_status =
                runtime::BackendPackPerformanceStatus::Measured;
        }
        if (preferred &&
            decision.verification_status ==
                runtime::BackendPackRouteVerificationStatus::Passed &&
            decision.training_authorization ==
                runtime::BackendPackTrainingAuthorizationStatus::Authorized &&
            decision.recommendation_target_eligible &&
            decision.eligibility !=
                runtime::BackendPackEligibility::Incompatible) {
            decision.install_recommendation =
                runtime::BackendPackInstallRecommendation::Recommended;
            decision.remediation = runtime::BackendPackRemediation::None;
        }
    }
}

}  // namespace cyxwiz
