#include "installer_pack_presentation.h"

namespace cyxwiz {
namespace {

std::string IncompatibilityReason(
    runtime::BackendPackCompatibilityRule rule) {
    switch (rule) {
        case runtime::BackendPackCompatibilityRule::Platform:
        case runtime::BackendPackCompatibilityRule::Architecture:
            return "This package targets a different operating system or processor architecture.";
        case runtime::BackendPackCompatibilityRule::RuntimeSet:
        case runtime::BackendPackCompatibilityRule::CompanionBase:
        case runtime::BackendPackCompatibilityRule::ArrayFireAbi:
            return "This package requires a different CyxWiz Engine runtime.";
        case runtime::BackendPackCompatibilityRule::Provider:
        case runtime::BackendPackCompatibilityRule::DeviceKind:
        case runtime::BackendPackCompatibilityRule::CpuFeatures:
            return "Known device or provider requirements do not match this machine.";
        case runtime::BackendPackCompatibilityRule::MinimumDriver:
            return "The detected provider driver is older than this package requires.";
        case runtime::BackendPackCompatibilityRule::CatalogSupport:
            return "The current signed release does not offer this package.";
        default:
            return "Known machine or runtime requirements do not match this package.";
    }
}

std::string RemediationAction(runtime::BackendPackRemediation remediation) {
    switch (remediation) {
        case runtime::BackendPackRemediation::VerifyRoute:
            return "Install the package and run Verify All.";
        case runtime::BackendPackRemediation::UpdateDriver:
            return "Update the provider driver, then run Verify All again.";
        case runtime::BackendPackRemediation::ImproveDeviceIdentity:
            return "Refresh device detection and run Verify All again.";
        case runtime::BackendPackRemediation::InstallMatchingBase:
            return "Select the matching CyxWiz Engine package.";
        case runtime::BackendPackRemediation::SelectSupportedPack:
            return "Choose a package offered by the current signed catalog.";
        case runtime::BackendPackRemediation::SelectAlternativeBackend:
            return "Choose another compatible verified backend.";
        case runtime::BackendPackRemediation::DiagnosticOnly:
            return "Use only for isolated diagnostics, not production training.";
        case runtime::BackendPackRemediation::None:
            return {};
    }
    return {};
}

}  // namespace

InstallerPackPresentation BuildInstallerPackPresentation(
    const BackendPackManagerRecord& record) {
    InstallerPackPresentation result;
    if (!record.delivery_metadata_available && !record.installed) {
        result.status = "Unavailable";
        result.explanation = record.delivery_metadata_error.empty()
            ? "No verified delivery metadata is available for this package."
            : "The signed package metadata could not be verified.";
        result.tone = InstallerPackPresentationTone::Danger;
        return result;
    }
    if (!record.compatibility) {
        result.status = record.active ? "Active; compatibility unknown"
            : record.installed ? "Installed; compatibility unknown"
                               : "Unavailable";
        result.explanation = record.installed
            ? "Compatibility could not be evaluated for this installed package."
            : "Compatibility could not be evaluated, so installation is disabled.";
        result.action =
            "Refresh the signed package catalog and compatibility details.";
        result.tone = record.installed
            ? InstallerPackPresentationTone::Warning
            : InstallerPackPresentationTone::Danger;
        return result;
    }

    const auto& decision = *record.compatibility;
    result.action = RemediationAction(decision.remediation);
    if (decision.eligibility == runtime::BackendPackEligibility::Incompatible ||
        decision.install_recommendation ==
            runtime::BackendPackInstallRecommendation::NotOffered) {
        result.status = "Not compatible";
        result.explanation = IncompatibilityReason(decision.rule);
        result.tone = InstallerPackPresentationTone::Danger;
        return result;
    }
    if (decision.install_recommendation ==
        runtime::BackendPackInstallRecommendation::DiagnosticOnly) {
        result.status = "Diagnostic only";
        result.explanation =
            "This package is available only for explicit isolated diagnostics.";
        result.tone = InstallerPackPresentationTone::Warning;
        return result;
    }

    switch (decision.verification_status) {
        case runtime::BackendPackRouteVerificationStatus::Crashed:
            result.status = "Verification crashed";
            result.explanation =
                "The exact local route crashed inside an isolated verification process.";
            result.action =
                "Update the provider or driver and verify again, or use another verified route.";
            result.tone = InstallerPackPresentationTone::Danger;
            return result;
        case runtime::BackendPackRouteVerificationStatus::TimedOut:
            result.status = "Verification timed out";
            result.explanation =
                "The exact local route exceeded the verification time limit.";
            result.action =
                "Update the provider or driver and run Verify All again.";
            result.tone = InstallerPackPresentationTone::Warning;
            return result;
        case runtime::BackendPackRouteVerificationStatus::Failed:
            result.status = "Verification failed";
            result.explanation =
                "The exact local route did not pass the required operation set.";
            result.action =
                "Check provider requirements and run Verify All again.";
            result.tone = InstallerPackPresentationTone::Danger;
            return result;
        case runtime::BackendPackRouteVerificationStatus::Stale:
            result.status = "Verification outdated";
            result.explanation =
                "Saved results belong to a different runtime or package state.";
            result.action = "Run Verify All with the current runtime.";
            result.tone = InstallerPackPresentationTone::Warning;
            return result;
        default:
            break;
    }

    if (decision.install_recommendation ==
            runtime::BackendPackInstallRecommendation::Recommended &&
        decision.performance_status ==
            runtime::BackendPackPerformanceStatus::PreferredMeasured) {
        result.status = "Best verified";
        result.explanation =
            "This was the fastest comparable route in the latest local verification.";
        result.action.clear();
        result.tone = InstallerPackPresentationTone::Success;
        return result;
    }
    if (decision.verification_status ==
        runtime::BackendPackRouteVerificationStatus::Passed) {
        result.status = decision.performance_status ==
                runtime::BackendPackPerformanceStatus::Measured
            ? "Verified and measured" : "Verified and ready";
        result.explanation =
            "The exact local route passed all required verification operations.";
        result.action.clear();
        result.tone = InstallerPackPresentationTone::Success;
        return result;
    }
    if (record.update_available) {
        result.status = "Update available";
        result.explanation =
            "A newer signed package is available for this backend.";
        result.tone = InstallerPackPresentationTone::Warning;
        return result;
    }
    if (decision.eligibility == runtime::BackendPackEligibility::Unknown) {
        result.status = "Verification needed";
        result.explanation =
            "Current machine details are not sufficient to prove compatibility.";
        result.tone = InstallerPackPresentationTone::Warning;
        return result;
    }

    result.status = record.active ? "Installed; verify" : "Compatible; verify";
    result.explanation =
        "The signed requirements match, but this exact route is not verified yet.";
    result.tone = InstallerPackPresentationTone::Accent;
    return result;
}

}  // namespace cyxwiz
