#pragma once

#include "backend_pack_manager_model.h"
#include "installer_verification_summary.h"

#include <vector>

namespace cyxwiz {

void ReconcileBackendPackDecisionEvidence(
    std::vector<BackendPackManagerRecord>& records,
    const InstallerVerificationSummary& verification);

}  // namespace cyxwiz
