#pragma once

#include "backend_pack_manager_model.h"

#include "backend_pack_lifecycle_service.h"
#include "backend_pack_state_service.h"

#include <vector>

namespace cyxwiz {

std::vector<BackendPackManagerRecord> BuildBackendPackCatalogRecords(
    const runtime::VerifiedBackendPackCatalogSnapshot& catalog,
    const runtime::ActiveRuntimeState& active_runtime);

}  // namespace cyxwiz
