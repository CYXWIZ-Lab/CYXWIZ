#pragma once

#include "backend_pack_manager_model.h"

#include "backend_pack_lifecycle_service.h"
#include "backend_pack_state_service.h"

#include <vector>

namespace cyxwiz {

// One record per catalog pack for this platform, plus installed packs the
// catalog no longer lists. A plugin-pack manifest becomes a Plugin record
// (plugin id, no backend); its installed state comes from active_plugins
// (active-plugins.json), which counts only for the active base.
std::vector<BackendPackManagerRecord> BuildBackendPackCatalogRecords(
    const runtime::VerifiedBackendPackCatalogSnapshot& catalog,
    const runtime::ActiveRuntimeState& active_runtime,
    const runtime::ActivePluginsState& active_plugins = {});

std::vector<BackendPackManagerRecord> BuildBackendPackCatalogRecords(
    const runtime::VerifiedBackendPackCatalogSnapshot& catalog,
    const runtime::ActiveRuntimeState& active_runtime,
    const runtime::BackendPackCompatibilityContext& compatibility_context,
    const runtime::ActivePluginsState& active_plugins = {});

}  // namespace cyxwiz
