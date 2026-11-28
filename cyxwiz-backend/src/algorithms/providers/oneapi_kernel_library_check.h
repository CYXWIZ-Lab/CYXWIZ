#pragma once

// Internal to the backend and its tests; not part of the plugin SDK. Exported
// from the backend library only so that the unit tests can reach them.

#include "cyxwiz/api_export.h"
#include "cyxwiz/backend_fallback_reason.h"

#include <string>

namespace cyxwiz {

// Why the oneAPI kernel library at `path` cannot be used, from what loading it
// finds: BackendInternalError when it can; ProviderAbiMismatch when an entry
// point is missing or its ABI number is not this Engine's
// (CYXWIZ_ONEAPI_KERNELS_ABI); OneapiProviderUnavailable when it does not
// load. The registry checks the installed library the same way.
CYXWIZ_API BackendFallbackReason OneapiKernelLibraryVerdict(bool loaded, bool entry_points, int abi);
// Test-only: loads `path` as the kernel library and gives the verdict, with
// why in detail, whether or not ArrayFire's oneAPI backend is loaded.
CYXWIZ_API BackendFallbackReason CheckOneapiKernelLibraryForTesting(const std::string& path, std::string& detail);

}  // namespace cyxwiz
