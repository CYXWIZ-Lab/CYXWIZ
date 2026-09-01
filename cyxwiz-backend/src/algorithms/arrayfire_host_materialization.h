#pragma once

#include "arrayfire_backend_utils.h"

#include <string>

#ifdef CYXWIZ_HAS_ARRAYFIRE
namespace af {
class array;
}

namespace cyxwiz {

// The only direct af::array::host boundary outside Tensor. Requiring explicit
// attribution here keeps synchronization truth complete for non-Tensor APIs.
CYXWIZ_API void MaterializeArrayFireToHost(
    const af::array& source,
    void* destination,
    ArrayFireHostSyncCategory category,
    std::string operation_name,
    std::string layout,
    std::string reason_code = "direct_arrayfire_host_materialization",
    std::string context = {});

} // namespace cyxwiz
#endif
