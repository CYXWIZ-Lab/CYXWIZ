#pragma once

// TOFIX118 P8 profiling: with CYXWIZ_PROFILE_STAGE_SYNC=1, waits for the
// device so the next measured duration is device work, not dispatch. No-op
// otherwise (it removes CPU/GPU overlap).

#include <cstdlib>
#include <string>

#ifdef CYXWIZ_HAS_ARRAYFIRE
#include <arrayfire.h>
#endif

namespace cyxwiz {

inline void TrainingProfileStageFence() {
    static const bool enabled = [] {
        const char* value = std::getenv("CYXWIZ_PROFILE_STAGE_SYNC");
        return value && *value && std::string(value) != "0";
    }();
#ifdef CYXWIZ_HAS_ARRAYFIRE
    if (enabled) af::sync();
#else
    (void)enabled;
#endif
}

}  // namespace cyxwiz
