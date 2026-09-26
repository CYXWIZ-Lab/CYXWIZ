#pragma once

#include "cyxwiz/api_export.h"

#include <functional>
#include <string>

namespace cyxwiz {

// Backend -> host debug events (layer timings, recurrent route decisions,
// fallback reasons). The callback lives inside the backend library, so a
// host that sets it sees events emitted from backend code too (it used to be
// a header-local static, one copy per module, so backend events never reached
// the Engine).
class CYXWIZ_API BackendDebugHooks {
public:
    using DebugEventCallback = std::function<void(const std::string& source,
                                                  const std::string& message)>;

    static void SetDebugEventCallback(DebugEventCallback callback);
    static void EmitDebugEvent(const std::string& source, const std::string& message);
    static bool HasDebugEventCallback();
};

} // namespace cyxwiz
