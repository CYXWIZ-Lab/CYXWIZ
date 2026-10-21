#pragma once

#include <string>

namespace cyxwiz {

// ArrayFire's Windows unified loader probes a conventional Program Files
// installation after its normal DLL search paths. Packaged CyxWiz runtimes
// temporarily redirect that fallback while the unified loader discovers the
// immutable active runtime. Developer launches without an active-runtime
// identity are left unchanged.
class ScopedArrayFireBackendDiscoveryIsolation {
public:
    ScopedArrayFireBackendDiscoveryIsolation() = default;
    ~ScopedArrayFireBackendDiscoveryIsolation();

    ScopedArrayFireBackendDiscoveryIsolation(
        const ScopedArrayFireBackendDiscoveryIsolation&) = delete;
    ScopedArrayFireBackendDiscoveryIsolation& operator=(
        const ScopedArrayFireBackendDiscoveryIsolation&) = delete;

    bool Apply(std::string& error);

private:
#ifdef _WIN32
    std::wstring previous_program_files_;
    bool previous_program_files_present_ = false;
    bool applied_ = false;
#endif
};

}  // namespace cyxwiz
