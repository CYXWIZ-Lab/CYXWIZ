// Linux: keep ArrayFire backends of a packaged runtime inside that runtime.
//
// ArrayFire's unified loader dlopen()s each backend first by leaf name
// (LD_LIBRARY_PATH, the caller's RUNPATH, then ld.so.cache) and then from
// fixed directories such as /opt/arrayfire/lib. When a GPU pack is not
// installed, a developer ArrayFire registered in ld.so.cache would be loaded
// into the packaged process and abort it (both copies register the same
// spdlog loggers). ld.so.cache cannot be disabled through the environment, so
// the executable defines dlopen: it takes precedence over glibc's for every
// library the process loads. Requests for ArrayFire backend libraries resolve
// only to files inside CYXWIZ_ACTIVE_RUNTIME_ROOT; everything else passes
// through unchanged. Developer launches without an active runtime are not
// affected.
#if defined(__linux__)

#include <dlfcn.h>
#include <limits.h>
#include <stdlib.h>

#include <string>
#include <string_view>

namespace {

using DlopenFunction = void* (*)(const char*, int);

DlopenFunction RealDlopen() {
    static const DlopenFunction function =
        reinterpret_cast<DlopenFunction>(dlsym(RTLD_NEXT, "dlopen"));
    return function;
}

bool IsArrayFireBackend(std::string_view leaf) {
    for (const std::string_view prefix :
         {"libafcpu.so", "libafcuda.so", "libafopencl.so", "libafoneapi.so"}) {
        if (leaf.substr(0, prefix.size()) == prefix) {
            return true;
        }
    }
    return false;
}

bool ResolveInside(const std::string& candidate, const std::string& root,
                   std::string& resolved) {
    char buffer[PATH_MAX];
    if (realpath(candidate.c_str(), buffer) == nullptr) {
        return false;
    }
    resolved = buffer;
    return resolved.size() > root.size() &&
           resolved.compare(0, root.size(), root) == 0 &&
           resolved[root.size()] == '/';
}

// Fails like a missing library, so the unified loader moves on.
void* Refuse(int mode) {
    return RealDlopen()("/dev/null/cyxwiz-arrayfire-backend-outside-runtime", mode);
}

}  // namespace

extern "C" __attribute__((visibility("default"))) void* dlopen(const char* file,
                                                              int mode) {
    const DlopenFunction real = RealDlopen();
    const char* root_value = getenv("CYXWIZ_ACTIVE_RUNTIME_ROOT");
    if (file == nullptr || root_value == nullptr || *root_value == '\0') {
        return real(file, mode);
    }
    const std::string_view path(file);
    const auto slash = path.rfind('/');
    const std::string_view leaf =
        slash == std::string_view::npos ? path : path.substr(slash + 1);
    if (!IsArrayFireBackend(leaf)) {
        return real(file, mode);
    }

    char root_buffer[PATH_MAX];
    if (realpath(root_value, root_buffer) == nullptr) {
        return Refuse(mode);
    }
    const std::string root = root_buffer;
    std::string resolved;
    if (slash != std::string_view::npos) {
        return ResolveInside(std::string(path), root, resolved)
            ? real(resolved.c_str(), mode)
            : Refuse(mode);
    }

    // Leaf name: only the runtime's own library directories count.
    const char* search = getenv("LD_LIBRARY_PATH");
    std::string_view directories = search ? search : "";
    while (!directories.empty()) {
        const auto colon = directories.find(':');
        const std::string_view directory = directories.substr(0, colon);
        if (!directory.empty() &&
            ResolveInside(std::string(directory) + "/" + std::string(leaf), root,
                          resolved)) {
            return real(resolved.c_str(), mode);
        }
        if (colon == std::string_view::npos) break;
        directories.remove_prefix(colon + 1);
    }
    return Refuse(mode);
}

#endif  // __linux__
