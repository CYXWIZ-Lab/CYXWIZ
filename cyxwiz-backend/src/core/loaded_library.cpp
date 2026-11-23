#include "loaded_library.h"

#include <string>

#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#ifdef __APPLE__
#include <mach-o/dyld.h>
#else
#include <link.h>
#endif
#endif

namespace cyxwiz::detail {
namespace {

#ifndef _WIN32
// Whether a loaded image's file name is lib<stem>.<...>.dylib on macOS, or
// lib<stem>.so[.<...>] elsewhere. ArrayFire's unified library opens its
// backends by full path, under versioned names, so the short name alone is
// not enough to find them.
bool NamesLibrary(const char* path, const std::string& stem) {
    std::string name(path);
    const auto slash = name.find_last_of('/');
    if (slash != std::string::npos) name.erase(0, slash + 1);
    const std::string prefix = "lib" + stem + ".";
    if (name.compare(0, prefix.size(), prefix) != 0) return false;
#ifdef __APPLE__
    const std::string suffix = ".dylib";
    return name.size() >= prefix.size() + suffix.size() - 1 &&
           name.compare(name.size() - suffix.size(), suffix.size(), suffix) == 0;
#else
    return name.compare(prefix.size() - 1, 3, ".so") == 0;
#endif
}

// The path the loader recorded for the library, or empty.
std::string LoadedPath(const std::string& stem) {
#ifdef __APPLE__
    const uint32_t count = _dyld_image_count();
    for (uint32_t i = 0; i < count; ++i) {
        const char* image = _dyld_get_image_name(i);
        if (image != nullptr && NamesLibrary(image, stem)) return image;
    }
    return {};
#else
    struct Search {
        const std::string* stem;
        std::string path;
    } search{&stem, {}};
    dl_iterate_phdr(
        [](dl_phdr_info* info, size_t, void* data) {
            auto* found = static_cast<Search*>(data);
            if (info->dlpi_name != nullptr && info->dlpi_name[0] != '\0' &&
                NamesLibrary(info->dlpi_name, *found->stem)) {
                found->path = info->dlpi_name;
                return 1;
            }
            return 0;
        },
        &search);
    return search.path;
#endif
}
#endif

} // namespace

void* FindLoadedSymbol(const char* stem, const char* symbol) {
#ifdef _WIN32
    const HMODULE module = GetModuleHandleA((std::string(stem) + ".dll").c_str());
    return module != nullptr ? reinterpret_cast<void*>(GetProcAddress(module, symbol)) : nullptr;
#else
    const std::string path = LoadedPath(stem);
    if (path.empty()) return nullptr;
    void* handle = dlopen(path.c_str(), RTLD_LAZY | RTLD_NOLOAD);
    if (handle == nullptr) return nullptr;
    void* found = dlsym(handle, symbol);
    // Drops only the reference RTLD_NOLOAD added; whoever loaded it keeps it.
    dlclose(handle);
    return found;
#endif
}

bool IsLibraryLoaded(const char* stem) {
#ifdef _WIN32
    return GetModuleHandleA((std::string(stem) + ".dll").c_str()) != nullptr;
#else
    return !LoadedPath(stem).empty();
#endif
}

} // namespace cyxwiz::detail
