#pragma once

// A driver library opened at run time (CUDA, OpenCL, NVML). The portable
// backend must not import these, so a machine without them still loads it.

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace cyxwiz::detail {

class DynamicLibrary {
public:
#ifdef _WIN32
    // A driver library is opened from the Windows system folder only: a
    // file of the same name beside a project or on PATH is never loaded.
    explicit DynamicLibrary(const wchar_t* name, bool already_loaded = false)
        : handle_(already_loaded ? GetModuleHandleW(name)
                                 : LoadLibraryExW(name, nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32)),
          owns_(!already_loaded && handle_ != nullptr) {}

    ~DynamicLibrary() {
        if (owns_) FreeLibrary(handle_);
    }

    void* Find(const char* symbol) const {
        return handle_ != nullptr
            ? reinterpret_cast<void*>(GetProcAddress(handle_, symbol))
            : nullptr;
    }

private:
    HMODULE handle_ = nullptr;
    bool owns_ = false;
#else
    explicit DynamicLibrary(const char* name, bool already_loaded = false)
        : handle_(dlopen(name,
                         RTLD_LAZY |
                             (already_loaded ? RTLD_NOLOAD : RTLD_LOCAL))) {}

    ~DynamicLibrary() {
        if (handle_ != nullptr) dlclose(handle_);
    }

    void* Find(const char* symbol) const {
        return handle_ != nullptr ? dlsym(handle_, symbol) : nullptr;
    }

private:
    void* handle_ = nullptr;
#endif

public:
    DynamicLibrary(const DynamicLibrary&) = delete;
    DynamicLibrary& operator=(const DynamicLibrary&) = delete;
    bool loaded() const { return handle_ != nullptr; }
};

template <typename Function>
Function FindFunction(const DynamicLibrary& library, const char* symbol) {
    return reinterpret_cast<Function>(library.Find(symbol));
}

}  // namespace cyxwiz::detail
