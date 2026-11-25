// Every test, benchmark and tool executable of the Engine and its plugins
// gets this file (cmake/CyxWizTestScratchUserFolders.cmake): before main()
// runs, the user folders of core/user_folders.h point into a scratch profile
// of its own, <temp>/cyxwiz-tests/<executable name>-<process id>, so a test
// never writes into the user's real settings, logs or caches (no-project run
// folders, plugin library copies, presets). The folder is removed at exit.
//
// In-process only: the environment (APPDATA, LOCALAPPDATA, HOME) is not
// changed, so Python's user site and downloaded models are still found, and
// child processes use the real folders. A test that needs its own profile
// still opens a ScopedUserFoldersOverrideForTesting; it restores this one.

#include "../../src/core/user_folders.h"

#include <cstdio>
#include <filesystem>
#include <string>
#include <system_error>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <process.h>
#include <windows.h>
#elif defined(__APPLE__)
#include <mach-o/dyld.h>
#include <unistd.h>
#include <vector>
#else
#include <unistd.h>
#endif

namespace {

namespace fs = std::filesystem;

std::string ExecutableStem() {
    fs::path exe;
#if defined(_WIN32)
    std::wstring buffer(MAX_PATH, L'\0');
    for (;;) {
        const DWORD length = ::GetModuleFileNameW(nullptr, buffer.data(), static_cast<DWORD>(buffer.size()));
        if (length == 0) break;
        if (length < buffer.size()) {
            buffer.resize(length);
            exe = fs::path(buffer);
            break;
        }
        buffer.resize(buffer.size() * 2);
    }
#elif defined(__APPLE__)
    uint32_t size = 0;
    _NSGetExecutablePath(nullptr, &size);
    std::vector<char> buffer(size + 1, '\0');
    if (_NSGetExecutablePath(buffer.data(), &size) == 0) exe = fs::path(buffer.data());
#else
    std::error_code ec;
    exe = fs::read_symlink("/proc/self/exe", ec);
#endif
    const std::string stem = exe.stem().string();
    return stem.empty() ? std::string("test") : stem;
}

long ProcessId() {
#if defined(_WIN32)
    return static_cast<long>(::_getpid());
#else
    return static_cast<long>(::getpid());
#endif
}

fs::path ScratchRoot() {
    std::error_code ec;
    fs::path temp = fs::temp_directory_path(ec);
    if (ec || temp.empty()) return {};
    return (temp / "cyxwiz-tests" / (ExecutableStem() + "-" + std::to_string(ProcessId()))).lexically_normal();
}

class ScratchUserFolders {
public:
    ScratchUserFolders() : root_(ScratchRoot()) {
        if (root_.empty()) {
            std::fprintf(stderr, "No temp folder: this run uses the real user folders\n");
            return;
        }
        // A crashed run with the same executable and process id left it: start empty.
        if (IsScratchRoot(root_)) {
            std::error_code ec;
            fs::remove_all(root_, ec);
        }
        // Never released: a thread or a static destructor that runs after this
        // object still writes into the scratch folder, never the real one.
        override_ = new cyxwiz::ScopedUserFoldersOverrideForTesting(root_);
    }

    ~ScratchUserFolders() {
        // The override stays set; only the folder goes (a library still loaded
        // from it stays, and a late write may make it again: it is scratch).
        if (!override_ || !IsScratchRoot(root_)) return;
        std::error_code ec;
        fs::remove_all(root_, ec);
    }

    ScratchUserFolders(const ScratchUserFolders&) = delete;
    ScratchUserFolders& operator=(const ScratchUserFolders&) = delete;

private:
    // Only this process's folder, and only under <temp>/cyxwiz-tests.
    static bool IsScratchRoot(const fs::path& root) {
        return !root.empty() && root.parent_path().filename() == "cyxwiz-tests";
    }

    fs::path root_;
    cyxwiz::ScopedUserFoldersOverrideForTesting* override_ = nullptr;  // never deleted (see above)
};

}  // namespace

// Before the executable's own static objects, so none of them sees the real
// folders; its folder is removed after them.
#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4073)  // initializers put in library initialization area: intended
#pragma init_seg(lib)
#pragma warning(pop)
static ScratchUserFolders scratch_user_folders;
#elif defined(__GNUC__) && !defined(__APPLE__)
static ScratchUserFolders scratch_user_folders __attribute__((init_priority(101)));
#else
static ScratchUserFolders scratch_user_folders;
#endif
