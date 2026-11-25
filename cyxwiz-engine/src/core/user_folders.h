#pragma once

// Where the Engine keeps the files it writes. The Engine only reads its own
// folder: an installed base is a signed package (read-only for an all-users
// install) and a base update replaces it. Settings, layout, logs, caches and
// run state go to the user's folders below; a development build keeps its
// log, layout and tutorial state in its build folder.
//
// The folder functions are header-only (any target may include this file).
// The run-file functions at the end are the Engine's own (user_folders.cpp).
//
//                 Windows                 macOS                                 Linux
// UserConfigDir   %APPDATA%\CyxWiz        ~/.cyxwiz                             ~/.cyxwiz
// UserStateDir    %LOCALAPPDATA%\CyxWiz   ~/Library/Application Support/CyxWiz  $XDG_STATE_HOME/cyxwiz (1)
// EngineStateDir  <state>\Engine          <state>/Engine                        <state>/engine
// UserLogDir      <state>\Engine\logs     ~/Library/Logs/CyxWiz                 <state>/engine/logs
// UserCacheDir    <state>\cache           ~/Library/Caches/CyxWiz               $XDG_CACHE_HOME/cyxwiz (2)
// (1) ~/.local/state when unset; (2) ~/.cache when unset.
//
// Never relative: with no user folder in the environment the folders are
// under <temp>/cyxwiz; if even that fails they are empty and the caller
// skips its write. Never under $XDG_DATA_HOME/cyxwiz (the Linux install root,
// which product removal deletes).

#include <cstdlib>
#include <filesystem>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

#if !defined(_WIN32)
#include <pwd.h>
#include <unistd.h>
#endif

namespace cyxwiz {

enum class UserFolderPlatform { Windows, MacOS, Linux };

constexpr UserFolderPlatform CurrentUserFolderPlatform() {
#if defined(_WIN32)
    return UserFolderPlatform::Windows;
#elif defined(__APPLE__)
    return UserFolderPlatform::MacOS;
#else
    return UserFolderPlatform::Linux;
#endif
}

// The environment the folders come from. Relative values count as unset.
struct UserFolderEnvironment {
    std::filesystem::path appdata;        // Windows: APPDATA
    std::filesystem::path local_appdata;  // Windows: LOCALAPPDATA
    std::filesystem::path user_profile;   // Windows: USERPROFILE
    std::filesystem::path home;           // macOS, Linux: HOME
    std::filesystem::path xdg_state_home;
    std::filesystem::path xdg_cache_home;
    std::filesystem::path temp;           // the last resort
};

struct UserFolders {
    std::filesystem::path config;
    std::filesystem::path state;
    std::filesystem::path engine_state;
    std::filesystem::path logs;
    std::filesystem::path cache;
};

namespace user_folders_detail {

inline std::filesystem::path Absolute(std::filesystem::path path) {
    return path.is_absolute() ? path.lexically_normal() : std::filesystem::path{};
}

// `base / leaf`, or empty when `base` is empty (never a relative path).
inline std::filesystem::path Under(const std::filesystem::path& base, const std::filesystem::path& leaf) {
    return base.empty() ? std::filesystem::path{} : (base / leaf).lexically_normal();
}

inline std::mutex& OverrideMutex() {
    static std::mutex mutex;
    return mutex;
}

inline std::optional<std::filesystem::path>& TestRootOverride() {
    static std::optional<std::filesystem::path> root;
    return root;
}

inline std::filesystem::path EnvironmentPath(const char* name) {
#if defined(_WIN32)
    const std::wstring wide_name(name, name + std::char_traits<char>::length(name));
    wchar_t* value = nullptr;
    std::size_t length = 0;
    std::filesystem::path path;
    if (_wdupenv_s(&value, &length, wide_name.c_str()) == 0 && value != nullptr) {
        path = std::filesystem::path(value);
    }
    std::free(value);
    return Absolute(std::move(path));
#else
    const char* value = std::getenv(name);
    return value && *value ? Absolute(std::filesystem::path(value)) : std::filesystem::path{};
#endif
}

inline std::filesystem::path TempDirectory() {
    std::error_code ec;
    auto temp = std::filesystem::temp_directory_path(ec);
    return ec ? std::filesystem::path{} : Absolute(std::move(temp));
}

inline UserFolderEnvironment ProcessEnvironment() {
    UserFolderEnvironment env;
    env.appdata = EnvironmentPath("APPDATA");
    env.local_appdata = EnvironmentPath("LOCALAPPDATA");
    env.user_profile = EnvironmentPath("USERPROFILE");
    env.home = EnvironmentPath("HOME");
#if !defined(_WIN32)
    if (env.home.empty()) {
        if (const passwd* entry = getpwuid(getuid()); entry && entry->pw_dir) {
            env.home = Absolute(std::filesystem::path(entry->pw_dir));
        }
    }
#endif
    env.xdg_state_home = EnvironmentPath("XDG_STATE_HOME");
    env.xdg_cache_home = EnvironmentPath("XDG_CACHE_HOME");
    env.temp = TempDirectory();
    return env;
}

// A test's fake profile: the user folders of the platform under `root`.
inline UserFolderEnvironment OverrideEnvironment(const std::filesystem::path& root) {
    UserFolderEnvironment env;
    env.appdata = root / "AppData" / "Roaming";
    env.local_appdata = root / "AppData" / "Local";
    env.user_profile = root;
    env.home = root;
    env.temp = root / "tmp";
    return env;
}

inline UserFolderEnvironment CurrentEnvironment() {
    {
        std::lock_guard<std::mutex> lock(OverrideMutex());
        if (const auto& root = TestRootOverride(); root.has_value()) {
            return OverrideEnvironment(*root);
        }
    }
    return ProcessEnvironment();
}

}  // namespace user_folders_detail

// The table at the top of this file for one platform and environment.
inline UserFolders ResolveUserFolders(UserFolderPlatform platform, const UserFolderEnvironment& env) {
    using user_folders_detail::Absolute;
    using user_folders_detail::Under;
    const auto temp_root = Under(Absolute(env.temp), "cyxwiz");
    const auto home = Absolute(env.home);
    UserFolders folders;
    switch (platform) {
    case UserFolderPlatform::Windows: {
        const auto profile = Absolute(env.user_profile);
        auto roaming = Absolute(env.appdata);
        if (roaming.empty()) roaming = Under(profile, std::filesystem::path("AppData") / "Roaming");
        auto local = Absolute(env.local_appdata);
        if (local.empty()) local = Under(profile, std::filesystem::path("AppData") / "Local");
        folders.config = roaming.empty() ? Under(temp_root, "config") : Under(roaming, "CyxWiz");
        folders.state = local.empty() ? temp_root : Under(local, "CyxWiz");
        folders.engine_state = Under(folders.state, "Engine");
        folders.logs = Under(folders.engine_state, "logs");
        folders.cache = Under(folders.state, "cache");
        break;
    }
    case UserFolderPlatform::MacOS:
        if (home.empty()) {
            folders.config = Under(temp_root, "config");
            folders.state = temp_root;
            folders.engine_state = Under(folders.state, "Engine");
            folders.logs = Under(folders.engine_state, "logs");
            folders.cache = Under(temp_root, "cache");
        } else {
            folders.config = Under(home, ".cyxwiz");
            folders.state = Under(home, std::filesystem::path("Library") / "Application Support" / "CyxWiz");
            folders.engine_state = Under(folders.state, "Engine");
            folders.logs = Under(home, std::filesystem::path("Library") / "Logs" / "CyxWiz");
            folders.cache = Under(home, std::filesystem::path("Library") / "Caches" / "CyxWiz");
        }
        break;
    case UserFolderPlatform::Linux: {
        auto state_home = Absolute(env.xdg_state_home);
        if (state_home.empty()) state_home = Under(home, std::filesystem::path(".local") / "state");
        auto cache_home = Absolute(env.xdg_cache_home);
        if (cache_home.empty()) cache_home = Under(home, ".cache");
        folders.config = home.empty() ? Under(temp_root, "config") : Under(home, ".cyxwiz");
        folders.state = state_home.empty() ? temp_root : Under(state_home, "cyxwiz");
        folders.engine_state = Under(folders.state, "engine");
        folders.logs = Under(folders.engine_state, "logs");
        folders.cache = cache_home.empty() ? Under(temp_root, "cache") : Under(cache_home, "cyxwiz");
        break;
    }
    }
    return folders;
}

inline UserFolders CurrentUserFolders() {
    return ResolveUserFolders(CurrentUserFolderPlatform(), user_folders_detail::CurrentEnvironment());
}

// Settings, layout and tutorial state; also user plugins, nodes, presets.
inline std::filesystem::path UserConfigDir() { return CurrentUserFolders().config; }
// Machine-local data (compute choice, route evidence).
inline std::filesystem::path UserStateDir() { return CurrentUserFolders().state; }
// The Engine's own machine-local data (debug runs; an installed Engine's working folder).
inline std::filesystem::path EngineStateDir() { return CurrentUserFolders().engine_state; }
// An installed Engine's log (engine_log.txt and the four before it).
inline std::filesystem::path UserLogDir() { return CurrentUserFolders().logs; }
// Data the Engine can rebuild (plugin library copies, Python bytecode).
inline std::filesystem::path UserCacheDir() { return CurrentUserFolders().cache; }

// `folder / name`, or empty when the folder is unknown: never a relative path.
inline std::filesystem::path UserFile(const std::filesystem::path& folder, const std::filesystem::path& name) {
    return user_folders_detail::Under(folder, name);
}

// Where Python keeps compiled bytecode (PYTHONPYCACHEPREFIX) for an installed
// Engine: the user's cache folder, never __pycache__ beside the .py files.
inline std::filesystem::path PythonBytecodeDir() {
    return UserFile(UserCacheDir(), std::filesystem::path("python") / "pycache");
}

// Points this process's Python, and every Python it starts (children inherit
// the environment), at `folder` for compiled bytecode. Empty `folder`: no
// bytecode is written at all. False when the environment cannot be changed.
inline bool SetPythonBytecodeEnvironment(const std::filesystem::path& folder) {
#if defined(_WIN32)
    if (folder.empty()) return _wputenv_s(L"PYTHONDONTWRITEBYTECODE", L"1") == 0;
    return _wputenv_s(L"PYTHONPYCACHEPREFIX", folder.wstring().c_str()) == 0;
#else
    if (folder.empty()) return setenv("PYTHONDONTWRITEBYTECODE", "1", 1) == 0;
    return setenv("PYTHONPYCACHEPREFIX", folder.string().c_str(), 1) == 0;
#endif
}

// What a run with no project open writes ("artifacts", "checkpoints", "prepared").
inline std::filesystem::path NoProjectRunDir(const std::filesystem::path& kind) {
    return UserFile(UserCacheDir(), std::filesystem::path("no-project") / kind);
}

// NoProjectRunDir, or an error that says why there is none: a run never
// writes to a path relative to the working folder.
inline std::filesystem::path RequireNoProjectRunDir(const std::filesystem::path& kind) {
    auto dir = NoProjectRunDir(kind);
    if (dir.empty()) {
        throw std::runtime_error("No user folder is known for the " + kind.string() +
                                 " of a run with no project open. Open a project first.");
    }
    return dir;
}

// The installed runtime the bootstrapper started this process from
// (CYXWIZ_ACTIVE_RUNTIME_ROOT, read wide on Windows); empty in a development
// build or when it is not an absolute path.
inline std::filesystem::path ActiveRuntimeRootFromEnvironment() {
    return user_folders_detail::EnvironmentPath("CYXWIZ_ACTIVE_RUNTIME_ROOT");
}

// Points every folder above into `root` (a fake profile) for a test.
class ScopedUserFoldersOverrideForTesting {
public:
    explicit ScopedUserFoldersOverrideForTesting(std::filesystem::path root) {
        std::error_code ec;
        auto absolute_root = std::filesystem::absolute(root, ec);
        if (ec) absolute_root = std::move(root);
        std::lock_guard<std::mutex> lock(user_folders_detail::OverrideMutex());
        previous_ = user_folders_detail::TestRootOverride();
        user_folders_detail::TestRootOverride() = absolute_root.lexically_normal();
    }

    ~ScopedUserFoldersOverrideForTesting() {
        std::lock_guard<std::mutex> lock(user_folders_detail::OverrideMutex());
        user_folders_detail::TestRootOverride() = std::move(previous_);
    }

    ScopedUserFoldersOverrideForTesting(const ScopedUserFoldersOverrideForTesting&) = delete;
    ScopedUserFoldersOverrideForTesting& operator=(const ScopedUserFoldersOverrideForTesting&) = delete;

private:
    std::optional<std::filesystem::path> previous_;
};

}  // namespace cyxwiz

namespace cyxwiz::core {

// The folder of the running executable (core/executable_directory.cpp, in the
// training-hooks library): what ships beside the Engine is read from here.
std::filesystem::path ExecutableDirectory();

// The files one Engine run keeps for itself, and its working folder.
struct RunFileLocations {
    bool installed = false;
    std::filesystem::path log_dir;            // engine_log.txt, engine_log.1.txt ... engine_log.4.txt
    std::filesystem::path imgui_ini;          // dock layout, open panels, separate windows
    std::filesystem::path tutorial_progress;  // tutorial_progress.json
    std::filesystem::path working_dir;        // set at start; a stray relative write lands here
};

// An installed Engine (`installed`): the user's folders. A development build:
// its build folder (`exe_dir`), so each worktree keeps its own log and layout.
inline RunFileLocations RunFileLocationsFor(bool installed, const std::filesystem::path& exe_dir) {
    RunFileLocations locations;
    locations.installed = installed;
    const auto dev_dir = user_folders_detail::Absolute(exe_dir);
    if (!installed && !dev_dir.empty()) {
        locations.log_dir = dev_dir;
        locations.imgui_ini = dev_dir / "imgui.ini";
        locations.tutorial_progress = dev_dir / "tutorial_progress.json";
        locations.working_dir = dev_dir;
        return locations;
    }
    const auto folders = CurrentUserFolders();
    locations.log_dir = folders.logs;
    locations.imgui_ini = UserFile(folders.config, "imgui.ini");
    locations.tutorial_progress = UserFile(folders.config, "tutorial_progress.json");
    locations.working_dir = folders.engine_state;
    return locations;
}

// Installed: CYXWIZ_ACTIVE_RUNTIME_ROOT is set, or the Engine runs from
// <root>/runtime/base/<id>/ (the rule the Engine's ownership check uses).
bool IsInstalledRun();
// The installed runtime this Engine runs in, launched by the bootstrapper or
// directly (runtime::InstalledRuntimeRoot); empty for a development build.
std::filesystem::path InstalledRuntimeRoot();
RunFileLocations CurrentRunFileLocations();
// engine_log.txt of this run (empty when no folder is known).
std::filesystem::path EngineLogPath();

}  // namespace cyxwiz::core
