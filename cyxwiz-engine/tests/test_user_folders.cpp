// Where the Engine writes (core/user_folders.h): the folder table per
// platform from a redirected environment, never a relative path, the test
// override, and the run files of an installed run and a development build.

#include "core/user_folders.h"

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>

namespace {

namespace fs = std::filesystem;
using cyxwiz::ResolveUserFolders;
using cyxwiz::UserFolderEnvironment;
using cyxwiz::UserFolderPlatform;
using cyxwiz::UserFolders;

int failures = 0;

void Check(bool ok, const std::string& what) {
    if (ok) return;
    ++failures;
    std::printf("FAIL: %s\n", what.c_str());
}

void CheckPath(const fs::path& actual, const fs::path& expected, const std::string& what) {
    Check(actual == expected.lexically_normal(), what + ": got '" + actual.string() + "', expected '" +
                                                     expected.lexically_normal().string() + "'");
}

bool StartsWith(const fs::path& path, const fs::path& prefix) {
    return path.string().rfind(prefix.string(), 0) == 0;
}

void SetVariable(const char* name, const char* value) {
#ifdef _WIN32
    ::_putenv_s(name, value);
#else
    if (*value)
        ::setenv(name, value, 1);
    else
        ::unsetenv(name);
#endif
}

fs::path Root() { return (fs::temp_directory_path() / "cyxwiz-user-folders").lexically_normal(); }

void CheckWindowsTable() {
    const fs::path root = Root();
    UserFolderEnvironment env;
    env.appdata = root / "Roaming";
    env.local_appdata = root / "Local";
    env.temp = root / "tmp";
    const UserFolders f = ResolveUserFolders(UserFolderPlatform::Windows, env);
    CheckPath(f.config, root / "Roaming" / "CyxWiz", "Windows config");
    CheckPath(f.state, root / "Local" / "CyxWiz", "Windows state");
    CheckPath(f.engine_state, root / "Local" / "CyxWiz" / "Engine", "Windows Engine state");
    CheckPath(f.logs, root / "Local" / "CyxWiz" / "Engine" / "logs", "Windows logs");
    CheckPath(f.cache, root / "Local" / "CyxWiz" / "cache", "Windows cache");

    // No APPDATA / LOCALAPPDATA: the profile's AppData folders.
    UserFolderEnvironment profile_only;
    profile_only.user_profile = root / "me";
    const UserFolders p = ResolveUserFolders(UserFolderPlatform::Windows, profile_only);
    CheckPath(p.config, root / "me" / "AppData" / "Roaming" / "CyxWiz", "Windows config from the profile");
    CheckPath(p.logs, root / "me" / "AppData" / "Local" / "CyxWiz" / "Engine" / "logs",
              "Windows logs from the profile");
}

void CheckMacTable() {
    const fs::path root = Root();
    UserFolderEnvironment env;
    env.home = root / "home";
    const UserFolders f = ResolveUserFolders(UserFolderPlatform::MacOS, env);
    const fs::path library = root / "home" / "Library";
    CheckPath(f.config, root / "home" / ".cyxwiz", "macOS config");
    CheckPath(f.state, library / "Application Support" / "CyxWiz", "macOS state");
    CheckPath(f.engine_state, library / "Application Support" / "CyxWiz" / "Engine", "macOS Engine state");
    CheckPath(f.logs, library / "Logs" / "CyxWiz", "macOS logs");
    CheckPath(f.cache, library / "Caches" / "CyxWiz", "macOS cache");
}

void CheckLinuxTable() {
    const fs::path root = Root();
    UserFolderEnvironment env;
    env.home = root / "home";
    const UserFolders f = ResolveUserFolders(UserFolderPlatform::Linux, env);
    const fs::path state = root / "home" / ".local" / "state" / "cyxwiz";
    CheckPath(f.config, root / "home" / ".cyxwiz", "Linux config");
    CheckPath(f.state, state, "Linux state");
    CheckPath(f.engine_state, state / "engine", "Linux Engine state");
    CheckPath(f.logs, state / "engine" / "logs", "Linux logs");
    CheckPath(f.cache, root / "home" / ".cache" / "cyxwiz", "Linux cache");

    env.xdg_state_home = root / "state";
    env.xdg_cache_home = root / "cache";
    const UserFolders x = ResolveUserFolders(UserFolderPlatform::Linux, env);
    CheckPath(x.state, root / "state" / "cyxwiz", "Linux state from XDG_STATE_HOME");
    CheckPath(x.cache, root / "cache" / "cyxwiz", "Linux cache from XDG_CACHE_HOME");
    for (const auto& folder : {x.config, x.state, x.engine_state, x.logs, x.cache}) {
        Check(folder.string().find("share") == std::string::npos,
              "Linux never uses the data folder (the install root): " + folder.string());
    }
}

void CheckNeverRelative() {
    const fs::path temp_root = Root() / "tmp" / "cyxwiz";
    for (const auto platform : {UserFolderPlatform::Windows, UserFolderPlatform::MacOS, UserFolderPlatform::Linux}) {
        // Relative values count as unset: the temp folder is used.
        UserFolderEnvironment relative;
        relative.appdata = "roaming";
        relative.local_appdata = "local";
        relative.user_profile = ".";
        relative.home = ".";
        relative.xdg_state_home = "state";
        relative.xdg_cache_home = "cache";
        relative.temp = Root() / "tmp";
        const UserFolders t = ResolveUserFolders(platform, relative);
        for (const auto& folder : {t.config, t.state, t.engine_state, t.logs, t.cache}) {
            Check(folder.is_absolute(), "a relative environment gives an absolute folder: " + folder.string());
            Check(StartsWith(folder, temp_root), "with no user folder the temp folder is used: " + folder.string());
        }
        // Nothing at all: empty folders (the caller skips its write), never ".".
        const UserFolders none = ResolveUserFolders(platform, UserFolderEnvironment{});
        for (const auto& folder : {none.config, none.state, none.engine_state, none.logs, none.cache}) {
            Check(folder.empty(), "with no folder at all the folder is empty: " + folder.string());
        }
    }
    Check(cyxwiz::UserFile({}, "x.json").empty(), "a file in an unknown folder is an empty path");
}

void CheckOverrideAndRunFiles() {
    const fs::path profile = Root() / "profile";
    cyxwiz::ScopedUserFoldersOverrideForTesting scope(profile);
    const fs::path config = cyxwiz::UserConfigDir();
    Check(StartsWith(config, profile), "the override holds the config folder: " + config.string());
    Check(StartsWith(cyxwiz::UserLogDir(), profile), "the override holds the log folder");
    Check(StartsWith(cyxwiz::UserCacheDir(), profile), "the override holds the cache folder");
    CheckPath(cyxwiz::NoProjectRunDir("artifacts"), cyxwiz::UserCacheDir() / "no-project" / "artifacts",
              "the no-project artifacts folder");
    CheckPath(cyxwiz::RequireNoProjectRunDir("checkpoints"), cyxwiz::UserCacheDir() / "no-project" / "checkpoints",
              "the no-project checkpoints folder");

    const fs::path build = Root() / "build" / "bin";
    const auto dev = cyxwiz::core::RunFileLocationsFor(false, build);
    CheckPath(dev.log_dir, build, "a development build logs in its build folder");
    CheckPath(dev.imgui_ini, build / "imgui.ini", "a development build keeps its layout in its build folder");
    CheckPath(dev.tutorial_progress, build / "tutorial_progress.json", "development tutorial state");
    CheckPath(dev.working_dir, build, "a development build works in its build folder");

    const fs::path base = Root() / "install" / "runtime" / "base" / "base-v1";
    const auto installed = cyxwiz::core::RunFileLocationsFor(true, base);
    CheckPath(installed.log_dir, cyxwiz::UserLogDir(), "an installed Engine logs in the user's log folder");
    CheckPath(installed.imgui_ini, config / "imgui.ini", "an installed Engine keeps its layout in the config folder");
    CheckPath(installed.tutorial_progress, config / "tutorial_progress.json", "installed tutorial state");
    CheckPath(installed.working_dir, cyxwiz::EngineStateDir(), "an installed Engine works in its state folder");
    for (const auto& path : {installed.log_dir, installed.imgui_ini, installed.tutorial_progress,
                             installed.working_dir}) {
        Check(!StartsWith(path, base), "nothing of an installed run is in the base: " + path.string());
    }
}

// A test executable runs in its own scratch profile from the start
// (tests/support/scratch_user_folders.cpp), never in the user's folders.
void CheckTestScratchDefault() {
    const fs::path scratch = (fs::temp_directory_path() / "cyxwiz-tests").lexically_normal();
    for (const auto& folder : {cyxwiz::UserConfigDir(), cyxwiz::UserStateDir(), cyxwiz::UserLogDir(),
                               cyxwiz::UserCacheDir()}) {
        Check(StartsWith(folder, scratch), "a test's user folder is in its scratch profile: " + folder.string());
        Check(folder.string().find("test_user_folders-") != std::string::npos,
              "the scratch profile is named after the test executable: " + folder.string());
    }
    const fs::path before = cyxwiz::UserStateDir();
    {
        cyxwiz::ScopedUserFoldersOverrideForTesting scope(Root() / "inner");
        Check(StartsWith(cyxwiz::UserStateDir(), Root() / "inner"), "a test's own override wins");
    }
    CheckPath(cyxwiz::UserStateDir(), before, "closing a test's own override restores the scratch profile");
}

// The real folders as the Engine computes them from this process's
// environment (what the scratch profile stands in for). Only computed, never
// written.
std::string Environment(const char* name) {
#ifdef _WIN32
    char* value = nullptr;
    std::size_t length = 0;
    std::string text;
    if (_dupenv_s(&value, &length, name) == 0 && value) text = value;
    std::free(value);
    return text;
#else
    const char* value = std::getenv(name);
    return value ? value : "";
#endif
}

void CheckProcessEnvironmentResolution() {
    const UserFolders real = ResolveUserFolders(cyxwiz::CurrentUserFolderPlatform(),
                                                cyxwiz::user_folders_detail::ProcessEnvironment());
#ifdef _WIN32
    if (const fs::path local = Environment("LOCALAPPDATA"); local.is_absolute()) {
        CheckPath(real.state, local / "CyxWiz", "the real state folder comes from LOCALAPPDATA");
        CheckPath(real.cache, local / "CyxWiz" / "cache", "the real cache folder");
    }
    if (const fs::path roaming = Environment("APPDATA"); roaming.is_absolute()) {
        CheckPath(real.config, roaming / "CyxWiz", "the real config folder comes from APPDATA");
    }
#else
    if (const fs::path home = Environment("HOME"); home.is_absolute()) {
        CheckPath(real.config, home / ".cyxwiz", "the real config folder comes from HOME");
    }
#endif
    const fs::path scratch = (fs::temp_directory_path() / "cyxwiz-tests").lexically_normal();
    Check(!StartsWith(real.state, scratch), "the real folders are not the scratch profile: " + real.state.string());
}

void CheckInstalledRun() {
    SetVariable("CYXWIZ_ACTIVE_RUNTIME_ROOT", "");
    Check(!cyxwiz::core::IsInstalledRun(), "a test in the build folder is a development run");
    const std::string runtime = (Root() / "install" / "runtime").string();
    SetVariable("CYXWIZ_ACTIVE_RUNTIME_ROOT", runtime.c_str());
    Check(cyxwiz::core::IsInstalledRun(), "the bootstrapper's runtime variable makes an installed run");
    Check(cyxwiz::core::CurrentRunFileLocations().installed, "the run files follow the installed run");
    SetVariable("CYXWIZ_ACTIVE_RUNTIME_ROOT", "");
}

}  // namespace

int main() {
    CheckTestScratchDefault();
    CheckProcessEnvironmentResolution();
    CheckWindowsTable();
    CheckMacTable();
    CheckLinuxTable();
    CheckNeverRelative();
    CheckOverrideAndRunFiles();
    CheckInstalledRun();
    if (failures) {
        std::printf("%d user folder check(s) failed\n", failures);
        return 1;
    }
    std::printf("User folder checks passed\n");
    return 0;
}
