#include "application.h"
#include "plugin/plugin_manager.h"
#include <cyxwiz/cyxwiz.h>
#include <spdlog/spdlog.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include "core/runtime_log_sink.h"
#include "core/compute_runtime_config.h"
#include "core/compute_runtime_paths.h"
#include "core/execution_device_preferences.h"
#include "core/route_qualification_snapshot.h"
#ifdef _WIN32
#include "windows_dll_search.h"
#endif
#include <iostream>
#include <filesystem>
#include <cstdlib>

#ifdef _WIN32
#include <windows.h>
#elif defined(__APPLE__)
#include <mach-o/dyld.h>
#include <limits.h>
#elif defined(__linux__)
#include <unistd.h>
#include <limits.h>
#endif

namespace {

std::filesystem::path GetExecutableDir() {
#ifdef _WIN32
    char buffer[MAX_PATH];
    DWORD len = GetModuleFileNameA(nullptr, buffer, MAX_PATH);
    if (len == 0 || len == MAX_PATH) {
        return {};
    }
    return std::filesystem::path(std::string(buffer, len)).parent_path();
#elif defined(__APPLE__)
    char exec_path[PATH_MAX];
    uint32_t size = sizeof(exec_path);
    if (_NSGetExecutablePath(exec_path, &size) != 0) {
        return {};
    }
    return std::filesystem::path(exec_path).parent_path();
#elif defined(__linux__)
    char exec_path[PATH_MAX];
    ssize_t len = readlink("/proc/self/exe", exec_path, sizeof(exec_path) - 1);
    if (len <= 0) {
        return {};
    }
    exec_path[len] = '\0';
    return std::filesystem::path(exec_path).parent_path();
#else
    return {};
#endif
}

void SetLaunchCwdEnv(const std::filesystem::path& cwd) {
#ifdef _WIN32
    _putenv_s("CYXWIZ_LAUNCH_CWD", cwd.string().c_str());
#else
    setenv("CYXWIZ_LAUNCH_CWD", cwd.string().c_str(), 1);
#endif
}

} // namespace

int main(int argc, char** argv) {
#ifdef _WIN32
    std::string runtime_search_error;
    if (!cyxwiz::runtime::ConfigureActiveRuntimeDllSearchFromEnvironment(
            runtime_search_error)) {
        std::cerr << "CyxWiz runtime setup failed: "
                  << runtime_search_error << std::endl;
        return 78;
    }
#endif
    std::filesystem::path launch_cwd = std::filesystem::current_path();
    SetLaunchCwdEnv(launch_cwd);

    std::filesystem::path exe_dir = GetExecutableDir();
    bool cwd_set = false;
    if (!exe_dir.empty()) {
        std::error_code ec;
        std::filesystem::current_path(exe_dir, ec);
        if (ec) {
            std::cerr << "Failed to set working directory to executable directory: " << ec.message() << std::endl;
        } else {
            cwd_set = true;
        }
    }

    // Setup logging with file output
    try {
        // Get executable directory for log file
        std::filesystem::path exe_path = std::filesystem::current_path();
        std::filesystem::path log_path = exe_path / "engine_log.txt";

        auto file_sink = std::make_shared<spdlog::sinks::basic_file_sink_mt>(log_path.string(), true);
        auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
        auto runtime_sink = std::make_shared<cyxwiz::RuntimeLogSinkMt>(
            cyxwiz::RuntimeLogStore::Instance());

        auto logger = std::make_shared<spdlog::logger>(
            "cyxwiz",
            spdlog::sinks_init_list{file_sink, console_sink, runtime_sink});
        logger->set_level(spdlog::level::info);
        spdlog::set_default_logger(logger);
        spdlog::flush_on(spdlog::level::info);
    } catch (const spdlog::spdlog_ex& ex) {
        std::cerr << "Log initialization failed: " << ex.what() << std::endl;
    }

#ifdef CYXWIZ_DEBUG
    spdlog::set_level(spdlog::level::debug);
#endif

    if (cwd_set) {
        spdlog::info("Working directory set to executable directory: {}", exe_dir.string());
    }
    spdlog::info("Launch working directory: {}", launch_cwd.string());

    spdlog::info("Starting CyxWiz Engine v{}", cyxwiz::GetVersionString());

    const auto qualification_cache =
        cyxwiz::GetRouteQualificationCachePath();
    const auto legacy_qualification =
        std::filesystem::current_path() /
        "arrayfire-route-qualification.json";
    std::error_code qualification_path_error;
    const bool qualification_cache_exists =
        std::filesystem::exists(
            qualification_cache, qualification_path_error) &&
        !qualification_path_error;
    auto qualification =
        qualification_cache_exists
            ? cyxwiz::LoadAndInstallRouteQualificationSnapshot(
                  qualification_cache)
            : cyxwiz::LoadAndInstallRouteQualificationSnapshot(
                  legacy_qualification);
    if (qualification.loaded && !qualification_cache_exists) {
        std::string migration_error;
        const auto snapshot = cyxwiz::GetRouteQualificationSnapshot();
        if (snapshot.has_value() &&
            cyxwiz::SaveRouteQualificationSnapshotAtomic(
                qualification_cache, *snapshot, migration_error)) {
            spdlog::info(
                "Migrated route qualification evidence to machine-local cache {}",
                qualification_cache.string());
        } else {
            spdlog::warn(
                "Could not migrate route qualification evidence to {}: {}",
                qualification_cache.string(), migration_error);
        }
    }
    if (qualification.loaded) {
        spdlog::info(
            "Loaded route qualification evidence for {} routes",
            qualification.route_count);
    } else {
        spdlog::warn(
            "Compute route changes are disabled until qualification evidence "
            "is available: {}",
            qualification.message);
    }

    // Initialize backend
    if (!cyxwiz::Initialize()) {
        spdlog::error("Failed to initialize CyxWiz backend");
        return 1;
    }

    const auto runtime_config_path =
        cyxwiz::GetComputeRuntimeConfigPath();
    const auto runtime_config =
        cyxwiz::LoadComputeRuntimeConfig(runtime_config_path);
    if (runtime_config.loaded) {
        cyxwiz::SetNextRunExecutionPolicy(
            runtime_config.config.default_fallback_policy);
        if (runtime_config.config.preferred_route.has_value()) {
            const auto& preferred =
                *runtime_config.config.preferred_route;
            cyxwiz::CommitExecutionDeviceSelectionState(
                {preferred.type,
                 preferred.last_device_id,
                 preferred.physical_fingerprint});
            spdlog::info(
                "Loaded machine compute preference: backend={} device_hint={} identity={}",
                cyxwiz::ExecutionDeviceSelectionBackendName(preferred.type),
                preferred.last_device_id,
                preferred.physical_fingerprint.empty()
                    ? "backend_local"
                    : preferred.physical_fingerprint);
        }
    } else if (runtime_config.file_exists) {
        spdlog::warn(
            "Machine compute runtime configuration was not applied: {}",
            runtime_config.message);
    } else {
        std::string config_error;
        if (!cyxwiz::SaveComputeRuntimeConfigAtomic(
                runtime_config_path, runtime_config.config, config_error)) {
            spdlog::warn(
                "Could not initialize machine compute runtime configuration: {}",
                config_error);
        }
    }

    // List available devices
    auto devices = cyxwiz::Device::GetAvailableDevices();
    spdlog::info("Available compute devices:");
    for (const auto& device : devices) {
        spdlog::info("  - {} ({})", device.name, static_cast<int>(device.type));
    }

    // Create and run application
    try {
        CyxWizApp app(argc, argv);

        // Initialize plugin system
        {
            auto& pm = cyxwiz::plugin::PluginManager::Instance();
            std::filesystem::path plugin_base_dir = std::filesystem::current_path();
            std::filesystem::path user_dir;
#ifdef _WIN32
            if (auto* appdata = std::getenv("APPDATA"))
                user_dir = std::filesystem::path(appdata) / "cyxwiz" / "plugins";
#else
            if (auto* home = std::getenv("HOME"))
                user_dir = std::filesystem::path(home) / ".cyxwiz" / "plugins";
#endif
            std::vector<std::filesystem::path> search_paths;
            search_paths.push_back(plugin_base_dir / "plugins");
            if (!user_dir.empty()) search_paths.push_back(user_dir);
            pm.SetSearchPaths(search_paths);
            // Plugins are loaded on-demand from the Plugin Manager panel (sidebar)
            spdlog::info("Plugin system: {} search paths configured", search_paths.size());
        }

        int result = app.Run();

        // Cleanup plugins before backend
        cyxwiz::plugin::PluginManager::Instance().ShutdownAll();
        cyxwiz::plugin::PluginManager::Instance().UnloadAll();

        cyxwiz::Shutdown();

        return result;
    } catch (const std::exception& e) {
        spdlog::error("Fatal error: {}", e.what());
        cyxwiz::Shutdown();
        return 1;
    }
}
