#include "application.h"
#include "engine_runtime_ownership.h"
#include "runtime_layout.h"
#include "backend_pack_platform.h"
#include "plugin/plugin_manager.h"
#include "plugin/plugin_pack_trust.h"
#include "plugin/plugin_data_formats.h"
#include "plugin/plugin_session.h"
#include "plugin/engine_plugin_context.h"
#include "core/engine_config.h"
#include "core/project_manager.h"
#include "gui/plugin_ui_kit.h"
#include <cyxwiz/cyxwiz.h>
#include <spdlog/spdlog.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include "core/runtime_log_sink.h"
#include "core/arrayfire_backend_discovery_isolation.h"
#include "core/compute_runtime_config.h"
#include "core/compute_runtime_paths.h"
#include "core/execution_device_preferences.h"
#include "core/machine_compute_preference.h"
#include "core/route_qualification_snapshot.h"
#include "core/python_package_smoke.h"
#include "core/user_folders.h"
#include "gui/node_editor.h"
#include <imgui.h>
#include <imnodes.h>
#include <fstream>
#ifdef _WIN32
#include "windows_dll_search.h"
#endif
#include <iostream>
#include <filesystem>
#include <map>
#include <cstdlib>
#include <exception>
#include <cmath>
#include <string_view>
#include <vector>

namespace {

void SetLaunchCwdEnv(const std::filesystem::path& cwd) {
#ifdef _WIN32
    _putenv_s("CYXWIZ_LAUNCH_CWD", cwd.string().c_str());
#else
    setenv("CYXWIZ_LAUNCH_CWD", cwd.string().c_str(), 1);
#endif
}

// cyxwiz-engine --export-code <pytorch|tensorflow|keras|pycyxwiz> <graph> <out>:
// the code the Node Editor generates for a saved graph, written headless (no
// window) - the TOFIX112 export harness runs it against CyxWiz's model.
bool IsExportCodeRequested(int argc, char** argv) {
    return argc == 5 && std::string_view(argv[1]) == "--export-code";
}

int RunExportCode(const std::string& framework_name, const std::filesystem::path& graph,
                  const std::filesystem::path& out) {
    cyxwiz::CodeFramework framework;
    if (framework_name == "pytorch") framework = cyxwiz::CodeFramework::PyTorch;
    else if (framework_name == "tensorflow") framework = cyxwiz::CodeFramework::TensorFlow;
    else if (framework_name == "keras") framework = cyxwiz::CodeFramework::Keras;
    else if (framework_name == "pycyxwiz") framework = cyxwiz::CodeFramework::PyCyxWiz;
    else {
        std::cerr << "unknown framework '" << framework_name << "' (pytorch, tensorflow, keras, pycyxwiz)\n";
        return 2;
    }
    ImGui::CreateContext();
    ImNodes::CreateContext();
    int result = 0;
    {
        gui::NodeEditor editor;
        if (!editor.LoadGraph(graph.string())) {
            std::cerr << "cannot load " << graph.string() << "\n";
            result = 2;
        } else {
            const std::string code = editor.GenerateCodeText(framework);
            std::ofstream file(out, std::ios::binary | std::ios::trunc);
            if (code.empty() || !file || !(file << code)) {
                std::cerr << "no code generated for " << graph.string() << "\n";
                result = 1;
            }
        }
    }
    ImNodes::DestroyContext();
    ImGui::DestroyContext();
    return result;
}

bool IsPackageSmokeRequested(int argc, char** argv) {
    return argc == 2 && std::string_view(argv[1]) == "--package-smoke";
}

const char* EnvironmentValue(const char* name) {
    const char* value = std::getenv(name);
    return value != nullptr ? value : "";
}

int RunPackageSmoke() {
    constexpr const char* kForbiddenOverrides[] = {
        "AF_PLUGIN_PATH",
        "CYXWIZ_ARRAYFIRE_DIR",
        "AF_BUILD_PATH",
        "AF_BUILD_LIB_CUSTOM_PATH",
        "PYTHONHOME",
        "PYTHONPATH",
    };
    if (std::string_view(EnvironmentValue("CYXWIZ_ACTIVE_RUNTIME_ROOT")).empty()) {
        std::cerr << "package_smoke schema=1 status=fail "
                     "reason=active_runtime_missing\n";
        return 78;
    }
    const auto af_path = std::filesystem::path(EnvironmentValue("AF_PATH"));
    const auto expected_af_path = cyxwiz::core::ExecutableDirectory() / "arrayfire";
    if (!af_path.empty() &&
        af_path.lexically_normal() != expected_af_path.lexically_normal()) {
        std::cerr << "package_smoke schema=1 status=fail "
                     "reason=runtime_override_present variable=AF_PATH\n";
        return 78;
    }
    for (const char* name : kForbiddenOverrides) {
        if (!std::string_view(EnvironmentValue(name)).empty()) {
            std::cerr << "package_smoke schema=1 status=fail "
                         "reason=runtime_override_present variable="
                      << name << '\n';
            return 78;
        }
    }
    if (std::string_view(EnvironmentValue("PATH")).find(
            "cyxwiz-untrusted-marker") != std::string_view::npos) {
        std::cerr << "package_smoke schema=1 status=fail "
                     "reason=inherited_path_present\n";
        return 78;
    }

    try {
        const auto activation =
            cyxwiz::Device(cyxwiz::DeviceType::CPU, 0).ActivateExact(true);
        if (!activation.success || !activation.execution_validated ||
            activation.effective_type != cyxwiz::DeviceType::CPU ||
            activation.effective_device_id != 0) {
            std::cerr << "package_smoke schema=1 status=fail "
                         "reason=cpu_activation_failed stage="
                      << cyxwiz::DeviceActivationStageName(activation.stage)
                      << " error=" << activation.error_code << '\n';
            return 1;
        }

        const float left_values[] = {1.0f, 2.0f, 3.0f, 4.0f};
        const float right_values[] = {5.0f, 6.0f, 7.0f, 8.0f};
        const cyxwiz::Tensor left(
            {2, 2}, left_values, cyxwiz::DataType::Float32);
        const cyxwiz::Tensor right(
            {2, 2}, right_values, cyxwiz::DataType::Float32);
        const cyxwiz::Tensor checksum = (left * right).Sum();
        const float value = checksum.ReadData<float>()[0];
        if (!std::isfinite(value) || std::abs(value - 70.0f) > 0.001f) {
            std::cerr << "package_smoke schema=1 status=fail "
                         "reason=cpu_result_mismatch\n";
            return 1;
        }

        std::string python_error;
        const std::string python_version =
            cyxwiz::core::RunBundledPythonSmoke(cyxwiz::core::ExecutableDirectory(), &python_error);
        if (python_version.empty()) {
            std::cerr << "package_smoke schema=1 status=fail reason="
                      << python_error << '\n';
            return 1;
        }

        std::cout << "package_smoke schema=1 status=pass "
                     "effective_backend=cpu effective_device=0 "
                     "runtime_isolation=pass checksum="
                  << value << " python=" << python_version << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "package_smoke schema=1 status=fail "
                     "reason=execution_error detail='"
                  << error.what() << "'\n";
        return 1;
    }
}

} // namespace

int main(int argc, char** argv) {
    cyxwiz::runtime::RuntimeOperationLock engine_ownership;
    std::string ownership_error;
    if (!cyxwiz::runtime::AcquirePackagedEngineOwnership(
            cyxwiz::core::ExecutableDirectory() / cyxwiz::runtime::CurrentEngineExecutableName(),
            cyxwiz::ActiveRuntimeRootFromEnvironment(), engine_ownership, ownership_error)) {
        std::cerr << "CyxWiz Engine launch is blocked: " << ownership_error << '\n';
        return 78;
    }
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

    std::filesystem::path exe_dir = cyxwiz::core::ExecutableDirectory();
    // The Engine only reads its own folder. An installed Engine keeps its log,
    // layout and working folder in the user's folders; a development build in
    // its build folder (core/user_folders.h). A stray relative write then
    // lands in the working folder, never in the installed base.
    const auto run_files = cyxwiz::core::CurrentRunFileLocations();
    std::string working_dir_problem;  // logged once the logger exists
    if (run_files.working_dir.empty()) {
        working_dir_problem = "no user state folder is known";
    } else {
        std::error_code ec;
        std::filesystem::create_directories(run_files.working_dir, ec);
        std::filesystem::current_path(run_files.working_dir, ec);
        if (ec) working_dir_problem = ec.message();
    }
    // An installed Engine's Python, and every Python it starts (the Python
    // check, venv creation, a project's venv, pip), keeps compiled bytecode in
    // the user's cache folder: the bundled Python in the base is read-only and
    // inventoried. Set before anything starts Python; children inherit it.
    const bool bytecode_set =
        !run_files.installed || cyxwiz::SetPythonBytecodeEnvironment(cyxwiz::PythonBytecodeDir());

    // Logging: the console, the Console panel and engine_log.txt. Without a
    // writable log folder the Engine still runs and says why it has no file.
    const auto log_path = cyxwiz::UserFile(run_files.log_dir, "engine_log.txt");
    std::string log_file_problem;
    try {
        std::vector<spdlog::sink_ptr> sinks;
        if (log_path.empty()) {
            log_file_problem = "no log folder is known";
        } else {
            std::error_code ec;
            std::filesystem::create_directories(run_files.log_dir, ec);
            // Keep the last five launches (TOFIX159 D7): engine_log.txt is this one,
            // engine_log.1.txt the one before, ... engine_log.4.txt the oldest. A
            // crashed launch's log survives the restart that follows it.
            const auto numbered = [&run_files](int n) {
                return run_files.log_dir / ("engine_log." + std::to_string(n) + ".txt");
            };
            std::filesystem::remove(numbered(4), ec);
            for (int n = 3; n >= 1; --n) std::filesystem::rename(numbered(n), numbered(n + 1), ec);
            std::filesystem::rename(log_path, numbered(1), ec);
            try {
                sinks.push_back(std::make_shared<spdlog::sinks::basic_file_sink_mt>(log_path.string(), true));
            } catch (const spdlog::spdlog_ex& ex) {
                log_file_problem = ex.what();
            }
        }
        sinks.push_back(std::make_shared<spdlog::sinks::stdout_color_sink_mt>());
        sinks.push_back(std::make_shared<cyxwiz::RuntimeLogSinkMt>(cyxwiz::RuntimeLogStore::Instance()));

        auto logger = std::make_shared<spdlog::logger>("cyxwiz", sinks.begin(), sinks.end());
        logger->set_level(spdlog::level::info);
        spdlog::set_default_logger(logger);
        spdlog::flush_on(spdlog::level::info);
    } catch (const spdlog::spdlog_ex& ex) {
        std::cerr << "Log initialization failed: " << ex.what() << std::endl;
    }

#ifdef CYXWIZ_DEBUG
    spdlog::set_level(spdlog::level::debug);
#endif

    // An exception nobody catches used to end the Engine with no trace in
    // engine_log.txt (TOFIX146); record what it was before the process aborts.
    std::set_terminate([] {
        std::string reason = "no active exception";
        if (const auto current = std::current_exception()) {
            try {
                std::rethrow_exception(current);
            } catch (const std::exception& e) {
                reason = e.what();
            } catch (...) {
                reason = "non-standard exception";
            }
        }
        spdlog::critical("Unhandled exception, the Engine must close: {}", reason);
        spdlog::shutdown();
        std::abort();
    });

    if (working_dir_problem.empty()) {
        spdlog::debug("Working folder: {}", run_files.working_dir.string());
    } else {
        spdlog::warn("The Engine could not work in its folder {} ({}); it works in {}",
                     run_files.working_dir.string(), working_dir_problem,
                     std::filesystem::current_path().string());
    }
    if (!bytecode_set)
        spdlog::warn("Python's bytecode folder could not be set; Python may write __pycache__ beside its files");
    spdlog::debug("Launch working directory: {}", launch_cwd.string());

    spdlog::info("Starting CyxWiz Engine {} from {}", cyxwiz::GetVersionString(), exe_dir.string());
    if (log_file_problem.empty()) {
        spdlog::debug("Engine log: {}", log_path.string());
    } else {
        spdlog::warn("The Engine log is not written to a file ({}): {}", log_path.string(), log_file_problem);
    }

    const auto qualification_cache =
        cyxwiz::GetRouteQualificationCachePath();
    auto qualification =
        cyxwiz::LoadAndInstallRouteQualificationSnapshot(qualification_cache);
    if (qualification.loaded) {
        spdlog::debug(
            "Loaded route qualification evidence for {} routes",
            qualification.route_count);
    } else {
        spdlog::warn(
            "Compute route changes are disabled until qualification evidence "
            "is available: {}",
            qualification.message);
    }

    // Initialize ArrayFire while packaged backend discovery is constrained to
    // the immutable active runtime. The scope restores the ordinary process
    // environment after ArrayFire has cached its backend handles.
    {
        cyxwiz::ScopedArrayFireBackendDiscoveryIsolation isolation;
        std::string isolation_error;
        if (!isolation.Apply(isolation_error)) {
            spdlog::error(
                "Failed to isolate ArrayFire backend discovery: {}",
                isolation_error);
            return 78;
        }
        if (!cyxwiz::Initialize()) {
            spdlog::error("Failed to initialize CyxWiz backend");
            return 1;
        }
    }

    if (IsPackageSmokeRequested(argc, argv)) {
        const int result = RunPackageSmoke();
        cyxwiz::Shutdown();
        return result;
    }
    if (IsExportCodeRequested(argc, argv)) {
        const int result = RunExportCode(argv[2], argv[3], argv[4]);
        cyxwiz::Shutdown();
        return result;
    }

    const auto runtime_config_path =
        cyxwiz::GetComputeRuntimeConfigPath();
    const auto runtime_config = cyxwiz::ApplyMachineComputePreference();
    if (runtime_config.loaded) {
        if (runtime_config.config.preferred_route.has_value()) {
            const auto& preferred =
                *runtime_config.config.preferred_route;
            spdlog::info(
                "Loaded machine compute preference: backend={} device_hint={} identity={}",
                cyxwiz::ExecutionDeviceSelectionBackendName(preferred.type),
                preferred.last_device_id,
                preferred.physical_fingerprint.empty()
                    ? "backend_local"
                    : preferred.physical_fingerprint);
        }
    } else if (runtime_config.older_schema) {
        spdlog::warn(
            "Saved compute choice not applied: {}. Automatic device selection "
            "is in use; choose the device again in Preferences > Devices to "
            "save it ({})",
            runtime_config.message, runtime_config_path.string());
    } else if (runtime_config.file_exists) {
        spdlog::warn(
            "Saved compute choice not applied: {}. Automatic device selection "
            "is in use ({})",
            runtime_config.message, runtime_config_path.string());
    } else {
        std::string config_error;
        if (!cyxwiz::SaveComputeRuntimeConfigAtomic(
                runtime_config_path, runtime_config.config, config_error)) {
            spdlog::warn(
                "Could not initialize machine compute runtime configuration: {}",
                config_error);
        }
    }

    // The devices: one line each at Debug with its backend and index (the
    // number used to be the backend's enum value); one count per backend at Info.
    auto devices = cyxwiz::Device::GetAvailableDevices();
    std::map<std::string, int> per_backend;
    for (const auto& device : devices) {
        const std::string backend = cyxwiz::ExecutionDeviceSelectionBackendName(device.type);
        spdlog::debug("Compute device: {} ({} device {})", device.name, backend, device.device_id);
        ++per_backend[backend];
    }
    std::string counts;
    for (const auto& [backend, count] : per_backend)
        counts += (counts.empty() ? "" : ", ") + backend + " " + std::to_string(count);
    spdlog::info("Compute devices: {} ({})", devices.size(), counts);

    // Create and run application
    try {
        CyxWizApp app(argc, argv);

        // Initialize plugin system
        {
            auto& pm = cyxwiz::plugin::PluginManager::Instance();
            // The Engine's own plugins/ folder, then the user's.
            const auto user_dir = cyxwiz::UserFile(cyxwiz::UserConfigDir(), "plugins");
            std::vector<std::filesystem::path> search_paths;
            if (!exe_dir.empty()) search_paths.push_back(exe_dir / "plugins");
            if (!user_dir.empty()) search_paths.push_back(user_dir);
            pm.SetSearchPaths(search_paths);
            spdlog::debug("Plugin system: {} search paths configured", search_paths.size());
            // An installed Engine also loads the add-ons (plugin packs) active
            // in its runtime (runtime/active-plugins.json), from their exact
            // folders, each checked against its signed inventory first.
            // Also when the installed Engine was started directly (a shortcut,
            // a double-click in the install folder): the same add-ons and the
            // same checks as through the bootstrapper.
            if (const auto runtime_root = cyxwiz::core::InstalledRuntimeRoot(); !runtime_root.empty()) {
                cyxwiz::plugin::ConfigureInstalledPluginPacks(pm, runtime_root);
            }
            // Plugins loaded in an earlier session and set to load at startup
            // (TOFIX125 step 2.9); others load from the Plugin Manager.
            // Plugin dataset formats reach Data Input through the core's hook (TOFIX181).
            cyxwiz::plugin::InstallPluginDataFormats();
            // Plugin panels draw the Engine's widgets (plugin API 1.4).
            cyxwiz::plugin::EnginePluginContext::SetUi(&cyxwiz::ui::PluginUiKit());
            // The open project's folder and the plugins' saved settings (plugin API 1.4)
            cyxwiz::plugin::EnginePluginContext::SetHostServices({
                [] {
                    auto& projects = cyxwiz::ProjectManager::Instance();
                    return projects.HasActiveProject() ? std::filesystem::path(projects.GetProjectRoot())
                                                       : std::filesystem::path{};
                },
                [](const std::string& plugin_id, const std::string& key, std::string& value) {
                    return cyxwiz::core::EngineConfig::Instance().PluginSetting(plugin_id, key, value);
                },
                [](const std::string& plugin_id, const std::string& key, const std::string& value) {
                    auto& config = cyxwiz::core::EngineConfig::Instance();
                    config.SetPluginSetting(plugin_id, key, value);
                    config.Save();
                },
            });
            cyxwiz::plugin::LoadStartupPlugins(pm);
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
