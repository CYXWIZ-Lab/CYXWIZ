#include "runtime_layout.h"
#include "backend_pack_maintenance_request.h"
#include "backend_pack_platform.h"

#include <cerrno>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <string>
#include <string_view>
#include <vector>

#include <fcntl.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#ifdef __APPLE__
#include <mach-o/dyld.h>
#endif

namespace {

std::filesystem::path ExecutableDirectory() {
#ifdef __APPLE__
    std::uint32_t size = 0;
    ::_NSGetExecutablePath(nullptr, &size);
    std::vector<char> buffer(size);
    if (size == 0 || ::_NSGetExecutablePath(buffer.data(), &size) != 0) {
        return {};
    }
    std::error_code error;
    const auto executable = std::filesystem::weakly_canonical(
        std::filesystem::path(buffer.data()), error);
    return error ? std::filesystem::path{} : executable.parent_path();
#else
    std::vector<char> buffer(4096);
    const auto length = ::readlink(
        "/proc/self/exe", buffer.data(), buffer.size());
    if (length <= 0 || static_cast<std::size_t>(length) >= buffer.size()) {
        return {};
    }
    return std::filesystem::path(
        std::string(buffer.data(), static_cast<std::size_t>(length)))
        .parent_path();
#endif
}

int Fail(const std::filesystem::path& runtime_root,
         const std::string& message) {
    cyxwiz::runtime::AppendBootstrapDiagnostic(
        runtime_root, "launch failed: " + message);
    std::cerr << "CyxWiz launch failed: " << message << '\n';
    std::cerr << "Diagnostic log: "
              << (runtime_root / "bootstrapper.log").string() << '\n';
    return 78;
}

bool SetEnvironmentValue(const char* name, const std::string& value) {
    return ::setenv(name, value.c_str(), 1) == 0;
}

bool ClearEnvironmentValue(const char* name) {
    return ::unsetenv(name) == 0;
}

std::string RuntimeLibraryPath(
    const cyxwiz::runtime::ActiveRuntime& runtime) {
    std::string value;
    for (const auto& directory : runtime.dll_directories) {
        if (!value.empty()) value.push_back(':');
        value += directory.string();
    }
    return value;
}

bool ConfigureRuntimeEnvironment(
    const cyxwiz::runtime::ActiveRuntime& runtime) {
    if (!SetEnvironmentValue(
            "CYXWIZ_ACTIVE_RUNTIME_ROOT", runtime.runtime_root.string()) ||
        !SetEnvironmentValue("CYXWIZ_RUNTIME_SET_ID", runtime.runtime_set_id) ||
        !SetEnvironmentValue(
            "CYXWIZ_RUNTIME_GENERATION",
            std::to_string(runtime.generation)) ||
        !SetEnvironmentValue("CYXWIZ_BASE_PACK_ID", runtime.base_pack_id) ||
        !SetEnvironmentValue("PATH", "/usr/bin:/bin:/usr/sbin:/sbin")) {
        return false;
    }

    for (const char* name : {
             "CYXWIZ_RUNTIME_PACK_CUDA",
             "CYXWIZ_RUNTIME_PACK_OPENCL",
             "CYXWIZ_RUNTIME_PACK_ONEAPI"}) {
        if (!ClearEnvironmentValue(name)) return false;
    }
    for (const auto& pack : runtime.packs) {
        const char* name = pack.backend == "cuda"
            ? "CYXWIZ_RUNTIME_PACK_CUDA"
            : pack.backend == "opencl"
                ? "CYXWIZ_RUNTIME_PACK_OPENCL"
                : "CYXWIZ_RUNTIME_PACK_ONEAPI";
        if (!SetEnvironmentValue(name, pack.pack_id)) return false;
    }

    for (const char* name : {
             "AF_PATH", "AF_PLUGIN_PATH", "CYXWIZ_ARRAYFIRE_DIR",
             "AF_BUILD_PATH", "AF_BUILD_LIB_CUSTOM_PATH",
             "PYTHONHOME", "PYTHONPATH", "LD_PRELOAD",
             "DYLD_INSERT_LIBRARIES", "DYLD_FRAMEWORK_PATH",
             "DYLD_FALLBACK_LIBRARY_PATH"}) {
        if (!ClearEnvironmentValue(name)) return false;
    }
#ifdef __APPLE__
    return SetEnvironmentValue(
        "DYLD_LIBRARY_PATH", RuntimeLibraryPath(runtime)) &&
        ClearEnvironmentValue("LD_LIBRARY_PATH");
#else
    return SetEnvironmentValue(
        "LD_LIBRARY_PATH", RuntimeLibraryPath(runtime)) &&
        ClearEnvironmentValue("DYLD_LIBRARY_PATH");
#endif
}

struct ChildResult {
    bool started = false;
    int exit_code = 78;
    std::string message;
};

ChildResult LaunchAndWait(
    const std::filesystem::path& executable,
    std::vector<std::string> arguments,
    const std::filesystem::path& working_directory,
    const cyxwiz::runtime::ActiveRuntime& runtime) {
    ChildResult result;
    int error_pipe[2] = {-1, -1};
    if (::pipe(error_pipe) != 0 ||
        ::fcntl(error_pipe[1], F_SETFD, FD_CLOEXEC) == -1) {
        if (error_pipe[0] != -1) ::close(error_pipe[0]);
        if (error_pipe[1] != -1) ::close(error_pipe[1]);
        result.message = "Cannot create the child launch boundary: " +
            std::string(std::strerror(errno));
        return result;
    }

    const pid_t child = ::fork();
    if (child < 0) {
        const int code = errno;
        ::close(error_pipe[0]);
        ::close(error_pipe[1]);
        result.message = "Cannot fork the CyxWiz child: " +
            std::string(std::strerror(code));
        return result;
    }
    if (child == 0) {
        ::close(error_pipe[0]);
        int child_error = 0;
        if (::chdir(working_directory.c_str()) != 0 ||
            !ConfigureRuntimeEnvironment(runtime)) {
            child_error = errno == 0 ? EINVAL : errno;
        } else {
            std::vector<char*> values;
            values.reserve(arguments.size() + 1);
            for (auto& argument : arguments) values.push_back(argument.data());
            values.push_back(nullptr);
            ::execv(executable.c_str(), values.data());
            child_error = errno;
        }
        const auto ignored = ::write(
            error_pipe[1], &child_error, sizeof(child_error));
        (void)ignored;
        ::_exit(78);
    }

    ::close(error_pipe[1]);
    int child_error = 0;
    const auto error_bytes = ::read(
        error_pipe[0], &child_error, sizeof(child_error));
    ::close(error_pipe[0]);
    result.started = error_bytes == 0;
    if (!result.started) {
        result.message = error_bytes < 0
            ? "Cannot read the child launch result: " +
                  std::string(std::strerror(errno))
            : "Cannot start the CyxWiz child: " +
                  std::string(std::strerror(child_error));
    }

    int status = 0;
    if (::waitpid(child, &status, 0) < 0) {
        result.message = "Waiting for the CyxWiz child failed: " +
            std::string(std::strerror(errno));
        result.started = false;
        return result;
    }
    if (WIFEXITED(status)) {
        result.exit_code = WEXITSTATUS(status);
    } else if (WIFSIGNALED(status)) {
        result.exit_code = 128 + WTERMSIG(status);
    }
    return result;
}

bool RunBackendPackRepair(
    const cyxwiz::runtime::ActiveRuntime& runtime,
    const cyxwiz::runtime::BackendPackMaintenanceRequest& request,
    std::string& message) {
    const auto helper = runtime.base_directory /
        cyxwiz::runtime::CurrentBackendPackInstallerExecutableName();
    if (!std::filesystem::is_regular_file(helper)) {
        message = "Backend-pack repair helper is missing";
        return false;
    }
    std::vector<std::string> arguments{
        helper.string(), "--runtime-root", runtime.runtime_root.string(),
        "--pack-id", request.pack_id, "--repair"};
    const auto child = LaunchAndWait(
        helper, std::move(arguments), runtime.base_directory, runtime);
    if (!child.started) {
        message = child.message;
        return false;
    }
    message = child.exit_code == 0
        ? "Backend pack repaired, locally qualified, and reactivated"
        : "Backend-pack repair helper failed with exit code " +
              std::to_string(child.exit_code);
    return child.exit_code == 0;
}

}  // namespace

int main(int argc, char** argv) {
    const auto executable_directory = ExecutableDirectory();
    if (executable_directory.empty()) {
        std::cerr << "CyxWiz launch failed: cannot resolve bootstrapper location\n";
        return 78;
    }

    std::filesystem::path runtime_root = executable_directory / "runtime";
    int first_forwarded_argument = 1;
    if (argc >= 3 && std::string_view(argv[1]) == "--runtime-root") {
        runtime_root = argv[2];
        first_forwarded_argument = 3;
    }
    bool installer_mode = false;
    if (argc > first_forwarded_argument &&
        std::string_view(argv[first_forwarded_argument]) == "--installer") {
        installer_mode = true;
        ++first_forwarded_argument;
    }

    cyxwiz::runtime::ActiveRuntime runtime;
    std::string error;
    if (!cyxwiz::runtime::ResolveActiveRuntime(runtime_root, runtime, error)) {
        return Fail(runtime_root, error);
    }
    const auto launched_executable = installer_mode
        ? runtime.base_directory /
              cyxwiz::runtime::CurrentInstallerManagerExecutableName()
        : runtime.engine_executable;
    if (!std::filesystem::is_regular_file(launched_executable)) {
        return Fail(
            runtime.runtime_root,
            installer_mode
                ? "active base does not contain the CyxWiz Installer"
                : "active base does not contain the CyxWiz Engine");
    }

    std::vector<std::string> child_arguments{launched_executable.string()};
    if (installer_mode) {
        child_arguments.push_back("--runtime-root");
        child_arguments.push_back(runtime.runtime_root.string());
    }
    for (int index = first_forwarded_argument; index < argc; ++index) {
        child_arguments.emplace_back(argv[index]);
    }
    const auto child = LaunchAndWait(
        launched_executable, std::move(child_arguments),
        runtime.base_directory, runtime);
    if (!child.started) return Fail(runtime.runtime_root, child.message);

    cyxwiz::runtime::AppendBootstrapDiagnostic(
        runtime.runtime_root,
        std::string(installer_mode ? "launched installer runtime_set="
                                   : "launched runtime_set=") +
            runtime.runtime_set_id +
            " generation=" + std::to_string(runtime.generation) +
            " base=" + runtime.base_pack_id);
    if (!installer_mode) {
        cyxwiz::runtime::ActiveRuntimeState launched_runtime;
        launched_runtime.runtime_set_id = runtime.runtime_set_id;
        launched_runtime.generation = runtime.generation;
        launched_runtime.base_pack_id = runtime.base_pack_id;
        for (const auto& pack : runtime.packs) {
            launched_runtime.packs.push_back({pack.backend, pack.pack_id});
        }
        const auto maintenance =
            cyxwiz::runtime::ApplyPendingBackendPackMaintenance(
                runtime.runtime_root, launched_runtime,
                [&](const auto& request, std::string& message) {
                    return RunBackendPackRepair(runtime, request, message);
                });
        if (maintenance.status != cyxwiz::runtime::
                BackendPackMaintenanceApplyStatus::NoRequest) {
            cyxwiz::runtime::AppendBootstrapDiagnostic(
                runtime.runtime_root,
                std::string("backend maintenance ") +
                    cyxwiz::runtime::BackendPackMaintenanceApplyStatusName(
                        maintenance.status) +
                    ": " + maintenance.message);
        }
    }
    return child.exit_code;
}
