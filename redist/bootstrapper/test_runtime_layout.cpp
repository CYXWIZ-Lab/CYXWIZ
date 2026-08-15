#include "runtime_layout.h"
#include "backend_pack_platform.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>

#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>

namespace {

class Fixture {
public:
    Fixture() {
        root = std::filesystem::temp_directory_path() /
               ("cyxwiz-runtime-test-" + std::to_string(::GetCurrentProcessId()) + "-" +
                std::to_string(++sequence));
        std::filesystem::create_directories(root / "base" / "base-v1");
        Touch(
            root / "base" / "base-v1" /
            cyxwiz::runtime::CurrentEngineExecutableName());
    }

    ~Fixture() {
        std::error_code error;
        std::filesystem::remove_all(root, error);
    }

    void WriteState(const std::string& packs, const std::string& extra = "") const {
        std::ofstream stream(root / "active-runtime.json", std::ios::binary);
        stream << "{\"schema_version\":1,\"runtime_set_id\":\"set-v1\",";
        stream << "\"generation\":1,\"base_pack_id\":\"base-v1\",";
        stream << "\"packs\":" << packs << extra << "}";
    }

    void AddOpenClPack(bool plugin = true) const {
        const auto directory = root / "packs" / "opencl" / "opencl-v1" / "runtime";
        std::filesystem::create_directories(directory);
        if (plugin) {
            Touch(directory / "afopencl.dll");
        }
    }

    static void Touch(const std::filesystem::path& path) {
        std::ofstream(path, std::ios::binary).put('\0');
    }

    std::filesystem::path root;
    static inline int sequence = 0;
};

bool Expect(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
    }
    return condition;
}

int RunBootstrapper(
    const std::filesystem::path& bootstrapper,
    const std::filesystem::path& runtime_root) {
    std::wstring command = L"\"" + bootstrapper.native() +
                           L"\" --runtime-root \"" + runtime_root.native() + L"\"";
    std::vector<wchar_t> mutable_command(command.begin(), command.end());
    mutable_command.push_back(L'\0');
    STARTUPINFOW startup{};
    startup.cb = sizeof(startup);
    PROCESS_INFORMATION process{};
    if (!::CreateProcessW(
            bootstrapper.c_str(), mutable_command.data(), nullptr, nullptr,
            FALSE, CREATE_NO_WINDOW, nullptr, nullptr, &startup, &process)) {
        return -1;
    }
    ::CloseHandle(process.hThread);
    ::WaitForSingleObject(process.hProcess, 30000);
    DWORD exit_code = 999;
    ::GetExitCodeProcess(process.hProcess, &exit_code);
    ::CloseHandle(process.hProcess);
    return static_cast<int>(exit_code);
}

std::string ReadText(const std::filesystem::path& path) {
    std::ifstream stream(path, std::ios::binary);
    return std::string(
        std::istreambuf_iterator<char>(stream),
        std::istreambuf_iterator<char>());
}

}  // namespace

int main() {
    int failures = 0;
    {
        Fixture fixture;
        fixture.WriteState("[]");
        cyxwiz::runtime::ActiveRuntime runtime;
        std::string error;
        failures += !Expect(
            cyxwiz::runtime::ResolveActiveRuntime(fixture.root, runtime, error),
            "valid CPU runtime must resolve: " + error);
        failures += !Expect(runtime.packs.empty(), "CPU runtime must not invent optional packs");
        failures += !Expect(runtime.dll_directories.size() == 1,
                            "CPU runtime must resolve exactly its base directory");
    }
    {
        Fixture fixture;
        fixture.AddOpenClPack();
        fixture.WriteState("[{\"backend\":\"opencl\",\"pack_id\":\"opencl-v1\"}]");
        cyxwiz::runtime::ActiveRuntime runtime;
        std::string error;
        failures += !Expect(
            cyxwiz::runtime::ResolveActiveRuntime(fixture.root, runtime, error),
            "valid OpenCL composition must resolve: " + error);
        failures += !Expect(runtime.packs.size() == 1 && runtime.dll_directories.size() == 2,
                            "OpenCL composition must add one selected runtime directory");
    }
    {
        Fixture fixture;
        fixture.WriteState("[]", ",\"unknown\":true");
        cyxwiz::runtime::ActiveRuntime runtime;
        std::string error;
        failures += !Expect(
            !cyxwiz::runtime::ResolveActiveRuntime(fixture.root, runtime, error) &&
                error.find("unknown or missing") != std::string::npos,
            "unknown state fields must fail closed");
    }
    {
        Fixture fixture;
        fixture.AddOpenClPack();
        fixture.WriteState(
            "[{\"backend\":\"opencl\",\"pack_id\":\"opencl-v1\"},"
            "{\"backend\":\"opencl\",\"pack_id\":\"opencl-v1\"}]");
        cyxwiz::runtime::ActiveRuntime runtime;
        std::string error;
        failures += !Expect(
            !cyxwiz::runtime::ResolveActiveRuntime(fixture.root, runtime, error) &&
                error.find("duplicate active backend") != std::string::npos,
            "duplicate backend selection must fail closed");
    }
    {
        Fixture fixture;
        fixture.AddOpenClPack(false);
        fixture.WriteState("[{\"backend\":\"opencl\",\"pack_id\":\"opencl-v1\"}]");
        cyxwiz::runtime::ActiveRuntime runtime;
        std::string error;
        failures += !Expect(
            !cyxwiz::runtime::ResolveActiveRuntime(fixture.root, runtime, error) &&
                error.find("afopencl.dll") != std::string::npos,
            "missing selected plugin must report its exact closure failure");
    }
    {
        Fixture fixture;
        fixture.WriteState("[]");
        std::ofstream stream(fixture.root / "active-runtime.json", std::ios::binary);
        stream << "{\"schema_version\":1,\"runtime_set_id\":\"../escape\",";
        stream << "\"generation\":1,\"base_pack_id\":\"base-v1\",\"packs\":[]}";
        stream.close();
        cyxwiz::runtime::ActiveRuntime runtime;
        std::string error;
        failures += !Expect(
            !cyxwiz::runtime::ResolveActiveRuntime(fixture.root, runtime, error) &&
                error.find("safe identifier") != std::string::npos,
            "path-like identities must fail before path resolution");
    }
    {
        Fixture fixture;
        fixture.AddOpenClPack();
        fixture.WriteState(
            "[{\"backend\":\"opencl\",\"pack_id\":\"opencl-v1\"}]");
        std::vector<wchar_t> executable(32768);
        const DWORD executable_length = ::GetModuleFileNameW(
            nullptr, executable.data(), static_cast<DWORD>(executable.size()));
        const auto binary_directory =
            std::filesystem::path(std::wstring(executable.data(), executable_length)).parent_path();
        const auto bootstrapper = binary_directory / "cyxwiz-runtime-bootstrapper.exe";
        const auto child = binary_directory / "test_runtime_bootstrapper_child.exe";
        std::filesystem::copy_file(
            child, fixture.root / "base" / "base-v1" / "cyxwiz-engine.exe",
            std::filesystem::copy_options::overwrite_existing);
        ::SetEnvironmentVariableW(L"AF_PATH", L"C:\\cyxwiz-untrusted-marker");
        ::SetEnvironmentVariableW(L"AF_PLUGIN_PATH", L"C:\\cyxwiz-untrusted-marker");
        ::SetEnvironmentVariableW(L"AF_BUILD_PATH", L"C:\\cyxwiz-untrusted-marker");
        ::SetEnvironmentVariableW(L"AF_BUILD_LIB_CUSTOM_PATH", L"C:\\cyxwiz-untrusted-marker");
        ::SetEnvironmentVariableW(L"PYTHONPATH", L"C:\\cyxwiz-untrusted-marker");
        const auto original_path_size = ::GetEnvironmentVariableW(L"PATH", nullptr, 0);
        std::vector<wchar_t> original_path(original_path_size);
        ::GetEnvironmentVariableW(L"PATH", original_path.data(), original_path_size);
        const std::wstring marked_path =
            std::wstring(original_path.data()) + L";C:\\cyxwiz-untrusted-marker";
        ::SetEnvironmentVariableW(L"PATH", marked_path.c_str());

        const int exit_code = RunBootstrapper(bootstrapper, fixture.root);

        ::SetEnvironmentVariableW(L"AF_PATH", nullptr);
        ::SetEnvironmentVariableW(L"AF_PLUGIN_PATH", nullptr);
        ::SetEnvironmentVariableW(L"AF_BUILD_PATH", nullptr);
        ::SetEnvironmentVariableW(L"AF_BUILD_LIB_CUSTOM_PATH", nullptr);
        ::SetEnvironmentVariableW(L"PYTHONPATH", nullptr);
        ::SetEnvironmentVariableW(L"PATH", original_path.data());
        failures += !Expect(
            exit_code == 0,
            "launcher must start a child with isolated runtime environment; exit=" +
                std::to_string(exit_code));
        const auto diagnostic = ReadText(fixture.root / "bootstrapper.log");
        failures += !Expect(
            diagnostic.find("launched runtime_set=set-v1") != std::string::npos &&
                diagnostic.find("engine DLL search configured") != std::string::npos,
            "successful launch must record package-local runtime diagnostics");
    }
    {
        Fixture fixture;
        std::filesystem::remove(fixture.root / "active-runtime.json");
        std::vector<wchar_t> executable(32768);
        const DWORD executable_length = ::GetModuleFileNameW(
            nullptr, executable.data(), static_cast<DWORD>(executable.size()));
        const auto binary_directory =
            std::filesystem::path(std::wstring(executable.data(), executable_length)).parent_path();
        const int exit_code = RunBootstrapper(
            binary_directory / "cyxwiz-runtime-bootstrapper.exe", fixture.root);
        failures += !Expect(exit_code == 78, "invalid active state must fail before launch");
        failures += !Expect(
            ReadText(fixture.root / "bootstrapper.log").find(
                "active-runtime.json is missing") != std::string::npos,
            "state validation failure must be recorded in the package-local log");
    }

    if (failures == 0) {
        std::cout << "runtime bootstrapper contract tests passed\n";
    }
    return failures == 0 ? 0 : 1;
}
