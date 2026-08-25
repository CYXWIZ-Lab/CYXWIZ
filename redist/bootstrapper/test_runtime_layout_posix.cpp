#include "runtime_layout.h"
#include "backend_pack_platform.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#ifdef __APPLE__
#include <mach-o/dyld.h>
#endif

namespace {

class Fixture {
public:
    Fixture() {
        root = std::filesystem::temp_directory_path() /
            ("cyxwiz-runtime-posix-test-" + std::to_string(::getpid()) +
             "-" + std::to_string(++sequence));
        std::filesystem::create_directories(root / "base" / "base-v1");
        std::ofstream(root / "active-runtime.json", std::ios::binary)
            << "{\"schema_version\":1,\"runtime_set_id\":\"set-v1\","
               "\"generation\":1,\"base_pack_id\":\"base-v1\","
               "\"packs\":[]}";
    }

    ~Fixture() {
        std::error_code error;
        std::filesystem::remove_all(root, error);
    }

    std::filesystem::path root;
    static inline int sequence = 0;
};

bool Expect(bool condition, const std::string& message) {
    if (!condition) std::cerr << "FAIL: " << message << '\n';
    return condition;
}

std::filesystem::path ExecutableDirectory() {
#ifdef __APPLE__
    std::uint32_t size = 0;
    ::_NSGetExecutablePath(nullptr, &size);
    std::vector<char> buffer(size);
    if (size == 0 || ::_NSGetExecutablePath(buffer.data(), &size) != 0) {
        return {};
    }
    return std::filesystem::weakly_canonical(buffer.data()).parent_path();
#else
    std::vector<char> buffer(4096);
    const auto length = ::readlink(
        "/proc/self/exe", buffer.data(), buffer.size());
    return length <= 0
        ? std::filesystem::path{}
        : std::filesystem::path(
              std::string(buffer.data(), static_cast<std::size_t>(length)))
              .parent_path();
#endif
}

int Run(const std::filesystem::path& executable,
        const std::filesystem::path& runtime_root,
        bool installer = false,
        const char* invocation_name = nullptr) {
    const pid_t child = ::fork();
    if (child < 0) return -1;
    if (child == 0) {
        const auto root = runtime_root.string();
        const char* argument_zero = invocation_name
            ? invocation_name : executable.c_str();
        if (installer) {
            ::execl(executable.c_str(), argument_zero, "--runtime-root",
                    root.c_str(), "--installer", nullptr);
        } else {
            ::execl(executable.c_str(), argument_zero, "--runtime-root",
                    root.c_str(), nullptr);
        }
        ::_exit(127);
    }
    int status = 0;
    if (::waitpid(child, &status, 0) < 0 || !WIFEXITED(status)) return -1;
    return WEXITSTATUS(status);
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
    const auto binary_directory = ExecutableDirectory();
    const auto bootstrapper = binary_directory /
        cyxwiz::runtime::CurrentRuntimeBootstrapperExecutableName();
    const auto child = binary_directory / "test_runtime_bootstrapper_child";
    {
        Fixture fixture;
        const auto base = fixture.root / "base" / "base-v1";
        std::filesystem::copy_file(
            child, base / cyxwiz::runtime::CurrentEngineExecutableName());
        std::filesystem::copy_file(
            child,
            base / cyxwiz::runtime::CurrentInstallerManagerExecutableName());
        std::filesystem::permissions(
            base / cyxwiz::runtime::CurrentEngineExecutableName(),
            std::filesystem::perms::owner_exec |
                std::filesystem::perms::group_exec |
                std::filesystem::perms::others_exec,
            std::filesystem::perm_options::add);
        std::filesystem::permissions(
            base / cyxwiz::runtime::CurrentInstallerManagerExecutableName(),
            std::filesystem::perms::owner_exec |
                std::filesystem::perms::group_exec |
                std::filesystem::perms::others_exec,
            std::filesystem::perm_options::add);
        ::setenv("AF_PATH", "cyxwiz-untrusted-marker", 1);
        ::setenv("PYTHONPATH", "cyxwiz-untrusted-marker", 1);
#ifdef __APPLE__
        // An invalid DYLD_INSERT_LIBRARIES value terminates the bootstrapper
        // before main(). Contaminate the non-host injection variable instead;
        // the child still requires both injection variables to be absent.
        ::setenv("LD_PRELOAD", "cyxwiz-untrusted-marker", 1);
#else
        ::setenv("DYLD_INSERT_LIBRARIES", "cyxwiz-untrusted-marker", 1);
#endif
        ::setenv("LD_LIBRARY_PATH", "cyxwiz-untrusted-marker", 1);
        ::setenv("DYLD_LIBRARY_PATH", "cyxwiz-untrusted-marker", 1);

        failures += !Expect(
            Run(bootstrapper, fixture.root) == 0,
            "stable launcher should start the active Engine with a clean environment");
        failures += !Expect(
            Run(bootstrapper, fixture.root, true) == 0,
            "stable launcher should start the installed manager");
#ifdef __APPLE__
        failures += !Expect(
            Run(bootstrapper, fixture.root, false, "CyxWiz Installer") == 0,
            "macOS Installer bundle name should select installed-manager mode");
#endif
        const auto log = ReadText(fixture.root / "bootstrapper.log");
        failures += !Expect(
            log.find("launched runtime_set=set-v1") != std::string::npos &&
                log.find("launched installer runtime_set=set-v1") !=
                    std::string::npos,
            "successful Engine and installer launches should be diagnosed");
    }
    {
        Fixture fixture;
        std::filesystem::remove(fixture.root / "active-runtime.json");
        failures += !Expect(
            Run(bootstrapper, fixture.root) == 78,
            "missing active state should fail before child launch");
        failures += !Expect(
            ReadText(fixture.root / "bootstrapper.log").find(
                "active-runtime.json is missing") != std::string::npos,
            "invalid state should be recorded in the package-local log");
    }
    if (failures == 0) {
        std::cout << "POSIX runtime bootstrapper contract tests passed\n";
    }
    return failures == 0 ? 0 : 1;
}
