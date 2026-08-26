#include <cstdlib>
#include <filesystem>
#include <string>

namespace {

std::string EnvironmentValue(const char* name) {
    const char* value = std::getenv(name);
    return value ? value : std::string{};
}

}  // namespace

int main(int argc, char** argv) {
    const auto runtime_root = EnvironmentValue("CYXWIZ_ACTIVE_RUNTIME_ROOT");
    if (runtime_root.empty() ||
        EnvironmentValue("CYXWIZ_RUNTIME_SET_ID") != "set-v1" ||
        EnvironmentValue("CYXWIZ_RUNTIME_GENERATION") != "1" ||
        EnvironmentValue("CYXWIZ_BASE_PACK_ID") != "base-v1") {
        return 10;
    }
#ifdef __APPLE__
    const auto library_path = EnvironmentValue("DYLD_LIBRARY_PATH");
#else
    const auto library_path = EnvironmentValue("LD_LIBRARY_PATH");
#endif
    if (library_path.find("cyxwiz-untrusted-marker") != std::string::npos ||
        library_path.find("base-v1") == std::string::npos ||
        !EnvironmentValue("AF_PATH").empty() ||
        !EnvironmentValue("PYTHONPATH").empty() ||
        !EnvironmentValue("LD_PRELOAD").empty() ||
        !EnvironmentValue("DYLD_INSERT_LIBRARIES").empty()) {
        return 11;
    }
    if (argc == 1) return 0;
    if (argc == 3 && std::string(argv[1]) == "--runtime-root" &&
        std::filesystem::path(argv[2]) == std::filesystem::path(runtime_root)) {
        return 0;
    }
    return 12;
}
