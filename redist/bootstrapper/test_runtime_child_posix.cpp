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
    const auto expected_af_path =
        std::filesystem::path(runtime_root) / "base" / "base-v1" / "arrayfire";
    if (library_path.find("cyxwiz-untrusted-marker") != std::string::npos ||
        library_path.find("base-v1") == std::string::npos ||
        std::filesystem::path(EnvironmentValue("AF_PATH")) != expected_af_path ||
        !EnvironmentValue("PYTHONPATH").empty() ||
        !EnvironmentValue("LD_PRELOAD").empty() ||
        !EnvironmentValue("DYLD_INSERT_LIBRARIES").empty()) {
        return 11;
    }
    if (argc == 1) return 0;
    if ((argc == 3 || argc == 4) &&
        std::string(argv[1]) == "--runtime-root" &&
        std::filesystem::path(argv[2]) == std::filesystem::path(runtime_root) &&
        (argc == 3 || std::string(argv[3]) == "--product-removal-host")) {
        return 0;
    }
    return 12;
}
