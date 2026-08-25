#include "product_registration.h"

#include "backend_pack_platform.h"

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

#ifndef _WIN32
#include <sys/stat.h>
#endif

namespace {

void Check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        std::exit(1);
    }
}

class TemporaryDirectory {
public:
    TemporaryDirectory() {
        path_ = std::filesystem::temp_directory_path() /
            ("cyxwiz-product-registration-test-" + std::to_string(
                std::chrono::steady_clock::now().time_since_epoch().count()));
        std::filesystem::create_directories(path_);
    }
    ~TemporaryDirectory() {
        std::error_code ignored;
        std::filesystem::remove_all(path_, ignored);
    }
    const std::filesystem::path& path() const { return path_; }

private:
    std::filesystem::path path_;
};

std::string ReadText(const std::filesystem::path& path) {
    std::ifstream stream(path, std::ios::binary);
    return {std::istreambuf_iterator<char>(stream),
            std::istreambuf_iterator<char>()};
}

cyxwiz::runtime::ProductRegistrationRequest Request(
    const std::filesystem::path& install_root) {
    cyxwiz::runtime::ProductRegistrationRequest request;
    request.install_root = install_root;
    request.runtime_root = install_root / "runtime";
    request.product_version = "0.2.0";
    return request;
}

void TestValidation() {
    const auto relative =
        cyxwiz::runtime::RegisterInstalledProduct(Request("relative"));
    Check(!relative.registered && !relative.message.empty(),
          "Product registration must reject a relative installation root");

    TemporaryDirectory temporary;
    const auto missing =
        cyxwiz::runtime::RegisterInstalledProduct(Request(temporary.path()));
    Check(!missing.registered &&
              missing.message.find("stable CyxWiz launcher") !=
                  std::string::npos,
          "Product registration must require the verified stable launcher");
}

#ifndef _WIN32
void CreateLauncher(const std::filesystem::path& install_root) {
    std::filesystem::create_directories(install_root / "runtime");
    const auto launcher = install_root /
        std::string(
            cyxwiz::runtime::CurrentRuntimeBootstrapperExecutableName());
    std::ofstream(launcher, std::ios::binary) << "launcher";
    ::chmod(launcher.c_str(), 0755);
}

void TestNativeRegistration() {
    TemporaryDirectory temporary;
    const auto install_root = temporary.path() / "CyxWiz Product";
    CreateLauncher(install_root);
    const std::string path_before = std::getenv("PATH")
        ? std::getenv("PATH") : "";

#if defined(__APPLE__)
    const auto result =
        cyxwiz::runtime::RegisterInstalledProduct(Request(install_root));
    Check(result.registered,
          "macOS product application bundles must register successfully");
    const auto engine_bundle = install_root / "CyxWiz.app" / "Contents";
    const auto installer_bundle =
        install_root / "CyxWiz Installer.app" / "Contents";
    Check(ReadText(engine_bundle / "Info.plist").find(
              "com.cyxwiz.engine") != std::string::npos &&
              ReadText(installer_bundle / "Info.plist").find(
                  "com.cyxwiz.installer") != std::string::npos,
          "macOS bundles must retain distinct Engine and Installer identities");
    Check(std::filesystem::read_symlink(
              engine_bundle / "MacOS" / "CyxWiz") ==
              "../../../cyxwiz-runtime-bootstrapper" &&
              std::filesystem::read_symlink(
                  installer_bundle / "MacOS" / "CyxWiz Installer") ==
              "../../../cyxwiz-runtime-bootstrapper",
          "macOS bundles must point only at the stable bootstrapper");
#else
    const auto data_home = temporary.path() / "xdg-data";
    Check(::setenv("XDG_DATA_HOME", data_home.c_str(), 1) == 0,
          "Linux registration test must isolate XDG_DATA_HOME");
    const auto result =
        cyxwiz::runtime::RegisterInstalledProduct(Request(install_root));
    Check(result.registered,
          "Linux desktop entries must register successfully");
    const auto applications = data_home / "applications";
    const auto engine = ReadText(applications / "cyxwiz.desktop");
    const auto installer =
        ReadText(applications / "cyxwiz-installer.desktop");
    Check(engine.find("Name=CyxWiz\n") != std::string::npos &&
              engine.find(" --installer") == std::string::npos &&
              installer.find("Name=CyxWiz Installer\n") !=
                  std::string::npos &&
              installer.find(" --installer\n") != std::string::npos,
          "Linux must register separate Engine and maintenance launch entries");
#endif

    const std::string path_after = std::getenv("PATH")
        ? std::getenv("PATH") : "";
    Check(path_after == path_before,
          "Product registration must not mutate the loader or executable PATH");
}
#endif

}  // namespace

int main() {
    TestValidation();
#ifndef _WIN32
    TestNativeRegistration();
#endif
    std::cout << "Product registration contracts passed\n";
    return 0;
}
