#include "product_registration_internal.h"

#include "backend_pack_platform.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string_view>
#include <system_error>

#include <sys/stat.h>

namespace cyxwiz::runtime::detail {
namespace {

#if !defined(__APPLE__)
std::string PathUtf8(const std::filesystem::path& path) {
    const auto value = path.u8string();
    return {reinterpret_cast<const char*>(value.data()), value.size()};
}
#endif

bool WriteTextAtomic(
    const std::filesystem::path& destination,
    const std::string& content,
    std::string& error) {
    std::error_code filesystem_error;
    std::filesystem::create_directories(
        destination.parent_path(), filesystem_error);
    if (filesystem_error) {
        error = "Cannot create product integration directory: " +
            filesystem_error.message();
        return false;
    }
    auto temporary = destination;
    temporary += ".part-" + std::to_string(
        std::chrono::steady_clock::now().time_since_epoch().count());
    std::ofstream stream(temporary, std::ios::binary | std::ios::trunc);
    if (!stream) {
        error = "Cannot create temporary product integration file";
        return false;
    }
    stream.write(content.data(), static_cast<std::streamsize>(content.size()));
    stream.flush();
    if (!stream) {
        stream.close();
        std::filesystem::remove(temporary, filesystem_error);
        error = "Cannot write complete product integration file";
        return false;
    }
    stream.close();
    if (::chmod(temporary.c_str(), 0644) != 0) {
        std::filesystem::remove(temporary, filesystem_error);
        error = "Cannot set permissions on product integration file";
        return false;
    }
    std::filesystem::rename(temporary, destination, filesystem_error);
    if (filesystem_error) {
        std::error_code ignored;
        std::filesystem::remove(temporary, ignored);
        error = "Cannot publish product integration file: " +
            filesystem_error.message();
        return false;
    }
    return true;
}

bool ReadTextFile(
    const std::filesystem::path& path,
    std::string& content,
    std::string& error) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) {
        error = "Cannot read CyxWiz product integration file";
        return false;
    }
    content.assign(
        std::istreambuf_iterator<char>(stream),
        std::istreambuf_iterator<char>());
    if (stream.bad()) {
        error = "Cannot read complete CyxWiz product integration file";
        return false;
    }
    return true;
}

#if !defined(__APPLE__)
bool ValidateOwnedTextFile(
    const std::filesystem::path& path,
    const std::string& expected,
    bool& present,
    std::string& error) {
    present = false;
    std::error_code filesystem_error;
    const auto status = std::filesystem::symlink_status(
        path, filesystem_error);
    if (status.type() == std::filesystem::file_type::not_found) return true;
    if (filesystem_error ||
        status.type() != std::filesystem::file_type::regular) {
        error = "CyxWiz product integration is not a managed regular file";
        return false;
    }
    std::string content;
    if (!ReadTextFile(path, content, error)) return false;
    if (content != expected) {
        error = "CyxWiz product integration contains unmanaged changes";
        return false;
    }
    present = true;
    return true;
}

bool RemoveOwnedTextFile(
    const std::filesystem::path& path,
    bool present,
    std::string& error) {
    if (!present) return true;
    std::error_code filesystem_error;
    if (!std::filesystem::remove(path, filesystem_error) || filesystem_error) {
        error = "Cannot remove CyxWiz product integration file: " +
            filesystem_error.message();
        return false;
    }
    return true;
}
#endif

#if !defined(__APPLE__)
std::string QuoteDesktopExec(const std::filesystem::path& path) {
    std::string quoted = "\"";
    for (const char character : PathUtf8(path)) {
        if (character == '%') {
            quoted += "%%";
            continue;
        }
        if (character == '\\' || character == '\"' ||
            character == '`' || character == '$') {
            quoted.push_back('\\');
        }
        quoted.push_back(character);
    }
    quoted.push_back('\"');
    return quoted;
}
#endif

#if defined(__APPLE__)

std::string InfoPlist(
    std::string_view bundle_name,
    std::string_view executable,
    std::string_view bundle_id,
    std::string_view version) {
    std::ostringstream output;
    output << "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
           << "<!DOCTYPE plist PUBLIC \"-//Apple//DTD PLIST 1.0//EN\" "
              "\"http://www.apple.com/DTDs/PropertyList-1.0.dtd\">\n"
           << "<plist version=\"1.0\">\n<dict>\n"
           << "  <key>CFBundleDevelopmentRegion</key><string>en</string>\n"
           << "  <key>CFBundleDisplayName</key><string>" << bundle_name
           << "</string>\n"
           << "  <key>CFBundleExecutable</key><string>" << executable
           << "</string>\n"
           << "  <key>CFBundleIdentifier</key><string>" << bundle_id
           << "</string>\n"
           << "  <key>CFBundleInfoDictionaryVersion</key><string>6.0</string>\n"
           << "  <key>CFBundleName</key><string>" << bundle_name
           << "</string>\n"
           << "  <key>CFBundlePackageType</key><string>APPL</string>\n"
           << "  <key>CFBundleShortVersionString</key><string>" << version
           << "</string>\n"
           << "  <key>CFBundleVersion</key><string>" << version
           << "</string>\n"
           << "</dict>\n</plist>\n";
    return output.str();
}

bool PublishBundle(
    const std::filesystem::path& install_root,
    std::string_view bundle_name,
    std::string_view executable_name,
    std::string_view bundle_id,
    std::string_view version,
    std::string& error) {
    const auto bundle = install_root / (std::string(bundle_name) + ".app");
    const auto macos = bundle / "Contents" / "MacOS";
    const auto executable = macos / std::string(executable_name);
    std::error_code filesystem_error;
    std::filesystem::create_directories(macos, filesystem_error);
    if (filesystem_error) {
        error = "Cannot create the macOS CyxWiz application bundle: " +
            filesystem_error.message();
        return false;
    }
    const std::filesystem::path target =
        "../../../" + std::string(CurrentRuntimeBootstrapperExecutableName());
    const auto status = std::filesystem::symlink_status(
        executable, filesystem_error);
    if (!filesystem_error && status.type() ==
            std::filesystem::file_type::symlink) {
        if (std::filesystem::read_symlink(executable, filesystem_error) !=
                target ||
            filesystem_error) {
            std::filesystem::remove(executable, filesystem_error);
        }
    } else if (!filesystem_error && status.type() !=
                   std::filesystem::file_type::not_found) {
        error = "The macOS CyxWiz bundle executable is not a managed symlink";
        return false;
    }
    filesystem_error.clear();
    if (!std::filesystem::is_symlink(executable, filesystem_error)) {
        filesystem_error.clear();
        std::filesystem::create_symlink(target, executable, filesystem_error);
    }
    if (filesystem_error) {
        error = "Cannot publish the macOS CyxWiz bundle launcher: " +
            filesystem_error.message();
        return false;
    }
    return WriteTextAtomic(
        bundle / "Contents" / "Info.plist",
        InfoPlist(
            bundle_name, executable_name, bundle_id, version), error);
}

bool RemoveManagedBundle(
    const std::filesystem::path& install_root,
    std::string_view bundle_name,
    std::string_view executable_name,
    std::string_view bundle_id,
    std::string_view version,
    bool remove,
    std::string& error) {
    const auto bundle = install_root / (std::string(bundle_name) + ".app");
    const auto contents = bundle / "Contents";
    const auto plist = contents / "Info.plist";
    const auto macos = contents / "MacOS";
    const auto executable = macos / std::string(executable_name);
    const std::array expected_paths{contents, plist, macos, executable};

    std::error_code filesystem_error;
    const auto bundle_status = std::filesystem::symlink_status(
        bundle, filesystem_error);
    if (bundle_status.type() == std::filesystem::file_type::not_found) {
        return true;
    }
    if (filesystem_error ||
        bundle_status.type() != std::filesystem::file_type::directory) {
        error = "The macOS CyxWiz application bundle is not managed";
        return false;
    }
    std::size_t discovered = 0;
    for (std::filesystem::recursive_directory_iterator iterator(
             bundle, filesystem_error), end;
         iterator != end && !filesystem_error;
         iterator.increment(filesystem_error)) {
        const bool expected = std::find(
            expected_paths.begin(), expected_paths.end(), iterator->path()) !=
            expected_paths.end();
        if (!expected) {
            error = "The macOS CyxWiz application bundle contains unmanaged files";
            return false;
        }
        ++discovered;
    }
    if (filesystem_error || discovered != expected_paths.size()) {
        error = "The macOS CyxWiz application bundle is incomplete";
        return false;
    }
    std::string plist_content;
    if (!ReadTextFile(plist, plist_content, error) ||
        plist_content != InfoPlist(
            bundle_name, executable_name, bundle_id, version)) {
        if (error.empty()) {
            error = "The macOS CyxWiz application bundle contains unmanaged metadata";
        }
        return false;
    }
    const std::filesystem::path target =
        "../../../" + std::string(CurrentRuntimeBootstrapperExecutableName());
    if (!std::filesystem::is_symlink(executable, filesystem_error) ||
        filesystem_error ||
        std::filesystem::read_symlink(executable, filesystem_error) != target ||
        filesystem_error) {
        error = "The macOS CyxWiz bundle launcher is not managed";
        return false;
    }
    if (!remove) return true;
    for (const auto& path :
         std::array{executable, macos, plist, contents, bundle}) {
        filesystem_error.clear();
        if (!std::filesystem::remove(path, filesystem_error) ||
            filesystem_error) {
            error = "Cannot remove the macOS CyxWiz application bundle: " +
                filesystem_error.message();
            return false;
        }
    }
    return true;
}

#else

std::filesystem::path LinuxApplicationsDirectory(
    ProductInstallScope scope) {
    if (scope == ProductInstallScope::AllUsers) {
        return "/usr/share/applications";
    }
    const char* data_home = std::getenv("XDG_DATA_HOME");
    if (data_home && *data_home) {
        const std::filesystem::path candidate(data_home);
        if (candidate.is_absolute()) return candidate / "applications";
    }
    const char* home = std::getenv("HOME");
    return home && *home
        ? std::filesystem::path(home) / ".local" / "share" / "applications"
        : std::filesystem::path{};
}

std::string DesktopEntry(
    const std::filesystem::path& launcher,
    bool installer) {
    std::ostringstream output;
    output << "[Desktop Entry]\n"
           << "Type=Application\n"
           << "Version=1.0\n"
           << "Name=" << (installer ? "CyxWiz Installer" : "CyxWiz") << "\n"
           << "Comment="
           << (installer ? "Modify, repair, or remove CyxWiz"
                         : "Launch CyxWiz Engine") << "\n"
           << "Exec=" << QuoteDesktopExec(launcher)
           << (installer ? " --installer" : "") << "\n"
           << "Terminal=false\n"
           << "StartupNotify=true\n"
           << "X-CyxWiz-Managed=true\n"
           << "Categories=Development;Science;\n";
    return output.str();
}

#endif

}  // namespace

ProductRegistrationResult RegisterPlatformProduct(
    const ProductRegistrationRequest& request) {
    ProductRegistrationResult result;
    std::string error;
#if defined(__APPLE__)
    if (!PublishBundle(
            request.install_root, "CyxWiz", "CyxWiz",
            "com.cyxwiz.engine", request.product_version, error) ||
        !PublishBundle(
            request.install_root, "CyxWiz Installer", "CyxWiz Installer",
            "com.cyxwiz.installer", request.product_version, error)) {
        result.message = std::move(error);
        return result;
    }
    result.registered = true;
    result.message =
        "CyxWiz Engine and Installer application bundles were registered";
#else
    const auto applications = LinuxApplicationsDirectory(request.scope);
    if (!applications.is_absolute()) {
        result.message =
            "Cannot resolve the Linux desktop applications directory";
        return result;
    }
    const auto launcher = request.install_root /
        std::string(CurrentRuntimeBootstrapperExecutableName());
    if (!WriteTextAtomic(
            applications / "cyxwiz.desktop",
            DesktopEntry(launcher, false), error) ||
        !WriteTextAtomic(
            applications / "cyxwiz-installer.desktop",
            DesktopEntry(launcher, true), error)) {
        result.message = std::move(error);
        return result;
    }
    result.registered = true;
    result.message =
        "CyxWiz Engine and Installer desktop entries were registered";
#endif
    return result;
}

ProductUnregistrationResult UnregisterPlatformProduct(
    const ProductRegistrationRequest& request) {
    ProductUnregistrationResult result;
    std::string error;
#if defined(__APPLE__)
    const auto unregister_bundle = [&](
        std::string_view bundle_name,
        std::string_view executable_name,
        std::string_view bundle_id,
        bool remove) {
        return RemoveManagedBundle(
            request.install_root, bundle_name, executable_name, bundle_id,
            request.product_version, remove, error);
    };
    if (!unregister_bundle(
            "CyxWiz", "CyxWiz", "com.cyxwiz.engine", false) ||
        !unregister_bundle(
            "CyxWiz Installer", "CyxWiz Installer",
            "com.cyxwiz.installer", false) ||
        !unregister_bundle(
            "CyxWiz", "CyxWiz", "com.cyxwiz.engine", true) ||
        !unregister_bundle(
            "CyxWiz Installer", "CyxWiz Installer",
            "com.cyxwiz.installer", true)) {
        result.message = std::move(error);
        return result;
    }
    result.unregistered = true;
    result.message =
        "CyxWiz Engine and Installer application bundles were removed";
#else
    const auto applications = LinuxApplicationsDirectory(request.scope);
    if (!applications.is_absolute()) {
        result.message =
            "Cannot resolve the Linux desktop applications directory";
        return result;
    }
    const auto launcher = request.install_root /
        std::string(CurrentRuntimeBootstrapperExecutableName());
    const auto engine_entry = applications / "cyxwiz.desktop";
    const auto installer_entry = applications / "cyxwiz-installer.desktop";
    bool engine_present = false;
    bool installer_present = false;
    if (!ValidateOwnedTextFile(
            engine_entry, DesktopEntry(launcher, false),
            engine_present, error) ||
        !ValidateOwnedTextFile(
            installer_entry, DesktopEntry(launcher, true),
            installer_present, error)) {
        result.message = std::move(error);
        return result;
    }
    if (!RemoveOwnedTextFile(engine_entry, engine_present, error) ||
        !RemoveOwnedTextFile(installer_entry, installer_present, error)) {
        result.message = std::move(error);
        return result;
    }
    std::error_code ignored;
    std::filesystem::remove(applications, ignored);
    result.unregistered = true;
    result.message =
        "CyxWiz Engine and Installer desktop entries were removed";
#endif
    return result;
}

}  // namespace cyxwiz::runtime::detail
