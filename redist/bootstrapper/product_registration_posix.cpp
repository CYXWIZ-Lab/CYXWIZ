#include "product_registration_internal.h"

#include "backend_pack_platform.h"

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

}  // namespace cyxwiz::runtime::detail
