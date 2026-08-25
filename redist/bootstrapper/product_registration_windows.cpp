#include "product_registration_internal.h"

#include "backend_pack_platform.h"

#include <string>
#include <string_view>
#include <system_error>

#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <objbase.h>
#include <shlobj.h>
#include <shobjidl.h>

namespace cyxwiz::runtime::detail {
namespace {

constexpr std::string_view kProductName = "CyxWiz";
constexpr std::string_view kPublisher = "CyxWiz";

class ComScope {
public:
    ComScope() : result_(::CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED)) {}
    ~ComScope() {
        if (result_ == S_OK || result_ == S_FALSE) ::CoUninitialize();
    }
    bool available() const {
        return SUCCEEDED(result_) || result_ == RPC_E_CHANGED_MODE;
    }

private:
    HRESULT result_;
};

template <typename Interface>
class ComOwner {
public:
    ~ComOwner() {
        if (value_) value_->Release();
    }
    Interface** output() { return &value_; }
    Interface* operator->() const { return value_; }
    explicit operator bool() const { return value_ != nullptr; }

private:
    Interface* value_ = nullptr;
};

class RegistryKey {
public:
    ~RegistryKey() {
        if (value_) ::RegCloseKey(value_);
    }
    HKEY* output() { return &value_; }
    HKEY get() const { return value_; }

private:
    HKEY value_ = nullptr;
};

bool ResolveProgramsDirectory(
    ProductInstallScope scope,
    bool create,
    std::filesystem::path& programs,
    std::string& error) {
    PWSTR raw_programs = nullptr;
    const KNOWNFOLDERID& folder = scope == ProductInstallScope::AllUsers
        ? FOLDERID_CommonPrograms : FOLDERID_Programs;
    const HRESULT result = ::SHGetKnownFolderPath(
        folder, create ? KF_FLAG_CREATE : KF_FLAG_DEFAULT,
        nullptr, &raw_programs);
    if (FAILED(result) || !raw_programs) {
        error = "Cannot resolve the Windows Start Menu directory";
        return false;
    }
    programs = raw_programs;
    ::CoTaskMemFree(raw_programs);
    return true;
}

std::wstring QuoteWindowsArgument(const std::wstring& value) {
    std::wstring quoted = L"\"";
    std::size_t backslashes = 0;
    for (const wchar_t character : value) {
        if (character == L'\\') {
            ++backslashes;
            continue;
        }
        if (character == L'\"') {
            quoted.append(backslashes * 2 + 1, L'\\');
            quoted.push_back(character);
            backslashes = 0;
            continue;
        }
        quoted.append(backslashes, L'\\');
        backslashes = 0;
        quoted.push_back(character);
    }
    quoted.append(backslashes * 2, L'\\');
    quoted.push_back(L'\"');
    return quoted;
}

bool SetRegistryString(
    HKEY key,
    const wchar_t* name,
    const std::wstring& value,
    std::string& error) {
    const auto bytes = static_cast<DWORD>(
        (value.size() + 1) * sizeof(wchar_t));
    const LSTATUS status = ::RegSetValueExW(
        key, name, 0, REG_SZ,
        reinterpret_cast<const BYTE*>(value.c_str()), bytes);
    if (status != ERROR_SUCCESS) {
        error = "Cannot write CyxWiz product registration; Win32 error " +
            std::to_string(status);
        return false;
    }
    return true;
}

bool CreateShortcut(
    const std::filesystem::path& destination,
    const std::filesystem::path& launcher,
    const wchar_t* arguments,
    const wchar_t* description,
    std::string& error) {
    ComOwner<IShellLinkW> link;
    HRESULT result = ::CoCreateInstance(
        CLSID_ShellLink, nullptr, CLSCTX_INPROC_SERVER,
        IID_IShellLinkW, reinterpret_cast<void**>(link.output()));
    if (FAILED(result) || !link) {
        error = "Cannot create the CyxWiz Start Menu link";
        return false;
    }
    if (FAILED(link->SetPath(launcher.c_str())) ||
        FAILED(link->SetArguments(arguments)) ||
        FAILED(link->SetWorkingDirectory(launcher.parent_path().c_str())) ||
        FAILED(link->SetDescription(description)) ||
        FAILED(link->SetIconLocation(launcher.c_str(), 0))) {
        error = "Cannot configure the CyxWiz Start Menu link";
        return false;
    }
    ComOwner<IPersistFile> persistence;
    result = link->QueryInterface(
        IID_IPersistFile, reinterpret_cast<void**>(persistence.output()));
    if (FAILED(result) || !persistence) {
        error = "Cannot persist the CyxWiz Start Menu link";
        return false;
    }
    auto temporary = destination;
    temporary += L".part";
    std::error_code ignored;
    std::filesystem::remove(temporary, ignored);
    result = persistence->Save(temporary.c_str(), TRUE);
    if (FAILED(result)) {
        std::filesystem::remove(temporary, ignored);
        error = "Cannot write the CyxWiz Start Menu link; HRESULT " +
            std::to_string(static_cast<unsigned long>(result));
        return false;
    }
    if (!::MoveFileExW(
            temporary.c_str(), destination.c_str(),
            MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) {
        const DWORD code = ::GetLastError();
        std::filesystem::remove(temporary, ignored);
        error = "Cannot publish the CyxWiz Start Menu link; Win32 error " +
            std::to_string(code);
        return false;
    }
    return true;
}

bool RemoveFileIfPresent(
    const std::filesystem::path& path,
    std::string& error) {
    std::error_code filesystem_error;
    std::filesystem::remove(path, filesystem_error);
    if (filesystem_error) {
        error = "Cannot remove CyxWiz product integration: " +
            filesystem_error.message();
        return false;
    }
    return true;
}

}  // namespace

ProductRegistrationResult RegisterPlatformProduct(
    const ProductRegistrationRequest& request) {
    ProductRegistrationResult result;
    ComScope com;
    if (!com.available()) {
        result.message = "Cannot initialize Windows product registration";
        return result;
    }
    std::filesystem::path programs;
    if (!ResolveProgramsDirectory(
            request.scope, true, programs, result.message)) {
        return result;
    }
    const auto product_folder = programs / L"CyxWiz";
    std::error_code filesystem_error;
    std::filesystem::create_directories(product_folder, filesystem_error);
    if (filesystem_error) {
        result.message = "Cannot create the CyxWiz Start Menu directory: " +
            filesystem_error.message();
        return result;
    }
    const auto launcher = request.install_root /
        std::string(CurrentRuntimeBootstrapperExecutableName());
    std::string error;
    if (!CreateShortcut(
            product_folder / L"CyxWiz.lnk", launcher, L"",
            L"Launch CyxWiz Engine", error) ||
        !CreateShortcut(
            product_folder / L"CyxWiz Installer.lnk", launcher,
            L"--installer", L"Modify or repair CyxWiz", error)) {
        result.message = std::move(error);
        return result;
    }

    const HKEY hive = request.scope == ProductInstallScope::AllUsers
        ? HKEY_LOCAL_MACHINE : HKEY_CURRENT_USER;
    RegistryKey key;
    const LSTATUS create_status = ::RegCreateKeyExW(
        hive,
        L"Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\CyxWiz",
        0, nullptr, REG_OPTION_NON_VOLATILE, KEY_WRITE, nullptr,
        key.output(), nullptr);
    if (create_status != ERROR_SUCCESS) {
        result.message =
            "Cannot register CyxWiz with Windows Apps & Features; Win32 error " +
            std::to_string(create_status);
        return result;
    }
    const auto uninstall = QuoteWindowsArgument(launcher.native()) +
        L" --installer";
    const std::wstring display_name(kProductName.begin(), kProductName.end());
    const std::wstring publisher(kPublisher.begin(), kPublisher.end());
    const std::wstring version(
        request.product_version.begin(), request.product_version.end());
    if (!SetRegistryString(key.get(), L"DisplayName", display_name, error) ||
        !SetRegistryString(key.get(), L"Publisher", publisher, error) ||
        !SetRegistryString(key.get(), L"DisplayVersion", version, error) ||
        !SetRegistryString(
            key.get(), L"InstallLocation", request.install_root.native(), error) ||
        !SetRegistryString(
            key.get(), L"DisplayIcon", launcher.native(), error) ||
        !SetRegistryString(key.get(), L"UninstallString", uninstall, error) ||
        !SetRegistryString(key.get(), L"ModifyPath", uninstall, error)) {
        result.message = std::move(error);
        return result;
    }
    result.registered = true;
    result.message =
        "CyxWiz was registered in the Start Menu and Apps & Features";
    return result;
}

ProductUnregistrationResult UnregisterPlatformProduct(
    const ProductRegistrationRequest& request) {
    ProductUnregistrationResult result;
    ComScope com;
    if (!com.available()) {
        result.message = "Cannot initialize Windows product unregistration";
        return result;
    }
    std::filesystem::path programs;
    if (!ResolveProgramsDirectory(
            request.scope, false, programs, result.message)) {
        return result;
    }
    const auto product_folder = programs / L"CyxWiz";
    std::string error;
    if (!RemoveFileIfPresent(product_folder / L"CyxWiz.lnk", error) ||
        !RemoveFileIfPresent(
            product_folder / L"CyxWiz Installer.lnk", error)) {
        result.message = std::move(error);
        return result;
    }
    std::error_code ignored;
    std::filesystem::remove(product_folder, ignored);

    const HKEY hive = request.scope == ProductInstallScope::AllUsers
        ? HKEY_LOCAL_MACHINE : HKEY_CURRENT_USER;
    const LSTATUS delete_status = ::RegDeleteKeyW(
        hive,
        L"Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\CyxWiz");
    if (delete_status != ERROR_SUCCESS &&
        delete_status != ERROR_FILE_NOT_FOUND) {
        result.message =
            "Cannot remove CyxWiz from Windows Apps & Features; Win32 error " +
            std::to_string(delete_status);
        return result;
    }
    result.unregistered = true;
    result.message =
        "CyxWiz Start Menu and Apps & Features entries were removed";
    return result;
}

}  // namespace cyxwiz::runtime::detail
