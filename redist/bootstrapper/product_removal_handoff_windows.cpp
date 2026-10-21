#include "product_removal_handoff_platform.h"

#include "backend_pack_platform.h"

#include <chrono>
#include <cstddef>
#include <filesystem>
#include <string>
#include <vector>

#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>

namespace cyxwiz::runtime::detail {
namespace {

constexpr LONGLONG kMaximumFinalizerBytes = 16 * 1024 * 1024;

std::wstring QuoteArgument(const std::wstring& argument) {
    if (!argument.empty() &&
        argument.find_first_of(L" \t\n\v\"") == std::wstring::npos) {
        return argument;
    }
    std::wstring quoted = L"\"";
    std::size_t backslashes = 0;
    for (const wchar_t character : argument) {
        if (character == L'\\') {
            ++backslashes;
        } else if (character == L'\"') {
            quoted.append(backslashes * 2 + 1, L'\\');
            quoted.push_back(character);
            backslashes = 0;
        } else {
            quoted.append(backslashes, L'\\');
            backslashes = 0;
            quoted.push_back(character);
        }
    }
    quoted.append(backslashes * 2, L'\\');
    quoted.push_back(L'\"');
    return quoted;
}

void RemoveStaging(
    const std::filesystem::path& finalizer,
    const std::filesystem::path& directory) {
    if (!finalizer.empty()) ::DeleteFileW(finalizer.c_str());
    if (!directory.empty()) ::RemoveDirectoryW(directory.c_str());
}

bool CopyExactExecutable(
    const std::filesystem::path& source,
    const std::filesystem::path& destination,
    std::string& error) {
    const HANDLE input = ::CreateFileW(
        source.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
        FILE_ATTRIBUTE_NORMAL | FILE_FLAG_OPEN_REPARSE_POINT, nullptr);
    if (input == INVALID_HANDLE_VALUE) {
        error = "Cannot open the exact product removal finalizer; Win32 error " +
            std::to_string(::GetLastError());
        return false;
    }
    FILE_ATTRIBUTE_TAG_INFO source_info{};
    LARGE_INTEGER source_size{};
    if (!::GetFileInformationByHandleEx(
            input, FileAttributeTagInfo, &source_info, sizeof(source_info)) ||
        !::GetFileSizeEx(input, &source_size) || source_size.QuadPart <= 0 ||
        source_size.QuadPart > kMaximumFinalizerBytes ||
        (source_info.FileAttributes &
            (FILE_ATTRIBUTE_DIRECTORY | FILE_ATTRIBUTE_REPARSE_POINT)) != 0) {
        const DWORD code = ::GetLastError();
        ::CloseHandle(input);
        error = "The product removal finalizer is not an exact file; Win32 "
            "error " + std::to_string(code);
        return false;
    }
    const HANDLE output = ::CreateFileW(
        destination.c_str(), GENERIC_WRITE, 0, nullptr, CREATE_NEW,
        FILE_ATTRIBUTE_NORMAL, nullptr);
    if (output == INVALID_HANDLE_VALUE) {
        const DWORD code = ::GetLastError();
        ::CloseHandle(input);
        error = "Cannot create the staged product removal finalizer; Win32 "
            "error " + std::to_string(code);
        return false;
    }
    bool succeeded = true;
    char buffer[64 * 1024];
    for (;;) {
        DWORD bytes_read = 0;
        if (!::ReadFile(input, buffer, sizeof(buffer), &bytes_read, nullptr)) {
            succeeded = false;
            break;
        }
        if (bytes_read == 0) break;
        DWORD offset = 0;
        while (offset < bytes_read) {
            DWORD written = 0;
            if (!::WriteFile(
                    output, buffer + offset, bytes_read - offset,
                    &written, nullptr) || written == 0) {
                succeeded = false;
                break;
            }
            offset += written;
        }
        if (!succeeded) break;
    }
    if (succeeded && !::FlushFileBuffers(output)) succeeded = false;
    const DWORD operation_error = ::GetLastError();
    ::CloseHandle(output);
    ::CloseHandle(input);
    if (!succeeded) {
        ::DeleteFileW(destination.c_str());
        error = "Cannot copy the complete product removal finalizer; Win32 "
            "error " + std::to_string(operation_error);
    }
    return succeeded;
}

bool CreateStagingDirectory(
    std::string_view install_id,
    std::filesystem::path& directory,
    std::string& error) {
    std::vector<wchar_t> buffer(32768);
    const DWORD length = ::GetTempPathW(
        static_cast<DWORD>(buffer.size()), buffer.data());
    if (length == 0 || length >= buffer.size()) {
        error = "Cannot resolve the product-removal temporary directory";
        return false;
    }
    const std::filesystem::path temporary(
        std::wstring(buffer.data(), length));
    const auto nonce = std::chrono::steady_clock::now()
        .time_since_epoch().count();
    for (unsigned int attempt = 0; attempt < 32; ++attempt) {
        directory = temporary /
            ("cyxwiz-removal-" + std::string(install_id) + "-" +
             std::to_string(::GetCurrentProcessId()) + "-" +
             std::to_string(nonce) + "-" + std::to_string(attempt));
        if (::CreateDirectoryW(directory.c_str(), nullptr)) return true;
        if (::GetLastError() != ERROR_ALREADY_EXISTS) break;
    }
    error = "Cannot create an exclusive product-removal temporary directory; "
        "Win32 error " + std::to_string(::GetLastError());
    directory.clear();
    return false;
}

}  // namespace

bool LaunchDetachedProductRemovalFinalizer(
    const std::filesystem::path& source_finalizer,
    const std::filesystem::path& install_root,
    std::string_view install_id,
    ProductRemovalHandoff& handoff,
    std::string& error) {
    std::filesystem::path directory;
    if (!CreateStagingDirectory(install_id, directory, error)) return false;
    const auto staged = directory /
        std::string(CurrentProductRemovalFinalizerExecutableName());
    const auto result = directory / "result.txt";
    if (!CopyExactExecutable(source_finalizer, staged, error)) {
        RemoveStaging(staged, directory);
        return false;
    }
    const DWORD attributes = ::GetFileAttributesW(staged.c_str());
    if (attributes == INVALID_FILE_ATTRIBUTES ||
        (attributes & (FILE_ATTRIBUTE_DIRECTORY |
                       FILE_ATTRIBUTE_REPARSE_POINT)) != 0) {
        error = "The staged product removal finalizer is not an exact file";
        RemoveStaging(staged, directory);
        return false;
    }

    SECURITY_ATTRIBUTES security{};
    security.nLength = sizeof(security);
    security.bInheritHandle = TRUE;
    HANDLE read_handle = nullptr;
    HANDLE write_handle = nullptr;
    if (!::CreatePipe(&read_handle, &write_handle, &security, 0) ||
        !::SetHandleInformation(
            write_handle, HANDLE_FLAG_INHERIT, 0)) {
        const DWORD code = ::GetLastError();
        if (read_handle != nullptr) ::CloseHandle(read_handle);
        if (write_handle != nullptr) ::CloseHandle(write_handle);
        RemoveStaging(staged, directory);
        error = "Cannot create the product-removal lifetime boundary; Win32 "
            "error " + std::to_string(code);
        return false;
    }

    SIZE_T attribute_bytes = 0;
    ::InitializeProcThreadAttributeList(nullptr, 1, 0, &attribute_bytes);
    std::vector<std::byte> attribute_storage(attribute_bytes);
    auto* attributes_list = reinterpret_cast<LPPROC_THREAD_ATTRIBUTE_LIST>(
        attribute_storage.data());
    const bool attributes_initialized = attribute_bytes != 0 &&
        ::InitializeProcThreadAttributeList(
            attributes_list, 1, 0, &attribute_bytes);
    const bool handle_list_set = attributes_initialized &&
        ::UpdateProcThreadAttribute(
            attributes_list, 0, PROC_THREAD_ATTRIBUTE_HANDLE_LIST,
            &read_handle, sizeof(read_handle), nullptr, nullptr);
    if (!handle_list_set) {
        const DWORD code = ::GetLastError();
        if (attributes_initialized) {
            ::DeleteProcThreadAttributeList(attributes_list);
        }
        ::CloseHandle(read_handle);
        ::CloseHandle(write_handle);
        RemoveStaging(staged, directory);
        error = "Cannot restrict product-removal handle inheritance; Win32 "
            "error " + std::to_string(code);
        return false;
    }

    std::wstring command = QuoteArgument(staged.native()) +
        L" --install-root " + QuoteArgument(install_root.native()) +
        L" --parent-lifetime-handle " +
        std::to_wstring(reinterpret_cast<std::uintptr_t>(read_handle));
    std::vector<wchar_t> mutable_command(command.begin(), command.end());
    mutable_command.push_back(L'\0');
    STARTUPINFOEXW startup{};
    startup.StartupInfo.cb = sizeof(startup);
    startup.lpAttributeList = attributes_list;
    PROCESS_INFORMATION process{};
    const BOOL created = ::CreateProcessW(
        staged.c_str(), mutable_command.data(), nullptr, nullptr, TRUE,
        EXTENDED_STARTUPINFO_PRESENT | CREATE_NO_WINDOW, nullptr,
        directory.c_str(), &startup.StartupInfo, &process);
    const DWORD create_error = created ? ERROR_SUCCESS : ::GetLastError();
    ::DeleteProcThreadAttributeList(attributes_list);
    ::CloseHandle(read_handle);
    if (!created) {
        ::CloseHandle(write_handle);
        RemoveStaging(staged, directory);
        error = "Cannot launch the detached product removal finalizer; Win32 "
            "error " + std::to_string(create_error);
        return false;
    }
    ::CloseHandle(process.hThread);
    ::CloseHandle(process.hProcess);
    handoff.staged_finalizer = staged;
    handoff.result_path = result;
    handoff.parent_lifetime = ProductRemovalParentLifetime(
        reinterpret_cast<std::uintptr_t>(write_handle));
    return true;
}

void CloseProductRemovalLifetimeToken(std::uintptr_t token) noexcept {
    ::CloseHandle(reinterpret_cast<HANDLE>(token));
}

}  // namespace cyxwiz::runtime::detail
