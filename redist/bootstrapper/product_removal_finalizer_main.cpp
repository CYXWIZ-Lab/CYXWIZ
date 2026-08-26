#include "product_removal_finalizer.h"

#include <charconv>
#include <cstdint>
#include <iostream>
#include <string>
#include <string_view>

#ifdef _WIN32
#include <cerrno>
#include <cwchar>
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <cerrno>
#include <fcntl.h>
#include <unistd.h>
#endif

namespace {

#ifdef _WIN32
bool ParseToken(const wchar_t* text, std::uintptr_t& token) {
    if (text == nullptr || *text == L'\0' || *text == L'-') return false;
    wchar_t* end = nullptr;
    errno = 0;
    const auto value = std::wcstoull(text, &end, 10);
    if (errno != 0 || end == text || *end != L'\0') return false;
    token = static_cast<std::uintptr_t>(value);
    return static_cast<unsigned long long>(token) == value;
}
#else
bool ParseToken(std::string_view text, std::uintptr_t& token) {
    if (text.empty() || text.front() == '-') return false;
    std::uintptr_t value = 0;
    const auto parsed = std::from_chars(
        text.data(), text.data() + text.size(), value);
    if (parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size()) {
        return false;
    }
    token = value;
    return true;
}
#endif

bool PublishResult(
    const std::filesystem::path& executable,
    std::string_view value) {
    if (!executable.is_absolute()) return false;
    const auto path = executable.parent_path() / "result.txt";
#ifdef _WIN32
    const HANDLE file = ::CreateFileW(
        path.c_str(), GENERIC_WRITE, 0, nullptr, CREATE_NEW,
        FILE_ATTRIBUTE_NORMAL, nullptr);
    if (file == INVALID_HANDLE_VALUE || value.size() > MAXDWORD) return false;
    DWORD written = 0;
    const BOOL succeeded = ::WriteFile(
        file, value.data(), static_cast<DWORD>(value.size()),
        &written, nullptr);
    ::CloseHandle(file);
    return succeeded && written == value.size();
#else
    const int file = ::open(
        path.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC | O_NOFOLLOW,
        0600);
    if (file < 0) return false;
    std::size_t offset = 0;
    while (offset < value.size()) {
        const auto written = ::write(
            file, value.data() + offset, value.size() - offset);
        if (written < 0 && errno == EINTR) continue;
        if (written <= 0) {
            ::close(file);
            return false;
        }
        offset += static_cast<std::size_t>(written);
    }
    return ::close(file) == 0;
#endif
}

}  // namespace

#ifdef _WIN32
int wmain(int argc, wchar_t** argv) {
    if (argc != 5 || std::wstring_view(argv[1]) != L"--install-root" ||
        std::wstring_view(argv[3]) != L"--parent-lifetime-handle") {
        std::cerr << "CyxWiz product removal finalizer: invalid arguments\n";
        return 78;
    }
    std::uintptr_t token = 0;
    if (!ParseToken(argv[4], token)) {
        std::cerr << "CyxWiz product removal finalizer: invalid lifetime handle\n";
        return 78;
    }
    cyxwiz::runtime::ProductRemovalAuthorization authorization;
    std::string error;
    if (!cyxwiz::runtime::AwaitAuthorizedProductRemoval(
            std::filesystem::path(argv[2]).lexically_normal(), token,
            authorization, error)) {
        PublishResult(std::filesystem::path(argv[0]), "rejected\n");
        std::cerr << "CyxWiz product removal finalizer: " << error << '\n';
        return 78;
    }
    return PublishResult(std::filesystem::path(argv[0]), "authorized\n")
        ? 0 : 78;
}
#else
int main(int argc, char** argv) {
    if (argc != 5 || std::string_view(argv[1]) != "--install-root" ||
        std::string_view(argv[3]) != "--parent-lifetime-fd") {
        std::cerr << "CyxWiz product removal finalizer: invalid arguments\n";
        return 78;
    }
    std::uintptr_t token = 0;
    if (!ParseToken(argv[4], token)) {
        std::cerr << "CyxWiz product removal finalizer: invalid lifetime descriptor\n";
        return 78;
    }
    cyxwiz::runtime::ProductRemovalAuthorization authorization;
    std::string error;
    if (!cyxwiz::runtime::AwaitAuthorizedProductRemoval(
            std::filesystem::path(argv[2]).lexically_normal(), token,
            authorization, error)) {
        PublishResult(std::filesystem::path(argv[0]), "rejected\n");
        std::cerr << "CyxWiz product removal finalizer: " << error << '\n';
        return 78;
    }
    return PublishResult(std::filesystem::path(argv[0]), "authorized\n")
        ? 0 : 78;
}
#endif
