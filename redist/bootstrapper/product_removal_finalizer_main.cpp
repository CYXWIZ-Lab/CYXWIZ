#include "product_removal_finalizer.h"

#include <charconv>
#include <cstdint>
#include <iostream>
#include <string>
#include <string_view>

#ifdef _WIN32
#include <cerrno>
#include <cwchar>
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
        std::cerr << "CyxWiz product removal finalizer: " << error << '\n';
        return 78;
    }
    return 0;
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
        std::cerr << "CyxWiz product removal finalizer: " << error << '\n';
        return 78;
    }
    return 0;
}
#endif
