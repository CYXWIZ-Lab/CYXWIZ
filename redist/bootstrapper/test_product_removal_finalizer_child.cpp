#include "product_removal_finalizer.h"

#include <charconv>
#include <cstdint>
#include <fstream>
#include <string>
#include <string_view>

#ifdef _WIN32
#include <cerrno>
#include <cwchar>
#else
#include <cerrno>
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
    const auto parsed = std::from_chars(
        text.data(), text.data() + text.size(), token);
    return parsed.ec == std::errc{} &&
        parsed.ptr == text.data() + text.size();
}
#endif

bool PublishResult(
    const std::filesystem::path& executable,
    std::string_view value) {
    std::ofstream stream(
        executable.parent_path() / "result.txt",
        std::ios::binary | std::ios::trunc);
    stream.write(value.data(), static_cast<std::streamsize>(value.size()));
    stream.flush();
    return static_cast<bool>(stream);
}

}  // namespace

#ifdef _WIN32
int wmain(int argc, wchar_t** argv) {
    if (argc != 5 || std::wstring_view(argv[1]) != L"--install-root" ||
        std::wstring_view(argv[3]) != L"--parent-lifetime-handle") {
        return 78;
    }
    std::uintptr_t token = 0;
    if (!ParseToken(argv[4], token)) return 78;
    cyxwiz::runtime::ProductRemovalAuthorization authorization;
    std::string error;
    const bool authorized = cyxwiz::runtime::AwaitAuthorizedProductRemoval(
        std::filesystem::path(argv[2]).lexically_normal(), token,
        authorization, error);
    return PublishResult(
        std::filesystem::path(argv[0]),
        authorized ? "authorized\n" : "rejected\n") ? 0 : 78;
}
#else
int main(int argc, char** argv) {
    if (argc != 5 || std::string_view(argv[1]) != "--install-root" ||
        std::string_view(argv[3]) != "--parent-lifetime-fd") {
        return 78;
    }
    std::uintptr_t token = 0;
    if (!ParseToken(argv[4], token)) return 78;
    cyxwiz::runtime::ProductRemovalAuthorization authorization;
    std::string error;
    const bool authorized = cyxwiz::runtime::AwaitAuthorizedProductRemoval(
        std::filesystem::path(argv[2]).lexically_normal(), token,
        authorization, error);
    return PublishResult(
        std::filesystem::path(argv[0]),
        authorized ? "authorized\n" : "rejected\n") ? 0 : 78;
}
#endif
