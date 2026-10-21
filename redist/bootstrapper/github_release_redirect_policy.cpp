#include "github_release_redirect_policy.h"

#include <array>
#include <cstddef>

namespace cyxwiz::runtime {
namespace {

constexpr std::size_t kMaximumUrlBytes = 4096;
constexpr std::string_view kGithubPrefix = "https://github.com/";
constexpr std::string_view kReleaseAssetPrefix =
    "https://release-assets.githubusercontent.com/";
constexpr std::string_view kReleaseAssetPathPrefix =
    "github-production-release-asset/";

bool IsBoundedAscii(std::string_view value) {
    if (value.empty() || value.size() > kMaximumUrlBytes) return false;
    for (const unsigned char character : value) {
        if (character <= 0x20 || character >= 0x7f || character == '\\') {
            return false;
        }
    }
    return true;
}

bool IsSafePathCharacter(char character) {
    return (character >= 'a' && character <= 'z') ||
           (character >= 'A' && character <= 'Z') ||
           (character >= '0' && character <= '9') || character == '.' ||
           character == '_' || character == '-';
}

bool IsSafePathSegment(std::string_view value, std::size_t maximum_size) {
    if (value.empty() || value.size() > maximum_size || value == "." ||
        value == "..") {
        return false;
    }
    for (const char character : value) {
        if (!IsSafePathCharacter(character)) return false;
    }
    return true;
}

bool IsSafeAssetName(std::string_view value) {
    if (value.empty() || value.size() > 255 || value == "." || value == "..") {
        return false;
    }
    for (const char character : value) {
        if (!IsSafePathCharacter(character) && character != '+') return false;
    }
    return true;
}

bool EqualsAsciiCaseInsensitive(
    std::string_view left,
    std::string_view right) {
    if (left.size() != right.size()) return false;
    for (std::size_t index = 0; index < left.size(); ++index) {
        const auto lower = [](char character) {
            return character >= 'A' && character <= 'Z'
                ? static_cast<char>(character - 'A' + 'a')
                : character;
        };
        if (lower(left[index]) != lower(right[index])) return false;
    }
    return true;
}

template <std::size_t Size>
bool SplitExact(
    std::string_view value,
    std::array<std::string_view, Size>& segments) {
    std::size_t start = 0;
    for (std::size_t index = 0; index < Size; ++index) {
        const auto separator = value.find('/', start);
        if (index + 1 == Size) {
            if (separator != std::string_view::npos) return false;
            segments[index] = value.substr(start);
        } else {
            if (separator == std::string_view::npos) return false;
            segments[index] = value.substr(start, separator - start);
            start = separator + 1;
        }
    }
    return true;
}

bool IsCanonicalGithubReleaseAssetUrl(std::string_view url) {
    if (!IsBoundedAscii(url) || !url.starts_with(kGithubPrefix) ||
        url.find_first_of("?#@") != std::string_view::npos) {
        return false;
    }
    std::array<std::string_view, 6> segments;
    if (!SplitExact(url.substr(kGithubPrefix.size()), segments)) return false;
    return IsSafePathSegment(segments[0], 100) &&
           IsSafePathSegment(segments[1], 100) &&
           segments[2] == "releases" && segments[3] == "download" &&
           IsSafePathSegment(segments[4], 128) &&
           !EqualsAsciiCaseInsensitive(segments[4], "latest") &&
           IsSafeAssetName(segments[5]);
}

bool IsGithubReleaseAssetLocation(std::string_view location) {
    if (!IsBoundedAscii(location) ||
        !location.starts_with(kReleaseAssetPrefix) ||
        location.find('#') != std::string_view::npos) {
        return false;
    }
    const auto remainder = location.substr(kReleaseAssetPrefix.size());
    const auto query = remainder.find('?');
    if (query == std::string_view::npos || query == 0 ||
        query + 1 == remainder.size()) {
        return false;
    }
    const auto path = remainder.substr(0, query);
    if (!path.starts_with(kReleaseAssetPathPrefix)) return false;
    std::array<std::string_view, 2> segments;
    if (!SplitExact(path.substr(kReleaseAssetPathPrefix.size()), segments)) {
        return false;
    }
    if (segments[0].empty()) return false;
    for (const char character : segments[0]) {
        if (character < '0' || character > '9') return false;
    }
    return IsSafePathSegment(segments[1], 128);
}

}  // namespace

bool AuthorizeGithubReleaseAssetRedirect(
    std::string_view original_url,
    unsigned int response_status,
    std::string_view location,
    std::string& authorized_url,
    std::string& error) {
    authorized_url.clear();
    if (response_status != 302U) {
        error = "HTTPS redirect status is not authorized";
        return false;
    }
    if (!IsCanonicalGithubReleaseAssetUrl(original_url)) {
        error = "HTTPS redirect source is not an immutable GitHub release asset";
        return false;
    }
    if (!IsGithubReleaseAssetLocation(location)) {
        error = "HTTPS redirect destination authority is not authorized";
        return false;
    }
    authorized_url.assign(location);
    error.clear();
    return true;
}

}  // namespace cyxwiz::runtime
