#include "github_release_redirect_policy.h"

#include <iostream>
#include <string>
#include <string_view>
#include <vector>

namespace {

using cyxwiz::runtime::AuthorizeGithubReleaseAssetRedirect;

constexpr std::string_view kSource =
    "https://github.com/CYXWIZ-Lab/CYXWIZ/releases/download/"
    "v0.2.0-alpha.1/cyxwiz-installer.zip";
constexpr std::string_view kLocation =
    "https://release-assets.githubusercontent.com/"
    "github-production-release-asset/12345/"
    "01234567-89ab-cdef-0123-456789abcdef?sp=r&sig=example";

bool Rejects(
    std::string_view source,
    unsigned int status,
    std::string_view location) {
    std::string authorized = "stale";
    std::string error;
    return !AuthorizeGithubReleaseAssetRedirect(
               source, status, location, authorized, error) &&
           authorized.empty() && !error.empty() &&
           (location.empty() || error.find(location) == std::string::npos);
}

}  // namespace

int main() {
    int failures = 0;
    const auto expect = [&](bool condition, const char* message) {
        if (condition) return;
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    };

    std::string authorized;
    std::string error;
    expect(
        AuthorizeGithubReleaseAssetRedirect(
            kSource, 302, kLocation, authorized, error),
        "canonical GitHub release redirect must be authorized");
    expect(authorized == kLocation && error.empty(),
           "authorization must return only the accepted destination");
    expect(
        AuthorizeGithubReleaseAssetRedirect(
            "https://github.com/o/r/releases/download/v1/a+build.zip",
            302, kLocation, authorized, error),
        "safe semantic-version asset names must be authorized");

    const std::vector<std::string> invalid_sources = {
        "http://github.com/o/r/releases/download/v1/a.zip",
        "https://github.com.evil.test/o/r/releases/download/v1/a.zip",
        "https://user@github.com/o/r/releases/download/v1/a.zip",
        "https://github.com:443/o/r/releases/download/v1/a.zip",
        "https://github.com/o/r/releases/latest/download/a.zip",
        "https://github.com/o/r/releases/download/latest/a.zip",
        "https://github.com/o/r/releases/download/LATEST/a.zip",
        "https://github.com/o/r/releases/download/v1/nested/a.zip",
        "https://github.com/o/../releases/download/v1/a.zip",
        "https://github.com/o/r/releases/download/v1/a%20b.zip",
        "https://github.com/o/r/releases/download/v1/a.zip?token=x",
        "https://github.com/o/r/releases/download/v1/a.zip#fragment",
    };
    for (const auto& source : invalid_sources) {
        expect(Rejects(source, 302, kLocation),
               "unsafe or mutable source must be rejected");
    }

    const std::vector<std::string> invalid_locations = {
        "",
        "/relative-release-asset?sig=x",
        "http://release-assets.githubusercontent.com/"
        "github-production-release-asset/1/a?sig=x",
        "https://user@release-assets.githubusercontent.com/"
        "github-production-release-asset/1/a?sig=x",
        "https://release-assets.githubusercontent.com:443/"
        "github-production-release-asset/1/a?sig=x",
        "https://release-assets.githubusercontent.com.evil.test/"
        "github-production-release-asset/1/a?sig=x",
        "https://objects.githubusercontent.com/"
        "github-production-release-asset/1/a?sig=x",
        "https://release-assets.githubusercontent.com/other/1/a?sig=x",
        "https://release-assets.githubusercontent.com/"
        "github-production-release-asset/not-a-number/a?sig=x",
        "https://release-assets.githubusercontent.com/"
        "github-production-release-asset/1/a",
        "https://release-assets.githubusercontent.com/"
        "github-production-release-asset/1/a/extra?sig=x",
        "https://release-assets.githubusercontent.com/"
        "github-production-release-asset/1/a?sig=x#fragment",
    };
    for (const auto& location : invalid_locations) {
        expect(Rejects(kSource, 302, location),
               "unrecognized redirect destination must be rejected");
    }

    expect(Rejects(kSource, 301, kLocation), "301 must be rejected");
    expect(Rejects(kSource, 307, kLocation), "307 must be rejected");
    expect(Rejects(kSource, 308, kLocation), "308 must be rejected");
    return failures == 0 ? 0 : 1;
}
