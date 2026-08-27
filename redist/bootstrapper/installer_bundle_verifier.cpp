#include "installer_bundle_verifier.h"

#include "backend_pack_hash.h"
#include "backend_pack_path.h"

#include <algorithm>
#include <cctype>
#include <limits>
#include <set>
#include <string_view>

#include <nlohmann/json.hpp>

namespace cyxwiz::runtime {
namespace {

using Json = nlohmann::json;
constexpr std::uint64_t kMaximumArchiveBytes = 256U * 1024U * 1024U;
constexpr std::uint64_t kMaximumComponentBytes = 512U * 1024U * 1024U;
constexpr std::size_t kMaximumComponents = 4096;

bool HasExactKeys(
    const Json& object, std::initializer_list<const char*> expected) {
    return object.is_object() && object.size() == expected.size() &&
        std::all_of(expected.begin(), expected.end(),
                    [&](const char* key) { return object.contains(key); });
}

bool IsIdentifier(const std::string& value) {
    return !value.empty() && value.size() <= 128 &&
        std::isalnum(static_cast<unsigned char>(value.front())) &&
        std::all_of(value.begin(), value.end(), [](unsigned char character) {
            return std::islower(character) || std::isdigit(character) ||
                character == '.' || character == '_' || character == '-';
        });
}

bool IsVersion(const std::string& value) {
    return !value.empty() && value.size() <= 64 &&
        std::isalnum(static_cast<unsigned char>(value.front())) &&
        std::all_of(value.begin(), value.end(), [](unsigned char character) {
            return std::isalnum(character) || character == '.' ||
                character == '_' || character == '+' || character == '-';
        });
}

bool IsLeapYear(int year) {
    return year % 4 == 0 && (year % 100 != 0 || year % 400 == 0);
}

bool IsUtc(const std::string& value) {
    if (value.size() != 20 || value[4] != '-' || value[7] != '-' ||
        value[10] != 'T' || value[13] != ':' || value[16] != ':' ||
        value[19] != 'Z') {
        return false;
    }
    for (std::size_t index = 0; index < value.size(); ++index) {
        if (index == 4 || index == 7 || index == 10 || index == 13 ||
            index == 16 || index == 19) continue;
        if (!std::isdigit(static_cast<unsigned char>(value[index]))) return false;
    }
    const int year = std::stoi(value.substr(0, 4));
    const int month = std::stoi(value.substr(5, 2));
    const int day = std::stoi(value.substr(8, 2));
    const int hour = std::stoi(value.substr(11, 2));
    const int minute = std::stoi(value.substr(14, 2));
    const int second = std::stoi(value.substr(17, 2));
    if (year < 1970 || month < 1 || month > 12 || hour > 23 ||
        minute > 59 || second > 59) return false;
    constexpr int days[] = {0, 31, 28, 31, 30, 31, 30,
                            31, 31, 30, 31, 30, 31};
    const int maximum_day = days[month] + (month == 2 && IsLeapYear(year));
    return day >= 1 && day <= maximum_day;
}

bool ReadString(
    const Json& object, const char* key, std::string& output) {
    if (!object.contains(key) || !object[key].is_string()) return false;
    output = object[key].get<std::string>();
    return !output.empty();
}

bool ParseNumericVersion(
    const std::string& value, std::vector<unsigned int>& parts) {
    parts.clear();
    std::size_t begin = 0;
    while (begin < value.size()) {
        const auto end = value.find('.', begin);
        const auto part = value.substr(
            begin, (end == std::string::npos ? value.size() : end) - begin);
        if (part.empty() || !std::all_of(
                part.begin(), part.end(), [](unsigned char character) {
                    return std::isdigit(character);
                })) return false;
        try {
            const auto parsed = std::stoul(part);
            if (parsed > std::numeric_limits<unsigned int>::max()) return false;
            parts.push_back(static_cast<unsigned int>(parsed));
        } catch (...) {
            return false;
        }
        begin = end == std::string::npos ? value.size() : end + 1;
    }
    return !parts.empty();
}

bool VersionAtLeast(const std::string& current, const std::string& minimum) {
    std::vector<unsigned int> current_parts;
    std::vector<unsigned int> minimum_parts;
    if (!ParseNumericVersion(current, current_parts) ||
        !ParseNumericVersion(minimum, minimum_parts)) return false;
    const auto count = std::max(current_parts.size(), minimum_parts.size());
    for (std::size_t index = 0; index < count; ++index) {
        const auto left = index < current_parts.size() ? current_parts[index] : 0;
        const auto right = index < minimum_parts.size() ? minimum_parts[index] : 0;
        if (left != right) return left > right;
    }
    return true;
}

bool ReadUnsigned(
    const Json& object, const char* key, std::uint64_t maximum,
    bool allow_zero, std::uint64_t& output) {
    if (!object.contains(key) || !object[key].is_number_unsigned()) return false;
    output = object[key].get<std::uint64_t>();
    return (allow_zero || output > 0) && output <= maximum;
}

}  // namespace

std::string VerifiedInstallerBundle::InstallerEntryPoint() const {
    return platform == "windows" ? "cyxwiz-installer.exe" : "cyxwiz-installer";
}

InstallerBundleVerifier::InstallerBundleVerifier(
    BackendPackTrustStore trust_store, std::string setup_version,
    std::string platform, std::string architecture)
    : trust_store_(std::move(trust_store)),
      setup_version_(std::move(setup_version)),
      platform_(std::move(platform)),
      architecture_(std::move(architecture)) {}

bool InstallerBundleVerifier::Verify(
    const std::filesystem::path& descriptor_path,
    const std::string& current_utc,
    VerifiedInstallerBundle& output,
    std::string& error) const {
    output = {};
    if (!IsUtc(current_utc) || !IsVersion(setup_version_)) {
        error = "Setup version or trusted UTC clock is invalid";
        return false;
    }
    std::string body_bytes;
    if (!trust_store_.VerifySignedDocument(
            descriptor_path, "cyxwiz-installer-bundle",
            TrustedMetadataRole::Installer, body_bytes, error)) return false;
    Json body;
    try {
        body = Json::parse(body_bytes);
    } catch (const std::exception& exception) {
        error = std::string("Verified installer body is invalid JSON: ") +
            exception.what();
        return false;
    }
    if (!HasExactKeys(
            body, {"bundle_id", "bundle_version", "cyxwiz_release",
                   "release_channel", "platform", "architecture",
                   "minimum_setup_version", "generated_utc", "expires_utc",
                   "archive", "components"}) ||
        !ReadString(body, "bundle_id", output.bundle_id) ||
        !ReadString(body, "bundle_version", output.bundle_version) ||
        !ReadString(body, "cyxwiz_release", output.cyxwiz_release) ||
        !ReadString(body, "release_channel", output.release_channel) ||
        !ReadString(body, "platform", output.platform) ||
        !ReadString(body, "architecture", output.architecture) ||
        !ReadString(body, "minimum_setup_version", output.minimum_setup_version) ||
        !ReadString(body, "generated_utc", output.generated_utc) ||
        !ReadString(body, "expires_utc", output.expires_utc) ||
        !IsIdentifier(output.bundle_id) || !IsVersion(output.bundle_version) ||
        !IsVersion(output.cyxwiz_release) ||
        !IsVersion(output.minimum_setup_version) ||
        (output.release_channel != "alpha" &&
         output.release_channel != "beta" &&
         output.release_channel != "stable") ||
        (output.platform != "windows" && output.platform != "linux" &&
         output.platform != "macos") ||
        (output.architecture != "x86_64" && output.architecture != "arm64") ||
        !IsUtc(output.generated_utc) || !IsUtc(output.expires_utc) ||
        output.expires_utc <= output.generated_utc) {
        error = "Signed installer bundle body violates schema 1";
        return false;
    }
    if (output.platform != platform_ || output.architecture != architecture_) {
        error = "Installer bundle targets a different platform or architecture";
        return false;
    }
    if (current_utc < output.generated_utc || current_utc >= output.expires_utc) {
        error = "Installer bundle is not current for the trusted clock";
        return false;
    }
    if (!VersionAtLeast(setup_version_, output.minimum_setup_version)) {
        error = "Installer bundle requires a newer setup program";
        return false;
    }
    const auto& archive = body["archive"];
    if (!HasExactKeys(archive, {"file_name", "size", "sha256"}) ||
        !ReadString(archive, "file_name", output.archive.file_name) ||
        !ReadUnsigned(archive, "size", kMaximumArchiveBytes, false,
                      output.archive.size) ||
        !ReadString(archive, "sha256", output.archive.sha256) ||
        output.archive.file_name != output.bundle_id + ".zip" ||
        !IsCanonicalBackendPackRelativePath(output.archive.file_name) ||
        !IsLowercaseSha256(output.archive.sha256)) {
        error = "Signed installer archive identity violates schema 1";
        return false;
    }
    const auto& components = body["components"];
    if (!components.is_array() || components.empty() ||
        components.size() > kMaximumComponents) {
        error = "Signed installer component count is outside its bounds";
        return false;
    }
    std::set<std::string> seen;
    std::uint64_t total_bytes = 0;
    std::string previous_path;
    std::set<std::string> paths;
    std::set<std::string> executable_paths;
    for (const auto& component : components) {
        VerifiedInstallerBundleComponent parsed;
        if (!HasExactKeys(component, {"path", "size", "sha256", "executable"}) ||
            !ReadString(component, "path", parsed.relative_path) ||
            !ReadUnsigned(component, "size", kMaximumComponentBytes, true,
                          parsed.size) ||
            !ReadString(component, "sha256", parsed.sha256) ||
            !component["executable"].is_boolean() ||
            !IsCanonicalBackendPackRelativePath(parsed.relative_path) ||
            parsed.relative_path.size() > 512 ||
            !IsLowercaseSha256(parsed.sha256) ||
            (!previous_path.empty() && parsed.relative_path <= previous_path) ||
            !seen.insert(FoldBackendPackPath(parsed.relative_path)).second ||
            total_bytes > kMaximumComponentBytes - parsed.size) {
            error = "Signed installer component inventory violates schema 1";
            return false;
        }
        parsed.executable = component["executable"].get<bool>();
        total_bytes += parsed.size;
        previous_path = parsed.relative_path;
        paths.insert(parsed.relative_path);
        if (parsed.executable) executable_paths.insert(parsed.relative_path);
        output.components.push_back(std::move(parsed));
    }
    const std::string suffix = output.platform == "windows" ? ".exe" : "";
    const std::set<std::string> required_executables = {
        "cyxwiz-installer" + suffix,
        "cyxwiz-backend-pack-installer" + suffix};
    const std::set<std::string> required_files = {
        "runtime/trust/trusted-keys.json", "runtime/catalogs/current.json"};
    if (!std::all_of(required_executables.begin(), required_executables.end(),
                     [&](const auto& path) {
                         return paths.contains(path) &&
                             executable_paths.contains(path);
                     }) ||
        !std::all_of(required_files.begin(), required_files.end(),
                     [&](const auto& path) { return paths.contains(path); })) {
        error = "Installer bundle is missing a signed bootstrap entry point";
        return false;
    }
    return true;
}

}  // namespace cyxwiz::runtime
