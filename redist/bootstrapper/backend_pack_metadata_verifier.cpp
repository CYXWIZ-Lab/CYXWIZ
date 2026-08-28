#include "backend_pack_metadata_verifier.h"
#include "backend_pack_hash.h"
#include "backend_pack_path.h"

#include <openssl/evp.h>

#include <algorithm>
#include <cctype>
#include <fstream>
#include <limits>
#include <memory>
#include <set>
#include <utility>

#include <nlohmann/json.hpp>

namespace cyxwiz::runtime {
namespace {

using Json = nlohmann::json;

constexpr std::uintmax_t kMaximumMetadataBytes = 4U * 1024U * 1024U;

bool IsIdentifier(const std::string& value) {
    if (value.empty() || value.size() > 128 ||
        !std::isalnum(static_cast<unsigned char>(value.front()))) {
        return false;
    }
    return std::all_of(value.begin(), value.end(), [](unsigned char c) {
        return std::islower(c) || std::isdigit(c) || c == '.' || c == '_' ||
               c == '-';
    });
}

bool IsVersion(const std::string& value) {
    if (value.empty() || value.size() > 64 ||
        !std::isalnum(static_cast<unsigned char>(value.front()))) {
        return false;
    }
    return std::all_of(value.begin(), value.end(), [](unsigned char c) {
        return std::isalnum(c) || c == '.' || c == '_' || c == '+' ||
               c == '-';
    });
}

bool IsUtc(const std::string& value) {
    if (value.size() != 20 || value[4] != '-' || value[7] != '-' ||
        value[10] != 'T' || value[13] != ':' || value[16] != ':' ||
        value[19] != 'Z') {
        return false;
    }
    for (std::size_t i = 0; i < value.size(); ++i) {
        if (i == 4 || i == 7 || i == 10 || i == 13 || i == 16 || i == 19)
            continue;
        if (!std::isdigit(static_cast<unsigned char>(value[i]))) return false;
    }
    return true;
}

bool HasExactKeys(
    const Json& object,
    std::initializer_list<const char*> expected) {
    if (!object.is_object() || object.size() != expected.size()) return false;
    return std::all_of(expected.begin(), expected.end(),
                       [&](const char* key) { return object.contains(key); });
}

bool ContainsFloat(const Json& value) {
    if (value.is_number_float()) return true;
    if (value.is_array()) {
        return std::any_of(value.begin(), value.end(), ContainsFloat);
    }
    if (value.is_object()) {
        return std::any_of(
            value.begin(), value.end(),
            [](const auto& item) { return ContainsFloat(item); });
    }
    return false;
}

bool ReadDocument(
    const std::filesystem::path& path,
    Json& output,
    std::string& bytes,
    std::string& error) {
    std::error_code filesystem_error;
    const auto status = std::filesystem::symlink_status(path, filesystem_error);
    if (filesystem_error || !std::filesystem::is_regular_file(status) ||
        std::filesystem::is_symlink(status)) {
        error = "Signed metadata is missing or is not a regular file";
        return false;
    }
    const auto size = std::filesystem::file_size(path, filesystem_error);
    if (filesystem_error || size == 0 || size > kMaximumMetadataBytes) {
        error = "Signed metadata is missing, empty, or exceeds 4 MiB";
        return false;
    }
    std::ifstream stream(path, std::ios::binary);
    if (!stream) {
        error = "Cannot open signed metadata: " + path.string();
        return false;
    }
    bytes.resize(static_cast<std::size_t>(size));
    stream.read(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    if (stream.gcount() != static_cast<std::streamsize>(bytes.size()) ||
        stream.peek() != std::char_traits<char>::eof()) {
        error = "Signed metadata changed while it was being read";
        return false;
    }
    try {
        output = Json::parse(bytes);
    } catch (const std::exception& exception) {
        error = std::string("Signed metadata is not valid JSON: ") +
                exception.what();
        return false;
    }
    return true;
}

bool ReadString(
    const Json& object,
    const char* key,
    std::string& output,
    std::string& error) {
    if (!object.contains(key) || !object[key].is_string() ||
        object[key].get_ref<const std::string&>().empty()) {
        error = std::string(key) + " must be a non-empty string";
        return false;
    }
    output = object[key].get<std::string>();
    return true;
}

bool ReadIdentifier(
    const Json& object,
    const char* key,
    std::string& output,
    std::string& error) {
    return ReadString(object, key, output, error) &&
           (IsIdentifier(output) ||
            (error = std::string(key) + " is not a canonical identifier",
             false));
}

bool ReadVersion(
    const Json& object,
    const char* key,
    std::string& output,
    std::string& error) {
    return ReadString(object, key, output, error) &&
           (IsVersion(output) ||
            (error = std::string(key) + " is not a valid version", false));
}

bool ReadSha256(
    const Json& object,
    const char* key,
    std::string& output,
    std::string& error) {
    return ReadString(object, key, output, error) &&
           (IsLowercaseSha256(output) ||
            (error = std::string(key) + " is not a lowercase SHA-256", false));
}

bool ReadUnsigned(
    const Json& object,
    const char* key,
    std::uint64_t& output,
    std::uint64_t minimum,
    std::string& error) {
    if (!object.contains(key) || !object[key].is_number_unsigned()) {
        error = std::string(key) + " must be an unsigned integer";
        return false;
    }
    output = object[key].get<std::uint64_t>();
    if (output < minimum) {
        error = std::string(key) + " is below its minimum";
        return false;
    }
    return true;
}

bool ReadIdentifierArray(
    const Json& value,
    std::vector<std::string>& output,
    std::string& error) {
    if (!value.is_array()) {
        error = "Expected an identifier array";
        return false;
    }
    std::set<std::string> seen;
    output.clear();
    for (const auto& item : value) {
        if (!item.is_string() || !IsIdentifier(item.get<std::string>()) ||
            !seen.insert(item.get<std::string>()).second) {
            error = "Identifier array contains an invalid or duplicate value";
            return false;
        }
        output.push_back(item.get<std::string>());
    }
    return true;
}

bool ReadStringMap(
    const Json& value,
    std::map<std::string, std::string>& output,
    std::string& error) {
    if (!value.is_object()) {
        error = "Expected a string map";
        return false;
    }
    output.clear();
    for (auto iterator = value.begin(); iterator != value.end(); ++iterator) {
        if (!IsIdentifier(iterator.key()) || !iterator.value().is_string() ||
            iterator.value().get_ref<const std::string&>().empty()) {
            error = "String map contains an invalid key or value";
            return false;
        }
        output.emplace(iterator.key(), iterator.value().get<std::string>());
    }
    return true;
}

bool DecodeBase64Url(
    const std::string& encoded,
    std::size_t expected_size,
    std::vector<unsigned char>& output) {
    if (encoded.empty() || encoded.find('=') != std::string::npos ||
        !std::all_of(encoded.begin(), encoded.end(), [](unsigned char c) {
            return std::isalnum(c) || c == '-' || c == '_';
        })) {
        return false;
    }
    std::string padded = encoded;
    std::replace(padded.begin(), padded.end(), '-', '+');
    std::replace(padded.begin(), padded.end(), '_', '/');
    const auto padding = (4 - padded.size() % 4) % 4;
    padded.append(padding, '=');
    output.assign(3 * padded.size() / 4, 0);
    const int decoded = EVP_DecodeBlock(
        output.data(),
        reinterpret_cast<const unsigned char*>(padded.data()),
        static_cast<int>(padded.size()));
    if (decoded < 0 || static_cast<std::size_t>(decoded) < padding) return false;
    output.resize(static_cast<std::size_t>(decoded) - padding);
    return output.size() == expected_size;
}

bool VerifyEd25519(
    const std::vector<unsigned char>& public_key,
    const std::string& payload,
    const std::string& encoded_signature) {
    std::vector<unsigned char> signature;
    if (public_key.size() != 32 ||
        !DecodeBase64Url(encoded_signature, 64, signature)) {
        return false;
    }
    EVP_PKEY* raw_key = EVP_PKEY_new_raw_public_key(
        EVP_PKEY_ED25519, nullptr, public_key.data(), public_key.size());
    if (!raw_key) return false;
    const auto free_key = [](EVP_PKEY* key) { EVP_PKEY_free(key); };
    std::unique_ptr<EVP_PKEY, decltype(free_key)> key(raw_key, free_key);
    EVP_MD_CTX* raw_context = EVP_MD_CTX_new();
    if (!raw_context) return false;
    const auto free_context = [](EVP_MD_CTX* context) {
        EVP_MD_CTX_free(context);
    };
    std::unique_ptr<EVP_MD_CTX, decltype(free_context)> context(
        raw_context, free_context);
    return EVP_DigestVerifyInit(
               context.get(), nullptr, nullptr, nullptr, key.get()) == 1 &&
           EVP_DigestVerify(
               context.get(), signature.data(), signature.size(),
               reinterpret_cast<const unsigned char*>(payload.data()),
               payload.size()) == 1;
}

bool ValidateEnvelope(
    const Json& document,
    const char* expected_kind,
    const BackendPackTrustStore& trust_store,
    const std::optional<std::string>& required_key_id,
    TrustedMetadataRole role,
    const Json*& signed_body,
    std::string& error) {
    if (!HasExactKeys(
            document, {"schema_version", "kind", "signed", "signatures"}) ||
        !document["schema_version"].is_number_unsigned() ||
        document["schema_version"].get<std::uint64_t>() != 1 ||
        !document["kind"].is_string() ||
        document["kind"].get<std::string>() != expected_kind ||
        !document["signed"].is_object() ||
        ContainsFloat(document["signed"]) ||
        !document["signatures"].is_array() ||
        document["signatures"].empty()) {
        error = "Signed envelope violates schema 1";
        return false;
    }
    const std::string canonical = document["signed"].dump();
    std::set<std::string> signature_ids;
    bool verified = false;
    for (const auto& signature : document["signatures"]) {
        std::string key_id;
        std::string algorithm;
        std::string value;
        if (!HasExactKeys(signature, {"key_id", "algorithm", "value"}) ||
            !ReadIdentifier(signature, "key_id", key_id, error) ||
            !ReadString(signature, "algorithm", algorithm, error) ||
            algorithm != "ed25519" ||
            !ReadString(signature, "value", value, error) ||
            value.size() != 86 || !signature_ids.insert(key_id).second) {
            error = "Signature entry violates schema 1";
            return false;
        }
        if (required_key_id && key_id != *required_key_id) continue;
        const auto* key = trust_store.Find(key_id);
        const bool authorized = key && !key->revoked &&
            ((role == TrustedMetadataRole::Catalog && key->catalog) ||
             (role == TrustedMetadataRole::Pack && key->pack) ||
             (role == TrustedMetadataRole::Installer && key->installer));
        if (!authorized) {
            continue;
        }
        verified = VerifyEd25519(key->public_key, canonical, value) || verified;
    }
    if (verified) {
        signed_body = &document["signed"];
        return true;
    }
    error = required_key_id
        ? "No valid trusted signature from the catalog-authorized key"
        : "No valid trusted signature for the required document role";
    return false;
}

std::optional<BackendPackSupportStatus> ParseSupportStatus(
    const Json& value) {
    if (!value.is_string()) return std::nullopt;
    const auto status = value.get<std::string>();
    if (status == "supported") return BackendPackSupportStatus::Supported;
    if (status == "diagnostic") return BackendPackSupportStatus::Diagnostic;
    if (status == "blocked") return BackendPackSupportStatus::Blocked;
    if (status == "revoked") return BackendPackSupportStatus::Revoked;
    return std::nullopt;
}

bool ParseNumericVersion(
    const std::string& value,
    bool allow_wildcard,
    std::vector<unsigned int>& parts,
    bool& wildcard) {
    parts.clear();
    wildcard = false;
    std::size_t begin = 0;
    while (begin < value.size()) {
        const auto end = value.find('.', begin);
        const auto part = value.substr(
            begin, (end == std::string::npos ? value.size() : end) - begin);
        if (allow_wildcard && part == "x" && end == std::string::npos) {
            wildcard = true;
            return !parts.empty();
        }
        if (part.empty() ||
            !std::all_of(part.begin(), part.end(), [](unsigned char c) {
                return std::isdigit(c);
            }))
            return false;
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

int CompareParts(
    const std::vector<unsigned int>& left,
    const std::vector<unsigned int>& right,
    std::size_t count = std::numeric_limits<std::size_t>::max()) {
    const auto size = std::min(
        count, std::max(left.size(), right.size()));
    for (std::size_t i = 0; i < size; ++i) {
        const auto a = i < left.size() ? left[i] : 0;
        const auto b = i < right.size() ? right[i] : 0;
        if (a < b) return -1;
        if (a > b) return 1;
    }
    return 0;
}

bool VersionAtLeast(
    const std::string& current,
    const std::string& minimum) {
    std::vector<unsigned int> current_parts;
    std::vector<unsigned int> minimum_parts;
    bool ignored = false;
    return ParseNumericVersion(current, false, current_parts, ignored) &&
           ParseNumericVersion(minimum, false, minimum_parts, ignored) &&
           CompareParts(current_parts, minimum_parts) >= 0;
}

bool VersionAtMost(
    const std::string& current,
    const std::string& maximum) {
    std::vector<unsigned int> current_parts;
    std::vector<unsigned int> maximum_parts;
    bool ignored = false;
    bool wildcard = false;
    if (!ParseNumericVersion(current, false, current_parts, ignored) ||
        !ParseNumericVersion(maximum, true, maximum_parts, wildcard)) {
        return false;
    }
    if (!wildcard) return CompareParts(current_parts, maximum_parts) <= 0;
    return CompareParts(current_parts, maximum_parts, maximum_parts.size()) <= 0;
}

bool IsAllowedPlatform(const std::string& value) {
    return value == "win64" || value == "linux64" || value == "macos";
}

bool IsAllowedArchitecture(const std::string& value) {
    return value == "x86_64" || value == "arm64";
}

bool IsAllowedBackend(const std::string& value) {
    return value == "cpu" || value == "cuda" || value == "opencl" ||
           value == "oneapi";
}

bool ParseTrustStore(
    const Json& document,
    std::vector<BackendPackTrustStore::Key>& keys,
    std::string& error) {
    if (!HasExactKeys(document, {"schema_version", "keys"}) ||
        !document["schema_version"].is_number_unsigned() ||
        document["schema_version"].get<std::uint64_t>() != 1 ||
        !document["keys"].is_array() || document["keys"].empty()) {
        error = "trusted-keys.json violates schema 1";
        return false;
    }
    std::set<std::string> key_ids;
    for (const auto& entry : document["keys"]) {
        if (!HasExactKeys(
                entry,
                {"key_id", "algorithm", "public_key", "roles", "revoked"})) {
            error = "Trust key entry contains unknown or missing fields";
            return false;
        }
        BackendPackTrustStore::Key key;
        std::string algorithm;
        std::string public_key;
        if (!ReadIdentifier(entry, "key_id", key.key_id, error) ||
            !key_ids.insert(key.key_id).second ||
            !ReadString(entry, "algorithm", algorithm, error) ||
            algorithm != "ed25519" ||
            !ReadString(entry, "public_key", public_key, error) ||
            !DecodeBase64Url(public_key, 32, key.public_key) ||
            !entry["roles"].is_array() || entry["roles"].empty() ||
            !entry["revoked"].is_boolean()) {
            error = "Trust key entry is invalid";
            return false;
        }
        std::set<std::string> roles;
        for (const auto& role : entry["roles"]) {
            if (!role.is_string() ||
                !roles.insert(role.get<std::string>()).second) {
                error = "Trust key roles are invalid or duplicated";
                return false;
            }
            if (role == "catalog") key.catalog = true;
            else if (role == "pack") key.pack = true;
            else if (role == "installer") key.installer = true;
            else {
                error = "Trust key contains an unsupported role";
                return false;
            }
        }
        key.revoked = entry["revoked"].get<bool>();
        keys.push_back(std::move(key));
    }
    return true;
}

}  // namespace

VerifiedBackendPackPayload VerifiedBackendPackManifest::BindExtractedDirectory(
    std::filesystem::path source_directory) const {
    return {runtime_set_id, companion_base_id, backend, pack_id,
            std::move(source_directory), components};
}

std::optional<BackendPackTrustStore> BackendPackTrustStore::Load(
    const std::filesystem::path& path,
    std::string& error) {
    Json document;
    std::string bytes;
    if (!ReadDocument(path, document, bytes, error)) return std::nullopt;
    BackendPackTrustStore output;
    if (!ParseTrustStore(document, output.keys_, error)) return std::nullopt;
    return output;
}

std::optional<BackendPackTrustStore> BackendPackTrustStore::LoadJson(
    std::string_view bytes,
    std::string& error) {
    if (bytes.empty() || bytes.size() > kMaximumMetadataBytes) {
        error = "Embedded trust metadata is empty or exceeds 4 MiB";
        return std::nullopt;
    }
    Json document;
    try {
        document = Json::parse(bytes);
    } catch (const std::exception& exception) {
        error = std::string("Embedded trust metadata is not valid JSON: ") +
            exception.what();
        return std::nullopt;
    }
    BackendPackTrustStore output;
    if (!ParseTrustStore(document, output.keys_, error)) return std::nullopt;
    return output;
}

const BackendPackTrustStore::Key* BackendPackTrustStore::Find(
    const std::string& key_id) const {
    const auto found = std::find_if(
        keys_.begin(), keys_.end(), [&](const Key& key) {
            return key.key_id == key_id;
        });
    return found == keys_.end() ? nullptr : &*found;
}

bool BackendPackTrustStore::VerifySignedDocument(
    const std::filesystem::path& path,
    const std::string& expected_kind,
    TrustedMetadataRole role,
    std::string& canonical_signed_body,
    std::string& error) const {
    canonical_signed_body.clear();
    if (expected_kind.empty()) {
        error = "Expected signed document kind is required";
        return false;
    }
    Json document;
    std::string bytes;
    if (!ReadDocument(path, document, bytes, error)) return false;
    const Json* signed_body = nullptr;
    if (!ValidateEnvelope(
            document, expected_kind.c_str(), *this, std::nullopt, role,
            signed_body, error)) {
        return false;
    }
    canonical_signed_body = signed_body->dump();
    return true;
}

BackendPackMetadataVerifier::BackendPackMetadataVerifier(
    BackendPackTrustStore trust_store,
    std::string client_version,
    std::string platform,
    std::string architecture)
    : trust_store_(std::move(trust_store)),
      client_version_(std::move(client_version)),
      platform_(std::move(platform)),
      architecture_(std::move(architecture)) {}

bool BackendPackMetadataVerifier::VerifyCatalog(
    const std::filesystem::path& catalog_path,
    const std::string& current_utc,
    VerifiedBackendPackCatalog& output,
    std::string& error) const {
    output = {};
    if (!IsUtc(current_utc) || !IsVersion(client_version_)) {
        error = "Current UTC time or client version is invalid";
        return false;
    }
    Json document;
    std::string bytes;
    if (!ReadDocument(catalog_path, document, bytes, error)) return false;
    const Json* signed_body = nullptr;
    if (!ValidateEnvelope(
            document, "cyxwiz-backend-pack-catalog", trust_store_,
            std::nullopt, TrustedMetadataRole::Catalog, signed_body, error)) {
        return false;
    }
    if (!HasExactKeys(
            *signed_body,
            {"catalog_id", "generated_utc", "expires_utc",
             "minimum_client_version", "packs"}) ||
        !ReadIdentifier(*signed_body, "catalog_id", output.catalog_id, error) ||
        !ReadString(*signed_body, "generated_utc", output.generated_utc, error) ||
        !ReadString(*signed_body, "expires_utc", output.expires_utc, error) ||
        !ReadVersion(
            *signed_body, "minimum_client_version",
            output.minimum_client_version, error) ||
        !IsUtc(output.generated_utc) || !IsUtc(output.expires_utc) ||
        output.expires_utc <= output.generated_utc ||
        !(*signed_body)["packs"].is_array()) {
        error = "Signed catalog body violates schema 1";
        return false;
    }
    if (current_utc < output.generated_utc || current_utc >= output.expires_utc) {
        error = "Catalog is not current for the trusted clock";
        return false;
    }
    if (!VersionAtLeast(client_version_, output.minimum_client_version)) {
        error = "Catalog requires a newer CyxWiz client";
        return false;
    }
    std::set<std::string> pack_ids;
    for (const auto& entry : (*signed_body)["packs"]) {
        BackendPackCatalogEntry parsed;
        std::string support;
        if (!HasExactKeys(
                entry,
                {"pack_id", "manifest_url", "manifest_sha256",
                 "signing_key_id", "support_status"}) ||
            !ReadIdentifier(entry, "pack_id", parsed.pack_id, error) ||
            !pack_ids.insert(parsed.pack_id).second ||
            !ReadString(entry, "manifest_url", parsed.manifest_url, error) ||
            parsed.manifest_url.rfind("https://", 0) != 0 ||
            !ReadSha256(
                entry, "manifest_sha256", parsed.manifest_sha256, error) ||
            !ReadIdentifier(
                entry, "signing_key_id", parsed.signing_key_id, error) ||
            !ReadString(entry, "support_status", support, error)) {
            error = "Catalog pack entry violates schema 1";
            return false;
        }
        const auto status = ParseSupportStatus(support);
        const auto* signing_key = trust_store_.Find(parsed.signing_key_id);
        if (!status || !signing_key || signing_key->revoked ||
            !signing_key->pack) {
            error = "Catalog references an invalid, revoked, or untrusted pack key";
            return false;
        }
        parsed.support_status = *status;
        output.packs.push_back(std::move(parsed));
    }
    return true;
}

bool BackendPackMetadataVerifier::VerifyManifest(
    const std::filesystem::path& manifest_path,
    const BackendPackCatalogEntry& catalog_entry,
    VerifiedBackendPackManifest& output,
    std::string& error,
    BackendPackManifestKind expected_kind) const {
    output = {};
    if (catalog_entry.support_status == BackendPackSupportStatus::Blocked ||
        catalog_entry.support_status == BackendPackSupportStatus::Revoked) {
        error = "Catalog policy blocks this backend pack";
        return false;
    }
    Json document;
    std::string bytes;
    if (!ReadDocument(manifest_path, document, bytes, error)) return false;
    std::string manifest_digest;
    if (!Sha256Bytes(bytes, manifest_digest, error) ||
        manifest_digest != catalog_entry.manifest_sha256) {
        error = "Manifest SHA-256 differs from the signed catalog";
        return false;
    }
    const Json* signed_body = nullptr;
    if (!ValidateEnvelope(
            document, "cyxwiz-backend-pack-manifest", trust_store_,
            catalog_entry.signing_key_id, TrustedMetadataRole::Pack,
            signed_body, error)) {
        return false;
    }
    std::string pack_kind;
    if (!HasExactKeys(
            *signed_body,
            {"pack_id", "pack_kind", "backend", "package_version",
             "platform", "architecture", "runtime_set_id", "cyxwiz_release",
             "arrayfire", "companion_base_id", "conflicts", "compatibility",
             "components", "licenses", "archive", "generated_utc"}) ||
        !ReadIdentifier(*signed_body, "pack_id", output.pack_id, error) ||
        output.pack_id != catalog_entry.pack_id ||
        !ReadIdentifier(*signed_body, "backend", output.backend, error) ||
        !IsAllowedBackend(output.backend) ||
        !ReadVersion(
            *signed_body, "package_version", output.package_version, error) ||
        !ReadString(*signed_body, "platform", output.platform, error) ||
        !IsAllowedPlatform(output.platform) || output.platform != platform_ ||
        !ReadString(
            *signed_body, "architecture", output.architecture, error) ||
        !IsAllowedArchitecture(output.architecture) ||
        output.architecture != architecture_ ||
        !ReadIdentifier(
            *signed_body, "runtime_set_id", output.runtime_set_id, error) ||
        !ReadString(*signed_body, "generated_utc", output.generated_utc, error) ||
        !IsUtc(output.generated_utc) ||
        !ReadString(*signed_body, "pack_kind", pack_kind, error)) {
        error = "Signed manifest identity violates schema 1 or this client target";
        return false;
    }
    if (expected_kind == BackendPackManifestKind::BackendPack) {
        if (pack_kind != "backend_pack" || output.backend == "cpu" ||
            !ReadIdentifier(
                *signed_body, "companion_base_id",
                output.companion_base_id, error) ||
            output.companion_base_id == output.pack_id) {
            error = "Signed manifest is not an optional backend pack";
            return false;
        }
        output.kind = BackendPackManifestKind::BackendPack;
    } else {
        if (pack_kind != "base" || output.backend != "cpu" ||
            !(*signed_body)["companion_base_id"].is_null()) {
            error = "Signed manifest is not a CPU base pack";
            return false;
        }
        output.kind = BackendPackManifestKind::Base;
        output.companion_base_id.clear();
    }

    const auto& release = (*signed_body)["cyxwiz_release"];
    if (!HasExactKeys(release, {"minimum", "maximum"}) ||
        !ReadVersion(
            release, "minimum", output.minimum_cyxwiz_release, error) ||
        !ReadVersion(
            release, "maximum", output.maximum_cyxwiz_release, error) ||
        !VersionAtLeast(client_version_, output.minimum_cyxwiz_release) ||
        !VersionAtMost(client_version_, output.maximum_cyxwiz_release)) {
        error = "Backend pack does not support this CyxWiz release";
        return false;
    }
    const auto& arrayfire = (*signed_body)["arrayfire"];
    if (!HasExactKeys(arrayfire, {"version", "abi"}) ||
        !ReadVersion(arrayfire, "version", output.arrayfire_version, error) ||
        !ReadIdentifier(arrayfire, "abi", output.arrayfire_abi, error)) {
        error = "ArrayFire manifest identity is invalid";
        return false;
    }
    if (!ReadIdentifierArray(
            (*signed_body)["conflicts"], output.conflicts, error) ||
        std::find(output.conflicts.begin(), output.conflicts.end(),
                  output.pack_id) != output.conflicts.end()) {
        error = "Manifest conflicts are invalid";
        return false;
    }

    const auto& compatibility = (*signed_body)["compatibility"];
    if (!HasExactKeys(
            compatibility,
            {"device_kinds", "cpu_features", "provider_types",
             "minimum_driver_versions", "tested_driver_ranges",
             "minimum_identity_confidence", "recommendation_targets",
             "operation_matrix_id", "training_scope", "support_status"}) ||
        !ReadIdentifierArray(
            compatibility["device_kinds"],
            output.compatibility.device_kinds, error) ||
        !ReadIdentifierArray(
            compatibility["cpu_features"],
            output.compatibility.cpu_features, error) ||
        !ReadIdentifierArray(
            compatibility["provider_types"],
            output.compatibility.provider_types, error) ||
        !ReadStringMap(
            compatibility["minimum_driver_versions"],
            output.compatibility.minimum_driver_versions, error) ||
        !ReadStringMap(
            compatibility["tested_driver_ranges"],
            output.compatibility.tested_driver_ranges, error) ||
        !ReadIdentifier(
            compatibility, "minimum_identity_confidence",
            output.compatibility.minimum_identity_confidence, error) ||
        !ReadIdentifierArray(
            compatibility["recommendation_targets"],
            output.compatibility.recommendation_targets, error) ||
        !ReadIdentifier(
            compatibility, "operation_matrix_id",
            output.compatibility.operation_matrix_id, error) ||
        !ReadIdentifierArray(
            compatibility["training_scope"],
            output.compatibility.training_scope, error)) {
        error = "Manifest compatibility contract is invalid";
        return false;
    }
    static const std::set<std::string> kConfidence = {
        "unknown", "backend_local", "provider_reported", "stable_hardware"};
    if (!kConfidence.contains(
            output.compatibility.minimum_identity_confidence)) {
        error = "Manifest identity-confidence policy is invalid";
        return false;
    }
    for (const auto& backend : output.compatibility.recommendation_targets) {
        if (!IsAllowedBackend(backend)) {
            error = "Manifest recommends an unknown backend";
            return false;
        }
    }
    const auto support = ParseSupportStatus(compatibility["support_status"]);
    if (!support || *support != catalog_entry.support_status) {
        error = "Manifest and catalog support policy differ";
        return false;
    }
    output.compatibility.support_status = *support;

    const auto& components = (*signed_body)["components"];
    if (!components.is_array() || components.empty()) {
        error = "Manifest components must not be empty";
        return false;
    }
    std::set<std::string> component_paths;
    std::set<std::string> folded_paths;
    for (const auto& component : components) {
        VerifiedPackComponent parsed;
        std::string source;
        if (!HasExactKeys(
                component,
                {"path", "size", "sha256", "source", "executable"}) ||
            !ReadString(component, "path", parsed.relative_path, error) ||
            !IsCanonicalBackendPackRelativePath(parsed.relative_path) ||
            !folded_paths.insert(
                FoldBackendPackPath(parsed.relative_path)).second ||
            !ReadUnsigned(component, "size", parsed.size, 0, error) ||
            !ReadSha256(component, "sha256", parsed.sha256, error) ||
            !ReadIdentifier(component, "source", source, error) ||
            !component["executable"].is_boolean()) {
            error = "Manifest component inventory is invalid";
            return false;
        }
        component_paths.insert(parsed.relative_path);
        output.components.push_back(std::move(parsed));
    }

    const auto& licenses = (*signed_body)["licenses"];
    if (!licenses.is_array() || licenses.empty()) {
        error = "Manifest licenses must not be empty";
        return false;
    }
    for (const auto& license : licenses) {
        std::string component;
        std::string path;
        if (!HasExactKeys(license, {"component", "path"}) ||
            !ReadIdentifier(license, "component", component, error) ||
            !ReadString(license, "path", path, error) ||
            !IsCanonicalBackendPackRelativePath(path) ||
            !component_paths.contains(path)) {
            error = "Manifest license inventory is invalid";
            return false;
        }
        if (std::find(
                output.licenses.begin(), output.licenses.end(), component) ==
            output.licenses.end()) {
            output.licenses.push_back(std::move(component));
        }
    }

    const auto& archive = (*signed_body)["archive"];
    if (!HasExactKeys(archive, {"file_name", "size", "sha256"}) ||
        !ReadString(archive, "file_name", output.archive.file_name, error) ||
        !IsCanonicalBackendPackRelativePath(output.archive.file_name) ||
        output.archive.file_name.find('/') != std::string::npos ||
        !ReadUnsigned(archive, "size", output.archive.size, 1, error) ||
        !ReadSha256(archive, "sha256", output.archive.sha256, error)) {
        error = "Manifest archive identity is invalid";
        return false;
    }
    return true;
}

const char* BackendPackSupportStatusName(BackendPackSupportStatus status) {
    switch (status) {
        case BackendPackSupportStatus::Supported: return "supported";
        case BackendPackSupportStatus::Diagnostic: return "diagnostic";
        case BackendPackSupportStatus::Blocked: return "blocked";
        case BackendPackSupportStatus::Revoked: return "revoked";
        default: return "unknown";
    }
}

}  // namespace cyxwiz::runtime
