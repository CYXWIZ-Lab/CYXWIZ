#include "backend_pack_metadata_verifier.h"

#include <openssl/evp.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

namespace {

using Json = nlohmann::json;
using namespace cyxwiz::runtime;

class TemporaryDirectory {
public:
    TemporaryDirectory() {
        root_ = std::filesystem::temp_directory_path() /
            ("cyxwiz-pack-metadata-" + std::to_string(
                std::chrono::steady_clock::now().time_since_epoch().count()));
        std::filesystem::create_directories(root_);
    }
    ~TemporaryDirectory() {
        std::error_code error;
        std::filesystem::remove_all(root_, error);
    }
    const std::filesystem::path& Path() const { return root_; }

private:
    std::filesystem::path root_;
};

struct KeyPair {
    explicit KeyPair(EVP_PKEY* value) : value(value, EVP_PKEY_free) {}
    std::unique_ptr<EVP_PKEY, decltype(&EVP_PKEY_free)> value;
};

KeyPair GenerateKey() {
    EVP_PKEY_CTX* raw_context = EVP_PKEY_CTX_new_id(EVP_PKEY_ED25519, nullptr);
    if (!raw_context) return KeyPair(nullptr);
    const auto free_context = [](EVP_PKEY_CTX* context) {
        EVP_PKEY_CTX_free(context);
    };
    std::unique_ptr<EVP_PKEY_CTX, decltype(free_context)> context(
        raw_context, free_context);
    EVP_PKEY* key = nullptr;
    if (EVP_PKEY_keygen_init(context.get()) != 1 ||
        EVP_PKEY_keygen(context.get(), &key) != 1) {
        return KeyPair(nullptr);
    }
    return KeyPair(key);
}

std::string Base64Url(const unsigned char* bytes, std::size_t size) {
    std::string output(4 * ((size + 2) / 3), '\0');
    const auto written = EVP_EncodeBlock(
        reinterpret_cast<unsigned char*>(output.data()), bytes,
        static_cast<int>(size));
    output.resize(static_cast<std::size_t>(written));
    while (!output.empty() && output.back() == '=') output.pop_back();
    std::replace(output.begin(), output.end(), '+', '-');
    std::replace(output.begin(), output.end(), '/', '_');
    return output;
}

std::string PublicKey(const KeyPair& key) {
    std::array<unsigned char, 32> bytes{};
    std::size_t size = bytes.size();
    if (!key.value || EVP_PKEY_get_raw_public_key(
            key.value.get(), bytes.data(), &size) != 1 ||
        size != bytes.size()) {
        return {};
    }
    return Base64Url(bytes.data(), bytes.size());
}

std::string Sign(const Json& signed_body, const KeyPair& key) {
    const auto payload = signed_body.dump();
    EVP_MD_CTX* raw_context = EVP_MD_CTX_new();
    if (!raw_context) return {};
    const auto free_context = [](EVP_MD_CTX* context) {
        EVP_MD_CTX_free(context);
    };
    std::unique_ptr<EVP_MD_CTX, decltype(free_context)> context(
        raw_context, free_context);
    std::array<unsigned char, 64> signature{};
    std::size_t size = signature.size();
    if (!key.value || EVP_DigestSignInit(
            context.get(), nullptr, nullptr, nullptr, key.value.get()) != 1 ||
        EVP_DigestSign(
            context.get(), signature.data(), &size,
            reinterpret_cast<const unsigned char*>(payload.data()),
            payload.size()) != 1 || size != signature.size()) {
        return {};
    }
    return Base64Url(signature.data(), signature.size());
}

std::string Sha256(const std::string& bytes) {
    std::array<unsigned char, 32> digest{};
    unsigned int size = 0;
    if (EVP_Digest(
            bytes.data(), bytes.size(), digest.data(), &size,
            EVP_sha256(), nullptr) != 1 || size != digest.size()) {
        return {};
    }
    static constexpr char kHex[] = "0123456789abcdef";
    std::string output(64, '0');
    for (std::size_t i = 0; i < digest.size(); ++i) {
        output[2 * i] = kHex[digest[i] >> 4];
        output[2 * i + 1] = kHex[digest[i] & 0x0f];
    }
    return output;
}

std::string WriteJson(const std::filesystem::path& path, const Json& value) {
    const auto bytes = value.dump(2) + "\n";
    std::ofstream stream(path, std::ios::binary | std::ios::trunc);
    stream << bytes;
    return bytes;
}

Json Envelope(
    const char* kind,
    Json signed_body,
    const char* key_id,
    const KeyPair& key) {
    const auto signature = Sign(signed_body, key);
    return {
        {"schema_version", std::uint64_t{1}},
        {"kind", kind},
        {"signed", std::move(signed_body)},
        {"signatures", Json::array({{
            {"key_id", key_id},
            {"algorithm", "ed25519"},
            {"value", signature}}})}};
}

Json Manifest(const KeyPair& pack_key) {
    Json signed_body = {
        {"pack_id", "opencl-v1"},
        {"pack_kind", "backend_pack"},
        {"backend", "opencl"},
        {"package_version", "1.0.0"},
        {"platform", "win64"},
        {"architecture", "x86_64"},
        {"runtime_set_id", "set-v1"},
        {"cyxwiz_release", {{"minimum", "0.2.0"}, {"maximum", "0.2.x"}}},
        {"arrayfire", {{"version", "3.10.0"}, {"abi", "arrayfire-3.10"}}},
        {"companion_base_id", "base-v1"},
        {"conflicts", Json::array()},
        {"compatibility", {
            {"device_kinds", Json::array({"gpu"})},
            {"cpu_features", Json::array()},
            {"provider_types", Json::array({"opencl-icd"})},
            {"minimum_driver_versions", {{"intel", "31.0.101.2115"}}},
            {"tested_driver_ranges", {{"intel", ">=31.0.101.2115"}}},
            {"minimum_identity_confidence", "stable_hardware"},
            {"recommendation_targets", Json::array({"cpu"})},
            {"operation_matrix_id", "matrix-v1"},
            {"training_scope", Json::array({"dense"})},
            {"support_status", "supported"}}},
        {"components", Json::array({{
            {"path", "runtime/afopencl.dll"},
            {"size", std::uint64_t{1}},
            {"sha256", "6e340b9cffb37a989ca544e6bb780a2c78901d3fb33738768511a30617afa01d"},
            {"source", "arrayfire"},
            {"executable", true}}, {
            {"path", "THIRD_PARTY_LICENSES/ArrayFire/LICENSE.txt"},
            {"size", std::uint64_t{1}},
            {"sha256", "6e340b9cffb37a989ca544e6bb780a2c78901d3fb33738768511a30617afa01d"},
            {"source", "arrayfire-license"},
            {"executable", false}}})},
        {"licenses", Json::array({{
            {"component", "arrayfire"},
            {"path", "THIRD_PARTY_LICENSES/ArrayFire/LICENSE.txt"}}})},
        {"archive", {
            {"file_name", "opencl-v1.zip"},
            {"size", std::uint64_t{4096}},
            {"sha256", std::string(64, 'a')}}},
        {"generated_utc", "2026-08-13T20:00:00Z"}};
    return Envelope(
        "cyxwiz-backend-pack-manifest", std::move(signed_body),
        "pack-2026", pack_key);
}

Json BaseManifest(const KeyPair& pack_key) {
    auto signed_body = Manifest(pack_key)["signed"];
    signed_body["pack_id"] = "base-v1";
    signed_body["pack_kind"] = "base";
    signed_body["backend"] = "cpu";
    signed_body["companion_base_id"] = nullptr;
    signed_body["compatibility"]["device_kinds"] = Json::array({"cpu"});
    signed_body["compatibility"]["provider_types"] = Json::array();
    signed_body["compatibility"]["minimum_identity_confidence"] =
        "backend_local";
    signed_body["archive"]["file_name"] = "base-v1.zip";
    return Envelope(
        "cyxwiz-backend-pack-manifest", std::move(signed_body),
        "pack-2026", pack_key);
}

Json Catalog(
    const KeyPair& catalog_key,
    const std::string& manifest_sha256,
    const std::string& support_status = "supported",
    const std::string& pack_id = "opencl-v1") {
    Json signed_body = {
        {"catalog_id", "production-2026-08"},
        {"generated_utc", "2026-08-13T20:00:00Z"},
        {"expires_utc", "2026-09-13T20:00:00Z"},
        {"minimum_client_version", "0.2.0"},
        {"packs", Json::array({{
            {"pack_id", pack_id},
            {"manifest_url", "https://downloads.cyxwiz.com/" + pack_id + ".json"},
            {"manifest_sha256", manifest_sha256},
            {"signing_key_id", "pack-2026"},
            {"support_status", support_status}}})}};
    return Envelope(
        "cyxwiz-backend-pack-catalog", std::move(signed_body),
        "catalog-2026", catalog_key);
}

Json TrustRoot(
    const KeyPair& catalog_key,
    const KeyPair& pack_key,
    bool revoke_pack = false) {
    return {
        {"schema_version", std::uint64_t{1}},
        {"keys", Json::array({{
            {"key_id", "catalog-2026"},
            {"algorithm", "ed25519"},
            {"public_key", PublicKey(catalog_key)},
            {"roles", Json::array({"catalog"})},
            {"revoked", false}}, {
            {"key_id", "pack-2026"},
            {"algorithm", "ed25519"},
            {"public_key", PublicKey(pack_key)},
            {"roles", Json::array({"pack"})},
            {"revoked", revoke_pack}}})}};
}

bool Expect(bool condition, const char* message) {
    if (!condition) std::cerr << message << '\n';
    return condition;
}

}  // namespace

int main() {
    TemporaryDirectory temporary;
    const auto catalog_key = GenerateKey();
    const auto pack_key = GenerateKey();
    if (!Expect(catalog_key.value != nullptr && pack_key.value != nullptr,
                "key generation failed")) return 1;

    const auto trust_path = temporary.Path() / "trusted-keys.json";
    const auto manifest_path = temporary.Path() / "manifest.json";
    const auto catalog_path = temporary.Path() / "catalog.json";
    WriteJson(trust_path, TrustRoot(catalog_key, pack_key));
    const auto manifest_bytes = WriteJson(manifest_path, Manifest(pack_key));
    WriteJson(catalog_path, Catalog(catalog_key, Sha256(manifest_bytes)));

    std::string error;
    const auto trust_json = TrustRoot(catalog_key, pack_key).dump();
    auto memory_trust = BackendPackTrustStore::LoadJson(trust_json, error);
    if (!Expect(
            memory_trust.has_value() &&
                memory_trust->Find("catalog-2026") != nullptr,
            error.c_str()) ||
        !Expect(
            !BackendPackTrustStore::LoadJson("", error).has_value(),
            "empty in-memory trust was accepted")) {
        return 1;
    }
    auto trust = BackendPackTrustStore::Load(trust_path, error);
    if (!Expect(trust.has_value(), error.c_str())) return 1;
    BackendPackMetadataVerifier verifier(
        std::move(*trust), "0.2.0", "win64", "x86_64");
    VerifiedBackendPackCatalog catalog;
    if (!Expect(
            verifier.VerifyCatalog(
                catalog_path, "2026-08-14T12:00:00Z", catalog, error),
            error.c_str()) ||
        !Expect(catalog.packs.size() == 1, "catalog pack count differs")) {
        return 1;
    }
    VerifiedBackendPackManifest manifest;
    if (!Expect(
            verifier.VerifyManifest(
                manifest_path, catalog.packs.front(), manifest, error),
            error.c_str()) ||
        !Expect(manifest.pack_id == "opencl-v1", "manifest ID differs") ||
        !Expect(manifest.components.size() == 2, "component count differs")) {
        return 1;
    }
    const auto payload = manifest.BindExtractedDirectory(temporary.Path());
    if (!Expect(payload.pack_id == "opencl-v1", "bound payload differs"))
        return 1;

    const auto base_manifest_path = temporary.Path() / "base-manifest.json";
    const auto base_catalog_path = temporary.Path() / "base-catalog.json";
    const auto base_manifest_bytes = WriteJson(
        base_manifest_path, BaseManifest(pack_key));
    WriteJson(
        base_catalog_path,
        Catalog(
            catalog_key, Sha256(base_manifest_bytes), "supported",
            "base-v1"));
    VerifiedBackendPackCatalog base_catalog;
    if (!Expect(
            verifier.VerifyCatalog(
                base_catalog_path, "2026-08-14T12:00:00Z",
                base_catalog, error),
            error.c_str())) {
        return 1;
    }
    VerifiedBackendPackManifest base_manifest;
    if (!Expect(
            verifier.VerifyManifest(
                base_manifest_path, base_catalog.packs.front(),
                base_manifest, error, BackendPackManifestKind::Base),
            error.c_str()) ||
        !Expect(
            base_manifest.kind == BackendPackManifestKind::Base &&
                base_manifest.backend == "cpu" &&
                base_manifest.companion_base_id.empty(),
            "base manifest identity differs") ||
        !Expect(
            !verifier.VerifyManifest(
                base_manifest_path, base_catalog.packs.front(),
                base_manifest, error),
            "optional-pack verification accepted a base manifest")) {
        return 1;
    }

    auto tampered = Manifest(pack_key);
    tampered["signed"]["backend"] = "cuda";
    WriteJson(manifest_path, tampered);
    if (!Expect(
            !verifier.VerifyManifest(
                manifest_path, catalog.packs.front(), manifest, error),
            "tampered manifest hash was accepted")) return 1;

    auto widened = Manifest(pack_key);
    auto widened_body = widened["signed"];
    widened_body["post_install_script"] = "setup.cmd";
    widened = Envelope(
        "cyxwiz-backend-pack-manifest", std::move(widened_body),
        "pack-2026", pack_key);
    const auto widened_bytes = WriteJson(manifest_path, widened);
    WriteJson(catalog_path, Catalog(catalog_key, Sha256(widened_bytes)));
    if (!Expect(
            verifier.VerifyCatalog(
                catalog_path, "2026-08-14T12:00:00Z", catalog, error),
            error.c_str()) ||
        !Expect(
            !verifier.VerifyManifest(
                manifest_path, catalog.packs.front(), manifest, error),
            "signed manifest with an unknown field was accepted")) {
        return 1;
    }

    auto altered_catalog = Catalog(catalog_key, Sha256(widened_bytes));
    altered_catalog["signed"]["expires_utc"] = "2027-09-13T20:00:00Z";
    WriteJson(catalog_path, altered_catalog);
    if (!Expect(
            !verifier.VerifyCatalog(
                catalog_path, "2026-08-14T12:00:00Z", catalog, error),
            "catalog with an invalid signature was accepted")) return 1;

    auto duplicate_signature = Catalog(catalog_key, Sha256(widened_bytes));
    duplicate_signature["signatures"].push_back(
        duplicate_signature["signatures"].front());
    WriteJson(catalog_path, duplicate_signature);
    if (!Expect(
            !verifier.VerifyCatalog(
                catalog_path, "2026-08-14T12:00:00Z", catalog, error),
            "catalog with duplicate signatures was accepted")) return 1;

    WriteJson(catalog_path, Catalog(catalog_key, Sha256(widened_bytes)));

    WriteJson(trust_path, TrustRoot(catalog_key, pack_key, true));
    trust = BackendPackTrustStore::Load(trust_path, error);
    if (!Expect(trust.has_value(), error.c_str())) return 1;
    BackendPackMetadataVerifier revoked_verifier(
        std::move(*trust), "0.2.0", "win64", "x86_64");
    if (!Expect(
            !revoked_verifier.VerifyCatalog(
                catalog_path, "2026-08-14T12:00:00Z", catalog, error),
            "catalog referencing a revoked pack key was accepted")) return 1;

    auto future_pack = Manifest(pack_key);
    future_pack["signed"]["cyxwiz_release"] = {
        {"minimum", "0.3.0"}, {"maximum", "0.3.x"}};
    auto future_body = future_pack["signed"];
    future_pack = Envelope(
        "cyxwiz-backend-pack-manifest", std::move(future_body),
        "pack-2026", pack_key);
    const auto future_bytes = WriteJson(manifest_path, future_pack);
    WriteJson(catalog_path, Catalog(catalog_key, Sha256(future_bytes)));
    WriteJson(trust_path, TrustRoot(catalog_key, pack_key));
    trust = BackendPackTrustStore::Load(trust_path, error);
    BackendPackMetadataVerifier downgrade_verifier(
        std::move(*trust), "0.2.0", "win64", "x86_64");
    if (!Expect(
            downgrade_verifier.VerifyCatalog(
                catalog_path, "2026-08-14T12:00:00Z", catalog, error),
            error.c_str()) ||
        !Expect(
            !downgrade_verifier.VerifyManifest(
                manifest_path, catalog.packs.front(), manifest, error),
            "a pack for a newer CyxWiz release was accepted after downgrade")) {
        return 1;
    }

    WriteJson(trust_path, TrustRoot(catalog_key, pack_key));
    trust = BackendPackTrustStore::Load(trust_path, error);
    BackendPackMetadataVerifier policy_verifier(
        std::move(*trust), "0.2.0", "win64", "x86_64");
    if (!Expect(
            !policy_verifier.VerifyCatalog(
                catalog_path, "2026-10-14T12:00:00Z", catalog, error),
            "expired catalog was accepted")) return 1;

    std::cout << "backend pack metadata verifier contract tests passed\n";
    return 0;
}
