#include "backend_pack_metadata_verifier.h"
#include "backend_pack_metadata_limits.h"
#include "backend_pack_platform.h"

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
    const KeyPair& key,
    std::uint64_t schema_version = 1) {
    const auto signature = Sign(signed_body, key);
    return {
        {"schema_version", schema_version},
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
            {"path", CurrentArrayFireBackendPluginRelativePath("opencl")},
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

// Manifest schema 2: a plugin pack has no backend, ArrayFire or device fields.
Json PluginBody(const KeyPair& pack_key) {
    auto signed_body = Manifest(pack_key)["signed"];
    for (const char* name : {"backend", "runtime_set_id", "arrayfire", "compatibility"})
        signed_body.erase(name);
    signed_body["pack_id"] = "mujoco-v1";
    signed_body["pack_kind"] = "plugin_pack";
    signed_body["plugin_id"] = "com.cyxwiz.simulation.mujoco";
    signed_body["support_status"] = "supported";
    signed_body["archive"]["file_name"] = "mujoco-v1.zip";
    return signed_body;
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

    auto foreign_body = BaseManifest(pack_key)["signed"];
    foreign_body["platform"] = "macos";
    foreign_body["architecture"] = "arm64";
    const auto foreign_path = temporary.Path() / "foreign-base.json";
    const auto foreign_bytes = WriteJson(foreign_path, Envelope(
        "cyxwiz-backend-pack-manifest", foreign_body, "pack-2026", pack_key));
    auto foreign_entry = base_catalog.packs.front();
    foreign_entry.manifest_sha256 = Sha256(foreign_bytes);
    if (!Expect(!verifier.VerifyManifest(
                    foreign_path, foreign_entry, base_manifest, error,
                    BackendPackManifestKind::Base),
                "delivery accepted a foreign target") ||
        !Expect(verifier.VerifyManifest(
                    foreign_path, foreign_entry, base_manifest, error,
                    BackendPackManifestKind::Base,
                    BackendPackManifestTargetScope::CatalogDiscovery),
                error.c_str()) ||
        !Expect(base_manifest.platform == "macos" &&
                    base_manifest.architecture == "arm64",
                "catalog discovery lost authenticated target identity")) return 1;
    foreign_entry.manifest_sha256 = std::string(64, '0');
    if (!Expect(!verifier.VerifyManifest(
                    foreign_path, foreign_entry, base_manifest, error,
                    BackendPackManifestKind::Base,
                    BackendPackManifestTargetScope::CatalogDiscovery),
                "catalog discovery bypassed manifest hash verification")) return 1;
    auto forged = Envelope("cyxwiz-backend-pack-manifest", foreign_body,
                           "pack-2026", pack_key);
    forged["signed"]["package_version"] = "forged";
    foreign_entry.manifest_sha256 = Sha256(WriteJson(foreign_path, forged));
    if (!Expect(!verifier.VerifyManifest(
                    foreign_path, foreign_entry, base_manifest, error,
                    BackendPackManifestKind::Base,
                    BackendPackManifestTargetScope::CatalogDiscovery),
                "catalog discovery accepted an invalid signature")) return 1;

    auto large_body = Manifest(pack_key)["signed"];
    auto& large_components = large_body["components"];
    for (std::uint64_t index = 0; index < 18000; ++index) {
        large_components.push_back({
            {"path", "python/site-packages/package_" +
                 std::to_string(index) +
                 "/module_with_a_bounded_inventory_name.py"},
            {"size", std::uint64_t{1}},
            {"sha256", "6e340b9cffb37a989ca544e6bb780a2c78901d3fb33738768511a30617afa01d"},
            {"source", "python"},
            {"executable", false}});
    }
    const auto large_manifest_path = temporary.Path() / "large-manifest.json";
    const auto large_catalog_path = temporary.Path() / "large-catalog.json";
    const auto large_manifest_bytes = WriteJson(
        large_manifest_path,
        Envelope(
            "cyxwiz-backend-pack-manifest", std::move(large_body),
            "pack-2026", pack_key));
    if (!Expect(
            large_manifest_bytes.size() > 4U * 1024U * 1024U &&
                large_manifest_bytes.size() <=
                    kMaximumBackendPackMetadataBytes,
            "large manifest does not exercise the shared metadata bound")) {
        return 1;
    }
    WriteJson(
        large_catalog_path,
        Catalog(catalog_key, Sha256(large_manifest_bytes)));
    VerifiedBackendPackCatalog large_catalog;
    if (!Expect(
            verifier.VerifyCatalog(
                large_catalog_path, "2026-08-14T12:00:00Z",
                large_catalog, error),
            error.c_str()) ||
        !Expect(
            verifier.VerifyManifest(
                large_manifest_path, large_catalog.packs.front(),
                manifest, error),
            error.c_str())) {
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

    if (!Expect(verifier.VerifyInstalledManifest(manifest_path, "opencl-v1",
                    BackendPackManifestKind::BackendPack, manifest, error),
                "Historical signed identity must survive an installer version change") ||
        !Expect(!verifier.VerifyInstalledManifest(manifest_path, "wrong-id",
                    BackendPackManifestKind::BackendPack, manifest, error),
                "Historical evidence must bind the active package ID") ||
        !Expect(!revoked_verifier.VerifyInstalledManifest(manifest_path, "opencl-v1",
                    BackendPackManifestKind::BackendPack, manifest, error),
                "Historical evidence must honor current signer revocations")) return 1;

    auto historical_body = future_pack["signed"];
    historical_body["architecture"] = "arm64";
    WriteJson(manifest_path, Envelope("cyxwiz-backend-pack-manifest",
        historical_body, "pack-2026", pack_key));
    if (!Expect(!verifier.VerifyInstalledManifest(manifest_path, "opencl-v1",
                    BackendPackManifestKind::BackendPack, manifest, error),
                "Historical evidence must match the local target")) return 1;
    future_pack["signed"]["package_version"] = "9.0.0";
    WriteJson(manifest_path, future_pack);
    if (!Expect(!verifier.VerifyInstalledManifest(manifest_path, "opencl-v1",
                    BackendPackManifestKind::BackendPack, manifest, error),
                "Historical evidence must reject tampering without a catalog hash")) return 1;

    WriteJson(trust_path, TrustRoot(catalog_key, pack_key));
    trust = BackendPackTrustStore::Load(trust_path, error);
    BackendPackMetadataVerifier policy_verifier(
        std::move(*trust), "0.2.0", "win64", "x86_64");
    if (!Expect(
            !policy_verifier.VerifyCatalog(
                catalog_path, "2026-10-14T12:00:00Z", catalog, error),
            "expired catalog was accepted")) return 1;

    // Plugin packs: schema 2 only, verified only when a plugin pack is expected.
    {
        BackendPackMetadataVerifier plugin_verifier(
            *BackendPackTrustStore::Load(trust_path, error), "0.2.0", "win64", "x86_64");
        const auto write_plugin = [&](const std::filesystem::path& path, const Json& document,
                                      const char* support) {
            const auto bytes = WriteJson(path, document);
            const auto catalog_file = temporary.Path() / "plugin-catalog.json";
            WriteJson(catalog_file, Catalog(catalog_key, Sha256(bytes), support,
                                            document["signed"]["pack_id"].get<std::string>()));
            VerifiedBackendPackCatalog parsed;
            if (!plugin_verifier.VerifyCatalog(catalog_file, "2026-08-14T12:00:00Z", parsed, error))
                return BackendPackCatalogEntry{};
            return parsed.packs.front();
        };
        const auto plugin_path = temporary.Path() / "plugin-manifest.json";
        const auto plugin_document = Envelope("cyxwiz-backend-pack-manifest", PluginBody(pack_key),
                                              "pack-2026", pack_key, 2);
        const auto plugin_entry = write_plugin(plugin_path, plugin_document, "supported");
        VerifiedBackendPackManifest plugin;
        if (!Expect(plugin_verifier.VerifyManifest(plugin_path, plugin_entry, plugin, error,
                                                   BackendPackManifestKind::PluginPack),
                    error.c_str()) ||
            !Expect(plugin.kind == BackendPackManifestKind::PluginPack &&
                        plugin.plugin_id == "com.cyxwiz.simulation.mujoco" &&
                        plugin.companion_base_id == "base-v1" && plugin.backend.empty() &&
                        plugin.runtime_set_id.empty() && plugin.arrayfire_abi.empty() &&
                        plugin.compatibility.support_status == BackendPackSupportStatus::Supported,
                    "plugin manifest identity differs") ||
            !Expect(plugin.BindExtractedDirectory(temporary.Path()).plugin_id ==
                        "com.cyxwiz.simulation.mujoco",
                    "bound plugin payload lost its plugin ID") ||
            !Expect(!plugin_verifier.VerifyManifest(plugin_path, plugin_entry, plugin, error),
                    "backend-pack verification accepted a plugin manifest") ||
            !Expect(!plugin_verifier.VerifyManifest(plugin_path, plugin_entry, plugin, error,
                                                    BackendPackManifestKind::Base),
                    "base verification accepted a plugin manifest") ||
            !Expect(plugin_verifier.VerifyInstalledManifest(plugin_path, "mujoco-v1",
                                                            BackendPackManifestKind::PluginPack,
                                                            plugin, error),
                    "installed plugin evidence was rejected"))
            return 1;

        // A schema-1 backend manifest is never a plugin pack.
        const auto backend_path = temporary.Path() / "backend-as-plugin.json";
        const auto backend_entry = write_plugin(backend_path, Manifest(pack_key), "supported");
        if (!Expect(!plugin_verifier.VerifyManifest(backend_path, backend_entry, plugin, error,
                                                    BackendPackManifestKind::PluginPack),
                    "plugin verification accepted a schema-1 backend manifest"))
            return 1;

        // Schema 2 fails closed on ArrayFire fields and on a schema-1 envelope.
        auto with_arrayfire = PluginBody(pack_key);
        with_arrayfire["arrayfire"] = {{"version", "3.10.0"}, {"abi", "arrayfire-3.10"}};
        const auto bad_path = temporary.Path() / "plugin-bad.json";
        auto bad_entry = write_plugin(
            bad_path, Envelope("cyxwiz-backend-pack-manifest", with_arrayfire, "pack-2026", pack_key, 2),
            "supported");
        if (!Expect(!plugin_verifier.VerifyManifest(bad_path, bad_entry, plugin, error,
                                                    BackendPackManifestKind::PluginPack),
                    "a plugin manifest with ArrayFire identity was accepted"))
            return 1;
        bad_entry = write_plugin(
            bad_path, Envelope("cyxwiz-backend-pack-manifest", PluginBody(pack_key), "pack-2026", pack_key, 1),
            "supported");
        if (!Expect(!plugin_verifier.VerifyManifest(bad_path, bad_entry, plugin, error,
                                                    BackendPackManifestKind::PluginPack),
                    "a plugin body in a schema-1 envelope was accepted"))
            return 1;
        auto no_base = PluginBody(pack_key);
        no_base["companion_base_id"] = nullptr;
        bad_entry = write_plugin(
            bad_path, Envelope("cyxwiz-backend-pack-manifest", no_base, "pack-2026", pack_key, 2),
            "supported");
        if (!Expect(!plugin_verifier.VerifyManifest(bad_path, bad_entry, plugin, error,
                                                    BackendPackManifestKind::PluginPack),
                    "a plugin pack without a companion base was accepted"))
            return 1;

        // A pack made before manifests carried the plugin's name and
        // description (PluginBody) has neither.
        if (!Expect(plugin.plugin_name.empty() && plugin.plugin_description.empty(),
                    "an older plugin manifest gained a name or description"))
            return 1;
        // The plugin's own name and description are signed with the pack, for
        // any plugin; each is optional on its own.
        auto named = PluginBody(pack_key);
        named["plugin_id"] = "org.example.fluid-lab";
        named["plugin_name"] = "Fluid Lab";
        const auto named_path = temporary.Path() / "plugin-named.json";
        auto named_entry = write_plugin(
            named_path, Envelope("cyxwiz-backend-pack-manifest", named, "pack-2026", pack_key, 2),
            "supported");
        if (!Expect(plugin_verifier.VerifyManifest(named_path, named_entry, plugin, error,
                                                   BackendPackManifestKind::PluginPack),
                    error.c_str()) ||
            !Expect(plugin.plugin_id == "org.example.fluid-lab" && plugin.plugin_name == "Fluid Lab" &&
                        plugin.plugin_description.empty(),
                    "a plugin manifest's name was not kept"))
            return 1;
        named["plugin_description"] = "Fluid simulation: tanks, pipes and pumps (\xC3\xA9tude).";
        named_entry = write_plugin(
            named_path, Envelope("cyxwiz-backend-pack-manifest", named, "pack-2026", pack_key, 2),
            "supported");
        if (!Expect(plugin_verifier.VerifyManifest(named_path, named_entry, plugin, error,
                                                   BackendPackManifestKind::PluginPack),
                    error.c_str()) ||
            !Expect(plugin.plugin_name == "Fluid Lab" &&
                        plugin.plugin_description ==
                            "Fluid simulation: tanks, pipes and pumps (\xC3\xA9tude).",
                    "a plugin manifest's description was not kept") ||
            !Expect(plugin_verifier.VerifyInstalledManifest(named_path, "mujoco-v1",
                                                            BackendPackManifestKind::PluginPack,
                                                            plugin, error) &&
                        plugin.plugin_name == "Fluid Lab",
                    "installed plugin evidence lost the plugin's name"))
            return 1;
        // They are text for the user: a wrong type, an empty or oversized
        // string and control characters fail closed, as does any other field.
        const std::pair<const char*, Json> unusable[] = {
            {"plugin_name", ""},
            {"plugin_name", 7},
            {"plugin_name", nullptr},
            {"plugin_name", std::string(129, 'x')},
            {"plugin_name", "Fluid\nLab"},
            {"plugin_description", ""},
            {"plugin_description", Json::array({"Fluids."})},
            {"plugin_description", std::string(513, 'x')},
            {"plugin_description", std::string("tab\there")},
            {"plugin_author", "Example"},
        };
        for (const auto& [field, value] : unusable) {
            auto body = PluginBody(pack_key);
            body["plugin_name"] = "Fluid Lab";
            body[field] = value;
            bad_entry = write_plugin(
                bad_path, Envelope("cyxwiz-backend-pack-manifest", body, "pack-2026", pack_key, 2),
                "supported");
            if (!Expect(!plugin_verifier.VerifyManifest(bad_path, bad_entry, plugin, error,
                                                        BackendPackManifestKind::PluginPack),
                        (std::string("a plugin manifest with an unusable ") + field +
                         " was accepted: " + value.dump()).c_str()))
                return 1;
        }

        // Catalog and manifest support policy must agree.
        const auto diagnostic_entry = write_plugin(plugin_path, plugin_document, "diagnostic");
        if (!Expect(!plugin_verifier.VerifyManifest(plugin_path, diagnostic_entry, plugin, error,
                                                    BackendPackManifestKind::PluginPack),
                    "plugin support policy differing from the catalog was accepted"))
            return 1;
    }

    // A plugin's name and description: the cases packaging's display_text
    // runs too (redist/tests/test_backend_pack_contract.py), judged alike.
    {
        const std::filesystem::path fixtures = CYXWIZ_REDIST_TEST_FIXTURES;
        std::ifstream input(fixtures / "plugin_display_text_cases.json", std::ios::binary);
        const Json document = Json::parse(input, nullptr, false);
        if (!Expect(document.is_object() && document.contains("cases") && document["cases"].size() > 30,
                    "the shared display-text cases could not be read"))
            return 1;
        WriteJson(trust_path, TrustRoot(catalog_key, pack_key));
        BackendPackMetadataVerifier case_verifier(
            *BackendPackTrustStore::Load(trust_path, error), "0.2.0", "win64", "x86_64");
        for (const auto& item : document["cases"]) {
            std::string text;
            const auto piece = item["text"].get<std::string>();
            for (std::size_t count = item.value("repeat", std::size_t{1}); count > 0; --count) text += piece;
            text += item.value("suffix", std::string{});
            const auto maximum = item["maximum_bytes"].get<std::size_t>();
            if (!Expect(IsPluginDisplayText(text, maximum) == item["accepted"].get<bool>(),
                        ("display text judged unlike packaging: " + item["why"].get<std::string>()).c_str()))
                return 1;
            // The verifier applies the same rule to the signed manifest
            if (maximum != kMaximumPluginNameBytes && maximum != kMaximumPluginDescriptionBytes) continue;
            auto body = PluginBody(pack_key);
            body[maximum == kMaximumPluginNameBytes ? "plugin_name" : "plugin_description"] = text;
            const auto path = temporary.Path() / "plugin-display-text.json";
            const auto bytes = WriteJson(
                path, Envelope("cyxwiz-backend-pack-manifest", body, "pack-2026", pack_key, 2));
            BackendPackCatalogEntry entry;
            entry.pack_id = "mujoco-v1";
            entry.manifest_url = "https://downloads.cyxwiz.com/mujoco-v1.json";
            entry.manifest_sha256 = Sha256(bytes);
            entry.signing_key_id = "pack-2026";
            entry.support_status = BackendPackSupportStatus::Supported;
            VerifiedBackendPackManifest verified;
            const bool accepted = case_verifier.VerifyManifest(
                path, entry, verified, error, BackendPackManifestKind::PluginPack);
            if (!Expect(accepted == item["accepted"].get<bool>(),
                        ("signed manifest judged unlike packaging: " + item["why"].get<std::string>() +
                         (accepted ? "" : " (" + error + ")")).c_str()) ||
                !Expect(!accepted || (maximum == kMaximumPluginNameBytes ? verified.plugin_name
                                                                         : verified.plugin_description) == text,
                        "an accepted name or description was not kept as signed"))
                return 1;
        }
        // Bytes that are not UTF-8 never reach a signed manifest (the JSON
        // reader refuses them); the rule refuses them as well.
        for (const std::string text : {std::string("Fluid\xFFLab"), std::string("Fluid\xC3"),
                                       std::string("\xC0\xAF"), std::string("\xE0\x80\xAF"),
                                       std::string("\xED\xA0\x80"), std::string("\xF4\x90\x80\x80"),
                                       std::string("\x80"), std::string("\xF8\x88\x80\x80\x80")}) {
            if (!Expect(!IsPluginDisplayText(text, kMaximumPluginNameBytes),
                        "text that is not well-formed UTF-8 was accepted"))
                return 1;
        }
    }

    // The cross-language round trip: a plugin manifest that Python packaging
    // built and the release signing step signed, with a name and description
    // that are not ASCII (redist/tests/test_package_release.py writes it).
    // The verifier re-serialises the signed body before checking the
    // signature, so acceptance means both sides agree on the canonical bytes.
    {
        const std::filesystem::path fixture =
            std::filesystem::path(CYXWIZ_REDIST_TEST_FIXTURES) / "plugin-manifest-non-ascii";
        const auto read = [](const std::filesystem::path& path) {
            std::ifstream input(path, std::ios::binary);
            return std::string(std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>());
        };
        const auto manifest_bytes = read(fixture / "manifest.json");
        const auto signed_bytes = read(fixture / "signed.json");
        const Json document = Json::parse(manifest_bytes, nullptr, false);
        auto python_trust = BackendPackTrustStore::Load(fixture / "trusted-keys.json", error);
        if (!Expect(!manifest_bytes.empty() && !signed_bytes.empty() && document.is_object(),
                    "the Python-signed plugin manifest fixture could not be read") ||
            !Expect(python_trust.has_value(), error.c_str()) ||
            !Expect(document["signed"].dump() == signed_bytes,
                    "the canonical bytes differ from the bytes Python signed"))
            return 1;
        BackendPackMetadataVerifier python_verifier(std::move(*python_trust), "1.2.3", "win64", "x86_64");
        BackendPackCatalogEntry entry;
        entry.pack_id = document["signed"]["pack_id"].get<std::string>();
        entry.manifest_url = "https://downloads.cyxwiz.com/" + entry.pack_id + ".json";
        entry.manifest_sha256 = Sha256(manifest_bytes);
        entry.signing_key_id = "pack-fixture";
        entry.support_status = BackendPackSupportStatus::Diagnostic;
        VerifiedBackendPackManifest python_manifest;
        if (!Expect(python_verifier.VerifyManifest(fixture / "manifest.json", entry, python_manifest, error,
                                                   BackendPackManifestKind::PluginPack),
                    ("the verifier refused a manifest Python signed: " + error).c_str()) ||
            !Expect(python_manifest.plugin_id == "org.example.simulacion" &&
                        python_manifest.plugin_name == "Simulaci\xC3\xB3n \xC3\xBC" &&
                        python_manifest.plugin_description ==
                            "F\xC3\xADsica para rob\xC3\xB3tica: p\xC3\xA9ndulos, "
                            "\xE6\xB5\x81\xE4\xBD\x93 \xF0\x9F\x8C\x8A.",
                    "the non-ASCII name or description did not survive the round trip"))
            return 1;
        // One changed byte of the name and the signature no longer holds
        Json altered = document;
        altered["signed"]["plugin_name"] = "Simulacion \xC3\xBC";
        const auto altered_path = temporary.Path() / "python-altered.json";
        entry.manifest_sha256 = Sha256(WriteJson(altered_path, altered));
        if (!Expect(!python_verifier.VerifyManifest(altered_path, entry, python_manifest, error,
                                                    BackendPackManifestKind::PluginPack),
                    "a Python-signed manifest with a changed name was accepted"))
            return 1;
    }

    std::cout << "backend pack metadata verifier contract tests passed\n";
    return 0;
}
