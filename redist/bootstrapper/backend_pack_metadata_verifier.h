#pragma once

#include "backend_pack_installer.h"

#include <cstdint>
#include <filesystem>
#include <map>
#include <optional>
#include <string>
#include <vector>

namespace cyxwiz::runtime {

enum class BackendPackSupportStatus {
    Supported,
    Diagnostic,
    Blocked,
    Revoked
};

struct BackendPackCatalogEntry {
    std::string pack_id;
    std::string manifest_url;
    std::string manifest_sha256;
    std::string signing_key_id;
    BackendPackSupportStatus support_status =
        BackendPackSupportStatus::Blocked;
};

struct VerifiedBackendPackCatalog {
    std::string catalog_id;
    std::string generated_utc;
    std::string expires_utc;
    std::string minimum_client_version;
    std::vector<BackendPackCatalogEntry> packs;
};

struct BackendPackCompatibility {
    std::vector<std::string> device_kinds;
    std::vector<std::string> cpu_features;
    std::vector<std::string> provider_types;
    std::map<std::string, std::string> minimum_driver_versions;
    std::map<std::string, std::string> tested_driver_ranges;
    std::string minimum_identity_confidence;
    std::vector<std::string> recommendation_targets;
    std::string operation_matrix_id;
    std::vector<std::string> training_scope;
    BackendPackSupportStatus support_status =
        BackendPackSupportStatus::Blocked;
};

struct BackendPackArchiveIdentity {
    std::string file_name;
    std::uint64_t size = 0;
    std::string sha256;
};

struct VerifiedBackendPackManifest {
    std::string pack_id;
    std::string backend;
    std::string package_version;
    std::string platform;
    std::string architecture;
    std::string runtime_set_id;
    std::string minimum_cyxwiz_release;
    std::string maximum_cyxwiz_release;
    std::string arrayfire_version;
    std::string arrayfire_abi;
    std::string companion_base_id;
    std::vector<std::string> conflicts;
    BackendPackCompatibility compatibility;
    std::vector<VerifiedPackComponent> components;
    BackendPackArchiveIdentity archive;
    std::string generated_utc;

    VerifiedBackendPackPayload BindExtractedDirectory(
        std::filesystem::path source_directory) const;
};

class BackendPackTrustStore {
public:
    struct Key {
        std::string key_id;
        std::vector<unsigned char> public_key;
        bool catalog = false;
        bool pack = false;
        bool revoked = false;
    };

    static std::optional<BackendPackTrustStore> Load(
        const std::filesystem::path& path,
        std::string& error);
    const Key* Find(const std::string& key_id) const;

private:
    std::vector<Key> keys_;
};

class BackendPackMetadataVerifier {
public:
    BackendPackMetadataVerifier(
        BackendPackTrustStore trust_store,
        std::string client_version,
        std::string platform,
        std::string architecture);

    bool VerifyCatalog(
        const std::filesystem::path& catalog_path,
        const std::string& current_utc,
        VerifiedBackendPackCatalog& output,
        std::string& error) const;

    bool VerifyManifest(
        const std::filesystem::path& manifest_path,
        const BackendPackCatalogEntry& catalog_entry,
        VerifiedBackendPackManifest& output,
        std::string& error) const;

private:
    BackendPackTrustStore trust_store_;
    std::string client_version_;
    std::string platform_;
    std::string architecture_;
};

const char* BackendPackSupportStatusName(BackendPackSupportStatus status);

}  // namespace cyxwiz::runtime
