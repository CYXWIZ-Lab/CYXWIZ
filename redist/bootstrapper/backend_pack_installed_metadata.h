#pragma once
#include "backend_pack_metadata_verifier.h"

namespace cyxwiz::runtime {
bool ReadInstalledBackendPackMetadata(
    const std::filesystem::path &runtime_root, const std::string &pack_id,
    BackendPackManifestKind kind, const BackendPackMetadataVerifier &verifier,
    VerifiedBackendPackManifest &output, std::string &error);
bool RetainInstalledBackendPackMetadata(
    const std::filesystem::path &runtime_root,
    const std::filesystem::path &source, const BackendPackCatalogEntry &entry,
    BackendPackManifestKind kind, const BackendPackMetadataVerifier &verifier,
    std::string &error);
} // namespace cyxwiz::runtime
