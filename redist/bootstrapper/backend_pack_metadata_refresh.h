#pragma once

#include "backend_pack_metadata_verifier.h"

#include <cstdint>
#include <filesystem>
#include <string>

namespace cyxwiz::runtime {

class BackendPackMetadataSource {
public:
    virtual ~BackendPackMetadataSource() = default;
    virtual bool Fetch(
        const std::string& url,
        const std::filesystem::path& destination,
        std::uint64_t maximum_bytes,
        std::string& error) = 0;
};

class HttpsBackendPackMetadataSource final
    : public BackendPackMetadataSource {
public:
    bool Fetch(
        const std::string& url,
        const std::filesystem::path& destination,
        std::uint64_t maximum_bytes,
        std::string& error) override;
};

enum class BackendPackMetadataRefreshStatus {
    Refreshed,
    InvalidRequest,
    SourceFailure,
    VerificationFailure,
    PublicationFailure
};

struct BackendPackMetadataRefreshRequest {
    std::string catalog_url;
    std::filesystem::path trusted_keys_path;
    std::filesystem::path destination_root;
    std::string current_utc;
};

struct BackendPackMetadataRefreshResult {
    BackendPackMetadataRefreshStatus status =
        BackendPackMetadataRefreshStatus::InvalidRequest;
    std::string message;
    std::string catalog_id;
    std::size_t verified_pack_count = 0;
    // Entries of a refreshed catalog that are listed but not delivered, by
    // reason. The verified catalog is published with its policy all the same;
    // none of their manifests is.
    // The manifest is the one the signed catalog names, and this client
    // refuses it (a format it does not know, a key it does not trust):
    std::size_t refused_pack_count = 0;
    // The downloaded manifest is not the bytes the signed catalog names:
    std::size_t mismatched_pack_count = 0;
    // The manifest could not be downloaded:
    std::size_t unfetched_pack_count = 0;
};

BackendPackMetadataRefreshResult RefreshBackendPackMetadata(
    const BackendPackMetadataRefreshRequest& request,
    const BackendPackMetadataVerifier& verifier,
    BackendPackMetadataSource& source);

const char* BackendPackMetadataRefreshStatusName(
    BackendPackMetadataRefreshStatus status);

}  // namespace cyxwiz::runtime
