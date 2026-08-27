#pragma once

#include "backend_pack_acquisition.h"
#include "backend_pack_metadata_verifier.h"

#include <filesystem>
#include <string>

namespace cyxwiz::runtime {

enum class InstallerSetupStatus {
    Ready,
    InvalidRequest,
    TrustFailure,
    AcquisitionFailure,
    ExtractionFailure,
    CacheFailure
};

struct InstallerSetupRequest {
    std::filesystem::path descriptor_path;
    std::filesystem::path cache_root;
    std::string current_utc;
    std::string setup_version;
    std::string platform;
    std::string architecture;
};

struct InstallerSetupResult {
    InstallerSetupStatus status = InstallerSetupStatus::InvalidRequest;
    std::string message;
    std::string bundle_id;
    std::filesystem::path installer_path;
};

InstallerSetupResult PrepareInstallerBundle(
    const InstallerSetupRequest& request,
    const BackendPackTrustStore& trust_store,
    BackendPackArtifactSource& archive_source);

const char* InstallerSetupStatusName(InstallerSetupStatus status);

}  // namespace cyxwiz::runtime
