#pragma once

#include "backend_pack_metadata_verifier.h"

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace cyxwiz::runtime {

struct VerifiedInstallerBundleComponent {
    std::string relative_path;
    std::uint64_t size = 0;
    std::string sha256;
    bool executable = false;
};

struct VerifiedInstallerBundle {
    std::string bundle_id;
    std::string bundle_version;
    std::string cyxwiz_release;
    std::string release_channel;
    std::string platform;
    std::string architecture;
    std::string minimum_setup_version;
    std::string generated_utc;
    std::string expires_utc;
    BackendPackArchiveIdentity archive;
    std::vector<VerifiedInstallerBundleComponent> components;

    std::string InstallerEntryPoint() const;
};

class InstallerBundleVerifier {
public:
    InstallerBundleVerifier(
        BackendPackTrustStore trust_store,
        std::string setup_version,
        std::string platform,
        std::string architecture);

    bool Verify(
        const std::filesystem::path& descriptor_path,
        const std::string& current_utc,
        VerifiedInstallerBundle& output,
        std::string& error) const;

private:
    BackendPackTrustStore trust_store_;
    std::string setup_version_;
    std::string platform_;
    std::string architecture_;
};

}  // namespace cyxwiz::runtime
