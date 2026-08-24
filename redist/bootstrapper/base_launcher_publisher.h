#pragma once

#include "backend_pack_metadata_verifier.h"

#include <filesystem>
#include <string>

namespace cyxwiz::runtime {

struct BaseLauncherPublishResult {
    bool published = false;
    std::filesystem::path installed_path;
    std::string message;
};

BaseLauncherPublishResult PublishVerifiedBaseLauncher(
    const VerifiedBackendPackManifest& manifest,
    const std::filesystem::path& installed_base_directory,
    const std::filesystem::path& runtime_root);

}  // namespace cyxwiz::runtime
