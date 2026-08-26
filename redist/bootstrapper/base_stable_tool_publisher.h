#pragma once

#include "backend_pack_metadata_verifier.h"

#include <filesystem>
#include <string>

namespace cyxwiz::runtime {

struct BaseStableToolsPublishResult {
    bool published = false;
    std::filesystem::path launcher_path;
    std::filesystem::path finalizer_path;
    std::string message;
};

BaseStableToolsPublishResult PublishVerifiedBaseStableTools(
    const VerifiedBackendPackManifest& manifest,
    const std::filesystem::path& installed_base_directory,
    const std::filesystem::path& runtime_root);

}  // namespace cyxwiz::runtime
