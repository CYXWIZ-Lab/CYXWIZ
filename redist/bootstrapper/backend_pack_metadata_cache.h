#pragma once

#include "backend_pack_lifecycle_service.h"

#include <filesystem>
#include <string>

namespace cyxwiz::runtime {

// Publishes already-verified bootstrap metadata into a runtime cache. Files
// are replaced independently and the signed catalog is published last, so a
// runtime never observes a new catalog before its verified manifests exist.
bool PublishVerifiedBackendPackMetadata(
    const std::filesystem::path& trusted_keys_path,
    const VerifiedBackendPackCatalogSnapshot& snapshot,
    const std::filesystem::path& runtime_root,
    std::string& error);

}  // namespace cyxwiz::runtime
