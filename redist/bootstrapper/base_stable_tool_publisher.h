#pragma once

#include "backend_pack_metadata_verifier.h"

#include <filesystem>
#include <functional>
#include <string>

namespace cyxwiz::runtime {

struct BaseStableToolsPublishResult {
  bool published = false;
  bool commit_uncertain = false;
  std::filesystem::path launcher_path;
  std::filesystem::path finalizer_path;
  // Retained if rollback/cleanup fails; never automatically discard recovery
  // evidence.
  std::filesystem::path recovery_directory;
  std::string message;
};

// Runs synchronously after both tools are published, with backups still held.
// False MUST mean activation was not committed. True means committed. An
// exception is an unknown outcome: retain new tools and backups for recovery.
// Caller must hold installation ownership throughout publication and commit.
using StableToolsCommit = std::function<bool(std::string &error)>;

BaseStableToolsPublishResult PublishVerifiedBaseStableTools(
    const VerifiedBackendPackManifest &manifest,
    const std::filesystem::path &installed_base_directory,
    const std::filesystem::path &runtime_root, StableToolsCommit commit = {});

} // namespace cyxwiz::runtime
