#include "base_stable_tool_publisher.h"

#include "atomic_file_publisher.h"
#include "backend_pack_hash.h"
#include "backend_pack_platform.h"

#include <algorithm>
#include <chrono>
#include <exception>
#include <string_view>
#include <system_error>

namespace cyxwiz::runtime {
namespace {

struct ToolBackup {
  std::filesystem::path directory;
  std::filesystem::path file;
  std::uintmax_t size = 0;
  std::string hash;
  bool existed = false;
};

bool CaptureTool(const std::filesystem::path &destination, ToolBackup &backup,
                 std::string &error) {
  std::error_code code;
  const auto status = std::filesystem::symlink_status(destination, code);
  if (status.type() == std::filesystem::file_type::not_found &&
      (!code || code == std::errc::no_such_file_or_directory))
    return true;
  if (code || status.type() != std::filesystem::file_type::regular) {
    error = "Existing stable tool is not a plain file: " + destination.string();
    return false;
  }
  backup.existed = true;
  backup.size = std::filesystem::file_size(destination, code);
  // Bound recovery storage independently of untrusted existing file sizes.
  if (code || backup.size == 0 || backup.size > 256ULL * 1024 * 1024) {
    error = "Existing stable tool has an invalid recovery size";
    return false;
  }
  if (!Sha256File(destination, backup.hash, error))
    return false;
  const auto directory =
      destination.parent_path() /
      (".cyxwiz-tool-backup-" +
       std::to_string(
           std::chrono::steady_clock::now().time_since_epoch().count()));
  // Never adopt an existing backup directory or overwrite its contents.
  if (!std::filesystem::create_directory(directory, code)) {
    error = "Cannot create an exclusive stable-tool recovery directory: " +
            code.message();
    return false;
  }
  backup.directory = directory;
  backup.file = directory / destination.filename();
  return PublishRegularFileAtomic(
      destination, backup.file, backup.size, error,
      [&](const auto &candidate, std::string &reason) {
        std::string hash;
        if (!Sha256File(candidate, hash, reason))
          return false;
        if (hash == backup.hash)
          return true;
        reason = "Stable tool changed while preserving its recovery copy";
        return false;
      });
}

bool RestoreTool(const std::filesystem::path &destination,
                 const ToolBackup &backup, std::string &error) {
  if (backup.existed) {
    return PublishRegularFileAtomic(
        backup.file, destination, backup.size, error,
        [&](const auto &candidate, std::string &reason) {
          std::string hash;
          if (!Sha256File(candidate, hash, reason))
            return false;
          if (hash == backup.hash)
            return true;
          reason = "Stable-tool recovery copy changed";
          return false;
        });
  }
  std::error_code code;
  std::filesystem::remove(destination, code);
  if (code)
    error =
        "Cannot remove the new stable tool during rollback: " + code.message();
  return !code;
}

void DiscardBackup(const ToolBackup &backup,
                   BaseStableToolsPublishResult &result) {
  if (backup.directory.empty())
    return;
  // Delete only the exact owned file and empty directory, never recursively.
  std::error_code code;
  std::filesystem::remove(backup.file, code);
  if (!code)
    std::filesystem::remove(backup.directory, code);
  if (code) {
    result.recovery_directory = backup.directory;
    result.message += "; stable-tool backup cleanup failed: " + code.message();
    result.message += "; recovery directory: " + backup.directory.string();
  }
}

const VerifiedPackComponent *
ValidateStableTool(const VerifiedBackendPackManifest &manifest,
                   const std::filesystem::path &installed_base_directory,
                   std::string_view tool_name, std::string &error) {
  const auto component =
      std::find_if(manifest.components.begin(), manifest.components.end(),
                   [&](const VerifiedPackComponent &candidate) {
                     return candidate.relative_path == tool_name;
                   });
  if (component == manifest.components.end() || component->size == 0) {
    error = "The verified CPU base does not contain required stable tool " +
            std::string(tool_name);
    return nullptr;
  }

  const auto source = installed_base_directory / std::string(tool_name);
  std::error_code filesystem_error;
  const auto source_status =
      std::filesystem::symlink_status(source, filesystem_error);
  if (filesystem_error ||
      source_status.type() != std::filesystem::file_type::regular) {
    error = "The installed stable tool is not a regular file: " +
            std::string(tool_name);
    return nullptr;
  }
  const auto source_size = std::filesystem::file_size(source, filesystem_error);
  if (filesystem_error || source_size != component->size) {
    error = "The installed stable tool differs from its verified manifest: " +
            std::string(tool_name);
    return nullptr;
  }
  std::string source_hash;
  if (!Sha256File(source, source_hash, error))
    return nullptr;
  if (source_hash != component->sha256) {
    error =
        "The installed stable tool hash differs from its verified manifest: " +
        std::string(tool_name);
    return nullptr;
  }
  return &*component;
}

bool PublishVerifiedStableTool(
    const VerifiedPackComponent &component,
    const std::filesystem::path &installed_base_directory,
    const std::filesystem::path &install_root,
    std::filesystem::path &published_path, std::string &error) {
  const auto source = installed_base_directory / component.relative_path;
  published_path = install_root / component.relative_path;
  if (!PublishRegularFileAtomic(
          source, published_path, component.size, error,
          [&](const std::filesystem::path &candidate,
              std::string &validation_error) {
            std::string candidate_hash;
            if (!Sha256File(candidate, candidate_hash, validation_error)) {
              return false;
            }
            if (candidate_hash != component.sha256) {
              validation_error = "The copied stable tool hash differs from the "
                                 "verified manifest";
              return false;
            }
            return true;
          })) {
    published_path.clear();
    return false;
  }
  return true;
}

} // namespace

BaseStableToolsPublishResult PublishVerifiedBaseStableTools(
    const VerifiedBackendPackManifest &manifest,
    const std::filesystem::path &installed_base_directory,
    const std::filesystem::path &runtime_root, StableToolsCommit commit) {
  BaseStableToolsPublishResult result;
  if (manifest.kind != BackendPackManifestKind::Base ||
      manifest.backend != "cpu" || !runtime_root.is_absolute() ||
      !installed_base_directory.is_absolute() ||
      installed_base_directory.lexically_normal() !=
          (runtime_root / "base" / manifest.pack_id).lexically_normal()) {
    result.message =
        "Verified CPU-base identity and absolute installed paths are required";
    return result;
  }

  const auto install_root = runtime_root.parent_path();
  // Reject a malformed pair before touching either installed tool. The copy
  // validator still checks each staged file against source changes afterward.
  const auto *finalizer = ValidateStableTool(
      manifest, installed_base_directory,
      CurrentProductRemovalFinalizerExecutableName(), result.message);
  if (!finalizer)
    return result;
  const auto *launcher = ValidateStableTool(
      manifest, installed_base_directory,
      CurrentRuntimeBootstrapperExecutableName(), result.message);
  if (!launcher)
    return result;
  ToolBackup backup, launcher_backup;
  const auto finalizer_destination = install_root / finalizer->relative_path;
  const auto launcher_destination = install_root / launcher->relative_path;
  if (!CaptureTool(finalizer_destination, backup, result.message)) {
    result.recovery_directory = backup.directory;
    if (!backup.directory.empty())
      result.message += "; recovery directory: " + backup.directory.string();
    return result;
  }
  if (commit &&
      !CaptureTool(launcher_destination, launcher_backup, result.message)) {
    DiscardBackup(backup, result);
    if (!launcher_backup.directory.empty()) {
      result.recovery_directory = launcher_backup.directory;
      result.message +=
          "; recovery directory: " + launcher_backup.directory.string();
    }
    return result;
  }
  bool finalizer_published = false;
  bool launcher_published = false;
  try {
    finalizer_published = PublishVerifiedStableTool(
        *finalizer, installed_base_directory, install_root,
        result.finalizer_path, result.message);
    launcher_published = finalizer_published &&
                         PublishVerifiedStableTool(
                             *launcher, installed_base_directory, install_root,
                             result.launcher_path, result.message);
    result.published = launcher_published;
  } catch (const std::exception &exception) {
    result.message =
        "Stable-tool publication failed: " + std::string(exception.what());
  }
  if (result.published && commit) {
    try {
      result.published = commit(result.message);
      if (!result.published && result.message.empty())
        result.message = "Activation rejected before commit";
    } catch (...) {
      result.published = false;
      result.commit_uncertain = true;
      result.message =
          "Activation outcome is uncertain; tools and recovery copies retained";
      for (const auto *saved : {&backup, &launcher_backup}) {
        if (!saved->directory.empty()) {
          result.recovery_directory = saved->directory;
          result.message +=
              "; recovery directory: " + saved->directory.string();
        }
      }
      return result;
    }
  }
  if (!result.published) {
    bool restored = true;
    if (launcher_published) {
      std::string rollback_error;
      if (!RestoreTool(launcher_destination, launcher_backup, rollback_error)) {
        restored = false;
        result.message += "; launcher rollback failed: " + rollback_error;
      } else {
        result.launcher_path.clear();
      }
    }
    if (finalizer_published) {
      std::string rollback_error;
      if (!RestoreTool(finalizer_destination, backup, rollback_error)) {
        restored = false;
        result.message += "; finalizer rollback failed: " + rollback_error;
      } else {
        result.finalizer_path.clear();
        result.message += "; previous finalizer state restored";
      }
    }
    if (!restored) {
      for (const auto *saved : {&backup, &launcher_backup}) {
        if (!saved->directory.empty()) {
          result.recovery_directory = saved->directory;
          result.message +=
              "; recovery directory: " + saved->directory.string();
        }
      }
      return result;
    }
    DiscardBackup(launcher_backup, result);
    DiscardBackup(backup, result);
    return result;
  }
  result.message = "Verified stable launcher and removal finalizer published "
                   "with per-file atomic replacement";
  DiscardBackup(launcher_backup, result);
  DiscardBackup(backup, result);
  return result;
}

} // namespace cyxwiz::runtime
