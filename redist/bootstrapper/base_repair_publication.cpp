#include "base_repair_publication.h"

#include <algorithm>
#include <exception>
#include <limits>
#include <system_error>

namespace cyxwiz::runtime {
namespace {
bool SameState(const ActiveRuntimeState &a, const ActiveRuntimeState &b) {
  return a.generation == b.generation && a.base_pack_id == b.base_pack_id &&
         a.runtime_set_id == b.runtime_set_id &&
         a.packs.size() == b.packs.size() &&
         std::equal(a.packs.begin(), a.packs.end(), b.packs.begin(),
                    [](const auto &x, const auto &y) {
                      return x.backend == y.backend && x.pack_id == y.pack_id;
                    });
}

bool PlainDirectory(const std::filesystem::path &path, std::error_code &error) {
  const auto status = std::filesystem::symlink_status(path, error);
  return !error && status.type() == std::filesystem::file_type::directory &&
         std::filesystem::canonical(path, error) == path && !error;
}
} // namespace

BackendPackInstallResult
PublishBaseRepair(const std::filesystem::path &runtime_root,
                  const std::filesystem::path &staging_root,
                  const ActiveRuntimeState &expected,
                  const BackendPackExecutionActiveCheck &execution_active,
                  const BackendPackInstallCheckpointHook &checkpoint,
                  const std::atomic<bool> &cancelled) {
  namespace fs = std::filesystem;
  using S = BackendPackInstallStatus;
  if (!execution_active ||
      expected.generation == std::numeric_limits<std::uint64_t>::max())
    return {S::InvalidRequest, "CPU repair requires an execution guard and an "
                               "advanceable runtime identity"};
  std::error_code ec;
  const auto root = fs::canonical(runtime_root, ec);
  if (ec)
    return {S::FilesystemFailure, "Cannot resolve the CPU repair root"};
  const auto stage = fs::canonical(staging_root, ec);
  if (ec || stage.parent_path() != root / "staging" ||
      !PlainDirectory(stage, ec) || !PlainDirectory(stage / "payload", ec))
    return {S::FilesystemFailure,
            "CPU repair staging is redirected or outside the runtime"};
  const auto staged = stage / "payload";
  const auto destination = root / "base" / expected.base_pack_id;
  fs::create_directories(root / "base", ec);
  if (ec || !PlainDirectory(root / "base", ec))
    return {S::FilesystemFailure, "CPU base directory is unsafe to repair"};
  const auto destination_status = fs::symlink_status(destination, ec);
  if (ec == std::errc::no_such_file_or_directory)
    ec.clear();
  const bool existed = destination_status.type() != fs::file_type::not_found;
  if (ec || (existed && !PlainDirectory(destination, ec)))
    return {S::FilesystemFailure,
            "Installed CPU base is redirected or unreadable"};
  std::string error;
  const auto unchanged = [&] {
    ActiveRuntimeState current;
    return LoadActiveRuntimeState(root / "active-runtime.json", current,
                                  error) &&
           SameState(current, expected);
  };
  if (!unchanged())
    return {S::InvalidRequest, "Runtime identity changed before CPU repair"};
  if (cancelled.load())
    return {S::Interrupted, "CPU repair cancelled before publication"};
  if (execution_active())
    return {S::ExecutionActive,
            "Close CyxWiz Engine before repairing its CPU base"};

  const auto backups = root / "repair-backups";
  fs::create_directories(backups, ec);
  if (ec || !PlainDirectory(backups, ec))
    return {S::FilesystemFailure, "CPU repair backup directory is unsafe"};
  const auto backup = backups / stage.filename();
  if (!fs::create_directory(backup, ec) || ec)
    return {S::FilesystemFailure, "Cannot reserve a unique CPU repair backup"};
  if (!SaveActiveRuntimeStateAtomic(backup / "previous-active-runtime.json",
                                    expected, error))
    return {S::FilesystemFailure,
            "Cannot preserve CPU repair recovery evidence: " + error};

  bool quarantined = false;
  bool published = false;
  const auto failed = [&](S status,
                          std::string reason) -> BackendPackInstallResult {
    std::error_code restore_error;
    if (published)
      fs::rename(destination, staged, restore_error);
    if (!restore_error && quarantined)
      fs::rename(backup / "payload", destination, restore_error);
    if (restore_error) {
      status = S::FilesystemFailure;
      reason += "; automatic restoration failed: " + restore_error.message();
    }
    reason += "; recovery backup: " + backup.string();
    return {status, std::move(reason), destination};
  };
  const auto interrupted = [&](BackendPackInstallCheckpoint point) {
    return cancelled.load() || (checkpoint && !checkpoint(point));
  };
  auto candidate = expected;
  ++candidate.generation;
  candidate.packs.clear();
  try {
    if (existed) {
      fs::rename(destination, backup / "payload", ec);
      if (ec)
        return failed(S::FilesystemFailure,
                      "Cannot preserve the old CPU base: " + ec.message());
      quarantined = true;
    }
    if (interrupted(BackendPackInstallCheckpoint::AfterQuarantine))
      return failed(S::Interrupted, "CPU repair interrupted after backup");
    fs::rename(staged, destination, ec);
    if (ec)
      return failed(S::FilesystemFailure,
                    "Cannot publish the repaired CPU base: " + ec.message());
    published = true;
    if (interrupted(BackendPackInstallCheckpoint::AfterPackPublish) ||
        interrupted(BackendPackInstallCheckpoint::BeforeActivation))
      return failed(S::Interrupted, "CPU repair interrupted before activation");
    if (execution_active())
      return failed(S::ExecutionActive, "Execution started during CPU repair");
    if (!unchanged())
      return failed(S::InvalidRequest,
                    "Runtime identity changed during CPU repair");

    ActiveRuntime resolved;
    if (!ResolveRuntimeState(root, candidate, resolved, error))
      return failed(S::IntegrityFailure,
                    "Repaired CPU runtime is incomplete: " + error);
    if (!SaveActiveRuntimeStateAtomic(root / "active-runtime.json", candidate,
                                      error))
      return failed(S::FilesystemFailure,
                    "Cannot activate the repaired CPU base: " + error);
  } catch (const std::exception &exception) {
    return failed(S::FilesystemFailure,
                  std::string("CPU repair failed before commit: ") +
                      exception.what());
  } catch (...) {
    return failed(S::FilesystemFailure,
                  "CPU repair failed before commit with an unknown exception");
  }
  // Commit complete. Never delete the former directory (which may include user
  // files). A killed process also leaves that directory outside staging
  // cleanup.
  std::string message =
      "CPU base repaired; optional packs are retained but inactive. "
      "Verify compute routes in CyxWiz Engine. Recovery backup: " +
      backup.string();
  fs::remove(stage,
             ec); // Empty staging only; never recurse over recovery data.
  if (ec)
    message += "; empty staging cleanup failed: " + ec.message();
  return {S::InstalledAndActivated, message, destination,
          BackendPackStateResult{BackendPackStateStatus::Completed, message,
                                 expected, candidate}};
}
} // namespace cyxwiz::runtime
