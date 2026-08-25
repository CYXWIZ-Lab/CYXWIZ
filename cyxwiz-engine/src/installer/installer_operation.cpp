#include "installer_operation.h"

#include <utility>

namespace cyxwiz::installer {
namespace {

void AppendMessage(std::string &output, const std::string &message) {
  if (message.empty())
    return;
  if (!output.empty())
    output += '\n';
  output += message;
}

void Report(const InstallerPlanExecutionObserver &observer,
            std::size_t completed_steps, std::size_t total_steps,
            std::string activity) {
  if (observer) {
    observer({completed_steps, total_steps, std::move(activity)});
  }
}

} // namespace

InstallerPlanExecutionResult ExecuteInstallerPlan(
    BackendPackInstallerPlatform &platform, const BackendPackInstallerPlan &plan,
    const InstallerPlanExecutionObserver &observer) {
  InstallerPlanExecutionResult batch;
  const std::size_t total_steps = static_cast<std::size_t>(plan.install_base) +
                                  plan.pack_ids.size() +
                                  plan.deactivate_backends.size();
  if (total_steps == 0) {
    batch.succeeded = true;
    batch.message = "No installation changes were required";
    return batch;
  }

  std::size_t completed_steps = 0;
  if (plan.install_base) {
    Report(observer, completed_steps, total_steps,
           "Installing and qualifying the required CPU Engine");
    const auto result = platform.InstallBase(plan.base_pack_id);
    AppendMessage(batch.message, result.message);
    if (!result.succeeded || !result.activated)
      return batch;
    ++completed_steps;
  }

  for (const auto &pack_id : plan.pack_ids) {
    Report(observer, completed_steps, total_steps,
           "Downloading, verifying, and qualifying " + pack_id);
    const auto result = platform.InstallOrUpdate(pack_id);
    AppendMessage(batch.message, result.message);
    if (!result.succeeded || !result.activated)
      return batch;
    ++completed_steps;
  }

  for (const auto &backend : plan.deactivate_backends) {
    Report(observer, completed_steps, total_steps,
           "Deactivating the " + backend + " route");
    const auto result = platform.DeactivateBackend(backend);
    AppendMessage(batch.message, result.message);
    if (!result.succeeded)
      return batch;
    ++completed_steps;
  }

  batch.succeeded = true;
  Report(observer, completed_steps, total_steps,
         "Installation changes completed");
  return batch;
}

} // namespace cyxwiz::installer
