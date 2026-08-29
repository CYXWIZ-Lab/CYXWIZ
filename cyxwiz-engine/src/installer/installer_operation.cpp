#include "installer_operation.h"

#include <algorithm>
#include <utility>

namespace cyxwiz::installer {
namespace {

struct InstallerPhase {
  std::size_t index = 0;
  std::size_t count = 0;
  const char *label = "";
};

InstallerPhase DescribePhase(const std::string &stage,
                             bool include_registration) {
  const std::size_t install_phase_count = include_registration ? 8 : 7;
  if (stage == "verifying_catalog")
    return {1, install_phase_count, "Verify catalog"};
  if (stage == "verifying_manifest")
    return {2, install_phase_count, "Verify package"};
  if (stage == "acquiring")
    return {3, install_phase_count, "Download package"};
  if (stage == "extracting")
    return {4, install_phase_count, "Extract package"};
  if (stage == "installing")
    return {5, install_phase_count, "Install files"};
  if (stage == "qualifying")
    return {6, install_phase_count, "Verify compatibility"};
  if (stage == "activating")
    return {7, install_phase_count, "Activate runtime"};
  if (stage == "registering" && include_registration)
    return {8, install_phase_count, "Register application"};
  if (stage == "complete")
    return {install_phase_count, install_phase_count,
            include_registration ? "Finalize application"
                                 : "Finalize package"};
  if (stage == "removing")
    return {1, 1, "Deactivate runtime"};
  return {};
}

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
    const float fraction = total_steps == 0
                               ? 0.0f
                               : static_cast<float>(completed_steps) /
                                     static_cast<float>(total_steps);
    InstallerPlanExecutionProgress progress;
    progress.completed_steps = completed_steps;
    progress.total_steps = total_steps;
    progress.overall_fraction = fraction;
    progress.activity = std::move(activity);
    observer(progress);
  }
}

float DetailFraction(const InstallerHelperProgress &progress) {
  const float inner = progress.total_bytes == 0
                          ? 0.0f
                          : std::clamp(
                                static_cast<float>(progress.completed_bytes) /
                                    static_cast<float>(progress.total_bytes),
                                0.0f, 1.0f);
  if (progress.stage == "verifying_catalog") return 0.02f;
  if (progress.stage == "verifying_manifest") return 0.05f;
  if (progress.stage == "acquiring") return 0.08f + 0.42f * inner;
  if (progress.stage == "extracting") return 0.50f + 0.22f * inner;
  if (progress.stage == "installing") return 0.72f + 0.12f * inner;
  if (progress.stage == "qualifying") return 0.86f;
  if (progress.stage == "activating") return 0.95f;
  if (progress.stage == "registering") return 0.98f;
  if (progress.stage == "removing") return 0.35f;
  if (progress.stage == "complete") return 1.0f;
  return 0.0f;
}

void ReportDetail(const InstallerPlanExecutionObserver &observer,
                  std::size_t completed_steps, std::size_t total_steps,
                  const InstallerHelperProgress &detail,
                  bool include_registration) {
  if (!observer || total_steps == 0) return;
  const float fraction = std::clamp(
      (static_cast<float>(completed_steps) + DetailFraction(detail)) /
          static_cast<float>(total_steps),
      0.0f, 1.0f);
  const auto phase = DescribePhase(detail.stage, include_registration);
  InstallerPlanExecutionProgress progress;
  progress.completed_steps = completed_steps;
  progress.total_steps = total_steps;
  progress.phase_index = phase.index;
  progress.phase_count = phase.count;
  progress.overall_fraction = fraction;
  progress.completed_bytes = detail.completed_bytes;
  progress.total_bytes = detail.total_bytes;
  progress.stage = detail.stage;
  progress.phase_label = phase.label;
  progress.activity = detail.message;
  observer(progress);
}

} // namespace

InstallerPlanExecutionResult ExecuteInstallerPlan(
    BackendPackInstallerPlatform &platform, const BackendPackInstallerPlan &plan,
    const InstallerPlanExecutionObserver &observer) {
  InstallerPlanExecutionResult batch;
  const std::size_t total_steps = static_cast<std::size_t>(plan.install_base) +
                                  static_cast<std::size_t>(plan.update_base) +
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
    const auto result = platform.InstallBase(
        plan.base_pack_id,
        [&](const InstallerHelperProgress &detail) {
          ReportDetail(observer, completed_steps, total_steps, detail, true);
        });
    AppendMessage(batch.message, result.message);
    if (!result.succeeded || !result.activated)
      return batch;
    ++completed_steps;
  }

  if (plan.update_base) {
    Report(observer, completed_steps, total_steps,
           "Updating and qualifying the CyxWiz Engine/CPU base");
    const auto result = platform.UpdateBase(
        plan.base_pack_id,
        [&](const InstallerHelperProgress &detail) {
          ReportDetail(observer, completed_steps, total_steps, detail, true);
        });
    AppendMessage(batch.message, result.message);
    if (!result.succeeded || !result.activated)
      return batch;
    ++completed_steps;
  }

  for (const auto &pack_id : plan.pack_ids) {
    Report(observer, completed_steps, total_steps,
           "Downloading, verifying, and qualifying " + pack_id);
    const auto result = platform.InstallOrUpdate(
        pack_id, [&](const InstallerHelperProgress &detail) {
          ReportDetail(observer, completed_steps, total_steps, detail, false);
        });
    AppendMessage(batch.message, result.message);
    if (!result.succeeded || !result.activated)
      return batch;
    ++completed_steps;
  }

  for (const auto &backend : plan.deactivate_backends) {
    Report(observer, completed_steps, total_steps,
           "Deactivating the " + backend + " route");
    const auto result = platform.DeactivateBackend(
        backend, [&](const InstallerHelperProgress &detail) {
          ReportDetail(observer, completed_steps, total_steps, detail, false);
        });
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
