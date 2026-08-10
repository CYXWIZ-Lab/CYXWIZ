#include "viewport.h"
#include "panels/training_plot_panel.h"
#include "../core/execution_device_preferences.h"
#include "../core/training_trace_collector.h"

#include <algorithm>
#include <cstddef>
#include <exception>
#include <optional>
#include <string>
#include <utility>

#include <cyxwiz/cyxwiz.h>
#include <imgui.h>

namespace gui {
namespace {

const char* DeviceTypeName(int type) {
    switch (type) {
        case 0: return "CPU";
        case 1: return "CUDA";
        case 2: return "OpenCL";
        case 3: return "Metal (unsupported)";
        case 4: return "Vulkan (unsupported)";
        case 5: return "oneAPI";
        default: return "Unknown";
    }
}

std::optional<int> DeviceTypeFromExecutionBackend(
    const std::string& backend) {
    if (backend == "arrayfire_cpu") {
        return 0;
    }
    if (backend == "arrayfire_cuda") {
        return 1;
    }
    if (backend == "arrayfire_opencl") {
        return 2;
    }
    if (backend == "arrayfire_oneapi") {
        return 5;
    }
    return std::nullopt;
}

ImVec4 DeviceTypeColor(int type) {
    switch (type) {
        case 0: return ImVec4(0.5f, 0.8f, 1.0f, 1.0f);
        case 1: return ImVec4(0.35f, 0.95f, 0.45f, 1.0f);
        case 2: return ImVec4(1.0f, 0.75f, 0.35f, 1.0f);
        case 5: return ImVec4(0.8f, 0.6f, 1.0f, 1.0f);
        default: return ImVec4(0.75f, 0.75f, 0.75f, 1.0f);
    }
}

const CachedDeviceInfo* FindDevice(
    const std::vector<CachedDeviceInfo>& devices,
    int type,
    int device_id) {
    for (const auto& device : devices) {
        if (device.type == type && device.device_id == device_id) {
            return &device;
        }
    }
    return nullptr;
}

bool HasRunBoundDevice(const cyxwiz::TrainingTraceSummary& trace) {
    return !trace.run_id.empty() && !trace.effective_backend.empty();
}

bool IsTrainingTraceActive(const cyxwiz::TrainingTraceSummary& trace) {
    return trace.status == "running";
}

void SummaryRow(const char* label, const char* value) {
    ImGui::TableNextRow();
    ImGui::TableNextColumn();
    ImGui::TextDisabled("%s", label);
    ImGui::TableNextColumn();
    ImGui::TextUnformatted(value);
}

}  // namespace

Viewport::Viewport() : show_window_(true), devices_initialized_(false) {
}

Viewport::~Viewport() = default;

void Viewport::Render() {
    if (!show_window_) return;

    if (ImGui::Begin("Viewport", &show_window_)) {
        ImGui::TextColored(ImVec4(0.4f, 0.8f, 1.0f, 1.0f),
                           "System Information");
        ImGui::Separator();

        if (ImGui::BeginTable("##engine_summary", 2,
                              ImGuiTableFlags_SizingStretchProp |
                                  ImGuiTableFlags_BordersInnerV)) {
            ImGui::TableSetupColumn("Label",
                                    ImGuiTableColumnFlags_WidthFixed,
                                    120.0f);
            ImGui::TableSetupColumn("Value",
                                    ImGuiTableColumnFlags_WidthStretch);
            SummaryRow("Engine version", cyxwiz::GetVersionString());
            ImGui::EndTable();
        }

        ImGui::Spacing();
        ImGui::TextColored(ImVec4(0.3f, 1.0f, 0.3f, 1.0f),
                           "Active Compute Runtime");
        ImGui::Separator();

        try {
            if (!devices_initialized_) {
                cached_devices_.clear();
                const auto raw_devices = cyxwiz::Device::GetAvailableDevices();
                cached_devices_.reserve(raw_devices.size());
                for (const auto& device : raw_devices) {
                    CachedDeviceInfo cached;
                    cached.type = static_cast<int>(device.type);
                    cached.device_id = device.device_id;
                    cached.name = device.name;
                    cached.memory_total = device.memory_total;
                    cached.memory_available = device.memory_available;
                    cached.compute_units = device.compute_units;
                    cached.supports_fp64 = device.supports_fp64;
                    cached.supports_fp16 = device.supports_fp16;
                    cached_devices_.push_back(std::move(cached));
                }
                devices_initialized_ = true;
            }

            const auto trace =
                cyxwiz::TrainingTraceCollector::LatestTrace();
            const bool has_run_bound_device = HasRunBoundDevice(trace);
            const auto pending_selection =
                cyxwiz::GetPendingExecutionDeviceSelection();
            const auto next_run_policy =
                cyxwiz::GetNextRunExecutionPolicy();
            const auto trace_type = has_run_bound_device
                ? DeviceTypeFromExecutionBackend(trace.effective_backend)
                : std::optional<int>{};
            const int pending_type = pending_selection.has_value()
                ? static_cast<int>(pending_selection->type)
                : -1;
            const auto* pending_info = pending_selection.has_value()
                ? FindDevice(cached_devices_,
                             pending_type,
                             pending_selection->device_id)
                : nullptr;
            auto* process_device = cyxwiz::Device::GetCurrentDevice();
            if (has_run_bound_device || process_device) {
                const int active_type = trace_type.has_value()
                    ? *trace_type
                    : (process_device
                           ? static_cast<int>(process_device->GetType())
                           : -1);
                const int active_id = has_run_bound_device
                    ? trace.effective_device_id
                    : (process_device ? process_device->GetDeviceId() : -1);
                const auto* active_info =
                    FindDevice(cached_devices_, active_type, active_id);
                const std::string active_name = has_run_bound_device
                    ? (!trace.effective_device_name.empty()
                           ? trace.effective_device_name
                           : (active_info ? active_info->name
                                          : "Runtime device not in discovery list"))
                    : (active_info ? active_info->name
                                   : "Runtime device not in discovery list");
                const char* source_label = has_run_bound_device
                    ? (IsTrainingTraceActive(trace)
                           ? "Current training trace"
                           : "Last training trace")
                    : "Process runtime";
                const std::string backend_label = has_run_bound_device
                    ? trace.effective_backend
                    : DeviceTypeName(active_type);

                ImGui::TextColored(DeviceTypeColor(active_type),
                                   "%s  %s",
                                   has_run_bound_device
                                       ? (IsTrainingTraceActive(trace)
                                              ? "ACTIVE RUN"
                                              : "LAST RUN")
                                       : "ACTIVE",
                                   DeviceTypeName(active_type));
                if (ImGui::BeginTable("##active_runtime", 2,
                                      ImGuiTableFlags_SizingStretchProp |
                                          ImGuiTableFlags_BordersInnerV)) {
                    ImGui::TableSetupColumn("Label",
                                            ImGuiTableColumnFlags_WidthFixed,
                                            120.0f);
                    ImGui::TableSetupColumn("Value",
                                            ImGuiTableColumnFlags_WidthStretch);
                    SummaryRow("Source", source_label);
                    if (has_run_bound_device) {
                        SummaryRow(
                            "Requested backend",
                            trace.requested_backend.empty()
                                ? "Not recorded"
                                : trace.requested_backend.c_str());
                        SummaryRow("Effective backend",
                                   backend_label.c_str());
                    } else {
                        SummaryRow("Backend", backend_label.c_str());
                    }
                    SummaryRow("Device", active_name.c_str());

                    ImGui::TableNextRow();
                    ImGui::TableNextColumn();
                    ImGui::TextDisabled("Device ID");
                    ImGui::TableNextColumn();
                    ImGui::Text("%d", active_id);

                    ImGui::TableNextRow();
                    ImGui::TableNextColumn();
                    ImGui::TextDisabled("Total memory");
                    ImGui::TableNextColumn();
                    if (active_info && active_info->memory_total > 0) {
                        const double total_gb =
                            active_info->memory_total /
                            (1024.0 * 1024.0 * 1024.0);
                        ImGui::Text("%.2f GB", total_gb);
                    } else {
                        ImGui::TextDisabled("Not reported by backend");
                    }

                    if (has_run_bound_device) {
                        SummaryRow(
                            "Placement",
                            trace.placement_fingerprint.empty()
                                ? "Not recorded"
                                : trace.placement_fingerprint.c_str());
                        ImGui::TableNextRow();
                        ImGui::TableNextColumn();
                        ImGui::TextDisabled("Placement entries");
                        ImGui::TableNextColumn();
                        ImGui::Text(
                            "%llu",
                            static_cast<unsigned long long>(
                                trace.placement_entry_count));
                        if (!trace.fallback_policy.empty()) {
                            const bool run_is_strict =
                                trace.fallback_policy ==
                                "forbid_native_cpu_fallback";
                            SummaryRow(
                                "Run policy",
                                run_is_strict
                                    ? "Strict ArrayFire residency"
                                    : "Compatibility with recorded fallback");
                        }
                    }

                    const auto selected_policy =
                        next_run_policy.value_or(
                            cyxwiz::ArrayFireFallbackPolicy::
                                AllowNativeCpuFallback);
                    const std::string next_policy_label =
                        std::string(cyxwiz::ExecutionPolicyDisplayName(
                            selected_policy)) +
                        (next_run_policy.has_value() ? "" : " (default)");
                    SummaryRow("Next policy", next_policy_label.c_str());

                    if (pending_selection.has_value()) {
                        const std::string pending_backend =
                            cyxwiz::ExecutionDeviceSelectionBackendName(
                                pending_selection->type);
                        const std::string pending_name = pending_info
                            ? pending_info->name
                            : "Queued device not in discovery list";
                        SummaryRow("Next backend", pending_backend.c_str());
                        SummaryRow("Next device", pending_name.c_str());

                        ImGui::TableNextRow();
                        ImGui::TableNextColumn();
                        ImGui::TextDisabled("Next device ID");
                        ImGui::TableNextColumn();
                        ImGui::Text("%d", pending_selection->device_id);
                    }
                    ImGui::EndTable();
                }
            } else {
                ImGui::TextColored(ImVec4(1.0f, 0.75f, 0.3f, 1.0f),
                                   "ArrayFire runtime device is unavailable");
            }

            ImGui::Spacing();
            ImGui::Text("Available Devices (%zu)", cached_devices_.size());
            ImGui::SameLine();
            if (ImGui::SmallButton("Refresh##viewport_devices")) {
                RefreshDevices();
            }
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip(
                    "Re-enumerate available devices; queued selection is applied at training start");
            }

            if (cached_devices_.empty()) {
                ImGui::TextDisabled("No compute devices discovered");
            } else {
                for (const auto& device : cached_devices_) {
                    const bool is_run_bound =
                        has_run_bound_device && trace_type.has_value() &&
                        *trace_type == device.type &&
                        trace.effective_device_id == device.device_id;
                    const bool is_process_active =
                        !has_run_bound_device && process_device &&
                        static_cast<int>(process_device->GetType()) ==
                            device.type &&
                        process_device->GetDeviceId() == device.device_id;
                    const bool is_pending =
                        pending_selection.has_value() &&
                        pending_type == device.type &&
                        pending_selection->device_id == device.device_id;
                    ImGui::BulletText("%s", device.name.c_str());
                    ImGui::SameLine();
                    ImGui::TextDisabled("[%s:%d]",
                                        DeviceTypeName(device.type),
                                        device.device_id);
                    if (is_run_bound || is_process_active) {
                        ImGui::SameLine();
                        ImGui::TextColored(DeviceTypeColor(device.type),
                                           "%s",
                                           is_run_bound
                                               ? (IsTrainingTraceActive(trace)
                                                      ? "Active run"
                                                      : "Last run")
                                               : "Active");
                    }
                    if (is_pending) {
                        ImGui::SameLine();
                        ImGui::TextColored(DeviceTypeColor(device.type),
                                           "Next run");
                    }
                }
            }

            ImGui::Spacing();
            ImGui::TextWrapped(
                "Training trace reports the run-bound ArrayFire "
                "backend/device when a run exists. Process runtime is shown "
                "only when no run-bound trace is available. Preferences "
                "selection is shown separately as Next run and is applied at "
                "training start. Native CPU fallback is reported separately "
                "by run diagnostics.");
        } catch (const std::exception& error) {
            ImGui::TextColored(ImVec4(1.0f, 0.3f, 0.3f, 1.0f),
                               "Device query failed: %s",
                               error.what());
        }

        ImGui::Spacing();
        ImGui::Spacing();

        cyxwiz::TrainingStatusSnapshot training;
        const bool has_training_panel = training_panel_ != nullptr;
        if (has_training_panel) {
            training = training_panel_->GetStatusSnapshot();
        }
        const bool has_training_summary =
            has_training_panel &&
            (training.has_data || training.is_training || training.is_preparing ||
             training.preparation_failed || !training.terminal_status.empty());

        ImGui::TextColored(ImVec4(1.0f, 0.6f, 0.2f, 1.0f),
                           "Training Summary");
        ImGui::Separator();

        if (!has_training_summary) {
            ImGui::TextDisabled("No active or completed training run");
        } else {
            const char* status = "Last run";
            ImVec4 status_color(0.7f, 0.7f, 0.7f, 1.0f);
            if (training.preparation_failed) {
                status = "Preparation failed";
                status_color = ImVec4(1.0f, 0.35f, 0.35f, 1.0f);
            } else if (training.is_preparing) {
                status = "Preparing";
                status_color = ImVec4(1.0f, 0.75f, 0.3f, 1.0f);
            } else if (training.is_training) {
                status = "Training";
                status_color = ImVec4(0.35f, 1.0f, 0.4f, 1.0f);
            } else if (!training.terminal_status.empty()) {
                status = training.terminal_status.c_str();
                status_color = training.terminal_status == "completed"
                    ? ImVec4(0.35f, 1.0f, 0.4f, 1.0f)
                    : ImVec4(1.0f, 0.75f, 0.3f, 1.0f);
            }

            ImGui::Text("Status:");
            ImGui::SameLine();
            ImGui::TextColored(status_color, "%s", status);

            if (!training.status_message.empty()) {
                ImGui::TextWrapped("%s", training.status_message.c_str());
            }
            if (training.is_preparing) {
                ImGui::ProgressBar(
                    std::clamp(training.preparation_progress, 0.0f, 1.0f),
                    ImVec2(-1.0f, 0.0f));
            }

            if (training.current_epoch > 0) {
                if (training.total_epochs > 0) {
                    ImGui::Text("Epoch: %d / %d",
                                training.current_epoch,
                                training.total_epochs);
                } else {
                    ImGui::Text("Epoch: %d", training.current_epoch);
                }
            }
            if (training.is_training && training.total_batches > 0) {
                ImGui::Text("Batch: %d / %d",
                            training.current_batch,
                            training.total_batches);
            }

            if (ImGui::BeginTable("##training_metrics", 3,
                                  ImGuiTableFlags_SizingStretchProp |
                                      ImGuiTableFlags_BordersInnerV |
                                      ImGuiTableFlags_RowBg)) {
                ImGui::TableSetupColumn("Metric");
                ImGui::TableSetupColumn("Train");
                ImGui::TableSetupColumn("Validation");
                ImGui::TableHeadersRow();

                if (training.train_loss >= 0.0) {
                    ImGui::TableNextRow();
                    ImGui::TableNextColumn();
                    ImGui::TextUnformatted("Loss");
                    ImGui::TableNextColumn();
                    ImGui::Text("%.6f", training.train_loss);
                    ImGui::TableNextColumn();
                    if (training.val_loss >= 0.0) {
                        ImGui::Text("%.6f", training.val_loss);
                    } else {
                        ImGui::TextDisabled("--");
                    }
                }

                if (training.train_accuracy >= 0.0) {
                    ImGui::TableNextRow();
                    ImGui::TableNextColumn();
                    ImGui::TextUnformatted("Accuracy");
                    ImGui::TableNextColumn();
                    ImGui::Text("%.2f%%", training.train_accuracy);
                    ImGui::TableNextColumn();
                    if (training.val_accuracy >= 0.0) {
                        ImGui::Text("%.2f%%", training.val_accuracy);
                    } else {
                        ImGui::TextDisabled("--");
                    }
                }
                ImGui::EndTable();
            }

            for (const auto& metric : training.latest_custom_metrics) {
                ImGui::Text("%s: %.4f",
                            metric.first.c_str(),
                            metric.second);
            }
            if (training.samples_per_second > 0.0f) {
                ImGui::Text("Throughput: %.0f samples/s",
                            training.samples_per_second);
            }
            if (!training.is_training && training.total_training_time > 0.0f) {
                ImGui::Text("Run time: %.1f s", training.total_training_time);
            }
            if (!training.is_training && training.checkpoint_epoch > 0) {
                ImGui::Text("Restored checkpoint epoch: %d",
                            training.checkpoint_epoch);
            }
            ImGui::TextDisabled("Metric samples: %zu", training.metric_points);
        }

        ImGui::Spacing();
        ImGui::Separator();
        const bool busy = has_training_panel &&
                          (training.is_training || training.is_preparing);
        ImGui::Text("Engine status:");
        ImGui::SameLine();
        ImGui::TextColored(
            busy ? ImVec4(0.35f, 1.0f, 0.4f, 1.0f)
                 : ImVec4(0.7f, 0.7f, 0.7f, 1.0f),
            "%s",
            training.is_preparing ? "Preparing"
                                  : (training.is_training ? "Training" : "Ready"));
    }
    ImGui::End();
}

void Viewport::RefreshDevices() {
    devices_initialized_ = false;
}

}  // namespace gui
