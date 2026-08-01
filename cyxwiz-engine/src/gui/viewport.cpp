#include "viewport.h"
#include "panels/training_plot_panel.h"

#include <algorithm>
#include <cstddef>
#include <exception>
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
        case 3: return "Metal";
        case 4: return "Vulkan";
        default: return "Unknown";
    }
}

ImVec4 DeviceTypeColor(int type) {
    switch (type) {
        case 0: return ImVec4(0.5f, 0.8f, 1.0f, 1.0f);
        case 1: return ImVec4(0.35f, 0.95f, 0.45f, 1.0f);
        case 2: return ImVec4(1.0f, 0.75f, 0.35f, 1.0f);
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

            auto* active_device = cyxwiz::Device::GetCurrentDevice();
            if (active_device) {
                const int active_type =
                    static_cast<int>(active_device->GetType());
                const int active_id = active_device->GetDeviceId();
                const auto* active_info =
                    FindDevice(cached_devices_, active_type, active_id);

                ImGui::TextColored(DeviceTypeColor(active_type),
                                   "ACTIVE  %s",
                                   DeviceTypeName(active_type));
                if (ImGui::BeginTable("##active_runtime", 2,
                                      ImGuiTableFlags_SizingStretchProp |
                                          ImGuiTableFlags_BordersInnerV)) {
                    ImGui::TableSetupColumn("Label",
                                            ImGuiTableColumnFlags_WidthFixed,
                                            120.0f);
                    ImGui::TableSetupColumn("Value",
                                            ImGuiTableColumnFlags_WidthStretch);
                    SummaryRow("Backend", DeviceTypeName(active_type));
                    SummaryRow("Device",
                               active_info ? active_info->name.c_str()
                                           : "Runtime device not in discovery list");

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
                    ImGui::EndTable();
                }
            } else {
                ImGui::TextColored(ImVec4(1.0f, 0.75f, 0.3f, 1.0f),
                                   "Active ArrayFire runtime is unavailable");
            }

            ImGui::Spacing();
            ImGui::Text("Available Devices (%zu)", cached_devices_.size());
            ImGui::SameLine();
            if (ImGui::SmallButton("Refresh##viewport_devices")) {
                RefreshDevices();
            }

            if (cached_devices_.empty()) {
                ImGui::TextDisabled("No compute devices discovered");
            } else {
                const auto* active = cyxwiz::Device::GetCurrentDevice();
                for (const auto& device : cached_devices_) {
                    const bool is_active =
                        active &&
                        static_cast<int>(active->GetType()) == device.type &&
                        active->GetDeviceId() == device.device_id;
                    ImGui::BulletText("%s", device.name.c_str());
                    ImGui::SameLine();
                    ImGui::TextDisabled("[%s:%d]",
                                        DeviceTypeName(device.type),
                                        device.device_id);
                    if (is_active) {
                        ImGui::SameLine();
                        ImGui::TextColored(DeviceTypeColor(device.type),
                                           "Active");
                    }
                }
            }

            ImGui::Spacing();
            ImGui::TextWrapped(
                "Active runtime reports the current ArrayFire backend/device "
                "selection. Individual graph operators can still use an "
                "explicit CPU fallback; compiler and run diagnostics remain "
                "the source for per-operator placement.");
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
