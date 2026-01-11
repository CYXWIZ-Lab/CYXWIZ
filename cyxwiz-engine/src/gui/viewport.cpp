#include "viewport.h"
#include "panels/training_plot_panel.h"
#include <imgui.h>
#include <cyxwiz/cyxwiz.h>

namespace gui {

Viewport::Viewport() : show_window_(true), devices_initialized_(false) {
}

Viewport::~Viewport() = default;

void Viewport::Render() {
    if (!show_window_) return;

    if (ImGui::Begin("Viewport", &show_window_)) {
        ImGui::TextColored(ImVec4(0.4f, 0.8f, 1.0f, 1.0f), "System Information");
        ImGui::Separator();

        // CyxWiz Backend Version
        ImGui::Text("CyxWiz Backend: %s", cyxwiz::GetVersionString());
        ImGui::Spacing();

        // Device information
        ImGui::TextColored(ImVec4(0.3f, 1.0f, 0.3f, 1.0f), "Compute Devices");
        ImGui::Separator();

        try {
            // Use cached devices to avoid querying on every frame
            if (!devices_initialized_) {
                cached_devices_.clear();
                auto raw_devices = cyxwiz::Device::GetAvailableDevices();
                for (const auto& dev : raw_devices) {
                    CachedDeviceInfo cached;
                    cached.type = static_cast<int>(dev.type);
                    cached.device_id = dev.device_id;
                    cached.name = dev.name;
                    cached.memory_total = dev.memory_total;
                    cached.memory_available = dev.memory_available;
                    cached.compute_units = dev.compute_units;
                    cached.supports_fp64 = dev.supports_fp64;
                    cached.supports_fp16 = dev.supports_fp16;
                    cached_devices_.push_back(cached);
                }
                devices_initialized_ = true;
            }
            const auto& devices = cached_devices_;

            if (!devices.empty()) {
                // Show first device info
                const auto& info = devices[0];

                const char* type_name = "Unknown";
                switch (info.type) {
                    case 0: type_name = "CPU"; break;
                    case 1: type_name = "CUDA"; break;
                    case 2: type_name = "OpenCL"; break;
                    case 3: type_name = "Metal"; break;
                    case 4: type_name = "Vulkan"; break;
                }

                ImGui::Text("Backend: %s", type_name);
                ImGui::Text("Device: %s", info.name.c_str());
                ImGui::Text("Device ID: %d", info.device_id);

                if (info.memory_total > 0) {
                    double mem_gb = info.memory_total / (1024.0 * 1024.0 * 1024.0);
                    ImGui::Text("Memory: %.2f GB", mem_gb);
                }

                if (info.compute_units > 0) {
                    ImGui::Text("Compute Units: %d", info.compute_units);
                }

                ImGui::Spacing();

                // Device capabilities
                ImGui::TextColored(ImVec4(0.3f, 1.0f, 0.3f, 1.0f), "Capabilities");
                ImGui::Separator();

                ImGui::BulletText("Double Precision: %s", info.supports_fp64 ? "Yes" : "No");
                ImGui::BulletText("Half Precision: %s", info.supports_fp16 ? "Yes" : "No");

                ImGui::Spacing();

                // Show all available devices
                if (devices.size() > 1) {
                    ImGui::TextColored(ImVec4(0.3f, 1.0f, 0.3f, 1.0f), "Available Devices (%zu)", devices.size());
                    ImGui::Separator();
                    for (size_t i = 0; i < devices.size(); ++i) {
                        const char* dev_type = "Unknown";
                        switch (devices[i].type) {
                            case 0: dev_type = "CPU"; break;
                            case 1: dev_type = "CUDA"; break;
                            case 2: dev_type = "OpenCL"; break;
                            case 3: dev_type = "Metal"; break;
                            case 4: dev_type = "Vulkan"; break;
                        }
                        ImGui::BulletText("%s [%s] (ID: %d)", devices[i].name.c_str(), dev_type, devices[i].device_id);
                    }
                }
            } else {
                ImGui::Text("No compute devices found");
            }

        } catch (const std::exception& e) {
            ImGui::TextColored(ImVec4(1.0f, 0.3f, 0.3f, 1.0f), "Error: %s", e.what());
        }

        ImGui::Spacing();
        ImGui::Spacing();

        // Quick tips
        ImGui::TextColored(ImVec4(0.8f, 0.8f, 0.3f, 1.0f), "Quick Tips");
        ImGui::Separator();
        ImGui::BulletText("Training Dashboard: View real-time training metrics");
        ImGui::BulletText("Script Editor: Write and run Python training scripts");
        ImGui::BulletText("Command Window: Execute Python commands interactively");
        ImGui::BulletText("Node Editor: Build ML models visually (coming soon)");

        ImGui::Spacing();
        ImGui::Spacing();

        // Training Status (if training is active)
        if (training_panel_ && training_panel_->HasData()) {
            ImGui::TextColored(ImVec4(1.0f, 0.6f, 0.2f, 1.0f), "Training Status");
            ImGui::Separator();

            int epoch = training_panel_->GetCurrentEpoch();
            double train_loss = training_panel_->GetCurrentTrainLoss();
            double val_loss = training_panel_->GetCurrentValLoss();
            double train_acc = training_panel_->GetCurrentTrainAccuracy();
            double val_acc = training_panel_->GetCurrentValAccuracy();
            size_t data_points = training_panel_->GetDataPointCount();

            // Epoch counter with animated effect
            ImGui::Text("Current Epoch: ");
            ImGui::SameLine();
            ImGui::TextColored(ImVec4(0.3f, 1.0f, 0.3f, 1.0f), "%d", epoch);

            ImGui::Spacing();

            // Loss metrics
            ImGui::Text("Training Loss: ");
            ImGui::SameLine();
            ImVec4 loss_color = train_loss < 0.1 ? ImVec4(0.3f, 1.0f, 0.3f, 1.0f) :
                               train_loss < 0.5 ? ImVec4(1.0f, 1.0f, 0.3f, 1.0f) :
                                                  ImVec4(1.0f, 0.3f, 0.3f, 1.0f);
            ImGui::TextColored(loss_color, "%.6f", train_loss);

            if (val_loss >= 0.0) {
                ImGui::Text("Validation Loss: ");
                ImGui::SameLine();
                ImGui::TextColored(ImVec4(0.4f, 0.8f, 1.0f, 1.0f), "%.6f", val_loss);
            }

            // Accuracy metrics (if available)
            if (train_acc >= 0.0) {
                ImGui::Spacing();
                ImGui::Text("Training Accuracy: ");
                ImGui::SameLine();
                ImGui::TextColored(ImVec4(0.3f, 1.0f, 0.3f, 1.0f), "%.2f%%", train_acc);
            }

            if (val_acc >= 0.0) {
                ImGui::Text("Validation Accuracy: ");
                ImGui::SameLine();
                ImGui::TextColored(ImVec4(0.4f, 0.8f, 1.0f, 1.0f), "%.2f%%", val_acc);
            }

            ImGui::Spacing();
            ImGui::Text("Data Points: %zu", data_points);

            ImGui::Spacing();
            ImGui::Spacing();
        }

        // System Status
        ImGui::TextColored(ImVec4(0.4f, 0.8f, 1.0f, 1.0f), "System Status");
        ImGui::Separator();

        const char* status = (training_panel_ && training_panel_->HasData()) ? "Training Active" : "Ready";
        ImVec4 status_color = (training_panel_ && training_panel_->HasData()) ?
                              ImVec4(0.3f, 1.0f, 0.3f, 1.0f) : ImVec4(0.7f, 0.7f, 0.7f, 1.0f);
        ImGui::Text("Status: ");
        ImGui::SameLine();
        ImGui::TextColored(status_color, "%s", status);
    }
    ImGui::End();
}

void Viewport::RefreshDevices() {
    devices_initialized_ = false;
}

} // namespace gui
