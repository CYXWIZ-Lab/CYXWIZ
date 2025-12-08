// deployment_panel.cpp - Deploy and manage model deployments
#include "gui/panels/deployment_panel.h"
#include "gui/icons.h"
#include "core/backend_manager.h"
#include <imgui.h>
#include <spdlog/spdlog.h>

namespace cyxwiz::servernode::gui {

void DeploymentPanel::Render() {
    ImGui::PushFont(GetSafeFont(FONT_LARGE));
    ImGui::Text("%s Deployment", ICON_FA_ROCKET);
    ImGui::PopFont();
    ImGui::Separator();

    // Connection status
    if (IsDaemonConnected()) {
        ImGui::TextColored(ImVec4(0.3f, 0.8f, 0.3f, 1.0f), "%s Daemon Connected", ICON_FA_LINK);
    } else {
        ImGui::TextColored(ImVec4(0.8f, 0.3f, 0.3f, 1.0f), "%s Daemon Disconnected", ICON_FA_LINK_SLASH);
        ImGui::TextDisabled("Connect to daemon to deploy models.");
        return;
    }

    ImGui::Spacing();

    // Tab bar for Deploy and Active sections
    if (ImGui::BeginTabBar("DeploymentTabs")) {
        if (ImGui::BeginTabItem(ICON_FA_ROCKET " Deploy New")) {
            RenderDeploySection();
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem(ICON_FA_SERVER " Active Deployments")) {
            RenderActiveDeployments();
            ImGui::EndTabItem();
        }
        ImGui::EndTabBar();
    }

    // Dialogs
    RenderUndeployDialog();
}

void DeploymentPanel::RenderDeploySection() {
    // Load models if not loaded
    if (!models_loaded_) {
        RefreshModels();
        models_loaded_ = true;
    }

    ImGui::Spacing();
    ImGui::Text("%s Select Model", ICON_FA_CUBE);
    ImGui::Spacing();

    // Refresh button
    if (ImGui::Button(ICON_FA_ROTATE " Refresh Models")) {
        RefreshModels();
    }

    ImGui::Spacing();

    // Model dropdown
    if (models_.empty()) {
        ImGui::TextDisabled("No models available. Add models to the models directory.");
    } else {
        ImGui::SetNextItemWidth(400);
        const char* preview = (selected_model_idx_ >= 0 && selected_model_idx_ < (int)models_.size())
            ? models_[selected_model_idx_].name.c_str()
            : "Select a model...";

        if (ImGui::BeginCombo("##ModelSelect", preview)) {
            for (int i = 0; i < (int)models_.size(); i++) {
                const auto& model = models_[i];
                bool is_selected = (selected_model_idx_ == i);

                // Show model info in dropdown
                std::string label = model.name;
                if (model.is_deployed) {
                    label += " [DEPLOYED]";
                }

                if (ImGui::Selectable(label.c_str(), is_selected)) {
                    selected_model_idx_ = i;
                }

                // Tooltip with details
                if (ImGui::IsItemHovered()) {
                    ImGui::BeginTooltip();
                    ImGui::Text("Format: %s", model.format.c_str());
                    ImGui::Text("Size: %.2f GB", model.size_bytes / (1024.0 * 1024.0 * 1024.0));
                    if (!model.architecture.empty()) {
                        ImGui::Text("Architecture: %s", model.architecture.c_str());
                    }
                    ImGui::EndTooltip();
                }

                if (is_selected) {
                    ImGui::SetItemDefaultFocus();
                }
            }
            ImGui::EndCombo();
        }
    }

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    // Server configuration
    ImGui::Text("%s Server Configuration", ICON_FA_GEAR);
    ImGui::Spacing();

    ImGui::SetNextItemWidth(150);
    ImGui::InputInt("Port", &port_);
    if (port_ < 1024) port_ = 1024;
    if (port_ > 65535) port_ = 65535;
    ImGui::SameLine();
    ImGui::TextDisabled("(1024-65535)");

    ImGui::SetNextItemWidth(150);
    ImGui::InputInt("GPU Layers", &gpu_layers_);
    if (gpu_layers_ < 0) gpu_layers_ = 0;
    if (gpu_layers_ > 100) gpu_layers_ = 100;
    ImGui::SameLine();
    ImGui::TextDisabled("(0 = CPU only)");

    ImGui::SetNextItemWidth(150);
    ImGui::InputInt("Context Size", &context_size_);
    if (context_size_ < 512) context_size_ = 512;
    if (context_size_ > 131072) context_size_ = 131072;
    ImGui::SameLine();
    ImGui::TextDisabled("(512-131072)");

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    // Success/Error messages
    if (!deploy_success_.empty()) {
        ImGui::TextColored(ImVec4(0.3f, 0.8f, 0.3f, 1.0f), "%s %s", ICON_FA_CIRCLE_CHECK, deploy_success_.c_str());
        ImGui::Spacing();
    }

    if (!deploy_error_.empty()) {
        ImGui::TextColored(ImVec4(1.0f, 0.3f, 0.3f, 1.0f), "%s %s", ICON_FA_TRIANGLE_EXCLAMATION, deploy_error_.c_str());
        ImGui::Spacing();
    }

    // Deploy button
    bool can_deploy = selected_model_idx_ >= 0 && selected_model_idx_ < (int)models_.size() && !deploying_;
    if (!can_deploy) {
        ImGui::BeginDisabled();
    }

    if (ImGui::Button(ICON_FA_ROCKET " Deploy Model", ImVec2(200, 40))) {
        auto* client = GetDaemonClient();
        if (client && client->IsConnected()) {
            const auto& model = models_[selected_model_idx_];
            deploying_ = true;
            deploy_error_.clear();
            deploy_success_.clear();

            std::string deployment_id, error;
            if (client->DeployModel(model.path, port_, gpu_layers_, context_size_,
                                   deployment_id, error)) {
                deploy_success_ = "Model deployed successfully on port " + std::to_string(port_);
                spdlog::info("Deployed model {} on port {}, id={}", model.name, port_, deployment_id);
                RefreshModels();  // Refresh to show deployed status
            } else {
                deploy_error_ = error.empty() ? "Deployment failed" : error;
                spdlog::error("Deploy failed: {}", deploy_error_);
            }
            deploying_ = false;
        }
    }

    if (!can_deploy) {
        ImGui::EndDisabled();
    }

    if (deploying_) {
        ImGui::SameLine();
        ImGui::Text("%s Deploying...", ICON_FA_SPINNER);
    }
}

void DeploymentPanel::RenderActiveDeployments() {
    // Load deployments if not loaded
    if (!deployments_loaded_) {
        RefreshDeployments();
        deployments_loaded_ = true;
    }

    ImGui::Spacing();

    // Refresh button
    if (ImGui::Button(ICON_FA_ROTATE " Refresh")) {
        RefreshDeployments();
    }

    ImGui::Spacing();

    if (deployments_.empty()) {
        ImGui::TextDisabled("No active deployments.");
        ImGui::Text("Deploy a model from the 'Deploy New' tab.");
        return;
    }

    // Deployments table
    if (ImGui::BeginTable("DeploymentsTable", 7, ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg |
                          ImGuiTableFlags_Resizable | ImGuiTableFlags_ScrollY)) {
        ImGui::TableSetupColumn("Model", ImGuiTableColumnFlags_WidthStretch);
        ImGui::TableSetupColumn("Status", ImGuiTableColumnFlags_WidthFixed, 80);
        ImGui::TableSetupColumn("Port", ImGuiTableColumnFlags_WidthFixed, 60);
        ImGui::TableSetupColumn("GPU", ImGuiTableColumnFlags_WidthFixed, 50);
        ImGui::TableSetupColumn("Context", ImGuiTableColumnFlags_WidthFixed, 70);
        ImGui::TableSetupColumn("Requests", ImGuiTableColumnFlags_WidthFixed, 80);
        ImGui::TableSetupColumn("Actions", ImGuiTableColumnFlags_WidthFixed, 80);
        ImGui::TableHeadersRow();

        for (const auto& deploy : deployments_) {
            ImGui::TableNextRow();
            ImGui::PushID(deploy.id.c_str());

            // Model name
            ImGui::TableNextColumn();
            ImGui::Text("%s %s", ICON_FA_CUBE, deploy.model_name.c_str());
            if (ImGui::IsItemHovered()) {
                ImGui::BeginTooltip();
                ImGui::Text("Path: %s", deploy.model_path.c_str());
                ImGui::Text("Format: %s", deploy.format.c_str());
                ImGui::Text("ID: %s", deploy.id.c_str());
                ImGui::EndTooltip();
            }

            // Status
            ImGui::TableNextColumn();
            ImGui::TextColored(GetStatusColor(deploy.status), "%s", GetStatusText(deploy.status));

            // Port
            ImGui::TableNextColumn();
            ImGui::Text("%d", deploy.port);

            // GPU layers
            ImGui::TableNextColumn();
            ImGui::Text("%d", deploy.gpu_layers);

            // Context size
            ImGui::TableNextColumn();
            ImGui::Text("%d", deploy.context_size);

            // Request count
            ImGui::TableNextColumn();
            ImGui::Text("%lld", (long long)deploy.request_count);

            // Actions
            ImGui::TableNextColumn();
            if (ImGui::SmallButton(ICON_FA_STOP " Stop")) {
                pending_undeploy_id_ = deploy.id;
                pending_undeploy_name_ = deploy.model_name;
                undeploy_error_.clear();
                show_undeploy_dialog_ = true;
            }

            ImGui::PopID();
        }

        ImGui::EndTable();
    }
}

void DeploymentPanel::RenderUndeployDialog() {
    if (!show_undeploy_dialog_) return;

    ImGui::OpenPopup("Stop Deployment?");

    ImVec2 center = ImGui::GetMainViewport()->GetCenter();
    ImGui::SetNextWindowPos(center, ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));

    if (ImGui::BeginPopupModal("Stop Deployment?", &show_undeploy_dialog_, ImGuiWindowFlags_AlwaysAutoResize)) {
        ImGui::Text("%s Are you sure you want to stop:", ICON_FA_TRIANGLE_EXCLAMATION);
        ImGui::Spacing();
        ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.3f, 1.0f), "  %s", pending_undeploy_name_.c_str());
        ImGui::Spacing();
        ImGui::Text("This will terminate the inference server.");
        ImGui::Spacing();

        if (!undeploy_error_.empty()) {
            ImGui::TextColored(ImVec4(1.0f, 0.3f, 0.3f, 1.0f), "%s %s", ICON_FA_TRIANGLE_EXCLAMATION, undeploy_error_.c_str());
            ImGui::Spacing();
        }

        ImGui::Separator();
        ImGui::Spacing();

        if (ImGui::Button(ICON_FA_STOP " Stop", ImVec2(120, 0))) {
            auto* client = GetDaemonClient();
            if (client && client->IsConnected()) {
                std::string error;
                if (client->UndeployModel(pending_undeploy_id_, error)) {
                    spdlog::info("Undeployed: {}", pending_undeploy_name_);
                    show_undeploy_dialog_ = false;
                    RefreshDeployments();
                    RefreshModels();
                } else {
                    undeploy_error_ = error.empty() ? "Undeploy failed" : error;
                }
            } else {
                undeploy_error_ = "Daemon not connected";
            }
        }

        ImGui::SameLine();
        if (ImGui::Button("Cancel", ImVec2(120, 0))) {
            show_undeploy_dialog_ = false;
            undeploy_error_.clear();
        }

        ImGui::EndPopup();
    }
}

void DeploymentPanel::RefreshModels() {
    models_.clear();
    auto* client = GetDaemonClient();
    if (client && client->IsConnected()) {
        client->ListLocalModels(models_);
        spdlog::debug("Loaded {} models", models_.size());
    }
    // Reset selection if invalid
    if (selected_model_idx_ >= (int)models_.size()) {
        selected_model_idx_ = -1;
    }
}

void DeploymentPanel::RefreshDeployments() {
    deployments_.clear();
    auto* client = GetDaemonClient();
    if (client && client->IsConnected()) {
        client->ListDeployments(deployments_);
        spdlog::debug("Loaded {} deployments", deployments_.size());
    }
}

const char* DeploymentPanel::GetStatusText(int status) {
    switch (status) {
        case 1: return "Loading";
        case 2: return "Running";
        case 3: return "Stopped";
        case 4: return "Error";
        default: return "Unknown";
    }
}

ImVec4 DeploymentPanel::GetStatusColor(int status) {
    switch (status) {
        case 1: return ImVec4(1.0f, 0.8f, 0.3f, 1.0f);  // Yellow - loading
        case 2: return ImVec4(0.3f, 0.8f, 0.3f, 1.0f);  // Green - running
        case 3: return ImVec4(0.5f, 0.5f, 0.5f, 1.0f);  // Gray - stopped
        case 4: return ImVec4(1.0f, 0.3f, 0.3f, 1.0f);  // Red - error
        default: return ImVec4(1.0f, 1.0f, 1.0f, 1.0f);
    }
}

} // namespace cyxwiz::servernode::gui
