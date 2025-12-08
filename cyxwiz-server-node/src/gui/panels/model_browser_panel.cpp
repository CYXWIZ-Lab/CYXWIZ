// model_browser_panel.cpp - Browse and manage models with daemon integration
#include "gui/panels/model_browser_panel.h"
#include "gui/icons.h"
#include "core/backend_manager.h"
#include <imgui.h>
#include <filesystem>
#include <spdlog/spdlog.h>

namespace cyxwiz::servernode::gui {

void ModelBrowserPanel::Render() {
    ImGui::PushFont(GetSafeFont(FONT_LARGE));
    ImGui::Text("%s Models", ICON_FA_CUBE);
    ImGui::PopFont();
    ImGui::Separator();

    // Search bar
    ImGui::SetNextItemWidth(300);
    ImGui::InputTextWithHint("##Search", "Search models...", search_query_, sizeof(search_query_));
    ImGui::SameLine();
    if (ImGui::Button(ICON_FA_ROTATE " Refresh")) {
        RefreshModels();
    }

    // Connection status indicator
    ImGui::SameLine();
    if (IsDaemonConnected()) {
        ImGui::TextColored(ImVec4(0.3f, 0.8f, 0.3f, 1.0f), "%s Daemon", ICON_FA_LINK);
    } else {
        ImGui::TextColored(ImVec4(0.8f, 0.5f, 0.3f, 1.0f), "%s Local", ICON_FA_FOLDER);
    }

    ImGui::Spacing();

    // Tab bar
    if (ImGui::BeginTabBar("ModelTabs")) {
        if (ImGui::BeginTabItem("Local Models")) {
            selected_tab_ = 0;
            RenderLocalModels();
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("Downloaded")) {
            selected_tab_ = 1;
            RenderDownloaded();
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("Marketplace")) {
            selected_tab_ = 2;
            RenderMarketplace();
            ImGui::EndTabItem();
        }
        ImGui::EndTabBar();
    }

    // Dialogs
    RenderDeployDialog();
    RenderDeleteConfirmDialog();
}

void ModelBrowserPanel::RenderLocalModels() {
    if (!models_scanned_) {
        ScanLocalModels();
        models_scanned_ = true;
    }

    if (local_models_.empty()) {
        ImGui::TextDisabled("No local models found.");
        ImGui::Text("Place model files (.gguf, .onnx, .safetensors) in the models directory.");
        return;
    }

    // Filter by search query
    std::string query_lower = search_query_;
    std::transform(query_lower.begin(), query_lower.end(), query_lower.begin(), ::tolower);

    for (const auto& model : local_models_) {
        // Apply search filter
        if (!query_lower.empty()) {
            std::string name_lower = model.name;
            std::transform(name_lower.begin(), name_lower.end(), name_lower.begin(), ::tolower);
            if (name_lower.find(query_lower) == std::string::npos) {
                continue;
            }
        }

        ImGui::PushID(model.path.c_str());

        // Model card
        ImGui::BeginGroup();

        // Model name and format
        ImGui::Text("%s %s", ICON_FA_FILE, model.name.c_str());
        ImGui::SameLine(ImGui::GetContentRegionAvail().x - 180);

        // Deployed badge
        if (model.is_deployed) {
            ImGui::TextColored(ImVec4(0.3f, 0.8f, 0.3f, 1.0f), "%s Deployed", ICON_FA_CIRCLE_CHECK);
        } else {
            ImGui::TextDisabled("%s", model.format.c_str());
        }

        // Size and path
        ImGui::TextDisabled("Size: %.2f GB", model.size_bytes / (1024.0 * 1024.0 * 1024.0));
        ImGui::SameLine();
        ImGui::TextDisabled("| %s", model.architecture.empty() ? "Unknown" : model.architecture.c_str());

        // Action buttons
        if (!model.is_deployed) {
            if (ImGui::Button(ICON_FA_ROCKET " Deploy")) {
                pending_deploy_model_ = model.path;
                deploy_error_.clear();
                show_deploy_dialog_ = true;
            }
        } else {
            ImGui::BeginDisabled();
            ImGui::Button(ICON_FA_ROCKET " Deploy");
            ImGui::EndDisabled();
            if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) {
                ImGui::SetTooltip("Model is already deployed");
            }
        }

        ImGui::SameLine();
        if (ImGui::Button(ICON_FA_TRASH " Delete")) {
            pending_delete_model_ = model.path;
            delete_error_.clear();
            show_delete_dialog_ = true;
        }

        ImGui::EndGroup();
        ImGui::Separator();
        ImGui::PopID();
    }
}

void ModelBrowserPanel::RenderDownloaded() {
    ImGui::TextDisabled("Downloaded models from CyxWiz Hub");
    ImGui::Spacing();

    // Placeholder - would fetch from daemon's downloaded models list
    ImGui::Text("No downloaded models yet.");
    ImGui::Spacing();
    if (ImGui::Button(ICON_FA_CLOUD_ARROW_DOWN " Browse Hub")) {
        // Switch to marketplace tab
        selected_tab_ = 2;
    }
}

void ModelBrowserPanel::RenderMarketplace() {
    ImGui::TextDisabled("Browse models from CyxWiz Marketplace");
    ImGui::Spacing();

    // Placeholder - would connect to marketplace API
    ImGui::Text("Marketplace coming soon...");
    ImGui::Text("Browse and download models from the CyxWiz community.");
}

void ModelBrowserPanel::RenderDeployDialog() {
    if (!show_deploy_dialog_) return;

    ImGui::OpenPopup("Deploy Model");

    ImVec2 center = ImGui::GetMainViewport()->GetCenter();
    ImGui::SetNextWindowPos(center, ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));
    ImGui::SetNextWindowSize(ImVec2(450, 350), ImGuiCond_Appearing);

    if (ImGui::BeginPopupModal("Deploy Model", &show_deploy_dialog_, ImGuiWindowFlags_NoResize)) {
        // Model path
        std::filesystem::path p(pending_deploy_model_);
        ImGui::Text("%s Model: %s", ICON_FA_FILE, p.filename().string().c_str());
        ImGui::Separator();
        ImGui::Spacing();

        // Server configuration
        ImGui::Text("Server Configuration");
        ImGui::Spacing();

        ImGui::SetNextItemWidth(150);
        ImGui::InputInt("Port", &deploy_port_);
        if (deploy_port_ < 1024) deploy_port_ = 1024;
        if (deploy_port_ > 65535) deploy_port_ = 65535;

        ImGui::SetNextItemWidth(150);
        ImGui::InputInt("GPU Layers", &deploy_gpu_layers_);
        if (deploy_gpu_layers_ < 0) deploy_gpu_layers_ = 0;
        if (deploy_gpu_layers_ > 100) deploy_gpu_layers_ = 100;
        ImGui::SameLine();
        ImGui::TextDisabled("(0 = CPU only)");

        ImGui::SetNextItemWidth(150);
        ImGui::InputInt("Context Size", &deploy_context_size_);
        if (deploy_context_size_ < 512) deploy_context_size_ = 512;
        if (deploy_context_size_ > 131072) deploy_context_size_ = 131072;

        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();

        // Error display
        if (!deploy_error_.empty()) {
            ImGui::TextColored(ImVec4(1.0f, 0.3f, 0.3f, 1.0f), "%s %s", ICON_FA_TRIANGLE_EXCLAMATION, deploy_error_.c_str());
            ImGui::Spacing();
        }

        // Deploy button
        if (deploying_) {
            ImGui::BeginDisabled();
            ImGui::Button("Deploying...", ImVec2(120, 0));
            ImGui::EndDisabled();
            ImGui::SameLine();
            ImGui::Text("%s", ICON_FA_SPINNER);
        } else {
            if (ImGui::Button(ICON_FA_ROCKET " Deploy", ImVec2(120, 0))) {
                auto* client = GetDaemonClient();
                if (client && client->IsConnected()) {
                    deploying_ = true;
                    std::string deployment_id, error;
                    if (client->DeployModel(pending_deploy_model_, deploy_port_,
                                           deploy_gpu_layers_, deploy_context_size_,
                                           deployment_id, error)) {
                        spdlog::info("Model deployed: {} on port {}", pending_deploy_model_, deploy_port_);
                        show_deploy_dialog_ = false;
                        RefreshModels();  // Refresh to show deployed status
                    } else {
                        deploy_error_ = error.empty() ? "Deployment failed" : error;
                        spdlog::error("Deploy failed: {}", deploy_error_);
                    }
                    deploying_ = false;
                } else {
                    deploy_error_ = "Daemon not connected. Start the daemon first.";
                }
            }
        }

        ImGui::SameLine();
        if (ImGui::Button("Cancel", ImVec2(120, 0))) {
            show_deploy_dialog_ = false;
            deploy_error_.clear();
        }

        ImGui::EndPopup();
    }
}

void ModelBrowserPanel::RenderDeleteConfirmDialog() {
    if (!show_delete_dialog_) return;

    ImGui::OpenPopup("Delete Model?");

    ImVec2 center = ImGui::GetMainViewport()->GetCenter();
    ImGui::SetNextWindowPos(center, ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));

    if (ImGui::BeginPopupModal("Delete Model?", &show_delete_dialog_, ImGuiWindowFlags_AlwaysAutoResize)) {
        std::filesystem::path p(pending_delete_model_);
        ImGui::Text("%s Are you sure you want to delete:", ICON_FA_TRIANGLE_EXCLAMATION);
        ImGui::Spacing();
        ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.3f, 1.0f), "  %s", p.filename().string().c_str());
        ImGui::Spacing();
        ImGui::Text("This action cannot be undone.");
        ImGui::Spacing();

        if (!delete_error_.empty()) {
            ImGui::TextColored(ImVec4(1.0f, 0.3f, 0.3f, 1.0f), "%s %s", ICON_FA_TRIANGLE_EXCLAMATION, delete_error_.c_str());
            ImGui::Spacing();
        }

        ImGui::Separator();
        ImGui::Spacing();

        if (ImGui::Button(ICON_FA_TRASH " Delete", ImVec2(120, 0))) {
            auto* client = GetDaemonClient();
            if (client && client->IsConnected()) {
                std::string error;
                if (client->DeleteModel(pending_delete_model_, error)) {
                    spdlog::info("Model deleted: {}", pending_delete_model_);
                    show_delete_dialog_ = false;
                    RefreshModels();
                } else {
                    delete_error_ = error.empty() ? "Delete failed" : error;
                }
            } else {
                // Fallback: delete locally
                try {
                    std::filesystem::remove(pending_delete_model_);
                    spdlog::info("Model deleted locally: {}", pending_delete_model_);
                    show_delete_dialog_ = false;
                    RefreshModels();
                } catch (const std::exception& e) {
                    delete_error_ = e.what();
                }
            }
        }

        ImGui::SameLine();
        if (ImGui::Button("Cancel", ImVec2(120, 0))) {
            show_delete_dialog_ = false;
            delete_error_.clear();
        }

        ImGui::EndPopup();
    }
}

void ModelBrowserPanel::ScanLocalModels() {
    local_models_.clear();

    // Try daemon first
    auto* client = GetDaemonClient();
    if (client && client->IsConnected()) {
        if (client->ListLocalModels(local_models_)) {
            spdlog::debug("Loaded {} models from daemon", local_models_.size());
            return;
        }
    }

    // Fallback to local filesystem scan
    auto& config = GetBackend().GetConfig();
    std::string models_dir = config.models_directory;

    if (!std::filesystem::exists(models_dir)) {
        return;
    }

    for (const auto& entry : std::filesystem::directory_iterator(models_dir)) {
        if (!entry.is_regular_file()) continue;

        std::string ext = entry.path().extension().string();
        std::string format;

        if (ext == ".gguf") format = "GGUF";
        else if (ext == ".onnx") format = "ONNX";
        else if (ext == ".safetensors") format = "SafeTensors";
        else if (ext == ".pt" || ext == ".pth") format = "PyTorch";
        else continue;

        ipc::ModelInfo info;
        info.name = entry.path().stem().string();
        info.path = entry.path().string();
        info.format = format;
        info.size_bytes = entry.file_size();
        info.is_deployed = false;
        info.architecture = "";

        local_models_.push_back(info);
    }
}

void ModelBrowserPanel::RefreshModels() {
    models_scanned_ = false;

    // Also trigger a rescan on daemon if connected
    auto* client = GetDaemonClient();
    if (client && client->IsConnected()) {
        std::vector<std::string> dirs;
        int found = 0;
        client->ScanModels(dirs, found);  // Empty dirs = use default
    }
}

} // namespace cyxwiz::servernode::gui
