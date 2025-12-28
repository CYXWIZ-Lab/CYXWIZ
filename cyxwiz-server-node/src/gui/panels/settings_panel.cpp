// settings_panel.cpp
#include "gui/panels/settings_panel.h"
#include "gui/icons.h"
#include <imgui.h>
#include <thread>
#include <cstring>

namespace cyxwiz::servernode::gui {

void SettingsPanel::Render() {
    if (!config_loaded_) {
        config_ = GetBackend().GetConfig();
        // Initialize daemon address from config
        strncpy(daemon_address_, config_.ipc_address.c_str(), sizeof(daemon_address_) - 1);
        daemon_address_[sizeof(daemon_address_) - 1] = '\0';
        config_loaded_ = true;
    }

    ImGui::PushFont(ImGui::GetIO().Fonts->Fonts[2]);
    ImGui::Text("%s Settings", ICON_FA_GEAR);
    ImGui::PopFont();
    ImGui::Separator();
    ImGui::Spacing();

    if (ImGui::BeginTabBar("SettingsTabs")) {
        if (ImGui::BeginTabItem("General")) {
            ImGui::Text("Node Name");
            char name_buf[256];
            strncpy(name_buf, config_.node_name.c_str(), sizeof(name_buf));
            if (ImGui::InputText("##NodeName", name_buf, sizeof(name_buf))) {
                config_.node_name = name_buf;
                config_changed_ = true;
            }

            ImGui::Text("Region");
            char region_buf[64];
            strncpy(region_buf, config_.region.c_str(), sizeof(region_buf));
            if (ImGui::InputText("##Region", region_buf, sizeof(region_buf))) {
                config_.region = region_buf;
                config_changed_ = true;
            }
            ImGui::EndTabItem();
        }

        if (ImGui::BeginTabItem("Network")) {
            ImGui::Text("Central Server");
            char server_buf[256];
            strncpy(server_buf, config_.central_server.c_str(), sizeof(server_buf));
            if (ImGui::InputText("##CentralServer", server_buf, sizeof(server_buf))) {
                config_.central_server = server_buf;
                config_changed_ = true;
            }

            if (ImGui::InputInt("HTTP API Port", &config_.http_api_port)) {
                config_changed_ = true;
            }

            if (ImGui::Checkbox("Enable TLS", &config_.enable_tls)) {
                config_changed_ = true;
            }
            ImGui::EndTabItem();
        }

        // Remote Connection Tab - for connecting GUI to remote daemon
        if (ImGui::BeginTabItem("Remote Connection")) {
            RenderRemoteConnectionTab();
            ImGui::EndTabItem();
        }

        if (ImGui::BeginTabItem("Training")) {
            if (ImGui::Checkbox("Enable Training", &config_.training_enabled)) {
                config_changed_ = true;
            }

            if (ImGui::InputInt("Max Concurrent Jobs", &config_.max_concurrent_jobs)) {
                config_changed_ = true;
            }

            if (ImGui::SliderFloat("GPU Allocation", &config_.gpu_allocation, 0.0f, 1.0f, "%.0f%%")) {
                config_changed_ = true;
            }
            ImGui::EndTabItem();
        }

        if (ImGui::BeginTabItem("API")) {
            if (ImGui::Checkbox("Require Authentication", &config_.api_require_auth)) {
                config_changed_ = true;
            }

            if (ImGui::InputInt("Default Rate Limit", &config_.api_default_rate_limit)) {
                config_changed_ = true;
            }
            ImGui::EndTabItem();
        }

        ImGui::EndTabBar();
    }

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    if (config_changed_) {
        if (ImGui::Button(ICON_FA_FLOPPY_DISK " Save Changes")) {
            GetBackend().UpdateConfig(config_);
            config_changed_ = false;
        }
        ImGui::SameLine();
        if (ImGui::Button(ICON_FA_ROTATE_LEFT " Discard")) {
            config_ = GetBackend().GetConfig();
            config_changed_ = false;
        }
    }
}

void SettingsPanel::RenderRemoteConnectionTab() {
    ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f),
        "Configure connection to a remote CyxWiz Server Daemon");
    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    // Daemon Address
    ImGui::Text("%s Daemon Address", ICON_FA_SERVER);
    ImGui::SetNextItemWidth(-1);
    ImGui::InputText("##DaemonAddress", daemon_address_, sizeof(daemon_address_));
    ImGui::TextColored(ImVec4(0.5f, 0.5f, 0.5f, 1.0f), "Format: hostname:port (e.g., 192.168.1.100:50054)");
    ImGui::Spacing();

    // TLS Settings Section
    ImGui::Spacing();
    if (ImGui::CollapsingHeader(ICON_FA_LOCK " TLS Security Settings", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::Indent();

        // Enable TLS checkbox
        ImGui::Checkbox("Enable TLS Encryption", &tls_settings_.enabled);
        if (tls_settings_.enabled) {
            ImGui::TextColored(ImVec4(0.3f, 0.8f, 0.3f, 1.0f), ICON_FA_LOCK " Secure connection");
        } else {
            ImGui::TextColored(ImVec4(0.8f, 0.5f, 0.2f, 1.0f), ICON_FA_TRIANGLE_EXCLAMATION " Insecure connection");
        }
        ImGui::Spacing();

        // TLS options (only show when TLS is enabled)
        if (tls_settings_.enabled) {
            // CA Certificate Path
            ImGui::Text("CA Certificate Path");
            ImGui::SetNextItemWidth(-80);
            if (ImGui::InputText("##CACertPath", ca_cert_path_, sizeof(ca_cert_path_))) {
                tls_settings_.ca_cert_path = ca_cert_path_;
            }
            ImGui::SameLine();
            if (ImGui::Button(ICON_FA_FOLDER_OPEN "##BrowseCA")) {
                // TODO: File browser dialog
            }
            ImGui::TextColored(ImVec4(0.5f, 0.5f, 0.5f, 1.0f), "Required for TLS verification");
            ImGui::Spacing();

            // Mutual TLS (optional)
            if (ImGui::TreeNode("Mutual TLS (Optional)")) {
                ImGui::TextColored(ImVec4(0.5f, 0.5f, 0.5f, 1.0f),
                    "For servers requiring client certificate authentication");
                ImGui::Spacing();

                // Client Certificate
                ImGui::Text("Client Certificate Path");
                ImGui::SetNextItemWidth(-80);
                if (ImGui::InputText("##ClientCertPath", client_cert_path_, sizeof(client_cert_path_))) {
                    tls_settings_.client_cert_path = client_cert_path_;
                }
                ImGui::SameLine();
                if (ImGui::Button(ICON_FA_FOLDER_OPEN "##BrowseClientCert")) {
                    // TODO: File browser dialog
                }

                // Client Key
                ImGui::Text("Client Private Key Path");
                ImGui::SetNextItemWidth(-80);
                if (ImGui::InputText("##ClientKeyPath", client_key_path_, sizeof(client_key_path_))) {
                    tls_settings_.client_key_path = client_key_path_;
                }
                ImGui::SameLine();
                if (ImGui::Button(ICON_FA_FOLDER_OPEN "##BrowseClientKey")) {
                    // TODO: File browser dialog
                }

                ImGui::TreePop();
            }
            ImGui::Spacing();

            // Advanced TLS Options
            if (ImGui::TreeNode("Advanced Options")) {
                // Target Name Override
                ImGui::Text("Server Name Override");
                ImGui::SetNextItemWidth(-1);
                if (ImGui::InputText("##TargetNameOverride", target_name_override_, sizeof(target_name_override_))) {
                    tls_settings_.target_name_override = target_name_override_;
                }
                ImGui::TextColored(ImVec4(0.5f, 0.5f, 0.5f, 1.0f),
                    "Override expected server name for certificate verification");
                ImGui::Spacing();

                // Skip Verification (development only)
                ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.9f, 0.3f, 0.3f, 1.0f));
                if (ImGui::Checkbox("Skip Certificate Verification", &tls_settings_.skip_verification)) {
                    if (tls_settings_.skip_verification) {
                        ImGui::OpenPopup("SecurityWarning");
                    }
                }
                ImGui::PopStyleColor();
                ImGui::TextColored(ImVec4(0.9f, 0.3f, 0.3f, 1.0f),
                    ICON_FA_TRIANGLE_EXCLAMATION " DEVELOPMENT ONLY - Not for production!");

                // Security warning popup
                if (ImGui::BeginPopupModal("SecurityWarning", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
                    ImGui::Text(ICON_FA_TRIANGLE_EXCLAMATION " Security Warning");
                    ImGui::Separator();
                    ImGui::TextWrapped(
                        "Skipping certificate verification makes the connection vulnerable to "
                        "man-in-the-middle attacks. Only use this option for development and testing.");
                    ImGui::Spacing();
                    if (ImGui::Button("I Understand", ImVec2(120, 0))) {
                        ImGui::CloseCurrentPopup();
                    }
                    ImGui::SameLine();
                    if (ImGui::Button("Cancel", ImVec2(120, 0))) {
                        tls_settings_.skip_verification = false;
                        ImGui::CloseCurrentPopup();
                    }
                    ImGui::EndPopup();
                }

                ImGui::TreePop();
            }
        }

        ImGui::Unindent();
    }

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    // Connection Test Section
    ImGui::Text("%s Connection Test", ICON_FA_PLUG);
    ImGui::Spacing();

    if (testing_connection_) {
        ImGui::Text(ICON_FA_SPINNER " Testing connection...");
    } else {
        if (ImGui::Button(ICON_FA_PLUG " Test Connection", ImVec2(150, 0))) {
            TestRemoteConnection();
        }
    }

    // Show test result
    if (test_result_valid_) {
        ImGui::SameLine();
        if (test_result_success_) {
            ImGui::TextColored(ImVec4(0.3f, 0.8f, 0.3f, 1.0f),
                ICON_FA_CHECK " Connected successfully!");
        } else {
            ImGui::TextColored(ImVec4(0.9f, 0.3f, 0.3f, 1.0f),
                ICON_FA_XMARK " %s", test_result_message_.c_str());
        }
    }

    ImGui::Spacing();

    // Connection status indicator
    ImGui::Separator();
    ImGui::Spacing();
    ImGui::Text("Current Status:");
    ImGui::SameLine();

    auto* daemon_client = GetDaemonClient();
    bool is_connected = daemon_client && daemon_client->IsConnected();
    if (is_connected) {
        bool is_tls = daemon_client->IsTLSEnabled();
        if (is_tls) {
            ImGui::TextColored(ImVec4(0.3f, 0.8f, 0.3f, 1.0f),
                ICON_FA_LOCK " Connected (TLS)");
        } else {
            ImGui::TextColored(ImVec4(0.8f, 0.8f, 0.3f, 1.0f),
                ICON_FA_PLUG " Connected (Insecure)");
        }
    } else {
        ImGui::TextColored(ImVec4(0.5f, 0.5f, 0.5f, 1.0f),
            ICON_FA_PLUG " Not connected");
    }
}

void SettingsPanel::TestRemoteConnection() {
    testing_connection_ = true;
    test_result_valid_ = false;

    // Update TLS settings from buffers
    tls_settings_.ca_cert_path = ca_cert_path_;
    tls_settings_.client_cert_path = client_cert_path_;
    tls_settings_.client_key_path = client_key_path_;
    tls_settings_.target_name_override = target_name_override_;

    // Run test in a detached thread to avoid blocking the UI
    std::thread([this]() {
        std::string error;
        bool success = ipc::DaemonClient::TestConnection(
            daemon_address_,
            tls_settings_,
            error,
            5  // 5 second timeout
        );

        // Update result (safe because ImGui is single-threaded for rendering)
        test_result_success_ = success;
        test_result_message_ = success ? "Success" : error;
        test_result_valid_ = true;
        testing_connection_ = false;
    }).detach();
}

} // namespace cyxwiz::servernode::gui
