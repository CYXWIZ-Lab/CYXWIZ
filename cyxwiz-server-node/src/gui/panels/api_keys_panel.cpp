// api_keys_panel.cpp - API key management with daemon integration
#include "gui/panels/api_keys_panel.h"
#include "gui/icons.h"
#include <imgui.h>
#include <spdlog/spdlog.h>
#include <chrono>
#include <ctime>
#include <iomanip>
#include <sstream>

namespace cyxwiz::servernode::gui {

void APIKeysPanel::Render() {
    ImGui::PushFont(GetSafeFont(FONT_LARGE));
    ImGui::Text("%s API Keys", ICON_FA_KEY);
    ImGui::PopFont();
    ImGui::Separator();

    // Connection status
    if (IsDaemonConnected()) {
        ImGui::TextColored(ImVec4(0.3f, 0.8f, 0.3f, 1.0f), "%s Daemon Connected", ICON_FA_LINK);
    } else {
        ImGui::TextColored(ImVec4(0.8f, 0.3f, 0.3f, 1.0f), "%s Daemon Disconnected", ICON_FA_LINK_SLASH);
        ImGui::TextDisabled("Connect to daemon to manage API keys.");
        return;
    }

    ImGui::Spacing();

    // Generate new key button
    if (ImGui::Button(ICON_FA_PLUS " Generate New Key", ImVec2(180, 0))) {
        show_generate_dialog_ = true;
        memset(new_key_name_, 0, sizeof(new_key_name_));
        new_key_rate_limit_ = 100;
        generate_error_.clear();
    }

    ImGui::SameLine();
    if (ImGui::Button(ICON_FA_ROTATE " Refresh")) {
        RefreshKeys();
    }

    ImGui::Spacing();
    ImGui::TextDisabled("API keys grant access to deployed models via REST API.");
    ImGui::Spacing();

    // Key list
    RenderKeyList();

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    // Endpoint info
    ImGui::Text("%s Endpoint Information", ICON_FA_CIRCLE_INFO);
    ImGui::TextDisabled("Base URL: http://localhost:<port>/v1");
    ImGui::TextDisabled("Header: Authorization: Bearer cyx_sk_...");
    ImGui::Spacing();

    ImGui::PushFont(GetSafeFont(FONT_MONO));
    ImGui::TextDisabled("curl -H \"Authorization: Bearer <key>\" http://localhost:8082/v1/models");
    ImGui::PopFont();

    // Dialogs
    RenderGenerateDialog();
    RenderNewKeyDialog();
    RenderRevokeDialog();
}

void APIKeysPanel::RenderKeyList() {
    // Load keys if not loaded
    if (!keys_loaded_) {
        RefreshKeys();
        keys_loaded_ = true;
    }

    if (keys_.empty()) {
        ImGui::TextDisabled("No API keys generated yet.");
        ImGui::Text("Click 'Generate New Key' to create your first API key.");
        return;
    }

    // Keys table
    if (ImGui::BeginTable("APIKeysTable", 6, ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg |
                          ImGuiTableFlags_Resizable | ImGuiTableFlags_ScrollY)) {
        ImGui::TableSetupColumn("Name", ImGuiTableColumnFlags_WidthStretch);
        ImGui::TableSetupColumn("Key Prefix", ImGuiTableColumnFlags_WidthFixed, 120);
        ImGui::TableSetupColumn("Rate Limit", ImGuiTableColumnFlags_WidthFixed, 90);
        ImGui::TableSetupColumn("Requests", ImGuiTableColumnFlags_WidthFixed, 80);
        ImGui::TableSetupColumn("Last Used", ImGuiTableColumnFlags_WidthFixed, 120);
        ImGui::TableSetupColumn("Actions", ImGuiTableColumnFlags_WidthFixed, 80);
        ImGui::TableHeadersRow();

        for (const auto& key : keys_) {
            ImGui::TableNextRow();
            ImGui::PushID(key.id.c_str());

            // Name
            ImGui::TableNextColumn();
            if (key.is_active) {
                ImGui::Text("%s %s", ICON_FA_KEY, key.name.c_str());
            } else {
                ImGui::TextDisabled("%s %s (revoked)", ICON_FA_KEY, key.name.c_str());
            }

            // Key prefix
            ImGui::TableNextColumn();
            ImGui::PushFont(GetSafeFont(FONT_MONO));
            ImGui::Text("%s", key.key_prefix.c_str());
            ImGui::PopFont();
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("Click to copy prefix");
            }
            if (ImGui::IsItemClicked()) {
                ImGui::SetClipboardText(key.key_prefix.c_str());
            }

            // Rate limit
            ImGui::TableNextColumn();
            ImGui::Text("%d/min", key.rate_limit_rpm);

            // Request count
            ImGui::TableNextColumn();
            ImGui::Text("%lld", (long long)key.request_count);

            // Last used
            ImGui::TableNextColumn();
            if (key.last_used_at > 0) {
                ImGui::Text("%s", FormatTimestamp(key.last_used_at).c_str());
            } else {
                ImGui::TextDisabled("Never");
            }

            // Actions
            ImGui::TableNextColumn();
            if (key.is_active) {
                if (ImGui::SmallButton(ICON_FA_TRASH " Revoke")) {
                    pending_revoke_id_ = key.id;
                    pending_revoke_name_ = key.name;
                    revoke_error_.clear();
                    show_revoke_dialog_ = true;
                }
            } else {
                ImGui::TextDisabled("Revoked");
            }

            ImGui::PopID();
        }

        ImGui::EndTable();
    }
}

void APIKeysPanel::RenderGenerateDialog() {
    if (!show_generate_dialog_) return;

    ImGui::OpenPopup("Generate API Key");

    ImVec2 center = ImGui::GetMainViewport()->GetCenter();
    ImGui::SetNextWindowPos(center, ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));
    ImGui::SetNextWindowSize(ImVec2(400, 250), ImGuiCond_Appearing);

    if (ImGui::BeginPopupModal("Generate API Key", &show_generate_dialog_, ImGuiWindowFlags_NoResize)) {
        ImGui::Text("%s Create New API Key", ICON_FA_KEY);
        ImGui::Separator();
        ImGui::Spacing();

        ImGui::Text("Key Name:");
        ImGui::SetNextItemWidth(-1);
        ImGui::InputTextWithHint("##KeyName", "e.g., production, development", new_key_name_, sizeof(new_key_name_));

        ImGui::Spacing();
        ImGui::Text("Rate Limit (requests/minute):");
        ImGui::SetNextItemWidth(150);
        ImGui::InputInt("##RateLimit", &new_key_rate_limit_);
        if (new_key_rate_limit_ < 1) new_key_rate_limit_ = 1;
        if (new_key_rate_limit_ > 10000) new_key_rate_limit_ = 10000;
        ImGui::SameLine();
        ImGui::TextDisabled("(1-10000)");

        ImGui::Spacing();

        if (!generate_error_.empty()) {
            ImGui::TextColored(ImVec4(1.0f, 0.3f, 0.3f, 1.0f), "%s %s", ICON_FA_TRIANGLE_EXCLAMATION, generate_error_.c_str());
            ImGui::Spacing();
        }

        ImGui::Separator();
        ImGui::Spacing();

        bool can_generate = strlen(new_key_name_) > 0 && !generating_;
        if (!can_generate) {
            ImGui::BeginDisabled();
        }

        if (ImGui::Button(ICON_FA_PLUS " Generate", ImVec2(120, 0))) {
            auto* client = GetDaemonClient();
            if (client && client->IsConnected()) {
                generating_ = true;
                generate_error_.clear();

                std::string full_key, error;
                if (client->CreateAPIKey(new_key_name_, new_key_rate_limit_, full_key, error)) {
                    spdlog::info("Created API key: {}", new_key_name_);
                    generated_full_key_ = full_key;
                    key_copied_ = false;
                    show_generate_dialog_ = false;
                    show_new_key_dialog_ = true;
                    RefreshKeys();
                } else {
                    generate_error_ = error.empty() ? "Failed to generate key" : error;
                }
                generating_ = false;
            }
        }

        if (!can_generate) {
            ImGui::EndDisabled();
        }

        ImGui::SameLine();
        if (ImGui::Button("Cancel", ImVec2(120, 0))) {
            show_generate_dialog_ = false;
            generate_error_.clear();
        }

        ImGui::EndPopup();
    }
}

void APIKeysPanel::RenderNewKeyDialog() {
    if (!show_new_key_dialog_) return;

    ImGui::OpenPopup("API Key Generated");

    ImVec2 center = ImGui::GetMainViewport()->GetCenter();
    ImGui::SetNextWindowPos(center, ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));
    ImGui::SetNextWindowSize(ImVec2(500, 220), ImGuiCond_Appearing);

    if (ImGui::BeginPopupModal("API Key Generated", &show_new_key_dialog_, ImGuiWindowFlags_NoResize)) {
        ImGui::TextColored(ImVec4(0.3f, 0.8f, 0.3f, 1.0f), "%s API Key Created Successfully!", ICON_FA_CIRCLE_CHECK);
        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();

        ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.3f, 1.0f), "%s Important: Copy this key now!", ICON_FA_TRIANGLE_EXCLAMATION);
        ImGui::Text("This is the only time you will see the full key.");
        ImGui::Spacing();

        ImGui::Text("Your API Key:");
        ImGui::PushFont(GetSafeFont(FONT_MONO));

        // Show key in a selectable text area
        ImGui::PushStyleColor(ImGuiCol_FrameBg, ImVec4(0.15f, 0.15f, 0.15f, 1.0f));
        ImGui::SetNextItemWidth(-1);
        char key_buf[256];
        strncpy(key_buf, generated_full_key_.c_str(), sizeof(key_buf) - 1);
        key_buf[sizeof(key_buf) - 1] = '\0';
        ImGui::InputText("##FullKey", key_buf, sizeof(key_buf), ImGuiInputTextFlags_ReadOnly);
        ImGui::PopStyleColor();
        ImGui::PopFont();

        ImGui::Spacing();

        if (ImGui::Button(ICON_FA_CLIPBOARD " Copy to Clipboard", ImVec2(180, 0))) {
            ImGui::SetClipboardText(generated_full_key_.c_str());
            key_copied_ = true;
        }

        if (key_copied_) {
            ImGui::SameLine();
            ImGui::TextColored(ImVec4(0.3f, 0.8f, 0.3f, 1.0f), "%s Copied!", ICON_FA_CHECK);
        }

        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();

        if (ImGui::Button("Done", ImVec2(120, 0))) {
            show_new_key_dialog_ = false;
            generated_full_key_.clear();
        }

        ImGui::EndPopup();
    }
}

void APIKeysPanel::RenderRevokeDialog() {
    if (!show_revoke_dialog_) return;

    ImGui::OpenPopup("Revoke API Key?");

    ImVec2 center = ImGui::GetMainViewport()->GetCenter();
    ImGui::SetNextWindowPos(center, ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));

    if (ImGui::BeginPopupModal("Revoke API Key?", &show_revoke_dialog_, ImGuiWindowFlags_AlwaysAutoResize)) {
        ImGui::Text("%s Are you sure you want to revoke:", ICON_FA_TRIANGLE_EXCLAMATION);
        ImGui::Spacing();
        ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.3f, 1.0f), "  %s", pending_revoke_name_.c_str());
        ImGui::Spacing();
        ImGui::Text("This action cannot be undone.");
        ImGui::Text("All requests using this key will be rejected.");
        ImGui::Spacing();

        if (!revoke_error_.empty()) {
            ImGui::TextColored(ImVec4(1.0f, 0.3f, 0.3f, 1.0f), "%s %s", ICON_FA_TRIANGLE_EXCLAMATION, revoke_error_.c_str());
            ImGui::Spacing();
        }

        ImGui::Separator();
        ImGui::Spacing();

        if (ImGui::Button(ICON_FA_TRASH " Revoke", ImVec2(120, 0))) {
            auto* client = GetDaemonClient();
            if (client && client->IsConnected()) {
                std::string error;
                if (client->RevokeAPIKey(pending_revoke_id_, error)) {
                    spdlog::info("Revoked API key: {}", pending_revoke_name_);
                    show_revoke_dialog_ = false;
                    RefreshKeys();
                } else {
                    revoke_error_ = error.empty() ? "Revoke failed" : error;
                }
            } else {
                revoke_error_ = "Daemon not connected";
            }
        }

        ImGui::SameLine();
        if (ImGui::Button("Cancel", ImVec2(120, 0))) {
            show_revoke_dialog_ = false;
            revoke_error_.clear();
        }

        ImGui::EndPopup();
    }
}

void APIKeysPanel::RefreshKeys() {
    keys_.clear();
    auto* client = GetDaemonClient();
    if (client && client->IsConnected()) {
        client->ListAPIKeys(keys_);
        spdlog::debug("Loaded {} API keys", keys_.size());
    }
}

std::string APIKeysPanel::FormatTimestamp(int64_t timestamp) {
    if (timestamp <= 0) return "Never";

    std::time_t time = static_cast<std::time_t>(timestamp);
    std::tm* tm = std::localtime(&time);
    if (!tm) return "Invalid";

    std::ostringstream oss;
    oss << std::put_time(tm, "%m/%d %H:%M");
    return oss.str();
}

} // namespace cyxwiz::servernode::gui
