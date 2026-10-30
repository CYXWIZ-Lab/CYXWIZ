// Preferences > Servers (TOFIX168 step 3b): the one server address the Engine
// reaches central, the CyxCloud gateway and sign-in through, what it resolves
// to, whether each service answers, and the separate per-service addresses
// for development. The words come from cyxwiz-protocol's
// server_address_view, which the server node's screens render too.

#include "toolbar.h"

#include "../../core/engine_config.h"
#include "../ui_buttons.h"
#include "../ui_tokens.h"

#include "common/server_address_probe.h"
#include "common/server_address_view.h"

#include <imgui.h>
#include <spdlog/spdlog.h>

#include <chrono>
#include <cstring>

namespace cyxwiz {
namespace {

namespace proto = cyxwiz::protocol;

std::string Trimmed(const char* text) {
    const std::string s(text);
    const auto first = s.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) return {};
    return s.substr(first, s.find_last_not_of(" \t\r\n") - first + 1);
}

void CopyTo(char* buffer, size_t size, const std::string& text) {
    std::strncpy(buffer, text.c_str(), size - 1);
    buffer[size - 1] = '\0';
}

ImVec4 ReachColour(proto::ServiceReach reach, const ui::Tokens& t) {
    switch (reach) {
        case proto::ServiceReach::Reachable: return t.success;
        case proto::ServiceReach::NotReachable: return t.caution;
        case proto::ServiceReach::Checking: return t.running;
        case proto::ServiceReach::NotChecked: break;
    }
    return t.pending;
}

void SectionHeading(const char* title) {
    ImGui::Spacing();
    ImGui::TextUnformatted(title);
    ImGui::Separator();
    ImGui::Spacing();
}

}  // namespace

void ToolbarPanel::LoadServerSettings() {
    auto& config = core::EngineConfig::Instance();
    CopyTo(server_address_buf_, sizeof(server_address_buf_), config.GetServerAddress());
    const auto separate = config.GetSeparateAddresses();
    CopyTo(separate_central_buf_, sizeof(separate_central_buf_), separate.central);
    CopyTo(separate_gateway_buf_, sizeof(separate_gateway_buf_), separate.gateway);
    CopyTo(separate_api_buf_, sizeof(separate_api_buf_), separate.auth_api);
    server_change_confirm_ = false;
    server_probed_.clear();
    servers_loaded_ = true;
}

void ToolbarPanel::PollServerProbe() {
    if (!server_probe_.valid() ||
        server_probe_.wait_for(std::chrono::seconds(0)) != std::future_status::ready) {
        return;
    }
    server_reach_ = server_probe_.get();
}

void ToolbarPanel::StartServerProbe(const std::string& key) {
    if (server_probe_.valid()) return;  // one at a time; the next frame asks again
    server_probed_ = key;
    auto& config = core::EngineConfig::Instance();
    const std::string address = config.GetServerAddress();
    proto::ServerEndpoints separate;
    separate.central = config.GetCentralServerAddress();
    separate.gateway = config.GetCloudGatewayAddress();
    separate.api_base_url = config.GetAuthApiUrl();
    server_reach_ = {};
    server_reach_.central = server_reach_.gateway = server_reach_.api = proto::ServiceReach::Checking;
    server_probe_ = std::async(std::launch::async, [address, separate]() {
        const proto::HttpGetFn get = [](const std::string& url, std::string* body) {
            return auth::AuthClient::HttpGet(url, body);
        };
        return address.empty() ? proto::ProbeEndpoints(separate, proto::ServiceSet::EngineServices, get)
                               : proto::ProbeServer(address, proto::ServiceSet::EngineServices, get);
    });
}

void ToolbarPanel::SignOut() {
    auth::AuthClient::Instance().Logout();
    is_logged_in_ = false;
    logged_in_user_.clear();
    std::memset(login_identifier_, 0, sizeof(login_identifier_));
    std::memset(login_password_, 0, sizeof(login_password_));
    login_error_message_.clear();
    login_success_message_.clear();
    spdlog::info("User signed out");
    if (on_logout_callback_) on_logout_callback_();
}

void ToolbarPanel::ApplyServerAddress(const std::string& address) {
    auto& auth = auth::AuthClient::Instance();
    // A sign-in belongs to the server that issued it.
    if (auth.IsAuthenticated()) SignOut();
    auto& config = core::EngineConfig::Instance();
    config.SetServerAddress(address);
    config.Save();
    auth.SetApiBaseUrl(config.GetAuthApiUrl());
    server_change_confirm_ = false;
    server_probed_.clear();
    spdlog::info("Server address set to {}", address.empty() ? "none (the separate addresses apply)" : address);
    if (server_changed_callback_) server_changed_callback_();
}

void ToolbarPanel::RenderServersTab() {
    const ui::Tokens& t = ui::CurrentTokens();
    auto& config = core::EngineConfig::Instance();
    auto& auth = auth::AuthClient::Instance();
    if (!servers_loaded_) LoadServerSettings();
    PollServerProbe();

    const std::string saved = config.GetServerAddress();
    const std::string typed = Trimmed(server_address_buf_);
    const auto typed_view = proto::DescribeServerAddress(typed, proto::ServiceSet::EngineServices);
    const bool changed = typed != saved;

    ImGui::PushTextWrapPos(0.0f);

    // ---- The address
    SectionHeading("CyxWiz server");
    ImGui::TextUnformatted("Server address");
    const char* use_label = "Use this server";
    const float use_width = ui::ButtonWidth(use_label, ui::ButtonSize::Regular);
    ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x - use_width - ImGui::GetStyle().ItemSpacing.x);
    if (ImGui::InputTextWithHint("##server_address", "http://localhost:8000", server_address_buf_,
                                 sizeof(server_address_buf_))) {
        server_change_confirm_ = false;
    }
    ImGui::SameLine();
    const bool can_use = changed && (typed_view.usable || typed_view.blank);
    const char* use_reason = !changed ? "This is the server in use" : "Fix the address first";
    if (ui::SecondaryButton(use_label, can_use, use_reason, ui::ButtonSize::Regular)) {
        if (auth.IsAuthenticated()) {
            server_change_confirm_ = true;
        } else {
            ApplyServerAddress(typed);
        }
    }

    ImGui::TextDisabled("Examples:");
    for (const auto& example : proto::ServerAddressExamples()) {
        const std::string link = std::string(example.label) + "##example";
        const float needed = ui::ButtonWidth(link.c_str(), ui::ButtonSize::Small) +
                             ImGui::CalcTextSize(example.value).x + ImGui::GetStyle().ItemSpacing.x * 3;
        if (ImGui::GetContentRegionAvail().x > needed) ImGui::SameLine();
        ImGui::PushID(example.value);
        if (ui::LinkButton(link.c_str())) {
            CopyTo(server_address_buf_, sizeof(server_address_buf_), example.value);
            server_change_confirm_ = false;
        }
        ImGui::PopID();
        ImGui::SameLine();
        ImGui::TextDisabled("%s", example.value);
    }

    if (!typed_view.problem.empty()) {
        ImGui::TextColored(t.caution, "%s", typed_view.problem.c_str());
    } else {
        ImGui::TextDisabled("%s", typed_view.kind_line.c_str());
    }

    // ---- Signed in: changing the server signs out, so ask first
    if (server_change_confirm_) {
        ImGui::Spacing();
        ImGui::PushStyleColor(ImGuiCol_ChildBg, t.bg_panel);
        ImGui::BeginChild("##server_change_confirm", ImVec2(0, 0),
                          ImGuiChildFlags_AutoResizeY | ImGuiChildFlags_AlwaysUseWindowPadding);
        const std::string target = typed.empty() ? "the separate addresses" : typed;
        const std::string current = saved.empty() ? "the separate addresses" : saved;
        ImGui::TextColored(t.text_bright, "Change the server to %s?", target.c_str());
        ImGui::TextColored(t.text_dim,
                           "You are signed in at %s. Changing the server signs you out and disconnects from it; "
                           "sign in again at the new server.",
                           current.c_str());
        if (ui::SecondaryButton("Change and sign out", true, nullptr, ui::ButtonSize::Regular)) {
            ApplyServerAddress(typed);
        }
        ImGui::SameLine();
        const std::string keep = "Keep " + current;
        if (ui::LinkButton(keep.c_str())) {
            CopyTo(server_address_buf_, sizeof(server_address_buf_), saved);
            server_change_confirm_ = false;
        }
        ImGui::EndChild();
        ImGui::PopStyleColor();
    }

    // ---- What is in use, and whether it answers
    ImGui::Spacing();
    ImGui::Spacing();
    ImGui::TextUnformatted("What the Engine connects to");
    const char* check_label = "Check again";
    ImGui::SameLine(ImGui::GetContentRegionMax().x - ui::ButtonWidth(check_label, ui::ButtonSize::Small));
    if (ui::LinkButton(check_label)) server_probed_.clear();
    ImGui::Separator();
    ImGui::Spacing();

    auto in_use = saved.empty() ? proto::DescribeSeparateAddresses(config.GetCentralServerAddress(),
                                                                   config.GetCloudGatewayAddress(),
                                                                   config.GetAuthApiUrl())
                                : proto::DescribeServerAddress(saved, proto::ServiceSet::EngineServices);
    const std::string probe_key = saved.empty() ? "separate:" + config.GetCentralServerAddress() + "|" +
                                                      config.GetCloudGatewayAddress() + "|" + config.GetAuthApiUrl()
                                                : saved;
    if (probe_key != server_probed_ && (in_use.usable || in_use.blank)) StartServerProbe(probe_key);
    proto::ApplyReach(in_use, server_reach_);

    if (ImGui::BeginTable("##server_services", 3, ImGuiTableFlags_SizingStretchProp)) {
        ImGui::TableSetupColumn("Service", ImGuiTableColumnFlags_WidthStretch, 1.0f);
        ImGui::TableSetupColumn("Address", ImGuiTableColumnFlags_WidthStretch, 2.2f);
        ImGui::TableSetupColumn("Status", ImGuiTableColumnFlags_WidthStretch, 1.0f);
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::TextDisabled("Service");
        ImGui::TableNextColumn();
        ImGui::TextDisabled("Address");
        ImGui::TableNextColumn();
        ImGui::TextDisabled("Status");
        for (const auto& row : in_use.rows) {
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::TextUnformatted(row.name.c_str());
            ImGui::TableNextColumn();
            ImGui::TextUnformatted(row.address.c_str());
            ImGui::TableNextColumn();
            ImGui::TextColored(ReachColour(row.reach, t), "%s", proto::ReachWord(row.reach));
        }
        ImGui::EndTable();
    }
    if (saved.empty()) {
        ImGui::TextDisabled("%s", in_use.kind_line.c_str());
    } else {
        ImGui::TextDisabled("In use and saved: %s. Central, the gateway and sign-in all follow it.", saved.c_str());
    }

    // ---- Account
    ImGui::Spacing();
    SectionHeading("Account");
    if (auth.IsAuthenticated()) {
        const auto user = auth.GetUserInfo();
        const std::string who = !user.email.empty() ? user.email : user.username;
        ImGui::Text("Signed in as %s at %s.", who.c_str(), auth.GetApiBaseUrl().c_str());
    } else {
        ImGui::Text("Signed out. Sign in at %s (Account, top right).", auth.GetApiBaseUrl().c_str());
    }

    // ---- Separate addresses (development)
    ImGui::Spacing();
    if (ui::LinkButton(show_separate_addresses_ ? "Hide separate addresses" : "Separate addresses (development)")) {
        show_separate_addresses_ = !show_separate_addresses_;
    }
    if (show_separate_addresses_) {
        ImGui::PushStyleColor(ImGuiCol_ChildBg, t.bg_panel);
        ImGui::BeginChild("##separate_addresses", ImVec2(0, 0),
                          ImGuiChildFlags_AutoResizeY | ImGuiChildFlags_AlwaysUseWindowPadding);
        ImGui::TextColored(t.text_dim,
                           "Separate addresses, for development. The Engine uses them only while the server "
                           "address above is empty.");
        const bool in_effect = saved.empty();
        bool edited = false;
        if (ImGui::BeginTable("##separate_fields", 2, ImGuiTableFlags_SizingStretchProp)) {
            ImGui::TableSetupColumn("Label", ImGuiTableColumnFlags_WidthStretch, 1.0f);
            ImGui::TableSetupColumn("Field", ImGuiTableColumnFlags_WidthStretch, 3.2f);
            struct Field {
                const char* label;
                const char* id;
                char* buffer;
                size_t size;
                const char* hint;
            };
            const Field fields[] = {
                {"Central server", "##separate_central", separate_central_buf_, sizeof(separate_central_buf_),
                 "localhost:50051"},
                {"CyxCloud gateway", "##separate_gateway", separate_gateway_buf_, sizeof(separate_gateway_buf_),
                 "next to central, port 50060"},
                {"Sign-in", "##separate_api", separate_api_buf_, sizeof(separate_api_buf_),
                 "http://127.0.0.1:3002/api"},
            };
            for (const auto& field : fields) {
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::AlignTextToFramePadding();
                ImGui::TextUnformatted(field.label);
                ImGui::TableNextColumn();
                ImGui::BeginDisabled(!in_effect);
                ImGui::SetNextItemWidth(-1.0f);
                ImGui::InputTextWithHint(field.id, field.hint, field.buffer, field.size);
                edited |= ImGui::IsItemDeactivatedAfterEdit();
                ImGui::EndDisabled();
                if (!in_effect && ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) {
                    ImGui::SetTooltip("Clear the server address to use these");
                }
            }
            ImGui::EndTable();
        }
        if (!in_effect) ImGui::TextDisabled("Not in use: the server address is set.");
        if (edited) {
            config.SetCentralServerAddress(Trimmed(separate_central_buf_));
            config.SetCloudGatewayAddress(Trimmed(separate_gateway_buf_));
            config.SetAuthApiUrl(Trimmed(separate_api_buf_));
            config.Save();
            auth.SetApiBaseUrl(config.GetAuthApiUrl());
            server_probed_.clear();
            spdlog::info("Separate server addresses saved");
            if (server_changed_callback_) server_changed_callback_();
        }
        ImGui::EndChild();
        ImGui::PopStyleColor();
    }

    ImGui::PopTextWrapPos();
}

}  // namespace cyxwiz
