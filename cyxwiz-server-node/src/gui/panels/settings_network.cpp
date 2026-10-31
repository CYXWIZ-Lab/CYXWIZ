// settings_network.cpp - Settings > Network (TOFIX168 step 3b): the one CyxWiz
// server address the node signs in at and reports to central through, what
// it resolves to and whether each service answers, the separate addresses
// for development, and this node's own API.
#include "gui/panels/settings_panel.h"

#include "auth/auth_manager.h"
#include "common/server_address_view.h"
#include "gui/server_address_widgets.h"
#include "gui/theme.h"
#include "gui/ui_widgets.h"

#include <imgui.h>

#include <cstring>

namespace cyxwiz::servernode::gui {

namespace proto = cyxwiz::protocol;

namespace {

std::string Trimmed(const std::string& s) {
    const auto first = s.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) return {};
    return s.substr(first, s.find_last_not_of(" \t\r\n") - first + 1);
}

// A text field bound to a config string; true when it changed.
bool StringField(const char* id, std::string& value, const char* hint, bool enabled = true) {
    char buffer[512];
    std::strncpy(buffer, value.c_str(), sizeof(buffer) - 1);
    buffer[sizeof(buffer) - 1] = '\0';
    if (!ui::TextField(id, buffer, sizeof(buffer), hint, 0, enabled)) return false;
    value = buffer;
    return true;
}

void Heading(const char* title) {
    ImGui::Spacing();
    ImGui::TextUnformatted(title);
    ImGui::Separator();
    ImGui::Spacing();
}

}  // namespace

bool SettingsPanel::ServerSettingsChanged() const {
    const auto& saved = core::BackendManager::Instance().GetConfig();
    return Trimmed(config_.cyxwiz_server) != saved.cyxwiz_server || config_.central_server != saved.central_server ||
           config_.auth_api_url != saved.auth_api_url;
}

void SettingsPanel::RenderNetworkTab() {
    const auto& semantic = ::gui::GetTheme().Semantic();
    const ImVec4 muted = ui::Muted();
    ImGui::PushTextWrapPos(0.0f);

    // ---- The one address
    Heading("CyxWiz server");
    ImGui::TextUnformatted("Server address");
    if (StringField("##CyxWizServer", config_.cyxwiz_server, "http://localhost:8000")) {
        config_changed_ = true;
        save_note_.clear();
        server_probe_.Edited();
    }
    ImGui::TextColored(muted, "Examples:");
    for (const auto& example : proto::ServerAddressExamples()) {
        const std::string link = std::string(example.label) + " " + example.value;
        const float needed = ImGui::CalcTextSize(link.c_str()).x + ImGui::GetStyle().ItemSpacing.x * 2;
        if (ImGui::GetContentRegionAvail().x > needed) ImGui::SameLine();
        if (ui::LinkButton(link.c_str())) {
            config_.cyxwiz_server = example.value;
            config_changed_ = true;
            save_note_.clear();
        }
    }

    const std::string typed = Trimmed(config_.cyxwiz_server);
    auto view = proto::DescribeServerAddress(typed, proto::ServiceSet::NodeServices);
    if (view.blank) {
        view = proto::DescribeSeparateAddresses(config_.central_server, "", config_.auth_api_url);
    }
    if (!view.problem.empty()) {
        ImGui::TextColored(semantic.warning, "%s", view.problem.c_str());
    } else {
        ImGui::TextColored(muted, "%s", view.kind_line.c_str());
        std::string key = typed;
        if (view.usable) {
            server_probe_.Update(typed, true);
        } else {
            server_probe_.UpdateSeparate(config_.central_server, config_.auth_api_url);
            key = ServerProbe::SeparateKey(config_.central_server, config_.auth_api_url);
        }
        proto::ApplyReach(view, server_probe_.ReachFor(key));
        RenderServerRows(view);
        if (ui::LinkButton("Check again")) server_probe_.Forget();
    }
    ImGui::TextColored(muted, "Datasets come from the gateway address in each job; nothing to set here.");
    if (ServerSettingsChanged() && auth::AuthManager::Instance().IsAuthenticated()) {
        ImGui::TextColored(semantic.warning,
                           "Saving a new server signs this node out; sign in again at the new server.");
    }

    // ---- Separate addresses (development)
    ImGui::Spacing();
    if (ui::LinkButton(show_separate_addresses_ ? "Hide separate addresses" : "Separate addresses (development)")) {
        show_separate_addresses_ = !show_separate_addresses_;
    }
    if (show_separate_addresses_) {
        const bool in_effect = typed.empty();
        ImGui::TextColored(muted, "Used only while the server address is empty (development).");
        ImGui::TextUnformatted("Central server");
        if (StringField("##CentralServer", config_.central_server, "localhost:50051", in_effect)) {
            server_probe_.Edited();
            config_changed_ = true;
            save_note_.clear();
        }
        ImGui::TextUnformatted("Sign-in");
        if (StringField("##AuthApi", config_.auth_api_url, "http://127.0.0.1:3002/api", in_effect)) {
            server_probe_.Edited();
            config_changed_ = true;
            save_note_.clear();
        }
        if (!in_effect) ImGui::TextColored(muted, "Not in use: the server address is set.");
    }

    // ---- This node's own API
    Heading("This node's own API");
    if (ImGui::InputInt("HTTP API port", &config_.http_api_port)) {
        config_changed_ = true;
        save_note_.clear();
    }
    if (ImGui::Checkbox("Serve this node's API over TLS", &config_.enable_tls)) {
        config_changed_ = true;
        save_note_.clear();
    }
    ImGui::TextColored(muted,
                       "TLS for this node's own API, not for the connection to CyxWiz (an https:// server "
                       "address is).");

    ImGui::PopTextWrapPos();
}

}  // namespace cyxwiz::servernode::gui
