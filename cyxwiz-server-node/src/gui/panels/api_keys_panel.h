// api_keys_panel.h - API key management with daemon integration
#pragma once
#include "gui/server_panel.h"
#include "ipc/daemon_client.h"
#include <vector>
#include <string>

namespace cyxwiz::servernode::gui {

class APIKeysPanel : public ServerPanel {
public:
    APIKeysPanel() : ServerPanel("API Keys") {}
    void Render() override;

private:
    void RenderKeyList();
    void RenderGenerateDialog();
    void RenderNewKeyDialog();
    void RenderRevokeDialog();
    void RefreshKeys();
    std::string FormatTimestamp(int64_t timestamp);

    // API keys list
    std::vector<ipc::APIKeyInfo> keys_;
    bool keys_loaded_ = false;

    // Generate new key dialog
    bool show_generate_dialog_ = false;
    char new_key_name_[64] = "";
    int new_key_rate_limit_ = 100;
    std::string generate_error_;
    bool generating_ = false;

    // Show generated key (one-time display)
    bool show_new_key_dialog_ = false;
    std::string generated_full_key_;
    bool key_copied_ = false;

    // Revoke confirmation dialog
    bool show_revoke_dialog_ = false;
    std::string pending_revoke_id_;
    std::string pending_revoke_name_;
    std::string revoke_error_;
};

} // namespace cyxwiz::servernode::gui
