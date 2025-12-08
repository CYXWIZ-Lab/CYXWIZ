// settings_panel.h - Node configuration
#pragma once
#include "gui/server_panel.h"
#include "core/backend_manager.h"
#include "ipc/daemon_client.h"
#include <string>

namespace cyxwiz::servernode::gui {

class SettingsPanel : public ServerPanel {
public:
    SettingsPanel() : ServerPanel("Settings") {}
    void Render() override;

private:
    void RenderRemoteConnectionTab();
    void TestRemoteConnection();

    core::NodeConfig config_;
    bool config_loaded_ = false;
    bool config_changed_ = false;

    // Remote connection settings
    char daemon_address_[256] = "localhost:50054";
    ipc::TLSConnectionSettings tls_settings_;

    // Path buffers for TLS certificate files
    char ca_cert_path_[512] = "";
    char client_cert_path_[512] = "";
    char client_key_path_[512] = "";
    char target_name_override_[256] = "";

    // Connection test state
    bool testing_connection_ = false;
    bool test_result_valid_ = false;
    bool test_result_success_ = false;
    std::string test_result_message_;
};

} // namespace cyxwiz::servernode::gui
