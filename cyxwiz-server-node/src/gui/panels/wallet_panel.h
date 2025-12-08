// wallet_panel.h - Wallet and earnings
#pragma once
#include "gui/server_panel.h"

namespace cyxwiz::servernode::gui {

class WalletPanel : public ServerPanel {
public:
    WalletPanel() : ServerPanel("Wallet") {}
    void Render() override;
private:
    char wallet_address_[128] = "";
    bool is_connected_ = false;
};

} // namespace cyxwiz::servernode::gui
