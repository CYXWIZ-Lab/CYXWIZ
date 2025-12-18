// wallet_panel.h - Wallet and earnings
#pragma once
#include "gui/server_panel.h"
#include <string>
#include <vector>

namespace cyxwiz::servernode::gui {

struct WalletTransaction {
    std::string signature;
    std::string type;
    std::string status;
    double amount;
    int64_t timestamp;
    std::string description;
};

class WalletPanel : public ServerPanel {
public:
    WalletPanel() : ServerPanel("Wallet") {}
    void Render() override;
private:
    void RenderBalanceSection(const std::string& wallet_address, double sol_balance, double cyxwiz_balance);
    void RenderTransactionHistory();
    std::string FormatAddress(const std::string& address) const;
    std::string FormatTimestamp(int64_t timestamp) const;

    char wallet_address_[128] = "";
    bool is_connected_ = false;
    
    // USD prices
    double sol_usd_price_ = 220.50;
    double cyxwiz_usd_price_ = 0.25;
    
    // Mock balances (will be replaced with real data)
    double sol_balance_ = 5.0;
    double cyxwiz_balance_ = 1000.0;
    
    // Transactions
    std::vector<WalletTransaction> transactions_;
};

} // namespace cyxwiz::servernode::gui
