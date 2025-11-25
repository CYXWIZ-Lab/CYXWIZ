#pragma once

#include "../panel.h"
#include <string>
#include <vector>
#include <memory>

namespace gui {

enum class WalletConnectionStatus {
    Disconnected,
    Connecting,
    Connected,
    Error
};

struct WalletTransaction {
    std::string signature;
    std::string type;
    std::string status;
    double amount;
    int64_t timestamp;
    std::string description;
    uint64_t job_id;
};

class WalletPanel : public cyxwiz::Panel {
public:
    WalletPanel();
    ~WalletPanel() override;

    void Render() override;

    // Wallet operations
    void ConnectWallet(const std::string& address);
    void DisconnectWallet();
    void RefreshBalance();
    void RefreshTransactions();

    // Getters
    bool IsConnected() const { return status_ == WalletConnectionStatus::Connected; }
    const std::string& GetWalletAddress() const { return wallet_address_; }
    double GetCyxwizBalance() const { return cyxwiz_balance_; }
    double GetSolBalance() const { return sol_balance_; }

private:
    // UI rendering
    void RenderConnectionStatus();
    void RenderWalletInfo();
    void RenderBalance();
    void RenderActions();
    void RenderTransactionHistory();
    void RenderConnectionDialog();

    // Helper functions
    std::string FormatAddress(const std::string& address) const;
    std::string FormatBalance(double balance) const;
    std::string FormatTimestamp(int64_t timestamp) const;
    std::string GetTransactionIcon(const std::string& type) const;

    // Wallet state
    WalletConnectionStatus status_;
    std::string wallet_address_;
    double sol_balance_;
    double cyxwiz_balance_;
    std::string token_mint_;

    // Transactions
    std::vector<WalletTransaction> transactions_;
    int transaction_display_limit_;

    // UI state
    bool show_connection_dialog_;
    char wallet_address_buffer_[128];
    std::string error_message_;
    float last_refresh_time_;
    bool auto_refresh_enabled_;
};

} // namespace gui
