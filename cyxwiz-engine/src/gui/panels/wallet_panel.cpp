#include "wallet_panel.h"
#include <imgui.h>
#include <ctime>
#include <iomanip>
#include <sstream>

namespace gui {

WalletPanel::WalletPanel()
    : Panel("Wallet", true)
    , status_(WalletConnectionStatus::Disconnected)
    , wallet_address_("")
    , sol_balance_(0.0)
    , cyxwiz_balance_(0.0)
    , token_mint_("")
    , transaction_display_limit_(10)
    , show_connection_dialog_(false)
    , error_message_("")
    , last_refresh_time_(0.0f)
    , auto_refresh_enabled_(true)
{
    memset(wallet_address_buffer_, 0, sizeof(wallet_address_buffer_));
}

WalletPanel::~WalletPanel() = default;

void WalletPanel::Render() {
    if (!visible_) return;

    ImGui::Begin(name_.c_str(), &visible_);

    RenderConnectionStatus();
    ImGui::Separator();

    if (status_ == WalletConnectionStatus::Connected) {
        RenderWalletInfo();
        ImGui::Separator();
        RenderBalance();
        ImGui::Separator();
        RenderActions();
        ImGui::Separator();
        RenderTransactionHistory();
    } else {
        ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f), "Connect your wallet to get started");
        ImGui::Spacing();
        if (ImGui::Button("Connect Wallet", ImVec2(-1, 40))) {
            show_connection_dialog_ = true;
        }
    }

    ImGui::End();

    // Connection dialog
    if (show_connection_dialog_) {
        RenderConnectionDialog();
    }
}

void WalletPanel::RenderConnectionStatus() {
    ImGui::Text("Status:");
    ImGui::SameLine();

    switch (status_) {
        case WalletConnectionStatus::Disconnected:
            ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f), "Disconnected");
            break;
        case WalletConnectionStatus::Connecting:
            ImGui::TextColored(ImVec4(1.0f, 1.0f, 0.0f, 1.0f), "Connecting...");
            break;
        case WalletConnectionStatus::Connected:
            ImGui::TextColored(ImVec4(0.0f, 1.0f, 0.0f, 1.0f), "Connected");
            break;
        case WalletConnectionStatus::Error:
            ImGui::TextColored(ImVec4(1.0f, 0.0f, 0.0f, 1.0f), "Error");
            break;
    }

    if (!error_message_.empty()) {
        ImGui::SameLine();
        ImGui::TextColored(ImVec4(1.0f, 0.0f, 0.0f, 1.0f), "- %s", error_message_.c_str());
    }
}

void WalletPanel::RenderWalletInfo() {
    ImGui::Text("Address:");
    ImGui::SameLine();
    ImGui::TextColored(ImVec4(0.5f, 0.8f, 1.0f, 1.0f), "%s", FormatAddress(wallet_address_).c_str());

    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("%s", wallet_address_.c_str());
    }

    ImGui::SameLine();
    if (ImGui::SmallButton("Copy")) {
        ImGui::SetClipboardText(wallet_address_.c_str());
    }

    ImGui::SameLine();
    if (ImGui::SmallButton("View on Explorer")) {
        std::string url = "https://explorer.solana.com/address/" + wallet_address_ + "?cluster=devnet";
        // TODO: Open URL in browser
    }
}

void WalletPanel::RenderBalance() {
    ImGui::BeginChild("Balance", ImVec2(0, 120), true, ImGuiWindowFlags_NoScrollbar);

    // CYXWIZ Balance (primary)
    ImGui::PushFont(ImGui::GetIO().Fonts->Fonts[0]); // Use default font, can be larger
    ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.2f, 1.0f), "CYXWIZ Balance");
    ImGui::PopFont();

    ImGui::PushFont(ImGui::GetIO().Fonts->Fonts[0]);
    ImGui::Text("%s", FormatBalance(cyxwiz_balance_).c_str());
    ImGui::PopFont();

    ImGui::Spacing();

    // SOL Balance (secondary)
    ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f), "SOL Balance:");
    ImGui::SameLine();
    ImGui::Text("%.4f SOL", sol_balance_);

    ImGui::Spacing();

    // Refresh button
    if (ImGui::Button("Refresh Balance")) {
        RefreshBalance();
    }

    ImGui::SameLine();
    ImGui::Checkbox("Auto-refresh", &auto_refresh_enabled_);

    ImGui::EndChild();
}

void WalletPanel::RenderActions() {
    ImGui::Text("Quick Actions:");
    ImGui::Spacing();

    ImGui::BeginGroup();
    if (ImGui::Button("Disconnect", ImVec2(120, 0))) {
        DisconnectWallet();
    }
    ImGui::EndGroup();
}

void WalletPanel::RenderTransactionHistory() {
    ImGui::Text("Recent Transactions:");
    ImGui::SameLine();
    if (ImGui::SmallButton("Refresh")) {
        RefreshTransactions();
    }

    if (transactions_.empty()) {
        ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f), "No transactions yet");
        return;
    }

    ImGui::BeginChild("TransactionList", ImVec2(0, 0), true);

    for (const auto& tx : transactions_) {
        ImGui::PushID(tx.signature.c_str());

        // Transaction icon and type
        ImGui::Text("%s", GetTransactionIcon(tx.type).c_str());
        ImGui::SameLine();
        ImGui::TextColored(ImVec4(0.8f, 0.8f, 0.8f, 1.0f), "%s", tx.description.c_str());

        // Amount
        ImGui::SameLine(ImGui::GetWindowWidth() - 150);
        if (tx.amount > 0) {
            ImGui::TextColored(ImVec4(0.0f, 1.0f, 0.0f, 1.0f), "+%.2f", tx.amount);
        } else {
            ImGui::TextColored(ImVec4(1.0f, 0.0f, 0.0f, 1.0f), "%.2f", tx.amount);
        }
        ImGui::SameLine();
        ImGui::Text("CYXWIZ");

        // Timestamp and status
        ImGui::Indent(20.0f);
        ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "%s", FormatTimestamp(tx.timestamp).c_str());
        ImGui::SameLine();

        if (tx.status == "CONFIRMED") {
            ImGui::TextColored(ImVec4(0.0f, 1.0f, 0.0f, 1.0f), "[Confirmed]");
        } else if (tx.status == "PENDING") {
            ImGui::TextColored(ImVec4(1.0f, 1.0f, 0.0f, 1.0f), "[Pending]");
        } else {
            ImGui::TextColored(ImVec4(1.0f, 0.0f, 0.0f, 1.0f), "[Failed]");
        }

        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Signature: %s", tx.signature.c_str());
        }

        ImGui::Unindent(20.0f);
        ImGui::Spacing();

        ImGui::PopID();
    }

    ImGui::EndChild();
}

void WalletPanel::RenderConnectionDialog() {
    ImGui::OpenPopup("Connect Wallet");

    ImVec2 center = ImGui::GetMainViewport()->GetCenter();
    ImGui::SetNextWindowPos(center, ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));
    ImGui::SetNextWindowSize(ImVec2(500, 200), ImGuiCond_Appearing);

    if (ImGui::BeginPopupModal("Connect Wallet", &show_connection_dialog_, ImGuiWindowFlags_NoResize)) {
        ImGui::Text("Enter your Solana wallet address:");
        ImGui::Spacing();

        ImGui::PushItemWidth(-1);
        ImGui::InputText("##wallet_address", wallet_address_buffer_, sizeof(wallet_address_buffer_));
        ImGui::PopItemWidth();

        ImGui::Spacing();
        ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.0f, 1.0f), "Note: Your private key is never stored. Only the public address is used.");

        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();

        if (ImGui::Button("Connect", ImVec2(120, 0))) {
            std::string address(wallet_address_buffer_);
            if (!address.empty()) {
                ConnectWallet(address);
                show_connection_dialog_ = false;
            }
        }

        ImGui::SameLine();
        if (ImGui::Button("Cancel", ImVec2(120, 0))) {
            show_connection_dialog_ = false;
        }

        ImGui::EndPopup();
    }
}

// Wallet operations
void WalletPanel::ConnectWallet(const std::string& address) {
    status_ = WalletConnectionStatus::Connecting;
    wallet_address_ = address;
    error_message_ = "";

    // TODO: Call gRPC WalletService.ConnectWallet
    // For now, simulate connection
    status_ = WalletConnectionStatus::Connected;

    // Mock data
    sol_balance_ = 2.5;
    cyxwiz_balance_ = 1250.0;
    token_mint_ = "CYXWiZ1111111111111111111111111111111111111";

    RefreshTransactions();
}

void WalletPanel::DisconnectWallet() {
    status_ = WalletConnectionStatus::Disconnected;
    wallet_address_ = "";
    sol_balance_ = 0.0;
    cyxwiz_balance_ = 0.0;
    token_mint_ = "";
    transactions_.clear();
    error_message_ = "";
}

void WalletPanel::RefreshBalance() {
    if (status_ != WalletConnectionStatus::Connected) return;

    // TODO: Call gRPC WalletService.GetBalance
    // For now, use existing values
}

void WalletPanel::RefreshTransactions() {
    if (status_ != WalletConnectionStatus::Connected) return;

    // TODO: Call gRPC WalletService.GetTransactionHistory

    // Mock data
    transactions_.clear();
    transactions_.push_back({
        "5T7K...9dF2",
        "JOB_PAYMENT",
        "CONFIRMED",
        -10.5,
        std::time(nullptr) - 120,
        "Job Payment #12345",
        12345
    });
    transactions_.push_back({
        "8Hn2...3kL9",
        "REWARD_CLAIM",
        "CONFIRMED",
        5.2,
        std::time(nullptr) - 3600,
        "Reward Claim",
        0
    });
}

// Helper functions
std::string WalletPanel::FormatAddress(const std::string& address) const {
    if (address.length() <= 16) return address;
    return address.substr(0, 8) + "..." + address.substr(address.length() - 8);
}

std::string WalletPanel::FormatBalance(double balance) const {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2) << balance << " CYXWIZ";
    return oss.str();
}

std::string WalletPanel::FormatTimestamp(int64_t timestamp) const {
    time_t now = std::time(nullptr);
    int64_t diff = now - timestamp;

    if (diff < 60) {
        return std::to_string(diff) + "s ago";
    } else if (diff < 3600) {
        return std::to_string(diff / 60) + "m ago";
    } else if (diff < 86400) {
        return std::to_string(diff / 3600) + "h ago";
    } else {
        return std::to_string(diff / 86400) + "d ago";
    }
}

std::string WalletPanel::GetTransactionIcon(const std::string& type) const {
    if (type == "JOB_PAYMENT") return "[OUT]";
    if (type == "REWARD_CLAIM") return "[IN]";
    if (type == "STAKE") return "[STAKE]";
    if (type == "UNSTAKE") return "[UNSTAKE]";
    if (type == "REFUND") return "[REFUND]";
    return "[?]";
}

} // namespace gui
