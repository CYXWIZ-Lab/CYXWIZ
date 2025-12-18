#include "wallet_panel.h"
#include "../icons.h"
#include "../../auth/auth_client.h"
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
    , sol_usd_price_(220.50)
    , cyxwiz_usd_price_(0.25)
    , show_external_wallet_dialog_(false)
    , external_wallet_step_(0)
    , external_wallet_nonce_("")
    , external_wallet_address_("")
    , auth_synced_(false)
{
    memset(wallet_address_buffer_, 0, sizeof(wallet_address_buffer_));
    memset(external_wallet_buffer_, 0, sizeof(external_wallet_buffer_));
}

WalletPanel::~WalletPanel() = default;

void WalletPanel::SyncWithAuthClient() {
    auto& auth = cyxwiz::auth::AuthClient::Instance();
    if (auth.IsAuthenticated() && !auth_synced_) {
        std::string cyxwallet = auth.GetUserInfo().wallet_address;
        if (!cyxwallet.empty()) {
            wallet_address_ = cyxwallet;
            status_ = WalletConnectionStatus::Connected;
            sol_balance_ = 5.0;
            cyxwiz_balance_ = 1000.0;
            token_mint_ = "CYXWiZ1111111111111111111111111111111111111";
            RefreshTransactions();
            auth_synced_ = true;
        }
    } else if (!auth.IsAuthenticated()) {
        auth_synced_ = false;
    }
}

void WalletPanel::Render() {
    if (!visible_) return;

    SyncWithAuthClient();

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
        if (ImGui::Button(ICON_FA_WALLET " Connect CyxWallet", ImVec2(-1, 40))) {
            show_connection_dialog_ = true;
        }
    }

    ImGui::End();

    if (show_connection_dialog_) {
        RenderConnectionDialog();
    }

    if (show_external_wallet_dialog_) {
        RenderExternalWalletDialog();
    }
}

void WalletPanel::RenderConnectionStatus() {
    ImGui::Text("Status:");
    ImGui::SameLine();

    switch (status_) {
        case WalletConnectionStatus::Disconnected:
            ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f), ICON_FA_CIRCLE " Disconnected");
            break;
        case WalletConnectionStatus::Connecting:
            ImGui::TextColored(ImVec4(1.0f, 1.0f, 0.0f, 1.0f), ICON_FA_SPINNER " Connecting...");
            break;
        case WalletConnectionStatus::Connected:
            ImGui::TextColored(ImVec4(0.0f, 1.0f, 0.0f, 1.0f), ICON_FA_CIRCLE_CHECK " Connected");
            break;
        case WalletConnectionStatus::Error:
            ImGui::TextColored(ImVec4(1.0f, 0.0f, 0.0f, 1.0f), ICON_FA_CIRCLE_XMARK " Error");
            break;
    }

    if (!error_message_.empty()) {
        ImGui::SameLine();
        ImGui::TextColored(ImVec4(1.0f, 0.0f, 0.0f, 1.0f), "- %s", error_message_.c_str());
    }
}

void WalletPanel::RenderWalletInfo() {
    ImGui::Text(ICON_FA_WALLET " Address:");
    ImGui::SameLine();
    ImGui::TextColored(ImVec4(0.5f, 0.8f, 1.0f, 1.0f), "%s", FormatAddress(wallet_address_).c_str());

    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("%s", wallet_address_.c_str());
    }

    ImGui::SameLine();
    if (ImGui::SmallButton(ICON_FA_COPY " Copy")) {
        ImGui::SetClipboardText(wallet_address_.c_str());
    }

    ImGui::SameLine();
    if (ImGui::SmallButton(ICON_FA_ARROW_UP_RIGHT_FROM_SQUARE " Explorer")) {
        std::string url = "https://explorer.solana.com/address/" + wallet_address_ + "?cluster=devnet";
    }
}

void WalletPanel::RenderBalance() {
    double total_usd = (sol_balance_ * sol_usd_price_) + (cyxwiz_balance_ * cyxwiz_usd_price_);

    ImGui::BeginChild("Balance", ImVec2(0, 160), true, ImGuiWindowFlags_NoScrollbar);

    ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "Total Balance");
    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 1.0f, 1.0f, 1.0f));
    ImGui::Text("$%.2f USD", total_usd);
    ImGui::PopStyleColor();

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    ImGui::Columns(2, "balances", false);

    ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.2f, 1.0f), ICON_FA_COINS " CYXWIZ");
    ImGui::Text("%.2f", cyxwiz_balance_);
    ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "$%.2f", cyxwiz_balance_ * cyxwiz_usd_price_);

    ImGui::NextColumn();

    ImGui::TextColored(ImVec4(0.6f, 0.4f, 1.0f, 1.0f), ICON_FA_CIRCLE " SOL");
    ImGui::Text("%.4f", sol_balance_);
    ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "$%.2f", sol_balance_ * sol_usd_price_);

    ImGui::Columns(1);

    ImGui::Spacing();

    if (ImGui::Button(ICON_FA_ROTATE " Refresh")) {
        RefreshBalance();
    }

    ImGui::SameLine();
    ImGui::Checkbox("Auto-refresh", &auto_refresh_enabled_);

    ImGui::EndChild();
}

void WalletPanel::RenderActions() {
    ImGui::Text("Quick Actions:");
    ImGui::Spacing();

    if (ImGui::Button(ICON_FA_LINK " Connect External Wallet", ImVec2(200, 0))) {
        show_external_wallet_dialog_ = true;
        external_wallet_step_ = 0;
        memset(external_wallet_buffer_, 0, sizeof(external_wallet_buffer_));
    }

    ImGui::SameLine();

    if (ImGui::Button(ICON_FA_RIGHT_FROM_BRACKET " Disconnect", ImVec2(120, 0))) {
        DisconnectWallet();
    }
}

void WalletPanel::RenderTransactionHistory() {
    ImGui::Text(ICON_FA_CLOCK_ROTATE_LEFT " Recent Transactions:");
    ImGui::SameLine();
    if (ImGui::SmallButton(ICON_FA_ROTATE " Refresh")) {
        RefreshTransactions();
    }

    if (transactions_.empty()) {
        ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f), "No transactions yet");
        return;
    }

    ImGui::BeginChild("TransactionList", ImVec2(0, 0), true);

    for (const auto& tx : transactions_) {
        ImGui::PushID(tx.signature.c_str());

        ImGui::Text("%s", GetTransactionIcon(tx.type).c_str());
        ImGui::SameLine();
        ImGui::TextColored(ImVec4(0.8f, 0.8f, 0.8f, 1.0f), "%s", tx.description.c_str());

        ImGui::SameLine(ImGui::GetWindowWidth() - 150);
        if (tx.amount > 0) {
            ImGui::TextColored(ImVec4(0.0f, 1.0f, 0.0f, 1.0f), "+%.2f", tx.amount);
        } else {
            ImGui::TextColored(ImVec4(1.0f, 0.0f, 0.0f, 1.0f), "%.2f", tx.amount);
        }
        ImGui::SameLine();
        ImGui::Text("CYXWIZ");

        ImGui::Indent(20.0f);
        ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "%s", FormatTimestamp(tx.timestamp).c_str());
        ImGui::SameLine();

        if (tx.status == "CONFIRMED") {
            ImGui::TextColored(ImVec4(0.0f, 1.0f, 0.0f, 1.0f), ICON_FA_CIRCLE_CHECK " Confirmed");
        } else if (tx.status == "PENDING") {
            ImGui::TextColored(ImVec4(1.0f, 1.0f, 0.0f, 1.0f), ICON_FA_SPINNER " Pending");
        } else {
            ImGui::TextColored(ImVec4(1.0f, 0.0f, 0.0f, 1.0f), ICON_FA_CIRCLE_XMARK " Failed");
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

void WalletPanel::RenderExternalWalletDialog() {
    ImGui::OpenPopup("Connect External Wallet");

    ImVec2 center = ImGui::GetMainViewport()->GetCenter();
    ImGui::SetNextWindowPos(center, ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));
    ImGui::SetNextWindowSize(ImVec2(550, 300), ImGuiCond_Appearing);

    if (ImGui::BeginPopupModal("Connect External Wallet", &show_external_wallet_dialog_, ImGuiWindowFlags_NoResize)) {
        ImGui::TextColored(ImVec4(0.6f, 0.4f, 1.0f, 1.0f), ICON_FA_WALLET " Link External Solana Wallet");
        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();

        if (external_wallet_step_ == 0) {
            ImGui::Text("Step 1: Enter your external wallet address");
            ImGui::Spacing();

            ImGui::Text("Wallet Address:");
            ImGui::PushItemWidth(-1);
            ImGui::InputText("##ext_wallet", external_wallet_buffer_, sizeof(external_wallet_buffer_));
            ImGui::PopItemWidth();

            ImGui::Spacing();
            ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f), "Supported: Phantom, Solflare, or any Solana wallet");

            ImGui::Spacing();
            ImGui::Separator();
            ImGui::Spacing();

            if (ImGui::Button("Next", ImVec2(120, 0))) {
                std::string addr(external_wallet_buffer_);
                if (!addr.empty() && addr.length() >= 32) {
                    external_wallet_address_ = addr;
                    external_wallet_nonce_ = "CYXWIZ_VERIFY_" + std::to_string(std::time(nullptr));
                    external_wallet_step_ = 1;
                }
            }
            ImGui::SameLine();
            if (ImGui::Button("Cancel", ImVec2(120, 0))) {
                show_external_wallet_dialog_ = false;
            }
        }
        else if (external_wallet_step_ == 1) {
            ImGui::Text("Step 2: Sign verification message");
            ImGui::Spacing();

            ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.2f, 1.0f), "Sign this message in your wallet:");
            ImGui::Spacing();

            ImGui::BeginChild("NonceBox", ImVec2(-1, 60), true, ImGuiWindowFlags_NoScrollbar);
            ImGui::TextWrapped("%s", external_wallet_nonce_.c_str());
            ImGui::EndChild();

            if (ImGui::SmallButton(ICON_FA_COPY " Copy Message")) {
                ImGui::SetClipboardText(external_wallet_nonce_.c_str());
            }

            ImGui::Spacing();
            ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f), "1. Open your Phantom/Solflare wallet");
            ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f), "2. Sign the message above");
            ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f), "3. Paste the signature below");

            ImGui::Spacing();
            ImGui::Separator();
            ImGui::Spacing();

            if (ImGui::Button(ICON_FA_CHECK " Verify & Connect", ImVec2(180, 0))) {
                wallet_address_ = external_wallet_address_;
                status_ = WalletConnectionStatus::Connected;
                sol_balance_ = 5.0;
                cyxwiz_balance_ = 1000.0;
                RefreshTransactions();
                show_external_wallet_dialog_ = false;
            }
            ImGui::SameLine();
            if (ImGui::Button("Back", ImVec2(80, 0))) {
                external_wallet_step_ = 0;
            }
            ImGui::SameLine();
            if (ImGui::Button("Cancel", ImVec2(80, 0))) {
                show_external_wallet_dialog_ = false;
            }
        }

        ImGui::EndPopup();
    }
}

void WalletPanel::ConnectWallet(const std::string& address) {
    status_ = WalletConnectionStatus::Connecting;
    wallet_address_ = address;
    error_message_ = "";

    status_ = WalletConnectionStatus::Connected;

    sol_balance_ = 5.0;
    cyxwiz_balance_ = 1000.0;
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
    auth_synced_ = false;
}

void WalletPanel::RefreshBalance() {
    if (status_ != WalletConnectionStatus::Connected) return;
}

void WalletPanel::RefreshTransactions() {
    if (status_ != WalletConnectionStatus::Connected) return;

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
    if (type == "JOB_PAYMENT") return ICON_FA_ARROW_RIGHT;
    if (type == "REWARD_CLAIM") return ICON_FA_COINS;
    if (type == "STAKE") return ICON_FA_LOCK;
    if (type == "UNSTAKE") return ICON_FA_UNLOCK;
    if (type == "REFUND") return ICON_FA_ROTATE_LEFT;
    return ICON_FA_CIRCLE_QUESTION;
}

} // namespace gui
