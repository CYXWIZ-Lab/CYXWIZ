// wallet_panel.cpp
#include "gui/panels/wallet_panel.h"
#include "gui/icons.h"
#include "core/backend_manager.h"
#include "core/state_manager.h"
#include "auth/auth_manager.h"
#include <imgui.h>
#include <ctime>
#include <sstream>
#include <iomanip>

namespace cyxwiz::servernode::gui {

void WalletPanel::Render() {
    ImGui::PushFont(ImGui::GetIO().Fonts->Fonts[2]);
    ImGui::Text("%s Wallet", ICON_FA_WALLET);
    ImGui::PopFont();
    ImGui::Separator();
    ImGui::Spacing();

    auto* state = GetState();
    auto& auth = auth::AuthManager::Instance();

    // Get wallet address - prefer auth user's wallet, fallback to state
    std::string current_address;
    if (auth.IsAuthenticated()) {
        auto user_info = auth.GetUserInfo();
        current_address = user_info.wallet_address;
    }
    if (current_address.empty() && state) {
        current_address = state->GetWalletAddress();
    }

    // Show login prompt if not authenticated
    if (!auth.IsAuthenticated()) {
        ImGui::TextColored(ImVec4(0.8f, 0.8f, 0.3f, 1.0f),
            ICON_FA_TRIANGLE_EXCLAMATION " Not Logged In");
        ImGui::Spacing();
        ImGui::TextWrapped(
            "Sign in to your CyxWiz account to use your linked wallet and view earnings.");
        ImGui::Spacing();
        ImGui::TextDisabled("Go to Login to sign in.");
        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();
    }

    if (current_address.empty()) {
        // Not connected
        if (auth.IsAuthenticated()) {
            ImGui::TextColored(ImVec4(0.8f, 0.5f, 0.2f, 1.0f),
                ICON_FA_LINK_SLASH " No Wallet Linked");
            ImGui::Spacing();
            ImGui::TextWrapped(
                "Your CyxWiz account doesn't have a wallet linked yet. "
                "Link a Solana wallet on the web dashboard to receive earnings.");
            ImGui::Spacing();

            if (ImGui::Button(ICON_FA_ARROW_UP_RIGHT_FROM_SQUARE " Link Wallet on Dashboard", ImVec2(250, 35))) {
#ifdef _WIN32
                std::system("start https://cyxwiz.com/dashboard/wallet");
#elif __APPLE__
                std::system("open https://cyxwiz.com/dashboard/wallet");
#else
                std::system("xdg-open https://cyxwiz.com/dashboard/wallet");
#endif
            }
        } else {
            ImGui::TextWrapped("Connect your Solana wallet to receive earnings.");
            ImGui::Spacing();

            ImGui::Text("Wallet Address (Solana)");
            ImGui::InputText("##WalletAddress", wallet_address_, sizeof(wallet_address_));

            ImGui::Spacing();
            if (ImGui::Button(ICON_FA_LINK " Connect Wallet", ImVec2(200, 40))) {
                if (strlen(wallet_address_) > 0) {
                    is_connected_ = true;
                }
            }
        }

        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();
        ImGui::TextDisabled("Supported wallets: Phantom, Solflare, Backpack");
    } else {
        ImGui::Text(ICON_FA_WALLET " Connected Wallet");
        ImGui::TextColored(ImVec4(0.4f, 0.7f, 1.0f, 1.0f), "%s", FormatAddress(current_address).c_str());
        
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("%s", current_address.c_str());
        }

        ImGui::SameLine();
        if (ImGui::SmallButton(ICON_FA_COPY " Copy")) {
            ImGui::SetClipboardText(current_address.c_str());
        }
        ImGui::SameLine();
        if (ImGui::SmallButton(ICON_FA_ARROW_UP_RIGHT_FROM_SQUARE " Explorer")) {
            std::string url = "start https://explorer.solana.com/address/" + current_address + "?cluster=devnet";
#ifdef _WIN32
            std::system(url.c_str());
#endif
        }

        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();

        RenderBalanceSection(current_address, sol_balance_, cyxwiz_balance_);

        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();

        ImGui::Text(ICON_FA_CHART_LINE " Earnings Summary");

        auto today = state ? state->GetEarningsToday() : core::EarningsInfo{};
        auto week = state ? state->GetEarningsThisWeek() : core::EarningsInfo{};
        auto month = state ? state->GetEarningsThisMonth() : core::EarningsInfo{};

        if (ImGui::BeginTable("Earnings", 3, ImGuiTableFlags_BordersInnerV)) {
            ImGui::TableSetupColumn("Period");
            ImGui::TableSetupColumn("CYXWIZ");
            ImGui::TableSetupColumn("USD");
            ImGui::TableHeadersRow();

            ImGui::TableNextRow();
            ImGui::TableNextColumn(); ImGui::Text("Today");
            ImGui::TableNextColumn(); ImGui::Text("%.4f", today.total_earnings);
            ImGui::TableNextColumn(); ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "$%.2f", today.total_earnings * cyxwiz_usd_price_);

            ImGui::TableNextRow();
            ImGui::TableNextColumn(); ImGui::Text("This Week");
            ImGui::TableNextColumn(); ImGui::Text("%.4f", week.total_earnings);
            ImGui::TableNextColumn(); ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "$%.2f", week.total_earnings * cyxwiz_usd_price_);

            ImGui::TableNextRow();
            ImGui::TableNextColumn(); ImGui::Text("This Month");
            ImGui::TableNextColumn(); ImGui::Text("%.4f", month.total_earnings);
            ImGui::TableNextColumn(); ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "$%.2f", month.total_earnings * cyxwiz_usd_price_);

            ImGui::EndTable();
        }

        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();

        RenderTransactionHistory();
    }
}

void WalletPanel::RenderBalanceSection(const std::string& wallet_address, double sol_balance, double cyxwiz_balance) {
    double total_usd = (sol_balance * sol_usd_price_) + (cyxwiz_balance * cyxwiz_usd_price_);

    ImGui::BeginChild("Balance", ImVec2(0, 140), true, ImGuiWindowFlags_NoScrollbar);

    ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "Total Balance");
    ImGui::PushFont(ImGui::GetIO().Fonts->Fonts[2]);
    ImGui::Text("$%.2f USD", total_usd);
    ImGui::PopFont();

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    ImGui::Columns(2, "balances", false);

    ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.2f, 1.0f), ICON_FA_COINS " CYXWIZ");
    ImGui::Text("%.2f", cyxwiz_balance);
    ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "$%.2f", cyxwiz_balance * cyxwiz_usd_price_);

    ImGui::NextColumn();

    ImGui::TextColored(ImVec4(0.6f, 0.4f, 1.0f, 1.0f), ICON_FA_CIRCLE " SOL");
    ImGui::Text("%.4f", sol_balance);
    ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "$%.2f", sol_balance * sol_usd_price_);

    ImGui::Columns(1);

    ImGui::Spacing();

    if (ImGui::Button(ICON_FA_ROTATE " Refresh")) {
        // TODO: Refresh balance from blockchain
    }

    ImGui::EndChild();
}

void WalletPanel::RenderTransactionHistory() {
    ImGui::Text(ICON_FA_CLOCK_ROTATE_LEFT " Recent Transactions");
    ImGui::SameLine();
    if (ImGui::SmallButton(ICON_FA_ROTATE " Refresh##tx")) {
        // TODO: Refresh transactions
    }

    ImGui::BeginChild("Transactions", ImVec2(0, 150), true);
    
    if (transactions_.empty()) {
        ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f), "No transactions yet");
    } else {
        for (const auto& tx : transactions_) {
            ImGui::PushID(tx.signature.c_str());
            
            if (tx.type == "EARNING") {
                ImGui::TextColored(ImVec4(0.0f, 1.0f, 0.0f, 1.0f), ICON_FA_ARROW_DOWN);
            } else if (tx.type == "WITHDRAW") {
                ImGui::TextColored(ImVec4(1.0f, 0.0f, 0.0f, 1.0f), ICON_FA_ARROW_UP);
            } else {
                ImGui::Text(ICON_FA_ARROW_RIGHT);
            }
            
            ImGui::SameLine();
            ImGui::Text("%s", tx.description.c_str());
            
            ImGui::SameLine(ImGui::GetWindowWidth() - 120);
            if (tx.amount > 0) {
                ImGui::TextColored(ImVec4(0.0f, 1.0f, 0.0f, 1.0f), "+%.4f", tx.amount);
            } else {
                ImGui::TextColored(ImVec4(1.0f, 0.0f, 0.0f, 1.0f), "%.4f", tx.amount);
            }
            
            ImGui::Indent(20.0f);
            ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "%s", FormatTimestamp(tx.timestamp).c_str());
            ImGui::SameLine();
            if (tx.status == "CONFIRMED") {
                ImGui::TextColored(ImVec4(0.0f, 1.0f, 0.0f, 1.0f), ICON_FA_CIRCLE_CHECK);
            } else if (tx.status == "PENDING") {
                ImGui::TextColored(ImVec4(1.0f, 1.0f, 0.0f, 1.0f), ICON_FA_SPINNER);
            }
            ImGui::Unindent(20.0f);
            
            ImGui::PopID();
        }
    }
    
    ImGui::EndChild();
}

std::string WalletPanel::FormatAddress(const std::string& address) const {
    if (address.length() <= 16) return address;
    return address.substr(0, 8) + "..." + address.substr(address.length() - 8);
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

} // namespace cyxwiz::servernode::gui
