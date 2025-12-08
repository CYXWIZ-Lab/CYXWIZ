// wallet_panel.cpp
#include "gui/panels/wallet_panel.h"
#include "gui/icons.h"
#include "core/backend_manager.h"
#include "core/state_manager.h"
#include "auth/auth_manager.h"
#include <imgui.h>

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

    double balance = state ? state->GetWalletBalance() : 0.0;

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
            // Logged in but no wallet linked
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
            // Not logged in - show manual entry option
            ImGui::TextWrapped("Connect your Solana wallet to receive earnings.");
            ImGui::Spacing();

            ImGui::Text("Wallet Address (Solana)");
            ImGui::InputText("##WalletAddress", wallet_address_, sizeof(wallet_address_));

            ImGui::Spacing();
            if (ImGui::Button(ICON_FA_LINK " Connect Wallet", ImVec2(200, 40))) {
                // TODO: Validate and connect wallet
                if (strlen(wallet_address_) > 0) {
                    is_connected_ = true;
                    // state->UpdateWallet(wallet_address_, 0.0);
                }
            }
        }

        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();

        ImGui::TextDisabled("Supported wallets: Phantom, Solflare, Backpack");
    } else {
        // Connected
        ImGui::Text("Connected Wallet");
        ImGui::TextColored(ImVec4(0.4f, 0.7f, 1.0f, 1.0f), "%s", current_address.c_str());

        if (ImGui::Button(ICON_FA_COPY " Copy")) {
            ImGui::SetClipboardText(current_address.c_str());
        }
        ImGui::SameLine();
        if (ImGui::Button(ICON_FA_LINK_SLASH " Disconnect")) {
            // TODO: Disconnect wallet
        }

        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();

        // Balance
        ImGui::Text("Balance");
        ImGui::PushFont(ImGui::GetIO().Fonts->Fonts[2]);
        ImGui::TextColored(ImVec4(0.4f, 0.8f, 0.4f, 1.0f), "%.4f CYXWIZ", balance);
        ImGui::PopFont();
        ImGui::TextDisabled("~$%.2f USD", balance * 0.20);  // Placeholder rate

        ImGui::Spacing();

        if (ImGui::Button(ICON_FA_ARROW_UP " Withdraw", ImVec2(150, 0))) {
            // TODO: Withdraw funds
        }
    }

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    // Earnings summary
    ImGui::Text("Earnings Summary");

    auto today = state ? state->GetEarningsToday() : core::EarningsInfo{};
    auto week = state ? state->GetEarningsThisWeek() : core::EarningsInfo{};
    auto month = state ? state->GetEarningsThisMonth() : core::EarningsInfo{};

    if (ImGui::BeginTable("Earnings", 2)) {
        ImGui::TableNextRow();
        ImGui::TableNextColumn(); ImGui::Text("Today");
        ImGui::TableNextColumn(); ImGui::Text("%.4f CYXWIZ", today.total_earnings);

        ImGui::TableNextRow();
        ImGui::TableNextColumn(); ImGui::Text("This Week");
        ImGui::TableNextColumn(); ImGui::Text("%.4f CYXWIZ", week.total_earnings);

        ImGui::TableNextRow();
        ImGui::TableNextColumn(); ImGui::Text("This Month");
        ImGui::TableNextColumn(); ImGui::Text("%.4f CYXWIZ", month.total_earnings);

        ImGui::EndTable();
    }

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    // Transaction history
    ImGui::Text("Recent Transactions");
    ImGui::BeginChild("Transactions", ImVec2(0, 150), true);
    ImGui::TextDisabled("No transactions yet");
    ImGui::EndChild();
}

} // namespace cyxwiz::servernode::gui
