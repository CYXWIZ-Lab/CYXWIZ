// pool_mining_panel.cpp - Pool mining with daemon integration
#include "gui/panels/pool_mining_panel.h"
#include "gui/icons.h"
#include <imgui.h>
#include <spdlog/spdlog.h>

namespace cyxwiz::servernode::gui {

// Static pool data
const char* PoolMiningPanel::GetPoolName(int index) {
    static const char* names[] = {
        "CyxWiz Official Pool",
        "Community Pool Alpha",
        "Community Pool Beta"
    };
    return (index >= 0 && index < NUM_POOLS) ? names[index] : "Unknown";
}

const char* PoolMiningPanel::GetPoolAddress(int index) {
    static const char* addresses[] = {
        "pool.cyxwiz.io:3333",
        "alpha.cyxpool.net:3333",
        "beta.cyxpool.net:3333"
    };
    return (index >= 0 && index < NUM_POOLS) ? addresses[index] : "";
}

const char* PoolMiningPanel::GetPoolFee(int index) {
    static const char* fees[] = {"1%", "0.5%", "0.8%"};
    return (index >= 0 && index < NUM_POOLS) ? fees[index] : "N/A";
}

void PoolMiningPanel::Render() {
    ImGui::PushFont(GetSafeFont(FONT_LARGE));
    ImGui::Text("%s Pool Mining", ICON_FA_COINS);
    ImGui::PopFont();
    ImGui::Separator();

    // Connection status
    if (IsDaemonConnected()) {
        ImGui::TextColored(ImVec4(0.3f, 0.8f, 0.3f, 1.0f), "%s Daemon Connected", ICON_FA_LINK);
    } else {
        ImGui::TextColored(ImVec4(0.8f, 0.3f, 0.3f, 1.0f), "%s Daemon Disconnected", ICON_FA_LINK_SLASH);
        ImGui::TextDisabled("Connect to daemon to configure pool mining.");
        return;
    }

    ImGui::Spacing();

    // Refresh button
    if (ImGui::Button(ICON_FA_ROTATE " Refresh")) {
        RefreshStatus();
    }

    ImGui::Spacing();
    ImGui::TextWrapped("Join a mining pool to earn CYXWIZ rewards by contributing your compute power.");
    ImGui::Spacing();

    // Load status on first render
    if (!status_loaded_) {
        RefreshStatus();
        status_loaded_ = true;
    }

    // Render sections
    RenderPoolSelector();
    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    RenderMiningStats();
    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    RenderConfiguration();
    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    RenderControls();

    // Error popup
    if (show_error_popup_) {
        ImGui::OpenPopup("Mining Error");
        show_error_popup_ = false;
    }

    if (ImGui::BeginPopupModal("Mining Error", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
        ImGui::Text("%s %s", ICON_FA_TRIANGLE_EXCLAMATION, error_message_.c_str());
        ImGui::Spacing();
        if (ImGui::Button("OK", ImVec2(120, 0))) {
            ImGui::CloseCurrentPopup();
            error_message_.clear();
        }
        ImGui::EndPopup();
    }
}

void PoolMiningPanel::RefreshStatus() {
    auto* client = GetDaemonClient();
    if (client && client->IsConnected()) {
        if (client->GetPoolStatus(pool_status_)) {
            intensity_slider_ = pool_status_.mining_intensity;
            spdlog::debug("Pool status refreshed: mining={}, intensity={:.0f}%",
                          pool_status_.is_mining, pool_status_.mining_intensity * 100);
        }
    }
}

void PoolMiningPanel::RenderPoolSelector() {
    ImGui::Text("%s Select Pool", ICON_FA_SERVER);

    // Pool dropdown
    const char* pool_names[NUM_POOLS];
    for (int i = 0; i < NUM_POOLS; i++) {
        pool_names[i] = GetPoolName(i);
    }

    ImGui::SetNextItemWidth(300);
    if (ImGui::Combo("##Pool", &selected_pool_, pool_names, NUM_POOLS)) {
        // Pool selection changed
    }

    // Show selected pool info
    ImGui::TextDisabled("Address: %s", GetPoolAddress(selected_pool_));
    ImGui::TextDisabled("Fee: %s", GetPoolFee(selected_pool_));

    // Current pool status
    ImGui::Spacing();
    if (pool_status_.is_joined) {
        ImGui::TextColored(ImVec4(0.3f, 0.8f, 0.3f, 1.0f), "%s Joined: %s",
                           ICON_FA_CIRCLE_CHECK, pool_status_.pool_name.c_str());

        // Pool stats
        ImGui::TextDisabled("Active Miners: %d | Pool Hashrate: %.2f MH/s",
                            pool_status_.active_miners, pool_status_.pool_hashrate);

        if (ImGui::Button(ICON_FA_RIGHT_FROM_BRACKET " Leave Pool")) {
            auto* client = GetDaemonClient();
            if (client) {
                std::string error;
                if (client->LeavePool(error)) {
                    spdlog::info("Left pool: {}", pool_status_.pool_name);
                    RefreshStatus();
                } else {
                    error_message_ = error.empty() ? "Failed to leave pool" : error;
                    show_error_popup_ = true;
                }
            }
        }
    } else {
        ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.3f, 1.0f), "%s Not joined to any pool",
                           ICON_FA_CIRCLE_EXCLAMATION);

        if (ImGui::Button(ICON_FA_RIGHT_TO_BRACKET " Join Pool")) {
            auto* client = GetDaemonClient();
            if (client) {
                std::string error;
                if (client->JoinPool(GetPoolAddress(selected_pool_), error)) {
                    spdlog::info("Joined pool: {}", GetPoolName(selected_pool_));
                    RefreshStatus();
                } else {
                    error_message_ = error.empty() ? "Failed to join pool" : error;
                    show_error_popup_ = true;
                }
            }
        }
    }
}

void PoolMiningPanel::RenderMiningStats() {
    ImGui::Text("%s Mining Stats", ICON_FA_CHART_LINE);

    ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding, 5.0f);
    ImGui::BeginChild("MiningStats", ImVec2(0, 150), true);

    if (pool_status_.is_mining) {
        const auto& stats = pool_status_.stats;

        // Hashrate
        ImGui::Text("Hashrate:");
        ImGui::SameLine(150);
        ImGui::TextColored(ImVec4(0.3f, 0.8f, 0.9f, 1.0f), "%s", FormatHashrate(stats.hashrate_mhs).c_str());

        // Shares
        ImGui::Text("Shares:");
        ImGui::SameLine(150);
        int total = (int)(stats.shares_accepted + stats.shares_rejected);
        float acceptance = total > 0 ? (float)stats.shares_accepted / total * 100.0f : 100.0f;
        ImGui::Text("%lld / %lld", (long long)stats.shares_accepted, (long long)stats.shares_submitted);
        ImGui::SameLine();
        ImVec4 color = acceptance >= 95.0f ? ImVec4(0.3f, 0.8f, 0.3f, 1.0f) :
                       acceptance >= 80.0f ? ImVec4(0.8f, 0.8f, 0.3f, 1.0f) :
                                             ImVec4(0.8f, 0.3f, 0.3f, 1.0f);
        ImGui::TextColored(color, "(%.1f%% accepted)", acceptance);

        // Estimated earnings
        ImGui::Spacing();
        ImGui::Text("Est. Daily:");
        ImGui::SameLine(150);
        ImGui::TextColored(ImVec4(0.3f, 0.9f, 0.3f, 1.0f), "%.4f CYXWIZ", stats.estimated_daily);

        ImGui::Text("Est. Monthly:");
        ImGui::SameLine(150);
        ImGui::TextColored(ImVec4(0.3f, 0.9f, 0.3f, 1.0f), "%.4f CYXWIZ", stats.estimated_monthly);

        // Pool earnings total
        ImGui::Spacing();
        ImGui::Text("Total Earned:");
        ImGui::SameLine(150);
        ImGui::TextColored(ImVec4(0.9f, 0.8f, 0.3f, 1.0f), "%.4f CYXWIZ", pool_status_.pool_earnings);

        // Mining uptime
        ImGui::Text("Uptime:");
        ImGui::SameLine(150);
        ImGui::TextDisabled("%s", FormatDuration(stats.mining_uptime_seconds).c_str());
    } else {
        ImGui::TextDisabled("Mining is not active.");
        ImGui::TextDisabled("Start mining to see statistics.");
    }

    ImGui::EndChild();
    ImGui::PopStyleVar();
}

void PoolMiningPanel::RenderConfiguration() {
    ImGui::Text("%s Configuration", ICON_FA_SLIDERS);

    // Intensity slider
    ImGui::Text("Mining Intensity");
    ImGui::SetNextItemWidth(300);
    if (ImGui::SliderFloat("##Intensity", &intensity_slider_, 0.0f, 1.0f, "%.0f%%")) {
        // Apply intensity change to daemon
        auto* client = GetDaemonClient();
        if (client && client->IsConnected()) {
            std::string error;
            if (!client->SetMiningIntensity(intensity_slider_, error)) {
                spdlog::warn("Failed to set mining intensity: {}", error);
            }
        }
    }

    // Intensity explanation
    if (intensity_slider_ < 0.3f) {
        ImGui::TextDisabled("Low: Minimal GPU usage, lower earnings");
    } else if (intensity_slider_ < 0.7f) {
        ImGui::TextDisabled("Medium: Balanced performance and earnings");
    } else {
        ImGui::TextDisabled("High: Maximum earnings, higher GPU usage");
    }

    ImGui::Spacing();

    // Options (local UI state - these would need daemon support to persist)
    ImGui::Checkbox("Auto-start mining on daemon startup", &auto_start_);
    ImGui::Checkbox("Mine only when GPU is idle", &mine_when_idle_);
}

void PoolMiningPanel::RenderControls() {
    // Main mining control button
    if (pool_status_.is_mining) {
        // Stop button - red
        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.6f, 0.2f, 0.2f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.7f, 0.3f, 0.3f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.5f, 0.1f, 0.1f, 1.0f));

        if (ImGui::Button(ICON_FA_STOP " Stop Mining", ImVec2(200, 40))) {
            auto* client = GetDaemonClient();
            if (client) {
                std::string error;
                if (client->StopMining(error)) {
                    spdlog::info("Mining stopped");
                    RefreshStatus();
                } else {
                    error_message_ = error.empty() ? "Failed to stop mining" : error;
                    show_error_popup_ = true;
                }
            }
        }

        ImGui::PopStyleColor(3);

        ImGui::SameLine();
        ImGui::TextDisabled("Mining for %s", FormatDuration(pool_status_.stats.mining_uptime_seconds).c_str());

        // Show current status indicator
        ImGui::SameLine(ImGui::GetContentRegionAvail().x - 100);
        ImGui::TextColored(ImVec4(0.3f, 0.9f, 0.3f, 1.0f), "%s MINING", ICON_FA_CIRCLE);
    } else {
        // Check if joined to pool first
        if (!pool_status_.is_joined) {
            ImGui::BeginDisabled();
            ImGui::Button(ICON_FA_PLAY " Start Mining", ImVec2(200, 40));
            ImGui::EndDisabled();
            ImGui::SameLine();
            ImGui::TextDisabled("Join a pool first to start mining");
        } else {
            // Start button - green
            ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.2f, 0.5f, 0.2f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.3f, 0.6f, 0.3f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.1f, 0.4f, 0.1f, 1.0f));

            if (ImGui::Button(ICON_FA_PLAY " Start Mining", ImVec2(200, 40))) {
                auto* client = GetDaemonClient();
                if (client) {
                    std::string error;
                    if (client->StartMining(error)) {
                        spdlog::info("Mining started");
                        RefreshStatus();
                    } else {
                        error_message_ = error.empty() ? "Failed to start mining" : error;
                        show_error_popup_ = true;
                    }
                }
            }

            ImGui::PopStyleColor(3);

            ImGui::SameLine();
            ImGui::TextDisabled("Ready to mine");
        }
    }
}

std::string PoolMiningPanel::FormatDuration(int64_t seconds) {
    if (seconds < 60) {
        return std::to_string(seconds) + "s";
    } else if (seconds < 3600) {
        return std::to_string(seconds / 60) + "m " + std::to_string(seconds % 60) + "s";
    } else if (seconds < 86400) {
        int64_t hours = seconds / 3600;
        int64_t mins = (seconds % 3600) / 60;
        return std::to_string(hours) + "h " + std::to_string(mins) + "m";
    } else {
        int64_t days = seconds / 86400;
        int64_t hours = (seconds % 86400) / 3600;
        return std::to_string(days) + "d " + std::to_string(hours) + "h";
    }
}

std::string PoolMiningPanel::FormatHashrate(float mhs) {
    if (mhs >= 1000.0f) {
        return std::to_string((int)(mhs / 1000.0f)) + "." +
               std::to_string((int)(mhs) % 1000 / 100) + " GH/s";
    } else if (mhs >= 1.0f) {
        return std::to_string((int)mhs) + "." +
               std::to_string((int)(mhs * 10) % 10) + " MH/s";
    } else {
        return std::to_string((int)(mhs * 1000)) + " KH/s";
    }
}

} // namespace cyxwiz::servernode::gui
