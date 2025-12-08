// pool_mining_panel.h - Pool mining with daemon integration
#pragma once
#include "gui/server_panel.h"
#include "ipc/daemon_client.h"
#include <string>

namespace cyxwiz::servernode::gui {

class PoolMiningPanel : public ServerPanel {
public:
    PoolMiningPanel() : ServerPanel("Pool Mining") {}
    void Render() override;

private:
    void RefreshStatus();
    void RenderPoolSelector();
    void RenderMiningStats();
    void RenderConfiguration();
    void RenderControls();
    std::string FormatDuration(int64_t seconds);
    std::string FormatHashrate(float mhs);

    // Daemon data
    ipc::PoolStatus pool_status_;
    bool status_loaded_ = false;

    // UI state
    int selected_pool_ = 0;
    float intensity_slider_ = 0.5f;
    bool auto_start_ = false;
    bool mine_when_idle_ = true;

    // Error handling
    std::string error_message_;
    bool show_join_error_ = false;
    bool show_error_popup_ = false;

    // Available pools (hardcoded for now)
    static const char* GetPoolName(int index);
    static const char* GetPoolAddress(int index);
    static const char* GetPoolFee(int index);
    static constexpr int NUM_POOLS = 3;
};

} // namespace cyxwiz::servernode::gui
