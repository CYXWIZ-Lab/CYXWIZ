#pragma once

#include <string>
#include <functional>
#include <vector>
#include <chrono>
#include <memory>
#include "../../network/grpc_client.h"
#include "../../network/reservation_client.h"
#include "../../network/p2p_client.h"
#include "p2p_training_panel.h"

namespace network {
    class GRPCClient;
    class JobManager;
    class ReservationClient;
    class P2PClient;
}

namespace gui {
    class NodeEditor;
    class WalletPanel;
}

namespace cyxwiz {

class P2PTrainingPanel;

class ConnectionDialog {
public:
    ConnectionDialog(network::GRPCClient* client, network::JobManager* job_manager);
    ~ConnectionDialog();

    void Render();
    void Show() { show_ = true; }
    void Hide() { show_ = false; }
    bool IsVisible() const { return show_; }

    // Set callback for when connection state changes
    void SetConnectionCallback(std::function<void(bool)> callback) {
        connection_callback_ = callback;
    }

    // Set node editor for accessing model graph
    void SetNodeEditor(gui::NodeEditor* editor) { node_editor_ = editor; }

    // Set wallet panel for accessing wallet address
    void SetWalletPanel(gui::WalletPanel* panel) { wallet_panel_ = panel; }

    // Set P2P training panel for monitoring
    void SetP2PTrainingPanel(P2PTrainingPanel* panel) { p2p_training_panel_ = panel; }

    // Set reservation and P2P clients
    void SetReservationClient(std::shared_ptr<network::ReservationClient> client) {
        reservation_client_ = client;
    }
    void SetP2PClient(std::shared_ptr<network::P2PClient> client) {
        p2p_client_ = client;
    }

    // Get active reservation info
    bool HasActiveReservation() const;
    const network::ReservationInfo& GetReservation() const;

private:
    void RenderConnectionPanel();
    void RenderNodeDiscoveryPanel();
    void RenderNodeTable();
    void RenderNodeSearchFilters();
    void RenderSelectedNodeInfo();
    void RenderReservationPanel();       // Reserve node UI
    void RenderReservationConfirmDialog(); // Confirmation dialog
    void RenderActiveReservationPanel();  // Show active reservation
    // RenderActiveJobsPanel removed - jobs tracked via P2P Training Progress panel

    // Node discovery actions
    void RefreshNodeList();
    void SearchNodes();

    // Reservation actions
    void StartReservation();
    void CancelReservation();
    void ConnectToReservedNode();
    void StartP2PTraining();     // Send job directly to Server Node via P2P
    void StartNewP2PTraining();  // Start new job within same reservation

    network::GRPCClient* client_;
    network::JobManager* job_manager_;
    std::shared_ptr<network::ReservationClient> reservation_client_;
    std::shared_ptr<network::P2PClient> p2p_client_;

    bool show_;
    char server_address_[256];
    bool connecting_;
    std::string connection_error_;

    // Node discovery state
    std::vector<network::NodeDisplayInfo> discovered_nodes_;
    int selected_node_index_ = -1;
    std::string selected_node_id_;
    bool show_search_filters_ = false;
    network::NodeSearchCriteria search_criteria_;

    // Search filter UI buffers
    int filter_device_type_ = 0;        // 0=Any, 1=CUDA, 2=OpenCL, 3=CPU
    float filter_min_vram_gb_ = 0.0f;
    float filter_max_price_ = 0.0f;
    float filter_min_reputation_ = 0.0f;
    bool filter_free_tier_only_ = false;
    char filter_region_[64] = "";
    int filter_sort_by_ = 0;

    // Node list refresh timing
    std::chrono::steady_clock::time_point last_node_refresh_time_;
    static constexpr float node_refresh_interval_seconds_ = 10.0f;

    // Reservation state
    bool show_reservation_confirm_ = false;
    int reservation_duration_minutes_ = 60;
    int reservation_epochs_ = 10;           // Default training epochs
    int reservation_batch_size_ = 32;       // Default batch size
    char user_wallet_[128] = "";
    bool reserving_ = false;
    std::string reservation_error_;
    network::ReservationInfo active_reservation_;
    bool has_active_reservation_ = false;

    // Dataset URI for P2P training
    char dataset_uri_[512];

    std::function<void(bool)> connection_callback_;
    gui::NodeEditor* node_editor_ = nullptr;
    gui::WalletPanel* wallet_panel_ = nullptr;
    P2PTrainingPanel* p2p_training_panel_ = nullptr;
};

} // namespace cyxwiz
