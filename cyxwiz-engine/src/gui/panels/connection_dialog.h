#pragma once

#include "../../core/graph_job_memory_probe.h"
#include "../../core/remote_job_presentation.h"
#include "../../core/reservation_presentation.h"
#include <atomic>
#include <mutex>
#include <future>
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

    // Set P2P training panel for monitoring
    // Also wires the panel's Resume / Start over actions (TOFIX118 P4 GUI).
    void SetP2PTrainingPanel(P2PTrainingPanel* panel);

    // Set reservation and P2P clients
    void SetReservationClient(std::shared_ptr<network::ReservationClient> client);
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
    // The reservation card (connection_dialog_reservation.cpp, TOFIX118
    // gaps 5-6, mockup approved 2026-09-30).
    void RenderReservationPanel();        // reserve quote for the selected node
    void RenderActiveReservationPanel();  // time left, Extend, Leave, details
    void RenderTrainingOnNodePanel();     // fit card, start / stop, disconnect
    void RenderLeaveNodePopup();
    void RenderStopTrainingPopup();
    void RenderReservationReceipt();      // after a reservation ended
    void RenderReconnectPrompt();         // active reservations found on connect
    void RenderReservationError();        // reservation_error_, in every state
    static ReservationNodeFacts FactsFor(const network::NodeDisplayInfo& node);
    ActiveReservationInputs CurrentReservationInputs() const;
    const network::NodeDisplayInfo* SelectedNode() const;
    // RenderActiveJobsPanel removed - jobs tracked via P2P Training Progress panel

    // Node discovery actions
    void RefreshNodeList();
    void SearchNodes();

    // Reservation actions
    void StartReservation();
    // Leave the node (after the confirm popup): the reservation keeps running
    // and moves to the reconnect prompt; the reserved time stays the user's
    // (owner rule 2026-10-07: no early end, nothing returned).
    void LeaveNode();
    // The reservation is over: disconnect, stop the heartbeat, keep a receipt.
    void FinishReservation(ReservationEndReason reason, const std::string& error);
    void DisconnectFromNode();    // P2P only; the card stays
    // Add minutes to the active reservation and hand the node the new end
    // (TOFIX118 gap 5). False with reservation_error_ set when refused.
    bool ExtendActiveReservation(int additional_minutes);
    void ConnectToReservedNode();
    void StartP2PTraining();     // Send job directly to Server Node via P2P
    // Start a new job within the same reservation; with resume_job_id, send
    // that job again so the node continues from its checkpoint (P4e-3).
    void StartNewP2PTraining(const std::string& resume_job_id = std::string());
    // "Will this job fit?" (connection_dialog_job_fit.cpp, TOFIX118 P4 GUI):
    // measure the job once per graph and batch size, then draw the card.
    void UpdateJobEstimate();
    cyxwiz::JobFitCard CurrentJobFitCard() const;
    void RenderJobFitCard();

    // Reconnection support (after Engine restart)
    void CheckForActiveReservations();  // lists them in the reconnect prompt
    void ReconnectToReservation(const std::string& reservation_id);  // one of found_reservations_

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
    bool show_leave_confirm_ = false;     // opens the Leave node popup
    bool show_stop_confirm_ = false;      // opens the Stop training popup
    bool quote_details_open_ = false;
    bool reservation_details_open_ = false;
    bool receipt_details_open_ = false;
    int reservation_duration_minutes_ = 60;
    int reservation_epochs_ = 10;           // Default training epochs
    int reservation_batch_size_ = 32;       // Default batch size
    bool reserving_ = false;
    std::string reservation_error_;
    network::ReservationInfo active_reservation_;
    bool has_active_reservation_ = false;
    // From the Central Server's heartbeat (every 30 s, heartbeat thread):
    // seconds left, -1 before the first reply; and its "extend soon" hint.
    std::atomic<long long> reservation_seconds_left_{-1};
    std::atomic<long long> reservation_heartbeat_at_{0};  // Unix seconds of that reply
    std::atomic<bool> reservation_should_extend_{false};
    // The heartbeat found the reservation gone (heartbeat thread); the
    // render thread turns it into the receipt.
    std::atomic<bool> reservation_gone_{false};
    std::mutex reservation_gone_mutex_;
    std::string reservation_gone_message_;
    long long expired_at_ = 0;       // when the card first saw time run out
    int jobs_started_ = 0;           // in the active reservation
    bool has_receipt_ = false;       // show the receipt of the last reservation
    ReservationEndFacts receipt_facts_;
    std::vector<ActiveReservationListing> found_reservations_;  // reconnect prompt
    network::NodeDisplayInfo reserved_node_;  // the listing the reservation was made from

    // Job memory estimate (ProbeGraphJobMemory on a worker), keyed by the
    // graph and batch size it measured.
    std::future<cyxwiz::GraphJobMemoryProbe> estimate_future_;
    cyxwiz::GraphJobMemoryProbe estimate_result_;
    cyxwiz::JobEstimateState estimate_state_ = cyxwiz::JobEstimateState::NotStarted;
    std::size_t estimate_key_ = 0;
    std::string estimate_device_;
    double estimate_next_check_ = 0.0;
    std::string last_rejection_reason_;  // the node's words for the last refusal
    bool fit_details_open_ = false;

    // Dataset URI for P2P training
    char dataset_uri_[512];

    std::function<void(bool)> connection_callback_;
    gui::NodeEditor* node_editor_ = nullptr;
    P2PTrainingPanel* p2p_training_panel_ = nullptr;
};

} // namespace cyxwiz
