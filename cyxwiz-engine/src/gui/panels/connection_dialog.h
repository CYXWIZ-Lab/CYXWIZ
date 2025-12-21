#pragma once

#include <string>
#include <functional>
#include <vector>
#include <chrono>
#include "../../network/grpc_client.h"

namespace network {
    class GRPCClient;
    class JobManager;
}

namespace gui {
    class NodeEditor;
}

namespace cyxwiz {

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

private:
    void RenderConnectionPanel();
    void RenderNodeDiscoveryPanel();
    void RenderNodeTable();
    void RenderNodeSearchFilters();
    void RenderSelectedNodeInfo();
    void RenderJobSubmitPanel();
    void RenderActiveJobsPanel();

    // Node discovery actions
    void RefreshNodeList();
    void SearchNodes();

    network::GRPCClient* client_;
    network::JobManager* job_manager_;

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

    // Job submission fields
    char model_definition_[1024];
    char dataset_uri_[512];
    bool submitting_job_;
    std::string last_submitted_job_id_;

    std::function<void(bool)> connection_callback_;
    gui::NodeEditor* node_editor_ = nullptr;
};

} // namespace cyxwiz
