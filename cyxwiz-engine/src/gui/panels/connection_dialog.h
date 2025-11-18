#pragma once

#include <string>
#include <functional>

namespace network {
    class GRPCClient;
    class JobManager;
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

private:
    void RenderConnectionPanel();
    void RenderJobSubmitPanel();
    void RenderActiveJobsPanel();

    network::GRPCClient* client_;
    network::JobManager* job_manager_;

    bool show_;
    char server_address_[256];
    bool connecting_;
    std::string connection_error_;

    // Job submission fields
    char model_definition_[1024];
    char dataset_uri_[512];
    bool submitting_job_;
    std::string last_submitted_job_id_;

    std::function<void(bool)> connection_callback_;
};

} // namespace cyxwiz
