// deployment_panel.h - Deploy and manage model deployments
#pragma once
#include "gui/server_panel.h"
#include "ipc/daemon_client.h"
#include <vector>
#include <string>

namespace cyxwiz::servernode::gui {

class DeploymentPanel : public ServerPanel {
public:
    DeploymentPanel() : ServerPanel("Deployment") {}
    void Render() override;

private:
    void RenderDeploySection();
    void RenderActiveDeployments();
    void RenderUndeployDialog();
    void RefreshModels();
    void RefreshDeployments();
    const char* GetStatusText(int status);
    ImVec4 GetStatusColor(int status);

    // Model list for dropdown
    std::vector<ipc::ModelInfo> models_;
    bool models_loaded_ = false;
    int selected_model_idx_ = -1;

    // Deploy configuration
    int port_ = 8080;
    int context_size_ = 4096;
    int gpu_layers_ = 35;
    bool deploying_ = false;
    std::string deploy_error_;
    std::string deploy_success_;

    // Active deployments
    std::vector<ipc::DeploymentInfo> deployments_;
    bool deployments_loaded_ = false;

    // Undeploy confirmation dialog
    bool show_undeploy_dialog_ = false;
    std::string pending_undeploy_id_;
    std::string pending_undeploy_name_;
    std::string undeploy_error_;
};

} // namespace cyxwiz::servernode::gui
