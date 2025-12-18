// model_browser_panel.h - Browse and manage models
#pragma once
#include "gui/server_panel.h"
#include "ipc/daemon_client.h"
#include <vector>
#include <string>

namespace cyxwiz::servernode::gui {

class ModelBrowserPanel : public ServerPanel {
public:
    ModelBrowserPanel() : ServerPanel("Model Browser") {}
    void Render() override;

    // Set model for deployment (called from external code)
    void SetModelForDeploy(const std::string& model_path) {
        pending_deploy_model_ = model_path;
        show_deploy_dialog_ = true;
    }

private:
    void RenderLocalModels();
    void RenderDownloaded();
    void RenderMarketplace();
    void RenderDeployDialog();
    void RenderDeleteConfirmDialog();
    void ScanLocalModels();
    void RefreshModels();

    int selected_tab_ = 0;
    std::vector<ipc::ModelInfo> local_models_;
    bool models_scanned_ = false;
    char search_query_[256] = "";

    // Deployment dialog state
    bool show_deploy_dialog_ = false;
    std::string pending_deploy_model_;
    int deploy_port_ = 8082;
    int deploy_gpu_layers_ = 35;
    int deploy_context_size_ = 4096;
    std::string deploy_error_;
    bool deploying_ = false;

    // Delete confirmation dialog
    bool show_delete_dialog_ = false;
    std::string pending_delete_model_;
    std::string delete_error_;
};

} // namespace cyxwiz::servernode::gui
