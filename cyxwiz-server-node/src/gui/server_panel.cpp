// server_panel.cpp - Base panel implementation
#include "gui/server_panel.h"
#include "core/backend_manager.h"
#include "core/state_manager.h"
#include "ipc/daemon_client.h"

namespace cyxwiz::servernode::gui {

ServerPanel::ServerPanel(const std::string& name, bool visible)
    : name_(name), visible_(visible) {
}

core::BackendManager& ServerPanel::GetBackend() {
    return core::BackendManager::Instance();
}

core::StateManager* ServerPanel::GetState() {
    return core::BackendManager::Instance().GetStateManager();
}

bool ServerPanel::IsDaemonConnected() const {
    return daemon_client_ && daemon_client_->IsConnected();
}

} // namespace cyxwiz::servernode::gui
