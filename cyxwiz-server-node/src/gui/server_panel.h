// server_panel.h - Base class for server node panels
#pragma once

#include <string>
#include <imgui.h>

namespace cyxwiz::servernode::core {
    class BackendManager;
    class StateManager;
}

namespace cyxwiz::servernode::ipc {
    class DaemonClient;
}

namespace cyxwiz::servernode::gui {

// Safe font access helper - returns default font if index out of range
inline ImFont* GetSafeFont(int index) {
    ImGuiIO& io = ImGui::GetIO();
    if (index >= 0 && index < io.Fonts->Fonts.Size) {
        return io.Fonts->Fonts[index];
    }
    return io.Fonts->Fonts[0];  // Fallback to first font
}

// Font indices
constexpr int FONT_REGULAR = 0;
constexpr int FONT_MONO = 1;
constexpr int FONT_LARGE = 2;

class ServerPanel {
public:
    ServerPanel(const std::string& name, bool visible = true);
    virtual ~ServerPanel() = default;

    virtual void Render() = 0;
    virtual void Update() {}

    const std::string& GetName() const { return name_; }
    bool IsVisible() const { return visible_; }
    void SetVisible(bool visible) { visible_ = visible; }
    void Toggle() { visible_ = !visible_; }
    void Show() { visible_ = true; }
    void Hide() { visible_ = false; }
    bool* GetVisiblePtr() { return &visible_; }

    // Set daemon client for dual-process mode
    void SetDaemonClient(ipc::DaemonClient* client) { daemon_client_ = client; }

protected:
    core::BackendManager& GetBackend();
    core::StateManager* GetState();
    ipc::DaemonClient* GetDaemonClient() { return daemon_client_; }
    bool IsDaemonConnected() const;

    std::string name_;
    bool visible_ = true;
    ipc::DaemonClient* daemon_client_ = nullptr;
};

} // namespace cyxwiz::servernode::gui
