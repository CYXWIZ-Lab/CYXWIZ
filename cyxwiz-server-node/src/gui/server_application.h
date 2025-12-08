// server_application.h - GUI application main loop for server node
#pragma once

#include <memory>
#include <string>

struct GLFWwindow;
struct ImFont;

namespace cyxwiz::servernode::ipc {
    class DaemonClient;
}

namespace cyxwiz::servernode::gui {

class ServerMainWindow;

class ServerApplication {
public:
    // Constructor with optional daemon client (for dual-process mode)
    ServerApplication(int argc, char** argv,
                      std::shared_ptr<ipc::DaemonClient> daemon_client = nullptr);
    ~ServerApplication();

    // Run the application (blocking)
    int Run();

    // Access window
    GLFWwindow* GetWindow() const { return window_; }

    // Access fonts
    ImFont* GetRegularFont() const { return font_regular_; }
    ImFont* GetMonoFont() const { return font_mono_; }
    ImFont* GetLargeFont() const { return font_large_; }

private:
    bool Initialize();
    void Shutdown();
    void MainLoop();
    void Render();

    // Setup helpers
    bool InitializeGLFW();
    bool InitializeImGui();
    void LoadFonts();
    void ApplyTheme();

    // Window
    GLFWwindow* window_ = nullptr;
    int window_width_ = 1280;
    int window_height_ = 800;
    std::string window_title_ = "CyxWiz Server Node";

    // Main window
    std::unique_ptr<ServerMainWindow> main_window_;

    // Fonts
    ImFont* font_regular_ = nullptr;
    ImFont* font_mono_ = nullptr;
    ImFont* font_large_ = nullptr;

    // State
    bool running_ = false;
    bool is_idle_ = false;
    float last_frame_time_ = 0.0f;

    // Command line
    int argc_;
    char** argv_;

    // Daemon client (for dual-process mode)
    std::shared_ptr<ipc::DaemonClient> daemon_client_;
};

} // namespace cyxwiz::servernode::gui
