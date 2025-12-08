#pragma once

#include <memory>
#include <string>

struct GLFWwindow;
struct ImGuiIO;
struct ImFont;

namespace gui {
    class MainWindow;
}

namespace scripting {
    class PythonEngine;
}

namespace network {
    class GRPCClient;
    class JobManager;
}

class CyxWizApp {
public:
    CyxWizApp(int argc, char** argv);
    ~CyxWizApp();

    int Run();
    void Shutdown();

private:
    bool Initialize();
    void ProcessCommandLine(int argc, char** argv);
    void MainLoop();
    void HandleInput();
    void Update(float delta_time);
    void Render();
    void LoadFonts(ImGuiIO& io);

    GLFWwindow* window_;
    std::unique_ptr<gui::MainWindow> main_window_;
    std::unique_ptr<scripting::PythonEngine> python_engine_;
    std::unique_ptr<network::GRPCClient> grpc_client_;
    std::unique_ptr<network::JobManager> job_manager_;

    bool running_;
    double last_frame_time_;
    std::string imgui_ini_path_;  // Store ini file path

    // Close confirmation state
    bool show_close_confirmation_ = false;
    bool show_unsaved_confirmation_ = false;
    bool show_data_loaded_confirmation_ = false;
    bool force_close_ = false;

    // Close confirmation handling
    void HandleCloseConfirmation();
    void HandleUnsavedConfirmation();
    void HandleDataLoadedConfirmation();
    bool ShouldPreventClose();
    bool HasUnsavedWork();
    bool HasLoadedData();

    // Fonts
    ImFont* font_regular_ = nullptr;
    ImFont* font_medium_ = nullptr;
    ImFont* font_bold_ = nullptr;
    ImFont* font_mono_ = nullptr;
    ImFont* font_mono_bold_ = nullptr;

    // Idle detection for GPU power saving
    double last_activity_time_ = 0.0;
    bool is_idle_ = false;
    static constexpr double IDLE_TIMEOUT = 1.0;  // Seconds before entering idle mode
    static constexpr double IDLE_FRAME_TIME = 0.1;  // 10 FPS when idle

    // Debug logging flags (can be toggled via console commands)
    bool log_idle_transitions_ = false;  // Log idle mode enter/exit
};
