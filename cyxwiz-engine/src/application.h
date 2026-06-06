#pragma once

#include <memory>
#include <string>

struct GLFWwindow;
struct ImGuiIO;
struct ImFont;

namespace cyxwiz {
    class PythonSetupWizard;
    class StartPage;
}

namespace gui {
    class MainWindow;
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
    [[noreturn]] void Shutdown();

private:
    bool Initialize();
    void ProcessCommandLine(int argc, char** argv);
    void OpenStartupProjectIfRequested();
    void OpenStartupGraphIfRequested();
    void UpdateWindowTitle();
    void ScanForPython();  // Scan for Python on startup (no initialization)
    void MainLoop();
    void HandleInput();
    void Update(float delta_time);
    void Render();
    void LoadFonts(ImGuiIO& io);

    GLFWwindow* window_;
    std::unique_ptr<gui::MainWindow> main_window_;
    std::unique_ptr<network::GRPCClient> grpc_client_;
    std::unique_ptr<network::JobManager> job_manager_;

    bool running_;
    double last_frame_time_;
    std::string imgui_ini_path_;  // Store ini file path

    // Python setup wizard (shown on first launch if no Python configured)
    std::unique_ptr<cyxwiz::PythonSetupWizard> python_wizard_;
    bool python_configured_ = false;

    // Python detection (scanned on startup, not initialized)
    struct {
        bool scanned = false;
        bool found = false;
        std::string version;
        int major = 0;
        int minor = 0;
        std::string path;
        std::string warning;  // Version warning message
    } python_scan_;

    // Start page (shown after Python wizard if no project specified)
    std::unique_ptr<cyxwiz::StartPage> start_page_;
    bool project_selected_ = false;

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

    // Startup project path (from command line)
    std::string startup_project_path_;
    std::string startup_graph_path_;
};
