#pragma once

#include <memory>
#include <string>

struct GLFWwindow;

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

    GLFWwindow* window_;
    std::unique_ptr<gui::MainWindow> main_window_;
    std::unique_ptr<scripting::PythonEngine> python_engine_;
    std::unique_ptr<network::GRPCClient> grpc_client_;
    std::unique_ptr<network::JobManager> job_manager_;

    bool running_;
    double last_frame_time_;
    std::string imgui_ini_path_;  // Store ini file path
};
