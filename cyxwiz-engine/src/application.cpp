#include "application.h"
#include "gui/main_window.h"
#include "gui/console.h"
#include "gui/console_sink.h"
#include "gui/theme.h"
#include "gui/dialogs/python_setup_wizard.h"
#include "gui/dialogs/start_page.h"
#include "auth/auth_client.h"
#include "network/grpc_client.h"
#include "network/job_manager.h"
#include "core/async_task_manager.h"
#include "core/data_registry.h"
#include "core/project_manager.h"
#include "core/training_manager.h"
#include "core/engine_config.h"
#include "core/python_detector.h"
#include "core/texture_manager.h"

#include <cstdlib>  // for _exit()
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#ifdef _WIN32
#define GLFW_EXPOSE_NATIVE_WIN32
#include <GLFW/glfw3native.h>
#include <dwmapi.h>
#pragma comment(lib, "dwmapi.lib")
#elif defined(__APPLE__)
#include <mach-o/dyld.h>
#include <libgen.h>
#elif defined(__linux__)
#include <unistd.h>
#include <limits.h>
#include <libgen.h>
#endif
#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
#include <implot.h>
#include <imnodes.h>
#include <cyxwiz/device.h>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#include <filesystem>
#include <optional>
#include <algorithm>
#include <cctype>
#include <vector>

static void glfw_error_callback(int error, const char* description) {
    spdlog::error("GLFW Error {}: {}", error, description);
}

#ifdef _WIN32
// Enable dark mode for Windows title bar (Windows 10 1809+ / Windows 11)
static void enable_dark_title_bar(GLFWwindow* window) {
    HWND hwnd = glfwGetWin32Window(window);
    if (!hwnd) return;

    // DWMWA_USE_IMMERSIVE_DARK_MODE = 20 (Windows 10 20H1+)
    // For older Windows 10 builds, use undocumented value 19
    BOOL dark_mode = TRUE;

    // Try the official attribute first (Windows 10 20H1+)
    HRESULT hr = DwmSetWindowAttribute(hwnd, 20, &dark_mode, sizeof(dark_mode));

    if (FAILED(hr)) {
        // Fall back to undocumented attribute for older Windows 10 builds
        hr = DwmSetWindowAttribute(hwnd, 19, &dark_mode, sizeof(dark_mode));
    }

    if (SUCCEEDED(hr)) {
        spdlog::info("Dark title bar enabled");
    } else {
        spdlog::debug("Dark title bar not available on this Windows version");
    }
}
#endif

// Load window icon from resources
static bool load_window_icon(GLFWwindow* window) {
    // Try both possible locations
    std::filesystem::path icon_path = "cyxwiz-engine/resources/cyxwiz.png";

    if (!std::filesystem::exists(icon_path)) {
        icon_path = "resources/cyxwiz.png";
        if (!std::filesystem::exists(icon_path)) {
            spdlog::warn("Window icon not found at either location");
            return false;
        }
    }

    int width, height, channels;
    unsigned char* pixels = stbi_load(icon_path.string().c_str(), &width, &height, &channels, 4);

    if (!pixels) {
        spdlog::error("Failed to load window icon: {}", stbi_failure_reason());
        return false;
    }

    GLFWimage image;
    image.width = width;
    image.height = height;
    image.pixels = pixels;

    glfwSetWindowIcon(window, 1, &image);
    stbi_image_free(pixels);

    spdlog::info("Window icon loaded successfully ({}x{})", width, height);
    return true;
}

namespace {

std::string ToLower(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return value;
}

bool HasCyxwizExtension(const std::filesystem::path& path) {
    return ToLower(path.extension().string()) == ".cyxwiz";
}

std::optional<std::filesystem::path> FindProjectFileInDir(const std::filesystem::path& dir) {
    if (!std::filesystem::exists(dir) || !std::filesystem::is_directory(dir)) {
        return std::nullopt;
    }

    std::filesystem::path named = dir / (dir.filename().string() + ".cyxwiz");
    if (std::filesystem::exists(named)) {
        return std::filesystem::absolute(named);
    }

    std::filesystem::path first_match;
    try {
        for (const auto& entry : std::filesystem::directory_iterator(dir)) {
            if (!entry.is_regular_file()) {
                continue;
            }
            if (HasCyxwizExtension(entry.path())) {
                if (!first_match.empty()) {
                    spdlog::warn("Multiple .cyxwiz files found in {}, using first: {}",
                                 dir.string(), first_match.string());
                    return std::filesystem::absolute(first_match);
                }
                first_match = entry.path();
            }
        }
    } catch (const std::exception& e) {
        spdlog::warn("Failed to scan project directory {}: {}", dir.string(), e.what());
        return std::nullopt;
    }

    if (!first_match.empty()) {
        return std::filesystem::absolute(first_match);
    }

    return std::nullopt;
}

std::optional<std::filesystem::path> ResolveProjectArg(const std::string& arg) {
    if (arg.empty()) {
        return std::nullopt;
    }

    std::filesystem::path raw(arg);
    std::vector<std::filesystem::path> attempts;

    if (raw.is_absolute()) {
        attempts.push_back(raw);
    } else {
        if (const char* launch_cwd = std::getenv("CYXWIZ_LAUNCH_CWD")) {
            attempts.push_back(std::filesystem::path(launch_cwd) / raw);
        }
        attempts.push_back(std::filesystem::current_path() / raw);
    }

    for (const auto& attempt : attempts) {
        if (!std::filesystem::exists(attempt)) {
            continue;
        }
        if (std::filesystem::is_directory(attempt)) {
            if (auto found = FindProjectFileInDir(attempt)) {
                return found;
            }
            continue;
        }
        if (HasCyxwizExtension(attempt)) {
            return std::filesystem::absolute(attempt);
        }
    }

    return std::nullopt;
}

} // namespace

CyxWizApp::CyxWizApp(int argc, char** argv)
    : window_(nullptr), running_(true), last_frame_time_(0.0) {

    ProcessCommandLine(argc, argv);

    if (!Initialize()) {
        throw std::runtime_error("Failed to initialize application");
    }
}

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable: 4722)
#endif
CyxWizApp::~CyxWizApp() {
    // Shutdown intentionally ends the process with _exit(0) after
    // explicit resource cleanup to avoid unsafe singleton destruction.
    Shutdown();
}
#ifdef _MSC_VER
#pragma warning(pop)
#endif

void CyxWizApp::ProcessCommandLine(int argc, char** argv) {
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        spdlog::debug("Command line arg: {}", arg);
        if (startup_project_path_.empty()) {
            if (auto resolved = ResolveProjectArg(arg)) {
                startup_project_path_ = resolved->string();
                spdlog::info("Startup project detected: {}", startup_project_path_);
            }
        }
    }
}

void CyxWizApp::OpenStartupProjectIfRequested() {
    if (startup_project_path_.empty()) {
        return;
    }

    auto& pm = cyxwiz::ProjectManager::Instance();
    if (pm.OpenProject(startup_project_path_)) {
        spdlog::info("Opened project from command line: {}", startup_project_path_);
        UpdateWindowTitle();  // Update window title with project name
    } else {
        spdlog::error("Failed to open project from command line: {}", startup_project_path_);
    }
}

void CyxWizApp::UpdateWindowTitle() {
    if (!window_) {
        return;  // Window not created yet
    }

    auto& pm = cyxwiz::ProjectManager::Instance();
    std::string title = "CyxWiz Engine";

    if (pm.HasActiveProject()) {
        // Get project name from the project file path
        std::filesystem::path project_path(pm.GetProjectFilePath());
        std::string project_name = project_path.stem().string();  // Get filename without extension
        title = "CyxWiz Engine - " + project_name;
    }

    glfwSetWindowTitle(window_, title.c_str());
    spdlog::debug("Window title updated to: {}", title);
}

void CyxWizApp::ScanForPython() {
    spdlog::info("Scanning for Python installation...");

    auto& config = cyxwiz::core::EngineConfig::Instance();

    // First check if system Python is already configured
    if (config.HasSystemPython()) {
        auto python_info = cyxwiz::core::PythonDetector::ValidatePythonInstallation(config.GetSystemPythonPath());
        if (python_info) {
            python_scan_.scanned = true;
            python_scan_.found = true;
            python_scan_.version = python_info->version;
            python_scan_.major = python_info->major;
            python_scan_.minor = python_info->minor;
            python_scan_.path = python_info->executable_path;

            // Check version warnings
            if (python_scan_.major == 3 && python_scan_.minor >= 14) {
                python_scan_.warning = "Python 3.14+ detected. We recommend Python 3.12 for stable packages (built with 3.12 bindings).";
                spdlog::warn("{}", python_scan_.warning);
            } else if (python_scan_.major < 3 || (python_scan_.major == 3 && python_scan_.minor < 12)) {
                python_scan_.warning = "Python version < 3.12 detected. Please install Python 3.12 or higher.";
                spdlog::warn("{}", python_scan_.warning);
            }

            spdlog::info("Python {} detected at: {}", python_scan_.version, python_scan_.path);
            return;
        }
    }

    // No configured Python or validation failed - scan for available Python
    auto best_python = cyxwiz::core::PythonDetector::FindBestPython();
    if (best_python) {
        python_scan_.scanned = true;
        python_scan_.found = true;
        python_scan_.version = best_python->version;
        python_scan_.major = best_python->major;
        python_scan_.minor = best_python->minor;
        python_scan_.path = best_python->executable_path;

        // Check version warnings
        if (python_scan_.major == 3 && python_scan_.minor >= 14) {
            python_scan_.warning = "Python 3.14+ detected. We recommend Python 3.12 for stable packages (built with 3.12 bindings).";
            spdlog::warn("{}", python_scan_.warning);
        } else if (python_scan_.major < 3 || (python_scan_.major == 3 && python_scan_.minor < 12)) {
            python_scan_.warning = "Python version < 3.12 detected. Please install Python 3.12 or higher.";
            spdlog::warn("{}", python_scan_.warning);
        }

        spdlog::info("Python {} detected at: {}", python_scan_.version, python_scan_.path);
    } else {
        python_scan_.scanned = true;
        python_scan_.found = false;
        python_scan_.warning = "No Python installation found. Please install Python 3.12 or higher.";
        spdlog::warn("{}", python_scan_.warning);
    }
}

bool CyxWizApp::Initialize() {
    // Setup GLFW
    glfwSetErrorCallback(glfw_error_callback);
    if (!glfwInit()) {
        spdlog::error("Failed to initialize GLFW");
        return false;
    }

    // GL version configuration - try multiple versions for macOS compatibility
    const char* glsl_version = nullptr;

    // Window hints for resizable window (set before context hints)
    glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);
    glfwWindowHint(GLFW_MAXIMIZED, GLFW_TRUE);  // Start maximized
    glfwWindowHint(GLFW_DECORATED, GLFW_TRUE);

#ifdef __APPLE__
    // macOS: Simplify pixel format to avoid OpenCore Patcher issues
    spdlog::info("Attempting to create OpenGL context with minimal requirements");

    // Disable features that might cause pixel format issues
    glfwWindowHint(GLFW_SAMPLES, 0);              // No multisampling
    glfwWindowHint(GLFW_DEPTH_BITS, 24);          // Standard depth buffer
    glfwWindowHint(GLFW_STENCIL_BITS, 8);         // Standard stencil buffer
    glfwWindowHint(GLFW_STEREO, GLFW_FALSE);      // No stereo
    glfwWindowHint(GLFW_SRGB_CAPABLE, GLFW_FALSE); // No sRGB
    glfwWindowHint(GLFW_DOUBLEBUFFER, GLFW_TRUE); // Double buffering

    // Try OpenGL 2.1 (most compatible)
    glsl_version = "#version 120";
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);

    spdlog::info("Attempting OpenGL 2.1 with simplified pixel format");
    window_ = glfwCreateWindow(1280, 720, "CyxWiz Engine", nullptr, nullptr);

    // Try with even more minimal requirements
    if (window_ == nullptr) {
        spdlog::warn("OpenGL 2.1 failed, trying minimal configuration");
        glfwDefaultWindowHints();
        glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);
        glfwWindowHint(GLFW_SAMPLES, 0);
        glfwWindowHint(GLFW_STEREO, GLFW_FALSE);
        glfwWindowHint(GLFW_SRGB_CAPABLE, GLFW_FALSE);
        glsl_version = "#version 120";
        window_ = glfwCreateWindow(800, 600, "CyxWiz Engine", nullptr, nullptr);
    }
#else
    // GL 3.3 + GLSL 330 for Windows/Linux
    glsl_version = "#version 330";
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    // Create window
    window_ = glfwCreateWindow(1920, 1080, "CyxWiz Engine", nullptr, nullptr);
#endif

    if (window_ == nullptr) {
        spdlog::error("Failed to create GLFW window with all attempted OpenGL versions");
        return false;
    }

    // Load window icon
    load_window_icon(window_);

#ifdef _WIN32
    // Enable dark mode for Windows title bar
    enable_dark_title_bar(window_);
#endif

    // Make sure window is visible and focused
    glfwShowWindow(window_);
    glfwFocusWindow(window_);

    glfwMakeContextCurrent(window_);
    glfwSwapInterval(1); // Enable vsync

    // Initialize GLAD - Load OpenGL function pointers
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        spdlog::error("Failed to initialize GLAD");
        return false;
    }
    spdlog::info("OpenGL {}.{} initialized", GLVersion.major, GLVersion.minor);

    // Setup Dear ImGui context
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImPlot::CreateContext();  // Initialize ImPlot for plotting functionality
    ImNodes::CreateContext();  // Initialize ImNodes for visual node editor
    ImGuiIO& io = ImGui::GetIO();

    // Set persistent ini file path (same directory as executable)
    imgui_ini_path_ = "imgui.ini";
    io.IniFilename = imgui_ini_path_.c_str();

    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
    // TODO: ViewportsEnable causes crash on Windows - needs investigation
    // io.ConfigFlags |= ImGuiConfigFlags_ViewportsEnable;

    // Setup Dear ImGui style - Apply CyxWiz professional theme
    gui::GetTheme().ApplyPreset(gui::ThemePreset::CyxWizDark);

    // When viewports are enabled we tweak WindowRounding/WindowBg
    ImGuiStyle& style = ImGui::GetStyle();
    if (io.ConfigFlags & ImGuiConfigFlags_ViewportsEnable) {
        style.WindowRounding = 0.0f;
        style.Colors[ImGuiCol_WindowBg].w = 1.0f;
    }

    // Setup Platform/Renderer backends
    ImGui_ImplGlfw_InitForOpenGL(window_, true);
    ImGui_ImplOpenGL3_Init(glsl_version);

    // Load professional fonts
    LoadFonts(io);

    // Scan for Python on startup (no initialization yet)
    ScanForPython();

    // Check if Python is configured - show wizard if not
    auto& config = cyxwiz::core::EngineConfig::Instance();
    python_configured_ = config.HasSystemPython();

    if (!python_configured_) {
        spdlog::info("No system Python configured - showing setup wizard");
        python_wizard_ = std::make_unique<cyxwiz::PythonSetupWizard>();
        // Main window will be created after wizard completes
        return true;
    }

    // Python is configured - show start page
    spdlog::info("Python configured - showing start page");
    start_page_ = std::make_unique<cyxwiz::StartPage>();

    // If project was specified on command line, we'll still show the start page
    // but it can be skipped by the user
    return true;

    // Both Python and project are ready - create main window
    project_selected_ = true;
    spdlog::info("Python configured and project specified - creating main window");

    // Initialize components
    main_window_ = std::make_unique<gui::MainWindow>();
    OpenStartupProjectIfRequested();
    grpc_client_ = std::make_unique<network::GRPCClient>();
    job_manager_ = std::make_unique<network::JobManager>(grpc_client_.get());

    // Connect network components to main window
    main_window_->SetNetworkComponents(grpc_client_.get(), job_manager_.get());

    // Connect debug logging flags to main window (for View menu toggles)
    main_window_->SetIdleLogPtr(&log_idle_transitions_);

    // Set exit request callback (triggered by File > Exit menu)
    main_window_->SetExitRequestCallback([this]() {
        spdlog::info("Exit requested via menu");
        glfwSetWindowShouldClose(window_, GLFW_TRUE);
    });

    // Register console sink with spdlog to show logs in GUI
    if (main_window_ && main_window_->GetConsole()) {
        auto* console = main_window_->GetConsole();

        // Add welcome message directly to console
        console->AddSuccess("=== CyxWiz Engine Console ===");
        console->AddInfo("Console panel initialized - logs will appear here");

        // Show Python scan results
        if (python_scan_.scanned) {
            if (python_scan_.found) {
                console->AddSuccess("Python " + python_scan_.version + " detected at: " + python_scan_.path);
                if (!python_scan_.warning.empty()) {
                    console->AddWarning(python_scan_.warning);
                }
            } else {
                console->AddError("No Python installation found");
                console->AddWarning(python_scan_.warning);
            }
        }

        // Show initialization message based on project status
        if (!startup_project_path_.empty()) {
            console->AddInfo("Project loaded - Python will be initialized from project's virtual environment");
        } else {
            console->AddInfo("No project loaded - Python not initialized");
            console->AddInfo("Create or open a project to use Python");
        }

        // Register spdlog sink for future logs
        auto console_sink = std::make_shared<gui::ConsoleSinkMt>(console);
        auto logger = spdlog::default_logger();
        logger->sinks().push_back(console_sink);

        // Test log to verify spdlog integration
        spdlog::info("✓ Console logging enabled");
        console->AddSuccess("✓ spdlog integration working");
    }

    // Restore saved auth session at startup
    auto& auth = cyxwiz::auth::AuthClient::Instance();
    if (auth.LoadSavedSession()) {
        spdlog::info("Auth session restored for: {}", auth.GetUserInfo().email);
    }

    spdlog::info("Application initialized successfully");

    // Log device information to GUI console
    if (main_window_ && main_window_->GetConsole()) {
        auto* console = main_window_->GetConsole();

        // Log backend initialization status
        console->AddSuccess("CyxWiz Backend initialized");

        // Get and log available devices
        auto devices = cyxwiz::Device::GetAvailableDevices();
        console->AddInfo("Available compute devices:");

        bool has_gpu = false;
        for (const auto& device : devices) {
            std::string device_type_str;
            switch(device.type) {
                case cyxwiz::DeviceType::CPU: device_type_str = "CPU"; break;
                case cyxwiz::DeviceType::CUDA: device_type_str = "CUDA GPU"; has_gpu = true; break;
                case cyxwiz::DeviceType::OPENCL: device_type_str = "OpenCL GPU"; has_gpu = true; break;
                default: device_type_str = "Unknown"; break;
            }

            std::string log_msg = "  - " + device.name + " (" + device_type_str + ")";
            if (device.memory_total > 0) {
                log_msg += " - " + std::to_string(device.memory_total / (1024*1024)) + " MB";
            }
            console->AddInfo(log_msg);
        }

        if (has_gpu) {
            console->AddSuccess("GPU acceleration enabled!");
        }
    }

    return true;
}

int CyxWizApp::Run() {
    last_frame_time_ = glfwGetTime();

    while (running_) {
        // Check if user is trying to close the window
        if (glfwWindowShouldClose(window_)) {
            if (force_close_) {
                // User confirmed force close
                break;
            }

            // Check if we should prevent close (script running)
            if (ShouldPreventClose()) {
                // Cancel the close and show confirmation dialog
                glfwSetWindowShouldClose(window_, GLFW_FALSE);
                show_close_confirmation_ = true;
            }
            // Check for unsaved files
            else if (HasUnsavedWork()) {
                glfwSetWindowShouldClose(window_, GLFW_FALSE);
                show_unsaved_confirmation_ = true;
            }
            // Check for loaded data in memory
            else if (HasLoadedData()) {
                glfwSetWindowShouldClose(window_, GLFW_FALSE);
                show_data_loaded_confirmation_ = true;
            } else {
                // OK to close
                break;
            }
        }

        double current_time = glfwGetTime();
        float delta_time = static_cast<float>(current_time - last_frame_time_);
        last_frame_time_ = current_time;

        HandleInput();
        Update(delta_time);
        Render();
    }

    return 0;
}

bool CyxWizApp::ShouldPreventClose() {
    // Check if a script is running
    if (main_window_ && main_window_->IsScriptRunning()) {
        return true;
    }
    return false;
}

bool CyxWizApp::HasUnsavedWork() {
    // Check for unsaved files in script editor
    if (main_window_ && main_window_->HasUnsavedFiles()) {
        return true;
    }
    return false;
}

bool CyxWizApp::HasLoadedData() {
    // Check for loaded datasets in memory
    auto& registry = cyxwiz::DataRegistry::Instance();
    return !registry.GetDatasetNames().empty();
}

void CyxWizApp::HandleCloseConfirmation() {
    if (!show_close_confirmation_) return;

    // Center the popup
    ImVec2 center = ImGui::GetMainViewport()->GetCenter();
    ImGui::SetNextWindowPos(center, ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));

    if (ImGui::BeginPopupModal("Script Running###CloseConfirm", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
        ImGui::Text("A Python script is currently running.");
        ImGui::Spacing();
        ImGui::Text("What would you like to do?");
        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();

        if (ImGui::Button("Stop Script & Close", ImVec2(150, 0))) {
            // Stop the script and close
            if (main_window_) {
                main_window_->StopRunningScript();
            }
            show_close_confirmation_ = false;
            running_ = false;
            ImGui::CloseCurrentPopup();
        }

        ImGui::SameLine();

        if (ImGui::Button("Force Close", ImVec2(100, 0))) {
            // Force close without stopping
            show_close_confirmation_ = false;
            force_close_ = true;
            running_ = false;
            ImGui::CloseCurrentPopup();
        }

        ImGui::SameLine();

        if (ImGui::Button("Cancel", ImVec2(80, 0))) {
            show_close_confirmation_ = false;
            ImGui::CloseCurrentPopup();
        }

        ImGui::EndPopup();
    }

    // Open the popup if we need to show it
    if (show_close_confirmation_ && !ImGui::IsPopupOpen("Script Running###CloseConfirm")) {
        ImGui::OpenPopup("Script Running###CloseConfirm");
    }
}

void CyxWizApp::HandleUnsavedConfirmation() {
    if (!show_unsaved_confirmation_) return;

    // Center the popup
    ImVec2 center = ImGui::GetMainViewport()->GetCenter();
    ImGui::SetNextWindowPos(center, ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));

    if (ImGui::BeginPopupModal("Unsaved Changes###UnsavedConfirm", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
        ImGui::Text("You have unsaved changes in the following files:");
        ImGui::Spacing();

        // List unsaved files
        if (main_window_) {
            auto unsaved_files = main_window_->GetUnsavedFileNames();
            for (const auto& filename : unsaved_files) {
                ImGui::BulletText("%s", filename.c_str());
            }
        }

        ImGui::Spacing();
        ImGui::Text("What would you like to do?");
        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();

        if (ImGui::Button("Save All & Close", ImVec2(130, 0))) {
            // Save all files and close
            if (main_window_) {
                main_window_->SaveAllFiles();
            }
            show_unsaved_confirmation_ = false;
            running_ = false;
            ImGui::CloseCurrentPopup();
        }

        ImGui::SameLine();

        if (ImGui::Button("Discard & Close", ImVec2(120, 0))) {
            // Close without saving
            show_unsaved_confirmation_ = false;
            force_close_ = true;
            running_ = false;
            ImGui::CloseCurrentPopup();
        }

        ImGui::SameLine();

        if (ImGui::Button("Cancel", ImVec2(80, 0))) {
            show_unsaved_confirmation_ = false;
            ImGui::CloseCurrentPopup();
        }

        ImGui::EndPopup();
    }

    // Open the popup if we need to show it
    if (show_unsaved_confirmation_ && !ImGui::IsPopupOpen("Unsaved Changes###UnsavedConfirm")) {
        ImGui::OpenPopup("Unsaved Changes###UnsavedConfirm");
    }
}

void CyxWizApp::HandleDataLoadedConfirmation() {
    if (!show_data_loaded_confirmation_) return;

    // Center the popup
    ImVec2 center = ImGui::GetMainViewport()->GetCenter();
    ImGui::SetNextWindowPos(center, ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));

    if (ImGui::BeginPopupModal("Data Loaded###DataLoadedConfirm", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
        auto& registry = cyxwiz::DataRegistry::Instance();
        auto dataset_names = registry.GetDatasetNames();
        auto stats = registry.GetMemoryStats();

        ImGui::Text("You have datasets loaded in memory:");
        ImGui::Spacing();

        // List loaded datasets
        for (const auto& name : dataset_names) {
            ImGui::BulletText("%s", name.c_str());
        }

        ImGui::Spacing();
        ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f), "Total memory usage: %s",
                          stats.FormatBytes(stats.total_allocated).c_str());
        ImGui::Spacing();
        ImGui::Text("Closing will unload all data from memory.");
        ImGui::Text("Make sure you've saved any work that depends on this data.");
        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();

        if (ImGui::Button("Unload & Close", ImVec2(130, 0))) {
            // Unload all datasets and close
            registry.UnloadAll();
            show_data_loaded_confirmation_ = false;
            running_ = false;
            ImGui::CloseCurrentPopup();
        }

        ImGui::SameLine();

        if (ImGui::Button("Cancel", ImVec2(80, 0))) {
            show_data_loaded_confirmation_ = false;
            ImGui::CloseCurrentPopup();
        }

        ImGui::EndPopup();
    }

    // Open the popup if we need to show it
    if (show_data_loaded_confirmation_ && !ImGui::IsPopupOpen("Data Loaded###DataLoadedConfirm")) {
        ImGui::OpenPopup("Data Loaded###DataLoadedConfirm");
    }
}

void CyxWizApp::HandleInput() {
    double current_time = glfwGetTime();

    // Check for ACTUAL user activity (not just "ImGui wants input")
    ImGuiIO& io = ImGui::GetIO();

    // Check for real mouse movement (not just hovering)
    bool mouse_moved = io.MouseDelta.x != 0.0f || io.MouseDelta.y != 0.0f;
    bool mouse_clicked = io.MouseClicked[0] || io.MouseClicked[1] || io.MouseClicked[2];
    bool mouse_scrolled = io.MouseWheel != 0.0f || io.MouseWheelH != 0.0f;

    // Check for any key/text input (ImGui 1.91+ compatible)
    bool key_pressed = !io.InputQueueCharacters.empty() ||
                       io.KeyCtrl || io.KeyShift || io.KeyAlt || io.KeySuper;

    bool has_activity = mouse_moved || mouse_clicked || mouse_scrolled || key_pressed;

    // Check if training is active (need full frame rate)
    bool training_active = cyxwiz::TrainingManager::Instance().IsTrainingActive();

    if (has_activity || training_active) {
        last_activity_time_ = current_time;
        is_idle_ = false;
    } else if (current_time - last_activity_time_ > IDLE_TIMEOUT) {
        is_idle_ = true;
    }

    // Track state transitions for debugging
    static bool was_idle = false;
    static int idle_frame_count = 0;
    static int active_frame_count = 0;

    if (is_idle_ && !training_active) {
        // Use wait with timeout for reduced CPU/GPU usage when idle
        glfwWaitEventsTimeout(IDLE_FRAME_TIME);

        idle_frame_count++;
        if (!was_idle) {
            if (log_idle_transitions_) {
                spdlog::info("Entering IDLE mode (reduced GPU usage)");
            }
            was_idle = true;
            active_frame_count = 0;
        }
    } else {
        glfwPollEvents();

        active_frame_count++;
        if (was_idle) {
            if (log_idle_transitions_) {
                spdlog::info("Exiting IDLE mode (full frame rate)");
            }
            was_idle = false;
            idle_frame_count = 0;
        }
    }
}

void CyxWizApp::Update(float delta_time) {
    (void)delta_time;

    // Update components
    if (job_manager_) {
        job_manager_->Update();
    }

    // Process async task completion callbacks
    cyxwiz::AsyncTaskManager::Instance().ProcessCompletedCallbacks();
}

void CyxWizApp::Render() {
    // Start ImGui frame
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    // Render Python setup wizard if active (shown on first launch)
    if (python_wizard_) {
        bool wizard_still_active = python_wizard_->Render();

        if (!wizard_still_active) {
            // Wizard completed or cancelled
            auto result = python_wizard_->GetResult();

            if (result == cyxwiz::PythonSetupWizard::Result::Completed) {
                spdlog::info("Python setup wizard completed successfully");
                python_configured_ = true;
                python_wizard_.reset();

                // Check if project needs to be selected
                // Show start page after Python wizard completes
                spdlog::info("Python setup complete - showing start page");
                start_page_ = std::make_unique<cyxwiz::StartPage>();

            } else if (result == cyxwiz::PythonSetupWizard::Result::Cancelled) {
                spdlog::info("Python setup wizard cancelled - exiting application");
                glfwSetWindowShouldClose(window_, GLFW_TRUE);
            }
        }
    }

    // Render start page if active (shown after Python wizard)
    if (start_page_) {
        bool page_still_active = start_page_->Render();

        if (!page_still_active) {
            // Start page completed or user wants to exit
            auto result = start_page_->GetResult();

            if (result == cyxwiz::StartPage::Result::ProjectSelected) {
                spdlog::info("Project selected from start page");
                startup_project_path_ = start_page_->GetSelectedProjectPath();
                project_selected_ = true;
                start_page_.reset();

            } else if (result == cyxwiz::StartPage::Result::ContinueWithout) {
                spdlog::info("User chose to continue without project");
                startup_project_path_ = "";  // No project
                project_selected_ = true;    // But allow main window to open
                start_page_.reset();

            } else if (result == cyxwiz::StartPage::Result::Exit) {
                spdlog::info("User exited start page - closing application");
                glfwSetWindowShouldClose(window_, GLFW_TRUE);
                start_page_.reset();
            }
        }
    }

    // Create main window once Python is configured and project is selected
    if (python_configured_ && project_selected_ && !main_window_) {
        spdlog::info("Creating main window with project: {}", startup_project_path_);

        main_window_ = std::make_unique<gui::MainWindow>();
        OpenStartupProjectIfRequested();  // This will call UpdateWindowTitle() if project opens
        UpdateWindowTitle();  // Update window title regardless (shows project name or just "CyxWiz Engine")
        grpc_client_ = std::make_unique<network::GRPCClient>();
        job_manager_ = std::make_unique<network::JobManager>(grpc_client_.get());

        // Connect network components to main window
        main_window_->SetNetworkComponents(grpc_client_.get(), job_manager_.get());

        // Connect debug logging flags to main window (for View menu toggles)
        main_window_->SetIdleLogPtr(&log_idle_transitions_);

        // Set exit request callback (triggered by File > Exit menu)
        main_window_->SetExitRequestCallback([this]() {
            spdlog::info("Exit requested via menu");
            glfwSetWindowShouldClose(window_, GLFW_TRUE);
        });

        // Register console sink with spdlog
        if (main_window_ && main_window_->GetConsole()) {
            auto* console = main_window_->GetConsole();

            console->AddSuccess("=== CyxWiz Engine Console ===");
            console->AddInfo("Console panel initialized - logs will appear here");

            auto console_sink = std::make_shared<gui::ConsoleSinkMt>(console);
            auto logger = spdlog::default_logger();
            logger->sinks().push_back(console_sink);

            spdlog::info("✓ Console logging enabled");
            console->AddSuccess("✓ spdlog integration working");

            // Show Python scan results
            if (python_scan_.scanned) {
                if (python_scan_.found) {
                    console->AddSuccess("Python " + python_scan_.version + " detected at: " + python_scan_.path);
                    if (!python_scan_.warning.empty()) {
                        console->AddWarning(python_scan_.warning);
                    }
                } else {
                    console->AddError("No Python installation found");
                    console->AddWarning(python_scan_.warning);
                }
            }

            // Show initialization message based on project status
            if (!startup_project_path_.empty()) {
                console->AddInfo("Project loaded - Python will be initialized from project's virtual environment");
            } else {
                console->AddInfo("No project loaded - Python not initialized");
                console->AddInfo("Create or open a project to use Python");
            }
        }

        // Restore saved auth session
        auto& auth = cyxwiz::auth::AuthClient::Instance();
        if (auth.LoadSavedSession()) {
            spdlog::info("Auth session restored for: {}", auth.GetUserInfo().email);
        }

        // Log device information
        if (main_window_ && main_window_->GetConsole()) {
            auto* console = main_window_->GetConsole();
            console->AddSuccess("CyxWiz Backend initialized");

            auto devices = cyxwiz::Device::GetAvailableDevices();
            console->AddInfo("Available compute devices:");

            for (const auto& device : devices) {
                std::string device_type_str;
                switch(device.type) {
                    case cyxwiz::DeviceType::CPU: device_type_str = "CPU"; break;
                    case cyxwiz::DeviceType::CUDA: device_type_str = "CUDA GPU"; break;
                    case cyxwiz::DeviceType::OPENCL: device_type_str = "OpenCL GPU"; break;
                    default: device_type_str = "Unknown"; break;
                }

                std::string log_msg = "  - " + device.name + " (" + device_type_str + ")";
                console->AddInfo(log_msg);
            }
        }
    }

    // Render main window (with docking)
    if (main_window_) {
        try {
            main_window_->Render();
        } catch (const std::exception& e) {
            spdlog::error("Exception in main_window_->Render(): {}", e.what());
        } catch (...) {
            spdlog::error("Unknown exception in main_window_->Render()");
        }
    }

    // Handle close confirmation dialogs
    HandleCloseConfirmation();
    HandleUnsavedConfirmation();
    HandleDataLoadedConfirmation();

    // Rendering
    ImGui::Render();
    int display_w, display_h;
    glfwGetFramebufferSize(window_, &display_w, &display_h);
    glViewport(0, 0, display_w, display_h);
    glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    // Safely render ImGui draw data with null check
    ImDrawData* draw_data = ImGui::GetDrawData();
    if (draw_data != nullptr) {
        ImGui_ImplOpenGL3_RenderDrawData(draw_data);
    } else {
        spdlog::error("ImGui::GetDrawData() returned nullptr - skipping render");
    }

    // Update and Render additional Platform Windows
    ImGuiIO& io = ImGui::GetIO();
    if (io.ConfigFlags & ImGuiConfigFlags_ViewportsEnable) {
        GLFWwindow* backup_current_context = glfwGetCurrentContext();
        ImGui::UpdatePlatformWindows();
        ImGui::RenderPlatformWindowsDefault();
        glfwMakeContextCurrent(backup_current_context);
    }

    glfwSwapBuffers(window_);
}

void CyxWizApp::Shutdown() {
    spdlog::info("Shutting down application...");

    // Remove console sink from spdlog BEFORE destroying main_window
    // The console sink points to the Console panel which will be destroyed
    {
        auto logger = spdlog::default_logger();
        auto& sinks = logger->sinks();
        // Keep only the first sink (stdout) - remove any additional sinks like ConsoleSink
        if (sinks.size() > 1) {
            sinks.resize(1);
        }
    }

    // Cleanup components
    job_manager_.reset();
    grpc_client_.reset();
    main_window_.reset();

    // Cleanup OpenGL resources BEFORE ImGui shutdown
    // TextureManager uses OpenGL calls that require valid ImGui/GL state
    cyxwiz::TextureManager::Instance().DeleteAllTextures();

    // Cleanup ImGui
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImNodes::DestroyContext();  // Cleanup ImNodes context
    ImPlot::DestroyContext();  // Cleanup ImPlot context
    ImGui::DestroyContext();

    // Cleanup GLFW
    if (window_) {
        glfwDestroyWindow(window_);
    }
    glfwTerminate();

    spdlog::info("Application shut down complete");

    // Use _exit() to skip static destruction - many singletons have destructors
    // that try to log or use resources that are already destroyed.
    // This is safe because all important cleanup is already done above.
    _exit(0);
}

void CyxWizApp::LoadFonts(ImGuiIO& io) {
    // Font configuration for crisp rendering (high quality)
    ImFontConfig font_config;
    font_config.OversampleH = 3;  // Higher oversampling for sharper text/icons
    font_config.OversampleV = 2;  // Vertical oversampling for better quality
    font_config.PixelSnapH = true;

    // Try multiple font paths (running from different directories)
    std::vector<std::string> font_paths = {
        "resources/fonts/",
        "cyxwiz-engine/resources/fonts/",
        "../resources/fonts/",
        "../Resources/fonts/"  // macOS app bundle
    };

#ifdef __APPLE__
    // On macOS, also check paths relative to the executable
    char exec_path[PATH_MAX];
    uint32_t size = sizeof(exec_path);
    if (_NSGetExecutablePath(exec_path, &size) == 0) {
        std::string exec_dir = dirname(exec_path);
        font_paths.insert(font_paths.begin(), exec_dir + "/resources/fonts/");
        font_paths.insert(font_paths.begin(), exec_dir + "/../Resources/fonts/");  // App bundle
        font_paths.insert(font_paths.begin(), exec_dir + "/../resources/fonts/");
        spdlog::debug("macOS executable dir: {}", exec_dir);
    }
#elif defined(__linux__)
    // On Linux, check paths relative to the executable using /proc/self/exe
    char exec_path[PATH_MAX];
    ssize_t len = readlink("/proc/self/exe", exec_path, sizeof(exec_path) - 1);
    if (len != -1) {
        exec_path[len] = ' ';
        char* exec_path_copy = strdup(exec_path);
        std::string exec_dir = dirname(exec_path_copy);
        free(exec_path_copy);
        font_paths.insert(font_paths.begin(), exec_dir + "/resources/fonts/");
        font_paths.insert(font_paths.begin(), exec_dir + "/../resources/fonts/");
        font_paths.insert(font_paths.begin(), exec_dir + "/../../../cyxwiz-engine/resources/fonts/");  // From build/bin/Release/
        spdlog::debug("Linux executable dir: {}", exec_dir);
    }
#endif

    std::string font_base_path;
    for (const auto& path : font_paths) {
        std::string test_path = path + "Inter-Regular.ttf";
        spdlog::debug("Checking font path: {}", test_path);
        if (std::filesystem::exists(test_path)) {
            font_base_path = path;
            spdlog::info("Found fonts at: {}", path);
            break;
        }
    }

    if (font_base_path.empty()) {
        spdlog::warn("Custom fonts not found in any of the search paths, using default ImGui font");
        spdlog::warn("Current working directory: {}", std::filesystem::current_path().string());
        io.Fonts->AddFontDefault();
        return;
    }

    spdlog::info("Loading fonts from: {}", font_base_path);

    // Define font sizes (scaled for high DPI)
    const float base_font_size = 15.0f;
    const float mono_font_size = 14.0f;

    // Load Inter font family (UI font)
    std::string inter_regular = font_base_path + "Inter-Regular.ttf";
    std::string inter_medium = font_base_path + "Inter-Medium.ttf";
    std::string inter_bold = font_base_path + "Inter-Bold.ttf";

    // Load JetBrains Mono (code font)
    std::string mono_regular = font_base_path + "JetBrainsMono-Regular.ttf";
    std::string mono_bold = font_base_path + "JetBrainsMono-Bold.ttf";

    // FontAwesome icon font
    std::string fa_solid = font_base_path + "fa-solid-900.ttf";

    // Tabler Icons font (POC for node icon themes)
    std::string tabler_icons = font_base_path + "tabler-icons.ttf";

    // Additional icon packs
    std::string remix_icons = font_base_path + "remixicon.ttf";
    std::string lucide_icons = font_base_path + "lucide.ttf";
    std::string iconoir_icons = font_base_path + "iconoir.ttf";
    std::string phosphor_icons = font_base_path + "phosphor.ttf";

    // Icon font glyph ranges (FontAwesome 6)
    static const ImWchar icon_ranges[] = { 0xe000, 0xf8ff, 0 };

    // Tabler Icons glyph ranges (0xea00 - 0xf9ff)
    static const ImWchar tabler_icon_ranges[] = { 0xea00, 0xf9ff, 0 };

    // Additional icon pack glyph ranges (all use Private Use Area)
    static const ImWchar remix_icon_ranges[] = { 0xea01, 0xf2ff, 0 };
    static const ImWchar lucide_icon_ranges[] = { 0xe900, 0xefff, 0 };
    static const ImWchar iconoir_icon_ranges[] = { 0xe900, 0xefff, 0 };
    static const ImWchar phosphor_icon_ranges[] = { 0xe000, 0xf8ff, 0 };

    // Icon font config (for merging) - high quality
    ImFontConfig icon_config;
    icon_config.MergeMode = true;
    icon_config.PixelSnapH = true;
    icon_config.OversampleH = 3;  // Sharp icons
    icon_config.OversampleV = 2;
    icon_config.GlyphMinAdvanceX = base_font_size;  // Make icons monospaced

    // Load regular font (this becomes the default)
    if (std::filesystem::exists(inter_regular)) {
        font_regular_ = io.Fonts->AddFontFromFileTTF(inter_regular.c_str(), base_font_size, &font_config);
        if (font_regular_) {
            spdlog::info("Loaded Inter-Regular ({}px)", base_font_size);

            // Merge FontAwesome icons into regular font
            if (std::filesystem::exists(fa_solid)) {
                io.Fonts->AddFontFromFileTTF(fa_solid.c_str(), base_font_size - 1.0f, &icon_config, icon_ranges);
                spdlog::info("Merged FontAwesome icons into regular font");
            }

            // Merge Tabler icons into regular font (POC for node icon themes)
            if (std::filesystem::exists(tabler_icons)) {
                io.Fonts->AddFontFromFileTTF(tabler_icons.c_str(), base_font_size - 1.0f, &icon_config, tabler_icon_ranges);
                spdlog::info("Merged Tabler icons into regular font");
            }

            // Merge Remix icons into regular font
            if (std::filesystem::exists(remix_icons)) {
                io.Fonts->AddFontFromFileTTF(remix_icons.c_str(), base_font_size - 1.0f, &icon_config, remix_icon_ranges);
                spdlog::info("Merged Remix icons into regular font");
            }

            // Merge Lucide icons into regular font
            if (std::filesystem::exists(lucide_icons)) {
                io.Fonts->AddFontFromFileTTF(lucide_icons.c_str(), base_font_size - 1.0f, &icon_config, lucide_icon_ranges);
                spdlog::info("Merged Lucide icons into regular font");
            }

            // Merge Iconoir icons into regular font
            if (std::filesystem::exists(iconoir_icons)) {
                io.Fonts->AddFontFromFileTTF(iconoir_icons.c_str(), base_font_size - 1.0f, &icon_config, iconoir_icon_ranges);
                spdlog::info("Merged Iconoir icons into regular font");
            }

            // Merge Phosphor icons into regular font
            if (std::filesystem::exists(phosphor_icons)) {
                io.Fonts->AddFontFromFileTTF(phosphor_icons.c_str(), base_font_size - 1.0f, &icon_config, phosphor_icon_ranges);
                spdlog::info("Merged Phosphor icons into regular font");
            }
        }
    }

    // Load medium font
    if (std::filesystem::exists(inter_medium)) {
        font_medium_ = io.Fonts->AddFontFromFileTTF(inter_medium.c_str(), base_font_size, &font_config);
        if (font_medium_) {
            spdlog::info("Loaded Inter-Medium ({}px)", base_font_size);

            // Merge FontAwesome icons
            if (std::filesystem::exists(fa_solid)) {
                io.Fonts->AddFontFromFileTTF(fa_solid.c_str(), base_font_size - 1.0f, &icon_config, icon_ranges);
            }
        }
    }

    // Load bold font
    if (std::filesystem::exists(inter_bold)) {
        font_bold_ = io.Fonts->AddFontFromFileTTF(inter_bold.c_str(), base_font_size, &font_config);
        if (font_bold_) {
            spdlog::info("Loaded Inter-Bold ({}px)", base_font_size);

            // Merge FontAwesome icons
            if (std::filesystem::exists(fa_solid)) {
                io.Fonts->AddFontFromFileTTF(fa_solid.c_str(), base_font_size - 1.0f, &icon_config, icon_ranges);
            }
        }
    }

    // Load monospace font (for code/console)
    if (std::filesystem::exists(mono_regular)) {
        font_mono_ = io.Fonts->AddFontFromFileTTF(mono_regular.c_str(), mono_font_size, &font_config);
        if (font_mono_) {
            spdlog::info("Loaded JetBrainsMono-Regular ({}px)", mono_font_size);

            // Merge FontAwesome icons
            if (std::filesystem::exists(fa_solid)) {
                icon_config.GlyphMinAdvanceX = mono_font_size;
                io.Fonts->AddFontFromFileTTF(fa_solid.c_str(), mono_font_size - 1.0f, &icon_config, icon_ranges);
                icon_config.GlyphMinAdvanceX = base_font_size;  // Reset
            }
        }
    }

    // Load monospace bold font
    if (std::filesystem::exists(mono_bold)) {
        font_mono_bold_ = io.Fonts->AddFontFromFileTTF(mono_bold.c_str(), mono_font_size, &font_config);
        if (font_mono_bold_) {
            spdlog::info("Loaded JetBrainsMono-Bold ({}px)", mono_font_size);

            // Merge FontAwesome icons
            if (std::filesystem::exists(fa_solid)) {
                icon_config.GlyphMinAdvanceX = mono_font_size;
                io.Fonts->AddFontFromFileTTF(fa_solid.c_str(), mono_font_size - 1.0f, &icon_config, icon_ranges);
            }
        }
    }

    // If no fonts were loaded, add default
    if (!font_regular_) {
        spdlog::warn("Failed to load Inter-Regular, using default font");
        io.Fonts->AddFontDefault();
    }

    // Build font atlas
    spdlog::info("Building font atlas...");
    spdlog::default_logger()->flush();  // Force flush before potential crash
    io.Fonts->Build();
    spdlog::info("Font atlas built successfully");
}
