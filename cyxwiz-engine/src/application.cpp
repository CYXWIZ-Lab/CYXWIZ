#include "application.h"
#include "gui/main_window.h"
#include "gui/console.h"
#include "gui/console_sink.h"
#include "gui/theme.h"
#include "scripting/python_engine.h"
#include "network/grpc_client.h"
#include "network/job_manager.h"
#include "core/async_task_manager.h"

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#ifdef _WIN32
#define GLFW_EXPOSE_NATIVE_WIN32
#include <GLFW/glfw3native.h>
#include <dwmapi.h>
#pragma comment(lib, "dwmapi.lib")
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

CyxWizApp::CyxWizApp(int argc, char** argv)
    : window_(nullptr), running_(true), last_frame_time_(0.0) {

    ProcessCommandLine(argc, argv);

    if (!Initialize()) {
        throw std::runtime_error("Failed to initialize application");
    }
}

CyxWizApp::~CyxWizApp() {
    Shutdown();
}

void CyxWizApp::ProcessCommandLine(int argc, char** argv) {
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        spdlog::debug("Command line arg: {}", arg);
        // TODO: Process command line arguments
    }
}

bool CyxWizApp::Initialize() {
    // Setup GLFW
    glfwSetErrorCallback(glfw_error_callback);
    if (!glfwInit()) {
        spdlog::error("Failed to initialize GLFW");
        return false;
    }

    // GL 3.3 + GLSL 330
    const char* glsl_version = "#version 330";
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

    // Window hints for resizable window
    glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);
    glfwWindowHint(GLFW_MAXIMIZED, GLFW_TRUE);  // Start maximized
    glfwWindowHint(GLFW_DECORATED, GLFW_TRUE);

    // Create window
    window_ = glfwCreateWindow(1920, 1080, "CyxWiz Engine", nullptr, nullptr);
    if (window_ == nullptr) {
        spdlog::error("Failed to create GLFW window");
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

    // Initialize components
    main_window_ = std::make_unique<gui::MainWindow>();
    python_engine_ = std::make_unique<scripting::PythonEngine>();
    grpc_client_ = std::make_unique<network::GRPCClient>();
    job_manager_ = std::make_unique<network::JobManager>(grpc_client_.get());

    // Connect network components to main window
    main_window_->SetNetworkComponents(grpc_client_.get(), job_manager_.get());

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

        // Register spdlog sink for future logs
        auto console_sink = std::make_shared<gui::ConsoleSinkMt>(console);
        auto logger = spdlog::default_logger();
        logger->sinks().push_back(console_sink);

        // Test log to verify spdlog integration
        spdlog::info("✓ Console logging enabled");
        console->AddSuccess("✓ spdlog integration working");
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

void CyxWizApp::HandleInput() {
    glfwPollEvents();
}

void CyxWizApp::Update(float delta_time) {
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

    // Cleanup components
    job_manager_.reset();
    grpc_client_.reset();
    python_engine_.reset();
    main_window_.reset();

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
}

void CyxWizApp::LoadFonts(ImGuiIO& io) {
    // Font configuration for crisp rendering
    ImFontConfig font_config;
    font_config.OversampleH = 2;
    font_config.OversampleV = 1;
    font_config.PixelSnapH = true;

    // Try multiple font paths (running from different directories)
    std::vector<std::string> font_paths = {
        "resources/fonts/",
        "cyxwiz-engine/resources/fonts/",
        "../resources/fonts/"
    };

    std::string font_base_path;
    for (const auto& path : font_paths) {
        if (std::filesystem::exists(path + "Inter-Regular.ttf")) {
            font_base_path = path;
            break;
        }
    }

    if (font_base_path.empty()) {
        spdlog::warn("Custom fonts not found, using default ImGui font");
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

    // Icon font glyph ranges (FontAwesome 6)
    static const ImWchar icon_ranges[] = { 0xe000, 0xf8ff, 0 };

    // Icon font config (for merging)
    ImFontConfig icon_config;
    icon_config.MergeMode = true;
    icon_config.PixelSnapH = true;
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
