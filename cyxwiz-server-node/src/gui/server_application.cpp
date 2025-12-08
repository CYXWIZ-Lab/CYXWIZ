// server_application.cpp - GUI application implementation
#include "gui/server_application.h"
#include "gui/server_main_window.h"
#include "gui/theme.h"
#include "gui/icons.h"
#include "gui/IconsFontAwesome6.h"
#include "ipc/daemon_client.h"

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
#include <implot.h>
#include <spdlog/spdlog.h>

#include <filesystem>

#ifdef _WIN32
#define GLFW_EXPOSE_NATIVE_WIN32
#include <GLFW/glfw3native.h>
#include <windows.h>
#include <dwmapi.h>
#pragma comment(lib, "dwmapi.lib")
#endif

namespace cyxwiz::servernode::gui {

// Error callback for GLFW
static void glfw_error_callback(int error, const char* description) {
    spdlog::error("GLFW Error {}: {}", error, description);
}

ServerApplication::ServerApplication(int argc, char** argv,
                                     std::shared_ptr<ipc::DaemonClient> daemon_client)
    : argc_(argc), argv_(argv), daemon_client_(std::move(daemon_client)) {
}

ServerApplication::~ServerApplication() {
    Shutdown();
}

int ServerApplication::Run() {
    if (!Initialize()) {
        spdlog::error("Failed to initialize ServerApplication");
        return 1;
    }

    running_ = true;
    MainLoop();

    return 0;
}

bool ServerApplication::Initialize() {
    spdlog::info("Initializing ServerApplication GUI...");

    if (!InitializeGLFW()) {
        return false;
    }

    if (!InitializeImGui()) {
        glfwDestroyWindow(window_);
        glfwTerminate();
        return false;
    }

    LoadFonts();
    ApplyTheme();

    // Create main window with daemon client
    main_window_ = std::make_unique<ServerMainWindow>(daemon_client_.get());

    spdlog::info("ServerApplication initialized successfully");
    return true;
}

bool ServerApplication::InitializeGLFW() {
    glfwSetErrorCallback(glfw_error_callback);

    if (!glfwInit()) {
        spdlog::error("Failed to initialize GLFW");
        return false;
    }

    // OpenGL 3.3 Core Profile
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

    // Create window
    window_ = glfwCreateWindow(window_width_, window_height_, window_title_.c_str(), nullptr, nullptr);
    if (!window_) {
        spdlog::error("Failed to create GLFW window");
        glfwTerminate();
        return false;
    }

    glfwMakeContextCurrent(window_);
    glfwSwapInterval(1);  // VSync

    // Initialize GLAD
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        spdlog::error("Failed to initialize GLAD");
        return false;
    }

#ifdef _WIN32
    // Enable dark title bar on Windows 10/11
    HWND hwnd = glfwGetWin32Window(window_);
    if (hwnd) {
        BOOL dark = TRUE;
        DwmSetWindowAttribute(hwnd, 20, &dark, sizeof(dark));  // DWMWA_USE_IMMERSIVE_DARK_MODE
    }
#endif

    spdlog::info("GLFW initialized with OpenGL {}", (const char*)glGetString(GL_VERSION));
    return true;
}

bool ServerApplication::InitializeImGui() {
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImPlot::CreateContext();

    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
    // ViewportsEnable can cause issues, disable for now
    // io.ConfigFlags |= ImGuiConfigFlags_ViewportsEnable;

    io.IniFilename = "imgui_server_node.ini";

    // Initialize backends
    if (!ImGui_ImplGlfw_InitForOpenGL(window_, true)) {
        spdlog::error("Failed to initialize ImGui GLFW backend");
        return false;
    }

    if (!ImGui_ImplOpenGL3_Init("#version 330 core")) {
        spdlog::error("Failed to initialize ImGui OpenGL3 backend");
        return false;
    }

    spdlog::info("ImGui initialized");
    return true;
}

void ServerApplication::LoadFonts() {
    ImGuiIO& io = ImGui::GetIO();

    // Find fonts directory - check from project root and build directory
    std::vector<std::string> font_paths = {
        "./resources/fonts/",
        "./cyxwiz-server-node/resources/fonts/",  // From project root
        "./cyxwiz-engine/resources/fonts/",       // Fallback to engine fonts
        "../resources/fonts/",
        "../../resources/fonts/",
        "../../../cyxwiz-server-node/resources/fonts/",  // From build/bin/Release/
        "../../../cyxwiz-engine/resources/fonts/",
        "../../cyxwiz-server-node/resources/fonts/",
        "../cyxwiz-engine/resources/fonts/"
    };

    std::string font_dir;
    for (const auto& path : font_paths) {
        if (std::filesystem::exists(path + "Inter-Regular.ttf")) {
            font_dir = path;
            break;
        }
    }

    if (font_dir.empty()) {
        spdlog::warn("Font directory not found, using default fonts");
        font_regular_ = io.Fonts->AddFontDefault();
        font_mono_ = io.Fonts->AddFontDefault();
        font_large_ = io.Fonts->AddFontDefault();
        return;
    }

    // Font configuration
    ImFontConfig font_config;
    font_config.OversampleH = 2;
    font_config.OversampleV = 2;
    font_config.PixelSnapH = true;

    // Load regular font
    std::string inter_path = font_dir + "Inter-Regular.ttf";
    font_regular_ = io.Fonts->AddFontFromFileTTF(inter_path.c_str(), 15.0f, &font_config);

    // Merge FontAwesome icons
    ImFontConfig icon_config;
    icon_config.MergeMode = true;
    icon_config.PixelSnapH = true;
    icon_config.GlyphMinAdvanceX = 14.0f;
    static const ImWchar icon_ranges[] = { ICON_MIN_FA, ICON_MAX_FA, 0 };

    std::string fa_path = font_dir + "fa-solid-900.ttf";
    if (std::filesystem::exists(fa_path)) {
        io.Fonts->AddFontFromFileTTF(fa_path.c_str(), 14.0f, &icon_config, icon_ranges);
    }

    // Load mono font
    std::string mono_path = font_dir + "JetBrainsMono-Regular.ttf";
    if (std::filesystem::exists(mono_path)) {
        font_mono_ = io.Fonts->AddFontFromFileTTF(mono_path.c_str(), 14.0f, &font_config);
        // Merge icons into mono font too
        if (std::filesystem::exists(fa_path)) {
            io.Fonts->AddFontFromFileTTF(fa_path.c_str(), 14.0f, &icon_config, icon_ranges);
        }
    } else {
        font_mono_ = font_regular_;
    }

    // Load large font
    font_large_ = io.Fonts->AddFontFromFileTTF(inter_path.c_str(), 24.0f, &font_config);
    if (std::filesystem::exists(fa_path)) {
        io.Fonts->AddFontFromFileTTF(fa_path.c_str(), 22.0f, &icon_config, icon_ranges);
    }

    io.Fonts->Build();
    spdlog::info("Fonts loaded from: {}", font_dir);
}

void ServerApplication::ApplyTheme() {
    ::gui::GetTheme().ApplyPreset(::gui::ThemePreset::CyxWizDark);
    spdlog::info("Theme applied: CyxWizDark");
}

void ServerApplication::MainLoop() {
    int last_width = 0, last_height = 0;
    glfwGetWindowSize(window_, &last_width, &last_height);

    while (!glfwWindowShouldClose(window_) && running_) {
        // Calculate delta time
        float current_time = static_cast<float>(glfwGetTime());
        float delta_time = current_time - last_frame_time_;
        last_frame_time_ = current_time;
        (void)delta_time;  // May be used later

        // Check if window is being resized
        int current_width, current_height;
        glfwGetWindowSize(window_, &current_width, &current_height);
        bool is_resizing = (current_width != last_width || current_height != last_height);
        last_width = current_width;
        last_height = current_height;

        // Poll events - always poll during resize for smooth rendering
        if (is_idle_ && !is_resizing) {
            glfwWaitEventsTimeout(0.05);  // 20 FPS when idle (was 10)
        } else {
            glfwPollEvents();
        }

        // Detect idle state - not idle if resizing
        ImGuiIO& io = ImGui::GetIO();
        is_idle_ = !is_resizing &&
                   !io.WantCaptureMouse && !io.WantCaptureKeyboard &&
                   io.MouseDelta.x == 0 && io.MouseDelta.y == 0;

        Render();
    }
}

void ServerApplication::Render() {
    // Start new frame
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    // Render main window
    if (main_window_) {
        main_window_->Render();
    }

    // Render ImGui
    ImGui::Render();

    // Get framebuffer size
    int display_w, display_h;
    glfwGetFramebufferSize(window_, &display_w, &display_h);
    glViewport(0, 0, display_w, display_h);

    // Clear
    ImVec4 clear_color = ImGui::GetStyle().Colors[ImGuiCol_WindowBg];
    glClearColor(clear_color.x, clear_color.y, clear_color.z, clear_color.w);
    glClear(GL_COLOR_BUFFER_BIT);

    // Draw
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

    // Swap buffers
    glfwSwapBuffers(window_);
}

void ServerApplication::Shutdown() {
    spdlog::info("Shutting down ServerApplication...");

    main_window_.reset();

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImPlot::DestroyContext();
    ImGui::DestroyContext();

    if (window_) {
        glfwDestroyWindow(window_);
        window_ = nullptr;
    }
    glfwTerminate();

    spdlog::info("ServerApplication shutdown complete");
}

} // namespace cyxwiz::servernode::gui
