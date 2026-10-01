#include "application.h"
#include "gui/appearance_settings.h"
#include "gui/main_window.h"
#include "gui/console.h"
#include "gui/display_density.h"
#include "gui/editor_fonts.h"
#include "gui/ui_fonts.h"
#include "gui/ui_buttons.h"
#include "gui/ui_tokens.h"
#include "gui/panel_memory.h"
#include "gui/ui_widgets.h"
#include "gui/theme.h"
#include "gui/dialogs/python_setup_dialog.h"
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

#include <chrono>
#include <cmath>
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
#ifdef __APPLE__
    (void)window;
    spdlog::debug("macOS uses the application bundle icon for regular windows");
    return true;
#else
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
#endif
}

namespace {

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
        if (auto resolved = cyxwiz::ProjectManager::ResolveProjectFilePath(attempt.string())) {
            return std::filesystem::path(*resolved);
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
                continue;
            }
        }
        // Scripts open in the Script Editor once the workspace is up
        // (cyxwiz-engine train.py, like an editor's command line).
        std::error_code ec;
        const std::filesystem::path script = std::filesystem::absolute(arg, ec);
        const std::string ext = script.extension().string();
        if (!ec && (ext == ".py" || ext == ".cyx") && std::filesystem::is_regular_file(script, ec)) {
            startup_scripts_.push_back(script.string());
            spdlog::info("Startup script: {}", startup_scripts_.back());
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

void CyxWizApp::OpenStartupGraphIfRequested() {
    if (startup_graph_path_.empty() || !main_window_) {
        return;
    }

    if (main_window_->OpenGraphInNodeEditor(startup_graph_path_)) {
        spdlog::info("Opened starter graph from start page: {}", startup_graph_path_);
    } else {
        spdlog::error("Failed to open starter graph from start page: {}", startup_graph_path_);
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
    glfwWindowHint(GLFW_MAXIMIZED, GLFW_FALSE);
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
#ifdef _WIN32
    if (HWND hwnd = glfwGetWin32Window(window_)) {
        ShowWindow(hwnd, SW_RESTORE);
        SetWindowPos(hwnd, HWND_TOP, 100, 100, 1280, 720, SWP_SHOWWINDOW);
        SetForegroundWindow(hwnd);
        spdlog::info("Native Windows window shown: HWND={}", reinterpret_cast<void*>(hwnd));
    } else {
        spdlog::error("GLFW window was created but no native HWND is available");
    }
#endif

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
    // Absolute: Python script runs change the working directory to the
    // project root, and a relative path then wrote imgui.ini there.
    {
        std::error_code ec;
        const auto ini = std::filesystem::absolute("imgui.ini", ec);
        imgui_ini_path_ = ec ? std::string("imgui.ini") : ini.string();
    }
    io.IniFilename = imgui_ini_path_.c_str();
    // Open panels are kept in imgui.ini next to the layout (TOFIX129 0.6).
    gui::InstallPanelMemory();

    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
    // TODO: ViewportsEnable causes crash on Windows - needs investigation
    // io.ConfigFlags |= ImGuiConfigFlags_ViewportsEnable;

    // Engine-wide appearance (theme, code text size, sidebar) from the
    // Engine settings; the interface text size is used by LoadFonts.
    gui::ApplyStartupAppearance();

    // When viewports are enabled we tweak WindowRounding/WindowBg
    ImGuiStyle& style = ImGui::GetStyle();
    if (io.ConfigFlags & ImGuiConfigFlags_ViewportsEnable) {
        style.WindowRounding = 0.0f;
        style.Colors[ImGuiCol_WindowBg].w = 1.0f;
    }

    // Setup Platform/Renderer backends
    ImGui_ImplGlfw_InitForOpenGL(window_, true);
    ImGui_ImplOpenGL3_Init(glsl_version);

    // Match font rasterization to the physical framebuffer while preserving
    // logical font metrics and UI layout.
    font_rasterizer_density_ = DetectFontRasterizerDensity(true);

    // Load professional fonts
    LoadFonts(io);

    GLint max_texture_size = 0;
    glGetIntegerv(GL_MAX_TEXTURE_SIZE, &max_texture_size);
    spdlog::info("Font atlas dimensions: {}x{} (GPU maximum texture size: {})",
                 io.Fonts->TexWidth, io.Fonts->TexHeight, max_texture_size);
    if (io.Fonts->TexWidth > max_texture_size ||
        io.Fonts->TexHeight > max_texture_size) {
        spdlog::error("Font atlas exceeds the GPU texture-size limit");
        return false;
    }
    if (!ImGui_ImplOpenGL3_CreateFontsTexture()) {
        spdlog::error("Failed to upload initial font atlas texture");
        return false;
    }
    spdlog::info("Initial font atlas texture uploaded successfully");

#ifdef CYXWIZ_HAS_PYTHON
    // The Python scan runs on a worker while the start page is up (TOFIX129
    // A2-3); the start page chip shows its state.
    python_setup_ = std::make_unique<cyxwiz::PythonSetupDialog>();
    python_setup_->StartScan();
#else
    // Do not block the core Engine on an interpreter when scripting was not built.
    python_configured_ = true;
    spdlog::info("Python scripting support is disabled in this build; skipping interpreter setup");
#endif
    start_page_ = std::make_unique<cyxwiz::StartPage>();

    // If project was specified on command line, we'll still show the start page
    // but it can be skipped by the user
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

            // Anything still open (running script, unsaved scripts, loaded
            // data) is listed in one dialog; otherwise close at once.
            if (ShouldPreventClose() || HasUnsavedWork() || HasLoadedData()) {
                glfwSetWindowShouldClose(window_, GLFW_FALSE);
                show_close_dialog_ = true;
            } else {
                break;
            }
        }

        double current_time = glfwGetTime();
        float delta_time = static_cast<float>(current_time - last_frame_time_);
        last_frame_time_ = current_time;

        HandleInput();
        Update(delta_time);
        // A minimised window has nothing to show: keep background work and
        // event handling going, skip the frame (TOFIX129 step 0.7).
        if (glfwGetWindowAttrib(window_, GLFW_ICONIFIED)) {
            glfwWaitEventsTimeout(MINIMISED_WAIT_TIME);
            continue;
        }
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

void CyxWizApp::HandleCloseRequest() {
    // One dialog for every way of closing (File > Exit and the window's close
    // box): unsaved scripts, a running script and loaded datasets are listed
    // together, and the choice made here is final (TOFIX129 A2-2).
    if (!show_close_dialog_) return;
    using namespace cyxwiz::ui;
    const Tokens& t = CurrentTokens();
    constexpr const char* kTitle = "Close CyxWiz Engine?###close_engine";
    if (!ImGui::IsPopupOpen(kTitle)) ImGui::OpenPopup(kTitle);

    const ImGuiViewport* viewport = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos(viewport->GetWorkCenter(), ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));
    ImGui::SetNextWindowSizeConstraints(ImVec2(480.0f, 0.0f), ImVec2(640.0f, viewport->WorkSize.y * 0.85f));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, t.rounding_card);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(t.space_xl, t.space_lg));
    const bool open = ImGui::BeginPopupModal(kTitle, nullptr, ImGuiWindowFlags_AlwaysAutoResize);
    ImGui::PopStyleVar(2);
    if (!open) return;

    const std::vector<std::string> unsaved = main_window_ ? main_window_->GetUnsavedFileNames() : std::vector<std::string>{};
    const bool script_running = main_window_ && main_window_->IsScriptRunning();
    auto& registry = cyxwiz::DataRegistry::Instance();
    const std::vector<std::string> datasets = registry.GetDatasetNames();

    ImGui::PushTextWrapPos(ImGui::GetCursorPosX() + 600.0f);
    ImGui::TextUnformatted("Some work is still open. Choose what happens to it.");
    ImGui::Spacing();

    auto section = [&](const char* title, const std::vector<std::string>& lines, const char* note) {
        BeginCard(title);
        ImGui::TextColored(t.text_dim, "%s", title);
        for (const auto& line : lines) ImGui::TextUnformatted(line.c_str());
        if (note && note[0]) ImGui::TextColored(t.text_dim, "%s", note);
        EndCard();
    };
    if (!unsaved.empty()) {
        section(("Unsaved scripts (" + std::to_string(unsaved.size()) + ")").c_str(), unsaved, nullptr);
    }
    if (script_running) {
        section("Running script", {"A Python script is running. Closing stops it."}, nullptr);
    }
    if (!datasets.empty()) {
        const auto stats = registry.GetMemoryStats();
        std::string names;
        for (size_t i = 0; i < datasets.size() && i < 6; ++i) names += (i ? ", " : "") + datasets[i];
        if (datasets.size() > 6) names += ", and " + std::to_string(datasets.size() - 6) + " more";
        section(("Datasets in memory (" + stats.FormatBytes(stats.total_allocated) + ")").c_str(), {names},
                "Closing unloads them. Prepared data stays on disk and reloads next time.");
    }
    if (!close_error_.empty()) StatusText(Status::NotSupported, close_error_.c_str());
    ImGui::PopTextWrapPos();
    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    auto close_now = [&]() {
        if (script_running && main_window_) main_window_->StopRunningScript();
        registry.UnloadAll();
        show_close_dialog_ = false;
        force_close_ = true;
        running_ = false;
        ImGui::CloseCurrentPopup();
    };

    if (!unsaved.empty()) {
        if (PrimaryButton("Save scripts and close")) {
            main_window_->SaveAllFiles();
            if (main_window_->HasUnsavedFiles()) {
                close_error_ = "Some scripts could not be saved; they are still listed above. Details are in the log.";
            } else {
                close_now();
            }
        }
        ImGui::SameLine(0.0f, t.space_md);
        if (DangerButton("Close without saving", true, nullptr, ButtonSize::Regular)) close_now();
    } else {
        if (PrimaryButton("Close")) close_now();
    }
    const float keep_w = ButtonWidth("Keep working", ButtonSize::Regular);
    ImGui::SameLine(ImGui::GetWindowContentRegionMax().x - keep_w);
    if (SecondaryButton("Keep working", true, nullptr, ButtonSize::Regular) || ImGui::IsKeyPressed(ImGuiKey_Escape)) {
        show_close_dialog_ = false;
        close_error_.clear();
        ImGui::CloseCurrentPopup();
    }
    ImGui::EndPopup();
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

    const bool focused = glfwGetWindowAttrib(window_, GLFW_FOCUSED) != 0;

    if (is_idle_ && !training_active) {
        // Use wait with timeout for reduced CPU/GPU usage when idle
        glfwWaitEventsTimeout(IDLE_FRAME_TIME);

        if (!was_idle) {
            if (log_idle_transitions_) {
                spdlog::info("Entering IDLE mode (reduced GPU usage)");
            }
            was_idle = true;
        }
    } else if (!focused) {
        // Another window has focus: 30 frames a second is plenty for a
        // training chart in the background (TOFIX129 step 0.7).
        glfwWaitEventsTimeout(UNFOCUSED_FRAME_TIME);
        was_idle = false;
    } else {
        glfwPollEvents();

        if (was_idle) {
            if (log_idle_transitions_) {
                spdlog::info("Exiting IDLE mode (full frame rate)");
            }
            was_idle = false;
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

void CyxWizApp::RenderPythonWait() {
    const ImGuiViewport* vp = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos(vp->GetCenter(), ImGuiCond_Always, ImVec2(0.5f, 0.5f));
    ImGui::Begin("##python_wait", nullptr,
                 ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoSavedSettings |
                     ImGuiWindowFlags_NoMove);
    cyxwiz::ui::StatusText(cyxwiz::ui::Status::Verifying, "Checking Python before opening the workspace...");
    ImGui::End();
}

void CyxWizApp::Render() {
    RefreshFontRasterizerDensity();
    // Interface text size changed in Preferences > Appearance: new fonts
    // are built between frames.
    if (gui::ConsumeFontRebuildRequest()) {
        spdlog::info("Interface text size changed; rebuilding font atlas");
        RebuildFontAtlas();
    }

    // Start ImGui frame
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    // Python scan and dialog (TOFIX129 A2-3).
    if (python_setup_) {
        python_setup_->Poll();
        if (start_page_) {
            const auto view = python_setup_->View();
            cyxwiz::StartPage::PythonStatus status;
            status.text = view.chip;
            status.level = view.level;
            status.on_click = [this]() { python_setup_->Open(); };
            start_page_->SetPythonStatus(std::move(status));
        }
        python_setup_->Render();
        if (python_setup_->Settled()) python_configured_ = true;
        // A project was chosen before the scan finished: say why the
        // workspace has not opened yet.
        if (!start_page_ && project_selected_ && python_setup_->Scanning()) RenderPythonWait();
    }

    // Render start page if active
    if (start_page_) {
        bool page_still_active = start_page_->Render();

        if (!page_still_active) {
            // Start page completed or user wants to exit
            auto result = start_page_->GetResult();

            if (result == cyxwiz::StartPage::Result::ProjectSelected) {
                spdlog::info("Project selected from start page");
                startup_project_path_ = start_page_->GetSelectedProjectPath();
                startup_graph_path_.clear();
                project_selected_ = true;
                start_page_.reset();

            } else if (result == cyxwiz::StartPage::Result::ExampleGraphSelected) {
                spdlog::info("Starter graph selected from start page");
                startup_project_path_.clear();
                startup_graph_path_ = start_page_->GetSelectedGraphPath();
                project_selected_ = true;
                start_page_.reset();

            } else if (result == cyxwiz::StartPage::Result::ContinueWithout) {
                spdlog::info("User chose to continue without project");
                startup_project_path_ = "";  // No project
                startup_graph_path_.clear();
                project_selected_ = true;    // But allow main window to open
                start_page_.reset();

            }
        }
    }

    // Create main window once Python is configured and project is selected
    if (python_configured_ && project_selected_ && !main_window_) {
        spdlog::info("Creating main window with project: {}", startup_project_path_);

        main_window_ = std::make_unique<gui::MainWindow>();
        OpenStartupProjectIfRequested();  // This will call UpdateWindowTitle() if project opens
        OpenStartupGraphIfRequested();
        for (const auto& script : startup_scripts_) main_window_->OpenScriptFile(script);
        startup_scripts_.clear();
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

        // Runtime startup evidence is already captured by the core spdlog
        // sink. The Commands transcript is reserved for interactive output.
        spdlog::info("Console panel initialized; runtime logs available");

        // Restore saved auth session
        auto& auth = cyxwiz::auth::AuthClient::Instance();
        if (auth.LoadSavedSession()) {
            spdlog::info("Auth session restored for: {}", auth.GetUserInfo().email);
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
    HandleCloseRequest();

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

    // Stop scripts, model downloads and the inference server first; this call
    // was dropped from Shutdown in b4599389 and is restored here.
    if (main_window_) {
        main_window_->PrepareForShutdown();
    }

    // Background work stops before the UI it reports to is destroyed: queued
    // tasks retire, running tasks are asked to stop, stale main-thread
    // delivery is discarded. Bounded, so a task that ignores cancellation
    // cannot hang exit; the process ends with _exit below regardless.
    const auto tasks =
        cyxwiz::AsyncTaskManager::Instance().Shutdown(std::chrono::seconds(5));
    if (!tasks.drained) {
        spdlog::warn("Shutdown continues with {} background task(s) still running",
                     tasks.unfinished_tasks.size());
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

float CyxWizApp::DetectFontRasterizerDensity(bool log_metrics) const {
    int window_width = 0;
    int window_height = 0;
    int framebuffer_width = 0;
    int framebuffer_height = 0;
    if (window_) {
        glfwGetWindowSize(window_, &window_width, &window_height);
        glfwGetFramebufferSize(window_, &framebuffer_width, &framebuffer_height);
    }

    float density = 1.0f;
#if defined(__APPLE__) || defined(__linux__)
    const bool has_valid_dimensions =
        window_width > 0 && window_height > 0 && framebuffer_width > 0 &&
        framebuffer_height > 0;
    density = has_valid_dimensions
                  ? cyxwiz::gui::CalculateFramebufferDensity(
                        window_width, window_height, framebuffer_width,
                        framebuffer_height)
                  : font_rasterizer_density_;
#endif

    if (log_metrics) {
        spdlog::info(
            "Display metrics: logical={}x{} framebuffer={}x{} font_density={:.2f}x",
            window_width, window_height, framebuffer_width, framebuffer_height,
            density);
    }
    return density;
}

void CyxWizApp::RefreshFontRasterizerDensity() {
#if defined(__APPLE__) || defined(__linux__)
    const float detected_density = DetectFontRasterizerDensity(false);
    if (!cyxwiz::gui::HasMaterialFontDensityChange(
            font_rasterizer_density_, detected_density)) {
        return;
    }

    ImGuiIO& io = ImGui::GetIO();
    if (io.Fonts->Locked) {
        spdlog::warn("Deferring font atlas density change while atlas is locked");
        return;
    }

    spdlog::info("Display density changed from {:.2f}x to {:.2f}x; rebuilding font atlas",
                 font_rasterizer_density_, detected_density);
    font_rasterizer_density_ = detected_density;
    RebuildFontAtlas();
#endif
}

void CyxWizApp::RebuildFontAtlas() {
    ImGuiIO& io = ImGui::GetIO();
    if (io.Fonts->Locked) {
        spdlog::warn("Deferring font atlas rebuild while the atlas is locked");
        return;
    }
    ImGui_ImplOpenGL3_DestroyFontsTexture();
    io.FontDefault = nullptr;
    cyxwiz::gui::ClearEditorMonoFonts();
    cyxwiz::ui::ClearFonts();
    font_regular_ = nullptr;
    font_medium_ = nullptr;
    font_bold_ = nullptr;
    font_mono_ = nullptr;
    font_mono_bold_ = nullptr;
    io.Fonts->Clear();
    LoadFonts(io);
    if (!ImGui_ImplOpenGL3_CreateFontsTexture()) {
        spdlog::error("Failed to upload rebuilt font atlas texture");
    }
}

void CyxWizApp::LoadFonts(ImGuiIO& io) {
    // Font configuration for crisp rendering (high quality)
    ImFontConfig font_config;
    const bool needs_manual_oversampling = font_rasterizer_density_ <= 1.0f;
    font_config.OversampleH = needs_manual_oversampling ? 3 : 1;
    font_config.OversampleV = needs_manual_oversampling ? 2 : 1;
    font_config.PixelSnapH = true;
    font_config.RasterizerDensity = font_rasterizer_density_;

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
        exec_path[len] = '\0';
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
    if (!resolved_font_base_path_.empty() &&
        std::filesystem::exists(resolved_font_base_path_ + "Inter-Regular.ttf")) {
        font_base_path = resolved_font_base_path_;
    }
    for (const auto& path : font_paths) {
        if (!font_base_path.empty()) break;
        std::string test_path = path + "Inter-Regular.ttf";
        spdlog::debug("Checking font path: {}", test_path);
        if (std::filesystem::exists(test_path)) {
            std::error_code ec;
            const auto absolute = std::filesystem::absolute(path, ec);
            font_base_path = ec ? path : absolute.string();
            if (!font_base_path.empty() && font_base_path.back() != '/' &&
                font_base_path.back() != '\\') {
                font_base_path += '/';
            }
            resolved_font_base_path_ = font_base_path;
            spdlog::info("Found fonts at: {}", font_base_path);
            break;
        }
    }

    if (font_base_path.empty()) {
        spdlog::warn("Custom fonts not found in any of the search paths, using default ImGui font");
        spdlog::warn("Current working directory: {}", std::filesystem::current_path().string());
        io.Fonts->AddFontDefault(&font_config);
        return;
    }

    spdlog::info("Loading fonts from: {}", font_base_path);

    // Define font sizes (scaled for high DPI)
    // Interface text size (Preferences > Appearance): 13, 15, 17 or 20 px.
    const float base_font_size = static_cast<float>(gui::UiTextPixels());
    const float mono_font_size = 14.0f;

    // Load Inter font family (UI font)
    std::string inter_regular = font_base_path + "Inter-Regular.ttf";
    std::string inter_medium = font_base_path + "Inter-Medium.ttf";
    std::string inter_bold = font_base_path + "Inter-Bold.ttf";

    // Load JetBrains Mono (code font)
    std::string mono_regular = font_base_path + "JetBrainsMono-Regular.ttf";
    std::string mono_bold = font_base_path + "JetBrainsMono-Bold.ttf";

    std::string terminal_symbol_fallback;
#ifdef _WIN32
    wchar_t windows_directory[MAX_PATH]{};
    const UINT windows_directory_length =
        GetWindowsDirectoryW(windows_directory, MAX_PATH);
    if (windows_directory_length > 0 && windows_directory_length < MAX_PATH) {
        const auto candidate = std::filesystem::path(windows_directory) /
                               "Fonts" / "seguisym.ttf";
        if (std::filesystem::is_regular_file(candidate))
            terminal_symbol_fallback = candidate.string();
    }
#endif

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
    icon_config.OversampleH = needs_manual_oversampling ? 3 : 1;
    icon_config.OversampleV = needs_manual_oversampling ? 2 : 1;
    icon_config.RasterizerDensity = font_rasterizer_density_;
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

    // Heading font: Inter-Medium at a larger size, for section and dialog
    // titles (TOFIX129). Screens get it through cyxwiz::ui::GetFont.
    const float heading_font_size = std::round(base_font_size * 1.35f);
    if (std::filesystem::exists(inter_medium)) {
        ImFont* heading = io.Fonts->AddFontFromFileTTF(inter_medium.c_str(), heading_font_size, &font_config);
        if (heading) {
            spdlog::info("Loaded Inter-Medium heading ({}px)", heading_font_size);
            if (std::filesystem::exists(fa_solid)) {
                icon_config.GlyphMinAdvanceX = heading_font_size;
                io.Fonts->AddFontFromFileTTF(fa_solid.c_str(), heading_font_size - 1.0f, &icon_config, icon_ranges);
                icon_config.GlyphMinAdvanceX = base_font_size;
            }
            cyxwiz::ui::RegisterFont(cyxwiz::ui::Font::Heading, heading);
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

    // Load monospace editor fonts as real atlas sizes. Runtime scaling blurs text.
    auto merge_code_icons = [&](float font_size) {
        if (std::filesystem::exists(fa_solid)) {
            icon_config.GlyphMinAdvanceX = font_size;
            io.Fonts->AddFontFromFileTTF(fa_solid.c_str(), font_size - 1.0f, &icon_config, icon_ranges);
            icon_config.GlyphMinAdvanceX = base_font_size;
        }
    };

    if (std::filesystem::exists(mono_regular)) {
        for (size_t i = 0; i < cyxwiz::gui::kEditorFontScales.size(); ++i) {
            const float scale = cyxwiz::gui::kEditorFontScales[i];
            const float pixel_size = cyxwiz::gui::kEditorMonoFontPixels[i];
            ImFont* mono_font = cyxwiz::gui::AddTerminalCapableMonoFont(
                io.Fonts, mono_regular.c_str(),
                terminal_symbol_fallback.empty()
                    ? nullptr
                    : terminal_symbol_fallback.c_str(),
                pixel_size, &font_config);
            if (mono_font) {
                cyxwiz::gui::RegisterEditorMonoFont(scale, mono_font);
                if (i == 0) {
                    font_mono_ = mono_font;
                }
                spdlog::info("Loaded JetBrainsMono-Regular ({}px) for editor scale {}", pixel_size, scale);
                merge_code_icons(pixel_size);
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
        io.Fonts->AddFontDefault(&font_config);
    }

    io.FontDefault = font_regular_;
    cyxwiz::ui::RegisterFont(cyxwiz::ui::Font::Regular, font_regular_);
    cyxwiz::ui::RegisterFont(cyxwiz::ui::Font::Medium, font_medium_);
    cyxwiz::ui::RegisterFont(cyxwiz::ui::Font::Bold, font_bold_);

    // Build font atlas
    spdlog::info("Building font atlas...");
    spdlog::default_logger()->flush();  // Force flush before potential crash
    io.Fonts->Build();
    spdlog::info("Font atlas built successfully at {:.2f}x rasterizer density",
                 font_rasterizer_density_);
}
