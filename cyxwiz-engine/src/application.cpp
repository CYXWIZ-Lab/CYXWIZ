#include "application.h"
#include "gui/main_window.h"
#include "gui/console.h"
#include "scripting/python_engine.h"
#include "network/grpc_client.h"
#include "network/job_manager.h"

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <spdlog/spdlog.h>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
#include <implot.h>
#include <imnodes.h>
#include <cyxwiz/device.h>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#include <filesystem>

static void glfw_error_callback(int error, const char* description) {
    spdlog::error("GLFW Error {}: {}", error, description);
}

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

    // Setup Dear ImGui style
    ImGui::StyleColorsDark();

    // When viewports are enabled we tweak WindowRounding/WindowBg
    ImGuiStyle& style = ImGui::GetStyle();
    if (io.ConfigFlags & ImGuiConfigFlags_ViewportsEnable) {
        style.WindowRounding = 0.0f;
        style.Colors[ImGuiCol_WindowBg].w = 1.0f;
    }

    // Setup Platform/Renderer backends
    ImGui_ImplGlfw_InitForOpenGL(window_, true);
    ImGui_ImplOpenGL3_Init(glsl_version);

    // Load fonts
    // io.Fonts->AddFontFromFileTTF("resources/fonts/Roboto-Medium.ttf", 16.0f);

    // Initialize components
    main_window_ = std::make_unique<gui::MainWindow>();
    python_engine_ = std::make_unique<scripting::PythonEngine>();
    grpc_client_ = std::make_unique<network::GRPCClient>();
    job_manager_ = std::make_unique<network::JobManager>(grpc_client_.get());

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

    while (!glfwWindowShouldClose(window_) && running_) {
        double current_time = glfwGetTime();
        float delta_time = static_cast<float>(current_time - last_frame_time_);
        last_frame_time_ = current_time;

        HandleInput();
        Update(delta_time);
        Render();
    }

    return 0;
}

void CyxWizApp::HandleInput() {
    glfwPollEvents();
}

void CyxWizApp::Update(float delta_time) {
    // Update components
    if (job_manager_) {
        job_manager_->Update();
    }
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
