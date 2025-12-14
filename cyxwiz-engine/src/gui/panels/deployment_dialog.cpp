// deployment_dialog.cpp - UI dialog for deploying models
#include "deployment_dialog.h"
#include "../icons.h"
#include <imgui.h>
#include <spdlog/spdlog.h>
#include <filesystem>
#include <cstring>

#ifdef _WIN32
#include <windows.h>
#include <commdlg.h>
#include <shobjidl.h>  // For IFileDialog
#include <objbase.h>   // For CoInitialize
#endif

namespace cyxwiz {

namespace fs = std::filesystem;

DeploymentDialog::DeploymentDialog()
    : Panel("Deployment", true) {
    embedded_server_ = std::make_unique<LocalInferenceServer>();
    deployment_client_ = std::make_unique<network::DeploymentClient>();
}

DeploymentDialog::~DeploymentDialog() {
    // Stop any running embedded server
    if (embedded_server_ && embedded_server_->IsRunning()) {
        embedded_server_->Stop();
    }

    // Wait for any async operations
    if (operation_thread_ && operation_thread_->joinable()) {
        operation_thread_->join();
    }
}

void DeploymentDialog::Open() {
    is_open_ = true;
    error_message_.clear();
    status_message_.clear();

    // Refresh server node deployments if connected
    if (deployment_client_ && deployment_client_->IsConnected()) {
        deployment_client_->ListDeployments(server_deployments_);
    }
}

void DeploymentDialog::Close() {
    is_open_ = false;
}

void DeploymentDialog::SetModelPath(const std::string& path) {
    strncpy(model_path_, path.c_str(), sizeof(model_path_) - 1);
    model_path_[sizeof(model_path_) - 1] = '\0';
}

bool DeploymentDialog::HasActiveDeployment() const {
    if (embedded_server_ && embedded_server_->IsRunning()) {
        return true;
    }
    if (!active_server_deployment_id_.empty()) {
        return true;
    }
    return false;
}

void DeploymentDialog::Render() {
    if (!is_open_) return;

    ImGui::SetNextWindowSize(ImVec2(550, 600), ImGuiCond_FirstUseEver);

    if (ImGui::Begin(ICON_FA_ROCKET " Deploy Model", &is_open_)) {
        RenderModeSelector();
        ImGui::Separator();
        RenderModelSection();
        ImGui::Separator();

        if (mode_ == DeploymentMode::Embedded) {
            RenderEmbeddedConfig();
        } else {
            RenderServerNodeConfig();
        }

        ImGui::Separator();
        RenderDeployButton();
        ImGui::Separator();
        RenderActiveDeployments();
        RenderStatusMessages();
    }
    ImGui::End();
}

void DeploymentDialog::RenderModeSelector() {
    ImGui::Text("%s Deployment Mode", ICON_FA_SLIDERS);
    ImGui::Spacing();

    // Tab-style mode selection
    float tab_width = 200.0f;

    bool is_embedded = (mode_ == DeploymentMode::Embedded);
    bool is_server = (mode_ == DeploymentMode::ServerNode);

    // Embedded tab
    if (is_embedded) {
        ImGui::PushStyleColor(ImGuiCol_Button, ImGui::GetStyleColorVec4(ImGuiCol_ButtonActive));
    }
    if (ImGui::Button(ICON_FA_MICROCHIP " Embedded", ImVec2(tab_width, 0))) {
        mode_ = DeploymentMode::Embedded;
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Run inference server directly in the Engine\n(No Server Node daemon required)");
    }
    if (is_embedded) {
        ImGui::PopStyleColor();
    }

    ImGui::SameLine();

    // Server Node tab
    if (is_server) {
        ImGui::PushStyleColor(ImGuiCol_Button, ImGui::GetStyleColorVec4(ImGuiCol_ButtonActive));
    }
    if (ImGui::Button(ICON_FA_SERVER " Server Node", ImVec2(tab_width, 0))) {
        mode_ = DeploymentMode::ServerNode;
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Deploy to Server Node daemon\n(Requires daemon running on target address)");
    }
    if (is_server) {
        ImGui::PopStyleColor();
    }

    ImGui::Spacing();
}

void DeploymentDialog::RenderModelSection() {
    ImGui::Text("%s Model", ICON_FA_CUBE);
    ImGui::Spacing();

    ImGui::SetNextItemWidth(300);
    ImGui::InputText("##ModelPath", model_path_, sizeof(model_path_));

    ImGui::SameLine();
    if (ImGui::Button(ICON_FA_FILE " File")) {
        std::string path = OpenModelFile();
        if (!path.empty()) {
            SetModelPath(path);
        }
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Select binary .cyxmodel file\n(from Tools > Save Trained Model)");
    }

    ImGui::SameLine();
    if (ImGui::Button(ICON_FA_FOLDER " Folder")) {
        std::string path = OpenModelFolder();
        if (!path.empty()) {
            SetModelPath(path);
        }
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Select .cyxmodel directory\n(from Deploy > Export Model)");
    }

    // Show model info if valid path
    if (strlen(model_path_) > 0 && fs::exists(model_path_)) {
        if (fs::is_directory(model_path_)) {
            ImGui::TextColored(ImVec4(0.3f, 0.8f, 0.3f, 1.0f), "%s Directory format", ICON_FA_CHECK);
            // Count files in directory
            size_t file_count = 0;
            for (const auto& entry : fs::directory_iterator(model_path_)) {
                if (entry.is_regular_file()) file_count++;
            }
            ImGui::SameLine();
            ImGui::TextDisabled("(%zu files)", file_count);
        } else {
            ImGui::TextColored(ImVec4(0.3f, 0.8f, 0.3f, 1.0f), "%s Binary format", ICON_FA_CHECK);
            auto file_size = fs::file_size(model_path_);
            ImGui::SameLine();
            ImGui::TextDisabled("(%.2f MB)", file_size / (1024.0 * 1024.0));
        }
    } else if (strlen(model_path_) > 0) {
        ImGui::TextColored(ImVec4(0.8f, 0.3f, 0.3f, 1.0f), "%s Path not found", ICON_FA_TRIANGLE_EXCLAMATION);
    }

    ImGui::Spacing();
}

void DeploymentDialog::RenderEmbeddedConfig() {
    ImGui::Text("%s Embedded Configuration", ICON_FA_GEAR);
    ImGui::Spacing();

    ImGui::SetNextItemWidth(150);
    ImGui::InputInt("Port", &embedded_port_);
    if (embedded_port_ < 1024) embedded_port_ = 1024;
    if (embedded_port_ > 65535) embedded_port_ = 65535;
    ImGui::SameLine();
    ImGui::TextDisabled("(1024-65535)");

    ImGui::Spacing();
}

void DeploymentDialog::RenderServerNodeConfig() {
    ImGui::Text("%s Server Node Configuration", ICON_FA_GEAR);
    ImGui::Spacing();

    // Connection section
    ImGui::Text("Connection");
    ImGui::SetNextItemWidth(250);
    ImGui::InputText("##ServerAddress", server_address_, sizeof(server_address_));

    ImGui::SameLine();
    bool is_connected = deployment_client_ && deployment_client_->IsConnected();

    if (is_connected) {
        if (ImGui::Button(ICON_FA_LINK_SLASH " Disconnect")) {
            DisconnectFromServerNode();
        }
    } else {
        if (ImGui::Button(ICON_FA_LINK " Connect")) {
            ConnectToServerNode();
        }
    }

    // Connection status
    if (is_connected) {
        ImGui::TextColored(ImVec4(0.3f, 0.8f, 0.3f, 1.0f), "%s Connected", ICON_FA_CIRCLE_CHECK);
    } else {
        ImGui::TextColored(ImVec4(0.5f, 0.5f, 0.5f, 1.0f), "%s Disconnected", ICON_FA_CIRCLE_XMARK);
    }

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    // Deployment settings (only if connected)
    if (is_connected) {
        ImGui::Text("Deployment Settings");
        ImGui::Spacing();

        ImGui::SetNextItemWidth(150);
        ImGui::InputInt("Port", &server_port_);
        if (server_port_ < 1024) server_port_ = 1024;
        if (server_port_ > 65535) server_port_ = 65535;

        ImGui::SetNextItemWidth(150);
        ImGui::InputInt("GPU Layers", &gpu_layers_);
        if (gpu_layers_ < 0) gpu_layers_ = 0;
        if (gpu_layers_ > 100) gpu_layers_ = 100;
        ImGui::SameLine();
        ImGui::TextDisabled("(0 = CPU only)");

        ImGui::SetNextItemWidth(150);
        ImGui::InputInt("Context Size", &context_size_);
        if (context_size_ < 512) context_size_ = 512;
        if (context_size_ > 131072) context_size_ = 131072;

        ImGui::Checkbox("Enable Terminal Access", &enable_terminal_);
    }

    ImGui::Spacing();
}

void DeploymentDialog::RenderDeployButton() {
    bool is_deploying = is_deploying_.load();

    // Determine if we can deploy
    bool can_deploy = false;
    bool model_valid = strlen(model_path_) > 0 && fs::exists(model_path_);

    if (mode_ == DeploymentMode::Embedded) {
        bool server_running = embedded_server_ && embedded_server_->IsRunning();
        can_deploy = model_valid && !server_running && !is_deploying;
    } else {
        bool is_connected = deployment_client_ && deployment_client_->IsConnected();
        can_deploy = model_valid && is_connected && !is_deploying;
    }

    // Deploy button
    if (!can_deploy || is_deploying) {
        ImGui::BeginDisabled();
    }

    ImVec2 button_size(200, 40);
    if (ImGui::Button(ICON_FA_ROCKET " Start Server", button_size)) {
        if (mode_ == DeploymentMode::Embedded) {
            StartEmbeddedDeployment();
        } else {
            StartServerNodeDeployment();
        }
    }

    if (!can_deploy || is_deploying) {
        ImGui::EndDisabled();
    }

    if (is_deploying) {
        ImGui::SameLine();
        ImGui::Text("%s Deploying...", ICON_FA_SPINNER);
    }

    ImGui::Spacing();
}

void DeploymentDialog::RenderActiveDeployments() {
    ImGui::Text("%s Active Deployments", ICON_FA_SERVER);
    ImGui::Spacing();

    bool has_deployments = false;

    // Embedded server
    if (embedded_server_ && embedded_server_->IsRunning()) {
        has_deployments = true;

        ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4(0.1f, 0.2f, 0.1f, 0.5f));
        ImGui::BeginChild("EmbeddedDeployment", ImVec2(0, 80), true);

        ImGui::Text("%s %s", ICON_FA_MICROCHIP, embedded_server_->GetModelName().c_str());
        ImGui::SameLine(ImGui::GetWindowWidth() - 80);
        if (ImGui::SmallButton(ICON_FA_STOP " Stop")) {
            StopEmbeddedDeployment();
        }

        ImGui::TextColored(ImVec4(0.3f, 0.8f, 0.3f, 1.0f), "%s Running", ICON_FA_CIRCLE);
        ImGui::SameLine();
        ImGui::TextDisabled("| Port: %d", embedded_server_->GetPort());
        ImGui::SameLine();
        ImGui::TextDisabled("| Requests: %llu", (unsigned long long)embedded_server_->GetRequestCount());

        ImGui::TextDisabled("Endpoint: %s", embedded_server_->GetEndpointUrl().c_str());

        ImGui::EndChild();
        ImGui::PopStyleColor();
    }

    // Server Node deployments
    if (mode_ == DeploymentMode::ServerNode && deployment_client_ && deployment_client_->IsConnected()) {
        // Refresh deployments
        deployment_client_->ListDeployments(server_deployments_);

        for (const auto& deploy : server_deployments_) {
            has_deployments = true;

            ImGui::PushID(deploy.id.c_str());
            ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4(0.1f, 0.1f, 0.2f, 0.5f));
            ImGui::BeginChild("ServerDeployment", ImVec2(0, 80), true);

            ImGui::Text("%s %s", ICON_FA_SERVER, deploy.model_name.c_str());
            ImGui::SameLine(ImGui::GetWindowWidth() - 80);
            if (ImGui::SmallButton(ICON_FA_STOP " Stop")) {
                StopServerNodeDeployment(deploy.id);
            }

            // Status color
            ImVec4 status_color;
            const char* status_text;
            switch (deploy.status) {
                case 4:  // READY
                case 5:  // RUNNING
                    status_color = ImVec4(0.3f, 0.8f, 0.3f, 1.0f);
                    status_text = "Running";
                    break;
                case 3:  // LOADING
                    status_color = ImVec4(1.0f, 0.8f, 0.3f, 1.0f);
                    status_text = "Loading";
                    break;
                default:
                    status_color = ImVec4(0.5f, 0.5f, 0.5f, 1.0f);
                    status_text = "Unknown";
                    break;
            }

            ImGui::TextColored(status_color, "%s %s", ICON_FA_CIRCLE, status_text);
            ImGui::SameLine();
            ImGui::TextDisabled("| Port: %d", deploy.port);
            ImGui::SameLine();
            ImGui::TextDisabled("| GPU: %d", deploy.gpu_layers);
            ImGui::SameLine();
            ImGui::TextDisabled("| Requests: %llu", (unsigned long long)deploy.request_count);

            ImGui::TextDisabled("ID: %s", deploy.id.c_str());

            ImGui::EndChild();
            ImGui::PopStyleColor();
            ImGui::PopID();
        }
    }

    if (!has_deployments) {
        ImGui::TextDisabled("No active deployments.");
    }

    ImGui::Spacing();
}

void DeploymentDialog::RenderStatusMessages() {
    if (!status_message_.empty()) {
        ImGui::TextColored(ImVec4(0.3f, 0.8f, 0.3f, 1.0f), "%s %s", ICON_FA_CIRCLE_CHECK, status_message_.c_str());
    }

    if (!error_message_.empty()) {
        ImGui::TextColored(ImVec4(1.0f, 0.3f, 0.3f, 1.0f), "%s %s", ICON_FA_TRIANGLE_EXCLAMATION, error_message_.c_str());
    }
}

void DeploymentDialog::StartEmbeddedDeployment() {
    error_message_.clear();
    status_message_.clear();

    if (!fs::exists(model_path_)) {
        error_message_ = "Model file not found";
        return;
    }

    spdlog::info("Starting embedded deployment: {} on port {}", model_path_, embedded_port_);

    // Load model
    if (!embedded_server_->LoadModel(model_path_)) {
        error_message_ = "Failed to load model: " + embedded_server_->GetLastError();
        spdlog::error("{}", error_message_);
        return;
    }

    // Start server
    if (!embedded_server_->Start(embedded_port_)) {
        error_message_ = "Failed to start server: " + embedded_server_->GetLastError();
        spdlog::error("{}", error_message_);
        return;
    }

    status_message_ = "Server started on port " + std::to_string(embedded_port_);
    spdlog::info("{}", status_message_);
}

void DeploymentDialog::StopEmbeddedDeployment() {
    if (embedded_server_ && embedded_server_->IsRunning()) {
        embedded_server_->Stop();
        embedded_server_->UnloadModel();
        status_message_ = "Embedded server stopped";
        spdlog::info("{}", status_message_);
    }
}

void DeploymentDialog::StartServerNodeDeployment() {
    error_message_.clear();
    status_message_.clear();

    if (!deployment_client_ || !deployment_client_->IsConnected()) {
        error_message_ = "Not connected to Server Node";
        return;
    }

    is_deploying_ = true;

    // Run deployment in background
    if (operation_thread_ && operation_thread_->joinable()) {
        operation_thread_->join();
    }

    operation_thread_ = std::make_unique<std::thread>([this]() {
        network::DeploymentConfig config;
        config.model_path = model_path_;
        config.port = server_port_;
        config.gpu_layers = gpu_layers_;
        config.context_size = context_size_;
        config.enable_terminal = enable_terminal_;

        auto result = deployment_client_->Deploy(config);

        std::lock_guard<std::mutex> lock(state_mutex_);
        is_deploying_ = false;

        if (result.success) {
            active_server_deployment_id_ = result.deployment_id;
            status_message_ = "Deployed: " + result.endpoint_url;
        } else {
            error_message_ = "Deployment failed: " + result.error_message;
        }
    });
}

void DeploymentDialog::StopServerNodeDeployment(const std::string& deployment_id) {
    if (!deployment_client_ || !deployment_client_->IsConnected()) {
        error_message_ = "Not connected to Server Node";
        return;
    }

    if (deployment_client_->StopDeployment(deployment_id)) {
        status_message_ = "Deployment stopped";
        if (active_server_deployment_id_ == deployment_id) {
            active_server_deployment_id_.clear();
        }
    } else {
        error_message_ = "Failed to stop: " + deployment_client_->GetLastError();
    }
}

void DeploymentDialog::ConnectToServerNode() {
    error_message_.clear();
    status_message_.clear();

    if (deployment_client_->Connect(server_address_)) {
        status_message_ = "Connected to Server Node";
        deployment_client_->ListDeployments(server_deployments_);
    } else {
        error_message_ = "Connection failed: " + deployment_client_->GetLastError();
    }
}

void DeploymentDialog::DisconnectFromServerNode() {
    deployment_client_->Disconnect();
    server_deployments_.clear();
    active_server_deployment_id_.clear();
    status_message_ = "Disconnected from Server Node";
}

std::string DeploymentDialog::OpenModelFile() {
#ifdef _WIN32
    char filename[MAX_PATH] = "";

    OPENFILENAMEA ofn = {};
    ofn.lStructSize = sizeof(ofn);
    ofn.hwndOwner = nullptr;
    ofn.lpstrFilter = "CyxWiz Model (*.cyxmodel)\0*.cyxmodel\0All Files (*.*)\0*.*\0";
    ofn.lpstrFile = filename;
    ofn.nMaxFile = MAX_PATH;
    ofn.lpstrTitle = "Select Binary .cyxmodel File";
    ofn.Flags = OFN_PATHMUSTEXIST | OFN_FILEMUSTEXIST | OFN_NOCHANGEDIR;

    if (GetOpenFileNameA(&ofn)) {
        return std::string(filename);
    }
#endif
    return "";
}

std::string DeploymentDialog::OpenModelFolder() {
#ifdef _WIN32
    std::string result;
    HRESULT hr = CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED | COINIT_DISABLE_OLE1DDE);
    if (SUCCEEDED(hr)) {
        IFileDialog* pfd = nullptr;
        hr = CoCreateInstance(CLSID_FileOpenDialog, nullptr, CLSCTX_INPROC_SERVER,
                              IID_IFileDialog, reinterpret_cast<void**>(&pfd));

        if (SUCCEEDED(hr)) {
            DWORD dwOptions;
            hr = pfd->GetOptions(&dwOptions);
            if (SUCCEEDED(hr)) {
                hr = pfd->SetOptions(dwOptions | FOS_PICKFOLDERS | FOS_PATHMUSTEXIST);
            }

            pfd->SetTitle(L"Select .cyxmodel Directory");
            hr = pfd->Show(nullptr);
            if (SUCCEEDED(hr)) {
                IShellItem* psi = nullptr;
                hr = pfd->GetResult(&psi);
                if (SUCCEEDED(hr)) {
                    PWSTR pszPath = nullptr;
                    hr = psi->GetDisplayName(SIGDN_FILESYSPATH, &pszPath);
                    if (SUCCEEDED(hr)) {
                        int size = WideCharToMultiByte(CP_UTF8, 0, pszPath, -1, nullptr, 0, nullptr, nullptr);
                        if (size > 0) {
                            result.resize(size - 1);
                            WideCharToMultiByte(CP_UTF8, 0, pszPath, -1, &result[0], size, nullptr, nullptr);
                        }
                        CoTaskMemFree(pszPath);
                    }
                    psi->Release();
                }
            }
            pfd->Release();
        }
        CoUninitialize();
    }
    return result;
#endif
    return "";
}

} // namespace cyxwiz
