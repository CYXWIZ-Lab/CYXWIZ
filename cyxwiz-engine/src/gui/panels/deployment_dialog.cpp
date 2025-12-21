// deployment_dialog.cpp - UI dialog for deploying models
#include "deployment_dialog.h"
#include "../icons.h"
#include "../../auth/auth_client.h"
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
    central_server_client_ = std::make_unique<network::GRPCClient>();
}

DeploymentDialog::~DeploymentDialog() {
    // Stop any running embedded server
    if (embedded_server_ && embedded_server_->IsRunning()) {
        embedded_server_->Stop();
    }

    // Disconnect from Central Server
    if (central_server_client_ && central_server_client_->IsConnected()) {
        central_server_client_->Disconnect();
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

    // Taller window when GGUF options are shown
    ImGui::SetNextWindowSize(ImVec2(550, is_gguf_model_ ? 750 : 600), ImGuiCond_FirstUseEver);

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

        // GGUF-specific config (shown for both modes when GGUF model detected)
        if (is_gguf_model_) {
            ImGui::Separator();
            RenderGGUFConfig();
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
        // Detect model format from extension
        std::string ext = fs::path(model_path_).extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        is_gguf_model_ = (ext == ".gguf");

        if (fs::is_directory(model_path_)) {
            ImGui::TextColored(ImVec4(0.3f, 0.8f, 0.3f, 1.0f), "%s Directory format", ICON_FA_CHECK);
            // Count files in directory
            size_t file_count = 0;
            for (const auto& entry : fs::directory_iterator(model_path_)) {
                if (entry.is_regular_file()) file_count++;
            }
            ImGui::SameLine();
            ImGui::TextDisabled("(%zu files)", file_count);
        } else if (is_gguf_model_) {
            ImGui::TextColored(ImVec4(0.3f, 0.6f, 1.0f, 1.0f), "%s GGUF/LLM Model", ICON_FA_BRAIN);
            auto file_size = fs::file_size(model_path_);
            ImGui::SameLine();
            ImGui::TextDisabled("(%.2f GB)", file_size / (1024.0 * 1024.0 * 1024.0));
        } else {
            ImGui::TextColored(ImVec4(0.3f, 0.8f, 0.3f, 1.0f), "%s Binary format", ICON_FA_CHECK);
            auto file_size = fs::file_size(model_path_);
            ImGui::SameLine();
            ImGui::TextDisabled("(%.2f MB)", file_size / (1024.0 * 1024.0));
        }
    } else if (strlen(model_path_) > 0) {
        ImGui::TextColored(ImVec4(0.8f, 0.3f, 0.3f, 1.0f), "%s Path not found", ICON_FA_TRIANGLE_EXCLAMATION);
        is_gguf_model_ = false;
    } else {
        is_gguf_model_ = false;
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

    // Check authentication status
    auto& auth = auth::AuthClient::Instance();
    bool is_authenticated = auth.IsAuthenticated();

    // ========== Step 1: Connect to Central Server ==========
    ImGui::Text("%s Step 1: Connect to Central Server", ICON_FA_CLOUD);
    ImGui::Spacing();

    ImGui::SetNextItemWidth(250);
    ImGui::InputText("##CentralServerAddress", central_server_address_, sizeof(central_server_address_));

    ImGui::SameLine();
    bool central_connected = central_server_client_ && central_server_client_->IsConnected();

    if (central_connected) {
        if (ImGui::Button(ICON_FA_LINK_SLASH " Disconnect##Central")) {
            DisconnectFromCentralServer();
        }
    } else {
        if (!is_authenticated) {
            ImGui::BeginDisabled();
        }
        if (ImGui::Button(ICON_FA_LINK " Connect##Central")) {
            ConnectToCentralServer();
        }
        if (!is_authenticated) {
            ImGui::EndDisabled();
            if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) {
                ImGui::SetTooltip("Please log in first to connect to Central Server");
            }
        }
    }

    // Central Server connection status
    if (central_connected) {
        ImGui::TextColored(ImVec4(0.3f, 0.8f, 0.3f, 1.0f), "%s Connected to Central Server", ICON_FA_CIRCLE_CHECK);
    } else if (!is_authenticated) {
        ImGui::TextColored(ImVec4(0.8f, 0.6f, 0.2f, 1.0f), "%s Login required", ICON_FA_LOCK);
    } else {
        ImGui::TextColored(ImVec4(0.5f, 0.5f, 0.5f, 1.0f), "%s Not connected", ICON_FA_CIRCLE_XMARK);
    }

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    // ========== Step 2: Discover and Select Node ==========
    if (central_connected) {
        RenderNodeDiscoverySection();
        ImGui::Separator();
        ImGui::Spacing();
    }

    // ========== Step 3: Direct Connection to Selected Node ==========
    bool has_selected_node = !selected_node_id_.empty();

    if (has_selected_node) {
        ImGui::Text("%s Step 3: Connect to Selected Node", ICON_FA_SERVER);
        ImGui::Spacing();

        // Show selected node address
        ImGui::Text("Target: %s", server_address_);
        ImGui::SameLine();

        bool node_connected = deployment_client_ && deployment_client_->IsConnected();

        if (node_connected) {
            if (ImGui::Button(ICON_FA_LINK_SLASH " Disconnect##Node")) {
                DisconnectFromServerNode();
            }
        } else {
            if (ImGui::Button(ICON_FA_LINK " Connect##Node")) {
                ConnectToServerNode();
            }
        }

        // Node connection status
        if (node_connected) {
            ImGui::TextColored(ImVec4(0.3f, 0.8f, 0.3f, 1.0f), "%s Connected to Node", ICON_FA_CIRCLE_CHECK);

            ImGui::Spacing();
            ImGui::Separator();
            ImGui::Spacing();

            // Deployment settings (only if connected to node)
            ImGui::Text("Deployment Settings");
            ImGui::Spacing();

            ImGui::SetNextItemWidth(150);
            ImGui::InputInt("Port", &server_port_);
            if (server_port_ < 1024) server_port_ = 1024;
            if (server_port_ > 65535) server_port_ = 65535;

            ImGui::SetNextItemWidth(150);
            ImGui::InputInt("GPU Layers", &n_gpu_layers_);
            if (n_gpu_layers_ < 0) n_gpu_layers_ = 0;
            if (n_gpu_layers_ > 100) n_gpu_layers_ = 100;
            ImGui::SameLine();
            ImGui::TextDisabled("(0 = CPU only)");

            ImGui::SetNextItemWidth(150);
            ImGui::InputInt("Context Size", &context_size_);
            if (context_size_ < 512) context_size_ = 512;
            if (context_size_ > 131072) context_size_ = 131072;

            ImGui::Checkbox("Enable Terminal Access", &enable_terminal_);
        } else {
            ImGui::TextColored(ImVec4(0.5f, 0.5f, 0.5f, 1.0f), "%s Not connected to node", ICON_FA_CIRCLE_XMARK);
        }
    } else if (central_connected) {
        ImGui::TextColored(ImVec4(0.8f, 0.6f, 0.2f, 1.0f),
            "%s Select a node from the list above to continue", ICON_FA_ARROW_UP);
    }

    ImGui::Spacing();
}

void DeploymentDialog::RenderGGUFConfig() {
    ImGui::Text("%s GGUF/LLM Configuration", ICON_FA_BRAIN);
    ImGui::Spacing();

    // GPU Offloading
    ImGui::SetNextItemWidth(200);
    ImGui::SliderInt("GPU Layers##GGUF", &n_gpu_layers_, 0, 100);
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Number of layers to offload to GPU\n"
                          "0 = CPU only\n"
                          "Higher = more GPU memory used, faster inference");
    }

    // Context Size with presets
    ImGui::SetNextItemWidth(200);
    const char* ctx_items[] = {"512", "1K", "2K", "4K", "8K", "16K", "32K", "64K", "128K"};
    static int ctx_sizes[] = {512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072};
    int ctx_idx = 2;  // Default 2048
    for (int i = 0; i < 9; ++i) {
        if (ctx_sizes[i] == context_size_) {
            ctx_idx = i;
            break;
        }
    }
    if (ImGui::Combo("Context Size##GGUF", &ctx_idx, ctx_items, 9)) {
        context_size_ = ctx_sizes[ctx_idx];
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Maximum context window size\n"
                          "Larger = more memory, longer prompts supported");
    }

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Text("Sampling Parameters");
    ImGui::Spacing();

    // Temperature
    ImGui::SetNextItemWidth(200);
    ImGui::SliderFloat("Temperature", &temperature_, 0.0f, 2.0f, "%.2f");
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Controls randomness\n"
                          "0 = deterministic\n"
                          "1 = balanced\n"
                          "2 = very creative");
    }

    // Max Tokens
    ImGui::SetNextItemWidth(200);
    ImGui::SliderInt("Max Tokens", &max_tokens_, 16, 4096);
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Maximum number of tokens to generate");
    }

    // Advanced sampling (collapsible)
    if (ImGui::CollapsingHeader("Advanced Sampling")) {
        ImGui::Indent();

        ImGui::SetNextItemWidth(200);
        ImGui::SliderFloat("Top-P", &top_p_, 0.0f, 1.0f, "%.2f");
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Nucleus sampling threshold\n"
                              "Lower = more focused\n"
                              "1.0 = disabled");
        }

        ImGui::SetNextItemWidth(200);
        ImGui::SliderInt("Top-K", &top_k_, 1, 100);
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Top-K sampling\n"
                              "Lower = more focused");
        }

        ImGui::SetNextItemWidth(200);
        ImGui::SliderFloat("Repeat Penalty", &repeat_penalty_, 1.0f, 2.0f, "%.2f");
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Penalize repeated tokens\n"
                              "1.0 = no penalty");
        }

        ImGui::Unindent();
    }

    ImGui::Spacing();

    // Embedding mode toggle
    ImGui::Checkbox("Embedding Mode", &enable_embeddings_);
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Run as embedding model (for semantic search)\n"
                          "Requires embedding-capable model like nomic-embed, e5, etc.");
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
        // Throttled refresh - only poll every refresh_interval_seconds_
        auto now = std::chrono::steady_clock::now();
        float elapsed = std::chrono::duration<float>(now - last_refresh_time_).count();

        // Use longer interval if last refresh failed (backoff)
        float interval = last_refresh_failed_ ? refresh_interval_seconds_ * 5.0f : refresh_interval_seconds_;

        if (elapsed >= interval) {
            last_refresh_time_ = now;
            last_refresh_failed_ = !deployment_client_->ListDeployments(server_deployments_);
        }

        for (const auto& deploy : server_deployments_) {
            has_deployments = true;

            ImGui::PushID(deploy.id.c_str());
            ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4(0.1f, 0.1f, 0.2f, 0.5f));
            ImGui::BeginChild("ServerDeployment", ImVec2(0, 80), true);

            ImGui::Text("%s %s", ICON_FA_SERVER, deploy.model_name.c_str());

            // Show Stop button for running deployments, Delete for stopped ones
            bool is_stopped = (deploy.status == 6 || deploy.status == 7 || deploy.status == 8);  // STOPPED, FAILED, TERMINATED
            bool is_running = (deploy.status == 4 || deploy.status == 5 || deploy.status == 3);  // READY, RUNNING, LOADING

            if (is_stopped) {
                ImGui::SameLine(ImGui::GetWindowWidth() - 80);
                ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.6f, 0.2f, 0.2f, 1.0f));
                if (ImGui::SmallButton(ICON_FA_TRASH " Delete")) {
                    DeleteServerNodeDeployment(deploy.id);
                }
                ImGui::PopStyleColor();
            } else if (is_running) {
                ImGui::SameLine(ImGui::GetWindowWidth() - 80);
                if (ImGui::SmallButton(ICON_FA_STOP " Stop")) {
                    StopServerNodeDeployment(deploy.id);
                }
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
                case 6:  // STOPPED
                    status_color = ImVec4(0.5f, 0.5f, 0.5f, 1.0f);
                    status_text = "Stopped";
                    break;
                case 7:  // FAILED
                    status_color = ImVec4(0.9f, 0.3f, 0.3f, 1.0f);
                    status_text = "Failed";
                    break;
                case 8:  // TERMINATED
                    status_color = ImVec4(0.6f, 0.4f, 0.4f, 1.0f);
                    status_text = "Terminated";
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
        config.gpu_layers = n_gpu_layers_;
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

void DeploymentDialog::DeleteServerNodeDeployment(const std::string& deployment_id) {
    if (!deployment_client_ || !deployment_client_->IsConnected()) {
        error_message_ = "Not connected to Server Node";
        return;
    }

    if (deployment_client_->DeleteDeployment(deployment_id)) {
        status_message_ = "Deployment deleted";
        if (active_server_deployment_id_ == deployment_id) {
            active_server_deployment_id_.clear();
        }
        // Force refresh of deployment list
        last_refresh_time_ = std::chrono::steady_clock::time_point{};
    } else {
        error_message_ = "Failed to delete: " + deployment_client_->GetLastError();
    }
}

void DeploymentDialog::ConnectToServerNode() {
    error_message_.clear();
    status_message_.clear();

    // Require authentication before connecting
    auto& auth = auth::AuthClient::Instance();
    if (!auth.IsAuthenticated()) {
        error_message_ = "Please log in first to connect to Server Node";
        spdlog::warn("Connection rejected - user not authenticated");
        return;
    }

    // Set JWT token for authenticated requests
    deployment_client_->SetAuthToken(auth.GetJwtToken());
    spdlog::debug("Set JWT token for deployment client");

    if (deployment_client_->Connect(server_address_)) {
        status_message_ = "Connected to Server Node";
        // Reset refresh state for immediate first refresh
        last_refresh_time_ = std::chrono::steady_clock::time_point{};
        last_refresh_failed_ = false;
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

// ============================================================================
// Central Server / Node Discovery
// ============================================================================

void DeploymentDialog::ConnectToCentralServer() {
    error_message_.clear();
    status_message_.clear();

    auto& auth = auth::AuthClient::Instance();
    if (!auth.IsAuthenticated()) {
        error_message_ = "Please log in first to connect to Central Server";
        return;
    }

    central_server_client_->SetAuthToken(auth.GetJwtToken());

    if (central_server_client_->Connect(central_server_address_)) {
        status_message_ = "Connected to Central Server";
        // Immediately refresh node list
        RefreshNodeList();
    } else {
        error_message_ = "Failed to connect: " + central_server_client_->GetLastError();
    }
}

void DeploymentDialog::DisconnectFromCentralServer() {
    central_server_client_->Disconnect();
    discovered_nodes_.clear();
    selected_node_index_ = -1;
    selected_node_id_.clear();
    status_message_ = "Disconnected from Central Server";
}

void DeploymentDialog::RefreshNodeList() {
    if (!central_server_client_ || !central_server_client_->IsConnected()) {
        return;
    }

    last_node_refresh_time_ = std::chrono::steady_clock::now();

    if (!central_server_client_->ListNodes(discovered_nodes_, true, 50)) {
        spdlog::warn("Failed to refresh node list: {}", central_server_client_->GetLastError());
    }
}

void DeploymentDialog::SearchNodes() {
    if (!central_server_client_ || !central_server_client_->IsConnected()) {
        return;
    }

    // Build search criteria from UI filters
    search_criteria_.required_device = "";
    if (filter_device_type_ == 1) search_criteria_.required_device = "CUDA";
    else if (filter_device_type_ == 2) search_criteria_.required_device = "OpenCL";
    else if (filter_device_type_ == 3) search_criteria_.required_device = "CPU";

    search_criteria_.min_vram = static_cast<int64_t>(filter_min_vram_gb_ * 1024 * 1024 * 1024);
    search_criteria_.max_price_per_hour = filter_max_price_;
    search_criteria_.min_reputation = filter_min_reputation_;
    search_criteria_.require_free_tier = filter_free_tier_only_;
    search_criteria_.preferred_region = filter_region_;
    search_criteria_.sort_by = filter_sort_by_;
    search_criteria_.max_results = 50;

    if (!central_server_client_->FindNodes(search_criteria_, discovered_nodes_)) {
        error_message_ = "Search failed: " + central_server_client_->GetLastError();
    }
}

void DeploymentDialog::RenderNodeDiscoverySection() {
    ImGui::Text("%s Step 2: Select a Compute Node", ICON_FA_NETWORK_WIRED);
    ImGui::Spacing();

    // Toolbar
    if (ImGui::Button(ICON_FA_ROTATE " Refresh")) {
        RefreshNodeList();
    }
    ImGui::SameLine();

    if (ImGui::Button(show_search_filters_ ? ICON_FA_FILTER " Hide Filters" : ICON_FA_FILTER " Filters")) {
        show_search_filters_ = !show_search_filters_;
    }
    ImGui::SameLine();
    ImGui::TextDisabled("| %zu nodes found", discovered_nodes_.size());

    // Auto-refresh
    auto now = std::chrono::steady_clock::now();
    float elapsed = std::chrono::duration<float>(now - last_node_refresh_time_).count();
    if (elapsed >= node_refresh_interval_seconds_) {
        RefreshNodeList();
    }

    // Search filters
    if (show_search_filters_) {
        RenderNodeSearchFilters();
    }

    ImGui::Spacing();

    // Node table
    RenderNodeTable();

    // Selected node info
    if (selected_node_index_ >= 0 && selected_node_index_ < static_cast<int>(discovered_nodes_.size())) {
        ImGui::Spacing();
        RenderSelectedNodeInfo();
    }
}

void DeploymentDialog::RenderNodeSearchFilters() {
    ImGui::BeginChild("SearchFilters", ImVec2(0, 120), true);

    ImGui::Columns(3, "FilterColumns", false);

    // Column 1: Hardware
    ImGui::Text("Hardware");
    const char* device_items[] = {"Any", "CUDA (NVIDIA)", "OpenCL", "CPU"};
    ImGui::SetNextItemWidth(120);
    ImGui::Combo("Device##Filter", &filter_device_type_, device_items, 4);

    ImGui::SetNextItemWidth(120);
    ImGui::SliderFloat("Min VRAM (GB)", &filter_min_vram_gb_, 0.0f, 48.0f, "%.1f");

    ImGui::NextColumn();

    // Column 2: Pricing
    ImGui::Text("Pricing");
    ImGui::SetNextItemWidth(120);
    ImGui::InputFloat("Max $/hr", &filter_max_price_, 0.01f, 0.1f, "%.2f");
    if (filter_max_price_ < 0) filter_max_price_ = 0;

    ImGui::Checkbox("Free tier only", &filter_free_tier_only_);

    ImGui::NextColumn();

    // Column 3: Trust & Location
    ImGui::Text("Trust & Location");
    ImGui::SetNextItemWidth(120);
    ImGui::SliderFloat("Min Reputation", &filter_min_reputation_, 0.0f, 1.0f, "%.2f");

    ImGui::SetNextItemWidth(120);
    ImGui::InputText("Region", filter_region_, sizeof(filter_region_));

    ImGui::Columns(1);

    ImGui::Spacing();

    // Sort options
    const char* sort_items[] = {"Price (Low to High)", "Performance", "Reputation", "Availability"};
    ImGui::SetNextItemWidth(150);
    ImGui::Combo("Sort By", &filter_sort_by_, sort_items, 4);

    ImGui::SameLine();
    if (ImGui::Button(ICON_FA_MAGNIFYING_GLASS " Search")) {
        SearchNodes();
    }
    ImGui::SameLine();
    if (ImGui::Button(ICON_FA_ERASER " Clear")) {
        filter_device_type_ = 0;
        filter_min_vram_gb_ = 0.0f;
        filter_max_price_ = 0.0f;
        filter_min_reputation_ = 0.0f;
        filter_free_tier_only_ = false;
        filter_region_[0] = '\0';
        filter_sort_by_ = 0;
        RefreshNodeList();
    }

    ImGui::EndChild();
}

void DeploymentDialog::RenderNodeTable() {
    // Table with scrolling
    ImGui::BeginChild("NodeTableChild", ImVec2(0, 200), true);

    if (ImGui::BeginTable("NodeTable", 7, ImGuiTableFlags_Borders |
                                           ImGuiTableFlags_RowBg |
                                           ImGuiTableFlags_Resizable |
                                           ImGuiTableFlags_ScrollY)) {
        // Headers
        ImGui::TableSetupColumn("", ImGuiTableColumnFlags_WidthFixed, 30);  // Radio button
        ImGui::TableSetupColumn("Name", ImGuiTableColumnFlags_WidthStretch);
        ImGui::TableSetupColumn("Device", ImGuiTableColumnFlags_WidthFixed, 70);
        ImGui::TableSetupColumn("VRAM", ImGuiTableColumnFlags_WidthFixed, 60);
        ImGui::TableSetupColumn("Price/hr", ImGuiTableColumnFlags_WidthFixed, 80);
        ImGui::TableSetupColumn("Rep.", ImGuiTableColumnFlags_WidthFixed, 50);
        ImGui::TableSetupColumn("Status", ImGuiTableColumnFlags_WidthFixed, 60);
        ImGui::TableHeadersRow();

        // Rows
        for (int i = 0; i < static_cast<int>(discovered_nodes_.size()); ++i) {
            const auto& node = discovered_nodes_[i];
            ImGui::PushID(i);

            ImGui::TableNextRow();

            // Radio button for selection
            ImGui::TableNextColumn();
            bool is_selected = (selected_node_index_ == i);
            if (ImGui::RadioButton("##Select", is_selected)) {
                selected_node_index_ = i;
                selected_node_id_ = node.node_id;

                // Set server address for P2P connection using actual endpoint
                if (!node.ip_address.empty()) {
                    snprintf(server_address_, sizeof(server_address_), "%s:%d",
                             node.ip_address.c_str(), node.port);
                } else {
                    // Fallback to node_id if ip_address not available
                    snprintf(server_address_, sizeof(server_address_), "%s:50052", node.node_id.c_str());
                    spdlog::warn("Node {} has no IP address, using node_id as fallback", node.name);
                }
            }

            // Name
            ImGui::TableNextColumn();
            ImGui::Text("%s", node.name.c_str());
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("ID: %s\nRegion: %s", node.node_id.c_str(), node.region.c_str());
            }

            // Device
            ImGui::TableNextColumn();
            if (node.device_type == "CUDA") {
                ImGui::TextColored(ImVec4(0.3f, 0.8f, 0.3f, 1.0f), "%s", node.device_type.c_str());
            } else if (node.device_type == "OpenCL") {
                ImGui::TextColored(ImVec4(0.3f, 0.6f, 0.9f, 1.0f), "%s", node.device_type.c_str());
            } else {
                ImGui::Text("%s", node.device_type.c_str());
            }

            // VRAM
            ImGui::TableNextColumn();
            if (node.vram_bytes > 0) {
                ImGui::Text("%.1fG", node.vram_bytes / (1024.0 * 1024.0 * 1024.0));
            } else {
                ImGui::TextDisabled("-");
            }

            // Price
            ImGui::TableNextColumn();
            if (node.free_tier_available) {
                ImGui::TextColored(ImVec4(0.3f, 0.8f, 0.3f, 1.0f), ICON_FA_GIFT " Free");
            } else if (node.price_per_hour > 0) {
                ImGui::Text("%.4f", node.price_per_hour);
                if (ImGui::IsItemHovered() && node.price_usd_equivalent > 0) {
                    ImGui::SetTooltip("$%.2f USD/hr", node.price_usd_equivalent);
                }
            } else {
                ImGui::TextDisabled("-");
            }

            // Reputation
            ImGui::TableNextColumn();
            if (node.reputation_score >= 0.8) {
                ImGui::TextColored(ImVec4(0.3f, 0.8f, 0.3f, 1.0f), "%.0f%%", node.reputation_score * 100);
            } else if (node.reputation_score >= 0.5) {
                ImGui::TextColored(ImVec4(0.9f, 0.7f, 0.2f, 1.0f), "%.0f%%", node.reputation_score * 100);
            } else {
                ImGui::TextColored(ImVec4(0.8f, 0.3f, 0.3f, 1.0f), "%.0f%%", node.reputation_score * 100);
            }

            // Status
            ImGui::TableNextColumn();
            if (node.is_online) {
                ImGui::TextColored(ImVec4(0.3f, 0.8f, 0.3f, 1.0f), ICON_FA_CIRCLE " Online");
            } else {
                ImGui::TextColored(ImVec4(0.5f, 0.5f, 0.5f, 1.0f), ICON_FA_CIRCLE " Offline");
            }

            ImGui::PopID();
        }

        ImGui::EndTable();
    }

    if (discovered_nodes_.empty()) {
        ImGui::TextDisabled("No nodes available. Try refreshing or adjusting filters.");
    }

    ImGui::EndChild();
}

void DeploymentDialog::RenderSelectedNodeInfo() {
    if (selected_node_index_ < 0 || selected_node_index_ >= static_cast<int>(discovered_nodes_.size())) {
        return;
    }

    const auto& node = discovered_nodes_[selected_node_index_];

    ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4(0.1f, 0.15f, 0.2f, 0.5f));
    ImGui::BeginChild("SelectedNodeInfo", ImVec2(0, 80), true);

    ImGui::Text("%s Selected: %s", ICON_FA_CIRCLE_CHECK, node.name.c_str());

    ImGui::Columns(3, "SelectedNodeCols", false);

    // Hardware
    ImGui::Text("%s %s", ICON_FA_MICROCHIP, node.device_type.c_str());
    if (node.vram_bytes > 0) {
        ImGui::SameLine();
        ImGui::TextDisabled("(%.1f GB VRAM)", node.vram_bytes / (1024.0 * 1024.0 * 1024.0));
    }
    ImGui::Text("Compute Score: %.1f", node.compute_score);

    ImGui::NextColumn();

    // Pricing
    ImGui::Text("%s Billing: %s", ICON_FA_DOLLAR_SIGN, node.billing_model.c_str());
    ImGui::Text("Price: %.4f CYXWIZ/hr", node.price_per_hour);
    if (node.free_tier_available) {
        ImGui::TextColored(ImVec4(0.3f, 0.8f, 0.3f, 1.0f), ICON_FA_GIFT " Free tier available");
    }

    ImGui::NextColumn();

    // Trust
    ImGui::Text("%s Reputation: %.0f%%", ICON_FA_STAR, node.reputation_score * 100);
    ImGui::Text("Jobs Completed: %lld", (long long)node.total_jobs_completed);
    ImGui::Text("Staked: %.2f CYXWIZ", node.staked_amount);

    ImGui::Columns(1);

    ImGui::EndChild();
    ImGui::PopStyleColor();
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
