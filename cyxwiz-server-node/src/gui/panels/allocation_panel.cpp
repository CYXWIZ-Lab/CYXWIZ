// allocation_panel.cpp - Resource allocation panel implementation
#include "gui/panels/allocation_panel.h"
#include "gui/icons.h"
#include "core/backend_manager.h"
#include "core/state_manager.h"
#include "ipc/daemon_client.h"
#include "auth/auth_manager.h"

#include <imgui.h>
#include <imgui_internal.h>
#include <fmt/format.h>
#include <spdlog/spdlog.h>
#include <fstream>
#include <nlohmann/json.hpp>

#ifdef _WIN32
#include <windows.h>
#include <shlobj.h>
#endif

namespace cyxwiz::servernode::gui {

// Color scheme
namespace AllocColors {
    const ImVec4 SectionBg = ImVec4(0.08f, 0.10f, 0.12f, 1.0f);
    const ImVec4 CardBg = ImVec4(0.10f, 0.12f, 0.14f, 1.0f);
    const ImVec4 CardBgEnabled = ImVec4(0.08f, 0.15f, 0.12f, 1.0f);
    const ImVec4 CardBorder = ImVec4(0.20f, 0.22f, 0.25f, 1.0f);
    const ImVec4 CardBorderEnabled = ImVec4(0.30f, 0.70f, 0.40f, 1.0f);
    const ImVec4 TextPrimary = ImVec4(1.0f, 1.0f, 1.0f, 1.0f);
    const ImVec4 TextSecondary = ImVec4(0.6f, 0.6f, 0.6f, 1.0f);
    const ImVec4 TextLabel = ImVec4(0.5f, 0.5f, 0.5f, 1.0f);
    const ImVec4 AccentGreen = ImVec4(0.45f, 0.80f, 0.45f, 1.0f);
    const ImVec4 AccentBlue = ImVec4(0.40f, 0.75f, 0.90f, 1.0f);
    const ImVec4 AccentOrange = ImVec4(0.85f, 0.65f, 0.35f, 1.0f);
    const ImVec4 AccentRed = ImVec4(0.90f, 0.40f, 0.40f, 1.0f);
    const ImVec4 SliderGrab = ImVec4(0.45f, 0.80f, 0.45f, 1.0f);
    const ImVec4 ButtonApply = ImVec4(0.30f, 0.65f, 0.35f, 1.0f);
    const ImVec4 ButtonApplyHover = ImVec4(0.35f, 0.75f, 0.40f, 1.0f);
}

AllocationPanel::AllocationPanel() : ServerPanel("Allocation") {
    last_refresh_ = std::chrono::steady_clock::now();
}

void AllocationPanel::Update() {
    if (!devices_initialized_) {
        RefreshDeviceList();
        LoadAllocations();
        devices_initialized_ = true;
    }
}

void AllocationPanel::RefreshDeviceList() {
    allocations_.clear();

    // Get GPU info from metrics
    std::vector<core::GPUMetrics> gpus;

    if (IsDaemonConnected()) {
        auto* daemon = GetDaemonClient();
        ipc::DaemonStatus status;
        if (daemon->GetStatus(status)) {
            // Use the full GPU list from daemon metrics (includes NVIDIA, Intel, AMD)
            for (const auto& gpu_info : status.metrics.gpus) {
                core::GPUMetrics gpu;
                gpu.device_id = gpu_info.device_id;
                gpu.name = gpu_info.name;
                gpu.vram_total_bytes = gpu_info.vram_total;
                gpus.push_back(gpu);
            }
        }
    } else {
        auto* state = GetState();
        if (state) {
            auto metrics = state->GetMetrics();
            gpus = metrics.gpus;
        }
    }

    // Create GPU allocations
    for (const auto& gpu : gpus) {
        ResourceAllocation alloc;
        alloc.device_type = ResourceAllocation::DeviceType::Gpu;
        alloc.device_id = gpu.device_id;
        alloc.device_name = gpu.name;
        alloc.vram_total_mb = gpu.vram_total_bytes / (1024 * 1024);
        alloc.vram_reserved_mb = std::min<size_t>(2048, alloc.vram_total_mb / 4); // Reserve 2GB or 25%
        alloc.vram_allocated_mb = alloc.vram_total_mb - alloc.vram_reserved_mb;
        alloc.is_enabled = false;
        allocations_.push_back(alloc);
    }

    // Create CPU allocation
#ifdef _WIN32
    SYSTEM_INFO sysInfo;
    GetSystemInfo(&sysInfo);

    ResourceAllocation cpu_alloc;
    cpu_alloc.device_type = ResourceAllocation::DeviceType::Cpu;
    cpu_alloc.device_id = 0;
    cpu_alloc.cores_total = sysInfo.dwNumberOfProcessors;
    cpu_alloc.cores_reserved = std::min(2, cpu_alloc.cores_total / 4);
    cpu_alloc.cores_allocated = cpu_alloc.cores_total - cpu_alloc.cores_reserved;
    cpu_alloc.is_enabled = false;

    // Get CPU name
    HKEY hKey;
    if (RegOpenKeyExA(HKEY_LOCAL_MACHINE,
                      "HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0",
                      0, KEY_READ, &hKey) == ERROR_SUCCESS) {
        char cpuName[256] = {0};
        DWORD bufSize = sizeof(cpuName);
        if (RegQueryValueExA(hKey, "ProcessorNameString", nullptr, nullptr,
                            (LPBYTE)cpuName, &bufSize) == ERROR_SUCCESS) {
            cpu_alloc.device_name = cpuName;
            // Trim
            size_t start = cpu_alloc.device_name.find_first_not_of(" ");
            size_t end = cpu_alloc.device_name.find_last_not_of(" ");
            if (start != std::string::npos) {
                cpu_alloc.device_name = cpu_alloc.device_name.substr(start, end - start + 1);
            }
        }
        RegCloseKey(hKey);
    }
    if (cpu_alloc.device_name.empty()) {
        cpu_alloc.device_name = "CPU";
    }

    allocations_.push_back(cpu_alloc);
#endif
}

void AllocationPanel::Render() {
    Update();

    ImVec2 content_size = ImGui::GetContentRegionAvail();

    if (ImGui::BeginChild("AllocationContent", ImVec2(0, 0), false)) {
        ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(8, 12));

        RenderHeader();
        ImGui::Spacing();

        RenderGpuAllocations();
        ImGui::Spacing();

        RenderCpuAllocation();
        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();

        RenderSummary();
        ImGui::Spacing();

        RenderActionButtons();

        ImGui::PopStyleVar();
    }
    ImGui::EndChild();
}

void AllocationPanel::RenderHeader() {
    ImGui::PushFont(GetSafeFont(FONT_LARGE));
    ImGui::TextColored(AllocColors::TextPrimary, ICON_FA_SLIDERS " Resource Allocation");
    ImGui::PopFont();

    ImGui::TextColored(AllocColors::TextSecondary,
        "Configure which resources to share with the CyxWiz network");
}

void AllocationPanel::RenderGpuAllocations() {
    ImGui::TextColored(AllocColors::TextPrimary, ICON_FA_MICROCHIP " GPU Resources");
    ImGui::Spacing();

    bool has_gpus = false;
    for (size_t i = 0; i < allocations_.size(); i++) {
        if (allocations_[i].device_type == ResourceAllocation::DeviceType::Gpu) {
            RenderGpuAllocationCard(static_cast<int>(i), allocations_[i]);
            ImGui::Spacing();
            has_gpus = true;
        }
    }

    if (!has_gpus) {
        ImGui::TextColored(AllocColors::TextSecondary, "No GPUs detected");
    }
}

void AllocationPanel::RenderGpuAllocationCard(int index, ResourceAllocation& alloc) {
    ImGui::PushID(index);

    ImVec4 bg_color = alloc.is_enabled ? AllocColors::CardBgEnabled : AllocColors::CardBg;
    ImVec4 border_color = alloc.is_enabled ? AllocColors::CardBorderEnabled : AllocColors::CardBorder;

    ImGui::PushStyleColor(ImGuiCol_ChildBg, bg_color);
    ImGui::PushStyleColor(ImGuiCol_Border, border_color);
    ImGui::PushStyleVar(ImGuiStyleVar_ChildBorderSize, 2.0f);

    float card_height = show_advanced_ ? 180.0f : 140.0f;
    if (ImGui::BeginChild("GPUAllocCard", ImVec2(0, card_height), true)) {
        // Header with toggle
        bool enabled = alloc.is_enabled;
        if (ImGui::Checkbox("##Enable", &enabled)) {
            alloc.is_enabled = enabled;
            allocations_dirty_ = true;
        }
        ImGui::SameLine();

        ImGui::PushFont(GetSafeFont(FONT_LARGE));
        ImGui::TextColored(AllocColors::TextPrimary, "GPU %d: %s", alloc.device_id, alloc.device_name.c_str());
        ImGui::PopFont();

        ImGui::SameLine(ImGui::GetContentRegionAvail().x - 100);
        if (alloc.is_enabled) {
            ImGui::TextColored(AllocColors::AccentGreen, "SHARING");
        } else {
            ImGui::TextColored(AllocColors::TextSecondary, "DISABLED");
        }

        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();

        // VRAM allocation slider
        ImGui::BeginDisabled(!alloc.is_enabled);

        ImGui::TextColored(AllocColors::TextLabel, "VRAM Allocation");
        ImGui::SameLine(200);
        ImGui::TextColored(AllocColors::TextPrimary, "%s / %s",
            FormatMB(alloc.vram_allocated_mb).c_str(),
            FormatMB(alloc.vram_total_mb).c_str());

        // Slider for VRAM
        ImGui::PushStyleColor(ImGuiCol_SliderGrab, AllocColors::SliderGrab);
        ImGui::PushStyleColor(ImGuiCol_SliderGrabActive, AllocColors::AccentGreen);

        size_t min_alloc = 0;
        size_t max_alloc = alloc.vram_total_mb - alloc.vram_reserved_mb;
        int vram_int = static_cast<int>(alloc.vram_allocated_mb);

        ImGui::SetNextItemWidth(-1);
        if (ImGui::SliderInt("##VRAM", &vram_int, static_cast<int>(min_alloc), static_cast<int>(max_alloc), "%d MB")) {
            alloc.vram_allocated_mb = static_cast<size_t>(vram_int);
            allocations_dirty_ = true;
        }

        ImGui::PopStyleColor(2);

        // Reserved display
        ImGui::TextColored(AllocColors::TextSecondary, "Reserved for system: %s",
            FormatMB(alloc.vram_reserved_mb).c_str());

        // Advanced options
        if (show_advanced_) {
            ImGui::Spacing();

            // Priority
            ImGui::TextColored(AllocColors::TextLabel, "Priority");
            ImGui::SameLine(200);
            const char* priorities[] = { "Low", "Normal", "High" };
            ImGui::SetNextItemWidth(100);
            if (ImGui::Combo("##Priority", &alloc.priority, priorities, 3)) {
                allocations_dirty_ = true;
            }
        }

        ImGui::EndDisabled();
    }
    ImGui::EndChild();

    ImGui::PopStyleVar();
    ImGui::PopStyleColor(2);
    ImGui::PopID();
}

void AllocationPanel::RenderCpuAllocation() {
    ImGui::TextColored(AllocColors::TextPrimary, ICON_FA_DESKTOP " CPU Resources");
    ImGui::Spacing();

    // Find CPU allocation
    ResourceAllocation* cpu_alloc = nullptr;
    for (auto& alloc : allocations_) {
        if (alloc.device_type == ResourceAllocation::DeviceType::Cpu) {
            cpu_alloc = &alloc;
            break;
        }
    }

    if (!cpu_alloc) {
        ImGui::TextColored(AllocColors::TextSecondary, "CPU not detected");
        return;
    }

    RenderCpuAllocationCard(*cpu_alloc);
}

void AllocationPanel::RenderCpuAllocationCard(ResourceAllocation& alloc) {
    ImGui::PushID("CPU");

    ImVec4 bg_color = alloc.is_enabled ? AllocColors::CardBgEnabled : AllocColors::CardBg;
    ImVec4 border_color = alloc.is_enabled ? AllocColors::CardBorderEnabled : AllocColors::CardBorder;

    ImGui::PushStyleColor(ImGuiCol_ChildBg, bg_color);
    ImGui::PushStyleColor(ImGuiCol_Border, border_color);
    ImGui::PushStyleVar(ImGuiStyleVar_ChildBorderSize, 2.0f);

    float card_height = show_advanced_ ? 160.0f : 120.0f;
    if (ImGui::BeginChild("CPUAllocCard", ImVec2(0, card_height), true)) {
        // Header with toggle
        bool enabled = alloc.is_enabled;
        if (ImGui::Checkbox("##Enable", &enabled)) {
            alloc.is_enabled = enabled;
            allocations_dirty_ = true;
        }
        ImGui::SameLine();

        ImGui::PushFont(GetSafeFont(FONT_LARGE));
        ImGui::TextColored(AllocColors::TextPrimary, "%s", alloc.device_name.c_str());
        ImGui::PopFont();

        ImGui::SameLine(ImGui::GetContentRegionAvail().x - 100);
        if (alloc.is_enabled) {
            ImGui::TextColored(AllocColors::AccentGreen, "SHARING");
        } else {
            ImGui::TextColored(AllocColors::TextSecondary, "DISABLED");
        }

        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();

        // Core allocation slider
        ImGui::BeginDisabled(!alloc.is_enabled);

        ImGui::TextColored(AllocColors::TextLabel, "Core Allocation");
        ImGui::SameLine(200);
        ImGui::TextColored(AllocColors::TextPrimary, "%d / %d cores",
            alloc.cores_allocated, alloc.cores_total);

        // Slider for cores
        ImGui::PushStyleColor(ImGuiCol_SliderGrab, AllocColors::SliderGrab);
        ImGui::PushStyleColor(ImGuiCol_SliderGrabActive, AllocColors::AccentGreen);

        int max_cores = alloc.cores_total - alloc.cores_reserved;
        ImGui::SetNextItemWidth(-1);
        if (ImGui::SliderInt("##Cores", &alloc.cores_allocated, 0, max_cores, "%d cores")) {
            allocations_dirty_ = true;
        }

        ImGui::PopStyleColor(2);

        ImGui::TextColored(AllocColors::TextSecondary, "Reserved for system: %d cores",
            alloc.cores_reserved);

        ImGui::EndDisabled();
    }
    ImGui::EndChild();

    ImGui::PopStyleVar();
    ImGui::PopStyleColor(2);
    ImGui::PopID();
}

void AllocationPanel::RenderSummary() {
    ImGui::PushStyleColor(ImGuiCol_ChildBg, AllocColors::SectionBg);

    if (ImGui::BeginChild("SummarySection", ImVec2(0, 70), true)) {
        ImGui::TextColored(AllocColors::TextPrimary, ICON_FA_CHART_PIE " Allocation Summary");
        ImGui::Spacing();

        ImGui::Columns(4, "SummaryCols", false);

        // Enabled GPUs
        ImGui::TextColored(AllocColors::TextLabel, "GPUs Enabled");
        int enabled_gpus = GetEnabledGpuCount();
        ImGui::TextColored(enabled_gpus > 0 ? AllocColors::AccentGreen : AllocColors::TextSecondary,
            "%d", enabled_gpus);
        ImGui::NextColumn();

        // Total VRAM
        ImGui::TextColored(AllocColors::TextLabel, "VRAM Allocated");
        size_t total_vram = GetTotalAllocatedVramMb();
        ImGui::TextColored(total_vram > 0 ? AllocColors::AccentBlue : AllocColors::TextSecondary,
            "%s", FormatMB(total_vram).c_str());
        ImGui::NextColumn();

        // CPU Cores
        ImGui::TextColored(AllocColors::TextLabel, "CPU Cores");
        int total_cores = GetTotalAllocatedCores();
        ImGui::TextColored(total_cores > 0 ? AllocColors::AccentOrange : AllocColors::TextSecondary,
            "%d", total_cores);
        ImGui::NextColumn();

        // Status
        ImGui::TextColored(AllocColors::TextLabel, "Status");
        if (HasAllocations()) {
            ImGui::TextColored(AllocColors::AccentGreen, "Ready");
        } else {
            ImGui::TextColored(AllocColors::TextSecondary, "No Resources");
        }

        ImGui::Columns(1);
    }
    ImGui::EndChild();
    ImGui::PopStyleColor();
}

void AllocationPanel::RenderActionButtons() {
    // Advanced toggle
    if (ImGui::Checkbox("Show Advanced Options", &show_advanced_)) {
        // Just toggle the flag
    }

    float button_width = show_retry_button_ ? 320.0f : 220.0f;
    ImGui::SameLine(ImGui::GetContentRegionAvail().x - button_width);

    // Save button
    if (ImGui::Button(ICON_FA_FLOPPY_DISK " Save", ImVec2(100, 30))) {
        SaveAllocations();
        status_message_ = "Settings saved!";
    }

    ImGui::SameLine();

    // Apply button
    ImGui::PushStyleColor(ImGuiCol_Button, AllocColors::ButtonApply);
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, AllocColors::ButtonApplyHover);

    ImGui::BeginDisabled(!HasAllocations() || is_applying_);
    if (ImGui::Button(ICON_FA_CHECK " Apply", ImVec2(100, 30))) {
        ApplyAllocations();
    }
    ImGui::EndDisabled();

    ImGui::PopStyleColor(2);

    // Retry button (shown when connection failed)
    if (show_retry_button_) {
        ImGui::SameLine();

        ImGui::PushStyleColor(ImGuiCol_Button, AllocColors::AccentOrange);
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.95f, 0.75f, 0.45f, 1.0f));

        ImGui::BeginDisabled(is_applying_);
        if (ImGui::Button(ICON_FA_ROTATE_RIGHT " Retry", ImVec2(100, 30))) {
            RetryConnection();
        }
        ImGui::EndDisabled();

        ImGui::PopStyleColor(2);
    }

    // Status message
    if (!status_message_.empty()) {
        ImGui::Spacing();
        ImVec4 msg_color = connection_failed_ ? AllocColors::AccentRed : AllocColors::AccentGreen;
        ImGui::TextColored(msg_color, "%s", status_message_.c_str());
    }
}

bool AllocationPanel::HasAllocations() const {
    for (const auto& alloc : allocations_) {
        if (alloc.is_enabled) {
            if (alloc.device_type == ResourceAllocation::DeviceType::Gpu && alloc.vram_allocated_mb > 0) {
                return true;
            }
            if (alloc.device_type == ResourceAllocation::DeviceType::Cpu && alloc.cores_allocated > 0) {
                return true;
            }
        }
    }
    return false;
}

size_t AllocationPanel::GetTotalAllocatedVramMb() const {
    size_t total = 0;
    for (const auto& alloc : allocations_) {
        if (alloc.is_enabled && alloc.device_type == ResourceAllocation::DeviceType::Gpu) {
            total += alloc.vram_allocated_mb;
        }
    }
    return total;
}

int AllocationPanel::GetTotalAllocatedCores() const {
    int total = 0;
    for (const auto& alloc : allocations_) {
        if (alloc.is_enabled && alloc.device_type == ResourceAllocation::DeviceType::Cpu) {
            total += alloc.cores_allocated;
        }
    }
    return total;
}

int AllocationPanel::GetEnabledGpuCount() const {
    int count = 0;
    for (const auto& alloc : allocations_) {
        if (alloc.is_enabled && alloc.device_type == ResourceAllocation::DeviceType::Gpu) {
            count++;
        }
    }
    return count;
}

std::string AllocationPanel::FormatMB(size_t mb) const {
    if (mb >= 1024) {
        return fmt::format("{:.1f} GB", mb / 1024.0);
    }
    return fmt::format("{} MB", mb);
}

void AllocationPanel::LoadAllocations() {
    // Load from config file
    std::string config_path;

#ifdef _WIN32
    char path[MAX_PATH];
    if (SUCCEEDED(SHGetFolderPathA(nullptr, CSIDL_APPDATA, nullptr, 0, path))) {
        config_path = std::string(path) + "\\CyxWiz\\allocations.json";
    }
#else
    config_path = std::string(getenv("HOME")) + "/.config/cyxwiz/allocations.json";
#endif

    if (config_path.empty()) return;

    try {
        std::ifstream file(config_path);
        if (!file.is_open()) return;

        nlohmann::json j;
        file >> j;

        // Apply saved settings to current allocations
        if (j.contains("allocations")) {
            for (const auto& saved : j["allocations"]) {
                int device_id = saved.value("device_id", -1);
                std::string type = saved.value("type", "");

                for (auto& alloc : allocations_) {
                    bool type_match = (type == "gpu" && alloc.device_type == ResourceAllocation::DeviceType::Gpu) ||
                                     (type == "cpu" && alloc.device_type == ResourceAllocation::DeviceType::Cpu);

                    if (type_match && alloc.device_id == device_id) {
                        alloc.is_enabled = saved.value("enabled", false);
                        alloc.priority = saved.value("priority", 1);

                        if (alloc.device_type == ResourceAllocation::DeviceType::Gpu) {
                            size_t saved_vram = saved.value("vram_allocated_mb", 0);
                            // Clamp to valid range
                            size_t max_alloc = alloc.vram_total_mb - alloc.vram_reserved_mb;
                            alloc.vram_allocated_mb = std::min(saved_vram, max_alloc);
                        } else {
                            int saved_cores = saved.value("cores_allocated", 0);
                            int max_cores = alloc.cores_total - alloc.cores_reserved;
                            alloc.cores_allocated = std::min(saved_cores, max_cores);
                        }
                        break;
                    }
                }
            }
        }
    } catch (...) {
        // Ignore load errors
    }
}

void AllocationPanel::SaveAllocations() {
    std::string config_path;

#ifdef _WIN32
    char path[MAX_PATH];
    if (SUCCEEDED(SHGetFolderPathA(nullptr, CSIDL_APPDATA, nullptr, 0, path))) {
        config_path = std::string(path) + "\\CyxWiz";
        CreateDirectoryA(config_path.c_str(), nullptr);
        config_path += "\\allocations.json";
    }
#else
    config_path = std::string(getenv("HOME")) + "/.config/cyxwiz";
    mkdir(config_path.c_str(), 0755);
    config_path += "/allocations.json";
#endif

    if (config_path.empty()) return;

    try {
        nlohmann::json j;
        j["allocations"] = nlohmann::json::array();

        for (const auto& alloc : allocations_) {
            nlohmann::json item;
            item["device_id"] = alloc.device_id;
            item["type"] = alloc.device_type == ResourceAllocation::DeviceType::Gpu ? "gpu" : "cpu";
            item["enabled"] = alloc.is_enabled;
            item["priority"] = alloc.priority;

            if (alloc.device_type == ResourceAllocation::DeviceType::Gpu) {
                item["vram_allocated_mb"] = alloc.vram_allocated_mb;
            } else {
                item["cores_allocated"] = alloc.cores_allocated;
            }

            j["allocations"].push_back(item);
        }

        std::ofstream file(config_path);
        file << j.dump(2);

        allocations_dirty_ = false;
    } catch (...) {
        status_message_ = "Failed to save settings";
    }
}

void AllocationPanel::ApplyAllocations() {
    // Save first
    SaveAllocations();

    // Reset state
    show_retry_button_ = false;
    connection_failed_ = false;
    is_applying_ = true;

    // Check if user is logged in
    auto& auth = auth::AuthManager::Instance();
    if (!auth.IsAuthenticated()) {
        status_message_ = "Please log in first to connect to Central Server";
        is_applying_ = false;
        return;
    }

    // Check if daemon is connected
    if (!IsDaemonConnected() || !GetDaemonClient()) {
        status_message_ = "Not connected to daemon";
        is_applying_ = false;
        return;
    }

    // Convert allocations to IPC format
    std::vector<ipc::DeviceAllocationInfo> ipc_allocations;
    for (const auto& alloc : allocations_) {
        if (alloc.is_enabled) {
            ipc::DeviceAllocationInfo info;
            info.device_type = (alloc.device_type == ResourceAllocation::DeviceType::Gpu)
                ? ipc::AllocDeviceType::Gpu : ipc::AllocDeviceType::Cpu;
            info.device_id = alloc.device_id;
            info.is_enabled = alloc.is_enabled;
            info.vram_allocation_mb = static_cast<int>(alloc.vram_allocated_mb);
            info.cpu_cores_allocation = alloc.cores_allocated;
            info.priority = static_cast<ipc::AllocPriority>(alloc.priority);
            ipc_allocations.push_back(info);
        }
    }

    // Get JWT token
    std::string jwt_token = auth.GetJwtToken();

    // Call daemon to set allocations and connect to Central Server
    auto result = GetDaemonClient()->SetAllocations(ipc_allocations, jwt_token, true);

    is_applying_ = false;

    if (result.success) {
        if (result.connected_to_central) {
            status_message_ = "Connected to Central Server! Node ID: " + result.node_id;
            show_retry_button_ = false;
            connection_failed_ = false;
            
            // Sync node_id with Web API so website shows Central Server ID
            if (!result.node_id.empty()) {
                bool synced = auth.SyncNodeIdWithWebApi(result.node_id);
                if (synced) {
                    spdlog::info("Node ID synced with Web API: {}", result.node_id);
                } else {
                    spdlog::warn("Failed to sync node_id with Web API");
                }
            }
        } else {
            status_message_ = "Allocations saved (offline mode)";
        }
    } else {
        status_message_ = "Connection failed: " + result.message;
        show_retry_button_ = true;
        connection_failed_ = true;
    }
}

void AllocationPanel::RetryConnection() {
    if (!IsDaemonConnected() || !GetDaemonClient()) {
        status_message_ = "Not connected to daemon";
        return;
    }

    is_applying_ = true;
    auto result = GetDaemonClient()->RetryConnection();
    is_applying_ = false;

    if (result.success && result.connected) {
        status_message_ = "Connected to Central Server! Node ID: " + result.node_id;
        show_retry_button_ = false;
        connection_failed_ = false;
        
        // Sync node_id with Web API so website shows Central Server ID
        if (!result.node_id.empty()) {
            auto& auth = auth::AuthManager::Instance();
            bool synced = auth.SyncNodeIdWithWebApi(result.node_id);
            if (synced) {
                spdlog::info("Node ID synced with Web API: {}", result.node_id);
            } else {
                spdlog::warn("Failed to sync node_id with Web API");
            }
        }
    } else {
        status_message_ = "Retry failed: " + result.message;
        show_retry_button_ = true;
        connection_failed_ = true;
    }
}

} // namespace cyxwiz::servernode::gui
