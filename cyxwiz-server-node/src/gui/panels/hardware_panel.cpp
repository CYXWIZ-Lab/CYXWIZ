// hardware_panel.cpp - Hardware detection display panel
#include "gui/panels/hardware_panel.h"
#include "gui/icons.h"
#include "core/backend_manager.h"
#include "core/state_manager.h"
#include "core/metrics_collector.h"
#include "ipc/daemon_client.h"

#include <imgui.h>
#include <imgui_internal.h>
#include <fmt/format.h>

#ifdef _WIN32
#include <windows.h>
#include <intrin.h>
#elif defined(__APPLE__)
#include <sys/sysctl.h>
#include <mach/mach.h>
#include <thread>
#endif

namespace cyxwiz::servernode::gui {

// Color scheme
namespace HWColors {
    const ImVec4 SectionBg = ImVec4(0.08f, 0.10f, 0.12f, 1.0f);
    const ImVec4 CardBg = ImVec4(0.10f, 0.12f, 0.14f, 1.0f);
    const ImVec4 CardBorder = ImVec4(0.20f, 0.22f, 0.25f, 1.0f);
    const ImVec4 TextPrimary = ImVec4(1.0f, 1.0f, 1.0f, 1.0f);
    const ImVec4 TextSecondary = ImVec4(0.6f, 0.6f, 0.6f, 1.0f);
    const ImVec4 TextLabel = ImVec4(0.5f, 0.5f, 0.5f, 1.0f);
    const ImVec4 AccentGreen = ImVec4(0.45f, 0.80f, 0.45f, 1.0f);
    const ImVec4 AccentBlue = ImVec4(0.40f, 0.75f, 0.90f, 1.0f);
    const ImVec4 AccentPurple = ImVec4(0.55f, 0.45f, 0.75f, 1.0f);
    const ImVec4 AccentOrange = ImVec4(0.85f, 0.65f, 0.35f, 1.0f);

    // Vendor-specific GPU colors
    const ImVec4 GPU_Intel = ImVec4(0.0f, 0.45f, 0.75f, 1.0f);    // Intel blue
    const ImVec4 GPU_NVIDIA = ImVec4(0.45f, 0.80f, 0.45f, 1.0f);  // NVIDIA green
    const ImVec4 GPU_AMD = ImVec4(0.90f, 0.35f, 0.35f, 1.0f);     // AMD red

    inline ImVec4 GetVendorColor(const std::string& vendor, bool is_nvidia) {
        if (is_nvidia || vendor == "NVIDIA") return GPU_NVIDIA;
        if (vendor == "Intel") return GPU_Intel;
        if (vendor == "AMD") return GPU_AMD;
        return AccentGreen;
    }
}

HardwarePanel::HardwarePanel() : ServerPanel("Hardware") {
    last_update_ = std::chrono::steady_clock::now();
}

void HardwarePanel::Update() {
    // Refresh hardware info periodically
    auto now = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_update_).count();

    if (!hardware_initialized_ || elapsed >= UPDATE_INTERVAL_MS) {
        RefreshHardwareInfo();
        last_update_ = now;
        hardware_initialized_ = true;
    }
}

void HardwarePanel::RefreshHardwareInfo() {
    // Get CPU info from registry (Windows)
#ifdef _WIN32
    SYSTEM_INFO sysInfo;
    GetSystemInfo(&sysInfo);
    cpu_info_.logical_cores = sysInfo.dwNumberOfProcessors;
    cpu_info_.physical_cores = cpu_info_.logical_cores / 2;
    if (cpu_info_.physical_cores < 1) cpu_info_.physical_cores = 1;

    // Get CPU name from registry
    HKEY hKey;
    if (RegOpenKeyExA(HKEY_LOCAL_MACHINE,
                      "HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0",
                      0, KEY_READ, &hKey) == ERROR_SUCCESS) {
        char cpuName[256] = {0};
        DWORD bufSize = sizeof(cpuName);
        if (RegQueryValueExA(hKey, "ProcessorNameString", nullptr, nullptr,
                            (LPBYTE)cpuName, &bufSize) == ERROR_SUCCESS) {
            cpu_info_.name = cpuName;
            // Trim whitespace
            size_t start = cpu_info_.name.find_first_not_of(" ");
            size_t end = cpu_info_.name.find_last_not_of(" ");
            if (start != std::string::npos) {
                cpu_info_.name = cpu_info_.name.substr(start, end - start + 1);
            }
        }

        char vendorId[256] = {0};
        bufSize = sizeof(vendorId);
        if (RegQueryValueExA(hKey, "VendorIdentifier", nullptr, nullptr,
                            (LPBYTE)vendorId, &bufSize) == ERROR_SUCCESS) {
            cpu_info_.vendor = vendorId;
        }

        DWORD mhz = 0;
        bufSize = sizeof(mhz);
        if (RegQueryValueExA(hKey, "~MHz", nullptr, nullptr,
                            (LPBYTE)&mhz, &bufSize) == ERROR_SUCCESS) {
            cpu_info_.base_speed_ghz = mhz / 1000.0f;
            cpu_info_.current_speed_ghz = cpu_info_.base_speed_ghz;
        }
        RegCloseKey(hKey);
    }

    // Get architecture
    switch (sysInfo.wProcessorArchitecture) {
        case PROCESSOR_ARCHITECTURE_AMD64:
            cpu_info_.architecture = "x64";
            break;
        case PROCESSOR_ARCHITECTURE_INTEL:
            cpu_info_.architecture = "x86";
            break;
        case PROCESSOR_ARCHITECTURE_ARM64:
            cpu_info_.architecture = "ARM64";
            break;
        default:
            cpu_info_.architecture = "Unknown";
    }

    // Get memory info
    MEMORYSTATUSEX memInfo;
    memInfo.dwLength = sizeof(MEMORYSTATUSEX);
    if (GlobalMemoryStatusEx(&memInfo)) {
        memory_info_.total_bytes = memInfo.ullTotalPhys;
        memory_info_.available_bytes = memInfo.ullAvailPhys;
        memory_info_.used_bytes = memory_info_.total_bytes - memory_info_.available_bytes;
    }
#elif defined(__APPLE__)
    // macOS CPU initialization
    size_t size = sizeof(cpu_info_.logical_cores);
    if (sysctlbyname("hw.logicalcpu", &cpu_info_.logical_cores, &size, nullptr, 0) != 0) {
        cpu_info_.logical_cores = std::thread::hardware_concurrency();
    }

    size = sizeof(cpu_info_.physical_cores);
    if (sysctlbyname("hw.physicalcpu", &cpu_info_.physical_cores, &size, nullptr, 0) != 0) {
        cpu_info_.physical_cores = cpu_info_.logical_cores / 2;
        if (cpu_info_.physical_cores < 1) cpu_info_.physical_cores = 1;
    }

    // Get CPU name
    char cpu_brand[256] = {0};
    size = sizeof(cpu_brand);
    if (sysctlbyname("machdep.cpu.brand_string", cpu_brand, &size, nullptr, 0) == 0) {
        cpu_info_.name = cpu_brand;
        // Trim whitespace
        size_t start = cpu_info_.name.find_first_not_of(" ");
        size_t end = cpu_info_.name.find_last_not_of(" ");
        if (start != std::string::npos) {
            cpu_info_.name = cpu_info_.name.substr(start, end - start + 1);
        }
    } else {
        cpu_info_.name = "Unknown CPU";
    }

    // Get CPU vendor
    char cpu_vendor[256] = {0};
    size = sizeof(cpu_vendor);
    if (sysctlbyname("machdep.cpu.vendor", cpu_vendor, &size, nullptr, 0) == 0) {
        cpu_info_.vendor = cpu_vendor;
    } else {
        cpu_info_.vendor = "Unknown";
    }

    // Get CPU frequency (in Hz)
    uint64_t freq_hz = 0;
    size = sizeof(freq_hz);
    if (sysctlbyname("hw.cpufrequency", &freq_hz, &size, nullptr, 0) == 0) {
        cpu_info_.base_speed_ghz = freq_hz / 1000000000.0f;
        cpu_info_.current_speed_ghz = cpu_info_.base_speed_ghz;
    } else {
        cpu_info_.base_speed_ghz = 0.0f;
        cpu_info_.current_speed_ghz = 0.0f;
    }

    // Get architecture
#if defined(__x86_64__)
    cpu_info_.architecture = "x86_64";
#elif defined(__aarch64__) || defined(__arm64__)
    cpu_info_.architecture = "ARM64";
#else
    cpu_info_.architecture = "Unknown";
#endif

    // Get memory info using Mach
    mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
    vm_statistics64_data_t vm_stat;
    if (host_statistics64(mach_host_self(), HOST_VM_INFO64,
                         (host_info64_t)&vm_stat, &count) == KERN_SUCCESS) {
        // Get page size
        vm_size_t page_size;
        host_page_size(mach_host_self(), &page_size);

        // Total physical memory
        uint64_t total_mem = 0;
        size = sizeof(total_mem);
        if (sysctlbyname("hw.memsize", &total_mem, &size, nullptr, 0) == 0) {
            memory_info_.total_bytes = total_mem;
        }

        // Calculate available memory (free + inactive + purgeable)
        uint64_t free_pages = vm_stat.free_count;
        uint64_t inactive_pages = vm_stat.inactive_count;
        uint64_t purgeable_pages = vm_stat.purgeable_count;
        memory_info_.available_bytes = (free_pages + inactive_pages + purgeable_pages) * page_size;
        memory_info_.used_bytes = memory_info_.total_bytes - memory_info_.available_bytes;
    }
#endif
}

void HardwarePanel::Render() {
    // Update hardware info
    Update();

    ImVec2 content_size = ImGui::GetContentRegionAvail();

    // Main scrollable area
    if (ImGui::BeginChild("HardwareContent", ImVec2(0, 0), false)) {
        ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(8, 12));

        // Summary section at top
        RenderSummarySection();

        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();

        // GPU Section
        RenderGpuSection();

        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();

        // CPU Section
        RenderCpuSection();

        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();

        // Memory Section
        RenderMemorySection();

        ImGui::PopStyleVar();
    }
    ImGui::EndChild();
}

void HardwarePanel::RenderSectionHeader(const char* title, const char* icon) {
    ImGui::PushFont(GetSafeFont(FONT_LARGE));
    if (icon) {
        ImGui::TextColored(HWColors::TextSecondary, "%s", icon);
        ImGui::SameLine();
    }
    ImGui::TextColored(HWColors::TextPrimary, "%s", title);
    ImGui::PopFont();
    ImGui::Spacing();
}

void HardwarePanel::RenderSummarySection() {
    // Get GPU count from metrics
    int gpu_count = 0;
    size_t total_vram = 0;

    if (IsDaemonConnected()) {
        auto* daemon = GetDaemonClient();
        ipc::DaemonStatus status;
        if (daemon->GetStatus(status)) {
            gpu_count = status.gpu_count;
            total_vram = status.metrics.vram_total;
        }
    } else {
        auto* state = GetState();
        if (state) {
            auto metrics = state->GetMetrics();
            gpu_count = metrics.gpu_count;
            for (const auto& gpu : metrics.gpus) {
                total_vram += gpu.vram_total_bytes;
            }
        }
    }

    ImGui::PushStyleColor(ImGuiCol_ChildBg, HWColors::SectionBg);
    if (ImGui::BeginChild("SummarySection", ImVec2(0, 80), true)) {
        ImGui::Columns(4, "SummaryCols", false);

        // GPU Count
        ImGui::TextColored(HWColors::TextLabel, "GPUs");
        ImGui::PushFont(GetSafeFont(FONT_LARGE));
        ImGui::TextColored(HWColors::AccentGreen, "%d", gpu_count);
        ImGui::PopFont();
        ImGui::NextColumn();

        // Total VRAM
        ImGui::TextColored(HWColors::TextLabel, "Total VRAM");
        ImGui::PushFont(GetSafeFont(FONT_LARGE));
        ImGui::TextColored(HWColors::AccentBlue, "%s", FormatBytes(total_vram).c_str());
        ImGui::PopFont();
        ImGui::NextColumn();

        // CPU Cores
        ImGui::TextColored(HWColors::TextLabel, "CPU Cores");
        ImGui::PushFont(GetSafeFont(FONT_LARGE));
        ImGui::TextColored(HWColors::AccentPurple, "%d / %d",
            cpu_info_.physical_cores, cpu_info_.logical_cores);
        ImGui::PopFont();
        ImGui::NextColumn();

        // Total RAM
        ImGui::TextColored(HWColors::TextLabel, "System RAM");
        ImGui::PushFont(GetSafeFont(FONT_LARGE));
        ImGui::TextColored(HWColors::AccentOrange, "%s", FormatBytes(memory_info_.total_bytes).c_str());
        ImGui::PopFont();

        ImGui::Columns(1);
    }
    ImGui::EndChild();
    ImGui::PopStyleColor();
}

void HardwarePanel::RenderGpuSection() {
    RenderSectionHeader("Graphics Processing Units", ICON_FA_MICROCHIP);

    // Get GPU metrics
    std::vector<core::GPUMetrics> gpus;

    if (IsDaemonConnected()) {
        auto* daemon = GetDaemonClient();
        ipc::DaemonStatus status;
        if (daemon->GetStatus(status)) {
            // Convert daemon GPU info to core::GPUMetrics
            for (const auto& gpu_info : status.metrics.gpus) {
                core::GPUMetrics gpu;
                gpu.device_id = gpu_info.device_id;
                gpu.name = gpu_info.name;
                gpu.vendor = gpu_info.vendor;
                gpu.usage_3d = gpu_info.usage_3d;
                gpu.usage_copy = gpu_info.usage_copy;
                gpu.usage_video_decode = gpu_info.usage_video_decode;
                gpu.usage_video_encode = gpu_info.usage_video_encode;
                gpu.memory_usage = gpu_info.memory_usage;
                gpu.vram_used_bytes = gpu_info.vram_used;
                gpu.vram_total_bytes = gpu_info.vram_total;
                gpu.temperature_celsius = gpu_info.temperature;
                gpu.power_watts = gpu_info.power_watts;
                gpu.is_nvidia = gpu_info.is_nvidia;
                gpus.push_back(gpu);
            }
            // Fallback: if no per-GPU data but gpu_count > 0, use legacy metrics
            if (gpus.empty() && status.gpu_count > 0) {
                core::GPUMetrics gpu;
                gpu.device_id = 0;
                gpu.name = status.gpu_name;
                gpu.usage_3d = status.metrics.gpu_usage;
                gpu.memory_usage = status.metrics.vram_usage;
                gpu.vram_used_bytes = status.metrics.vram_used;
                gpu.vram_total_bytes = status.metrics.vram_total;
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

    if (gpus.empty()) {
        ImGui::TextColored(HWColors::TextSecondary, "No GPUs detected");
        return;
    }

    // Render each GPU as a card
    for (size_t i = 0; i < gpus.size(); i++) {
        RenderGpuCard(static_cast<int>(i), gpus[i]);
        if (i < gpus.size() - 1) {
            ImGui::Spacing();
        }
    }
}

void HardwarePanel::RenderGpuCard(int index, const core::GPUMetrics& gpu) {
    ImGui::PushID(index);

    ImVec4 vendor_color = HWColors::GetVendorColor(gpu.vendor, gpu.is_nvidia);

    ImGui::PushStyleColor(ImGuiCol_ChildBg, HWColors::CardBg);
    ImGui::PushStyleColor(ImGuiCol_Border, vendor_color);
    ImGui::PushStyleVar(ImGuiStyleVar_ChildBorderSize, 2.0f);

    float card_height = 120.0f;
    if (ImGui::BeginChild("GPUCard", ImVec2(0, card_height), true)) {
        // Header row: GPU name and vendor badge
        ImGui::PushFont(GetSafeFont(FONT_LARGE));
        ImGui::TextColored(HWColors::TextPrimary, "GPU %d: %s", index, gpu.name.c_str());
        ImGui::PopFont();

        ImGui::SameLine(ImGui::GetContentRegionAvail().x - 80);
        ImGui::TextColored(vendor_color, "[%s]",
            gpu.is_nvidia ? "NVIDIA" :
            (gpu.vendor.empty() ? "Unknown" : gpu.vendor.c_str()));

        ImGui::Spacing();

        // Stats in columns
        ImGui::Columns(4, "GPUStats", false);

        // VRAM
        ImGui::TextColored(HWColors::TextLabel, "VRAM");
        ImGui::TextColored(HWColors::TextPrimary, "%s / %s",
            FormatBytes(gpu.vram_used_bytes).c_str(),
            FormatBytes(gpu.vram_total_bytes).c_str());

        // VRAM usage bar
        float vram_pct = gpu.memory_usage;
        ImGui::ProgressBar(vram_pct, ImVec2(-1, 4), "");
        ImGui::NextColumn();

        // Utilization
        ImGui::TextColored(HWColors::TextLabel, "Utilization");
        ImGui::TextColored(HWColors::TextPrimary, "%.0f%%", gpu.usage_3d * 100.0f);

        // Util bar
        ImGui::ProgressBar(gpu.usage_3d, ImVec2(-1, 4), "");
        ImGui::NextColumn();

        // Temperature
        ImGui::TextColored(HWColors::TextLabel, "Temperature");
        ImVec4 temp_color = gpu.temperature_celsius > 80 ? ImVec4(1, 0.3f, 0.3f, 1) :
                           (gpu.temperature_celsius > 70 ? HWColors::AccentOrange : HWColors::AccentGreen);
        ImGui::TextColored(temp_color, "%.0f C", gpu.temperature_celsius);
        ImGui::NextColumn();

        // Power
        ImGui::TextColored(HWColors::TextLabel, "Power");
        ImGui::TextColored(HWColors::TextPrimary, "%.0f W", gpu.power_watts);
        ImGui::NextColumn();

        ImGui::Columns(1);
    }
    ImGui::EndChild();

    ImGui::PopStyleVar();
    ImGui::PopStyleColor(2);
    ImGui::PopID();
}

void HardwarePanel::RenderCpuSection() {
    RenderSectionHeader("Central Processing Unit", ICON_FA_DESKTOP);

    ImGui::PushStyleColor(ImGuiCol_ChildBg, HWColors::CardBg);

    if (ImGui::BeginChild("CPUCard", ImVec2(0, 100), true)) {
        // CPU Name
        ImGui::PushFont(GetSafeFont(FONT_LARGE));
        ImGui::TextColored(HWColors::TextPrimary, "%s", cpu_info_.name.c_str());
        ImGui::PopFont();

        ImGui::Spacing();

        // Stats in columns
        ImGui::Columns(4, "CPUStats", false);

        ImGui::TextColored(HWColors::TextLabel, "Cores");
        ImGui::TextColored(HWColors::TextPrimary, "%d Physical", cpu_info_.physical_cores);
        ImGui::TextColored(HWColors::TextSecondary, "%d Logical", cpu_info_.logical_cores);
        ImGui::NextColumn();

        ImGui::TextColored(HWColors::TextLabel, "Base Speed");
        ImGui::TextColored(HWColors::TextPrimary, "%.2f GHz", cpu_info_.base_speed_ghz);
        ImGui::NextColumn();

        ImGui::TextColored(HWColors::TextLabel, "Architecture");
        ImGui::TextColored(HWColors::TextPrimary, "%s", cpu_info_.architecture.c_str());
        ImGui::NextColumn();

        ImGui::TextColored(HWColors::TextLabel, "Vendor");
        ImGui::TextColored(HWColors::TextPrimary, "%s",
            cpu_info_.vendor.empty() ? "Unknown" : cpu_info_.vendor.c_str());

        ImGui::Columns(1);
    }
    ImGui::EndChild();

    ImGui::PopStyleColor();
}

void HardwarePanel::RenderMemorySection() {
    RenderSectionHeader("System Memory", ICON_FA_MEMORY);

    ImGui::PushStyleColor(ImGuiCol_ChildBg, HWColors::CardBg);

    if (ImGui::BeginChild("MemoryCard", ImVec2(0, 80), true)) {
        // Stats in columns
        ImGui::Columns(4, "MemStats", false);

        ImGui::TextColored(HWColors::TextLabel, "Total");
        ImGui::PushFont(GetSafeFont(FONT_LARGE));
        ImGui::TextColored(HWColors::TextPrimary, "%s", FormatBytes(memory_info_.total_bytes).c_str());
        ImGui::PopFont();
        ImGui::NextColumn();

        ImGui::TextColored(HWColors::TextLabel, "Used");
        ImGui::PushFont(GetSafeFont(FONT_LARGE));
        ImGui::TextColored(HWColors::TextPrimary, "%s", FormatBytes(memory_info_.used_bytes).c_str());
        ImGui::PopFont();
        ImGui::NextColumn();

        ImGui::TextColored(HWColors::TextLabel, "Available");
        ImGui::PushFont(GetSafeFont(FONT_LARGE));
        ImGui::TextColored(HWColors::AccentGreen, "%s", FormatBytes(memory_info_.available_bytes).c_str());
        ImGui::PopFont();
        ImGui::NextColumn();

        ImGui::TextColored(HWColors::TextLabel, "Usage");
        float usage = memory_info_.total_bytes > 0 ?
            static_cast<float>(memory_info_.used_bytes) / memory_info_.total_bytes : 0.0f;
        ImGui::PushFont(GetSafeFont(FONT_LARGE));
        ImGui::TextColored(HWColors::TextPrimary, "%.1f%%", usage * 100.0f);
        ImGui::PopFont();

        ImGui::Columns(1);

        // Memory usage bar
        ImGui::Spacing();
        ImGui::ProgressBar(usage, ImVec2(-1, 6), "");
    }
    ImGui::EndChild();

    ImGui::PopStyleColor();
}

std::string HardwarePanel::FormatBytes(size_t bytes) const {
    const char* units[] = {"B", "KB", "MB", "GB", "TB"};
    int unit_idx = 0;
    double size = static_cast<double>(bytes);

    while (size >= 1024.0 && unit_idx < 4) {
        size /= 1024.0;
        unit_idx++;
    }

    if (unit_idx == 0) {
        return fmt::format("{} {}", static_cast<int>(size), units[unit_idx]);
    }
    return fmt::format("{:.1f} {}", size, units[unit_idx]);
}

std::string HardwarePanel::FormatSpeed(float ghz) const {
    return fmt::format("{:.2f} GHz", ghz);
}

} // namespace cyxwiz::servernode::gui
