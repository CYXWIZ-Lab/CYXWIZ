#include "memory_monitor.h"
#include "../icons.h"
#include <imgui.h>
#include <implot.h>
#include <spdlog/spdlog.h>
#include <cyxwiz/device.h>
#include <cstdio>
#include <algorithm>

#ifdef _WIN32
#include <windows.h>
#include <psapi.h>
#include <tlhelp32.h>
#pragma comment(lib, "pdh.lib")
#endif

namespace cyxwiz {

MemoryMonitor::MemoryMonitor() {
    cpu_history_.resize(SPARKLINE_POINTS, 0.0f);
    ram_history_.resize(SPARKLINE_POINTS, 0.0f);
    gpu_history_.resize(SPARKLINE_POINTS, 0.0f);
    process_mem_history_.resize(SPARKLINE_POINTS, 0.0f);
    last_update_ = std::chrono::steady_clock::now();

    InitializePDH();
    Update();
}

MemoryMonitor::~MemoryMonitor() {
    CleanupPDH();
}

void MemoryMonitor::InitializePDH() {
#ifdef _WIN32
    PDH_STATUS status = PdhOpenQuery(nullptr, 0, &cpu_query_);
    if (status != ERROR_SUCCESS) {
        spdlog::warn("Failed to open PDH query: {}", status);
        return;
    }

    status = PdhAddEnglishCounterA(cpu_query_, "\\Processor(_Total)\\% Processor Time", 0, &cpu_counter_);
    if (status != ERROR_SUCCESS) {
        spdlog::warn("Failed to add PDH counter: {}", status);
        PdhCloseQuery(cpu_query_);
        cpu_query_ = nullptr;
        return;
    }

    PdhCollectQueryData(cpu_query_);
    pdh_initialized_ = true;

    SYSTEM_INFO sysInfo;
    GetSystemInfo(&sysInfo);
    metrics_.cpu_core_count = sysInfo.dwNumberOfProcessors;
#endif
}

void MemoryMonitor::CleanupPDH() {
#ifdef _WIN32
    if (cpu_query_) {
        PdhCloseQuery(cpu_query_);
        cpu_query_ = nullptr;
    }
    pdh_initialized_ = false;
#endif
}

void MemoryMonitor::Update() {
    auto now = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_update_).count();

    if (elapsed < update_interval_ms_) return;
    last_update_ = now;

    UpdateCPUUsage();
    UpdateRAMUsage();
    UpdateGPUStatus();
    UpdateProcessInfo();
    ShiftHistory();
}

void MemoryMonitor::UpdateCPUUsage() {
#ifdef _WIN32
    if (pdh_initialized_ && cpu_query_) {
        PDH_STATUS status = PdhCollectQueryData(cpu_query_);
        if (status == ERROR_SUCCESS) {
            PDH_FMT_COUNTERVALUE counterVal;
            status = PdhGetFormattedCounterValue(cpu_counter_, PDH_FMT_DOUBLE, nullptr, &counterVal);
            if (status == ERROR_SUCCESS) {
                metrics_.cpu_usage_percent = static_cast<float>(counterVal.doubleValue);
            }
        }
    }
#else
    metrics_.cpu_usage_percent = 0.0f;
#endif
}

void MemoryMonitor::UpdateRAMUsage() {
#ifdef _WIN32
    MEMORYSTATUSEX memInfo;
    memInfo.dwLength = sizeof(MEMORYSTATUSEX);
    if (GlobalMemoryStatusEx(&memInfo)) {
        metrics_.ram_total_bytes = memInfo.ullTotalPhys;
        metrics_.ram_used_bytes = memInfo.ullTotalPhys - memInfo.ullAvailPhys;
        metrics_.ram_usage_percent = static_cast<float>(memInfo.dwMemoryLoad);
    }
#else
    metrics_.ram_total_bytes = 8ULL * 1024 * 1024 * 1024;
    metrics_.ram_used_bytes = 4ULL * 1024 * 1024 * 1024;
    metrics_.ram_usage_percent = 50.0f;
#endif
}

void MemoryMonitor::UpdateGPUStatus() {
    try {
        auto devices = Device::GetAvailableDevices();

        for (const auto& dev : devices) {
            if (dev.type == DeviceType::CUDA || dev.type == DeviceType::OPENCL) {
                metrics_.gpu_available = true;
                metrics_.gpu_name = dev.name;
                metrics_.gpu_vram_total = dev.memory_total;
                metrics_.gpu_vram_used = dev.memory_total - dev.memory_available;
                if (dev.memory_total > 0) {
                    metrics_.gpu_usage_percent = 100.0f *
                        static_cast<float>(metrics_.gpu_vram_used) /
                        static_cast<float>(metrics_.gpu_vram_total);
                }
                return;
            }
        }

        metrics_.gpu_available = false;
        metrics_.gpu_name = "None";
        metrics_.gpu_vram_used = 0;
        metrics_.gpu_vram_total = 0;
        metrics_.gpu_usage_percent = 0.0f;
    } catch (const std::exception& e) {
        spdlog::warn("Failed to query GPU: {}", e.what());
        metrics_.gpu_available = false;
    }
}

void MemoryMonitor::UpdateProcessInfo() {
#ifdef _WIN32
    HANDLE hProcess = GetCurrentProcess();

    PROCESS_MEMORY_COUNTERS_EX pmc;
    if (GetProcessMemoryInfo(hProcess, (PROCESS_MEMORY_COUNTERS*)&pmc, sizeof(pmc))) {
        metrics_.process_memory_bytes = pmc.WorkingSetSize;
    }

    DWORD processId = GetCurrentProcessId();
    HANDLE hSnapshot = CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0);
    if (hSnapshot != INVALID_HANDLE_VALUE) {
        THREADENTRY32 te32;
        te32.dwSize = sizeof(THREADENTRY32);
        int threadCount = 0;
        if (Thread32First(hSnapshot, &te32)) {
            do {
                if (te32.th32OwnerProcessID == processId) {
                    threadCount++;
                }
            } while (Thread32Next(hSnapshot, &te32));
        }
        CloseHandle(hSnapshot);
        metrics_.process_thread_count = threadCount;
    }

    FILETIME createTime, exitTime, kernelTime, userTime;
    if (GetProcessTimes(hProcess, &createTime, &exitTime, &kernelTime, &userTime)) {
        ULONGLONG currentProcessTime =
            (((ULONGLONG)kernelTime.dwHighDateTime << 32) | kernelTime.dwLowDateTime) +
            (((ULONGLONG)userTime.dwHighDateTime << 32) | userTime.dwLowDateTime);

        FILETIME sysIdle, sysKernel, sysUser;
        if (GetSystemTimes(&sysIdle, &sysKernel, &sysUser)) {
            ULONGLONG currentSystemTime =
                (((ULONGLONG)sysKernel.dwHighDateTime << 32) | sysKernel.dwLowDateTime) +
                (((ULONGLONG)sysUser.dwHighDateTime << 32) | sysUser.dwLowDateTime);

            if (last_system_time_ > 0) {
                ULONGLONG processTimeDelta = currentProcessTime - last_process_time_;
                ULONGLONG systemTimeDelta = currentSystemTime - last_system_time_;
                if (systemTimeDelta > 0) {
                    metrics_.process_cpu_percent = 100.0f *
                        static_cast<float>(processTimeDelta) /
                        static_cast<float>(systemTimeDelta);
                }
            }

            last_process_time_ = currentProcessTime;
            last_system_time_ = currentSystemTime;
        }
    }
#else
    metrics_.process_memory_bytes = 512 * 1024 * 1024;
    metrics_.process_cpu_percent = 5.0f;
    metrics_.process_thread_count = 8;
#endif
}

void MemoryMonitor::ShiftHistory() {
    for (size_t i = 0; i < SPARKLINE_POINTS - 1; ++i) {
        cpu_history_[i] = cpu_history_[i + 1];
        ram_history_[i] = ram_history_[i + 1];
        gpu_history_[i] = gpu_history_[i + 1];
        process_mem_history_[i] = process_mem_history_[i + 1];
    }
    cpu_history_[SPARKLINE_POINTS - 1] = metrics_.cpu_usage_percent;
    ram_history_[SPARKLINE_POINTS - 1] = metrics_.ram_usage_percent;
    gpu_history_[SPARKLINE_POINTS - 1] = metrics_.gpu_usage_percent;
    process_mem_history_[SPARKLINE_POINTS - 1] =
        static_cast<float>(metrics_.process_memory_bytes) / (1024.0f * 1024.0f);
}

ImU32 MemoryMonitor::ColorFromPercent(float percent) const {
    if (percent < 50.0f) return IM_COL32(76, 175, 80, 255);   // Green
    if (percent < 75.0f) return IM_COL32(255, 193, 7, 255);   // Yellow/Amber
    if (percent < 90.0f) return IM_COL32(255, 152, 0, 255);   // Orange
    return IM_COL32(244, 67, 54, 255);                         // Red
}

std::string MemoryMonitor::FormatBytes(size_t bytes) const {
    const char* units[] = {"B", "KB", "MB", "GB", "TB"};
    int unit = 0;
    double size = static_cast<double>(bytes);

    while (size >= 1024.0 && unit < 4) {
        size /= 1024.0;
        unit++;
    }

    char buf[32];
    if (unit == 0) {
        snprintf(buf, sizeof(buf), "%zu %s", bytes, units[unit]);
    } else if (size < 10.0) {
        snprintf(buf, sizeof(buf), "%.2f %s", size, units[unit]);
    } else if (size < 100.0) {
        snprintf(buf, sizeof(buf), "%.1f %s", size, units[unit]);
    } else {
        snprintf(buf, sizeof(buf), "%.0f %s", size, units[unit]);
    }
    return buf;
}

void MemoryMonitor::RenderMetricBar(const char* label, float percent, const char* value_text) {
    (void)label;  // Unused but kept for API consistency
    ImU32 color = ColorFromPercent(percent);
    ImVec4 colorVec = ImGui::ColorConvertU32ToFloat4(color);

    ImGui::PushStyleColor(ImGuiCol_PlotHistogram, colorVec);
    ImGui::ProgressBar(percent / 100.0f, ImVec2(-120, 0), "");
    ImGui::PopStyleColor();

    ImGui::SameLine();
    ImGui::TextColored(colorVec, "%5.1f%%", percent);
    ImGui::SameLine();
    ImGui::TextDisabled("%s", value_text);
}

void MemoryMonitor::RenderSparkline(const char* id, const std::vector<float>& data,
                                     float height, ImU32 color) {
    ImVec4 colorVec = ImGui::ColorConvertU32ToFloat4(color);

    if (ImPlot::BeginPlot(id, ImVec2(-1, height),
                          ImPlotFlags_NoTitle | ImPlotFlags_NoLegend |
                          ImPlotFlags_NoMouseText | ImPlotFlags_NoInputs |
                          ImPlotFlags_NoFrame)) {
        ImPlot::SetupAxes(nullptr, nullptr,
                          ImPlotAxisFlags_NoDecorations,
                          ImPlotAxisFlags_NoDecorations | ImPlotAxisFlags_AutoFit);
        ImPlot::SetupAxisLimits(ImAxis_X1, 0, SPARKLINE_POINTS, ImGuiCond_Always);
        ImPlot::SetupAxisLimits(ImAxis_Y1, 0, 100, ImGuiCond_Always);

        ImPlot::PushStyleVar(ImPlotStyleVar_FillAlpha, 0.25f);
        ImPlot::PushStyleColor(ImPlotCol_Line, colorVec);
        ImPlot::PushStyleColor(ImPlotCol_Fill, colorVec);
        ImPlot::PlotShaded("##fill", data.data(), static_cast<int>(data.size()));
        ImPlot::PlotLine("##line", data.data(), static_cast<int>(data.size()));
        ImPlot::PopStyleColor(2);
        ImPlot::PopStyleVar();

        ImPlot::EndPlot();
    }
}

void MemoryMonitor::Render() {
    if (!visible_) return;

    Update();

    ImGui::SetNextWindowSize(ImVec2(420, 480), ImGuiCond_FirstUseEver);

    if (ImGui::Begin(ICON_FA_CHART_LINE " System Monitor", &visible_)) {

        // ============ CPU SECTION ============
        ImGui::TextColored(ImVec4(0.4f, 0.8f, 1.0f, 1.0f),
                          ICON_FA_MICROCHIP " CPU");
        ImGui::SameLine(ImGui::GetWindowWidth() - 100);
        ImGui::TextDisabled("%d cores", metrics_.cpu_core_count);

        ImGui::Separator();

        RenderMetricBar("CPU", metrics_.cpu_usage_percent, "");
        RenderSparkline("##cpu_spark", cpu_history_, 50, ColorFromPercent(metrics_.cpu_usage_percent));

        ImGui::Spacing();

        // ============ RAM SECTION ============
        ImGui::TextColored(ImVec4(0.4f, 0.8f, 1.0f, 1.0f),
                          ICON_FA_MEMORY " Memory");
        ImGui::SameLine(ImGui::GetWindowWidth() - 150);
        ImGui::TextDisabled("%s / %s",
                           FormatBytes(metrics_.ram_used_bytes).c_str(),
                           FormatBytes(metrics_.ram_total_bytes).c_str());

        ImGui::Separator();

        char ramInfo[64];
        snprintf(ramInfo, sizeof(ramInfo), "%s free",
                FormatBytes(metrics_.ram_total_bytes - metrics_.ram_used_bytes).c_str());
        RenderMetricBar("RAM", metrics_.ram_usage_percent, ramInfo);
        RenderSparkline("##ram_spark", ram_history_, 50, ColorFromPercent(metrics_.ram_usage_percent));

        ImGui::Spacing();

        // ============ GPU SECTION ============
        if (metrics_.gpu_available) {
            ImGui::TextColored(ImVec4(0.4f, 0.8f, 1.0f, 1.0f),
                              ICON_FA_DESKTOP " GPU");
            ImGui::SameLine(ImGui::GetWindowWidth() - 200);
            ImGui::TextDisabled("%s", metrics_.gpu_name.c_str());

            ImGui::Separator();

            char gpuInfo[64];
            snprintf(gpuInfo, sizeof(gpuInfo), "%s / %s",
                    FormatBytes(metrics_.gpu_vram_used).c_str(),
                    FormatBytes(metrics_.gpu_vram_total).c_str());
            RenderMetricBar("VRAM", metrics_.gpu_usage_percent, gpuInfo);
            RenderSparkline("##gpu_spark", gpu_history_, 50, ColorFromPercent(metrics_.gpu_usage_percent));
        } else {
            ImGui::TextDisabled(ICON_FA_DESKTOP " GPU: Not available");
            ImGui::TextDisabled("(Requires CUDA or OpenCL device)");
        }

        ImGui::Spacing();

        // ============ PROCESS SECTION ============
        ImGui::TextColored(ImVec4(0.4f, 0.8f, 1.0f, 1.0f),
                          ICON_FA_TERMINAL " CyxWiz Process");

        ImGui::Separator();

        ImGui::Text(ICON_FA_DATABASE " Memory: %s",
                   FormatBytes(metrics_.process_memory_bytes).c_str());
        ImGui::SameLine(200);
        ImGui::Text(ICON_FA_BOLT " CPU: %.1f%%", metrics_.process_cpu_percent);
        ImGui::SameLine(320);
        ImGui::Text(ICON_FA_LAYER_GROUP " Threads: %d", metrics_.process_thread_count);

        float maxProcessMem = 0;
        for (float v : process_mem_history_) maxProcessMem = std::max(maxProcessMem, v);
        maxProcessMem = std::max(maxProcessMem * 1.2f, 100.0f);

        if (ImPlot::BeginPlot("##proc_mem", ImVec2(-1, 40),
                              ImPlotFlags_NoTitle | ImPlotFlags_NoLegend |
                              ImPlotFlags_NoMouseText | ImPlotFlags_NoInputs |
                              ImPlotFlags_NoFrame)) {
            ImPlot::SetupAxes(nullptr, nullptr,
                              ImPlotAxisFlags_NoDecorations,
                              ImPlotAxisFlags_NoDecorations);
            ImPlot::SetupAxisLimits(ImAxis_X1, 0, SPARKLINE_POINTS, ImGuiCond_Always);
            ImPlot::SetupAxisLimits(ImAxis_Y1, 0, maxProcessMem, ImGuiCond_Always);

            ImPlot::PushStyleVar(ImPlotStyleVar_FillAlpha, 0.25f);
            ImPlot::PushStyleColor(ImPlotCol_Line, ImVec4(0.5f, 0.5f, 1.0f, 1.0f));
            ImPlot::PushStyleColor(ImPlotCol_Fill, ImVec4(0.5f, 0.5f, 1.0f, 0.3f));
            ImPlot::PlotShaded("##fill", process_mem_history_.data(),
                              static_cast<int>(process_mem_history_.size()));
            ImPlot::PlotLine("##line", process_mem_history_.data(),
                            static_cast<int>(process_mem_history_.size()));
            ImPlot::PopStyleColor(2);
            ImPlot::PopStyleVar();

            ImPlot::EndPlot();
        }

        ImGui::Spacing();
        ImGui::Separator();

        // ============ FOOTER ============
        ImGui::TextDisabled("Update interval: %.0fms", update_interval_ms_);
        ImGui::SameLine(ImGui::GetWindowWidth() - 140);
        if (ImGui::SmallButton(ICON_FA_ARROWS_ROTATE " Refresh")) {
            last_update_ = std::chrono::steady_clock::time_point{};
            Update();
        }
    }
    ImGui::End();
}

} // namespace cyxwiz
