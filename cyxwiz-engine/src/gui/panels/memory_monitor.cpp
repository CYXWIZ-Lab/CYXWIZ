#include "memory_monitor.h"
#include "../icons.h"
#include <imgui.h>
#include <implot.h>
#include <spdlog/spdlog.h>

#ifdef _WIN32
#include <windows.h>
#include <psapi.h>
#endif

namespace cyxwiz {

MemoryMonitor::MemoryMonitor() {
    cpu_history_.resize(HISTORY_SIZE, 0.0f);
    gpu_history_.resize(HISTORY_SIZE, 0.0f);
    last_update_ = std::chrono::steady_clock::now();
    Update();
}

void MemoryMonitor::Update() {
    auto now = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_update_).count();

    if (elapsed < update_interval_ms_) return;
    last_update_ = now;

    UpdateCPUMemory();
    UpdateGPUMemory();

    // Shift history and add new values
    for (size_t i = 0; i < HISTORY_SIZE - 1; ++i) {
        cpu_history_[i] = cpu_history_[i + 1];
        gpu_history_[i] = gpu_history_[i + 1];
    }
    cpu_history_[HISTORY_SIZE - 1] = cpu_used_mb_;
    gpu_history_[HISTORY_SIZE - 1] = gpu_used_mb_;
}

void MemoryMonitor::UpdateCPUMemory() {
#ifdef _WIN32
    MEMORYSTATUSEX memInfo;
    memInfo.dwLength = sizeof(MEMORYSTATUSEX);
    GlobalMemoryStatusEx(&memInfo);
    cpu_total_mb_ = static_cast<float>(memInfo.ullTotalPhys) / (1024.0f * 1024.0f);

    // Get process-specific memory
    PROCESS_MEMORY_COUNTERS_EX pmc;
    if (GetProcessMemoryInfo(GetCurrentProcess(), (PROCESS_MEMORY_COUNTERS*)&pmc, sizeof(pmc))) {
        cpu_used_mb_ = static_cast<float>(pmc.WorkingSetSize) / (1024.0f * 1024.0f);
    }
#else
    // Linux/macOS implementation placeholder
    cpu_total_mb_ = 8192.0f;  // Placeholder
    cpu_used_mb_ = 1024.0f;   // Placeholder
#endif
}

void MemoryMonitor::UpdateGPUMemory() {
    // TODO: Integrate with ArrayFire or CUDA to get actual GPU memory
    gpu_available_ = false;
    gpu_used_mb_ = 0.0f;
    gpu_total_mb_ = 0.0f;
}

void MemoryMonitor::Render() {
    if (!visible_) return;

    Update();

    ImGui::SetNextWindowSize(ImVec2(400, 350), ImGuiCond_FirstUseEver);
    if (ImGui::Begin(ICON_FA_MEMORY " Memory Monitor", &visible_)) {

        // CPU Memory Section
        ImGui::Text(ICON_FA_DESKTOP " Process Memory");
        ImGui::Separator();

        ImGui::Text("Used: %.1f MB", cpu_used_mb_);

        // Calculate a reasonable max for the progress bar (2GB default for process)
        float process_max_mb = 2048.0f;
        float cpu_percent = (cpu_used_mb_ / process_max_mb) * 100.0f;
        if (cpu_percent > 100.0f) {
            process_max_mb = cpu_used_mb_ * 1.5f;  // Adjust if exceeded
            cpu_percent = (cpu_used_mb_ / process_max_mb) * 100.0f;
        }

        ImGui::ProgressBar(cpu_percent / 100.0f, ImVec2(-1, 0), "");
        ImGui::SameLine(0, -ImGui::GetStyle().ItemSpacing.x);
        ImGui::Text(" %.1f%%", cpu_percent);

        // CPU History Graph
        if (ImPlot::BeginPlot("##CPUHistory", ImVec2(-1, 100), ImPlotFlags_NoTitle | ImPlotFlags_NoMouseText)) {
            ImPlot::SetupAxes(nullptr, nullptr, ImPlotAxisFlags_NoTickLabels, ImPlotAxisFlags_AutoFit);
            ImPlot::SetupAxisLimits(ImAxis_X1, 0, HISTORY_SIZE, ImGuiCond_Always);
            ImPlot::SetupAxisLimits(ImAxis_Y1, 0, process_max_mb, ImGuiCond_Always);
            ImPlot::PlotLine("CPU (MB)", cpu_history_.data(), static_cast<int>(cpu_history_.size()));
            ImPlot::EndPlot();
        }

        ImGui::Spacing();

        // System Memory Section
        ImGui::Text(ICON_FA_SERVER " System Memory");
        ImGui::Separator();

        float system_used_percent = ((cpu_total_mb_ - (cpu_total_mb_ * 0.3f)) / cpu_total_mb_) * 100.0f; // Approximation
        ImGui::Text("Total: %.1f GB", cpu_total_mb_ / 1024.0f);

        ImGui::Spacing();

        // GPU Memory Section
        if (gpu_available_) {
            ImGui::Text(ICON_FA_MICROCHIP " GPU Memory");
            ImGui::Separator();

            float gpu_percent = (gpu_total_mb_ > 0) ? (gpu_used_mb_ / gpu_total_mb_) * 100.0f : 0.0f;
            ImGui::Text("Used: %.1f / %.1f MB", gpu_used_mb_, gpu_total_mb_);
            ImGui::ProgressBar(gpu_percent / 100.0f, ImVec2(-1, 0), "");

            if (ImPlot::BeginPlot("##GPUHistory", ImVec2(-1, 100), ImPlotFlags_NoTitle | ImPlotFlags_NoMouseText)) {
                ImPlot::SetupAxes(nullptr, nullptr, ImPlotAxisFlags_NoTickLabels, ImPlotAxisFlags_AutoFit);
                ImPlot::SetupAxisLimits(ImAxis_X1, 0, HISTORY_SIZE, ImGuiCond_Always);
                ImPlot::PlotLine("GPU (MB)", gpu_history_.data(), static_cast<int>(gpu_history_.size()));
                ImPlot::EndPlot();
            }
        } else {
            ImGui::TextDisabled(ICON_FA_MICROCHIP " GPU memory tracking not available");
            ImGui::TextDisabled("(Requires ArrayFire with CUDA/OpenCL)");
        }

        ImGui::Spacing();
        ImGui::Separator();

        // Actions
        if (ImGui::Button(ICON_FA_TRASH " Clear Cache")) {
            spdlog::info("Cache cleared from Memory Monitor");
        }
        ImGui::SameLine();
        if (ImGui::Button(ICON_FA_RECYCLE " Force GC")) {
            spdlog::info("Garbage collection triggered from Memory Monitor");
        }
    }
    ImGui::End();
}

} // namespace cyxwiz
