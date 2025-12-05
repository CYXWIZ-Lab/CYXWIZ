#include "memory_panel.h"
#include "../icons.h"
#include <imgui.h>
#include <imgui_internal.h>
#include <implot.h>
#include <spdlog/spdlog.h>
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <numeric>

namespace cyxwiz {

MemoryPanel::MemoryPanel() : Panel("Memory Monitor", false) {
    last_refresh_ = std::chrono::steady_clock::now();
    last_gpu_update_ = std::chrono::steady_clock::now();
}

void MemoryPanel::Render() {
    if (!visible_) return;

    // Auto-refresh GPU status
    if (auto_refresh_) {
        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_gpu_update_).count();
        if (elapsed >= refresh_interval_ms_) {
            UpdateGPUStatus();
            last_gpu_update_ = now;
        }
    }

    ImGui::SetNextWindowSize(ImVec2(700, 500), ImGuiCond_FirstUseEver);
    if (!ImGui::Begin(GetName(), &visible_)) {
        ImGui::End();
        return;
    }

    RenderToolbar();

    // View mode tabs
    if (ImGui::BeginTabBar("MemoryTabs")) {
        if (ImGui::BeginTabItem(ICON_FA_CHART_AREA " Overview")) {
            view_mode_ = 0;
            RenderOverviewChart();
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem(ICON_FA_LAYER_GROUP " Layer Breakdown")) {
            view_mode_ = 1;
            RenderLayerBreakdown();
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem(ICON_FA_MICROCHIP " GPU Details")) {
            view_mode_ = 2;
            RenderGPUDetails();
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem(ICON_FA_CIRCLE_INFO " Statistics")) {
            RenderMemoryStats();
            ImGui::EndTabItem();
        }
        ImGui::EndTabBar();
    }

    ImGui::End();
}

void MemoryPanel::RenderToolbar() {
    // Status indicator
    if (is_monitoring_) {
        ImGui::TextColored(ImVec4(0.3f, 0.8f, 0.3f, 1.0f), ICON_FA_CIRCLE " Monitoring");
    } else {
        ImGui::TextColored(ImVec4(0.5f, 0.5f, 0.5f, 1.0f), ICON_FA_CIRCLE " Idle");
    }

    ImGui::SameLine();
    ImGui::SeparatorEx(ImGuiSeparatorFlags_Vertical);
    ImGui::SameLine();

    // Clear button
    if (ImGui::Button(ICON_FA_TRASH " Clear")) {
        Clear();
    }

    ImGui::SameLine();

    // Export button
    if (ImGui::Button(ICON_FA_DOWNLOAD " Export")) {
        std::string filename = "memory_data.csv";
        if (ExportToCSV(filename)) {
            spdlog::info("Memory data exported to {}", filename);
        }
    }

    ImGui::SameLine();
    ImGui::SeparatorEx(ImGuiSeparatorFlags_Vertical);
    ImGui::SameLine();

    // Auto-refresh toggle
    ImGui::Checkbox("Auto-refresh", &auto_refresh_);

    ImGui::SameLine();

    // Refresh button (manual)
    if (ImGui::Button(ICON_FA_ARROWS_ROTATE)) {
        UpdateGPUStatus();
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Refresh GPU status now");
    }

    ImGui::SameLine();
    ImGui::SeparatorEx(ImGuiSeparatorFlags_Vertical);
    ImGui::SameLine();

    // Current memory status
    std::lock_guard<std::mutex> lock(data_mutex_);
    if (!history_.empty()) {
        const auto& latest = history_.back();
        ImGui::Text("CPU: %s | GPU: %s",
            FormatBytes(latest.cpu_heap_bytes + latest.cpu_tensors_bytes).c_str(),
            FormatBytes(latest.gpu_allocated_bytes).c_str());
    } else {
        ImGui::TextDisabled("No data");
    }

    ImGui::Separator();
}

void MemoryPanel::RenderOverviewChart() {
    std::lock_guard<std::mutex> lock(data_mutex_);

    if (history_.size() < 2) {
        ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f), "Collecting memory data...");
        ImGui::TextDisabled("Start training to see memory usage over time.");
        return;
    }

    // Chart options
    ImGui::Checkbox("CPU Heap", &show_cpu_heap_);
    ImGui::SameLine();
    ImGui::Checkbox("CPU Tensors", &show_cpu_tensors_);
    ImGui::SameLine();
    ImGui::Checkbox("GPU Allocated", &show_gpu_allocated_);
    ImGui::SameLine();
    ImGui::Checkbox("GPU Cached", &show_gpu_cached_);

    ImGui::SameLine();
    ImGui::SetNextItemWidth(100);
    ImGui::SliderFloat("Window (s)", &chart_time_window_, 10.0f, 300.0f);

    // Prepare data for plotting
    std::vector<double> timestamps;
    std::vector<double> cpu_heap;
    std::vector<double> cpu_tensors;
    std::vector<double> gpu_allocated;
    std::vector<double> gpu_cached;

    double latest_time = history_.back().timestamp;
    double start_time = latest_time - chart_time_window_;

    for (const auto& snapshot : history_) {
        if (snapshot.timestamp >= start_time) {
            timestamps.push_back(snapshot.timestamp);
            cpu_heap.push_back(snapshot.cpu_heap_bytes / (1024.0 * 1024.0));  // Convert to MB
            cpu_tensors.push_back(snapshot.cpu_tensors_bytes / (1024.0 * 1024.0));
            gpu_allocated.push_back(snapshot.gpu_allocated_bytes / (1024.0 * 1024.0));
            gpu_cached.push_back(snapshot.gpu_cached_bytes / (1024.0 * 1024.0));
        }
    }

    if (timestamps.empty()) {
        ImGui::TextDisabled("No data in selected time window");
        return;
    }

    // Plot
    if (ImPlot::BeginPlot("Memory Usage", ImVec2(-1, 300))) {
        ImPlot::SetupAxes("Time (s)", "Memory (MB)");
        ImPlot::SetupAxisLimits(ImAxis_X1, start_time, latest_time, ImGuiCond_Always);

        if (show_cpu_heap_ && !cpu_heap.empty()) {
            ImPlot::SetNextLineStyle(ImVec4(cpu_heap_color_.x, cpu_heap_color_.y, cpu_heap_color_.z, 1.0f));
            ImPlot::SetNextFillStyle(ImVec4(cpu_heap_color_.x, cpu_heap_color_.y, cpu_heap_color_.z, 0.3f));
            ImPlot::PlotShaded("CPU Heap", timestamps.data(), cpu_heap.data(), static_cast<int>(timestamps.size()), 0.0);
        }

        if (show_cpu_tensors_ && !cpu_tensors.empty()) {
            ImPlot::SetNextLineStyle(ImVec4(cpu_tensor_color_.x, cpu_tensor_color_.y, cpu_tensor_color_.z, 1.0f));
            ImPlot::PlotLine("CPU Tensors", timestamps.data(), cpu_tensors.data(), static_cast<int>(timestamps.size()));
        }

        if (show_gpu_allocated_ && !gpu_allocated.empty()) {
            ImPlot::SetNextLineStyle(ImVec4(gpu_allocated_color_.x, gpu_allocated_color_.y, gpu_allocated_color_.z, 1.0f));
            ImPlot::SetNextFillStyle(ImVec4(gpu_allocated_color_.x, gpu_allocated_color_.y, gpu_allocated_color_.z, 0.3f));
            ImPlot::PlotShaded("GPU Allocated", timestamps.data(), gpu_allocated.data(), static_cast<int>(timestamps.size()), 0.0);
        }

        if (show_gpu_cached_ && !gpu_cached.empty()) {
            ImPlot::SetNextLineStyle(ImVec4(gpu_cached_color_.x, gpu_cached_color_.y, gpu_cached_color_.z, 1.0f));
            ImPlot::PlotLine("GPU Cached", timestamps.data(), gpu_cached.data(), static_cast<int>(timestamps.size()));
        }

        ImPlot::EndPlot();
    }

    // Legend with current values
    ImGui::Separator();
    const auto& latest = history_.back();

    ImGui::Columns(4, nullptr, false);
    ImGui::ColorButton("##cpu_heap", cpu_heap_color_, ImGuiColorEditFlags_NoTooltip, ImVec2(12, 12));
    ImGui::SameLine();
    ImGui::Text("CPU Heap: %s", FormatBytes(latest.cpu_heap_bytes).c_str());
    ImGui::NextColumn();

    ImGui::ColorButton("##cpu_tensor", cpu_tensor_color_, ImGuiColorEditFlags_NoTooltip, ImVec2(12, 12));
    ImGui::SameLine();
    ImGui::Text("CPU Tensors: %s", FormatBytes(latest.cpu_tensors_bytes).c_str());
    ImGui::NextColumn();

    ImGui::ColorButton("##gpu_alloc", gpu_allocated_color_, ImGuiColorEditFlags_NoTooltip, ImVec2(12, 12));
    ImGui::SameLine();
    ImGui::Text("GPU Allocated: %s", FormatBytes(latest.gpu_allocated_bytes).c_str());
    ImGui::NextColumn();

    ImGui::ColorButton("##gpu_cache", gpu_cached_color_, ImGuiColorEditFlags_NoTooltip, ImVec2(12, 12));
    ImGui::SameLine();
    ImGui::Text("GPU Cached: %s", FormatBytes(latest.gpu_cached_bytes).c_str());

    ImGui::Columns(1);
}

void MemoryPanel::RenderLayerBreakdown() {
    std::lock_guard<std::mutex> lock(data_mutex_);

    if (history_.empty()) {
        ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f), "No layer memory data available.");
        return;
    }

    const auto& latest = history_.back();
    if (latest.per_layer_memory.empty()) {
        ImGui::TextDisabled("No per-layer memory data recorded.");
        ImGui::TextDisabled("Layer memory tracking is enabled during training.");
        return;
    }

    // Options
    ImGui::RadioButton("Bar Chart", &breakdown_mode_, 0);
    ImGui::SameLine();
    ImGui::RadioButton("Table", &breakdown_mode_, 1);
    ImGui::SameLine();
    ImGui::SetNextItemWidth(100);
    ImGui::SliderInt("Top N", &top_n_layers_, 5, 20);

    // Sort layers by memory usage
    std::vector<std::pair<std::string, size_t>> sorted_layers(
        latest.per_layer_memory.begin(), latest.per_layer_memory.end());

    std::sort(sorted_layers.begin(), sorted_layers.end(),
        [](const auto& a, const auto& b) { return a.second > b.second; });

    // Limit to top N
    if (static_cast<int>(sorted_layers.size()) > top_n_layers_) {
        sorted_layers.resize(top_n_layers_);
    }

    // Calculate total
    size_t total_memory = 0;
    for (const auto& [name, bytes] : latest.per_layer_memory) {
        total_memory += bytes;
    }

    if (breakdown_mode_ == 0) {
        // Bar chart
        std::vector<const char*> labels;
        std::vector<double> values;
        for (const auto& [name, bytes] : sorted_layers) {
            labels.push_back(name.c_str());
            values.push_back(bytes / (1024.0 * 1024.0));  // MB
        }

        if (ImPlot::BeginPlot("Layer Memory (MB)", ImVec2(-1, 300))) {
            ImPlot::SetupAxes(nullptr, "Memory (MB)", ImPlotAxisFlags_AutoFit, ImPlotAxisFlags_AutoFit);

            ImPlot::SetNextFillStyle(ImVec4(0.3f, 0.6f, 0.9f, 0.8f));
            ImPlot::PlotBars("Memory", values.data(), static_cast<int>(values.size()), 0.5);

            // Add labels
            for (size_t i = 0; i < labels.size(); i++) {
                ImPlot::Annotation(static_cast<double>(i), values[i], ImVec4(1, 1, 1, 1), ImVec2(0, -5), true, "%s", labels[i]);
            }

            ImPlot::EndPlot();
        }
    } else {
        // Table view
        if (ImGui::BeginTable("LayerMemoryTable", 4,
            ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg | ImGuiTableFlags_ScrollY,
            ImVec2(0, 300))) {

            ImGui::TableSetupScrollFreeze(0, 1);
            ImGui::TableSetupColumn("Layer", ImGuiTableColumnFlags_None);
            ImGui::TableSetupColumn("Memory", ImGuiTableColumnFlags_None);
            ImGui::TableSetupColumn("% of Total", ImGuiTableColumnFlags_None);
            ImGui::TableSetupColumn("Bar", ImGuiTableColumnFlags_None);
            ImGui::TableHeadersRow();

            for (const auto& [name, bytes] : sorted_layers) {
                ImGui::TableNextRow();

                double percentage = total_memory > 0 ? (static_cast<double>(bytes) / total_memory) * 100.0 : 0.0;

                ImGui::TableNextColumn();
                if (ImGui::Selectable(name.c_str(), false, ImGuiSelectableFlags_SpanAllColumns)) {
                    if (layer_click_callback_) {
                        layer_click_callback_(name);
                    }
                }

                ImGui::TableNextColumn();
                ImGui::Text("%s", FormatBytes(bytes).c_str());

                ImGui::TableNextColumn();
                ImGui::Text("%.1f%%", percentage);

                ImGui::TableNextColumn();
                ImGui::ProgressBar(static_cast<float>(percentage / 100.0), ImVec2(-1, 0), "");
            }

            ImGui::EndTable();
        }
    }

    // Summary
    ImGui::Separator();
    ImGui::Text("Total layer memory: %s across %zu layers",
        FormatBytes(total_memory).c_str(), latest.per_layer_memory.size());
}

void MemoryPanel::RenderGPUDetails() {
    ImGui::Text(ICON_FA_MICROCHIP " GPU Information");
    ImGui::Separator();

    if (gpu_info_.device_id < 0) {
        ImGui::TextColored(ImVec4(0.8f, 0.5f, 0.2f, 1.0f), "No GPU detected or GPU backend not available.");
        ImGui::TextDisabled("Ensure ArrayFire is initialized with CUDA or OpenCL backend.");

        if (ImGui::Button("Refresh GPU Status")) {
            UpdateGPUStatus();
        }
        return;
    }

    // GPU info display
    ImGui::BeginChild("GPUInfo", ImVec2(0, 200), true);

    ImGui::Text("Device: %s", gpu_info_.name.c_str());
    ImGui::Text("Backend: %s", gpu_info_.backend.c_str());
    ImGui::Text("Device ID: %d", gpu_info_.device_id);

    ImGui::Separator();

    // Memory bar
    size_t used_memory = gpu_info_.total_memory - gpu_info_.free_memory;
    float memory_usage = gpu_info_.total_memory > 0
        ? static_cast<float>(used_memory) / gpu_info_.total_memory
        : 0.0f;

    ImGui::Text("Memory Usage:");
    ImGui::SameLine();
    ImVec4 mem_color = GetMemoryColor(memory_usage * 100);
    ImGui::PushStyleColor(ImGuiCol_PlotHistogram, mem_color);
    ImGui::ProgressBar(memory_usage, ImVec2(-1, 20), "");
    ImGui::PopStyleColor();

    ImGui::Text("Used: %s / %s (%.1f%%)",
        FormatBytes(used_memory).c_str(),
        FormatBytes(gpu_info_.total_memory).c_str(),
        memory_usage * 100);
    ImGui::Text("Free: %s", FormatBytes(gpu_info_.free_memory).c_str());

    ImGui::Separator();

    // Utilization (if available)
    if (gpu_info_.utilization >= 0) {
        ImGui::Text("GPU Utilization:");
        ImVec4 util_color = GetMemoryColor(gpu_info_.utilization);
        ImGui::PushStyleColor(ImGuiCol_PlotHistogram, util_color);
        ImGui::ProgressBar(gpu_info_.utilization / 100.0f, ImVec2(-1, 20), "");
        ImGui::PopStyleColor();
        ImGui::Text("%.1f%%", gpu_info_.utilization);
    }

    // Temperature (if available)
    if (gpu_info_.temperature > 0) {
        ImGui::Text("Temperature: %.1f C", gpu_info_.temperature);
    }

    ImGui::EndChild();

    // Memory history chart for GPU only
    std::lock_guard<std::mutex> lock(data_mutex_);
    if (history_.size() >= 2) {
        std::vector<double> timestamps;
        std::vector<double> gpu_used;

        double latest_time = history_.back().timestamp;
        double start_time = latest_time - chart_time_window_;

        for (const auto& snapshot : history_) {
            if (snapshot.timestamp >= start_time) {
                timestamps.push_back(snapshot.timestamp);
                gpu_used.push_back(snapshot.gpu_allocated_bytes / (1024.0 * 1024.0));
            }
        }

        if (!timestamps.empty() && ImPlot::BeginPlot("GPU Memory Over Time", ImVec2(-1, 200))) {
            ImPlot::SetupAxes("Time (s)", "Memory (MB)");
            ImPlot::SetupAxisLimits(ImAxis_X1, start_time, latest_time, ImGuiCond_Always);

            ImPlot::SetNextLineStyle(ImVec4(gpu_allocated_color_.x, gpu_allocated_color_.y, gpu_allocated_color_.z, 1.0f));
            ImPlot::SetNextFillStyle(ImVec4(gpu_allocated_color_.x, gpu_allocated_color_.y, gpu_allocated_color_.z, 0.3f));
            ImPlot::PlotShaded("GPU Memory", timestamps.data(), gpu_used.data(), static_cast<int>(timestamps.size()), 0.0);

            ImPlot::EndPlot();
        }
    }
}

void MemoryPanel::RenderMemoryStats() {
    std::lock_guard<std::mutex> lock(data_mutex_);

    if (history_.empty()) {
        ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f), "No memory data collected yet.");
        return;
    }

    ImGui::Text("Statistics over %zu snapshots:", history_.size());
    ImGui::Separator();

    // Calculate statistics
    std::vector<size_t> cpu_totals, gpu_totals;
    for (const auto& s : history_) {
        cpu_totals.push_back(s.cpu_heap_bytes + s.cpu_tensors_bytes);
        gpu_totals.push_back(s.gpu_allocated_bytes);
    }

    auto calc_stats = [](const std::vector<size_t>& data) {
        struct Stats { size_t min, max, avg; };
        if (data.empty()) return Stats{0, 0, 0};
        size_t sum = std::accumulate(data.begin(), data.end(), size_t(0));
        return Stats{
            *std::min_element(data.begin(), data.end()),
            *std::max_element(data.begin(), data.end()),
            sum / data.size()
        };
    };

    auto cpu_stats = calc_stats(cpu_totals);
    auto gpu_stats = calc_stats(gpu_totals);

    // CPU Statistics
    ImGui::Text(ICON_FA_MICROCHIP " CPU Memory:");
    ImGui::Indent();
    ImGui::BulletText("Minimum: %s", FormatBytes(cpu_stats.min).c_str());
    ImGui::BulletText("Maximum: %s", FormatBytes(cpu_stats.max).c_str());
    ImGui::BulletText("Average: %s", FormatBytes(cpu_stats.avg).c_str());
    ImGui::BulletText("Current: %s", FormatBytes(cpu_totals.back()).c_str());
    ImGui::Unindent();

    ImGui::Separator();

    // GPU Statistics
    ImGui::Text(ICON_FA_CUBE " GPU Memory:");
    ImGui::Indent();
    ImGui::BulletText("Minimum: %s", FormatBytes(gpu_stats.min).c_str());
    ImGui::BulletText("Maximum: %s", FormatBytes(gpu_stats.max).c_str());
    ImGui::BulletText("Average: %s", FormatBytes(gpu_stats.avg).c_str());
    ImGui::BulletText("Current: %s", FormatBytes(gpu_totals.back()).c_str());
    if (gpu_info_.total_memory > 0) {
        float peak_usage = (static_cast<float>(gpu_stats.max) / gpu_info_.total_memory) * 100.0f;
        ImGui::BulletText("Peak Usage: %.1f%% of total", peak_usage);
    }
    ImGui::Unindent();

    ImGui::Separator();

    // Time info
    double duration = history_.back().timestamp - history_.front().timestamp;
    ImGui::Text("Monitoring Duration: %.1f seconds", duration);
    ImGui::Text("Snapshot Rate: %.1f Hz", history_.size() / std::max(duration, 0.001));
}

void MemoryPanel::BeginMonitoring() {
    std::lock_guard<std::mutex> lock(data_mutex_);
    is_monitoring_ = true;
    monitoring_start_time_ = std::chrono::steady_clock::now();
    current_snapshot_ = MemorySnapshot();
}

void MemoryPanel::EndMonitoring() {
    std::lock_guard<std::mutex> lock(data_mutex_);
    is_monitoring_ = false;
}

void MemoryPanel::RecordSnapshot(size_t cpu_heap, size_t cpu_tensors, size_t gpu_allocated, size_t gpu_cached) {
    std::lock_guard<std::mutex> lock(data_mutex_);
    if (!is_monitoring_) return;

    current_snapshot_.cpu_heap_bytes = cpu_heap;
    current_snapshot_.cpu_tensors_bytes = cpu_tensors;
    current_snapshot_.gpu_allocated_bytes = gpu_allocated;
    current_snapshot_.gpu_cached_bytes = gpu_cached;

    auto now = std::chrono::steady_clock::now();
    current_snapshot_.timestamp = std::chrono::duration<double>(now - monitoring_start_time_).count();
}

void MemoryPanel::RecordLayerMemory(const std::string& layer_name, size_t bytes) {
    std::lock_guard<std::mutex> lock(data_mutex_);
    if (!is_monitoring_) return;

    current_snapshot_.per_layer_memory[layer_name] = bytes;
}

void MemoryPanel::FinalizeSnapshot(int epoch, int step) {
    std::lock_guard<std::mutex> lock(data_mutex_);
    if (!is_monitoring_) return;

    current_snapshot_.epoch = epoch;
    current_snapshot_.step = step;

    history_.push_back(current_snapshot_);

    // Trim history if too large
    if (history_.size() > kMaxHistorySize) {
        history_.erase(history_.begin(), history_.begin() + (history_.size() - kMaxHistorySize));
    }

    // Reset for next snapshot
    current_snapshot_ = MemorySnapshot();
}

void MemoryPanel::UpdateGPUStatus() {
    // Try to get GPU info from ArrayFire if available
    // This is a placeholder - actual implementation depends on ArrayFire availability

    // For now, provide basic info that the GPU backend would fill in
    // In a real implementation, this would call af::info() or similar

    gpu_info_.device_id = 0;  // Placeholder
    gpu_info_.name = "GPU (Backend not queried)";
    gpu_info_.backend = "Unknown";
    gpu_info_.total_memory = 0;
    gpu_info_.free_memory = 0;
    gpu_info_.utilization = -1;  // Unknown
    gpu_info_.temperature = 0;   // Unknown

    // TODO: Query actual GPU info from ArrayFire
    // #ifdef CYXWIZ_HAS_ARRAYFIRE
    //   af::deviceInfo(gpu_info_.name, ...);
    //   gpu_info_.total_memory = af::getDeviceMemorySize();
    //   gpu_info_.free_memory = gpu_info_.total_memory - af::getAllocatedBytes();
    // #endif
}

void MemoryPanel::Clear() {
    std::lock_guard<std::mutex> lock(data_mutex_);
    history_.clear();
    current_snapshot_ = MemorySnapshot();
}

bool MemoryPanel::ExportToCSV(const std::string& path) {
    std::lock_guard<std::mutex> lock(data_mutex_);

    std::ofstream file(path);
    if (!file.is_open()) {
        spdlog::error("Failed to open file for export: {}", path);
        return false;
    }

    // Write header
    file << "Timestamp,Epoch,Step,CPU_Heap_Bytes,CPU_Tensors_Bytes,GPU_Allocated_Bytes,GPU_Cached_Bytes\n";

    // Write data
    for (const auto& snapshot : history_) {
        file << std::fixed << std::setprecision(3) << snapshot.timestamp << ","
             << snapshot.epoch << ","
             << snapshot.step << ","
             << snapshot.cpu_heap_bytes << ","
             << snapshot.cpu_tensors_bytes << ","
             << snapshot.gpu_allocated_bytes << ","
             << snapshot.gpu_cached_bytes << "\n";
    }

    file.close();
    return true;
}

std::string MemoryPanel::FormatBytes(size_t bytes) const {
    if (bytes < 1024) {
        return std::to_string(bytes) + " B";
    } else if (bytes < 1024 * 1024) {
        std::ostringstream ss;
        ss << std::fixed << std::setprecision(1) << (bytes / 1024.0) << " KB";
        return ss.str();
    } else if (bytes < 1024 * 1024 * 1024) {
        std::ostringstream ss;
        ss << std::fixed << std::setprecision(2) << (bytes / (1024.0 * 1024.0)) << " MB";
        return ss.str();
    } else {
        std::ostringstream ss;
        ss << std::fixed << std::setprecision(2) << (bytes / (1024.0 * 1024.0 * 1024.0)) << " GB";
        return ss.str();
    }
}

ImVec4 MemoryPanel::GetMemoryColor(double usage_percentage) const {
    // Green -> Yellow -> Red based on usage
    if (usage_percentage < 50) {
        return ImVec4(0.2f, 0.8f, 0.2f, 1.0f);  // Green
    } else if (usage_percentage < 75) {
        float t = (static_cast<float>(usage_percentage) - 50.0f) / 25.0f;
        return ImVec4(0.2f + 0.8f * t, 0.8f, 0.2f, 1.0f);  // Green to Yellow
    } else if (usage_percentage < 90) {
        float t = (static_cast<float>(usage_percentage) - 75.0f) / 15.0f;
        return ImVec4(1.0f, 0.8f - 0.6f * t, 0.2f, 1.0f);  // Yellow to Orange
    } else {
        return ImVec4(1.0f, 0.2f, 0.2f, 1.0f);  // Red
    }
}

} // namespace cyxwiz
