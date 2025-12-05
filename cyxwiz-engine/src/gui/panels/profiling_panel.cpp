#include "profiling_panel.h"
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

ProfilingPanel::ProfilingPanel() : Panel("Performance Profiler", false) {
}

void ProfilingPanel::Render() {
    if (!visible_) return;

    ImGui::SetNextWindowSize(ImVec2(800, 600), ImGuiCond_FirstUseEver);
    if (!ImGui::Begin(GetName(), &visible_)) {
        ImGui::End();
        return;
    }

    RenderToolbar();

    // View mode tabs
    if (ImGui::BeginTabBar("ProfilingTabs")) {
        if (ImGui::BeginTabItem(ICON_FA_CHART_BAR " Timeline")) {
            view_mode_ = 0;
            RenderTimelineView();
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem(ICON_FA_TABLE " Layer Breakdown")) {
            view_mode_ = 1;
            RenderLayerBreakdown();
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem(ICON_FA_CHART_LINE " History")) {
            view_mode_ = 2;
            RenderHistoryChart();
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem(ICON_FA_CIRCLE_INFO " Summary")) {
            RenderSummaryStats();
            ImGui::EndTabItem();
        }
        ImGui::EndTabBar();
    }

    ImGui::End();
}

void ProfilingPanel::RenderToolbar() {
    // Status indicator
    if (is_profiling_) {
        ImGui::TextColored(ImVec4(0.3f, 0.8f, 0.3f, 1.0f), ICON_FA_CIRCLE " Recording");
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
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Clear all profiling data");
    }

    ImGui::SameLine();

    // Export button
    if (ImGui::Button(ICON_FA_DOWNLOAD " Export CSV")) {
        std::string filename = "profiling_data.csv";
        if (ExportToCSV(filename)) {
            spdlog::info("Profiling data exported to {}", filename);
        }
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Export profiling data to CSV file");
    }

    ImGui::SameLine();
    ImGui::SeparatorEx(ImGuiSeparatorFlags_Vertical);
    ImGui::SameLine();

    // View options
    ImGui::Checkbox("Forward", &show_forward_time_);
    ImGui::SameLine();
    ImGui::Checkbox("Backward", &show_backward_time_);
    ImGui::SameLine();
    ImGui::Checkbox("Memory", &show_memory_);

    ImGui::SameLine();
    ImGui::SeparatorEx(ImGuiSeparatorFlags_Vertical);
    ImGui::SameLine();

    // Snapshot info
    std::lock_guard<std::mutex> lock(data_mutex_);
    ImGui::Text("Snapshots: %zu", history_.size());

    ImGui::Separator();
}

void ProfilingPanel::RenderTimelineView() {
    std::lock_guard<std::mutex> lock(data_mutex_);

    if (history_.empty()) {
        ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f), "No profiling data available.");
        ImGui::TextDisabled("Start training to collect profiling data.");
        return;
    }

    // Get the snapshot to display
    const ProfilingSnapshot& snapshot = selected_snapshot_index_ >= 0 &&
        selected_snapshot_index_ < static_cast<int>(history_.size())
        ? history_[selected_snapshot_index_]
        : history_.back();

    // Snapshot selector
    ImGui::Text("Epoch %d, Step %d", snapshot.epoch, snapshot.step);
    ImGui::SameLine();

    int snapshot_idx = selected_snapshot_index_ >= 0 ? selected_snapshot_index_ : static_cast<int>(history_.size()) - 1;
    ImGui::SetNextItemWidth(200);
    if (ImGui::SliderInt("##snapshot", &snapshot_idx, 0, static_cast<int>(history_.size()) - 1)) {
        selected_snapshot_index_ = snapshot_idx;
    }
    ImGui::SameLine();
    if (ImGui::Checkbox("Auto-scroll", &auto_scroll_timeline_)) {
        if (auto_scroll_timeline_) {
            selected_snapshot_index_ = -1;
        }
    }

    ImGui::Separator();

    // Calculate total time for scaling
    double max_time = 0.0;
    for (const auto& layer : snapshot.layer_profiles) {
        double total = layer.forward_time_ms + layer.backward_time_ms;
        if (total > max_time) max_time = total;
    }

    if (max_time <= 0) max_time = 1.0;

    // Draw timeline as horizontal bars
    ImGui::BeginChild("Timeline", ImVec2(0, 0), true);

    const float row_height = 24.0f;
    const float bar_height = 18.0f;
    const float label_width = 150.0f;
    const float time_label_width = 80.0f;
    const float bar_max_width = ImGui::GetContentRegionAvail().x - label_width - time_label_width - 20;

    for (size_t i = 0; i < snapshot.layer_profiles.size(); i++) {
        const auto& layer = snapshot.layer_profiles[i];

        // Filter check
        if (layer_filter_[0] != '\0') {
            std::string filter_lower = layer_filter_;
            std::string name_lower = layer.name;
            std::transform(filter_lower.begin(), filter_lower.end(), filter_lower.begin(), ::tolower);
            std::transform(name_lower.begin(), name_lower.end(), name_lower.begin(), ::tolower);
            if (name_lower.find(filter_lower) == std::string::npos) {
                continue;
            }
        }

        ImGui::PushID(static_cast<int>(i));

        // Layer name (clickable to highlight node)
        ImGui::BeginGroup();

        bool clicked = ImGui::Selectable(layer.name.c_str(), false, 0, ImVec2(label_width, row_height));
        if (clicked && node_highlight_callback_ && layer.node_id >= 0) {
            node_highlight_callback_(layer.node_id);
        }
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Node ID: %d\nParams: %s\nClick to highlight in editor",
                layer.node_id, FormatMemory(layer.param_count * sizeof(float)).c_str());
        }

        ImGui::EndGroup();
        ImGui::SameLine();

        // Draw bars
        ImVec2 cursor = ImGui::GetCursorScreenPos();
        ImDrawList* draw_list = ImGui::GetWindowDrawList();

        double total_time = layer.forward_time_ms + layer.backward_time_ms;
        float forward_width = static_cast<float>((layer.forward_time_ms / max_time) * bar_max_width);
        float backward_width = static_cast<float>((layer.backward_time_ms / max_time) * bar_max_width);

        // Forward time bar (blue)
        if (show_forward_time_ && layer.forward_time_ms > 0) {
            draw_list->AddRectFilled(
                cursor,
                ImVec2(cursor.x + forward_width, cursor.y + bar_height),
                ImGui::ColorConvertFloat4ToU32(forward_color_)
            );
        }

        // Backward time bar (orange, stacked after forward)
        if (show_backward_time_ && layer.backward_time_ms > 0) {
            draw_list->AddRectFilled(
                ImVec2(cursor.x + forward_width, cursor.y),
                ImVec2(cursor.x + forward_width + backward_width, cursor.y + bar_height),
                ImGui::ColorConvertFloat4ToU32(backward_color_)
            );
        }

        // Advance cursor
        ImGui::Dummy(ImVec2(bar_max_width, row_height));
        ImGui::SameLine();

        // Time label
        ImGui::Text("%s", FormatTime(total_time).c_str());

        ImGui::PopID();
    }

    ImGui::EndChild();

    // Legend
    ImGui::Separator();
    ImGui::ColorButton("##forward_legend", forward_color_, ImGuiColorEditFlags_NoTooltip, ImVec2(12, 12));
    ImGui::SameLine();
    ImGui::Text("Forward");
    ImGui::SameLine();
    ImGui::ColorButton("##backward_legend", backward_color_, ImGuiColorEditFlags_NoTooltip, ImVec2(12, 12));
    ImGui::SameLine();
    ImGui::Text("Backward");
}

void ProfilingPanel::RenderLayerBreakdown() {
    std::lock_guard<std::mutex> lock(data_mutex_);

    if (history_.empty()) {
        ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f), "No profiling data available.");
        return;
    }

    // Filter input
    ImGui::SetNextItemWidth(200);
    ImGui::InputTextWithHint("##filter", "Filter layers...", layer_filter_, sizeof(layer_filter_));

    ImGui::SameLine();
    if (ImGui::Button(ICON_FA_XMARK)) {
        layer_filter_[0] = '\0';
    }

    // Get latest snapshot
    const ProfilingSnapshot& snapshot = history_.back();

    // Table with sortable columns
    if (ImGui::BeginTable("LayerTable", 6,
        ImGuiTableFlags_Sortable | ImGuiTableFlags_Resizable |
        ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg |
        ImGuiTableFlags_ScrollY)) {

        ImGui::TableSetupScrollFreeze(0, 1);
        ImGui::TableSetupColumn("Layer", ImGuiTableColumnFlags_DefaultSort);
        ImGui::TableSetupColumn("Forward (ms)", ImGuiTableColumnFlags_None);
        ImGui::TableSetupColumn("Backward (ms)", ImGuiTableColumnFlags_None);
        ImGui::TableSetupColumn("Total (ms)", ImGuiTableColumnFlags_None);
        ImGui::TableSetupColumn("Memory", ImGuiTableColumnFlags_None);
        ImGui::TableSetupColumn("% of Total", ImGuiTableColumnFlags_None);
        ImGui::TableHeadersRow();

        // Handle sorting
        if (ImGuiTableSortSpecs* sort_specs = ImGui::TableGetSortSpecs()) {
            if (sort_specs->SpecsDirty) {
                sort_column_ = sort_specs->Specs[0].ColumnIndex;
                sort_ascending_ = sort_specs->Specs[0].SortDirection == ImGuiSortDirection_Ascending;
                sort_specs->SpecsDirty = false;
            }
        }

        // Calculate total time for percentage
        double total_time = 0.0;
        for (const auto& layer : snapshot.layer_profiles) {
            total_time += layer.forward_time_ms + layer.backward_time_ms;
        }

        // Create sorted copy
        std::vector<LayerProfile> sorted_profiles = snapshot.layer_profiles;

        // Apply filter
        if (layer_filter_[0] != '\0') {
            std::string filter_lower = layer_filter_;
            std::transform(filter_lower.begin(), filter_lower.end(), filter_lower.begin(), ::tolower);
            sorted_profiles.erase(
                std::remove_if(sorted_profiles.begin(), sorted_profiles.end(),
                    [&filter_lower](const LayerProfile& p) {
                        std::string name_lower = p.name;
                        std::transform(name_lower.begin(), name_lower.end(), name_lower.begin(), ::tolower);
                        return name_lower.find(filter_lower) == std::string::npos;
                    }),
                sorted_profiles.end());
        }

        // Sort
        std::sort(sorted_profiles.begin(), sorted_profiles.end(),
            [this](const LayerProfile& a, const LayerProfile& b) {
                double val_a = 0, val_b = 0;
                switch (sort_column_) {
                    case 0: return sort_ascending_ ? a.name < b.name : a.name > b.name;
                    case 1: val_a = a.forward_time_ms; val_b = b.forward_time_ms; break;
                    case 2: val_a = a.backward_time_ms; val_b = b.backward_time_ms; break;
                    case 3: val_a = a.forward_time_ms + a.backward_time_ms; val_b = b.forward_time_ms + b.backward_time_ms; break;
                    case 4: return sort_ascending_ ? a.memory_bytes < b.memory_bytes : a.memory_bytes > b.memory_bytes;
                    case 5: val_a = a.forward_time_ms + a.backward_time_ms; val_b = b.forward_time_ms + b.backward_time_ms; break;
                }
                return sort_ascending_ ? val_a < val_b : val_a > val_b;
            });

        // Render rows
        for (const auto& layer : sorted_profiles) {
            ImGui::TableNextRow();

            double layer_total = layer.forward_time_ms + layer.backward_time_ms;
            double percentage = total_time > 0 ? (layer_total / total_time) * 100.0 : 0.0;

            ImGui::TableNextColumn();
            if (ImGui::Selectable(layer.name.c_str(), false, ImGuiSelectableFlags_SpanAllColumns)) {
                if (node_highlight_callback_ && layer.node_id >= 0) {
                    node_highlight_callback_(layer.node_id);
                }
            }

            ImGui::TableNextColumn();
            ImGui::Text("%.3f", layer.forward_time_ms);

            ImGui::TableNextColumn();
            ImGui::Text("%.3f", layer.backward_time_ms);

            ImGui::TableNextColumn();
            ImGui::Text("%.3f", layer_total);

            ImGui::TableNextColumn();
            ImGui::Text("%s", FormatMemory(layer.memory_bytes).c_str());

            ImGui::TableNextColumn();
            // Progress bar showing percentage
            ImGui::ProgressBar(static_cast<float>(percentage / 100.0), ImVec2(-1, 0), "");
            ImGui::SameLine(0, 4);
            ImGui::Text("%.1f%%", percentage);
        }

        ImGui::EndTable();
    }
}

void ProfilingPanel::RenderHistoryChart() {
    std::lock_guard<std::mutex> lock(data_mutex_);

    if (history_.size() < 2) {
        ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f), "Need at least 2 snapshots for history chart.");
        return;
    }

    // Prepare data for plotting
    std::vector<double> steps;
    std::vector<double> forward_times;
    std::vector<double> backward_times;
    std::vector<double> total_times;
    std::vector<double> data_loading_times;

    for (size_t i = 0; i < history_.size(); i++) {
        steps.push_back(static_cast<double>(i));
        forward_times.push_back(history_[i].total_forward_ms);
        backward_times.push_back(history_[i].total_backward_ms);
        total_times.push_back(history_[i].total_forward_ms + history_[i].total_backward_ms);
        data_loading_times.push_back(history_[i].data_loading_ms);
    }

    // Plot
    if (ImPlot::BeginPlot("Training Step Timing", ImVec2(-1, 300))) {
        ImPlot::SetupAxes("Step", "Time (ms)");
        ImPlot::SetupAxisLimits(ImAxis_X1, 0, static_cast<double>(history_.size()), ImGuiCond_Always);

        if (show_forward_time_) {
            ImPlot::SetNextLineStyle(ImVec4(forward_color_.x, forward_color_.y, forward_color_.z, 1.0f));
            ImPlot::PlotLine("Forward", steps.data(), forward_times.data(), static_cast<int>(steps.size()));
        }

        if (show_backward_time_) {
            ImPlot::SetNextLineStyle(ImVec4(backward_color_.x, backward_color_.y, backward_color_.z, 1.0f));
            ImPlot::PlotLine("Backward", steps.data(), backward_times.data(), static_cast<int>(steps.size()));
        }

        ImPlot::SetNextLineStyle(ImVec4(0.8f, 0.8f, 0.8f, 1.0f));
        ImPlot::PlotLine("Total", steps.data(), total_times.data(), static_cast<int>(steps.size()));

        ImPlot::SetNextLineStyle(ImVec4(0.5f, 0.5f, 0.8f, 1.0f));
        ImPlot::PlotLine("Data Loading", steps.data(), data_loading_times.data(), static_cast<int>(steps.size()));

        ImPlot::EndPlot();
    }

    // Statistics
    ImGui::Separator();
    ImGui::Text("Statistics over %zu steps:", history_.size());

    if (!total_times.empty()) {
        double avg_total = std::accumulate(total_times.begin(), total_times.end(), 0.0) / total_times.size();
        double min_total = *std::min_element(total_times.begin(), total_times.end());
        double max_total = *std::max_element(total_times.begin(), total_times.end());

        ImGui::Columns(3, nullptr, false);
        ImGui::Text("Avg: %s", FormatTime(avg_total).c_str());
        ImGui::NextColumn();
        ImGui::Text("Min: %s", FormatTime(min_total).c_str());
        ImGui::NextColumn();
        ImGui::Text("Max: %s", FormatTime(max_total).c_str());
        ImGui::Columns(1);
    }
}

void ProfilingPanel::RenderSummaryStats() {
    std::lock_guard<std::mutex> lock(data_mutex_);

    if (history_.empty()) {
        ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f), "No profiling data available.");
        return;
    }

    const ProfilingSnapshot& latest = history_.back();

    ImGui::Text("Latest Snapshot: Epoch %d, Step %d", latest.epoch, latest.step);
    ImGui::Separator();

    // Overall timing
    ImGui::Text(ICON_FA_CLOCK " Timing Breakdown:");
    ImGui::Indent();

    double total_forward = latest.total_forward_ms;
    double total_backward = latest.total_backward_ms;
    double data_loading = latest.data_loading_ms;
    double optimizer = latest.optimizer_step_ms;
    double total = total_forward + total_backward + data_loading + optimizer;

    ImGui::BulletText("Forward Pass: %s (%.1f%%)", FormatTime(total_forward).c_str(),
        total > 0 ? (total_forward / total) * 100 : 0);
    ImGui::BulletText("Backward Pass: %s (%.1f%%)", FormatTime(total_backward).c_str(),
        total > 0 ? (total_backward / total) * 100 : 0);
    ImGui::BulletText("Data Loading: %s (%.1f%%)", FormatTime(data_loading).c_str(),
        total > 0 ? (data_loading / total) * 100 : 0);
    ImGui::BulletText("Optimizer Step: %s (%.1f%%)", FormatTime(optimizer).c_str(),
        total > 0 ? (optimizer / total) * 100 : 0);
    ImGui::BulletText("Total Step Time: %s", FormatTime(total).c_str());

    ImGui::Unindent();
    ImGui::Separator();

    // Layer count
    ImGui::Text(ICON_FA_LAYER_GROUP " Layer Statistics:");
    ImGui::Indent();
    ImGui::BulletText("Number of Layers: %zu", latest.layer_profiles.size());

    // Find slowest layers
    if (!latest.layer_profiles.empty()) {
        std::vector<LayerProfile> sorted = latest.layer_profiles;
        std::sort(sorted.begin(), sorted.end(), [](const LayerProfile& a, const LayerProfile& b) {
            return (a.forward_time_ms + a.backward_time_ms) > (b.forward_time_ms + b.backward_time_ms);
        });

        ImGui::Text("Top 5 Slowest Layers:");
        for (size_t i = 0; i < std::min(sorted.size(), size_t(5)); i++) {
            double layer_total = sorted[i].forward_time_ms + sorted[i].backward_time_ms;
            ImGui::BulletText("%s: %s", sorted[i].name.c_str(), FormatTime(layer_total).c_str());
        }
    }

    ImGui::Unindent();
    ImGui::Separator();

    // Throughput
    if (history_.size() > 1) {
        double elapsed = history_.back().timestamp - history_.front().timestamp;
        if (elapsed > 0) {
            double steps_per_sec = history_.size() / elapsed;
            ImGui::Text(ICON_FA_GAUGE_HIGH " Throughput:");
            ImGui::Indent();
            ImGui::BulletText("%.2f steps/second", steps_per_sec);
            ImGui::BulletText("%.2f samples/second (estimated)", steps_per_sec * 32);  // Assuming batch size of 32
            ImGui::Unindent();
        }
    }
}

void ProfilingPanel::BeginProfiling() {
    std::lock_guard<std::mutex> lock(data_mutex_);
    is_profiling_ = true;
    profiling_start_time_ = std::chrono::steady_clock::now();
    current_snapshot_ = ProfilingSnapshot();
    current_layer_data_.clear();
}

void ProfilingPanel::EndProfiling() {
    std::lock_guard<std::mutex> lock(data_mutex_);
    is_profiling_ = false;
}

void ProfilingPanel::RecordLayerForward(const std::string& layer_name, int node_id, double time_ms, size_t memory_bytes) {
    std::lock_guard<std::mutex> lock(data_mutex_);
    if (!is_profiling_) return;

    auto& layer = current_layer_data_[layer_name];
    layer.name = layer_name;
    layer.node_id = node_id;
    layer.forward_time_ms = time_ms;
    layer.memory_bytes = memory_bytes;

    current_snapshot_.total_forward_ms += time_ms;
}

void ProfilingPanel::RecordLayerBackward(const std::string& layer_name, int node_id, double time_ms) {
    std::lock_guard<std::mutex> lock(data_mutex_);
    if (!is_profiling_) return;

    auto& layer = current_layer_data_[layer_name];
    layer.name = layer_name;
    layer.node_id = node_id;
    layer.backward_time_ms = time_ms;

    current_snapshot_.total_backward_ms += time_ms;
}

void ProfilingPanel::RecordDataLoading(double time_ms) {
    std::lock_guard<std::mutex> lock(data_mutex_);
    if (!is_profiling_) return;

    current_snapshot_.data_loading_ms = time_ms;
}

void ProfilingPanel::RecordOptimizerStep(double time_ms) {
    std::lock_guard<std::mutex> lock(data_mutex_);
    if (!is_profiling_) return;

    current_snapshot_.optimizer_step_ms = time_ms;
}

void ProfilingPanel::FinalizeStep(int epoch, int step) {
    std::lock_guard<std::mutex> lock(data_mutex_);
    if (!is_profiling_) return;

    current_snapshot_.epoch = epoch;
    current_snapshot_.step = step;

    auto now = std::chrono::steady_clock::now();
    current_snapshot_.timestamp = std::chrono::duration<double>(now - profiling_start_time_).count();

    // Convert map to vector
    current_snapshot_.layer_profiles.clear();
    for (const auto& [name, profile] : current_layer_data_) {
        current_snapshot_.layer_profiles.push_back(profile);
    }

    // Add to history
    history_.push_back(current_snapshot_);

    // Trim history if too large
    if (history_.size() > kMaxHistorySize) {
        history_.erase(history_.begin(), history_.begin() + (history_.size() - kMaxHistorySize));
    }

    // Reset for next step
    current_snapshot_ = ProfilingSnapshot();
    current_layer_data_.clear();
}

void ProfilingPanel::Clear() {
    std::lock_guard<std::mutex> lock(data_mutex_);
    history_.clear();
    current_snapshot_ = ProfilingSnapshot();
    current_layer_data_.clear();
    selected_snapshot_index_ = -1;
}

bool ProfilingPanel::ExportToCSV(const std::string& path) {
    std::lock_guard<std::mutex> lock(data_mutex_);

    std::ofstream file(path);
    if (!file.is_open()) {
        spdlog::error("Failed to open file for export: {}", path);
        return false;
    }

    // Write header
    file << "Epoch,Step,Timestamp,Layer,Forward_ms,Backward_ms,Total_ms,Memory_bytes\n";

    // Write data
    for (const auto& snapshot : history_) {
        for (const auto& layer : snapshot.layer_profiles) {
            file << snapshot.epoch << ","
                 << snapshot.step << ","
                 << std::fixed << std::setprecision(3) << snapshot.timestamp << ","
                 << layer.name << ","
                 << std::fixed << std::setprecision(3) << layer.forward_time_ms << ","
                 << std::fixed << std::setprecision(3) << layer.backward_time_ms << ","
                 << std::fixed << std::setprecision(3) << (layer.forward_time_ms + layer.backward_time_ms) << ","
                 << layer.memory_bytes << "\n";
        }
    }

    file.close();
    return true;
}

std::string ProfilingPanel::FormatTime(double ms) const {
    if (ms < 1.0) {
        return std::to_string(static_cast<int>(ms * 1000)) + " us";
    } else if (ms < 1000.0) {
        std::ostringstream ss;
        ss << std::fixed << std::setprecision(2) << ms << " ms";
        return ss.str();
    } else {
        std::ostringstream ss;
        ss << std::fixed << std::setprecision(2) << (ms / 1000.0) << " s";
        return ss.str();
    }
}

std::string ProfilingPanel::FormatMemory(size_t bytes) const {
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

} // namespace cyxwiz
