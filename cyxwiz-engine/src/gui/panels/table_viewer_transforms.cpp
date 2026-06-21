#include "table_viewer.h"
#include <imgui.h>
#include <spdlog/spdlog.h>
#include <algorithm>
#include <cstring>

namespace cyxwiz {

// ============================================================================

void TableViewerPanel::ApplyColorMap(TableTab* tab, int column, ColorMapType type) {
    if (!tab || column < 0) return;

    // Ensure vector is sized
    if (tab->column_colormaps.size() <= static_cast<size_t>(column)) {
        tab->column_colormaps.resize(column + 1);
    }

    auto& cmap = tab->column_colormaps[column];
    cmap.type = type;
    cmap.auto_range = true;

    // Get column min/max for normalization
    if (column < static_cast<int>(tab->column_stats.size())) {
        cmap.min_val = static_cast<float>(tab->column_stats[column].min_val);
        cmap.max_val = static_cast<float>(tab->column_stats[column].max_val);
    }

    spdlog::info("Applied colormap {} to column {}", static_cast<int>(type), column);
}

void TableViewerPanel::ClearColorMap(TableTab* tab, int column) {
    if (!tab || column < 0) return;

    if (static_cast<size_t>(column) < tab->column_colormaps.size()) {
        tab->column_colormaps[column].type = ColorMapType::None;
    }
}

ImVec4 TableViewerPanel::GetColorMapColor(double normalized, ColorMapType type) const {
    // Clamp to [0, 1]
    float t = std::clamp(static_cast<float>(normalized), 0.0f, 1.0f);

    switch (type) {
        case ColorMapType::Viridis: {
            // Purple → Blue → Green → Yellow
            if (t < 0.25f) {
                float s = t / 0.25f;
                return ImVec4(
                    0.267f + s * (0.192f - 0.267f),
                    0.004f + s * (0.408f - 0.004f),
                    0.329f + s * (0.557f - 0.329f),
                    1.0f);
            } else if (t < 0.50f) {
                float s = (t - 0.25f) / 0.25f;
                return ImVec4(
                    0.192f + s * (0.208f - 0.192f),
                    0.408f + s * (0.718f - 0.408f),
                    0.557f + s * (0.475f - 0.557f),
                    1.0f);
            } else if (t < 0.75f) {
                float s = (t - 0.50f) / 0.25f;
                return ImVec4(
                    0.208f + s * (0.565f - 0.208f),
                    0.718f + s * (0.820f - 0.718f),
                    0.475f + s * (0.251f - 0.475f),
                    1.0f);
            } else {
                float s = (t - 0.75f) / 0.25f;
                return ImVec4(
                    0.565f + s * (0.992f - 0.565f),
                    0.820f + s * (0.906f - 0.820f),
                    0.251f + s * (0.145f - 0.251f),
                    1.0f);
            }
        }

        case ColorMapType::Inferno: {
            // Black → Magenta → Orange → Yellow
            if (t < 0.33f) {
                float s = t / 0.33f;
                return ImVec4(s * 0.737f, s * 0.216f, s * 0.329f, 1.0f);
            } else if (t < 0.66f) {
                float s = (t - 0.33f) / 0.33f;
                return ImVec4(
                    0.737f + s * (0.976f - 0.737f),
                    0.216f + s * (0.557f - 0.216f),
                    0.329f + s * (0.035f - 0.329f),
                    1.0f);
            } else {
                float s = (t - 0.66f) / 0.34f;
                return ImVec4(
                    0.976f + s * (0.988f - 0.976f),
                    0.557f + s * (0.996f - 0.557f),
                    0.035f + s * (0.643f - 0.035f),
                    1.0f);
            }
        }

        case ColorMapType::RdYlGn: {
            // Red → Yellow → Green (diverging)
            if (t < 0.5f) {
                float s = t / 0.5f;
                return ImVec4(
                    0.843f + s * (0.996f - 0.843f),
                    0.188f + s * (0.878f - 0.188f),
                    0.153f + s * (0.545f - 0.153f),
                    1.0f);
            } else {
                float s = (t - 0.5f) / 0.5f;
                return ImVec4(
                    0.996f + s * (0.102f - 0.996f),
                    0.878f + s * (0.596f - 0.878f),
                    0.545f + s * (0.314f - 0.545f),
                    1.0f);
            }
        }

        case ColorMapType::Blues: {
            // Light blue → Dark blue
            return ImVec4(
                0.031f + (1.0f - t) * 0.937f,
                0.188f + (1.0f - t) * 0.757f,
                0.420f + (1.0f - t) * 0.525f,
                1.0f);
        }

        case ColorMapType::Plasma: {
            // Purple → Pink → Orange → Yellow
            if (t < 0.33f) {
                float s = t / 0.33f;
                return ImVec4(
                    0.050f + s * (0.798f - 0.050f),
                    0.030f + s * (0.280f - 0.030f),
                    0.528f + s * (0.469f - 0.528f),
                    1.0f);
            } else if (t < 0.66f) {
                float s = (t - 0.33f) / 0.33f;
                return ImVec4(
                    0.798f + s * (0.973f - 0.798f),
                    0.280f + s * (0.580f - 0.280f),
                    0.469f + s * (0.254f - 0.469f),
                    1.0f);
            } else {
                float s = (t - 0.66f) / 0.34f;
                return ImVec4(
                    0.973f + s * (0.940f - 0.973f),
                    0.580f + s * (0.975f - 0.580f),
                    0.254f + s * (0.131f - 0.254f),
                    1.0f);
            }
        }

        default:
            return ImVec4(1.0f, 1.0f, 1.0f, 1.0f);
    }
}

// ============================================================================
// Filtering
// ============================================================================

void TableViewerPanel::ApplyFilter(TableTab* tab) {
    if (!tab || !tab->table) return;

    tab->filtered_indices.clear();

    if (tab->filter_text.empty()) {
        // No filter - show all
        tab->filtered_indices = tab->sorted_indices;
        return;
    }

    // Filter matching rows
    for (size_t idx : tab->sorted_indices) {
        bool match = false;
        for (size_t c = 0; c < tab->table->GetColumnCount(); c++) {
            std::string cell = tab->table->GetCellAsString(idx, c);
            if (cell.find(tab->filter_text) != std::string::npos) {
                match = true;
                break;
            }
        }
        if (match) {
            tab->filtered_indices.push_back(idx);
        }
    }

    spdlog::info("Filter applied: {} of {} rows match", tab->filtered_indices.size(), tab->sorted_indices.size());
}

void TableViewerPanel::ClearFilter(TableTab* tab) {
    if (!tab) return;

    tab->filter_text.clear();
    std::memset(tab->filter_buffer, 0, sizeof(tab->filter_buffer));
    tab->filtered_indices.clear();
    tab->filter_mode_hide = false;
}

// ============================================================================
// Quick Plot
// ============================================================================


}  // namespace cyxwiz

