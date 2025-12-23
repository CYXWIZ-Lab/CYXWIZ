#include "test_results_panel.h"
#include "../../core/test_manager.h"
#include "../../core/file_dialogs.h"
#include "../theme.h"
#include <spdlog/spdlog.h>
#include <algorithm>
#include <cmath>

namespace cyxwiz {

TestResultsPanel::TestResultsPanel()
    : Panel("Test Results", false)  // Hidden by default
{
}

void TestResultsPanel::SetResults(const TestingMetrics& results) {
    std::lock_guard<std::mutex> lock(results_mutex_);
    results_ = results;
    has_results_ = true;
    spdlog::info("TestResultsPanel: Received results with accuracy {:.2f}%",
                 results.test_accuracy * 100);
}

void TestResultsPanel::Clear() {
    std::lock_guard<std::mutex> lock(results_mutex_);
    results_ = TestingMetrics();
    has_results_ = false;
}

void TestResultsPanel::Render() {
    if (!visible_) return;

    ImGui::SetNextWindowSize(ImVec2(700, 500), ImGuiCond_FirstUseEver);

    if (ImGui::Begin(name_.c_str(), &visible_)) {
        if (!has_results_) {
            ImGui::TextDisabled("No test results available.");
            ImGui::TextDisabled("Run a test using Train > Run Test.");
        } else {
            RenderToolbar();

            ImGui::Separator();

            // Tab bar
            if (ImGui::BeginTabBar("TestResultsTabs")) {
                if (ImGui::BeginTabItem(ICON_FA_CHART_LINE " Overview")) {
                    selected_tab_ = 0;
                    RenderOverviewTab();
                    ImGui::EndTabItem();
                }

                if (ImGui::BeginTabItem(ICON_FA_TABLE " Confusion Matrix")) {
                    selected_tab_ = 1;
                    RenderConfusionMatrixTab();
                    ImGui::EndTabItem();
                }

                if (ImGui::BeginTabItem(ICON_FA_LIST_UL " Per-Class Metrics")) {
                    selected_tab_ = 2;
                    RenderPerClassTab();
                    ImGui::EndTabItem();
                }

                if (ImGui::BeginTabItem(ICON_FA_MAGNIFYING_GLASS " Predictions")) {
                    selected_tab_ = 3;
                    RenderPredictionsTab();
                    ImGui::EndTabItem();
                }

                ImGui::EndTabBar();
            }
        }
    }
    ImGui::End();
}

void TestResultsPanel::RenderToolbar() {
    // Export buttons
    if (ImGui::Button(ICON_FA_DOWNLOAD " Export CSV")) {
        ExportToCSV();
    }
    ImGui::SameLine();
    if (ImGui::Button(ICON_FA_FILE_CODE " Export JSON")) {
        ExportToJSON();
    }
    ImGui::SameLine();

    // Clear button
    if (ImGui::Button(ICON_FA_TRASH " Clear")) {
        Clear();
    }

    ImGui::SameLine();
    ImGui::Separator();
    ImGui::SameLine();

    // Quick stats
    std::lock_guard<std::mutex> lock(results_mutex_);
    ImVec4 acc_color = GetAccuracyColor(results_.test_accuracy);
    ImGui::TextColored(acc_color, "Accuracy: %.2f%%", results_.test_accuracy * 100);
    ImGui::SameLine();
    ImGui::Text("| Samples: %d", results_.total_samples);
    ImGui::SameLine();
    ImGui::Text("| Time: %.2fs", results_.total_time_seconds);
}

void TestResultsPanel::RenderOverviewTab() {
    std::lock_guard<std::mutex> lock(results_mutex_);

    // Main metrics section
    ImGui::BeginChild("OverviewMetrics", ImVec2(0, 0), true);

    // Accuracy with large display
    ImGui::PushFont(ImGui::GetIO().Fonts->Fonts[0]);  // Use default font
    ImVec4 acc_color = GetAccuracyColor(results_.test_accuracy);

    ImGui::TextColored(acc_color, "Accuracy");
    ImGui::SameLine();
    ImGui::SetCursorPosX(150);
    ImGui::TextColored(acc_color, "%.2f%%", results_.test_accuracy * 100);
    ImGui::PopFont();

    ImGui::Separator();

    // Two-column layout for metrics
    ImGui::Columns(2, "metrics_cols", true);

    // Left column - Loss & Timing
    ImGui::Text("Test Loss");
    ImGui::SameLine();
    ImGui::SetCursorPosX(120);
    ImGui::Text("%.4f", results_.test_loss);

    ImGui::Text("Samples");
    ImGui::SameLine();
    ImGui::SetCursorPosX(120);
    ImGui::Text("%d", results_.total_samples);

    ImGui::Text("Time");
    ImGui::SameLine();
    ImGui::SetCursorPosX(120);
    ImGui::Text("%.2f sec", results_.total_time_seconds);

    ImGui::Text("Throughput");
    ImGui::SameLine();
    ImGui::SetCursorPosX(120);
    ImGui::Text("%.0f samples/sec", results_.samples_per_second);

    ImGui::NextColumn();

    // Right column - Aggregate metrics
    ImGui::Text("Macro Precision");
    ImGui::SameLine();
    ImGui::SetCursorPosX(ImGui::GetColumnOffset(1) + 130);
    ImGui::TextColored(GetMetricColor(results_.macro_precision), "%.4f", results_.macro_precision);

    ImGui::Text("Macro Recall");
    ImGui::SameLine();
    ImGui::SetCursorPosX(ImGui::GetColumnOffset(1) + 130);
    ImGui::TextColored(GetMetricColor(results_.macro_recall), "%.4f", results_.macro_recall);

    ImGui::Text("Macro F1");
    ImGui::SameLine();
    ImGui::SetCursorPosX(ImGui::GetColumnOffset(1) + 130);
    ImGui::TextColored(GetMetricColor(results_.macro_f1), "%.4f", results_.macro_f1);

    ImGui::Text("Weighted F1");
    ImGui::SameLine();
    ImGui::SetCursorPosX(ImGui::GetColumnOffset(1) + 130);
    ImGui::TextColored(GetMetricColor(results_.weighted_f1), "%.4f", results_.weighted_f1);

    ImGui::Columns(1);

    ImGui::Separator();

    // Class summary
    if (!results_.per_class_metrics.empty()) {
        ImGui::Text("Classes: %d", static_cast<int>(results_.per_class_metrics.size()));

        // Find best and worst classes
        int best_class = 0, worst_class = 0;
        float best_f1 = results_.per_class_metrics[0].f1_score;
        float worst_f1 = results_.per_class_metrics[0].f1_score;

        for (size_t i = 1; i < results_.per_class_metrics.size(); ++i) {
            if (results_.per_class_metrics[i].f1_score > best_f1) {
                best_f1 = results_.per_class_metrics[i].f1_score;
                best_class = static_cast<int>(i);
            }
            if (results_.per_class_metrics[i].f1_score < worst_f1) {
                worst_f1 = results_.per_class_metrics[i].f1_score;
                worst_class = static_cast<int>(i);
            }
        }

        ImGui::Text("Best Class:");
        ImGui::SameLine();
        ImGui::TextColored(ImVec4(0.2f, 0.8f, 0.2f, 1.0f), "%s (F1: %.4f)",
                           results_.per_class_metrics[best_class].class_name.c_str(), best_f1);

        ImGui::Text("Worst Class:");
        ImGui::SameLine();
        ImGui::TextColored(ImVec4(0.8f, 0.3f, 0.3f, 1.0f), "%s (F1: %.4f)",
                           results_.per_class_metrics[worst_class].class_name.c_str(), worst_f1);
    }

    ImGui::EndChild();
}

void TestResultsPanel::RenderConfusionMatrixTab() {
    std::lock_guard<std::mutex> lock(results_mutex_);

    const auto& conf = results_.confusion_matrix;
    if (conf.num_classes == 0) {
        ImGui::TextDisabled("No confusion matrix data.");
        return;
    }

    // Controls
    ImGui::Checkbox("Normalize", &normalize_confusion_);
    ImGui::SameLine();
    ImGui::SetNextItemWidth(100);
    ImGui::SliderFloat("Cell Size", &cell_size_, 20.0f, 60.0f, "%.0f");

    ImGui::Separator();

    // Find max value for color scaling
    int max_value = 1;
    for (int i = 0; i < conf.num_classes; ++i) {
        for (int j = 0; j < conf.num_classes; ++j) {
            max_value = std::max(max_value, conf.matrix[i][j]);
        }
    }

    // Draw matrix
    ImGui::BeginChild("ConfMatrix", ImVec2(0, 0), true, ImGuiWindowFlags_HorizontalScrollbar);

    // Header row with predicted labels
    ImGui::SetCursorPosX(cell_size_ + 10);
    for (int j = 0; j < conf.num_classes; ++j) {
        ImGui::SetCursorPosX((j + 1) * (cell_size_ + 5) + 40);
        ImGui::Text("%d", j);
    }
    ImGui::Text("");  // New line

    // Matrix rows
    for (int i = 0; i < conf.num_classes; ++i) {
        // Row label
        ImGui::Text("%2d ", i);
        ImGui::SameLine();

        for (int j = 0; j < conf.num_classes; ++j) {
            int value = conf.matrix[i][j];
            float display_value = normalize_confusion_ && conf.matrix[i][j] > 0 ?
                static_cast<float>(value) / results_.per_class_metrics[i].support : static_cast<float>(value);

            ImVec4 bg_color = GetConfusionCellColor(value, max_value);
            if (i == j) {
                // Diagonal cells - good predictions
                bg_color = ImVec4(0.1f, 0.5f + 0.5f * (static_cast<float>(value) / max_value), 0.1f, 1.0f);
            }

            ImGui::PushStyleColor(ImGuiCol_Button, bg_color);
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, bg_color);
            ImGui::PushStyleColor(ImGuiCol_ButtonActive, bg_color);

            char label[32];
            if (normalize_confusion_) {
                snprintf(label, sizeof(label), "%.2f", display_value);
            } else {
                snprintf(label, sizeof(label), "%d", value);
            }

            ImGui::SetCursorPosX((j + 1) * (cell_size_ + 5) + 30);
            ImGui::Button(label, ImVec2(cell_size_, cell_size_));

            ImGui::PopStyleColor(3);

            // Tooltip
            if (ImGui::IsItemHovered()) {
                ImGui::BeginTooltip();
                ImGui::Text("Actual: %d, Predicted: %d", i, j);
                ImGui::Text("Count: %d", value);
                if (results_.per_class_metrics[i].support > 0) {
                    ImGui::Text("Percentage: %.1f%%",
                                100.0f * value / results_.per_class_metrics[i].support);
                }
                ImGui::EndTooltip();
            }

            ImGui::SameLine();
        }
        ImGui::Text("");  // New line
    }

    // Legend
    ImGui::Separator();
    ImGui::Text("Rows: Actual class | Columns: Predicted class");
    ImGui::Text("Diagonal: Correct predictions");

    ImGui::EndChild();
}

void TestResultsPanel::RenderPerClassTab() {
    std::lock_guard<std::mutex> lock(results_mutex_);

    if (results_.per_class_metrics.empty()) {
        ImGui::TextDisabled("No per-class metrics available.");
        return;
    }

    // Filter
    ImGui::SetNextItemWidth(200);
    ImGui::InputTextWithHint("##filter", "Filter classes...", filter_text_, sizeof(filter_text_));
    ImGui::SameLine();
    if (ImGui::Button("Clear")) {
        filter_text_[0] = '\0';
    }

    ImGui::Separator();

    // Table
    if (ImGui::BeginTable("PerClassTable", 7,
            ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg | ImGuiTableFlags_Resizable |
            ImGuiTableFlags_Sortable | ImGuiTableFlags_ScrollY,
            ImVec2(0, ImGui::GetContentRegionAvail().y))) {

        ImGui::TableSetupColumn("Class", ImGuiTableColumnFlags_WidthFixed, 80.0f);
        ImGui::TableSetupColumn("Precision", ImGuiTableColumnFlags_WidthFixed, 80.0f);
        ImGui::TableSetupColumn("Recall", ImGuiTableColumnFlags_WidthFixed, 80.0f);
        ImGui::TableSetupColumn("F1 Score", ImGuiTableColumnFlags_WidthFixed, 80.0f);
        ImGui::TableSetupColumn("TP", ImGuiTableColumnFlags_WidthFixed, 50.0f);
        ImGui::TableSetupColumn("FP", ImGuiTableColumnFlags_WidthFixed, 50.0f);
        ImGui::TableSetupColumn("Support", ImGuiTableColumnFlags_WidthFixed, 60.0f);
        ImGui::TableHeadersRow();

        std::string filter_str(filter_text_);
        std::transform(filter_str.begin(), filter_str.end(), filter_str.begin(), ::tolower);

        for (const auto& cm : results_.per_class_metrics) {
            // Filter
            if (!filter_str.empty()) {
                std::string class_lower = cm.class_name;
                std::transform(class_lower.begin(), class_lower.end(), class_lower.begin(), ::tolower);
                if (class_lower.find(filter_str) == std::string::npos) {
                    continue;
                }
            }

            ImGui::TableNextRow();

            ImGui::TableNextColumn();
            ImGui::Text("%s", cm.class_name.c_str());

            ImGui::TableNextColumn();
            ImGui::TextColored(GetMetricColor(cm.precision), "%.4f", cm.precision);

            ImGui::TableNextColumn();
            ImGui::TextColored(GetMetricColor(cm.recall), "%.4f", cm.recall);

            ImGui::TableNextColumn();
            ImGui::TextColored(GetMetricColor(cm.f1_score), "%.4f", cm.f1_score);

            ImGui::TableNextColumn();
            ImGui::Text("%d", cm.true_positives);

            ImGui::TableNextColumn();
            ImGui::Text("%d", cm.false_positives);

            ImGui::TableNextColumn();
            ImGui::Text("%d", cm.support);
        }

        ImGui::EndTable();
    }
}

void TestResultsPanel::RenderPredictionsTab() {
    std::lock_guard<std::mutex> lock(results_mutex_);

    if (results_.predictions.empty()) {
        ImGui::TextDisabled("No prediction data available.");
        return;
    }

    ImGui::Text("Total predictions: %zu", results_.predictions.size());

    // Show only misclassifications option
    static bool show_only_errors = false;
    ImGui::Checkbox("Show only misclassifications", &show_only_errors);

    ImGui::Separator();

    // Table
    if (ImGui::BeginTable("PredictionsTable", 4,
            ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg | ImGuiTableFlags_ScrollY,
            ImVec2(0, ImGui::GetContentRegionAvail().y))) {

        ImGui::TableSetupColumn("Index", ImGuiTableColumnFlags_WidthFixed, 60.0f);
        ImGui::TableSetupColumn("Predicted", ImGuiTableColumnFlags_WidthFixed, 80.0f);
        ImGui::TableSetupColumn("Actual", ImGuiTableColumnFlags_WidthFixed, 80.0f);
        ImGui::TableSetupColumn("Confidence", ImGuiTableColumnFlags_WidthStretch);
        ImGui::TableHeadersRow();

        // Use clipper for performance with large datasets
        ImGuiListClipper clipper;
        clipper.Begin(static_cast<int>(results_.predictions.size()));

        while (clipper.Step()) {
            for (int i = clipper.DisplayStart; i < clipper.DisplayEnd; ++i) {
                int pred = results_.predictions[i];
                int actual = results_.ground_truth[i];
                float conf = i < static_cast<int>(results_.confidences.size()) ?
                             results_.confidences[i] : 0.0f;
                bool correct = (pred == actual);

                if (show_only_errors && correct) {
                    continue;
                }

                ImGui::TableNextRow();

                // Index
                ImGui::TableNextColumn();
                ImGui::Text("%d", i);

                // Predicted
                ImGui::TableNextColumn();
                if (correct) {
                    ImGui::TextColored(ImVec4(0.2f, 0.8f, 0.2f, 1.0f), "%d", pred);
                } else {
                    ImGui::TextColored(ImVec4(0.9f, 0.3f, 0.3f, 1.0f), "%d", pred);
                }

                // Actual
                ImGui::TableNextColumn();
                ImGui::Text("%d", actual);

                // Confidence bar
                ImGui::TableNextColumn();
                ImVec4 conf_color = conf > 0.8f ? ImVec4(0.2f, 0.7f, 0.2f, 1.0f) :
                                    conf > 0.5f ? ImVec4(0.7f, 0.7f, 0.2f, 1.0f) :
                                                  ImVec4(0.7f, 0.3f, 0.2f, 1.0f);
                ImGui::PushStyleColor(ImGuiCol_PlotHistogram, conf_color);
                ImGui::ProgressBar(conf, ImVec2(-1, 0), "");
                ImGui::PopStyleColor();
                ImGui::SameLine();
                ImGui::Text("%.1f%%", conf * 100);
            }
        }

        ImGui::EndTable();
    }
}

ImVec4 TestResultsPanel::GetAccuracyColor(float accuracy) {
    if (accuracy >= 0.95f) return ImVec4(0.2f, 0.9f, 0.2f, 1.0f);  // Green
    if (accuracy >= 0.85f) return ImVec4(0.5f, 0.9f, 0.2f, 1.0f);  // Yellow-green
    if (accuracy >= 0.70f) return ImVec4(0.9f, 0.9f, 0.2f, 1.0f);  // Yellow
    if (accuracy >= 0.50f) return ImVec4(0.9f, 0.6f, 0.2f, 1.0f);  // Orange
    return ImVec4(0.9f, 0.2f, 0.2f, 1.0f);  // Red
}

ImVec4 TestResultsPanel::GetMetricColor(float value) {
    if (value >= 0.9f) return ImVec4(0.2f, 0.9f, 0.2f, 1.0f);
    if (value >= 0.7f) return ImVec4(0.9f, 0.9f, 0.2f, 1.0f);
    if (value >= 0.5f) return ImVec4(0.9f, 0.6f, 0.2f, 1.0f);
    return ImVec4(0.9f, 0.2f, 0.2f, 1.0f);
}

ImVec4 TestResultsPanel::GetConfusionCellColor(int value, int max_value) {
    if (max_value == 0) return ImVec4(0.2f, 0.2f, 0.2f, 1.0f);
    float intensity = static_cast<float>(value) / max_value;
    return ImVec4(0.3f + 0.5f * intensity, 0.2f, 0.2f, 1.0f);
}

void TestResultsPanel::ExportToCSV() {
    auto result = FileDialogs::SaveFile(
        "Export Test Results",
        {{"CSV Files", "csv"}, {"All Files", "*"}},
        nullptr,
        "test_results.csv"
    );
    if (result) {
        if (TestManager::Instance().ExportResultsToCSV(*result)) {
            spdlog::info("Exported results to: {}", *result);
        }
    }
}

void TestResultsPanel::ExportToJSON() {
    auto result = FileDialogs::SaveFile(
        "Export Test Results",
        {{"JSON Files", "json"}, {"All Files", "*"}},
        nullptr,
        "test_results.json"
    );
    if (result) {
        if (TestManager::Instance().ExportResultsToJSON(*result)) {
            spdlog::info("Exported results to: {}", *result);
        }
    }
}

} // namespace cyxwiz
