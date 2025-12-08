#include "random_generator_panel.h"
#include "../icons.h"
#include <spdlog/spdlog.h>
#include <algorithm>
#include <sstream>
#include <iomanip>
#include <cmath>

namespace cyxwiz {

RandomGeneratorPanel::RandomGeneratorPanel() {
    spdlog::info("RandomGeneratorPanel initialized");
}

RandomGeneratorPanel::~RandomGeneratorPanel() {
    if (compute_thread_ && compute_thread_->joinable()) {
        compute_thread_->join();
    }
}

void RandomGeneratorPanel::Render() {
    if (!visible_) return;

    ImGui::SetNextWindowSize(ImVec2(700, 550), ImGuiCond_FirstUseEver);

    if (ImGui::Begin(ICON_FA_DICE " Random Generator###RandomGeneratorPanel", &visible_)) {
        RenderToolbar();
        ImGui::Separator();

        if (is_computing_.load()) {
            RenderLoadingIndicator();
        } else {
            float panel_width = ImGui::GetContentRegionAvail().x;

            ImGui::BeginChild("SettingsPanel", ImVec2(panel_width * 0.35f, 0), true);
            RenderSettings();
            ImGui::EndChild();

            ImGui::SameLine();

            ImGui::BeginChild("ResultsPanel", ImVec2(0, 0), true);

            if (ImGui::BeginTabBar("RandomTabs")) {
                if (ImGui::BeginTabItem(ICON_FA_CHART_BAR " Histogram")) {
                    RenderHistogram();
                    ImGui::EndTabItem();
                }
                if (ImGui::BeginTabItem(ICON_FA_LIST_OL " Values")) {
                    RenderResults();
                    ImGui::EndTabItem();
                }
                if (ImGui::BeginTabItem(ICON_FA_CHART_PIE " Statistics")) {
                    RenderStatistics();
                    ImGui::EndTabItem();
                }
                if (ImGui::BeginTabItem(ICON_FA_FINGERPRINT " UUIDs")) {
                    RenderUUIDs();
                    ImGui::EndTabItem();
                }
                ImGui::EndTabBar();
            }

            ImGui::EndChild();
        }
    }
    ImGui::End();
}

void RandomGeneratorPanel::RenderToolbar() {
    if (ImGui::Button(ICON_FA_PLAY " Generate")) {
        GenerateAsync();
    }

    ImGui::SameLine();
    if (ImGui::Button(ICON_FA_ERASER " Clear")) {
        ClearResults();
    }

    ImGui::SameLine();
    if (ImGui::Button(ICON_FA_COPY " Copy")) {
        CopyValues();
    }

    ImGui::SameLine();
    ImGui::Separator();
    ImGui::SameLine();

    if (has_result_) {
        ImGui::Text("Count: %d | Mean: %.4f | StdDev: %.4f",
                    result_.count, result_.mean, result_.std_dev);
    } else {
        ImGui::TextDisabled("No data generated");
    }
}

void RandomGeneratorPanel::RenderSettings() {
    ImGui::Text(ICON_FA_SLIDERS " Distribution Settings");
    ImGui::Separator();

    // Distribution selection
    const char* distributions[] = {
        "Uniform", "Normal", "Exponential", "Poisson", "Integer"
    };
    ImGui::Combo("Distribution", &distribution_idx_, distributions, IM_ARRAYSIZE(distributions));

    ImGui::Spacing();

    // Count
    ImGui::DragInt("Count", &count_, 1.0f, 1, 100000);

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    // Distribution-specific parameters
    switch (distribution_idx_) {
        case 0: // Uniform
            ImGui::Text("Uniform Distribution:");
            ImGui::DragScalar("Min", ImGuiDataType_Double, &uniform_min_, 0.1f);
            ImGui::DragScalar("Max", ImGuiDataType_Double, &uniform_max_, 0.1f);
            break;

        case 1: // Normal
            ImGui::Text("Normal Distribution:");
            ImGui::DragScalar("Mean", ImGuiDataType_Double, &normal_mean_, 0.1f);
            ImGui::DragScalar("Std Dev", ImGuiDataType_Double, &normal_stddev_, 0.1f);
            if (normal_stddev_ < 0.001) normal_stddev_ = 0.001;
            break;

        case 2: // Exponential
            ImGui::Text("Exponential Distribution:");
            ImGui::DragScalar("Lambda", ImGuiDataType_Double, &exponential_lambda_, 0.1f);
            if (exponential_lambda_ < 0.001) exponential_lambda_ = 0.001;
            break;

        case 3: // Poisson
            ImGui::Text("Poisson Distribution:");
            ImGui::DragScalar("Lambda", ImGuiDataType_Double, &poisson_lambda_, 0.1f);
            if (poisson_lambda_ < 0.001) poisson_lambda_ = 0.001;
            break;

        case 4: // Integer
            ImGui::Text("Integer Distribution:");
            ImGui::DragInt("Min", &integer_min_, 1.0f);
            ImGui::DragInt("Max", &integer_max_, 1.0f);
            if (integer_max_ < integer_min_) integer_max_ = integer_min_;
            break;
    }

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    // Seed control
    ImGui::Checkbox("Use Custom Seed", &use_custom_seed_);
    if (use_custom_seed_) {
        ImGui::DragScalar("Seed", ImGuiDataType_S64, &custom_seed_, 1.0f);
    }

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    // Generate button at bottom of settings
    if (ImGui::Button(ICON_FA_DICE " Generate", ImVec2(-1, 0))) {
        GenerateAsync();
    }
}

void RandomGeneratorPanel::RenderResults() {
    if (!error_message_.empty()) {
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 0.4f, 0.4f, 1.0f));
        ImGui::TextWrapped("%s %s", ICON_FA_TRIANGLE_EXCLAMATION, error_message_.c_str());
        ImGui::PopStyleColor();
        return;
    }

    if (!has_result_ || result_.values.empty()) {
        ImGui::TextDisabled("No values generated. Click 'Generate' to create random numbers.");
        return;
    }

    ImGui::Text("Generated Values (%d):", static_cast<int>(result_.values.size()));
    ImGui::Separator();

    ImGui::BeginChild("ValuesList", ImVec2(0, 0), false);

    // Show values in columns
    int columns = 5;
    int row = 0;
    ImGui::Columns(columns, "values_cols", false);

    for (size_t i = 0; i < result_.values.size() && i < 1000; ++i) {
        ImGui::Text("%.6g", result_.values[i]);
        ImGui::NextColumn();
    }

    if (result_.values.size() > 1000) {
        ImGui::Columns(1);
        ImGui::TextDisabled("... and %d more values", static_cast<int>(result_.values.size() - 1000));
    }

    ImGui::Columns(1);
    ImGui::EndChild();
}

void RandomGeneratorPanel::RenderHistogram() {
    if (!has_result_ || result_.histogram.empty()) {
        ImGui::TextDisabled("No histogram data. Generate random numbers first.");
        return;
    }

    ImGui::Text(ICON_FA_CHART_BAR " Histogram");
    ImGui::Separator();

    // Find max count for scaling
    int max_count = 0;
    for (const auto& bin : result_.histogram) {
        max_count = std::max(max_count, bin.second);
    }

    if (max_count == 0) {
        ImGui::TextDisabled("No data in histogram");
        return;
    }

    ImGui::BeginChild("HistogramArea", ImVec2(0, 0), false);

    float bar_width = ImGui::GetContentRegionAvail().x - 100;
    float bar_height = 18.0f;

    // Calculate bin range for labels
    double bin_width = (result_.max_value - result_.min_value) / 20.0;

    for (const auto& bin : result_.histogram) {
        float fraction = static_cast<float>(bin.second) / max_count;
        float width = fraction * bar_width;

        // Bin label
        double bin_start = result_.min_value + bin.first * bin_width;
        ImGui::Text("[%.2f]", bin_start);
        ImGui::SameLine();

        // Draw bar
        ImVec2 cursor = ImGui::GetCursorScreenPos();
        ImGui::GetWindowDrawList()->AddRectFilled(
            cursor,
            ImVec2(cursor.x + width, cursor.y + bar_height - 2),
            IM_COL32(100, 150, 200, 255)
        );

        ImGui::SetCursorPosX(ImGui::GetCursorPosX() + width + 5);
        ImGui::Text("%d", bin.second);

        ImGui::Spacing();
    }

    ImGui::EndChild();
}

void RandomGeneratorPanel::RenderStatistics() {
    if (!has_result_) {
        ImGui::TextDisabled("No statistics available. Generate random numbers first.");
        return;
    }

    ImGui::Text(ICON_FA_CHART_PIE " Statistics");
    ImGui::Separator();
    ImGui::Spacing();

    ImGui::Columns(2, "stats_cols", false);

    ImGui::Text("Distribution:");
    ImGui::NextColumn();
    ImGui::Text("%s", result_.distribution.c_str());
    ImGui::NextColumn();

    ImGui::Text("Count:");
    ImGui::NextColumn();
    ImGui::Text("%d", result_.count);
    ImGui::NextColumn();

    ImGui::Text("Min Value:");
    ImGui::NextColumn();
    ImGui::Text("%.6g", result_.min_value);
    ImGui::NextColumn();

    ImGui::Text("Max Value:");
    ImGui::NextColumn();
    ImGui::Text("%.6g", result_.max_value);
    ImGui::NextColumn();

    ImGui::Text("Mean:");
    ImGui::NextColumn();
    ImGui::Text("%.6g", result_.mean);
    ImGui::NextColumn();

    ImGui::Text("Std Deviation:");
    ImGui::NextColumn();
    ImGui::Text("%.6g", result_.std_dev);
    ImGui::NextColumn();

    ImGui::Text("Seed Used:");
    ImGui::NextColumn();
    ImGui::Text("%llu", static_cast<unsigned long long>(result_.seed_used));
    ImGui::NextColumn();

    ImGui::Columns(1);

    // Additional stats
    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    if (!result_.values.empty()) {
        // Median (approximate - sort would be expensive)
        std::vector<double> sorted_values = result_.values;
        std::nth_element(sorted_values.begin(),
                        sorted_values.begin() + sorted_values.size() / 2,
                        sorted_values.end());
        double median = sorted_values[sorted_values.size() / 2];

        ImGui::Text("Median (approx): %.6g", median);

        // Range
        double range = result_.max_value - result_.min_value;
        ImGui::Text("Range: %.6g", range);

        // Variance
        double variance = result_.std_dev * result_.std_dev;
        ImGui::Text("Variance: %.6g", variance);
    }
}

void RandomGeneratorPanel::RenderUUIDs() {
    ImGui::Text(ICON_FA_FINGERPRINT " UUID Generator");
    ImGui::Separator();

    ImGui::DragInt("Count##uuid", &uuid_count_, 1.0f, 1, 100);
    ImGui::SameLine();
    if (ImGui::Button("Generate UUIDs")) {
        GenerateUUIDsAsync();
    }

    ImGui::Spacing();

    if (generated_uuids_.empty()) {
        ImGui::TextDisabled("No UUIDs generated");
        return;
    }

    if (ImGui::Button(ICON_FA_COPY " Copy All")) {
        std::ostringstream oss;
        for (const auto& uuid : generated_uuids_) {
            oss << uuid << "\n";
        }
        ImGui::SetClipboardText(oss.str().c_str());
    }

    ImGui::Separator();

    ImGui::BeginChild("UUIDList", ImVec2(0, 0), false);

    for (size_t i = 0; i < generated_uuids_.size(); ++i) {
        ImGui::PushID(static_cast<int>(i));

        if (ImGui::SmallButton(ICON_FA_COPY)) {
            ImGui::SetClipboardText(generated_uuids_[i].c_str());
        }
        ImGui::SameLine();

        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.6f, 0.8f, 1.0f, 1.0f));
        ImGui::TextUnformatted(generated_uuids_[i].c_str());
        ImGui::PopStyleColor();

        ImGui::PopID();
    }

    ImGui::EndChild();
}

void RandomGeneratorPanel::RenderLoadingIndicator() {
    ImGui::SetCursorPosY(ImGui::GetWindowHeight() / 2 - 20);
    float width = ImGui::GetWindowWidth();
    ImGui::SetCursorPosX(width / 2 - 80);
    ImGui::Text(ICON_FA_SPINNER " Generating...");
}

void RandomGeneratorPanel::GenerateAsync() {
    if (is_computing_.load()) return;

    if (compute_thread_ && compute_thread_->joinable()) {
        compute_thread_->join();
    }

    is_computing_ = true;
    has_result_ = false;
    error_message_.clear();

    int64_t seed = use_custom_seed_ ? custom_seed_ : -1;

    compute_thread_ = std::make_unique<std::thread>([this, seed]() {
        std::lock_guard<std::mutex> lock(result_mutex_);

        try {
            switch (distribution_idx_) {
                case 0: // Uniform
                    result_ = Utilities::GenerateUniform(count_, uniform_min_, uniform_max_, seed);
                    break;
                case 1: // Normal
                    result_ = Utilities::GenerateNormal(count_, normal_mean_, normal_stddev_, seed);
                    break;
                case 2: // Exponential
                    result_ = Utilities::GenerateExponential(count_, exponential_lambda_, seed);
                    break;
                case 3: // Poisson
                    result_ = Utilities::GeneratePoisson(count_, poisson_lambda_, seed);
                    break;
                case 4: // Integer
                    result_ = Utilities::GenerateIntegers(count_, integer_min_, integer_max_, seed);
                    break;
            }

            if (result_.success) {
                has_result_ = true;
                spdlog::info("Generated {} random numbers ({} distribution)",
                            count_, result_.distribution);
            } else {
                error_message_ = result_.error_message;
            }
        } catch (const std::exception& e) {
            error_message_ = std::string("Exception: ") + e.what();
        }

        is_computing_ = false;
    });
}

void RandomGeneratorPanel::GenerateUUIDsAsync() {
    generated_uuids_ = Utilities::GenerateUUIDs(uuid_count_);
    spdlog::info("Generated {} UUIDs", uuid_count_);
}

void RandomGeneratorPanel::CopyValues() {
    if (!has_result_ || result_.values.empty()) return;

    std::ostringstream oss;
    for (size_t i = 0; i < result_.values.size(); ++i) {
        if (i > 0) oss << "\n";
        oss << std::setprecision(10) << result_.values[i];
    }

    ImGui::SetClipboardText(oss.str().c_str());
    spdlog::info("Values copied to clipboard");
}

void RandomGeneratorPanel::ClearResults() {
    has_result_ = false;
    error_message_.clear();
    result_ = RandomNumberResult();
    generated_uuids_.clear();
}

} // namespace cyxwiz
