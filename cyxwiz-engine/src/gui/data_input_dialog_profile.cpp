// DataInputDialog profiling tab and Arrow column statistics.

#include "node_config_dialog.h"
#include "../core/arrow_dataset.h"
#include "../core/data_registry.h"

#include <arrow/api.h>
#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <unordered_set>

#include <spdlog/spdlog.h>

namespace gui {

void DataInputDialog::RenderDataProfilingTab() {
    const ImGuiStyle& style = ImGui::GetStyle();
    ImVec4 accent = style.Colors[ImGuiCol_HeaderActive];

    ImGui::Spacing();
    ImGui::TextColored(accent, "DATA QUALITY ANALYSIS");
    ImGui::Separator();
    ImGui::Spacing();

    if (data_load_state_ != DataLoadState::InMemory) {
        ImGui::TextDisabled("No data loaded in memory.");
        ImGui::TextDisabled("Click Apply to load data, then return here to analyze.");
        return;
    }

    if (profile_in_progress_) {
        ImGui::TextDisabled("Computing statistics...");
    } else {
        if (ImGui::Button("Analyze Data", ImVec2(150, 30))) {
            ComputeDataProfile();
        }
    }

    if (!profile_computed_) {
        ImGui::Spacing();
        ImGui::TextDisabled("Click 'Analyze Data' to compute column statistics");
        return;
    }

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    ImGui::TextColored(accent, "Data Quality Score");
    ImGui::Spacing();

    ImVec4 score_color;
    if (data_quality_score_ >= 0.8f)
        score_color = ImVec4(0.2f, 0.8f, 0.2f, 1.0f);
    else if (data_quality_score_ >= 0.5f)
        score_color = ImVec4(0.9f, 0.7f, 0.0f, 1.0f);
    else
        score_color = ImVec4(0.9f, 0.3f, 0.3f, 1.0f);

    ImGui::PushStyleColor(ImGuiCol_PlotHistogram, score_color);
    ImGui::ProgressBar(data_quality_score_, ImVec2(-1, 20));
    ImGui::PopStyleColor();
    ImGui::Text("%.1f%% complete data (%.1f%% missing values)",
        data_quality_score_ * 100.0f, (1.0f - data_quality_score_) * 100.0f);

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    ImGui::TextColored(accent, "COLUMN STATISTICS");
    ImGui::Spacing();

    if (column_stats_.empty()) {
        ImGui::TextDisabled("No column statistics available");
        return;
    }

    if (ImGui::BeginTable("ColumnStats", 8,
        ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg | ImGuiTableFlags_ScrollX |
        ImGuiTableFlags_ScrollY | ImGuiTableFlags_Resizable,
        ImVec2(0, ImGui::GetContentRegionAvail().y - 30))) {

        ImGui::TableSetupColumn("Column", ImGuiTableColumnFlags_WidthFixed, 120.0f);
        ImGui::TableSetupColumn("Type", ImGuiTableColumnFlags_WidthFixed, 70.0f);
        ImGui::TableSetupColumn("Count", ImGuiTableColumnFlags_WidthFixed, 60.0f);
        ImGui::TableSetupColumn("Unique", ImGuiTableColumnFlags_WidthFixed, 60.0f);
        ImGui::TableSetupColumn("Missing", ImGuiTableColumnFlags_WidthFixed, 70.0f);
        ImGui::TableSetupColumn("Min", ImGuiTableColumnFlags_WidthFixed, 80.0f);
        ImGui::TableSetupColumn("Max", ImGuiTableColumnFlags_WidthFixed, 80.0f);
        ImGui::TableSetupColumn("Mean", ImGuiTableColumnFlags_WidthFixed, 80.0f);
        ImGui::TableHeadersRow();

        for (const auto& stat : column_stats_) {
            ImGui::TableNextRow();

            ImGui::TableSetColumnIndex(0);
            ImGui::TextUnformatted(stat.name.c_str());

            ImGui::TableSetColumnIndex(1);
            ImGui::TextUnformatted(stat.dtype.c_str());

            ImGui::TableSetColumnIndex(2);
            ImGui::Text("%zu", stat.count);

            ImGui::TableSetColumnIndex(3);
            ImGui::Text("%zu", stat.unique_count);

            ImGui::TableSetColumnIndex(4);
            if (stat.null_percentage > 5.0f) {
                ImGui::TextColored(ImVec4(0.9f, 0.4f, 0.4f, 1.0f), "%.1f%%", stat.null_percentage);
            } else {
                ImGui::Text("%.1f%%", stat.null_percentage);
            }

            ImGui::TableSetColumnIndex(5);
            if (stat.dtype == "numeric" || stat.dtype == "integer") {
                ImGui::Text("%.2f", stat.min_val);
            } else {
                ImGui::TextDisabled("-");
            }

            ImGui::TableSetColumnIndex(6);
            if (stat.dtype == "numeric" || stat.dtype == "integer") {
                ImGui::Text("%.2f", stat.max_val);
            } else {
                ImGui::TextDisabled("-");
            }

            ImGui::TableSetColumnIndex(7);
            if (stat.dtype == "numeric" || stat.dtype == "integer") {
                ImGui::Text("%.2f", stat.mean);
            } else {
                ImGui::TextDisabled("-");
            }
        }

        ImGui::EndTable();
    }
}

void DataInputDialog::ComputeDataProfile() {
    profile_in_progress_ = true;
    column_stats_.clear();

    auto& registry = cyxwiz::DataRegistry::Instance();
    auto dataset = registry.GetArrowDataset(loaded_dataset_name_);

    if (!dataset) {
        profile_in_progress_ = false;
        spdlog::warn("Cannot compute profile: dataset '{}' not found", loaded_dataset_name_);
        return;
    }

    auto column_names = dataset->GetColumnNames();
    size_t total_nulls = 0;
    size_t total_values = 0;

    for (const auto& col_name : column_names) {
        ColumnStats stat;
        stat.name = col_name;

        auto arrow_table = dataset->GetArrowTable();
        if (arrow_table) {
            auto schema = arrow_table->schema();
            auto field = schema->GetFieldByName(col_name);
            if (field) {
                auto type = field->type();
                if (type->id() == arrow::Type::INT64 || type->id() == arrow::Type::INT32 ||
                    type->id() == arrow::Type::INT16 || type->id() == arrow::Type::INT8 ||
                    type->id() == arrow::Type::UINT64 || type->id() == arrow::Type::UINT32 ||
                    type->id() == arrow::Type::UINT16 || type->id() == arrow::Type::UINT8) {
                    stat.dtype = "integer";
                } else if (type->id() == arrow::Type::DOUBLE || type->id() == arrow::Type::FLOAT ||
                           type->id() == arrow::Type::HALF_FLOAT) {
                    stat.dtype = "numeric";
                } else if (type->id() == arrow::Type::STRING || type->id() == arrow::Type::LARGE_STRING) {
                    stat.dtype = "string";
                } else if (type->id() == arrow::Type::BOOL) {
                    stat.dtype = "boolean";
                } else {
                    stat.dtype = "other";
                }
            }

            auto column = arrow_table->GetColumnByName(col_name);
            if (column) {
                stat.count = static_cast<size_t>(column->length());
                stat.null_count = static_cast<size_t>(column->null_count());
                stat.null_percentage = stat.count > 0
                    ? (static_cast<float>(stat.null_count) / static_cast<float>(stat.count)) * 100.0f
                    : 0.0f;

                total_nulls += stat.null_count;
                total_values += stat.count;

                if (stat.dtype == "numeric" || stat.dtype == "integer") {
                    double sum = 0.0;
                    double min_v = std::numeric_limits<double>::max();
                    double max_v = std::numeric_limits<double>::lowest();
                    size_t valid_count = 0;

                    auto process_value = [&](double val) {
                        sum += val;
                        min_v = std::min(min_v, val);
                        max_v = std::max(max_v, val);
                        valid_count++;
                    };

                    for (int chunk_idx = 0; chunk_idx < column->num_chunks(); chunk_idx++) {
                        auto chunk = column->chunk(chunk_idx);
                        if (auto double_arr = std::dynamic_pointer_cast<arrow::DoubleArray>(chunk)) {
                            for (int64_t i = 0; i < double_arr->length(); i++) {
                                if (!double_arr->IsNull(i)) process_value(double_arr->Value(i));
                            }
                        } else if (auto float_arr = std::dynamic_pointer_cast<arrow::FloatArray>(chunk)) {
                            for (int64_t i = 0; i < float_arr->length(); i++) {
                                if (!float_arr->IsNull(i)) process_value(static_cast<double>(float_arr->Value(i)));
                            }
                        } else if (auto int64_arr = std::dynamic_pointer_cast<arrow::Int64Array>(chunk)) {
                            for (int64_t i = 0; i < int64_arr->length(); i++) {
                                if (!int64_arr->IsNull(i)) process_value(static_cast<double>(int64_arr->Value(i)));
                            }
                        } else if (auto int32_arr = std::dynamic_pointer_cast<arrow::Int32Array>(chunk)) {
                            for (int64_t i = 0; i < int32_arr->length(); i++) {
                                if (!int32_arr->IsNull(i)) process_value(static_cast<double>(int32_arr->Value(i)));
                            }
                        } else if (auto int16_arr = std::dynamic_pointer_cast<arrow::Int16Array>(chunk)) {
                            for (int64_t i = 0; i < int16_arr->length(); i++) {
                                if (!int16_arr->IsNull(i)) process_value(static_cast<double>(int16_arr->Value(i)));
                            }
                        } else if (auto int8_arr = std::dynamic_pointer_cast<arrow::Int8Array>(chunk)) {
                            for (int64_t i = 0; i < int8_arr->length(); i++) {
                                if (!int8_arr->IsNull(i)) process_value(static_cast<double>(int8_arr->Value(i)));
                            }
                        } else if (auto uint64_arr = std::dynamic_pointer_cast<arrow::UInt64Array>(chunk)) {
                            for (int64_t i = 0; i < uint64_arr->length(); i++) {
                                if (!uint64_arr->IsNull(i)) process_value(static_cast<double>(uint64_arr->Value(i)));
                            }
                        } else if (auto uint32_arr = std::dynamic_pointer_cast<arrow::UInt32Array>(chunk)) {
                            for (int64_t i = 0; i < uint32_arr->length(); i++) {
                                if (!uint32_arr->IsNull(i)) process_value(static_cast<double>(uint32_arr->Value(i)));
                            }
                        }
                    }

                    if (valid_count > 0) {
                        stat.min_val = min_v;
                        stat.max_val = max_v;
                        stat.mean = sum / static_cast<double>(valid_count);
                    }
                }

                std::unordered_set<std::string> unique_vals;
                if (stat.count <= 10000) {
                    for (int chunk_idx = 0; chunk_idx < column->num_chunks(); chunk_idx++) {
                        auto chunk = column->chunk(chunk_idx);
                        if (auto str_array = std::dynamic_pointer_cast<arrow::StringArray>(chunk)) {
                            for (int64_t i = 0; i < str_array->length(); i++) {
                                if (!str_array->IsNull(i)) {
                                    unique_vals.insert(str_array->GetString(i));
                                }
                            }
                        }
                    }
                }
                stat.unique_count = unique_vals.empty() ? 0 : unique_vals.size();
            }
        }

        column_stats_.push_back(stat);
    }

    if (total_values > 0) {
        data_quality_score_ = 1.0f - (static_cast<float>(total_nulls) / static_cast<float>(total_values));
    } else {
        data_quality_score_ = 1.0f;
    }

    profile_computed_ = true;
    profile_in_progress_ = false;

    spdlog::info("Data profile computed: {} columns, quality score: {:.1f}%",
        column_stats_.size(), data_quality_score_ * 100.0f);
}

} // namespace gui
