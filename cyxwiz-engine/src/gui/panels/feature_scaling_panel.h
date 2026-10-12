#pragma once

#include "../panel.h"
#include "../dataset_source.h"
#include <cyxwiz/data_transform.h>
#include <vector>
#include <string>
#include <memory>

namespace cyxwiz {

class DataTable;

enum class ScalingMethod {
    MinMax,
    ZScore,
    Robust,
    MaxAbs,
    Quantile
};

class FeatureScalingPanel : public Panel {
public:
    FeatureScalingPanel();
    void Render() override;

private:
    void RenderDataSelector();
    void RenderMethodTabs();
    void RenderMinMaxTab();
    void RenderZScoreTab();
    void RenderRobustTab();
    void RenderMaxAbsTab();
    void RenderQuantileTab();
    void RenderPreview();
    void RenderComparison();

    void UseTable(std::shared_ptr<DataTable> table);  // a picked dataset was read
    void ApplyScaling();
    void ExportData();

    std::shared_ptr<DataTable> current_table_;

    std::vector<std::string> column_names_;
    ui::DatasetSource source_;  // the picked dataset (Data Input or a node result)

    // Column selection
    std::vector<bool> selected_columns_;

    // Method selection
    ScalingMethod selected_method_ = ScalingMethod::MinMax;

    // MinMax settings
    float minmax_range_min_ = 0.0f;
    float minmax_range_max_ = 1.0f;

    // Quantile settings
    int quantile_n_quantiles_ = 1000;
    bool quantile_normal_output_ = false;

    // Results
    TransformResult transform_result_;
    std::vector<ColumnStats> original_stats_;
    std::vector<ColumnStats> transformed_stats_;

    // Comparison data for multiple methods
    std::map<ScalingMethod, TransformResult> method_results_;
    std::map<ScalingMethod, ColumnStats> method_stats_;  // first column of each result, computed once
    bool show_comparison_ = false;
    bool select_comparison_tab_ = false;  // bring the Comparison tab forward after Compare All

    std::string status_message_;
};

} // namespace cyxwiz
