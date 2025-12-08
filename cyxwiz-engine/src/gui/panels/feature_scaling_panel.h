#pragma once

#include "../panel.h"
#include <cyxwiz/data_transform.h>
#include <vector>
#include <string>
#include <memory>

namespace cyxwiz {

class DataTableRegistry;
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

    void SetDataRegistry(std::shared_ptr<DataTableRegistry> registry) { data_registry_ = registry; }

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

    void LoadSelectedData();
    void ApplyScaling();
    void ExportData();

    std::shared_ptr<DataTableRegistry> data_registry_;
    std::shared_ptr<DataTable> current_table_;

    std::vector<std::string> available_tables_;
    std::vector<std::string> column_names_;
    int selected_table_idx_ = -1;

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
    bool show_comparison_ = false;

    std::string status_message_;
};

} // namespace cyxwiz
