#pragma once

#include "../panel.h"
#include <cyxwiz/data_transform.h>
#include <vector>
#include <string>
#include <memory>

namespace cyxwiz {

class DataTableRegistry;
class DataTable;

class BoxCoxPanel : public Panel {
public:
    BoxCoxPanel();
    void Render() override;

    void SetDataRegistry(std::shared_ptr<DataTableRegistry> registry) { data_registry_ = registry; }

private:
    void RenderDataSelector();
    void RenderSettings();
    void RenderLambdaPlot();
    void RenderResults();

    void LoadSelectedData();
    void FindOptimalLambda();
    void ApplyBoxCox();

    std::shared_ptr<DataTableRegistry> data_registry_;
    std::shared_ptr<DataTable> current_table_;

    std::vector<std::string> available_tables_;
    std::vector<std::string> column_names_;
    int selected_table_idx_ = -1;
    int selected_column_ = 0;

    // Settings
    bool auto_lambda_ = true;
    float manual_lambda_ = 1.0f;
    bool use_yeo_johnson_ = false;  // For negative values

    // Lambda search results
    BoxCoxLambdaResult lambda_result_;

    // Transform results
    TransformResult transform_result_;
    ColumnStats original_stats_;
    ColumnStats transformed_stats_;

    // Normality test
    NormalityTestResult orig_normality_;
    NormalityTestResult trans_normality_;

    bool can_transform_ = true;
    std::string status_message_;
};

} // namespace cyxwiz
