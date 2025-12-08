#pragma once

#include "../panel.h"
#include <cyxwiz/data_transform.h>
#include <vector>
#include <string>
#include <memory>

namespace cyxwiz {

class DataTableRegistry;
class DataTable;

class StandardizationPanel : public Panel {
public:
    StandardizationPanel();
    void Render() override;

    void SetDataRegistry(std::shared_ptr<DataTableRegistry> registry) { data_registry_ = registry; }

private:
    void RenderDataSelector();
    void RenderSettings();
    void RenderPreview();
    void RenderResults();

    void LoadSelectedData();
    void ApplyStandardization();

    std::shared_ptr<DataTableRegistry> data_registry_;
    std::shared_ptr<DataTable> current_table_;

    std::vector<std::string> available_tables_;
    std::vector<std::string> column_names_;
    int selected_table_idx_ = -1;

    // Column selection
    std::vector<bool> selected_columns_;

    // Options
    bool use_robust_scaling_ = false;

    // Results
    TransformResult transform_result_;
    std::vector<ColumnStats> original_stats_;
    std::vector<ColumnStats> transformed_stats_;

    std::string status_message_;
};

} // namespace cyxwiz
