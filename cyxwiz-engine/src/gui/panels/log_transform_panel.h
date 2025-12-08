#pragma once

#include "../panel.h"
#include <cyxwiz/data_transform.h>
#include <vector>
#include <string>
#include <memory>

namespace cyxwiz {

class DataTableRegistry;
class DataTable;

class LogTransformPanel : public Panel {
public:
    LogTransformPanel();
    void Render() override;

    void SetDataRegistry(std::shared_ptr<DataTableRegistry> registry) { data_registry_ = registry; }

private:
    void RenderDataSelector();
    void RenderSettings();
    void RenderPreview();
    void RenderResults();

    void LoadSelectedData();
    void ApplyLogTransform();

    std::shared_ptr<DataTableRegistry> data_registry_;
    std::shared_ptr<DataTable> current_table_;

    std::vector<std::string> available_tables_;
    std::vector<std::string> column_names_;
    int selected_table_idx_ = -1;

    // Column selection
    std::vector<bool> selected_columns_;

    // Settings
    int log_base_ = 0;  // 0=natural, 1=log10, 2=log2
    bool use_log1p_ = true;

    // Results
    TransformResult transform_result_;
    std::vector<ColumnStats> original_stats_;
    std::vector<ColumnStats> transformed_stats_;
    bool can_transform_ = true;

    std::string status_message_;
};

} // namespace cyxwiz
