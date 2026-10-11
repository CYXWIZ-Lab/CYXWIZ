#pragma once

#include "../panel.h"
#include "../dataset_source.h"
#include <cyxwiz/data_transform.h>
#include <vector>
#include <string>
#include <memory>

namespace cyxwiz {

class DataTable;

class LogTransformPanel : public Panel {
public:
    LogTransformPanel();
    void Render() override;

private:
    void RenderDataSelector();
    void RenderSettings();
    void RenderPreview();
    void RenderResults();

    void UseTable(std::shared_ptr<DataTable> table);  // a picked dataset was read
    void ApplyLogTransform();

    std::shared_ptr<DataTable> current_table_;

    std::vector<std::string> column_names_;
    ui::DatasetSource source_;  // the picked dataset (Data Input or a node result)

    // Column selection
    std::vector<bool> selected_columns_;

    // Settings
    int log_base_ = 0;  // 0=natural, 1=log10, 2=log2
    bool use_log1p_ = true;

    // Results
    TransformResult transform_result_;
    std::vector<ColumnStats> original_stats_;
    std::vector<ColumnStats> transformed_stats_;
    std::vector<int> transformed_columns_;  // table column of each transformed column (data skips empty ones)
    bool can_transform_ = true;

    std::string status_message_;
};

} // namespace cyxwiz
