#pragma once

#include "../panel.h"
#include "../dataset_source.h"
#include <cyxwiz/data_transform.h>
#include <vector>
#include <string>
#include <memory>

namespace cyxwiz {

class DataTable;

class NormalizationPanel : public Panel {
public:
    NormalizationPanel();
    void Render() override;

private:
    void RenderDataSelector();
    void RenderSettings();
    void RenderPreview();
    void RenderResults();

    void UseTable(std::shared_ptr<DataTable> table);  // a picked dataset was read
    void ApplyNormalization();

    std::shared_ptr<DataTable> current_table_;

    std::vector<std::string> column_names_;
    ui::DatasetSource source_;  // the picked dataset (Data Input or a node result)

    // Column selection
    std::vector<bool> selected_columns_;

    // Range settings
    float range_min_ = 0.0f;
    float range_max_ = 1.0f;

    // Results
    TransformResult transform_result_;
    std::vector<ColumnStats> original_stats_;
    std::vector<ColumnStats> transformed_stats_;

    std::string status_message_;
};

} // namespace cyxwiz
