#pragma once

#include "../panel.h"
#include "../dataset_source.h"
#include <cyxwiz/model_evaluation.h>
#include <imgui.h>
#include <vector>
#include <string>
#include <memory>

namespace cyxwiz {

class DataTable;

class ConfusionMatrixPanel : public Panel {
public:
    ConfusionMatrixPanel();
    ~ConfusionMatrixPanel() override = default;

    void Render() override;

private:
    void RenderDataSelector();
    void RenderMatrixHeatmap();
    void RenderMetricsTable();
    void RenderNormalizedMatrix();
    void UseTable(std::shared_ptr<DataTable> table);  // a picked dataset was read
    void ComputeMatrix();

    std::shared_ptr<DataTable> current_table_;

    std::vector<std::string> column_names_;
    ui::DatasetSource source_;  // the picked dataset (Data Input or a node result)
    int true_label_column_ = 0;
    int pred_label_column_ = 1;

    ConfusionMatrixData result_;
    bool normalize_ = false;

    std::string status_message_;

    static constexpr int MAX_CLASSES = 20;
    static const ImU32 HEATMAP_COLORS[5];
};

} // namespace cyxwiz
