#pragma once

#include "../panel.h"
#include "../dataset_source.h"
#include <cyxwiz/model_evaluation.h>
#include <imgui.h>
#include <functional>
#include <utility>
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
    // One heatmap grid; cell_at(true, predicted) gives the colour position [0, 1] and the text.
    void RenderGrid(const std::function<std::pair<float, std::string>(int, int)>& cell_at);
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
};

} // namespace cyxwiz
