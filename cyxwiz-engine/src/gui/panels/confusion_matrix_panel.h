#pragma once

#include "../panel.h"
#include <cyxwiz/model_evaluation.h>
#include <imgui.h>
#include <vector>
#include <string>
#include <memory>

namespace cyxwiz {

class DataTableRegistry;
class DataTable;

class ConfusionMatrixPanel : public Panel {
public:
    ConfusionMatrixPanel();
    ~ConfusionMatrixPanel() override = default;

    void Render() override;

    void SetDataRegistry(std::shared_ptr<DataTableRegistry> registry) {
        data_registry_ = registry;
    }

private:
    void RenderDataSelector();
    void RenderMatrixHeatmap();
    void RenderMetricsTable();
    void RenderNormalizedMatrix();
    void LoadSelectedData();
    void ComputeMatrix();

    std::shared_ptr<DataTableRegistry> data_registry_;
    std::shared_ptr<DataTable> current_table_;

    std::vector<std::string> available_tables_;
    std::vector<std::string> column_names_;
    int selected_table_idx_ = -1;
    int true_label_column_ = 0;
    int pred_label_column_ = 1;

    ConfusionMatrixData result_;
    bool normalize_ = false;

    std::string status_message_;

    static constexpr int MAX_CLASSES = 20;
    static const ImU32 HEATMAP_COLORS[5];
};

} // namespace cyxwiz
