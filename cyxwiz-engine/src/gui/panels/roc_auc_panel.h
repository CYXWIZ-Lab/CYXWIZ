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

class ROCAUCPanel : public Panel {
public:
    ROCAUCPanel();
    ~ROCAUCPanel() override = default;

    void Render() override;

    void SetDataRegistry(std::shared_ptr<DataTableRegistry> registry) {
        data_registry_ = registry;
    }

private:
    void RenderDataSelector();
    void RenderROCCurve();
    void RenderThresholdAnalysis();
    void RenderMetrics();
    void LoadSelectedData();
    void ComputeROC();

    std::shared_ptr<DataTableRegistry> data_registry_;
    std::shared_ptr<DataTable> current_table_;

    std::vector<std::string> available_tables_;
    std::vector<std::string> column_names_;
    int selected_table_idx_ = -1;
    int true_label_column_ = 0;
    int score_column_ = 1;

    ROCCurveData roc_result_;
    BinaryMetrics current_metrics_;
    double selected_threshold_ = 0.5;

    std::string status_message_;
};

} // namespace cyxwiz
