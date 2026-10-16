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

class ROCAUCPanel : public Panel {
public:
    ROCAUCPanel();
    ~ROCAUCPanel() override = default;

    void Render() override;

private:
    void RenderDataSelector();
    void RenderROCCurve();
    void RenderThresholdAnalysis();
    void RenderMetrics();
    void UseTable(std::shared_ptr<DataTable> table);  // a picked dataset was read
    void ComputeROC();

    std::shared_ptr<DataTable> current_table_;

    std::vector<std::string> column_names_;
    ui::DatasetSource source_;  // the picked dataset (Data Input or a node result)
    int true_label_column_ = 0;
    int score_column_ = 1;

    ROCCurveData roc_result_;
    BinaryMetrics current_metrics_;
    double selected_threshold_ = 0.5;

    std::string status_message_;
};

} // namespace cyxwiz
