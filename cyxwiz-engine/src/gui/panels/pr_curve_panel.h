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

class PRCurvePanel : public Panel {
public:
    PRCurvePanel();
    ~PRCurvePanel() override = default;

    void Render() override;

private:
    void RenderDataSelector();
    void RenderPRCurve();
    void RenderF1Curve();
    void RenderMetrics();
    void UseTable(std::shared_ptr<DataTable> table);  // a picked dataset was read
    void ComputePRCurve();

    std::shared_ptr<DataTable> current_table_;

    std::vector<std::string> column_names_;
    ui::DatasetSource source_;  // the picked dataset (Data Input or a node result)
    int true_label_column_ = 0;
    int score_column_ = 1;

    PRCurveData pr_result_;
    std::vector<double> f1_scores_;  // F1 at each threshold

    std::string status_message_;
};

} // namespace cyxwiz
