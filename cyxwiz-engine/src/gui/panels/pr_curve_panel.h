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

class PRCurvePanel : public Panel {
public:
    PRCurvePanel();
    ~PRCurvePanel() override = default;

    void Render() override;

    void SetDataRegistry(std::shared_ptr<DataTableRegistry> registry) {
        data_registry_ = registry;
    }

private:
    void RenderDataSelector();
    void RenderPRCurve();
    void RenderF1Curve();
    void RenderMetrics();
    void LoadSelectedData();
    void ComputePRCurve();

    std::shared_ptr<DataTableRegistry> data_registry_;
    std::shared_ptr<DataTable> current_table_;

    std::vector<std::string> available_tables_;
    std::vector<std::string> column_names_;
    int selected_table_idx_ = -1;
    int true_label_column_ = 0;
    int score_column_ = 1;

    PRCurveData pr_result_;
    std::vector<double> f1_scores_;  // F1 at each threshold

    std::string status_message_;
};

} // namespace cyxwiz
