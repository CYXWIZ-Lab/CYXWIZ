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

class LearningCurvesPanel : public Panel {
public:
    LearningCurvesPanel();
    ~LearningCurvesPanel() override = default;

    void Render() override;

    void SetDataRegistry(std::shared_ptr<DataTableRegistry> registry) {
        data_registry_ = registry;
    }

    // Allow setting learning curve data from training
    void SetLearningCurveData(const LearningCurveData& data) {
        result_ = data;
    }

    // Add a single point to the learning curve
    void AddDataPoint(int train_size, double train_score, double val_score);

    // Clear all data
    void ClearData();

private:
    void RenderDataSelector();
    void RenderLearningCurve();
    void RenderBiasVarianceAnalysis();
    void RenderRecommendations();
    void LoadSelectedData();

    std::shared_ptr<DataTableRegistry> data_registry_;
    std::shared_ptr<DataTable> current_table_;

    std::vector<std::string> available_tables_;
    int selected_table_idx_ = -1;

    // Learning curve data
    LearningCurveData result_;

    // Manual data entry
    std::vector<int> manual_train_sizes_;
    std::vector<double> manual_train_scores_;
    std::vector<double> manual_val_scores_;

    std::string status_message_;
};

} // namespace cyxwiz
