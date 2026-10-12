#pragma once

#include "../panel.h"
#include "../dataset_source.h"
#include "../../core/table_class_labels.h"
#include <cyxwiz/model_evaluation.h>
#include <imgui.h>
#include <vector>
#include <string>
#include <memory>
#include <thread>
#include <atomic>

namespace cyxwiz {

class DataTable;

class CrossValidationPanel : public Panel {
public:
    CrossValidationPanel();
    ~CrossValidationPanel() override;

    void Render() override;

private:
    void RenderDataSelector();
    void RenderCVSettings();
    void RenderResults();
    void RenderFoldDetails();
    std::string ClassName(int cls) const;  // "delta", or "Class 1" for numeric labels
    void UseTable(std::shared_ptr<DataTable> table);  // a picked dataset was read
    void RunCrossValidation();

    std::shared_ptr<DataTable> current_table_;

    std::vector<std::string> column_names_;
    std::vector<bool> selected_features_;
    ui::DatasetSource source_;  // the picked dataset (Data Input or a node result)
    int target_column_ = 0;

    // CV settings
    int n_folds_ = 5;
    bool stratified_ = true;
    bool shuffle_ = true;
    int random_seed_ = 42;

    // Results
    CrossValidationResult result_;
    std::vector<std::pair<std::vector<int>, std::vector<int>>> fold_splits_;
    TableClassLabels class_labels_;  // target classes of a stratified split; empty otherwise

    std::thread compute_thread_;
    std::atomic<bool> is_computing_{false};
    std::shared_ptr<int> alive_ = std::make_shared<int>(0);  // results posted to the UI thread stop once the panel is gone
    std::string status_message_;
};

} // namespace cyxwiz
