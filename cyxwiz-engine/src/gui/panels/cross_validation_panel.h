#pragma once

#include "../panel.h"
#include <cyxwiz/model_evaluation.h>
#include <imgui.h>
#include <vector>
#include <string>
#include <memory>
#include <thread>
#include <atomic>

namespace cyxwiz {

class DataTableRegistry;
class DataTable;

class CrossValidationPanel : public Panel {
public:
    CrossValidationPanel();
    ~CrossValidationPanel() override;

    void Render() override;

    void SetDataRegistry(std::shared_ptr<DataTableRegistry> registry) {
        data_registry_ = registry;
    }

private:
    void RenderDataSelector();
    void RenderCVSettings();
    void RenderResults();
    void RenderFoldDetails();
    void LoadSelectedData();
    void RunCrossValidation();

    std::shared_ptr<DataTableRegistry> data_registry_;
    std::shared_ptr<DataTable> current_table_;

    std::vector<std::string> available_tables_;
    std::vector<std::string> column_names_;
    std::vector<bool> selected_features_;
    int selected_table_idx_ = -1;
    int target_column_ = 0;

    // CV settings
    int n_folds_ = 5;
    bool stratified_ = true;
    bool shuffle_ = true;
    int random_seed_ = 42;

    // Results
    CrossValidationResult result_;
    std::vector<std::pair<std::vector<int>, std::vector<int>>> fold_splits_;

    std::thread compute_thread_;
    std::atomic<bool> is_computing_{false};
    std::string status_message_;
};

} // namespace cyxwiz
