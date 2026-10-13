#pragma once

// Data Studio Profile tab (TOFIX134 P3.5, approved board 14; replaces
// Analyze): one profile of the picked dataset (core/dataset_profiler) and the
// column roles. Roles are edited here only (owner answer 2): contract (the
// graph) > roles set here (saved with the project) > inferred. Profiling runs
// in the background when a dataset is picked or re-loaded.

#include "../../core/dataset_contract.h"
#include "../../core/dataset_profiler.h"

#include <cstdint>
#include <memory>
#include <string>

namespace cyxwiz {

class ProfileView {
public:
    ProfileView();
    ~ProfileView();

    void Render();
    void SetActiveDataset(const std::string& dataset_name);
    void Reprofile();
    bool IsRunning() const { return task_id_ != 0; }
    float Progress() const { return progress_; }

private:
    void RenderOverview();
    void RenderColumns(float width);
    void RenderDetail();
    void RebuildContract();
    std::string SourceKey() const;

    std::string dataset_;           // catalog name
    std::string shown_;             // as the graph calls it
    std::string source_path_;
    std::string target_column_;     // the graph's target
    uint64_t generation_ = 0;       // of the data profiled
    std::shared_ptr<DatasetProfile> profile_;
    DatasetContract contract_;
    uint64_t task_id_ = 0;
    float progress_ = 0;
    std::string progress_text_;
    std::string error_;
    std::string profiled_at_;
    int selected_ = -1;
    char missing_buf_[256] = {};
    std::string missing_for_;
    std::string save_error_;
    std::shared_ptr<int> alive_ = std::make_shared<int>(0);
};

}  // namespace cyxwiz
