#pragma once

// Data Studio > ML readiness (TOFIX148 step 3, approved mockup v8, D5): the
// splits of a Data Split (or catalog datasets named <stem>_train / ...) side
// by side: roles, split sizes, leakage across splits, text length and class
// balance (core/ml_readiness). Measured in the background when the job opens
// and when a split's data changes; roles are set in Explore > Profile.

#include "../../core/data_studio_presentation.h"
#include "../../core/ml_readiness.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace cyxwiz {

class ReadinessView {
public:
    ReadinessView() = default;
    ~ReadinessView();

    // The sets the graph and the catalog offer (rebuilt twice a second) and
    // the graph's Data Loader balance mode.
    void SetSplitSets(const std::vector<studio::SplitSet>& sets, const std::string& balance);
    // The job tab became visible: measure again (roles may have changed in Profile).
    void Activate();
    void Render();

    bool IsRunning() const { return task_id_ != 0; }
    // The verdict of the set shown ("No leakage found"), "" before a result.
    std::string Verdict() const { return result_ ? view_.verdict : std::string(); }
    studio::Tone VerdictTone() const { return view_.verdict_tone; }

    // "Set roles in Profile" opens Explore > Profile on this dataset.
    std::function<void(const std::string&)> on_profile;
    // "Show the Data Split" on the canvas.
    std::function<void(int)> on_show_node;

private:
    void Measure();
    std::string Signature() const;  // the shown set's datasets and their generations

    std::vector<studio::SplitSet> sets_;
    std::string balance_;
    size_t chosen_ = 0;
    std::string measured_;          // Signature() of the result (or the running task)
    std::shared_ptr<readiness::Facts> result_;
    readiness::View view_;
    std::string error_;
    uint64_t task_id_ = 0;
    std::shared_ptr<int> alive_ = std::make_shared<int>(0);
};

}  // namespace cyxwiz
