#pragma once

// Explore > Profile for an image dataset (TOFIX178 1.3e), and the Data Input
// dialog's Profile tab: draws core/image_profile_presentation from the
// dataset's index table, the graph's Resize (the live compile) and the
// prepared sets in cache/images (read on a background task).

#include "../../core/image_profile_presentation.h"
#include "../../core/prepared_image_set.h"

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>

namespace cyxwiz {

class LiveGraphCompile;

class ImageProfileView {
public:
    ImageProfileView();
    ~ImageProfileView();

    void SetDataset(const std::string& dataset);
    // The canvas graph's compile, for its Resize and fixed transforms (may be null).
    void SetLiveCompile(const LiveGraphCompile* live_compile) { live_compile_ = live_compile; }
    void Render();

private:
    void Refresh();       // the index rows, when the index changed
    void ListSets();      // the prepared sets, on a task
    void MeasureStats();  // prepares the set when needed, then its channel statistics, on a task
    void MeasureQuality();  // the Quality Analyzer's measurements at the Resize size, on a task
    void ShowQuality(const std::shared_ptr<const struct ImageQualityAnalysis>& analysis,
                     const struct ImageQualityVerdict& verdict);
    std::string GraphIdentity() const;

    std::string dataset_;
    const LiveGraphCompile* live_compile_ = nullptr;
    image_profile::Input input_;
    image_profile::View view_;
    const void* index_seen_ = nullptr;   // the index table the rows came from
    std::string graph_seen_;             // the Resize + fixed transforms the sets were matched against
    uint64_t sets_task_ = 0;
    uint64_t stats_task_ = 0;
    std::string stats_for_;              // the graph steps the statistics are of
    // Progress the task writes and the UI thread reads; shared, so a task that
    // outlives the view writes into it safely.
    struct Progress {
        std::mutex mutex;
        image_profile::Input::StatsStep step = image_profile::Input::StatsStep::Idle;
        size_t done = 0, total = 0;
        std::string device;
    };
    std::shared_ptr<Progress> stats_progress_ = std::make_shared<Progress>();
    uint64_t quality_task_ = 0;
    struct QualityProgress {
        std::mutex mutex;
        size_t done = 0, total = 0;
    };
    std::shared_ptr<QualityProgress> quality_progress_ = std::make_shared<QualityProgress>();
    std::shared_ptr<int> alive_ = std::make_shared<int>(0);
};

}  // namespace cyxwiz
