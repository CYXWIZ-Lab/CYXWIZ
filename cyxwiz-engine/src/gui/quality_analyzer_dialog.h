#pragma once

#include "node_config_dialog.h"

#include "../core/data_registry.h"
#include "../core/quality_analyzer_presentation.h"

#include <atomic>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>

namespace gui {

// Quality Analyzer (TOFIX140): the checks on the left; on the right the
// verdict of the cached analysis (or "Not analyzed yet", or the progress of
// an analysis running in Task View). The presentation model words it all;
// this file only draws it.
class QualityAnalyzerDialog : public NodeConfigDialog {
public:
    explicit QualityAnalyzerDialog(MLNode* node);
    ~QualityAnalyzerDialog() override;

    void Apply() override;
    void Reset() override;
    ImVec2 GetDefaultSize() const override { return ImVec2(1000, 760); }
    bool IsBusy() const override;

protected:
    void RenderContent() override;
    float FooterHeight() const override;
    void RenderFooter(bool& should_close) override;

private:
    // An analysis running on a task worker; shared with it, so closing the
    // dialog never leaves the worker writing into freed memory.
    struct Job {
        std::atomic<size_t> done{0};
        std::atomic<size_t> total{0};
        std::atomic<bool> cancel{false};
        std::atomic<bool> finished{false};
        std::mutex mutex;
        std::string error;
        std::string key;
        bool cancelled = false;
    };

    void ResolveContext();
    void Rejudge();
    void StartAnalysis();
    void PollJob();
    void RenderChecks(bool locked);
    void RenderResults();
    void RenderNotAnalyzed();
    void RenderAnalyzing();
    void RenderReason(const cyxwiz::QualityReasonRow& row);
    void RenderHistograms();
    void RenderDetails();
    unsigned int Thumbnail(size_t index);
    void ReleaseThumbnails();

    cyxwiz::image::ImageQualityChecks checks_;
    bool context_resolved_ = false;
    std::string context_error_;  // why there is nothing to analyze
    std::string dataset_name_;
    std::optional<cyxwiz::DataRegistry::ImageDatasetEntry> entry_;
    int width_ = 0;
    int height_ = 0;
    std::string key_;  // the measurements these images and this size need

    std::shared_ptr<const cyxwiz::ImageQualityAnalysis> analysis_;
    bool stale_ = false;  // analysis_ was made for other images or another size
    std::optional<cyxwiz::QualityAnalyzerView> view_;
    std::string checks_error_;
    std::string job_error_;

    std::shared_ptr<Job> job_;
    uint64_t task_id_ = 0;
    std::string device_label_;

    bool show_details_ = false;
    std::shared_ptr<cyxwiz::Dataset> thumbnail_source_;
    std::map<size_t, unsigned int> thumbnails_;
};

}  // namespace gui
