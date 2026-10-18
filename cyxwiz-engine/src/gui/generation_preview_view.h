#pragma once

// The Generations view (TOFIX159 A4, design approved 2026-10-18): a training
// run's generation previews, shared by the Training Dashboard (live, while the
// run trains) and Data Studio > Results (finished runs), so both show the same
// words. Latest: one card per prompt. Compare two points side by side. One
// prompt over time. A rail says how the samples were made. The words come
// from core/generation_preview_presentation.

#include "../core/generation_preview_presentation.h"

#include <filesystem>
#include <set>
#include <string>
#include <vector>

namespace cyxwiz {

class GenerationPreviewView {
public:
    // The run's generation_previews folder; "" shows nothing.
    void SetFolder(const std::filesystem::path& folder);
    const std::filesystem::path& Folder() const { return folder_; }
    // Reads new reports every 2 s while shown.
    void Render();
    size_t ReportCount() const { return reports_.size(); }

private:
    void Refresh(bool force);
    void RenderToolbar();
    void RenderSummary(const GenerationPreviewReport& report);
    void RenderLatest(const GenerationPreviewReport& report);
    void RenderCompare();
    void RenderHistory();
    void RenderRail(const GenerationPreviewReport& report);
    void RenderOutput(const GenerationPreviewSample& sample, const GenerationPreviewReport& report, float height);

    std::filesystem::path folder_;
    std::vector<GenerationPreviewReport> reports_;
    std::vector<std::string> skipped_;
    std::string stamp_;
    double polled_ = -10.0;
    int view_ = 0;        // 0 latest, 1 compare, 2 one prompt over time
    int shown_ = -1;      // report shown by Latest; -1 = the newest
    int left_ = -1, right_ = -1;  // Compare; -1 = the two newest
    int prompt_ = 0;      // One prompt over time
    std::set<std::string> open_prompts_;  // "What the model saw" open per card
    std::string export_note_;
    bool export_failed_ = false;
};

}  // namespace cyxwiz
