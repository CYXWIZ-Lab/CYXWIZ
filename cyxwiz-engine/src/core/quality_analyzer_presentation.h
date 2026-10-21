#pragma once

// What the Quality Analyzer dialog shows for an analysis and a set of checks
// (TOFIX140): the headline, one row per reason with its worst examples, a
// class-imbalance warning, three histograms with the cut marked, and the
// details. Pure data in and out; the dialog only draws it.

#include "image_quality_analysis.h"

#include <cstddef>
#include <string>
#include <utility>
#include <vector>

namespace cyxwiz {

struct QualityExample {
    size_t index = 0;     // image index in the analysis
    std::string value;    // under the thumbnail, e.g. "blur 250"
    std::string tooltip;  // file name, class and value
};

struct QualityReasonRow {
    std::string title;  // "Blurry"
    size_t count = 0;
    std::string rule;   // "blur below 400"
    std::vector<QualityExample> examples;  // worst first
};

struct QualityHistogram {
    std::string title;
    std::vector<float> heights;  // per bin, 0 to 1 of the tallest
    std::vector<bool> cut;       // per bin: inside the rejected range
    std::string low, cut_label, high;
};

struct QualityAnalyzerView {
    std::string headline;   // "278 of 300 pass"
    std::string subline;    // "22 images left out of training (7.3%)"
    std::string imbalance;  // empty, or the warning
    std::vector<QualityReasonRow> reasons;  // only reasons that reject something
    std::string footnote;
    std::vector<QualityHistogram> histograms;
    std::vector<std::pair<std::string, std::string>> details;
};

constexpr size_t kQualityHistogramBins = 20;

QualityAnalyzerView BuildQualityAnalyzerView(const ImageQualityAnalysis& analysis,
                                             const ImageQualityVerdict& verdict,
                                             const image::ImageQualityChecks& checks,
                                             const std::string& dataset_name,
                                             size_t examples_per_reason = 5);

}  // namespace cyxwiz
