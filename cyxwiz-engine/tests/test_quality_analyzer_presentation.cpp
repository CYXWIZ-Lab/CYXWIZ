// Quality Analyzer dialog view model (TOFIX140).
//
// A ten-image analysis in two classes: the headline and share, one row per
// reason with its worst examples first, the class-imbalance warning, the
// footnote, the histograms' cut ranges and the details.
#include "../src/core/quality_analyzer_presentation.h"

#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

namespace {

namespace img = cyxwiz::image;

int checks = 0;

void Check(bool condition, const std::string& message) {
    ++checks;
    if (!condition) {
        std::cerr << "FAIL: " << message << "\n";
        std::exit(1);
    }
}

bool Has(const std::vector<std::pair<std::string, std::string>>& details, const std::string& key,
         const std::string& value) {
    for (const auto& [k, v] : details) {
        if (k == key) return v == value;
    }
    return false;
}

cyxwiz::ImageQualityAnalysis Analysis() {
    cyxwiz::ImageQualityAnalysis a;
    a.key = "00ff";
    a.folder_path = "D:/data/catdog";
    a.width = 64;
    a.height = 64;
    a.class_names = {"cat", "dog"};
    a.device = "CUDA - GeForce GTX 1050 Ti";
    a.seconds = 2.34;
    // cat: 0-4, dog: 5-9.
    const std::vector<img::ImageQualityMetrics> m = {
        {1000, 120, 0.20f, 1}, {250, 120, 0.20f, 2}, {300, 120, 0.20f, 3}, {900, 40, 0.05f, 4}, {1100, 240, 0.20f, 5},
        {1200, 120, 0.20f, 6}, {1300, 130, 0.21f, 7}, {800, 125, 0.22f, 8}, {950, 118, 0.19f, 9}, {1000, 120, 0.20f, 1},
    };
    for (size_t i = 0; i < m.size(); ++i) {
        a.files.push_back("D:/data/catdog/" + std::string(i < 5 ? "cat/cat." : "dog/dog.") + std::to_string(i) + ".jpg");
        a.labels.push_back(i < 5 ? 0 : 1);
        a.metrics.push_back(m[i]);
    }
    return a;
}

cyxwiz::ImageQualityVerdict Verdict(const cyxwiz::ImageQualityAnalysis& a, const img::ImageQualityChecks& c) {
    // Image 9 repeats image 0 (the device search is tested in test_image_quality).
    cyxwiz::ImageQualityVerdict v;
    v.duplicate_of.assign(a.metrics.size(), -1);
    v.duplicate_of[9] = 0;
    v.reasons = img::JudgeImageQuality(a.metrics, v.duplicate_of, c);
    v.class_total = {5, 5};
    v.class_rejected = {0, 0};
    for (size_t i = 0; i < v.reasons.size(); ++i) {
        const uint32_t r = v.reasons[i];
        if (!r) continue;
        ++v.rejected;
        ++v.class_rejected[static_cast<size_t>(a.labels[i])];
        if (r & img::kQualityBlurry) ++v.blurry;
        if (r & img::kQualityDark) ++v.dark;
        if (r & img::kQualityBright) ++v.bright;
        if (r & img::kQualityLowContrast) ++v.low_contrast;
        if (r & img::kQualityDuplicate) ++v.duplicates;
        if (r & (r - 1)) ++v.multiple;
    }
    return v;
}

void CheckDefaults() {
    const auto a = Analysis();
    const img::ImageQualityChecks c;
    const auto v = Verdict(a, c);
    const auto view = cyxwiz::BuildQualityAnalyzerView(a, v, c, "catdog_small");

    Check(view.headline == "5 of 10 pass", "headline: " + view.headline);
    Check(view.subline == "5 images left out of training (50.0%)", "subline: " + view.subline);
    Check(view.imbalance == "Rejections lean to one class: cat loses 4 of 5 (80.0%), dog 1 of 5 (20.0%).",
          "imbalance: " + view.imbalance);

    Check(view.reasons.size() == 5, "five reasons reject something");
    Check(view.reasons[0].title == "Blurry" && view.reasons[0].count == 2 && view.reasons[0].rule == "blur below 700",
          "blurry row");
    Check(view.reasons[0].examples.size() == 2 && view.reasons[0].examples[0].index == 1 &&
              view.reasons[0].examples[0].value == "blur 250" &&
              view.reasons[0].examples[0].tooltip == "cat.1 (cat), blur 250",
          "blurriest first, with its value and file");
    Check(view.reasons[1].title == "Low contrast" && view.reasons[1].examples[0].value == "0.050", "low contrast row");
    Check(view.reasons[2].title == "Too dark" && view.reasons[2].examples[0].value == "mean 40", "too dark row");
    Check(view.reasons[3].title == "Too bright" && view.reasons[3].rule == "brightness above 220", "too bright row");
    Check(view.reasons[4].title == "Near-duplicates" && view.reasons[4].examples[0].value == "like cat.0",
          "duplicate row names the image it repeats");
    Check(view.footnote == "1 image fails more than one check; each counts once in the 5.", "footnote: " + view.footnote);

    Check(view.histograms.size() == 3, "three histograms");
    const auto& blur = view.histograms[0];
    Check(blur.title == "Blur (all 10)" && blur.high == "1300" && blur.cut_label == "cut 700", "blur histogram range");
    Check(blur.heights.size() == cyxwiz::kQualityHistogramBins && blur.cut[0] && blur.cut[10] && !blur.cut[11],
          "blur bins below 700 are marked (bin width 65)");
    const auto& light = view.histograms[1];
    Check(light.cut[0] && !light.cut[4] && light.cut[19] && light.cut_label == "cut 50 / 220", "brightness cut both ends");

    Check(Has(view.details, "Measured on", "CUDA - GeForce GTX 1050 Ti"), "device in details");
    Check(Has(view.details, "Images", "10 measured at 64 x 64"), "size in details");
    Check(Has(view.details, "Time", "2.3 s"), "time in details");
    Check(Has(view.details, "Dataset", "catdog_small (2 classes)"), "dataset in details");
}

void CheckLenient() {
    const auto a = Analysis();
    img::ImageQualityChecks c;
    c.blur = c.brightness = c.contrast = c.duplicates = false;
    const auto view = cyxwiz::BuildQualityAnalyzerView(a, Verdict(a, c), c, "catdog_small");
    Check(view.headline == "All 10 pass" && view.subline == "No images left out of training", "nothing rejected");
    Check(view.reasons.empty() && view.imbalance.empty() && view.footnote.empty(), "no rows, warning or footnote");
    for (const auto& h : view.histograms) {
        Check(h.cut_label == "off", h.title + ": a check that is off says so");
        for (bool cut : h.cut) Check(!cut, h.title + ": nothing marked");
    }

    img::ImageQualityChecks dup_only;
    dup_only.blur = dup_only.brightness = dup_only.contrast = false;
    auto clean = a;
    clean.metrics[9].hash = 99;
    auto v = Verdict(clean, dup_only);
    v.duplicate_of[9] = -1;
    v.reasons = img::JudgeImageQuality(clean.metrics, v.duplicate_of, dup_only);
    v.rejected = 0;
    v.duplicates = 0;
    v.multiple = 0;
    v.class_rejected = {0, 0};
    Check(cyxwiz::BuildQualityAnalyzerView(clean, v, dup_only, "x").footnote == "Near-duplicates: none found.",
          "no duplicates is said");
}

}  // namespace

int main() {
    CheckDefaults();
    CheckLenient();
    std::cout << "quality analyzer presentation passed (" << checks << " checks)\n";
    return 0;
}
