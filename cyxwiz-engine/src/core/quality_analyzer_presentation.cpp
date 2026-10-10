#include "quality_analyzer_presentation.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <functional>

namespace cyxwiz {

namespace {

std::string Fixed(double value, int decimals) {
    char text[64];
    std::snprintf(text, sizeof(text), "%.*f", decimals, value);
    return text;
}

// Thresholds as typed: 400, 0.12, 50.
std::string Number(float value) {
    char text[64];
    std::snprintf(text, sizeof(text), "%g", static_cast<double>(value));
    return text;
}

std::string Percent(size_t part, size_t whole) {
    return whole == 0 ? "0%" : Fixed(100.0 * static_cast<double>(part) / static_cast<double>(whole), 1) + "%";
}

std::string Plural(size_t count, const char* one, const char* many) {
    return std::to_string(count) + " " + (count == 1 ? one : many);
}

std::string Stem(const std::string& file) {
    return std::filesystem::path(file).stem().string();
}

std::string ClassOf(const ImageQualityAnalysis& analysis, size_t index) {
    const int label = index < analysis.labels.size() ? analysis.labels[index] : -1;
    return label >= 0 && static_cast<size_t>(label) < analysis.class_names.size()
        ? analysis.class_names[static_cast<size_t>(label)] : std::string();
}

QualityReasonRow Row(const ImageQualityAnalysis& analysis, const ImageQualityVerdict& verdict, uint32_t reason,
                     std::string title, std::string rule, size_t limit,
                     const std::function<bool(size_t, size_t)>& worse,
                     const std::function<std::string(size_t)>& value) {
    QualityReasonRow row;
    row.title = std::move(title);
    row.rule = std::move(rule);
    std::vector<size_t> hits;
    for (size_t i = 0; i < verdict.reasons.size(); ++i) {
        if (verdict.reasons[i] & reason) hits.push_back(i);
    }
    row.count = hits.size();
    std::stable_sort(hits.begin(), hits.end(), worse);
    if (hits.size() > limit) hits.resize(limit);
    for (size_t i : hits) {
        const std::string cls = ClassOf(analysis, i);
        QualityExample example;
        example.index = i;
        example.value = value(i);
        example.tooltip = Stem(analysis.files[i]) + (cls.empty() ? "" : " (" + cls + ")") + ", " + example.value;
        row.examples.push_back(std::move(example));
    }
    return row;
}

QualityHistogram Histogram(std::string title, const std::vector<float>& values, float high,
                           const std::function<bool(float)>& rejected, std::string low_label,
                           std::string cut_label, std::string high_label) {
    QualityHistogram h;
    h.title = std::move(title);
    h.low = std::move(low_label);
    h.cut_label = std::move(cut_label);
    h.high = std::move(high_label);
    std::vector<size_t> counts(kQualityHistogramBins, 0);
    const float width = high / static_cast<float>(kQualityHistogramBins);
    for (float v : values) {
        const auto bin = static_cast<size_t>((std::max)(0.0f, v) / width);
        ++counts[(std::min)(bin, kQualityHistogramBins - 1)];
    }
    const size_t tallest = (std::max)(size_t{1}, *std::max_element(counts.begin(), counts.end()));
    for (size_t b = 0; b < kQualityHistogramBins; ++b) {
        h.heights.push_back(static_cast<float>(counts[b]) / static_cast<float>(tallest));
        h.cut.push_back(rejected((static_cast<float>(b) + 0.5f) * width));
    }
    return h;
}

}  // namespace

QualityAnalyzerView BuildQualityAnalyzerView(const ImageQualityAnalysis& analysis,
                                             const ImageQualityVerdict& verdict,
                                             const image::ImageQualityChecks& checks,
                                             const std::string& dataset_name,
                                             size_t examples_per_reason) {
    QualityAnalyzerView view;
    const size_t total = analysis.files.size();
    const auto& m = analysis.metrics;

    if (verdict.rejected == 0) {
        view.headline = "All " + std::to_string(total) + " pass";
        view.subline = "No images left out of training";
    } else {
        view.headline = std::to_string(total - verdict.rejected) + " of " + std::to_string(total) + " pass";
        view.subline = Plural(verdict.rejected, "image", "images") + " left out of training (" +
                       Percent(verdict.rejected, total) + ")";
    }

    // Warn when one class loses a clearly larger share than another.
    if (verdict.rejected > 0 && verdict.class_total.size() >= 2) {
        size_t most = 0, least = 0;
        const auto rate = [&](size_t c) {
            return verdict.class_total[c] == 0 ? 0.0
                : static_cast<double>(verdict.class_rejected[c]) / static_cast<double>(verdict.class_total[c]);
        };
        for (size_t c = 1; c < verdict.class_total.size(); ++c) {
            if (rate(c) > rate(most)) most = c;
            if (rate(c) < rate(least)) least = c;
        }
        if (rate(most) - rate(least) >= 0.02 && rate(most) >= 1.5 * rate(least)) {
            const auto share = [&](size_t c) {
                return std::to_string(verdict.class_rejected[c]) + " of " + std::to_string(verdict.class_total[c]) +
                       " (" + Percent(verdict.class_rejected[c], verdict.class_total[c]) + ")";
            };
            view.imbalance = "Rejections lean to one class: " + analysis.class_names[most] + " loses " + share(most) +
                             ", " + analysis.class_names[least] + " " + share(least) + ".";
        }
    }

    const size_t limit = examples_per_reason;
    const auto by = [&](auto key, bool ascending) {
        return [&m, key, ascending](size_t a, size_t b) {
            return ascending ? key(m[a]) < key(m[b]) : key(m[a]) > key(m[b]);
        };
    };
    const auto blur = [](const image::ImageQualityMetrics& x) { return x.blur; };
    const auto light = [](const image::ImageQualityMetrics& x) { return x.brightness; };
    const auto spread = [](const image::ImageQualityMetrics& x) { return x.contrast; };
    std::vector<QualityReasonRow> rows = {
        Row(analysis, verdict, image::kQualityBlurry, "Blurry", "blur below " + Number(checks.blur_min), limit,
            by(blur, true), [&](size_t i) { return "blur " + Fixed(m[i].blur, 0); }),
        Row(analysis, verdict, image::kQualityLowContrast, "Low contrast", "contrast below " + Number(checks.contrast_min),
            limit, by(spread, true), [&](size_t i) { return Fixed(m[i].contrast, 3); }),
        Row(analysis, verdict, image::kQualityDark, "Too dark", "brightness below " + Number(checks.brightness_min),
            limit, by(light, true), [&](size_t i) { return "mean " + Fixed(m[i].brightness, 0); }),
        Row(analysis, verdict, image::kQualityBright, "Too bright", "brightness above " + Number(checks.brightness_max),
            limit, by(light, false), [&](size_t i) { return "mean " + Fixed(m[i].brightness, 0); }),
        Row(analysis, verdict, image::kQualityDuplicate, "Near-duplicates",
            "within " + std::to_string(checks.duplicate_bits) + " bits of an earlier image", limit,
            [](size_t a, size_t b) { return a < b; },
            [&](size_t i) { return "like " + Stem(analysis.files[static_cast<size_t>(verdict.duplicate_of[i])]); }),
    };
    for (auto& row : rows) {
        if (row.count > 0) view.reasons.push_back(std::move(row));
    }

    std::vector<std::string> notes;
    if (checks.duplicates && verdict.duplicates == 0) notes.push_back("Near-duplicates: none found.");
    if (verdict.multiple > 0) {
        notes.push_back(Plural(verdict.multiple, "image fails", "images fail") +
                        " more than one check; each counts once in the " + std::to_string(verdict.rejected) + ".");
    }
    for (const auto& note : notes) view.footnote += (view.footnote.empty() ? "" : " ") + note;

    std::vector<float> blurs, means, spreads;
    for (const auto& x : m) {
        blurs.push_back(x.blur);
        means.push_back(x.brightness);
        spreads.push_back(x.contrast);
    }
    const float blur_high = std::ceil((blurs.empty() ? 1.0f : *std::max_element(blurs.begin(), blurs.end())) / 100.0f) * 100.0f;
    const float spread_high = std::ceil((spreads.empty() ? 0.1f : *std::max_element(spreads.begin(), spreads.end())) * 10.0f) / 10.0f;
    view.histograms.push_back(Histogram(
        "Blur (all " + std::to_string(total) + ")", blurs, (std::max)(blur_high, 100.0f),
        [&](float v) { return checks.blur && v < checks.blur_min; }, "0",
        checks.blur ? "cut " + Number(checks.blur_min) : "off", Number((std::max)(blur_high, 100.0f))));
    view.histograms.push_back(Histogram(
        "Brightness", means, 255.0f,
        [&](float v) { return checks.brightness && (v < checks.brightness_min || v > checks.brightness_max); }, "0",
        checks.brightness ? "cut " + Number(checks.brightness_min) + " / " + Number(checks.brightness_max) : "off",
        "255"));
    view.histograms.push_back(Histogram(
        "Contrast", spreads, (std::max)(spread_high, 0.1f),
        [&](float v) { return checks.contrast && v < checks.contrast_min; }, "0",
        checks.contrast ? "cut " + Number(checks.contrast_min) : "off", Number((std::max)(spread_high, 0.1f))));

    const std::string layout = std::to_string(analysis.class_names.size()) +
                               (analysis.class_names.size() == 1 ? " class" : " classes");
    view.details = {
        {"Dataset", dataset_name + " (" + layout + ")"},
        {"Folder", analysis.folder_path},
        {"Measured on", analysis.device.empty() ? std::string("unknown") : analysis.device},
        {"Images", std::to_string(total) + " measured at " + std::to_string(analysis.width) + " x " +
                       std::to_string(analysis.height)},
        {"Time", Fixed(analysis.seconds, 1) + " s"},
        {"Blur", "variance of the 3x3 Laplacian of the luminance"},
        {"Brightness / contrast", "mean / standard deviation of the luminance"},
        {"Duplicates", "8x8 difference hash, " + std::to_string(checks.duplicate_bits) + " bits apart"},
        {"Analysis", analysis.key},
    };
    return view;
}

}  // namespace cyxwiz
