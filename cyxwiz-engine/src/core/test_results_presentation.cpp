#include "test_results_presentation.h"

#include "compiled_node_presentation.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>

namespace cyxwiz::test_report {

namespace {

std::string Fixed(double value, int decimals) {
    char text[64];
    std::snprintf(text, sizeof(text), "%.*f", decimals, value);
    return text;
}

std::string Percent(double fraction) {
    return Fixed(fraction * 100.0, 2) + "%";
}

std::string Lower(std::string text) {
    std::transform(text.begin(), text.end(), text.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return text;
}

Tone AccuracyTone(double accuracy) {
    if (accuracy >= 0.85) return Tone::Ok;
    if (accuracy >= 0.5) return Tone::Warning;
    return Tone::Error;
}

std::string Count(long long n) {
    return FormatCompiledCount(n);
}

void AddComparison(View& v, const TestingMetrics& a, const TestingMetrics& b) {
    struct Row {
        const char* name;
        double previous, latest;
        bool percent, lower_is_better, show;
    };
    const bool classification = !b.regression_mode && !b.causal_lm_mode;
    const Row rows[] = {
        {"Loss", a.test_loss, b.test_loss, false, true, true},
        {"Accuracy", a.test_accuracy, b.test_accuracy, true, false, !b.regression_mode},
        {"Macro precision", a.macro_precision, b.macro_precision, true, false, classification},
        {"Macro recall", a.macro_recall, b.macro_recall, true, false, classification},
        {"Macro F1", a.macro_f1, b.macro_f1, true, false, classification},
        {"Weighted F1", a.weighted_f1, b.weighted_f1, true, false, classification},
        {"MAE", a.test_mae, b.test_mae, false, true, b.regression_mode},
        {"RMSE", a.test_rmse, b.test_rmse, false, true, b.regression_mode},
        {"Samples", static_cast<double>(a.total_samples), static_cast<double>(b.total_samples), false, false, true},
        {"Time (s)", a.total_time_seconds, b.total_time_seconds, false, true, true},
    };
    for (const Row& r : rows) {
        if (!r.show) continue;
        ComparisonRow out;
        out.metric = r.name;
        const double scale = r.percent ? 100.0 : 1.0;
        const auto format = [&](double x) {
            if (std::string(r.name) == "Samples") return Count(static_cast<long long>(x));
            return r.percent ? Fixed(x * scale, 2) + "%" : Fixed(x, 4);
        };
        out.previous = format(r.previous);
        out.latest = format(r.latest);
        const double delta = (r.latest - r.previous) * scale;
        char text[64];
        std::snprintf(text, sizeof(text), r.percent ? "%+.2f%%" : "%+.4f", delta);
        out.change = text;
        if (delta != 0.0) out.change_tone = (r.lower_is_better ? delta < 0.0 : delta > 0.0) ? Tone::Ok : Tone::Warning;
        v.comparison.push_back(std::move(out));
    }
    if (a.partial || b.partial)
        v.comparison_note = a.partial && b.partial ? "Both runs were quick tests (first batches only)."
                            : b.partial            ? "The latest run was a quick test (first batches only)."
                                                   : "The previous run was a quick test (first batches only).";
}

}  // namespace

bool IsFailedRun(const TestingMetrics& m) {
    return !m.is_complete && !m.status_message.empty();
}

std::string ClassName(const TestingMetrics& m, int index) {
    if (index >= 0 && static_cast<size_t>(index) < m.per_class_metrics.size() &&
        !m.per_class_metrics[static_cast<size_t>(index)].class_name.empty())
        return m.per_class_metrics[static_cast<size_t>(index)].class_name;
    if (index >= 0 && static_cast<size_t>(index) < m.confusion_matrix.class_names.size() &&
        !m.confusion_matrix.class_names[static_cast<size_t>(index)].empty())
        return m.confusion_matrix.class_names[static_cast<size_t>(index)];
    return "Class " + std::to_string(index);
}

View BuildView(const TestingMetrics& m, const TestingMetrics* previous) {
    View v;
    v.kind = m.causal_lm_mode ? "causal_lm" : m.regression_mode ? "regression" : m.metric_learning_mode ? "metric_learning"
                                                                                                      : "classification";
    v.classification = !m.causal_lm_mode && !m.regression_mode;
    v.subtitle = Count(m.total_samples) + (m.causal_lm_mode ? " windows" : m.regression_mode ? " forecast windows" : " test samples") +
                 ", " + Fixed(m.total_time_seconds, 2) + " s";
    if (m.partial)
        v.note = "Quick test: the first " + std::to_string(m.batch_limit) + " of " + std::to_string(m.total_batches) +
                 " batches. Train > Test > Run Test covers the whole test split.";

    if (m.causal_lm_mode) {
        v.headline = "Token accuracy " + Percent(m.test_accuracy);
        v.headline_tone = AccuracyTone(m.test_accuracy);
        v.chips = {{"loss " + Fixed(m.test_loss, 4), Tone::Neutral},
                   {Count(static_cast<long long>(m.total_target_values)) + " target tokens", Tone::Neutral}};
    } else if (m.regression_mode) {
        v.headline = "MAE " + Fixed(m.test_mae, 4);
        v.headline_tone = Tone::Neutral;
        v.chips = {{"RMSE " + Fixed(m.test_rmse, 4), Tone::Neutral}};
        if (const double r2 = RSquared(m); std::isfinite(r2)) v.chips.push_back({"R\xC2\xB2 " + Fixed(r2, 4), Tone::Neutral});
        v.chips.insert(v.chips.end(), {{"loss " + Fixed(m.test_loss, 4), Tone::Neutral},
                   {Count(static_cast<long long>(m.total_target_values)) + " target values", Tone::Neutral}});
    } else {
        v.headline = std::string(m.metric_learning_mode ? "1-NN accuracy " : "Accuracy ") + Percent(m.test_accuracy);
        v.headline_tone = AccuracyTone(m.test_accuracy);
        if (m.metric_learning_mode) {
            const std::string nn = "Each test row takes the class of its nearest other test row in the embedding space.";
            v.note = v.note.empty() ? nn : nn + " " + v.note;
        }
        v.chips = {{"loss " + Fixed(m.test_loss, 4), Tone::Neutral},
                   {"macro F1 " + Fixed(m.macro_f1, 4), Tone::Neutral},
                   {"weighted F1 " + Fixed(m.weighted_f1, 4), Tone::Neutral}};
        if (m.roc && m.roc->success) v.chips.push_back({"AUC " + Fixed(m.roc->auc, 4), Tone::Neutral});
        v.tiles = {{Fixed(m.macro_precision, 4), "macro precision"},
                   {Fixed(m.macro_recall, 4), "macro recall"},
                   {Fixed(m.macro_f1, 4), "macro F1"},
                   {Fixed(m.weighted_f1, 4), "weighted F1"}};
    }
    if (m.samples_per_second > 0.0f)
        v.chips.push_back({Count(static_cast<long long>(m.samples_per_second + 0.5f)) + " samples/s", Tone::Neutral});

    if (m.retrieval_metrics) {
        v.facts.emplace_back("Recall@" + std::to_string(m.retrieval_metrics->k), Fixed(m.retrieval_metrics->recall_at_k, 4));
        v.facts.emplace_back("Mean reciprocal rank", Fixed(m.retrieval_metrics->mean_reciprocal_rank, 4));
    }
    if (m.pair_metrics) {
        const auto& p = *m.pair_metrics;
        char threshold[32];
        std::snprintf(threshold, sizeof(threshold), "%g", p.threshold);
        v.facts.emplace_back("Pair accuracy", Fixed(p.accuracy, 4) + " (threshold " + threshold + ", " +
                                                  Count(static_cast<long long>(p.pair_count)) + " pairs)");
        v.facts.emplace_back("Mean distance", "same class " + Fixed(p.positive_distance_mean, 4) + ", other class " +
                                                  Fixed(p.negative_distance_mean, 4));
    }

    if (v.classification) {
        for (size_t i = 0; i < m.per_class_metrics.size(); ++i) {
            const auto& c = m.per_class_metrics[i];
            ClassRow row;
            row.index = static_cast<int>(i);
            row.name = ClassName(m, static_cast<int>(i));
            row.precision = c.precision;
            row.recall = c.recall;
            row.f1 = c.f1_score;
            row.found = c.true_positives;
            row.false_alarms = c.false_positives;
            row.support = c.support;
            row.never_predicted = c.true_positives == 0 && c.false_positives == 0;
            v.classes.push_back(std::move(row));
        }
        if (!v.classes.empty()) {
            const auto best = std::max_element(v.classes.begin(), v.classes.end(),
                                               [](const ClassRow& a, const ClassRow& b) { return a.f1 < b.f1; });
            v.facts.emplace_back("Best class", best->name + ", F1 " + Fixed(best->f1, 4) + " (" + Count(best->found) +
                                                   " of " + Count(best->support) + " found)");
            const float worst_f1 = std::min_element(v.classes.begin(), v.classes.end(), [](const ClassRow& a, const ClassRow& b) {
                                       return a.f1 < b.f1;
                                   })->f1;
            std::string worst;
            bool all_never = true;
            size_t worst_count = 0;
            for (const auto& c : v.classes) {
                if (c.f1 != worst_f1) continue;
                worst += (worst.empty() ? "" : ", ") + c.name;
                all_never = all_never && c.never_predicted;
                ++worst_count;
            }
            v.facts.emplace_back(worst_count == 1 ? "Worst class" : "Worst classes",
                                 worst + ", F1 " + Fixed(worst_f1, 4) + (all_never ? ": never predicted" : ""));
            v.facts.emplace_back("Classes", std::to_string(v.classes.size()));
        }

        const auto& conf = m.confusion_matrix;
        const int n = conf.num_classes;
        if (n > 0 && conf.matrix.size() == static_cast<size_t>(n)) {
            int largest = 0, li = -1, lj = -1;
            for (int i = 0; i < n; ++i) {
                v.matrix.names.push_back(ClassName(m, i));
                v.matrix.counts.push_back(conf.matrix[static_cast<size_t>(i)]);
                int total = 0;
                for (int j = 0; j < n && static_cast<size_t>(j) < conf.matrix[static_cast<size_t>(i)].size(); ++j) {
                    const int value = conf.matrix[static_cast<size_t>(i)][static_cast<size_t>(j)];
                    total += value;
                    v.matrix.max_count = std::max(v.matrix.max_count, value);
                    if (i != j && value > largest) {
                        largest = value;
                        li = i;
                        lj = j;
                    }
                }
                v.matrix.row_totals.push_back(total);
            }
            if (li >= 0)
                v.matrix.largest_confusion = "The largest confusion: true " + ClassName(m, li) + " predicted as " +
                                             ClassName(m, lj) + " (" + Count(largest) + ").";
        }

        const size_t samples = std::min(m.predictions.size(), m.ground_truth.size());
        for (size_t i = 0; i < samples; ++i) {
            if (m.predictions[i] == m.ground_truth[i]) ++v.correct;
            else v.mistakes.push_back(static_cast<int>(i));
        }
        // The most confident mistakes first: the rows worth a look.
        if (m.confidences.size() >= samples)
            std::stable_sort(v.mistakes.begin(), v.mistakes.end(), [&m](int a, int b) {
                return m.confidences[static_cast<size_t>(a)] > m.confidences[static_cast<size_t>(b)];
            });
    }

    // A regression's kept rows: the bias (mean of predicted - true over every
    // target value) and the rows by their mean |error|, largest first.
    if (m.regression_mode && m.regression_width > 0 && !m.regression_truth.empty() &&
        m.regression_truth.size() == m.regression_predictions.size()) {
        const size_t width = static_cast<size_t>(m.regression_width);
        const size_t rows = m.regression_truth.size() / width;
        std::vector<double> row_error(rows, 0.0);
        double bias = 0.0;
        for (size_t i = 0; i < rows * width; ++i) {
            const double e = static_cast<double>(m.regression_predictions[i]) - m.regression_truth[i];
            bias += e;
            row_error[i / width] += std::abs(e) / static_cast<double>(width);
        }
        bias /= static_cast<double>(rows * width);
        v.facts.emplace_back("Bias", (bias >= 0.0 ? "+" : "") + Fixed(bias, 4) +
                                         (bias > 0.0 ? " (predicts high)" : bias < 0.0 ? " (predicts low)" : ""));
        v.largest_errors.resize(rows);
        for (size_t i = 0; i < rows; ++i) v.largest_errors[i] = static_cast<int>(i);
        std::stable_sort(v.largest_errors.begin(), v.largest_errors.end(),
                         [&row_error](int a, int b) { return row_error[static_cast<size_t>(a)] > row_error[static_cast<size_t>(b)]; });
    }

    if (previous && !IsFailedRun(*previous)) AddComparison(v, *previous, m);
    return v;
}

ThresholdFigures AtThreshold(const ROCCurveData& roc, int positives, int negatives, double threshold) {
    ThresholdFigures f;
    f.threshold = threshold;
    const size_t points = std::min({roc.fpr.size(), roc.tpr.size(), roc.thresholds.size()});
    if (!roc.success || points == 0 || positives <= 0 || negatives <= 0) return f;
    // Thresholds fall along the curve; the last point at or above `threshold`
    // counts every score >= threshold as positive (point 0, +inf: none).
    size_t at = 0;
    for (size_t i = 1; i < points; ++i)
        if (roc.thresholds[i] >= threshold) at = i;
    f.tp = static_cast<int>(std::lround(roc.tpr[at] * positives));
    f.fp = static_cast<int>(std::lround(roc.fpr[at] * negatives));
    f.fn = positives - f.tp;
    f.tn = negatives - f.fp;
    const double tp = f.tp, fp = f.fp, tn = f.tn, fn = f.fn;
    f.precision = tp + fp > 0 ? tp / (tp + fp) : 0.0;
    f.recall = tp / positives;
    f.specificity = tn / negatives;
    f.f1 = f.precision + f.recall > 0.0 ? 2.0 * f.precision * f.recall / (f.precision + f.recall) : 0.0;
    f.balanced_accuracy = (f.recall + f.specificity) / 2.0;
    const double d = std::sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn));
    f.mcc = d > 0.0 ? (tp * tn - fp * fn) / d : 0.0;
    f.accuracy = (tp + tn) / static_cast<double>(positives + negatives);
    f.ok = true;
    return f;
}

double BestF1Threshold(const ROCCurveData& roc, int positives, int negatives) {
    const size_t points = std::min({roc.fpr.size(), roc.tpr.size(), roc.thresholds.size()});
    double best = NAN, best_f1 = -1.0;
    for (size_t i = 1; i < points; ++i) {
        const ThresholdFigures f = AtThreshold(roc, positives, negatives, roc.thresholds[i]);
        if (f.ok && f.f1 > best_f1) {
            best_f1 = f.f1;
            best = roc.thresholds[i];
        }
    }
    return best;
}

double RSquared(const TestingMetrics& m) {
    const size_t width = static_cast<size_t>(std::max(m.regression_width, 0));
    if (!m.regression_mode || width == 0 || m.regression_truth.empty() ||
        m.regression_truth.size() != m.regression_predictions.size())
        return NAN;
    const size_t rows = m.regression_truth.size() / width;
    double total = 0.0;
    for (size_t t = 0; t < width; ++t) {
        double mean = 0.0;
        for (size_t i = 0; i < rows; ++i) mean += m.regression_truth[i * width + t];
        mean /= static_cast<double>(rows);
        double sse = 0.0, sst = 0.0;
        for (size_t i = 0; i < rows; ++i) {
            const double y = m.regression_truth[i * width + t];
            const double e = y - m.regression_predictions[i * width + t];
            sse += e * e;
            sst += (y - mean) * (y - mean);
        }
        if (!(sst > 0.0)) return NAN;
        total += 1.0 - sse / sst;
    }
    return total / static_cast<double>(width);
}

std::vector<ClassRow> SortClasses(std::vector<ClassRow> rows, ClassSort by, bool ascending, const std::string& filter) {
    if (!filter.empty()) {
        const std::string needle = Lower(filter);
        rows.erase(std::remove_if(rows.begin(), rows.end(),
                                  [&](const ClassRow& r) { return Lower(r.name).find(needle) == std::string::npos; }),
                   rows.end());
    }
    const auto key = [by](const ClassRow& r) -> double {
        switch (by) {
            case ClassSort::Name: return r.index;
            case ClassSort::Precision: return r.precision;
            case ClassSort::Recall: return r.recall;
            case ClassSort::F1: return r.f1;
            case ClassSort::Found: return r.found;
            case ClassSort::FalseAlarms: return r.false_alarms;
            case ClassSort::Support: return r.support;
        }
        return r.index;
    };
    std::stable_sort(rows.begin(), rows.end(), [&](const ClassRow& a, const ClassRow& b) {
        if (by == ClassSort::Name) return ascending ? a.name < b.name : b.name < a.name;
        return ascending ? key(a) < key(b) : key(b) < key(a);
    });
    return rows;
}

}  // namespace cyxwiz::test_report
