#include "cross_validation_presentation.h"

#include "compiled_node_presentation.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <sstream>

namespace cyxwiz::folds_view {

namespace {

std::string Fixed(double v, int decimals) {
    char text[48];
    std::snprintf(text, sizeof(text), "%.*f", decimals, v);
    return text;
}

std::string Percent(double fraction, int decimals = 1) { return Fixed(fraction * 100.0, decimals) + "%"; }

// A number to about four significant figures ("318.2", "0.1609", "12,345").
std::string Num(double v) {
    if (!std::isfinite(v)) return "-";
    const double a = std::abs(v);
    if (a >= 10000.0) return FormatCompiledCount(static_cast<long long>(std::llround(v)));
    char text[48];
    std::snprintf(text, sizeof(text), a >= 100.0 ? "%.1f" : a >= 1.0 ? "%.3f" : "%.4f", v);
    return text;
}

std::string Duration(double seconds) {
    if (seconds < 60.0) return Fixed(seconds, seconds < 10.0 ? 1 : 0) + " s";
    const long long s = std::llround(seconds);
    if (s < 3600) return std::to_string(s / 60) + " min " + std::to_string(s % 60) + " s";
    return std::to_string(s / 3600) + " h " + std::to_string((s % 3600) / 60) + " min";
}

double Pick(const CrossValidationScores& s, CurveScore score) {
    switch (score) {
        case CurveScore::Accuracy: return s.accuracy;
        case CurveScore::MacroF1: return s.macro_f1;
        case CurveScore::Loss: return s.loss;
        case CurveScore::Mae: return s.mae;
        case CurveScore::Rmse: return s.rmse;
        case CurveScore::R2: return s.r2;
    }
    return 0.0;
}

bool IsFraction(CurveScore score) { return score == CurveScore::Accuracy || score == CurveScore::MacroF1; }
bool HigherIsBetter(CurveScore score) {
    return score == CurveScore::Accuracy || score == CurveScore::MacroF1 || score == CurveScore::R2;
}

struct Stats {
    double mean = 0.0, sd = 0.0, low = 0.0, high = 0.0;
};
Stats Of(const std::vector<double>& v) {
    Stats s;
    if (v.empty()) return s;
    for (const double x : v) s.mean += x;
    s.mean /= static_cast<double>(v.size());
    double squares = 0.0;
    for (const double x : v) squares += (x - s.mean) * (x - s.mean);
    s.sd = v.size() > 1 ? std::sqrt(squares / static_cast<double>(v.size() - 1)) : 0.0;
    s.low = *std::min_element(v.begin(), v.end());
    s.high = *std::max_element(v.begin(), v.end());
    return s;
}

}  // namespace

CrossValidationView BuildCrossValidationView(const CrossValidationOutcome& o, std::optional<double> single_test) {
    CrossValidationView v;
    v.regression = o.regression;
    std::vector<const CrossValidationRun*> runs;
    for (const auto& run : o.runs)
        if (std::abs(run.share - 1.0) < 1e-9) runs.push_back(&run);
    const int k = static_cast<int>(runs.size());
    std::vector<double> main, second, third;
    for (const auto* r : runs) {
        main.push_back(o.regression ? r->val.mae : r->val.accuracy);
        second.push_back(o.regression ? r->val.rmse : r->val.macro_f1);
        third.push_back(o.regression ? r->val.r2 : r->val.loss);
    }
    const Stats a = Of(main), b = Of(second), c = Of(third);
    const std::string folds = " over " + std::to_string(k) + " fold" + (k == 1 ? "" : "s");
    v.percent = !o.regression;
    v.bars = main;
    v.bars_mean = a.mean;
    v.bars_label = o.regression ? "MAE" : "accuracy";
    if (o.regression) {
        v.headline = "MAE " + Num(a.mean) + " \xC2\xB1 " + Num(a.sd) + folds;
        v.columns = {"MAE", "RMSE", "R\xC2\xB2"};
        v.pills = {"RMSE " + Num(b.mean) + " \xC2\xB1 " + Num(b.sd), "R\xC2\xB2 " + Fixed(c.mean, 3) + " \xC2\xB1 " + Fixed(c.sd, 3),
                   "range " + Num(a.low) + " .. " + Num(a.high)};
        v.sentence = "Each fold's MAE is " + Num(a.low) + " to " + Num(a.high) + ", so expect an error of about " + Num(a.mean) +
                     " on new rows.";
    } else {
        v.headline = "Accuracy " + Percent(a.mean) + " \xC2\xB1 " + Fixed(a.sd * 100.0, 1) + folds;
        v.columns = {"accuracy", "macro F1", "val loss"};
        v.pills = {"macro F1 " + Fixed(b.mean, 3) + " \xC2\xB1 " + Fixed(b.sd, 3), "val loss " + Fixed(c.mean, 3) + " \xC2\xB1 " + Fixed(c.sd, 3),
                   "range " + Percent(a.low) + " .. " + Percent(a.high)};
        v.sentence = "Each fold scores " + Percent(a.low) + " to " + Percent(a.high) + ", so expect about " +
                     Percent(std::round(a.mean * 100.0) / 100.0, 0) + " on new rows.";
    }
    if (o.seconds > 0.0) v.pills.push_back(Duration(o.seconds));
    if (single_test && std::isfinite(*single_test) && k > 1) {
        const double t = *single_test;
        const std::string shown = o.regression ? Num(t) : Percent(t);
        const bool better = o.regression ? t < a.low : t > a.high;
        const bool worse = o.regression ? t > a.high : t < a.low;
        if (better)
            v.sentence += " The single test said " + shown + ", better than every fold: that split was a lucky one, not a better model.";
        else if (worse)
            v.sentence += " The single test said " + shown + ", worse than every fold: that split was an unlucky one.";
        else
            v.sentence += " The single test's " + shown + " is inside that range.";
    }
    // A fold whose loss (classes) or MAE (numbers) stands well apart.
    const std::vector<double>& watch = o.regression ? main : third;
    const Stats w = o.regression ? a : c;
    for (int i = 0; i < k; ++i) {
        const auto* r = runs[static_cast<size_t>(i)];
        FoldRow row;
        row.fold = std::to_string(r->fold);
        row.train_rows = FormatCompiledCount(r->train_rows);
        row.val_rows = FormatCompiledCount(r->val_rows);
        row.scores = o.regression ? std::vector<std::string>{Num(main[static_cast<size_t>(i)]), Num(second[static_cast<size_t>(i)]),
                                                             Fixed(third[static_cast<size_t>(i)], 3)}
                                  : std::vector<std::string>{Percent(main[static_cast<size_t>(i)]), Fixed(second[static_cast<size_t>(i)], 3),
                                                             Fixed(third[static_cast<size_t>(i)], 3)};
        row.flagged = k >= 3 && w.sd > 0.0 && watch[static_cast<size_t>(i)] > w.mean + 1.5 * w.sd;
        if (row.flagged && v.note.empty()) {
            // High loss with an accuracy that is not low: a few mistakes made with
            // great confidence.
            const bool accuracy_ok = main[static_cast<size_t>(i)] >= a.mean - std::max(a.sd, 0.005);
            v.note = o.regression
                         ? "Fold " + row.fold + "'s MAE (" + Num(watch[static_cast<size_t>(i)]) + ") stands apart: its rows are harder to predict."
                     : accuracy_ok ? "Fold " + row.fold + "'s loss is high (" + Fixed(watch[static_cast<size_t>(i)], 3) +
                                         ") though its accuracy is not low: a few mistakes made with great confidence."
                                   : "Fold " + row.fold + "'s loss is high (" + Fixed(watch[static_cast<size_t>(i)], 3) + ") and its accuracy low.";
        }
        v.rows.push_back(std::move(row));
    }
    v.mean.fold = "mean";
    v.mean.scores = o.regression ? std::vector<std::string>{Num(a.mean), Num(b.mean), Fixed(c.mean, 3)}
                                 : std::vector<std::string>{Percent(a.mean), Fixed(b.mean, 3), Fixed(c.mean, 3)};
    return v;
}

std::vector<CurveScore> CurveScores(bool regression) {
    return regression ? std::vector<CurveScore>{CurveScore::Mae, CurveScore::Rmse, CurveScore::R2}
                      : std::vector<CurveScore>{CurveScore::Accuracy, CurveScore::MacroF1, CurveScore::Loss};
}

const char* CurveScoreName(CurveScore score) {
    switch (score) {
        case CurveScore::Accuracy: return "accuracy";
        case CurveScore::MacroF1: return "macro F1";
        case CurveScore::Loss: return "loss";
        case CurveScore::Mae: return "MAE";
        case CurveScore::Rmse: return "RMSE";
        case CurveScore::R2: return "R\xC2\xB2";
    }
    return "";
}

LearningCurveView BuildLearningCurveView(const CrossValidationOutcome& o, CurveScore score) {
    LearningCurveView v;
    v.score_name = CurveScoreName(score);
    v.percent = IsFraction(score);
    v.higher_is_better = HigherIsBetter(score);
    std::vector<double> shares;
    for (const auto& run : o.runs)
        if (std::find_if(shares.begin(), shares.end(), [&](double s) { return std::abs(s - run.share) < 1e-9; }) == shares.end())
            shares.push_back(run.share);
    std::sort(shares.begin(), shares.end());
    const auto show = [&](double x) { return v.percent ? Percent(x) : Num(x); };
    const auto gap_text = [&](double g) { return v.percent ? Fixed(g * 100.0, 1) : Num(g); };
    for (const double share : shares) {
        std::vector<double> train, val, rows;
        for (const auto& run : o.runs) {
            if (std::abs(run.share - share) > 1e-9) continue;
            train.push_back(Pick(run.train, score));
            val.push_back(Pick(run.val, score));
            rows.push_back(static_cast<double>(run.train_rows));
        }
        const Stats t = Of(train), va = Of(val), r = Of(rows);
        v.points.push_back({share, r.mean, t.mean, t.sd, va.mean, va.sd});
        const double gap = v.higher_is_better ? t.mean - va.mean : va.mean - t.mean;
        v.table.push_back({FormatCompiledCount(std::llround(r.mean)), show(t.mean), show(va.mean), gap_text(gap)});
    }
    if (v.points.size() < 2) return v;
    const auto& last = v.points.back();
    const auto& before = v.points[v.points.size() - 2];
    // "Levels off": validation within a point (or 1% of its value) of the last.
    const double close = v.percent ? 0.01 : std::max(std::abs(last.val_mean) * 0.01, 1e-12);
    const double rise = v.higher_is_better ? last.val_mean - before.val_mean : before.val_mean - last.val_mean;
    const std::string rows = FormatCompiledCount(std::llround(last.rows));
    if (rise > close) {
        v.sentence = "Validation " + v.score_name + " is still improving at " + rows + " rows (" + show(before.val_mean) + " to " +
                     show(last.val_mean) + "): more rows would likely help.";
    } else {
        // Back along the points that are the same within a point or the folds'
        // spread at full size (a dip inside the spread is not a fall).
        const double same = std::max(close, last.val_sd);
        size_t from = v.points.size() - 1;
        while (from > 0 && std::abs(v.points[from - 1].val_mean - last.val_mean) <= same) --from;
        v.sentence = "Validation " + v.score_name + " levels off at about " + show(last.val_mean) + " from " +
                     FormatCompiledCount(std::llround(v.points[from].rows)) + " rows; more of the same rows will not help much.";
    }
    const double gap = v.higher_is_better ? last.train_mean - last.val_mean : last.val_mean - last.train_mean;
    const double scale = v.percent ? 1.0 : std::max(std::abs(last.val_mean), 1e-12);
    const double relative = gap / scale;
    const std::string unit = v.percent ? " points" : "";
    if (relative < 0.03)
        v.advice = "The gap is " + gap_text(std::max(gap, 0.0)) + unit + " at full size, so the model is not memorising. To do better, give it "
                   "more information (more columns) or a larger model, not more rows.";
    else if (relative > 0.10)
        v.advice = "Training scores " + gap_text(gap) + unit + " better than validation at full size: the model memorises its rows. "
                   "More rows, dropout or a smaller model would help.";
    else
        v.advice = "A gap of " + gap_text(gap) + unit + " at full size: some memorising. More rows or a little dropout may help.";
    return v;
}

std::string ProgressLine(const CrossValidationProgress& p, double elapsed) {
    std::string line = "Fold " + std::to_string(p.fold) + " of " + std::to_string(p.folds);
    if (p.share < 1.0 - 1e-9) line += " (training rows " + Fixed(p.share * 100.0, 0) + "%)";
    if (p.epochs > 0) line += " \xC2\xB7 epoch " + std::to_string(std::max(p.epoch, 0)) + " of " + std::to_string(p.epochs);
    // Time left from the share of the work done (runs x epochs).
    const double done = (static_cast<double>(p.run - 1) + (p.epochs > 0 ? static_cast<double>(p.epoch) / p.epochs : 0.0)) /
                        std::max(1, p.runs);
    if (done > 0.02 && elapsed > 1.0) {
        const double left = elapsed * (1.0 - done) / done;
        line += " \xC2\xB7 about " + (left < 60.0 ? std::to_string(static_cast<int>(std::ceil(left / 5.0) * 5.0)) + " s"
                                                  : std::to_string(static_cast<int>(std::ceil(left / 60.0))) + " min") +
                " left";
    }
    return line;
}

bool ParseShares(const std::string& text, std::vector<double>& shares, std::string& error) {
    shares.clear();
    std::string cleaned = text;
    for (char& ch : cleaned)
        if (ch == ',' || ch == ';') ch = ' ';
    std::istringstream in(cleaned);
    std::string word;
    while (in >> word) {
        const bool percent = !word.empty() && word.back() == '%';
        if (percent) word.pop_back();
        char* end = nullptr;
        double v = std::strtod(word.c_str(), &end);
        if (word.empty() || end == word.c_str() || *end != '\0') {
            error = "'" + word + (percent ? "%" : "") + "' is not a number";
            return false;
        }
        if (percent || v > 1.0) v /= 100.0;
        if (!(v > 0.0 && v <= 1.0)) {
            error = "each share must be above 0 and at most 100%";
            return false;
        }
        shares.push_back(v);
    }
    std::sort(shares.begin(), shares.end());
    shares.erase(std::unique(shares.begin(), shares.end(), [](double x, double y) { return std::abs(x - y) < 1e-9; }), shares.end());
    if (shares.size() < 2) {
        error = "give two shares or more (for example 10%, 25%, 50%, 100%)";
        return false;
    }
    return true;
}

std::string SharesText(const std::vector<double>& shares) {
    std::string out;
    for (const double s : shares) {
        if (!out.empty()) out += ", ";
        char text[24];
        std::snprintf(text, sizeof(text), "%g%%", s * 100.0);
        out += text;
    }
    return out;
}

std::string OutcomeCsv(const CrossValidationOutcome& o) {
    std::ostringstream out;
    out.precision(10);
    const bool curve = std::any_of(o.runs.begin(), o.runs.end(), [](const CrossValidationRun& r) { return r.share < 1.0 - 1e-9; });
    out << "fold,share,train_rows,val_rows,val_" << (o.regression ? "mae,val_rmse,val_r2" : "accuracy,val_macro_f1,val_loss");
    if (curve) out << (o.regression ? ",train_mae,train_rmse,train_r2" : ",train_accuracy,train_macro_f1,train_loss");
    out << ",epochs,seconds\n";
    const auto triple = [&](const CrossValidationScores& s) {
        if (o.regression) out << s.mae << "," << s.rmse << "," << s.r2;
        else out << s.accuracy << "," << s.macro_f1 << "," << s.loss;
    };
    for (const auto& r : o.runs) {
        out << r.fold << "," << r.share << "," << r.train_rows << "," << r.val_rows << ",";
        triple(r.val);
        if (curve) {
            out << ",";
            triple(r.train);
        }
        out << "," << r.epochs << "," << r.seconds << "\n";
    }
    out << "\n# target," << o.target << "\n# rows used," << o.rows << "\n# graph test rows left out," << o.test_rows_left_out
        << "\n";
    return out.str();
}

}  // namespace cyxwiz::folds_view
