#include "test_predictions_metrics.h"

#include "arrow_number.h"
#include "test_predictions_table.h"

#include <arrow/api.h>
#include <arrow/util/key_value_metadata.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <optional>

namespace cyxwiz {

std::shared_ptr<arrow::Table> PredictionsTableFromMetrics(const TestingMetrics& m) {
    if (m.causal_lm_mode) return nullptr;
    TestPredictionsSource source{m.row_ids, m.source_dataset};
    source.loss = m.test_loss;
    source.seconds = m.total_time_seconds;
    source.partial = m.partial;
    if (m.regression_mode)
        return BuildRegressionPredictionsTable(m.regression_truth, m.regression_predictions, m.regression_width, source);
    if (m.ground_truth.empty()) return nullptr;
    std::vector<std::string> class_names;
    for (const auto& c : m.per_class_metrics) class_names.push_back(c.class_name);
    return BuildTestPredictionsTable(m.ground_truth, m.predictions, m.confidences, m.class_probabilities,
                                     m.probability_classes, class_names, source);
}

namespace {

std::string MetadataValue(const arrow::Table& table, const char* key) {
    const auto metadata = table.schema()->metadata();
    if (!metadata) return {};
    const int at = metadata->FindKey(key);
    return at >= 0 ? metadata->value(at) : std::string();
}

std::vector<std::string> SplitLines(const std::string& text) {
    std::vector<std::string> lines;
    size_t start = 0;
    while (start <= text.size()) {
        const size_t end = text.find('\n', start);
        lines.push_back(text.substr(start, end == std::string::npos ? std::string::npos : end - start));
        if (end == std::string::npos) break;
        start = end + 1;
    }
    return lines;
}

// A column of whole numbers (a class index, a row id); false when it is not a
// number column, has a missing value or a fraction.
bool Whole(const arrow::Table& table, const std::string& name, std::vector<int64_t>& out) {
    const std::vector<double> values = ArrowNumbers(table.GetColumnByName(name));
    if (values.size() != static_cast<size_t>(table.num_rows())) return false;
    out.clear();
    out.reserve(values.size());
    for (double v : values) {
        if (std::isnan(v) || v != std::floor(v)) return false;
        out.push_back(static_cast<int64_t>(v));
    }
    return true;
}

bool Real(const arrow::Table& table, const std::string& name, std::vector<double>& out) {
    out = ArrowNumbers(table.GetColumnByName(name));
    return out.size() == static_cast<size_t>(table.num_rows());
}

bool ReadClassifier(const arrow::Table& table, TestingMetrics& m, std::string& error) {
    const size_t rows = static_cast<size_t>(table.num_rows());
    std::vector<int64_t> truth, predicted;
    if (!Whole(table, "true_class", truth) || !Whole(table, "predicted_class", predicted)) {
        error = "the true_class / predicted_class columns are missing or not whole numbers";
        return false;
    }
    std::vector<double> confidence;
    if (!Real(table, "confidence", confidence)) confidence.assign(rows, 1.0);

    // The scores: binary "score" (the positive class), multi-class one
    // "score_..." column per class, in order.
    std::vector<std::vector<double>> scores;
    if (table.GetColumnByName("score")) {
        std::vector<double> positive;
        if (Real(table, "score", positive)) {
            std::vector<double> negative(rows);
            for (size_t i = 0; i < rows; ++i) negative[i] = 1.0 - positive[i];
            scores = {std::move(negative), std::move(positive)};
        }
    } else {
        for (const auto& field : table.schema()->fields()) {
            if (field->name().rfind("score_", 0) != 0) continue;
            std::vector<double> column;
            if (!Real(table, field->name(), column)) {
                scores.clear();
                break;
            }
            scores.push_back(std::move(column));
        }
    }

    std::vector<std::string> names;
    const std::string joined = MetadataValue(table, kTestPredictionsClassNamesKey);
    if (!joined.empty()) names = SplitLines(joined);
    int64_t highest = -1;
    for (size_t i = 0; i < rows; ++i) {
        if (truth[i] < 0 || predicted[i] < 0) {
            error = "a class index is negative";
            return false;
        }
        highest = std::max({highest, truth[i], predicted[i]});
    }
    const int classes = static_cast<int>(std::max<int64_t>({highest + 1, static_cast<int64_t>(names.size()),
                                                            static_cast<int64_t>(scores.size())}));

    m.confusion_matrix.Resize(classes);
    m.confusion_matrix.class_names = names;
    m.per_class_metrics.assign(static_cast<size_t>(classes), ClassMetrics{});
    for (int c = 0; c < classes; ++c)
        m.per_class_metrics[static_cast<size_t>(c)].class_name =
            static_cast<size_t>(c) < names.size() && !names[static_cast<size_t>(c)].empty() ? names[static_cast<size_t>(c)]
                                                                                            : "Class " + std::to_string(c);
    m.ground_truth.reserve(rows);
    m.predictions.reserve(rows);
    m.confidences.reserve(rows);
    for (size_t i = 0; i < rows; ++i) {
        const int t = static_cast<int>(truth[i]);
        const int p = static_cast<int>(predicted[i]);
        m.ground_truth.push_back(t);
        m.predictions.push_back(p);
        m.confidences.push_back(static_cast<float>(confidence[i]));
        m.confusion_matrix.Add(t, p);
        if (t == p) ++m.correct_predictions;
    }
    if (scores.size() >= 2 && static_cast<int>(scores.size()) == classes) {
        m.probability_classes = classes;
        m.class_probabilities.reserve(rows * scores.size());
        for (size_t i = 0; i < rows; ++i)
            for (const auto& column : scores) m.class_probabilities.push_back(static_cast<float>(column[i]));
    }
    FinishClassifierMetrics(m);
    return true;
}

bool ReadRegression(const arrow::Table& table, TestingMetrics& m, std::string& error) {
    const size_t rows = static_cast<size_t>(table.num_rows());
    std::vector<std::pair<std::string, std::string>> targets;  // true, predicted
    if (table.GetColumnByName("true_value")) {
        targets.emplace_back("true_value", "predicted_value");
    } else {
        for (int t = 0; table.GetColumnByName("true_" + std::to_string(t)); ++t)
            targets.emplace_back("true_" + std::to_string(t), "predicted_" + std::to_string(t));
    }
    if (targets.empty()) {
        error = "the table has neither class columns (true_class) nor target columns (true_value)";
        return false;
    }
    const size_t width = targets.size();
    std::vector<std::vector<double>> truth(width), predicted(width);
    for (size_t t = 0; t < width; ++t) {
        if (!Real(table, targets[t].first, truth[t]) || !Real(table, targets[t].second, predicted[t])) {
            error = "'" + targets[t].first + "' / '" + targets[t].second + "' are missing or not numbers";
            return false;
        }
    }
    m.regression_mode = true;
    m.regression_width = static_cast<int>(width);
    m.regression_truth.reserve(rows * width);
    m.regression_predictions.reserve(rows * width);
    double absolute = 0.0, squared = 0.0;
    size_t counted = 0;
    for (size_t i = 0; i < rows; ++i) {
        for (size_t t = 0; t < width; ++t) {
            m.regression_truth.push_back(static_cast<float>(truth[t][i]));
            m.regression_predictions.push_back(static_cast<float>(predicted[t][i]));
            const double e = predicted[t][i] - truth[t][i];
            if (std::isnan(e)) continue;
            absolute += std::abs(e);
            squared += e * e;
            ++counted;
        }
    }
    m.total_target_values = rows * width;
    m.test_mae = counted ? static_cast<float>(absolute / static_cast<double>(counted)) : 0.0f;
    m.test_rmse = counted ? static_cast<float>(std::sqrt(squared / static_cast<double>(counted))) : 0.0f;
    return true;
}

}  // namespace

bool MetricsFromPredictionsTable(const arrow::Table& table, TestingMetrics& out, std::string& error) {
    out = TestingMetrics{};
    if (table.num_rows() == 0) {
        error = "the table has no rows";
        return false;
    }
    const bool classifier = table.GetColumnByName("true_class") != nullptr;
    if (!(classifier ? ReadClassifier(table, out, error) : ReadRegression(table, out, error))) return false;

    out.total_samples = static_cast<int>(table.num_rows());
    out.source_dataset = MetadataValue(table, kTestPredictionsSourceKey);
    if (table.GetColumnByName("row_id") && !Whole(table, "row_id", out.row_ids)) out.row_ids.clear();
    const std::string loss = MetadataValue(table, kTestPredictionsLossKey);
    if (!loss.empty()) out.test_loss = std::strtof(loss.c_str(), nullptr);
    const std::string seconds = MetadataValue(table, kTestPredictionsSecondsKey);
    if (!seconds.empty()) out.total_time_seconds = std::strtof(seconds.c_str(), nullptr);
    if (out.total_time_seconds > 0.0f) out.samples_per_second = static_cast<float>(out.total_samples) / out.total_time_seconds;
    out.partial = MetadataValue(table, kTestPredictionsPartialKey) == "true";
    out.is_complete = true;
    out.status_message = "Testing complete";
    return true;
}

namespace {

std::string Lower(std::string text) {
    std::transform(text.begin(), text.end(), text.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return text;
}

// A cell as a class label: text as it is, a number without trailing zeros
// ("3", "0.5"), a bool as true / false; nullopt when missing (null, empty
// text, NaN).
std::vector<std::optional<std::string>> LabelTexts(const arrow::ChunkedArray& column) {
    std::vector<std::optional<std::string>> out;
    out.reserve(static_cast<size_t>(column.length()));
    for (const auto& chunk : column.chunks()) {
        const auto id = chunk->type_id();
        for (int64_t i = 0; i < chunk->length(); ++i) {
            if (chunk->IsNull(i)) {
                out.emplace_back();
            } else if (id == arrow::Type::STRING || id == arrow::Type::LARGE_STRING) {
                std::string text = id == arrow::Type::STRING ? static_cast<const arrow::StringArray&>(*chunk).GetString(i)
                                                             : static_cast<const arrow::LargeStringArray&>(*chunk).GetString(i);
                if (text.empty()) out.emplace_back();
                else out.emplace_back(std::move(text));
            } else if (id == arrow::Type::BOOL) {
                out.emplace_back(static_cast<const arrow::BooleanArray&>(*chunk).Value(i) ? "true" : "false");
            } else if (IsArrowNumber(id)) {
                const double v = ArrowNumberAt(*chunk, i);
                if (std::isnan(v)) {
                    out.emplace_back();
                    continue;
                }
                char text[40];
                if (v == std::floor(v) && std::abs(v) < 1e15) std::snprintf(text, sizeof(text), "%lld", static_cast<long long>(v));
                else std::snprintf(text, sizeof(text), "%.10g", v);
                out.emplace_back(text);
            } else {
                const auto scalar = chunk->GetScalar(i);
                if (scalar.ok() && (*scalar)->is_valid) out.emplace_back((*scalar)->ToString());
                else out.emplace_back();
            }
        }
    }
    return out;
}

}  // namespace

bool MetricsFromColumns(const arrow::Table& table, const PredictionColumns& c, TestingMetrics& out, int64_t& left_out,
                        std::string& error) {
    out = TestingMetrics{};
    left_out = 0;
    const auto truth_column = table.GetColumnByName(c.truth);
    const auto predicted_column = table.GetColumnByName(c.predicted);
    if (!truth_column || !predicted_column) {
        error = "pick a true column and a predicted column";
        return false;
    }
    const size_t rows = static_cast<size_t>(table.num_rows());
    TestPredictionsSource source;
    source.dataset = c.dataset;
    std::shared_ptr<arrow::Table> predictions;

    if (c.numbers) {
        const std::vector<double> truth = ArrowNumbers(truth_column), predicted = ArrowNumbers(predicted_column);
        if (truth.size() != rows || predicted.size() != rows) {
            error = "'" + (truth.size() != rows ? c.truth : c.predicted) + "' is not a number column";
            return false;
        }
        std::vector<float> t, p;
        for (size_t i = 0; i < rows; ++i) {
            if (!std::isfinite(truth[i]) || !std::isfinite(predicted[i])) {
                ++left_out;
                continue;
            }
            t.push_back(static_cast<float>(truth[i]));
            p.push_back(static_cast<float>(predicted[i]));
            source.row_ids.push_back(static_cast<int64_t>(i));
        }
        if (t.empty()) {
            error = "no row has both a true and a predicted value";
            return false;
        }
        predictions = BuildRegressionPredictionsTable(t, p, 1, source);
    } else {
        const auto truth = LabelTexts(*truth_column), predicted = LabelTexts(*predicted_column);
        // The class set: both columns' labels, numbers in numeric order, text sorted (the label encoder's order).
        std::vector<std::string> names;
        for (const auto* labels : {&truth, &predicted})
            for (const auto& label : *labels)
                if (label) names.push_back(*label);
        std::sort(names.begin(), names.end());
        names.erase(std::unique(names.begin(), names.end()), names.end());
        const bool numeric = IsArrowNumber(truth_column->type()->id()) && IsArrowNumber(predicted_column->type()->id());
        if (numeric)
            std::sort(names.begin(), names.end(), [](const std::string& a, const std::string& b) { return std::stod(a) < std::stod(b); });
        if (names.size() < 2) {
            error = names.empty() ? "no row has both a true and a predicted label" : "only one class (" + names[0] + ") in the two columns";
            return false;
        }
        if (names.size() > 1000) {
            error = std::to_string(names.size()) + " different labels: pick class columns, or read the columns as numbers";
            return false;
        }
        std::map<std::string, int> index;
        for (size_t i = 0; i < names.size(); ++i) index[names[i]] = static_cast<int>(i);
        const int classes = static_cast<int>(names.size());

        // Scores: one per class (score_<class> columns go in class order), or one for two classes.
        std::vector<std::string> score_names = c.scores;
        if (score_names.size() == names.size()) {
            std::vector<std::string> ordered(names.size());
            bool by_name = true;
            for (const auto& s : score_names) {
                const auto it = index.find(s.rfind("score_", 0) == 0 ? s.substr(6) : std::string());
                if (it == index.end() || !ordered[static_cast<size_t>(it->second)].empty()) by_name = false;
                else ordered[static_cast<size_t>(it->second)] = s;
            }
            if (by_name) score_names = ordered;
        } else if (score_names.size() == 1 && classes != 2) {
            error = "one score column fits two classes; these columns have " + std::to_string(classes) +
                    ": pick one score column per class";
            return false;
        } else if (score_names.size() > 1) {
            error = std::to_string(score_names.size()) + " score columns for " + std::to_string(classes) + " classes";
            return false;
        }
        std::vector<std::vector<double>> scores;
        for (const auto& name : score_names) {
            scores.push_back(ArrowNumbers(table.GetColumnByName(name)));
            if (scores.back().size() != rows) {
                error = "'" + name + "' is not a number column";
                return false;
            }
        }
        int positive = 1;
        if (scores.size() == 1 && !c.positive.empty()) {
            const auto it = index.find(c.positive);
            if (it == index.end()) {
                error = "the positive class '" + c.positive + "' is not a label of these columns";
                return false;
            }
            positive = it->second;
        }

        std::vector<int> gt, pred;
        std::vector<float> confidence, probabilities;
        for (size_t i = 0; i < rows; ++i) {
            bool missing = !truth[i] || !predicted[i];
            for (const auto& s : scores) missing = missing || !std::isfinite(s[i]);
            if (missing) {
                ++left_out;
                continue;
            }
            gt.push_back(index[*truth[i]]);
            pred.push_back(index[*predicted[i]]);
            source.row_ids.push_back(static_cast<int64_t>(i));
            if (scores.size() == 1) {
                const double s = scores[0][i];
                probabilities.push_back(static_cast<float>(positive == 1 ? 1.0 - s : s));
                probabilities.push_back(static_cast<float>(positive == 1 ? s : 1.0 - s));
            } else {
                for (const auto& column : scores) probabilities.push_back(static_cast<float>(column[i]));
            }
            confidence.push_back(scores.empty() ? 1.0f : probabilities[probabilities.size() - static_cast<size_t>(classes) + static_cast<size_t>(pred.back())]);
        }
        if (gt.empty()) {
            error = "no row has both a true and a predicted label";
            return false;
        }
        predictions = BuildTestPredictionsTable(gt, pred, confidence, probabilities, scores.empty() ? 0 : classes, names, source);
    }
    if (!predictions || !MetricsFromPredictionsTable(*predictions, out, error)) {
        if (error.empty()) error = "the columns cannot be read as a test";
        return false;
    }
    return true;
}

PredictionColumns GuessPredictionColumns(const arrow::Table& table) {
    PredictionColumns c;
    const auto fields = table.schema()->fields();
    const auto find = [&](std::initializer_list<const char*> exact, std::initializer_list<const char*> parts) {
        for (const char* name : exact)
            for (const auto& f : fields)
                if (Lower(f->name()) == name) return f->name();
        for (const char* part : parts)
            for (const auto& f : fields)
                if (Lower(f->name()).find(part) != std::string::npos) return f->name();
        return std::string();
    };
    c.predicted = find({"predicted_class", "predicted_value", "predicted", "prediction", "y_pred"}, {"pred"});
    for (const auto& f : fields) {
        const std::string lower = Lower(f->name());
        if (f->name() == c.predicted || lower.find("pred") != std::string::npos) continue;
        if (lower == "true_class" || lower == "true_value" || lower == "y_true" || lower == "true" || lower == "actual" ||
            lower == "label" || lower == "target" || lower == "y") {
            c.truth = f->name();
            break;
        }
    }
    if (c.truth.empty()) c.truth = find({}, {"true", "actual", "label", "target"});
    if (c.truth == c.predicted) c.truth.clear();
    for (const auto& f : fields)
        if (Lower(f->name()).rfind("score_", 0) == 0) c.scores.push_back(f->name());
    if (c.scores.empty()) {
        const std::string one = find({"score", "probability", "prob", "proba"}, {});
        if (!one.empty()) c.scores.push_back(one);
    }
    if (!c.truth.empty() && !c.predicted.empty()) {
        const auto truth = ArrowNumbers(table.GetColumnByName(c.truth));
        const auto predicted = ArrowNumbers(table.GetColumnByName(c.predicted));
        if (!truth.empty() && !predicted.empty()) {
            std::vector<double> distinct;
            bool fractions = false;
            for (const double v : truth) {
                if (std::isnan(v)) continue;
                fractions = fractions || v != std::floor(v);
                if (distinct.size() <= 20 && std::find(distinct.begin(), distinct.end(), v) == distinct.end()) distinct.push_back(v);
            }
            c.numbers = fractions || distinct.size() > 20;
        }
    }
    if (c.numbers) c.scores.clear();
    return c;
}

}  // namespace cyxwiz
