// A test's classifier figures (TOFIX157): the confusion matrix and
// FinishClassifierMetrics, apart from TestExecutor so the pipeline executor
// (Test Evaluation node) and the views share them without the test runner.
#include "test_executor.h"
#include "one_vs_rest_curves.h"

#include <algorithm>

namespace cyxwiz {

// ============================================================================
// ConfusionMatrix Implementation
// ============================================================================

void ConfusionMatrix::Resize(int n) {
    num_classes = n;
    matrix.clear();
    matrix.resize(n, std::vector<int>(n, 0));
}

void ConfusionMatrix::Add(int actual, int predicted) {
    if (actual >= 0 && actual < num_classes && predicted >= 0 && predicted < num_classes) {
        matrix[actual][predicted]++;
    }
}

float ConfusionMatrix::GetAccuracy() const {
    if (num_classes == 0) return 0.0f;

    int correct = 0;
    int total = 0;
    for (int i = 0; i < num_classes; ++i) {
        for (int j = 0; j < num_classes; ++j) {
            total += matrix[i][j];
            if (i == j) correct += matrix[i][j];
        }
    }
    return total > 0 ? static_cast<float>(correct) / total : 0.0f;
}

int ConfusionMatrix::GetTotal() const {
    int total = 0;
    for (int i = 0; i < num_classes; ++i) {
        for (int j = 0; j < num_classes; ++j) {
            total += matrix[i][j];
        }
    }
    return total;
}

void FinishClassifierMetrics(TestingMetrics& m) {
    // Per class, from the confusion matrix [actual][predicted].
    const int num_classes = m.confusion_matrix.num_classes;
    m.per_class_metrics.resize(static_cast<size_t>(std::max(num_classes, 0)));
    for (int c = 0; c < num_classes; ++c) {
        ClassMetrics& cm = m.per_class_metrics[static_cast<size_t>(c)];
        const auto& matrix = m.confusion_matrix.matrix;
        cm.class_id = c;
        if (cm.class_name.empty()) {
            cm.class_name = static_cast<size_t>(c) < m.confusion_matrix.class_names.size()
                                ? m.confusion_matrix.class_names[static_cast<size_t>(c)]
                                : "Class " + std::to_string(c);
        }
        cm.true_positives = matrix[c][c];
        cm.false_positives = 0;  // column c without the diagonal
        cm.false_negatives = 0;  // row c without the diagonal
        cm.support = 0;          // row c: the actual samples of the class
        for (int i = 0; i < num_classes; ++i) {
            if (i != c) {
                cm.false_positives += matrix[i][c];
                cm.false_negatives += matrix[c][i];
            }
            cm.support += matrix[c][i];
        }
        const int pred_positives = cm.true_positives + cm.false_positives;
        cm.precision = pred_positives > 0 ? static_cast<float>(cm.true_positives) / pred_positives : 0.0f;
        const int actual_positives = cm.true_positives + cm.false_negatives;
        cm.recall = actual_positives > 0 ? static_cast<float>(cm.true_positives) / actual_positives : 0.0f;
        const float pr_sum = cm.precision + cm.recall;
        cm.f1_score = pr_sum > 0 ? 2.0f * cm.precision * cm.recall / pr_sum : 0.0f;
    }

    // Macro (unweighted mean over classes) and support-weighted averages.
    if (num_classes > 0) {
        float sum_precision = 0.0f, sum_recall = 0.0f, sum_f1 = 0.0f, weighted_f1 = 0.0f;
        int total_support = 0;
        for (const auto& cm : m.per_class_metrics) {
            sum_precision += cm.precision;
            sum_recall += cm.recall;
            sum_f1 += cm.f1_score;
            weighted_f1 += cm.f1_score * cm.support;
            total_support += cm.support;
        }
        m.macro_precision = sum_precision / num_classes;
        m.macro_recall = sum_recall / num_classes;
        m.macro_f1 = sum_f1 / num_classes;
        m.weighted_f1 = total_support > 0 ? weighted_f1 / total_support : 0.0f;
        m.test_accuracy = m.confusion_matrix.GetAccuracy();
    }

    // ROC / precision-recall from the kept class probabilities.
    if (m.metric_learning_mode || !m.curves_note.empty()) return;
    const size_t classes = static_cast<size_t>(std::max(m.probability_classes, 0));
    const size_t samples = m.ground_truth.size();
    if (classes < 2 || samples == 0 || m.class_probabilities.size() != samples * classes) {
        m.curves_note = "ROC / PR need class scores for every test sample";
        return;
    }
    if (classes == 2) {
        // Binary: the positive class's probability against truth == 1.
        std::vector<double> positive(samples);
        for (size_t i = 0; i < samples; ++i) positive[i] = m.class_probabilities[i * 2 + 1];
        m.roc = ModelEvaluation::ComputeROC(m.ground_truth, positive);
        m.pr = ModelEvaluation::ComputePRCurve(m.ground_truth, positive);
    } else {
        // One-vs-rest; a class the test set has no sample of is skipped, not fatal.
        std::vector<std::string> names;
        for (const auto& c : m.per_class_metrics) names.push_back(c.class_name);
        OneVsRestCurves curves = ComputeOneVsRestCurves(m.ground_truth, m.class_probabilities, classes, names);
        m.roc = std::move(curves.roc);
        m.pr = std::move(curves.pr);
        m.curves_note = std::move(curves.note);
    }
    // Say which curves failed unless the message already does ("ROC / PR
    // need ...", "Precision-recall requires ..."); a successful multi-class
    // run may carry a note on skipped classes.
    const auto why = [](const std::string& error) {
        return error.rfind("ROC", 0) == 0 || error.rfind("Precision", 0) == 0 ? error : "ROC / PR: " + error;
    };
    if (!m.roc->success) m.curves_note = why(m.roc->error_message);
    else if (!m.pr->success) m.curves_note = why(m.pr->error_message);
}

}  // namespace cyxwiz
