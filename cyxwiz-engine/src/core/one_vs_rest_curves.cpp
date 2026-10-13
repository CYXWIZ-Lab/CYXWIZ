#include "one_vs_rest_curves.h"

namespace cyxwiz {

OneVsRestCurves ComputeOneVsRestCurves(const std::vector<int>& ground_truth, const std::vector<float>& scores,
                                       size_t classes, const std::vector<std::string>& class_names) {
    OneVsRestCurves out;
    const size_t samples = ground_truth.size();
    if (classes < 2 || samples == 0 || scores.size() != samples * classes) {
        out.roc.error_message = out.pr.error_message = "ROC / PR need class scores for every test sample";
        return out;
    }

    // A one-vs-rest curve of class c depends only on the labels and column c,
    // so the classes the test set has no sample of are dropped and the rest
    // renumbered for ModelEvaluation, then mapped back.
    std::vector<size_t> support(classes, 0);
    for (const int label : ground_truth)
        if (label >= 0 && static_cast<size_t>(label) < classes) ++support[static_cast<size_t>(label)];
    std::vector<size_t> scored;      // new index -> class
    std::vector<int> renumber(classes, -1);
    std::string skipped_names;
    for (size_t c = 0; c < classes; ++c) {
        if (support[c] > 0) {
            renumber[c] = static_cast<int>(scored.size());
            scored.push_back(c);
            continue;
        }
        out.skipped.push_back(c);
        const std::string name =
            c < class_names.size() && !class_names[c].empty() ? class_names[c] : "class " + std::to_string(c);
        skipped_names += (skipped_names.empty() ? "" : ", ") + name;
    }
    if (scored.size() < 2) {
        out.roc.error_message = out.pr.error_message = "ROC / PR need test samples of at least two classes";
        return out;
    }

    std::vector<int> labels(samples);
    std::vector<std::vector<double>> rows(samples, std::vector<double>(scored.size()));
    for (size_t i = 0; i < samples; ++i) {
        const int label = ground_truth[i];
        labels[i] = label >= 0 && static_cast<size_t>(label) < classes ? renumber[static_cast<size_t>(label)] : label;
        for (size_t k = 0; k < scored.size(); ++k) rows[i][k] = scores[i * classes + scored[k]];
    }
    const ROCCurveData roc = ModelEvaluation::ComputeMulticlassROC(labels, rows);
    const PRCurveData pr = ModelEvaluation::ComputeMulticlassPRCurve(labels, rows);
    if (!roc.success || !pr.success) {
        out.roc.error_message = out.pr.error_message = !roc.success ? roc.error_message : pr.error_message;
        return out;
    }

    // Back to the model's class numbers; a skipped class keeps empty curves.
    out.roc = roc;
    out.pr = pr;
    out.roc.class_fpr.assign(classes, {});
    out.roc.class_tpr.assign(classes, {});
    out.roc.class_thresholds.assign(classes, {});
    out.roc.class_auc.assign(classes, 0.0);
    out.pr.class_precision.assign(classes, {});
    out.pr.class_recall.assign(classes, {});
    out.pr.class_thresholds.assign(classes, {});
    out.pr.class_ap.assign(classes, 0.0);
    for (size_t k = 0; k < scored.size(); ++k) {
        const size_t c = scored[k];
        out.roc.class_fpr[c] = roc.class_fpr[k];
        out.roc.class_tpr[c] = roc.class_tpr[k];
        out.roc.class_thresholds[c] = roc.class_thresholds[k];
        out.roc.class_auc[c] = roc.class_auc[k];
        out.pr.class_precision[c] = pr.class_precision[k];
        out.pr.class_recall[c] = pr.class_recall[k];
        out.pr.class_thresholds[c] = pr.class_thresholds[k];
        out.pr.class_ap[c] = pr.class_ap[k];
    }
    if (!out.skipped.empty()) {
        out.note = "No curve for " + skipped_names + ": the test set has no sample of " +
                   (out.skipped.size() == 1 ? "it" : "them") + ". AUC and AP average the other " +
                   std::to_string(scored.size()) + " classes.";
    }
    return out;
}

}  // namespace cyxwiz
