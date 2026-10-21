#pragma once

// Multi-class ROC / PR for the Test step (TOFIX140): one-vs-rest per class on
// the device (ModelEvaluation's multi-class curves), skipping a class the test
// set has no sample of instead of failing every curve. The macro AUC / AP
// average the scored classes (scikit-learn's roc_auc_score with labels= the
// scored classes). A skipped class keeps empty curves.

#include <cyxwiz/model_evaluation.h>

#include <string>
#include <vector>

namespace cyxwiz {

struct OneVsRestCurves {
    ROCCurveData roc;
    PRCurveData pr;
    std::vector<size_t> skipped;  // classes without curves
    std::string note;             // which classes were skipped and why; empty when none
};

// scores: row-major [samples][classes]. class_names (optional) word the note.
OneVsRestCurves ComputeOneVsRestCurves(const std::vector<int>& ground_truth, const std::vector<float>& scores,
                                       size_t classes, const std::vector<std::string>& class_names = {});

}  // namespace cyxwiz
