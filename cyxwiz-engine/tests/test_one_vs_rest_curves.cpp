// Multi-class ROC / PR for the Test step (TOFIX140): a class with no test
// sample is skipped with a note instead of failing every curve; the macro
// averages cover the scored classes and match the binary curves per class.
// Runs on the requested ArrayFire device (CYXWIZ_TEST_ARRAYFIRE_BACKEND).
#include "../src/core/one_vs_rest_curves.h"
#include "computation_truth/test_device_selection.h"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

using namespace cyxwiz;

namespace {

int failures = 0;

void Check(bool condition, const std::string& message) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", message.c_str());
        ++failures;
    }
}

bool Near(double a, double b) { return std::abs(a - b) < 1e-9; }

}  // namespace

int main() {
    Check(cyxwiz::test::SelectTestDeviceFromEnvironment(), "requested test device");

    // 4 classes; class 2 has no test sample.
    const std::vector<int> truth = {0, 1, 3, 0, 1, 3, 0, 1};
    const std::vector<float> scores = {
        0.7f, 0.1f, 0.1f, 0.1f,  0.2f, 0.6f, 0.1f, 0.1f,  0.1f, 0.2f, 0.1f, 0.6f,  0.4f, 0.3f, 0.2f, 0.1f,
        0.3f, 0.3f, 0.2f, 0.2f,  0.1f, 0.1f, 0.2f, 0.6f,  0.5f, 0.2f, 0.2f, 0.1f,  0.2f, 0.5f, 0.2f, 0.1f};
    const OneVsRestCurves curves = ComputeOneVsRestCurves(truth, scores, 4, {"cat", "dog", "bird", "fish"});
    Check(curves.roc.success && curves.pr.success, "curves computed despite the empty class: " +
                                                       curves.roc.error_message);
    Check(curves.skipped == std::vector<size_t>({2}), "class 2 skipped");
    Check(curves.roc.class_fpr[2].empty() && curves.pr.class_recall[2].empty(), "skipped class has no curve");
    Check(curves.note.find("bird") != std::string::npos && curves.note.find("3 classes") != std::string::npos,
          "note names the class and the count: " + curves.note);

    // Each scored class matches the binary curve; the macro is over the 3 scored classes.
    double auc_sum = 0.0;
    double ap_sum = 0.0;
    for (const size_t c : {0u, 1u, 3u}) {
        std::vector<int> labels;
        std::vector<double> s;
        for (size_t i = 0; i < truth.size(); ++i) {
            labels.push_back(truth[i] == static_cast<int>(c) ? 1 : 0);
            s.push_back(scores[i * 4 + c]);
        }
        const auto roc = ModelEvaluation::ComputeROC(labels, s);
        const auto pr = ModelEvaluation::ComputePRCurve(labels, s);
        Check(Near(curves.roc.class_auc[c], roc.auc) && Near(curves.pr.class_ap[c], pr.average_precision),
              "class " + std::to_string(c) + " matches its binary curve");
        auc_sum += roc.auc;
        ap_sum += pr.average_precision;
    }
    Check(Near(curves.roc.auc, auc_sum / 3.0) && Near(curves.pr.average_precision, ap_sum / 3.0),
          "macro over the scored classes");

    // Every class present: no note.
    const std::vector<int> full = {0, 1, 2, 0, 1, 2};
    const std::vector<float> full_scores = {0.8f, 0.1f, 0.1f, 0.2f, 0.7f, 0.1f, 0.1f, 0.2f, 0.7f,
                                            0.5f, 0.3f, 0.2f, 0.3f, 0.4f, 0.3f, 0.2f, 0.3f, 0.5f};
    const OneVsRestCurves all = ComputeOneVsRestCurves(full, full_scores, 3);
    Check(all.roc.success && all.skipped.empty() && all.note.empty(), "no skip, no note");

    // Scores of the wrong size are refused.
    Check(!ComputeOneVsRestCurves(full, {0.5f}, 3).roc.success, "short scores refused");

    if (failures) {
        std::fprintf(stderr, "%d check(s) failed\n", failures);
        return 1;
    }
    std::printf("one-vs-rest curves: empty class skipped, per-class and macro values as expected. OK\n");
    return 0;
}
