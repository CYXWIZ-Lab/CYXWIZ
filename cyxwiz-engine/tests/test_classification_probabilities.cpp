// ClassificationClassProbabilities (TOFIX140 ROC / PR in test results): the
// per-class probabilities the Test step keeps for its curves. Binary modes give
// {1 - p, p}; multi-class logits go through softmax (numpy reference values);
// scores that already form a distribution pass through unchanged.
#include "../src/core/classification_decision.h"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

using cyxwiz::ClassificationClassProbabilities;
using cyxwiz::ClassificationDecisionMode;

namespace {

int failures = 0;

void Check(bool condition, const std::string& message) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", message.c_str());
        ++failures;
    }
}

void Near(const std::vector<float>& got, const std::vector<float>& want, const std::string& what) {
    Check(got.size() == want.size(), what + ": size");
    for (size_t i = 0; i < got.size() && i < want.size(); ++i) {
        Check(std::fabs(got[i] - want[i]) < 1e-6f, what + " [" + std::to_string(i) + "]: " +
                                                       std::to_string(got[i]) + " vs " + std::to_string(want[i]));
    }
}

}  // namespace

int main() {
    std::vector<float> out;

    // Binary logit 0.8 -> sigmoid 0.68997448.
    const float logit = 0.8f;
    Check(ClassificationClassProbabilities(&logit, 1, ClassificationDecisionMode::BinaryLogit, out) == 2,
          "binary logit gives two classes");
    Near(out, {1.0f - 0.68997448f, 0.68997448f}, "binary logit");

    // Binary probability 0.3 (and clamped above 1).
    out.clear();
    const float probability = 0.3f;
    ClassificationClassProbabilities(&probability, 1, ClassificationDecisionMode::BinaryProbability, out);
    Near(out, {0.7f, 0.3f}, "binary probability");
    out.clear();
    const float over = 1.2f;
    ClassificationClassProbabilities(&over, 1, ClassificationDecisionMode::BinaryProbability, out);
    Near(out, {0.0f, 1.0f}, "binary probability clamped");

    // Multi-class logits -> softmax (numpy: exp(x - max) / sum).
    out.clear();
    const float logits[] = {2.0f, 1.0f, 0.1f};
    Check(ClassificationClassProbabilities(logits, 3, ClassificationDecisionMode::MulticlassScores, out) == 3,
          "three classes");
    Near(out, {0.65900114f, 0.24243297f, 0.09856589f}, "softmax of logits");

    // A Softmax output layer's distribution is kept as it is (softmax again
    // would reorder samples within a class).
    out.clear();
    const float distribution[] = {0.1f, 0.7f, 0.2f};
    ClassificationClassProbabilities(distribution, 3, ClassificationDecisionMode::MulticlassScores, out);
    Near(out, {0.1f, 0.7f, 0.2f}, "distribution unchanged");

    // Probabilities append: two samples in one buffer.
    ClassificationClassProbabilities(logits, 3, ClassificationDecisionMode::MulticlassScores, out);
    Check(out.size() == 6, "rows append");

    if (failures) {
        std::fprintf(stderr, "%d check(s) failed\n", failures);
        return 1;
    }
    std::printf("classification probabilities: binary, softmax and distributions as expected. OK\n");
    return 0;
}
