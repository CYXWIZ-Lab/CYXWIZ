#pragma once

// What a trained classifier's scores become (TOFIX129 D11): a decision
// threshold for a yes / no model and a probability calibration, chosen in
// Results > Test (Threshold, Calibration) and kept with the model. Test
// applies it, a checkpoint and a .cyxmodel package carry it, and the local
// inference server returns probabilities and the predicted class with it.
// Pure data in and out.

#include "graph_model.h"

#include <string>
#include <vector>

namespace cyxwiz {

struct ModelDecision {
    enum class Calibration { None, Temperature, Platt, Isotonic };
    // How the model's outputs become probabilities (its loss): "softmax",
    // "sigmoid" (one logit per class, or one for a yes / no model) or
    // "probabilities" (already).
    std::string scores = "softmax";

    bool has_threshold = false;
    int positive = 1;          // the class the threshold is for
    double threshold = 0.5;    // predicted when its (calibrated) probability reaches this

    Calibration calibration = Calibration::None;
    double temperature = 1.0;
    double platt_a = 0.0, platt_b = 0.0;     // P = 1 / (1 + exp(a p + b)) of the positive class
    std::vector<double> isotonic_x, isotonic_y;

    bool Empty() const { return !has_threshold && calibration == Calibration::None; }
};

// Calibrates one row's class probabilities in place and returns the
// predicted class (the threshold for a yes / no model, else the top class).
int ApplyModelDecision(const ModelDecision& decision, std::vector<float>& probabilities);

// One row's raw outputs as probabilities, by `decision.scores`.
std::vector<float> DecisionProbabilities(const ModelDecision& decision, const float* outputs, size_t width);

std::string ModelDecisionToJson(const ModelDecision& decision);
// False (and `decision` unchanged) when `text` is not a decision.
bool ModelDecisionFromJson(const std::string& text, ModelDecision& decision);

// The `scores` word for a model trained with this loss.
std::string DecisionScoresForLoss(gui::NodeType loss_type);

// "threshold 0.30 for MALE; temperature 0.776"; "" when empty.
std::string DescribeModelDecision(const ModelDecision& decision, const std::vector<std::string>& class_names);

}  // namespace cyxwiz
