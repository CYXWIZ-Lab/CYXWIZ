#pragma once

// SVC class probabilities from decision values (TOFIX145): Platt's sigmoid per
// class pair, then libsvm's pairwise coupling. Used by ClassicML::PredictSVC.

#include "cyxwiz/classic_ml.h"

#include <arrayfire.h>

namespace cyxwiz::classic_detail {

// Probabilities [n, n_classes] from one-vs-one decision values [n, problems];
// `model` has prob_a / prob_b for every problem.
af::array SVCProbabilities(const af::array& decision, const SVMModel& model);

}  // namespace cyxwiz::classic_detail
