#pragma once

// pycyxwiz.model (TOFIX129 D11): the Engine's Tools > Model and Results >
// Test > Calibration from Python, on the same SDK code (model_cost.h,
// learning_rate_range.h, model_interpretability.h, calibration.h); checked
// against PyTorch and scikit-learn by test_python_model.py.

#include <pybind11/pybind11.h>

namespace pycyxwiz_bindings {

// pycyxwiz.model: summary, LearningRateRange, lr_range_test, saliency,
// smooth_grad, integrated_gradients, reliability, fit_temperature,
// with_temperature, fit_platt, apply_platt, fit_isotonic, apply_isotonic.
void BindModelTools(pybind11::module_& m);

}  // namespace pycyxwiz_bindings
