#pragma once

// pycyxwiz.optimize and pycyxwiz.calculus (TOFIX129 D10): the programs, the
// optimizer paths and the calculus of Tools > Optimization and Calculus,
// checked against scipy and torch by test_python_optimize.py.

#include <pybind11/pybind11.h>

namespace pycyxwiz_bindings {

// pycyxwiz.optimize: linprog, qp, portfolio, follow.
void BindOptimize(pybind11::module_& m);

// pycyxwiz.calculus: Expression, integrate, derivative, step_errors,
// curvature, curvature_over_range, gauss_legendre.
void BindCalculus(pybind11::module_& m);

}  // namespace pycyxwiz_bindings
