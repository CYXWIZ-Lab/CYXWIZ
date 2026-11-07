#pragma once

// Integrals, derivatives and curvature of a typed formula (TOFIX129 D10). The
// formula is a cyxwiz::Expression; its points are evaluated on the selected
// ArrayFire device, many at a time (float64 where the device has it).

#include "cyxwiz/api_export.h"
#include "cyxwiz/expression.h"

#include <limits>
#include <string>
#include <vector>

namespace cyxwiz {

enum class QuadratureRule {
    Adaptive,       // Gauss-Kronrod 21 on bisected intervals, as scipy.integrate.quad
    Trapezoid,      // n equal intervals
    Midpoint,       // n equal intervals
    Simpson,        // n equal intervals, n even
    GaussLegendre   // n points, exact for polynomials up to degree 2n - 1
};

struct CYXWIZ_API QuadratureOptions {
    QuadratureRule rule = QuadratureRule::Adaptive;
    int n = 100;                            // intervals, or Gauss points
    double absolute_tolerance = 1.49e-8;    // Adaptive: scipy's defaults
    double relative_tolerance = 1.49e-8;
    int max_intervals = 50;
};

struct CYXWIZ_API QuadratureResult {
    bool success = false;
    std::string error_message;
    double value = 0.0;
    // Adaptive: the Gauss-Kronrod estimate of |value - integral|; the fixed
    // rules have none (NaN).
    double error_estimate = std::numeric_limits<double>::quiet_NaN();
    bool converged = true;                  // Adaptive: met the tolerance
    int evaluations = 0;
    int intervals = 0;
};

enum class DifferenceRule { Forward, Backward, Central };

struct CYXWIZ_API DifferentiationResult {
    bool success = false;
    std::string error_message;
    bool has_exact = false;                 // false when f uses max, mod, ...
    std::string first_formula, second_formula;
    double exact_first = 0.0, exact_second = 0.0;
    double numerical_first = 0.0, numerical_second = 0.0;
    double step = 0.0, second_step = 0.0;   // the h used
};

// The numerical first derivative's error against the exact one, by step h.
struct CYXWIZ_API StepErrorCurve {
    bool success = false;
    std::string error_message;
    std::vector<double> steps;
    std::vector<double> forward_error, central_error;
};

struct CYXWIZ_API CurvatureResult {
    bool success = false;
    std::string error_message;
    std::vector<double> gradient;
    std::vector<std::vector<double>> hessian;
    std::vector<double> eigenvalues;        // ascending
};

// The Hessian's smallest eigenvalue on a grid of a two-variable formula.
struct CYXWIZ_API RangeCurvature {
    bool success = false;
    std::string error_message;
    int points = 0;
    int negative_points = 0;                // curving down in some direction
    int positive_points = 0;                // curving up in every direction
    double first_negative_x = 0.0, first_negative_y = 0.0;
};

class CYXWIZ_API Calculus {
public:
    // The integral of f (one variable) from a to b. Adaptive takes infinite
    // bounds; the fixed rules need finite ones.
    static QuadratureResult Integrate(const Expression& f, double a, double b,
                                      const QuadratureOptions& options = {});

    // f'(x) and f''(x): exact from the formula, and numerical by `rule` with
    // step h (0: the step that suits float64 for that rule).
    static DifferentiationResult Differentiate(const Expression& f, double x,
                                          DifferenceRule rule = DifferenceRule::Central, double h = 0.0);

    // Forward and central errors at `count` steps from h_from down to h_to,
    // evenly spaced in log10.
    static StepErrorCurve ErrorByStep(const Expression& f, double x,
                                      double h_from = 1e-1, double h_to = 1e-12, int count = 45);

    // Gradient, Hessian and its eigenvalues at `point`, all exact.
    static CurvatureResult CurvatureAt(const Expression& f, const std::vector<double>& point);

    // The Hessian's eigenvalues at per_side x per_side points of
    // [x_min, x_max] x [y_min, y_max] (f of two variables).
    static RangeCurvature CurvatureOverRange(const Expression& f, double x_min, double x_max,
                                             double y_min, double y_max, int per_side = 40);

    // Gauss-Legendre nodes and weights on [-1, 1].
    static void GaussLegendre(int n, std::vector<double>& nodes, std::vector<double>& weights);
};

} // namespace cyxwiz
