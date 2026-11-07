// SPDX-License-Identifier: Apache-2.0
// Part of the CyxWiz plugin SDK (see cmake/sdk/NOTICE).
#pragma once

// Linear and convex quadratic programs (HiGHS, the solver scipy uses) and
// the training optimizers' paths on a typed formula (TOFIX129 D10). The
// calculus of a formula is in calculus.h, the formula in expression.h.

#include "api_export.h"
#include "expression.h"
#include "optimizers/optimizer_base.h"

#include <functional>
#include <limits>
#include <string>
#include <vector>

namespace cyxwiz {

// ============================================================================
// Programs and optimizer paths (TOFIX129 D10)
// ============================================================================

enum class ConstraintSense { LessEqual, GreaterEqual, Equal };

// One row: coefficients . x  (<=, >= or =)  rhs.
struct CYXWIZ_API LinearConstraint {
    std::string name;
    std::vector<double> coefficients;
    ConstraintSense sense = ConstraintSense::LessEqual;
    double rhs = 0.0;
};

// lower <= x <= upper; +-infinity for no limit.
struct CYXWIZ_API VariableBound {
    double lower = 0.0;
    double upper = std::numeric_limits<double>::infinity();
};

// maximize or minimize objective . x subject to the rows and the bounds
// (empty bounds: every x >= 0).
struct CYXWIZ_API LinearProgram {
    bool maximize = false;
    std::vector<double> objective;
    std::vector<LinearConstraint> constraints;
    std::vector<VariableBound> bounds;
};

// minimize 0.5 x' Q x + c' x subject to the rows and the bounds (empty
// bounds: every x free). Q must be symmetric positive semidefinite.
struct CYXWIZ_API QuadraticProgram {
    std::vector<std::vector<double>> Q;
    std::vector<double> c;
    std::vector<LinearConstraint> constraints;
    std::vector<VariableBound> bounds;
};

enum class ProgramStatus { Optimal, Infeasible, Unbounded, NotConvex, Failed };

struct CYXWIZ_API ProgramResult {
    bool success = false;                   // Optimal
    std::string error_message;              // why not, in words
    ProgramStatus status = ProgramStatus::Failed;
    double objective = 0.0;
    std::vector<double> x;
    // Per row: room left (|rhs - row value|; 0 for a met equality), whether
    // it holds with no room, and its shadow price: how much the best
    // objective changes when rhs grows by 1.
    std::vector<double> slack;
    std::vector<bool> binding;
    std::vector<double> shadow_prices;
    int iterations = 0;
};

// The update rules training uses (cyxwiz::Optimizer), on a formula's
// variables as parameters.
struct CYXWIZ_API OptimizerSetting {
    OptimizerType type = OptimizerType::Adam;
    double learning_rate = 0.001;
    double momentum = 0.0;                  // SGD and RMSprop
};

struct CYXWIZ_API OptimizerPath {
    bool success = false;
    std::string error_message;
    std::vector<std::vector<double>> points;    // the start, then one per step
    std::vector<double> values;                 // f at each point
    std::vector<double> gradient_norms;         // |grad f| at each point
    bool converged = false;                     // |grad f| fell below the tolerance
    bool diverged = false;                      // left the finite numbers
    int diverged_at = 0;                        // the step that did
    bool stopped = false;                       // the stop query said so
};

// ============================================================================
// Optimization Class
// ============================================================================

class CYXWIZ_API Optimization {
public:
    // ==================== Programs (HiGHS) ====================

    // Any number of variables and rows. Infeasible and unbounded problems
    // come back with that status and the reason in words.
    static ProgramResult SolveLinearProgram(const LinearProgram& problem);

    // Refuses a Q that is not positive semidefinite (NotConvex, naming its
    // most negative eigenvalue).
    static ProgramResult SolveQuadraticProgram(const QuadraticProgram& problem);

    // ==================== Optimizer paths ====================

    // `steps` updates of f's variables from `start` by the training
    // optimizer `setting` names, the gradient exact from the formula. Ends
    // early when |grad f| < tolerance, when a value stops being finite
    // (diverged), or when `stop` returns true.
    static OptimizerPath FollowOptimizer(const Expression& f, const std::vector<double>& start,
                                         const OptimizerSetting& setting, int steps,
                                         double tolerance = 1e-8,
                                         const std::function<bool()>& stop = {});
};

} // namespace cyxwiz
