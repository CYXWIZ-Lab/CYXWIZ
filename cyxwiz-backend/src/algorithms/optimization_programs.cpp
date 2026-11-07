// Linear and convex quadratic programs on HiGHS (TOFIX129 D10): the solver
// scipy.optimize.linprog uses. Any number of variables and rows; the answer
// comes back with each row's room, binding flag and shadow price.

// arrayfire.h brings windows.h; its min / max macros break HiGHS's headers.
#ifndef NOMINMAX
#define NOMINMAX
#endif

#include "cyxwiz/optimization.h"

#include "cyxwiz/linear_algebra.h"
#include "arrayfire_host_values.h"

#include <Highs.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>

namespace cyxwiz {

namespace {

std::string RowName(const LinearConstraint& row, size_t index) {
    return row.name.empty() ? "Row " + std::to_string(index + 1) : row.name;
}

// "" when the sizes and numbers fit, else why not.
std::string CheckShape(size_t n, const std::vector<LinearConstraint>& rows, const std::vector<VariableBound>& bounds) {
    if (n == 0) return "The problem has no variables";
    for (size_t i = 0; i < rows.size(); ++i) {
        if (rows[i].coefficients.size() != n)
            return RowName(rows[i], i) + " has " + std::to_string(rows[i].coefficients.size()) +
                   " coefficients; the problem has " + std::to_string(n) + " variables";
        for (const double v : rows[i].coefficients)
            if (!std::isfinite(v)) return RowName(rows[i], i) + " has a coefficient that is not a number";
        if (!std::isfinite(rows[i].rhs)) return RowName(rows[i], i) + " has a right-hand side that is not a number";
    }
    if (!bounds.empty() && bounds.size() != n)
        return "There are " + std::to_string(bounds.size()) + " bounds for " + std::to_string(n) + " variables";
    for (size_t j = 0; j < bounds.size(); ++j)
        if (std::isnan(bounds[j].lower) || std::isnan(bounds[j].upper) || bounds[j].lower > bounds[j].upper)
            return "Variable " + std::to_string(j + 1) + " has a lower bound above its upper bound";
    return "";
}

HighsLp BuildLp(const std::vector<double>& cost, bool maximize, const std::vector<LinearConstraint>& rows,
                const std::vector<VariableBound>& bounds, double default_lower) {
    const size_t n = cost.size();
    HighsLp lp;
    lp.num_col_ = static_cast<HighsInt>(n);
    lp.num_row_ = static_cast<HighsInt>(rows.size());
    lp.sense_ = maximize ? ObjSense::kMaximize : ObjSense::kMinimize;
    lp.col_cost_ = cost;
    for (size_t j = 0; j < n; ++j) {
        lp.col_lower_.push_back(bounds.empty() ? default_lower : bounds[j].lower);
        lp.col_upper_.push_back(bounds.empty() ? kHighsInf : bounds[j].upper);
    }
    for (const auto& row : rows) {
        lp.row_lower_.push_back(row.sense == ConstraintSense::LessEqual ? -kHighsInf : row.rhs);
        lp.row_upper_.push_back(row.sense == ConstraintSense::GreaterEqual ? kHighsInf : row.rhs);
    }
    lp.a_matrix_.format_ = MatrixFormat::kColwise;
    lp.a_matrix_.num_col_ = lp.num_col_;
    lp.a_matrix_.num_row_ = lp.num_row_;
    lp.a_matrix_.start_.assign(1, 0);
    for (size_t j = 0; j < n; ++j) {
        for (size_t i = 0; i < rows.size(); ++i) {
            const double v = rows[i].coefficients[j];
            if (v == 0.0) continue;
            lp.a_matrix_.index_.push_back(static_cast<HighsInt>(i));
            lp.a_matrix_.value_.push_back(v);
        }
        lp.a_matrix_.start_.push_back(static_cast<HighsInt>(lp.a_matrix_.index_.size()));
    }
    return lp;
}

// Runs the model; on "unbounded or infeasible" runs again without presolve,
// which tells the two apart.
HighsModelStatus Run(Highs& highs, const HighsModel& model) {
    highs.setOptionValue("output_flag", false);
    if (highs.passModel(model) == HighsStatus::kError) return HighsModelStatus::kModelError;
    highs.run();
    HighsModelStatus status = highs.getModelStatus();
    if (status == HighsModelStatus::kUnboundedOrInfeasible) {
        highs.setOptionValue("presolve", "off");
        highs.clearSolver();
        highs.run();
        status = highs.getModelStatus();
    }
    return status;
}

ProgramResult Read(Highs& highs, HighsModelStatus status, const std::vector<LinearConstraint>& rows, bool maximize,
                   bool quadratic) {
    ProgramResult r;
    const HighsInfo& info = highs.getInfo();
    r.iterations = static_cast<int>(std::max<HighsInt>(0, info.simplex_iteration_count) +
                                    std::max<HighsInt>(0, info.qp_iteration_count) +
                                    std::max<HighsInt>(0, info.ipm_iteration_count));
    switch (status) {
        case HighsModelStatus::kOptimal: break;
        case HighsModelStatus::kInfeasible:
            r.status = ProgramStatus::Infeasible;
            r.error_message = "No point satisfies every constraint and bound together";
            return r;
        case HighsModelStatus::kUnbounded:
        case HighsModelStatus::kUnboundedOrInfeasible:
            r.status = ProgramStatus::Unbounded;
            r.error_message = quadratic || !maximize
                ? "The objective falls without limit: no constraint stops some direction"
                : "The objective grows without limit: no constraint stops some direction";
            return r;
        case HighsModelStatus::kIterationLimit:
            r.status = ProgramStatus::Failed;
            r.error_message = "The solver did not settle within " + std::to_string(r.iterations) + " steps";
            return r;
        default:
            r.status = ProgramStatus::Failed;
            r.error_message = "The solver stopped: " + highs.modelStatusToString(status);
            return r;
    }
    const HighsSolution& solution = highs.getSolution();
    r.status = ProgramStatus::Optimal;
    r.success = true;
    r.objective = info.objective_function_value;
    r.x = solution.col_value;
    for (size_t i = 0; i < rows.size(); ++i) {
        const double activity = solution.row_value[i];
        double room = 0.0;
        switch (rows[i].sense) {
            case ConstraintSense::LessEqual: room = rows[i].rhs - activity; break;
            case ConstraintSense::GreaterEqual: room = activity - rows[i].rhs; break;
            case ConstraintSense::Equal: room = std::abs(activity - rows[i].rhs); break;
        }
        room = std::max(0.0, room);
        r.slack.push_back(room);
        r.binding.push_back(room <= 1e-9 * (1.0 + std::abs(rows[i].rhs)));
        // HiGHS's row dual is d(objective) / d(row bound) for either sense.
        r.shadow_prices.push_back(solution.row_dual[i]);
    }
    return r;
}

}  // namespace

ProgramResult Optimization::SolveLinearProgram(const LinearProgram& problem) {
    ProgramResult r;
    r.error_message = CheckShape(problem.objective.size(), problem.constraints, problem.bounds);
    for (const double v : problem.objective)
        if (r.error_message.empty() && !std::isfinite(v)) r.error_message = "The objective has a coefficient that is not a number";
    if (!r.error_message.empty()) return r;
    try {
        Highs highs;
        HighsModel model;
        model.lp_ = BuildLp(problem.objective, problem.maximize, problem.constraints, problem.bounds, 0.0);
        const HighsModelStatus status = Run(highs, model);
        return Read(highs, status, problem.constraints, problem.maximize, false);
    } catch (const std::exception& e) {
        r.error_message = std::string("The solver failed: ") + e.what();
        return r;
    }
}

QuadraticProgram Optimization::PortfolioProgram(const std::vector<double>& means,
                                                const std::vector<std::vector<double>>& covariance,
                                                double target_return, bool long_only) {
    QuadraticProgram p;
    p.Q = covariance;
    p.c.assign(means.size(), 0.0);
    p.constraints.push_back({"Budget", std::vector<double>(means.size(), 1.0), ConstraintSense::Equal, 1.0});
    if (!std::isnan(target_return))
        p.constraints.push_back({"Return", means, ConstraintSense::GreaterEqual, target_return});
    p.bounds.assign(means.size(), long_only ? VariableBound{0.0, kHighsInf} : VariableBound{-kHighsInf, kHighsInf});
    return p;
}

ProgramResult Optimization::SolveQuadraticProgram(const QuadraticProgram& problem) {
    ProgramResult r;
    const size_t n = problem.c.size();
    r.error_message = CheckShape(n, problem.constraints, problem.bounds);
    if (r.error_message.empty() && problem.Q.size() != n)
        r.error_message = "Q has " + std::to_string(problem.Q.size()) + " rows; c has " + std::to_string(n) + " entries";
    for (size_t i = 0; r.error_message.empty() && i < n; ++i)
        if (problem.Q[i].size() != n) r.error_message = "Q must be square (" + std::to_string(n) + " x " + std::to_string(n) + ")";
    for (size_t i = 0; r.error_message.empty() && i < n; ++i) {
        for (size_t j = 0; r.error_message.empty() && j < n; ++j) {
            if (!std::isfinite(problem.Q[i][j]) || !std::isfinite(problem.c[i]))
                r.error_message = "Q and c must be numbers";
            else if (std::abs(problem.Q[i][j] - problem.Q[j][i]) > 1e-12 * (1.0 + std::abs(problem.Q[i][j])))
                r.error_message = "Q must be symmetric: Q[" + std::to_string(i + 1) + "][" + std::to_string(j + 1) +
                                  "] differs from Q[" + std::to_string(j + 1) + "][" + std::to_string(i + 1) + "]";
        }
    }
    if (!r.error_message.empty()) return r;

    // Convex only: the smallest eigenvalue of Q may not be negative.
    std::vector<std::vector<double>> symmetric = problem.Q;
    double frobenius = 0.0;
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < n; ++j) {
            symmetric[i][j] = 0.5 * (problem.Q[i][j] + problem.Q[j][i]);
            frobenius += symmetric[i][j] * symmetric[i][j];
        }
    EigenResult eigen;
    try {
        eigen = LinearAlgebra::Eigen(symmetric);
    } catch (const std::exception& e) {
        eigen.success = false;
        eigen.error_message = e.what();
    }
    if (!eigen.success) {
        r.error_message = "Q's eigenvalues could not be found: " + eigen.error_message;
        return r;
    }
    double low = 0.0, high = 0.0;
    for (size_t k = 0; k < eigen.eigenvalues.size(); ++k) {
        const double v = eigen.eigenvalues[k].real();
        if (k == 0 || v < low) low = v;
        if (k == 0 || v > high) high = v;
    }
    const double rounding = DeviceHasDouble() ? 1e-10 * std::max(1.0, std::abs(high)) : 1e-6 * (1.0 + std::sqrt(frobenius));
    if (low < -rounding) {
        char buffer[200];
        std::snprintf(buffer, sizeof buffer,
                      "Q is not positive semidefinite (its smallest eigenvalue is %.6g), so the problem has no single "
                      "minimum; this solver needs a convex Q",
                      low);
        r.status = ProgramStatus::NotConvex;
        r.error_message = buffer;
        return r;
    }

    // HiGHS's tolerances are absolute: an objective of entries near 1e-4 (a
    // covariance of returns) never meets them and its QP solver runs without
    // end. The objective is solved scaled to entries of size 1 (the minimizer
    // is the same), its value and the row duals scaled back.
    double largest = 0.0;
    for (size_t i = 0; i < n; ++i) {
        largest = std::max(largest, std::abs(problem.c[i]));
        for (size_t j = 0; j < n; ++j) largest = std::max(largest, std::abs(problem.Q[i][j]));
    }
    const double scale = largest > 0.0 ? 1.0 / largest : 1.0;
    std::vector<double> c = problem.c;
    for (double& v : c) v *= scale;

    try {
        Highs highs;
        HighsModel model;
        model.lp_ = BuildLp(c, false, problem.constraints, problem.bounds, -kHighsInf);
        HighsHessian& h = model.hessian_;
        h.dim_ = static_cast<HighsInt>(n);
        h.format_ = HessianFormat::kTriangular;  // the lower triangle, column by column
        h.start_.assign(1, 0);
        for (size_t j = 0; j < n; ++j) {
            for (size_t i = j; i < n; ++i) {
                if (problem.Q[i][j] == 0.0) continue;
                h.index_.push_back(static_cast<HighsInt>(i));
                h.value_.push_back(problem.Q[i][j] * scale);
            }
            h.start_.push_back(static_cast<HighsInt>(h.index_.size()));
        }
        if (h.index_.empty()) model.hessian_.clear();  // Q = 0: a linear program
        // HiGHS adds 1e-7 to Q's diagonal by default, which moves the answer
        // by about that much (0.75000004 for 0.75); 1e-12 keeps a guard for a
        // semidefinite Q without the bias.
        highs.setOptionValue("qp_regularization_value", 1e-12);
        for (const char* option : {"primal_feasibility_tolerance", "dual_feasibility_tolerance", "optimality_tolerance"})
            highs.setOptionValue(option, 1e-10);
        // A guard: an active-set QP of this size settles in far fewer steps.
        highs.setOptionValue("qp_iteration_limit", static_cast<HighsInt>(std::max<size_t>(10000, 100 * (n + problem.constraints.size()))));
        const HighsModelStatus status = Run(highs, model);
        ProgramResult solved = Read(highs, status, problem.constraints, false, true);
        solved.objective /= scale;
        for (double& price : solved.shadow_prices) price /= scale;
        return solved;
    } catch (const std::exception& e) {
        r.error_message = std::string("The solver failed: ") + e.what();
        return r;
    }
}

} // namespace cyxwiz
