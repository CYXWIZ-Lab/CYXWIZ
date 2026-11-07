#include "bindings_optimize.h"

#include <pybind11/stl.h>

#include "cyxwiz/calculus.h"
#include "cyxwiz/optimization.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <limits>
#include <optional>
#include <string>
#include <vector>

namespace py = pybind11;
using namespace pybind11::literals;

namespace pycyxwiz_bindings {
namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();

std::string Lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return s;
}

// x for one value, x and y for two, x1 ... xn for more.
std::vector<std::string> DefaultVariables(size_t n) {
    if (n == 1) return {"x"};
    if (n == 2) return {"x", "y"};
    std::vector<std::string> names;
    for (size_t i = 1; i <= n; ++i) names.push_back("x" + std::to_string(i));
    return names;
}

// A formula given as text (read with `variables`) or as a calculus.Expression.
cyxwiz::Expression Formula(const py::object& formula, const std::vector<std::string>& variables) {
    if (py::isinstance<cyxwiz::Expression>(formula)) return formula.cast<cyxwiz::Expression>();
    if (!py::isinstance<py::str>(formula)) throw py::type_error("formula must be text or a calculus.Expression");
    cyxwiz::Expression f = cyxwiz::Expression::Parse(formula.cast<std::string>(), variables);
    if (!f.Valid()) throw py::value_error(f.Error());
    return f;
}

double Limit(const py::handle& v, double none) { return v.is_none() ? none : v.cast<double>(); }

// None: the program's default; one (low, high) pair: every variable; else a
// pair per variable. None inside a pair is no limit.
std::vector<cyxwiz::VariableBound> Bounds(const py::object& bounds, size_t n) {
    std::vector<cyxwiz::VariableBound> out;
    if (bounds.is_none()) return out;
    const auto seq = bounds.cast<py::sequence>();
    const auto pair = [](const py::handle& item) {
        const auto p = item.cast<py::sequence>();
        if (py::len(p) != 2) throw py::value_error("a bound is (low, high), None for no limit");
        return cyxwiz::VariableBound{Limit(p[0], -kInf), Limit(p[1], kInf)};
    };
    if (py::len(seq) == 2 && !py::isinstance<py::sequence>(seq[0])) return std::vector<cyxwiz::VariableBound>(n, pair(seq));
    if (py::len(seq) != n) throw py::value_error("bounds needs one (low, high) per variable, or one pair for all");
    for (const auto item : seq) out.push_back(pair(item));
    return out;
}

cyxwiz::ConstraintSense Sense(const std::string& s) {
    if (s == "<=") return cyxwiz::ConstraintSense::LessEqual;
    if (s == ">=") return cyxwiz::ConstraintSense::GreaterEqual;
    if (s == "=" || s == "==") return cyxwiz::ConstraintSense::Equal;
    throw py::value_error("a constraint's sense is '<=', '>=' or '=' ('" + s + "' given)");
}

// (coefficients, sense, rhs) or (coefficients, sense, rhs, name) per row.
std::vector<cyxwiz::LinearConstraint> Rows(const py::object& constraints) {
    std::vector<cyxwiz::LinearConstraint> rows;
    if (constraints.is_none()) return rows;
    for (const auto item : constraints) {
        const auto t = item.cast<py::sequence>();
        const size_t k = py::len(t);
        if (k != 3 && k != 4)
            throw py::value_error("a constraint is (coefficients, sense, rhs) or (coefficients, sense, rhs, name)");
        cyxwiz::LinearConstraint row;
        row.coefficients = t[0].cast<std::vector<double>>();
        row.sense = Sense(t[1].cast<std::string>());
        row.rhs = t[2].cast<double>();
        row.name = k == 4 ? t[3].cast<std::string>() : "C" + std::to_string(rows.size() + 1);
        rows.push_back(std::move(row));
    }
    return rows;
}

const char* StatusName(cyxwiz::ProgramStatus s) {
    switch (s) {
        case cyxwiz::ProgramStatus::Optimal: return "optimal";
        case cyxwiz::ProgramStatus::Infeasible: return "infeasible";
        case cyxwiz::ProgramStatus::Unbounded: return "unbounded";
        case cyxwiz::ProgramStatus::NotConvex: return "not convex";
        case cyxwiz::ProgramStatus::Failed: break;
    }
    return "failed";
}

py::dict Answer(const cyxwiz::ProgramResult& r) {
    return py::dict("success"_a = r.success, "status"_a = StatusName(r.status), "message"_a = r.error_message,
                    "x"_a = r.x, "fun"_a = r.success ? py::object(py::float_(r.objective)) : py::object(py::none()),
                    "slack"_a = r.slack, "binding"_a = r.binding, "shadow_prices"_a = r.shadow_prices,
                    "iterations"_a = r.iterations);
}

cyxwiz::OptimizerType OptimizerOf(const std::string& name) {
    const std::string n = Lower(name);
    if (n == "sgd") return cyxwiz::OptimizerType::SGD;
    if (n == "adam") return cyxwiz::OptimizerType::Adam;
    if (n == "adamw") return cyxwiz::OptimizerType::AdamW;
    if (n == "nadam") return cyxwiz::OptimizerType::NAdam;
    if (n == "rmsprop") return cyxwiz::OptimizerType::RMSprop;
    if (n == "adagrad") return cyxwiz::OptimizerType::AdaGrad;
    throw py::value_error("optimizer is one of sgd, adam, adamw, nadam, rmsprop, adagrad ('" + name + "' given)");
}

cyxwiz::QuadratureRule RuleOf(const std::string& name) {
    const std::string n = Lower(name);
    if (n == "adaptive") return cyxwiz::QuadratureRule::Adaptive;
    if (n == "trapezoid") return cyxwiz::QuadratureRule::Trapezoid;
    if (n == "midpoint") return cyxwiz::QuadratureRule::Midpoint;
    if (n == "simpson") return cyxwiz::QuadratureRule::Simpson;
    if (n == "gauss" || n == "gauss_legendre") return cyxwiz::QuadratureRule::GaussLegendre;
    throw py::value_error("rule is one of adaptive, trapezoid, midpoint, simpson, gauss ('" + name + "' given)");
}

cyxwiz::DifferenceRule DifferenceOf(const std::string& name) {
    const std::string n = Lower(name);
    if (n == "central") return cyxwiz::DifferenceRule::Central;
    if (n == "forward") return cyxwiz::DifferenceRule::Forward;
    if (n == "backward") return cyxwiz::DifferenceRule::Backward;
    throw py::value_error("rule is one of central, forward, backward ('" + name + "' given)");
}

py::object NoneIfNaN(double v) { return std::isnan(v) ? py::object(py::none()) : py::object(py::float_(v)); }

}  // namespace

void BindCalculus(py::module_& m) {
    auto calculus = m.def_submodule("calculus", "Integrals, derivatives and curvature of a typed formula");

    py::class_<cyxwiz::Expression>(calculus, "Expression",
        "A typed formula, such as '(1 - x)^2 + 100*(y - x^2)^2': + - * / % ^ (right associative; -x^2 is -(x^2)), "
        "pi e phi, sin cos tan asin acos atan sinh cosh tanh sqrt cbrt exp log ln log10 log2 abs floor ceil round "
        "sign factorial, and pow mod fmod min max atan2. Its variables are named in order; any other name is refused.")
        .def(py::init([](const std::string& text, const std::vector<std::string>& variables, bool degrees) {
                 cyxwiz::Expression f = cyxwiz::Expression::Parse(
                     text, variables, degrees ? cyxwiz::Expression::AngleUnit::Degrees : cyxwiz::Expression::AngleUnit::Radians);
                 if (!f.Valid()) throw py::value_error(f.Error());
                 return f;
             }),
             "text"_a, "variables"_a = std::vector<std::string>{"x"}, "degrees"_a = false)
        .def("__call__", [](const cyxwiz::Expression& f, const py::args& args) {
                 std::vector<double> values;
                 for (const auto v : args) values.push_back(v.cast<double>());
                 if (values.size() != f.Variables().size())
                     throw py::value_error("the formula takes " + std::to_string(f.Variables().size()) + " values");
                 std::string problem;
                 const double value = f.Evaluate(values, &problem);
                 if (!problem.empty()) throw py::value_error(problem);
                 return value;
             }, "The value at one point, one number per variable in order; a division by zero or a value outside a "
                "function's domain raises ValueError")
        .def("evaluate_many", [](const cyxwiz::Expression& f, const std::vector<std::vector<double>>& columns) {
                 if (columns.size() != f.Variables().size())
                     throw py::value_error("one column of values per variable (" + std::to_string(f.Variables().size()) + ")");
                 for (const auto& column : columns)
                     if (column.size() != columns.front().size()) throw py::value_error("the columns must be the same length");
                 return f.EvaluateMany(columns);
             }, "The values at many points, a list of values per variable; on the device (float64 where it has it). "
                "Points without a value give nan or inf", "columns"_a)
        .def("derivative", [](const cyxwiz::Expression& f, const py::object& variable) {
                 size_t index = 0;
                 const auto& names = f.Variables();
                 if (variable.is_none()) {
                     if (names.size() != 1) throw py::value_error("name the variable to differentiate by");
                 } else if (py::isinstance<py::str>(variable)) {
                     const auto it = std::find(names.begin(), names.end(), variable.cast<std::string>());
                     if (it == names.end()) throw py::value_error("not a variable of the formula: " + variable.cast<std::string>());
                     index = static_cast<size_t>(it - names.begin());
                 } else {
                     index = variable.cast<size_t>();
                     if (index >= names.size()) throw py::value_error("variable index out of range");
                 }
                 cyxwiz::Expression d = f.Derivative(index);
                 if (!d.Valid()) throw py::value_error(d.Error());
                 return d;
             }, "The exact partial derivative by a variable (its name or index), simplified", "variable"_a = py::none())
        .def_property_readonly("variables", &cyxwiz::Expression::Variables)
        .def_property_readonly("text", &cyxwiz::Expression::Text)
        .def("__str__", &cyxwiz::Expression::ToString)
        .def("__repr__", [](const cyxwiz::Expression& f) { return "Expression('" + f.ToString() + "')"; });

    calculus.def("integrate", [](const py::object& formula, double a, double b, const std::string& rule, int n,
                                 double epsabs, double epsrel, int limit, const std::string& variable) {
        cyxwiz::QuadratureOptions options;
        options.rule = RuleOf(rule);
        options.n = n;
        options.absolute_tolerance = epsabs;
        options.relative_tolerance = epsrel;
        options.max_intervals = limit;
        const auto r = cyxwiz::Calculus::Integrate(Formula(formula, {variable}), a, b, options);
        if (!r.success) throw py::value_error(r.error_message);
        return py::dict("value"_a = r.value, "error"_a = NoneIfNaN(r.error_estimate), "converged"_a = r.converged,
                        "evaluations"_a = r.evaluations, "intervals"_a = r.intervals);
    }, "The integral of a formula of one variable from a to b (either may be +-inf with the adaptive rule). "
       "adaptive: Gauss-Kronrod 21 on bisected intervals, as scipy.integrate.quad, with its error estimate; "
       "trapezoid, midpoint, simpson: n equal intervals; gauss: n Gauss-Legendre points",
       "formula"_a, "a"_a, "b"_a, "rule"_a = "adaptive", "n"_a = 100, "epsabs"_a = 1.49e-8, "epsrel"_a = 1.49e-8,
       "limit"_a = 50, "variable"_a = "x");

    calculus.def("derivative", [](const py::object& formula, double x, const std::string& rule, double h,
                                  const std::string& variable) {
        const auto r = cyxwiz::Calculus::Differentiate(Formula(formula, {variable}), x, DifferenceOf(rule), h);
        if (!r.success) throw py::value_error(r.error_message);
        return py::dict("first"_a = r.has_exact ? py::object(py::float_(r.exact_first)) : py::object(py::none()),
                        "second"_a = r.has_exact ? py::object(py::float_(r.exact_second)) : py::object(py::none()),
                        "first_formula"_a = r.first_formula, "second_formula"_a = r.second_formula,
                        "numerical_first"_a = r.numerical_first, "numerical_second"_a = r.numerical_second,
                        "step"_a = r.step, "second_step"_a = r.second_step);
    }, "f'(x) and f''(x): exact from the formula (None when it has no exact derivative, as with max or mod) and "
       "by differences with step h (0: the step that suits float64 for the rule)",
       "formula"_a, "x"_a, "rule"_a = "central", "h"_a = 0.0, "variable"_a = "x");

    calculus.def("step_errors", [](const py::object& formula, double x, double h_from, double h_to, int count,
                                   const std::string& variable) {
        const auto r = cyxwiz::Calculus::ErrorByStep(Formula(formula, {variable}), x, h_from, h_to, count);
        if (!r.success) throw py::value_error(r.error_message);
        return py::dict("steps"_a = r.steps, "forward_error"_a = r.forward_error, "central_error"_a = r.central_error);
    }, "|numerical f' - exact f'| for the forward and central differences at `count` steps from h_from down to "
       "h_to, evenly spaced in log10",
       "formula"_a, "x"_a, "h_from"_a = 1e-1, "h_to"_a = 1e-12, "count"_a = 45, "variable"_a = "x");

    calculus.def("curvature", [](const py::object& formula, const std::vector<double>& point,
                                 const std::optional<std::vector<std::string>>& variables) {
        const auto r = cyxwiz::Calculus::CurvatureAt(Formula(formula, variables ? *variables : DefaultVariables(point.size())), point);
        if (!r.success) throw py::value_error(r.error_message);
        return py::dict("gradient"_a = r.gradient, "hessian"_a = r.hessian, "eigenvalues"_a = r.eigenvalues);
    }, "The gradient, the Hessian and its eigenvalues (ascending) at a point, all exact: every eigenvalue above 0 "
       "curves up in every direction at this point",
       "formula"_a, "point"_a, "variables"_a = py::none());

    calculus.def("curvature_over_range", [](const py::object& formula, std::pair<double, double> x_range,
                                            std::pair<double, double> y_range, int per_side,
                                            const std::vector<std::string>& variables) {
        const auto r = cyxwiz::Calculus::CurvatureOverRange(Formula(formula, variables), x_range.first, x_range.second,
                                                            y_range.first, y_range.second, per_side);
        if (!r.success) throw py::value_error(r.error_message);
        py::object first = py::none();
        if (r.negative_points > 0) first = py::make_tuple(r.first_negative_x, r.first_negative_y);
        return py::dict("points"_a = r.points, "negative_points"_a = r.negative_points,
                        "positive_points"_a = r.positive_points, "first_negative"_a = first);
    }, "The Hessian's smallest eigenvalue on per_side x per_side points of a two-variable formula: how many points "
       "curve down in some direction (not convex there), and the first of them",
       "formula"_a, "x_range"_a, "y_range"_a, "per_side"_a = 40,
       "variables"_a = std::vector<std::string>{"x", "y"});

    calculus.def("gauss_legendre", [](int n) {
        if (n < 1) throw py::value_error("n must be at least 1");
        std::vector<double> nodes, weights;
        cyxwiz::Calculus::GaussLegendre(n, nodes, weights);
        return py::make_tuple(nodes, weights);
    }, "The n Gauss-Legendre nodes and weights on [-1, 1] (numpy.polynomial.legendre.leggauss)", "n"_a);
}

void BindOptimize(py::module_& m) {
    auto optimize = m.def_submodule("optimize", "Linear and quadratic programs (HiGHS) and the training optimizers' paths");

    optimize.def("linprog", [](const std::vector<double>& c, const py::object& constraints, const py::object& bounds,
                               bool maximize) {
        cyxwiz::LinearProgram p;
        p.maximize = maximize;
        p.objective = c;
        p.constraints = Rows(constraints);
        p.bounds = Bounds(bounds, c.size());
        return Answer(cyxwiz::Optimization::SolveLinearProgram(p));
    }, "Minimize (or maximize) c . x subject to the constraints, each (coefficients, '<=' / '>=' / '=', rhs[, name]), "
       "and the bounds (None: every x >= 0). Returns a dict: success, status ('optimal', 'infeasible', 'unbounded', "
       "'failed'), message, x, fun, slack, binding and shadow_prices (per row: how much the best value changes when "
       "its rhs grows by 1), iterations",
       "c"_a, "constraints"_a = py::none(), "bounds"_a = py::none(), "maximize"_a = false);

    optimize.def("qp", [](const std::vector<std::vector<double>>& Q, const std::vector<double>& c,
                          const py::object& constraints, const py::object& bounds) {
        cyxwiz::QuadraticProgram p;
        p.Q = Q;
        p.c = c;
        p.constraints = Rows(constraints);
        p.bounds = Bounds(bounds, c.size());
        return Answer(cyxwiz::Optimization::SolveQuadraticProgram(p));
    }, "Minimize 0.5 x' Q x + c . x subject to the constraints and the bounds (None: every x free). Q must be "
       "symmetric positive semidefinite, else status 'not convex' names its most negative eigenvalue. Returns the "
       "same dict as linprog",
       "Q"_a, "c"_a, "constraints"_a = py::none(), "bounds"_a = py::none());

    optimize.def("portfolio", [](const std::vector<double>& means, const std::vector<std::vector<double>>& covariance,
                                 const std::optional<double>& target_return, bool long_only) {
        const auto p = cyxwiz::Optimization::PortfolioProgram(
            means, covariance, target_return ? *target_return : std::numeric_limits<double>::quiet_NaN(), long_only);
        return Answer(cyxwiz::Optimization::SolveQuadraticProgram(p));
    }, "The least risky mix of assets (Markowitz), as the Portfolio Weights node: minimize 0.5 w' C w with the "
       "weights summing to 1 (row 'Budget'), earning at least target_return a period (row 'Return'; None: the "
       "minimum-variance mix), and with long_only no weight below 0. means and covariance per period",
       "means"_a, "covariance"_a, "target_return"_a = py::none(), "long_only"_a = true);

    optimize.def("follow", [](const py::object& formula, const std::vector<double>& start, const std::string& optimizer,
                              double lr, double momentum, int steps, double tolerance,
                              const std::optional<std::vector<std::string>>& variables) {
        cyxwiz::OptimizerSetting setting;
        setting.type = OptimizerOf(optimizer);
        setting.learning_rate = lr;
        setting.momentum = momentum;
        const auto f = Formula(formula, variables ? *variables : DefaultVariables(start.size()));
        const auto path = cyxwiz::Optimization::FollowOptimizer(
            f, start, setting, steps, tolerance, [] { return PyErr_CheckSignals() != 0; });
        if (PyErr_Occurred()) throw py::error_already_set();
        if (!path.success) throw py::value_error(path.error_message);
        return py::dict("points"_a = path.points, "values"_a = path.values, "gradient_norms"_a = path.gradient_norms,
                        "converged"_a = path.converged, "diverged"_a = path.diverged,
                        "diverged_at"_a = path.diverged ? py::object(py::int_(path.diverged_at)) : py::object(py::none()));
    }, "The path of a training optimizer (the update rules training uses, in float32 as training) minimizing a "
       "formula from `start`, the gradient exact from the formula. Ends early when |grad f| < tolerance or a value "
       "stops being finite (diverged, with the step). Returns points (the start, then one per step), values, "
       "gradient_norms, converged, diverged, diverged_at",
       "formula"_a, "start"_a, "optimizer"_a = "adam", "lr"_a = 0.001, "momentum"_a = 0.0, "steps"_a = 1000,
       "tolerance"_a = 1e-8, "variables"_a = py::none());
}

}  // namespace pycyxwiz_bindings
