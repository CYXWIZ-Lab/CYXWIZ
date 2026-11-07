// The optimization window's backend against sympy / scipy / torch (TOFIX129
// D10): fixtures/optimization_truth.json from
// reference/generate_optimization_fixtures.py.
//
// expression  cyxwiz::Expression: values and exact derivatives at points
//             (host, and the device for many points at once), parse errors,
//             printing back, and the Calculator that reads through it.
// quad        Calculus::Integrate, adaptive, against the exact value and
//             scipy.integrate.quad; an unconverged answer's estimate covers
//             its error.
// fixed       trapezoid / midpoint / Simpson / Gauss-Legendre and the
//             Legendre nodes against numpy / scipy.
// derivative  exact f' and f'', the numerical ones, the error by step.
// curvature   Rosenbrock's Hessian eigenvalues and the range count.
// lp, qp      Optimization::SolveLinearProgram / SolveQuadraticProgram (HiGHS)
//             against scipy linprog / SLSQP: x, the objective, room, shadow
//             prices (scipy's by re-solving with a raised rhs), statuses.
// paths       Optimization::FollowOptimizer (the training optimizers) against
//             torch.optim in float32.

#include "test_device_selection.h"

#include <cyxwiz/calculus.h>
#include <cyxwiz/expression.h>
#include <cyxwiz/optimization.h>
#include <cyxwiz/utilities.h>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <vector>

using json = nlohmann::json;
using cyxwiz::Calculus;
using cyxwiz::Expression;

namespace {

int checks = 0;

void Check(bool condition, const std::string& message) {
    ++checks;
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        std::exit(1);
    }
}

void Near(double actual, double expected, double tolerance, const std::string& what) {
    Check(std::fabs(actual - expected) <= tolerance * (1.0 + std::fabs(expected)),
          what + ": " + std::to_string(actual) + " vs " + std::to_string(expected));
}

std::vector<double> Doubles(const json& j) {
    std::vector<double> out;
    for (const auto& v : j) out.push_back(v.get<double>());
    return out;
}

std::vector<std::string> Strings(const json& j) {
    std::vector<std::string> out;
    for (const auto& v : j) out.push_back(v.get<std::string>());
    return out;
}

void CheckExpressions(const json& cases) {
    for (const auto& c : cases) {
        const std::string text = c.at("text").get<std::string>();
        const auto variables = Strings(c.at("variables"));
        const Expression f = Expression::Parse(text, variables);
        Check(f.Valid(), text + " parses: " + f.Error());

        // Printed back and parsed again, it is the same formula.
        const Expression again = Expression::Parse(f.ToString(), variables);
        Check(again.Valid(), text + " printed as " + f.ToString() + " parses again: " + again.Error());

        std::vector<Expression> partials;
        for (size_t v = 0; v < variables.size(); ++v) {
            partials.push_back(f.Derivative(v));
            Check(partials.back().Valid(), text + " d/d" + variables[v] + ": " + partials.back().Error());
        }

        std::vector<std::vector<double>> columns(variables.size());
        const auto points = c.at("points");
        for (size_t p = 0; p < points.size(); ++p) {
            const std::vector<double> point = Doubles(points[p]);
            const std::string at = text + " at point " + std::to_string(p);
            Near(f.Evaluate(point), c.at("values")[p].get<double>(), 1e-12, at);
            Near(again.Evaluate(point), c.at("values")[p].get<double>(), 1e-12, at + " (printed " + f.ToString() + ")");
            for (size_t v = 0; v < variables.size(); ++v) {
                Near(partials[v].Evaluate(point), c.at("derivatives")[p][v].get<double>(), 1e-11,
                     at + " d/d" + variables[v] + " = " + partials[v].ToString());
                columns[v].push_back(point[v]);
            }
        }
        if (variables.empty()) continue;
        const std::vector<double> many = f.EvaluateMany(columns);
        Check(many.size() == points.size(), text + ": one device value per point");
        for (size_t p = 0; p < many.size(); ++p)
            Near(many[p], c.at("values")[p].get<double>(), 1e-12, text + " on the device at point " + std::to_string(p));
    }
}

void CheckRefusals() {
    struct Bad { const char* text; const char* says; };
    for (const Bad& bad : {Bad{"(x + 1", "Expected ')' at 7"}, Bad{"x +", "Expected a value at 4"},
                           Bad{"foo(x)", "Unknown function 'foo' at 1"}, Bad{"z * 2", "Unknown name 'z' at 1"},
                           Bad{"", "Empty expression"}, Bad{"2 $ 3", "Unexpected '$' at 3"},
                           Bad{"atan2(x)", "atan2 takes two values"}, Bad{"sin(x, 1)", "sin takes one value"}}) {
        const Expression e = Expression::Parse(bad.text, {"x"});
        Check(!e.Valid() && e.Error().find(bad.says) != std::string::npos,
              std::string("'") + bad.text + "' refused with \"" + bad.says + "\": got \"" + e.Error() + "\"");
    }
    const Expression rounding = Expression::Parse("max(x, 1) + floor(x)", {"x"}).Derivative(0);
    Check(!rounding.Valid() && rounding.Error().find("max has no derivative") != std::string::npos,
          "max has no derivative: " + rounding.Error());
    const Expression flat = Expression::Parse("floor(x) + 3*x", {"x"}).Derivative(0);
    Check(flat.Valid() && flat.ToString() == "3", "floor counts as flat: " + flat.ToString());
    std::string problem;
    Expression::Parse("1/(x - 2)", {"x"}).Evaluate({2.0}, &problem);
    Check(problem == "Division by zero", "division by zero named: " + problem);
}

void CheckReadable() {
    struct Pair { const char* text; const char* derivative; };
    for (const Pair& p : {Pair{"sin(x)", "cos(x)"}, Pair{"cos(x)", "-sin(x)"}, Pair{"x^3", "3*x^2"},
                          Pair{"exp(-x^2)", "-exp(-x^2)*2*x"}, Pair{"(1 - x)^2", "-2*(1 - x)"}}) {
        const Expression d = Expression::Parse(p.text, {"x"}).Derivative(0);
        Check(d.ToString() == p.derivative, std::string("d/dx ") + p.text + " reads " + p.derivative + ": got " + d.ToString());
    }
}

void CheckCalculator() {
    using cyxwiz::Utilities;
    auto value = [](const std::string& text, const std::string& angles = "radians") {
        const auto r = Utilities::Evaluate(text, {}, angles);
        Check(r.success, text + " evaluates: " + r.error_message);
        return r.result;
    };
    Near(value("-2^2"), -4.0, 1e-15, "-2^2 is -(2^2)");
    Near(value("exp(-1^2)"), std::exp(-1.0), 1e-15, "exp(-1^2)");
    Near(value("sin(30)", "degrees"), 0.5, 1e-15, "sin of 30 degrees");
    Near(value("atan(1)", "degrees"), 45.0, 1e-13, "atan in degrees");
    Near(value("factorial(5) + fact(3)"), 126.0, 0.0, "factorial");
    Near(value("7 % 3 + mod(-7, 3)"), 0.0, 0.0, "C fmod keeps the dividend's sign");
    Near(value("2^-1"), 0.5, 0.0, "2^-1");
    const auto x = Utilities::Evaluate("x^2 + y", {{"x", 3.0}, {"y", 1.0}});
    Check(x.success && x.result == 10.0, "variables");
    const auto zero = Utilities::Evaluate("1/0");
    Check(!zero.success && zero.error_message == "Division by zero", "1/0 refused: " + zero.error_message);
    const auto unknown = Utilities::Evaluate("2 * q");
    Check(!unknown.success && unknown.error_message.find("Unknown name 'q'") != std::string::npos, "unknown name");
}

double Bound(const json& j) {
    const std::string b = j.get<std::string>();
    if (b == "inf") return std::numeric_limits<double>::infinity();
    if (b == "-inf") return -std::numeric_limits<double>::infinity();
    if (b == "pi") return 3.14159265358979323846;
    return std::stod(b);
}

void CheckQuad(const json& cases) {
    for (const auto& c : cases) {
        const std::string text = c.at("text").get<std::string>();
        const std::string label = text + " from " + c.at("a").get<std::string>() + " to " + c.at("b").get<std::string>();
        const auto r = Calculus::Integrate(Expression::Parse(text, {"x"}), Bound(c.at("a")), Bound(c.at("b")));
        Check(r.success, label + ": " + r.error_message);
        const double exact = c.at("exact").get<double>(), miss = std::fabs(r.value - exact);
        std::cout << "  " << label << ": " << r.value << " (estimate " << r.error_estimate << ", " << r.evaluations
                  << " evaluations, " << r.intervals << " intervals" << (r.converged ? "" : ", not converged") << "; quad "
                  << c.at("quad_evaluations").get<int>() << ")\n";
        Check(miss <= (std::max)(r.error_estimate, 4e-16 * (1.0 + std::fabs(exact))),
              label + ": the estimate covers the error (" + std::to_string(miss) + " vs " + std::to_string(r.error_estimate) + ")");
        if (r.converged)
            Check(miss <= (std::max)(1.49e-8, 1.49e-8 * std::fabs(exact)), label + ": within quad's tolerance, off by " + std::to_string(miss));
        if (c.at("quad_evaluations").get<int>() == 21) {  // one interval: the same rule as QUADPACK
            Check(r.evaluations == 21, label + ": one interval");
            Near(r.value, c.at("quad").get<double>(), 1e-15, label + " = quad");
            Near(r.error_estimate, c.at("quad_error").get<double>(), 1e-6, label + " estimate = quad's");
        }
    }
    const auto odd = Calculus::Integrate(Expression::Parse("x", {"x"}), 0, 1, {cyxwiz::QuadratureRule::Simpson, 7});
    Check(!odd.success && odd.error_message.find("even number") != std::string::npos, "Simpson refuses an odd n");
    const auto open = Calculus::Integrate(Expression::Parse("exp(-x)", {"x"}), 0, std::numeric_limits<double>::infinity(),
                                          {cyxwiz::QuadratureRule::Trapezoid, 10});
    Check(!open.success && open.error_message.find("adaptive") != std::string::npos, "infinite bounds need the adaptive rule");
    const auto two = Calculus::Integrate(Expression::Parse("x*y", {"x", "y"}), 0, 1);
    Check(!two.success && two.error_message.find("one variable") != std::string::npos, "one variable");
    const auto pole = Calculus::Integrate(Expression::Parse("1/(x - 0.5)", {"x"}), 0, 1, {cyxwiz::QuadratureRule::Trapezoid, 2});
    Check(!pole.success && pole.error_message.find("not finite at x = 0.5") != std::string::npos, "a pole named: " + pole.error_message);
}

void CheckFixed(const json& c) {
    const Expression f = Expression::Parse(c.at("text").get<std::string>(), {"x"});
    const double a = Bound(c.at("a")), b = Bound(c.at("b"));
    const int n = c.at("n").get<int>();
    const std::pair<const char*, cyxwiz::QuadratureRule> rules[] = {{"trapezoid", cyxwiz::QuadratureRule::Trapezoid},
        {"midpoint", cyxwiz::QuadratureRule::Midpoint}, {"simpson", cyxwiz::QuadratureRule::Simpson}};
    for (const auto& [name, rule] : rules) {
        const auto r = Calculus::Integrate(f, a, b, {rule, n});
        Check(r.success && std::isnan(r.error_estimate), std::string(name) + " runs, no estimate");
        Near(r.value, c.at("rules").at(name).get<double>(), 1e-13, name);
    }
    for (const auto& [k, value] : c.at("gauss").items()) {
        const auto r = Calculus::Integrate(f, a, b, {cyxwiz::QuadratureRule::GaussLegendre, std::stoi(k)});
        Check(r.success && r.evaluations == std::stoi(k), "Gauss " + k + " points");
        Near(r.value, value.get<double>(), 1e-14, "Gauss-Legendre " + k);
    }
    for (const auto& [k, table] : c.at("legendre").items()) {
        std::vector<double> nodes, weights;
        Calculus::GaussLegendre(std::stoi(k), nodes, weights);
        const auto en = Doubles(table.at("nodes")), ew = Doubles(table.at("weights"));
        Check(nodes.size() == en.size(), "Legendre " + k + " size");
        for (size_t i = 0; i < en.size(); ++i) {
            Near(nodes[i], en[i], 1e-14, "Legendre " + k + " node " + std::to_string(i));
            Near(weights[i], ew[i], 1e-13, "Legendre " + k + " weight " + std::to_string(i));
        }
    }
}

void CheckDerivatives(const json& cases) {
    for (const auto& c : cases) {
        const std::string text = c.at("text").get<std::string>();
        const Expression f = Expression::Parse(text, {"x"});
        const double x = c.at("x").get<double>();
        const auto r = Calculus::Differentiate(f, x);
        Check(r.success && r.has_exact, text + " differentiates: " + r.error_message);
        Near(r.exact_first, c.at("first").get<double>(), 1e-14, text + " f'");
        Near(r.exact_second, c.at("second").get<double>(), 1e-13, text + " f''");
        Near(r.numerical_first, r.exact_first, 1e-9, text + " central f' at h = " + std::to_string(r.step));
        Near(r.numerical_second, r.exact_second, 1e-6, text + " f'' at h = " + std::to_string(r.second_step));
        const auto forward = Calculus::Differentiate(f, x, cyxwiz::DifferenceRule::Forward, 1e-4);
        const auto backward = Calculus::Differentiate(f, x, cyxwiz::DifferenceRule::Backward, 1e-4);
        Check(std::fabs(forward.numerical_first - r.exact_first) > 1e-12 &&
                  std::fabs(forward.numerical_first + backward.numerical_first - 2 * r.exact_first) <
                      std::fabs(forward.numerical_first - r.exact_first),
              text + ": forward and backward err on opposite sides");
        const auto steps = Doubles(c.at("steps"));
        const auto curve = Calculus::ErrorByStep(f, x, steps.front(), steps.back(), static_cast<int>(steps.size()));
        Check(curve.success && curve.steps.size() == steps.size(), text + " error curve");
        const double fx = std::fabs(f.Evaluate({x})) + 1.0;
        for (size_t i = 0; i < steps.size(); ++i) {
            Near(curve.steps[i], steps[i], 1e-12, text + " step " + std::to_string(i));
            const double noise = 8 * 2.2e-16 * fx / steps[i];
            const double ef = c.at("forward_error")[i].get<double>(), ec = c.at("central_error")[i].get<double>();
            Check(std::fabs(curve.forward_error[i] - ef) <= (std::max)(1e-3 * ef, noise), text + " forward error at h " + std::to_string(steps[i]));
            Check(std::fabs(curve.central_error[i] - ec) <= (std::max)(1e-3 * ec, noise), text + " central error at h " + std::to_string(steps[i]));
        }
    }
    const auto none = Calculus::Differentiate(Expression::Parse("max(x, 0)", {"x"}), 1.0);
    Check(none.success && !none.has_exact && std::fabs(none.numerical_first - 1.0) < 1e-9, "no exact derivative, numerical still");
}

void CheckCurvature(const json& c) {
    const Expression f = Expression::Parse(c.at("text").get<std::string>(), {"x", "y"});
    for (const auto& p : c.at("at")) {
        const auto point = Doubles(p.at("point"));
        const auto r = Calculus::CurvatureAt(f, point);
        Check(r.success, "curvature at a point: " + r.error_message);
        const auto e = Doubles(p.at("eigenvalues"));
        for (size_t i = 0; i < e.size(); ++i) Near(r.eigenvalues[i], e[i], 1e-10, "eigenvalue " + std::to_string(i));
    }
    const auto& range = c.at("range");
    const auto r = Calculus::CurvatureOverRange(f, range.at("x")[0].get<double>(), range.at("x")[1].get<double>(),
                                                range.at("y")[0].get<double>(), range.at("y")[1].get<double>(),
                                                range.at("per_side").get<int>());
    Check(r.success && r.points == 1600, "range check: " + r.error_message);
    Check(r.negative_points == range.at("negative").get<int>(),
          "negative-curvature points: " + std::to_string(r.negative_points));
}

std::vector<cyxwiz::LinearConstraint> Rows(const json& rows) {
    std::vector<cyxwiz::LinearConstraint> out;
    for (const auto& r : rows) {
        cyxwiz::LinearConstraint row;
        row.name = r.at("name").get<std::string>();
        row.coefficients = Doubles(r.at("coefficients"));
        const std::string sense = r.at("sense").get<std::string>();
        row.sense = sense == "<=" ? cyxwiz::ConstraintSense::LessEqual
                  : sense == ">=" ? cyxwiz::ConstraintSense::GreaterEqual : cyxwiz::ConstraintSense::Equal;
        row.rhs = r.at("rhs").get<double>();
        out.push_back(row);
    }
    return out;
}

std::vector<cyxwiz::VariableBound> Bounds(const json& bounds) {
    std::vector<cyxwiz::VariableBound> out;
    if (bounds.is_null()) return out;
    const double inf = std::numeric_limits<double>::infinity();
    for (const auto& b : bounds)
        out.push_back({b[0].is_null() ? -inf : b[0].get<double>(), b[1].is_null() ? inf : b[1].get<double>()});
    return out;
}

void CheckProgramAnswer(const cyxwiz::ProgramResult& r, const json& c, double price_tolerance, const std::string& label) {
    Check(r.success, label + " solves: " + r.error_message);
    const auto x = Doubles(c.at("x"));
    Check(r.x.size() == x.size(), label + ": one value per variable");
    for (size_t i = 0; i < x.size(); ++i) Near(r.x[i], x[i], 1e-7, label + " x" + std::to_string(i + 1));
    Near(r.objective, c.at("value").get<double>(), 1e-9, label + " objective");
    const auto prices = Doubles(c.at("shadow_prices"));
    Check(r.shadow_prices.size() == prices.size(), label + ": one shadow price per row");
    for (size_t i = 0; i < prices.size(); ++i)
        Near(r.shadow_prices[i], prices[i], price_tolerance, label + " shadow price of row " + std::to_string(i + 1));
}

void CheckLinearPrograms(const json& cases) {
    for (const auto& c : cases) {
        const std::string label = "LP " + c.at("label").get<std::string>();
        cyxwiz::LinearProgram lp;
        lp.maximize = c.at("maximize").get<bool>();
        lp.objective = Doubles(c.at("objective"));
        lp.constraints = Rows(c.at("rows"));
        lp.bounds = Bounds(c.at("bounds"));
        const auto r = cyxwiz::Optimization::SolveLinearProgram(lp);
        const std::string status = c.at("status").get<std::string>();
        if (status == "infeasible") {
            Check(!r.success && r.status == cyxwiz::ProgramStatus::Infeasible && r.error_message.find("No point") != std::string::npos,
                  label + ": infeasible, in words: " + r.error_message);
            continue;
        }
        if (status == "unbounded") {
            Check(!r.success && r.status == cyxwiz::ProgramStatus::Unbounded && r.error_message.find("grows without limit") != std::string::npos,
                  label + ": unbounded, in words: " + r.error_message);
            continue;
        }
        CheckProgramAnswer(r, c, 1e-6, label);
        const auto slack = Doubles(c.at("slack"));
        for (size_t i = 0; i < slack.size(); ++i) {
            Near(r.slack[i], slack[i], 1e-7, label + " room in row " + std::to_string(i + 1));
            Check(r.binding[i] == (slack[i] < 1e-9), label + " binding row " + std::to_string(i + 1));
        }
    }
    cyxwiz::LinearProgram bad;
    bad.objective = {1, 2};
    bad.constraints = {{"Wide", {1, 2, 3}, cyxwiz::ConstraintSense::LessEqual, 1}};
    const auto r = cyxwiz::Optimization::SolveLinearProgram(bad);
    Check(!r.success && r.error_message == "Wide has 3 coefficients; the problem has 2 variables", "row size named: " + r.error_message);
}

void CheckQuadraticPrograms(const json& cases) {
    for (const auto& c : cases) {
        const std::string label = "QP " + c.at("label").get<std::string>();
        cyxwiz::QuadraticProgram qp;
        for (const auto& row : c.at("Q")) qp.Q.push_back(Doubles(row));
        qp.c = Doubles(c.at("c"));
        qp.constraints = Rows(c.at("rows"));
        qp.bounds = Bounds(c.at("bounds"));
        // scipy's prices are finite differences of SLSQP answers.
        CheckProgramAnswer(cyxwiz::Optimization::SolveQuadraticProgram(qp), c, 2e-4, label);
    }
    cyxwiz::QuadraticProgram saddle;
    saddle.Q = {{1, 0}, {0, -2}};
    saddle.c = {0, 0};
    const auto r = cyxwiz::Optimization::SolveQuadraticProgram(saddle);
    Check(!r.success && r.status == cyxwiz::ProgramStatus::NotConvex && r.error_message.find("smallest eigenvalue is -2") != std::string::npos,
          "a saddle is refused: " + r.error_message);
    cyxwiz::QuadraticProgram lopsided;
    lopsided.Q = {{1, 0.5}, {0, 1}};
    lopsided.c = {0, 0};
    const auto l = cyxwiz::Optimization::SolveQuadraticProgram(lopsided);
    Check(!l.success && l.error_message.find("symmetric") != std::string::npos, "Q must be symmetric: " + l.error_message);
}

void CheckPaths(const json& cases) {
    const std::map<std::string, cyxwiz::OptimizerType> types = {
        {"SGD", cyxwiz::OptimizerType::SGD}, {"Adam", cyxwiz::OptimizerType::Adam}, {"AdamW", cyxwiz::OptimizerType::AdamW},
        {"NAdam", cyxwiz::OptimizerType::NAdam}, {"RMSprop", cyxwiz::OptimizerType::RMSprop}, {"Adagrad", cyxwiz::OptimizerType::AdaGrad}};
    for (const auto& c : cases) {
        const std::string label = "path " + c.at("label").get<std::string>();
        cyxwiz::OptimizerSetting setting;
        setting.type = types.at(c.at("optimizer").get<std::string>());
        setting.learning_rate = c.at("settings").at("lr").get<double>();
        setting.momentum = c.at("settings").value("momentum", 0.0);
        const int steps = c.at("steps").get<int>();
        const auto path = cyxwiz::Optimization::FollowOptimizer(Expression::Parse(c.at("text").get<std::string>(), {"x", "y"}),
                                                                Doubles(c.at("start")), setting, steps, 0.0);
        Check(path.success, label + ": " + path.error_message);
        if (!c.at("diverged_at").is_null()) {
            Check(path.diverged && path.diverged_at == c.at("diverged_at").get<int>(),
                  label + " diverges at torch's step " + std::to_string(c.at("diverged_at").get<int>()) + ": " +
                      (path.diverged ? std::to_string(path.diverged_at) : std::string("did not")));
            continue;
        }
        Check(!path.diverged && static_cast<int>(path.points.size()) == steps + 1, label + ": every step taken");
        for (const auto& e : c.at("every")) {
            const int step = e.at("step").get<int>();
            const auto expected = Doubles(e.at("point"));
            // A chaotic stretch is held to the path's own sensitivity (torch
            // against torch from a start one ulp away), the rest to 2e-4.
            const double sensitivity = 4.0 * e.at("sensitivity").get<double>();
            for (size_t i = 0; i < 2; ++i)
                Check(std::fabs(path.points[step][i] - expected[i]) <= (std::max)(2e-4 * (1.0 + std::fabs(expected[i])), sensitivity),
                      label + " step " + std::to_string(step) + " coordinate " + std::to_string(i) + ": " +
                          std::to_string(path.points[step][i]) + " vs torch " + std::to_string(expected[i]));
        }
    }
    int asked = 0;
    const auto stopped = cyxwiz::Optimization::FollowOptimizer(Expression::Parse("x^2", {"x"}), {3.0},
        {cyxwiz::OptimizerType::SGD, 0.1, 0.0}, 1000, 0.0, [&asked] { return ++asked > 5; });
    Check(stopped.success && stopped.stopped && stopped.points.size() == 6, "Stop ends the path");
    const auto flat = cyxwiz::Optimization::FollowOptimizer(Expression::Parse("x^2", {"x"}), {3.0},
        {cyxwiz::OptimizerType::SGD, 0.1, 0.0}, 1000, 1e-6);
    Check(flat.converged && flat.points.size() < 100, "a small gradient ends the path");
}

}  // namespace

int main() {
    if (!cyxwiz::test::SelectTestDeviceFromEnvironment()) return 1;
    std::ifstream in(CYXWIZ_OPTIMIZATION_FIXTURE);
    Check(in.good(), std::string("cannot open ") + CYXWIZ_OPTIMIZATION_FIXTURE);
    const json fixture = json::parse(in);
    CheckExpressions(fixture.at("expression"));
    CheckRefusals();
    CheckReadable();
    CheckCalculator();
    CheckQuad(fixture.at("quad"));
    CheckFixed(fixture.at("fixed"));
    CheckDerivatives(fixture.at("derivative"));
    CheckCurvature(fixture.at("curvature"));
    CheckLinearPrograms(fixture.at("lp"));
    CheckQuadraticPrograms(fixture.at("qp"));
    CheckPaths(fixture.at("paths"));
    std::cout << "Optimization truth test passed: " << checks << " checks\n";
    return 0;
}
