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

#include "test_device_selection.h"

#include <cyxwiz/calculus.h>
#include <cyxwiz/expression.h>
#include <cyxwiz/utilities.h>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <cstdlib>
#include <fstream>
#include <iostream>
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
    std::cout << "Optimization truth test passed: " << checks << " checks\n";
    return 0;
}
