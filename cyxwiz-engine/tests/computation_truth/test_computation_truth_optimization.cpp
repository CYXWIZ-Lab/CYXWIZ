// The optimization window's backend against sympy / scipy / torch (TOFIX129
// D10): fixtures/optimization_truth.json from
// reference/generate_optimization_fixtures.py.
//
// expression  cyxwiz::Expression: values and exact derivatives at points
//             (host, and the device for many points at once), parse errors,
//             printing back, and the Calculator that reads through it.

#include "test_device_selection.h"

#include <cyxwiz/expression.h>
#include <cyxwiz/utilities.h>

#include <nlohmann/json.hpp>

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

using json = nlohmann::json;
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
    const double device_tol = cyxwiz::test::DeviceTolerance(1e-12, 2e-5);
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
            Near(many[p], c.at("values")[p].get<double>(), device_tol, text + " on the device at point " + std::to_string(p));
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
    std::cout << "Optimization truth test passed: " << checks << " checks\n";
    return 0;
}
