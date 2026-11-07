#pragma once

// A typed formula such as "(1 - x)^2 + 100*(y - x^2)^2": parsed once into a
// tree, evaluated at one point on the host or at many points on the device,
// and differentiated exactly. The Calculator, the optimization window and the
// calculus functions all read formulas through it.
//
// Grammar, loosest first: + -, then * / %, then unary - +, then ^ (right
// associative), so -x^2 is -(x^2) and 2^-1 is 0.5. Numbers as 1.5, .5, 2e-3.
// Constants (any case): pi, e, phi. Functions (any case): sin cos tan asin
// acos atan sinh cosh tanh sqrt cbrt exp log ln log10 log2 abs floor ceil
// round sign factorial fact, and with two arguments pow mod fmod min max
// atan2.

#include "cyxwiz/api_export.h"

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace cyxwiz {

struct ExpressionNode;

class CYXWIZ_API Expression {
public:
    enum class AngleUnit { Radians, Degrees };

    Expression();

    // Parses `text`. The names in `variables` are its inputs, in that order;
    // any other name is refused ("Unknown name: z"). With Degrees, trig
    // functions take degrees and the inverse ones give degrees.
    static Expression Parse(const std::string& text,
                            const std::vector<std::string>& variables = {},
                            AngleUnit angles = AngleUnit::Radians);

    bool Valid() const { return root_ != nullptr; }
    // Why it did not parse (or has no derivative), with the column for a
    // parse error ("Expected ')' at 12").
    const std::string& Error() const { return error_; }
    const std::string& Text() const { return text_; }
    const std::vector<std::string>& Variables() const { return variables_; }
    // True when the formula reads variable `index`.
    bool Uses(size_t index) const;

    // The value at one point, `values` in Variables() order. A value outside
    // a function's domain is NaN (log of a negative number) or infinite (a
    // division by zero); `problem`, when given, receives the first such
    // event in words ("Division by zero").
    double Evaluate(const std::vector<double>& values, std::string* problem = nullptr) const;

    // The values at many points on the device (float64 where it has it):
    // columns[v][i] is variable v at point i; every column has the same
    // length. Returns one value per point.
    std::vector<double> EvaluateMany(const std::vector<std::vector<double>>& columns) const;

    // The exact derivative with respect to variable `index`, simplified.
    // floor, ceil, round and sign count as flat (derivative 0 where they are
    // defined); factorial, mod, min and max have none: the result is then
    // invalid with the reason in Error().
    Expression Derivative(size_t index) const;

    // The tree written back with the fewest parentheses ("cos(x)", "-sin(x)").
    std::string ToString() const;

private:
    std::shared_ptr<const ExpressionNode> root_;
    std::vector<std::string> variables_;
    std::string text_;
    std::string error_;
    AngleUnit angles_ = AngleUnit::Radians;
};

} // namespace cyxwiz
