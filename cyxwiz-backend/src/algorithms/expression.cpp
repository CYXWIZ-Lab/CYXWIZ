#include "cyxwiz/expression.h"

#include "arrayfire_host_values.h"

#include <arrayfire.h>

#include <algorithm>
#include <cctype>
#include <charconv>
#include <cmath>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <system_error>

namespace cyxwiz {

namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr double kE = 2.71828182845904523536;
constexpr double kPhi = 1.61803398874989484820;

enum class Op { Number, Variable, Negate, Add, Subtract, Multiply, Divide, Power, Function };

enum class Fn {
    Sin, Cos, Tan, Asin, Acos, Atan, Sinh, Cosh, Tanh, Sqrt, Cbrt, Exp, Log, Log10, Log2,
    Abs, Floor, Ceil, Round, Sign, Factorial, Mod, Min, Max, Atan2
};

struct FnInfo {
    const char* name;
    Fn fn;
    int arguments;
};

constexpr FnInfo kFunctions[] = {
    {"sin", Fn::Sin, 1},     {"cos", Fn::Cos, 1},         {"tan", Fn::Tan, 1},     {"asin", Fn::Asin, 1},
    {"acos", Fn::Acos, 1},   {"atan", Fn::Atan, 1},       {"sinh", Fn::Sinh, 1},   {"cosh", Fn::Cosh, 1},
    {"tanh", Fn::Tanh, 1},   {"sqrt", Fn::Sqrt, 1},       {"cbrt", Fn::Cbrt, 1},   {"exp", Fn::Exp, 1},
    {"log", Fn::Log, 1},     {"ln", Fn::Log, 1},          {"log10", Fn::Log10, 1}, {"log2", Fn::Log2, 1},
    {"abs", Fn::Abs, 1},     {"floor", Fn::Floor, 1},     {"ceil", Fn::Ceil, 1},   {"round", Fn::Round, 1},
    {"sign", Fn::Sign, 1},   {"factorial", Fn::Factorial, 1}, {"fact", Fn::Factorial, 1},
    {"mod", Fn::Mod, 2},     {"fmod", Fn::Mod, 2},        {"min", Fn::Min, 2},     {"max", Fn::Max, 2},
    {"atan2", Fn::Atan2, 2},
};

const char* FnName(Fn fn) {
    for (const auto& f : kFunctions)
        if (f.fn == fn) return f.name;
    return "?";
}

}  // namespace

struct ExpressionNode {
    Op op = Op::Number;
    Fn fn = Fn::Sin;
    double value = 0.0;
    size_t variable = 0;
    std::shared_ptr<const ExpressionNode> a, b;
};

namespace {

using NodePtr = std::shared_ptr<const ExpressionNode>;

bool IsNumber(const NodePtr& n, double v) { return n->op == Op::Number && n->value == v; }
bool IsNumber(const NodePtr& n) { return n->op == Op::Number; }

NodePtr Number(double v) {
    auto n = std::make_shared<ExpressionNode>();
    n->op = Op::Number;
    n->value = v;
    return n;
}

NodePtr Variable(size_t index) {
    auto n = std::make_shared<ExpressionNode>();
    n->op = Op::Variable;
    n->variable = index;
    return n;
}

NodePtr Raw(Op op, NodePtr a, NodePtr b = nullptr, Fn fn = Fn::Sin) {
    auto n = std::make_shared<ExpressionNode>();
    n->op = op;
    n->fn = fn;
    n->a = std::move(a);
    n->b = std::move(b);
    return n;
}

double Factorial(double x) {
    const double n = std::trunc(x);
    if (std::isnan(n) || n < 0) return std::numeric_limits<double>::quiet_NaN();
    double result = 1.0;
    for (double i = 2; i <= n && std::isfinite(result); ++i) result *= i;
    return result;
}

double Sign(double x) { return x > 0 ? 1.0 : (x < 0 ? -1.0 : 0.0); }

double Apply1(Fn fn, double x) {
    switch (fn) {
        case Fn::Sin: return std::sin(x);
        case Fn::Cos: return std::cos(x);
        case Fn::Tan: return std::tan(x);
        case Fn::Asin: return std::asin(x);
        case Fn::Acos: return std::acos(x);
        case Fn::Atan: return std::atan(x);
        case Fn::Sinh: return std::sinh(x);
        case Fn::Cosh: return std::cosh(x);
        case Fn::Tanh: return std::tanh(x);
        case Fn::Sqrt: return std::sqrt(x);
        case Fn::Cbrt: return std::cbrt(x);
        case Fn::Exp: return std::exp(x);
        case Fn::Log: return std::log(x);
        case Fn::Log10: return std::log10(x);
        case Fn::Log2: return std::log2(x);
        case Fn::Abs: return std::abs(x);
        case Fn::Floor: return std::floor(x);
        case Fn::Ceil: return std::ceil(x);
        case Fn::Round: return std::round(x);
        case Fn::Sign: return Sign(x);
        case Fn::Factorial: return Factorial(x);
        default: return std::numeric_limits<double>::quiet_NaN();
    }
}

double Apply2(Fn fn, double x, double y) {
    switch (fn) {
        case Fn::Mod: return std::fmod(x, y);
        case Fn::Min: return (std::min)(x, y);
        case Fn::Max: return (std::max)(x, y);
        case Fn::Atan2: return std::atan2(x, y);
        default: return std::numeric_limits<double>::quiet_NaN();
    }
}

// The constructors simplify as they build: constants fold, and 0 and 1 drop
// out, so derivatives stay readable.
NodePtr Neg(NodePtr a) {
    if (IsNumber(a)) return Number(-a->value);
    if (a->op == Op::Negate) return a->a;
    return Raw(Op::Negate, std::move(a));
}

NodePtr Add(NodePtr a, NodePtr b) {
    if (IsNumber(a) && IsNumber(b)) return Number(a->value + b->value);
    if (IsNumber(a, 0.0)) return b;
    if (IsNumber(b, 0.0)) return a;
    if (IsNumber(b) && b->value < 0) return Raw(Op::Subtract, std::move(a), Number(-b->value));
    if (b->op == Op::Negate) return Raw(Op::Subtract, std::move(a), b->a);
    return Raw(Op::Add, std::move(a), std::move(b));
}

NodePtr Sub(NodePtr a, NodePtr b) {
    if (IsNumber(a) && IsNumber(b)) return Number(a->value - b->value);
    if (IsNumber(b, 0.0)) return a;
    if (IsNumber(a, 0.0)) return Neg(std::move(b));
    if (b->op == Op::Negate) return Add(std::move(a), b->a);
    return Raw(Op::Subtract, std::move(a), std::move(b));
}

NodePtr Mul(NodePtr a, NodePtr b) {
    if (IsNumber(a) && IsNumber(b)) return Number(a->value * b->value);
    if (IsNumber(a, 0.0) || IsNumber(b, 0.0)) return Number(0.0);
    if (IsNumber(a, 1.0)) return b;
    if (IsNumber(b, 1.0)) return a;
    if (IsNumber(a, -1.0)) return Neg(std::move(b));
    if (IsNumber(b, -1.0)) return Neg(std::move(a));
    if (a->op == Op::Negate) return Neg(Mul(a->a, std::move(b)));
    if (b->op == Op::Negate) return Neg(Mul(std::move(a), b->a));
    if (IsNumber(b)) return Raw(Op::Multiply, std::move(b), std::move(a));  // 2*x, not x*2
    return Raw(Op::Multiply, std::move(a), std::move(b));
}

NodePtr Div(NodePtr a, NodePtr b) {
    if (IsNumber(a) && IsNumber(b) && b->value != 0.0) return Number(a->value / b->value);
    if (IsNumber(a, 0.0) && !IsNumber(b, 0.0)) return Number(0.0);
    if (IsNumber(b, 1.0)) return a;
    if (a->op == Op::Negate) return Neg(Div(a->a, std::move(b)));
    return Raw(Op::Divide, std::move(a), std::move(b));
}

NodePtr Pow(NodePtr a, NodePtr b) {
    if (IsNumber(a) && IsNumber(b)) return Number(std::pow(a->value, b->value));
    if (IsNumber(b, 1.0)) return a;
    if (IsNumber(b, 0.0)) return Number(1.0);
    return Raw(Op::Power, std::move(a), std::move(b));
}

NodePtr Call(Fn fn, NodePtr a, NodePtr b = nullptr) {
    if (IsNumber(a) && (!b || IsNumber(b)) && fn != Fn::Mod && fn != Fn::Factorial)
        return Number(b ? Apply2(fn, a->value, b->value) : Apply1(fn, a->value));
    return Raw(Op::Function, std::move(a), std::move(b), fn);
}

bool UsesVariable(const NodePtr& n, size_t index) {
    if (!n) return false;
    if (n->op == Op::Variable) return n->variable == index;
    return UsesVariable(n->a, index) || UsesVariable(n->b, index);
}

// ---------------------------------------------------------------------------
// Parsing

class Parser {
public:
    Parser(const std::string& text, const std::vector<std::string>& variables, Expression::AngleUnit angles)
        : text_(text), variables_(variables), degrees_(angles == Expression::AngleUnit::Degrees) {}

    NodePtr Parse(std::string& error) {
        NodePtr root;
        try {
            Skip();
            if (pos_ >= text_.size()) Fail("Empty expression");
            root = ParseSum();
            Skip();
            if (pos_ < text_.size()) Fail(std::string("Unexpected '") + text_[pos_] + "'");
        } catch (const std::runtime_error& e) {
            error = e.what();
            return nullptr;
        }
        return root;
    }

private:
    [[noreturn]] void Fail(const std::string& what) const {
        throw std::runtime_error(what + " at " + std::to_string(pos_ + 1));
    }

    void Skip() {
        while (pos_ < text_.size() && std::isspace(static_cast<unsigned char>(text_[pos_]))) ++pos_;
    }

    bool Take(char c) {
        Skip();
        if (pos_ < text_.size() && text_[pos_] == c) {
            ++pos_;
            return true;
        }
        return false;
    }

    NodePtr ParseSum() {
        NodePtr left = ParseProduct();
        while (true) {
            if (Take('+')) left = Add(left, ParseProduct());
            else if (Take('-')) left = Sub(left, ParseProduct());
            else return left;
        }
    }

    NodePtr ParseProduct() {
        NodePtr left = ParseUnary();
        while (true) {
            if (Take('*')) left = Mul(left, ParseUnary());
            else if (Take('/')) left = Div(left, ParseUnary());
            else if (Take('%')) left = Raw(Op::Function, left, ParseUnary(), Fn::Mod);
            else return left;
        }
    }

    NodePtr ParseUnary() {
        if (Take('-')) return Neg(ParseUnary());
        if (Take('+')) return ParseUnary();
        return ParsePower();
    }

    NodePtr ParsePower() {
        NodePtr base = ParsePrimary();
        if (Take('^')) return Pow(base, ParseUnary());  // right associative; 2^-1
        return base;
    }

    NodePtr ParsePrimary() {
        Skip();
        if (pos_ >= text_.size()) Fail("Expected a value");
        const char c = text_[pos_];
        if (Take('(')) {
            NodePtr inner = ParseSum();
            if (!Take(')')) Fail("Expected ')'");
            return inner;
        }
        if (std::isdigit(static_cast<unsigned char>(c)) || c == '.') return ParseNumber();
        if (std::isalpha(static_cast<unsigned char>(c)) || c == '_') return ParseName();
        Fail(std::string("Unexpected '") + c + "'");
    }

    NodePtr ParseNumber() {
        const size_t start = pos_;
        bool dot = false;
        while (pos_ < text_.size()) {
            const char c = text_[pos_];
            if (std::isdigit(static_cast<unsigned char>(c))) {
                ++pos_;
            } else if (c == '.' && !dot) {
                dot = true;
                ++pos_;
            } else {
                break;
            }
        }
        // An exponent only when digits follow it ("2e-3"; "2e" alone is refused).
        if (pos_ < text_.size() && (text_[pos_] == 'e' || text_[pos_] == 'E')) {
            size_t p = pos_ + 1;
            if (p < text_.size() && (text_[p] == '+' || text_[p] == '-')) ++p;
            if (p < text_.size() && std::isdigit(static_cast<unsigned char>(text_[p]))) {
                pos_ = p;
                while (pos_ < text_.size() && std::isdigit(static_cast<unsigned char>(text_[pos_]))) ++pos_;
            }
        }
        double value = 0.0;
        const char* first = text_.data() + start;
        const char* last = text_.data() + pos_;
        const auto [end, ec] = std::from_chars(first, last, value);
        if (ec == std::errc::result_out_of_range) {
            pos_ = start;
            Fail("Number out of range");
        }
        if (ec != std::errc() || end != last) {
            pos_ = start;
            Fail("Expected a number");
        }
        return Number(value);
    }

    NodePtr ParseName() {
        const size_t start = pos_;
        while (pos_ < text_.size() &&
               (std::isalnum(static_cast<unsigned char>(text_[pos_])) || text_[pos_] == '_'))
            ++pos_;
        const std::string name = text_.substr(start, pos_ - start);
        std::string lower = name;
        std::transform(lower.begin(), lower.end(), lower.begin(),
                       [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });

        Skip();
        const bool call = pos_ < text_.size() && text_[pos_] == '(';
        if (!call) {
            for (size_t i = 0; i < variables_.size(); ++i)
                if (variables_[i] == name) return Variable(i);
            if (lower == "pi") return Number(kPi);
            if (lower == "e") return Number(kE);
            if (lower == "phi") return Number(kPhi);
            pos_ = start;
            Fail("Unknown name '" + name + "'");
        }
        const FnInfo* info = nullptr;
        for (const auto& f : kFunctions)
            if (lower == f.name) info = &f;
        if (!info && lower != "pow") {
            pos_ = start;
            Fail("Unknown function '" + name + "'");
        }
        Take('(');
        NodePtr first = ParseSum();
        NodePtr second;
        if (Take(',')) second = ParseSum();
        if (!Take(')')) Fail("Expected ')'");
        if (lower == "pow") {
            if (!second) Fail("pow takes two values");
            return Pow(first, second);
        }
        if ((info->arguments == 2) != (second != nullptr)) {
            pos_ = start;
            Fail(std::string(info->name) + " takes " + (info->arguments == 2 ? "two values" : "one value"));
        }
        return Angled(info->fn, first, second);
    }

    // In degrees, trig functions take degrees and the inverse ones give them.
    NodePtr Angled(Fn fn, NodePtr a, NodePtr b) {
        if (!degrees_) return Call(fn, a, b);
        const NodePtr to_radians = Number(kPi / 180.0), to_degrees = Number(180.0 / kPi);
        switch (fn) {
            case Fn::Sin: case Fn::Cos: case Fn::Tan:
                return Call(fn, Mul(a, to_radians));
            case Fn::Asin: case Fn::Acos: case Fn::Atan:
                return Mul(Call(fn, a), to_degrees);
            case Fn::Atan2:
                return Mul(Call(fn, a, b), to_degrees);
            default:
                return Call(fn, a, b);
        }
    }

    const std::string& text_;
    const std::vector<std::string>& variables_;
    bool degrees_;
    size_t pos_ = 0;
};

// ---------------------------------------------------------------------------
// Evaluation

double EvaluateHost(const NodePtr& n, const std::vector<double>& values, std::string* problem) {
    auto note = [problem](const char* what) {
        if (problem && problem->empty()) *problem = what;
    };
    switch (n->op) {
        case Op::Number: return n->value;
        case Op::Variable: return values[n->variable];
        case Op::Negate: return -EvaluateHost(n->a, values, problem);
        case Op::Add: return EvaluateHost(n->a, values, problem) + EvaluateHost(n->b, values, problem);
        case Op::Subtract: return EvaluateHost(n->a, values, problem) - EvaluateHost(n->b, values, problem);
        case Op::Multiply: return EvaluateHost(n->a, values, problem) * EvaluateHost(n->b, values, problem);
        case Op::Divide: {
            const double num = EvaluateHost(n->a, values, problem), den = EvaluateHost(n->b, values, problem);
            if (den == 0.0) note("Division by zero");
            return num / den;
        }
        case Op::Power: return std::pow(EvaluateHost(n->a, values, problem), EvaluateHost(n->b, values, problem));
        case Op::Function: {
            const double x = EvaluateHost(n->a, values, problem);
            if (!n->b) {
                if (n->fn == Fn::Factorial && std::trunc(x) < 0) note("Factorial undefined for negative numbers");
                return Apply1(n->fn, x);
            }
            const double y = EvaluateHost(n->b, values, problem);
            if (n->fn == Fn::Mod && y == 0.0) note("Modulo by zero");
            return Apply2(n->fn, x, y);
        }
    }
    return std::numeric_limits<double>::quiet_NaN();
}

af::array EvaluateDevice(const NodePtr& n, const std::vector<af::array>& columns, dim_t count, af::dtype type) {
    auto eval = [&](const NodePtr& child) { return EvaluateDevice(child, columns, count, type); };
    switch (n->op) {
        case Op::Number: return af::constant(n->value, count, type);
        case Op::Variable: return columns[n->variable];
        case Op::Negate: return -eval(n->a);
        case Op::Add: return eval(n->a) + eval(n->b);
        case Op::Subtract: return eval(n->a) - eval(n->b);
        case Op::Multiply: return eval(n->a) * eval(n->b);
        case Op::Divide: return eval(n->a) / eval(n->b);
        case Op::Power: {
            // An integer power of a negative base is defined; af::pow gives NaN for it.
            if (IsNumber(n->b) && n->b->value == std::round(n->b->value) && std::abs(n->b->value) <= 64) {
                const af::array base = eval(n->a);
                const int k = static_cast<int>(std::abs(n->b->value));
                af::array result = af::constant(1.0, count, type), square = base;
                for (int bits = k; bits > 0; bits >>= 1) {
                    if (bits & 1) result = result * square;
                    if (bits > 1) square = square * square;
                }
                return n->b->value < 0 ? 1.0 / result : result;
            }
            return af::pow(eval(n->a), eval(n->b));
        }
        case Op::Function: break;
    }
    const af::array x = eval(n->a);
    switch (n->fn) {
        case Fn::Sin: return af::sin(x);
        case Fn::Cos: return af::cos(x);
        case Fn::Tan: return af::tan(x);
        case Fn::Asin: return af::asin(x);
        case Fn::Acos: return af::acos(x);
        case Fn::Atan: return af::atan(x);
        case Fn::Sinh: return af::sinh(x);
        case Fn::Cosh: return af::cosh(x);
        case Fn::Tanh: return af::tanh(x);
        case Fn::Sqrt: return af::sqrt(x);
        case Fn::Cbrt: return af::cbrt(x);
        case Fn::Exp: return af::exp(x);
        case Fn::Log: return af::log(x);
        case Fn::Log10: return af::log10(x);
        case Fn::Log2: return af::log2(x);
        case Fn::Abs: return af::abs(x);
        case Fn::Floor: return af::floor(x);
        case Fn::Ceil: return af::ceil(x);
        case Fn::Round: return af::round(x);
        case Fn::Sign: return (x > 0).as(type) - (x < 0).as(type);
        case Fn::Factorial: {
            const af::array n_ = af::trunc(x);
            return af::select(n_ < 0, af::constant(std::numeric_limits<double>::quiet_NaN(), count, type),
                              af::tgamma(n_ + 1.0));
        }
        case Fn::Mod: {
            const af::array y = eval(n->b);
            return x - y * af::trunc(x / y);  // C fmod: the sign of x
        }
        case Fn::Min: return (af::min)(x, eval(n->b));
        case Fn::Max: return (af::max)(x, eval(n->b));
        case Fn::Atan2: return af::atan2(x, eval(n->b));
    }
    return af::constant(std::numeric_limits<double>::quiet_NaN(), count, type);
}

// ---------------------------------------------------------------------------
// Derivatives

NodePtr Derive(const NodePtr& n, size_t index, std::string& error) {
    if (!error.empty()) return nullptr;
    if (!UsesVariable(n, index)) return Number(0.0);
    auto d = [&](const NodePtr& child) { return Derive(child, index, error); };
    switch (n->op) {
        case Op::Number: return Number(0.0);
        case Op::Variable: return Number(n->variable == index ? 1.0 : 0.0);
        case Op::Negate: {
            NodePtr du = d(n->a);
            return du ? Neg(du) : nullptr;
        }
        case Op::Add:
        case Op::Subtract: {
            NodePtr du = d(n->a), dv = d(n->b);
            if (!du || !dv) return nullptr;
            return n->op == Op::Add ? Add(du, dv) : Sub(du, dv);
        }
        case Op::Multiply: {
            NodePtr du = d(n->a), dv = d(n->b);
            if (!du || !dv) return nullptr;
            return Add(Mul(du, n->b), Mul(n->a, dv));
        }
        case Op::Divide: {
            NodePtr du = d(n->a), dv = d(n->b);
            if (!du || !dv) return nullptr;
            if (IsNumber(dv, 0.0)) return Div(du, n->b);
            return Div(Sub(Mul(du, n->b), Mul(n->a, dv)), Pow(n->b, Number(2.0)));
        }
        case Op::Power: {
            const NodePtr& u = n->a;
            const NodePtr& v = n->b;
            NodePtr du = d(u);
            if (!du) return nullptr;
            if (!UsesVariable(v, index))  // v u^(v-1) u'
                return Mul(Mul(v, Pow(u, Sub(v, Number(1.0)))), du);
            NodePtr dv = d(v);
            if (!dv) return nullptr;  // u^v (v' ln u + v u' / u)
            return Mul(n, Add(Mul(dv, Call(Fn::Log, u)), Div(Mul(v, du), u)));
        }
        case Op::Function: break;
    }
    const NodePtr& u = n->a;
    if (n->b) {
        if (n->fn == Fn::Atan2) {  // atan2(y, x)' = (x y' - y x') / (x^2 + y^2)
            NodePtr dy = d(n->a), dx = d(n->b);
            if (!dy || !dx) return nullptr;
            return Div(Sub(Mul(n->b, dy), Mul(n->a, dx)),
                       Add(Pow(n->b, Number(2.0)), Pow(n->a, Number(2.0))));
        }
        error = std::string(FnName(n->fn)) + " has no derivative formula";
        return nullptr;
    }
    if (n->fn == Fn::Factorial) {
        error = "factorial has no derivative formula";
        return nullptr;
    }
    if (n->fn == Fn::Floor || n->fn == Fn::Ceil || n->fn == Fn::Round || n->fn == Fn::Sign) return Number(0.0);
    NodePtr du = d(u);
    if (!du) return nullptr;
    const NodePtr two = Number(2.0), one = Number(1.0);
    NodePtr outer;
    switch (n->fn) {
        case Fn::Sin: outer = Call(Fn::Cos, u); break;
        case Fn::Cos: outer = Neg(Call(Fn::Sin, u)); break;
        case Fn::Tan: outer = Div(one, Pow(Call(Fn::Cos, u), two)); break;
        case Fn::Asin: outer = Div(one, Call(Fn::Sqrt, Sub(one, Pow(u, two)))); break;
        case Fn::Acos: outer = Neg(Div(one, Call(Fn::Sqrt, Sub(one, Pow(u, two))))); break;
        case Fn::Atan: outer = Div(one, Add(one, Pow(u, two))); break;
        case Fn::Sinh: outer = Call(Fn::Cosh, u); break;
        case Fn::Cosh: outer = Call(Fn::Sinh, u); break;
        case Fn::Tanh: outer = Div(one, Pow(Call(Fn::Cosh, u), two)); break;
        case Fn::Sqrt: outer = Div(one, Mul(two, Call(Fn::Sqrt, u))); break;
        case Fn::Cbrt: outer = Div(one, Mul(Number(3.0), Pow(Call(Fn::Cbrt, u), two))); break;
        case Fn::Exp: outer = n; break;
        case Fn::Log: outer = Div(one, u); break;
        case Fn::Log10: outer = Div(one, Mul(Number(std::log(10.0)), u)); break;
        case Fn::Log2: outer = Div(one, Mul(Number(std::log(2.0)), u)); break;
        case Fn::Abs: outer = Call(Fn::Sign, u); break;
        default: error = std::string(FnName(n->fn)) + " has no derivative formula"; return nullptr;
    }
    return Mul(outer, du);
}

// ---------------------------------------------------------------------------
// Printing

int Precedence(const NodePtr& n) {
    switch (n->op) {
        case Op::Add: case Op::Subtract: return 1;
        case Op::Multiply: case Op::Divide: return 2;
        case Op::Negate: return Precedence(n->a) == 2 ? 2 : 3;  // -x*y reads as a product
        case Op::Power: return 4;
        case Op::Number: return n->value < 0 ? 3 : 5;
        default: return 5;
    }
}

std::string Print(const NodePtr& n, const std::vector<std::string>& names);

std::string Wrapped(const NodePtr& n, const std::vector<std::string>& names, bool wrap) {
    const std::string s = Print(n, names);
    return wrap ? "(" + s + ")" : s;
}

std::string Print(const NodePtr& n, const std::vector<std::string>& names) {
    const int p = Precedence(n);
    switch (n->op) {
        case Op::Number: {
            char buffer[40];
            std::snprintf(buffer, sizeof buffer, "%.15g", n->value);
            return buffer;
        }
        case Op::Variable: return names[n->variable];
        case Op::Negate: return "-" + Wrapped(n->a, names, Precedence(n->a) < 2);  // -a*b = -(a*b)
        case Op::Add: return Wrapped(n->a, names, false) + " + " + Wrapped(n->b, names, Precedence(n->b) < p);
        case Op::Subtract: return Wrapped(n->a, names, false) + " - " + Wrapped(n->b, names, Precedence(n->b) <= p);
        case Op::Multiply: return Wrapped(n->a, names, Precedence(n->a) < p) + "*" + Wrapped(n->b, names, Precedence(n->b) < p);
        case Op::Divide: return Wrapped(n->a, names, Precedence(n->a) < p) + "/" + Wrapped(n->b, names, Precedence(n->b) <= p);
        case Op::Power: return Wrapped(n->a, names, Precedence(n->a) <= p) + "^" + Wrapped(n->b, names, Precedence(n->b) < 3);
        case Op::Function:
            return std::string(FnName(n->fn)) + "(" + Print(n->a, names) + (n->b ? ", " + Print(n->b, names) : "") + ")";
    }
    return "?";
}

}  // namespace

// ---------------------------------------------------------------------------

Expression::Expression() = default;

Expression Expression::Parse(const std::string& text, const std::vector<std::string>& variables, AngleUnit angles) {
    Expression e;
    e.text_ = text;
    e.variables_ = variables;
    e.angles_ = angles;
    Parser parser(text, variables, angles);
    e.root_ = parser.Parse(e.error_);
    return e;
}

bool Expression::Uses(size_t index) const { return UsesVariable(root_, index); }

double Expression::Evaluate(const std::vector<double>& values, std::string* problem) const {
    if (!root_) {
        if (problem) *problem = error_;
        return std::numeric_limits<double>::quiet_NaN();
    }
    if (values.size() != variables_.size()) {
        if (problem) *problem = "Expected " + std::to_string(variables_.size()) + " values";
        return std::numeric_limits<double>::quiet_NaN();
    }
    return EvaluateHost(root_, values, problem);
}

std::vector<double> Expression::EvaluateMany(const std::vector<std::vector<double>>& columns) const {
    if (!root_ || columns.size() != variables_.size()) return {};
    const size_t count = columns.empty() ? 1 : columns.front().size();
    for (const auto& c : columns)
        if (c.size() != count) return {};
    if (count == 0) return {};
    if (!DeviceHasDouble()) {
        std::vector<double> out(count), point(columns.size());
        for (size_t i = 0; i < count; ++i) {
            for (size_t v = 0; v < columns.size(); ++v) point[v] = columns[v][i];
            out[i] = EvaluateHost(root_, point, nullptr);
        }
        return out;
    }
    std::vector<af::array> device;
    device.reserve(columns.size());
    for (const auto& c : columns) device.push_back(HostValuesToDevice(c));
    return DeviceToHostValues(EvaluateDevice(root_, device, static_cast<dim_t>(count), f64), "Expression::EvaluateMany");
}

Expression Expression::Derivative(size_t index) const {
    Expression d;
    d.variables_ = variables_;
    d.angles_ = angles_;
    if (!root_) {
        d.error_ = error_;
        return d;
    }
    if (index >= variables_.size()) {
        d.error_ = "No variable " + std::to_string(index);
        return d;
    }
    d.root_ = Derive(root_, index, d.error_);
    if (!d.root_ && d.error_.empty()) d.error_ = "No derivative";
    if (d.root_) {
        d.text_ = Print(d.root_, variables_);
        d.error_.clear();
    }
    return d;
}

std::string Expression::ToString() const { return root_ ? Print(root_, variables_) : ""; }

} // namespace cyxwiz
