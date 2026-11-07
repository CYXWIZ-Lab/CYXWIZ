#include "optimization_presentation.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <sstream>

namespace cyxwiz::optcalc {

namespace {

constexpr const char* kMinus = "\xE2\x88\x92";  // U+2212
constexpr double kInf = std::numeric_limits<double>::infinity();

const char* SenseText(ConstraintSense sense) {
    switch (sense) {
        case ConstraintSense::LessEqual: return "\xE2\x89\xA4";     // ≤
        case ConstraintSense::GreaterEqual: return "\xE2\x89\xA5";  // ≥
        case ConstraintSense::Equal: return "=";
    }
    return "";
}

std::string Point(const std::vector<double>& p) {
    std::string s = "(";
    for (size_t i = 0; i < p.size(); ++i) s += (i ? ", " : "") + Number(p[i], 5);
    return s + ")";
}

std::string BoundText(const std::string& text) {
    if (text == "pi") return "\xCF\x80";  // π
    if (text == "inf") return "\xE2\x88\x9E";  // ∞
    if (text == "-inf") return std::string(kMinus) + "\xE2\x88\x9E";
    return text;
}

}  // namespace

const std::vector<OpInfo>& Ops() {
    static const std::vector<OpInfo> ops = {
        {Op::Optimizers, "optimizers", "Optimizers", "Minimize",
         "The training optimizers on a formula of x and y: their paths on its map"},
        {Op::Linear, "linear", "Linear", "Programs",
         "Maximize or minimize c . x under linear constraints, any number of variables"},
        {Op::Quadratic, "quadratic", "Quadratic", "Programs",
         "Minimize 0.5 x' Q x + c' x under linear constraints (Q positive semidefinite)"},
        {Op::Integral, "integral", "Integral", "Calculus",
         "The area under f(x) from a to b, adaptive with an error estimate, and the fixed rules"},
        {Op::Derivative, "derivative", "Derivative", "Calculus",
         "f'(x) and f''(x): exact from the formula, and by differences with the error of each step"},
        {Op::Curvature, "curvature", "Curvature", "Calculus",
         "How f(x, y) curves at a point (the Hessian's eigenvalues), and over the shown range"},
    };
    return ops;
}

const OpInfo& InfoOf(Op op) {
    for (const auto& info : Ops())
        if (info.op == op) return info;
    return Ops().front();
}

const OpInfo* FindOp(const std::string& id) {
    for (const auto& info : Ops())
        if (id == info.id) return &info;
    return nullptr;
}

std::vector<std::string> Groups() { return {"Minimize", "Programs", "Calculus"}; }

std::string Number(double value, int digits) {
    if (std::isnan(value)) return "NaN";
    if (std::isinf(value)) return value > 0 ? "\xE2\x88\x9E" : std::string(kMinus) + "\xE2\x88\x9E";
    if (value == 0.0) return "0";
    char buffer[64];
    // An exponent below 1e-3 (1.6e-4, not 0.00016), as the error figures read.
    if (std::abs(value) < 1e-3)
        std::snprintf(buffer, sizeof buffer, "%.*e", (std::max)(0, digits - 1), std::abs(value));
    else
        std::snprintf(buffer, sizeof buffer, "%.*g", digits, std::abs(value));
    std::string s = buffer;
    // Trailing zeros of a mantissa ("1.500e-05" -> "1.5e-05").
    const size_t e = s.find('e');
    std::string mantissa = s.substr(0, e), exponent = e == std::string::npos ? "" : s.substr(e);
    if (mantissa.find('.') != std::string::npos) {
        while (!mantissa.empty() && mantissa.back() == '0') mantissa.pop_back();
        if (!mantissa.empty() && mantissa.back() == '.') mantissa.pop_back();
    }
    if (!exponent.empty()) {  // e-05 -> e-5
        const char sign = exponent[1];
        size_t i = 2;
        while (i + 1 < exponent.size() && exponent[i] == '0') ++i;
        exponent = std::string("e") + (sign == '-' ? "-" : "") + exponent.substr(i);
    }
    return (value < 0 ? std::string(kMinus) : std::string()) + mantissa + exponent;
}

std::string Count(long long value) {
    std::string digits = std::to_string(value < 0 ? -value : value), out;
    for (size_t i = 0; i < digits.size(); ++i) {
        if (i && (digits.size() - i) % 3 == 0) out += ',';
        out += digits[i];
    }
    return (value < 0 ? std::string(kMinus) : std::string()) + out;
}

// ---------------------------------------------------------------------------

const std::vector<FormulaExample>& SurfaceExamples() {
    static const std::vector<FormulaExample> examples = {
        {"Rosenbrock", "(1 - x)^2 + 100*(y - x^2)^2", -2, 2, -2.5, 2, -1, 3, "(1, 1)"},
        {"Himmelblau", "(x^2 + y - 11)^2 + (x + y^2 - 7)^2", 0, 0, -5, 5, -5, 5,
         "(3, 2), (\xE2\x88\x92" "2.805, 3.131), (\xE2\x88\x92" "3.779, \xE2\x88\x92" "3.283), (3.584, \xE2\x88\x92" "1.848)"},
        {"Rastrigin", "20 + x^2 - 10*cos(2*pi*x) + y^2 - 10*cos(2*pi*y)", 2.2, 1.7, -3, 3, -3, 3, "(0, 0)"},
        {"Beale", "(1.5 - x + x*y)^2 + (2.25 - x + x*y^2)^2 + (2.625 - x + x*y^3)^2", 1, 1.5, -4.5, 4.5, -4.5, 4.5,
         "(3, 0.5)"},
        {"Sphere", "x^2 + y^2", 2, -1.5, -3, 3, -3, 3, "(0, 0)"},
    };
    return examples;
}

const std::vector<LineExample>& LineExamples() {
    static const std::vector<LineExample> examples = {
        {"sin(x)", "sin(x)", "0", "pi", 1.0},
        {"exp(-x^2)", "exp(-x^2)", "-inf", "inf", 0.5},
        {"1/(1+x^2)", "1/(1 + x^2)", "0", "1", 1.0},
        {"sqrt(x)", "sqrt(x)", "0", "1", 2.0},
        {"x^3 - 2x", "x^3 - 2*x", "-1", "2", -1.5},
    };
    return examples;
}

bool ReadValue(const std::string& text, double& value, std::string& error, bool allow_infinite) {
    std::string t;
    for (const char c : text)
        if (!std::isspace(static_cast<unsigned char>(c))) t += static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    if (t == "inf" || t == "+inf" || t == "-inf") {
        if (!allow_infinite) {
            error = "Infinity is not allowed here";
            return false;
        }
        value = t == "-inf" ? -kInf : kInf;
        return true;
    }
    const Expression e = Expression::Parse(text);
    if (!e.Valid()) {
        error = "'" + text + "' is not a number: " + e.Error();
        return false;
    }
    value = e.Evaluate({});
    if (!std::isfinite(value)) {
        error = "'" + text + "' is not a finite number";
        return false;
    }
    return true;
}

std::vector<Segment> ContourSegments(const std::vector<double>& v, int nx, int ny, double x_min, double x_max,
                                     double y_min, double y_max, double level) {
    std::vector<Segment> out;
    if (nx < 2 || ny < 2 || v.size() != static_cast<size_t>(nx) * ny) return out;
    const double dx = (x_max - x_min) / (nx - 1), dy = (y_max - y_min) / (ny - 1);
    for (int j = 0; j + 1 < ny; ++j)
        for (int i = 0; i + 1 < nx; ++i) {
            const double c[4] = {v[j * nx + i], v[j * nx + i + 1], v[(j + 1) * nx + i + 1], v[(j + 1) * nx + i]};
            if (!std::isfinite(c[0]) || !std::isfinite(c[1]) || !std::isfinite(c[2]) || !std::isfinite(c[3])) continue;
            const double px[4] = {x_min + dx * i, x_min + dx * (i + 1), x_min + dx * (i + 1), x_min + dx * i};
            const double py[4] = {y_min + dy * j, y_min + dy * j, y_min + dy * (j + 1), y_min + dy * (j + 1)};
            double hit[4][2];
            int hits = 0;
            for (int e = 0; e < 4; ++e) {
                const double a = c[e], b = c[(e + 1) % 4];
                if ((a < level) == (b < level)) continue;
                const double t = (level - a) / (b - a);
                hit[hits][0] = px[e] + (px[(e + 1) % 4] - px[e]) * t;
                hit[hits][1] = py[e] + (py[(e + 1) % 4] - py[e]) * t;
                ++hits;
            }
            for (int k = 0; k + 1 < hits; k += 2) out.push_back({hit[k][0], hit[k][1], hit[k + 1][0], hit[k + 1][1]});
        }
    return out;
}

SurfaceMap MapOf(const Expression& f, double x_min, double x_max, double y_min, double y_max, int n, int level_count) {
    SurfaceMap map;
    if (!f.Valid()) {
        map.error = f.Error();
        return map;
    }
    if (f.Variables().size() != 2) {
        map.error = "The map takes a formula of x and y";
        return map;
    }
    if (!(x_max > x_min) || !(y_max > y_min) || n < 2) {
        map.error = "The range needs min < max";
        return map;
    }
    map.nx = map.ny = n;
    map.x_min = x_min;
    map.x_max = x_max;
    map.y_min = y_min;
    map.y_max = y_max;
    std::vector<double> xs, ys;
    xs.reserve(static_cast<size_t>(n) * n);
    ys.reserve(xs.capacity());
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i) {
            xs.push_back(x_min + (x_max - x_min) * i / (n - 1));
            ys.push_back(y_min + (y_max - y_min) * j / (n - 1));
        }
    map.values = f.EvaluateMany({xs, ys});
    std::vector<double> finite;
    for (const double v : map.values)
        if (std::isfinite(v)) finite.push_back(v);
    if (finite.empty()) {
        map.error = "f has no finite value in this range";
        return map;
    }
    std::sort(finite.begin(), finite.end());
    const double low = finite.front(), high = finite.back(), median = finite[finite.size() / 2];
    // A formula whose top is far above its middle (Rosenbrock) gets its
    // lines in log10(f - min + 1), or every line crowds the walls.
    map.log_scale = (high - low) > 10.0 * (median - low + 1e-12);
    auto scaled = [&](double v) { return map.log_scale ? std::log10(v - low + 1.0) : v; };
    std::vector<double> drawn(map.values.size());
    for (size_t i = 0; i < drawn.size(); ++i)
        drawn[i] = std::isfinite(map.values[i]) ? scaled(map.values[i]) : std::numeric_limits<double>::quiet_NaN();
    const double top = scaled(high), bottom = scaled(low);
    if (!(top > bottom)) return map;  // flat: no lines
    for (int k = 1; k <= level_count; ++k) {
        const double level = bottom + (top - bottom) * k / (level_count + 1);
        map.levels.push_back({level, ContourSegments(drawn, n, n, x_min, x_max, y_min, y_max, level)});
    }
    return map;
}

// ---------------------------------------------------------------------------

std::string OptimizerName(const OptimizerChoice& c) {
    switch (c.type) {
        case OptimizerType::SGD: return c.momentum > 0 ? "SGD + momentum " + Number(c.momentum, 3) : "SGD";
        case OptimizerType::Adam: return "Adam";
        case OptimizerType::AdamW: return "AdamW";
        case OptimizerType::NAdam: return "NAdam";
        case OptimizerType::RMSprop: return c.momentum > 0 ? "RMSprop + momentum " + Number(c.momentum, 3) : "RMSprop";
        case OptimizerType::AdaGrad: return "AdaGrad";
        default: return "Optimizer";
    }
}

std::vector<OptimizerChoice> OfferedOptimizers() {
    return {{OptimizerType::SGD, 0.001, 0.9, true}, {OptimizerType::Adam, 0.01, 0.0, true},
            {OptimizerType::RMSprop, 0.01, 0.0, true}, {OptimizerType::SGD, 0.01, 0.0, true},
            {OptimizerType::AdamW, 0.01, 0.0, false}, {OptimizerType::NAdam, 0.01, 0.0, false},
            {OptimizerType::AdaGrad, 0.1, 0.0, false}};
}

MinimizeAnswer RunMinimize(const MinimizeRequest& request, const std::function<bool()>& stop) {
    MinimizeAnswer answer;
    const Expression f = Expression::Parse(request.formula, {"x", "y"});
    if (!f.Valid()) {
        answer.error = f.Error();
        return answer;
    }
    if (request.steps < 1) {
        answer.error = "Take at least 1 step";
        return answer;
    }
    int best = -1;
    std::vector<std::string> diverged;
    for (size_t i = 0; i < request.optimizers.size(); ++i) {
        const OptimizerChoice& c = request.optimizers[i];
        if (!c.on) continue;
        const OptimizerPath path = Optimization::FollowOptimizer(
            f, {request.x0, request.y0}, {c.type, c.learning_rate, c.momentum}, request.steps, 1e-8, stop);
        if (!path.success) {
            answer.error = OptimizerName(c) + ": " + path.error_message;
            return answer;
        }
        PathRow row;
        row.name = OptimizerName(c);
        row.learning_rate = Number(c.learning_rate, 4);
        row.choice = i;
        const int taken = static_cast<int>(path.points.size()) - 1;
        if (path.diverged) {
            row.ended_at = "left the finite numbers";
            row.value = row.gradient = "overflow";
            row.word = "Diverged at step " + Count(path.diverged_at);
            row.tone = Tone::Warning;
            diverged.push_back(row.name + " at learning rate " + row.learning_rate);
        } else {
            const double value = path.values.back(), norm = path.gradient_norms.back();
            row.ended_at = Point(path.points.back());
            row.value = Number(value, 5);
            row.gradient = Number(norm, 4);
            if (path.stopped) {
                row.word = "Stopped at step " + Count(taken);
            } else if (path.converged || norm <= 1e-2 * (std::max)(1.0, std::abs(value))) {
                const CurvatureResult here = Calculus::CurvatureAt(f, path.points.back());
                const bool bowl = here.success && !here.eigenvalues.empty() && here.eigenvalues.front() > 0;
                const bool saddle = here.success && !here.eigenvalues.empty() && here.eigenvalues.front() < 0 &&
                                    here.eigenvalues.back() > 0;
                row.word = bowl ? "Reached a minimum" : (saddle ? "Stopped on a saddle" : "Reached a flat point");
                row.tone = bowl ? Tone::Good : Tone::Plain;
            } else {
                row.word = "Still moving at step " + Count(taken);
            }
            if (best < 0 || value < answer.paths[static_cast<size_t>(best)].values.back()) best = static_cast<int>(answer.paths.size());
        }
        answer.paths.push_back(path);
        answer.rows.push_back(row);
    }
    if (answer.rows.empty()) {
        answer.error = "Turn on at least one optimizer";
        return answer;
    }
    std::string s;
    if (best >= 0) {
        const PathRow& row = answer.rows[static_cast<size_t>(best)];
        s = row.name + " ends lowest: f = " + row.value + " at " + row.ended_at;
        s += row.tone == Tone::Good ? ", a minimum." : (row.word.rfind("Still", 0) == 0 ? ", still moving." : ".");
    }
    if (!diverged.empty()) {
        std::string list;
        for (size_t i = 0; i < diverged.size(); ++i) list += (i ? (i + 1 == diverged.size() ? " and " : ", ") : "") + diverged[i];
        s += (s.empty() ? "" : " ") + list + " diverged: " + (diverged.size() == 1 ? "its" : "their") +
             " steps grew without limit. Try a smaller learning rate.";
    }
    answer.sentence = s;
    return answer;
}

// ---------------------------------------------------------------------------

namespace {

ProgramRequest Linear(bool maximize, std::vector<std::string> vars, std::vector<double> c, std::vector<ProgramRow> rows,
                      std::vector<VariableBound> bounds = {}) {
    ProgramRequest r;
    r.maximize = maximize;
    r.variables = std::move(vars);
    r.objective = std::move(c);
    r.rows = std::move(rows);
    r.bounds = bounds.empty() ? std::vector<VariableBound>(r.variables.size()) : std::move(bounds);
    return r;
}

ProgramRequest Quadratic(std::vector<std::string> vars, std::vector<std::vector<double>> Q, std::vector<double> c,
                         std::vector<ProgramRow> rows, std::vector<VariableBound> bounds) {
    ProgramRequest r;
    r.quadratic = true;
    r.maximize = false;
    r.variables = std::move(vars);
    r.Q = std::move(Q);
    r.objective = std::move(c);
    r.rows = std::move(rows);
    r.bounds = std::move(bounds);
    return r;
}

}  // namespace

const std::vector<ProgramExample>& LinearExamples() {
    using S = ConstraintSense;
    static const std::vector<ProgramExample> examples = {
        {"Two products (2 variables)",
         Linear(true, {"x1", "x2"}, {3, 2}, {{"C1", {1, 1}, S::LessEqual, 4}, {"C2", {1, 2}, S::LessEqual, 6}})},
        {"Production plan (4 variables, one =)",
         Linear(true, {"a", "b", "c", "d"}, {20, 30, 25, 15},
                {{"Labour", {1, 2, 1, 1}, S::LessEqual, 40}, {"Machine", {3, 1, 2, 1}, S::LessEqual, 60},
                 {"Order", {1, 1, 1, 0}, S::Equal, 25}})},
        {"Diet (\xE2\x89\xA5 rows)",
         Linear(false, {"bread", "milk"}, {0.6, 0.35},
                {{"Protein", {5, 7}, S::GreaterEqual, 8}, {"Fat", {4, 2}, S::GreaterEqual, 15},
                 {"Fibre", {2, 1}, S::GreaterEqual, 3}})},
        {"Transport (6 routes)",
         Linear(false, {"s1d1", "s1d2", "s1d3", "s2d1", "s2d2", "s2d3"}, {8, 6, 10, 9, 12, 13},
                {{"Supply 1", {1, 1, 1, 0, 0, 0}, S::LessEqual, 50}, {"Supply 2", {0, 0, 0, 1, 1, 1}, S::LessEqual, 60},
                 {"Demand 1", {1, 0, 0, 1, 0, 0}, S::GreaterEqual, 30}, {"Demand 2", {0, 1, 0, 0, 1, 0}, S::GreaterEqual, 40},
                 {"Demand 3", {0, 0, 1, 0, 0, 1}, S::GreaterEqual, 35}})},
        {"No solution", Linear(false, {"x1", "x2"}, {1, 1}, {{"C1", {1, 1}, S::LessEqual, 1}, {"C2", {1, 1}, S::GreaterEqual, 3}})},
    };
    return examples;
}

const std::vector<ProgramExample>& QuadraticExamples() {
    using S = ConstraintSense;
    const VariableBound free{-kInf, kInf}, positive{0.0, kInf};
    static const std::vector<ProgramExample> examples = {
        {"Quadratic (2 variables)",
         Quadratic({"x1", "x2"}, {{2, 0}, {0, 2}}, {-2, -5}, {{"C1", {1, 1}, S::LessEqual, 3}}, {free, free})},
        {"Portfolio (3 assets)",
         Quadratic({"A", "B", "C"}, {{0.04, 0.006, 0.002}, {0.006, 0.09, 0.009}, {0.002, 0.009, 0.0225}}, {0, 0, 0},
                   {{"Budget", {1, 1, 1}, S::Equal, 1}, {"Return 8 %", {0.06, 0.12, 0.08}, S::GreaterEqual, 0.08}},
                   {positive, positive, positive})},
        {"Box (bounds and a \xE2\x89\xA5 row)",
         Quadratic({"x1", "x2"}, {{4, 1}, {1, 2}}, {1, 1}, {{"R1", {1, 1}, S::GreaterEqual, 1}}, {positive, positive})},
    };
    return examples;
}

bool ParseRows(const std::string& text, size_t variables, std::vector<ProgramRow>& rows, std::string& error) {
    std::vector<ProgramRow> out;
    std::istringstream lines(text);
    std::string line;
    int number = 0;
    while (std::getline(lines, line)) {
        ++number;
        for (char& c : line)
            if (c == ',' || c == ';' || c == '\t') c = ' ';
        std::string trimmed = line;
        trimmed.erase(0, trimmed.find_first_not_of(" \r"));
        if (trimmed.empty()) continue;
        ProgramRow row;
        const size_t colon = trimmed.find(':');
        if (colon != std::string::npos) {
            row.name = trimmed.substr(0, colon);
            while (!row.name.empty() && row.name.back() == ' ') row.name.pop_back();
            trimmed = trimmed.substr(colon + 1);
        }
        std::istringstream words(trimmed);
        std::string word;
        bool sense = false;
        std::vector<double> after;
        while (words >> word) {
            if (word == "<=" || word == ">=" || word == "=" || word == "\xE2\x89\xA4" || word == "\xE2\x89\xA5") {
                if (sense) {
                    error = "Line " + std::to_string(number) + " has two signs";
                    return false;
                }
                sense = true;
                row.sense = (word == "<=" || word == "\xE2\x89\xA4") ? ConstraintSense::LessEqual
                          : (word == ">=" || word == "\xE2\x89\xA5") ? ConstraintSense::GreaterEqual
                                                                       : ConstraintSense::Equal;
                continue;
            }
            double v = 0.0;
            std::string why;
            if (!ReadValue(word, v, why)) {
                error = "Line " + std::to_string(number) + ": " + why;
                return false;
            }
            (sense ? after : row.coefficients).push_back(v);
        }
        if (!sense || after.size() != 1) {
            error = "Line " + std::to_string(number) + " needs <=, >= or = and one number after it";
            return false;
        }
        if (row.coefficients.size() != variables) {
            error = "Line " + std::to_string(number) + " has " + std::to_string(row.coefficients.size()) +
                    " coefficients; the problem has " + std::to_string(variables) + " variables";
            return false;
        }
        row.rhs = after.front();
        if (row.name.empty()) row.name = "C" + std::to_string(out.size() + 1);
        out.push_back(row);
    }
    if (out.empty()) {
        error = "No rows to read";
        return false;
    }
    rows = std::move(out);
    return true;
}

namespace {

using Polygon = std::vector<std::pair<double, double>>;

// The part of `poly` where a x + b y <= c (Sutherland-Hodgman).
Polygon Clip(const Polygon& poly, double a, double b, double c) {
    Polygon out;
    for (size_t i = 0; i < poly.size(); ++i) {
        const auto& p = poly[i];
        const auto& q = poly[(i + 1) % poly.size()];
        const double fp = a * p.first + b * p.second - c, fq = a * q.first + b * q.second - c;
        if (fp <= 0) out.push_back(p);
        if ((fp < 0 && fq > 0) || (fp > 0 && fq < 0)) {
            const double t = fp / (fp - fq);
            out.push_back({p.first + (q.first - p.first) * t, p.second + (q.second - p.second) * t});
        }
    }
    return out;
}

// The line a x + b y = c across the view, if it crosses it.
bool LineAcross(double a, double b, double c, const ProgramPlot& v, Segment& s) {
    std::vector<std::pair<double, double>> hits;
    auto add = [&](double x, double y) {
        if (x >= v.x_min - 1e-9 && x <= v.x_max + 1e-9 && y >= v.y_min - 1e-9 && y <= v.y_max + 1e-9) hits.push_back({x, y});
    };
    if (b != 0) {
        add(v.x_min, (c - a * v.x_min) / b);
        add(v.x_max, (c - a * v.x_max) / b);
    }
    if (a != 0) {
        add((c - b * v.y_min) / a, v.y_min);
        add((c - b * v.y_max) / a, v.y_max);
    }
    if (hits.size() < 2) return false;
    s = {hits[0].first, hits[0].second, hits[1].first, hits[1].second};
    for (const auto& h : hits)  // the two farthest apart
        if (std::hypot(h.first - s.x1, h.second - s.y1) > std::hypot(s.x2 - s.x1, s.y2 - s.y1)) {
            s.x2 = h.first;
            s.y2 = h.second;
        }
    return true;
}

ProgramPlot PlotOf(const ProgramRequest& request, const ProgramResult& result) {
    ProgramPlot plot;
    if (request.variables.size() != 2) return plot;
    // A view that holds the axes' intercepts, the bounds and the best point.
    double hx = 1.0, hy = 1.0, lx = 0.0, ly = 0.0;
    for (const auto& row : request.rows) {
        if (row.coefficients[0] != 0) hx = (std::max)(hx, std::abs(row.rhs / row.coefficients[0]));
        if (row.coefficients[1] != 0) hy = (std::max)(hy, std::abs(row.rhs / row.coefficients[1]));
    }
    const auto& bx = request.bounds[0];
    const auto& by = request.bounds[1];
    if (std::isfinite(bx.lower)) lx = (std::min)(lx, bx.lower);
    if (std::isfinite(by.lower)) ly = (std::min)(ly, by.lower);
    if (!std::isfinite(bx.lower)) lx = -hx;
    if (!std::isfinite(by.lower)) ly = -hy;
    if (result.success) {
        hx = (std::max)(hx, result.x[0]);
        hy = (std::max)(hy, result.x[1]);
        lx = (std::min)(lx, result.x[0]);
        ly = (std::min)(ly, result.x[1]);
    }
    plot.x_min = lx - 0.1 * (hx - lx);
    plot.x_max = hx + 0.15 * (hx - lx);
    plot.y_min = ly - 0.1 * (hy - ly);
    plot.y_max = hy + 0.15 * (hy - ly);
    Polygon region = {{plot.x_min, plot.y_min}, {plot.x_max, plot.y_min}, {plot.x_max, plot.y_max}, {plot.x_min, plot.y_max}};
    if (std::isfinite(bx.lower)) region = Clip(region, -1, 0, -bx.lower);
    if (std::isfinite(bx.upper)) region = Clip(region, 1, 0, bx.upper);
    if (std::isfinite(by.lower)) region = Clip(region, 0, -1, -by.lower);
    if (std::isfinite(by.upper)) region = Clip(region, 0, 1, by.upper);
    for (const auto& row : request.rows) {
        const double a = row.coefficients[0], b = row.coefficients[1];
        if (row.sense != ConstraintSense::GreaterEqual) region = Clip(region, a, b, row.rhs);
        if (row.sense != ConstraintSense::LessEqual) region = Clip(region, -a, -b, -row.rhs);
        Segment s{};
        if (LineAcross(a, b, row.rhs, plot, s)) plot.lines.push_back(s);
        else plot.lines.push_back({0, 0, 0, 0});
    }
    plot.region = region.size() >= 3 ? region : Polygon{};
    plot.has_best = result.success;
    if (result.success) {
        plot.best_x = result.x[0];
        plot.best_y = result.x[1];
    }
    plot.drawn = true;
    return plot;
}

}  // namespace

ProgramAnswer RunProgram(const ProgramRequest& request) {
    ProgramAnswer answer;
    const size_t n = request.variables.size();
    if (n == 0) {
        answer.error = "The problem has no variables";
        return answer;
    }
    if (request.objective.size() != n || (!request.bounds.empty() && request.bounds.size() != n) ||
        (request.quadratic && request.Q.size() != n)) {
        answer.error = "Every variable needs its objective entry and its bounds";
        return answer;
    }
    std::vector<LinearConstraint> rows;
    for (const auto& r : request.rows) rows.push_back({r.name, r.coefficients, r.sense, r.rhs});
    if (request.quadratic) {
        QuadraticProgram qp{request.Q, request.objective, rows, request.bounds};
        answer.result = Optimization::SolveQuadraticProgram(qp);
    } else {
        LinearProgram lp{request.maximize, request.objective, rows, request.bounds};
        answer.result = Optimization::SolveLinearProgram(lp);
    }
    const ProgramResult& r = answer.result;
    switch (r.status) {
        case ProgramStatus::Optimal: answer.status = "Optimal"; answer.tone = Tone::Good; break;
        case ProgramStatus::Infeasible: answer.status = "Infeasible"; answer.tone = Tone::Warning; break;
        case ProgramStatus::Unbounded: answer.status = "Unbounded"; answer.tone = Tone::Warning; break;
        case ProgramStatus::NotConvex: answer.status = "Not convex"; answer.tone = Tone::Warning; break;
        case ProgramStatus::Failed: answer.status = "Not solved"; answer.tone = Tone::Warning; break;
    }
    if (!r.success) {
        // A problem that cannot be posed (sizes, numbers) is an error; a
        // problem without an answer is an answer.
        if (r.status == ProgramStatus::Failed && r.iterations == 0 && r.error_message.find("solver") == std::string::npos)
            answer.error = r.error_message;
        answer.sentence = r.error_message + ".";
        answer.plot = PlotOf(request, r);
        return answer;
    }
    const std::string what = request.quadratic ? "the value" : "Z";
    std::string s = request.quadratic ? "Smallest value " + Number(r.objective) + " at "
                                      : std::string("Best Z = ") + Number(r.objective) + " at ";
    for (size_t i = 0; i < n; ++i) {
        s += (i ? (i + 1 == n ? " and " : ", ") : "") + request.variables[i] + " = " + Number(r.x[i]);
        answer.variables.push_back({request.variables[i], Number(r.x[i], 8)});
    }
    s += ".";
    std::vector<std::string> spare;
    for (size_t i = 0; i < request.rows.size(); ++i) {
        const auto& row = request.rows[i];
        const double price = std::abs(r.shadow_prices[i]) < 1e-12 ? 0.0 : r.shadow_prices[i];
        ConstraintRow out{row.name, Number(r.slack[i]), Number(price),
                          row.sense == ConstraintSense::Equal ? "equality" : (r.binding[i] ? "yes" : "no")};
        answer.constraints.push_back(out);
        if (row.sense == ConstraintSense::Equal || r.binding[i]) {
            if (price != 0.0)
                s += " " + row.name + (row.sense == ConstraintSense::Equal ? " holds exactly" : " is binding") +
                     ": one more unit of its right-hand side " + (price > 0 ? "adds " : "takes ") + Number(std::abs(price)) +
                     (price > 0 ? " to " : " from ") + what + ".";
        } else {
            spare.push_back(row.name + " has " + Number(r.slack[i]));
        }
    }
    if (!spare.empty()) {
        std::string list;
        for (size_t i = 0; i < spare.size(); ++i) list += (i ? (i + 1 == spare.size() ? " and " : ", ") : "") + spare[i];
        s += " " + list + " to spare.";
    }
    answer.sentence = s;
    answer.plot = PlotOf(request, r);
    return answer;
}

// ---------------------------------------------------------------------------

namespace {

const char* RuleName(QuadratureRule rule) {
    switch (rule) {
        case QuadratureRule::Adaptive: return "Adaptive (Gauss\xE2\x80\x93Kronrod)";
        case QuadratureRule::Trapezoid: return "Trapezoid";
        case QuadratureRule::Midpoint: return "Midpoint";
        case QuadratureRule::Simpson: return "Simpson";
        case QuadratureRule::GaussLegendre: return "Gauss\xE2\x80\x93Legendre";
    }
    return "";
}

}  // namespace

IntegralAnswer RunIntegral(const IntegralRequest& request) {
    IntegralAnswer answer;
    const Expression f = Expression::Parse(request.formula, {"x"});
    if (!f.Valid()) {
        answer.error = f.Error();
        return answer;
    }
    double a = 0, b = 0;
    std::string why;
    if (!ReadValue(request.a, a, why, true) || !ReadValue(request.b, b, why, true)) {
        answer.error = why;
        return answer;
    }
    const bool finite = std::isfinite(a) && std::isfinite(b);
    const QuadratureResult adaptive = Calculus::Integrate(f, a, b);
    if (!adaptive.success) {
        answer.error = adaptive.error_message;
        return answer;
    }
    QuadratureResult chosen = adaptive;
    if (request.rule != QuadratureRule::Adaptive) {
        if (!finite) {
            answer.error = "Infinite bounds need the adaptive rule";
            return answer;
        }
        chosen = Calculus::Integrate(f, a, b, {request.rule, request.n});
        if (!chosen.success) {
            answer.error = chosen.error_message;
            return answer;
        }
    }
    answer.value = chosen.value;
    answer.title = "\xE2\x88\xAB " + request.formula + " dx from " + BoundText(request.a) + " to " + BoundText(request.b) +
                   " = " + Number(chosen.value, 15);
    if (request.rule == QuadratureRule::Adaptive) {
        answer.sentence = adaptive.converged
            ? "Estimated error " + Number(adaptive.error_estimate, 2) + ", from " + Count(adaptive.evaluations) + " evaluations."
            : "Did not reach the tolerance in " + Count(adaptive.intervals) + " intervals (" + Count(adaptive.evaluations) +
                  " evaluations); the estimated error is " + Number(adaptive.error_estimate, 2) +
                  ". A singularity at an end slows it: split the range there.";
    } else {
        const std::string count = request.rule == QuadratureRule::GaussLegendre ? Count(request.n) + " points"
                                                                                : Count(request.n) + " intervals";
        answer.sentence = std::string(RuleName(request.rule)) + " with " + count + ", " + Count(chosen.evaluations) +
                          " evaluations. It differs from the adaptive answer by " +
                          Number(std::abs(chosen.value - adaptive.value), 2) + " (whose own estimated error is " +
                          Number(adaptive.error_estimate, 2) + ").";
    }
    answer.rules.push_back({RuleName(QuadratureRule::Adaptive), Number(adaptive.value, 15), Count(adaptive.evaluations),
                            "estimate " + Number(adaptive.error_estimate, 2)});
    if (finite) {
        const int n = (std::max)(2, request.n);
        const std::pair<QuadratureRule, int> rules[] = {{QuadratureRule::Trapezoid, n}, {QuadratureRule::Midpoint, n},
            {QuadratureRule::Simpson, n % 2 ? n + 1 : n}, {QuadratureRule::GaussLegendre, 5}, {QuadratureRule::GaussLegendre, 10}};
        for (const auto& [rule, k] : rules) {
            const QuadratureResult q = Calculus::Integrate(f, a, b, {rule, k});
            if (!q.success) continue;
            answer.rules.push_back({std::string(RuleName(rule)) + ", " + Count(k), Number(q.value, 15), Count(q.evaluations),
                                    Number(std::abs(q.value - adaptive.value), 2)});
        }
    }
    // The curve over the range, its finite part when a bound is infinite.
    double lo = a, hi = b;
    if (lo > hi) std::swap(lo, hi);
    if (!std::isfinite(lo) && !std::isfinite(hi)) {
        lo = -5;
        hi = 5;
    } else if (!std::isfinite(lo)) {
        lo = hi - 10;
    } else if (!std::isfinite(hi)) {
        hi = lo + 10;
    }
    answer.shade_from = lo;
    answer.shade_to = hi;
    const double pad = 0.08 * (hi - lo);
    for (int i = 0; i <= 240; ++i) answer.curve_x.push_back(lo - pad + (hi - lo + 2 * pad) * i / 240.0);
    answer.curve_y = f.EvaluateMany({answer.curve_x});
    return answer;
}

DerivativeAnswer RunDerivative(const DerivativeRequest& request) {
    DerivativeAnswer answer;
    const Expression f = Expression::Parse(request.formula, {"x"});
    if (!f.Valid()) {
        answer.error = f.Error();
        return answer;
    }
    double x = 0;
    if (!ReadValue(request.x, x, answer.error)) return answer;
    answer.result = Calculus::Differentiate(f, x, request.rule, request.h);
    if (!answer.result.success) {
        answer.error = answer.result.error_message;
        return answer;
    }
    const DifferentiationResult& r = answer.result;
    const char* rule = request.rule == DifferenceRule::Central ? "Central" : (request.rule == DifferenceRule::Forward ? "Forward" : "Backward");
    if (r.has_exact) {
        answer.title = "f\xE2\x80\xB2(" + request.x + ") = " + Number(r.exact_first, 12) + " \xC2\xB7 f\xE2\x80\xB3(" + request.x +
                       ") = " + Number(r.exact_second, 12);
        answer.sentence = "From the formula: f\xE2\x80\xB2 = " + r.first_formula + ", f\xE2\x80\xB3 = " + r.second_formula + ". " +
                          rule + " differences with h = " + Number(r.step, 2) + " give " + Number(r.numerical_first, 12) +
                          ", off by " + Number(std::abs(r.numerical_first - r.exact_first), 2) + ".";
        answer.curve = Calculus::ErrorByStep(f, x);
    } else {
        answer.title = "f\xE2\x80\xB2(" + request.x + ") \xE2\x89\x88 " + Number(r.numerical_first, 10) + " \xC2\xB7 f\xE2\x80\xB3(" +
                       request.x + ") \xE2\x89\x88 " + Number(r.numerical_second, 8);
        answer.sentence = std::string("The formula has no derivative formula (it uses min, max, mod or factorial); ") + rule +
                          " differences with h = " + Number(r.step, 2) + " give these.";
    }
    const double span = (std::max)(2.0, 0.5 * std::abs(x));
    for (int i = 0; i <= 200; ++i) answer.curve_x.push_back(x - span + 2 * span * i / 200.0);
    answer.curve_f = f.EvaluateMany({answer.curve_x});
    const Expression d = f.Derivative(0);
    if (d.Valid()) answer.curve_df = d.EvaluateMany({answer.curve_x});
    return answer;
}

CurvatureAnswer RunCurvature(const CurvatureRequest& request) {
    CurvatureAnswer answer;
    const Expression f = Expression::Parse(request.formula, {"x", "y"});
    if (!f.Valid()) {
        answer.error = f.Error();
        return answer;
    }
    double x = 0, y = 0;
    if (!ReadValue(request.x, x, answer.error) || !ReadValue(request.y, y, answer.error)) return answer;
    answer.result = Calculus::CurvatureAt(f, {x, y});
    if (!answer.result.success) {
        answer.error = answer.result.error_message;
        return answer;
    }
    const auto& e = answer.result.eigenvalues;
    for (const auto& row : answer.result.hessian)
        for (const double v : row) answer.hessian.push_back(Number(v, 6));
    for (const double v : e) answer.eigenvalues.push_back(Number(v, 6));
    const std::string at = "(" + Number(x, 5) + ", " + Number(y, 5) + ")";
    const double scale = (std::max)(1.0, (std::max)(std::abs(e.front()), std::abs(e.back())));
    const bool up = e.front() > 1e-9 * scale, down = e.back() < -1e-9 * scale;
    if (up) {
        answer.title = "Curves up in every direction at " + at;
        answer.sentence = "Both eigenvalues are positive, so this point sits in a bowl. That alone does not make f convex";
        const double ratio = e.back() / e.front();
        answer.sentence += ratio >= 10 ? ": the bowl is narrow, one direction " + Count(std::llround(ratio)) +
                                             " times steeper than the other, which is why plain gradient steps zig-zag here."
                                       : ".";
    } else if (down) {
        answer.title = "Curves down in every direction at " + at;
        answer.sentence = "Both eigenvalues are negative: a cap, a high point. f is not convex.";
    } else if (e.front() < -1e-9 * scale && e.back() > 1e-9 * scale) {
        answer.title = "Saddle at " + at;
        answer.sentence = "It curves up along one direction and down along another. A function with a saddle anywhere is not convex.";
    } else {
        answer.title = "Flat in one direction at " + at;
        answer.sentence = "An eigenvalue is 0: along that direction f is straight here, to second order.";
    }
    answer.range = Calculus::CurvatureOverRange(f, request.x_min, request.x_max, request.y_min, request.y_max, request.per_side);
    if (answer.range.success) {
        const std::string where = "Over the shown range (x " + Number(request.x_min) + ".." + Number(request.x_max) + ", y " +
                                  Number(request.y_min) + ".." + Number(request.y_max) + ", " + Count(request.per_side) + " \xC3\x97 " +
                                  Count(request.per_side) + " points): ";
        if (answer.range.negative_points > 0)
            answer.range_sentence = where + "not convex. The Hessian has a negative eigenvalue at " + Count(answer.range.negative_points) +
                                    " of the " + Count(answer.range.points) + " points, for example (" +
                                    Number(answer.range.first_negative_x, 4) + ", " + Number(answer.range.first_negative_y, 4) + ").";
        else
            answer.range_sentence = where + "no point curves down, so f looks convex here (a check of samples, not a proof).";
    }
    return answer;
}

} // namespace cyxwiz::optcalc
