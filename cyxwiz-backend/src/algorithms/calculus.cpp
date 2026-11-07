#include "cyxwiz/calculus.h"

#include "cyxwiz/linear_algebra.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <numeric>

namespace cyxwiz {

namespace {

constexpr double kEps = std::numeric_limits<double>::epsilon();
constexpr double kPi = 3.14159265358979323846;

// QUADPACK qk21: the Kronrod nodes (positive half, the last is the centre),
// their weights, and the 10-point Gauss weights of nodes 1, 3, 5, 7, 9.
constexpr double kXgk[11] = {
    0.995657163025808080735527280689003, 0.973906528517171720077964012084452,
    0.930157491355708226001207180059508, 0.865063366688984510732096688423493,
    0.780817726586416897063717578345042, 0.679409568299024406234327365114874,
    0.562757134668604683339000099272694, 0.433395394129247190799265943165784,
    0.294392862701460198131126603103866, 0.148874338981631210884826001129720, 0.0};
constexpr double kWgk[11] = {
    0.011694638867371874278064396062192, 0.032558162307964727478818972459390,
    0.054755896574351996031381300244580, 0.075039674810919952767043140916190,
    0.093125454583697605535065465083366, 0.109387158802297641899210590325805,
    0.123491976262065851077208969293981, 0.134709217311473325928054001771707,
    0.142775938577060080797094273138717, 0.147739104901338491374841515972068,
    0.149445554002916905664936468389821};
constexpr double kWg[5] = {
    0.066671344308688137593568809893332, 0.149451349150580593145776339657697,
    0.219086362515982043995534934228163, 0.269266719309996355091226921569469,
    0.295524224714752870173892994651338};

// The integrand in the variable the rule runs over: x itself on a finite
// range, else t in (0, 1] with x = a + (1 - t)/t (QUADPACK qagi's map).
class Integrand {
public:
    Integrand(const Expression& f, double a, double b) : f_(f) {
        if (std::isinf(a) && std::isinf(b)) {
            kind_ = Kind::Both;
        } else if (std::isinf(b)) {
            kind_ = Kind::Upper;
            bound_ = a;
        } else if (std::isinf(a)) {
            kind_ = Kind::Lower;
            bound_ = b;
        }
    }

    bool Infinite() const { return kind_ != Kind::Finite; }

    // Values at `t`; "" or why not (a value that is not finite).
    std::string Values(const std::vector<double>& t, std::vector<double>& out) const {
        if (kind_ == Kind::Finite) {
            out = f_.EvaluateMany({t});
            return Check(t, out);
        }
        std::vector<double> x(t.size());
        for (size_t i = 0; i < t.size(); ++i) {
            const double s = (1.0 - t[i]) / t[i];
            x[i] = kind_ == Kind::Lower ? bound_ - s : (kind_ == Kind::Upper ? bound_ + s : s);
        }
        out = f_.EvaluateMany({x});
        if (kind_ == Kind::Both) {
            std::vector<double> minus(x.size());
            std::transform(x.begin(), x.end(), minus.begin(), [](double v) { return -v; });
            const std::vector<double> other = f_.EvaluateMany({minus});
            for (size_t i = 0; i < out.size(); ++i) out[i] += other[i];
        }
        for (size_t i = 0; i < out.size(); ++i) out[i] /= t[i] * t[i];
        return Check(x, out);
    }

private:
    enum class Kind { Finite, Upper, Lower, Both };

    static std::string Check(const std::vector<double>& x, const std::vector<double>& values) {
        if (values.size() != x.size()) return "f could not be evaluated";
        for (size_t i = 0; i < values.size(); ++i)
            if (!std::isfinite(values[i])) {
                char buffer[96];
                std::snprintf(buffer, sizeof buffer, "f is not finite at x = %.10g", x[i]);
                return buffer;
            }
        return "";
    }

    const Expression& f_;
    Kind kind_ = Kind::Finite;
    double bound_ = 0.0;
};

struct Piece {
    double lo, hi;
    double value = 0.0, error = 0.0;
    bool done = false;
};

// Gauss-Kronrod 21 on every piece not yet done, all points in one call.
std::string EvaluatePieces(const Integrand& g, std::vector<Piece>& pieces, int& evaluations) {
    std::vector<size_t> todo;
    for (size_t i = 0; i < pieces.size(); ++i)
        if (!pieces[i].done) todo.push_back(i);
    if (todo.empty()) return "";
    std::vector<double> t;
    t.reserve(todo.size() * 21);
    for (const size_t i : todo) {
        const double c = 0.5 * (pieces[i].lo + pieces[i].hi), h = 0.5 * (pieces[i].hi - pieces[i].lo);
        t.push_back(c);
        for (int j = 0; j < 10; ++j) {
            t.push_back(c - h * kXgk[j]);
            t.push_back(c + h * kXgk[j]);
        }
    }
    std::vector<double> f;
    const std::string problem = g.Values(t, f);
    if (!problem.empty()) return problem;
    evaluations += static_cast<int>(t.size());
    for (size_t k = 0; k < todo.size(); ++k) {
        Piece& p = pieces[todo[k]];
        const double* v = f.data() + k * 21;
        const double h = 0.5 * (p.hi - p.lo), dh = std::abs(h);
        const double fc = v[0];
        double resk = kWgk[10] * fc, resg = 0.0, resabs = std::abs(resk);
        for (int j = 0; j < 10; ++j) {
            const double f1 = v[1 + 2 * j], f2 = v[2 + 2 * j];
            resk += kWgk[j] * (f1 + f2);
            resabs += kWgk[j] * (std::abs(f1) + std::abs(f2));
            if (j % 2 == 1) resg += kWg[j / 2] * (f1 + f2);
        }
        const double reskh = 0.5 * resk;
        double resasc = kWgk[10] * std::abs(fc - reskh);
        for (int j = 0; j < 10; ++j) resasc += kWgk[j] * (std::abs(v[1 + 2 * j] - reskh) + std::abs(v[2 + 2 * j] - reskh));
        resabs *= dh;
        resasc *= dh;
        double error = std::abs((resk - resg) * h);
        if (resasc != 0.0 && error != 0.0) error = resasc * std::min(1.0, std::pow(200.0 * error / resasc, 1.5));
        if (resabs > std::numeric_limits<double>::min() / (50.0 * kEps)) error = std::max(50.0 * kEps * resabs, error);
        p.value = resk * h;
        p.error = error;
        p.done = true;
    }
    return "";
}

QuadratureResult Adaptive(const Expression& f, double a, double b, const QuadratureOptions& o) {
    QuadratureResult r;
    const Integrand g(f, a, b);
    std::vector<Piece> pieces;
    if (g.Infinite()) pieces.push_back({0.0, 1.0});
    else pieces.push_back({a, b});
    const int limit = std::max(1, o.max_intervals);
    while (true) {
        const std::string problem = EvaluatePieces(g, pieces, r.evaluations);
        if (!problem.empty()) {
            r.error_message = problem;
            return r;
        }
        double total = 0.0, error = 0.0;
        for (const Piece& p : pieces) {
            total += p.value;
            error += p.error;
        }
        r.value = total;
        r.error_estimate = error;
        r.intervals = static_cast<int>(pieces.size());
        const double tolerance = std::max(o.absolute_tolerance, o.relative_tolerance * std::abs(total));
        if (error <= tolerance) break;
        if (static_cast<int>(pieces.size()) >= limit) {
            r.converged = false;
            break;
        }
        // Bisect, worst first, every piece whose error exceeds its share of
        // the tolerance, as many as the interval limit allows.
        std::vector<size_t> order(pieces.size());
        std::iota(order.begin(), order.end(), size_t{0});
        std::sort(order.begin(), order.end(), [&](size_t i, size_t j) { return pieces[i].error > pieces[j].error; });
        const double width = g.Infinite() ? 1.0 : std::abs(b - a);
        int room = limit - static_cast<int>(pieces.size());
        std::vector<Piece> next;
        std::vector<bool> split(pieces.size(), false);
        for (size_t rank = 0; rank < order.size() && room > 0; ++rank) {
            const Piece& p = pieces[order[rank]];
            if (rank > 0 && p.error <= tolerance * std::abs(p.hi - p.lo) / width) break;
            split[order[rank]] = true;
            --room;
        }
        for (size_t i = 0; i < pieces.size(); ++i) {
            if (!split[i]) {
                next.push_back(pieces[i]);
                continue;
            }
            const double mid = 0.5 * (pieces[i].lo + pieces[i].hi);
            if (mid == pieces[i].lo || mid == pieces[i].hi) {  // cannot halve further
                next.push_back(pieces[i]);
                continue;
            }
            next.push_back({pieces[i].lo, mid});
            next.push_back({mid, pieces[i].hi});
        }
        if (next.size() == pieces.size()) {
            r.converged = false;
            break;
        }
        pieces = std::move(next);
    }
    r.success = true;
    return r;
}

QuadratureResult Fixed(const Expression& f, double a, double b, const QuadratureOptions& o) {
    QuadratureResult r;
    const int n = o.n;
    if (n < 1) {
        r.error_message = "The rule needs at least 1 interval";
        return r;
    }
    if (o.rule == QuadratureRule::Simpson && n % 2 != 0) {
        r.error_message = "Simpson's rule needs an even number of intervals (" + std::to_string(n) + " given)";
        return r;
    }
    std::vector<double> x, w;
    const double h = (b - a) / n;
    switch (o.rule) {
        case QuadratureRule::Trapezoid:
            for (int i = 0; i <= n; ++i) {
                x.push_back(a + h * i);
                w.push_back(i == 0 || i == n ? 0.5 * h : h);
            }
            break;
        case QuadratureRule::Midpoint:
            for (int i = 0; i < n; ++i) {
                x.push_back(a + h * (i + 0.5));
                w.push_back(h);
            }
            break;
        case QuadratureRule::Simpson:
            for (int i = 0; i <= n; ++i) {
                x.push_back(a + h * i);
                w.push_back(h / 3.0 * (i == 0 || i == n ? 1.0 : (i % 2 == 1 ? 4.0 : 2.0)));
            }
            break;
        case QuadratureRule::GaussLegendre: {
            std::vector<double> nodes, weights;
            Calculus::GaussLegendre(n, nodes, weights);
            const double c = 0.5 * (a + b), half = 0.5 * (b - a);
            for (int i = 0; i < n; ++i) {
                x.push_back(c + half * nodes[i]);
                w.push_back(half * weights[i]);
            }
            break;
        }
        case QuadratureRule::Adaptive: break;
    }
    const std::vector<double> v = f.EvaluateMany({x});
    double sum = 0.0;
    for (size_t i = 0; i < v.size(); ++i) {
        if (!std::isfinite(v[i])) {
            char buffer[96];
            std::snprintf(buffer, sizeof buffer, "f is not finite at x = %.10g", x[i]);
            r.error_message = buffer;
            return r;
        }
        sum += w[i] * v[i];
    }
    r.value = sum;
    r.evaluations = static_cast<int>(x.size());
    r.intervals = o.rule == QuadratureRule::GaussLegendre ? 1 : n;
    r.success = true;
    return r;
}

std::string OneVariable(const Expression& f) {
    if (!f.Valid()) return f.Error();
    if (f.Variables().size() != 1) return "The formula must have one variable (x)";
    return "";
}

}  // namespace

QuadratureResult Calculus::Integrate(const Expression& f, double a, double b, const QuadratureOptions& options) {
    QuadratureResult r;
    r.error_message = OneVariable(f);
    if (!r.error_message.empty()) return r;
    if (std::isnan(a) || std::isnan(b)) {
        r.error_message = "The bounds must be numbers";
        return r;
    }
    if (a == b) {
        r.success = true;
        r.error_estimate = 0.0;
        return r;
    }
    // Integrate low to high, then flip the sign.
    const bool flipped = a > b;
    if (flipped) std::swap(a, b);
    if (std::isinf(a) && std::isinf(b) && a == b) {
        r.error_message = "Both bounds are the same infinity";
        return r;
    }
    const bool infinite = std::isinf(a) || std::isinf(b);
    if (infinite && options.rule != QuadratureRule::Adaptive) {
        r.error_message = "Infinite bounds need the adaptive rule";
        return r;
    }
    r = options.rule == QuadratureRule::Adaptive ? Adaptive(f, a, b, options) : Fixed(f, a, b, options);
    if (flipped) r.value = -r.value;
    return r;
}

void Calculus::GaussLegendre(int n, std::vector<double>& nodes, std::vector<double>& weights) {
    nodes.assign(static_cast<size_t>(std::max(n, 0)), 0.0);
    weights.assign(nodes.size(), 0.0);
    // Newton on the Legendre polynomial from Chebyshev starting points.
    for (int i = 0; i < (n + 1) / 2; ++i) {
        double z = std::cos(kPi * (i + 0.75) / (n + 0.5)), dp = 0.0;
        for (int iteration = 0; iteration < 100; ++iteration) {
            double p1 = 1.0, p2 = 0.0;
            for (int j = 1; j <= n; ++j) {
                const double p3 = p2;
                p2 = p1;
                p1 = ((2.0 * j - 1.0) * z * p2 - (j - 1.0) * p3) / j;
            }
            dp = n * (z * p1 - p2) / (z * z - 1.0);
            const double step = p1 / dp;
            z -= step;
            if (std::abs(step) < 1e-16) break;
        }
        nodes[i] = -z;
        nodes[n - 1 - i] = z;
        weights[i] = weights[n - 1 - i] = 2.0 / ((1.0 - z * z) * dp * dp);
    }
}

DifferentiationResult Calculus::Differentiate(const Expression& f, double x, DifferenceRule rule, double h) {
    DifferentiationResult r;
    r.error_message = OneVariable(f);
    if (!r.error_message.empty()) return r;
    const Expression d1 = f.Derivative(0);
    if (d1.Valid()) {
        const Expression d2 = d1.Derivative(0);
        r.has_exact = d2.Valid();
        if (r.has_exact) {
            r.first_formula = d1.ToString();
            r.second_formula = d2.ToString();
            r.exact_first = d1.Evaluate({x});
            r.exact_second = d2.Evaluate({x});
        }
    }
    const double scale = std::max(1.0, std::abs(x));
    if (h <= 0.0) h = (rule == DifferenceRule::Central ? std::cbrt(kEps) : std::sqrt(kEps)) * scale;
    const double h2 = std::pow(kEps, 0.25) * scale;
    // Steps that are exact in binary, so x + h - x is h.
    const volatile double xh = x + h;
    h = xh - x;
    const volatile double xh2 = x + h2;
    r.step = h;
    r.second_step = xh2 - x;
    auto at = [&f](double v) { return f.Evaluate({v}); };
    switch (rule) {
        case DifferenceRule::Forward: r.numerical_first = (at(x + h) - at(x)) / h; break;
        case DifferenceRule::Backward: r.numerical_first = (at(x) - at(x - h)) / h; break;
        case DifferenceRule::Central: r.numerical_first = (at(x + h) - at(x - h)) / (2.0 * h); break;
    }
    r.numerical_second = (at(x + r.second_step) - 2.0 * at(x) + at(x - r.second_step)) / (r.second_step * r.second_step);
    if (!std::isfinite(at(x))) {
        r.error_message = "f is not finite at x";
        return r;
    }
    r.success = true;
    return r;
}

StepErrorCurve Calculus::ErrorByStep(const Expression& f, double x, double h_from, double h_to, int count) {
    StepErrorCurve r;
    r.error_message = OneVariable(f);
    if (!r.error_message.empty()) return r;
    const Expression d1 = f.Derivative(0);
    if (!d1.Valid()) {
        r.error_message = "No exact derivative to measure against: " + d1.Error();
        return r;
    }
    if (count < 2 || h_from <= 0.0 || h_to <= 0.0) {
        r.error_message = "The steps must be positive, at least 2 of them";
        return r;
    }
    const double exact = d1.Evaluate({x});
    std::vector<double> plus(count), minus(count), here(static_cast<size_t>(count), x);
    for (int i = 0; i < count; ++i) {
        const double e = std::log10(h_from) + (std::log10(h_to) - std::log10(h_from)) * i / (count - 1);
        const double h = std::pow(10.0, e);
        r.steps.push_back(h);
        plus[i] = x + h;
        minus[i] = x - h;
    }
    const std::vector<double> fp = f.EvaluateMany({plus}), fm = f.EvaluateMany({minus}), f0 = f.EvaluateMany({here});
    for (int i = 0; i < count; ++i) {
        const double h = r.steps[i];
        r.forward_error.push_back(std::abs((fp[i] - f0[i]) / h - exact));
        r.central_error.push_back(std::abs((fp[i] - fm[i]) / (2.0 * h) - exact));
    }
    r.success = true;
    return r;
}

CurvatureResult Calculus::CurvatureAt(const Expression& f, const std::vector<double>& point) {
    CurvatureResult r;
    if (!f.Valid()) {
        r.error_message = f.Error();
        return r;
    }
    const size_t n = f.Variables().size();
    if (n == 0 || point.size() != n) {
        r.error_message = "The point needs one value per variable";
        return r;
    }
    r.hessian.assign(n, std::vector<double>(n, 0.0));
    for (size_t i = 0; i < n; ++i) {
        const Expression di = f.Derivative(i);
        if (!di.Valid()) {
            r.error_message = "No exact derivative: " + di.Error();
            return r;
        }
        r.gradient.push_back(di.Evaluate(point));
        for (size_t j = i; j < n; ++j) {
            const Expression dij = di.Derivative(j);
            if (!dij.Valid()) {
                r.error_message = "No exact second derivative: " + dij.Error();
                return r;
            }
            r.hessian[i][j] = r.hessian[j][i] = dij.Evaluate(point);
        }
    }
    for (const auto& row : r.hessian)
        for (const double v : row)
            if (!std::isfinite(v)) {
                r.error_message = "The Hessian is not finite at this point";
                return r;
            }
    const EigenResult eigen = LinearAlgebra::Eigen(r.hessian);
    if (!eigen.success) {
        r.error_message = eigen.error_message;
        return r;
    }
    for (const auto& e : eigen.eigenvalues) r.eigenvalues.push_back(e.real());
    std::sort(r.eigenvalues.begin(), r.eigenvalues.end());
    r.success = true;
    return r;
}

RangeCurvature Calculus::CurvatureOverRange(const Expression& f, double x_min, double x_max,
                                            double y_min, double y_max, int per_side) {
    RangeCurvature r;
    if (!f.Valid()) {
        r.error_message = f.Error();
        return r;
    }
    if (f.Variables().size() != 2) {
        r.error_message = "The range check takes a formula of two variables (x, y)";
        return r;
    }
    if (per_side < 2 || !(x_max > x_min) || !(y_max > y_min)) {
        r.error_message = "The range needs min < max and at least 2 points a side";
        return r;
    }
    Expression h[3];
    const size_t pairs[3][2] = {{0, 0}, {0, 1}, {1, 1}};
    for (int k = 0; k < 3; ++k) {
        const Expression d = f.Derivative(pairs[k][0]);
        h[k] = d.Valid() ? d.Derivative(pairs[k][1]) : d;
        if (!h[k].Valid()) {
            r.error_message = "No exact second derivative: " + h[k].Error();
            return r;
        }
    }
    // Row by row in y, x varying fastest.
    std::vector<double> xs, ys;
    for (int j = 0; j < per_side; ++j)
        for (int i = 0; i < per_side; ++i) {
            xs.push_back(x_min + (x_max - x_min) * i / (per_side - 1));
            ys.push_back(y_min + (y_max - y_min) * j / (per_side - 1));
        }
    const std::vector<double> a = h[0].EvaluateMany({xs, ys}), b = h[1].EvaluateMany({xs, ys}), d = h[2].EvaluateMany({xs, ys});
    bool first = true;
    for (size_t p = 0; p < xs.size(); ++p) {
        if (!std::isfinite(a[p]) || !std::isfinite(b[p]) || !std::isfinite(d[p])) continue;
        ++r.points;
        const double mean = 0.5 * (a[p] + d[p]), radius = std::hypot(0.5 * (a[p] - d[p]), b[p]);
        const double low = mean - radius, high = mean + radius;
        if (low < 0.0) {
            ++r.negative_points;
            if (first) {
                r.first_negative_x = xs[p];
                r.first_negative_y = ys[p];
                first = false;
            }
        } else if (low > 0.0 && high > 0.0) {
            ++r.positive_points;
        }
    }
    r.success = true;
    return r;
}

} // namespace cyxwiz
