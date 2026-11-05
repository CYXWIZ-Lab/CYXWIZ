#include "linear_algebra_presentation.h"

#include "session_query_engine.h"

#include <cyxwiz/linear_algebra.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <numeric>
#include <sstream>

namespace cyxwiz::linalg {

namespace {

const std::vector<OpInfo> kOps = {
    {Op::Add, "add", "A + B", "Combine", Second::MatrixB, "Adds B to A, cell by cell."},
    {Op::Subtract, "subtract", "A - B", "Combine", Second::MatrixB, "Takes B from A, cell by cell."},
    {Op::Multiply, "multiply", "A × B", "Combine", Second::MatrixB, "The matrix product: each cell a row of A times a column of B."},
    {Op::Scale, "scale", "k × A", "Combine", Second::Scalar, "Every cell of A times k."},
    {Op::Transpose, "transpose", "Transpose", "Describe", Second::None, "The rows of A become its columns."},
    {Op::Inverse, "inverse", "Inverse", "Describe", Second::None, "The matrix that A times gives the identity."},
    {Op::Determinant, "determinant", "Determinant", "Describe", Second::None, "How A scales volume; 0 when it is singular."},
    {Op::Rank, "rank", "Rank", "Describe", Second::None, "How many of A's columns are independent."},
    {Op::Norm, "norm", "Norm", "Describe", Second::None, "The size of A: the root of the sum of its squares (Frobenius)."},
    {Op::Condition, "condition", "Condition", "Describe", Second::None, "How much A magnifies errors: its largest over its smallest singular value."},
    {Op::Solve, "solve", "A x = b", "Solve", Second::VectorB, "The x with A x = b, for a square A that is not singular."},
    {Op::LeastSquares, "least_squares", "Least squares", "Solve", Second::VectorB, "The x that brings A x closest to b, for any A."},
    {Op::Eigen, "eigen", "Eigen", "Factor", Second::None, "The directions A only stretches, and by how much."},
    {Op::SVD, "svd", "SVD", "Factor", Second::None, "A = U S V^T: its directions and their strengths, for any A."},
    {Op::QR, "qr", "QR", "Factor", Second::None, "A = Q R: orthonormal columns times an upper triangle."},
    {Op::Cholesky, "cholesky", "Cholesky", "Factor", Second::None, "A = L L^T, for a symmetric positive definite A."},
    {Op::LU, "lu", "LU", "Factor", Second::None, "A with its rows reordered = L U, lower times upper triangle."},
};

std::string Size(size_t rows, size_t cols) { return std::to_string(rows) + " × " + std::to_string(cols); }
std::string Size(const Matrix& m) { return Size(m.size(), m.empty() ? 0 : m[0].size()); }

double Scale(const Matrix& m) {
    double s = 0;
    for (const auto& r : m)
        for (const double v : r) s = std::max(s, std::fabs(v));
    return s;
}

double MaxDifference(const Matrix& x, const Matrix& y) {
    double d = 0;
    for (size_t i = 0; i < x.size() && i < y.size(); ++i)
        for (size_t j = 0; j < x[i].size() && j < y[i].size(); ++j) d = std::max(d, std::fabs(x[i][j] - y[i][j]));
    return d;
}

std::string Short(double v) {
    char buffer[32];
    std::snprintf(buffer, sizeof(buffer), "%.2g", v);
    return buffer;
}

std::string Plain(double v) {  // a figure like 14.93 or 2.5: up to 4 decimals, trailing zeros dropped
    char buffer[48];
    std::snprintf(buffer, sizeof(buffer), "%.4f", v);
    std::string s = buffer;
    if (s.find('.') != std::string::npos) {
        while (s.back() == '0') s.pop_back();
        if (s.back() == '.') s.pop_back();
    }
    return s == "-0" ? "0" : s;
}

std::string Percent(double share) {
    char buffer[32];
    std::snprintf(buffer, sizeof(buffer), "%.2f %%", 100.0 * share);
    return buffer;
}

Table MatrixTable(const std::string& title, const Matrix& m, const std::string& column, const std::string& row = "r") {
    Table t;
    t.title = title;
    t.values = m;
    const size_t cols = m.empty() ? 0 : m[0].size();
    for (size_t j = 0; j < cols; ++j) t.columns.push_back(column + std::to_string(j + 1));
    for (size_t i = 0; i < m.size(); ++i) t.row_names.push_back(row + std::to_string(i + 1));
    return t;
}

// The reason a matrix cannot be read, or "".
std::string Malformed(const Matrix& m, const char* name) {
    if (m.empty() || m[0].empty()) return std::string(name) + " is empty.";
    for (const auto& r : m) {
        if (r.size() != m[0].size()) return std::string(name) + " has rows of different lengths.";
        for (const double v : r)
            if (!std::isfinite(v)) return std::string(name) + " has a value that is not a number.";
    }
    return "";
}

std::string ConditionWords(double c) {
    if (!std::isfinite(c)) return "A is singular: its condition is infinite.";
    if (c < 1e3) return "A is well conditioned (condition " + Plain(c) + "): small changes in the input move the answer little.";
    const int digits = static_cast<int>(std::floor(std::log10(c)));
    return "A is ill conditioned (condition " + Short(c) + "): the answer can lose about " + std::to_string(digits) + " of its 16 digits.";
}

// rank of A by the backend's rule.
int RankOf(const Matrix& a, std::string& error) {
    const ScalarResult r = LinearAlgebra::Rank(a);
    if (!r.success) error = r.error_message;
    return static_cast<int>(std::lround(r.value));
}

std::string Join(const std::vector<double>& v, double scale) {
    std::string s;
    for (const double x : v) s += (s.empty() ? "" : " · ") + Number(x, scale);
    return s;
}

// A's column names, when they fit `count` (A from a table).
bool HasNames(const Request& r, size_t count) {
    return r.a_columns.size() == count && std::any_of(r.a_columns.begin(), r.a_columns.end(), [](const std::string& n) { return !n.empty(); });
}

struct Stopper {
    const std::function<bool()>& stop;
    bool operator()() const { return stop && stop(); }
};

Answer Refuse(std::string why, std::vector<std::string> instead = {}) {
    Answer a;
    a.error = std::move(why);
    a.try_instead = std::move(instead);
    return a;
}

// ---- Combine and Describe

Answer RunCombine(const Request& r) {
    Answer a;
    const size_t m = r.a.size(), n = r.a[0].size();
    MatrixResult result;
    if (r.op == Op::Scale) {
        if (!std::isfinite(r.k)) return Refuse("k is not a number.");
        result = LinearAlgebra::ScalarMultiply(r.a, r.k);
        a.sentence = "k × A with k = " + Plain(r.k) + ": every cell of A times " + Plain(r.k) + ".";
    } else {
        if (const std::string bad = Malformed(r.b, "B"); !bad.empty()) return Refuse(bad);
        const size_t bm = r.b.size(), bn = r.b[0].size();
        if (r.op == Op::Multiply) {
            if (n != bm)
                return Refuse("A has " + std::to_string(n) + " columns and B has " + std::to_string(bm) + " rows: A × B needs them equal.");
            result = LinearAlgebra::Multiply(r.a, r.b);
            a.sentence = "A × B is " + Size(m, bn) + ": each cell is a row of A times a column of B.";
        } else {
            if (m != bm || n != bn)
                return Refuse("A is " + Size(m, n) + " and B is " + Size(bm, bn) + ": " + (r.op == Op::Add ? "A + B" : "A - B") +
                              " needs the same size.");
            result = r.op == Op::Add ? LinearAlgebra::Add(r.a, r.b) : LinearAlgebra::Subtract(r.a, r.b);
            a.sentence = std::string(r.op == Op::Add ? "A + B" : "A - B") + ", cell by cell (both " + Size(m, n) + ").";
        }
    }
    if (!result.success) return Refuse(result.error_message);
    a.tabs.push_back(MatrixTable("Result (" + Size(result.matrix) + ")", result.matrix, "c"));
    return a;
}

Answer RunDescribe(const Request& r, const Stopper& stopped) {
    Answer a;
    const size_t m = r.a.size(), n = r.a[0].size();
    const bool square = m == n;
    const double scale = Scale(r.a);
    std::string error;
    if (r.op == Op::Transpose) {
        const MatrixResult t = LinearAlgebra::Transpose(r.a);
        if (!t.success) return Refuse(t.error_message);
        a.sentence = "A^T is " + Size(n, m) + ": the rows of A become its columns.";
        a.tabs.push_back(MatrixTable("A^T (" + Size(n, m) + ")", t.matrix, "c"));
        return a;
    }
    if (r.op == Op::Norm) {
        const ScalarResult f = LinearAlgebra::FrobeniusNorm(r.a);
        if (!f.success) return Refuse(f.error_message);
        a.sentence = "‖A‖ = " + Number(f.value) + ": the root of the sum of the squares of its cells (Frobenius).";
        a.figures.push_back({"‖A‖", Number(f.value)});
        return a;
    }
    const int rank = RankOf(r.a, error);
    if (!error.empty()) return Refuse(error);
    if (stopped()) return Refuse("Cancelled.");
    const int full = static_cast<int>(std::min(m, n));
    if (r.op == Op::Rank) {
        a.sentence = "A (" + Size(m, n) + ") has rank " + std::to_string(rank) + " of " + std::to_string(full) + ": " +
                     (rank == full ? "its " + std::string(m >= n ? "columns" : "rows") + " are independent."
                                   : std::to_string(full - rank) + " of its " + (m >= n ? "columns are mixes" : "rows are mixes") +
                                         " of the others.");
        a.figures.push_back({"rank", std::to_string(rank) + " of " + std::to_string(full)});
        return a;
    }
    if (r.op == Op::Condition) {
        const ScalarResult c = LinearAlgebra::ConditionNumber(r.a);
        const double value = rank < full ? std::numeric_limits<double>::infinity() : c.value;
        if (!c.success && rank == full) return Refuse(c.error_message);
        a.sentence = ConditionWords(value);
        a.figures.push_back({"condition", std::isfinite(value) ? Plain(value) : "∞"});
        a.figures.push_back({"rank", std::to_string(rank) + " of " + std::to_string(full)});
        return a;
    }
    if (!square) return Refuse(std::string(r.op == Op::Inverse ? "Inverse" : "Determinant") + " needs a square A; A is " + Size(m, n) + ".",
                               r.op == Op::Inverse ? std::vector<std::string>{"least_squares"} : std::vector<std::string>{});
    const ScalarResult trace = LinearAlgebra::Trace(r.a);
    if (r.op == Op::Determinant) {
        if (rank < full) {
            a.sentence = "det A = 0: A is singular (rank " + std::to_string(rank) + " of " + std::to_string(full) + ").";
            a.figures.push_back({"det", "0"});
        } else {
            const ScalarResult d = LinearAlgebra::Determinant(r.a);
            if (!d.success) return Refuse(d.error_message);
            a.sentence = "det A = " + Number(d.value) + (d.value < 0 ? ": A flips orientation and scales volume by " + Number(-d.value) + "."
                                                                     : ": A scales volume by " + Number(d.value) + ".");
            a.figures.push_back({"det", Number(d.value)});
        }
        if (trace.success) a.figures.push_back({"trace", Number(trace.value, scale)});
        return a;
    }
    // Inverse
    if (rank < full)
        return Refuse("A is singular (rank " + std::to_string(rank) + " of " + std::to_string(full) +
                          "): it has no inverse. Least squares solves A x = b without it.",
                      {"least_squares"});
    const MatrixResult inv = LinearAlgebra::Inverse(r.a);
    if (!inv.success) return Refuse(inv.error_message);
    if (stopped()) return Refuse("Cancelled.");
    const ScalarResult c = LinearAlgebra::ConditionNumber(r.a);
    a.sentence = "A⁻¹ (" + Size(m, n) + "): A A⁻¹ is the identity. " + (c.success ? ConditionWords(c.value) : "");
    a.tabs.push_back(MatrixTable("A⁻¹ (" + Size(m, n) + ")", inv.matrix, "c"));
    if (c.success) a.figures.push_back({"condition", Plain(c.value)});
    const MatrixResult product = LinearAlgebra::Multiply(r.a, inv.matrix);
    const MatrixResult identity = LinearAlgebra::Identity(static_cast<int>(n));
    if (product.success && identity.success) {
        a.verify = "A A⁻¹ gives the identity: largest difference " + Short(MaxDifference(product.matrix, identity.matrix));
        a.verified = true;
    }
    return a;
}

// ---- Solve

Answer RunSolve(const Request& r, const Stopper& stopped) {
    Answer a;
    const size_t m = r.a.size(), n = r.a[0].size();
    if (const std::string bad = Malformed(r.b, "b"); !bad.empty()) return Refuse(bad);
    if (r.b.size() != m) return Refuse("A has " + std::to_string(m) + " rows and b has " + std::to_string(r.b.size()) + ": they must match.");
    std::string error;
    const int rank = RankOf(r.a, error);
    if (!error.empty()) return Refuse(error);
    if (stopped()) return Refuse("Cancelled.");
    const int full = static_cast<int>(std::min(m, n));
    MatrixResult x;
    if (r.op == Op::Solve) {
        if (m != n) return Refuse("A x = b needs a square A; A is " + Size(m, n) + ". Least squares takes any A.", {"least_squares"});
        if (rank < full)
            return Refuse("A is singular (rank " + std::to_string(rank) + " of " + std::to_string(full) +
                              "), so A x = b has no single solution. Least squares gives the x closest to it.",
                          {"least_squares"});
        x = LinearAlgebra::Solve(r.a, r.b);
    } else {
        x = LinearAlgebra::LeastSquares(r.a, r.b);
    }
    if (!x.success) return Refuse(x.error_message);
    if (stopped()) return Refuse("Cancelled.");
    const MatrixResult ax = LinearAlgebra::Multiply(r.a, x.matrix);
    double miss = 0;
    if (ax.success)
        for (size_t i = 0; i < m; ++i)
            for (size_t j = 0; j < r.b[i].size(); ++j) miss += (ax.matrix[i][j] - r.b[i][j]) * (ax.matrix[i][j] - r.b[i][j]);
    miss = std::sqrt(miss);
    const double scale = Scale(x.matrix);
    std::string xs;
    if (x.matrix[0].size() == 1) {
        for (const auto& row : x.matrix) xs += (xs.empty() ? "" : ", ") + Number(row[0], scale);
        xs = "x = (" + xs + ")";
    } else {
        xs = "x (" + Size(x.matrix) + ")";
    }
    Table t = MatrixTable("x (" + Size(x.matrix) + ")", x.matrix, x.matrix[0].size() == 1 ? "x" : "b", "x");
    if (HasNames(r, x.matrix.size())) t.row_names = r.a_columns;
    if (x.matrix[0].size() == 1) t.columns = {"x"};
    a.tabs.push_back(t);
    if (r.op == Op::Solve) {
        const ScalarResult c = LinearAlgebra::ConditionNumber(r.a);
        a.sentence = "A x = b has one solution: " + xs + ". " + (c.success ? ConditionWords(c.value) : "");
        if (c.success) a.figures.push_back({"condition", Plain(c.value)});
        a.verify = "A x gives b back: largest difference " + Short(ax.success ? MaxDifference(ax.matrix, r.b) : NAN);
        a.verified = ax.success;
    } else {
        a.sentence = "Least squares: " + xs + " brings A x closest to b" +
                     (rank < static_cast<int>(n) ? " (A has rank " + std::to_string(rank) + " of " + std::to_string(n) +
                                                       ", so many x do as well: this is the shortest)"
                                                 : "") +
                     ". A x misses b by " + Number(miss) + ".";
        a.figures.push_back({"rank", std::to_string(rank) + " of " + std::to_string(full)});
        a.figures.push_back({"‖A x - b‖", Number(miss)});
        a.verify = miss <= 1e-9 * std::max(1.0, Scale(r.b)) ? "A x gives b back: b is reachable exactly"
                                                            : "A x misses b by " + Number(miss) + ": the closest A x can get";
        a.verified = ax.success;
    }
    return a;
}

// ---- Factor

Answer RunEigen(const Request& r, const Stopper& stopped) {
    Answer a;
    const size_t n = r.a.size();
    if (r.a[0].size() != n) return Refuse("Eigen needs a square A; A is " + Size(r.a) + ". SVD takes any A.", {"svd"});
    bool symmetric = false;
    try {
        symmetric = LinearAlgebra::IsSymmetric(r.a);
    } catch (const std::exception& e) {
        return Refuse(e.what());
    }
    if (!symmetric && n > 2)
        return Refuse("Eigen takes symmetric matrices of any size, and others up to 2 × 2: the device has no general eigensolver. "
                      "Symmetrize A, or use SVD, which takes any matrix.",
                      {"svd"});
    const EigenResult e = LinearAlgebra::Eigen(r.a);
    if (!e.success) return Refuse(e.error_message);
    if (stopped()) return Refuse("Cancelled.");
    std::vector<size_t> order(e.eigenvalues.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](size_t x, size_t y) { return e.eigenvalues[x].real() > e.eigenvalues[y].real(); });
    const double scale = Scale(r.a);
    bool complex = false;
    std::vector<double> real;
    for (const size_t i : order) {
        real.push_back(e.eigenvalues[i].real());
        complex = complex || std::fabs(e.eigenvalues[i].imag()) > 1e-12 * std::max(1.0, scale);
    }
    Table values;
    values.title = "Eigenvalues";
    values.columns = {"real", "imaginary"};
    values.matrix = false;
    for (size_t k = 0; k < order.size(); ++k) {
        values.row_names.push_back("λ" + std::to_string(k + 1));
        values.values.push_back({e.eigenvalues[order[k]].real(), e.eigenvalues[order[k]].imag()});
    }
    values.note = "Largest first.";
    const double tol = 1e-10 * std::max(1.0, scale);
    const bool all_positive = std::all_of(real.begin(), real.end(), [&](double v) { return v > tol; });
    const bool all_nonnegative = std::all_of(real.begin(), real.end(), [&](double v) { return v > -tol; });
    const bool all_negative = std::all_of(real.begin(), real.end(), [&](double v) { return v < -tol; });
    double sum = 0, product = 1;
    for (const double v : real) sum += v, product *= v;
    if (complex) {
        a.sentence = "A is not symmetric and its eigenvalues are a complex pair, " + Number(real[0], scale) + " ± " +
                     Number(std::fabs(e.eigenvalues[order[0]].imag()), scale) + "i: A turns every direction as it stretches it.";
    } else {
        a.sentence = std::string(symmetric ? "A is symmetric, so its eigenvalues are real: " : "A's eigenvalues are real: ") + Join(real, scale) +
                     ". " +
                     (all_positive       ? "All are positive: A is positive definite. "
                      : all_nonnegative  ? "None is negative but one is 0: A is positive semi-definite and singular. "
                      : all_negative     ? "All are negative: A is negative definite. "
                                         : "Some are positive and some negative: A is indefinite. ") +
                     "Their product is det A = " + Number(product, scale) + ", their sum the trace, " + Number(sum, scale) + ".";
        for (size_t k = 0; k < real.size(); ++k) {
            const double positive = std::accumulate(real.begin(), real.end(), 0.0, [](double s, double v) { return s + std::max(v, 0.0); });
            a.bars.push_back({"λ" + std::to_string(k + 1), real[k], all_nonnegative && positive > 0 ? real[k] / positive : NAN});
        }
    }
    a.figures.push_back({"λ", complex ? Number(real[0], scale) + " ± " + Number(std::fabs(e.eigenvalues[order[0]].imag()), scale) + "i"
                                      : Join(real, scale)});
    a.figures.push_back({"det", Number(product, scale)});
    a.figures.push_back({"trace", Number(sum, scale)});
    a.tabs.push_back(values);
    if (!complex) {
        Matrix v(n, std::vector<double>(n));
        for (size_t k = 0; k < n; ++k)
            for (size_t i = 0; i < n; ++i) v[i][k] = e.eigenvectors[i][order[k]].real();
        Table vectors = MatrixTable("Eigenvectors", v, "v");
        if (HasNames(r, n)) vectors.row_names = r.a_columns;
        vectors.note = "One column per eigenvalue, of length 1; a vector and its negative are the same direction.";
        a.tabs.push_back(vectors);
        const MatrixResult av = LinearAlgebra::Multiply(r.a, v);
        if (av.success) {
            double worst = 0;
            for (size_t i = 0; i < n; ++i)
                for (size_t k = 0; k < n; ++k) worst = std::max(worst, std::fabs(av.matrix[i][k] - real[k] * v[i][k]));
            a.verify = "A v = λ v for each pair: largest difference " + Short(worst);
            a.verified = true;
        }
    }
    return a;
}

Answer RunSVD(const Request& r, const Stopper& stopped) {
    Answer a;
    const size_t m = r.a.size(), n = r.a[0].size();
    const SVDResult s = LinearAlgebra::SVD(r.a, false);
    if (!s.success) return Refuse(s.error_message);
    if (stopped()) return Refuse("Cancelled.");
    std::string error;
    const int rank = RankOf(r.a, error);
    if (!error.empty()) return Refuse(error);
    const int full = static_cast<int>(std::min(m, n));
    double energy = 0;
    for (const double v : s.S) energy += v * v;
    const double top = s.S.empty() ? 0 : s.S[0];
    for (size_t k = 0; k < s.S.size(); ++k)
        a.bars.push_back({"σ" + std::to_string(k + 1), s.S[k], energy > 0 ? s.S[k] * s.S[k] / energy : NAN});
    std::string sentence = "A (" + Size(m, n) + ") has rank " + std::to_string(rank) + " of " + std::to_string(full);
    if (rank < full)
        sentence += ": " + std::to_string(full - rank) + " of its singular values " + (full - rank == 1 ? "is" : "are") + " 0, so " +
                    (full - rank == 1 ? "one of its " : "some of its ") + (m >= n ? "columns are" : "rows are") + " mixes of the others.";
    else
        sentence += ".";
    if (energy > 0 && !a.bars.empty()) {
        size_t needed = 0;
        double kept = 0;
        while (needed < a.bars.size() && kept < 0.95) kept += a.bars[needed++].share;
        sentence += needed == 1 ? " One direction holds " + Percent(a.bars[0].share) + " of its spread."
                                : " " + std::to_string(needed) + " directions hold " + Percent(kept) + " of its spread.";
    }
    a.sentence = sentence;
    a.figures.push_back({"σ", Join(s.S, top)});
    a.figures.push_back({"rank", std::to_string(rank) + " of " + std::to_string(full)});
    a.figures.push_back({"condition", rank < full ? "∞ (singular)" : (s.S.back() > 0 ? Plain(top / s.S.back()) : "∞")});
    Table values;
    values.title = "Singular values";
    values.columns = {"σ", "share of σ²", "cumulative"};
    values.matrix = false;
    double cumulative = 0;
    for (size_t k = 0; k < s.S.size(); ++k) {
        cumulative += a.bars[k].share;
        values.row_names.push_back(std::to_string(k + 1));
        values.values.push_back({s.S[k], a.bars[k].share, cumulative});
    }
    values.note = "share of σ²: how much of the matrix each direction carries (σ² over the sum of σ²).";
    a.tabs.push_back(values);
    Table u = MatrixTable("U (" + Size(s.U) + ")", s.U, "u");
    u.note = "Reduced: as many columns as singular values.";
    a.tabs.push_back(u);
    Table vt = MatrixTable("V^T (" + Size(s.Vt) + ")", s.Vt, "c", "v");
    if (HasNames(r, n)) vt.columns = r.a_columns;
    a.tabs.push_back(vt);
    const int keep = std::clamp(r.low_rank, 1, std::max(1, full));
    const MatrixResult low = LinearAlgebra::LowRankApproximation(r.a, keep);
    if (low.success) {
        double off = 0, all = 0;
        for (size_t i = 0; i < m; ++i)
            for (size_t j = 0; j < n; ++j) {
                off += (r.a[i][j] - low.matrix[i][j]) * (r.a[i][j] - low.matrix[i][j]);
                all += r.a[i][j] * r.a[i][j];
            }
        Table lr = MatrixTable("Low rank (k = " + std::to_string(keep) + ")", low.matrix, "c");
        lr.note = "A kept to its first " + std::to_string(keep) + (keep == 1 ? " direction" : " directions") + ": off by " + Number(std::sqrt(off)) +
                  " (Frobenius), " + Percent(all > 0 ? std::sqrt(off / all) : 0) + " of ‖A‖.";
        a.tabs.push_back(lr);
    }
    // U S V^T gives A back.
    Matrix us = s.U;
    for (auto& row : us)
        for (size_t k = 0; k < row.size() && k < s.S.size(); ++k) row[k] *= s.S[k];
    const MatrixResult back = LinearAlgebra::Multiply(us, s.Vt);
    if (back.success) {
        a.verify = "U S V^T gives A back: largest difference " + Short(MaxDifference(back.matrix, r.a));
        a.verified = true;
    }
    return a;
}

Answer RunQR(const Request& r, const Stopper& stopped) {
    Answer a;
    const size_t m = r.a.size(), n = r.a[0].size();
    const QRResult q = LinearAlgebra::QR(r.a);
    if (!q.success) return Refuse(q.error_message);
    if (stopped()) return Refuse("Cancelled.");
    std::string error;
    const int rank = RankOf(r.a, error);
    if (!error.empty()) return Refuse(error);
    const int full = static_cast<int>(std::min(m, n));
    const MatrixResult qtq = LinearAlgebra::Multiply(LinearAlgebra::Transpose(q.Q).matrix, q.Q);
    const MatrixResult identity = LinearAlgebra::Identity(static_cast<int>(q.Q.empty() ? 0 : q.Q[0].size()));
    const double orthonormal = qtq.success && identity.success ? MaxDifference(qtq.matrix, identity.matrix) : NAN;
    a.sentence = "A = Q R: Q is " + Size(q.Q) + " with orthonormal columns, R is " + Size(q.R) + " upper triangular. " +
                 (rank == full ? "A has full rank (" + std::to_string(rank) + " of " + std::to_string(full) + ")."
                               : "A has rank " + std::to_string(rank) + " of " + std::to_string(full) + ": R's diagonal has a 0.");
    a.figures.push_back({"Q^T Q = I to", Short(orthonormal)});
    a.figures.push_back({"rank", std::to_string(rank) + " of " + std::to_string(full)});
    Table tq = MatrixTable("Q (" + Size(q.Q) + ")", q.Q, "q");
    tq.note = "Reduced Q: as many columns as R has rows.";
    a.tabs.push_back(tq);
    Table tr = MatrixTable("R (" + Size(q.R) + ")", q.R, "c");
    tr.note = "Upper triangular.";
    a.tabs.push_back(tr);
    const MatrixResult back = LinearAlgebra::Multiply(q.Q, q.R);
    if (back.success) {
        a.verify = "Q R gives A back: largest difference " + Short(MaxDifference(back.matrix, r.a));
        a.verified = true;
    }
    return a;
}

Answer RunCholesky(const Request& r, const Stopper& stopped) {
    Answer a;
    const size_t n = r.a.size();
    if (r.a[0].size() != n) return Refuse("Cholesky needs a square A; A is " + Size(r.a) + ".", {"qr"});
    bool symmetric = false;
    try {
        symmetric = LinearAlgebra::IsSymmetric(r.a);
    } catch (const std::exception& e) {
        return Refuse(e.what());
    }
    if (!symmetric) return Refuse("A is not symmetric, so it has no Cholesky factor (L L^T is always symmetric). Symmetrize A, or use LU.", {"lu"});
    const CholeskyResult c = LinearAlgebra::Cholesky(r.a);
    if (stopped()) return Refuse("Cancelled.");
    if (!c.success) {
        const EigenResult e = LinearAlgebra::Eigen(r.a);
        std::vector<double> values;
        if (e.success)
            for (const auto& v : e.eigenvalues) values.push_back(v.real());
        std::sort(values.rbegin(), values.rend());
        const double scale = Scale(r.a);
        std::string why = "A is symmetric but not positive definite";
        if (!values.empty()) {
            const bool zero = values.back() > -1e-10 * std::max(1.0, scale);
            why += ": its eigenvalues are " + Join(values, scale) + (zero ? ", and the 0 leaves it" : ", and the negative one leaves it");
            why += " without a Cholesky factor.";
        } else {
            why += ", so it has no Cholesky factor.";
        }
        return Refuse(why + " Eigen shows that direction; LU factors it anyway.", {"eigen", "lu"});
    }
    a.sentence = "A is symmetric and positive definite, so A = L L^T with L lower triangular and a positive diagonal.";
    a.figures.push_back({"symmetric", "yes"});
    a.figures.push_back({"positive definite", "yes"});
    Table l = MatrixTable("L (" + Size(c.L) + ")", c.L, "c");
    l.note = "Lower triangular.";
    a.tabs.push_back(l);
    const MatrixResult back = LinearAlgebra::Multiply(c.L, LinearAlgebra::Transpose(c.L).matrix);
    if (back.success) {
        a.verify = "L L^T gives A back: largest difference " + Short(MaxDifference(back.matrix, r.a));
        a.verified = true;
    }
    return a;
}

Answer RunLU(const Request& r, const Stopper& stopped) {
    Answer a;
    const size_t n = r.a.size();
    if (r.a[0].size() != n) return Refuse("LU needs a square A; A is " + Size(r.a) + ". QR takes any A.", {"qr"});
    const LUResult lu = LinearAlgebra::LU(r.a);
    if (!lu.success) return Refuse(lu.error_message);
    if (stopped()) return Refuse("Cancelled.");
    Matrix pa;
    std::string order;
    for (const int p : lu.P) {
        if (p < 0 || static_cast<size_t>(p) >= n) return Refuse("The device returned a row order out of range.");
        pa.push_back(r.a[static_cast<size_t>(p)]);
        order += (order.empty() ? "" : ", ") + std::to_string(p + 1);
    }
    bool moved = false;
    for (size_t i = 0; i < lu.P.size(); ++i) moved = moved || lu.P[i] != static_cast<int>(i);
    a.sentence = std::string(moved ? "With its rows in the order " + order + ", A = L U" : "A = L U") +
                 ": L lower triangular with 1s on its diagonal, U upper triangular. " +
                 (moved ? "The rows are reordered so each pivot is the largest left in its column." : "No row had to move.");
    a.figures.push_back({"row order", order});
    a.tabs.push_back(MatrixTable("L (" + Size(lu.L) + ")", lu.L, "c"));
    a.tabs.push_back(MatrixTable("U (" + Size(lu.U) + ")", lu.U, "c"));
    const MatrixResult back = LinearAlgebra::Multiply(lu.L, lu.U);
    if (back.success) {
        a.verify = std::string("L U gives A") + (moved ? " with its rows reordered" : "") + ": largest difference " +
                   Short(MaxDifference(back.matrix, pa));
        a.verified = true;
    }
    return a;
}

}  // namespace

const std::vector<OpInfo>& Ops() { return kOps; }

const OpInfo& InfoOf(Op op) {
    for (const auto& info : kOps)
        if (info.op == op) return info;
    return kOps.front();
}

const OpInfo* FindOp(const std::string& id) {
    for (const auto& info : kOps)
        if (id == info.id) return &info;
    return nullptr;
}

std::vector<std::string> Groups() { return {"Combine", "Describe", "Solve", "Factor"}; }

Answer Run(const Request& request, const std::function<bool()>& stop) {
    const Stopper stopped{stop};
    if (const std::string bad = Malformed(request.a, "A"); !bad.empty()) return Refuse(bad);
    try {
        switch (request.op) {
            case Op::Add:
            case Op::Subtract:
            case Op::Multiply:
            case Op::Scale: return RunCombine(request);
            case Op::Transpose:
            case Op::Inverse:
            case Op::Determinant:
            case Op::Rank:
            case Op::Norm:
            case Op::Condition: return RunDescribe(request, stopped);
            case Op::Solve:
            case Op::LeastSquares: return RunSolve(request, stopped);
            case Op::Eigen: return RunEigen(request, stopped);
            case Op::SVD: return RunSVD(request, stopped);
            case Op::QR: return RunQR(request, stopped);
            case Op::Cholesky: return RunCholesky(request, stopped);
            case Op::LU: return RunLU(request, stopped);
        }
    } catch (const std::exception& e) {
        return Refuse(std::string("The device could not compute it: ") + e.what());
    }
    return Refuse("Unknown operation.");
}

// ---- A matrix from a table

namespace {

bool IsNumber(const arrow::DataType& type) {
    switch (type.id()) {
        case arrow::Type::INT8:
        case arrow::Type::INT16:
        case arrow::Type::INT32:
        case arrow::Type::INT64:
        case arrow::Type::UINT8:
        case arrow::Type::UINT16:
        case arrow::Type::UINT32:
        case arrow::Type::UINT64:
        case arrow::Type::HALF_FLOAT:
        case arrow::Type::FLOAT:
        case arrow::Type::DOUBLE:
        case arrow::Type::DECIMAL128:
        case arrow::Type::DECIMAL256: return true;
        default: return false;
    }
}

std::string Rows(size_t n) { return std::to_string(n) + (n == 1 ? " row" : " rows"); }

}  // namespace

std::vector<std::string> NumberColumns(const std::string& table, const QueryRunner& run, std::string& error) {
    std::vector<std::string> out;
    QueryRequest q;
    q.sql = "SELECT * FROM " + SessionQueryEngine::QuoteIdentifier(table) + " LIMIT 0";
    q.label = "Linear Algebra: " + table;
    const QueryResult result = run(q);
    if (!result.ok || !result.table) {
        error = result.error.empty() ? "Cannot read " + table + "." : result.error;
        return out;
    }
    for (const auto& field : result.table->schema()->fields())
        if (IsNumber(*field->type())) out.push_back(field->name());
    return out;
}

TableMatrix ReadMatrix(const TableChoice& choice, const QueryRunner& run) {
    TableMatrix out;
    if (choice.table.empty()) return out.error = "Pick a table.", out;
    if (choice.columns.empty()) return out.error = "Pick the number columns of " + choice.table + ".", out;
    const size_t k = choice.columns.size();
    if (choice.matrix == FromTable::Rows) {
        // Counted before the read: a table too large to be a matrix is not loaded.
        QueryRequest count;
        count.sql = "SELECT COUNT(*) AS n FROM " + SessionQueryEngine::QuoteIdentifier(choice.table);
        count.label = "Linear Algebra: count";
        const QueryResult counted = run(count);
        if (!counted.ok || !counted.table) return out.error = counted.error.empty() ? "Cannot read " + choice.table + "." : counted.error, out;
        const auto rows = explore_read::Numbers(*counted.table, "n");
        const size_t n = rows.empty() || !rows[0] ? 0 : static_cast<size_t>(*rows[0]);
        if (n * k > kMaxTableValues)
            return out.error = choice.table + " has " + Rows(n) + " × " + std::to_string(k) + " columns: more than " +
                               std::to_string(kMaxTableValues) +
                               " values to read as a matrix. Take its correlation or covariance, or sample the rows in Prepare.",
                   out;
    }
    std::vector<QueryParam> params;
    std::vector<std::string> exprs;
    for (size_t j = 0; j < choice.columns.size(); ++j)
        exprs.push_back(explore_read::NumberExpr(choice.columns[j], {}, params, "c" + std::to_string(j)));
    const explore_read::Read read = explore_read::Select(choice.table, exprs, params, run, "Linear Algebra: read");
    if (!read.table) return out.error = read.error.empty() ? "Cannot read " + choice.table + "." : read.error, out;
    const size_t n = static_cast<size_t>(read.table->num_rows());
    std::vector<std::vector<std::optional<double>>> columns;
    for (size_t j = 0; j < k; ++j) columns.push_back(explore_read::Numbers(*read.table, "c" + std::to_string(j)));
    Matrix rows;
    rows.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        std::vector<double> row(k);
        bool missing = false;
        for (size_t j = 0; j < k && !missing; ++j) {
            if (!columns[j][i]) missing = true;
            else row[j] = *columns[j][i];
        }
        if (missing) ++out.dropped;
        else rows.push_back(std::move(row));
    }
    out.rows = n;
    out.names = choice.columns;
    const std::string left = out.dropped ? Rows(out.dropped) + " with a missing value left out" : "none left out";
    if (rows.empty()) return out.error = "Every row of " + choice.table + " has a missing value in the chosen columns.", out;
    if (choice.matrix == FromTable::Rows) {
        out.m = std::move(rows);
        out.note = "The rows of " + choice.table + ": " + Size(out.m) + " (" + left + ").";
        return out;
    }
    if (rows.size() < 2) return out.error = "A correlation or covariance needs 2 rows or more.", out;
    // On the device: the means, then the cross products about them / (n - 1).
    const double count = static_cast<double>(rows.size());
    const ColumnSummaryResult summary = LinearAlgebra::ColumnSummary(rows);
    if (!summary.success) return out.error = summary.error_message, out;
    const MatrixResult covariance = LinearAlgebra::ScalarMultiply(summary.cross, 1.0 / (count - 1.0));
    out.m = covariance.matrix;
    if (choice.matrix == FromTable::Correlation) {
        for (size_t j = 0; j < k; ++j)
            if (!(out.m[j][j] > 0)) return out.error = choice.columns[j] + " has one value in every row: it has no correlation.", out;
        Matrix c(k, std::vector<double>(k));
        for (size_t i = 0; i < k; ++i)
            for (size_t j = 0; j < k; ++j) c[i][j] = i == j ? 1.0 : out.m[i][j] / std::sqrt(out.m[i][i] * out.m[j][j]);
        out.m = std::move(c);
    }
    out.note = std::string(choice.matrix == FromTable::Correlation ? "The correlation" : "The covariance") + " of " + std::to_string(k) +
               " columns of " + choice.table + ", over " + Rows(rows.size()) + " (" + left + ").";
    return out;
}

TableAnswer RunWithTables(Request request, const Sources& sources, const QueryRunner& run, const std::function<bool()>& stop) {
    TableAnswer out;
    const auto stopped = [&] { return stop && stop(); };
    if (sources.a) {
        out.a = ReadMatrix(*sources.a, run);
        if (out.a.error == "Cancelled.") return out.error = "Cancelled.", out;
        if (!out.a.ok()) return out.answer = Refuse("A: " + out.a.error), out;
        request.a = out.a.m;
        request.a_columns = out.a.names;
    }
    if (stopped()) return out.error = "Cancelled.", out;
    if (sources.b) {
        out.b = ReadMatrix(*sources.b, run);
        if (out.b.error == "Cancelled.") return out.error = "Cancelled.", out;
        if (!out.b.ok()) return out.answer = Refuse("b: " + out.b.error), out;
        request.b = out.b.m;
        const bool same_table = sources.a && sources.a->table == sources.b->table;
        if (!same_table && (out.a.dropped || out.b.dropped))
            return out.answer = Refuse("A and b come from two tables and " + std::string(out.a.dropped ? "A" : "b") +
                                       " has rows with a missing value: the rows cannot be lined up with one side left out. Fill or remove them in Prepare first."),
                   out;
        if (same_table && (out.a.dropped || out.b.dropped)) {
            // One table: read both sides together, so a row missing on either side is left out of both.
            TableChoice both = *sources.a;
            both.columns.insert(both.columns.end(), sources.b->columns.begin(), sources.b->columns.end());
            const TableMatrix joined = ReadMatrix(both, run);
            if (!joined.ok()) return out.answer = Refuse("A and b read together (to leave out the rows missing on either side): " + joined.error), out;
            const size_t ka = sources.a->columns.size();
            request.a.clear();
            request.b.clear();
            for (const auto& row : joined.m) {
                request.a.emplace_back(row.begin(), row.begin() + static_cast<std::ptrdiff_t>(ka));
                request.b.emplace_back(row.begin() + static_cast<std::ptrdiff_t>(ka), row.end());
            }
            out.a.note = out.b.note = "The rows of " + sources.a->table + ": " + Rows(joined.m.size()) + " (" + Rows(joined.dropped) +
                                       " with a missing value in A or b left out).";
        }
    }
    if (stopped()) return out.error = "Cancelled.", out;
    out.answer = Run(request, stop);
    if (out.answer.error == "Cancelled.") out.error = "Cancelled.";
    return out;
}

bool ParseMatrix(const std::string& text, Matrix& out, std::string& error) {
    out.clear();
    std::vector<std::vector<std::string>> cells;
    std::istringstream lines(text);
    std::string line;
    while (std::getline(lines, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        std::vector<std::string> row;
        std::string cell;
        bool any = false;
        for (const char ch : line) {
            if (ch == '\t' || ch == ',' || ch == ';' || ch == ' ') {
                if (!cell.empty()) row.push_back(cell);
                cell.clear();
            } else {
                cell += ch;
                any = true;
            }
        }
        if (!cell.empty()) row.push_back(cell);
        if (any && !row.empty()) cells.push_back(row);
    }
    if (cells.empty()) return error = "Nothing to read: paste rows of numbers.", false;
    const auto number = [](const std::string& s, double& v) {
        char* end = nullptr;
        v = std::strtod(s.c_str(), &end);
        return end && *end == '\0' && std::isfinite(v);
    };
    double v = 0;
    // A first line of names only (a header), then a first column of names.
    if (std::none_of(cells[0].begin(), cells[0].end(), [&](const std::string& s) { return number(s, v); }) && cells.size() > 1)
        cells.erase(cells.begin());
    if (std::all_of(cells.begin(), cells.end(), [&](const auto& row) { return row.size() > 1 && !number(row[0], v); }))
        for (auto& row : cells) row.erase(row.begin());
    for (size_t i = 0; i < cells.size(); ++i) {
        if (cells[i].size() != cells[0].size())
            return error = "Row " + std::to_string(i + 1) + " has " + std::to_string(cells[i].size()) + " values, row 1 has " +
                           std::to_string(cells[0].size()) + ".",
                   out.clear(), false;
        std::vector<double> row;
        for (const std::string& s : cells[i]) {
            if (!number(s, v)) return error = "\"" + s + "\" in row " + std::to_string(i + 1) + " is not a number.", out.clear(), false;
            row.push_back(v);
        }
        out.push_back(row);
    }
    if (out.size() > static_cast<size_t>(kMaxTyped) || out[0].size() > static_cast<size_t>(kMaxTyped))
        return error = "The pasted matrix is " + Size(out) + "; a typed one goes up to " + Size(kMaxTyped, kMaxTyped) +
                       ". From a table takes larger ones.",
               out.clear(), false;
    return true;
}

std::string Number(double value, double scale) {
    if (!std::isfinite(value)) return std::isnan(value) ? "NaN" : (value > 0 ? "∞" : "-∞");
    if (value == 0.0 || (scale > 0 && std::fabs(value) <= 1e-10 * scale)) return "0";
    char buffer[48];
    const double magnitude = std::fabs(value);
    if (magnitude >= 1e6 || magnitude < 1e-4)
        std::snprintf(buffer, sizeof(buffer), "%.4e", value);
    else
        std::snprintf(buffer, sizeof(buffer), "%.4f", value);
    return buffer;
}

namespace {

std::string Exact(double v) {
    char buffer[48];
    std::snprintf(buffer, sizeof(buffer), "%.12g", v);
    return buffer;
}

std::string Rows(const Table& t, char separator, bool quote) {
    const auto field = [&](const std::string& s) {
        if (!quote || s.find_first_of(",\"\n") == std::string::npos) return s;
        std::string q = "\"";
        for (const char c : s) q += c == '"' ? std::string("\"\"") : std::string(1, c);
        return q + "\"";
    };
    std::string out;
    out += field("");
    for (const auto& c : t.columns) out += separator + field(c);
    out += '\n';
    for (size_t i = 0; i < t.values.size(); ++i) {
        out += field(i < t.row_names.size() ? t.row_names[i] : std::to_string(i + 1));
        for (const double v : t.values[i]) out += separator + Exact(v);
        out += '\n';
    }
    return out;
}

}  // namespace

std::string CopyText(const Answer& answer, size_t tab) {
    std::string out = answer.ok() ? answer.sentence : answer.error;
    if (answer.ok() && tab < answer.tabs.size()) out += "\n\n" + answer.tabs[tab].title + "\n" + Rows(answer.tabs[tab], '\t', false);
    return out;
}

std::string Csv(const Table& table) { return Rows(table, ',', true); }

std::vector<HandOff> HandOffs(Op op, const Sources& sources, int keep, const std::vector<std::string>& b_number_columns) {
    std::vector<HandOff> out;
    if (!sources.a || sources.a->columns.empty()) return out;
    // A step keeps the input's columns and adds its own beside them (as Explore's
    // steps do), so what reads the table after it still finds its columns.
    const auto beside = [](std::map<std::string, std::string> p) {
        p["keep_columns"] = "true";
        return p;
    };
    const TableChoice& a = *sources.a;
    std::string features;
    for (const auto& c : a.columns) features += (features.empty() ? "" : ",") + c;
    const std::string components = std::to_string(std::clamp<int>(keep, 1, static_cast<int>(a.columns.size())));
    if (op == Op::SVD && a.matrix == FromTable::Rows) {
        out.push_back({"Add as a step: Truncated SVD", "Fits the top " + components + " directions on the Train rows of every run and maps each row onto them",
                       "TruncatedSVD", "Truncated SVD (" + components + ")", beside({{"feature_cols", features}, {"n_components", components}})});
    }
    if ((op == Op::SVD || op == Op::Eigen) && a.matrix != FromTable::Rows) {
        const bool scale = a.matrix == FromTable::Correlation;
        out.push_back({"Add as a step: PCA", std::string("Fits the principal directions of the ") + (scale ? "correlation" : "covariance") +
                           " on the Train rows of every run and maps each row onto the top " + components,
                       "PCANode", "PCA (" + components + ")",
                       beside({{"feature_cols", features}, {"n_components", components}, {"center", "true"}, {"scale", scale ? "true" : "false"}})});
        if (!scale)
            out.push_back({"Add as a step: Whitening", "Rotates and rescales the columns on the Train rows' covariance of every run: no correlation, variance 1",
                           "Whitening", "Whitening (PCA)", beside({{"feature_cols", features}, {"method", "pca"}, {"n_components", "0"}})});
    }
    if ((op == Op::Solve || op == Op::LeastSquares) && a.matrix == FromTable::Rows && sources.b && !sources.b->columns.empty()) {
        std::string b_cols;
        for (const auto& c : sources.b->columns) b_cols += (b_cols.empty() ? "" : ",") + c;
        const bool two = sources.b->table != a.table;
        out.push_back({"Keep as a node: Linear Solve", std::string("Solves for x on the Train rows of every run") + (two ? " (b's table into its b input)" : ""),
                       "LinearSolve", "Linear Solve (" + b_cols + ")",
                       {{"a_cols", features}, {"b_cols", b_cols}, {"method", op == Op::Solve ? "solve" : "least_squares"}}, false, two});
    }
    if (op == Op::Multiply && a.matrix == FromTable::Rows && sources.b && sources.b->matrix == FromTable::Rows && sources.b->table != a.table &&
        sources.b->columns == b_number_columns) {
        out.push_back({"Add as a step: Linear Projection", "Multiplies each row of every run by B (B's table into its Matrix input)",
                       "LinearProjection", "Linear Projection", beside({{"feature_cols", features}}), true, true});
    }
    return out;
}

}  // namespace cyxwiz::linalg
