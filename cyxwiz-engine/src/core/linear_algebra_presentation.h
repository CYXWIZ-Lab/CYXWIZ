#pragma once

// Tools > Linear Algebra (TOFIX129 D9): one window for the matrix operations.
// The operation list in its four groups, the answer to a request (computed
// with the backend's LinearAlgebra on the device): a sentence, its figures,
// the bars of singular values or eigenvalues, a tab per result matrix and the
// check that the factors give A back, or the reason it cannot run; and the
// matrix read from pasted text, and the result as text to copy or save.

#include <functional>
#include <string>
#include <vector>

namespace cyxwiz::linalg {

using Matrix = std::vector<std::vector<double>>;

enum class Op {
    Add, Subtract, Multiply, Scale,                         // Combine
    Transpose, Inverse, Determinant, Rank, Norm, Condition, // Describe
    Solve, LeastSquares,                                    // Solve
    Eigen, SVD, QR, Cholesky, LU,                           // Factor
};

// What the second input is.
enum class Second { None, MatrixB, VectorB, Scalar };

struct OpInfo {
    Op op;
    const char* id;      // stable ("svd")
    const char* label;   // "SVD"
    const char* group;   // "Factor"
    Second second;
    const char* hint;    // one line on what it does
};
const std::vector<OpInfo>& Ops();
const OpInfo& InfoOf(Op op);
const OpInfo* FindOp(const std::string& id);
// The groups in order: Combine, Describe, Solve, Factor.
std::vector<std::string> Groups();

constexpr int kMaxTyped = 64;  // rows and columns of a typed matrix

struct Request {
    Op op = Op::SVD;
    Matrix a, b;           // b: B, or b as a column (n x 1, or n x k)
    double k = 2.0;        // k x A
    int low_rank = 1;      // SVD: the directions kept
};

struct Figure {
    std::string name, value;
};

struct Bar {
    std::string name;   // "σ1", "λ2"
    double value = 0;   // the singular value or eigenvalue (real part)
    double share = 0;   // its share (σ² over the sum of σ²; λ over the sum of λ when all are positive), NaN when none
};

struct Table {
    std::string title;                     // "U (4 × 3)"
    std::vector<std::string> columns;      // "u1", "c1", ...
    std::vector<std::string> row_names;    // "r1", ...
    Matrix values;
    std::string note;
    bool matrix = true;                    // a matrix result (Use as A), not a list of figures
};

struct Answer {
    std::string error;                     // why it cannot run: the answer then
    std::vector<std::string> try_instead;  // operation ids the reason points to
    bool ok() const { return error.empty(); }
    std::string sentence;
    std::vector<Figure> figures;
    std::vector<Bar> bars;                 // SVD and Eigen
    std::vector<Table> tabs;
    std::string verify;                    // "Q R gives A back: largest difference 1.4e-14"
    bool verified = false;
};

// The answer to `request`; `stop` is asked between the device calls.
Answer Run(const Request& request, const std::function<bool()>& stop = {});

// A matrix from pasted text: rows on lines, values split by tabs, commas,
// semicolons or spaces; a first line or column of names is left out. Every
// row must have the same count; at most kMaxTyped x kMaxTyped.
bool ParseMatrix(const std::string& text, Matrix& out, std::string& error);

// A value as the answer shows it: 4 decimals, scientific beyond 1e6 or below
// 1e-4, and 0 for what is below 1e-10 of the scale (rounding noise).
std::string Number(double value, double scale = 1.0);

// The sentence, then the table as tab-separated rows (Copy).
std::string CopyText(const Answer& answer, size_t tab);
// The table as CSV with a header row (Export CSV).
std::string Csv(const Table& table);

}  // namespace cyxwiz::linalg
