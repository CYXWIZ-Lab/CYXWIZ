#pragma once

// Tools > Linear Algebra (TOFIX129 D9): one window for the matrix operations.
// The operation list in its four groups, the answer to a request (computed
// with the backend's LinearAlgebra on the device): a sentence, its figures,
// the bars of singular values or eigenvalues, a tab per result matrix and the
// check that the factors give A back, or the reason it cannot run; and the
// matrix read from pasted text, and the result as text to copy or save.

#include "explore_read.h"

#include <functional>
#include <map>
#include <optional>
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
    std::vector<std::string> a_columns;  // A's column names (from a table), "" when typed
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

// ---- A matrix from a table of the session (a Data Input's data, a node's result)

enum class FromTable { Rows, Correlation, Covariance };
constexpr size_t kMaxTableValues = 1000000;  // the rows read as a matrix: rows × columns at most

struct TableChoice {
    std::string table;
    std::vector<std::string> columns;   // number columns, in order
    FromTable matrix = FromTable::Rows;
};

struct TableMatrix {
    std::string error;
    bool ok() const { return error.empty(); }
    Matrix m;
    std::vector<std::string> names;     // the columns (and, for a correlation or covariance, the rows)
    size_t rows = 0;                    // rows read
    size_t dropped = 0;                 // rows left out for a missing value
    std::string note;                   // "150 rows of iris (none left out)"
};

// The number columns of `table`, in order.
std::vector<std::string> NumberColumns(const std::string& table, const QueryRunner& run, std::string& error);
// The chosen columns as a matrix: the rows (a row with a missing value left
// out and counted), or their correlation or covariance (on the device, with
// the backend's LinearAlgebra).
TableMatrix ReadMatrix(const TableChoice& choice, const QueryRunner& run);

struct Sources {
    std::optional<TableChoice> a, b;    // set: that input comes from a table
};
struct TableAnswer {
    Answer answer;
    TableMatrix a, b;                   // as read (empty when typed)
    bool ok() const { return error.empty(); }
    std::string error;                  // "Cancelled." only
};
// Reads the inputs that come from tables, then answers. A and b from two
// tables line up row by row: their row counts must match, and a missing
// value is refused there (rows cannot be left out of one side alone).
TableAnswer RunWithTables(Request request, const Sources& sources, const QueryRunner& run, const std::function<bool()>& stop = {});

// ---- The answer as a node of the graph: the same work on every run, fitted on
// the Train rows (Add as a step / Keep as a node). Only an answer read from
// tables has one: SVD of rows -> Truncated SVD; SVD or eigen of a covariance
// or correlation -> PCA (and Whitening); A x = b or least squares with b from
// a table -> Linear Solve (a branch: its result is x); A B with B from a table
// -> Linear Projection (B into its Matrix input).
struct HandOff {
    std::string label;          // the button ("Add as a step: Truncated SVD")
    std::string hint;           // what the node does on a run
    std::string type;           // the node's import name ("TruncatedSVD")
    std::string name;           // the node's name
    std::map<std::string, std::string> parameters;
    bool step = true;           // in the data path, else a branch reading A's table
    bool second_input = false;  // B's table goes into the node's second input
};
// `b_number_columns`: every number column of B's table (Linear Projection
// takes them all as its outputs, so B must use them all).
std::vector<HandOff> HandOffs(Op op, const Sources& sources, int keep, const std::vector<std::string>& b_number_columns);

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
