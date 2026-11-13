#pragma once

// Tools > Optimization and Calculus (TOFIX129 D10): one window, three groups.
// Minimize (training optimizers on a typed formula), Programs (linear and
// quadratic programs), Calculus (integral, derivative, curvature). Each
// request is answered by the backend (Expression, Calculus, Optimization):
// a sentence in words, its figures, the rows of its tables and the shapes
// its plots draw, or the reason it cannot run.

#include "explore_read.h"

#include <cyxwiz/calculus.h>
#include <cyxwiz/expression.h>
#include <cyxwiz/optimization.h>

#include <functional>
#include <map>
#include <string>
#include <vector>

namespace cyxwiz::optcalc {

enum class Op { Optimizers, Linear, Quadratic, Integral, Derivative, Curvature };

struct OpInfo {
    Op op;
    const char* id;      // stable ("integral")
    const char* label;   // "Integral"
    const char* group;   // "Calculus"
    const char* hint;    // one line on what it does
};
const std::vector<OpInfo>& Ops();
const OpInfo& InfoOf(Op op);
const OpInfo* FindOp(const std::string& id);
std::vector<std::string> Groups();  // Minimize, Programs, Calculus

// A number for people: up to `digits` significant digits, no trailing
// zeros, a minus sign (U+2212) for negatives; "1,600" for whole counts.
std::string Number(double value, int digits = 6);
std::string Count(long long value);

// ---------------------------------------------------------------------------
// Formulas of x and y

struct FormulaExample {
    const char* name;      // "Rosenbrock"
    const char* formula;   // "(1 - x)^2 + 100*(y - x^2)^2"
    double x0, y0;         // a start
    double x_min, x_max, y_min, y_max;  // a range that shows it
    const char* minima;    // "(1, 1)"
};
const std::vector<FormulaExample>& SurfaceExamples();

struct LineExample {
    const char* name;      // "sin(x)"
    const char* formula;
    const char* a;         // bounds as typed ("0", "pi", "-inf")
    const char* b;
    double x;              // a point for the derivative
};
const std::vector<LineExample>& LineExamples();

// A bound or a point as typed: a number, a constant formula ("pi/2"),
// "inf" or "-inf". False with the reason when it is not one.
bool ReadValue(const std::string& text, double& value, std::string& error, bool allow_infinite = false);

// A grid of f over a range, and its contour lines.
struct Segment {
    double x1, y1, x2, y2;
};
struct ContourLevel {
    double level;                      // in the drawn scale
    std::vector<Segment> segments;
};
struct SurfaceMap {
    std::string error;
    bool ok() const { return error.empty(); }
    int nx = 0, ny = 0;
    double x_min = 0, x_max = 0, y_min = 0, y_max = 0;
    std::vector<double> values;        // f, row by row in y (x fastest); NaN where undefined
    bool log_scale = false;            // levels in log10(f - min + 1): f spans many decades
    std::vector<ContourLevel> levels;
};
// f on an n x n grid (on the device) and `level_count` contour lines.
SurfaceMap MapOf(const Expression& f, double x_min, double x_max, double y_min, double y_max,
                 int n = 120, int level_count = 14);
// Marching squares over `values` (nx x ny, x fastest) at `level`.
std::vector<Segment> ContourSegments(const std::vector<double>& values, int nx, int ny, double x_min, double x_max,
                                     double y_min, double y_max, double level);

// ---------------------------------------------------------------------------
// Minimize

struct OptimizerChoice {
    OptimizerType type = OptimizerType::Adam;
    double learning_rate = 0.01;
    double momentum = 0.0;
    bool on = true;
};
// "SGD + momentum 0.9", "Adam"
std::string OptimizerName(const OptimizerChoice& choice);
// The optimizers offered, with their usual learning rates.
std::vector<OptimizerChoice> OfferedOptimizers();

struct MinimizeRequest {
    std::string formula = "(1 - x)^2 + 100*(y - x^2)^2";
    double x0 = -2.0, y0 = 2.0;
    int steps = 1000;
    std::vector<OptimizerChoice> optimizers;
};

enum class Tone { Good, Plain, Warning };

struct PathRow {
    std::string name;              // "SGD + momentum 0.9"
    std::string learning_rate;
    std::string ended_at;          // "(0.9943, 0.9886)", "left the map"
    std::string value;             // f there
    std::string gradient;          // |grad f| there
    std::string word;              // "Reached a minimum", "Still moving at step 1,000", "Diverged at step 6"
    Tone tone = Tone::Plain;
    size_t choice = 0;             // index into the request's optimizers
};

struct MinimizeAnswer {
    std::string error;
    bool ok() const { return error.empty(); }
    std::string sentence;
    std::vector<OptimizerPath> paths;   // one per optimizer that is on
    std::vector<PathRow> rows;
};
MinimizeAnswer RunMinimize(const MinimizeRequest& request, const std::function<bool()>& stop = {});

// ---------------------------------------------------------------------------
// Programs

struct ProgramRow {
    std::string name;
    std::vector<double> coefficients;
    ConstraintSense sense = ConstraintSense::LessEqual;
    double rhs = 0.0;
};

struct ProgramRequest {
    bool quadratic = false;
    bool maximize = true;                       // linear only
    std::vector<std::string> variables;         // "x1", "x2"
    std::vector<double> objective;              // linear: c; quadratic: c of c' x
    std::vector<std::vector<double>> Q;         // quadratic
    std::vector<ProgramRow> rows;
    std::vector<VariableBound> bounds;          // one per variable
};

// The window's starting problems.
struct ProgramExample {
    const char* name;
    ProgramRequest request;
};
const std::vector<ProgramExample>& LinearExamples();
const std::vector<ProgramExample>& QuadraticExamples();

// Rows pasted as text, one per line: the coefficients, then <=, >= or =, then
// the right-hand side ("1 2 <= 6"); an optional leading name ending in ':'
// ("Labour: 1 2 1 1 <= 40"). False with the line and the reason when not.
bool ParseRows(const std::string& text, size_t variables, std::vector<ProgramRow>& rows, std::string& error);

struct VariableRow {
    std::string name, value;
};
struct ConstraintRow {
    std::string name, room, price, binding;     // binding: "yes", "no", "equality"
};

// Two variables: the feasible region (a polygon clipped to `view`), each
// row's line across the view, and the best point.
struct ProgramPlot {
    bool drawn = false;
    double x_min = 0, x_max = 0, y_min = 0, y_max = 0;
    std::vector<std::pair<double, double>> region;    // empty: no feasible point in view
    std::vector<Segment> lines;                        // one per row, in order
    bool has_best = false;
    double best_x = 0, best_y = 0;
};

struct ProgramAnswer {
    std::string error;                  // a problem that cannot be posed (sizes, numbers)
    bool ok() const { return error.empty(); }
    std::string status;                 // "Optimal", "Infeasible", "Unbounded", "Not convex", "Failed"
    Tone tone = Tone::Plain;
    std::string sentence;
    std::vector<VariableRow> variables;
    std::vector<ConstraintRow> constraints;
    ProgramResult result;
    ProgramPlot plot;
};
ProgramAnswer RunProgram(const ProgramRequest& request);

// A portfolio from a table of returns (one column per asset, one row per
// period): the rows read (a row with a missing value left out), their means
// and covariance (n - 1, on the device), and the quadratic program
// Optimization::PortfolioProgram poses. `target` as typed (empty: none).
struct PortfolioChoice {
    std::string table;
    std::vector<std::string> assets;
    std::string target;
    bool long_only = true;
};
struct PortfolioRead {
    std::string error;
    bool ok() const { return error.empty(); }
    ProgramRequest request;
    std::string note;           // "120 periods of returns (none left out)."
};
PortfolioRead PortfolioFromTable(const PortfolioChoice& choice, const QueryRunner& run);

// The same as a node of the graph, run on every run and fitted on the Train
// rows: a branch reading the table (Keep as a node: Portfolio weights).
struct GraphHandOff {
    std::string label;          // the button
    std::string hint;           // what the node does
    std::string type;           // its import name
    std::string name;
    std::map<std::string, std::string> parameters;
};
GraphHandOff PortfolioHandOff(const PortfolioChoice& choice);

// ---------------------------------------------------------------------------
// Calculus

struct IntegralRequest {
    std::string formula = "sin(x)";
    std::string a = "0", b = "pi";
    QuadratureRule rule = QuadratureRule::Adaptive;
    int n = 100;
};

struct RuleRow {
    std::string name;          // "Simpson, 100"
    std::string value;
    std::string evaluations;
    std::string difference;    // from the adaptive answer
    double raw_value = 0.0, raw_difference = 0.0;  // for Copy / Export (the adaptive row: its estimate)
    int raw_evaluations = 0;
};

struct IntegralAnswer {
    std::string error;
    bool ok() const { return error.empty(); }
    std::string title;         // "∫ sin(x) dx from 0 to π = 2"
    std::string sentence;
    double value = 0.0;
    std::vector<RuleRow> rules;                 // every rule, the adaptive one first
    std::vector<double> curve_x, curve_y;       // f over [a, b] (finite part)
    double shade_from = 0.0, shade_to = 0.0;
};
IntegralAnswer RunIntegral(const IntegralRequest& request);

struct DerivativeRequest {
    std::string formula = "sin(x)";
    std::string x = "1";
    DifferenceRule rule = DifferenceRule::Central;
    std::string step;          // h as typed; empty: the step that suits float64
};

struct DerivativeAnswer {
    std::string error;
    bool ok() const { return error.empty(); }
    std::string title;         // "f′(1) = 0.540302305868 · f″(1) = −0.841470984808"
    std::string sentence;
    DifferentiationResult result;
    StepErrorCurve curve;
    std::vector<double> curve_x, curve_f, curve_df;   // f and f' around x
};
DerivativeAnswer RunDerivative(const DerivativeRequest& request);

struct CurvatureRequest {
    std::string formula = "(1 - x)^2 + 100*(y - x^2)^2";
    std::string x = "0", y = "0";
    double x_min = -2, x_max = 2, y_min = -1, y_max = 3;
    int per_side = 40;
};

struct CurvatureAnswer {
    std::string error;
    bool ok() const { return error.empty(); }
    std::string title;         // "Saddle at (0, 1)"
    std::string sentence;
    std::string range_sentence;
    std::vector<std::string> hessian;      // row by row, as text
    std::vector<std::string> eigenvalues;  // ascending, as text
    CurvatureResult result;
    RangeCurvature range;
};
CurvatureAnswer RunCurvature(const CurvatureRequest& request);

// ---------------------------------------------------------------------------
// Copy and Export: the answer as rows of plain numbers (a '-' sign, full
// precision), separated by `separator` (',' for a CSV file, '\t' to paste).

std::string MinimizeText(const MinimizeRequest& request, const MinimizeAnswer& answer, char separator);
std::string ProgramText(const ProgramRequest& request, const ProgramAnswer& answer, char separator);
std::string IntegralText(const IntegralAnswer& answer, char separator);
std::string DerivativeText(const DerivativeAnswer& answer, char separator);
std::string CurvatureText(const CurvatureAnswer& answer, char separator);

} // namespace cyxwiz::optcalc
