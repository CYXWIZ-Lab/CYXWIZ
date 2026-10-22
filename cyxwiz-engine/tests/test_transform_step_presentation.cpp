// Data Studio > Prepare's scale / power / log step view (TOFIX129 D2): the
// method list and switch, the step's columns, before / after figures
// (numpy / scipy / scikit-learn values), the power figures and the words.

#include "../src/core/transform_step_presentation.h"

#include <arrow/api.h>

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

using namespace cyxwiz::transform_step;
using gui::NodeType;

namespace {
void Check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        std::exit(1);
    }
}

void Near(double got, double want, const std::string& what, double tol = 1e-9) {
    Check(std::fabs(got - want) <= tol * std::max(1.0, std::fabs(want)),
          what + ": got " + std::to_string(got) + ", want " + std::to_string(want));
}

std::shared_ptr<arrow::Array> Doubles(const std::vector<double>& values) {
    arrow::DoubleBuilder b;
    for (double v : values) {
        if (std::isnan(v)) (void)b.AppendNull();
        else (void)b.Append(v);
    }
    std::shared_ptr<arrow::Array> out;
    (void)b.Finish(&out);
    return out;
}

std::shared_ptr<arrow::Table> Table(const std::vector<std::pair<std::string, std::vector<double>>>& columns, bool with_text = true) {
    arrow::FieldVector fields;
    std::vector<std::shared_ptr<arrow::Array>> arrays;
    for (const auto& [name, values] : columns) {
        fields.push_back(arrow::field(name, arrow::float64()));
        arrays.push_back(Doubles(values));
    }
    if (with_text) {
        arrow::StringBuilder y;
        for (size_t i = 0; i < columns.front().second.size(); ++i) (void)y.Append(i % 2 ? "x" : "z");
        std::shared_ptr<arrow::Array> yy;
        (void)y.Finish(&yy);
        fields.push_back(arrow::field("y", arrow::utf8()));
        arrays.push_back(yy);
    }
    return arrow::Table::Make(arrow::schema(fields), arrays);
}
}  // namespace

int main() {
    const double M = NAN;

    // Methods and kinds.
    Check(ScaleMethods().size() == 5 && std::string(ScaleMethods()[0].label) == "Z-score" &&
              std::string(ScaleMethods()[4].label) == "Quantile",
          "five scaling methods, Z-score first");
    Check(KindOf(NodeType::RobustScaler) == Kind::Scale && KindOf(NodeType::PowerTransform) == Kind::Power &&
              KindOf(NodeType::LogTransform) == Kind::Log && KindOf(NodeType::FilterRows) == Kind::None,
          "kinds");
    Check(MethodOf(NodeType::MaxAbsScaler) && !MethodOf(NodeType::PowerTransform), "methods are the scalers");
    Check(Formula(NodeType::MinMaxScaler, {{"min", "-1"}, {"max", "1"}}) == "(x - min) / (max - min), to -1 .. 1",
          "Min-Max formula: " + Formula(NodeType::MinMaxScaler, {{"min", "-1"}, {"max", "1"}}));
    Check(Formula(NodeType::PowerTransform, {}) == "Yeo-Johnson, lambda found per column", "power formula");
    Check(Formula(NodeType::PowerTransform, {{"family", "box-cox"}, {"lambda_mode", "fixed"}, {"lambda", "0.5"}}) ==
              "Box-Cox, lambda 0.5",
          "typed lambda formula");
    Check(Formula(NodeType::LogTransform, {{"base", "log10"}, {"use_log1p", "false"}}) == "log(x), base 10", "log formula");
    Check(Formula(NodeType::QuantileTransform, {{"output_distribution", "normal"}}) == "through its quantiles to a normal shape",
          "quantile formula");

    // A method switch keeps the columns and the fit choice, not another method's saved values.
    const auto carried = CarriedParameters({{"columns", "a,b"}, {"label_col", "y"}, {"exclude_columns", "id"},
                                            {"operation_mode", "transform_only"}, {"state_path", "s.cyxstate.json"},
                                            {"save_state", "true"}, {"with_mean", "false"}});
    Check(carried.at("columns") == "a,b" && carried.at("label_col") == "y" && carried.at("exclude_columns") == "id" &&
              carried.at("save_state") == "true" && carried.at("operation_mode") == "fit_transform" &&
              !carried.count("state_path") && !carried.count("with_mean"),
          "method switch carries columns and fit choice only");

    // The step's columns: chosen, else number columns bar label and excluded.
    const auto input = Table({{"a", {1, 2, M, 4, 10}}, {"b", {0, -3, 6, M, 3}}, {"id", {1, 2, 3, 4, 5}}});
    Check(NumberColumns(*input->schema()) == std::vector<std::string>{"a", "b", "id"}, "number columns");
    Check(StepColumns({{"columns", " b , a"}}, *input->schema()) == std::vector<std::string>{"b", "a"}, "chosen columns");
    Check(StepColumns({{"exclude_columns", "id"}}, *input->schema()) == std::vector<std::string>{"a", "b"}, "auto columns");
    Check(StepColumns({{"label_col", "a"}}, *input->schema()) == std::vector<std::string>{"b", "id"}, "label left out");

    // Describe: numpy / scipy on the present values of a.
    const auto a = ColumnValues(*input, "a");
    Check(a.size() == 5 && std::isnan(a[2]), "missing read as NaN");
    Check(ColumnValues(*input, "y").empty(), "a text column has no values");
    const Stats s = Describe(a);
    Check(s.ok && s.count == 4 && s.missing == 1, "count and missing");
    Near(s.mean, 4.25, "mean");
    Near(s.std, 3.491060010942235, "population std");
    Near(s.skew, 0.8483206969390205, "skewness", 1e-7);
    Check(s.bins == std::vector<int64_t>{1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1},
          "numpy histogram, 24 bins");
    Check(Range(s) == "1 .. 10", "range: " + Range(s));
    Check(!Describe({M, M}).ok, "no values: nothing to describe");

    // A scaling step's figures, from its input and output.
    const auto scaled = Table({{"a", {-0.9309493362512627, -0.6445033866354896, M, -0.0716114874039433, 1.6470642102906956}},
                               {"b", {0, -3, 6, M, 3}},
                               {"id", {1, 2, 3, 4, 5}}});
    const StepFigures z = ComputeFigures(NodeType::StandardScaler, {{"columns", "a"}}, *input, *scaled);
    Check(z.ok() && z.columns.size() == 1 && z.rows_in == 5 && z.rows_out == 5, "one column figured");
    Near(z.columns[0].after.mean, 0.0, "after mean", 1e-6);
    Near(z.columns[0].after.std, 1.0, "after std", 1e-6);
    Check(z.columns[0].after.missing == 1, "missing kept in the after figures");
    Check(BeyondZ3Line(z) == "Values beyond |z| 3: none in these columns.", BeyondZ3Line(z));
    StepFigures far = z;
    far.columns[0].after.beyond_z3 = 2;
    Check(BeyondZ3Line(far) == "Values beyond |z| 3: 2 (a 2).", BeyondZ3Line(far));
    Check(!ComputeFigures(NodeType::StandardScaler, {{"columns", "y"}}, *input, *scaled).ok(), "a text column is no step column");

    // A power step: scipy Box-Cox on a (lambda -0.1307).
    const auto boxed = Table({{"a", {0.0, 0.6626702252299034, M, 1.2679334797267825, 1.9883366130517623}},
                              {"b", {0, -3, 6, M, 3}},
                              {"id", {1, 2, 3, 4, 5}}});
    const StepFigures p = ComputeFigures(NodeType::PowerTransform, {{"columns", "a"}, {"family", "box-cox"}}, *input, *boxed);
    Check(p.ok() && p.columns[0].has_power && p.columns[0].power.box_cox, "power figures");
    Near(p.columns[0].power.lambda, -0.13072833065209727, "Box-Cox lambda (scipy)", 1e-3);
    Check(p.columns[0].power.lambdas.size() == p.columns[0].power.log_likelihoods.size() &&
              p.columns[0].power.lambdas.size() >= 100 && p.columns[0].power.lambdas.front() == -5.0,
          "likelihood curve over -5 .. 5");
    Near(p.columns[0].after.skew, 0.04890206694721448, "skewness after", 1e-6);
    Check(std::isnan(p.columns[0].power.p_after), "no normality test under 20 values");
    Check(PowerSentence(p.columns[0]) == "Skewness 0.85 -> 0.05 (the normality test needs at least 20 values).",
          PowerSentence(p.columns[0]));
    Check(BoxCoxNote(p) == "Box-Cox needs values above 0; the column's are.", BoxCoxNote(p));
    const StepFigures bad = ComputeFigures(NodeType::PowerTransform, {{"columns", "a,b"}, {"family", "box-cox"}}, *input, *boxed);
    Check(BoxCoxNote(bad) == "Box-Cox needs values above 0; b has 0 or less: use Yeo-Johnson.", BoxCoxNote(bad));
    Check(PowerSentence(bad.columns[1]).find("Use Yeo-Johnson") != std::string::npos, PowerSentence(bad.columns[1]));
    const StepFigures typed = ComputeFigures(
        NodeType::PowerTransform, {{"columns", "a"}, {"lambda_mode", "fixed"}, {"lambda", "0.5"}}, *input, *boxed);
    Near(typed.columns[0].power.lambda, 0.5, "typed lambda");
    const StepFigures reused = ComputeFigures(NodeType::PowerTransform, {{"columns", "a"}, {"operation_mode", "transform_only"}},
                                              *input, *boxed, {{"a", 0.25}});
    Near(reused.columns[0].power.lambda, 0.25, "saved lambda");

    ColumnFigures words;
    words.has_power = true;
    words.before.ok = words.after.ok = true;
    words.before.skew = 0.47;
    words.after.skew = 0.03;
    words.power.p_after = 1.3e-8;
    Check(PowerSentence(words) ==
              "Skewness 0.47 -> 0.03, but the values still do not look normal (p 1.3e-08): no single power transform "
              "makes them normal; the column may mix groups (more than one hump).",
          PowerSentence(words));
    words.power.p_after = 0.21;
    Check(PowerSentence(words) == "Skewness 0.47 -> 0.03; the values now look normal (p 0.21).", PowerSentence(words));
    Check(PValue(NAN) == "-" && PValue(0.21) == "0.21", "p values");
    Check(Value(-2.2e-16) == "0" && Value(-3e-10) == "0" && Value(3.616) == "3.616" && Value(-0.5) == "-0.5",
          "values: " + Value(-3e-10));
    Check(RunLine(344, 344) == "Last run passed \xC2\xB7 344 rows in, 344 out", RunLine(344, 344));

    // Compare methods: scikit-learn on a.
    const CompareFigures c = CompareMethods("a", a);
    Check(c.ok() && c.methods.size() == 5 && c.original.count == 4, "every method compared");
    Near(c.methods[0].min, -0.9309493362512627, "Z-score min", 1e-7);
    Near(c.methods[0].max, 1.6470642102906956, "Z-score max", 1e-7);
    Near(c.methods[1].min, 0.0, "Min-Max min");
    Near(c.methods[1].max, 1.0, "Min-Max max");
    Near(c.methods[2].min, -0.5333333333333333, "Robust min", 1e-7);
    Near(c.methods[3].min, 0.1, "Max-Abs min", 1e-7);
    Near(c.methods[4].max, 5.19933758270342, "Quantile (normal) max", 1e-4);
    Check(!CompareMethods("a", {M}).ok(), "no values: nothing to compare");

    std::cout << "transform step presentation: methods, columns, figures, power and words match numpy / scipy / sklearn. OK\n";
    return 0;
}
