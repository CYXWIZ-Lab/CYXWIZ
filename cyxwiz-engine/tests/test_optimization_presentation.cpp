// Tools > Optimization and Calculus (TOFIX129 D10): the window's model. The
// figures are the oracle's (scipy, torch; see
// tests/computation_truth/reference/generate_optimization_fixtures.py); this
// test checks the words, rows and shapes the window shows for them.

#include "core/linear_algebra_presentation.h"
#include "core/optimization_presentation.h"
#include "core/session_query_engine.h"

#include <arrow/api.h>
#include <nlohmann/json.hpp>

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

using namespace cyxwiz;
using namespace cyxwiz::optcalc;

namespace {

int checks = 0;

void Check(bool condition, const std::string& message) {
    ++checks;
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        std::exit(1);
    }
}

bool Has(const std::string& text, const std::string& part) { return text.find(part) != std::string::npos; }

void CheckBasics() {
    Check(Ops().size() == 6 && Groups().size() == 3, "six operations in three groups");
    Check(FindOp("integral") && FindOp("integral")->op == Op::Integral && !FindOp("nope"), "operations by id");
    Check(Number(-0.5) == "\xE2\x88\x92" "0.5", "a minus sign: " + Number(-0.5));
    Check(Number(3.286e-5, 3) == "3.29e-5", "short exponent: " + Number(3.286e-5, 3));
    Check(Number(12.0) == "12" && Number(2.0, 15) == "2", "whole numbers without a point");
    Check(Count(1600) == "1,600" && Count(825) == "825", "counts with commas");
    double v = 0;
    std::string why;
    Check(ReadValue("pi/2", v, why) && std::abs(v - 1.5707963267948966) < 1e-15, "a constant formula is a value");
    Check(!ReadValue("inf", v, why) && ReadValue("-inf", v, why, true) && std::isinf(v) && v < 0, "infinity only where allowed");
    Check(!ReadValue("2x", v, why) && Has(why, "is not a number"), "2x refused: " + why);
}

void CheckMap() {
    // x + y on a 3 x 3 grid of [0, 2]^2: the level 1 line crosses 2 cells.
    const std::vector<double> plane = {0, 1, 2, 1, 2, 3, 2, 3, 4};
    const auto segments = ContourSegments(plane, 3, 3, 0, 2, 0, 2, 1.5);
    Check(segments.size() == 3, "three cells cross x + y = 1.5: " + std::to_string(segments.size()));
    for (const auto& s : segments)
        Check(std::abs(s.x1 + s.y1 - 1.5) < 1e-12 && std::abs(s.x2 + s.y2 - 1.5) < 1e-12, "segment ends on the line");
    const auto map = MapOf(Expression::Parse(SurfaceExamples()[0].formula, {"x", "y"}), -2.5, 2, -1, 3);
    Check(map.ok() && map.log_scale && map.levels.size() == 14, "Rosenbrock's map is drawn in log scale, 14 lines");
    size_t drawn = 0;
    for (const auto& level : map.levels) drawn += level.segments.empty() ? 0 : 1;
    Check(drawn == 14, "every Rosenbrock level has a line in view");
    const auto sphere = MapOf(Expression::Parse("x^2 + y^2", {"x", "y"}), -3, 3, -3, 3);
    Check(sphere.ok() && !sphere.log_scale, "a sphere stays in plain scale");
    Check(!MapOf(Expression::Parse("x", {"x"}), 0, 1, 0, 1).ok(), "the map needs x and y");
}

void CheckMinimize() {
    MinimizeRequest request;
    request.optimizers = OfferedOptimizers();  // momentum, Adam, RMSprop, SGD 0.01 on; the rest off
    const auto answer = RunMinimize(request);
    Check(answer.ok() && answer.rows.size() == 4, "four optimizers on: " + answer.error);
    const auto& momentum = answer.rows[0];
    Check(momentum.name == "SGD + momentum 0.9" && momentum.word == "Reached a minimum" && momentum.tone == Tone::Good,
          "momentum reaches the minimum: " + momentum.word);
    Check(momentum.ended_at == "(0.99427, 0.98855)", "momentum ends where torch does: " + momentum.ended_at);
    Check(answer.rows[1].name == "Adam" && answer.rows[1].word == "Still moving at step 1,000", "Adam still moving: " + answer.rows[1].word);
    Check(answer.rows[1].ended_at == "(\xE2\x88\x92" "1.5059, 2.2735)", "Adam's end: " + answer.rows[1].ended_at);
    const auto& sgd = answer.rows[3];
    Check(sgd.word == "Diverged at step 4" && sgd.tone == Tone::Warning && sgd.value == "overflow", "SGD 0.01 diverges: " + sgd.word);
    Check(Has(answer.sentence, "SGD + momentum 0.9 ends lowest: f = 3.2866e-5 at (0.99427, 0.98855)") && Has(answer.sentence, ", a minimum.") &&
              Has(answer.sentence, "SGD at learning rate 0.01 diverged: its steps grew without limit"),
          "sentence: " + answer.sentence);
    Check(answer.paths[0].points.size() == 1001, "a path holds every step");
    const std::string csv = MinimizeText(request, answer, ',');
    Check(Has(csv, "optimizer,learning_rate,step,x,y,f,gradient_norm\nSGD + momentum 0.9,0.001,0,-2,2,409,"), "paths as CSV");
    int asked = 0;
    const auto stopped = RunMinimize(request, [&asked] { return ++asked > 10; });
    Check(stopped.ok() && Has(stopped.rows[0].word, "Stopped at step"), "Stop: " + stopped.rows[0].word);
    MinimizeRequest none = request;
    for (auto& o : none.optimizers) o.on = false;
    Check(RunMinimize(none).error == "Turn on at least one optimizer", "no optimizer on");
    MinimizeRequest bad = request;
    bad.formula = "(1 - x";
    Check(Has(RunMinimize(bad).error, "Expected ')'"), "a formula that does not parse");
}

void CheckPrograms() {
    const auto two = RunProgram(LinearExamples()[0].request);
    Check(two.ok() && two.status == "Optimal" && two.tone == Tone::Good, "two products solve");
    Check(two.sentence == "Best Z = 12 at x1 = 4 and x2 = 0. C1 is binding: one more unit of its right-hand side adds 3 to Z. "
                          "C2 has 2 to spare.",
          "two products sentence: " + two.sentence);
    Check(two.constraints.size() == 2 && two.constraints[0].binding == "yes" && two.constraints[0].price == "3" &&
              two.constraints[1].room == "2" && two.constraints[1].binding == "no",
          "constraint rows");
    Check(two.plot.drawn && two.plot.has_best && two.plot.best_x == 4 && two.plot.region.size() == 4 && two.plot.lines.size() == 2,
          "two variables: region with 4 corners, 2 lines, the best point");
    bool corner = false;
    for (const auto& p : two.plot.region) corner = corner || (std::abs(p.first - 2) < 1e-9 && std::abs(p.second - 2) < 1e-9);
    Check(corner, "the region's corner at (2, 2)");

    const auto plan = RunProgram(LinearExamples()[1].request);
    Check(plan.ok() && Has(plan.sentence, "Best Z = 825 at a = 0, b = 2.5, c = 22.5 and d = 12.5.") &&
              Has(plan.sentence, "Labour is binding: one more unit of its right-hand side adds 10 to Z.") &&
              Has(plan.sentence, "Order holds exactly: one more unit of its right-hand side adds 5 to Z."),
          "production plan: " + plan.sentence);
    Check(!plan.plot.drawn && plan.constraints[2].binding == "equality", "four variables: no region; the equality row");

    const auto none = RunProgram(LinearExamples()[4].request);
    Check(none.ok() && none.status == "Infeasible" && none.tone == Tone::Warning &&
              none.sentence == "No point satisfies every constraint and bound together.",
          "infeasible in words: " + none.sentence);
    Check(none.plot.drawn && none.plot.region.empty(), "infeasible: no region drawn");

    const auto quad = RunProgram(QuadraticExamples()[0].request);
    Check(quad.ok() && quad.sentence == "Smallest value \xE2\x88\x92" "7.125 at x1 = 0.75 and x2 = 2.25. C1 is binding: one more unit of its "
                                        "right-hand side takes 0.5 from the value.",
          "QP sentence: " + quad.sentence);
    const auto folio = RunProgram(QuadraticExamples()[1].request);
    Check(folio.ok() && folio.variables[0].value == "0.27352572" && folio.variables[2].value == "0.58971142",
          "portfolio weights: " + (folio.variables.empty() ? folio.sentence : folio.variables[0].value));

    ProgramRequest saddle = QuadraticExamples()[0].request;
    saddle.Q = {{1, 0}, {0, -2}};
    const auto refused = RunProgram(saddle);
    Check(refused.ok() && refused.status == "Not convex" && Has(refused.sentence, "smallest eigenvalue is -2"),
          "a saddle Q is an answer in words: " + refused.sentence);
    ProgramRequest wide = LinearExamples()[0].request;
    wide.rows[0].coefficients = {1, 1, 1};
    Check(RunProgram(wide).error == "C1 has 3 coefficients; the problem has 2 variables", "a wrong row size is an error");

    std::vector<ProgramRow> rows;
    std::string why;
    Check(ParseRows("Labour: 1 2 1 1 <= 40\n3, 1, 2, 1 >= 60\n\n1 1 1 0 = 25\n", 4, rows, why) && rows.size() == 3 &&
              rows[0].name == "Labour" && rows[1].name == "C2" && rows[1].sense == ConstraintSense::GreaterEqual && rows[2].rhs == 25,
          "rows pasted: " + why);
    Check(!ParseRows("1 2 3 <= 4", 2, rows, why) && why == "Line 1 has 3 coefficients; the problem has 2 variables", "row size: " + why);
    Check(!ParseRows("1 2 4", 2, rows, why) && Has(why, "needs <=, >= or ="), "no sign: " + why);

    const std::string text = ProgramText(LinearExamples()[0].request, two, ',');
    Check(Has(text, "status,Optimal\nobjective,12\n") && Has(text, "C1,<=,4,0,3,yes\n") &&
              Has(text, "C2,<=,6,2,0,no\n"),
          "the program as CSV: " + text);
    Check(Has(ProgramText(QuadraticExamples()[0].request, quad, '\t'), "x1\t0.75"), "tab-separated to paste");
}

void CheckCalculus() {
    const auto sine = RunIntegral({});
    Check(sine.ok() && sine.title == "\xE2\x88\xAB sin(x) dx from 0 to \xCF\x80 = 2", "integral title: " + sine.title);
    Check(sine.sentence == "Estimated error 2.2e-14, from 21 evaluations.", "integral sentence: " + sine.sentence);
    Check(sine.rules.size() == 6 && sine.rules[3].name == "Simpson, 100" && sine.rules[3].difference == "1.1e-8",
          "rules compared with the adaptive answer");
    Check(sine.curve_x.size() == 241 && sine.curve_y.size() == 241, "the curve");
    Check(Has(IntegralText(sine, ','), "\"Simpson, 100\",2.00000001082450") && Has(IntegralText(sine, ','), "Adaptive"),
          "rules as CSV, a name with a comma quoted: " + IntegralText(sine, ','));
    IntegralRequest gauss{"exp(-x^2)", "-inf", "inf"};
    const auto bell = RunIntegral(gauss);
    Check(bell.ok() && std::abs(bell.value - 1.7724538509055159) < 1e-10 && bell.rules.size() == 1 &&
              Has(bell.title, "from \xE2\x88\x92\xE2\x88\x9E to \xE2\x88\x9E"),
          "the whole line: " + bell.title);
    IntegralRequest trap{"sin(x)", "0", "pi", QuadratureRule::Trapezoid, 100};
    const auto t = RunIntegral(trap);
    Check(t.ok() && Has(t.sentence, "Trapezoid with 100 intervals, 101 evaluations. It differs from the adaptive answer by 1.6e-4"),
          "a fixed rule: " + t.sentence);
    IntegralRequest pole{"1/sqrt(x)", "0", "1"};
    Check(Has(RunIntegral(pole).sentence, "Did not reach the tolerance in 50 intervals"), "an end singularity said");
    IntegralRequest open{"exp(-x)", "0", "inf", QuadratureRule::Simpson, 10};
    Check(RunIntegral(open).error == "Infinite bounds need the adaptive rule", "infinite with a fixed rule");

    const auto d = RunDerivative({});
    Check(d.ok() && d.title == "f\xE2\x80\xB2(1) = 0.540302305868 \xC2\xB7 f\xE2\x80\xB3(1) = \xE2\x88\x92" "0.841470984808",
          "derivative title: " + d.title);
    Check(Has(d.sentence, "From the formula: f\xE2\x80\xB2 = cos(x), f\xE2\x80\xB3 = -sin(x). Central differences with h = 6.1e-6"),
          "derivative sentence: " + d.sentence);
    Check(d.curve.success && d.curve.steps.size() == 45 && d.curve_df.size() == d.curve_x.size(), "error curve and f'");
    Check(RunDerivative({"sin(x)", "1", DifferenceRule::Central, "0"}).error == "The step h must be above 0 (empty: the step that suits float64)",
          "a step of 0 refused");
    Check(Has(RunDerivative({"sin(x)", "1", DifferenceRule::Central, "1e-"}).error, "is not a number"), "a step that is not a number refused");
    Check(RunDerivative({"sin(x)", "1", DifferenceRule::Central, "1e-5"}).ok(), "a typed step");
    DerivativeRequest kink;
    kink.formula = "max(x, 0)";
    const auto k = RunDerivative(kink);
    Check(k.ok() && Has(k.title, "\xE2\x89\x88") && k.curve_df.empty(), "no derivative formula: " + k.title);

    const auto origin = RunCurvature({});
    Check(origin.ok() && origin.title == "Curves up in every direction at (0, 0)" && origin.eigenvalues.front() == "2",
          "curvature at (0, 0): " + origin.title);
    Check(Has(origin.range_sentence, "not convex. The Hessian has a negative eigenvalue at 672 of the 1,600 points"),
          "range: " + origin.range_sentence);
    CurvatureRequest one;
    one.x = "1";
    one.y = "1";
    Check(Has(RunCurvature(one).sentence, "one direction 2,508 times steeper"), "a narrow bowl");
    CurvatureRequest saddle;
    saddle.y = "1";
    Check(RunCurvature(saddle).title == "Saddle at (0, 1)", "a saddle");
    CurvatureRequest bowl;
    bowl.formula = "x^2 + 3*y^2";
    Check(Has(RunCurvature(bowl).range_sentence, "no point curves down"), "a convex bowl over the range");
}

void CheckPortfolioFromTable() {
    std::ifstream in(CYXWIZ_OPTIMIZATION_FIXTURE);
    Check(in.good(), "the optimization fixture");
    const auto p = nlohmann::json::parse(in).at("portfolio");
    std::vector<std::shared_ptr<arrow::Field>> fields;
    std::vector<std::shared_ptr<arrow::Array>> arrays;
    std::vector<std::string> assets;
    for (size_t a = 0; a < p.at("assets").size(); ++a) {
        arrow::DoubleBuilder b;
        for (const auto& row : p.at("returns")) (void)b.Append(row[a].get<double>());
        std::shared_ptr<arrow::Array> column;
        (void)b.Finish(&column);
        assets.push_back(p.at("assets")[a].get<std::string>());
        fields.push_back(arrow::field(assets.back(), arrow::float64()));
        arrays.push_back(column);
    }
    const auto root = std::filesystem::temp_directory_path() / "cyxwiz_test_optimization_presentation";
    std::filesystem::remove_all(root);
    std::filesystem::create_directories(root);
    SessionQueryEngine engine(root.string());
    std::string error;
    Check(engine.AttachArrow("returns", 1, arrow::Table::Make(arrow::schema(fields), arrays), &error), "attach: " + error);
    const QueryRunner run = [&](const QueryRequest& q) { return engine.Run(q); };
    // The every-row case: long only, with a target.
    const auto& c = p.at("cases").back();
    char target[32];
    std::snprintf(target, sizeof target, "%.17g", c.at("target_return").get<double>());
    PortfolioChoice choice{"returns", assets, target, true};
    const auto read = PortfolioFromTable(choice, run);
    Check(read.ok() && Has(read.note, "120 periods of returns (none left out); a target of"), "read the returns: " + read.error + read.note);
    const auto answer = RunProgram(read.request);
    Check(answer.ok() && answer.result.success, "the portfolio solves: " + answer.sentence);
    for (size_t i = 0; i < assets.size(); ++i)
        Check(std::abs(answer.result.x[i] - c.at("weight")[i].get<double>()) < 1e-7, "weight of " + assets[i]);
    Check(Has(answer.sentence, "Budget holds exactly"), "the budget row in words: " + answer.sentence);
    const auto hand = PortfolioHandOff(choice);
    Check(hand.type == "PortfolioWeights" && hand.parameters.at("asset_cols") == "equity,growth,credit,bonds" &&
              hand.parameters.at("long_only") == "true" && hand.parameters.at("target_return") == target,
          "the hand-off carries the choice");
    Check(PortfolioFromTable({"", assets, "", true}, run).error == "Pick a table of returns", "no table picked");
    Check(PortfolioFromTable({"returns", {"bonds"}, "", true}, run).error == "Tick at least two asset columns", "one asset refused");
    Check(Has(PortfolioFromTable({"returns", assets, "lots", true}, run).error, "is not a number"), "a target that is not a number");
    // The window hands the node a number, whatever formula the target was typed as.
    Check(PortfolioHandOff({"returns", assets, " 1/100 ", true}).parameters.at("target_return") == "0.01",
          "a typed target reaches the node as a number");

    // A table a Data Split marked: the Train rows, as the node reads them; the
    // split marker is not offered as a column.
    arrow::Int8Builder marks;
    for (const auto& v : p.at("partition")) (void)marks.Append(static_cast<int8_t>(v.get<int>()));
    std::shared_ptr<arrow::Array> marked;
    (void)marks.Finish(&marked);
    auto split_fields = fields;
    auto split_arrays = arrays;
    split_fields.push_back(arrow::field("__partition__", arrow::int8()));
    split_arrays.push_back(marked);
    Check(engine.AttachArrow("split_returns", 2, arrow::Table::Make(arrow::schema(split_fields), split_arrays), &error),
          "attach the split table: " + error);
    std::string columns_error;
    Check(linalg::NumberColumns("split_returns", run, columns_error) == assets, "the split marker is not a column to tick");
    for (const auto& sc : p.at("cases")) {
        if (!sc.at("split").get<bool>() || !sc.at("long_only").get<bool>() || sc.at("target_return").is_null()) continue;
        char split_target[32];
        std::snprintf(split_target, sizeof split_target, "%.17g", sc.at("target_return").get<double>());
        const auto train = PortfolioFromTable({"split_returns", assets, split_target, true}, run);
        Check(train.ok() && Has(train.note, "90 periods of split_returns (its Train rows, 30 Validation / Test rows left out"),
              "the Train rows read: " + train.error + train.note);
        const auto solved = RunProgram(train.request);
        for (size_t i = 0; i < assets.size(); ++i)
            Check(solved.ok() && std::abs(solved.result.x[i] - sc.at("weight")[i].get<double>()) < 1e-7,
                  "Train-row weight of " + assets[i] + " = the node's");
    }
}

}  // namespace

int main() {
    CheckBasics();
    CheckMap();
    CheckMinimize();
    CheckPrograms();
    CheckCalculus();
    CheckPortfolioFromTable();
    std::cout << "Optimization presentation test passed: " << checks << " checks\n";
    return 0;
}
