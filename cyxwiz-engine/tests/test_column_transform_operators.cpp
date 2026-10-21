// Fitted column transforms (TOFIX129 D2): Standard, Min-Max, Robust and
// Max-Abs scalers, Quantile and Power transforms against scikit-learn / scipy
// on a small table with missing values; missing stays missing in its row;
// fitted values saved and reused; settings that differ from the saved ones,
// and Box-Cox on values of 0 or less, are refused.

#include "../src/core/node_executors/preprocessing_operators.h"

#include <arrow/api.h>

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <map>
#include <string>
#include <vector>

using namespace cyxwiz;
namespace fs = std::filesystem;

namespace {
void Check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        std::exit(1);
    }
}

// a = 1, 2, (missing), 4, 10; b = 0, -3, 6, (missing), 3; y = text (left alone).
std::shared_ptr<arrow::Table> Table() {
    arrow::DoubleBuilder a, b;
    arrow::StringBuilder y;
    const double av[] = {1, 2, 0, 4, 10}, bv[] = {0, -3, 6, 0, 3};
    for (int i = 0; i < 5; ++i) {
        if (i == 2) (void)a.AppendNull();
        else (void)a.Append(av[i]);
        if (i == 3) (void)b.AppendNull();
        else (void)b.Append(bv[i]);
        (void)y.Append(i % 2 ? "x" : "z");
    }
    std::shared_ptr<arrow::Array> aa, bb, yy;
    (void)a.Finish(&aa);
    (void)b.Finish(&bb);
    (void)y.Finish(&yy);
    return arrow::Table::Make(arrow::schema({arrow::field("a", arrow::float64()), arrow::field("b", arrow::float64()),
                                             arrow::field("y", arrow::utf8())}),
                              {aa, bb, yy});
}

std::shared_ptr<arrow::Table> Run(IPipelineOperator& op, const std::map<std::string, std::string>& params,
                                  const std::shared_ptr<arrow::Table>& input, std::string* failure = nullptr) {
    std::string error;
    if (!op.Configure(params, error)) {
        if (failure) {
            *failure = error;
            return nullptr;
        }
        Check(false, op.GetName() + " configure: " + error);
    }
    auto out = op.Apply(input);
    if (!out.ok()) {
        if (failure) {
            *failure = out.status().ToString();
            return nullptr;
        }
        Check(false, op.GetName() + " apply: " + out.status().ToString());
    }
    return *out;
}

// The column's values; NaN marks a missing (null) cell.
std::vector<double> Column(const std::shared_ptr<arrow::Table>& t, const std::string& name) {
    std::vector<double> out;
    auto col = t->GetColumnByName(name);
    for (int c = 0; c < col->num_chunks(); ++c) {
        auto chunk = col->chunk(c);
        for (int64_t i = 0; i < chunk->length(); ++i) {
            if (chunk->IsNull(i)) {
                out.push_back(NAN);
                continue;
            }
            auto s = *chunk->GetScalar(i);
            out.push_back(std::static_pointer_cast<arrow::DoubleScalar>(*s->CastTo(arrow::float64()))->value);
        }
    }
    return out;
}

// `got` equals `want` (present values; NaN = missing), to float32 precision.
void Expect(const std::vector<double>& got, const std::vector<double>& want, const std::string& what, double tol = 2e-5) {
    Check(got.size() == want.size(), what + ": row count");
    for (size_t i = 0; i < want.size(); ++i) {
        if (std::isnan(want[i])) {
            Check(std::isnan(got[i]), what + ": row " + std::to_string(i + 1) + " should stay missing");
            continue;
        }
        Check(std::fabs(got[i] - want[i]) <= tol * std::max(1.0, std::fabs(want[i])),
              what + ": row " + std::to_string(i + 1) + " got " + std::to_string(got[i]) + ", want " + std::to_string(want[i]));
    }
}
}  // namespace

int main() {
    const auto table = Table();
    const double M = NAN;

    // scikit-learn on the present values of a (1, 2, 4, 10) and b (0, -3, 6, 3).
    {
        StandardScalerOperator op;
        auto out = Run(op, {{"columns", "a"}}, table);
        Expect(Column(out, "a"), {-0.9309493362512627, -0.6445033866354896, M, -0.0716114874039433, 1.6470642102906956}, "Standard a");
        Check(out->GetColumnByName("y")->type()->id() == arrow::Type::STRING, "text column left alone");
    }
    {
        MinMaxScalerOperator op;
        auto out = Run(op, {{"columns", "a"}, {"min", "-1"}, {"max", "1"}}, table);
        Expect(Column(out, "a"), {-1.0, -1.0 + 2.0 / 9.0, M, -1.0 + 6.0 / 9.0, 1.0}, "Min-Max a into -1 .. 1");
    }
    {
        RobustScalerOperator op;
        auto out = Run(op, {{"columns", "a"}}, table);
        Expect(Column(out, "a"), {-0.5333333333333333, -0.26666666666666666, M, 0.26666666666666666, 1.8666666666666667}, "Robust a");
    }
    {
        MaxAbsScalerOperator op;
        auto out = Run(op, {}, table);  // every number column
        Expect(Column(out, "b"), {0.0, -0.5, 1.0, M, 0.5}, "Max-Abs b");
        Expect(Column(out, "a"), {0.1, 0.2, M, 0.4, 1.0}, "Max-Abs a");
    }
    {
        QuantileTransformOperator op;
        auto out = Run(op, {{"columns", "a"}, {"n_quantiles", "4"}}, table);
        Expect(Column(out, "a"), {0.0, 0.33333333333333337, M, 0.6666666666666667, 1.0}, "Quantile uniform a");
        QuantileTransformOperator normal;
        out = Run(normal, {{"columns", "a"}, {"n_quantiles", "4"}, {"output_distribution", "normal"}}, table);
        Expect(Column(out, "a"), {-5.199337582605575, -0.43072729929545744, M, 0.4307272992954577, 5.19933758270342}, "Quantile normal a", 1e-4);
    }
    {
        PowerTransformOperator op;  // Yeo-Johnson, lambda found (scipy -0.4105)
        auto out = Run(op, {{"columns", "a,b"}}, table);
        Expect(Column(out, "a"), {0.6032589141288834, 0.884281261923128, M, 1.1778284620017958, 1.525738202524031}, "Yeo-Johnson a", 2e-4);
        Expect(Column(out, "b"), {0.0, -3.188248329217059, 5.484182931517227, M, 2.8249605329936855}, "Yeo-Johnson b", 2e-4);
        PowerTransformOperator box;  // Box-Cox (scipy lambda -0.1307)
        out = Run(box, {{"columns", "a"}, {"family", "box-cox"}}, table);
        Expect(Column(out, "a"), {0.0, 0.6626702252299034, M, 1.2679334797267825, 1.9883366130517623}, "Box-Cox a", 2e-4);
        PowerTransformOperator fixed;  // typed lambda 0: the natural log
        out = Run(fixed, {{"columns", "a"}, {"family", "box-cox"}, {"lambda_mode", "fixed"}, {"lambda", "0"}}, table);
        Expect(Column(out, "a"), {0.0, std::log(2.0), M, std::log(4.0), std::log(10.0)}, "Box-Cox lambda 0 = log");
        std::string why;
        PowerTransformOperator refused;
        Check(!Run(refused, {{"columns", "b"}, {"family", "box-cox"}}, table, &why) &&
                  why.find("Yeo-Johnson") != std::string::npos,
              "Box-Cox on 0 and below is refused and suggests Yeo-Johnson: " + why);
    }

    // Fitted values saved, then reused on other rows (validation / test).
    const fs::path dir = fs::temp_directory_path() / "cyxwiz_test_column_transforms";
    std::error_code ec;
    fs::remove_all(dir, ec);
    fs::create_directories(dir);
    for (const std::string name : {"StandardScaler", "MinMaxScaler", "RobustScaler", "MaxAbsScaler", "QuantileTransform", "PowerTransform"}) {
        const std::string path = (dir / (name + ".cyxstate.json")).string();
        auto make = [&]() -> std::unique_ptr<IPipelineOperator> {
            if (name == "StandardScaler") return std::make_unique<StandardScalerOperator>();
            if (name == "MinMaxScaler") return std::make_unique<MinMaxScalerOperator>();
            if (name == "RobustScaler") return std::make_unique<RobustScalerOperator>();
            if (name == "MaxAbsScaler") return std::make_unique<MaxAbsScalerOperator>();
            if (name == "QuantileTransform") return std::make_unique<QuantileTransformOperator>();
            return std::make_unique<PowerTransformOperator>();
        };
        auto fit = make();
        const auto fitted = Run(*fit, {{"columns", "a"}, {"save_state", "true"}, {"state_path", path}}, table);
        Check(fs::exists(path), name + ": fitted values saved");
        // Reused on the same rows: the same answer as fitting.
        auto reuse = make();
        const auto again = Run(*reuse, {{"columns", "a"}, {"operation_mode", "transform_only"}, {"state_path", path}}, table);
        Expect(Column(again, "a"), Column(fitted, "a"), name + ": saved values give the fitted answer", 1e-6);
    }
    {
        std::string why;
        StandardScalerOperator differs;
        Check(!Run(differs, {{"columns", "a"}, {"with_mean", "false"}, {"operation_mode", "transform_only"},
                             {"state_path", (dir / "StandardScaler.cyxstate.json").string()}},
                   table, &why),
              "settings that differ from the saved ones are refused");
        MinMaxScalerOperator no_path;
        Check(!Run(no_path, {{"operation_mode", "transform_only"}}, table, &why), "Use saved values needs a state path");
        MinMaxScalerOperator inverted;
        Check(!Run(inverted, {{"min", "1"}, {"max", "0"}}, table, &why), "Min-Max range must rise");
    }
    fs::remove_all(dir, ec);
    std::cout << "column transforms: six transforms match scikit-learn / scipy, missing values kept, saved values reused. OK\n";
    return 0;
}
