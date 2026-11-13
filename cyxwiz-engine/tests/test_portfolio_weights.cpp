// The Portfolio Weights step (TOFIX129 D10) against the exact KKT solutions
// of tests/computation_truth/reference/generate_optimization_fixtures.py
// ("portfolio"): 120 periods of four assets' returns, the first 90 marked
// Train. The step fits on the Train rows (all rows without a split).

#include "../src/core/graph_model.h"
#include "../src/core/node_executors/pipeline_operator_factory.h"
#include "../src/core/split_partitioning.h"

#include <arrow/api.h>
#include <nlohmann/json.hpp>

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

using json = nlohmann::json;

namespace {

int checks = 0;

void Check(bool condition, const std::string& message) {
    ++checks;
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        std::exit(1);
    }
}

std::shared_ptr<arrow::Table> Returns(const json& p, bool split) {
    std::vector<std::shared_ptr<arrow::Field>> fields;
    std::vector<std::shared_ptr<arrow::Array>> arrays;
    const auto& rows = p.at("returns");
    for (size_t a = 0; a < p.at("assets").size(); ++a) {
        arrow::DoubleBuilder b;
        for (const auto& row : rows) (void)b.Append(row[a].get<double>());
        std::shared_ptr<arrow::Array> column;
        (void)b.Finish(&column);
        fields.push_back(arrow::field(p.at("assets")[a].get<std::string>(), arrow::float64()));
        arrays.push_back(column);
    }
    if (split) {
        arrow::Int8Builder b;
        for (const auto& v : p.at("partition")) (void)b.Append(static_cast<int8_t>(v.get<int>()));
        std::shared_ptr<arrow::Array> column;
        (void)b.Finish(&column);
        fields.push_back(arrow::field(cyxwiz::kSplitPartitionColumn, arrow::int8()));
        arrays.push_back(column);
    }
    return arrow::Table::Make(arrow::schema(fields), arrays);
}

std::vector<double> Column(const std::shared_ptr<arrow::Table>& t, const char* name) {
    std::vector<double> out;
    for (const auto& chunk : t->GetColumnByName(name)->chunks()) {
        const auto values = std::static_pointer_cast<arrow::DoubleArray>(chunk);
        for (int64_t i = 0; i < values->length(); ++i) out.push_back(values->Value(i));
    }
    return out;
}

arrow::Result<std::shared_ptr<arrow::Table>> Run(const std::shared_ptr<arrow::Table>& input, const std::map<std::string, std::string>& params,
                                                 std::string& error) {
    auto op = cyxwiz::PipelineOperatorFactory::Instance().Create(gui::NodeType::PortfolioWeights);
    Check(op != nullptr, "the factory makes a Portfolio Weights step");
    if (!op->Configure(params, error)) return arrow::Status::Invalid(error);
    return op->Apply(input);
}

}  // namespace

int main() {
    std::ifstream in(CYXWIZ_OPTIMIZATION_FIXTURE);
    Check(in.good(), std::string("cannot open ") + CYXWIZ_OPTIMIZATION_FIXTURE);
    const json p = json::parse(in).at("portfolio");
    for (const auto& c : p.at("cases")) {
        const bool split = c.at("split").get<bool>();
        std::map<std::string, std::string> params = {{"long_only", c.at("long_only").get<bool>() ? "true" : "false"}};
        std::string label = std::string(split ? "Train rows" : "every row") + (c.at("long_only").get<bool>() ? ", long only" : ", long and short");
        if (!c.at("target_return").is_null()) {
            char buffer[32];
            std::snprintf(buffer, sizeof buffer, "%.17g", c.at("target_return").get<double>());
            params["target_return"] = buffer;
            label += ", target " + std::string(buffer);
        }
        std::string error;
        const auto result = Run(Returns(p, split), params, error);
        Check(result.ok(), label + ": " + result.status().ToString() + error);
        const auto table = *result;
        Check(table->num_rows() == 4, label + ": one row per asset");
        for (const char* name : {"weight", "mean_return", "sd", "risk_share"}) {
            const auto got = Column(table, name);
            for (size_t i = 0; i < got.size(); ++i) {
                const double want = c.at(name)[i].get<double>();
                Check(std::fabs(got[i] - want) <= 1e-7 + 1e-7 * std::fabs(want),
                      label + ": " + name + " of " + p.at("assets")[i].get<std::string>() + " " + std::to_string(got[i]) + " vs " +
                          std::to_string(want));
            }
        }
    }
    std::string error;
    char buffer[32];
    std::snprintf(buffer, sizeof buffer, "%.6g", p.at("unreachable").get<double>());
    const auto refused = Run(Returns(p, true), {{"target_return", buffer}}, error);
    Check(!refused.ok() && refused.status().message().find("no mix of these assets earns") != std::string::npos &&
              refused.status().message().find("the highest mean is") != std::string::npos,
          "an unreachable target named: " + refused.status().ToString());
    const auto one = Run(Returns(p, false), {{"asset_cols", "bonds"}}, error);
    Check(!one.ok() && one.status().message().find("at least two asset columns") != std::string::npos, "one asset refused");
    std::string bad;
    auto op = cyxwiz::PipelineOperatorFactory::Instance().Create(gui::NodeType::PortfolioWeights);
    Check(!op->Configure({{"target_return", "high"}}, bad) && bad.find("not a number") != std::string::npos, "a target that is not a number");
    std::cout << "Portfolio weights test passed: " << checks << " checks\n";
    return 0;
}
