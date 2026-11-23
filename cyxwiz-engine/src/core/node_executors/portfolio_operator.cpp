#include "portfolio_operator.h"

#include "linear_map.h"

#include <cyxwiz/linear_algebra.h>
#include <cyxwiz/optimization.h>

#include <spdlog/spdlog.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace cyxwiz {

namespace {

std::shared_ptr<arrow::Array> Doubles(const std::vector<double>& values) {
    arrow::DoubleBuilder b;
    (void)b.AppendValues(values);
    std::shared_ptr<arrow::Array> a;
    (void)b.Finish(&a);
    return a;
}

std::string Figure(double v) {
    char buffer[32];
    std::snprintf(buffer, sizeof buffer, "%.6g", v);
    return buffer;
}

}  // namespace

bool PortfolioWeightsOperator::Configure(const std::map<std::string, std::string>& params, std::string& error) {
    const auto cols = params.find("asset_cols");
    asset_cols_ = ParseColumnList(cols != params.end() ? cols->second : "");
    target_return_ = NAN;
    const auto target = params.find("target_return");
    if (target != params.end() && !target->second.empty()) {
        char* end = nullptr;
        target_return_ = std::strtod(target->second.c_str(), &end);
        if (!end || *end != '\0' || !std::isfinite(target_return_))
            return error = "PortfolioWeights: target_return is not a number: " + target->second, false;
    }
    return ReadBoolParameter(params, "long_only", true, long_only_, GetName(), error);
}

arrow::Result<std::shared_ptr<arrow::Table>> PortfolioWeightsOperator::Apply(const std::shared_ptr<arrow::Table>& input) {
    if (!input) return arrow::Status::Invalid("PortfolioWeights: input table is null");
    std::vector<std::string> assets;
    ARROW_RETURN_NOT_OK(ResolveNumberFeatures(input, asset_cols_, "", GetName(), assets));
    if (assets.size() < 2) return arrow::Status::Invalid("PortfolioWeights: a mix needs at least two asset columns");
    NumberColumns all;
    ARROW_RETURN_NOT_OK(ReadNumberColumns(input, assets, GetName(), all, GetCancellationQuery()));
    const NumberColumns train = TrainColumns(input, all);
    const size_t n = train.rows.size();
    if (n < 2) return arrow::Status::Invalid("PortfolioWeights: at least two periods (rows) with every return are needed");
    ARROW_RETURN_NOT_OK(CheckCancellation(GetName()));

    // The means and the covariance (n - 1) of the Train rows, on the device.
    const ColumnSummaryResult summary = LinearAlgebra::ColumnSummary(train.View());
    if (!summary.success) return arrow::Status::ExecutionError("PortfolioWeights: " + summary.error_message);
    std::vector<std::vector<double>> covariance = SecondMoments(summary, false, {}, static_cast<double>(n - 1));

    const QuadraticProgram problem = Optimization::PortfolioProgram(summary.mean, covariance, target_return_, long_only_);
    const ProgramResult r = Optimization::SolveQuadraticProgram(problem);
    if (!r.success) {
        if (r.status == ProgramStatus::Infeasible && !std::isnan(target_return_)) {
            size_t best = 0;
            for (size_t i = 1; i < assets.size(); ++i)
                if (summary.mean[i] > summary.mean[best]) best = i;
            return arrow::Status::Invalid("PortfolioWeights: no mix of these assets earns " + Figure(target_return_) +
                                          " a period" + (long_only_ ? " without short positions" : "") +
                                          "; the highest mean is " + Figure(summary.mean[best]) + " (" + assets[best] + ")");
        }
        return arrow::Status::Invalid("PortfolioWeights: " + r.error_message);
    }

    // Each asset's share of the risk: w_i (C w)_i / w' C w.
    const size_t d = assets.size();
    std::vector<double> sd(d), share(d), cw(d, 0.0);
    double variance = 0.0, mean = 0.0;
    for (size_t i = 0; i < d; ++i) {
        for (size_t j = 0; j < d; ++j) cw[i] += covariance[i][j] * r.x[j];
        variance += r.x[i] * cw[i];
        mean += r.x[i] * summary.mean[i];
        sd[i] = std::sqrt((std::max)(0.0, covariance[i][i]));
    }
    for (size_t i = 0; i < d; ++i) share[i] = variance > 0 ? r.x[i] * cw[i] / variance : 0.0;
    spdlog::info("PortfolioWeights: {} assets, {} periods: mean return {:.6g}, sd {:.6g} a period{}", d, n, mean,
                 std::sqrt((std::max)(0.0, variance)), std::isnan(target_return_) ? " (the minimum-variance mix)" : "");

    arrow::StringBuilder names;
    for (const auto& a : assets) (void)names.Append(a);
    std::shared_ptr<arrow::Array> name_array;
    (void)names.Finish(&name_array);
    const auto schema = arrow::schema({arrow::field("asset", arrow::utf8()), arrow::field("weight", arrow::float64()),
                                       arrow::field("mean_return", arrow::float64()), arrow::field("sd", arrow::float64()),
                                       arrow::field("risk_share", arrow::float64())});
    return arrow::Table::Make(schema, {name_array, Doubles(r.x), Doubles(summary.mean), Doubles(sd), Doubles(share)});
}

}  // namespace cyxwiz
