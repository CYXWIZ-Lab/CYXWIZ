// DataTransform (the data transformation panels) against PyTorch float64 references
// of the scikit-learn / scipy definitions (TOFIX140, data transforms on the device).
//
// fixtures/data_transform_pytorch.json (generate_data_transform_fixtures.py, which
// also cross-checks scipy). Runs on the active ArrayFire backend
// (CYXWIZ_TEST_ARRAYFIRE_BACKEND=cuda|opencl|cpu).
#include "test_device_selection.h"

#include <cyxwiz/data_transform.h>

#include <nlohmann/json.hpp>

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

namespace {

using json = nlohmann::json;
using Columns = std::vector<std::vector<double>>;
using cyxwiz::DataTransform;
using cyxwiz::test::DeviceTolerance;

// float64 rounding, or float32 rounding on a device without float64 (worst
// measured on the Intel Iris Xe, TOFIX145 item 7: 1.5e-5, the normaltest K^2).
double Tolerance() { return DeviceTolerance(1e-8, 5e-5); }
// The maximum-likelihood lambda, and values transformed with it: the float32
// likelihood is flat to ~1e-7, which moves its optimum by ~5e-4.
double LambdaTolerance() { return DeviceTolerance(1e-7, 2e-3); }
double AutoTolerance() { return DeviceTolerance(1e-6, 2e-3); }

void Check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << std::endl;
        std::exit(1);
    }
}

void CheckValue(double actual, double expected, const std::string& what, double tolerance = Tolerance()) {
    Check(std::abs(actual - expected) <= tolerance * (1.0 + std::abs(expected)),
          what + ": " + std::to_string(actual) + ", PyTorch " + std::to_string(expected));
}

void CheckColumns(const cyxwiz::TransformResult& result, const json& expected, const std::string& what,
                  double tolerance = Tolerance()) {
    Check(result.success, what + ": " + result.error_message);
    const auto columns = expected.get<Columns>();
    Check(result.transformed_data.size() == columns.size(), what + ": column count");
    for (size_t c = 0; c < columns.size(); ++c) {
        Check(result.transformed_data[c].size() == columns[c].size(), what + ": column length");
        for (size_t i = 0; i < columns[c].size(); ++i) {
            CheckValue(result.transformed_data[c][i], columns[c][i],
                       what + " [" + std::to_string(c) + "][" + std::to_string(i) + "]", tolerance);
        }
    }
}

void CheckRefused(const std::string& error, const json& c, const std::string& what) {
    const std::string message = c.at("message").get<std::string>();
    Check(error.find(message) != std::string::npos,
          what + ": expected a refusal containing '" + message + "', got '" + error + "'");
}

std::filesystem::path FixturePath(const char* argv0) {
    const auto beside = std::filesystem::path(argv0).parent_path() / "computation_truth_fixtures" /
                        "data_transform_pytorch.json";
    if (std::filesystem::exists(beside)) return beside;
    return std::filesystem::path(CYXWIZ_DATA_TRANSFORM_FIXTURE);
}

void RunCase(const json& c, const std::string& name) {
    const std::string op = c.at("op").get<std::string>();
    const Columns inputs = c.at("inputs").get<Columns>();
    const std::vector<double>& x = inputs.front();

    if (op == "stats") {
        const auto stats = DataTransform::ComputeColumnStats(x);
        const json& e = c.at("stats");
        Check(stats.count == e.at("count").get<int>(), name + ": count (" + stats.error_message + ")");
        for (const auto& [field, value] : std::initializer_list<std::pair<const char*, double>>{
                 {"min", stats.min}, {"max", stats.max}, {"mean", stats.mean}, {"std_dev", stats.std_dev},
                 {"q1", stats.q1}, {"median", stats.median}, {"q3", stats.q3}, {"iqr", stats.iqr},
                 {"skewness", stats.skewness}}) {
            CheckValue(value, e.at(field).get<double>(), name + " " + field);
        }
        return;
    }
    if (op == "normalize") {
        const double low = c.value("range_min", 0.0), high = c.value("range_max", 1.0);
        return CheckColumns(DataTransform::Normalize(inputs, low, high), c.at("outputs"), name);
    }
    if (op == "standardize") {
        cyxwiz::StandardizeOptions options;
        options.with_mean = c.value("with_mean", true);
        options.with_std = c.value("with_std", true);
        return CheckColumns(DataTransform::Standardize(inputs, options), c.at("outputs"), name);
    }
    if (op == "scale_columns") {
        return CheckColumns(DataTransform::ScaleColumns(inputs, c.at("centers").get<std::vector<double>>(),
                                                        c.at("scales").get<std::vector<double>>()),
                            c.at("outputs"), name);
    }
    if (op == "outliers") {
        const auto result = DataTransform::FindOutliers(x, c.at("method").get<std::string>(), c.at("threshold").get<double>());
        Check(result.success, name + ": " + result.error_message);
        const auto flags = c.at("flags").get<std::vector<int>>();
        Check(result.is_outlier.size() == flags.size(), name + ": flag count");
        size_t count = 0;
        for (size_t i = 0; i < flags.size(); ++i) {
            Check(result.is_outlier[i] == flags[i], name + ": flag " + std::to_string(i));
            count += static_cast<size_t>(flags[i]);
        }
        return Check(result.count == count, name + ": outlier count");
    }
    if (op == "robust_scale") {
        cyxwiz::RobustScaleOptions options;
        options.quantile_min = c.value("quantile_min", 25.0);
        options.quantile_max = c.value("quantile_max", 75.0);
        options.with_centering = c.value("with_centering", true);
        options.with_scaling = c.value("with_scaling", true);
        return CheckColumns(DataTransform::RobustScale(inputs, options), c.at("outputs"), name);
    }
    if (op == "maxabs_scale") return CheckColumns(DataTransform::MaxAbsScale(inputs), c.at("outputs"), name);
    if (op == "log") {
        return CheckColumns(DataTransform::LogTransform(inputs, c.at("base").get<std::string>(),
                                                        c.at("use_log1p").get<bool>()),
                            c.at("outputs"), name);
    }
    if (op == "boxcox" || op == "yeojohnson") {
        const double lambda = c.at("lambda_in").get<double>();
        const auto result = op == "boxcox" ? DataTransform::BoxCoxColumn(x, lambda, false)
                                           : DataTransform::YeoJohnsonColumn(x, lambda, false);
        CheckColumns(result, c.at("outputs"), name);
        return CheckValue(result.params.at("lambda"), lambda, name + " lambda param");
    }
    if (op == "boxcox_auto" || op == "yeojohnson_auto") {
        const bool boxcox = op == "boxcox_auto";
        const double lambda = c.at("lambda").get<double>();
        const auto search = boxcox ? DataTransform::FindOptimalLambda(x) : DataTransform::FindOptimalYeoJohnsonLambda(x);
        Check(search.success, name + " search: " + search.error_message);
        CheckValue(search.optimal_lambda, lambda, name + " optimal lambda", LambdaTolerance());
        CheckValue(search.best_log_likelihood, c.at("log_likelihood").get<double>(), name + " log-likelihood");
        const auto result = boxcox ? DataTransform::BoxCoxColumn(x, 0.0, true) : DataTransform::YeoJohnsonColumn(x, 0.0, true);
        CheckColumns(result, c.at("outputs"), name, AutoTolerance());
        return CheckValue(result.params.at("lambda"), lambda, name + " lambda param", LambdaTolerance());
    }
    if (op == "boxcox_lambda_profile" || op == "yeojohnson_lambda_profile") {
        const auto search = op == "boxcox_lambda_profile" ? DataTransform::FindOptimalLambda(x, -5.0, 5.0, 100)
                                                          : DataTransform::FindOptimalYeoJohnsonLambda(x, -5.0, 5.0, 100);
        Check(search.success, name + ": " + search.error_message);
        const auto lambdas = c.at("lambdas").get<std::vector<double>>();
        const auto lls = c.at("log_likelihoods").get<std::vector<double>>();
        Check(search.lambdas_tested.size() == lambdas.size() && search.log_likelihoods.size() == lls.size(),
              name + ": profile length");
        for (size_t i = 0; i < lambdas.size(); ++i) {
            CheckValue(search.lambdas_tested[i], lambdas[i], name + " lambda[" + std::to_string(i) + "]");
            CheckValue(search.log_likelihoods[i], lls[i], name + " llf[" + std::to_string(i) + "]");
        }
        return;
    }
    if (op == "quantile") {
        const std::string distribution = c.at("distribution").get<std::string>();
        const int n_quantiles = c.at("n_quantiles").get<int>();
        CheckColumns(DataTransform::QuantileTransform(inputs, distribution, n_quantiles), c.at("outputs"), name);
        // Fitted and re-applied (the Quantile Transform node's saved state) gives the same answer.
        std::vector<std::vector<double>> fitted;
        for (const auto& column : inputs) {
            std::vector<double> quantiles;
            std::string error;
            Check(DataTransform::QuantileFit(column, n_quantiles, quantiles, error), name + " fit: " + error);
            fitted.push_back(std::move(quantiles));
        }
        return CheckColumns(DataTransform::QuantileApply(inputs, fitted, distribution), c.at("outputs"),
                            name + " (fit, then apply)");
    }
    if (op == "normaltest") {
        const auto test = DataTransform::DAgostinoPearsonTest(x);
        Check(test.computed, name + ": " + test.error_message);
        CheckValue(test.statistic, c.at("statistic").get<double>(), name + " K^2");
        CheckValue(test.p_value, c.at("p_value").get<double>(), name + " p");
        return Check(test.is_normal == (test.p_value > 0.05), name + ": is_normal");
    }
    if (op == "refuse_boxcox") {
        CheckRefused(DataTransform::BoxCoxColumn(x, 0.5, false).error_message, c, name);
        CheckRefused(DataTransform::FindOptimalLambda(x).error_message, c, name + " search");
        return Check(!DataTransform::CanApplyBoxCox(x), name + ": CanApplyBoxCox");
    }
    if (op == "refuse_log" || op == "refuse_log1p") {
        const bool log1p = op == "refuse_log1p";
        CheckRefused(DataTransform::LogTransform(inputs, "natural", log1p).error_message, c, name);
        return Check(!DataTransform::CanApplyLogTransform(x, log1p), name + ": CanApplyLogTransform");
    }
    if (op == "refuse_normaltest") {
        const auto test = DataTransform::DAgostinoPearsonTest(x);
        Check(!test.computed, name + ": must not compute");
        return CheckRefused(test.error_message, c, name);
    }
    Check(false, name + ": unknown op " + op);
}

}  // namespace

int main(int, char** argv) {
    Check(cyxwiz::test::SelectTestDeviceFromEnvironment(), "requested test device");
    std::ifstream in(FixturePath(argv[0]));
    Check(static_cast<bool>(in), "cannot open the data transform fixture");
    const json fixture = json::parse(in);
    size_t passed = 0;
    for (const auto& c : fixture.at("cases")) {
        const std::string name = c.at("op").get<std::string>() + " " + c.at("label").get<std::string>();
        try {
            RunCase(c, name);
        } catch (const std::exception& error) {
            Check(false, name + ": " + error.what());
        }
        ++passed;
    }
    std::cout << "data transforms match PyTorch: " << passed << " cases" << std::endl;
    return 0;
}
