#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cyxwiz/model_evaluation.h>
#include <nlohmann/json.hpp>

#include <cmath>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

namespace {

using json = nlohmann::json;

json LoadRegressionMetricsFixture() {
    std::ifstream stream(CYXWIZ_REGRESSION_METRICS_FIXTURE_PATH);
    REQUIRE(stream.is_open());
    json fixture;
    stream >> fixture;
    REQUIRE(fixture.value("schema_version", 0) == 1);
    REQUIRE(fixture.at("oracle").value("name", "") == "scikit-learn");
    REQUIRE(!fixture.at("oracle").value("version", "").empty());
    REQUIRE(fixture.value("dtype", "") == "float64");
    REQUIRE(fixture.value("mape_units", "") == "relative_ratio");
    return fixture;
}

void CheckMetric(double actual, const json& expected, double tolerance) {
    if (expected.is_null()) {
        CHECK(std::isnan(actual));
        return;
    }
    CHECK(actual == Catch::Approx(expected.get<double>()).margin(tolerance));
}

} // namespace

TEST_CASE("Regression evaluation metrics match scikit-learn fixtures",
          "[model_evaluation][regression_metrics][sklearn]") {
    const auto fixture = LoadRegressionMetricsFixture();
    const double tolerance = fixture.at("tolerance").get<double>();

    for (const auto& test_case : fixture.at("cases")) {
        INFO("case=" << test_case.at("name").get<std::string>());
        const auto result = cyxwiz::ModelEvaluation::ComputeRegressionMetrics(
            test_case.at("y_true").get<std::vector<double>>(),
            test_case.at("y_pred").get<std::vector<double>>());
        REQUIRE(result.success);
        REQUIRE(result.error_message.empty());
        const auto& expected = test_case.at("expected");
        CheckMetric(result.mse, expected.at("mse"), tolerance);
        CheckMetric(result.rmse, expected.at("rmse"), tolerance);
        CheckMetric(result.mae, expected.at("mae"), tolerance);
        CheckMetric(result.r_squared, expected.at("r_squared"), tolerance);
        CheckMetric(result.mape, expected.at("mape"), tolerance);
        CheckMetric(result.max_error, expected.at("max_error"), tolerance);
    }
}

TEST_CASE("Regression evaluation rejects invalid public inputs",
          "[model_evaluation][regression_metrics][contract]") {
    const auto empty =
        cyxwiz::ModelEvaluation::ComputeRegressionMetrics({}, {});
    CHECK_FALSE(empty.success);
    CHECK_FALSE(empty.error_message.empty());

    const auto mismatched =
        cyxwiz::ModelEvaluation::ComputeRegressionMetrics({1.0}, {1.0, 2.0});
    CHECK_FALSE(mismatched.success);
    CHECK_FALSE(mismatched.error_message.empty());

    const auto non_finite = cyxwiz::ModelEvaluation::ComputeRegressionMetrics(
        {1.0, std::numeric_limits<double>::quiet_NaN()}, {1.0, 2.0});
    CHECK_FALSE(non_finite.success);
    CHECK_FALSE(non_finite.error_message.empty());
}
