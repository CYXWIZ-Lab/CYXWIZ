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

json LoadClassificationMetricsFixture() {
    std::ifstream stream(CYXWIZ_CLASSIFICATION_METRICS_FIXTURE_PATH);
    REQUIRE(stream.is_open());
    json fixture;
    stream >> fixture;
    REQUIRE(fixture.value("schema_version", 0) == 1);
    REQUIRE(fixture.at("oracles").at("metrics").value("name", "") ==
            "scikit-learn");
    REQUIRE(fixture.at("oracles").at("decisions").value("name", "") ==
            "PyTorch");
    REQUIRE(!fixture.at("oracles").at("metrics").value("version", "").empty());
    REQUIRE(!fixture.at("oracles").at("decisions").value("version", "").empty());
    REQUIRE(fixture.value("dtype", "") == "float64");
    REQUIRE(fixture.value("zero_division", -1) == 0);
    REQUIRE_FALSE(fixture.value("roc_drop_intermediate", true));
    REQUIRE(fixture.value("pr_average", "") ==
            "non-interpolated-average-precision");
    return fixture;
}

void CheckMetric(double actual, const json& expected, double tolerance) {
    if (expected.is_null()) {
        CHECK(std::isnan(actual));
        return;
    }
    CHECK(actual == Catch::Approx(expected.get<double>()).margin(tolerance));
}

void CheckVector(const std::vector<double>& actual,
                 const json& expected,
                 double tolerance,
                 bool null_is_positive_infinity = false) {
    REQUIRE(actual.size() == expected.size());
    for (size_t index = 0; index < actual.size(); ++index) {
        CAPTURE(index);
        if (expected.at(index).is_null()) {
            if (null_is_positive_infinity) {
                CHECK(std::isinf(actual[index]));
                CHECK(actual[index] > 0.0);
            } else {
                CHECK(std::isnan(actual[index]));
            }
        } else {
            CHECK(actual[index] ==
                  Catch::Approx(expected.at(index).get<double>())
                      .margin(tolerance));
        }
    }
}

void CheckNestedVectors(const std::vector<std::vector<double>>& actual,
                        const json& expected,
                        double tolerance,
                        bool null_is_positive_infinity = false) {
    REQUIRE(actual.size() == expected.size());
    for (size_t index = 0; index < actual.size(); ++index) {
        CAPTURE(index);
        CheckVector(actual[index], expected.at(index), tolerance,
                    null_is_positive_infinity);
    }
}

void CheckConfusionMatrix(const cyxwiz::ConfusionMatrixData& actual,
                          const json& expected,
                          double tolerance) {
    REQUIRE(actual.success);
    REQUIRE(actual.error_message.empty());
    CHECK(actual.matrix == expected.at("matrix").get<std::vector<std::vector<int>>>());
    CHECK(actual.class_names ==
          expected.at("class_names").get<std::vector<std::string>>());
    CHECK(actual.n_classes == static_cast<int>(expected.at("labels").size()));
    CHECK(actual.total_samples == expected.at("total_samples").get<int>());
    CheckMetric(actual.accuracy, expected.at("accuracy"), tolerance);
    CheckMetric(actual.macro_precision, expected.at("macro_precision"), tolerance);
    CheckMetric(actual.macro_recall, expected.at("macro_recall"), tolerance);
    CheckMetric(actual.macro_f1, expected.at("macro_f1"), tolerance);
    CheckMetric(actual.weighted_f1, expected.at("weighted_f1"), tolerance);
    CheckVector(actual.precision, expected.at("precision"), tolerance);
    CheckVector(actual.recall, expected.at("recall"), tolerance);
    CheckVector(actual.f1_scores, expected.at("f1"), tolerance);
    CHECK(actual.support == expected.at("support").get<std::vector<int>>());
}

void CheckBinaryMetrics(const cyxwiz::BinaryMetrics& actual,
                        const json& expected,
                        double tolerance) {
    REQUIRE(actual.success);
    REQUIRE(actual.error_message.empty());
    CHECK(actual.tp == expected.at("tp").get<int>());
    CHECK(actual.fp == expected.at("fp").get<int>());
    CHECK(actual.tn == expected.at("tn").get<int>());
    CHECK(actual.fn == expected.at("fn").get<int>());
    CheckMetric(actual.precision, expected.at("precision"), tolerance);
    CheckMetric(actual.recall, expected.at("recall"), tolerance);
    CheckMetric(actual.specificity, expected.at("specificity"), tolerance);
    CheckMetric(actual.f1, expected.at("f1"), tolerance);
    CheckMetric(actual.balanced_accuracy,
                expected.at("balanced_accuracy"), tolerance);
    CheckMetric(actual.mcc, expected.at("mcc"), tolerance);
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

TEST_CASE("Classification reports match sklearn fixtures",
          "[model_evaluation][classification_metrics][sklearn]") {
    const auto fixture = LoadClassificationMetricsFixture();
    const double tolerance = fixture.at("tolerance").get<double>();

    for (const auto& test_case : fixture.at("confusion_cases")) {
        INFO("case=" << test_case.at("name").get<std::string>());
        const auto y_true = test_case.at("y_true").get<std::vector<int>>();
        const auto y_pred = test_case.at("y_pred").get<std::vector<int>>();
        const auto class_names = test_case.contains("class_names")
            ? test_case.at("class_names").get<std::vector<std::string>>()
            : std::vector<std::string>{};
        const auto result = cyxwiz::ModelEvaluation::ComputeConfusionMatrix(
            y_true, y_pred, class_names);
        CheckConfusionMatrix(result, test_case.at("expected"), tolerance);

        const auto report = cyxwiz::ModelEvaluation::GenerateClassificationReport(
            y_true, y_pred, class_names);
        REQUIRE(report.success);
        REQUIRE(report.error_message.empty());
        CheckConfusionMatrix(
            report.confusion_matrix, test_case.at("expected"), tolerance);
        const auto& expected = test_case.at("expected");
        CheckMetric(report.overall_metrics.at("accuracy"),
                    expected.at("accuracy"), tolerance);
        CheckMetric(report.overall_metrics.at("macro_precision"),
                    expected.at("macro_precision"), tolerance);
        CheckMetric(report.overall_metrics.at("macro_recall"),
                    expected.at("macro_recall"), tolerance);
        CheckMetric(report.overall_metrics.at("macro_f1"),
                    expected.at("macro_f1"), tolerance);
        CheckMetric(report.overall_metrics.at("weighted_f1"),
                    expected.at("weighted_f1"), tolerance);
        for (size_t index = 0; index < result.class_names.size(); ++index) {
            CAPTURE(index);
            const auto& metrics = report.per_class_metrics.at(
                result.class_names[index]);
            CheckMetric(metrics.at("precision"),
                        expected.at("precision").at(index), tolerance);
            CheckMetric(metrics.at("recall"),
                        expected.at("recall").at(index), tolerance);
            CheckMetric(metrics.at("f1"),
                        expected.at("f1").at(index), tolerance);
            CheckMetric(metrics.at("support"),
                        expected.at("support").at(index), tolerance);
        }
    }
}

TEST_CASE("Binary classification metrics match sklearn and PyTorch decisions",
          "[model_evaluation][classification_metrics][sklearn][pytorch]") {
    const auto fixture = LoadClassificationMetricsFixture();
    const double tolerance = fixture.at("tolerance").get<double>();

    for (const auto& test_case : fixture.at("binary_cases")) {
        INFO("case=" << test_case.at("name").get<std::string>());
        if (test_case.contains("pytorch_y_pred")) {
            CHECK(test_case.at("pytorch_y_pred") ==
                  test_case.at("expected").at("y_pred"));
        }
        const auto result = cyxwiz::ModelEvaluation::ComputeBinaryMetrics(
            test_case.at("y_true").get<std::vector<int>>(),
            test_case.at("y_scores").get<std::vector<double>>(),
            test_case.at("threshold").get<double>());
        CHECK(result.threshold == test_case.at("threshold").get<double>());
        CheckBinaryMetrics(result, test_case.at("expected"), tolerance);
    }
}

TEST_CASE("ROC and precision-recall curves match sklearn fixtures",
          "[model_evaluation][classification_metrics][curves][sklearn]") {
    const auto fixture = LoadClassificationMetricsFixture();
    const double tolerance = fixture.at("tolerance").get<double>();

    for (const auto& test_case : fixture.at("curve_cases")) {
        INFO("case=" << test_case.at("name").get<std::string>());
        const auto y_true = test_case.at("y_true").get<std::vector<int>>();
        const auto y_scores =
            test_case.at("y_scores").get<std::vector<double>>();
        const auto& expected = test_case.at("expected");

        const auto roc = cyxwiz::ModelEvaluation::ComputeROC(y_true, y_scores);
        REQUIRE(roc.success);
        REQUIRE(roc.error_message.empty());
        CheckVector(roc.fpr, expected.at("roc").at("fpr"), tolerance);
        CheckVector(roc.tpr, expected.at("roc").at("tpr"), tolerance);
        CheckVector(roc.thresholds, expected.at("roc").at("thresholds"),
                    tolerance, true);
        CheckMetric(roc.auc, expected.at("roc").at("auc"), tolerance);

        const auto pr =
            cyxwiz::ModelEvaluation::ComputePRCurve(y_true, y_scores);
        REQUIRE(pr.success);
        REQUIRE(pr.error_message.empty());
        CheckVector(pr.precision, expected.at("pr").at("precision"), tolerance);
        CheckVector(pr.recall, expected.at("pr").at("recall"), tolerance);
        CheckVector(pr.thresholds, expected.at("pr").at("thresholds"), tolerance);
        CheckMetric(pr.average_precision,
                    expected.at("pr").at("average_precision"), tolerance);
        CHECK(pr.precision.size() == pr.thresholds.size() + 1);
        CHECK(pr.recall.size() == pr.thresholds.size() + 1);
    }
}

TEST_CASE("Multiclass curves match sklearn after PyTorch score adaptation",
          "[model_evaluation][classification_metrics][multiclass][sklearn][pytorch]") {
    const auto fixture = LoadClassificationMetricsFixture();
    const double tolerance = fixture.at("tolerance").get<double>();

    for (const auto& test_case : fixture.at("multiclass_cases")) {
        INFO("case=" << test_case.at("name").get<std::string>());
        const auto y_true = test_case.at("y_true").get<std::vector<int>>();
        const auto y_pred = test_case.at("y_pred").get<std::vector<int>>();
        const auto y_scores =
            test_case.at("y_scores").get<std::vector<std::vector<double>>>();
        const auto& expected = test_case.at("expected");

        CheckConfusionMatrix(
            cyxwiz::ModelEvaluation::ComputeConfusionMatrix(y_true, y_pred),
            expected.at("confusion"), tolerance);

        const auto roc =
            cyxwiz::ModelEvaluation::ComputeMulticlassROC(y_true, y_scores);
        REQUIRE(roc.success);
        REQUIRE(roc.error_message.empty());
        CheckNestedVectors(roc.class_fpr,
                           expected.at("roc").at("class_fpr"), tolerance);
        CheckNestedVectors(roc.class_tpr,
                           expected.at("roc").at("class_tpr"), tolerance);
        CheckNestedVectors(roc.class_thresholds,
                           expected.at("roc").at("class_thresholds"),
                           tolerance, true);
        CheckVector(roc.class_auc,
                    expected.at("roc").at("class_auc"), tolerance);
        CheckMetric(roc.auc, expected.at("roc").at("auc"), tolerance);

        const auto pr =
            cyxwiz::ModelEvaluation::ComputeMulticlassPRCurve(y_true, y_scores);
        REQUIRE(pr.success);
        REQUIRE(pr.error_message.empty());
        CheckNestedVectors(pr.class_precision,
                           expected.at("pr").at("class_precision"), tolerance);
        CheckNestedVectors(pr.class_recall,
                           expected.at("pr").at("class_recall"), tolerance);
        CheckNestedVectors(pr.class_thresholds,
                           expected.at("pr").at("class_thresholds"),
                           tolerance);
        CheckVector(pr.class_ap, expected.at("pr").at("class_ap"), tolerance);
        CheckMetric(pr.average_precision,
                    expected.at("pr").at("average_precision"), tolerance);
    }
}

TEST_CASE("Threshold selection and AUC match declared sklearn contracts",
          "[model_evaluation][classification_metrics][threshold][sklearn]") {
    const auto fixture = LoadClassificationMetricsFixture();
    const double tolerance = fixture.at("tolerance").get<double>();

    for (const auto& test_case : fixture.at("threshold_cases")) {
        const auto y_true = test_case.at("y_true").get<std::vector<int>>();
        const auto y_scores =
            test_case.at("y_scores").get<std::vector<double>>();
        for (const auto& item : test_case.at("expected").items()) {
            INFO("case=" << test_case.at("name").get<std::string>()
                          << ", criterion=" << item.key());
            CHECK(cyxwiz::ModelEvaluation::FindOptimalThreshold(
                      y_true, y_scores, item.key()) ==
                  Catch::Approx(item.value().get<double>()).margin(tolerance));
        }
    }

    for (const auto& test_case : fixture.at("auc_cases")) {
        INFO("case=" << test_case.at("name").get<std::string>());
        CHECK(cyxwiz::ModelEvaluation::ComputeAUC(
                  test_case.at("x").get<std::vector<double>>(),
                  test_case.at("y").get<std::vector<double>>()) ==
              Catch::Approx(test_case.at("expected").get<double>())
                  .margin(tolerance));
    }
}

TEST_CASE("Classification evaluation rejects invalid public inputs",
          "[model_evaluation][classification_metrics][contract]") {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const double inf = std::numeric_limits<double>::infinity();

    const auto empty_confusion =
        cyxwiz::ModelEvaluation::ComputeConfusionMatrix({}, {});
    CHECK_FALSE(empty_confusion.success);
    CHECK_FALSE(empty_confusion.error_message.empty());
    const auto mismatched_confusion =
        cyxwiz::ModelEvaluation::ComputeConfusionMatrix({0}, {0, 1});
    CHECK_FALSE(mismatched_confusion.success);
    CHECK_FALSE(mismatched_confusion.error_message.empty());
    const auto mismatched_names = cyxwiz::ModelEvaluation::ComputeConfusionMatrix(
        {0, 1}, {0, 1}, {"only-one-name"});
    CHECK_FALSE(mismatched_names.success);
    CHECK_FALSE(mismatched_names.error_message.empty());

    for (const auto& result : {
             cyxwiz::ModelEvaluation::ComputeBinaryMetrics({}, {}),
             cyxwiz::ModelEvaluation::ComputeBinaryMetrics({0}, {0.1, 0.2}),
             cyxwiz::ModelEvaluation::ComputeBinaryMetrics({2}, {0.1}),
             cyxwiz::ModelEvaluation::ComputeBinaryMetrics({0}, {nan}),
             cyxwiz::ModelEvaluation::ComputeBinaryMetrics({0}, {0.1}, inf),
         }) {
        CHECK_FALSE(result.success);
        CHECK_FALSE(result.error_message.empty());
    }

    for (const auto& result : {
             cyxwiz::ModelEvaluation::ComputeROC({}, {}),
             cyxwiz::ModelEvaluation::ComputeROC({0}, {0.1, 0.2}),
             cyxwiz::ModelEvaluation::ComputeROC({0, 2}, {0.1, 0.2}),
             cyxwiz::ModelEvaluation::ComputeROC({0, 1}, {0.1, nan}),
             cyxwiz::ModelEvaluation::ComputeROC({1, 1}, {0.1, 0.2}),
         }) {
        CHECK_FALSE(result.success);
        CHECK_FALSE(result.error_message.empty());
    }

    for (const auto& result : {
             cyxwiz::ModelEvaluation::ComputePRCurve({}, {}),
             cyxwiz::ModelEvaluation::ComputePRCurve({0}, {0.1, 0.2}),
             cyxwiz::ModelEvaluation::ComputePRCurve({0, 2}, {0.1, 0.2}),
             cyxwiz::ModelEvaluation::ComputePRCurve({0, 1}, {0.1, nan}),
             cyxwiz::ModelEvaluation::ComputePRCurve({0, 0}, {0.1, 0.2}),
         }) {
        CHECK_FALSE(result.success);
        CHECK_FALSE(result.error_message.empty());
    }

    for (const auto& result : {
             cyxwiz::ModelEvaluation::ComputeMulticlassROC(
                 {0, 1}, {{0.9, 0.1}}),
             cyxwiz::ModelEvaluation::ComputeMulticlassROC(
                 {0, 1}, {{0.9, 0.1}, {0.2}}),
             cyxwiz::ModelEvaluation::ComputeMulticlassROC(
                 {0, 2}, {{0.9, 0.1}, {0.2, 0.8}}),
             cyxwiz::ModelEvaluation::ComputeMulticlassROC(
                 {0, 1}, {{0.9, 0.1}, {nan, 0.8}}),
         }) {
        CHECK_FALSE(result.success);
        CHECK_FALSE(result.error_message.empty());
    }
    for (const auto& result : {
             cyxwiz::ModelEvaluation::ComputeMulticlassPRCurve(
                 {0, 1}, {{0.9, 0.1}}),
             cyxwiz::ModelEvaluation::ComputeMulticlassPRCurve(
                 {0, 1}, {{0.9, 0.1}, {0.2}}),
             cyxwiz::ModelEvaluation::ComputeMulticlassPRCurve(
                 {0, 2}, {{0.9, 0.1}, {0.2, 0.8}}),
             cyxwiz::ModelEvaluation::ComputeMulticlassPRCurve(
                 {0, 1}, {{0.9, 0.1}, {nan, 0.8}}),
         }) {
        CHECK_FALSE(result.success);
        CHECK_FALSE(result.error_message.empty());
    }

    CHECK(std::isnan(cyxwiz::ModelEvaluation::ComputeAUC({}, {})));
    CHECK(std::isnan(cyxwiz::ModelEvaluation::ComputeAUC(
        {0.0, 1.0, 0.5}, {0.0, 1.0, 0.5})));
    CHECK(std::isnan(cyxwiz::ModelEvaluation::ComputeAUC(
        {0.0, inf}, {0.0, 1.0})));
    CHECK(std::isnan(cyxwiz::ModelEvaluation::FindOptimalThreshold(
        {0}, {0.1, 0.2}, "f1")));
    CHECK(std::isnan(cyxwiz::ModelEvaluation::FindOptimalThreshold(
        {0}, {nan}, "f1")));
    CHECK(std::isnan(cyxwiz::ModelEvaluation::FindOptimalThreshold(
        {0}, {0.1}, "unknown")));
}
