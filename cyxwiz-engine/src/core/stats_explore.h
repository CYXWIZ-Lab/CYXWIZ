#pragma once

// The answers behind Data Studio's Explore tabs Tests and Regression
// (TOFIX129 D3, approved mockup d3_statistics_mockup.html). The chosen columns
// are read through a QueryRunner, so tables kept on disk work the same way. A
// value is missing when it is NULL, is not a number (number columns), or is
// one of the texts the project marks as missing for its column. The tests and
// fits run on the device (cyxwiz::StatisticalTests, checked against scipy and
// statsmodels).

#include "data_explore.h"

#include <cyxwiz/statistical_tests.h>

#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

namespace cyxwiz::stats {

enum class Question { TwoGroups, Paired, SeveralGroups, OneValue, TwoCategories };

inline constexpr size_t kMaxGroups = 20;  // more values than this in a group column is refused

struct TestRequest {
    Question question = Question::TwoGroups;
    std::string values;       // the number column (paired: the first one)
    std::string groups;       // the group column (two / several groups); two categories: the rows
    std::string second;       // paired, or two groups as two columns: the second number column;
                              // two categories: the columns
    bool two_columns = false; // two groups given as two number columns
    bool welch = true;        // unequal variances (two / several groups)
    double confidence = 0.95; // 1 - alpha
    double value = 0.0;       // one value: what the mean is compared with
};

struct GroupFigures {
    std::string name;
    int64_t n = 0;
    double mean = NAN, sd = NAN;
    double min = NAN, q1 = NAN, median = NAN, q3 = NAN, max = NAN;  // the box plot
    double normal_p = NAN;  // D'Agostino-Pearson; NaN below 20 values
    std::vector<double> values;
};

struct TestAnswer {
    std::string error;
    bool ok() const { return error.empty(); }
    TestRequest request;
    int64_t rows = 0;        // rows in the table
    int64_t left_out = 0;    // rows without the values (or group) the test needs
    std::vector<GroupFigures> groups;  // paired: the first column, the second, the differences
    TestResult main;         // the test asked for
    TestResult classic;      // several groups with Welch: the classic F, for comparison
    TestResult rank;         // Mann-Whitney U / Kruskal-Wallis beside it
    TestResult variances;    // Levene (two / several groups)
    std::vector<TukeyPair> pairs;  // several groups: which pairs differ
    // Two categories.
    std::vector<std::string> row_names, col_names;
    std::vector<std::vector<double>> counts, expected;
    double min_expected = NAN;
};

TestAnswer RunTest(const std::string& table, const TestRequest& request, const QueryRunner& run,
                   const explore::MissingTexts& texts);

struct RegressionRequest {
    std::string response;
    std::vector<std::string> predictors;
    bool polynomial = false;  // one predictor, powers 1 .. degree
    int degree = 2;
    double confidence = 0.95;
};

struct RegressionAnswer {
    std::string error;
    bool ok() const { return error.empty(); }
    RegressionRequest request;
    LinearFit fit;
    LinearFit line;     // polynomial: the straight line, for comparison
    LinearFit reduced;  // several predictors, some whose CI includes 0: the fit on the others
    std::vector<std::string> reduced_names;
    int64_t rows = 0, used = 0;
    std::vector<double> x;  // one predictor: its values (plots)
    std::vector<double> y;
    std::vector<std::vector<double>> columns;  // the predictors' values, rows used (export)
    double residual_normal_p = NAN;
};

RegressionAnswer RunRegression(const std::string& table, const RegressionRequest& request, const QueryRunner& run,
                               const explore::MissingTexts& texts);

}  // namespace cyxwiz::stats
