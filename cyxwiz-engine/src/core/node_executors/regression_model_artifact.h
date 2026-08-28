#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace cyxwiz {

enum class RegressionModelType {
    Linear,
    Polynomial,
};

struct RegressionModelArtifact {
    RegressionModelType type = RegressionModelType::Linear;
    std::vector<std::string> feature_names;
    std::string target_name;
    bool fit_intercept = true;
    int degree = 1;
    std::vector<double> coefficients;
    size_t sample_count = 0;
    double r_squared = 0.0;
    double adjusted_r_squared = 0.0;
    double mse = 0.0;
    double rmse = 0.0;
    double mae = 0.0;
    double residual_variance = 0.0;
    double residual_standard_error = 0.0;
};

bool SaveRegressionModelArtifact(const RegressionModelArtifact& model,
                                 const std::string& path,
                                 std::string* error = nullptr);
bool LoadRegressionModelArtifact(const std::string& path,
                                 RegressionModelArtifact& model,
                                 std::string* error = nullptr);

} // namespace cyxwiz
