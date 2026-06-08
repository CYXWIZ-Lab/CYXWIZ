#include "clustering_operators.h"
#include "feature_matrix_utils.h"

#include <cyxwiz/clustering.h>

#include <arrow/api.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <cctype>
#include <set>
#include <string>

namespace cyxwiz {

namespace {

template <typename T>
bool ParseIntParam(const std::map<std::string, std::string>& params,
                   const std::string& key,
                   T& out,
                   const std::string& op_name,
                   std::string& error) {
    auto it = params.find(key);
    if (it == params.end() || it->second.empty()) return true;  // keep default
    try {
        out = static_cast<T>(std::stoll(it->second));
    } catch (...) {
        error = op_name + ": '" + key + "' is not a valid integer: " + it->second;
        return false;
    }
    return true;
}

bool ParseDoubleParam(const std::map<std::string, std::string>& params,
                      const std::string& key,
                      double& out,
                      const std::string& op_name,
                      std::string& error) {
    auto it = params.find(key);
    if (it == params.end() || it->second.empty()) return true;
    try {
        out = std::stod(it->second);
    } catch (...) {
        error = op_name + ": '" + key + "' is not a valid float: " + it->second;
        return false;
    }
    return true;
}

std::string TrimString(const std::string& value) {
    const auto first = std::find_if_not(value.begin(), value.end(), [](unsigned char c) {
        return std::isspace(c) != 0;
    });
    const auto last = std::find_if_not(value.rbegin(), value.rend(), [](unsigned char c) {
        return std::isspace(c) != 0;
    }).base();
    if (first >= last) return {};
    return std::string(first, last);
}

std::string ToLowerAscii(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return value;
}

bool ValidateChoice(const std::string& value,
                    const std::vector<std::string>& allowed,
                    const std::string& key,
                    const std::string& op_name,
                    std::string& error) {
    if (std::find(allowed.begin(), allowed.end(), value) != allowed.end()) {
        return true;
    }
    std::string list;
    for (size_t i = 0; i < allowed.size(); ++i) {
        if (i > 0) list += "/";
        list += "'" + allowed[i] + "'";
    }
    error = op_name + ": '" + key + "' must be " + list + " (got '" + value + "')";
    return false;
}

int CountUniqueClusters(const std::vector<int>& labels) {
    std::set<int> unique;
    for (int l : labels) {
        if (l >= 0) unique.insert(l);  // -1 = noise (DBSCAN), not a cluster
    }
    return static_cast<int>(unique.size());
}

} // namespace

// ============================================================================
// ClusteringOperatorBase
// ============================================================================

bool ClusteringOperatorBase::ConfigureBase(
    const std::map<std::string, std::string>& params,
    std::string& error) {
    feature_cols_.clear();
    label_col_.clear();

    auto fc = params.find("feature_cols");
    const std::string fc_str = (fc != params.end()) ? fc->second : "";
    ParseCommaList(fc_str, feature_cols_);

    auto lc = params.find("label_col");
    if (lc != params.end()) label_col_ = lc->second;

    (void)error;
    return true;
}

// ============================================================================
// KMeansOperator
// ============================================================================

bool KMeansOperator::Configure(
    const std::map<std::string, std::string>& params,
    std::string& error) {
    n_clusters_ = 8;
    max_iter_ = 300;
    init_ = "kmeans++";
    n_init_ = 10;
    tol_ = 1e-4;
    seed_ = 0;

    if (!ConfigureBase(params, error)) return false;

    if (!ParseIntParam(params, "n_clusters", n_clusters_, GetName(), error)) return false;
    if (!ParseIntParam(params, "max_iter", max_iter_, GetName(), error)) return false;
    if (!ParseIntParam(params, "n_init", n_init_, GetName(), error)) return false;
    if (!ParseDoubleParam(params, "tol", tol_, GetName(), error)) return false;

    int seed_tmp = 0;
    if (!ParseIntParam(params, "seed", seed_tmp, GetName(), error)) return false;
    seed_ = static_cast<unsigned int>(seed_tmp);

    auto it = params.find("init");
    if (it != params.end() && !it->second.empty()) {
        init_ = ToLowerAscii(TrimString(it->second));
        if (!ValidateChoice(init_, {"random", "kmeans++"}, "init", GetName(), error))
            return false;
    }

    if (n_clusters_ < 1) {
        error = GetName() + ": n_clusters must be >= 1 (got " +
                std::to_string(n_clusters_) + ")";
        return false;
    }
    return true;
}

arrow::Result<std::shared_ptr<arrow::Table>>
KMeansOperator::Apply(const std::shared_ptr<arrow::Table>& input) {
    if (!input) return arrow::Status::Invalid(GetName() + ": input table is null");

    std::vector<std::string> resolved;
    ARROW_RETURN_NOT_OK(ResolveFeatureColumns(
        input, feature_cols_, label_col_, GetName(), resolved));

    std::vector<std::vector<double>> matrix;
    int64_t n_samples = 0;
    ARROW_RETURN_NOT_OK(ReadFeatureMatrix(input, resolved, GetName(), matrix, n_samples));

    if (n_samples < n_clusters_) {
        return arrow::Status::Invalid(
            GetName() + ": n_samples (" + std::to_string(n_samples) +
            ") < n_clusters (" + std::to_string(n_clusters_) + ")");
    }

    auto result = Clustering::KMeans(
        matrix, n_clusters_, max_iter_, init_, n_init_, tol_, seed_);
    if (!result.success) {
        return arrow::Status::ExecutionError(
            GetName() + ": KMeans failed: " + result.error_message);
    }

    spdlog::info("KMeans: {} samples x {} features -> {} clusters, "
                 "inertia={:.4f}, iters={}, converged={}",
                 n_samples, resolved.size(), result.n_clusters,
                 result.inertia, result.n_iterations, result.converged);

    return AppendClusterIdColumn(input, result.labels);
}

// ============================================================================
// DBSCANOperator
// ============================================================================

bool DBSCANOperator::Configure(
    const std::map<std::string, std::string>& params,
    std::string& error) {
    eps_ = 0.5;
    min_samples_ = 5;
    metric_ = "euclidean";

    if (!ConfigureBase(params, error)) return false;

    if (!ParseDoubleParam(params, "eps", eps_, GetName(), error)) return false;
    if (!ParseIntParam(params, "min_samples", min_samples_, GetName(), error)) return false;

    auto it = params.find("metric");
    if (it != params.end() && !it->second.empty()) {
        metric_ = ToLowerAscii(TrimString(it->second));
        if (!ValidateChoice(metric_, {"euclidean", "manhattan", "cosine"},
                            "metric", GetName(), error))
            return false;
    }

    if (eps_ <= 0.0) {
        error = GetName() + ": eps must be > 0 (got " + std::to_string(eps_) + ")";
        return false;
    }
    if (min_samples_ < 1) {
        error = GetName() + ": min_samples must be >= 1 (got " +
                std::to_string(min_samples_) + ")";
        return false;
    }
    return true;
}

arrow::Result<std::shared_ptr<arrow::Table>>
DBSCANOperator::Apply(const std::shared_ptr<arrow::Table>& input) {
    if (!input) return arrow::Status::Invalid(GetName() + ": input table is null");

    std::vector<std::string> resolved;
    ARROW_RETURN_NOT_OK(ResolveFeatureColumns(
        input, feature_cols_, label_col_, GetName(), resolved));

    std::vector<std::vector<double>> matrix;
    int64_t n_samples = 0;
    ARROW_RETURN_NOT_OK(ReadFeatureMatrix(input, resolved, GetName(), matrix, n_samples));

    auto result = Clustering::DBSCAN(matrix, eps_, min_samples_, metric_);
    if (!result.success) {
        return arrow::Status::ExecutionError(
            GetName() + ": DBSCAN failed: " + result.error_message);
    }

    spdlog::info("DBSCAN: {} samples x {} features -> {} clusters, "
                 "{} noise points, eps={}, min_samples={}, metric={}",
                 n_samples, resolved.size(), result.n_clusters,
                 result.n_noise_points, eps_, min_samples_, metric_);

    return AppendClusterIdColumn(input, result.labels);
}

// ============================================================================
// HierarchicalOperator
// ============================================================================

bool HierarchicalOperator::Configure(
    const std::map<std::string, std::string>& params,
    std::string& error) {
    n_clusters_ = 3;
    linkage_ = "ward";
    metric_ = "euclidean";

    if (!ConfigureBase(params, error)) return false;

    if (!ParseIntParam(params, "n_clusters", n_clusters_, GetName(), error)) return false;

    auto lk = params.find("linkage");
    if (lk != params.end() && !lk->second.empty()) {
        linkage_ = ToLowerAscii(TrimString(lk->second));
        if (!ValidateChoice(linkage_, {"ward", "complete", "average", "single"},
                            "linkage", GetName(), error))
            return false;
    }

    auto m = params.find("metric");
    if (m != params.end() && !m->second.empty()) {
        metric_ = ToLowerAscii(TrimString(m->second));
        if (!ValidateChoice(metric_, {"euclidean", "manhattan", "cosine"},
                            "metric", GetName(), error))
            return false;
    }

    if (n_clusters_ < 1) {
        error = GetName() + ": n_clusters must be >= 1 (got " +
                std::to_string(n_clusters_) + ")";
        return false;
    }
    // Ward linkage is only valid with Euclidean distance.
    if (linkage_ == "ward" && metric_ != "euclidean") {
        error = GetName() + ": linkage='ward' requires metric='euclidean' (got '" +
                metric_ + "')";
        return false;
    }
    return true;
}

arrow::Result<std::shared_ptr<arrow::Table>>
HierarchicalOperator::Apply(const std::shared_ptr<arrow::Table>& input) {
    if (!input) return arrow::Status::Invalid(GetName() + ": input table is null");

    std::vector<std::string> resolved;
    ARROW_RETURN_NOT_OK(ResolveFeatureColumns(
        input, feature_cols_, label_col_, GetName(), resolved));

    std::vector<std::vector<double>> matrix;
    int64_t n_samples = 0;
    ARROW_RETURN_NOT_OK(ReadFeatureMatrix(input, resolved, GetName(), matrix, n_samples));

    if (n_samples < n_clusters_) {
        return arrow::Status::Invalid(
            GetName() + ": n_samples (" + std::to_string(n_samples) +
            ") < n_clusters (" + std::to_string(n_clusters_) + ")");
    }

    auto result = Clustering::Hierarchical(matrix, n_clusters_, linkage_, metric_);
    if (!result.success) {
        return arrow::Status::ExecutionError(
            GetName() + ": Hierarchical failed: " + result.error_message);
    }

    spdlog::info("Hierarchical: {} samples x {} features -> {} clusters, "
                 "linkage={}, metric={}, cophenetic_corr={:.4f}",
                 n_samples, resolved.size(), result.n_clusters,
                 linkage_, metric_, result.cophenetic_correlation);

    return AppendClusterIdColumn(input, result.labels);
}

// ============================================================================
// GMMOperator
// ============================================================================

bool GMMOperator::Configure(
    const std::map<std::string, std::string>& params,
    std::string& error) {
    n_components_ = 3;
    covariance_type_ = "full";
    max_iter_ = 100;
    tol_ = 1e-3;
    n_init_ = 1;
    seed_ = 0;

    if (!ConfigureBase(params, error)) return false;

    if (!ParseIntParam(params, "n_components", n_components_, GetName(), error)) return false;
    if (!ParseIntParam(params, "max_iter", max_iter_, GetName(), error)) return false;
    if (!ParseDoubleParam(params, "tol", tol_, GetName(), error)) return false;
    if (!ParseIntParam(params, "n_init", n_init_, GetName(), error)) return false;

    int seed_tmp = 0;
    if (!ParseIntParam(params, "seed", seed_tmp, GetName(), error)) return false;
    seed_ = static_cast<unsigned int>(seed_tmp);

    auto c = params.find("covariance_type");
    if (c != params.end() && !c->second.empty()) {
        covariance_type_ = ToLowerAscii(TrimString(c->second));
        if (!ValidateChoice(covariance_type_,
                            {"full", "tied", "diag", "spherical"},
                            "covariance_type", GetName(), error))
            return false;
    }

    if (n_components_ < 1) {
        error = GetName() + ": n_components must be >= 1 (got " +
                std::to_string(n_components_) + ")";
        return false;
    }
    return true;
}

arrow::Result<std::shared_ptr<arrow::Table>>
GMMOperator::Apply(const std::shared_ptr<arrow::Table>& input) {
    if (!input) return arrow::Status::Invalid(GetName() + ": input table is null");

    std::vector<std::string> resolved;
    ARROW_RETURN_NOT_OK(ResolveFeatureColumns(
        input, feature_cols_, label_col_, GetName(), resolved));

    std::vector<std::vector<double>> matrix;
    int64_t n_samples = 0;
    ARROW_RETURN_NOT_OK(ReadFeatureMatrix(input, resolved, GetName(), matrix, n_samples));

    if (n_samples < n_components_) {
        return arrow::Status::Invalid(
            GetName() + ": n_samples (" + std::to_string(n_samples) +
            ") < n_components (" + std::to_string(n_components_) + ")");
    }

    auto result = Clustering::GMM(
        matrix, n_components_, covariance_type_, max_iter_, tol_, n_init_, seed_);
    if (!result.success) {
        return arrow::Status::ExecutionError(
            GetName() + ": GMM failed: " + result.error_message);
    }

    const int effective_k = CountUniqueClusters(result.labels);
    spdlog::info("GMM: {} samples x {} features -> {} components "
                 "(effective k={}), covariance={}, log_lik={:.4f}, "
                 "BIC={:.4f}, AIC={:.4f}, iters={}, converged={}",
                 n_samples, resolved.size(), result.n_components, effective_k,
                 covariance_type_, result.log_likelihood, result.bic, result.aic,
                 result.n_iterations, result.converged);

    return AppendClusterIdColumn(input, result.labels);
}

} // namespace cyxwiz
