#pragma once

#include "pipeline_operator.h"
#include <string>
#include <vector>

namespace cyxwiz {

/**
 * Shared base for all four clustering Cat-1 operators. Each subclass
 * runs one backend clustering algorithm; all emit `input + cluster_id
 * (int32)` so downstream nodes see the original columns plus the
 * cluster annotation.
 *
 * Closes the KMeansCluster / DBSCANCluster / HierarchicalCluster /
 * GMMCluster dead NodeTypes from the "Tool-to-Node Migration —
 * clustering" block in tofix.md.
 */
class ClusteringOperatorBase : public IPipelineOperator {
public:
    PipelineBand GetBand() const override { return PipelineBand::DataPrep; }

protected:
    std::vector<std::string> feature_cols_;
    std::string label_col_;

    // Shared Configure step — reads feature_cols (comma-sep, optional)
    // and label_col (optional, excluded from auto-detect). Returns true
    // on success. Subclasses call this first, then parse their own params.
    bool ConfigureBase(
        const std::map<std::string, std::string>& params,
        std::string& error);
};

/**
 * KMeansOperator — GPU-accelerated K-Means (ArrayFire backend).
 *
 * Params:
 *   feature_cols (optional, comma-sep)  — empty = auto-detect numeric.
 *   label_col    (optional)             — excluded from feature auto-detect.
 *   n_clusters   (default 8)            — k.
 *   max_iter     (default 300)          — EM iteration cap.
 *   init         (default "kmeans++")   — "random" / "kmeans++".
 *   n_init       (default 10)           — random restarts, best-inertia wins.
 *   tol          (default 1e-4)         — convergence tolerance.
 *   seed         (default 0)            — 0 = non-deterministic.
 */
class KMeansOperator : public ClusteringOperatorBase {
public:
    std::string GetName() const override { return "KMeansCluster"; }

    bool Configure(
        const std::map<std::string, std::string>& params,
        std::string& error) override;

    arrow::Result<std::shared_ptr<arrow::Table>> Apply(
        const std::shared_ptr<arrow::Table>& input) override;

private:
    int n_clusters_ = 8;
    int max_iter_ = 300;
    std::string init_ = "kmeans++";
    int n_init_ = 10;
    double tol_ = 1e-4;
    unsigned int seed_ = 0;
};

/**
 * DBSCANOperator — Density-Based Spatial Clustering.
 *
 * Params:
 *   feature_cols (optional, comma-sep)
 *   label_col    (optional)
 *   eps          (default 0.5)          — neighborhood radius.
 *   min_samples  (default 5)            — core-point threshold.
 *   metric       (default "euclidean")  — "euclidean"/"manhattan"/"cosine".
 *
 * Noise points get cluster_id = -1 (matches sklearn convention).
 */
class DBSCANOperator : public ClusteringOperatorBase {
public:
    std::string GetName() const override { return "DBSCANCluster"; }

    bool Configure(
        const std::map<std::string, std::string>& params,
        std::string& error) override;

    arrow::Result<std::shared_ptr<arrow::Table>> Apply(
        const std::shared_ptr<arrow::Table>& input) override;

private:
    double eps_ = 0.5;
    int min_samples_ = 5;
    std::string metric_ = "euclidean";
};

/**
 * HierarchicalOperator — Agglomerative clustering cut at n_clusters.
 *
 * Params:
 *   feature_cols (optional, comma-sep)
 *   label_col    (optional)
 *   n_clusters   (default 3)
 *   linkage      (default "ward")       — "ward"/"complete"/"average"/"single".
 *   metric       (default "euclidean")
 */
class HierarchicalOperator : public ClusteringOperatorBase {
public:
    std::string GetName() const override { return "HierarchicalCluster"; }

    bool Configure(
        const std::map<std::string, std::string>& params,
        std::string& error) override;

    arrow::Result<std::shared_ptr<arrow::Table>> Apply(
        const std::shared_ptr<arrow::Table>& input) override;

private:
    int n_clusters_ = 3;
    std::string linkage_ = "ward";
    std::string metric_ = "euclidean";
};

/**
 * GMMOperator — Gaussian Mixture Model with hard assignments.
 *
 * Params:
 *   feature_cols    (optional, comma-sep)
 *   label_col       (optional)
 *   n_components    (default 3)
 *   covariance_type (default "full")    — "full"/"tied"/"diag"/"spherical".
 *   max_iter        (default 100)
 *   tol             (default 1e-3)
 *   n_init          (default 1)
 *   seed            (default 0)
 *
 * v1 emits hard cluster assignments via cluster_id. Soft responsibilities
 * (per-component probabilities) are available in the backend result but
 * we don't expand them to columns — stays a Cat-2 inspection concern.
 */
class GMMOperator : public ClusteringOperatorBase {
public:
    std::string GetName() const override { return "GMMCluster"; }

    bool Configure(
        const std::map<std::string, std::string>& params,
        std::string& error) override;

    arrow::Result<std::shared_ptr<arrow::Table>> Apply(
        const std::shared_ptr<arrow::Table>& input) override;

private:
    int n_components_ = 3;
    std::string covariance_type_ = "full";
    int max_iter_ = 100;
    double tol_ = 1e-3;
    int n_init_ = 1;
    unsigned int seed_ = 0;
};

} // namespace cyxwiz
