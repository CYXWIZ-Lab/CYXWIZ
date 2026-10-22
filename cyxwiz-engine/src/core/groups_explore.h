#pragma once

// The answers behind Data Studio's Explore tabs Groups and Projection
// (TOFIX129 D5, approved mockup d5_clustering_mockup.html). The chosen number
// columns are read through a QueryRunner (explore_read), rows missing a value
// are left out and counted, and the columns are standardised (z-scores, as
// scikit-learn's StandardScaler) unless asked not to. The clustering and the
// projections run on the device (cyxwiz::Clustering, DimensionalityReduction;
// their PyTorch / scikit-learn truth is test_computation_truth_clustering and
// _dimensionality_reduction). What is new here (adjusted Rand, the k-distance
// knee, the elbow pick, the top of the dendrogram, the group profiles) is
// checked against scikit-learn / scipy in test_groups_explore.

#include "data_explore.h"

#include <cyxwiz/clustering.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

namespace cyxwiz::groups {

enum class Method { KMeans, DBSCAN, Hierarchical, Mixture };

struct GroupsRequest {
    std::vector<std::string> columns;
    bool standardise = true;
    Method method = Method::KMeans;
    int k = 3;                       // K-Means, Hierarchical, Mixture
    uint32_t seed = 42;              // K-Means, Mixture (0 is taken as 42: a run always repeats)
    double eps = 0.5;                // DBSCAN
    int min_samples = 5;             // DBSCAN
    std::string metric = "euclidean";  // DBSCAN, Hierarchical
    std::string linkage = "ward";    // Hierarchical
    std::string covariance = "full"; // Mixture
    std::string compare;             // a column to compare the groups with ("" none)
};

// One group: its rows, the mean of each column (in the column's own units)
// and its mean silhouette.
struct GroupProfile {
    int label = 0;
    int64_t rows = 0;
    std::vector<double> means;
    double silhouette = NAN;
};

struct GroupsAnswer {
    std::string error;
    bool ok() const { return error.empty(); }
    GroupsRequest request;
    int64_t rows = 0;      // rows in the table
    int64_t used = 0;      // rows with every column
    std::vector<int> labels;            // per used row; -1 is DBSCAN's noise
    std::vector<double> x, y;           // the used rows on the first two principal components (the scatter)
    double x_ratio = NAN, y_ratio = NAN;  // their share of the variance
    std::vector<double> overall_means;  // each column over the used rows
    std::vector<double> overall_sds;
    std::vector<GroupProfile> groups;   // by label, ascending; noise not a group
    int noise = 0;
    // Scores (euclidean, on the data clustered); unset below two groups.
    bool scored = false;
    double silhouette = NAN, davies_bouldin = NAN, calinski_harabasz = NAN;
    // What the method says about itself.
    double inertia = NAN;
    int iterations = 0;
    bool converged = false;
    double cophenetic = NAN;
    double bic = NAN, aic = NAN, log_likelihood = NAN;
    std::vector<std::vector<double>> probabilities;  // Mixture: per used row, per group
    // Compared with a column: its values (sorted), the counts [group][value],
    // and how far they agree (adjusted Rand; noise rows left out).
    std::vector<std::string> compare_values;
    std::vector<std::vector<int64_t>> crosstab;
    double adjusted_rand = NAN;
};

GroupsAnswer RunGroups(const std::string& table, const GroupsRequest& request, const QueryRunner& run,
                       const explore::MissingTexts& texts);

// How many groups: for the method, a score per number of groups (K-Means:
// inertia and silhouette; Mixture: BIC and AIC; Hierarchical: silhouette over
// the cuts, and the tree), or DBSCAN's sorted k-distance curve and its knee.
struct HowManyRequest {
    GroupsRequest base;  // columns, scaling, method and its other settings
    int from = 2, to = 8;
};

struct HowManyAnswer {
    std::string error;
    bool ok() const { return error.empty(); }
    Method method = Method::KMeans;
    int64_t used = 0;
    std::vector<int> ks;
    std::vector<double> inertia, silhouette, bic, aic;
    std::vector<double> kdistances;  // DBSCAN: sorted ascending
    double knee = NAN;               // DBSCAN: the suggested eps
    std::vector<std::vector<double>> linkage;  // Hierarchical: scipy's linkage matrix
    int suggested = 0;               // the suggested number of groups (0: none)
};

HowManyAnswer RunHowMany(const std::string& table, const HowManyRequest& request, const QueryRunner& run,
                         const explore::MissingTexts& texts);

// Projection: PCA, t-SNE or UMAP of the chosen columns. t-SNE and UMAP read at
// most `max_rows` rows (a seeded random sample beyond, in table order).
enum class ProjectionMethod { PCA, TSNE, UMAP };

struct ProjectionRequest {
    std::vector<std::string> columns;
    bool standardise = true;
    ProjectionMethod method = ProjectionMethod::PCA;
    int perplexity = 30;
    double learning_rate = 200.0;
    int iterations = 1000;
    int neighbors = 15;
    double min_dist = 0.1;
    uint32_t seed = 42;
    int64_t max_rows = 5000;
    std::string colour;  // a column to colour the points by ("" none)
};

struct ProjectionAnswer {
    std::string error;
    bool ok() const { return error.empty(); }
    ProjectionRequest request;
    int64_t rows = 0, used = 0;
    bool sampled = false;            // t-SNE / UMAP on a sample of the used rows
    std::vector<double> x, y;
    std::vector<std::string> colour; // per point ("" when missing)
    bool colour_numeric = false;     // the colour column is a number (a scale, not groups)
    // PCA: each component's share of the variance (all of them) and the
    // loadings of the first two [component][column].
    std::vector<double> ratio;
    std::vector<std::vector<double>> loadings;
    double kl_divergence = NAN;      // t-SNE
};

ProjectionAnswer RunProjection(const std::string& table, const ProjectionRequest& request, const QueryRunner& run,
                               const explore::MissingTexts& texts);

// ---- Pure helpers (checked against scikit-learn / scipy) ----

// Adjusted Rand index of two labelings of the same rows (sklearn's
// adjusted_rand_score); NaN when they differ in length or are empty.
double AdjustedRand(const std::vector<int>& a, const std::vector<int>& b);

// The knee of an ascending curve (the Kneedle rule: the point furthest above
// the chord, both axes scaled to 0 .. 1): its value; NaN below 3 points.
double KneeOf(const std::vector<double>& ascending);

// The elbow of a falling curve (inertia by k): the k with the largest second
// difference (Clustering::ComputeElbowAnalysis's rule); 0 below 3 points.
int ElbowOf(const std::vector<int>& ks, const std::vector<double>& inertia);

// The top of a dendrogram (scipy's dendrogram with truncate_mode="lastp"):
// the last `leaves - 1` merges drawn as links over `leaves` leaves; x in leaf
// units (leaf i at 10 * i + 5, as scipy), y the merge heights.
struct DendrogramTop {
    std::vector<std::array<double, 4>> xs, ys;  // one link: the four corners
    std::vector<int64_t> leaf_rows;             // rows under each leaf, left to right
};
DendrogramTop TopOfTree(const std::vector<std::vector<double>>& linkage, int64_t samples, int leaves);

}  // namespace cyxwiz::groups
