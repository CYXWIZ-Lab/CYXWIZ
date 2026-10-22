#include "groups_explore.h"

#include "explore_read.h"

#include <cyxwiz/dimensionality_reduction.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <map>
#include <numeric>
#include <optional>
#include <random>
#include <set>

namespace cyxwiz::groups {

namespace {

using explore_read::LabelExpr;
using explore_read::Labels;
using explore_read::NumberExpr;
using explore_read::Numbers;

// The chosen columns of the rows that have them all, in table order, and a
// label column beside them (compare / colour; empty when not asked).
struct Matrix {
    std::string error;
    int64_t rows = 0;
    std::vector<std::vector<double>> raw;   // [row][column]
    std::vector<std::vector<double>> data;  // standardised (or raw)
    std::vector<std::optional<std::string>> label;
    std::vector<double> means, sds;
};

Matrix ReadMatrix(const std::string& table, const std::vector<std::string>& columns, bool standardise,
                  const std::string& label_column, const QueryRunner& run, const explore::MissingTexts& texts,
                  const char* what) {
    Matrix m;
    if (columns.empty()) {
        m.error = "Add the number columns to use.";
        return m;
    }
    std::vector<QueryParam> params;
    std::vector<std::string> exprs;
    for (size_t i = 0; i < columns.size(); ++i) exprs.push_back(NumberExpr(columns[i], texts, params, "c" + std::to_string(i)));
    if (!label_column.empty()) exprs.push_back(LabelExpr(label_column, texts, params, "label"));
    const auto read = explore_read::Select(table, exprs, std::move(params), run, what);
    if (!read.table) {
        m.error = read.error;
        return m;
    }
    m.rows = read.table->num_rows();
    std::vector<std::vector<std::optional<double>>> cols;
    for (size_t i = 0; i < columns.size(); ++i) cols.push_back(Numbers(*read.table, "c" + std::to_string(i)));
    const auto labels = label_column.empty() ? std::vector<std::optional<std::string>>{} : Labels(*read.table, "label");
    for (int64_t r = 0; r < m.rows; ++r) {
        std::vector<double> row;
        row.reserve(columns.size());
        for (const auto& c : cols) {
            if (static_cast<size_t>(r) >= c.size() || !c[static_cast<size_t>(r)]) break;
            row.push_back(*c[static_cast<size_t>(r)]);
        }
        if (row.size() != columns.size()) continue;
        m.raw.push_back(std::move(row));
        if (!label_column.empty()) m.label.push_back(static_cast<size_t>(r) < labels.size() ? labels[static_cast<size_t>(r)] : std::nullopt);
    }
    const size_t n = m.raw.size(), d = columns.size();
    m.means.assign(d, 0.0);
    m.sds.assign(d, 0.0);
    for (const auto& row : m.raw)
        for (size_t j = 0; j < d; ++j) m.means[j] += row[j];
    for (auto& v : m.means) v /= std::max<size_t>(n, 1);
    for (const auto& row : m.raw)
        for (size_t j = 0; j < d; ++j) m.sds[j] += (row[j] - m.means[j]) * (row[j] - m.means[j]);
    for (auto& v : m.sds) v = std::sqrt(v / std::max<size_t>(n, 1));  // population SD, as StandardScaler
    m.data = m.raw;
    if (standardise)
        for (auto& row : m.data)
            for (size_t j = 0; j < d; ++j) row[j] = m.sds[j] > 0.0 ? (row[j] - m.means[j]) / m.sds[j] : 0.0;
    for (size_t j = 0; j < d; ++j)
        if (!(m.sds[j] > 0.0)) {
            m.error = columns[j] + " has the same value in every row used, so it cannot tell rows apart.";
            return m;
        }
    return m;
}

uint32_t Seed(uint32_t seed) { return seed == 0 ? 42u : seed; }

// Labels after cutting a linkage into k groups (the cut between the merges
// that leave k and k - 1 groups).
std::vector<int> CutInto(const std::vector<std::vector<double>>& linkage, int64_t samples, int k) {
    const int64_t merges = static_cast<int64_t>(linkage.size());
    if (k < 1 || k > samples) return {};
    const int64_t below = samples - k - 1;  // the last merge kept
    const double low = below >= 0 && below < merges ? linkage[static_cast<size_t>(below)][2] : 0.0;
    const double high = below + 1 < merges ? linkage[static_cast<size_t>(below + 1)][2] : low + 1.0;
    return Clustering::CutDendrogram(linkage, (low + high) / 2.0, static_cast<int>(samples));
}

}  // namespace

GroupsAnswer RunGroups(const std::string& table, const GroupsRequest& req, const QueryRunner& run,
                       const explore::MissingTexts& texts) {
    GroupsAnswer a;
    a.request = req;
    Matrix m = ReadMatrix(table, req.columns, req.standardise, req.compare, run, texts, "Groups: read the columns");
    a.rows = m.rows;
    a.used = static_cast<int64_t>(m.raw.size());
    if (!m.error.empty()) return a.error = m.error, a;
    const bool counted = req.method != Method::DBSCAN;
    if (counted && (req.k < 2 || req.k > 50)) return a.error = "The number of groups must be 2 to 50.", a;
    if (a.used < (counted ? req.k + 1 : 3))
        return a.error = std::to_string(a.used) + " rows have every column: too few for " +
                         (counted ? std::to_string(req.k) + " groups" : std::string("DBSCAN")) + ".", a;

    switch (req.method) {
        case Method::KMeans: {
            const auto r = Clustering::KMeans(m.data, req.k, 300, "kmeans++", 10, 1e-4, Seed(req.seed));
            if (!r.success) return a.error = "K-Means: " + r.error_message, a;
            a.labels = r.labels;
            a.inertia = r.inertia;
            a.iterations = r.n_iterations;
            a.converged = r.converged;
            break;
        }
        case Method::DBSCAN: {
            if (!(req.eps > 0.0)) return a.error = "eps must be above 0.", a;
            if (req.min_samples < 1) return a.error = "min samples must be 1 or more.", a;
            const auto r = Clustering::DBSCAN(m.data, req.eps, req.min_samples, req.metric);
            if (!r.success) return a.error = "DBSCAN: " + r.error_message, a;
            a.labels = r.labels;
            a.noise = r.n_noise_points;
            break;
        }
        case Method::Hierarchical: {
            if (req.linkage == "ward" && req.metric != "euclidean")
                return a.error = "Ward linkage measures euclidean distance; pick euclidean, or another linkage.", a;
            const auto r = Clustering::Hierarchical(m.data, req.k, req.linkage, req.metric);
            if (!r.success) return a.error = "Hierarchical: " + r.error_message, a;
            a.labels = r.labels;
            a.cophenetic = r.cophenetic_correlation;
            break;
        }
        case Method::Mixture: {
            const auto r = Clustering::GMM(m.data, req.k, req.covariance, 100, 1e-3, 1, Seed(req.seed));
            if (!r.success) return a.error = "Gaussian mixture: " + r.error_message, a;
            a.labels = r.labels;
            a.bic = r.bic;
            a.aic = r.aic;
            a.log_likelihood = r.log_likelihood;
            a.converged = r.converged;
            a.iterations = r.n_iterations;
            a.probabilities = r.responsibilities;
            break;
        }
    }

    // The groups (ascending label; noise is not one) and their scores.
    std::set<int> found(a.labels.begin(), a.labels.end());
    found.erase(-1);
    const size_t d = req.columns.size();
    a.overall_means = m.means;
    a.overall_sds = m.sds;
    for (const int g : found) {
        GroupProfile p;
        p.label = g;
        p.means.assign(d, 0.0);
        for (size_t i = 0; i < m.raw.size(); ++i) {
            if (a.labels[i] != g) continue;
            ++p.rows;
            for (size_t j = 0; j < d; ++j) p.means[j] += m.raw[i][j];
        }
        for (auto& v : p.means) v /= static_cast<double>(std::max<int64_t>(p.rows, 1));
        a.groups.push_back(std::move(p));
    }
    if (found.size() >= 2 && static_cast<int64_t>(found.size()) < a.used - a.noise) {
        const auto metrics = Clustering::EvaluateClustering(m.data, a.labels);
        if (metrics.success) {
            a.scored = true;
            a.silhouette = metrics.silhouette_score;
            a.davies_bouldin = metrics.davies_bouldin_index;
            a.calinski_harabasz = metrics.calinski_harabasz_score;
            for (size_t g = 0; g < a.groups.size() && g < metrics.cluster_silhouettes.size(); ++g)
                a.groups[g].silhouette = metrics.cluster_silhouettes[g];
        }
    }

    // The scatter: the first two principal components of the data clustered.
    if (d >= 2) {
        const auto pca = DimensionalityReduction::ComputePCA(m.data, 2, true, false);
        if (pca.success && pca.transformed.size() == m.data.size()) {
            for (const auto& row : pca.transformed) {
                a.x.push_back(row[0]);
                a.y.push_back(row.size() > 1 ? row[1] : 0.0);
            }
            a.x_ratio = pca.explained_variance_ratio.size() > 0 ? pca.explained_variance_ratio[0] : NAN;
            a.y_ratio = pca.explained_variance_ratio.size() > 1 ? pca.explained_variance_ratio[1] : NAN;
        }
    } else {
        for (const auto& row : m.data) {
            a.x.push_back(row[0]);
            a.y.push_back(0.0);
        }
    }

    // Compared with a column: the counts and the adjusted Rand (noise left out).
    if (!req.compare.empty()) {
        std::set<std::string> values;
        for (const auto& v : m.label)
            if (v) values.insert(*v);
        a.compare_values.assign(values.begin(), values.end());
        std::map<std::string, int> index;
        for (size_t i = 0; i < a.compare_values.size(); ++i) index[a.compare_values[i]] = static_cast<int>(i);
        std::map<int, size_t> group_row;
        for (size_t g = 0; g < a.groups.size(); ++g) group_row[a.groups[g].label] = g;
        a.crosstab.assign(a.groups.size(), std::vector<int64_t>(a.compare_values.size(), 0));
        std::vector<int> mine, theirs;
        for (size_t i = 0; i < m.label.size(); ++i) {
            if (!m.label[i] || a.labels[i] < 0) continue;
            const int v = index[*m.label[i]];
            ++a.crosstab[group_row[a.labels[i]]][static_cast<size_t>(v)];
            mine.push_back(a.labels[i]);
            theirs.push_back(v);
        }
        a.adjusted_rand = AdjustedRand(mine, theirs);
    }
    return a;
}

HowManyAnswer RunHowMany(const std::string& table, const HowManyRequest& req, const QueryRunner& run,
                         const explore::MissingTexts& texts) {
    HowManyAnswer a;
    a.method = req.base.method;
    Matrix m = ReadMatrix(table, req.base.columns, req.base.standardise, "", run, texts, "Groups: how many");
    a.used = static_cast<int64_t>(m.raw.size());
    if (!m.error.empty()) return a.error = m.error, a;
    if (a.used < 4) return a.error = std::to_string(a.used) + " rows have every column: too few to compare.", a;

    if (req.base.method == Method::DBSCAN) {
        // DBSCAN counts the point itself in min_samples: the (min_samples - 1)-th other point.
        a.kdistances = Clustering::ComputeKDistances(m.data, std::max(1, req.base.min_samples - 1), req.base.metric);
        if (a.kdistances.empty()) return a.error = "The distances to the neighbours could not be computed.", a;
        a.knee = KneeOf(a.kdistances);
        return a;
    }
    const int from = std::max(2, req.from);
    const int to = static_cast<int>(std::min<int64_t>(std::max(from, req.to), a.used - 1));
    if (to < from) return a.error = "Too few rows for " + std::to_string(from) + " groups.", a;
    if (req.base.method == Method::Hierarchical) {
        if (req.base.linkage == "ward" && req.base.metric != "euclidean")
            return a.error = "Ward linkage measures euclidean distance; pick euclidean, or another linkage.", a;
        const auto r = Clustering::Hierarchical(m.data, 2, req.base.linkage, req.base.metric);
        if (!r.success) return a.error = "Hierarchical: " + r.error_message, a;
        a.linkage = r.linkage_matrix;
    }
    double best = -INFINITY;
    for (int k = from; k <= to; ++k) {
        a.ks.push_back(k);
        switch (req.base.method) {
            case Method::KMeans: {
                const auto r = Clustering::KMeans(m.data, k, 300, "kmeans++", 10, 1e-4, Seed(req.base.seed));
                if (!r.success) return a.error = "K-Means at " + std::to_string(k) + ": " + r.error_message, a;
                a.inertia.push_back(r.inertia);
                a.silhouette.push_back(Clustering::ComputeSilhouetteScore(m.data, r.labels));
                break;
            }
            case Method::Mixture: {
                const auto r = Clustering::GMM(m.data, k, req.base.covariance, 100, 1e-3, 1, Seed(req.base.seed));
                if (!r.success) return a.error = "Gaussian mixture at " + std::to_string(k) + ": " + r.error_message, a;
                a.bic.push_back(r.bic);
                a.aic.push_back(r.aic);
                if (-r.bic > best) {
                    best = -r.bic;
                    a.suggested = k;
                }
                break;
            }
            case Method::Hierarchical: {
                const double s = Clustering::ComputeSilhouetteScore(m.data, CutInto(a.linkage, a.used, k));
                a.silhouette.push_back(s);
                if (std::isfinite(s) && s > best) {
                    best = s;
                    a.suggested = k;
                }
                break;
            }
            case Method::DBSCAN: break;
        }
    }
    if (req.base.method == Method::KMeans) a.suggested = ElbowOf(a.ks, a.inertia);
    return a;
}

ProjectionAnswer RunProjection(const std::string& table, const ProjectionRequest& req, const QueryRunner& run,
                               const explore::MissingTexts& texts, const ProjectionHooks& hooks) {
    ProjectionAnswer a;
    a.request = req;
    if (req.columns.size() < 2) return a.error = "Add two number columns or more to project.", a;
    Matrix m = ReadMatrix(table, req.columns, req.standardise, req.colour, run, texts, "Projection: read the columns");
    a.rows = m.rows;
    a.used = static_cast<int64_t>(m.raw.size());
    if (!m.error.empty()) return a.error = m.error, a;
    if (a.used < 3) return a.error = std::to_string(a.used) + " rows have every column: too few to project.", a;

    // t-SNE and UMAP: at most max_rows rows (a seeded sample, kept in table order).
    std::vector<size_t> keep(m.data.size());
    std::iota(keep.begin(), keep.end(), size_t{0});
    if (req.method != ProjectionMethod::PCA && req.max_rows > 0 && a.used > req.max_rows) {
        std::mt19937 rng(Seed(req.seed));
        std::shuffle(keep.begin(), keep.end(), rng);
        keep.resize(static_cast<size_t>(req.max_rows));
        std::sort(keep.begin(), keep.end());
        a.sampled = true;
    }
    std::vector<std::vector<double>> data;
    data.reserve(keep.size());
    for (const size_t i : keep) {
        data.push_back(m.data[i]);
        a.used_rows.push_back(static_cast<int64_t>(i));
    }
    if (hooks.should_stop && hooks.should_stop()) return a.error = "Cancelled.", a;

    std::vector<std::vector<double>> points;
    switch (req.method) {
        case ProjectionMethod::PCA: {
            const auto r = DimensionalityReduction::ComputePCA(data, 0, true, false);
            if (!r.success) return a.error = "PCA: " + r.error_message, a;
            points = r.transformed;
            a.ratio = r.explained_variance_ratio;
            for (size_t c = 0; c < 2 && c < r.components.size(); ++c) a.loadings.push_back(r.components[c]);
            break;
        }
        case ProjectionMethod::TSNE: {
            if (req.perplexity < 1 || req.perplexity >= static_cast<int>(data.size()))
                return a.error = "Perplexity must be at least 1 and below the number of rows (" + std::to_string(data.size()) + ").", a;
            // The progress callback is the only way back into the loop: it throws to stop it.
            struct Stopped {};
            tSNEResult r;
            try {
                r = DimensionalityReduction::ComputetSNE(data, 2, req.perplexity, req.learning_rate, req.iterations,
                                                         [&](int iteration, double kl) {
                                                             if (hooks.progress) {
                                                                 char text[64];
                                                                 std::snprintf(text, sizeof(text), "t-SNE iteration %d, KL %.3f", iteration, kl);
                                                                 hooks.progress(static_cast<float>(iteration) / static_cast<float>(std::max(1, req.iterations)), text);
                                                             }
                                                             if (hooks.should_stop && hooks.should_stop()) throw Stopped{};
                                                         });
            } catch (const Stopped&) {
                return a.error = "Cancelled.", a;
            }
            if (!r.success) return a.error = "t-SNE: " + r.error_message, a;
            points = r.embeddings;
            a.kl_divergence = r.final_kl_divergence;
            break;
        }
        case ProjectionMethod::UMAP: {
            if (req.neighbors < 2 || req.neighbors > static_cast<int>(data.size()))
                return a.error = "Neighbours must be 2 to the number of rows (" + std::to_string(data.size()) + ").", a;
            const auto r = DimensionalityReduction::ComputeUMAP(data, 2, req.neighbors, req.min_dist, 0);
            if (!r.success) return a.error = "UMAP: " + r.error_message, a;
            points = r.embeddings;
            break;
        }
    }
    for (const auto& p : points) {
        a.x.push_back(p.empty() ? 0.0 : p[0]);
        a.y.push_back(p.size() > 1 ? p[1] : 0.0);
    }
    if (!req.colour.empty()) {
        std::set<std::string> distinct;
        bool numbers = true;
        for (const size_t i : keep) {
            const auto& v = m.label[i];
            a.colour.push_back(v ? *v : std::string());
            if (!v) continue;
            distinct.insert(*v);
            char* end = nullptr;
            std::strtod(v->c_str(), &end);
            numbers = numbers && end && *end == '\0';
        }
        a.colour_numeric = numbers && distinct.size() > 20;
    }
    return a;
}

double AdjustedRand(const std::vector<int>& a, const std::vector<int>& b) {
    if (a.empty() || a.size() != b.size()) return NAN;
    std::map<std::pair<int, int>, int64_t> cells;
    std::map<int, int64_t> rows, cols;
    for (size_t i = 0; i < a.size(); ++i) {
        ++cells[{a[i], b[i]}];
        ++rows[a[i]];
        ++cols[b[i]];
    }
    const auto pairs = [](double n) { return n * (n - 1.0) / 2.0; };
    double index = 0.0, sum_a = 0.0, sum_b = 0.0;
    for (const auto& [key, n] : cells) index += pairs(static_cast<double>(n));
    for (const auto& [key, n] : rows) sum_a += pairs(static_cast<double>(n));
    for (const auto& [key, n] : cols) sum_b += pairs(static_cast<double>(n));
    const double expected = sum_a * sum_b / pairs(static_cast<double>(a.size()));
    const double most = (sum_a + sum_b) / 2.0;
    if (most == expected) return 1.0;  // scikit-learn's special case (one cluster each, or all singletons)
    return (index - expected) / (most - expected);
}

double KneeOf(const std::vector<double>& v) {
    if (v.size() < 3) return NAN;
    const double lo = v.front(), hi = v.back();
    if (!(hi > lo)) return NAN;
    size_t at = 0;
    double best = -INFINITY;
    for (size_t i = 0; i < v.size(); ++i) {
        const double x = static_cast<double>(i) / static_cast<double>(v.size() - 1);
        const double y = (v[i] - lo) / (hi - lo);
        if (x - y > best) {
            best = x - y;
            at = i;
        }
    }
    return v[at];
}

int ElbowOf(const std::vector<int>& ks, const std::vector<double>& inertia) {
    if (ks.size() < 3 || inertia.size() != ks.size()) return 0;
    size_t at = 1;
    double best = -INFINITY;
    for (size_t i = 1; i + 1 < inertia.size(); ++i) {
        const double bend = inertia[i - 1] - 2.0 * inertia[i] + inertia[i + 1];
        if (bend > best) {
            best = bend;
            at = i;
        }
    }
    return ks[at];
}

DendrogramTop TopOfTree(const std::vector<std::vector<double>>& Z, int64_t n, int leaves) {
    DendrogramTop top;
    const int64_t merges = static_cast<int64_t>(Z.size());
    if (merges != n - 1 || n < 2) return top;
    const int64_t p = std::clamp<int64_t>(leaves, 2, n);
    const int64_t first_drawn = n + merges - (p - 1);  // ids of the clusters the drawn merges make
    struct Placed {
        double x, h;
    };
    // Depth-first, left child (Z[.,0]) first, as scipy draws.
    std::function<Placed(int64_t)> place = [&](int64_t id) -> Placed {
        if (id < first_drawn) {
            const double x = 10.0 * static_cast<double>(top.leaf_rows.size()) + 5.0;
            top.leaf_rows.push_back(id < n ? 1 : static_cast<int64_t>(Z[static_cast<size_t>(id - n)][3]));
            return {x, 0.0};
        }
        const auto& row = Z[static_cast<size_t>(id - n)];
        const Placed l = place(static_cast<int64_t>(row[0]));
        const Placed r = place(static_cast<int64_t>(row[1]));
        top.xs.push_back({l.x, l.x, r.x, r.x});
        top.ys.push_back({l.h, row[2], row[2], r.h});
        return {(l.x + r.x) / 2.0, row[2]};
    };
    place(n + merges - 1);
    return top;
}

}  // namespace cyxwiz::groups
