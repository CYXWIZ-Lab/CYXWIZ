// PCA, t-SNE and UMAP on ArrayFire (TOFIX140): the CPU option is ArrayFire's CPU
// backend, there are no hand-written CPU loops over the data, and a build without
// ArrayFire refuses. PCA and t-SNE follow scikit-learn; UMAP follows umap-learn's
// graph construction with its layout updates taken in expectation (full batch).

#ifdef _WIN32
#define NOMINMAX
#endif

#include "cyxwiz/dimensionality_reduction.h"
#include "analysis_device.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <vector>

#ifdef min
#undef min
#endif
#ifdef max
#undef max
#endif

namespace cyxwiz {

using namespace analysis_detail;

namespace {

constexpr double kMachineEpsilon = std::numeric_limits<double>::epsilon();

af::array Constant(double value, const af::array& like) {
    return af::constant(value, like.dims(), like.type());
}

af::array Clip(const af::array& values, double limit) {
    return af::max(af::min(values, Constant(limit, values)), Constant(-limit, values));
}

// Zeros on the diagonal of a square matrix.
af::array WithoutDiagonal(const af::array& square) {
    return af::select(af::identity(square.dims(0), square.dims(1), b8), 0.0, square);
}

struct Projection {
    af::array components;   // [k, d]
    af::array transformed;  // [n, k]
    af::array variances;    // [k, 1]
    double total_variance = 0.0;
};

// PCA of x [n, d]: eigenvectors of the n - 1 covariance via SVD (symmetric PSD),
// signed so each component's largest-magnitude entry is positive (svd_flip on V).
Projection Project(af::array x, int k, bool center, bool scale) {
    const dim_t n = x.dims(0), d = x.dims(1);
    const af::array centered = x - af::tile(af::mean(x, 0), static_cast<unsigned>(n), 1);
    af::array divisor = Constant(1.0, af::mean(x, 0));
    if (scale) {
        divisor = af::sqrt(af::mean(centered * centered, 0));
        divisor = af::select(divisor < 1e-10, 1.0, divisor);
    }
    x = (center ? centered : x) / af::tile(divisor, static_cast<unsigned>(n), 1);
    const af::array covariance = af::matmul(x, x, AF_MAT_TRANS, AF_MAT_NONE) / static_cast<double>(n - 1);
    af::array u, s, vt;
    af::svd(u, s, vt, covariance);
    af::array components = vt(af::seq(0, k - 1), af::span);
    af::array peak, column;
    af::max(peak, column, af::abs(components), 1);
    const af::array flat_index = af::range(af::dim4(k), 0, u32) + column * static_cast<unsigned>(k);
    const af::array sign = af::select(af::lookup(af::flat(components), flat_index) < 0.0, -1.0,
                                      Constant(1.0, peak));
    components = components * af::tile(sign, 1, static_cast<unsigned>(d));
    Projection result;
    result.components = components;
    result.transformed = af::matmul(x, components, AF_MAT_NONE, AF_MAT_TRANS);
    result.variances = s(af::seq(0, k - 1));
    result.total_variance = af::sum<double>(af::diag(covariance));
    return result;
}

// ---------------------------------------------------------------- t-SNE

// scikit-learn's _binary_search_perplexity on squared distances, every row at
// once: conditional P [n, n] with a zero diagonal.
af::array ConditionalProbabilities(const af::array& sq_dist, int perplexity) {
    const dim_t n = sq_dist.dims(0);
    const double inf = std::numeric_limits<double>::infinity();
    const double desired = std::log(static_cast<double>(perplexity));
    const af::array off_diagonal = !af::identity(n, n, b8);
    af::array beta = af::constant(1.0, n, sq_dist.type());
    af::array beta_min = af::constant(-inf, n, sq_dist.type());
    af::array beta_max = af::constant(inf, n, sq_dist.type());
    af::array done = af::constant(0, n, b8);
    af::array result = af::constant(0.0, n, n, sq_dist.type());
    for (int step = 0; step < 100; ++step) {
        af::array p = af::select(off_diagonal, af::exp(-sq_dist * af::tile(beta, 1, static_cast<unsigned>(n))), 0.0);
        af::array sum_p = af::sum(p, 1);
        sum_p = af::select(sum_p == 0.0, 1e-8, sum_p);
        const af::array mean_dist = af::sum(sq_dist * p, 1) / sum_p;
        p = p / af::tile(sum_p, 1, static_cast<unsigned>(n));
        const af::array entropy = af::log(sum_p) + beta * mean_dist;
        result = af::select(af::tile(done, 1, static_cast<unsigned>(n)), result, p);
        const af::array diff = entropy - desired;
        const af::array open = !done && !(af::abs(diff) <= 1e-5);
        const af::array raise = open && diff > 0.0, lower = open && !(diff > 0.0);
        const af::array raised = af::select(af::isInf(beta_max), beta * 2.0, (beta + beta_max) / 2.0);
        const af::array lowered = af::select(af::isInf(beta_min), beta / 2.0, (beta + beta_min) / 2.0);
        beta_min = af::select(raise, beta, beta_min);
        beta_max = af::select(lower, beta, beta_max);
        beta = af::select(raise, raised, af::select(lower, lowered, beta));
        done = !open;
        beta.eval();
        result.eval();
        if (af::allTrue<bool>(done)) break;
    }
    return result;
}

struct Gradient {
    double kl = 0.0;
    af::array grad;  // [n, dims]
};

// scikit-learn's exact _kl_divergence and its gradient.
Gradient KlGradient(const af::array& y, const af::array& p, double dof) {
    const af::array weights = WithoutDiagonal(af::pow(1.0 + SquaredDistances(y, y) / dof, -(dof + 1.0) / 2.0));
    const af::array q = af::max(weights / af::sum<double>(weights), Constant(kMachineEpsilon, weights));
    Gradient result;
    result.kl = af::sum<double>(p * af::log(af::max(p, Constant(kMachineEpsilon, p)) / q));
    const af::array pq = WithoutDiagonal((p - q) * weights);
    const dim_t dims = y.dims(1);
    result.grad = (2.0 * (dof + 1.0) / dof) *
                  (y * af::tile(af::sum(pq, 1), 1, static_cast<unsigned>(dims)) - af::matmul(pq, y));
    return result;
}

// ----------------------------------------------------------------- UMAP

// umap-learn find_ab_params (spread 1): least-squares fit of 1 / (1 + a x^(2b)) to
// the target curve on 300 points in [0, 3], Levenberg-Marquardt from (1, 1).
std::pair<double, double> FitAb(double min_dist, af::dtype type) {
    const af::array x = af::range(af::dim4(300), 0, type) * (3.0 / 299.0);
    const af::array target = af::select(x < min_dist, 1.0, af::exp(-(x - min_dist)));
    const af::array log_x = af::log(af::select(x > 0.0, x, 1.0));
    const auto evaluate = [&](double a, double b, af::array* da, af::array* db) {
        const af::array power = af::select(x > 0.0, af::pow(x, 2.0 * b), 0.0);
        const af::array denom = 1.0 + a * power;
        if (da != nullptr) {
            *da = -power / (denom * denom);
            *db = -a * power * 2.0 * log_x / (denom * denom);
        }
        return 1.0 / denom - target;
    };
    double a = 1.0, b = 1.0, damping = 1e-3;
    af::array da, db;
    af::array residual = evaluate(a, b, &da, &db);
    double cost = af::sum<double>(residual * residual);
    for (int iter = 0; iter < 500; ++iter) {
        const double jaa = af::sum<double>(da * da), jab = af::sum<double>(da * db), jbb = af::sum<double>(db * db);
        const double ga = af::sum<double>(da * residual), gb = af::sum<double>(db * residual);
        const double maa = jaa * (1.0 + damping), mbb = jbb * (1.0 + damping);
        const double det = maa * mbb - jab * jab;
        const double step_a = -(mbb * ga - jab * gb) / det, step_b = -(maa * gb - jab * ga) / det;
        const af::array trial = evaluate(a + step_a, b + step_b, nullptr, nullptr);
        const double trial_cost = af::sum<double>(trial * trial);
        if (trial_cost < cost) {
            a += step_a;
            b += step_b;
            residual = evaluate(a, b, &da, &db);
            cost = trial_cost;
            damping = std::max(damping / 10.0, 1e-15);
            if (std::abs(step_a) <= 1e-15 * (1.0 + std::abs(a)) && std::abs(step_b) <= 1e-15 * (1.0 + std::abs(b))) break;
        } else {
            damping *= 10.0;
            if (damping > 1e15) break;
        }
    }
    return {a, b};
}

// umap-learn's fuzzy simplicial set: k nearest neighbours (self included),
// smooth_knn_dist, membership strengths, fuzzy union. Dense [n, n].
af::array FuzzyGraph(const af::array& x, int k) {
    const dim_t n = x.dims(0);
    af::array knn_dist = af::constant(0.0, n, k, x.type());
    af::array knn_index = af::constant(0, n, k, u32);
    const dim_t block = RowBlock(n);
    for (dim_t start = 0; start < n; start += block) {
        const dim_t stop = std::min(n, start + block);
        const af::seq rows(static_cast<double>(start), static_cast<double>(stop - 1));
        af::array sorted, order;
        af::sort(sorted, order, PairwiseDistances(x(rows, af::span), x, "euclidean"), 1);
        knn_dist(rows, af::span) = sorted(af::span, af::seq(0, k - 1));
        knn_index(rows, af::span) = order(af::span, af::seq(0, k - 1));
    }
    const double inf = std::numeric_limits<double>::infinity();
    af::array rho = af::min(af::select(knn_dist > 0.0, knn_dist, inf), 1);
    rho = af::select(af::isInf(rho), 0.0, rho);

    // Binary search for sigma: sum_{j>=1} exp(-(d_ij - rho_i) / sigma) = log2(k).
    const double target = std::log2(static_cast<double>(k));
    const af::array shifted = knn_dist(af::span, af::seq(1, k - 1)) - af::tile(rho, 1, static_cast<unsigned>(k - 1));
    af::array low = af::constant(0.0, n, x.type()), high = af::constant(inf, n, x.type()), mid = af::constant(1.0, n, x.type());
    af::array done = af::constant(0, n, b8);
    for (int step = 0; step < 64; ++step) {
        const af::array terms = af::select(shifted > 0.0, af::exp(-shifted / af::tile(mid, 1, static_cast<unsigned>(k - 1))), 1.0);
        const af::array psum = af::sum(terms, 1);
        const af::array open = !done && !(af::abs(psum - target) < 1e-5);
        const af::array above = open && psum > target, below = open && !(psum > target);
        high = af::select(above, mid, high);
        low = af::select(below, mid, low);
        mid = af::select(above, (low + high) / 2.0,
                         af::select(below, af::select(af::isInf(high), mid * 2.0, (low + high) / 2.0), mid));
        done = !open;
        mid.eval();
        if (af::allTrue<bool>(done)) break;
    }
    const af::array mean_row = af::mean(knn_dist, 1);
    const double mean_all = af::mean<double>(knn_dist);
    const af::array sigma = af::select(rho > 0.0, af::max(mid, 1e-3 * mean_row), af::max(mid, Constant(1e-3 * mean_all, mid)));

    const af::array self = af::tile(af::range(af::dim4(n), 0, u32), 1, static_cast<unsigned>(k));
    const af::array gap = knn_dist - af::tile(rho, 1, static_cast<unsigned>(k));
    const af::array sig = af::tile(sigma, 1, static_cast<unsigned>(k));
    af::array strength = af::select(gap <= 0.0 || sig == 0.0, 1.0, af::exp(-gap / sig));
    strength = af::select(knn_index == self, 0.0, strength);

    af::array graph = af::constant(0.0, n, n, x.type());
    graph(af::flat(self + knn_index * static_cast<unsigned>(n))) = af::flat(strength);
    const af::array transposed = graph.T();
    return graph + transposed - graph * transposed;
}

// Scaled PCA start: umap-learn init="pca" (max |coord| 10), then the [0, 10] rescale
// of simplicial_set_embedding.
af::array UmapStart(const af::array& x, int dims) {
    af::array y = Project(x, dims, true, false).transformed;
    y = y * (10.0 / af::max<double>(af::abs(y)));
    const dim_t n = y.dims(0);
    const af::array low = af::tile(af::min(y, 0), static_cast<unsigned>(n), 1);
    const af::array high = af::tile(af::max(y, 0), static_cast<unsigned>(n), 1);
    return 10.0 * (y - low) / (high - low);
}

// One epoch of umap-learn's layout updates in expectation: each edge (i, j) moves
// both ends 2 * rate times, each head draws 5 * sum_j rate_ij uniform negatives.
af::array UmapEpoch(const af::array& y, const af::array& rate, const af::array& repulse_scale, double a, double b,
                    double alpha) {
    const dim_t n = y.dims(0), dims = y.dims(1);
    const af::array d2 = SquaredDistances(y, y);
    const af::array positive = d2 > 0.0;
    const af::array safe = af::select(positive, d2, 1.0);
    const af::array attract = af::select(positive, -2.0 * a * b * af::pow(safe, b - 1.0) / (a * af::pow(safe, b) + 1.0), 0.0);
    const af::array repel = 2.0 * b / ((0.001 + d2) * (a * af::pow(d2, b) + 1.0));
    const af::array others = !af::identity(n, n, b8);
    af::array updated = y;
    for (dim_t f = 0; f < dims; ++f) {
        const int column = static_cast<int>(f);
        const af::array diff = af::tile(y(af::span, column), 1, static_cast<unsigned>(n)) -
                               af::tile(y(af::span, column).T(), static_cast<unsigned>(n), 1);
        const af::array pull = 2.0 * af::sum(rate * Clip(attract * diff, 4.0), 1);
        const af::array push_terms = af::select(others, af::select(positive, Clip(repel * diff, 4.0), 4.0), 0.0);
        const af::array push = repulse_scale * af::sum(push_terms, 1);
        updated(af::span, column) = y(af::span, column) + alpha * (pull + push);
    }
    updated.eval();
    return updated;
}

}  // namespace

// ================================================================== PCA

PCAResult DimensionalityReduction::ComputePCA(
    const std::vector<std::vector<double>>& data,
    int n_components,
    bool center,
    bool scale)
{
    PCAResult result;
    if (const std::string refusal = MatrixRefusal(data); !refusal.empty()) {
        result.error_message = refusal;
        return result;
    }
    const int n = static_cast<int>(data.size()), d = static_cast<int>(data[0].size());
    if (n < 2) {
        result.error_message = "PCA needs at least 2 samples";
        return result;
    }
    const int limit = std::min(n, d);
    const int k = n_components <= 0 || n_components > limit ? limit : n_components;
    try {
        const Projection projection = Project(MatrixToDevice(data), k, center, scale);
        if (!(projection.total_variance > 0.0)) {
            result.error_message = "Data has zero variance";
            return result;
        }
        constexpr const char* kOperation = "DimensionalityReduction::ComputePCA";
        result.components = MatrixFromDevice(projection.components, kOperation);
        result.transformed = MatrixFromDevice(projection.transformed, kOperation);
        result.explained_variance = DeviceToHostValues(projection.variances, kOperation);
        for (double variance : result.explained_variance) {
            result.explained_variance_ratio.push_back(variance / projection.total_variance);
            result.singular_values.push_back(std::sqrt(variance * (n - 1)));
            result.total_variance_explained += result.explained_variance_ratio.back();
        }
        result.n_components = k;
        result.success = true;
    } catch (const af::exception& e) {
        result = PCAResult{};
        result.error_message = DeviceError("DimensionalityReduction::ComputePCA", e);
    }
    return result;
}

// ================================================================ t-SNE

tSNEResult DimensionalityReduction::ComputetSNE(
    const std::vector<std::vector<double>>& data,
    int n_dims,
    int perplexity,
    double learning_rate,
    int n_iterations,
    std::function<void(int, double)> progress_callback)
{
    tSNEResult result;
    if (const std::string refusal = MatrixRefusal(data); !refusal.empty()) {
        result.error_message = refusal;
        return result;
    }
    const int n = static_cast<int>(data.size()), d = static_cast<int>(data[0].size());
    if (n_dims < 1 || n_dims > std::min(n, d)) {
        result.error_message = "Output dimensions must be between 1 and min(samples, features) (PCA start)";
        return result;
    }
    if (perplexity < 1 || perplexity >= n) {
        result.error_message = "Perplexity must be at least 1 and below the number of samples (" + std::to_string(n) + ")";
        return result;
    }
    if (!(learning_rate > 0.0) || n_iterations < 1) {
        result.error_message = "Learning rate must be positive and iterations at least 1";
        return result;
    }
    try {
        const af::array x = MatrixToDevice(data);
        af::array p = ConditionalProbabilities(SquaredDistances(x, x), perplexity);
        p = p + p.T();
        p = WithoutDiagonal(af::max(p / std::max(af::sum<double>(p), kMachineEpsilon), Constant(kMachineEpsilon, p)));

        af::array y = Project(x, n_dims, true, false).transformed;
        const af::array first = y(af::span, 0);
        const double first_std = std::sqrt(af::mean<double>((first - af::mean<double>(first)) * (first - af::mean<double>(first))));
        y = y / first_std * 1e-4;

        const double dof = std::max(n_dims - 1, 1);
        constexpr int kExploration = 250, kCheckEvery = 50, kWithoutProgress = 300;
        constexpr double kExaggeration = 12.0, kMinGain = 0.01, kMinGradNorm = 1e-7;
        // scikit-learn's _gradient_descent; the two stages start with fresh update / gains.
        const auto descend = [&](const af::array& probabilities, double momentum, int first_iter, int end_iter) {
            af::array update = Constant(0.0, y), gains = Constant(1.0, y);
            double best_error = std::numeric_limits<double>::max();
            int best_iter = first_iter, last = first_iter - 1;
            for (int i = first_iter; i < end_iter; ++i) {
                Gradient step = KlGradient(y, probabilities, dof);
                const double grad_norm = std::sqrt(af::sum<double>(step.grad * step.grad));
                const af::array increase = update * step.grad < 0.0;
                gains = af::max(af::select(increase, gains + 0.2, gains * 0.8), Constant(kMinGain, gains));
                update = momentum * update - learning_rate * (step.grad * gains);
                y = y + update;
                y.eval();
                last = i;
                result.final_kl_divergence = step.kl;
                if ((i + 1) % kCheckEvery == 0) {
                    result.kl_divergence_history.push_back(step.kl);
                    if (progress_callback) progress_callback(i + 1, step.kl);
                    if (step.kl < best_error) {
                        best_error = step.kl;
                        best_iter = i;
                    } else if (i - best_iter > kWithoutProgress) {
                        break;
                    }
                    if (grad_norm <= kMinGradNorm) break;
                }
            }
            return last;
        };
        const int explored = descend(p * kExaggeration, 0.5, 0, std::min(kExploration, n_iterations));
        const int finished = descend(p, 0.8, explored + 1, n_iterations);

        result.embeddings = MatrixFromDevice(y, "DimensionalityReduction::ComputetSNE");
        result.n_iterations = std::max(explored, finished) + 1;
        result.perplexity = perplexity;
        result.success = true;
    } catch (const af::exception& e) {
        result = tSNEResult{};
        result.error_message = DeviceError("DimensionalityReduction::ComputetSNE", e);
    }
    return result;
}

// ================================================================= UMAP

UMAPResult DimensionalityReduction::ComputeUMAP(
    const std::vector<std::vector<double>>& data,
    int n_dims,
    int n_neighbors,
    double min_dist,
    int n_epochs)
{
    UMAPResult result;
    if (const std::string refusal = MatrixRefusal(data); !refusal.empty()) {
        result.error_message = refusal;
        return result;
    }
    const int n = static_cast<int>(data.size()), d = static_cast<int>(data[0].size());
    if (n_dims < 1 || n_dims > std::min(n, d)) {
        result.error_message = "Output dimensions must be between 1 and min(samples, features) (PCA start)";
        return result;
    }
    if (n_neighbors < 2 || n_neighbors > n) {
        result.error_message = "Neighbours must be between 2 and the number of samples (" + std::to_string(n) + ")";
        return result;
    }
    if (!(min_dist >= 0.0) || min_dist > 1.0 || n_epochs < 0) {
        result.error_message = "min_dist must be in [0, 1] and n_epochs non-negative";
        return result;
    }
    const int epochs = n_epochs > 0 ? n_epochs : (n <= 10000 ? 500 : 200);
    try {
        const af::array x = MatrixToDevice(data);
        const auto [a, b] = FitAb(min_dist, x.type());
        af::array graph = FuzzyGraph(x, n_neighbors);
        const double peak = af::max<double>(graph);
        graph = af::select(graph < peak / epochs, 0.0, graph);  // simplicial_set_embedding's pruning
        const af::array rate = graph / peak;
        const af::array repulse_scale = 5.0 * af::sum(rate, 1) / static_cast<double>(n);

        af::array y = UmapStart(x, n_dims);
        for (int epoch = 0; epoch < epochs; ++epoch) {
            y = UmapEpoch(y, rate, repulse_scale, a, b, 1.0 - static_cast<double>(epoch) / epochs);
        }
        result.embeddings = MatrixFromDevice(y, "DimensionalityReduction::ComputeUMAP");
        result.n_neighbors = n_neighbors;
        result.min_dist = min_dist;
        result.a = a;
        result.b = b;
        result.n_epochs = epochs;
        result.success = true;
    } catch (const af::exception& e) {
        result = UMAPResult{};
        result.error_message = DeviceError("DimensionalityReduction::ComputeUMAP", e);
    }
    return result;
}

} // namespace cyxwiz
