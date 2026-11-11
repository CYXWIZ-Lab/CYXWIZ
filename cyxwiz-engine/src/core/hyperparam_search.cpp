#include "hyperparam_search.h"
#include <spdlog/spdlog.h>
#include <cmath>
#include <random>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <limits>
#include <sstream>

namespace cyxwiz {

// ===== SearchParameter Factory Methods =====

SearchParameter SearchParameter::Continuous(const std::string& name, float min, float max) {
    SearchParameter p;
    p.name = name;
    p.type = ParamType::Continuous;
    p.min_value = min;
    p.max_value = max;
    p.step = 0.0f;
    return p;
}

SearchParameter SearchParameter::Discrete(const std::string& name, int min, int max, int step) {
    SearchParameter p;
    p.name = name;
    p.type = ParamType::Discrete;
    p.min_value = static_cast<float>(min);
    p.max_value = static_cast<float>(max);
    p.step = static_cast<float>(step);
    return p;
}

SearchParameter SearchParameter::LogScale(const std::string& name, float min, float max) {
    SearchParameter p;
    p.name = name;
    p.type = ParamType::LogScale;
    p.min_value = min;
    p.max_value = max;
    p.step = 0.0f;
    return p;
}

SearchParameter SearchParameter::Categorical(const std::string& name, const std::vector<std::string>& options) {
    SearchParameter p;
    p.name = name;
    p.type = ParamType::Categorical;
    p.options = options;
    return p;
}

// ===== SearchSpace Methods =====

namespace {

// A whole-number parameter's value on its grid: min + k * step, inside the range.
float OnStep(const SearchParameter& p, float value) {
    if (p.type != ParamType::Discrete || p.step <= 0) return value;
    const float snapped = p.min_value + std::round((value - p.min_value) / p.step) * p.step;
    return std::clamp(snapped > p.max_value ? snapped - p.step : snapped, p.min_value, p.max_value);
}

}  // namespace

size_t SearchSpace::GetTotalCombinations() const {
    if (parameters.empty()) return 0;

    size_t total = 1;
    for (const auto& param : parameters) {
        size_t count = 1;
        switch (param.type) {
            case ParamType::Discrete:
                if (param.step > 0) {
                    count = static_cast<size_t>((param.max_value - param.min_value) / param.step) + 1;
                }
                break;
            case ParamType::Categorical:
                count = param.options.size();
                break;
            case ParamType::Continuous:
            case ParamType::LogScale:
                // For continuous, we'd need a grid resolution - use 10 points as default
                count = 10;
                break;
        }
        total *= count;
    }
    return total;
}

bool SearchSpace::IsValid() const {
    if (parameters.empty()) return false;

    for (const auto& param : parameters) {
        if (param.name.empty()) return false;
        if (param.node_id < 0 || param.property.empty()) return false;
        if (param.type == ParamType::Categorical && param.options.empty()) return false;
        if (param.type == ParamType::LogScale && param.min_value <= 0.0f) return false;
        if (param.type != ParamType::Categorical && param.min_value >= param.max_value) return false;
    }
    return true;
}

void SearchSpace::Clear() {
    parameters.clear();
}

// ===== HyperparamSearch Implementation =====

HyperparamSearch::HyperparamSearch() {
    // Initialize RNG with random seed
    std::random_device rd;
    rng_.seed(rd());
}

HyperparamSearch::~HyperparamSearch() {
    Stop();
    if (search_thread_ && search_thread_->joinable()) {
        search_thread_->join();
    }
}

void HyperparamSearch::SetConfig(const SearchConfig& config) {
    if (is_running_.load()) {
        spdlog::warn("Cannot change config while search is running");
        return;
    }
    config_ = config;

    // Set RNG seed if specified
    if (config_.random_seed != 0) {
        rng_.seed(config_.random_seed);
    }
}

bool HyperparamSearch::Start() {
    if (is_running_.load()) {
        spdlog::warn("Search already running");
        return false;
    }

    if (!config_.space.IsValid()) {
        spdlog::error("Hyperparameter search: every parameter needs a node property and a valid range");
        return false;
    }
    if (!trial_function_) {
        spdlog::error("Hyperparameter search: no graph to train (start it from a compiled graph)");
        return false;
    }
    if (search_thread_ && search_thread_->joinable()) search_thread_->join();

    is_running_ = true;
    is_paused_ = false;
    stop_requested_ = false;
    is_complete_ = false;
    end_ = SearchEnd::AllTrials;
    current_trial_ = 0;

    {
        std::lock_guard<std::mutex> lock(results_mutex_);
        results_.clear();
        best_trial_id_ = -1;
    }

    start_time_ = std::chrono::steady_clock::now();

    // Launch search thread
    search_thread_ = std::make_unique<std::thread>([this]() {
        RunSearchThread();
    });

    spdlog::info("Hyperparameter search started with {} trials", GetTotalTrials());
    return true;
}

void HyperparamSearch::Stop() {
    if (!is_running_.load()) return;

    stop_requested_ = true;
    spdlog::info("Hyperparameter search stop requested");
}

void HyperparamSearch::Pause() {
    if (!is_running_.load()) return;
    is_paused_ = true;
    spdlog::info("Hyperparameter search paused");
}

void HyperparamSearch::Resume() {
    if (!is_running_.load()) return;
    is_paused_ = false;
    spdlog::info("Hyperparameter search resumed");
}

int HyperparamSearch::GetTotalTrials() const {
    if (config_.strategy == SearchStrategy::Grid) {
        return static_cast<int>(std::min<size_t>(config_.space.GetTotalCombinations(),
                                                 static_cast<size_t>(std::numeric_limits<int>::max())));
    }
    return config_.max_trials;
}

float HyperparamSearch::GetProgress() const {
    int total = GetTotalTrials();
    if (total <= 0) return 0.0f;
    return static_cast<float>(current_trial_.load()) / static_cast<float>(total);
}

float HyperparamSearch::GetElapsedSeconds() const {
    if (!is_running_.load() && !is_complete_.load()) return 0.0f;

    auto now = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - start_time_);
    return elapsed.count() / 1000.0f;
}

std::vector<TrialResult> HyperparamSearch::GetResults() const {
    std::lock_guard<std::mutex> lock(results_mutex_);
    return results_;
}

TrialResult HyperparamSearch::GetBestResult() const {
    std::lock_guard<std::mutex> lock(results_mutex_);
    if (best_trial_id_ < 0 || results_.empty()) {
        return TrialResult{};
    }

    for (const auto& result : results_) {
        if (result.trial_id == best_trial_id_) {
            return result;
        }
    }
    return TrialResult{};
}

int HyperparamSearch::GetBestTrialId() const {
    std::lock_guard<std::mutex> lock(results_mutex_);
    return best_trial_id_;
}

void HyperparamSearch::RunSearchThread() {
    spdlog::info("Search thread started");

    int total_trials = GetTotalTrials();
    // A failed trial (a value that does not compile, a run that fails) is
    // kept and the search goes on; three in a row mean the graph itself
    // cannot train, and the search stops.
    int consecutive_failures = 0;

    for (int trial = 0; trial < total_trials && !stop_requested_.load(); ++trial) {
        // Wait while paused
        while (is_paused_.load() && !stop_requested_.load()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }

        if (stop_requested_.load()) break;

        // Check time limit
        if (GetElapsedSeconds() > config_.max_time_seconds) {
            spdlog::info("Time limit reached, stopping search");
            end_ = SearchEnd::TimeLimit;
            break;
        }

        current_trial_ = trial;

        // Sample parameters based on strategy
        std::map<std::string, float> params;
        std::map<std::string, std::string> categorical_params;

        switch (config_.strategy) {
            case SearchStrategy::Grid:
                std::tie(params, categorical_params) = SampleGrid(trial);
                break;
            case SearchStrategy::Random:
                std::tie(params, categorical_params) = SampleRandom();
                break;
            case SearchStrategy::Bayesian:
                std::tie(params, categorical_params) = SampleBayesian();
                break;
        }

        // Notify trial start
        if (on_trial_start_) {
            on_trial_start_(trial, params);
        }

        spdlog::info("Starting trial {}/{}", trial + 1, total_trials);

        // Run the trial
        TrialResult result = RunTrial(trial, params, categorical_params);
        if (stop_requested_.load() && !result.completed) break;  // cancelled mid-trial
        consecutive_failures = result.completed ? 0 : consecutive_failures + 1;

        // Store result
        {
            std::lock_guard<std::mutex> lock(results_mutex_);
            results_.push_back(result);

            // Update best result
            if (result.completed) {
                bool is_better = false;
                if (best_trial_id_ < 0) {
                    is_better = true;
                } else {
                    // Find current best
                    const TrialResult* best = nullptr;
                    for (const auto& r : results_) {
                        if (r.trial_id == best_trial_id_) {
                            best = &r;
                            break;
                        }
                    }

                    if (best) {
                        const float current_val = ObjectiveValue(config_, result);
                        const float best_val = ObjectiveValue(config_, *best);
                        is_better = config_.minimize ? current_val < best_val : current_val > best_val;
                    }
                }

                if (is_better) {
                    best_trial_id_ = trial;
                    spdlog::info("New best trial: {} (val_loss={:.4f}, val_acc={:.2f}%)",
                                trial, result.val_loss, result.val_accuracy * 100);
                }
            }
        }

        // Notify trial complete
        if (on_trial_complete_) {
            on_trial_complete_(result);
        }
        if (consecutive_failures >= 3) {
            spdlog::error("Hyperparameter search stopped: three trials in a row failed ({})", result.error_message);
            end_ = SearchEnd::Failures;
            break;
        }
    }

    if (stop_requested_.load() && end_.load() == SearchEnd::AllTrials) end_ = SearchEnd::Stopped;
    is_running_ = false;
    is_complete_ = true;
    current_trial_ = total_trials;

    spdlog::info("Search completed. {} trials finished.", results_.size());

    // Notify search complete
    if (on_search_complete_) {
        std::lock_guard<std::mutex> lock(results_mutex_);
        on_search_complete_(results_);
    }
}

TrialResult HyperparamSearch::RunTrial(
    int trial_id,
    const std::map<std::string, float>& params,
    const std::map<std::string, std::string>& categorical_params)
{
    TrialResult result;
    result.trial_id = trial_id;
    result.parameters = params;
    result.categorical_params = categorical_params;

    auto trial_start = std::chrono::steady_clock::now();

    // Log parameters
    std::string param_str;
    for (const auto& [name, value] : params) {
        if (!param_str.empty()) param_str += ", ";
        param_str += name + "=" + std::to_string(value);
    }
    for (const auto& [name, value] : categorical_params) {
        if (!param_str.empty()) param_str += ", ";
        param_str += name + "=" + value;
    }
    spdlog::info("Trial {} params: {}", trial_id, param_str);

    trial_epoch_ = 0;
    trial_epochs_ = 0;
    TrialCallbacks callbacks;
    callbacks.should_cancel = [this] { return stop_requested_.load(); };
    callbacks.on_epoch = [this, trial_id](const TrialProgress& progress) {
        trial_epoch_ = progress.epoch;
        trial_epochs_ = progress.epochs;
        if (on_trial_progress_) on_trial_progress_(trial_id, progress.epoch, progress.val_loss, progress.val_accuracy);
    };
    TrialOutcome outcome;
    try {
        outcome = trial_function_(TrialChangesFor(config_.space, params, categorical_params), callbacks);
    } catch (const std::exception& e) {
        outcome.error = e.what();
    }
    result.completed = outcome.ok;
    result.error_message = outcome.ok ? std::string() : outcome.error;
    result.epochs_completed = outcome.epochs;
    result.regression = outcome.score.regression;
    result.val_loss = outcome.score.val_loss;
    result.val_accuracy = outcome.score.val_accuracy;
    result.val_mae = outcome.score.val_mae;
    result.val_rmse = outcome.score.val_rmse;
    result.train_loss = outcome.score.train_loss;
    result.train_accuracy = outcome.score.train_accuracy;
    if (!outcome.ok) spdlog::warn("Trial {}: {}", trial_id, outcome.error);

    auto trial_end = std::chrono::steady_clock::now();
    result.duration_seconds = std::chrono::duration_cast<std::chrono::milliseconds>(
        trial_end - trial_start).count() / 1000.0f;

    return result;
}

std::pair<std::map<std::string, float>, std::map<std::string, std::string>>
HyperparamSearch::SampleGrid(int trial_idx) {
    std::map<std::string, float> params;
    std::map<std::string, std::string> categorical_params;

    // Calculate indices for each parameter
    size_t remaining = static_cast<size_t>(std::max(0, trial_idx));
    for (const auto& param : config_.space.parameters) {
        if (param.type == ParamType::Categorical) {
            if (param.options.empty()) {
                spdlog::warn("Skipping categorical parameter '{}' with no options", param.name);
                continue;
            }
            const size_t idx = remaining % param.options.size();
            remaining /= param.options.size();
            categorical_params[param.name] = param.options[idx];
        } else {
            int num_steps = 10;  // Default grid resolution for continuous
            if (param.type == ParamType::Discrete && param.step > 0) {
                num_steps = static_cast<int>((param.max_value - param.min_value) / param.step) + 1;
            }

            const size_t step_count = static_cast<size_t>(std::max(1, num_steps));
            const size_t idx = remaining % step_count;
            remaining /= step_count;

            float t = (step_count > 1)
                ? static_cast<float>(idx) / static_cast<float>(step_count - 1)
                : 0.5f;

            if (param.type == ParamType::LogScale) {
                float log_min = std::log(param.min_value);
                float log_max = std::log(param.max_value);
                params[param.name] = std::exp(log_min + t * (log_max - log_min));
            } else {
                params[param.name] = param.min_value + t * (param.max_value - param.min_value);
            }

            // A whole-number parameter's grid is its own steps from the minimum.
            if (param.type == ParamType::Discrete && param.step > 0)
                params[param.name] = OnStep(param, param.min_value + static_cast<float>(idx) * param.step);
        }
    }

    return {params, categorical_params};
}

std::pair<std::map<std::string, float>, std::map<std::string, std::string>>
HyperparamSearch::SampleRandom() {
    std::map<std::string, float> params;
    std::map<std::string, std::string> categorical_params;

    std::uniform_real_distribution<float> uniform(0.0f, 1.0f);

    for (const auto& param : config_.space.parameters) {
        if (param.type == ParamType::Categorical) {
            std::uniform_int_distribution<int> dist(0, static_cast<int>(param.options.size()) - 1);
            categorical_params[param.name] = param.options[dist(rng_)];
        } else {
            float t = uniform(rng_);

            if (param.type == ParamType::LogScale) {
                float log_min = std::log(param.min_value);
                float log_max = std::log(param.max_value);
                params[param.name] = std::exp(log_min + t * (log_max - log_min));
            } else {
                params[param.name] = param.min_value + t * (param.max_value - param.min_value);
            }

            params[param.name] = OnStep(param, params[param.name]);
        }
    }

    return {params, categorical_params};
}

namespace {

// A parameter's value as a coordinate in [0, 1] (log scale on its log), and
// back; a categorical option by its index.
double ToUnit(const SearchParameter& p, float value) {
    if (p.type == ParamType::LogScale)
        return (std::log(value) - std::log(p.min_value)) / (std::log(p.max_value) - std::log(p.min_value));
    return (value - p.min_value) / (p.max_value - p.min_value);
}

float FromUnit(const SearchParameter& p, double t) {
    const float u = static_cast<float>(std::clamp(t, 0.0, 1.0));
    const float value = p.type == ParamType::LogScale
        ? std::exp(std::log(p.min_value) + u * (std::log(p.max_value) - std::log(p.min_value)))
        : p.min_value + u * (p.max_value - p.min_value);
    return OnStep(p, value);
}

size_t OptionIndex(const SearchParameter& p, double t) {
    return std::min(p.options.size() - 1, static_cast<size_t>(std::clamp(t, 0.0, 1.0) * p.options.size()));
}

// Cholesky factor (lower triangle, row-major) of a symmetric n x n matrix;
// false when it is not positive definite.
bool Cholesky(std::vector<double>& a, size_t n) {
    for (size_t j = 0; j < n; ++j) {
        double d = a[j * n + j];
        for (size_t k = 0; k < j; ++k) d -= a[j * n + k] * a[j * n + k];
        if (d <= 0.0) return false;
        a[j * n + j] = std::sqrt(d);
        for (size_t i = j + 1; i < n; ++i) {
            double v = a[i * n + j];
            for (size_t k = 0; k < j; ++k) v -= a[i * n + k] * a[j * n + k];
            a[i * n + j] = v / a[j * n + j];
        }
    }
    return true;
}

// L y = b, then L^T x = y, in place.
void SolveLower(const std::vector<double>& l, size_t n, std::vector<double>& b) {
    for (size_t i = 0; i < n; ++i) {
        for (size_t k = 0; k < i; ++k) b[i] -= l[i * n + k] * b[k];
        b[i] /= l[i * n + i];
    }
}

void SolveUpper(const std::vector<double>& l, size_t n, std::vector<double>& b) {
    for (size_t i = n; i-- > 0;) {
        for (size_t k = i + 1; k < n; ++k) b[i] -= l[k * n + i] * b[k];
        b[i] /= l[i * n + i];
    }
}

double SquaredDistance(const std::vector<double>& a, const std::vector<double>& b) {
    double d = 0.0;
    for (size_t i = 0; i < a.size(); ++i) d += (a[i] - b[i]) * (a[i] - b[i]);
    return d;
}

// A Gaussian process over the unit cube: RBF kernel, the length scale picked
// by marginal likelihood from a few, observations standardised by the caller.
struct GaussianProcess {
    std::vector<std::vector<double>> x;
    std::vector<double> chol, alpha;
    double length = 0.2;
    double noise = 1e-4;

    bool Fit(const std::vector<std::vector<double>>& points, const std::vector<double>& y) {
        x = points;
        const size_t n = x.size();
        double best_evidence = -std::numeric_limits<double>::infinity();
        bool fitted = false;
        for (const double l : {0.05, 0.1, 0.2, 0.4, 0.8}) {
            std::vector<double> k(n * n);
            for (size_t i = 0; i < n; ++i)
                for (size_t j = 0; j < n; ++j)
                    k[i * n + j] = std::exp(-SquaredDistance(x[i], x[j]) / (2.0 * l * l)) + (i == j ? noise : 0.0);
            if (!Cholesky(k, n)) continue;
            std::vector<double> a = y;
            SolveLower(k, n, a);
            SolveUpper(k, n, a);
            // log p(y) = -1/2 y^T K^-1 y - sum log L_ii (constant dropped)
            double evidence = 0.0;
            for (size_t i = 0; i < n; ++i) evidence -= 0.5 * y[i] * a[i] + std::log(k[i * n + i]);
            if (evidence > best_evidence) {
                best_evidence = evidence;
                chol = std::move(k);
                alpha = std::move(a);
                length = l;
                fitted = true;
            }
        }
        return fitted;
    }

    void Predict(const std::vector<double>& point, double& mean, double& sd) const {
        const size_t n = x.size();
        std::vector<double> k(n);
        for (size_t i = 0; i < n; ++i) k[i] = std::exp(-SquaredDistance(point, x[i]) / (2.0 * length * length));
        mean = 0.0;
        for (size_t i = 0; i < n; ++i) mean += k[i] * alpha[i];
        SolveLower(chol, n, k);
        double var = 1.0;
        for (const double v : k) var -= v * v;
        sd = std::sqrt(std::max(var, 1e-12));
    }
};

// Expected improvement below `best` (minimising).
double ExpectedImprovement(double mean, double sd, double best) {
    const double xi = 0.01;
    const double z = (best - mean - xi) / sd;
    const double cdf = 0.5 * std::erfc(-z / std::sqrt(2.0));
    const double pdf = std::exp(-0.5 * z * z) / std::sqrt(2.0 * 3.14159265358979323846);
    return (best - mean - xi) * cdf + sd * pdf;
}

std::string CsvField(const std::string& text) {
    if (text.find_first_of(",\"\r\n") == std::string::npos) return text;
    std::string out = "\"";
    for (const char c : text) out += c == '"' ? std::string("\"\"") : std::string(1, c);
    return out + "\"";
}

}  // namespace

std::pair<std::map<std::string, float>, std::map<std::string, std::string>>
HyperparamSearch::SampleBayesian() {
    const auto& parameters = config_.space.parameters;
    const int initial = config_.initial_random_trials > 0
        ? config_.initial_random_trials
        : std::max(3, static_cast<int>(parameters.size()) + 1);

    // The finished trials as points in the unit cube, scored lower = better.
    std::vector<std::vector<double>> points;
    std::vector<double> scores;
    {
        std::lock_guard<std::mutex> lock(results_mutex_);
        for (const auto& r : results_) {
            if (!r.completed) continue;
            const float objective = ObjectiveValue(config_, r);
            if (!std::isfinite(objective)) continue;
            std::vector<double> point;
            for (const auto& p : parameters) {
                if (p.type == ParamType::Categorical) {
                    const auto it = r.categorical_params.find(p.name);
                    const auto at = it == r.categorical_params.end()
                        ? p.options.end() : std::find(p.options.begin(), p.options.end(), it->second);
                    const size_t index = at == p.options.end() ? 0 : static_cast<size_t>(at - p.options.begin());
                    point.push_back((index + 0.5) / static_cast<double>(p.options.size()));
                } else {
                    const auto it = r.parameters.find(p.name);
                    point.push_back(it == r.parameters.end() ? 0.5 : ToUnit(p, it->second));
                }
            }
            points.push_back(std::move(point));
            scores.push_back(config_.minimize ? objective : -objective);
        }
    }
    if (static_cast<int>(points.size()) < initial) return SampleRandom();

    // Standardise, fit, and take the candidate with the highest expected
    // improvement among random points and points near the best so far.
    double mean = 0.0, var = 0.0;
    for (const double v : scores) mean += v;
    mean /= static_cast<double>(scores.size());
    for (const double v : scores) var += (v - mean) * (v - mean);
    const double sd = std::sqrt(var / static_cast<double>(scores.size())) + 1e-12;
    std::vector<double> y;
    for (const double v : scores) y.push_back((v - mean) / sd);
    GaussianProcess gp;
    if (!gp.Fit(points, y)) return SampleRandom();
    const size_t best_index = static_cast<size_t>(std::min_element(y.begin(), y.end()) - y.begin());
    const double best = y[best_index];

    std::uniform_real_distribution<double> uniform(0.0, 1.0);
    std::normal_distribution<double> nudge(0.0, 0.05);
    std::vector<double> chosen;
    double chosen_ei = -1.0;
    for (int c = 0; c < 1200; ++c) {
        std::vector<double> candidate(parameters.size());
        for (size_t d = 0; d < candidate.size(); ++d)
            candidate[d] = c < 1000 ? uniform(rng_) : std::clamp(points[best_index][d] + nudge(rng_), 0.0, 1.0);
        double m = 0.0, s = 0.0;
        gp.Predict(candidate, m, s);
        const double ei = ExpectedImprovement(m, s, best);
        if (ei > chosen_ei) {
            chosen_ei = ei;
            chosen = std::move(candidate);
        }
    }

    std::map<std::string, float> params;
    std::map<std::string, std::string> categorical_params;
    for (size_t d = 0; d < parameters.size(); ++d) {
        const auto& p = parameters[d];
        if (p.type == ParamType::Categorical) categorical_params[p.name] = p.options[OptionIndex(p, chosen[d])];
        else params[p.name] = FromUnit(p, chosen[d]);
    }
    return {params, categorical_params};
}

float ObjectiveValue(const SearchConfig& config, const TrialResult& result) {
    if (config.objective == "val_accuracy") return result.val_accuracy;
    if (config.objective == "val_mae") return result.val_mae;
    return result.val_loss;
}

std::vector<TrialChange> TrialChangesFor(const SearchSpace& space, const std::map<std::string, float>& params,
                                         const std::map<std::string, std::string>& categorical_params) {
    std::vector<TrialChange> changes;
    for (const auto& p : space.parameters) {
        TrialChange change{p.node_id, p.property, {}};
        if (p.type == ParamType::Categorical) {
            const auto it = categorical_params.find(p.name);
            if (it == categorical_params.end()) continue;
            change.value = it->second;
        } else {
            const auto it = params.find(p.name);
            if (it == params.end()) continue;
            char text[32];
            if (p.type == ParamType::Discrete) {
                std::snprintf(text, sizeof(text), "%ld", std::lround(it->second));
            } else {
                std::snprintf(text, sizeof(text), "%.6g", static_cast<double>(it->second));
            }
            change.value = text;
        }
        changes.push_back(std::move(change));
    }
    return changes;
}

std::string SearchResultsCsv(const SearchConfig& config, const std::vector<TrialResult>& results, int best_trial_id) {
    std::ostringstream csv;
    csv << "trial,best";
    for (const auto& p : config.space.parameters) csv << ',' << CsvField(p.name);
    csv << ",val_loss,val_accuracy,val_mae,val_rmse,train_loss,train_accuracy,epochs,seconds,error\n";
    for (const auto& r : results) {
        csv << r.trial_id + 1 << ',' << (r.trial_id == best_trial_id ? "yes" : "");
        for (const auto& p : config.space.parameters) {
            csv << ',';
            if (p.type == ParamType::Categorical) {
                const auto it = r.categorical_params.find(p.name);
                if (it != r.categorical_params.end()) csv << CsvField(it->second);
            } else {
                const auto it = r.parameters.find(p.name);
                if (it != r.parameters.end()) csv << it->second;
            }
        }
        csv << ',';
        if (r.completed) {
            csv << r.val_loss << ',' << r.val_accuracy << ',' << r.val_mae << ',' << r.val_rmse << ',' << r.train_loss
                << ',' << r.train_accuracy;
        } else {
            csv << ",,,,,";
        }
        csv << ',' << r.epochs_completed << ',' << r.duration_seconds << ',' << CsvField(r.error_message) << '\n';
    }
    return csv.str();
}

} // namespace cyxwiz
