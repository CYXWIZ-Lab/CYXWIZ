#include "hyperparam_search.h"
#include <spdlog/spdlog.h>
#include <cmath>
#include <random>
#include <algorithm>
#include <chrono>

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

void SearchSpace::AddContinuous(const std::string& name, float min, float max) {
    parameters.push_back(SearchParameter::Continuous(name, min, max));
}

void SearchSpace::AddDiscrete(const std::string& name, int min, int max, int step) {
    parameters.push_back(SearchParameter::Discrete(name, min, max, step));
}

void SearchSpace::AddLogScale(const std::string& name, float min, float max) {
    parameters.push_back(SearchParameter::LogScale(name, min, max));
}

void SearchSpace::AddCategorical(const std::string& name, const std::vector<std::string>& options) {
    parameters.push_back(SearchParameter::Categorical(name, options));
}

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
        if (param.type == ParamType::Categorical && param.options.empty()) return false;
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

void HyperparamSearch::Start() {
    if (is_running_.load()) {
        spdlog::warn("Search already running");
        return;
    }

    if (!config_.space.IsValid()) {
        spdlog::error("Invalid search space configuration");
        return;
    }

    is_running_ = true;
    is_paused_ = false;
    stop_requested_ = false;
    is_complete_ = false;
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
        return static_cast<int>(config_.space.GetTotalCombinations());
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

    for (int trial = 0; trial < total_trials && !stop_requested_.load(); ++trial) {
        // Wait while paused
        while (is_paused_.load() && !stop_requested_.load()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }

        if (stop_requested_.load()) break;

        // Check time limit
        if (GetElapsedSeconds() > config_.max_time_seconds) {
            spdlog::info("Time limit reached, stopping search");
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
                        if (config_.minimize) {
                            float current_val = (config_.objective == "val_loss") ? result.val_loss : result.val_accuracy;
                            float best_val = (config_.objective == "val_loss") ? best->val_loss : best->val_accuracy;
                            is_better = current_val < best_val;
                        } else {
                            float current_val = (config_.objective == "val_accuracy") ? result.val_accuracy : result.val_loss;
                            float best_val = (config_.objective == "val_accuracy") ? best->val_accuracy : best->val_loss;
                            is_better = current_val > best_val;
                        }
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
    }

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

    // TODO: Integrate with actual TrainingExecutor
    // For now, simulate training with random results
    std::uniform_real_distribution<float> loss_dist(0.1f, 2.0f);
    std::uniform_real_distribution<float> acc_dist(0.5f, 0.99f);

    // Simulate epochs
    for (int epoch = 0; epoch < config_.epochs_per_trial && !stop_requested_.load(); ++epoch) {
        // Wait while paused
        while (is_paused_.load() && !stop_requested_.load()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }

        if (stop_requested_.load()) break;

        // Simulate training time
        std::this_thread::sleep_for(std::chrono::milliseconds(100));

        // Update progress
        result.epochs_completed = epoch + 1;
        result.train_loss = loss_dist(rng_);
        result.train_accuracy = acc_dist(rng_);
        result.val_loss = loss_dist(rng_);
        result.val_accuracy = acc_dist(rng_);

        if (on_trial_progress_) {
            on_trial_progress_(trial_id, epoch + 1, result.val_loss, result.val_accuracy);
        }
    }

    result.completed = !stop_requested_.load();

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
    int remaining = trial_idx;
    for (const auto& param : config_.space.parameters) {
        if (param.type == ParamType::Categorical) {
            int idx = remaining % param.options.size();
            remaining /= param.options.size();
            categorical_params[param.name] = param.options[idx];
        } else {
            int num_steps = 10;  // Default grid resolution for continuous
            if (param.type == ParamType::Discrete && param.step > 0) {
                num_steps = static_cast<int>((param.max_value - param.min_value) / param.step) + 1;
            }

            int idx = remaining % num_steps;
            remaining /= num_steps;

            float t = (num_steps > 1) ? static_cast<float>(idx) / (num_steps - 1) : 0.5f;

            if (param.type == ParamType::LogScale) {
                float log_min = std::log(param.min_value);
                float log_max = std::log(param.max_value);
                params[param.name] = std::exp(log_min + t * (log_max - log_min));
            } else {
                params[param.name] = param.min_value + t * (param.max_value - param.min_value);
            }

            if (param.type == ParamType::Discrete) {
                params[param.name] = std::round(params[param.name] / param.step) * param.step;
            }
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

            if (param.type == ParamType::Discrete && param.step > 0) {
                params[param.name] = std::round(params[param.name] / param.step) * param.step;
            }
        }
    }

    return {params, categorical_params};
}

std::pair<std::map<std::string, float>, std::map<std::string, std::string>>
HyperparamSearch::SampleBayesian() {
    // TODO: Implement Gaussian Process-based sampling
    // For now, fall back to random sampling
    spdlog::warn("Bayesian optimization not yet implemented, using random sampling");
    return SampleRandom();
}

void HyperparamSearch::ApplyParameters(
    const std::map<std::string, float>& params,
    const std::map<std::string, std::string>& categorical_params)
{
    // TODO: Apply parameters to training configuration
    // This will map parameter names like "learning_rate", "batch_size", etc.
    // to the actual TrainingConfiguration struct
}

} // namespace cyxwiz
