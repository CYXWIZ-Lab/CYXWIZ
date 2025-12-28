#pragma once

#include <string>
#include <vector>
#include <map>
#include <functional>
#include <atomic>
#include <mutex>
#include <memory>
#include <thread>
#include <limits>
#include <random>

namespace cyxwiz {

// Forward declarations
class TrainingExecutor;
class DatasetHandle;

// Parameter types for search space
enum class ParamType {
    Continuous,   // float range (e.g., learning_rate: 1e-5 to 1e-1)
    Discrete,     // integer range (e.g., batch_size: 16 to 128)
    Categorical,  // options list (e.g., optimizer: ["SGD", "Adam", "AdamW"])
    LogScale      // log-uniform (e.g., learning_rate with log scale)
};

// Single search parameter definition
struct SearchParameter {
    std::string name;
    ParamType type = ParamType::Continuous;

    // For continuous/discrete/logscale
    float min_value = 0.0f;
    float max_value = 1.0f;
    float step = 0.0f;  // 0 = continuous, >0 = discrete step

    // For categorical
    std::vector<std::string> options;

    // Current sampled value
    float current_value = 0.0f;
    std::string current_option;

    // Constructor helpers
    static SearchParameter Continuous(const std::string& name, float min, float max);
    static SearchParameter Discrete(const std::string& name, int min, int max, int step = 1);
    static SearchParameter LogScale(const std::string& name, float min, float max);
    static SearchParameter Categorical(const std::string& name, const std::vector<std::string>& options);
};

// Search space containing all parameters to optimize
struct SearchSpace {
    std::vector<SearchParameter> parameters;

    // Convenience methods to add parameters
    void AddContinuous(const std::string& name, float min, float max);
    void AddDiscrete(const std::string& name, int min, int max, int step = 1);
    void AddLogScale(const std::string& name, float min, float max);
    void AddCategorical(const std::string& name, const std::vector<std::string>& options);

    // Calculate total combinations for grid search
    size_t GetTotalCombinations() const;

    // Check if search space is valid
    bool IsValid() const;

    // Clear all parameters
    void Clear();
};

// Search strategy
enum class SearchStrategy {
    Grid,      // Exhaustive grid search (all combinations)
    Random,    // Random sampling from parameter space
    Bayesian   // Gaussian Process-based optimization (future)
};

// Result of a single trial
struct TrialResult {
    int trial_id = 0;
    std::map<std::string, float> parameters;        // Numeric parameters
    std::map<std::string, std::string> categorical_params;  // Categorical parameters

    float val_loss = std::numeric_limits<float>::max();
    float val_accuracy = 0.0f;
    float train_loss = 0.0f;
    float train_accuracy = 0.0f;

    float duration_seconds = 0.0f;
    bool completed = false;
    std::string error_message;

    // For tracking
    int epochs_completed = 0;
};

// Search configuration
struct SearchConfig {
    SearchStrategy strategy = SearchStrategy::Random;
    SearchSpace space;

    int max_trials = 20;
    int epochs_per_trial = 10;
    int early_stopping_patience = 3;

    std::string objective = "val_loss";  // "val_loss" or "val_accuracy"
    bool minimize = true;  // true for loss, false for accuracy

    // Resource limits
    float max_time_seconds = 3600.0f;  // 1 hour default

    // Dataset configuration
    float validation_split = 0.2f;
    int batch_size = 32;

    // Seed for reproducibility (0 = random)
    unsigned int random_seed = 0;
};

/**
 * Hyperparameter Search Engine
 * Coordinates search strategy with TrainingExecutor
 */
class HyperparamSearch {
public:
    HyperparamSearch();
    ~HyperparamSearch();

    // Configuration
    void SetConfig(const SearchConfig& config);
    const SearchConfig& GetConfig() const { return config_; }

    // Search control
    void Start();
    void Stop();
    void Pause();
    void Resume();

    bool IsRunning() const { return is_running_.load(); }
    bool IsPaused() const { return is_paused_.load(); }
    bool IsComplete() const { return is_complete_.load(); }

    // Progress
    int GetCurrentTrial() const { return current_trial_.load(); }
    int GetTotalTrials() const;
    float GetProgress() const;
    float GetElapsedSeconds() const;

    // Results
    std::vector<TrialResult> GetResults() const;
    TrialResult GetBestResult() const;
    int GetBestTrialId() const;

    // Callbacks
    using TrialStartCallback = std::function<void(int trial_id, const std::map<std::string, float>& params)>;
    using TrialProgressCallback = std::function<void(int trial_id, int epoch, float loss, float accuracy)>;
    using TrialCompleteCallback = std::function<void(const TrialResult&)>;
    using SearchCompleteCallback = std::function<void(const std::vector<TrialResult>&)>;

    void SetOnTrialStart(TrialStartCallback cb) { on_trial_start_ = cb; }
    void SetOnTrialProgress(TrialProgressCallback cb) { on_trial_progress_ = cb; }
    void SetOnTrialComplete(TrialCompleteCallback cb) { on_trial_complete_ = cb; }
    void SetOnSearchComplete(SearchCompleteCallback cb) { on_search_complete_ = cb; }

private:
    void RunSearchThread();
    TrialResult RunTrial(int trial_id, const std::map<std::string, float>& params,
                         const std::map<std::string, std::string>& categorical_params);

    // Strategy implementations
    std::pair<std::map<std::string, float>, std::map<std::string, std::string>>
        SampleGrid(int trial_idx);
    std::pair<std::map<std::string, float>, std::map<std::string, std::string>>
        SampleRandom();
    std::pair<std::map<std::string, float>, std::map<std::string, std::string>>
        SampleBayesian();

    // Helper to convert sampled parameters to training config
    void ApplyParameters(const std::map<std::string, float>& params,
                        const std::map<std::string, std::string>& categorical_params);

    SearchConfig config_;

    std::atomic<bool> is_running_{false};
    std::atomic<bool> is_paused_{false};
    std::atomic<bool> stop_requested_{false};
    std::atomic<bool> is_complete_{false};
    std::atomic<int> current_trial_{0};

    mutable std::mutex results_mutex_;
    std::vector<TrialResult> results_;
    int best_trial_id_ = -1;

    std::unique_ptr<std::thread> search_thread_;
    std::chrono::steady_clock::time_point start_time_;

    // Callbacks
    TrialStartCallback on_trial_start_;
    TrialProgressCallback on_trial_progress_;
    TrialCompleteCallback on_trial_complete_;
    SearchCompleteCallback on_search_complete_;

    // Random number generator
    std::mt19937 rng_;
};

} // namespace cyxwiz
