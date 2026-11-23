// Hyperparameter search logic (TOFIX178 T3) with a stand-in trial function:
// a known smooth score of the sampled values, so the search itself is what
// is tested (real trials train through RunTrial: test_trial_runner).
// - each parameter becomes a node property change of the trial;
// - Grid runs every combination, Random max_trials;
// - a failed trial is kept and the search goes on; three in a row stop it;
//   the search says why it ended (all trials, Stop, three failures);
// - no trial function, or a parameter without a node property: no start;
// - Bayesian (Gaussian process + expected improvement) beats Random on the
//   same budget, over several seeds;
// - the CSV has a row per trial and marks the best.
#include "../src/core/hyperparam_search.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>

using namespace cyxwiz;

namespace {

int g_failures = 0;

void Check(bool ok, const std::string& what) {
    std::printf("%s %s\n", ok ? "  ok  " : "  FAIL", what.c_str());
    if (!ok) ++g_failures;
}

constexpr int kOptimizer = 7;
constexpr int kDense = 3;

// A bowl with its floor at learning rate 0.003 (log scale) and 96 units.
float Bowl(float lr, float units) {
    const float a = (std::log10(lr) - std::log10(0.003f)) / 2.0f;
    const float b = (units - 96.0f) / 112.0f;
    return 0.2f + a * a + b * b;
}

TrialOutcome BowlTrial(const std::vector<TrialChange>& changes, const TrialCallbacks& callbacks) {
    float lr = 0.0f, units = 0.0f;
    for (const auto& c : changes) {
        if (c.node_id == kOptimizer && c.property == "learning_rate") lr = std::stof(c.value);
        if (c.node_id == kDense && c.property == "units") units = std::stof(c.value);
    }
    if (callbacks.on_epoch) callbacks.on_epoch({1, 1, 1.0f, Bowl(lr, units), 0.5f});
    TrialOutcome o;
    o.ok = true;
    o.epochs = 1;
    o.score.has_validation = true;
    o.score.val_loss = Bowl(lr, units);
    return o;
}

SearchConfig BowlConfig(SearchStrategy strategy, int trials, unsigned seed) {
    SearchConfig config;
    config.strategy = strategy;
    config.max_trials = trials;
    config.random_seed = seed;
    auto lr = SearchParameter::LogScale("learning rate", 1e-4f, 1e-1f);
    lr.node_id = kOptimizer;
    lr.property = "learning_rate";
    auto units = SearchParameter::Discrete("Dense units", 16, 256, 16);
    units.node_id = kDense;
    units.property = "units";
    config.space.parameters = {lr, units};
    return config;
}

std::vector<TrialResult> RunToEnd(HyperparamSearch& search) {
    if (!search.Start()) return {};
    while (search.IsRunning()) std::this_thread::sleep_for(std::chrono::milliseconds(2));
    return search.GetResults();
}

float BestLoss(const std::vector<TrialResult>& results) {
    float best = 1e9f;
    for (const auto& r : results)
        if (r.completed) best = std::min(best, r.val_loss);
    return best;
}

}  // namespace

int main() {
    // Changes: the values as node properties, units as a whole number.
    {
        SearchSpace space = BowlConfig(SearchStrategy::Random, 1, 1).space;
        const auto changes = TrialChangesFor(space, {{"learning rate", 0.0025f}, {"Dense units", 96.0f}}, {});
        Check(changes.size() == 2 && changes[0].node_id == kOptimizer && changes[0].property == "learning_rate" &&
                  changes[0].value == "0.0025" && changes[1].node_id == kDense && changes[1].value == "96",
              "values become node property changes");
    }

    // No start without a trial function or a target.
    {
        HyperparamSearch search;
        search.SetConfig(BowlConfig(SearchStrategy::Random, 3, 1));
        Check(!search.Start(), "no trial function: no start");
        auto config = BowlConfig(SearchStrategy::Random, 3, 1);
        config.space.parameters[0].node_id = -1;
        search.SetConfig(config);
        search.SetTrialFunction(BowlTrial);
        Check(!search.Start(), "a parameter without a node property: no start");
    }

    // Grid and Random run their trials and report progress.
    {
        HyperparamSearch search;
        auto config = BowlConfig(SearchStrategy::Grid, 0, 1);
        config.space.parameters[0] = SearchParameter::Categorical("optimizer", {"Adam", "SGD"});
        config.space.parameters[0].node_id = kOptimizer;
        config.space.parameters[0].property = "learning_rate";  // stand-in: any property
        search.SetConfig(config);
        search.SetTrialFunction([](const std::vector<TrialChange>&, const TrialCallbacks&) {
            TrialOutcome o;
            o.ok = true;
            o.score.has_validation = true;
            o.score.val_loss = 1.0f;
            return o;
        });
        const auto results = RunToEnd(search);
        Check(results.size() == 2 * 16, "grid runs every combination: " + std::to_string(results.size()));

        HyperparamSearch random;
        random.SetConfig(BowlConfig(SearchStrategy::Random, 12, 3));
        random.SetTrialFunction(BowlTrial);
        int progress = 0;
        random.SetOnTrialProgress([&](int, int, float, float) { ++progress; });
        Check(RunToEnd(random).size() == 12 && progress == 12, "random runs max_trials and reports each epoch");
    }

    // Failures: kept, the search goes on; three in a row stop it.
    {
        HyperparamSearch search;
        search.SetConfig(BowlConfig(SearchStrategy::Random, 10, 5));
        int calls = 0;
        search.SetTrialFunction([&](const std::vector<TrialChange>& c, const TrialCallbacks& cb) {
            ++calls;
            if (calls == 2) {
                TrialOutcome o;
                o.error = "the changed graph does not compile";
                return o;
            }
            return BowlTrial(c, cb);
        });
        const auto results = RunToEnd(search);
        Check(results.size() == 10 && !results[1].completed && results[1].error_message.find("compile") != std::string::npos,
              "a failed trial is kept with its error and the search goes on");
        Check(search.GetEnd() == SearchEnd::AllTrials, "it ended with all its trials run");

        HyperparamSearch broken;
        broken.SetConfig(BowlConfig(SearchStrategy::Random, 10, 5));
        broken.SetTrialFunction([](const std::vector<TrialChange>&, const TrialCallbacks&) {
            TrialOutcome o;
            o.error = "the graph has no validation part to score the trial on";
            return o;
        });
        Check(RunToEnd(broken).size() == 3, "three failures in a row stop the search");
        Check(broken.GetEnd() == SearchEnd::Failures, "and it says so");

        // Stop keeps the trials done and says the search was stopped.
        HyperparamSearch stopped;
        stopped.SetConfig(BowlConfig(SearchStrategy::Random, 10, 5));
        int runs = 0;
        stopped.SetTrialFunction([&](const std::vector<TrialChange>& c, const TrialCallbacks& cb) {
            if (++runs == 3) stopped.Stop();
            return BowlTrial(c, cb);
        });
        Check(RunToEnd(stopped).size() == 3 && stopped.GetEnd() == SearchEnd::Stopped,
              "Stop keeps the trials done and ends as stopped");
    }

    // Bayesian beats Random on the same budget (mean best over seeds).
    {
        constexpr int kTrials = 18;
        float bayes_sum = 0.0f, random_sum = 0.0f;
        int bayes_wins = 0;
        for (unsigned seed = 1; seed <= 8; ++seed) {
            HyperparamSearch bayes, random;
            bayes.SetConfig(BowlConfig(SearchStrategy::Bayesian, kTrials, seed));
            random.SetConfig(BowlConfig(SearchStrategy::Random, kTrials, seed));
            bayes.SetTrialFunction(BowlTrial);
            random.SetTrialFunction(BowlTrial);
            const float b = BestLoss(RunToEnd(bayes));
            const float r = BestLoss(RunToEnd(random));
            bayes_sum += b;
            random_sum += r;
            if (b <= r) ++bayes_wins;
        }
        char line[160];
        std::snprintf(line, sizeof(line), "Bayesian beats Random: mean best excess %.4f vs %.4f, %d of 8 seeds",
                      bayes_sum / 8.0f - 0.2f, random_sum / 8.0f - 0.2f, bayes_wins);
        Check(bayes_sum < random_sum && bayes_wins >= 5, line);
    }

    // CSV: a row per trial, the best marked, a failed trial's error quoted.
    {
        SearchConfig config = BowlConfig(SearchStrategy::Random, 2, 1);
        TrialResult good;
        good.trial_id = 0;
        good.completed = true;
        good.parameters = {{"learning rate", 0.01f}, {"Dense units", 64.0f}};
        good.val_loss = 0.5f;
        TrialResult bad;
        bad.trial_id = 1;
        bad.parameters = {{"learning rate", 0.02f}, {"Dense units", 32.0f}};
        bad.error_message = "failed, badly";
        const std::string csv = SearchResultsCsv(config, {good, bad}, 0);
        Check(csv.rfind("trial,best,learning rate,Dense units,val_loss", 0) == 0, "CSV header names the parameters");
        Check(csv.find("\n1,yes,0.01,64,0.5,") != std::string::npos, "the best trial is marked");
        Check(csv.find("\"failed, badly\"") != std::string::npos, "a failed trial's error is quoted");
    }

    if (g_failures) {
        std::printf("%d hyperparameter search checks failed\n", g_failures);
        return EXIT_FAILURE;
    }
    std::printf("Hyperparameter search passed\n");
    return EXIT_SUCCESS;
}
