#pragma once

// Stress test of trained policies (TOFIX212 P4 board 6): N episodes at each
// difficulty of one disturbance, for one or two saved policies, so they can be
// compared on the same grounds. Each level's episodes run as copies of one
// batch reset with the level's own seed, so both policies meet the same
// grounds and the same pushes. The disturbance reaches the environment as its
// robustness settings (the MuJoCo plugin's "Robustness (training)" section);
// each policy keeps its own other settings (Look ahead among them). Runs on a
// worker; the UI thread reads the progress. Engine side: it links the plugin
// host.

#include "trained_policy.h"

#include <atomic>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace cyxwiz::rl {

enum class StressKind { Rough = 0, Slopes, Steps, Obstacles, Kicks };
constexpr int kStressKindCount = 5;
const char* StressKindName(StressKind kind);  // "Rough"
// Every name, in StressKind order (the choices of a segmented control)
const char* const* StressKindNames();

struct StressTestSpec {
    StressKind kind = StressKind::Rough;
    int levels = 5;     // difficulty 0, 1/(levels-1), ..., 1
    int episodes = 10;  // a level
    uint64_t seed = 7000;
};

// The environment takes the robustness settings a stress test sets (the
// MuJoCo plugin's tasks; not the built-in CartPole)
bool SupportsStress(const plugin::PluginEnvironmentInfo& info);
// The environment settings a policy is tested with at `difficulty`: its own,
// with the robustness keys set for the test (no random physics, no
// curriculum; the ground or the pushes of the disturbance)
std::map<std::string, std::string> StressSettings(const std::map<std::string, std::string>& own, StressKind kind,
                                                  double difficulty);
// The level's difficulty (levels >= 2: evenly from 0 to 1)
double StressDifficulty(const StressTestSpec& spec, int level);
// The seed of a level: the same for every policy
uint64_t StressSeed(const StressTestSpec& spec, int level);

struct StressLevelResult {
    double difficulty = 0.0;
    std::vector<double> returns;  // one an episode
    int falls = 0;                // episodes that ended early or upside down
    double MeanReturn() const;
};

struct StressPolicyResult {
    std::string name;  // the policy's file name, or "this session's run"
    std::vector<StressLevelResult> levels;
};

class PolicyStressTest {
public:
    enum class State { Idle, Running, Finished, Stopped, Failed };

    struct Snapshot {
        State state = State::Idle;
        std::string message;  // why it failed
        int done = 0;         // levels finished, over every policy
        int total = 0;
        std::vector<StressPolicyResult> results;
    };

    PolicyStressTest() = default;
    ~PolicyStressTest();
    PolicyStressTest(const PolicyStressTest&) = delete;
    PolicyStressTest& operator=(const PolicyStressTest&) = delete;

    // Tests each policy in turn (one or two). False with the reason.
    bool Start(std::vector<std::pair<std::string, std::shared_ptr<TrainedPolicy>>> policies, StressTestSpec spec,
               std::string& error);
    void Stop();  // waits for the worker
    Snapshot Get() const;
    const StressTestSpec& Spec() const { return spec_; }

private:
    void Run();

    std::vector<std::pair<std::string, std::shared_ptr<TrainedPolicy>>> policies_;
    StressTestSpec spec_;
    std::thread worker_;
    std::atomic<bool> stop_{false};
    mutable std::mutex mutex_;
    Snapshot snapshot_;
};

// The results as CSV: difficulty, then each policy's mean return and falls
std::string StressResultsCsv(const std::vector<StressPolicyResult>& results, int episodes);

}  // namespace cyxwiz::rl
