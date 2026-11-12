#pragma once

// Copies of one MuJoCo task for training (IEnvironmentBatch): one read-only
// mjModel, one mjData and one random stream per copy, the copies stepped on a
// small worker pool. It holds nothing of the plugin object, so it outlives
// the plugin's shutdown safely.

#include "mj_task.h"
#include "mj_view_slot.h"
#include "mj_worker_pool.h"
#include "plugin/interfaces/i_environment_provider.h"

#include <memory>
#include <string>
#include <vector>

namespace cyxwiz::plugin::mujoco {

class MjEnvBatch final : public IEnvironmentBatch {
public:
    // Null with the reason when the model does not load or the settings do
    // not fit the task. view: where ShowCopy publishes (null: no picture).
    static MjEnvBatch* Create(const MjTaskSpec& spec, const std::string& model_path,
                              const std::map<std::string, std::string>& settings, int count,
                              std::shared_ptr<MjViewSlot> view, std::string& error);
    ~MjEnvBatch();

    int Count() const override { return count_; }
    int ObservationSize() const override { return observation_size_; }
    int ActionSize() const override { return action_size_; }
    bool Reset(uint64_t seed, float* observations) override;
    bool Step(const float* actions, const PluginEnvironmentStepBuffers& out) override;
    std::string LastError() const override { return error_; }
    bool ShowCopy(int copy) override;
    void Release() override { delete this; }

    int Threads() const { return pool_ ? pool_->Threads() : 1; }
    int MaxEpisodeSteps() const { return max_episode_steps_; }

private:
    MjEnvBatch() = default;

    void StartCopy(int copy, float* observation);
    // The shown copy's state into the view slot (after Reset and Step), with
    // its episode numbers; step: the actions and results just stepped (null
    // after Reset)
    struct SteppedCopy {
        const float* action = nullptr;
        float reward = 0.0f;
        bool episode_ended = false;
    };
    void Publish(const SteppedCopy* step);

    mjModel* model_ = nullptr;
    std::vector<mjData*> data_;
    std::vector<MjRandom> random_;
    std::vector<int> steps_;
    std::unique_ptr<MjTask> task_;
    std::unique_ptr<MjWorkerPool> pool_;
    int count_ = 0;
    int observation_size_ = 0;
    int action_size_ = 0;
    int max_episode_steps_ = 0;
    bool started_ = false;
    std::string error_;
    std::shared_ptr<MjViewSlot> view_;
    std::string model_path_;
    std::string title_;
    int shown_copy_ = -1;
};

// The environment's description for IEnvironmentProvider::GetEnvironments,
// read from the task's model. False with the reason when the model does
// not load.
bool DescribeMjTask(const MjTaskSpec& spec, const std::string& model_path, PluginEnvironmentInfo& info,
                    std::string& error);

}  // namespace cyxwiz::plugin::mujoco
