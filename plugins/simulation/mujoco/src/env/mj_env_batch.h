#pragma once

// Copies of one MuJoCo task for training (IEnvironmentBatch): one mjData and
// one random stream per copy, the copies stepped on a small worker pool. The
// copies share the task's model, unless the Environment node's robustness
// settings give each episode its own ground or factors (a model per copy,
// built when its episode starts) or the Stress panel changes the shown copy's
// scene. It holds nothing of the plugin object, so it outlives the plugin's
// shutdown safely.

#include "mj_robustness.h"
#include "mj_stress.h"
#include "mj_task.h"
#include "mj_view_slot.h"
#include "mj_worker_pool.h"
#include "plugin/interfaces/i_environment_provider.h"

#include <deque>
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
    std::vector<PluginEnvironmentStat> Stats() const override;
    void LabelShownCopy(const PluginShownCopyLabel& label) override;
    bool ShowCopyAsGhost(int copy, const std::string& name) override;
    void Release() override { delete this; }

    int Threads() const { return pool_ ? pool_->Threads() : 1; }
    int MaxEpisodeSteps() const { return max_episode_steps_; }
    // The scene copy `copy` plays in now (tests)
    const MjScene& SceneOf(int copy) const { return scenes_[static_cast<size_t>(copy)]; }
    const mjModel* ModelOf(int copy) const { return models_[static_cast<size_t>(copy)].get(); }
    const MjGroundProfile& ProfileOf(int copy) const { return profiles_[static_cast<size_t>(copy)]; }
    double RootX(int copy) const { return root_ >= 0 ? data_[static_cast<size_t>(copy)]->xpos[3 * root_] : 0.0; }

private:
    MjEnvBatch() = default;

    // A copy's start state (the reset noise from its stream) and observation
    void StartCopy(int copy, float* observation);
    // The task's observation of the copy, then the ground ahead (Look ahead)
    void ObserveCopy(int copy, float* observation) const;
    // A new episode's ground and factors for the copy (robustness, or the
    // Stress panel's ground for the shown copy). After StartCopy, outside the
    // workers; true when its model changed (observe it again).
    bool NewEpisode(int copy);
    // The copy's model with `scene` and its episode's factors, its state carried over
    bool BuildCopy(int copy, const MjScene& scene, std::string& error);

    // The shown copy's state into the view slot (after Reset and Step), with
    // its episode numbers; step: the actions and results just stepped (null
    // after Reset)
    struct SteppedCopy {
        const float* action = nullptr;
        float reward = 0.0f;
        bool episode_ended = false;
        bool terminated = false;  // ended by the task's own end
    };
    void Publish(const SteppedCopy* step);
    // The ghost copy's state into the view slot's ghost part, the same way
    void PublishGhost(const SteppedCopy* step);
    // The Stress panel's requests for the shown copy, taken from the view slot
    // and applied before a step; then its pushes
    void ApplyStress();
    MjStressHost StressHost();  // the shown copy
    double StepSeconds() const;

    MjSceneBuilder builder_;
    std::shared_ptr<mjModel> base_;                 // the task's model: copies share it until they need their own
    std::vector<std::shared_ptr<mjModel>> models_;  // each copy's
    std::vector<MjScene> scenes_;                   // each copy's ground and thrown objects
    std::vector<MjEpisodeDraw> draws_;              // each copy's episode factors
    std::vector<MjGroundProfile> profiles_;         // each copy's ground, for Look ahead
    std::vector<mjData*> data_;
    std::vector<MjRandom> random_;
    std::vector<int> steps_;
    std::vector<double> returns_;     // the episode's return so far
    std::vector<double> next_push_;   // simulated time of the copy's next push
    std::vector<double> ended_return_;// the return of the episode that ended this step
    std::vector<uint8_t> fell_;       // and whether it ended in a fall (ended early, or upside down)
    std::deque<bool> falls_;          // the last 100 episodes
    std::unique_ptr<MjTask> task_;
    std::unique_ptr<MjWorkerPool> pool_;
    MjRobustness robust_;
    MjCurriculum curriculum_;
    int count_ = 0;
    int task_observation_size_ = 0;
    int observation_size_ = 0;
    int action_size_ = 0;
    int max_episode_steps_ = 0;
    int root_ = -1;
    bool started_ = false;
    std::string error_;
    std::shared_ptr<MjViewSlot> view_;
    std::string model_path_;
    std::string title_;
    int shown_copy_ = -1;
    int ghost_copy_ = -1;  // ShowCopyAsGhost
    // The Stress panel's run on the shown copy (TOFIX212 P4)
    MjStressRun stress_;
};

// The environment's description for IEnvironmentProvider::GetEnvironments,
// read from the task's model: the task's settings, the batch's and the
// robustness section. False with the reason when the model does not load.
bool DescribeMjTask(const MjTaskSpec& spec, const std::string& model_path, PluginEnvironmentInfo& info,
                    std::string& error);
// The description as these settings make it (DescribeEnvironment): Look ahead
// adds its heights to the observation. False with the reason for settings
// that do not fit.
bool DescribeMjTaskWith(const MjTaskSpec& spec, const PluginEnvironmentInfo& listed,
                        const std::map<std::string, std::string>& settings, PluginEnvironmentInfo& info,
                        std::string& error);

}  // namespace cyxwiz::plugin::mujoco
