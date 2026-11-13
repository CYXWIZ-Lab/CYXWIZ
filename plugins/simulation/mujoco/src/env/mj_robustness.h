#pragma once

// Training for robustness (TOFIX212 P4 board 5): the Environment node's
// "Robustness (training)" settings for MuJoCo tasks, read and checked, and
// what each copy draws for an episode. All off by default: the plain task
// stays Gymnasium's. No ImGui here.

#include "mj_scene.h"
#include "mj_task.h"
#include "plugin/interfaces/i_environment_provider.h"

#include <array>
#include <map>
#include <string>
#include <vector>

namespace cyxwiz::plugin::mujoco {

struct MjRobustness {
    std::vector<MjGroundKind> grounds;  // each episode picks one; none: the model's own floor
    double difficulty_from = 0.0;       // 0..1
    double difficulty_to = 1.0;
    double curriculum_mark = 0.0;       // a mean return that raises the level; 0: no curriculum
    double push_every = 0.0;            // simulated seconds between pushes; 0: none
    double push_speed = 0.5;            // the largest push, m/s
    double friction_percent = 0.0;      // each episode scales friction by 1 +/- this
    double mass_percent = 0.0;          // the robot's masses
    double motor_percent = 0.0;         // the motors' strength (gear)
    bool look_ahead = false;            // the ground's heights ahead join the observation

    // Copies need their own models (a ground or factors per episode)
    bool PerCopyModels() const {
        return !grounds.empty() || friction_percent > 0.0 || mass_percent > 0.0 || motor_percent > 0.0;
    }
    bool Any() const { return PerCopyModels() || push_every > 0.0 || look_ahead; }
};

constexpr int kLookAheadCount = 10;         // heights in the observation
constexpr double kLookAheadSpacing = 0.2;   // m between them, the first 0.2 m ahead of the root
constexpr double kCurriculumStep = 0.1;     // the level rises this much at a time

// The settings, as the Environment node shows them (section "Robustness (training)")
const std::vector<PluginEnvironmentSetting>& RobustnessSettings();
bool IsRobustnessKey(const std::string& key);
// Reads the robustness keys of `given` (others are left for the task). False
// with the reason for a value that does not fit.
bool ReadMjRobustness(const std::map<std::string, std::string>& given, MjRobustness& out, std::string& error);
// `given` without the robustness keys (what the task reads)
std::map<std::string, std::string> WithoutRobustness(const std::map<std::string, std::string>& given);

// What one copy plays an episode with
struct MjEpisodeDraw {
    MjGround ground;              // Flat when the copy keeps the model's own floor
    double friction = 1.0;        // factors
    double mass = 1.0;
    double motor = 1.0;
};
// Draws from the copy's stream: a ground among the chosen ones with a
// difficulty in [from, level] (the level: the curriculum's, else `to`), and
// the factors
MjEpisodeDraw DrawEpisode(const MjRobustness& r, double level, MjRandom& random);
// The factors applied to a copy's model (built for the episode): friction of
// every geom, the robot's masses and inertia, the motors' gear. The caller
// runs mj_setConst after a mass change.
void ApplyFactors(const MjEpisodeDraw& draw, const MjRobotSize& robot, mjModel* m);

// The curriculum: the mean return of the last `window` finished episodes
// passing the mark raises the level by kCurriculumStep (up to `to`)
class MjCurriculum {
public:
    void Start(const MjRobustness& r, int copies);
    // An episode ended with this return; true when the level rose
    bool Finished(double episode_return);
    double Level() const { return level_; }
    bool Rising() const { return mark_ > 0.0 && level_ < to_; }

private:
    double level_ = 1.0;
    double to_ = 1.0;
    double mark_ = 0.0;
    size_t window_ = 20;
    std::vector<double> recent_;
};

}  // namespace cyxwiz::plugin::mujoco
