#pragma once

// What the Viewport's Inspect panel shows and changes (TOFIX212 P2): the
// selected body (mass, centre of mass, speed, its joints), the actuators, a
// joint's position, and keyframes (a named state saved with the project).
// No ImGui: the panel draws these, the tests check them. The caller holds the
// physics lock around every call that reads or writes mjData.

#include "mj_scene.h"

#include "mj_model_ref.h"

#include <mujoco/mujoco.h>

#include <array>
#include <filesystem>
#include <string>
#include <vector>

namespace cyxwiz::plugin::mujoco {

struct MjJointInfo {
    int id = -1;
    std::string name;
    int type = mjJNT_HINGE;  // mjtJoint
    bool scalar = false;     // hinge or slide: one value, typed in the panel
    double value = 0.0;      // rad or m (a scalar joint)
    double velocity = 0.0;   // rad/s or m/s
    bool limited = false;
    double low = 0.0, high = 0.0;
    double stiffness = 0.0;
    double damping = 0.0;
};

struct MjBodyInfo {
    int id = -1;
    std::string name;    // its name, or "body 3"
    int parent = -1;
    std::string parent_name;
    double mass = 0.0;        // kg
    double robot_mass = 0.0;  // the robot's bodies together (not a scene's objects)
    std::array<double, 3> centre_of_mass{0.0, 0.0, 0.0};  // world, m
    double speed = 0.0;          // m/s, of its centre of mass
    double angular_speed = 0.0;  // rad/s
    std::vector<MjJointInfo> joints;
};

struct MjActuatorInfo {
    int id = -1;
    std::string name;
    double control = 0.0;
    bool limited = false;
    double low = 0.0, high = 0.0;
    double force = 0.0;  // at the joint (gear x the actuator's force): N or N·m
};

// "bthigh", or "body 3" for a body with no name
std::string BodyName(const mjModel* m, int body);
MjBodyInfo InspectBody(const mjModel* m, const mjData* d, int body, const MjRobotSize& robot);
std::vector<MjActuatorInfo> InspectActuators(const mjModel* m, const mjData* d);
// A hinge or slide joint's position, clamped to its range, and the state
// computed again (mj_forward). False for another kind of joint.
bool SetJointValue(const mjModel* m, mjData* d, int joint, double value);

// A keyframe: the state MuJoCo integrates (mjSTATE_INTEGRATION: time,
// positions, velocities, activations, controls, applied forces ...)
struct MjKeyframe {
    std::string name;
    std::string saved;  // when it was saved, as shown ("12:04")
    std::vector<double> state;
};
constexpr int kKeyframeState = mjSTATE_INTEGRATION;

MjKeyframe CaptureKeyframe(const mjModel* m, const mjData* d, std::string name, std::string saved);
// The state set and computed again. False with the reason when the keyframe
// does not fit the model (saved for another version of the file).
bool ApplyKeyframe(const mjModel* m, mjData* d, const MjKeyframe& keyframe, std::string& error);
// The model file's own <key> elements: their names
std::vector<std::string> ModelKeyframes(const mjModel* m);

// <project>/mujoco/keyframes/<model file name>-<id>.json, the id from the
// model's reference (TOFIX219), so another PC finds the same keyframes
std::filesystem::path KeyframesFile(const std::filesystem::path& project, const MjModelRef& model);
std::string EncodeKeyframes(const std::vector<MjKeyframe>& keyframes, const std::string& model_path);
std::vector<MjKeyframe> DecodeKeyframes(const std::string& text);  // broken text: none
bool SaveKeyframes(const std::filesystem::path& file, const std::vector<MjKeyframe>& keyframes,
                   const std::string& model_path, std::string& error);
std::vector<MjKeyframe> LoadKeyframes(const std::filesystem::path& file);  // none when it is missing

}  // namespace cyxwiz::plugin::mujoco
