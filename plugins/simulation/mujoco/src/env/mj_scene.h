#pragma once

// What the Viewport's Stress panel adds to a model (TOFIX212 P4): a ground
// along the run and objects to throw at the robot, plus the disturbances that
// act on a running copy (kicks, throws). The training batch and the Viewport
// both build a model from (model file, MjScene) with CompileMjScene, so the two
// agree to the last state value. No ImGui here.
//
// The robot keeps its own slice of the state: the ground adds no state, and
// the objects are free bodies placed after everything the model file has, so
// qpos[0, robot.nq) and qvel[0, robot.nv) stay the robot's, as in the file.

#include <mujoco/mujoco.h>

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace cyxwiz::plugin::mujoco {

enum class MjGroundKind { Flat, Rough, Slopes, Steps, Obstacles };

// The ground for an episode: it runs along +x from 2 m ahead of the robot's
// start, 4 m wide, on the model's floor plane. difficulty 0..1 scales its
// heights; the same seed gives the same ground.
struct MjGround {
    MjGroundKind kind = MjGroundKind::Flat;
    double difficulty = 0.4;
    uint64_t seed = 7;
    bool operator==(const MjGround& o) const {
        return kind == o.kind && (kind == MjGroundKind::Flat || (difficulty == o.difficulty && seed == o.seed));
    }
    bool operator!=(const MjGround& o) const { return !(*this == o); }
};

enum class MjShape { Ball, Box, Capsule, Cylinder };
constexpr int kShapeCount = 4;
const char* ShapeName(MjShape shape);  // "Box"
// The sizes a shape uses: a ball 1 (diameter), a box 3 (length, width,
// height along its x, y, z), a capsule or cylinder 2 (length end to end and
// diameter, along its z)
int ShapeSizes(MjShape shape);

// An object to throw. Sizes a shape does not use are ignored (and compare equal).
struct MjThrownObject {
    MjShape shape = MjShape::Ball;
    std::array<double, 3> size{0.2, 0.2, 0.2};  // m
    double mass = 2.0;                          // kg
    bool operator==(const MjThrownObject& o) const {
        if (shape != o.shape || mass != o.mass) return false;
        for (int i = 0; i < ShapeSizes(shape); ++i)
            if (size[i] != o.size[i]) return false;
        return true;
    }
    bool operator!=(const MjThrownObject& o) const { return !(*this == o); }
};

// How an object is thrown: from `distance` m away in the direction `around`
// degrees (from +x, the way the robot runs, towards +y: 0 ahead, 90 left, 180
// behind, 270 right) and `up` degrees above level, at `speed`; turned by
// `quat` and spinning at `spin` rad/s (world axes) when it leaves.
struct MjThrow {
    double around = 0.0;
    double up = 0.0;
    double distance = 3.0;
    double speed = 6.0;
    std::array<double, 4> quat{1.0, 0.0, 0.0, 0.0};
    std::array<double, 3> spin{0.0, 0.0, 0.0};
    bool operator==(const MjThrow&) const = default;
};

// The unit vector `around` degrees from +x towards +y and `up` degrees above level
std::array<double, 3> Direction(double around, double up);

constexpr int kMaxThrownObjects = 8;
constexpr double kGroundLength = 300.0;  // m of ground along the run
constexpr double kGroundStart = 2.0;     // m ahead of the robot's start
constexpr double kGroundHalfWidth = 2.0;

struct MjScene {
    MjGround ground;
    std::vector<MjThrownObject> objects;  // the object slots, at most kMaxThrownObjects
    bool Empty() const { return ground.kind == MjGroundKind::Flat && objects.empty(); }
    bool operator==(const MjScene& o) const { return ground == o.ground && objects == o.objects; }
    bool operator!=(const MjScene& o) const { return !(*this == o); }
};

// The model file's own sizes: the robot's slice of the state and its bodies.
struct MjRobotSize {
    int nq = 0;
    int nv = 0;
    int nbody = 0;
};

const char* GroundName(MjGroundKind kind);  // "Rough"

// Builds the model file with a scene, again and again: the file is parsed and
// compiled once, each scene builds on a copy of it (a training batch builds a
// ground per copy and per episode).
class MjSceneBuilder {
public:
    bool Load(const std::string& model_path, std::string& error);
    const std::string& Path() const { return path_; }  // the file Load read
    // The model with the scene; null with the reason. An empty scene gives the
    // same model as mj_loadXML. The caller deletes it (mj_deleteModel).
    mjModel* Build(const MjScene& scene, std::string& error) const;
    bool Loaded() const { return plain_ != nullptr; }
    const MjRobotSize& Robot() const { return robot_; }
    double StartX() const { return start_[0]; }  // the robot's root at rest: the ground starts kGroundStart ahead

private:
    std::string path_;
    std::shared_ptr<mjSpec> spec_;
    std::shared_ptr<mjModel> plain_;
    MjRobotSize robot_;
    double start_[3] = {0.0, 0.0, 0.0};
};

// After masses or inertias changed: the constants derived from them again,
// the statistics kept as `stat` (MuJoCo sizes the view and scales the solver
// from them). False when MuJoCo could not make a state (out of memory).
bool RefreshConstants(mjModel* m, const mjStatistic& stat);

// The ground's heights along the run, for the heightfield kinds (Rough,
// Slopes): one sample every `spacing` m from the ground's start; empty for the
// others. The boxes of the box kinds (Steps, Obstacles): x from, x to, height.
struct MjGroundBox {
    double x0 = 0.0;
    double x1 = 0.0;
    double height = 0.0;
};
std::vector<double> GroundHeights(const MjGround& ground, double& spacing);
std::vector<MjGroundBox> GroundBoxes(const MjGround& ground);

// The ground's height above the floor along x (world), exactly as built: the
// heightfield sloping between its samples, the boxes flat on top. For the
// Look ahead observation.
struct MjGroundProfile {
    double x0 = 0.0;  // where the ground starts
    double spacing = 0.0;
    std::vector<double> heights;
    std::vector<MjGroundBox> boxes;
    double At(double x) const;
};
// The profile of `ground` for a robot that starts at start_x
MjGroundProfile GroundProfile(const MjGround& ground, double start_x);

// The body the robot moves by (the first body under the world with joints)
int RobotRootBody(const mjModel* m, const MjRobotSize& robot);
// The robot's root upside down (its up axis points below the horizon)
bool RobotFlipped(const mjModel* m, const mjData* d, const MjRobotSize& robot);

// A sudden velocity change `dv` (world, m/s) of the robot's root: a free
// joint takes all of it, a slide joint the part along its axis; hinge joints
// of the root are left.
void KickRobot(const mjModel* m, mjData* d, const MjRobotSize& robot, const std::array<double, 3>& dv);
// The robot's mass (its root's subtree), for a kick spread over time as a force
double RobotMass(const mjModel* m, const MjRobotSize& robot);

// Puts object slot `slot` where `how` says, turned and spinning as it says,
// and sends it at its speed to where the robot will be when it arrives (the
// robot's velocity times the flight time, gravity allowed for). False: no
// such slot.
bool ThrowObject(const mjModel* m, mjData* d, const MjRobotSize& robot, int slot, const MjThrow& how);

// The slot a throw of `object` uses in `scene`: a slot holding the same
// object, else a new slot, else the slot after `last_slot` is replaced.
// Changes `scene` when no slot holds the object (the model must be rebuilt).
int PickObjectSlot(MjScene& scene, const MjThrownObject& object, int last_slot);

// A copy's state from a model to the same model file with another scene: the
// time, the robot's state, the controls and the object slots both have; new
// slots wait where the model parks them.
void CopySceneState(const mjModel* from_m, const mjData* from_d, const mjModel* to_m, mjData* to_d,
                    const MjRobotSize& robot);

}  // namespace cyxwiz::plugin::mujoco
