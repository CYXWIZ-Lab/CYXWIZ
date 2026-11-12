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

#include <cstdint>
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

// An object to throw: a ball (size = diameter) or a cube (size = edge).
struct MjThrownObject {
    bool ball = true;
    double size = 0.2;  // m
    double mass = 2.0;  // kg
    bool operator==(const MjThrownObject& o) const { return ball == o.ball && size == o.size && mass == o.mass; }
    bool operator!=(const MjThrownObject& o) const { return !(*this == o); }
};

constexpr int kMaxThrownObjects = 4;
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

// The model file with the scene added; null with the reason when it does not
// load. robot: the model file's own sizes. An empty scene gives the same model
// as mj_loadXML.
mjModel* CompileMjScene(const std::string& model_path, const MjScene& scene, MjRobotSize& robot, std::string& error);

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

// The body the robot moves by (the first body under the world with joints)
int RobotRootBody(const mjModel* m, const MjRobotSize& robot);
// The robot's root upside down (its up axis points below the horizon)
bool RobotFlipped(const mjModel* m, const mjData* d, const MjRobotSize& robot);

// A sudden velocity change of the robot's root: back or forward along +x
// (the way Gymnasium's runners go), or up. Hinge joints of the root are left.
enum class MjKick { Back, Forward, Up };
void KickRobot(const mjModel* m, mjData* d, const MjRobotSize& robot, MjKick kick, double speed);

// Puts object slot `slot` 3 m ahead of or behind the robot, or 2.5 m above it,
// and sends it at `speed` to where the robot will be when it arrives (its
// velocity times the flight time, gravity allowed for). False: no such slot.
enum class MjThrowFrom { Ahead, Behind, Above };
bool ThrowObject(const mjModel* m, mjData* d, const MjRobotSize& robot, int slot, MjThrowFrom from, double speed);

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
