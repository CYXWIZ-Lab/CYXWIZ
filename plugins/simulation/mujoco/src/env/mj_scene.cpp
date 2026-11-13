#include "mj_scene.h"

#include "mj_task.h"  // MjRandom

#include <algorithm>
#include <cmath>
#include <memory>

namespace cyxwiz::plugin::mujoco {

namespace {

constexpr const char* kGroundName = "cyx_stress_ground";
constexpr const char* kGroundMaterial = "cyx_stress_ground_material";
constexpr double kParkY = 1000.0;  // object slots wait this far to the side (out of sight), one 3 m apart
constexpr float kObjectRgba[4] = {0.91f, 0.69f, 0.29f, 1.0f};
// The stress ground's colour: sand, so it stands out from the model's own floor
constexpr float kGroundRgba[4] = {0.95f, 0.72f, 0.45f, 1.0f};

// The model's floor: a plane on the world body (the first one)
struct Floor {
    int geom = -1;
    double z = 0.0;
};

Floor FindFloor(const mjModel* m) {
    Floor floor;
    for (int g = 0; g < m->ngeom; ++g) {
        if (m->geom_type[g] == mjGEOM_PLANE && m->geom_bodyid[g] == 0) {
            floor.geom = g;
            floor.z = m->geom_pos[3 * g + 2];
            break;
        }
    }
    return floor;
}

// Contact bits that meet every geom of the model that collides
int ContactMask(const mjModel* m) {
    int mask = 0;
    for (int g = 0; g < m->ngeom; ++g) mask |= m->geom_contype[g] | m->geom_conaffinity[g];
    return mask ? mask : 1;
}

// A geom that touches like the floor (friction, softness), or MuJoCo's defaults without one
void TouchLikeFloor(mjsGeom* geom, const mjModel* m, const Floor& floor, int mask) {
    geom->contype = mask;
    geom->conaffinity = mask;
    if (floor.geom < 0) return;
    const int g = floor.geom;
    for (int i = 0; i < 3; ++i) geom->friction[i] = m->geom_friction[3 * g + i];
    for (int i = 0; i < mjNREF; ++i) geom->solref[i] = m->geom_solref[mjNREF * g + i];
    for (int i = 0; i < mjNIMP; ++i) geom->solimp[i] = m->geom_solimp[mjNIMP * g + i];
    geom->solmix = m->geom_solmix[g];
    geom->condim = m->geom_condim[g];
    geom->priority = m->geom_priority[g];
    geom->margin = m->geom_margin[g];
    geom->gap = m->geom_gap[g];
}

// The ground's look: sand-coloured, so it is plain where it starts and how it
// rises; with the floor's texture laid at half the floor's period (finer squares
// show the bumps and slopes; texuniform keeps them the same size on every face)
void LookLikeGround(mjSpec* spec, mjsGeom* geom, const mjModel* m, const Floor& floor) {
    mjs_setString(geom->material, "");
    std::copy(kGroundRgba, kGroundRgba + 4, geom->rgba);
    if (floor.geom < 0) return;
    const int g = floor.geom;
    const int mat = m->geom_matid[g];
    if (mat < 0) return;
    const int tex = m->mat_texid[mat * mjNTEXROLE + mjTEXROLE_RGB];
    const char* tex_name = tex >= 0 ? mj_id2name(m, mjOBJ_TEXTURE, tex) : nullptr;
    if (!mjs_findElement(spec, mjOBJ_MATERIAL, kGroundMaterial)) {
        mjsMaterial* material = mjs_addMaterial(spec, nullptr);
        mjs_setName(material->element, kGroundMaterial);
        std::copy(kGroundRgba, kGroundRgba + 4, material->rgba);
        material->reflectance = 0.0f;  // a mirror finish looks wrong on bumps
        if (tex_name && mjs_setInStringVec(material->textures, mjTEXROLE_RGB, tex_name)) {
            // One texture repeat spans `period` m of the floor
            const double repeat = m->mat_texrepeat[2 * mat];
            double period = 1.0;
            if (repeat > 0.0)
                period = m->mat_texuniform[mat] ? 1.0 / repeat : 2.0 * m->geom_size[3 * g] / repeat;
            material->texuniform = 1;
            material->texrepeat[0] = material->texrepeat[1] = static_cast<float>(2.0 / period);
        }
    }
    mjs_setString(geom->material, kGroundMaterial);
}

void Normalise(double v[3]) {
    const double n = std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
    if (n > 0.0) for (int i = 0; i < 3; ++i) v[i] /= n;
}

// The object's geom: MuJoCo's sizes from ours (half extents, radius, half length)
void ShapeGeom(const MjThrownObject& object, mjsGeom* geom) {
    const auto& s = object.size;
    switch (object.shape) {
        case MjShape::Ball:
            geom->type = mjGEOM_SPHERE;
            geom->size[0] = s[0] / 2.0;
            break;
        case MjShape::Box:
            geom->type = mjGEOM_BOX;
            for (int i = 0; i < 3; ++i) geom->size[i] = s[i] / 2.0;
            break;
        case MjShape::Capsule:  // length end to end: the cylinder between the caps is length - diameter
            geom->type = mjGEOM_CAPSULE;
            geom->size[0] = s[1] / 2.0;
            // MuJoCo needs a middle: a capsule no longer than it is wide keeps 1 mm (it is then a ball)
            geom->size[1] = std::max(0.0005, (s[0] - s[1]) / 2.0);
            break;
        case MjShape::Cylinder:
            geom->type = mjGEOM_CYLINDER;
            geom->size[0] = s[1] / 2.0;
            geom->size[1] = s[0] / 2.0;
            break;
    }
}

// The object's principal inertia (a capsule as its cylinder, a little high)
std::array<double, 3> ShapeInertia(const MjThrownObject& object) {
    const auto& s = object.size;
    const double m = object.mass;
    switch (object.shape) {
        case MjShape::Ball: {
            const double r = s[0] / 2.0;
            return {0.4 * m * r * r, 0.4 * m * r * r, 0.4 * m * r * r};
        }
        case MjShape::Box:
            return {m * (s[1] * s[1] + s[2] * s[2]) / 12.0, m * (s[0] * s[0] + s[2] * s[2]) / 12.0,
                    m * (s[0] * s[0] + s[1] * s[1]) / 12.0};
        case MjShape::Capsule:
        case MjShape::Cylinder: {
            const double r = s[1] / 2.0;
            const double side = m * (3.0 * r * r + s[0] * s[0]) / 12.0;
            return {side, side, 0.5 * m * r * r};
        }
    }
    return {m, m, m};
}

// Half the object's largest size: how high it rests when parked
double ShapeReach(const MjThrownObject& object) {
    double reach = 0.0;
    for (int i = 0; i < ShapeSizes(object.shape); ++i) reach = std::max(reach, object.size[i] / 2.0);
    return reach;
}

// The robot's root velocity in the world (its free or slide joints)
void RootVelocity(const mjModel* m, const mjData* d, int root, double v[3]) {
    v[0] = v[1] = v[2] = 0.0;
    if (root < 0) return;
    for (int j = m->body_jntadr[root]; j < m->body_jntadr[root] + m->body_jntnum[root]; ++j) {
        const int dof = m->jnt_dofadr[j];
        if (m->jnt_type[j] == mjJNT_FREE) {
            for (int i = 0; i < 3; ++i) v[i] += d->qvel[dof + i];
        } else if (m->jnt_type[j] == mjJNT_SLIDE) {
            for (int i = 0; i < 3; ++i) v[i] += d->xaxis[3 * j + i] * d->qvel[dof];
        }
    }
}

double Height(const MjGround& ground) {
    const double d = std::clamp(ground.difficulty, 0.0, 1.0);
    switch (ground.kind) {
        case MjGroundKind::Rough: return 0.02 + 0.13 * d;
        case MjGroundKind::Steps: return 0.02 + 0.13 * d;
        case MjGroundKind::Obstacles: return 0.05 + 0.25 * d;
        default: return 0.0;
    }
}

}  // namespace

const char* ShapeName(MjShape shape) {
    switch (shape) {
        case MjShape::Ball: return "Ball";
        case MjShape::Box: return "Box";
        case MjShape::Capsule: return "Capsule";
        case MjShape::Cylinder: return "Cylinder";
    }
    return "Ball";
}

int ShapeSizes(MjShape shape) {
    switch (shape) {
        case MjShape::Ball: return 1;
        case MjShape::Box: return 3;
        case MjShape::Capsule:
        case MjShape::Cylinder: return 2;
    }
    return 1;
}

std::array<double, 3> Direction(double around, double up) {
    constexpr double kDeg = 3.14159265358979323846 / 180.0;
    const double a = around * kDeg;
    const double u = up * kDeg;
    return {std::cos(u) * std::cos(a), std::cos(u) * std::sin(a), std::sin(u)};
}

const char* GroundName(MjGroundKind kind) {
    switch (kind) {
        case MjGroundKind::Flat: return "Flat";
        case MjGroundKind::Rough: return "Rough";
        case MjGroundKind::Slopes: return "Slopes";
        case MjGroundKind::Steps: return "Steps";
        case MjGroundKind::Obstacles: return "Obstacles";
    }
    return "Flat";
}

std::vector<double> GroundHeights(const MjGround& ground, double& spacing) {
    std::vector<double> heights;
    MjRandom random(ground.seed);
    if (ground.kind == MjGroundKind::Rough) {
        // Random bumps a cell apart; the heightfield slopes between them
        spacing = 0.4;
        const int n = static_cast<int>(kGroundLength / spacing) + 1;
        const double h = Height(ground);
        heights.resize(n);
        for (int i = 0; i < n; ++i) heights[i] = i == 0 ? 0.0 : random.Uniform(0.0, h);
    } else if (ground.kind == MjGroundKind::Slopes) {
        // Hills: up and down at the slope, 2-4 m a side, never below the floor
        spacing = 0.1;
        const double slope = std::tan(0.05 + 0.30 * std::clamp(ground.difficulty, 0.0, 1.0));
        const int n = static_cast<int>(kGroundLength / spacing) + 1;
        heights.resize(n);
        double height = 0.0;
        double side_left = random.Uniform(2.0, 4.0);
        bool up = true;
        for (int i = 0; i < n; ++i) {
            heights[i] = height;
            height = std::max(0.0, height + (up ? slope : -slope) * spacing);
            side_left -= spacing;
            if (side_left <= 0.0) {
                up = !up;
                side_left = random.Uniform(2.0, 4.0);
            }
        }
    }
    return heights;
}

std::vector<MjGroundBox> GroundBoxes(const MjGround& ground) {
    std::vector<MjGroundBox> boxes;
    MjRandom random(ground.seed);
    const double h = Height(ground);
    if (ground.kind == MjGroundKind::Steps) {
        // Flights of 3-5 steps up, a landing, as many down, then flat ground
        double x = random.Uniform(1.0, 2.0);
        while (x < kGroundLength) {
            const int steps = 3 + static_cast<int>(random.Next() % 3);
            for (int level = 1; level <= steps; ++level) {
                const double tread = level == steps ? random.Uniform(1.0, 2.0) : random.Uniform(0.5, 0.9);
                boxes.push_back({x, x + tread, level * h});
                x += tread;
            }
            for (int level = steps - 1; level >= 1; --level) {
                const double tread = random.Uniform(0.5, 0.9);
                boxes.push_back({x, x + tread, level * h});
                x += tread;
            }
            x += random.Uniform(2.0, 3.0);
        }
    } else if (ground.kind == MjGroundKind::Obstacles) {
        // Blocks across the path, 2-5 m apart
        double x = random.Uniform(1.0, 3.0);
        while (x < kGroundLength) {
            const double length = random.Uniform(0.2, 0.6);
            boxes.push_back({x, x + length, random.Uniform(0.5, 1.0) * h});
            x += length + random.Uniform(2.0, 5.0);
        }
    }
    while (!boxes.empty() && boxes.back().x1 > kGroundLength) boxes.pop_back();
    return boxes;
}

bool MjSceneBuilder::Load(const std::string& model_path, std::string& error) {
    path_ = model_path;
    char parse_error[1000] = "";
    spec_.reset(mj_parseXML(model_path.c_str(), nullptr, parse_error, sizeof(parse_error)), mj_deleteSpec);
    if (!spec_) {
        error = "The model " + model_path + " did not load: " + parse_error;
        return false;
    }
    plain_.reset(mj_compile(spec_.get(), nullptr), mj_deleteModel);
    if (!plain_) {
        error = "The model " + model_path + " did not load: " + mjs_getError(spec_.get());
        spec_.reset();
        return false;
    }
    const mjModel* plain = plain_.get();
    robot_ = {static_cast<int>(plain->nq), static_cast<int>(plain->nv), static_cast<int>(plain->nbody)};
    // Where the robot starts: the ground starts ahead of it
    const int root = RobotRootBody(plain, robot_);
    mjData* d = mj_makeData(plain);
    if (!d) {
        error = "MuJoCo could not make a state for " + model_path + " (out of memory?).";
        return false;
    }
    mj_forward(plain, d);
    if (root >= 0) for (int i = 0; i < 3; ++i) start_[i] = d->xpos[3 * root + i];
    mj_deleteData(d);
    return true;
}

bool RefreshConstants(mjModel* m, const mjStatistic& stat) {
    mjData* d = mj_makeData(m);
    if (!d) return false;
    mj_setConst(m, d);
    mj_deleteData(d);
    m->stat = stat;
    return true;
}

mjModel* MjSceneBuilder::Build(const MjScene& scene, std::string& error) const {
    if (!plain_) {
        error = "No model is loaded.";
        return nullptr;
    }
    const mjModel* plain = plain_.get();
    if (scene.Empty()) return mj_copyModel(nullptr, plain);  // the file as it is
    std::unique_ptr<mjSpec, void (*)(mjSpec*)> spec(mj_copySpec(spec_.get()), mj_deleteSpec);
    if (!spec) {
        error = "MuJoCo could not copy the model (out of memory?).";
        return nullptr;
    }
    const double* start = start_;
    const Floor floor = FindFloor(plain);
    const int mask = ContactMask(plain);
    mjsBody* world = mjs_findBody(spec.get(), "world");
    const double x0 = start[0] + kGroundStart;

    double spacing = 0.0;
    const std::vector<double> heights = GroundHeights(scene.ground, spacing);
    if (heights.size() > 1) {
        // A heightfield: MuJoCo scales its samples to [0, 1] of size[2]
        const double top = *std::max_element(heights.begin(), heights.end());
        if (top > 0.0) {
            constexpr int kRows = 3;
            const int cols = static_cast<int>(heights.size());
            std::vector<float> data(static_cast<size_t>(kRows) * cols);
            for (int r = 0; r < kRows; ++r)
                for (int c = 0; c < cols; ++c) data[static_cast<size_t>(r) * cols + c] = static_cast<float>(heights[c] / top);
            mjsHField* field = mjs_addHField(spec.get());
            mjs_setName(field->element, kGroundName);
            const double half_length = spacing * (cols - 1) / 2.0;
            field->size[0] = half_length;
            field->size[1] = kGroundHalfWidth;
            field->size[2] = top;
            field->size[3] = 0.1;
            field->nrow = kRows;
            field->ncol = cols;
            mjs_setFloat(field->userdata, data.data(), static_cast<int>(data.size()));
            mjsGeom* geom = mjs_addGeom(world, nullptr);
            mjs_setName(geom->element, kGroundName);
            geom->type = mjGEOM_HFIELD;
            mjs_setString(geom->hfieldname, kGroundName);
            geom->pos[0] = x0 + half_length;
            geom->pos[1] = start[1];
            geom->pos[2] = floor.z;
            TouchLikeFloor(geom, plain, floor, mask);
            LookLikeGround(spec.get(), geom, plain, floor);
        }
    }
    const std::vector<MjGroundBox> boxes = GroundBoxes(scene.ground);
    for (size_t i = 0; i < boxes.size(); ++i) {
        const MjGroundBox& box = boxes[i];
        mjsGeom* geom = mjs_addGeom(world, nullptr);
        mjs_setName(geom->element, (std::string(kGroundName) + "_" + std::to_string(i)).c_str());
        geom->type = mjGEOM_BOX;
        geom->size[0] = (box.x1 - box.x0) / 2.0;
        geom->size[1] = kGroundHalfWidth;
        geom->size[2] = box.height / 2.0;
        geom->pos[0] = x0 + (box.x0 + box.x1) / 2.0;
        geom->pos[1] = start[1];
        geom->pos[2] = floor.z + box.height / 2.0;
        TouchLikeFloor(geom, plain, floor, mask);
        LookLikeGround(spec.get(), geom, plain, floor);
    }

    // The object slots: free bodies after everything else, parked to the side
    const size_t object_count = std::min<size_t>(scene.objects.size(), kMaxThrownObjects);
    for (size_t i = 0; i < object_count; ++i) {
        const MjThrownObject& object = scene.objects[i];
        const std::string name = "cyx_stress_object_" + std::to_string(i);
        mjsBody* body = mjs_addBody(world, nullptr);
        mjs_setName(body->element, name.c_str());
        body->pos[0] = start[0];
        body->pos[1] = start[1] + kParkY + 3.0 * static_cast<double>(i);
        body->pos[2] = floor.z + ShapeReach(object);
        mjs_addFreeJoint(body);
        mjsGeom* geom = mjs_addGeom(body, nullptr);
        mjs_setName(geom->element, name.c_str());
        ShapeGeom(object, geom);
        geom->mass = object.mass;
        TouchLikeFloor(geom, plain, floor, mask);
        mjs_setString(geom->material, "");
        std::copy(kObjectRgba, kObjectRgba + 4, geom->rgba);
    }

    mjModel* m = mj_compile(spec.get(), nullptr);
    if (!m) {
        error = std::string("The stress scene did not build: ") + mjs_getError(spec.get());
        return nullptr;
    }
    // The model's statistics stay the robot's: MuJoCo sizes the view from them
    // (300 m of ground would push the camera's near plane past the robot) and
    // scales the solver by the mean inertia
    m->stat = plain->stat;
    if (object_count > 0) {
        // A model may rescale its masses to a total (settotalmass): the robot keeps
        // the masses of its own file, each object the mass asked for
        for (int b = 1; b < plain->nbody; ++b) {
            m->body_mass[b] = plain->body_mass[b];
            for (int i = 0; i < 3; ++i) m->body_inertia[3 * b + i] = plain->body_inertia[3 * b + i];
            for (int i = 0; i < 3; ++i) m->body_ipos[3 * b + i] = plain->body_ipos[3 * b + i];
            for (int i = 0; i < 4; ++i) m->body_iquat[4 * b + i] = plain->body_iquat[4 * b + i];
        }
        for (size_t i = 0; i < object_count; ++i) {
            const int b = static_cast<int>(plain->nbody) + static_cast<int>(i);
            const MjThrownObject& object = scene.objects[i];
            const std::array<double, 3> inertia = ShapeInertia(object);
            m->body_mass[b] = object.mass;
            for (int k = 0; k < 3; ++k) m->body_inertia[3 * b + k] = inertia[k];
            for (int k = 0; k < 3; ++k) m->body_ipos[3 * b + k] = 0.0;
            m->body_iquat[4 * b] = 1.0;
            for (int k = 1; k < 4; ++k) m->body_iquat[4 * b + k] = 0.0;
        }
        if (!RefreshConstants(m, plain->stat)) {
            mj_deleteModel(m);
            error = "MuJoCo could not make a state for the stress scene (out of memory?).";
            return nullptr;
        }
    }
    return m;
}

double MjGroundProfile::At(double x) const {
    const double along = x - x0;
    if (along < 0.0) return 0.0;
    if (!heights.empty() && spacing > 0.0) {
        const double at = along / spacing;
        const size_t i = static_cast<size_t>(at);
        if (i + 1 >= heights.size()) return 0.0;
        const double t = at - static_cast<double>(i);
        return heights[i] + (heights[i + 1] - heights[i]) * t;  // the heightfield slopes between samples
    }
    for (const auto& box : boxes)
        if (along >= box.x0 && along < box.x1) return box.height;
    return 0.0;
}

MjGroundProfile GroundProfile(const MjGround& ground, double start_x) {
    MjGroundProfile profile;
    profile.x0 = start_x + kGroundStart;
    profile.heights = GroundHeights(ground, profile.spacing);
    profile.boxes = GroundBoxes(ground);
    return profile;
}

int RobotRootBody(const mjModel* m, const MjRobotSize& robot) {
    const int bodies = std::min(robot.nbody, static_cast<int>(m->nbody));
    for (int b = 1; b < bodies; ++b) {
        if (m->body_parentid[b] == 0 && m->body_jntnum[b] > 0) return b;
    }
    return -1;
}

bool RobotFlipped(const mjModel* m, const mjData* d, const MjRobotSize& robot) {
    const int root = RobotRootBody(m, robot);
    if (root < 0) return false;
    // The z component of the root's own z axis
    return d->xmat[9 * root + 8] < 0.0;
}

double RobotMass(const mjModel* m, const MjRobotSize& robot) {
    const int root = RobotRootBody(m, robot);
    return root < 0 ? 0.0 : m->body_subtreemass[root];
}

void KickRobot(const mjModel* m, mjData* d, const MjRobotSize& robot, const std::array<double, 3>& dv) {
    const int root = RobotRootBody(m, robot);
    if (root < 0) return;
    for (int j = m->body_jntadr[root]; j < m->body_jntadr[root] + m->body_jntnum[root]; ++j) {
        const int dof = m->jnt_dofadr[j];
        if (m->jnt_type[j] == mjJNT_FREE) {
            for (int i = 0; i < 3; ++i) d->qvel[dof + i] += dv[i];
        } else if (m->jnt_type[j] == mjJNT_SLIDE) {
            const mjtNum* axis = d->xaxis + 3 * j;
            d->qvel[dof] += axis[0] * dv[0] + axis[1] * dv[1] + axis[2] * dv[2];
        }
    }
}

bool ThrowObject(const mjModel* m, mjData* d, const MjRobotSize& robot, int slot, const MjThrow& how) {
    const int body = robot.nbody + slot;
    if (slot < 0 || body >= m->nbody || m->body_jntnum[body] < 1) return false;
    const int joint = m->body_jntadr[body];
    if (m->jnt_type[joint] != mjJNT_FREE) return false;
    const int root = RobotRootBody(m, robot);
    double target[3] = {0.0, 0.0, 0.5};
    if (root >= 0) for (int i = 0; i < 3; ++i) target[i] = d->subtree_com[3 * root + i];
    double velocity[3];
    RootVelocity(m, d, root, velocity);

    const std::array<double, 3> from = Direction(how.around, how.up);
    const double distance = std::max(how.distance, 0.5);
    double spawn[3];
    for (int i = 0; i < 3; ++i) spawn[i] = target[i] + from[i] * distance;
    const double speed = std::max(how.speed, 0.1);
    // Lead the robot: where it will be when the object arrives
    double aim[3] = {target[0], target[1], target[2]};
    double flight = 0.0;
    for (int pass = 0; pass < 3; ++pass) {
        double to[3] = {aim[0] - spawn[0], aim[1] - spawn[1], aim[2] - spawn[2]};
        flight = std::sqrt(to[0] * to[0] + to[1] * to[1] + to[2] * to[2]) / speed;
        for (int i = 0; i < 3; ++i) aim[i] = target[i] + velocity[i] * flight;
    }
    double direction[3] = {aim[0] - spawn[0], aim[1] - spawn[1], aim[2] - spawn[2]};
    Normalise(direction);
    const double gravity = -m->opt.gravity[2];
    const int qpos = m->jnt_qposadr[joint];
    const int dof = m->jnt_dofadr[joint];
    for (int i = 0; i < 3; ++i) d->qpos[qpos + i] = spawn[i];
    double quat[4] = {how.quat[0], how.quat[1], how.quat[2], how.quat[3]};
    if (mju_normalize4(quat) < 1e-9) quat[0] = 1.0;
    for (int i = 0; i < 4; ++i) d->qpos[qpos + 3 + i] = quat[i];
    for (int i = 0; i < 3; ++i) d->qvel[dof + i] = direction[i] * speed;
    d->qvel[dof + 2] += 0.5 * gravity * flight;  // it falls that much on the way
    // A free joint's angular velocity is in the body's own frame: the world spin turned into it
    mjtNum spin_world[3] = {how.spin[0], how.spin[1], how.spin[2]};
    mjtNum spin_local[3];
    mjtNum inverse[4];
    mju_negQuat(inverse, quat);
    mju_rotVecQuat(spin_local, spin_world, inverse);
    for (int i = 0; i < 3; ++i) d->qvel[dof + 3 + i] = spin_local[i];
    return true;
}

int PickObjectSlot(MjScene& scene, const MjThrownObject& object, int last_slot) {
    const int count = static_cast<int>(scene.objects.size());
    for (int k = 1; k <= count; ++k) {
        const int i = (last_slot + k + count) % count;
        if (i != last_slot && scene.objects[i] == object) return i;
    }
    if (count < kMaxThrownObjects) {
        scene.objects.push_back(object);
        return count;
    }
    if (last_slot >= 0 && last_slot < count && scene.objects[last_slot] == object) return last_slot;
    const int replaced = (last_slot + 1 + count) % count;
    scene.objects[replaced] = object;
    return replaced;
}

void CopySceneState(const mjModel* from_m, const mjData* from_d, const mjModel* to_m, mjData* to_d,
                    const MjRobotSize& robot) {
    to_d->time = from_d->time;
    for (int i = 0; i < robot.nq; ++i) to_d->qpos[i] = from_d->qpos[i];
    for (int i = 0; i < robot.nv; ++i) {
        to_d->qvel[i] = from_d->qvel[i];
        to_d->qacc_warmstart[i] = from_d->qacc_warmstart[i];
    }
    const int na = static_cast<int>(std::min(from_m->na, to_m->na));
    const int nu = static_cast<int>(std::min(from_m->nu, to_m->nu));
    for (int i = 0; i < na; ++i) to_d->act[i] = from_d->act[i];
    for (int i = 0; i < nu; ++i) to_d->ctrl[i] = from_d->ctrl[i];
    for (int b = 0; b < robot.nbody; ++b)
        for (int i = 0; i < 6; ++i) to_d->xfrc_applied[6 * b + i] = from_d->xfrc_applied[6 * b + i];
    // The object slots both have keep where they are and how they move
    const int kept = static_cast<int>(std::min(from_m->nbody, to_m->nbody)) - robot.nbody;
    for (int s = 0; s < kept; ++s) {
        const int b = robot.nbody + s;
        const int fj = from_m->body_jntadr[b];
        const int tj = to_m->body_jntadr[b];
        for (int i = 0; i < 7; ++i) to_d->qpos[to_m->jnt_qposadr[tj] + i] = from_d->qpos[from_m->jnt_qposadr[fj] + i];
        for (int i = 0; i < 6; ++i) to_d->qvel[to_m->jnt_dofadr[tj] + i] = from_d->qvel[from_m->jnt_dofadr[fj] + i];
    }
    mj_forward(to_m, to_d);
}

}  // namespace cyxwiz::plugin::mujoco
