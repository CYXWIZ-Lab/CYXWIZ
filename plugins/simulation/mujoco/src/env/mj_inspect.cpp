#include "mj_inspect.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <system_error>

namespace cyxwiz::plugin::mujoco {

std::string BodyName(const mjModel* m, int body) {
    const char* name = mj_id2name(m, mjOBJ_BODY, body);
    return name && *name ? std::string(name) : "body " + std::to_string(body);
}

MjBodyInfo InspectBody(const mjModel* m, const mjData* d, int body, const MjRobotSize& robot) {
    MjBodyInfo info;
    if (!m || !d || body <= 0 || body >= m->nbody) return info;
    info.id = body;
    info.name = BodyName(m, body);
    info.parent = m->body_parentid[body];
    info.parent_name = info.parent > 0 ? BodyName(m, info.parent) : std::string("the world");
    info.mass = m->body_mass[body];
    const int robot_bodies = std::min(robot.nbody, static_cast<int>(m->nbody));
    for (int b = 1; b < robot_bodies; ++b) info.robot_mass += m->body_mass[b];
    for (int i = 0; i < 3; ++i) info.centre_of_mass[i] = d->xipos[3 * body + i];
    // [angular, linear] of its centre of mass, in world axes
    mjtNum velocity[6];
    mj_objectVelocity(m, d, mjOBJ_BODY, body, velocity, 0);
    info.angular_speed = mju_norm3(velocity);
    info.speed = mju_norm3(velocity + 3);
    for (int j = m->body_jntadr[body]; j >= 0 && j < m->body_jntadr[body] + m->body_jntnum[body]; ++j) {
        MjJointInfo joint;
        joint.id = j;
        const char* name = mj_id2name(m, mjOBJ_JOINT, j);
        joint.name = name && *name ? name : "joint " + std::to_string(j);
        joint.type = m->jnt_type[j];
        joint.scalar = joint.type == mjJNT_HINGE || joint.type == mjJNT_SLIDE;
        if (joint.scalar) {
            joint.value = d->qpos[m->jnt_qposadr[j]];
            joint.velocity = d->qvel[m->jnt_dofadr[j]];
            joint.damping = m->dof_damping[m->jnt_dofadr[j]];
        }
        joint.limited = m->jnt_limited[j] != 0;
        joint.low = m->jnt_range[2 * j];
        joint.high = m->jnt_range[2 * j + 1];
        joint.stiffness = m->jnt_stiffness[j];
        info.joints.push_back(std::move(joint));
    }
    return info;
}

std::vector<MjActuatorInfo> InspectActuators(const mjModel* m, const mjData* d) {
    std::vector<MjActuatorInfo> out;
    if (!m || !d) return out;
    for (int a = 0; a < m->nu; ++a) {
        MjActuatorInfo info;
        info.id = a;
        const char* name = mj_id2name(m, mjOBJ_ACTUATOR, a);
        info.name = name && *name ? name : "actuator " + std::to_string(a);
        info.control = d->ctrl[a];
        info.limited = m->actuator_ctrllimited[a] != 0;
        info.low = m->actuator_ctrlrange[2 * a];
        info.high = m->actuator_ctrlrange[2 * a + 1];
        info.force = d->actuator_force[a] * m->actuator_gear[6 * a];  // at the joint: the gear times its force
        out.push_back(std::move(info));
    }
    return out;
}

bool SetJointValue(const mjModel* m, mjData* d, int joint, double value) {
    if (!m || !d || joint < 0 || joint >= m->njnt) return false;
    const int type = m->jnt_type[joint];
    if (type != mjJNT_HINGE && type != mjJNT_SLIDE) return false;
    if (m->jnt_limited[joint]) value = std::clamp(value, m->jnt_range[2 * joint], m->jnt_range[2 * joint + 1]);
    d->qpos[m->jnt_qposadr[joint]] = value;
    mj_forward(m, d);
    return true;
}

MjKeyframe CaptureKeyframe(const mjModel* m, const mjData* d, std::string name, std::string saved) {
    MjKeyframe key;
    key.name = std::move(name);
    key.saved = std::move(saved);
    key.state.resize(static_cast<size_t>(mj_stateSize(m, kKeyframeState)));
    mj_getState(m, d, key.state.data(), kKeyframeState);
    return key;
}

bool ApplyKeyframe(const mjModel* m, mjData* d, const MjKeyframe& keyframe, std::string& error) {
    const size_t size = static_cast<size_t>(mj_stateSize(m, kKeyframeState));
    if (keyframe.state.size() != size) {
        error = "\"" + keyframe.name + "\" was saved for another version of this model (" +
                std::to_string(keyframe.state.size()) + " values; the model has " + std::to_string(size) + ").";
        return false;
    }
    mj_setState(m, d, keyframe.state.data(), kKeyframeState);
    mj_forward(m, d);
    return true;
}

std::vector<std::string> ModelKeyframes(const mjModel* m) {
    std::vector<std::string> names;
    if (!m) return names;
    for (int k = 0; k < m->nkey; ++k) {
        const char* name = mj_id2name(m, mjOBJ_KEY, k);
        names.push_back(name && *name ? name : "key " + std::to_string(k));
    }
    return names;
}

std::filesystem::path KeyframesFile(const std::filesystem::path& project, const std::string& model_path) {
    // The model file within the project (it moves with it), else its full path;
    // FNV-1a, the same on every machine
    const std::filesystem::path model = std::filesystem::path(model_path).lexically_normal();
    const std::filesystem::path inside = model.lexically_relative(project.lexically_normal());
    const bool in_project = !inside.empty() && *inside.begin() != "..";
    const std::string key = (in_project ? inside : model).generic_string();
    uint32_t hash = 2166136261u;
    for (const unsigned char c : key) hash = (hash ^ c) * 16777619u;
    char suffix[16];
    std::snprintf(suffix, sizeof(suffix), "-%08x", hash);
    return project / "mujoco" / "keyframes" / (model.stem().string() + suffix + ".json");
}

std::string EncodeKeyframes(const std::vector<MjKeyframe>& keyframes, const std::string& model_path) {
    nlohmann::json j;
    j["model"] = std::filesystem::path(model_path).filename().string();
    j["state"] = "integration";
    j["keyframes"] = nlohmann::json::array();
    for (const auto& k : keyframes) j["keyframes"].push_back({{"name", k.name}, {"saved", k.saved}, {"values", k.state}});
    return j.dump(1);
}

std::vector<MjKeyframe> DecodeKeyframes(const std::string& text) {
    std::vector<MjKeyframe> out;
    const nlohmann::json j = nlohmann::json::parse(text, nullptr, false);
    if (!j.is_object() || !j.contains("keyframes") || !j["keyframes"].is_array()) return out;
    for (const auto& k : j["keyframes"]) {
        if (!k.is_object() || !k.contains("name") || !k["name"].is_string() || !k.contains("values") ||
            !k["values"].is_array())
            continue;
        MjKeyframe key;
        key.name = k["name"].get<std::string>();
        if (k.contains("saved") && k["saved"].is_string()) key.saved = k["saved"].get<std::string>();
        bool numbers = true;
        for (const auto& v : k["values"]) {
            if (!v.is_number()) {
                numbers = false;
                break;
            }
            key.state.push_back(v.get<double>());
        }
        if (numbers) out.push_back(std::move(key));
    }
    return out;
}

bool SaveKeyframes(const std::filesystem::path& file, const std::vector<MjKeyframe>& keyframes,
                   const std::string& model_path, std::string& error) {
    std::error_code ec;
    std::filesystem::create_directories(file.parent_path(), ec);
    if (ec) {
        error = "Could not create " + file.parent_path().string() + ": " + ec.message();
        return false;
    }
    std::ofstream out(file, std::ios::binary | std::ios::trunc);
    out << EncodeKeyframes(keyframes, model_path);
    if (!out) {
        error = "Could not write " + file.string() + ".";
        return false;
    }
    return true;
}

std::vector<MjKeyframe> LoadKeyframes(const std::filesystem::path& file) {
    std::ifstream in(file, std::ios::binary);
    if (!in) return {};
    std::stringstream text;
    text << in.rdbuf();
    return DecodeKeyframes(text.str());
}

}  // namespace cyxwiz::plugin::mujoco
