#include "mj_robustness.h"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <numeric>

namespace cyxwiz::plugin::mujoco {

namespace {

constexpr const char* kSection = "Robustness (training)";

struct GroundOption {
    const char* value;
    MjGroundKind kind;
};
constexpr GroundOption kGroundOptions[] = {{"flat", MjGroundKind::Flat},
                                           {"rough", MjGroundKind::Rough},
                                           {"slopes", MjGroundKind::Slopes},
                                           {"steps", MjGroundKind::Steps},
                                           {"obstacles", MjGroundKind::Obstacles}};

PluginEnvironmentSetting Number(const char* key, const char* label, const char* value, double min, double max,
                                const char* unit, const char* description) {
    PluginEnvironmentSetting s;
    s.key = key;
    s.label = label;
    s.default_value = value;
    s.description = description;
    s.kind = PluginSettingKind::Number;
    s.section = kSection;
    s.min = min;
    s.max = max;
    s.unit = unit;
    return s;
}

bool ParseNumber(const std::string& text, double& out) {
    const char* begin = text.data();
    const char* end = begin + text.size();
    const auto result = std::from_chars(begin, end, out);
    return result.ec == std::errc() && result.ptr == end && std::isfinite(out);
}

}  // namespace

const std::vector<PluginEnvironmentSetting>& RobustnessSettings() {
    static const std::vector<PluginEnvironmentSetting> settings = [] {
        std::vector<PluginEnvironmentSetting> list;
        PluginEnvironmentSetting grounds;
        grounds.key = "grounds";
        grounds.label = "Ground";
        grounds.default_value = "";
        grounds.description = "Each episode picks one of the chosen grounds; none: the model's own floor. The ground "
                              "runs along the run from 2 m ahead of the start.";
        grounds.kind = PluginSettingKind::Choices;
        for (const auto& option : kGroundOptions) grounds.options.push_back({option.value, GroundName(option.kind)});
        grounds.section = kSection;
        list.push_back(grounds);
        list.push_back(Number("difficulty_from", "Difficulty from", "0", 0.0, 1.0, "",
                              "The easiest ground an episode gets (0 gentle, 1 hard)"));
        list.push_back(Number("difficulty_to", "Difficulty to", "1", 0.0, 1.0, "", "The hardest ground an episode gets"));
        list.push_back(Number("curriculum", "Curriculum mark", "0", 0.0, 1.0e9, "",
                              "When the mean return of recent episodes passes this, the hardest ground drawn rises by "
                              "0.1, from 'from' up to 'to'. 0: no curriculum (the whole range from the start)"));
        list.push_back(Number("push_every", "Push every", "0", 0.0, 60.0, "s",
                              "A push in a random direction this often (simulated seconds, varied by half); 0: none"));
        list.push_back(Number("push_speed", "Push up to", "0.5", 0.0, 10.0, "m/s",
                              "The largest velocity change of a push"));
        list.push_back(Number("friction_percent", "Friction \xC2\xB1", "0", 0.0, 90.0, "%",
                              "Each episode scales every friction by a factor drawn within this"));
        list.push_back(Number("mass_percent", "Mass \xC2\xB1", "0", 0.0, 90.0, "%",
                              "Each episode scales the robot's masses by a factor drawn within this"));
        list.push_back(Number("motor_percent", "Motor strength \xC2\xB1", "0", 0.0, 90.0, "%",
                              "Each episode scales the motors' strength by a factor drawn within this"));
        PluginEnvironmentSetting look;
        look.key = "look_ahead";
        look.label = "Look ahead";
        look.default_value = "0";
        look.description = "The ground's height 0.2 to 2.0 m ahead (10 values) joins what the robot observes. Without "
                           "it the robot is blind to a step or block ahead. A policy trained with it plays only with it.";
        look.kind = PluginSettingKind::Flag;
        look.section = kSection;
        list.push_back(look);
        return list;
    }();
    return settings;
}

bool IsRobustnessKey(const std::string& key) {
    for (const auto& s : RobustnessSettings())
        if (s.key == key) return true;
    return false;
}

std::map<std::string, std::string> WithoutRobustness(const std::map<std::string, std::string>& given) {
    std::map<std::string, std::string> rest;
    for (const auto& [key, value] : given)
        if (!IsRobustnessKey(key)) rest[key] = value;
    return rest;
}

bool ReadMjRobustness(const std::map<std::string, std::string>& given, MjRobustness& out, std::string& error) {
    out = MjRobustness{};
    for (const auto& setting : RobustnessSettings()) {
        const auto it = given.find(setting.key);
        if (it == given.end()) continue;
        const std::string& text = it->second;
        if (setting.kind == PluginSettingKind::Choices) {
            size_t start = 0;
            while (start <= text.size()) {
                const size_t comma = text.find(',', start);
                const std::string part = text.substr(start, comma == std::string::npos ? std::string::npos : comma - start);
                if (!part.empty()) {
                    bool known = false;
                    for (const auto& option : kGroundOptions) {
                        if (part == option.value) {
                            if (std::find(out.grounds.begin(), out.grounds.end(), option.kind) == out.grounds.end())
                                out.grounds.push_back(option.kind);
                            known = true;
                        }
                    }
                    if (!known) {
                        error = "Ground: '" + part + "' is not one of flat, rough, slopes, steps, obstacles.";
                        return false;
                    }
                }
                if (comma == std::string::npos) break;
                start = comma + 1;
            }
            continue;
        }
        if (setting.kind == PluginSettingKind::Flag) {
            if (text != "0" && text != "1") {
                error = setting.label + " must be 1 (on) or 0 (off) (got '" + text + "').";
                return false;
            }
            out.look_ahead = text == "1";
            continue;
        }
        double value = 0.0;
        if (!ParseNumber(text, value)) {
            error = setting.label + " must be a number (got '" + text + "').";
            return false;
        }
        if (value < setting.min || value > setting.max) {
            char range[96];
            std::snprintf(range, sizeof(range), " must be from %g to %g (got '%s').", setting.min, setting.max, text.c_str());
            error = setting.label + range;
            return false;
        }
        if (setting.key == "difficulty_from") out.difficulty_from = value;
        else if (setting.key == "difficulty_to") out.difficulty_to = value;
        else if (setting.key == "curriculum") out.curriculum_mark = value;
        else if (setting.key == "push_every") out.push_every = value;
        else if (setting.key == "push_speed") out.push_speed = value;
        else if (setting.key == "friction_percent") out.friction_percent = value;
        else if (setting.key == "mass_percent") out.mass_percent = value;
        else if (setting.key == "motor_percent") out.motor_percent = value;
    }
    if (out.difficulty_from > out.difficulty_to) {
        error = "Difficulty from must not be above Difficulty to.";
        return false;
    }
    return true;
}

MjEpisodeDraw DrawEpisode(const MjRobustness& r, double level, MjRandom& random) {
    MjEpisodeDraw draw;
    if (!r.grounds.empty()) {
        draw.ground.kind = r.grounds[static_cast<size_t>(random.Next() % r.grounds.size())];
        const double top = std::clamp(level, r.difficulty_from, r.difficulty_to);
        draw.ground.difficulty = random.Uniform(r.difficulty_from, top);
        draw.ground.seed = random.Next() >> 16;  // a seed that reads back as a whole number
    }
    const auto factor = [&random](double percent) {
        return percent > 0.0 ? 1.0 + random.Uniform(-percent, percent) / 100.0 : 1.0;
    };
    draw.friction = factor(r.friction_percent);
    draw.mass = factor(r.mass_percent);
    draw.motor = factor(r.motor_percent);
    return draw;
}

void ApplyFactors(const MjEpisodeDraw& draw, const MjRobotSize& robot, mjModel* m) {
    if (draw.friction != 1.0)
        for (int g = 0; g < m->ngeom; ++g) m->geom_friction[3 * g] *= draw.friction;  // sliding friction
    if (draw.mass != 1.0) {
        const int bodies = std::min(robot.nbody, static_cast<int>(m->nbody));
        for (int b = 1; b < bodies; ++b) {
            m->body_mass[b] *= draw.mass;
            for (int i = 0; i < 3; ++i) m->body_inertia[3 * b + i] *= draw.mass;
        }
    }
    if (draw.motor != 1.0) {
        // The force is gain * control + bias: both scale (a muscle's own model is left as it is)
        for (int a = 0; a < m->nu; ++a) {
            const int gain = m->actuator_gaintype[a];
            if (gain != mjGAIN_FIXED && gain != mjGAIN_AFFINE) continue;
            for (int k = 0; k < 3; ++k) m->actuator_gainprm[mjNGAIN * a + k] *= draw.motor;
            if (m->actuator_biastype[a] == mjBIAS_AFFINE)
                for (int k = 0; k < 3; ++k) m->actuator_biasprm[mjNBIAS * a + k] *= draw.motor;
        }
    }
}

void MjCurriculum::Start(const MjRobustness& r, int copies) {
    mark_ = r.curriculum_mark;
    to_ = r.difficulty_to;
    level_ = mark_ > 0.0 ? r.difficulty_from : r.difficulty_to;
    window_ = static_cast<size_t>(std::max(20, 2 * copies));
    recent_.clear();
}

bool MjCurriculum::Finished(double episode_return) {
    if (mark_ <= 0.0 || level_ >= to_) return false;
    recent_.push_back(episode_return);
    if (recent_.size() < window_) return false;
    const double mean = std::accumulate(recent_.begin(), recent_.end(), 0.0) / static_cast<double>(recent_.size());
    recent_.erase(recent_.begin());
    if (mean < mark_) return false;
    level_ = std::min(to_, level_ + kCurriculumStep);
    recent_.clear();  // the next rise is judged on the harder ground
    return true;
}

}  // namespace cyxwiz::plugin::mujoco
