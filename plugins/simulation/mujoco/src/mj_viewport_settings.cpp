#include "mj_viewport_settings.h"
#include "mj_renderer.h"

#include "plugin/plugin_context.h"

#include <mujoco/mujoco.h>
#include <nlohmann/json.hpp>

#include <algorithm>

namespace cyxwiz::plugin::mujoco {
namespace {

constexpr const char* kPlaybackCamera = "viewport.playback_camera";
constexpr const char* kLiveNumbers = "viewport.live_numbers";
constexpr const char* kRecordFormat = "viewport.record_format";
constexpr const char* kRecordLength = "viewport.record_length";
constexpr const char* kScene = "viewport.scene";
constexpr const char* kStress = "viewport.stress";

// The switches of SceneOptions by their saved names
struct Switch {
    const char* name;
    bool SceneOptions::*field;
};
constexpr Switch kSwitches[] = {
    {"contact_points", &SceneOptions::contact_points}, {"contact_forces", &SceneOptions::contact_forces},
    {"joints", &SceneOptions::joints},                 {"actuators", &SceneOptions::actuators},
    {"centre_of_mass", &SceneOptions::centre_of_mass}, {"inertia", &SceneOptions::inertia},
    {"transparent", &SceneOptions::transparent},       {"convex_hulls", &SceneOptions::convex_hulls},
    {"wireframe", &SceneOptions::wireframe},           {"shadows", &SceneOptions::shadows},
    {"reflections", &SceneOptions::reflections},       {"sky", &SceneOptions::sky},
    {"fog", &SceneOptions::fog},                       {"endless_floor", &SceneOptions::endless_floor},
};

std::string Groups(const std::array<bool, 6>& groups) {
    std::string text;
    for (const bool on : groups) text += on ? '1' : '0';
    return text;
}

void ReadGroups(const nlohmann::json& j, const char* key, std::array<bool, 6>& groups) {
    if (!j.contains(key) || !j[key].is_string()) return;
    const std::string text = j[key].get<std::string>();
    if (text.size() != groups.size()) return;
    for (size_t i = 0; i < groups.size(); ++i) groups[i] = text[i] == '1';
}

// A saved number in [low, high], else the default
void ReadNumber(const nlohmann::json& j, const char* key, double low, double high, double& value) {
    if (!j.contains(key) || !j[key].is_number()) return;
    const double v = j[key].get<double>();
    if (v >= low && v <= high) value = v;
}

const char* FromKey(MjThrowFrom from) {
    switch (from) {
        case MjThrowFrom::Behind: return "behind";
        case MjThrowFrom::Above: return "above";
        case MjThrowFrom::Ahead: break;
    }
    return "ahead";
}

}  // namespace

std::string EncodeStress(const StressOptions& s) {
    nlohmann::json j;
    j["panel_open"] = s.panel_open;
    j["kick_speed"] = s.kick_speed;
    j["object"] = s.object.ball ? "ball" : "box";
    j["size"] = s.object.size;
    j["mass"] = s.object.mass;
    j["throw_speed"] = s.throw_speed;
    j["from"] = FromKey(s.from);
    j["auto_disturb"] = s.auto_disturb;
    j["auto_seconds"] = s.auto_seconds;
    j["ground"] = GroundName(s.ground.kind);
    j["difficulty"] = s.ground.difficulty;
    j["seed"] = s.ground.seed;
    return j.dump();
}

StressOptions DecodeStress(const std::string& text) {
    using L = StressLimits;
    StressOptions s;
    if (text.empty()) return s;
    const nlohmann::json j = nlohmann::json::parse(text, nullptr, false);
    if (!j.is_object()) return s;
    if (j.contains("panel_open") && j["panel_open"].is_boolean()) s.panel_open = j["panel_open"].get<bool>();
    if (j.contains("auto_disturb") && j["auto_disturb"].is_boolean()) s.auto_disturb = j["auto_disturb"].get<bool>();
    ReadNumber(j, "kick_speed", L::kKickMin, L::kKickMax, s.kick_speed);
    if (j.contains("object") && j["object"].is_string()) s.object.ball = j["object"].get<std::string>() != "box";
    ReadNumber(j, "size", L::kSizeMin, L::kSizeMax, s.object.size);
    ReadNumber(j, "mass", L::kMassMin, L::kMassMax, s.object.mass);
    ReadNumber(j, "throw_speed", L::kSpeedMin, L::kSpeedMax, s.throw_speed);
    if (j.contains("from") && j["from"].is_string()) {
        const std::string from = j["from"].get<std::string>();
        s.from = from == "behind" ? MjThrowFrom::Behind : from == "above" ? MjThrowFrom::Above : MjThrowFrom::Ahead;
    }
    ReadNumber(j, "auto_seconds", L::kAutoMin, L::kAutoMax, s.auto_seconds);
    if (j.contains("ground") && j["ground"].is_string()) {
        const std::string name = j["ground"].get<std::string>();
        for (const MjGroundKind kind : {MjGroundKind::Flat, MjGroundKind::Rough, MjGroundKind::Slopes,
                                        MjGroundKind::Steps, MjGroundKind::Obstacles})
            if (name == GroundName(kind)) s.ground.kind = kind;
    }
    ReadNumber(j, "difficulty", 0.0, 1.0, s.ground.difficulty);
    if (j.contains("seed") && j["seed"].is_number_unsigned()) s.ground.seed = j["seed"].get<uint64_t>();
    return s;
}

std::string EncodeScene(const SceneOptions& scene) {
    nlohmann::json j;
    for (const auto& s : kSwitches) j[s.name] = scene.*(s.field);
    j["frames"] = scene.frames;
    j["labels"] = scene.labels;
    j["geom_groups"] = Groups(scene.geom_groups);
    j["site_groups"] = Groups(scene.site_groups);
    return j.dump();
}

SceneOptions DecodeScene(const std::string& text) {
    SceneOptions scene;
    if (text.empty()) return scene;
    const nlohmann::json j = nlohmann::json::parse(text, nullptr, false);
    if (!j.is_object()) return scene;
    for (const auto& s : kSwitches)
        if (j.contains(s.name) && j[s.name].is_boolean()) scene.*(s.field) = j[s.name].get<bool>();
    if (j.contains("frames") && j["frames"].is_number_integer()) {
        const int frames = j["frames"].get<int>();
        if (frames >= 0 && frames < mjNFRAME) scene.frames = frames;
    }
    if (j.contains("labels") && j["labels"].is_number_integer()) {
        const int labels = j["labels"].get<int>();
        if (labels >= 0 && labels < mjNLABEL) scene.labels = labels;
    }
    ReadGroups(j, "geom_groups", scene.geom_groups);
    ReadGroups(j, "site_groups", scene.site_groups);
    return scene;
}

const char* PlaybackCameraKey(PlaybackCamera camera) {
    switch (camera) {
        case PlaybackCamera::FollowRoot: return "follow";
        case PlaybackCamera::Free: return "free";
        case PlaybackCamera::ModelTrack: break;
    }
    return "track";
}

PlaybackCamera PlaybackCameraFromKey(const std::string& key) {
    if (key == "follow") return PlaybackCamera::FollowRoot;
    if (key == "free") return PlaybackCamera::Free;
    return PlaybackCamera::ModelTrack;
}

ViewportDefaults LoadViewportDefaults(const PluginContext& ctx) {
    ViewportDefaults d;
    d.playback_camera = PlaybackCameraFromKey(ctx.GetSetting(kPlaybackCamera, "track"));
    d.live_numbers = ctx.GetSetting(kLiveNumbers, "on") != "off";
    d.record_format = ctx.GetSetting(kRecordFormat, "gif") == "png" ? MjRecorder::Format::PngFrames
                                                                    : MjRecorder::Format::Gif;
    d.record_until_stop = ctx.GetSetting(kRecordLength, "episode") == "stop";
    d.scene = DecodeScene(ctx.GetSetting(kScene));
    d.stress = DecodeStress(ctx.GetSetting(kStress));
    return d;
}

void SaveViewportDefaults(PluginContext& ctx, const ViewportDefaults& d) {
    ctx.SetSetting(kPlaybackCamera, PlaybackCameraKey(d.playback_camera));
    ctx.SetSetting(kLiveNumbers, d.live_numbers ? "on" : "off");
    ctx.SetSetting(kRecordFormat, d.record_format == MjRecorder::Format::PngFrames ? "png" : "gif");
    ctx.SetSetting(kRecordLength, d.record_until_stop ? "stop" : "episode");
    ctx.SetSetting(kScene, EncodeScene(d.scene));
    ctx.SetSetting(kStress, EncodeStress(d.stress));
}

SceneOptions ReadScene(const MjRenderer& r) {
    SceneOptions s;
    s.contact_points = r.VisFlag(mjVIS_CONTACTPOINT);
    s.contact_forces = r.VisFlag(mjVIS_CONTACTFORCE);
    s.joints = r.VisFlag(mjVIS_JOINT);
    s.actuators = r.VisFlag(mjVIS_ACTUATOR);
    s.centre_of_mass = r.VisFlag(mjVIS_COM);
    s.inertia = r.VisFlag(mjVIS_INERTIA);
    s.transparent = r.VisFlag(mjVIS_TRANSPARENT);
    s.convex_hulls = r.VisFlag(mjVIS_CONVEXHULL);
    s.wireframe = r.RenderFlag(mjRND_WIREFRAME);
    s.shadows = r.RenderFlag(mjRND_SHADOW);
    s.reflections = r.RenderFlag(mjRND_REFLECTION);
    s.sky = r.RenderFlag(mjRND_SKYBOX);
    s.fog = r.RenderFlag(mjRND_FOG);
    s.endless_floor = r.EndlessFloor();
    s.frames = r.Frames();
    s.labels = r.Labels();
    for (int g = 0; g < 6; ++g) {
        s.geom_groups[g] = r.GeomGroup(g);
        s.site_groups[g] = r.SiteGroup(g);
    }
    return s;
}

void ApplyScene(const SceneOptions& s, MjRenderer& r) {
    r.SetVisFlag(mjVIS_CONTACTPOINT, s.contact_points);
    r.SetVisFlag(mjVIS_CONTACTFORCE, s.contact_forces);
    r.SetVisFlag(mjVIS_JOINT, s.joints);
    r.SetVisFlag(mjVIS_ACTUATOR, s.actuators);
    r.SetVisFlag(mjVIS_COM, s.centre_of_mass);
    r.SetVisFlag(mjVIS_INERTIA, s.inertia);
    r.SetVisFlag(mjVIS_TRANSPARENT, s.transparent);
    r.SetVisFlag(mjVIS_CONVEXHULL, s.convex_hulls);
    r.SetRenderFlag(mjRND_WIREFRAME, s.wireframe);
    r.SetRenderFlag(mjRND_SHADOW, s.shadows);
    r.SetRenderFlag(mjRND_REFLECTION, s.reflections);
    r.SetRenderFlag(mjRND_SKYBOX, s.sky);
    r.SetRenderFlag(mjRND_FOG, s.fog);
    r.SetEndlessFloor(s.endless_floor);
    r.SetFrames(s.frames);
    r.SetLabels(s.labels);
    for (int g = 0; g < 6; ++g) {
        r.SetGeomGroup(g, s.geom_groups[g]);
        r.SetSiteGroup(g, s.site_groups[g]);
    }
}

}  // namespace cyxwiz::plugin::mujoco
