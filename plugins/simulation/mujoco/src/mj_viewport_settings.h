#pragma once

// The MuJoCo Viewport's saved choices (PluginContext::GetSetting /
// SetSetting, shown in Preferences > Simulation) and the View menu's scene
// options. No ImGui here: the panel and the Preferences section both use it,
// and the tests check it.

#include "mj_recorder.h"

#include <array>
#include <string>

namespace cyxwiz::plugin {
class PluginContext;
}

namespace cyxwiz::plugin::mujoco {

class MjRenderer;

// The camera a policy playback starts with
enum class PlaybackCamera {
    ModelTrack,  // the model's camera named "track" (Gymnasium's), else follow the root body
    FollowRoot,  // follow the root body
    Free,        // the free camera, aimed at the whole model
};

// What the View menu shows in the scene
struct SceneOptions {
    // Show in the scene
    bool contact_points = false;
    bool contact_forces = false;
    bool joints = false;
    bool actuators = false;
    bool centre_of_mass = false;
    bool inertia = false;
    bool transparent = false;
    bool convex_hulls = false;
    bool wireframe = false;
    // Rendering
    bool shadows = true;
    bool reflections = false;
    bool sky = true;
    bool fog = false;
    bool endless_floor = true;  // the model's floor drawn wherever the camera looks
    // Frames (mjtFrame) and labels (mjtLabel)
    int frames = 0;
    int labels = 0;
    // The model's visibility groups 0..5 (MuJoCo shows 0, 1 and 2)
    std::array<bool, 6> geom_groups{true, true, true, false, false, false};
    std::array<bool, 6> site_groups{true, true, true, false, false, false};

    bool operator==(const SceneOptions&) const = default;
};

struct ViewportDefaults {
    PlaybackCamera playback_camera = PlaybackCamera::ModelTrack;
    bool live_numbers = true;  // episode, step, return and action on the scene during playback
    MjRecorder::Format record_format = MjRecorder::Format::Gif;
    bool record_until_stop = false;  // false: until the episode ends
    SceneOptions scene;              // what View shows when the Viewport starts
};

// Saved text <-> values. Unknown or broken text gives the default.
std::string EncodeScene(const SceneOptions& scene);
SceneOptions DecodeScene(const std::string& text);
const char* PlaybackCameraKey(PlaybackCamera camera);
PlaybackCamera PlaybackCameraFromKey(const std::string& key);

ViewportDefaults LoadViewportDefaults(const PluginContext& ctx);
void SaveViewportDefaults(PluginContext& ctx, const ViewportDefaults& defaults);

// The renderer's options as a SceneOptions, and back
SceneOptions ReadScene(const MjRenderer& renderer);
void ApplyScene(const SceneOptions& scene, MjRenderer& renderer);

}  // namespace cyxwiz::plugin::mujoco
