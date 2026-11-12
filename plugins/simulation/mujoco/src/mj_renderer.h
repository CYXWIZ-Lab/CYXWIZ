#pragma once

// MjRenderer — Renders MuJoCo scenes to an OpenGL texture for ImGui display.
//
// MuJoCo's built-in renderer uses fixed-function OpenGL (compatibility profile).
// CyxWiz Engine uses OpenGL 3.3 Core Profile on Windows/Linux.
//
// Strategy: CPU readback approach.
// 1. mjr_makeContext + mjr_render into MuJoCo's internal offscreen FBO
//    (requires compatibility context — we create a hidden GLFW window for this)
// 2. mjr_readPixels reads pixels into CPU buffer
// 3. Upload CPU buffer to a Core Profile texture via glTexImage2D
// 4. Display in ImGui via ImGui::Image((ImTextureID)(intptr_t)texture_id, size)
//
// This decouples MuJoCo's GL requirements from the engine's GL context.

#include <mujoco/mujoco.h>
#include <mutex>
#include <vector>

struct GLFWwindow;  // Forward declare — avoid pulling in GLFW headers

namespace cyxwiz::plugin::mujoco {

class MjRenderer {
public:
    MjRenderer() = default;
    ~MjRenderer();

    // Non-copyable
    MjRenderer(const MjRenderer&) = delete;
    MjRenderer& operator=(const MjRenderer&) = delete;

    // Initialize renderer for a given model.
    // Creates a hidden GLFW compatibility context window for MuJoCo rendering.
    // Must be called from the main thread (GLFW requirement).
    // |main_window| is the engine's GLFW window (for context sharing, may be nullptr).
    bool Initialize(const mjModel* model, int width, int height, GLFWwindow* main_window);

    // Release all resources (MuJoCo context, GL texture, hidden window).
    void Shutdown();

    // The same model file with another stress scene (a ground, thrown
    // objects): the scene and the render context are made again for it; the
    // camera and the View options stay. Call from the main thread.
    bool ChangeModel(const mjModel* model);

    bool IsInitialized() const { return initialized_; }
    bool IsEnabled() const { return enabled_; }
    void SetEnabled(bool enabled) { enabled_ = enabled; }

    // Render one frame. Call from main thread.
    // Returns the Core Profile texture ID suitable for ImGui::Image().
    // Returns 0 if rendering is disabled or not initialized.
    unsigned int RenderFrame(const mjModel* model, const mjData* data);

    // Camera controls (called from ImGui mouse input). Orbit and zoom move
    // the free and the following camera; pan moves the free camera only (a
    // following camera keeps its body in the centre); a model camera stays
    // where the model puts it.
    void RotateCamera(float dx, float dy);
    void ZoomCamera(float dz);
    void PanCamera(float dx, float dy);
    // The free camera, aimed at the whole model
    void ResetCamera(const mjModel* model);

    // The camera the scene is seen through
    enum class CameraMode { Free, FollowBody, ModelCamera };
    CameraMode GetCameraMode() const;
    // The followed body or the model camera's id; -1 for the free camera
    int CameraTarget() const;
    // The free camera from where the view is now
    void UseFreeCamera();
    // Follows a body (its subtree's centre of mass) from the current angle, at
    // a distance that frames the body and what hangs from it
    void FollowBody(const mjModel* model, const mjData* data, int body_id);
    // One of the model's own <camera> elements
    void UseModelCamera(int camera_id);

    // The body under a point of the last frame (0..1 from the left and from
    // the top); -1 for none or the floor and sky (the world body)
    int BodyAt(const mjModel* model, const mjData* data, float x, float y);

    // Ctrl+drag pulls a body with a spring (MuJoCo's perturbation, as in its
    // own viewer). StartPush: the body under a point of the last frame (as
    // BodyAt), false for none. MovePush: the mouse's travel over the frame
    // height. PushForce: the pull for the body's state now, written into
    // data->xfrc_applied and `force`; the held body, -1 when none.
    bool StartPush(const mjModel* model, mjData* data, float x, float y);
    void MovePush(const mjModel* model, const mjData* data, float dx, float dy);
    int PushForce(const mjModel* model, mjData* data, mjtNum force[6]);
    void EndPush();
    bool IsPushing() const { return pert_.select > 0; }
    int PushedBody() const { return pert_.select; }

    // Render settings
    void SetResolution(int width, int height);
    int GetWidth() const { return width_; }
    int GetHeight() const { return height_; }

    // Visualization options (the View menu): an mjtVisFlag (contacts, joints,
    // centre of mass ...), an mjtRndFlag (wireframe, shadows, sky ...), the
    // frames and labels drawn, and the geom and site groups shown (0..5)
    bool VisFlag(int flag) const { return opt_.flags[flag] != 0; }
    void SetVisFlag(int flag, bool on) { opt_.flags[flag] = on ? 1 : 0; }
    bool RenderFlag(int flag) const { return scn_.flags[flag] != 0; }
    void SetRenderFlag(int flag, bool on) { scn_.flags[flag] = on ? 1 : 0; }
    int Frames() const { return opt_.frame; }
    void SetFrames(int frame) { opt_.frame = frame; }
    int Labels() const { return opt_.label; }
    void SetLabels(int label) { opt_.label = label; }
    bool GeomGroup(int group) const { return opt_.geomgroup[group] != 0; }
    void SetGeomGroup(int group, bool on) { opt_.geomgroup[group] = on ? 1 : 0; }
    bool SiteGroup(int group) const { return opt_.sitegroup[group] != 0; }
    void SetSiteGroup(int group, bool on) { opt_.sitegroup[group] = on ? 1 : 0; }

    // An endless floor: a model's level floor plane is drawn where the camera
    // looks, moved in whole texture repeats so its pattern stays put (a plane
    // collides everywhere; the model draws only its own size)
    bool EndlessFloor() const { return endless_floor_; }
    void SetEndlessFloor(bool on) { endless_floor_ = on; }

    // The last frame's pixels: RGB, bottom row first, GetWidth() x GetHeight()
    // (empty before the first frame)
    const std::vector<unsigned char>& LastFrame() const { return pixel_buffer_; }

    const mjvCamera& GetCamera() const { return cam_; }
    const mjvOption& GetOption() const { return opt_; }

    // Set physics mutex for thread-safe rendering during simulation
    void SetPhysicsMutex(std::mutex* mtx) { physics_mutex_ = mtx; }

private:
    void KeepModelCameraView(bool aim);
    void MoveFloorUnderView(const mjModel* model);
    bool CreateCompatContext(GLFWwindow* main_window);
    void MakeMainContextCurrent();
    void MakeCompatContextCurrent();
    bool AllocateDisplayTexture();
    void FreeDisplayTexture();

    // MuJoCo visualization structs
    mjvCamera cam_{};
    mjvOption opt_{};
    mjvScene scn_{};
    mjrContext con_{};
    mjvPerturb pert_{};  // the Ctrl+drag pull, drawn with the scene while held

    // Hidden GLFW window for compatibility context
    GLFWwindow* compat_window_ = nullptr;
    GLFWwindow* main_window_ = nullptr;

    // Core Profile display texture (owned by main context)
    unsigned int display_texture_ = 0;

    // CPU pixel buffer for readback
    std::vector<unsigned char> pixel_buffer_;

    int width_ = 640;
    int height_ = 480;
    bool initialized_ = false;
    bool enabled_ = true;
    bool endless_floor_ = true;
    bool resolution_changed_ = false;
    std::mutex* physics_mutex_ = nullptr;
};

} // namespace cyxwiz::plugin::mujoco
