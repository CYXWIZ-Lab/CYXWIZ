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

    bool IsInitialized() const { return initialized_; }
    bool IsEnabled() const { return enabled_; }
    void SetEnabled(bool enabled) { enabled_ = enabled; }

    // Render one frame. Call from main thread.
    // Returns the Core Profile texture ID suitable for ImGui::Image().
    // Returns 0 if rendering is disabled or not initialized.
    unsigned int RenderFrame(const mjModel* model, const mjData* data);

    // Camera controls (called from ImGui mouse input)
    void RotateCamera(float dx, float dy);
    void ZoomCamera(float dz);
    void PanCamera(float dx, float dy);
    void ResetCamera(const mjModel* model);
    void TrackBody(int body_id);

    // Render settings
    void SetResolution(int width, int height);
    int GetWidth() const { return width_; }
    int GetHeight() const { return height_; }

    // Visualization options
    void ToggleContactPoints() { opt_.flags[mjVIS_CONTACTPOINT] ^= 1; }
    void ToggleContactForces() { opt_.flags[mjVIS_CONTACTFORCE] ^= 1; }
    void ToggleWireframe() { scn_.flags[mjRND_WIREFRAME] ^= 1; }
    void ToggleJoints() { opt_.flags[mjVIS_JOINT] ^= 1; }
    void ToggleActuators() { opt_.flags[mjVIS_ACTUATOR] ^= 1; }

    const mjvCamera& GetCamera() const { return cam_; }
    const mjvOption& GetOption() const { return opt_; }

    // Set physics mutex for thread-safe rendering during simulation
    void SetPhysicsMutex(std::mutex* mtx) { physics_mutex_ = mtx; }

private:
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
    bool resolution_changed_ = false;
    std::mutex* physics_mutex_ = nullptr;
};

} // namespace cyxwiz::plugin::mujoco
