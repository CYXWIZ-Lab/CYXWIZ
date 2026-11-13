#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include "mj_renderer.h"

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <cmath>
#include <cstring>

namespace cyxwiz::plugin::mujoco {

MjRenderer::~MjRenderer() {
    Shutdown();
}

// =============================================================================
// Initialization
// =============================================================================

bool MjRenderer::Initialize(const mjModel* model, int width, int height,
                             GLFWwindow* main_window) {
    if (initialized_) {
        spdlog::warn("[MjRenderer] Already initialized — call Shutdown() first");
        return false;
    }
    if (!model) {
        spdlog::error("[MjRenderer] Cannot initialize with null model");
        return false;
    }

    width_ = width;
    height_ = height;
    main_window_ = main_window;

    // Step 1: Create hidden compatibility context for MuJoCo rendering
    if (!CreateCompatContext(main_window)) {
        return false;
    }

    // Step 2: Initialize MuJoCo visualization structs (in compat context)
    MakeCompatContextCurrent();

    // Load glad for the compat context (DLL has its own glad copy)
    gladLoadGLLoader(reinterpret_cast<GLADloadproc>(glfwGetProcAddress));

    mjv_defaultCamera(&cam_);
    mjv_defaultOption(&opt_);
    mjv_defaultScene(&scn_);
    mjr_defaultContext(&con_);
    mjv_defaultPerturb(&pert_);

    // Enable headlight for consistent scene lighting
    scn_.enabletransform = 0;
    scn_.flags[mjRND_SHADOW] = 1;
    scn_.flags[mjRND_REFLECTION] = 0;
    scn_.flags[mjRND_SKYBOX] = 1;

    // Create MuJoCo scene (up to 2000 geometric objects)
    mjv_makeScene(model, &scn_, 2000);

    // Create MuJoCo rendering context — this calls OpenGL functions
    // and needs the compatibility context active
    mjr_makeContext(model, &con_, mjFONTSCALE_150);

    if (con_.currentBuffer == -1) {
        spdlog::error("[MjRenderer] mjr_makeContext failed — no valid GL context");
        mjv_freeScene(&scn_);
        Shutdown();
        return false;
    }

    // The offscreen buffer starts at the model's <visual> size (640x480 by
    // default): size it to the panel
    mjr_resizeOffscreen(width_, height_, &con_);

    // Use the actual offscreen FBO size — may be smaller than requested
    if (con_.offWidth > 0 && con_.offHeight > 0) {
        if (con_.offWidth < width_ || con_.offHeight < height_) {
            spdlog::warn("[MjRenderer] Offscreen FBO {}x{} smaller than requested {}x{}, clamping",
                         con_.offWidth, con_.offHeight, width_, height_);
            width_ = con_.offWidth;
            height_ = con_.offHeight;
        }
    }

    spdlog::info("[MjRenderer] MuJoCo rendering context created ({}x{}, offscreen={}x{})",
                 width_, height_, con_.offWidth, con_.offHeight);

    // Step 3: Switch back to main context and create display texture
    MakeMainContextCurrent();

    // Reload glad function pointers for the DLL's copy of glad.
    // The engine initialized glad for its own copy, but this DLL has
    // a separate static copy that needs initialization.
    if (!gladLoadGLLoader(reinterpret_cast<GLADloadproc>(glfwGetProcAddress))) {
        spdlog::error("[MjRenderer] Failed to reload glad GL function pointers");
        Shutdown();
        return false;
    }

    if (!AllocateDisplayTexture()) {
        Shutdown();
        return false;
    }

    // Allocate CPU pixel buffer (RGB, bottom-up from MuJoCo)
    pixel_buffer_.resize(width_ * height_ * 3);

    initialized_ = true;
    spdlog::info("[MjRenderer] Initialized — texture_id={}", display_texture_);
    return true;
}

void MjRenderer::Shutdown() {
    if (display_texture_ != 0) {
        // Ensure main context is current for GL cleanup
        if (main_window_) {
            MakeMainContextCurrent();
        }
        FreeDisplayTexture();
    }

    if (compat_window_) {
        MakeCompatContextCurrent();
        mjr_freeContext(&con_);
        mjv_freeScene(&scn_);

        glfwDestroyWindow(compat_window_);
        compat_window_ = nullptr;
    }

    if (main_window_) {
        MakeMainContextCurrent();
    }

    pixel_buffer_.clear();
    initialized_ = false;
    spdlog::info("[MjRenderer] Shut down");
}

// =============================================================================
// Rendering
// =============================================================================

unsigned int MjRenderer::RenderFrame(const mjModel* model, const mjData* data) {
    if (!initialized_ || !enabled_ || !model || !data) {
        return 0;
    }

    // Handle resolution change
    if (resolution_changed_) {
        resolution_changed_ = false;

        // Reallocate CPU buffer
        pixel_buffer_.resize(width_ * height_ * 3);

        // Resize only the offscreen buffer (compat context); the meshes and
        // textures stay uploaded
        MakeCompatContextCurrent();
        mjr_resizeOffscreen(width_, height_, &con_);

        // Clamp to actual offscreen FBO size
        if (con_.offWidth > 0 && con_.offHeight > 0) {
            if (con_.offWidth < width_) width_ = con_.offWidth;
            if (con_.offHeight < height_) height_ = con_.offHeight;
            pixel_buffer_.resize(width_ * height_ * 3);
        }

        // Switch back to main context and reallocate display texture
        MakeMainContextCurrent();
        gladLoadGLLoader(reinterpret_cast<GLADloadproc>(glfwGetProcAddress));
        FreeDisplayTexture();
        AllocateDisplayTexture();
    }

    // Step 1: Render in compatibility context
    MakeCompatContextCurrent();

    // Update abstract scene (lock physics mutex to prevent concurrent mjData access)
    // The pull while held, else the selected body: MuJoCo draws it highlighted,
    // and names it when no labels are on
    mjvPerturb selection;
    mjv_defaultPerturb(&selection);
    const bool selected = !IsPushing() && selected_ > 0 && selected_ < model->nbody;
    selection.select = selected ? selected_ : 0;
    const mjvPerturb* pert = IsPushing() ? &pert_ : selected ? &selection : nullptr;
    mjvOption opt = opt_;
    if (selected && opt.label == mjLABEL_NONE) opt.label = mjLABEL_SELECTION;
    mjtNum marker[3] = {0.0, 0.0, 0.0};
    const auto update = [&]() {
        mjv_updateScene(model, const_cast<mjData*>(data), &opt, pert, &cam_, mjCAT_ALL, &scn_);
        if (selected) mju_copy3(marker, data->xipos + 3 * selected_);
    };
    if (physics_mutex_) {
        std::lock_guard<std::mutex> phys_lock(*physics_mutex_);
        update();
    } else {
        update();
    }
    // A small sphere at the selected body's centre of mass
    if (selected && marker_rgba_[3] > 0.0f && scn_.ngeom < scn_.maxgeom) {
        const mjtNum size[3] = {0.03, 0.03, 0.03};
        const mjtNum identity[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
        mjvGeom* geom = scn_.geoms + scn_.ngeom++;
        mjv_initGeom(geom, mjGEOM_SPHERE, size, marker, identity, marker_rgba_);
        geom->category = mjCAT_DECOR;
    }

    if (endless_floor_) MoveFloorUnderView(model);

    // Render to offscreen buffer
    mjrRect viewport = {0, 0, width_, height_};
    mjr_setBuffer(mjFB_OFFSCREEN, &con_);
    mjr_render(viewport, &scn_, &con_);

    // Read pixels into CPU buffer (RGB, bottom-up)
    // Re-set offscreen buffer (mjr_render may change it) and set pack alignment
    // for tightly packed RGB rows (mjr_readPixels calls glReadPixels internally)
    mjr_setBuffer(mjFB_OFFSCREEN, &con_);
    glPixelStorei(GL_PACK_ALIGNMENT, 1);
    mjr_readPixels(pixel_buffer_.data(), nullptr, viewport, &con_);
    glPixelStorei(GL_PACK_ALIGNMENT, 4);

    // Step 2: Upload to display texture (main context)
    MakeMainContextCurrent();

    glBindTexture(GL_TEXTURE_2D, display_texture_);

    // RGB data (3 bytes/pixel) rows may not be 4-byte aligned.
    // Set alignment to 1 to avoid shifted/garbled pixel rows.
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

    // MuJoCo outputs bottom-up RGB. OpenGL glTexImage2D with default
    // unpack settings also expects bottom-up, so no flip needed.
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width_, height_,
                    GL_RGB, GL_UNSIGNED_BYTE, pixel_buffer_.data());

    glPixelStorei(GL_UNPACK_ALIGNMENT, 4);  // Restore default
    glBindTexture(GL_TEXTURE_2D, 0);

    return display_texture_;
}

void MjRenderer::MoveFloorUnderView(const mjModel* model) {
    const mjvGLCamera head = mjv_averageCamera(&scn_.camera[0], &scn_.camera[1]);
    for (int i = 0; i < scn_.ngeom; ++i) {
        mjvGeom& g = scn_.geoms[i];
        if (g.type != mjGEOM_PLANE || g.objtype != mjOBJ_GEOM || g.objid < 0 || g.objid >= model->ngeom) continue;
        if (model->geom_bodyid[g.objid] != 0 || g.mat[8] < 0.999f) continue;  // level and part of the world only
        if (g.size[0] <= 0.0f || g.size[1] <= 0.0f) continue;                  // drawn endless already
        // Where the view is: ahead of the camera by half the floor's size
        float fx = head.forward[0], fy = head.forward[1];
        const float flat = std::sqrt(fx * fx + fy * fy);
        if (flat > 1e-6f) {
            fx /= flat;
            fy /= flat;
        }
        const float target[2] = {head.pos[0] + fx * 0.5f * g.size[0], head.pos[1] + fy * 0.5f * g.size[1]};
        // One texture repeat along each side, so the moved floor looks the same
        float period[2] = {2.0f * g.size[0], 2.0f * g.size[1]};
        const int mat = model->geom_matid[g.objid];
        if (mat >= 0) {
            for (int a = 0; a < 2; ++a) {
                const float repeat = model->mat_texrepeat[2 * mat + a];
                if (repeat > 0.0f)
                    period[a] = model->mat_texuniform[mat] ? 1.0f / repeat : 2.0f * g.size[a] / repeat;
            }
        }
        for (int a = 0; a < 2; ++a) {
            const float shift = target[a] - static_cast<float>(model->geom_pos[3 * g.objid + a]);
            g.pos[a] = static_cast<float>(model->geom_pos[3 * g.objid + a]) + std::round(shift / period[a]) * period[a];
        }
    }
}

// =============================================================================
// Camera Controls
// =============================================================================

void MjRenderer::RotateCamera(float dx, float dy) {
    // Scale for reasonable rotation speed
    cam_.azimuth += dx * 0.3f;
    cam_.elevation += dy * 0.3f;
    // Clamp elevation to avoid gimbal issues
    if (cam_.elevation > 89.0f) cam_.elevation = 89.0f;
    if (cam_.elevation < -89.0f) cam_.elevation = -89.0f;
}

void MjRenderer::ZoomCamera(float dz) {
    cam_.distance *= (1.0f - dz * 0.12f);
    if (cam_.distance < 0.01f) cam_.distance = 0.01f;
    if (cam_.distance > 100.0f) cam_.distance = 100.0f;
}

void MjRenderer::PanCamera(float dx, float dy) {
    if (cam_.type != mjCAMERA_FREE) return;  // a following camera re-centres on its body
    // Pan by adjusting lookat position
    double azimuth_rad = cam_.azimuth * M_PI / 180.0;
    cam_.lookat[0] += (-dx * cos(azimuth_rad) - dy * sin(azimuth_rad)) * 0.01;
    cam_.lookat[1] += (dx * sin(azimuth_rad) - dy * cos(azimuth_rad)) * 0.01;
    cam_.lookat[2] += dy * 0.01;
}

void MjRenderer::ResetCamera(const mjModel* model) {
    mjv_defaultCamera(&cam_);
    if (model) {
        // Point camera at model center with good viewing angle
        cam_.lookat[0] = model->stat.center[0];
        cam_.lookat[1] = model->stat.center[1];
        cam_.lookat[2] = model->stat.center[2] + 0.3 * model->stat.extent;
        cam_.distance = 2.0 * model->stat.extent;
        cam_.azimuth = 135.0;    // 3/4 view
        cam_.elevation = -20.0;  // Slightly above
    }
}

MjRenderer::CameraMode MjRenderer::GetCameraMode() const {
    if (cam_.type == mjCAMERA_TRACKING) return CameraMode::FollowBody;
    if (cam_.type == mjCAMERA_FIXED) return CameraMode::ModelCamera;
    return CameraMode::Free;
}

int MjRenderer::CameraTarget() const {
    if (cam_.type == mjCAMERA_TRACKING) return cam_.trackbodyid;
    if (cam_.type == mjCAMERA_FIXED) return cam_.fixedcamid;
    return -1;
}

// Leaving a model camera: the free or following camera starts from the view
// it showed (the last frame's head pose), not from where it was before
void MjRenderer::KeepModelCameraView(bool aim) {
    if (cam_.type != mjCAMERA_FIXED || scn_.camera[0].frustum_near <= 0.0f) return;
    const mjvGLCamera head = mjv_averageCamera(&scn_.camera[0], &scn_.camera[1]);
    const float* f = head.forward;
    cam_.azimuth = std::atan2(f[1], f[0]) * 180.0 / M_PI;
    cam_.elevation = std::asin(std::clamp(f[2], -1.0f, 1.0f)) * 180.0 / M_PI;
    if (!aim) return;
    const double distance = cam_.distance > 0.0 ? cam_.distance : 2.0;
    for (int i = 0; i < 3; ++i) cam_.lookat[i] = head.pos[i] + f[i] * distance;
}

void MjRenderer::UseFreeCamera() {
    KeepModelCameraView(true);
    cam_.type = mjCAMERA_FREE;  // otherwise lookat, angle and distance stay where they are
}

void MjRenderer::FollowBody(const mjModel* model, const mjData* data, int body_id) {
    KeepModelCameraView(false);  // the body comes into the centre; the angle stays
    cam_.type = mjCAMERA_TRACKING;
    cam_.trackbodyid = body_id;
    if (!model || !data || body_id <= 0 || body_id >= model->nbody) return;
    // The subtree's reach: its geoms' distance from its centre of mass plus their size
    double radius = 0.0;
    const auto measure = [&] {
        const mjtNum* com = data->subtree_com + 3 * body_id;
        for (int g = 0; g < model->ngeom; ++g) {
            int b = model->geom_bodyid[g];
            while (b > 0 && b != body_id) b = model->body_parentid[b];
            if (b != body_id) continue;
            const mjtNum* p = data->geom_xpos + 3 * g;
            const double dx = p[0] - com[0], dy = p[1] - com[1], dz = p[2] - com[2];
            radius = std::max(radius, std::sqrt(dx * dx + dy * dy + dz * dz) + model->geom_rbound[g]);
        }
    };
    if (physics_mutex_) {
        std::lock_guard<std::mutex> phys_lock(*physics_mutex_);
        measure();
    } else {
        measure();
    }
    if (radius > 0.0) cam_.distance = std::clamp(4.0 * radius, 0.5, 100.0);
}

void MjRenderer::UseModelCamera(int camera_id) {
    cam_.type = mjCAMERA_FIXED;
    cam_.fixedcamid = camera_id;
}

int MjRenderer::BodyAt(const mjModel* model, const mjData* data, float x, float y) {
    if (!initialized_ || !model || !data || width_ < 1 || height_ < 1) return -1;
    mjtNum point[3];
    int geom = -1, flex = -1, skin = -1;
    const mjtNum aspect = static_cast<mjtNum>(width_) / height_;
    int body = -1;
    // The scene is the last frame's; the model state is read under the physics lock
    if (physics_mutex_) {
        std::lock_guard<std::mutex> phys_lock(*physics_mutex_);
        body = mjv_select(model, data, &opt_, aspect, x, 1.0 - y, &scn_, point, &geom, &flex, &skin);
    } else {
        body = mjv_select(model, data, &opt_, aspect, x, 1.0 - y, &scn_, point, &geom, &flex, &skin);
    }
    return body > 0 ? body : -1;
}

bool MjRenderer::StartPush(const mjModel* model, mjData* data, float x, float y) {
    EndPush();
    if (!initialized_ || !model || !data || width_ < 1 || height_ < 1) return false;
    mjtNum point[3];
    int geom = -1, flex = -1, skin = -1;
    const mjtNum aspect = static_cast<mjtNum>(width_) / height_;
    std::unique_lock<std::mutex> phys_lock;
    if (physics_mutex_) phys_lock = std::unique_lock<std::mutex>(*physics_mutex_);
    const int body = mjv_select(model, data, &opt_, aspect, x, 1.0 - y, &scn_, point, &geom, &flex, &skin);
    if (body <= 0) return false;
    // Held at the picked point, in the body's frame
    mjtNum offset[3];
    mju_sub3(offset, point, data->xpos + 3 * body);
    mju_mulMatTVec(pert_.localpos, data->xmat + 9 * body, offset, 3, 3);
    pert_.select = body;
    pert_.flexselect = -1;
    pert_.skinselect = -1;
    pert_.active = mjPERT_TRANSLATE;
    mjv_initPerturb(model, data, &scn_, &pert_);
    return true;
}

void MjRenderer::MovePush(const mjModel* model, const mjData* data, float dx, float dy) {
    if (!IsPushing() || !model || !data) return;
    std::unique_lock<std::mutex> phys_lock;
    if (physics_mutex_) phys_lock = std::unique_lock<std::mutex>(*physics_mutex_);
    mjv_movePerturb(model, data, mjMOUSE_MOVE_V, dx, dy, &scn_, &pert_);
}

int MjRenderer::PushForce(const mjModel* model, mjData* data, mjtNum force[6]) {
    if (!IsPushing() || !model || !data || pert_.select >= model->nbody) return -1;
    std::unique_lock<std::mutex> phys_lock;
    if (physics_mutex_) phys_lock = std::unique_lock<std::mutex>(*physics_mutex_);
    const int body = pert_.select;
    mju_zero(data->xfrc_applied + 6 * body, 6);
    mjv_applyPerturbForce(model, data, &pert_);
    mju_copy(force, data->xfrc_applied + 6 * body, 6);
    return body;
}

void MjRenderer::EndPush() {
    pert_.select = 0;
    pert_.active = 0;
}

bool MjRenderer::ChangeModel(const mjModel* model) {
    if (!initialized_ || !model) return false;
    EndPush();
    MakeCompatContextCurrent();
    // The View's flags live in the scene: kept across the new scene
    mjtByte flags[mjNRNDFLAG];
    std::copy(scn_.flags, scn_.flags + mjNRNDFLAG, flags);
    const mjtByte enabletransform = scn_.enabletransform;
    mjr_freeContext(&con_);
    mjv_freeScene(&scn_);
    mjv_defaultScene(&scn_);
    mjv_makeScene(model, &scn_, 2000);
    std::copy(flags, flags + mjNRNDFLAG, scn_.flags);
    scn_.enabletransform = enabletransform;
    mjr_defaultContext(&con_);
    mjr_makeContext(model, &con_, mjFONTSCALE_150);
    const bool made = con_.currentBuffer != -1;
    if (made) {
        mjr_resizeOffscreen(width_, height_, &con_);
        if (con_.offWidth > 0 && con_.offHeight > 0) {
            width_ = std::min(width_, con_.offWidth);
            height_ = std::min(height_, con_.offHeight);
        }
        pixel_buffer_.resize(static_cast<size_t>(width_) * height_ * 3);
    } else {
        spdlog::error("[MjRenderer] mjr_makeContext failed for the changed model");
    }
    MakeMainContextCurrent();
    if (!made) {
        Shutdown();
        return false;
    }
    resolution_changed_ = true;  // the display texture follows the offscreen size
    return true;
}

// =============================================================================
// Resolution
// =============================================================================

void MjRenderer::SetResolution(int width, int height) {
    if (width == width_ && height == height_) return;
    if (width < 1 || height < 1) return;

    width_ = width;
    height_ = height;
    resolution_changed_ = true;
}

// =============================================================================
// Internal — GL Context Management
// =============================================================================

bool MjRenderer::CreateCompatContext([[maybe_unused]] GLFWwindow* main_window) {
    // Save current GLFW hints and set compatibility profile
    glfwDefaultWindowHints();
    glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);  // Hidden window
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);
    // Don't set GLFW_OPENGL_PROFILE — defaults to any (compatibility) profile

    // Create hidden window. Share with main window if possible (not required).
    compat_window_ = glfwCreateWindow(width_, height_, "MuJoCo Renderer",
                                       nullptr, nullptr);
    if (!compat_window_) {
        spdlog::error("[MjRenderer] Failed to create hidden compatibility GL context");
        return false;
    }

    spdlog::info("[MjRenderer] Created hidden compatibility GL context for MuJoCo");
    return true;
}

void MjRenderer::MakeMainContextCurrent() {
    if (main_window_) {
        glfwMakeContextCurrent(main_window_);
    }
}

void MjRenderer::MakeCompatContextCurrent() {
    if (compat_window_) {
        glfwMakeContextCurrent(compat_window_);
    }
}

bool MjRenderer::AllocateDisplayTexture() {
    glGenTextures(1, &display_texture_);
    if (display_texture_ == 0) {
        spdlog::error("[MjRenderer] Failed to allocate display texture");
        return false;
    }

    glBindTexture(GL_TEXTURE_2D, display_texture_);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width_, height_, 0,
                 GL_RGB, GL_UNSIGNED_BYTE, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glBindTexture(GL_TEXTURE_2D, 0);

    return true;
}

void MjRenderer::FreeDisplayTexture() {
    if (display_texture_ != 0) {
        glDeleteTextures(1, &display_texture_);
        display_texture_ = 0;
    }
}

} // namespace cyxwiz::plugin::mujoco
