#include "mj_renderer.h"

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <spdlog/spdlog.h>

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

        // Recreate MuJoCo context with new size (compat context)
        MakeCompatContextCurrent();
        mjr_freeContext(&con_);
        mjr_defaultContext(&con_);
        mjr_makeContext(model, &con_, mjFONTSCALE_150);

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

    // Update abstract scene
    mjv_updateScene(model, const_cast<mjData*>(data), &opt_, nullptr, &cam_,
                    mjCAT_ALL, &scn_);

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
    cam_.distance *= (1.0f - dz * 0.05f);
    if (cam_.distance < 0.01f) cam_.distance = 0.01f;
    if (cam_.distance > 100.0f) cam_.distance = 100.0f;
}

void MjRenderer::PanCamera(float dx, float dy) {
    // Pan by adjusting lookat position
    double azimuth_rad = cam_.azimuth * 3.14159265 / 180.0;
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

void MjRenderer::TrackBody(int body_id) {
    cam_.type = mjCAMERA_TRACKING;
    cam_.trackbodyid = body_id;
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
