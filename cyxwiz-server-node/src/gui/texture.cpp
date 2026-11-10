// texture.cpp - images for ImGui (TOFIX139 P7).
#include "gui/texture.h"

#include <glad/glad.h>
#include <spdlog/spdlog.h>
#include <stb_image.h>

#include <filesystem>
#include <system_error>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#elif defined(__APPLE__)
#include <mach-o/dyld.h>
#include <climits>
#endif

namespace cyxwiz::servernode::gui {

namespace {

// Where node resources may be: beside the executable, where the build and
// the install put them; in a macOS bundle's Resources; then the working
// folder.
std::vector<std::filesystem::path> ResourceDirs() {
    std::vector<std::filesystem::path> dirs;
    std::filesystem::path exe;
#ifdef _WIN32
    wchar_t path[MAX_PATH] = {};
    const DWORD length = GetModuleFileNameW(nullptr, path, MAX_PATH);
    if (length > 0 && length < MAX_PATH) exe = path;
#elif defined(__APPLE__)
    char path[PATH_MAX];
    uint32_t size = sizeof(path);
    if (_NSGetExecutablePath(path, &size) == 0) exe = path;
#elif defined(__linux__)
    std::error_code link_error;
    exe = std::filesystem::read_symlink("/proc/self/exe", link_error);
#endif
    if (!exe.empty()) {
        std::error_code canonical_error;
        const auto resolved = std::filesystem::weakly_canonical(exe, canonical_error);
        const auto dir = (canonical_error ? exe : resolved).parent_path();
        dirs.push_back(dir / "resources");
#ifdef __APPLE__
        dirs.push_back(dir.parent_path() / "Resources");
#endif
    }
    std::error_code cwd_error;
    const auto cwd = std::filesystem::current_path(cwd_error);
    if (!cwd_error) dirs.push_back(cwd / "resources");
    return dirs;
}

}  // namespace

bool LoadTexture(const std::string& path, Texture& out) {
    int width = 0, height = 0, channels = 0;
    unsigned char* pixels = stbi_load(path.c_str(), &width, &height, &channels, 4);
    if (!pixels) {
        spdlog::warn("Could not load image {}: {}", path, stbi_failure_reason());
        return false;
    }
    GLuint id = 0;
    glGenTextures(1, &id);
    glBindTexture(GL_TEXTURE_2D, id);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, pixels);
    out.id = id;
    out.width = width;
    out.height = height;
    out.edge_color = ImVec4(pixels[0] / 255.0f, pixels[1] / 255.0f, pixels[2] / 255.0f, pixels[3] / 255.0f);
    stbi_image_free(pixels);
    spdlog::info("Loaded image {} ({}x{})", path, width, height);
    return true;
}

std::string FindResourceFile(const std::string& file_name) {
    for (const auto& dir : ResourceDirs()) {
        const auto path = dir / file_name;
        std::error_code ec;
        if (std::filesystem::exists(path, ec)) {
            // UTF-8, never the Windows code page: the executable's folder may
            // hold any character (a user folder under C:\Users).
            const auto utf8 = path.u8string();
            return std::string(utf8.begin(), utf8.end());
        }
    }
    return {};
}

}  // namespace cyxwiz::servernode::gui
