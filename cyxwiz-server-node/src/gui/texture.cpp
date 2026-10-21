// texture.cpp - images for ImGui (TOFIX139 P7).
#include "gui/texture.h"

#include <glad/glad.h>
#include <spdlog/spdlog.h>
#include <stb_image.h>

#include <filesystem>

namespace cyxwiz::servernode::gui {

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
    const char* dirs[] = {"./cyxwiz-server-node/resources/", "./resources/", "../../../cyxwiz-server-node/resources/",
                          "../../cyxwiz-server-node/resources/", "../cyxwiz-server-node/resources/"};
    for (const char* dir : dirs) {
        const std::string path = std::string(dir) + file_name;
        std::error_code ec;
        if (std::filesystem::exists(path, ec)) return path;
    }
    return {};
}

}  // namespace cyxwiz::servernode::gui
