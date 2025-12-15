#include "output_renderer.h"
#include "../icons.h"
#include <spdlog/spdlog.h>
#include <sstream>
#include <algorithm>
#include <cstring>
#include <fstream>

// stb_image for PNG loading (implementation in plotting/stb_image_impl.cpp)
#include <stb_image.h>

#ifdef _WIN32
#include <windows.h>
#include <commdlg.h>
#include <shellapi.h>
#endif

namespace cyxwiz {

// Static member definition
OpenPlotWindowCallback OutputRenderer::open_plot_window_callback_;

// ========== Text Output ==========

void OutputRenderer::RenderText(const std::string& text, bool wrap) {
    if (text.empty()) return;

    ImGui::PushStyleColor(ImGuiCol_Text, GetStdoutColor());

    if (wrap) {
        ImGui::TextWrapped("%s", text.c_str());
    } else {
        ImGui::TextUnformatted(text.c_str());
    }

    ImGui::PopStyleColor();
}

void OutputRenderer::RenderError(const std::string& error) {
    if (error.empty()) return;

    ImGui::PushStyleColor(ImGuiCol_Text, GetErrorColor());

    // Add error icon
    ImGui::TextUnformatted(ICON_FA_CIRCLE_EXCLAMATION " ");
    ImGui::SameLine();

    // Render error text
    ImGui::TextWrapped("%s", error.c_str());

    ImGui::PopStyleColor();
}

void OutputRenderer::RenderStream(const std::string& text, const std::string& stream_name) {
    if (text.empty()) return;

    ImVec4 color = (stream_name == "stderr") ? GetStderrColor() : GetStdoutColor();
    ImGui::PushStyleColor(ImGuiCol_Text, color);

    // Show stream name if stderr
    if (stream_name == "stderr") {
        ImGui::TextUnformatted("[stderr] ");
        ImGui::SameLine();
    }

    ImGui::TextWrapped("%s", text.c_str());

    ImGui::PopStyleColor();
}

// ========== Rich Output ==========

void OutputRenderer::RenderImage(GLuint texture_id, int width, int height, float max_width) {
    if (texture_id == 0 || width <= 0 || height <= 0) return;

    // Calculate size
    float display_width = static_cast<float>(width);
    float display_height = static_cast<float>(height);

    if (max_width <= 0.0f) {
        max_width = ImGui::GetContentRegionAvail().x - 20.0f;
    }

    // Scale down if needed
    if (display_width > max_width) {
        float scale = max_width / display_width;
        display_width = max_width;
        display_height *= scale;
    }

    ImVec2 size(display_width, display_height);
    ImGui::Image((ImTextureID)(intptr_t)texture_id, size);
}

void OutputRenderer::RenderPlot(GLuint texture_id, int width, int height, const std::string& plot_id) {
    // Call with empty data (context menu actions won't work without data)
    static std::vector<unsigned char> empty_data;
    RenderPlotWithData(texture_id, width, height, empty_data, plot_id);
}

void OutputRenderer::RenderPlotWithData(GLuint texture_id, int width, int height,
                                         const std::vector<unsigned char>& png_data,
                                         const std::string& plot_id) {
    if (texture_id == 0 || width <= 0 || height <= 0) return;

    // Calculate size
    float max_width = ImGui::GetContentRegionAvail().x - 20.0f;
    float display_width = static_cast<float>(width);
    float display_height = static_cast<float>(height);

    if (display_width > max_width) {
        float scale = max_width / display_width;
        display_width = max_width;
        display_height *= scale;
    }

    ImVec2 size(display_width, display_height);

    // Render image
    ImGui::Image((ImTextureID)(intptr_t)texture_id, size);

    // Context menu for plot
    std::string popup_id = "##plot_context_" + plot_id;
    if (ImGui::BeginPopupContextItem(popup_id.c_str())) {
        bool has_data = !png_data.empty();

        // Copy Plot
        if (ImGui::MenuItem(ICON_FA_COPY " Copy Plot", nullptr, false, has_data)) {
            if (CopyImageToClipboard(png_data)) {
                spdlog::info("Plot copied to clipboard");
            } else {
                spdlog::warn("Failed to copy plot to clipboard");
            }
        }

        // Save As...
        if (ImGui::MenuItem(ICON_FA_DOWNLOAD " Save As...", nullptr, false, has_data)) {
            if (SaveImageToFile(png_data, plot_id.empty() ? "plot" : plot_id)) {
                spdlog::info("Plot saved to file");
            } else {
                spdlog::warn("Failed to save plot");
            }
        }

        // Open in Window
        if (ImGui::MenuItem(ICON_FA_EXPAND " Open in Window")) {
            if (open_plot_window_callback_) {
                open_plot_window_callback_(texture_id, width, height,
                    plot_id.empty() ? "Plot" : plot_id);
            } else {
                spdlog::warn("No callback set for opening plot in window");
            }
        }

        ImGui::EndPopup();
    }
}

void OutputRenderer::SetOpenPlotWindowCallback(OpenPlotWindowCallback callback) {
    open_plot_window_callback_ = callback;
}

bool OutputRenderer::CopyImageToClipboard(const std::vector<unsigned char>& png_data) {
    if (png_data.empty()) return false;

#ifdef _WIN32
    // Decode PNG to get raw RGBA data
    int width, height, channels;
    unsigned char* pixels = stbi_load_from_memory(
        png_data.data(), static_cast<int>(png_data.size()),
        &width, &height, &channels, 4);

    if (!pixels) {
        spdlog::error("Failed to decode PNG for clipboard");
        return false;
    }

    // Create DIB (Device Independent Bitmap) for clipboard
    BITMAPINFOHEADER bi = {};
    bi.biSize = sizeof(BITMAPINFOHEADER);
    bi.biWidth = width;
    bi.biHeight = -height;  // Negative for top-down DIB
    bi.biPlanes = 1;
    bi.biBitCount = 32;
    bi.biCompression = BI_RGB;
    bi.biSizeImage = width * height * 4;

    // Allocate memory for DIB
    size_t dib_size = sizeof(BITMAPINFOHEADER) + bi.biSizeImage;
    HGLOBAL hMem = GlobalAlloc(GMEM_MOVEABLE, dib_size);
    if (!hMem) {
        stbi_image_free(pixels);
        return false;
    }

    // Copy data to DIB
    unsigned char* dib_ptr = static_cast<unsigned char*>(GlobalLock(hMem));
    if (!dib_ptr) {
        GlobalFree(hMem);
        stbi_image_free(pixels);
        return false;
    }

    memcpy(dib_ptr, &bi, sizeof(BITMAPINFOHEADER));

    // Convert RGBA to BGRA for Windows
    unsigned char* dest = dib_ptr + sizeof(BITMAPINFOHEADER);
    for (int i = 0; i < width * height; i++) {
        dest[i * 4 + 0] = pixels[i * 4 + 2];  // B <- R
        dest[i * 4 + 1] = pixels[i * 4 + 1];  // G <- G
        dest[i * 4 + 2] = pixels[i * 4 + 0];  // R <- B
        dest[i * 4 + 3] = pixels[i * 4 + 3];  // A <- A
    }

    GlobalUnlock(hMem);
    stbi_image_free(pixels);

    // Copy to clipboard
    if (!OpenClipboard(nullptr)) {
        GlobalFree(hMem);
        return false;
    }

    EmptyClipboard();
    HANDLE result = SetClipboardData(CF_DIB, hMem);
    CloseClipboard();

    if (!result) {
        GlobalFree(hMem);
        return false;
    }

    return true;
#else
    // Non-Windows platforms: not implemented
    spdlog::warn("Copy to clipboard not implemented on this platform");
    return false;
#endif
}

bool OutputRenderer::SaveImageToFile(const std::vector<unsigned char>& png_data, const std::string& default_name) {
    if (png_data.empty()) return false;

#ifdef _WIN32
    // Create save file dialog
    char filename[MAX_PATH] = "";
    std::string default_filename = default_name + ".png";
    strncpy_s(filename, default_filename.c_str(), MAX_PATH - 1);

    OPENFILENAMEA ofn = {};
    ofn.lStructSize = sizeof(ofn);
    ofn.hwndOwner = nullptr;
    ofn.lpstrFilter = "PNG Files (*.png)\0*.png\0All Files (*.*)\0*.*\0";
    ofn.lpstrFile = filename;
    ofn.nMaxFile = MAX_PATH;
    ofn.lpstrTitle = "Save Plot Image";
    ofn.Flags = OFN_OVERWRITEPROMPT | OFN_PATHMUSTEXIST;
    ofn.lpstrDefExt = "png";

    if (!GetSaveFileNameA(&ofn)) {
        // User cancelled or error
        return false;
    }

    // Write PNG data to file
    std::ofstream file(filename, std::ios::binary);
    if (!file) {
        spdlog::error("Failed to create file: {}", filename);
        return false;
    }

    file.write(reinterpret_cast<const char*>(png_data.data()), png_data.size());
    file.close();

    spdlog::info("Plot saved to: {}", filename);
    return true;
#else
    // Non-Windows platforms: save to current directory with timestamp
    std::string filename = default_name + ".png";
    std::ofstream file(filename, std::ios::binary);
    if (!file) {
        spdlog::error("Failed to create file: {}", filename);
        return false;
    }

    file.write(reinterpret_cast<const char*>(png_data.data()), png_data.size());
    file.close();

    spdlog::info("Plot saved to: {}", filename);
    return true;
#endif
}

void OutputRenderer::RenderTable(const std::vector<std::vector<std::string>>& data,
                                  const std::vector<std::string>& headers) {
    if (data.empty()) return;

    size_t num_columns = headers.empty() ? (data.empty() ? 0 : data[0].size()) : headers.size();
    if (num_columns == 0) return;

    ImGuiTableFlags flags = ImGuiTableFlags_Borders |
                            ImGuiTableFlags_RowBg |
                            ImGuiTableFlags_Resizable |
                            ImGuiTableFlags_ScrollX |
                            ImGuiTableFlags_ScrollY;

    // Calculate table height (max 10 rows visible)
    float row_height = ImGui::GetTextLineHeightWithSpacing();
    float table_height = std::min(static_cast<float>(data.size() + 1), 11.0f) * row_height;

    if (ImGui::BeginTable("##output_table", static_cast<int>(num_columns), flags,
                          ImVec2(0.0f, table_height))) {

        // Setup columns
        for (size_t i = 0; i < num_columns; i++) {
            std::string header = (i < headers.size()) ? headers[i] : "Col " + std::to_string(i);
            ImGui::TableSetupColumn(header.c_str());
        }

        // Headers row
        if (!headers.empty()) {
            ImGui::TableHeadersRow();
        }

        // Data rows
        for (const auto& row : data) {
            ImGui::TableNextRow();
            for (size_t col = 0; col < num_columns; col++) {
                ImGui::TableSetColumnIndex(static_cast<int>(col));
                if (col < row.size()) {
                    ImGui::TextUnformatted(row[col].c_str());
                }
            }
        }

        ImGui::EndTable();
    }
}

void OutputRenderer::RenderMarkdown(const std::string& markdown) {
    if (markdown.empty()) return;

    std::istringstream stream(markdown);
    std::string line;
    bool in_code_block = false;

    while (std::getline(stream, line)) {
        // Code block toggle
        if (IsCodeBlock(line)) {
            in_code_block = !in_code_block;
            continue;
        }

        if (in_code_block) {
            // Render code with monospace styling
            ImGui::PushStyleColor(ImGuiCol_Text, GetCodeColor());
            ImGui::TextUnformatted(("    " + line).c_str());
            ImGui::PopStyleColor();
        } else {
            RenderMarkdownLine(line);
        }
    }
}

void OutputRenderer::RenderMarkdownLine(const std::string& line) {
    int header_level;
    std::string list_content;

    if (line.empty()) {
        ImGui::Spacing();
    }
    // Check for HTML header tags: <h1>...</h1>, <h2>...</h2>, etc.
    else if (IsHtmlHeader(line, header_level, list_content)) {
        ImGui::PushStyleColor(ImGuiCol_Text, GetHeaderColor());

        // Different font sizes for different header levels
        float scale = 1.0f + (4 - std::min(header_level, 4)) * 0.2f;  // h1=1.6, h2=1.4, h3=1.2
        ImGui::SetWindowFontScale(scale);

        ImGui::TextWrapped("%s", list_content.c_str());

        ImGui::SetWindowFontScale(1.0f);
        ImGui::PopStyleColor();

        // Underline for h1
        if (header_level == 1) {
            ImGui::Separator();
        }
    }
    else if (IsHeaderLine(line, header_level)) {
        // Markdown headers: # Heading
        std::string header_text = line.substr(header_level + 1);
        // Trim leading space
        if (!header_text.empty() && header_text[0] == ' ') {
            header_text = header_text.substr(1);
        }

        ImGui::PushStyleColor(ImGuiCol_Text, GetHeaderColor());

        // Different font sizes for different header levels
        float scale = 1.0f + (4 - header_level) * 0.2f;  // h1=1.6, h2=1.4, h3=1.2
        ImGui::SetWindowFontScale(scale);

        ImGui::TextWrapped("%s", header_text.c_str());

        ImGui::SetWindowFontScale(1.0f);
        ImGui::PopStyleColor();

        // Underline for h1
        if (header_level == 1) {
            ImGui::Separator();
        }
    }
    else if (IsListItem(line, list_content)) {
        ImGui::Bullet();
        ImGui::SameLine();
        RenderMarkdownInline(list_content);
    }
    else if (line.find("---") == 0 || line.find("***") == 0 || line.find("<hr") == 0) {
        // Horizontal rule (markdown or HTML)
        ImGui::Separator();
    }
    else if (line.find("<br") == 0 || line.find("<BR") == 0) {
        // Line break
        ImGui::Spacing();
    }
    else {
        // Regular text - handle inline formatting (including HTML tags)
        RenderMarkdownInline(line);
    }
}

void OutputRenderer::RenderMarkdownInline(const std::string& text) {
    // Simple inline formatting: **bold**, *italic*, `code`
    std::string current;
    bool bold = false;
    bool italic = false;
    bool code = false;

    for (size_t i = 0; i < text.size(); i++) {
        // Bold: **text**
        if (i + 1 < text.size() && text[i] == '*' && text[i + 1] == '*') {
            if (!current.empty()) {
                if (code) {
                    ImGui::PushStyleColor(ImGuiCol_Text, GetCodeColor());
                }
                ImGui::TextUnformatted(current.c_str());
                if (code) {
                    ImGui::PopStyleColor();
                }
                ImGui::SameLine(0, 0);
                current.clear();
            }
            bold = !bold;
            i++;  // Skip second *
            continue;
        }
        // Italic: *text*
        else if (text[i] == '*' && (i == 0 || text[i-1] != '*') &&
                 (i + 1 >= text.size() || text[i+1] != '*')) {
            if (!current.empty()) {
                ImGui::TextUnformatted(current.c_str());
                ImGui::SameLine(0, 0);
                current.clear();
            }
            italic = !italic;
            continue;
        }
        // Code: `text`
        else if (text[i] == '`') {
            if (!current.empty()) {
                ImGui::TextUnformatted(current.c_str());
                ImGui::SameLine(0, 0);
                current.clear();
            }
            code = !code;
            continue;
        }

        current += text[i];
    }

    // Render remaining text
    if (!current.empty()) {
        if (code) {
            ImGui::PushStyleColor(ImGuiCol_Text, GetCodeColor());
        }
        ImGui::TextUnformatted(current.c_str());
        if (code) {
            ImGui::PopStyleColor();
        }
    }
}

bool OutputRenderer::IsHeaderLine(const std::string& line, int& level) {
    level = 0;
    for (char c : line) {
        if (c == '#') {
            level++;
        } else {
            break;
        }
    }
    return level > 0 && level <= 6 && line.size() > static_cast<size_t>(level);
}

bool OutputRenderer::IsHtmlHeader(const std::string& line, int& level, std::string& content) {
    // Check for <h1>...</h1>, <h2>...</h2>, etc. (case insensitive)
    std::string lower_line = line;
    for (auto& c : lower_line) c = std::tolower(c);

    for (int h = 1; h <= 6; h++) {
        std::string open_tag = "<h" + std::to_string(h) + ">";
        std::string close_tag = "</h" + std::to_string(h) + ">";

        size_t start = lower_line.find(open_tag);
        if (start != std::string::npos) {
            size_t content_start = start + open_tag.length();
            size_t end = lower_line.find(close_tag, content_start);
            if (end != std::string::npos) {
                level = h;
                content = line.substr(content_start, end - content_start);
                return true;
            } else {
                // No closing tag - take rest of line as content
                level = h;
                content = line.substr(content_start);
                return true;
            }
        }
    }
    return false;
}

bool OutputRenderer::IsListItem(const std::string& line, std::string& content) {
    std::string trimmed = line;
    // Find first non-space
    size_t start = trimmed.find_first_not_of(" \t");
    if (start == std::string::npos) return false;

    if (start < line.size() && (line[start] == '-' || line[start] == '*' || line[start] == '+')) {
        if (start + 1 < line.size() && line[start + 1] == ' ') {
            content = line.substr(start + 2);
            return true;
        }
    }

    // Numbered list: 1. item
    if (start < line.size() && std::isdigit(line[start])) {
        size_t dot_pos = line.find('.', start);
        if (dot_pos != std::string::npos && dot_pos + 1 < line.size() && line[dot_pos + 1] == ' ') {
            content = line.substr(dot_pos + 2);
            return true;
        }
    }

    return false;
}

bool OutputRenderer::IsCodeBlock(const std::string& line) {
    std::string trimmed = line;
    size_t start = trimmed.find_first_not_of(" \t");
    if (start == std::string::npos) return false;
    return trimmed.substr(start, 3) == "```";
}

// ========== Cell Output ==========

void OutputRenderer::RenderCellOutput(const CellOutput& output) {
    switch (output.type) {
        case OutputType::Text:
            RenderText(output.data);
            break;

        case OutputType::Error:
            RenderError(output.data);
            break;

        case OutputType::Stream:
            RenderStream(output.data, output.name);
            break;

        case OutputType::Image:
        case OutputType::Plot:
            if (output.texture_id != 0) {
                // Use RenderPlotWithData if we have image_data for copy/save
                if (!output.image_data.empty()) {
                    RenderPlotWithData(output.texture_id, output.width, output.height,
                                       output.image_data, output.name);
                } else {
                    RenderPlot(output.texture_id, output.width, output.height, output.name);
                }
            } else if (!output.image_data.empty()) {
                // Create texture from raw PNG data on first render
                // Note: We cast away const to update texture_id (cached texture)
                CellOutput& mutable_output = const_cast<CellOutput&>(output);

                int width, height, channels;
                unsigned char* pixels = stbi_load_from_memory(
                    output.image_data.data(),
                    static_cast<int>(output.image_data.size()),
                    &width, &height, &channels, 4);

                if (pixels) {
                    mutable_output.texture_id = CreateTextureFromRGBA(pixels, width, height);
                    mutable_output.width = width;
                    mutable_output.height = height;
                    stbi_image_free(pixels);
                    // Use RenderPlotWithData to enable copy/save
                    RenderPlotWithData(mutable_output.texture_id, width, height,
                                       output.image_data, output.name);
                    spdlog::info("Created plot texture: {}x{}", width, height);
                } else {
                    ImGui::TextDisabled("[Failed to load image: %s]", stbi_failure_reason());
                }
            } else if (!output.data.empty()) {
                // Try base64-encoded PNG in data field
                CellOutput& mutable_output = const_cast<CellOutput&>(output);
                int width, height;
                GLuint texture = CreateTextureFromBase64PNG(output.data, width, height);
                if (texture != 0) {
                    mutable_output.texture_id = texture;
                    mutable_output.width = width;
                    mutable_output.height = height;
                    // Decode base64 to get raw data for copy/save
                    std::vector<unsigned char> png_data = Base64Decode(output.data);
                    RenderPlotWithData(texture, width, height, png_data, output.name);
                } else {
                    ImGui::TextDisabled("[Failed to decode image]");
                }
            } else {
                ImGui::TextDisabled("[Image: no data]");
            }
            break;

        case OutputType::Table:
            // TODO: Parse table data from output.data
            ImGui::TextDisabled("[Table output]");
            break;

        case OutputType::Html:
            // HTML not fully supported, show as text
            ImGui::TextDisabled("[HTML output]");
            RenderText(output.data);
            break;

        case OutputType::Markdown:
            RenderMarkdown(output.data);
            break;

        default:
            RenderText(output.data);
            break;
    }
}

// ========== Texture Utilities ==========

GLuint OutputRenderer::CreateTextureFromBase64PNG(const std::string& base64_data, int& out_width, int& out_height) {
    // Decode base64
    std::vector<unsigned char> png_data = Base64Decode(base64_data);

    if (png_data.empty()) {
        spdlog::error("Failed to decode base64 PNG data");
        return 0;
    }

    // Load PNG using stb_image
    int width, height, channels;
    unsigned char* pixels = stbi_load_from_memory(
        png_data.data(), static_cast<int>(png_data.size()),
        &width, &height, &channels, 4);  // Force RGBA

    if (!pixels) {
        spdlog::error("Failed to load PNG from memory: {}", stbi_failure_reason());
        return 0;
    }

    out_width = width;
    out_height = height;

    // Create OpenGL texture
    GLuint texture = CreateTextureFromRGBA(pixels, width, height);

    stbi_image_free(pixels);
    return texture;
}

GLuint OutputRenderer::CreateTextureFromRGBA(const unsigned char* data, int width, int height) {
    if (!data || width <= 0 || height <= 0) return 0;

    GLuint texture;
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);

    // Set texture parameters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    // Upload texture data
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0,
                 GL_RGBA, GL_UNSIGNED_BYTE, data);

    glBindTexture(GL_TEXTURE_2D, 0);

    spdlog::debug("Created texture {} ({}x{})", texture, width, height);
    return texture;
}

void OutputRenderer::DeleteTexture(GLuint texture_id) {
    if (texture_id != 0) {
        glDeleteTextures(1, &texture_id);
    }
}

// ========== Base64 Utilities ==========

static const std::string base64_chars =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "abcdefghijklmnopqrstuvwxyz"
    "0123456789+/";

static inline bool is_base64(unsigned char c) {
    return (isalnum(c) || (c == '+') || (c == '/'));
}

std::vector<unsigned char> OutputRenderer::Base64Decode(const std::string& encoded) {
    size_t in_len = encoded.size();
    int i = 0;
    int j = 0;
    int in_ = 0;
    unsigned char char_array_4[4], char_array_3[3];
    std::vector<unsigned char> result;

    while (in_len-- && (encoded[in_] != '=') && is_base64(encoded[in_])) {
        char_array_4[i++] = encoded[in_]; in_++;
        if (i == 4) {
            for (i = 0; i < 4; i++) {
                char_array_4[i] = static_cast<unsigned char>(base64_chars.find(char_array_4[i]));
            }

            char_array_3[0] = (char_array_4[0] << 2) + ((char_array_4[1] & 0x30) >> 4);
            char_array_3[1] = ((char_array_4[1] & 0xf) << 4) + ((char_array_4[2] & 0x3c) >> 2);
            char_array_3[2] = ((char_array_4[2] & 0x3) << 6) + char_array_4[3];

            for (i = 0; (i < 3); i++) {
                result.push_back(char_array_3[i]);
            }
            i = 0;
        }
    }

    if (i) {
        for (j = 0; j < i; j++) {
            char_array_4[j] = static_cast<unsigned char>(base64_chars.find(char_array_4[j]));
        }

        char_array_3[0] = (char_array_4[0] << 2) + ((char_array_4[1] & 0x30) >> 4);
        char_array_3[1] = ((char_array_4[1] & 0xf) << 4) + ((char_array_4[2] & 0x3c) >> 2);

        for (j = 0; (j < i - 1); j++) {
            result.push_back(char_array_3[j]);
        }
    }

    return result;
}

std::string OutputRenderer::Base64Encode(const unsigned char* data, size_t length) {
    std::string result;
    int i = 0;
    int j = 0;
    unsigned char char_array_3[3];
    unsigned char char_array_4[4];

    while (length--) {
        char_array_3[i++] = *(data++);
        if (i == 3) {
            char_array_4[0] = (char_array_3[0] & 0xfc) >> 2;
            char_array_4[1] = ((char_array_3[0] & 0x03) << 4) + ((char_array_3[1] & 0xf0) >> 4);
            char_array_4[2] = ((char_array_3[1] & 0x0f) << 2) + ((char_array_3[2] & 0xc0) >> 6);
            char_array_4[3] = char_array_3[2] & 0x3f;

            for (i = 0; (i < 4) ; i++) {
                result += base64_chars[char_array_4[i]];
            }
            i = 0;
        }
    }

    if (i) {
        for (j = i; j < 3; j++) {
            char_array_3[j] = '\0';
        }

        char_array_4[0] = (char_array_3[0] & 0xfc) >> 2;
        char_array_4[1] = ((char_array_3[0] & 0x03) << 4) + ((char_array_3[1] & 0xf0) >> 4);
        char_array_4[2] = ((char_array_3[1] & 0x0f) << 2) + ((char_array_3[2] & 0xc0) >> 6);

        for (j = 0; (j < i + 1); j++) {
            result += base64_chars[char_array_4[j]];
        }

        while ((i++ < 3)) {
            result += '=';
        }
    }

    return result;
}

} // namespace cyxwiz
