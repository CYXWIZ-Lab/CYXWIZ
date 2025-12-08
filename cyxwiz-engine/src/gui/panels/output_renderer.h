#pragma once

#include "../../scripting/cell.h"
#include <string>
#include <vector>
#include <functional>
#include <imgui.h>

#ifdef _WIN32
#include <glad/glad.h>
#else
#include <GL/gl.h>
#endif

namespace cyxwiz {

// Callback for "Open in Window" action
using OpenPlotWindowCallback = std::function<void(GLuint texture_id, int width, int height, const std::string& title)>;

/**
 * Utility class for rendering cell outputs in the notebook-style editor
 * Handles text, errors, images, plots, tables, and markdown
 */
class OutputRenderer {
public:
    // ========== Text Output ==========

    /**
     * Render plain text output
     */
    static void RenderText(const std::string& text, bool wrap = true);

    /**
     * Render error/traceback output with red styling
     */
    static void RenderError(const std::string& error);

    /**
     * Render stream output (stdout/stderr)
     */
    static void RenderStream(const std::string& text, const std::string& stream_name);

    // ========== Rich Output ==========

    /**
     * Render an image from OpenGL texture
     */
    static void RenderImage(GLuint texture_id, int width, int height, float max_width = 0.0f);

    /**
     * Render a plot (same as image but with plot-specific context menu)
     */
    static void RenderPlot(GLuint texture_id, int width, int height, const std::string& plot_id = "");

    /**
     * Render a plot with raw PNG data for copy/save operations
     */
    static void RenderPlotWithData(GLuint texture_id, int width, int height,
                                   const std::vector<unsigned char>& png_data,
                                   const std::string& plot_id = "");

    /**
     * Set callback for "Open in Window" action
     */
    static void SetOpenPlotWindowCallback(OpenPlotWindowCallback callback);

    /**
     * Copy image to clipboard (Windows only)
     */
    static bool CopyImageToClipboard(const std::vector<unsigned char>& png_data);

    /**
     * Save image to file with dialog
     */
    static bool SaveImageToFile(const std::vector<unsigned char>& png_data, const std::string& default_name = "plot");

    /**
     * Render a table
     */
    static void RenderTable(const std::vector<std::vector<std::string>>& data,
                           const std::vector<std::string>& headers = {});

    /**
     * Render basic markdown
     */
    static void RenderMarkdown(const std::string& markdown);

    // ========== Cell Output ==========

    /**
     * Render a CellOutput (dispatches to appropriate renderer)
     */
    static void RenderCellOutput(const CellOutput& output);

    // ========== Texture Utilities ==========

    /**
     * Create OpenGL texture from base64-encoded PNG
     * @return Texture ID and dimensions
     */
    static GLuint CreateTextureFromBase64PNG(const std::string& base64_data, int& out_width, int& out_height);

    /**
     * Create OpenGL texture from raw RGBA data
     */
    static GLuint CreateTextureFromRGBA(const unsigned char* data, int width, int height);

    /**
     * Delete a texture
     */
    static void DeleteTexture(GLuint texture_id);

    // ========== Base64 Utilities ==========

    /**
     * Decode base64 string to bytes
     */
    static std::vector<unsigned char> Base64Decode(const std::string& encoded);

    /**
     * Encode bytes to base64 string
     */
    static std::string Base64Encode(const unsigned char* data, size_t length);

private:
    // Markdown rendering helpers
    static void RenderMarkdownLine(const std::string& line);
    static void RenderMarkdownInline(const std::string& text);
    static bool IsHeaderLine(const std::string& line, int& level);
    static bool IsListItem(const std::string& line, std::string& content);
    static bool IsCodeBlock(const std::string& line);

    // Style helpers
    static ImVec4 GetErrorColor() { return ImVec4(1.0f, 0.4f, 0.4f, 1.0f); }
    static ImVec4 GetStdoutColor() { return ImVec4(0.9f, 0.9f, 0.9f, 1.0f); }
    static ImVec4 GetStderrColor() { return ImVec4(1.0f, 0.6f, 0.4f, 1.0f); }
    static ImVec4 GetHeaderColor() { return ImVec4(0.4f, 0.8f, 1.0f, 1.0f); }
    static ImVec4 GetCodeColor() { return ImVec4(0.6f, 0.9f, 0.6f, 1.0f); }

    // Callback for opening plots in new window
    static OpenPlotWindowCallback open_plot_window_callback_;
};

} // namespace cyxwiz
