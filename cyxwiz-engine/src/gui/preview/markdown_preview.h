#pragma once

#include <string>

namespace gui {

/**
 * Markdown preview renderer
 * Wraps OutputRenderer::RenderMarkdown with file loading
 */
class MarkdownPreviewRenderer {
public:
    MarkdownPreviewRenderer() = default;

    void SetContent(const std::string& markdown);
    void Render();

    bool HasContent() const { return !content_.empty(); }

private:
    std::string content_;
};

} // namespace gui
