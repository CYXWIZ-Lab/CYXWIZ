#include "markdown_preview.h"
#include "../panels/output_renderer.h"
#include <imgui.h>

namespace gui {

void MarkdownPreviewRenderer::SetContent(const std::string& markdown) {
    content_ = markdown;
}

void MarkdownPreviewRenderer::Render() {
    if (content_.empty()) {
        ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "No content.");
        return;
    }

    ImGui::BeginChild("##MarkdownContent", ImVec2(0, 0), false);
    cyxwiz::OutputRenderer::RenderMarkdown(content_);
    ImGui::EndChild();
}

} // namespace gui
