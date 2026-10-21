#include "markdown_preview.h"
#include "../panels/output_renderer.h"
#include "../ui_tokens.h"
#include <imgui.h>

namespace gui {

void MarkdownPreviewRenderer::SetContent(const std::string& markdown) {
    content_ = markdown;
}

void MarkdownPreviewRenderer::Render() {
    if (content_.empty()) {
        ImGui::TextColored(cyxwiz::ui::CurrentTokens().text_dim, "No content.");
        return;
    }

    ImGui::BeginChild("##MarkdownContent", ImVec2(0, 0), false);
    cyxwiz::OutputRenderer::RenderMarkdown(content_);
    ImGui::EndChild();
}

} // namespace gui
