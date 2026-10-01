#include "markdown_view.h"

#include "../core/markdown_blocks.h"
#include "editor_fonts.h"
#include "ui_fonts.h"
#include "ui_platform.h"
#include "ui_tokens.h"

#include <imgui.h>

#include <algorithm>
#include <cfloat>
#include <string>
#include <vector>

namespace cyxwiz::ui {

namespace {
struct Style {
    ImFont* font = nullptr;
    float size = 13.0f;
    ImVec4 colour;
};

// Lays out runs from the cursor, wrapping at `right`; returns the height used.
float DrawRuns(const std::vector<md::Run>& runs, const Style& base, float left, float right, bool bright_bold) {
    const Tokens& t = CurrentTokens();
    ImDrawList* dl = ImGui::GetWindowDrawList();
    ImFont* bold = GetFont(Font::Bold) ? GetFont(Font::Bold) : base.font;
    ImFont* mono = gui::GetCodeFont() ? gui::GetCodeFont() : base.font;
    const float mono_size = std::max(10.0f, base.size - 1.0f);
    const float line_h = std::floor(base.size * 1.5f);
    const ImVec2 origin = ImGui::GetCursorScreenPos();
    float x = left;
    float y = origin.y;
    for (const md::Run& run : runs) {
        ImFont* font = run.code ? mono : (run.bold ? bold : base.font);
        const float size = run.code ? mono_size : base.size;
        ImVec4 colour = base.colour;
        if (run.bold && bright_bold) colour = t.text_bright;
        if (run.italic) colour = Mix(colour, t.text_dim, 0.35f);
        if (!run.url.empty()) colour = t.accent_text;
        // Words keep their trailing space; a forced break is its own token.
        size_t i = 0;
        const std::string& s = run.text;
        while (i < s.size()) {
            if (s[i] == '\n') {
                x = left;
                y += line_h;
                ++i;
                continue;
            }
            size_t j = i;
            while (j < s.size() && s[j] != ' ' && s[j] != '\n') ++j;
            size_t k = j;
            while (k < s.size() && s[k] == ' ') ++k;
            const char* begin = s.c_str() + i;
            const float word_w = font->CalcTextSizeA(size, FLT_MAX, 0.0f, begin, s.c_str() + j).x;
            const float full_w = font->CalcTextSizeA(size, FLT_MAX, 0.0f, begin, s.c_str() + k).x;
            const float pad = run.code ? 4.0f : 0.0f;
            if (x + word_w + 2.0f * pad > right && x > left) {
                x = left;
                y += line_h;
            }
            const float text_y = y + (line_h - size) * 0.5f;
            if (run.code) {
                dl->AddRectFilled(ImVec2(x, y + 2.0f), ImVec2(x + word_w + 2.0f * pad, y + line_h - 2.0f),
                                  ToU32(t.bg_raised), 3.0f);
                dl->AddText(font, size, ImVec2(x + pad, text_y), ToU32(t.text_bright), begin, s.c_str() + j);
                x += word_w + 2.0f * pad + (full_w - word_w);
            } else {
                dl->AddText(font, size, ImVec2(x, text_y), ToU32(colour), begin, s.c_str() + j);
                if (!run.url.empty()) {
                    const ImVec2 a(x, y), b(x + word_w, y + line_h);
                    if (ImGui::IsMouseHoveringRect(a, b) && ImGui::IsWindowHovered()) {
                        dl->AddLine(ImVec2(a.x, text_y + size + 1.0f), ImVec2(b.x, text_y + size + 1.0f), ToU32(colour));
                        ImGui::SetMouseCursor(ImGuiMouseCursor_Hand);
                        ImGui::SetTooltip("%s", run.url.c_str());
                        if (ImGui::IsMouseClicked(ImGuiMouseButton_Left)) OpenUrl(run.url);
                    }
                }
                x += full_w;
            }
            i = k;
        }
    }
    return (y - origin.y) + line_h;
}
}  // namespace

void MarkdownView(const std::string& markdown) {
    const Tokens& t = CurrentTokens();
    const auto blocks = md::Parse(markdown);
    ImFont* regular = ImGui::GetFont();
    const float base = ImGui::GetFontSize();
    const float left0 = ImGui::GetCursorScreenPos().x;
    const float right = left0 + ImGui::GetContentRegionAvail().x;
    ImDrawList* dl = ImGui::GetWindowDrawList();

    for (size_t n = 0; n < blocks.size(); ++n) {
        const md::Block& b = blocks[n];
        if (n > 0) ImGui::Dummy(ImVec2(0.0f, b.kind == md::Block::Kind::Heading ? 6.0f : 3.0f));
        const ImVec2 at = ImGui::GetCursorScreenPos();
        float height = 0.0f;
        switch (b.kind) {
            case md::Block::Kind::Heading: {
                Style s;
                ImFont* heading = GetFont(Font::Heading);
                ImFont* bold = GetFont(Font::Bold);
                s.font = b.level <= 2 && heading ? heading : (bold ? bold : regular);
                s.size = b.level == 1 ? s.font->FontSize * 1.15f : (b.level == 2 ? s.font->FontSize : base * 1.05f);
                s.colour = t.text_bright;
                height = DrawRuns(b.runs, s, at.x, right, true);
                break;
            }
            case md::Block::Kind::Paragraph: {
                height = DrawRuns(b.runs, {regular, base, t.text}, at.x, right, true);
                break;
            }
            case md::Block::Kind::Bullet:
            case md::Block::Kind::Numbered: {
                const float indent = 18.0f * static_cast<float>(b.level);
                const float marker_w = 18.0f;
                const float line_h = std::floor(base * 1.5f);
                if (b.kind == md::Block::Kind::Bullet) {
                    dl->AddCircleFilled(ImVec2(at.x + indent + 6.0f, at.y + line_h * 0.5f), 2.5f, ToU32(t.text_dim));
                } else {
                    const std::string num = std::to_string(b.number) + ".";
                    dl->AddText(ImVec2(at.x + indent, at.y + (line_h - base) * 0.5f), ToU32(t.text_dim), num.c_str());
                }
                const float text_x = at.x + indent + marker_w + (b.kind == md::Block::Kind::Numbered ? 4.0f : 0.0f);
                ImGui::SetCursorScreenPos(ImVec2(text_x, at.y));
                height = DrawRuns(b.runs, {regular, base, t.text}, text_x, right, true);
                ImGui::SetCursorScreenPos(at);
                break;
            }
            case md::Block::Kind::Quote: {
                ImGui::SetCursorScreenPos(ImVec2(at.x + 12.0f, at.y));
                height = DrawRuns(b.runs, {regular, base, t.text_dim}, at.x + 12.0f, right, true);
                dl->AddRectFilled(at, ImVec2(at.x + 3.0f, at.y + height), ToU32(WithAlpha(t.accent_text, 0.5f)), 1.5f);
                ImGui::SetCursorScreenPos(at);
                break;
            }
            case md::Block::Kind::Code: {
                ImFont* mono = gui::GetCodeFont() ? gui::GetCodeFont() : regular;
                const float size = std::max(10.0f, base - 1.0f);
                const float line_h = std::floor(size * 1.5f);
                int lines = 1;
                for (char c : b.code) lines += c == '\n';
                height = lines * line_h + 16.0f;
                dl->AddRectFilled(at, ImVec2(right, at.y + height), ToU32(t.bg_raised), 6.0f);
                dl->PushClipRect(at, ImVec2(right - 8.0f, at.y + height), true);
                float y = at.y + 8.0f;
                size_t start = 0;
                while (start <= b.code.size()) {
                    size_t end = b.code.find('\n', start);
                    if (end == std::string::npos) end = b.code.size();
                    dl->AddText(mono, size, ImVec2(at.x + 12.0f, y + (line_h - size) * 0.5f), ToU32(t.text),
                                b.code.c_str() + start, b.code.c_str() + end);
                    y += line_h;
                    start = end + 1;
                }
                dl->PopClipRect();
                break;
            }
            case md::Block::Kind::Rule: {
                height = 12.0f;
                dl->AddLine(ImVec2(at.x, at.y + 6.0f), ImVec2(right, at.y + 6.0f), ToU32(t.border_soft));
                break;
            }
        }
        ImGui::SetCursorScreenPos(at);
        ImGui::Dummy(ImVec2(right - at.x, height));
    }
}

}  // namespace cyxwiz::ui
