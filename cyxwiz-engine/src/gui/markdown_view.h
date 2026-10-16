#pragma once

// Draws markdown (core/markdown_blocks) with wrapped text: headings in the
// heading/bold fonts, real bold, inline code chips, links that open in the
// browser, lists, quotes, code blocks, rules (TOFIX133 P4 step 4.3b,
// board 4). Uses the full content width of the current window.

#include <string>

namespace cyxwiz::ui {

void MarkdownView(const std::string& markdown);

}  // namespace cyxwiz::ui
