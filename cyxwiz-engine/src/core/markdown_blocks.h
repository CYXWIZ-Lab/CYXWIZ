// Markdown for notebook text cells (TOFIX133 P4 step 4.3b, board 4): a
// small CommonMark subset parsed into blocks of styled runs. Pure data; the
// renderer (gui/markdown_view) wraps and draws it.
#pragma once

#include <string>
#include <vector>

namespace cyxwiz::md {

struct Run {
    std::string text;
    bool bold = false;
    bool italic = false;
    bool code = false;
    std::string url;  // a link when not empty
};

struct Block {
    enum class Kind { Heading, Paragraph, Bullet, Numbered, Quote, Code, Rule };
    Kind kind = Kind::Paragraph;
    int level = 0;            // heading 1-6; list nesting (0 = top)
    int number = 0;           // numbered list item
    std::vector<Run> runs;    // text blocks
    std::string code;         // Code: the lines, '\n' separated
    std::string language;     // Code: the fence's info string
};

// Headings (# and <h1>..<h6>), paragraphs (lines joined), - * + and 1.
// lists (indent by two or more spaces nests), > quotes, ``` / ~~~ fences,
// --- *** ___ rules. Inline: **bold** __bold__ *italic* _italic_ `code`
// [text](url), <b> <strong> <i> <em> <code>, backslash escapes, <br>.
std::vector<Block> Parse(const std::string& text);

// Inline runs of one line or paragraph.
std::vector<Run> ParseInline(const std::string& text);

}  // namespace cyxwiz::md
