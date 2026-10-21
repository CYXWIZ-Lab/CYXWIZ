// Markdown for notebook text cells (TOFIX133 P4 step 4.3b).
#include "../src/core/markdown_blocks.h"

#include <cstdlib>
#include <iostream>
#include <string>

using namespace cyxwiz::md;

namespace {
void Check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        std::exit(1);
    }
}

std::string Plain(const std::vector<Run>& runs) {
    std::string s;
    for (const auto& r : runs) s += r.text;
    return s;
}
}  // namespace

int main() {
    // Board 4's text cell.
    const auto blocks = Parse(
        "## Sentiment model check\n"
        "Loads the **prepared vocabulary** and checks the tokenizer on statements from\n"
        "`sentiment_mental_health.csv` before calling the embedded server.\n");
    Check(blocks.size() == 2, "heading and one paragraph (lines joined)");
    Check(blocks[0].kind == Block::Kind::Heading && blocks[0].level == 2 && Plain(blocks[0].runs) == "Sentiment model check",
          "heading");
    const auto& p = blocks[1].runs;
    Check(Plain(p) == "Loads the prepared vocabulary and checks the tokenizer on statements from "
                      "sentiment_mental_health.csv before calling the embedded server.",
          "paragraph text");
    bool saw_bold = false, saw_code = false;
    for (const auto& r : p) {
        saw_bold = saw_bold || (r.bold && r.text == "prepared vocabulary");
        saw_code = saw_code || (r.code && r.text == "sentiment_mental_health.csv");
    }
    Check(saw_bold && saw_code, "bold and code runs");

    const auto list = Parse("- one\n- *two* and [docs](https://docs.python.org)\n  - nested\n1. first\n2) second\n");
    Check(list.size() == 5, "five list items");
    Check(list[0].kind == Block::Kind::Bullet && list[2].level == 1, "nested bullet");
    Check(list[1].runs[0].italic && list[1].runs[2].url == "https://docs.python.org" && list[1].runs[2].text == "docs",
          "italic and link");
    Check(list[3].kind == Block::Kind::Numbered && list[3].number == 1 && list[4].number == 2, "numbered items");

    const auto code = Parse("Before\n```python\nx = 1\n\ny = 2\n```\nAfter");
    Check(code.size() == 3 && code[1].kind == Block::Kind::Code && code[1].language == "python" &&
              code[1].code == "x = 1\n\ny = 2",
          "fenced code keeps blank lines");

    const auto misc = Parse("<h3>Results</h3>\n---\n> note **this**\n# Title #\n");
    Check(misc[0].kind == Block::Kind::Heading && misc[0].level == 3 && Plain(misc[0].runs) == "Results", "html heading");
    Check(misc[1].kind == Block::Kind::Rule, "rule");
    Check(misc[2].kind == Block::Kind::Quote && misc[2].runs.back().bold, "quote with bold");
    Check(Plain(misc[3].runs) == "Title", "closing hashes dropped");

    Check(Plain(ParseInline("snake_case_name and 2 * 3 * 4")) == "snake_case_name and 2 * 3 * 4", "no emphasis inside words or around spaces");
    Check(Plain(ParseInline("a \\*literal\\* star")) == "a *literal* star", "escapes");
    const auto html = ParseInline("<b>bold</b> <code>x</code><br>next");
    Check(html[0].bold && html[2].code && Plain(html) == "bold x\nnext", "html inline tags");
    Check(Parse("#hashtag").front().kind == Block::Kind::Paragraph, "#word is not a heading");
    Check(Parse("```\nnot closed").front().code == "not closed", "unclosed fence");

    std::cout << "markdown blocks: headings, paragraphs, lists, code, quotes, inline styles. OK\n";
    return 0;
}
