#include "markdown_blocks.h"

#include <cctype>
#include <sstream>

namespace cyxwiz::md {

namespace {
std::string TrimRight(std::string s) {
    while (!s.empty() && (s.back() == ' ' || s.back() == '\t' || s.back() == '\r')) s.pop_back();
    return s;
}

std::string TrimLeft(const std::string& s) {
    size_t i = 0;
    while (i < s.size() && (s[i] == ' ' || s[i] == '\t')) ++i;
    return s.substr(i);
}

int Indent(const std::string& s) {
    int n = 0;
    for (char c : s) {
        if (c == ' ') ++n;
        else if (c == '\t') n += 4;
        else break;
    }
    return n;
}

bool StartsWithCI(const std::string& s, size_t at, const char* word) {
    for (size_t k = 0; word[k]; ++k) {
        if (at + k >= s.size()) return false;
        if (std::tolower(static_cast<unsigned char>(s[at + k])) != word[k]) return false;
    }
    return true;
}

bool IsRule(const std::string& t) {
    if (t.size() < 3) return false;
    const char c = t[0];
    if (c != '-' && c != '*' && c != '_') return false;
    int count = 0;
    for (char ch : t) {
        if (ch == c) ++count;
        else if (ch != ' ') return false;
    }
    return count >= 3;
}

bool IsWordChar(char c) { return std::isalnum(static_cast<unsigned char>(c)) != 0; }

// <h2>Title</h2> on one line.
bool HtmlHeading(const std::string& t, int* level, std::string* inner) {
    if (t.size() < 9 || t[0] != '<' || (t[1] != 'h' && t[1] != 'H') || t[2] < '1' || t[2] > '6' || t[3] != '>') return false;
    const std::string close = std::string("</") + t[1] + t[2] + ">";
    const size_t end = t.find(close);
    if (end == std::string::npos) return false;
    *level = t[2] - '0';
    *inner = t.substr(4, end - 4);
    return true;
}
}  // namespace

std::vector<Run> ParseInline(const std::string& text) {
    std::vector<Run> runs;
    Run cur;
    bool bold = false, italic = false;
    auto flush = [&]() {
        if (!cur.text.empty()) runs.push_back(cur);
        cur = Run{};
        cur.bold = bold;
        cur.italic = italic;
    };
    for (size_t i = 0; i < text.size(); ++i) {
        const char c = text[i];
        if (c == '\\' && i + 1 < text.size() && std::ispunct(static_cast<unsigned char>(text[i + 1]))) {
            cur.text += text[++i];
            continue;
        }
        if (c == '`') {
            const size_t end = text.find('`', i + 1);
            if (end != std::string::npos) {
                flush();
                Run code;
                code.text = text.substr(i + 1, end - i - 1);
                code.code = true;
                runs.push_back(code);
                i = end;
                continue;
            }
        }
        if (c == '[') {
            const size_t close = text.find(']', i + 1);
            if (close != std::string::npos && close + 1 < text.size() && text[close + 1] == '(') {
                const size_t paren = text.find(')', close + 2);
                if (paren != std::string::npos) {
                    flush();
                    Run link;
                    link.text = text.substr(i + 1, close - i - 1);
                    link.url = text.substr(close + 2, paren - close - 2);
                    link.bold = bold;
                    link.italic = italic;
                    runs.push_back(link);
                    i = paren;
                    continue;
                }
            }
        }
        if ((c == '*' || c == '_') && i + 1 < text.size() && text[i + 1] == c) {
            // __ only between non-word characters, like CommonMark.
            const bool word_before = i > 0 && IsWordChar(text[i - 1]);
            const bool word_after = i + 2 < text.size() && IsWordChar(text[i + 2]);
            if (c == '*' || !(word_before && word_after)) {
                flush();
                bold = !bold;
                cur.bold = bold;
                ++i;
                continue;
            }
        }
        if (c == '*' || c == '_') {
            const bool word_before = i > 0 && IsWordChar(text[i - 1]);
            const bool word_after = i + 1 < text.size() && IsWordChar(text[i + 1]);
            const bool opens = !italic && i + 1 < text.size() && text[i + 1] != ' ';
            const bool closes = italic && i > 0 && text[i - 1] != ' ';
            if ((opens || closes) && (c == '*' || !(word_before && word_after))) {
                flush();
                italic = !italic;
                cur.italic = italic;
                continue;
            }
        }
        if (c == '<') {
            struct Tag { const char* name; int kind; };  // 1 bold, 2 italic, 3 code, 4 break
            static const Tag tags[] = {{"<b>", 1},       {"</b>", 1},  {"<strong>", 1}, {"</strong>", 1},
                                       {"<i>", 2},       {"</i>", 2},  {"<em>", 2},     {"</em>", 2},
                                       {"<br>", 4},      {"<br/>", 4}, {"<br />", 4}};
            bool matched = false;
            for (const Tag& tag : tags) {
                if (StartsWithCI(text, i, tag.name)) {
                    flush();
                    if (tag.kind == 1) bold = !bold;
                    if (tag.kind == 2) italic = !italic;
                    if (tag.kind == 4) cur.text += '\n';
                    cur.bold = bold;
                    cur.italic = italic;
                    i += std::char_traits<char>::length(tag.name) - 1;
                    matched = true;
                    break;
                }
            }
            if (matched) continue;
            if (StartsWithCI(text, i, "<code>")) {
                const size_t end = text.find("</code>", i);
                if (end != std::string::npos) {
                    flush();
                    Run code;
                    code.text = text.substr(i + 6, end - i - 6);
                    code.code = true;
                    runs.push_back(code);
                    i = end + 6;
                    continue;
                }
            }
        }
        cur.text += c;
    }
    flush();
    return runs;
}

std::vector<Block> Parse(const std::string& text) {
    std::vector<Block> blocks;
    std::istringstream in(text);
    std::string raw;
    std::string paragraph;
    bool in_fence = false;
    std::string fence;
    Block code;

    auto end_paragraph = [&]() {
        if (paragraph.empty()) return;
        Block b;
        b.kind = Block::Kind::Paragraph;
        b.runs = ParseInline(paragraph);
        blocks.push_back(std::move(b));
        paragraph.clear();
    };

    while (std::getline(in, raw)) {
        const std::string line = TrimRight(raw);
        const std::string t = TrimLeft(line);
        if (in_fence) {
            if (t.rfind(fence, 0) == 0) {
                in_fence = false;
                blocks.push_back(code);
                continue;
            }
            if (!code.code.empty()) code.code += '\n';
            code.code += line;
            continue;
        }
        if (t.rfind("```", 0) == 0 || t.rfind("~~~", 0) == 0) {
            end_paragraph();
            in_fence = true;
            fence = t.substr(0, 3);
            code = Block{};
            code.kind = Block::Kind::Code;
            code.language = TrimLeft(t.substr(3));
            continue;
        }
        if (t.empty()) {
            end_paragraph();
            continue;
        }
        int level = 0;
        std::string inner;
        if (t[0] == '#') {
            while (level < static_cast<int>(t.size()) && t[level] == '#') ++level;
            if (level <= 6 && (level == static_cast<int>(t.size()) || t[level] == ' ')) {
                end_paragraph();
                Block b;
                b.kind = Block::Kind::Heading;
                b.level = level;
                std::string title = TrimLeft(t.substr(level));
                while (!title.empty() && title.back() == '#') title.pop_back();  // closing #s
                b.runs = ParseInline(TrimRight(title));
                blocks.push_back(std::move(b));
                continue;
            }
        }
        if (HtmlHeading(t, &level, &inner)) {
            end_paragraph();
            Block b;
            b.kind = Block::Kind::Heading;
            b.level = level;
            b.runs = ParseInline(inner);
            blocks.push_back(std::move(b));
            continue;
        }
        if (IsRule(t) || t.rfind("<hr", 0) == 0) {
            end_paragraph();
            Block b;
            b.kind = Block::Kind::Rule;
            blocks.push_back(b);
            continue;
        }
        const int indent = Indent(line);
        if ((t[0] == '-' || t[0] == '*' || t[0] == '+') && t.size() > 1 && t[1] == ' ') {
            end_paragraph();
            Block b;
            b.kind = Block::Kind::Bullet;
            b.level = indent / 2;
            b.runs = ParseInline(TrimLeft(t.substr(2)));
            blocks.push_back(std::move(b));
            continue;
        }
        size_t digits = 0;
        while (digits < t.size() && std::isdigit(static_cast<unsigned char>(t[digits]))) ++digits;
        if (digits > 0 && digits + 1 < t.size() && (t[digits] == '.' || t[digits] == ')') && t[digits + 1] == ' ') {
            end_paragraph();
            Block b;
            b.kind = Block::Kind::Numbered;
            b.level = indent / 2;
            b.number = std::stoi(t.substr(0, digits));
            b.runs = ParseInline(TrimLeft(t.substr(digits + 2)));
            blocks.push_back(std::move(b));
            continue;
        }
        if (t[0] == '>') {
            end_paragraph();
            Block b;
            b.kind = Block::Kind::Quote;
            b.runs = ParseInline(TrimLeft(t.substr(1)));
            blocks.push_back(std::move(b));
            continue;
        }
        if (!paragraph.empty()) paragraph += ' ';
        paragraph += t;
    }
    if (in_fence) blocks.push_back(code);  // an unclosed fence runs to the end
    end_paragraph();
    return blocks;
}

}  // namespace cyxwiz::md
