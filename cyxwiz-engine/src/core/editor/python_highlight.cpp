#include "python_highlight.h"

#include "../python_tokenizer.h"
#include "text_document.h"

#include <algorithm>
#include <cctype>
#include <cstring>

namespace cyxwiz::editor {

namespace {
bool In(std::string_view word, std::initializer_list<const char*> list) {
    for (const char* w : list)
        if (word == w) return true;
    return false;
}

bool IsKeyword(std::string_view w) {
    return In(w, {"False", "None", "True", "and", "as", "assert", "async", "await", "break", "class", "continue",
                  "def", "del", "elif", "else", "except", "finally", "for", "from", "global", "if", "import", "in",
                  "is", "lambda", "nonlocal", "not", "or", "pass", "raise", "return", "try", "while", "with",
                  "yield", "match", "case"});
}

bool IsBuiltin(std::string_view w) {
    return In(w, {"abs", "all", "any", "ascii", "bin", "bool", "bytearray", "bytes", "callable", "chr",
                  "classmethod", "compile", "complex", "delattr", "dict", "dir", "divmod", "enumerate", "eval",
                  "exec", "filter", "float", "format", "frozenset", "getattr", "globals", "hasattr", "hash", "help",
                  "hex", "id", "input", "int", "isinstance", "issubclass", "iter", "len", "list", "locals", "map",
                  "max", "memoryview", "min", "next", "object", "oct", "open", "ord", "pow", "print", "property",
                  "range", "repr", "reversed", "round", "set", "setattr", "slice", "sorted", "staticmethod", "str",
                  "sum", "super", "tuple", "type", "vars", "zip", "Exception", "ValueError", "TypeError",
                  "KeyError", "IndexError", "RuntimeError", "AttributeError", "ImportError", "OSError",
                  "NotImplementedError", "StopIteration", "KeyboardInterrupt"});
}

bool IsConstantName(std::string_view w) {
    if (w == "self" || w == "cls") return true;
    bool letter = false;
    for (const char c : w) {
        if (std::islower(static_cast<unsigned char>(c))) return false;
        if (std::isupper(static_cast<unsigned char>(c))) letter = true;
    }
    return letter && w.size() > 1;
}

// From an opening triple quote at `from` (just after it), the end of the
// string on this line, or npos when it stays open.
size_t TripleEnd(std::string_view line, size_t from, char q) {
    for (size_t i = from; i < line.size(); ++i) {
        if (line[i] == '\\') {
            ++i;
            continue;
        }
        if (line[i] == q && i + 2 < line.size() + 0 && line[i + 1] == q && line[i + 2] == q) return i + 3;
    }
    return std::string_view::npos;
}

void Push(std::vector<Span>& spans, size_t start, size_t end, TokenKind kind) {
    if (end > start) spans.push_back({static_cast<int>(start), static_cast<int>(end - start), kind});
}
}  // namespace

LineState HighlightLine(std::string_view line, LineState in, std::vector<Span>& spans) {
    spans.clear();
    size_t pos = 0;
    if (in.triple) {
        const size_t end = TripleEnd(line, 0, in.triple);
        if (end == std::string_view::npos) {
            Push(spans, 0, line.size(), TokenKind::String);
            return in;
        }
        Push(spans, 0, end, TokenKind::String);
        pos = end;
    }

    // Cell markers: "%%..." and "# %%..." lines.
    {
        const size_t first = line.find_first_not_of(" \t");
        if (pos == 0 && first != std::string_view::npos &&
            (line.compare(first, 2, "%%") == 0 || line.compare(first, 4, "# %%") == 0)) {
            Push(spans, first, line.size(), TokenKind::CellMarker);
            return {};
        }
    }

    bool after_def = false;
    while (pos < line.size()) {
        const char c = line[pos];
        if (c == ' ' || c == '\t') {
            ++pos;
            continue;
        }
        if (c == '#') {
            Push(spans, pos, line.size(), TokenKind::Comment);
            return {};
        }
        // A triple-quoted string (with optional prefix) may run past the line.
        size_t q = pos;
        while (q < line.size() && q - pos < 2 && std::strchr("rRbBfFuU", line[q])) ++q;
        if (q < line.size() && (line[q] == '"' || line[q] == '\'') && q + 2 < line.size() && line[q + 1] == line[q] &&
            line[q + 2] == line[q] && (q == pos || q - pos <= 2)) {
            const bool prefix_ok = q == pos || !(pos > 0 && (std::isalnum(static_cast<unsigned char>(line[pos - 1])) || line[pos - 1] == '_'));
            if (prefix_ok) {
                const char quote = line[q];
                const size_t end = TripleEnd(line, q + 3, quote);
                if (end == std::string_view::npos) {
                    Push(spans, pos, line.size(), TokenKind::String);
                    return {quote};
                }
                Push(spans, pos, end, TokenKind::String);
                pos = end;
                after_def = false;
                continue;
            }
        }
        const char* b = nullptr;
        const char* e = nullptr;
        pytokens::Kind kind;
        if (!pytokens::Next(line.data() + pos, line.data() + line.size(), b, e, kind)) {
            ++pos;
            continue;
        }
        const size_t start = static_cast<size_t>(b - line.data());
        const size_t end = static_cast<size_t>(e - line.data());
        TokenKind out = TokenKind::Default;
        switch (kind) {
            case pytokens::Kind::String: out = TokenKind::String; break;
            case pytokens::Kind::Number: out = TokenKind::Number; break;
            case pytokens::Kind::Punctuation: out = TokenKind::Punctuation; break;
            case pytokens::Kind::Decorator: out = TokenKind::Decorator; break;
            case pytokens::Kind::Identifier: {
                const std::string_view word = line.substr(start, end - start);
                size_t next = end;
                while (next < line.size() && (line[next] == ' ' || line[next] == '\t')) ++next;
                if (IsKeyword(word)) out = TokenKind::Keyword;
                else if (after_def) out = TokenKind::Function;
                else if (IsBuiltin(word)) out = TokenKind::Builtin;
                else if (IsConstantName(word)) out = TokenKind::Constant;
                else if (next < line.size() && line[next] == '(') out = TokenKind::Function;
                after_def = word == "def" || word == "class";
                Push(spans, start, end, out);
                pos = end;
                continue;
            }
        }
        after_def = false;
        Push(spans, start, end, out);
        pos = end;
    }
    return {};
}

void Highlighter::Update(const Document& document) {
    recoloured_ = 0;
    if (primed_ && document.Version() == version_ && static_cast<int>(lines_.size()) == document.LineCount()) return;
    primed_ = true;
    version_ = document.Version();
    const int count = document.LineCount();

    // Keep cached lines where the text still matches at the same index
    // (cheap and good enough: edits that shift lines recolour from there).
    std::vector<Line> next(static_cast<size_t>(count));
    LineState state{};
    for (int i = 0; i < count; ++i) {
        const std::string& text = document.Line(i);
        Line& line = next[static_cast<size_t>(i)];
        if (i < static_cast<int>(lines_.size())) {
            Line& old = lines_[static_cast<size_t>(i)];
            if (old.valid && old.in == state && old.text == text) {
                line = std::move(old);
                state = line.out;
                continue;
            }
        }
        line.text = text;
        line.in = state;
        line.out = HighlightLine(text, state, line.spans);
        line.valid = true;
        state = line.out;
        ++recoloured_;
    }
    lines_ = std::move(next);
}

}  // namespace cyxwiz::editor
