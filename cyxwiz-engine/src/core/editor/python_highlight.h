// Python colouring for the Script Editor (TOFIX133 P1). Each line is
// coloured from the state its previous line ends in (inside a ''' or """
// string or not), so multi-line strings colour correctly; the cache only
// recolours lines whose text or incoming state changed. No ImGui.
#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace cyxwiz::editor {

class Document;

enum class TokenKind : uint8_t {
    Default,
    Keyword,
    Builtin,
    String,
    Number,
    Comment,
    Function,   // name after def/class, or a name being called
    Decorator,
    Constant,   // UPPER_CASE names, self, cls
    Punctuation,
    CellMarker  // %% / # %% lines
};

struct Span {
    int start = 0;
    int length = 0;
    TokenKind kind = TokenKind::Default;
};

struct LineState {
    char triple = 0;  // the quote of an open ''' or """ string, else 0
    friend bool operator==(const LineState& a, const LineState& b) { return a.triple == b.triple; }
    friend bool operator!=(const LineState& a, const LineState& b) { return !(a == b); }
};

// Colours one line given the state it starts in; returns the end state.
LineState HighlightLine(std::string_view line, LineState in, std::vector<Span>& spans);

class Highlighter {
public:
    // Brings the cache up to date with the document; cheap when little changed.
    void Update(const Document& document);
    // A line the cache does not hold yet has no spans (drawn plain).
    const std::vector<Span>& Spans(int line) const {
        static const std::vector<Span> kNone;
        return line >= 0 && static_cast<size_t>(line) < lines_.size() ? lines_[static_cast<size_t>(line)].spans : kNone;
    }
    LineState StateAtStart(int line) const { return lines_[static_cast<size_t>(line)].in; }
    int RecolouredLastUpdate() const { return recoloured_; }

private:
    struct Line {
        std::string text;
        LineState in;
        LineState out;
        std::vector<Span> spans;
        bool valid = false;
    };
    std::vector<Line> lines_;
    uint64_t version_ = 0;
    bool primed_ = false;
    int recoloured_ = 0;
};

}  // namespace cyxwiz::editor
