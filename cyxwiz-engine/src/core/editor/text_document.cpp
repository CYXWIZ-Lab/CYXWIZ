#include "text_document.h"

#include <algorithm>
#include <cctype>

namespace cyxwiz::editor {

namespace {
bool IsContinuation(char c) { return (static_cast<unsigned char>(c) & 0xC0) == 0x80; }

enum class CharClass { Space, Word, Other };
CharClass ClassOf(char c) {
    const unsigned char u = static_cast<unsigned char>(c);
    if (c == ' ' || c == '\t') return CharClass::Space;
    if (std::isalnum(u) || c == '_' || u >= 0x80) return CharClass::Word;
    return CharClass::Other;
}

char Closer(char open) {
    switch (open) {
        case '(': return ')';
        case '[': return ']';
        case '{': return '}';
        case '"': return '"';
        case '\'': return '\'';
        default: return 0;
    }
}
bool IsCloser(char c) { return c == ')' || c == ']' || c == '}'; }

std::vector<std::string> SplitLines(std::string_view text) {
    std::vector<std::string> out(1);
    for (size_t i = 0; i < text.size(); ++i) {
        const char c = text[i];
        if (c == '\r') {
            if (i + 1 < text.size() && text[i + 1] == '\n') ++i;
            out.emplace_back();
        } else if (c == '\n') {
            out.emplace_back();
        } else {
            out.back() += c;
        }
    }
    return out;
}

// Where `text` ends when inserted at `start`.
Pos EndOf(Pos start, std::string_view text) {
    Pos p = start;
    for (const char c : text) {
        if (c == '\n') {
            ++p.line;
            p.col = 0;
        } else {
            ++p.col;
        }
    }
    return p;
}

// A position after a replacement of [start, old_end) that now ends at new_end.
Pos MapPos(Pos p, Pos start, Pos old_end, Pos new_end) {
    if (p < start) return p;
    if (old_end <= p) {
        if (p.line == old_end.line) return {new_end.line, new_end.col + (p.col - old_end.col)};
        return {p.line + (new_end.line - old_end.line), p.col};
    }
    return new_end;  // inside the replaced text
}

int LeadingSpaces(const std::string& line) {
    int n = 0;
    while (n < static_cast<int>(line.size()) && (line[static_cast<size_t>(n)] == ' ' || line[static_cast<size_t>(n)] == '\t')) ++n;
    return n;
}

bool IsBlank(const std::string& line) { return LeadingSpaces(line) == static_cast<int>(line.size()); }
}  // namespace

int VisualColumn(std::string_view line, int byte, int tab_size) {
    int col = 0;
    const int end = std::min(byte, static_cast<int>(line.size()));
    for (int i = 0; i < end; ++i) {
        const char c = line[static_cast<size_t>(i)];
        if (c == '\t') col = (col / tab_size + 1) * tab_size;
        else if (!IsContinuation(c)) ++col;
    }
    return col;
}

int ByteForVisual(std::string_view line, int visual, int tab_size) {
    int col = 0;
    int i = 0;
    const int n = static_cast<int>(line.size());
    while (i < n) {
        const char c = line[static_cast<size_t>(i)];
        const int next = c == '\t' ? (col / tab_size + 1) * tab_size : col + 1;
        if (next > visual) {
            // Nearer to the start or the end of this character.
            if (visual - col > next - visual) {
                ++i;
                while (i < n && IsContinuation(line[static_cast<size_t>(i)])) ++i;
            }
            return i;
        }
        col = next;
        ++i;
        while (i < n && IsContinuation(line[static_cast<size_t>(i)])) ++i;
    }
    return n;
}

Document::Document() : lines_(1), selections_(1) {}

Document::Document(std::string_view text) : Document() { SetText(text); }

void Document::SetText(std::string_view text) {
    const int old_count = static_cast<int>(lines_.size());
    lines_ = SplitLines(text);
    line_edits_.assign(1, LineEdit{0, old_count - 1, static_cast<int>(lines_.size()) - 1});
    selections_.assign(1, Selection{});
    primary_ = 0;
    undo_.clear();
    redo_.clear();
    merge_open_ = false;
    saved_group_id_ = 0;
    ++version_;
}

std::string Document::Text() const {
    std::string out;
    for (size_t i = 0; i < lines_.size(); ++i) {
        if (i) out += '\n';
        out += lines_[i];
    }
    return out;
}

Pos Document::Clamp(Pos p) const {
    p.line = std::clamp(p.line, 0, LineCount() - 1);
    const std::string& line = Line(p.line);
    p.col = std::clamp(p.col, 0, static_cast<int>(line.size()));
    while (p.col > 0 && p.col < static_cast<int>(line.size()) && IsContinuation(line[static_cast<size_t>(p.col)])) --p.col;
    return p;
}

std::string Document::GetRange(Pos a, Pos b) const {
    a = Clamp(a);
    b = Clamp(b);
    if (b < a) std::swap(a, b);
    if (a.line == b.line) return Line(a.line).substr(static_cast<size_t>(a.col), static_cast<size_t>(b.col - a.col));
    std::string out = Line(a.line).substr(static_cast<size_t>(a.col));
    for (int l = a.line + 1; l < b.line; ++l) {
        out += '\n';
        out += Line(l);
    }
    out += '\n';
    out += Line(b.line).substr(0, static_cast<size_t>(b.col));
    return out;
}

// ---------------------------------------------------------------- cursors

void Document::Normalize() {
    if (selections_.empty()) selections_.push_back(Selection{});
    primary_ = std::clamp(primary_, 0, static_cast<int>(selections_.size()) - 1);
    for (auto& s : selections_) {
        s.anchor = Clamp(s.anchor);
        s.head = Clamp(s.head);
    }
    const Selection primary = selections_[static_cast<size_t>(primary_)];
    std::stable_sort(selections_.begin(), selections_.end(),
                     [](const Selection& a, const Selection& b) { return a.Start() < b.Start(); });
    std::vector<Selection> merged;
    for (const auto& s : selections_) {
        if (!merged.empty()) {
            Selection& last = merged.back();
            const bool overlap = s.Start() < last.End() || s.Start() == last.Start() ||
                                 (s.Empty() && last.End() == s.Start()) || (last.Empty() && last.Start() == s.Start());
            if (overlap) {
                const Pos start = std::min(last.Start(), s.Start());
                const Pos end = std::max(last.End(), s.End());
                const bool forward = !(last.head < last.anchor);
                last.anchor = forward ? start : end;
                last.head = forward ? end : start;
                continue;
            }
        }
        merged.push_back(s);
    }
    selections_ = std::move(merged);
    primary_ = 0;
    for (size_t i = 0; i < selections_.size(); ++i) {
        if (selections_[i].Start() <= primary.head && primary.head <= selections_[i].End()) {
            primary_ = static_cast<int>(i);
            break;
        }
    }
}

void Document::SetSelections(std::vector<Selection> selections, int primary) {
    selections_ = std::move(selections);
    primary_ = primary < 0 ? static_cast<int>(selections_.size()) - 1 : primary;
    merge_open_ = false;
    Normalize();
}

void Document::SetCursor(Pos p, bool extend) {
    Selection s;
    s.anchor = extend ? Primary().anchor : p;
    s.head = p;
    SetSelections({s}, 0);
}

void Document::AddCursor(Pos p) {
    auto selections = selections_;
    selections.push_back(Selection{p, p, -1});
    SetSelections(std::move(selections));
}

void Document::AddCursorVertical(int direction) {
    auto selections = selections_;
    const Selection primary = Primary();
    int new_primary = -1;
    for (const auto& s : selections_) {
        const int line = s.head.line + direction;
        if (line < 0 || line >= LineCount()) continue;
        const int goal = s.goal >= 0 ? s.goal : VisualColumn(Line(s.head.line), s.head.col, options_.tab_size);
        const Pos p{line, ByteForVisual(Line(line), goal, options_.tab_size)};
        selections.push_back(Selection{p, p, goal});
        if (s.head == primary.head) new_primary = static_cast<int>(selections.size()) - 1;
    }
    SetSelections(std::move(selections), new_primary);
}

void Document::SelectAll() {
    const Pos end{LineCount() - 1, static_cast<int>(Line(LineCount() - 1).size())};
    SetSelections({Selection{{0, 0}, end, -1}}, 0);
}

void Document::SelectWord(Pos p) {
    p = Clamp(p);
    const std::string& line = Line(p.line);
    int at = p.col;
    if (at >= static_cast<int>(line.size()) || ClassOf(line[static_cast<size_t>(at)]) == CharClass::Space) {
        if (at > 0 && ClassOf(line[static_cast<size_t>(at - 1)]) != CharClass::Space) --at;
    }
    if (at >= static_cast<int>(line.size())) {
        SetCursor(p);
        return;
    }
    const CharClass cls = ClassOf(line[static_cast<size_t>(at)]);
    int a = at;
    int b = at;
    while (a > 0 && ClassOf(line[static_cast<size_t>(a - 1)]) == cls) --a;
    while (b < static_cast<int>(line.size()) && ClassOf(line[static_cast<size_t>(b)]) == cls) ++b;
    SetSelections({Selection{{p.line, a}, {p.line, b}, -1}}, 0);
}

void Document::SelectLine(int line) {
    line = std::clamp(line, 0, LineCount() - 1);
    const Pos end = line + 1 < LineCount() ? Pos{line + 1, 0} : Pos{line, static_cast<int>(Line(line).size())};
    SetSelections({Selection{{line, 0}, end, -1}}, 0);
}

void Document::SelectAllOccurrences() {
    Selection base = Primary();
    if (base.Empty()) {
        SelectWord(base.head);
        base = Primary();
        if (base.Empty()) return;
    }
    const std::string needle = GetRange(base.Start(), base.End());
    if (needle.empty() || needle.find('\n') != std::string::npos) return;
    std::vector<Selection> found;
    int primary = 0;
    for (int l = 0; l < LineCount(); ++l) {
        const std::string& line = Line(l);
        for (size_t at = line.find(needle); at != std::string::npos; at = line.find(needle, at + needle.size())) {
            const Pos a{l, static_cast<int>(at)};
            const Pos b{l, static_cast<int>(at + needle.size())};
            if (a == base.Start()) primary = static_cast<int>(found.size());
            found.push_back(Selection{a, b, -1});
        }
    }
    if (!found.empty()) SetSelections(std::move(found), primary);
}

Pos Document::PrevChar(Pos p) const {
    if (p.col == 0) return p.line > 0 ? Pos{p.line - 1, static_cast<int>(Line(p.line - 1).size())} : p;
    const std::string& line = Line(p.line);
    int c = p.col - 1;
    while (c > 0 && IsContinuation(line[static_cast<size_t>(c)])) --c;
    return {p.line, c};
}

Pos Document::NextChar(Pos p) const {
    const std::string& line = Line(p.line);
    if (p.col >= static_cast<int>(line.size())) return p.line + 1 < LineCount() ? Pos{p.line + 1, 0} : p;
    int c = p.col + 1;
    while (c < static_cast<int>(line.size()) && IsContinuation(line[static_cast<size_t>(c)])) ++c;
    return {p.line, c};
}

Pos Document::WordLeft(Pos p) const {
    if (p.col == 0) return PrevChar(p);
    const std::string& line = Line(p.line);
    int c = p.col;
    while (c > 0 && ClassOf(line[static_cast<size_t>(c - 1)]) == CharClass::Space) --c;
    if (c == 0) return {p.line, 0};
    const CharClass cls = ClassOf(line[static_cast<size_t>(c - 1)]);
    while (c > 0 && ClassOf(line[static_cast<size_t>(c - 1)]) == cls) --c;
    return {p.line, c};
}

Pos Document::WordRight(Pos p) const {
    const std::string& line = Line(p.line);
    const int n = static_cast<int>(line.size());
    if (p.col >= n) return NextChar(p);
    int c = p.col;
    while (c < n && ClassOf(line[static_cast<size_t>(c)]) == CharClass::Space) ++c;
    if (c == n) return {p.line, n};
    const CharClass cls = ClassOf(line[static_cast<size_t>(c)]);
    while (c < n && ClassOf(line[static_cast<size_t>(c)]) == cls) ++c;
    return {p.line, c};
}

Pos Document::MovePos(Pos p, Motion motion, int& goal, int page_lines) const {
    const int tab = options_.tab_size;
    auto vertical = [&](int delta) {
        if (goal < 0) goal = VisualColumn(Line(p.line), p.col, tab);
        const int line = p.line + delta;
        if (line < 0) return Pos{0, 0};
        if (line >= LineCount()) return Pos{LineCount() - 1, static_cast<int>(Line(LineCount() - 1).size())};
        return Pos{line, ByteForVisual(Line(line), goal, tab)};
    };
    switch (motion) {
        case Motion::Left: goal = -1; return PrevChar(p);
        case Motion::Right: goal = -1; return NextChar(p);
        case Motion::Up: return vertical(-1);
        case Motion::Down: return vertical(1);
        case Motion::PageUp: return vertical(-page_lines);
        case Motion::PageDown: return vertical(page_lines);
        case Motion::WordLeft: goal = -1; return WordLeft(p);
        case Motion::WordRight: goal = -1; return WordRight(p);
        case Motion::Home: {
            // First press: the first non-blank character; again: column 0.
            goal = -1;
            const int indent = LeadingSpaces(Line(p.line));
            return {p.line, p.col == indent ? 0 : indent};
        }
        case Motion::End: goal = -1; return {p.line, static_cast<int>(Line(p.line).size())};
        case Motion::DocStart: goal = -1; return {0, 0};
        case Motion::DocEnd: goal = -1; return {LineCount() - 1, static_cast<int>(Line(LineCount() - 1).size())};
    }
    return p;
}

void Document::Move(Motion motion, bool extend, int page_lines) {
    for (auto& s : selections_) {
        if (!extend && !s.Empty() && (motion == Motion::Left || motion == Motion::Right)) {
            // An arrow first collapses a selection to its side.
            const Pos to = motion == Motion::Left ? s.Start() : s.End();
            s = Selection{to, to, -1};
            continue;
        }
        s.head = MovePos(s.head, motion, s.goal, page_lines);
        if (!extend) s.anchor = s.head;
    }
    merge_open_ = false;
    Normalize();
}

std::string Document::SelectedText() const {
    std::string out;
    bool any = false;
    for (const auto& s : selections_) {
        if (s.Empty()) continue;
        if (any) out += '\n';
        out += GetRange(s.Start(), s.End());
        any = true;
    }
    return out;
}

// ---------------------------------------------------------------- editing core

Pos Document::RawReplace(Pos a, Pos b, std::string_view text) {
    const std::string prefix = Line(a.line).substr(0, static_cast<size_t>(a.col));
    const std::string suffix = Line(b.line).substr(static_cast<size_t>(b.col));
    std::vector<std::string> parts = SplitLines(text);
    const Pos end{a.line + static_cast<int>(parts.size()) - 1,
                  parts.size() == 1 ? a.col + static_cast<int>(parts[0].size()) : static_cast<int>(parts.back().size())};
    parts.front() = prefix + parts.front();
    parts.back() += suffix;
    const int removed = b.line - a.line;
    const int added = static_cast<int>(parts.size()) - 1;
    lines_.erase(lines_.begin() + a.line, lines_.begin() + b.line + 1);
    lines_.insert(lines_.begin() + a.line, parts.begin(), parts.end());
    if (removed != 0 || added != 0) line_edits_.push_back({a.line, removed, added});
    return end;
}

void Document::ApplyEdits(std::vector<Edit> edits, GroupKind kind, CaretRule rule, const std::vector<int>* caret_offsets) {
    // Index edits so carets can be given back in their original order.
    std::vector<size_t> order(edits.size());
    for (size_t i = 0; i < edits.size(); ++i) {
        edits[i].a = Clamp(edits[i].a);
        edits[i].b = Clamp(edits[i].b);
        if (edits[i].b < edits[i].a) std::swap(edits[i].a, edits[i].b);
        order[i] = i;
    }
    std::sort(order.begin(), order.end(), [&](size_t x, size_t y) { return edits[y].a < edits[x].a; });

    Group group;
    group.before = selections_;
    group.primary_before = primary_;
    group.kind = kind;

    std::vector<Pos> carets(edits.size());
    std::vector<bool> applied(edits.size(), false);
    Pos floor{LineCount(), 0};  // start of the last applied edit: later ones must end before it
    for (const size_t i : order) {
        Edit& e = edits[i];
        if (floor < e.b) continue;  // overlaps an edit already applied: dropped
        Change change{e.a, GetRange(e.a, e.b), e.text};
        if (change.removed.empty() && change.inserted.empty()) {
            carets[i] = e.a;
            applied[i] = true;
            floor = e.a;
            continue;
        }
        const Pos new_end = RawReplace(e.a, e.b, e.text);
        for (auto& s : selections_) {
            s.anchor = MapPos(s.anchor, e.a, e.b, new_end);
            s.head = MapPos(s.head, e.a, e.b, new_end);
        }
        for (size_t j = 0; j < carets.size(); ++j)
            if (applied[j]) carets[j] = MapPos(carets[j], e.a, e.b, new_end);
        const int offset = caret_offsets ? (*caret_offsets)[i] : static_cast<int>(e.text.size());
        carets[i] = EndOf(e.a, std::string_view(e.text).substr(0, static_cast<size_t>(offset)));
        applied[i] = true;
        group.changes.push_back(std::move(change));
        floor = e.a;
    }

    if (rule == CaretRule::EndOfInsert) {
        std::vector<Selection> next;
        for (size_t i = 0; i < edits.size(); ++i)
            if (applied[i]) next.push_back(Selection{carets[i], carets[i], -1});
        if (!next.empty()) {
            const int primary = std::min(primary_, static_cast<int>(next.size()) - 1);
            selections_ = std::move(next);
            primary_ = primary;
        }
    }
    Normalize();
    if (group.changes.empty()) return;

    ++version_;
    redo_.clear();
    group.after = selections_;
    group.primary_after = primary_;
    if (merge_open_ && kind != GroupKind::Other && !undo_.empty() && undo_.back().kind == kind) {
        Group& top = undo_.back();
        for (auto& c : group.changes) top.changes.push_back(std::move(c));
        top.after = group.after;
        top.primary_after = group.primary_after;
    } else {
        group.id = next_group_id_++;
        undo_.push_back(std::move(group));
    }
    merge_open_ = kind != GroupKind::Other;
}

// ---------------------------------------------------------------- commands

void Document::Type(std::string_view text) {
    if (text.empty()) return;
    if (text.find('\n') != std::string_view::npos || text.find('\r') != std::string_view::npos) {
        Paste(text);
        return;
    }
    const bool single = text.size() == 1;
    const char ch = single ? text[0] : 0;

    if (options_.auto_close && single) {
        // Typing a closer that is already next to every cursor steps over it.
        const bool closer_or_quote = IsCloser(ch) || ch == '"' || ch == '\'';
        if (closer_or_quote) {
            bool over = true;
            for (const auto& s : selections_) {
                const std::string& line = Line(s.head.line);
                if (!s.Empty() || s.head.col >= static_cast<int>(line.size()) || line[static_cast<size_t>(s.head.col)] != ch) {
                    over = false;
                    break;
                }
            }
            if (over) {
                for (auto& s : selections_) s.anchor = s.head = Pos{s.head.line, s.head.col + 1};
                Normalize();
                return;
            }
        }
    }

    std::vector<Edit> edits;
    std::vector<int> carets;
    bool typing = true;
    for (const auto& s : selections_) {
        const std::string& line = Line(s.Start().line);
        const char next = s.End().col < static_cast<int>(Line(s.End().line).size()) ? Line(s.End().line)[static_cast<size_t>(s.End().col)] : 0;
        const char prev = s.Start().col > 0 ? line[static_cast<size_t>(s.Start().col - 1)] : 0;
        const char close = options_.auto_close && single ? Closer(ch) : 0;
        if (close && !s.Empty()) {
            // Wrap the selection.
            const std::string inner = GetRange(s.Start(), s.End());
            edits.push_back({s.Start(), s.End(), std::string(1, ch) + inner + close});
            carets.push_back(static_cast<int>(inner.size()) + 1);
            typing = false;
            continue;
        }
        const bool next_free = next == 0 || next == ' ' || next == '\t' || IsCloser(next) || next == ',' || next == ':';
        const bool quote = ch == '"' || ch == '\'';
        const bool pair = close && next_free && (!quote || (ClassOf(prev) != CharClass::Word && prev != ch));
        if (pair) {
            edits.push_back({s.Start(), s.End(), std::string(1, ch) + close});
            carets.push_back(1);
            continue;
        }
        edits.push_back({s.Start(), s.End(), std::string(text)});
        carets.push_back(static_cast<int>(text.size()));
    }
    // A space after a word closes the typing step, as in other editors.
    if (single && (ch == ' ' || ch == '\t')) merge_open_ = false;
    ApplyEdits(std::move(edits), typing ? GroupKind::Typing : GroupKind::Other, CaretRule::EndOfInsert, &carets);
}

void Document::Paste(std::string_view text) {
    std::string clean;
    clean.reserve(text.size());
    for (size_t i = 0; i < text.size(); ++i) {
        if (text[i] == '\r') {
            if (i + 1 < text.size() && text[i + 1] == '\n') continue;
            clean += '\n';
        } else {
            clean += text[i];
        }
    }
    std::vector<std::string> parts = SplitLines(clean);
    if (parts.size() > 1 && parts.back().empty()) parts.pop_back();
    const bool distribute = selections_.size() > 1 && parts.size() == selections_.size();
    std::vector<Edit> edits;
    for (size_t i = 0; i < selections_.size(); ++i) {
        const auto& s = selections_[i];
        edits.push_back({s.Start(), s.End(), distribute ? parts[i] : clean});
    }
    merge_open_ = false;
    ApplyEdits(std::move(edits), GroupKind::Other, CaretRule::EndOfInsert);
}

void Document::Newline() {
    std::vector<Edit> edits;
    std::vector<int> carets;
    for (const auto& s : selections_) {
        const std::string& line = Line(s.Start().line);
        const int col = s.Start().col;
        std::string indent = options_.auto_indent ? line.substr(0, static_cast<size_t>(std::min(LeadingSpaces(line), col))) : "";
        std::string before = line.substr(0, static_cast<size_t>(col));
        while (!before.empty() && (before.back() == ' ' || before.back() == '\t')) before.pop_back();
        const char last = before.empty() ? 0 : before.back();
        const std::string& end_line = Line(s.End().line);
        const char next = s.End().col < static_cast<int>(end_line.size()) ? end_line[static_cast<size_t>(s.End().col)] : 0;
        const bool opens = options_.auto_indent && (last == ':' || last == '(' || last == '[' || last == '{');
        if (opens && Closer(last) != 0 && next == Closer(last)) {
            // Between a pair: the closer goes to its own line.
            const std::string inner = indent + IndentUnit();
            edits.push_back({s.Start(), s.End(), "\n" + inner + "\n" + indent});
            carets.push_back(1 + static_cast<int>(inner.size()));
            continue;
        }
        if (opens) indent += IndentUnit();
        edits.push_back({s.Start(), s.End(), "\n" + indent});
        carets.push_back(1 + static_cast<int>(indent.size()));
    }
    merge_open_ = false;
    ApplyEdits(std::move(edits), GroupKind::Other, CaretRule::EndOfInsert, &carets);
}

void Document::Backspace() {
    std::vector<Edit> edits;
    for (const auto& s : selections_) {
        if (!s.Empty()) {
            edits.push_back({s.Start(), s.End(), ""});
            continue;
        }
        const Pos p = s.head;
        if (p.col == 0) {
            // At the very start there is nothing to delete; the cursor stays.
            edits.push_back(p.line > 0 ? Edit{PrevChar(p), p, ""} : Edit{p, p, ""});
            continue;
        }
        const std::string& line = Line(p.line);
        const std::string before = line.substr(0, static_cast<size_t>(p.col));
        if (before.find_first_not_of(' ') == std::string::npos) {
            // In the indentation: back to the previous tab stop.
            const int n = (p.col - 1) % options_.tab_size + 1;
            edits.push_back({{p.line, p.col - n}, p, ""});
            continue;
        }
        const char prev = line[static_cast<size_t>(p.col - 1)];
        const char next = p.col < static_cast<int>(line.size()) ? line[static_cast<size_t>(p.col)] : 0;
        if (options_.auto_close && Closer(prev) != 0 && next == Closer(prev)) {
            edits.push_back({{p.line, p.col - 1}, {p.line, p.col + 1}, ""});
            continue;
        }
        edits.push_back({PrevChar(p), p, ""});
    }
    ApplyEdits(std::move(edits), GroupKind::Typing, CaretRule::EndOfInsert);
}

void Document::DeleteForward() {
    std::vector<Edit> edits;
    for (const auto& s : selections_) {
        if (!s.Empty()) edits.push_back({s.Start(), s.End(), ""});
        else edits.push_back({s.head, NextChar(s.head), ""});
    }
    merge_open_ = false;
    ApplyEdits(std::move(edits), GroupKind::Other, CaretRule::EndOfInsert);
}

void Document::DeleteWordLeft() {
    std::vector<Edit> edits;
    for (const auto& s : selections_) edits.push_back(s.Empty() ? Edit{WordLeft(s.head), s.head, ""} : Edit{s.Start(), s.End(), ""});
    merge_open_ = false;
    ApplyEdits(std::move(edits), GroupKind::Other, CaretRule::EndOfInsert);
}

void Document::DeleteWordRight() {
    std::vector<Edit> edits;
    for (const auto& s : selections_) edits.push_back(s.Empty() ? Edit{s.head, WordRight(s.head), ""} : Edit{s.Start(), s.End(), ""});
    merge_open_ = false;
    ApplyEdits(std::move(edits), GroupKind::Other, CaretRule::EndOfInsert);
}

std::vector<std::pair<int, int>> Document::LineBlocks() const {
    std::vector<std::pair<int, int>> blocks;
    for (const auto& s : selections_) {
        int first = s.Start().line;
        int last = s.End().line;
        if (last > first && s.End().col == 0) --last;  // a selection ending at column 0 leaves that line alone
        if (!blocks.empty() && first <= blocks.back().second) {
            blocks.back().second = std::max(blocks.back().second, last);
        } else {
            blocks.emplace_back(first, last);
        }
    }
    return blocks;
}

void Document::Tab() {
    bool multi_line = false;
    for (const auto& s : selections_) multi_line = multi_line || s.Start().line != s.End().line;
    if (multi_line) {
        IndentLines();
        return;
    }
    std::vector<Edit> edits;
    for (const auto& s : selections_) {
        const int vcol = VisualColumn(Line(s.Start().line), s.Start().col, options_.tab_size);
        const int n = options_.tab_size - vcol % options_.tab_size;
        edits.push_back({s.Start(), s.End(), std::string(static_cast<size_t>(n), ' ')});
    }
    merge_open_ = false;
    ApplyEdits(std::move(edits), GroupKind::Other, CaretRule::EndOfInsert);
}

void Document::IndentLines() {
    std::vector<Edit> edits;
    for (const auto& [first, last] : LineBlocks())
        for (int l = first; l <= last; ++l)
            if (!IsBlank(Line(l))) edits.push_back({{l, 0}, {l, 0}, IndentUnit()});
    merge_open_ = false;
    ApplyEdits(std::move(edits), GroupKind::Other, CaretRule::KeepMapped);
}

void Document::Outdent() {
    std::vector<Edit> edits;
    for (const auto& [first, last] : LineBlocks()) {
        for (int l = first; l <= last; ++l) {
            const std::string& line = Line(l);
            int n = 0;
            if (!line.empty() && line[0] == '\t') n = 1;
            else
                while (n < options_.tab_size && n < static_cast<int>(line.size()) && line[static_cast<size_t>(n)] == ' ') ++n;
            if (n > 0) edits.push_back({{l, 0}, {l, n}, ""});
        }
    }
    merge_open_ = false;
    ApplyEdits(std::move(edits), GroupKind::Other, CaretRule::KeepMapped);
}

void Document::ToggleLineComment(std::string_view prefix) {
    const auto blocks = LineBlocks();
    bool all_commented = true;
    int min_indent = 1 << 30;
    for (const auto& [first, last] : blocks) {
        for (int l = first; l <= last; ++l) {
            const std::string& line = Line(l);
            if (IsBlank(line)) continue;
            const int indent = LeadingSpaces(line);
            min_indent = std::min(min_indent, indent);
            if (line.compare(static_cast<size_t>(indent), prefix.size(), prefix) != 0) all_commented = false;
        }
    }
    if (min_indent == (1 << 30)) return;  // only blank lines
    std::vector<Edit> edits;
    for (const auto& [first, last] : blocks) {
        for (int l = first; l <= last; ++l) {
            const std::string& line = Line(l);
            if (IsBlank(line)) continue;
            if (all_commented) {
                const int at = LeadingSpaces(line);
                int n = static_cast<int>(prefix.size());
                if (at + n < static_cast<int>(line.size()) && line[static_cast<size_t>(at + n)] == ' ') ++n;
                edits.push_back({{l, at}, {l, at + n}, ""});
            } else {
                edits.push_back({{l, min_indent}, {l, min_indent}, std::string(prefix) + " "});
            }
        }
    }
    merge_open_ = false;
    ApplyEdits(std::move(edits), GroupKind::Other, CaretRule::KeepMapped);
}

void Document::DuplicateLines() {
    // The copy goes above, so the cursors (mapped past it) end up on the copy below.
    std::vector<Edit> edits;
    for (const auto& [first, last] : LineBlocks()) edits.push_back({{first, 0}, {first, 0}, GetRange({first, 0}, {last, static_cast<int>(Line(last).size())}) + "\n"});
    merge_open_ = false;
    ApplyEdits(std::move(edits), GroupKind::Other, CaretRule::KeepMapped);
}

void Document::MoveLines(int direction) {
    if (direction == 0) return;
    const auto blocks = LineBlocks();
    for (const auto& [first, last] : blocks) {
        if (direction < 0 && first == 0) return;
        if (direction > 0 && last == LineCount() - 1) return;
    }
    std::vector<Selection> moved = selections_;
    std::vector<Edit> edits;
    for (const auto& [first, last] : blocks) {
        const std::string block = GetRange({first, 0}, {last, static_cast<int>(Line(last).size())});
        if (direction < 0) {
            const std::string& above = Line(first - 1);
            edits.push_back({{first - 1, 0}, {last, static_cast<int>(Line(last).size())}, block + "\n" + above});
        } else {
            const std::string& below = Line(last + 1);
            edits.push_back({{first, 0}, {last + 1, static_cast<int>(below.size())}, below + "\n" + block});
        }
    }
    for (auto& s : moved) {
        s.anchor.line += direction;
        s.head.line += direction;
    }
    merge_open_ = false;
    const int primary = primary_;
    ApplyEdits(std::move(edits), GroupKind::Other, CaretRule::KeepMapped);
    // Same columns, one line further: the moved text keeps its cursors.
    selections_ = std::move(moved);
    primary_ = primary;
    Normalize();
    if (!undo_.empty()) {
        undo_.back().after = selections_;
        undo_.back().primary_after = primary_;
    }
}

void Document::DeleteLines() {
    std::vector<Edit> edits;
    for (const auto& [first, last] : LineBlocks()) {
        if (last + 1 < LineCount()) edits.push_back({{first, 0}, {last + 1, 0}, ""});
        else if (first > 0) edits.push_back({{first - 1, static_cast<int>(Line(first - 1).size())}, {last, static_cast<int>(Line(last).size())}, ""});
        else edits.push_back({{0, 0}, {last, static_cast<int>(Line(last).size())}, ""});
    }
    merge_open_ = false;
    ApplyEdits(std::move(edits), GroupKind::Other, CaretRule::EndOfInsert);
}

void Document::Replace(Pos a, Pos b, std::string_view text) {
    merge_open_ = false;
    ApplyEdits({Edit{a, b, std::string(text)}}, GroupKind::Other, CaretRule::KeepMapped);
}

void Document::TransformSelections(const std::function<std::string(const std::string&)>& transform) {
    std::vector<Edit> edits;
    std::vector<Selection> after;
    for (const auto& s : selections_) {
        if (s.Empty()) continue;
        edits.push_back({s.Start(), s.End(), transform(GetRange(s.Start(), s.End()))});
    }
    if (edits.empty()) return;
    merge_open_ = false;
    std::vector<int> zero(edits.size(), 0);
    // Land each cursor at its edit's start, then select the new text.
    const std::vector<Edit> copy = edits;
    ApplyEdits(std::move(edits), GroupKind::Other, CaretRule::EndOfInsert, &zero);
    std::vector<Selection> selected;
    for (size_t i = 0; i < selections_.size() && i < copy.size(); ++i) {
        const Pos a = selections_[i].head;
        selected.push_back(Selection{a, EndOf(a, copy[i].text), -1});
    }
    selections_ = std::move(selected);
    Normalize();
    if (!undo_.empty()) undo_.back().after = selections_;
}

// ---------------------------------------------------------------- undo

bool Document::Undo() {
    if (undo_.empty()) return false;
    Group g = std::move(undo_.back());
    undo_.pop_back();
    for (auto it = g.changes.rbegin(); it != g.changes.rend(); ++it)
        RawReplace(it->start, EndOf(it->start, it->inserted), it->removed);
    selections_ = g.before;
    primary_ = g.primary_before;
    Normalize();
    redo_.push_back(std::move(g));
    merge_open_ = false;
    ++version_;
    return true;
}

bool Document::Redo() {
    if (redo_.empty()) return false;
    Group g = std::move(redo_.back());
    redo_.pop_back();
    for (const auto& c : g.changes) RawReplace(c.start, EndOf(c.start, c.removed), c.inserted);
    selections_ = g.after;
    primary_ = g.primary_after;
    Normalize();
    undo_.push_back(std::move(g));
    merge_open_ = false;
    ++version_;
    return true;
}

bool Document::Modified() const {
    const uint64_t top = undo_.empty() ? 0 : undo_.back().id;
    return top != saved_group_id_;
}

void Document::MarkSaved() {
    saved_group_id_ = undo_.empty() ? 0 : undo_.back().id;
    merge_open_ = false;  // later typing is a new step, so undoing it returns to "saved"
}

}  // namespace cyxwiz::editor
