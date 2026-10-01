#include "folding.h"

#include "python_highlight.h"

#include <algorithm>
#include <map>

namespace cyxwiz::editor {

namespace {
int IndentOf(const std::string& line, int tab_size) {
    int col = 0;
    for (const char c : line) {
        if (c == ' ') ++col;
        else if (c == '\t') col = (col / tab_size + 1) * tab_size;
        else return col;
    }
    return -1;  // blank
}

bool IsCellMarker(const std::string& line) {
    const size_t first = line.find_first_not_of(" \t");
    return first != std::string::npos && (line.compare(first, 2, "%%") == 0 || line.compare(first, 4, "# %%") == 0);
}
}  // namespace

std::vector<FoldRegion> FindFoldRegions(const Document& document) {
    const int count = document.LineCount();
    const int tab = document.Settings().tab_size;
    std::map<int, int> best;  // start -> end
    auto add = [&](int start, int end) {
        if (end <= start) return;
        auto it = best.find(start);
        if (it == best.end() || it->second < end) best[start] = end;
    };

    // Multi-line strings: the line that opens one to the line that closes it.
    std::vector<bool> in_string(static_cast<size_t>(count), false);
    {
        LineState state{};
        std::vector<Span> spans;
        int open = -1;
        for (int i = 0; i < count; ++i) {
            if (state.triple) in_string[static_cast<size_t>(i)] = true;
            const LineState out = HighlightLine(document.Line(i), state, spans);
            if (!state.triple && out.triple) open = i;
            if (state.triple && !out.triple && open >= 0) {
                add(open, i);
                open = -1;
            }
            state = out;
        }
    }

    // Indentation: a line followed by deeper lines folds them (blank lines
    // inside count, trailing blank lines do not).
    std::vector<int> indent(static_cast<size_t>(count));
    for (int i = 0; i < count; ++i)
        indent[static_cast<size_t>(i)] = in_string[static_cast<size_t>(i)] ? -2 : IndentOf(document.Line(i), tab);
    for (int i = 0; i < count; ++i) {
        const int base = indent[static_cast<size_t>(i)];
        if (base < 0) continue;
        int last = i;
        for (int j = i + 1; j < count; ++j) {
            const int ind = indent[static_cast<size_t>(j)];
            if (ind == -1) continue;               // blank
            if (ind == -2 || ind > base) last = j;  // string continuation or deeper
            else break;
        }
        add(i, last);
    }

    // Cell sections: a marker to the line before the next marker.
    int marker = -1;
    for (int i = 0; i < count; ++i) {
        if (!IsCellMarker(document.Line(i))) continue;
        if (marker >= 0) {
            int end = i - 1;
            while (end > marker && IndentOf(document.Line(end), tab) == -1) --end;
            add(marker, end);
        }
        marker = i;
    }
    if (marker >= 0) {
        int end = count - 1;
        while (end > marker && IndentOf(document.Line(end), tab) == -1) --end;
        add(marker, end);
    }

    std::vector<FoldRegion> regions;
    for (const auto& [start, end] : best) regions.push_back({start, end});
    return regions;
}

const FoldRegion* FoldState::RegionAt(int line) const {
    const auto it = std::lower_bound(regions_.begin(), regions_.end(), line,
                                     [](const FoldRegion& r, int l) { return r.start < l; });
    return it != regions_.end() && it->start == line ? &*it : nullptr;
}

void FoldState::Update(Document& document) {
    // Move folded headers with the edits.
    for (const LineEdit& e : document.TakeLineEdits()) {
        std::set<int> moved;
        for (const int start : folded_) {
            if (start <= e.line) moved.insert(start);                       // above (or the edited header itself)
            else if (start > e.line + e.removed) moved.insert(start - e.removed + e.added);
            // inside the replaced lines: the fold goes away
        }
        folded_ = std::move(moved);
    }
    regions_ = FindFoldRegions(document);
    for (auto it = folded_.begin(); it != folded_.end();) it = RegionAt(*it) ? std::next(it) : folded_.erase(it);
}

bool FoldState::CanFold(int line) const { return RegionAt(line) != nullptr; }

void FoldState::Toggle(int line) {
    if (IsFolded(line)) Unfold(line);
    else Fold(line);
}

void FoldState::Fold(int line) {
    if (CanFold(line)) folded_.insert(line);
}

void FoldState::Unfold(int line) { folded_.erase(line); }

void FoldState::FoldAll() {
    for (const auto& r : regions_) folded_.insert(r.start);
}

void FoldState::Reveal(int line) {
    for (auto it = folded_.begin(); it != folded_.end();) {
        const FoldRegion* r = RegionAt(*it);
        it = (r && r->start < line && line <= r->end) ? folded_.erase(it) : std::next(it);
    }
}

bool FoldState::IsHidden(int line) const {
    for (const int start : folded_) {
        if (start >= line) break;
        const FoldRegion* r = RegionAt(start);
        if (r && line <= r->end) return true;
    }
    return false;
}

std::vector<int> FoldState::VisibleLines(int line_count) const {
    std::vector<int> out;
    out.reserve(static_cast<size_t>(line_count));
    for (int i = 0; i < line_count; ++i) {
        out.push_back(i);
        if (IsFolded(i)) {
            const FoldRegion* r = RegionAt(i);
            if (r) i = r->end;  // skip the hidden lines (nested folds inside are hidden too)
        }
    }
    return out;
}

int FoldState::HiddenCount(int line) const {
    if (!IsFolded(line)) return 0;
    const FoldRegion* r = RegionAt(line);
    return r ? r->end - r->start : 0;
}

}  // namespace cyxwiz::editor
