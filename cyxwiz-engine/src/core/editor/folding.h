// Folding for the Script Editor (TOFIX133 P1): where blocks can fold
// (indentation, multi-line strings, %% / # %% sections) and which are
// folded, kept in step with edits. No ImGui.
#pragma once

#include "text_document.h"

#include <set>
#include <vector>

namespace cyxwiz::editor {

struct FoldRegion {
    int start = 0;  // header line, stays visible
    int end = 0;    // last hidden line
};

// Regions sorted by start; one per start line (the largest).
std::vector<FoldRegion> FindFoldRegions(const Document& document);

class FoldState {
public:
    // Recomputes regions and moves folded headers with the line edits since
    // the last call; folds whose header disappeared are dropped.
    void Update(Document& document);
    const std::vector<FoldRegion>& Regions() const { return regions_; }

    bool CanFold(int line) const;
    bool IsFolded(int line) const { return folded_.count(line) != 0; }
    void Toggle(int line);
    void Fold(int line);
    void Unfold(int line);
    void FoldAll();
    void UnfoldAll() { folded_.clear(); }
    // Opens every fold that hides `line` (a cursor or a search hit went there).
    void Reveal(int line);

    bool IsHidden(int line) const;
    // Lines shown, in order (headers of folded regions included).
    std::vector<int> VisibleLines(int line_count) const;
    // Lines a folded header hides (for the "··· N lines" label), 0 if not folded.
    int HiddenCount(int line) const;

private:
    const FoldRegion* RegionAt(int line) const;
    std::vector<FoldRegion> regions_;
    std::set<int> folded_;
};

}  // namespace cyxwiz::editor
