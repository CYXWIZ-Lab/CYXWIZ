#pragma once

// Text as the interface draws it (TOFIX129). ImGui draws one glyph per
// character, left to right, with no shaping, so Arabic came out as separate
// letters in reverse order. DisplayText gives Arabic letters their joined
// forms (isolated, initial, medial, final and the lam-alef ligatures) and puts
// right-to-left runs in the order they are seen (the Unicode bidirectional
// algorithm for one line: no explicit embeddings, numbers kept left to right,
// brackets mirrored). For drawing only: keep the original text for matching,
// filters and saving.

#include <string>
#include <string_view>

namespace cyxwiz::core {

// True when the text has a right-to-left letter (Hebrew, Arabic and its
// presentation forms).
bool HasRightToLeft(std::string_view utf8);

// The text to draw. Text with no right-to-left letter comes back unchanged.
std::string DisplayText(std::string_view utf8);

}  // namespace cyxwiz::core
