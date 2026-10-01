#include "cyx_format.h"

#include <cctype>
#include <sstream>

namespace cyxwiz::cyx {

namespace {
constexpr const char* kHeader1 = "# CyxWiz Script v0.3.0";
constexpr const char* kHeader2 = "# Cell markers: %%code, %%markdown, %%raw";

std::string Trim(const std::string& s) {
    const size_t b = s.find_first_not_of(" \t\r");
    if (b == std::string::npos) return {};
    const size_t e = s.find_last_not_of(" \t\r");
    return s.substr(b, e - b + 1);
}

bool IsMarker(const std::string& trimmed) { return trimmed.rfind("%%", 0) == 0; }

CellKind KindOf(const std::string& marker) {
    // "%%markdown", "%% markdown" and "%%Markdown" are all markdown.
    std::string word = Trim(marker.substr(2));
    for (char& c : word) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    if (word == "markdown" || word == "md") return CellKind::Markdown;
    if (word == "raw") return CellKind::Raw;
    return CellKind::Code;
}

const char* MarkerOf(CellKind kind) {
    switch (kind) {
        case CellKind::Markdown: return "%%markdown";
        case CellKind::Raw: return "%%raw";
        case CellKind::Code: break;
    }
    return "%%code";
}

void StripTrailingBlankLines(std::string& s) {
    while (!s.empty()) {
        const size_t nl = s.find_last_of('\n');
        const std::string last = nl == std::string::npos ? s : s.substr(nl + 1);
        if (!Trim(last).empty()) break;
        if (nl == std::string::npos) {
            s.clear();
            break;
        }
        s.erase(nl);
    }
}
}  // namespace

bool HasCellMarkers(const std::string& content) {
    std::istringstream in(content);
    std::string line;
    while (std::getline(in, line))
        if (IsMarker(Trim(line))) return true;
    return false;
}

std::vector<Cell> Parse(const std::string& content) {
    std::vector<Cell> cells;
    std::istringstream in(content);
    std::string line;
    bool in_cell = false;       // after a marker
    std::string preamble;       // text before the first marker
    Cell current;
    auto flush = [&]() {
        StripTrailingBlankLines(current.source);
        cells.push_back(current);
        current = Cell{};
    };
    while (std::getline(in, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        const std::string trimmed = Trim(line);
        if (IsMarker(trimmed)) {
            if (in_cell) {
                flush();
            } else {
                StripTrailingBlankLines(preamble);
                const size_t first = preamble.find_first_not_of(" \t\n");
                if (first != std::string::npos) cells.push_back({CellKind::Code, preamble.substr(first)});
            }
            current.kind = KindOf(trimmed);
            in_cell = true;
            continue;
        }
        if (!in_cell) {
            if (trimmed == kHeader1 || trimmed == kHeader2) continue;  // written by Serialize
            preamble += line + "\n";
            continue;
        }
        current.source += line + "\n";
    }
    if (in_cell) {
        flush();
    } else {
        StripTrailingBlankLines(preamble);
        cells.push_back({CellKind::Code, preamble});
    }
    return cells;
}

std::string Serialize(const std::vector<Cell>& cells) {
    std::string out;
    out += kHeader1;
    out += "\n";
    out += kHeader2;
    out += "\n\n";
    for (const auto& cell : cells) {
        std::string source = cell.source;
        StripTrailingBlankLines(source);
        out += MarkerOf(cell.kind);
        out += "\n";
        if (!source.empty()) {
            out += source;
            out += "\n";
        }
        out += "\n";
    }
    return out;
}

}  // namespace cyxwiz::cyx
