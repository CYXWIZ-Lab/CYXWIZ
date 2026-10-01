#include "html_table.h"

#include <cctype>
#include <cstdlib>

namespace cyxwiz::html {

namespace {
std::string Lower(const std::string& s) {
    std::string out = s;
    for (auto& c : out) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return out;
}

// Text of an element's inner HTML: tags dropped, whitespace collapsed.
std::string InnerText(const std::string& inner) {
    std::string text;
    bool in_tag = false;
    for (char c : inner) {
        if (c == '<') in_tag = true;
        else if (c == '>') in_tag = false;
        else if (!in_tag) text += c;
    }
    std::string collapsed;
    bool space = false;
    for (char c : text) {
        if (c == ' ' || c == '\n' || c == '\r' || c == '\t') {
            space = !collapsed.empty();
        } else {
            if (space) collapsed += ' ';
            collapsed += c;
            space = false;
        }
    }
    return DecodeEntities(collapsed);
}

void AppendUtf8(std::string& out, unsigned long cp) {
    if (cp < 0x80) out += static_cast<char>(cp);
    else if (cp < 0x800) {
        out += static_cast<char>(0xC0 | (cp >> 6));
        out += static_cast<char>(0x80 | (cp & 0x3F));
    } else if (cp < 0x10000) {
        out += static_cast<char>(0xE0 | (cp >> 12));
        out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
        out += static_cast<char>(0x80 | (cp & 0x3F));
    } else {
        out += static_cast<char>(0xF0 | (cp >> 18));
        out += static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
        out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
        out += static_cast<char>(0x80 | (cp & 0x3F));
    }
}

struct Cell {
    bool header = false;
    std::string text;
};

// Cells of one <tr>...</tr>.
std::vector<Cell> RowCells(const std::string& row, const std::string& lower) {
    std::vector<Cell> cells;
    size_t pos = 0;
    while (true) {
        const size_t th = lower.find("<th", pos);
        const size_t td = lower.find("<td", pos);
        const size_t at = std::min(th, td);
        if (at == std::string::npos) break;
        const bool header = at == th;
        const size_t open_end = lower.find('>', at);
        if (open_end == std::string::npos) break;
        const std::string close = header ? "</th>" : "</td>";
        size_t end = lower.find(close, open_end);
        if (end == std::string::npos) end = row.size();
        cells.push_back({header, InnerText(row.substr(open_end + 1, end - open_end - 1))});
        pos = end;
    }
    return cells;
}
}  // namespace

std::string DecodeEntities(const std::string& text) {
    std::string out;
    for (size_t i = 0; i < text.size(); ++i) {
        if (text[i] != '&') {
            out += text[i];
            continue;
        }
        const size_t semi = text.find(';', i);
        if (semi == std::string::npos || semi - i > 10) {
            out += '&';
            continue;
        }
        const std::string name = text.substr(i + 1, semi - i - 1);
        if (name == "amp") out += '&';
        else if (name == "lt") out += '<';
        else if (name == "gt") out += '>';
        else if (name == "quot") out += '"';
        else if (name == "apos" || name == "#39") out += '\'';
        else if (name == "nbsp") out += ' ';
        else if (name == "times") out += "\xC3\x97";
        else if (!name.empty() && name[0] == '#') {
            const bool hex = name.size() > 1 && (name[1] == 'x' || name[1] == 'X');
            AppendUtf8(out, std::strtoul(name.c_str() + (hex ? 2 : 1), nullptr, hex ? 16 : 10));
        } else {
            out += text.substr(i, semi - i + 1);
        }
        i = semi;
    }
    return out;
}

bool ParseTable(const std::string& html_text, Table& out) {
    out = Table{};
    const std::string lower = Lower(html_text);
    const size_t start = lower.find("<table");
    if (start == std::string::npos) return false;
    size_t end = lower.find("</table>", start);
    if (end == std::string::npos) end = lower.size();
    const size_t thead = lower.find("<thead", start);
    const size_t tbody = lower.find("<tbody", start);

    size_t pos = start;
    bool first_header_row = true;
    while (true) {
        const size_t tr = lower.find("<tr", pos);
        if (tr == std::string::npos || tr > end) break;
        size_t tr_end = lower.find("</tr>", tr);
        if (tr_end == std::string::npos || tr_end > end) tr_end = end;
        const std::vector<Cell> cells = RowCells(html_text.substr(tr, tr_end - tr), lower.substr(tr, tr_end - tr));
        pos = tr_end;
        if (cells.empty()) continue;
        const bool in_head = thead != std::string::npos && tr > thead && (tbody == std::string::npos || tr < tbody);
        const bool all_header = [&] {
            for (const auto& c : cells)
                if (!c.header) return false;
            return true;
        }();
        if (in_head || (out.rows.empty() && out.columns.empty() && all_header)) {
            if (first_header_row) {
                for (const auto& c : cells) out.columns.push_back(c.text);
                first_header_row = false;
            } else if (!cells.empty() && !cells[0].text.empty() && !out.columns.empty() && out.columns[0].empty()) {
                out.columns[0] = cells[0].text;  // pandas puts the index name on its own header row
            }
            continue;
        }
        std::vector<std::string> row;
        bool dots = true;
        for (const auto& c : cells) {
            row.push_back(c.text);
            dots = dots && (c.text == "..." || c.text == "\xE2\x80\xA6");
        }
        if (dots) {
            out.truncated = true;
            continue;
        }
        if (!cells.empty() && cells[0].header) out.has_index = true;
        out.rows.push_back(std::move(row));
    }
    // pandas' "<p>N rows × M columns</p>" right after the table.
    const size_t p = lower.find("<p>", end);
    if (p != std::string::npos && p < end + 40) {
        const size_t p_end = lower.find("</p>", p);
        if (p_end != std::string::npos) out.footer = InnerText(html_text.substr(p + 3, p_end - p - 3));
    }
    return !out.columns.empty() || !out.rows.empty();
}

}  // namespace cyxwiz::html
