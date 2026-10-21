#include "table_grid_rows.h"

#include <algorithm>
#include <cctype>
#include <charconv>
#include <cmath>
#include <cstdio>
#include <limits>
#include <numeric>

namespace cyxwiz {

namespace {

std::string Folded(std::string text) {
    for (char& ch : text) ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    return text;
}

// Progress is checked every this many cells read.
constexpr size_t kCheckCells = 1 << 16;

}  // namespace

std::string TableCellText(const DataTable& table, size_t row, size_t col) {
    const DataTable::CellValue v = table.GetCell(row, col);
    if (const auto* s = std::get_if<std::string>(&v)) return *s;
    if (const auto* n = std::get_if<int64_t>(&v)) return std::to_string(*n);
    if (const auto* d = std::get_if<double>(&v)) {
        char buf[32];
        const auto r = std::to_chars(buf, buf + sizeof(buf), *d);
        if (r.ptr - buf <= 12) return std::string(buf, r.ptr);
        std::snprintf(buf, sizeof(buf), "%.10g", *d);
        return buf;
    }
    return "";
}

TableColumnKind TableColumnKindOf(const DataTable& table, size_t col) {
    bool numeric = false;
    double min_v = std::numeric_limits<double>::max();
    double max_v = std::numeric_limits<double>::lowest();
    for (size_t r = 0; r < table.GetRowCount(); r++) {
        const auto cell = table.GetCell(r, col);
        double v = 0.0;
        if (const auto* d = std::get_if<double>(&cell)) v = *d;
        else if (const auto* n = std::get_if<int64_t>(&cell)) v = static_cast<double>(*n);
        else continue;
        if (std::isnan(v) || std::isinf(v)) continue;
        min_v = std::min(min_v, v);
        max_v = std::max(max_v, v);
        numeric = true;
    }
    TableColumnKind kind;
    kind.type = numeric ? "Numeric" : "Text";
    if (numeric) {
        kind.min_val = min_v;
        kind.max_val = max_v;
    }
    return kind;
}

std::vector<TableColumnKind> TableColumnKindsOf(const DataTable& table) {
    std::vector<TableColumnKind> kinds(table.GetColumnCount());
    for (size_t c = 0; c < kinds.size(); c++) kinds[c] = TableColumnKindOf(table, c);
    return kinds;
}

TableFilter::TableFilter(std::string text, bool case_sensitive, bool whole_cell)
    : text_(std::move(text)), case_sensitive_(case_sensitive), whole_cell_(whole_cell) {
    if (!case_sensitive_) folded_ = Folded(text_);
}

bool TableFilter::Matches(const std::string& cell) const {
    if (text_.empty()) return false;
    if (case_sensitive_) return whole_cell_ ? cell == text_ : cell.find(text_) != std::string::npos;
    const std::string folded = Folded(cell);
    return whole_cell_ ? folded == folded_ : folded.find(folded_) != std::string::npos;
}

TableRowOrder ComputeTableRowOrder(const DataTable& table, const TableRowRequest& request,
                                   const TableRowProgress& progress) {
    TableRowOrder out;
    const size_t rows = table.GetRowCount();
    const size_t cols = table.GetColumnCount();
    const bool filtering = !request.filter.empty() && (request.filter_rows || request.count_matches);
    // The share of the work the sort takes, for the progress.
    const float sort_share = request.sort_column >= 0 && request.order.empty() ? (filtering ? 0.3f : 1.0f) : 0.0f;
    size_t read = 0;
    const auto keep_going = [&](float fraction) {
        if (++read % kCheckCells != 0 || !progress) return true;
        if (progress(fraction)) return true;
        out.stopped = true;
        return false;
    };

    // The order.
    if (!request.order.empty()) {
        out.sorted = request.order;
    } else if (request.sort_column < 0 || static_cast<size_t>(request.sort_column) >= cols) {
        out.sorted.resize(rows);
        std::iota(out.sorted.begin(), out.sorted.end(), size_t(0));
        if (!request.ascending) std::reverse(out.sorted.begin(), out.sorted.end());
    } else {
        const size_t col = static_cast<size_t>(request.sort_column);
        const bool ascending = request.ascending;
        out.sorted.resize(rows);
        std::iota(out.sorted.begin(), out.sorted.end(), size_t(0));
        // Each cell is read once into a key, then the rows sort on the keys.
        if (request.numeric) {
            const double missing = ascending ? std::numeric_limits<double>::max() : std::numeric_limits<double>::lowest();
            std::vector<double> keys(rows, missing);
            for (size_t r = 0; r < rows; r++) {
                const auto cell = table.GetCell(r, col);
                if (const auto* d = std::get_if<double>(&cell)) keys[r] = *d;
                else if (const auto* n = std::get_if<int64_t>(&cell)) keys[r] = static_cast<double>(*n);
                if (!keep_going(sort_share * 0.5f * static_cast<float>(r) / static_cast<float>(rows))) return out;
            }
            std::stable_sort(out.sorted.begin(), out.sorted.end(), [&](size_t a, size_t b) {
                return ascending ? keys[a] < keys[b] : keys[a] > keys[b];
            });
        } else {
            std::vector<std::string> keys(rows);
            for (size_t r = 0; r < rows; r++) {
                keys[r] = TableCellText(table, r, col);
                if (!keep_going(sort_share * 0.5f * static_cast<float>(r) / static_cast<float>(rows))) return out;
            }
            std::stable_sort(out.sorted.begin(), out.sorted.end(), [&](size_t a, size_t b) {
                return ascending ? keys[a] < keys[b] : keys[a] > keys[b];
            });
        }
    }
    if (progress && !progress(sort_share)) {
        out.stopped = true;
        return out;
    }
    if (!filtering) return out;

    // The rows with a matching cell; when counting, every cell is read.
    if (request.count_matches) out.match_cells = 0;
    const size_t total = out.sorted.size();
    for (size_t k = 0; k < total; k++) {
        const size_t r = out.sorted[k];
        bool row_hit = false;
        for (size_t c = 0; c < cols; c++) {
            if (!keep_going(sort_share + (1.0f - sort_share) * static_cast<float>(k) / static_cast<float>(total)))
                return out;
            if (!request.filter.Matches(TableCellText(table, r, c))) continue;
            row_hit = true;
            if (!request.count_matches) break;
            ++out.match_cells;
        }
        if (!row_hit) continue;
        ++out.match_rows;
        if (request.filter_rows) out.filtered.push_back(r);
    }
    return out;
}

}  // namespace cyxwiz
