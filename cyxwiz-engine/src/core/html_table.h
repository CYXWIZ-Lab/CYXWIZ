// The first <table> of an HTML output (pandas' _repr_html_, Jupyter
// text/html) as rows of text, for the notebook's table output (TOFIX133 P4
// step 4.3c, board 5). No ImGui.
#pragma once

#include <string>
#include <vector>

namespace cyxwiz::html {

struct Table {
    std::vector<std::string> columns;            // header cells; the index column's header first when has_index
    std::vector<std::vector<std::string>> rows;  // body cells, same order
    bool has_index = false;                      // body rows start with a <th> (pandas index)
    bool truncated = false;                      // pandas left rows out ("..." row)
    std::string footer;                          // "53043 rows × 2 columns" when pandas printed it
};

// False when there is no table with at least one row or column.
bool ParseTable(const std::string& html, Table& out);

// &amp; &lt; &gt; &quot; &#39; &nbsp; and numeric entities.
std::string DecodeEntities(const std::string& text);

}  // namespace cyxwiz::html
