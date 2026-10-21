// Tables from HTML outputs (TOFIX133 P4 step 4.3c): pandas _repr_html_ of
// the sentiment dataset, a shortened frame, an index name row, entities.
#include "../src/core/html_table.h"

#include <cstdlib>
#include <iostream>
#include <string>

using namespace cyxwiz::html;

namespace {
void Check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        std::exit(1);
    }
}

// df.head(3)._repr_html_() for sentiment_mental_health.csv (pandas 2.x).
const char* kHead =
    "<div>\n<style scoped>\n    .dataframe tbody tr th:only-of-type {\n        vertical-align: middle;\n    }\n</style>\n"
    "<table border=\"1\" class=\"dataframe\">\n  <thead>\n    <tr style=\"text-align: right;\">\n      <th></th>\n"
    "      <th>statement</th>\n      <th>status</th>\n    </tr>\n  </thead>\n  <tbody>\n    <tr>\n      <th>0</th>\n"
    "      <td>oh my gosh</td>\n      <td>Anxiety</td>\n    </tr>\n    <tr>\n      <th>1</th>\n"
    "      <td>trouble sleeping, confused mind, restless hear...</td>\n      <td>Anxiety</td>\n    </tr>\n"
    "    <tr>\n      <th>2</th>\n      <td>All wrong, back off dear, forward doubt. Stay ...</td>\n      <td>Anxiety</td>\n"
    "    </tr>\n  </tbody>\n</table>\n</div>";

// The whole frame: pandas shortens it and prints the shape.
const char* kLong =
    "<table border=\"1\" class=\"dataframe\"><thead><tr><th></th><th>statement</th><th>status</th></tr>"
    "<tr><th>id</th><th></th><th></th></tr></thead><tbody>"
    "<tr><th>0</th><td>oh my gosh</td><td>Anxiety</td></tr>"
    "<tr><th>...</th><td>...</td><td>...</td></tr>"
    "<tr><th>53042</th><td>I have really bad door anxiety! It&#x27;s not about...</td><td>Anxiety</td></tr>"
    "</tbody></table>\n<p>53043 rows \xC3\x97 2 columns</p>\n</div>";
}  // namespace

int main() {
    Table t;
    Check(ParseTable(kHead, t), "head parses");
    Check(t.columns.size() == 3 && t.columns[0].empty() && t.columns[1] == "statement" && t.columns[2] == "status",
          "header row with the index column first");
    Check(t.rows.size() == 3 && t.has_index && t.rows[0][0] == "0" && t.rows[0][1] == "oh my gosh" && t.rows[2][2] == "Anxiety",
          "body rows");
    Check(!t.truncated && t.footer.empty(), "a head is complete");

    Check(ParseTable(kLong, t), "long frame parses");
    Check(t.columns[0] == "id", "index name from pandas' second header row");
    Check(t.rows.size() == 2 && t.truncated, "the ... row marks a shortened table");
    Check(t.rows[1][1] == "I have really bad door anxiety! It's not about...", "entities decoded");
    Check(t.footer == "53043 rows \xC3\x97 2 columns", "pandas shape footer");

    Check(ParseTable("<table><tr><td>a &amp; b</td><td>&lt;1&gt;</td></tr></table>", t) && t.rows[0][0] == "a & b" &&
              t.rows[0][1] == "<1>" && !t.has_index && t.columns.empty(),
          "plain table without header");
    Check(!ParseTable("<p>no table</p>", t), "no table");
    Check(DecodeEntities("&#233;t&eacute;") == "\xC3\xA9t&eacute;", "numeric entity; unknown names kept");

    std::cout << "html table: pandas head, shortened frame, index name, entities. OK\n";
    return 0;
}
