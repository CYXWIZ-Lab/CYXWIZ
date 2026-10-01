// CSV records (RFC 4180) for the Table Viewer: quoted commas, line breaks
// and quotes, as pandas writes the sentiment dataset.
#include "../src/core/csv_records.h"

#include <cstdlib>
#include <iostream>
#include <string>

using namespace cyxwiz::csv;

namespace {
void Check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        std::exit(1);
    }
}
}  // namespace

int main() {
    // As pandas to_csv writes statements with commas, quotes and line breaks.
    const std::string text =
        "\xEF\xBB\xBF,statement,status\r\n"
        "0,oh my gosh,Anxiety\r\n"
        "1,\"trouble sleeping, confused mind, restless heart. All out of tune\",Anxiety\r\n"
        "2,\"first line\nsecond line\",Normal\r\n"
        "3,\"she said \"\"hi\"\"\",Stress\r\n"
        "4,,Normal\r\n"
        "\r\n";
    const auto rows = ReadRecords(text);
    Check(rows.size() == 6, "header + five records, blank line skipped");
    Check(rows[0].size() == 3 && rows[0][0].empty() && rows[0][1] == "statement", "header without the BOM");
    Check(rows[2][1] == "trouble sleeping, confused mind, restless heart. All out of tune", "quoted commas");
    Check(rows[3][1] == "first line\nsecond line" && rows[3][2] == "Normal", "quoted line break stays in the field");
    Check(rows[4][1] == "she said \"hi\"", "doubled quotes");
    Check(rows[5].size() == 3 && rows[5][1].empty(), "empty field");
    Check(ReadRecords("a,b\n1,2").size() == 2, "no final newline");

    Check(Field("plain") == "plain" && Field("a,b") == "\"a,b\"" && Field("say \"x\"") == "\"say \"\"x\"\"\"" &&
              Field("two\nlines") == "\"two\nlines\"",
          "fields quoted for writing");
    const auto round = ReadRecords(Field("a,\"b\"\nc") + "," + Field("d") + "\n");
    Check(round.size() == 1 && round[0][0] == "a,\"b\"\nc" && round[0][1] == "d", "write then read");

    long long i = 0;
    double d = 0;
    Check(ParseInt("53043", i) && i == 53043 && !ParseInt("3 rows", i) && !ParseInt("", i) && !ParseInt("1.5", i),
          "whole integers only");
    Check(ParseDouble("2.5e3", d) && d == 2500.0 && !ParseDouble("2.5x", d), "whole numbers only");

    std::cout << "csv records: quoted commas, line breaks, quotes, BOM, numbers. OK\n";
    return 0;
}
