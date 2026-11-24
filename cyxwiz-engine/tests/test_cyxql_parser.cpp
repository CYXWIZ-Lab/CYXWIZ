#include "../src/gui/query/cyxql_examples.h"
#include "../src/gui/query/cyxql_parser.h"

#include <iostream>
#include <string>

namespace {

int failures = 0;

void Check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << "\n";
        ++failures;
    }
}

// Parses the query; on a refusal, says where and why.
bool Parses(const std::string& query) {
    cyxwiz::query::Parser parser(query);
    const auto ast = parser.parse();
    if (parser.hasError() || !ast) {
        std::cerr << "  " << query << "\n  -> " << parser.getError() << " (line "
                  << parser.getErrorLine() << ", column " << parser.getErrorColumn() << ")\n";
        return false;
    }
    return true;
}

}  // namespace

int main() {
    // Every example the Query Console offers parses: Run on an example never
    // ends in a syntax error.
    for (const auto& example : cyxwiz::query::Examples()) {
        Check(Parses(example.query), std::string("example parses: ") + example.name);
    }

    // Function names are words, not reserved: an alias or a property may use them.
    Check(Parses("MATCH (n) RETURN type(n) AS type, count(n) AS count"), "function words as aliases");
    Check(Parses("MATCH (n) RETURN n.size, n.id, n.keys"), "function words as property names");
    Check(Parses("MATCH (n) WHERE n.size > 2 RETURN n.name AS size"), "function words in WHERE and AS");

    // A query that is wrong is still refused.
    Check(!Parses("MATCH (n) RETURN n AS"), "an AS with no alias is refused");
    Check(!Parses("MATCH (n) RETURN n."), "a dot with no property is refused");

    if (failures > 0) {
        std::cerr << failures << " check(s) failed\n";
        return 1;
    }
    std::cout << "test_cyxql_parser: all checks passed\n";
    return 0;
}
