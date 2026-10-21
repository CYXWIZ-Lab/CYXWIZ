// Data Studio Explore tabs' words (TOFIX129 D1): counts, shares, the
// missing-value advice tiers, the Query tab's SQL and the Keep rows step
// "Drop these rows in Prepare" adds.

#include "../src/core/explore_presentation.h"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>

using namespace cyxwiz;
namespace ev = cyxwiz::explore_view;

namespace {
void Check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        std::exit(1);
    }
}
}  // namespace

int main() {
    Check(ev::Number(2408) == "2,408" && ev::Number(7) == "7", "thousands");
    Check(ev::Count(333, "complete row", "complete rows") == "333 complete rows", "plural");
    Check(ev::Count(1, "row", "rows") == "1 row", "singular");
    Check(ev::Share(19, 2408) == "0.8%" && ev::Share(11, 344) == "3.2%" && ev::Share(0, 0) == "0%", "shares");

    // The old window's tiers: below 5, 20, 50, then 50 or more.
    Check(ev::MissingAdvice(0).empty(), "nothing missing: no advice");
    Check(ev::MissingAdvice(3.2).front().find("dropping them") != std::string::npos, "under 5%: drop or fill");
    Check(ev::MissingAdvice(12).front().find("similar rows") != std::string::npos, "under 20%: fill from similar rows");
    Check(ev::MissingAdvice(30).front().find("Much is missing") != std::string::npos, "under 50%: indicator, model");
    Check(ev::MissingAdvice(50).front().find("More than half") != std::string::npos, "50% and more: drop the column");
    Check(ev::GeneralMissingAdvice().size() == 4, "general recommendations kept");

    Check(ev::MissingRule(0).find("No column has texts") != std::string::npos, "rule without texts");
    Check(ev::MissingRule(2).find("in 2 columns") != std::string::npos, "rule with texts");

    Check(ev::MissingRowsSql("Penguins", "sex", {}) == "SELECT *\nFROM \"Penguins\"\nWHERE \"sex\" IS NULL", "rows SQL");
    Check(ev::MissingRowsSql("t", "a\"b", {"N/A", "it's"}) ==
              "SELECT *\nFROM \"t\"\nWHERE \"a\"\"b\" IS NULL OR CAST(\"a\"\"b\" AS VARCHAR) IN ('N/A', 'it''s')",
          "quoted names and texts");

    Check(ev::Figure(4201.754) == "4202" && ev::Figure(0.18772859) == "0.1877" && ev::Figure(2700) == "2700", "figures");
    Check(ev::Figure(1437000) == "1.437e+06" && ev::Figure(-0.4719) == "-0.4719" && ev::Figure(NAN).empty(), "large, negative, NaN");
    Check(ev::OutlierRowsSql("Penguins", "body_mass_g", 2597.5, 5805.75) ==
              "SELECT *\nFROM \"Penguins\"\nWHERE \"body_mass_g\" < 2597.5 OR \"body_mass_g\" > 5805.75",
          "outlier rows SQL: " + ev::OutlierRowsSql("Penguins", "body_mass_g", 2597.5, 5805.75));

    // Drop these rows: a guided Keep rows step whose SQL keeps the rows with a value.
    guided::GuidedStep plain = ev::KeepRowsWithValue("sex", {});
    Check(plain.ops.size() == 1 && plain.ops[0].type == "keep_rows", "one Keep rows operation");
    guided::Context context;
    context.inputs = {"input"};
    context.columns["input"] = {"species", "sex"};
    const guided::Generated g = guided::GenerateSql(plain, context);
    Check(g.ok && g.sql.find("\"sex\" IS NOT NULL") != std::string::npos, "keeps rows where sex is not empty: " + g.sql);
    const guided::Generated t = guided::GenerateSql(ev::KeepRowsWithValue("sex", {"N/A", "-"}), context);
    Check(t.ok && t.sql.find("NOT IN ('N/A', '-')") != std::string::npos, "and is not one of the missing texts: " + t.sql);

    std::cout << "explore presentation: counts, advice tiers, rule line, rows SQL, Keep rows step. OK\n";
    return 0;
}
