// Data Studio's ML readiness (TOFIX148 step 3): the wording on the Berean
// kjv80_base_v1 splits as measured on 2026-10-14 (1,228 / 68 / 66 chapters,
// 1,066 / 62 / 63 families, no overlap, median words 641.5 / 555 / 549), and
// Measure through a real query engine on small splits with a leaked family,
// a repeated text and a target.

#include "../src/core/ml_readiness.h"
#include "../src/core/session_query_engine.h"

#include <arrow/api.h>

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>

using namespace cyxwiz;
using namespace cyxwiz::readiness;
namespace fs = std::filesystem;

namespace {
void Check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        std::exit(1);
    }
}

const Card* CardNamed(const View& v, const std::string& title) {
    for (const auto& c : v.cards)
        if (c.title == title) return &c;
    return nullptr;
}

bool HasLine(const Card* c, const std::string& text, studio::Tone tone) {
    if (!c) return false;
    for (const auto& l : c->lines)
        if (l.text == text && l.tone == tone) return true;
    return false;
}

std::string Lines(const Card* c) {
    std::string s;
    if (c)
        for (const auto& l : c->lines) s += "\n  " + l.text;
    return s;
}

SplitFacts Split(const char* split, int64_t rows, int64_t groups, int64_t ids, double median, double mean) {
    SplitFacts s;
    s.table = {split, std::string("documents_") + split, ""};
    s.rows = rows;
    s.groups = groups;
    s.ids = ids;
    s.words_median = median;
    s.words_mean = mean;
    return s;
}

std::shared_ptr<arrow::Table> Docs(const std::vector<std::string>& ids, const std::vector<std::string>& families,
                                   const std::vector<std::string>& texts, const std::vector<std::string>& labels) {
    arrow::StringBuilder a, b, c, d;
    for (size_t i = 0; i < ids.size(); ++i) {
        (void)a.Append(ids[i]);
        (void)b.Append(families[i]);
        (void)c.Append(texts[i]);
        (void)d.Append(labels[i]);
    }
    std::shared_ptr<arrow::Array> x, y, z, w;
    (void)a.Finish(&x);
    (void)b.Finish(&y);
    (void)c.Finish(&z);
    (void)d.Finish(&w);
    return arrow::Table::Make(arrow::schema({arrow::field("document_id", arrow::utf8()), arrow::field("source family", arrow::utf8()),
                                             arrow::field("text", arrow::utf8()), arrow::field("book", arrow::utf8())}),
                              {x, y, z, w});
}
}  // namespace

int main() {
    using studio::Tone;

    // Roles: yours first, then the contract, then inferred.
    {
        DatasetContract c;
        c.columns = {{"text", "text", ColumnRole::Text, RoleSource::Inferred, "", {}},
                     {"source_text", "text", ColumnRole::Text, RoleSource::Inferred, "", {}},
                     {"document_id", "text", ColumnRole::Id, RoleSource::Inferred, "", {}},
                     {"source_family_id", "text", ColumnRole::Group, RoleSource::User, "", {}},
                     {"partition_document_id", "text", ColumnRole::Id, RoleSource::User, "", {}}};
        // kjv80 as profiled: archive_sha256 (a book's hash, 64 characters) reads as Text before text.
        c.columns.insert(c.columns.begin(), {"archive_sha256", "text", ColumnRole::Text, RoleSource::Inferred, "", {}});
        std::vector<ColumnFacts> facts(3);
        facts[0].name = "archive_sha256";
        facts[0].avg_length = 64;
        facts[1].name = "text";
        facts[1].avg_length = 3790;
        facts[2].name = "source_text";
        facts[2].avg_length = 3790;
        const Roles r = PickRoles(c, facts);
        Check(r.text == "text" && r.group == "source_family_id" && r.id == "partition_document_id" && r.target.empty(),
              "roles: the longest Text (not the hash), the Group, the ID you set over the inferred one: " + r.text);
        c.columns[2].source = RoleSource::User;  // source_text set by you wins over longer inferred text
        Check(PickRoles(c, facts).text == "source_text", "a Text you set wins");
    }

    // kjv80_base_v1 as the Engine measures it.
    {
        Facts f;
        f.name = "kjv80_base_v1";
        f.roles = {"text", "document_id", "source_family_id", ""};
        f.splits = {Split("train", 1228, 1066, 1228, 641.5, 690.687), Split("validation", 68, 62, 68, 555, 630.0),
                    Split("test", 66, 63, 66, 549, 610.879)};
        for (size_t a = 0; a < 3; ++a)
            for (size_t b = a + 1; b < 3; ++b) f.overlaps.push_back({a, b, 0, 0, 0});
        const View v = Build(f);
        Check(v.title == "kjv80_base_v1" && v.splits == "train \xC2\xB7 validation \xC2\xB7 test", "header: " + v.splits);
        Check(v.verdict == "No leakage found" && v.caption == "no leakage" && v.verdict_tone == Tone::Ok, "verdict: " + v.verdict);
        const Card* sizes = CardNamed(v, "Split sizes");
        Check(sizes && sizes->header == std::vector<std::string>({"", "rows", "groups", "IDs"}) && sizes->rows.size() == 3 &&
                  sizes->rows[0] == std::vector<std::string>({"train", "1,228", "1,066", "1,228"}) && sizes->lines.empty(),
              "split sizes as measured");
        const Card* leak = CardNamed(v, "Leakage across splits");
        Check(HasLine(leak, "0 groups in two splits (source_family_id)", Tone::Ok) &&
                  HasLine(leak, "0 IDs in two splits (document_id)", Tone::Ok) &&
                  HasLine(leak, "0 identical texts in two splits (text)", Tone::Ok),
              "no leakage of any kind:" + Lines(leak));
        const Card* len = CardNamed(v, "Text length (words)");
        Check(len && len->rows[0] == std::vector<std::string>({"train", "641.5", "690.7"}) &&
                  len->rows[1] == std::vector<std::string>({"validation", "555", "630"}),
              "text length figures");
        Check(HasLine(len, "Validation and test are about 14% shorter than train.", Tone::Warn), "length warning:" + Lines(len));
        const Card* roles = CardNamed(v, "Roles (dataset contract)");
        Check(HasLine(roles, "Group  source_family_id", Tone::Plain) && roles->lines[3].text == "Target: none (a text corpus)",
              "roles card:" + Lines(roles));
        const Card* bal = CardNamed(v, "Class balance");
        Check(bal && bal->rows.empty() && bal->lines.size() == 1 && bal->lines[0].faint, "no target: class balance explains");

        // Without a Group the family check is not there; with no role at all nothing is checked.
        Facts no_group = f;
        no_group.roles.group.clear();
        for (auto& o : no_group.overlaps) o.groups = -1;
        const View ng = Build(no_group);
        Check(ng.verdict == "No leakage found" && HasLine(CardNamed(ng, "Leakage across splits"), "Groups: no Group column", Tone::Plain),
              "no group column says so");
        Facts none = no_group;
        none.roles = {};
        for (auto& o : none.overlaps) o = {o.a, o.b, -1, -1, -1};
        Check(Build(none).verdict == "Leakage not checked" && Build(none).verdict_tone == Tone::Warn, "nothing to compare on");
        Facts one = f;
        one.splits.resize(1);
        one.overlaps.clear();
        Check(Build(one).verdict == "One split only" && !CardNamed(Build(one), "Leakage across splits"), "one split");
    }

    // Measure on a real engine: family f2 and a text are in train and test; one train ID repeats.
    {
        const fs::path root = fs::temp_directory_path() / "cyxwiz_test_ml_readiness";
        std::error_code ec;
        fs::remove_all(root, ec);
        {
            SessionQueryEngine engine((root / "attach").string());
            std::string error;
            Check(engine.AttachArrow("docs_train", 1,
                                     Docs({"d1", "d2", "d3", "d3", "d4"}, {"f1", "f1", "f2", "f2", "f3"},
                                          {"In the beginning", "  and the earth  ", "was without form", "was without form", ""},
                                          {"gen", "gen", "gen", "exo", "exo"}),
                                     &error),
                  "attach train: " + error);
            Check(engine.AttachArrow("docs test", 1,
                                     Docs({"d9", "d10"}, {"f2", "f9"}, {"In the beginning", "and void"}, {"exo", "exo"}), &error),
                  "attach test: " + error);
            const QueryRunner run = [&](const QueryRequest& r) { return engine.Run(r); };
            const Roles roles{"text", "document_id", "source family", "book"};
            const Facts f = Measure("docs", {{"train", "docs_train", ""}, {"test", "docs test", ""}}, roles, run);
            Check(f.error.empty() && f.splits.size() == 2 && f.overlaps.size() == 1, "measured: " + f.error);
            const SplitFacts& tr = f.splits[0];
            Check(tr.error.empty() && tr.rows == 5 && tr.groups == 3 && tr.ids == 4, "train counts: " + tr.error);
            Check(tr.words_median == 3.0 && std::abs(tr.words_mean - 2.4) < 1e-9, "words: 3, 3, 3, 3, 0 (spaces trimmed, empty is 0)");
            Check(tr.classes.size() == 2 && tr.classes[0] == std::make_pair(std::string("gen"), int64_t{3}), "classes, most first");
            Check(f.overlaps[0].groups == 1 && f.overlaps[0].ids == 0 && f.overlaps[0].texts == 1, "f2 and one text leaked");

            const View v = Build(f);
            Check(v.verdict == "Leakage: 1 group and 1 identical text in two splits" && v.verdict_tone == Tone::Error,
                  "verdict: " + v.verdict);
            const Card* leak = CardNamed(v, "Leakage across splits");
            Check(HasLine(leak, "   1 in train and test", Tone::Error), "pair detail:" + Lines(leak));
            Check(HasLine(CardNamed(v, "Split sizes"), "train: 1 row repeats an ID", Tone::Warn), "repeated ID");
            const Card* bal = CardNamed(v, "Class balance");
            Check(bal && bal->header == std::vector<std::string>({"book", "train", "test"}) &&
                      bal->rows[0] == std::vector<std::string>({"gen", "60%", "0%"}),
                  "class shares per split");
            Check(HasLine(bal, "gen: 60% in train, 0% in test.", Tone::Warn), "class share moved:" + Lines(bal));

            // A missing table is that split's error, not a crash.
            const Facts bad = Measure("x", {{"train", "docs_train", ""}, {"test", "nope", ""}}, roles, run);
            Check(bad.splits[1].error.size() > 0 && bad.overlaps[0].groups == -1, "a split that cannot be read");
            // Stopping between queries.
            const Facts stop = Measure("x", {{"train", "docs_train", ""}}, roles, run, [] { return true; });
            Check(stop.error == "Cancelled.", "cancelled");
        }
        fs::remove_all(root, ec);
    }

    std::cout << "ml readiness: roles picked yours first, kjv80 splits worded (sizes, no leakage, length warning), "
                 "leaked family and text counted on a real engine, class shares, missing tables, cancel. OK\n";
    return 0;
}
