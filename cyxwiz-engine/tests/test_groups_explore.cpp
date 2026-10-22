// Data Studio's Explore tabs Groups and Projection (TOFIX129 D5), against
// scikit-learn and scipy on the Palmer penguins table
// (fixtures/generate_groups_fixtures.py): rows missing a value left out, the
// standardised columns, the four methods' groups and scores, what each group
// is, the comparison with a column (adjusted Rand), how many groups (elbow,
// silhouette by cut, BIC, the k-distance knee with each metric), the top of
// the dendrogram, PCA, the t-SNE / UMAP row cap, and the words.

#include "../src/core/groups_explore.h"
#include "../src/core/groups_presentation.h"
#include "../src/core/session_query_engine.h"

#include <arrow/api.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <vector>

using namespace cyxwiz;
using nlohmann::json;
namespace fs = std::filesystem;

namespace {
void Check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        std::exit(1);
    }
}

void Near(double a, double b, double tol, const std::string& what) {
    Check(std::fabs(a - b) <= tol * (1.0 + std::fabs(b)), what + ": " + std::to_string(a) + " vs " + std::to_string(b));
}

void Contains(const std::string& text, const std::string& part, const std::string& what) {
    Check(text.find(part) != std::string::npos, what + ": \"" + part + "\" not in \"" + text + "\"");
}

std::shared_ptr<arrow::Table> Penguins(const json& f) {
    std::vector<std::shared_ptr<arrow::Field>> fields;
    std::vector<std::shared_ptr<arrow::Array>> arrays;
    for (const auto& c : f["columns"]) {
        arrow::DoubleBuilder b;
        for (const auto& v : f["table"][c.get<std::string>()]) {
            if (v.is_null()) (void)b.AppendNull();
            else (void)b.Append(v.get<double>());
        }
        std::shared_ptr<arrow::Array> a;
        (void)b.Finish(&a);
        fields.push_back(arrow::field(c.get<std::string>(), arrow::float64()));
        arrays.push_back(a);
    }
    arrow::StringBuilder s;
    for (const auto& v : f["table"]["species"]) (void)s.Append(v.get<std::string>());
    std::shared_ptr<arrow::Array> a;
    (void)s.Finish(&a);
    fields.push_back(arrow::field("species", arrow::utf8()));
    arrays.push_back(a);
    return arrow::Table::Make(arrow::schema(fields), arrays);
}

// Our labels -> scikit-learn's, when the two partitions are the same (else empty).
std::map<int, int> SamePartition(const std::vector<int>& ours, const std::vector<int>& theirs) {
    std::map<int, int> to, from;
    if (ours.size() != theirs.size()) return {};
    for (size_t i = 0; i < ours.size(); ++i) {
        const auto [a, inserted_a] = to.emplace(ours[i], theirs[i]);
        const auto [b, inserted_b] = from.emplace(theirs[i], ours[i]);
        if (a->second != theirs[i] || b->second != ours[i]) return {};
    }
    return to;
}

std::vector<int> Ints(const json& j) { return j.get<std::vector<int>>(); }
std::vector<double> Doubles(const json& j) { return j.get<std::vector<double>>(); }
}  // namespace

int main() {
    std::ifstream in(CYXWIZ_GROUPS_FIXTURE);
    Check(in.good(), std::string("cannot open ") + CYXWIZ_GROUPS_FIXTURE);
    const json f = json::parse(in);
    const auto columns = f["columns"].get<std::vector<std::string>>();

    const fs::path root = fs::temp_directory_path() / "cyxwiz_test_groups_explore";
    std::error_code ec;
    fs::remove_all(root, ec);
    SessionQueryEngine engine(root.string());
    std::string error;
    Check(engine.AttachArrow("penguins", 1, Penguins(f), &error), "attach: " + error);
    const QueryRunner run = [&](const QueryRequest& r) { return engine.Run(r); };
    const int64_t used = f["used"].get<int64_t>();

    // ---- pure helpers
    for (const auto& c : f["ari"]) Near(groups::AdjustedRand(Ints(c["a"]), Ints(c["b"])), c["ari"].get<double>(), 1e-12, "adjusted Rand");
    Check(std::isnan(groups::AdjustedRand({0, 1}, {0})), "adjusted Rand of different lengths is NaN");
    Check(std::isnan(groups::KneeOf({1.0, 2.0})), "no knee below 3 points");
    Check(groups::ElbowOf({2, 3}, {5.0, 4.0}) == 0, "no elbow below 3 points");
    Check(groups::ElbowOf(Ints(f["elbow"]["ks"]), Doubles(f["elbow"]["inertia"])) == f["elbow"]["elbow"].get<int>(),
          "elbow of scikit-learn's inertias");

    // ---- K-Means, k = 3, compared with species
    groups::GroupsRequest km;
    km.columns = columns;
    km.compare = "species";
    const auto a = groups::RunGroups("penguins", km, run, {});
    Check(a.ok(), "K-Means: " + a.error);
    Check(a.rows == 344 && a.used == used, "K-Means rows " + std::to_string(a.rows) + " used " + std::to_string(a.used));
    const auto& sk = f["kmeans"];
    const auto map = SamePartition(a.labels, Ints(sk["labels"]));
    Check(map.size() == 3, "K-Means groups are scikit-learn's");
    Near(a.inertia, sk["inertia"].get<double>(), 1e-6, "inertia");
    Check(a.scored, "K-Means scored");
    Near(a.silhouette, sk["silhouette"].get<double>(), 1e-6, "silhouette");
    Near(a.davies_bouldin, sk["davies_bouldin"].get<double>(), 1e-6, "Davies-Bouldin");
    Near(a.calinski_harabasz, sk["calinski_harabasz"].get<double>(), 1e-6, "Calinski-Harabasz");
    Check(a.groups.size() == 3, "three groups");
    for (size_t j = 0; j < columns.size(); ++j) {
        Near(a.overall_means[j], sk["overall_means"][j].get<double>(), 1e-12, "overall mean");
        Near(a.overall_sds[j], sk["overall_sds"][j].get<double>(), 1e-12, "overall sd (population)");
    }
    for (const auto& g : a.groups) {
        const int theirs = map.at(g.label);
        Near(g.silhouette, sk["group_silhouette"][theirs].get<double>(), 1e-6, "group silhouette");
        for (size_t j = 0; j < columns.size(); ++j) Near(g.means[j], sk["means"][theirs][j].get<double>(), 1e-12, "group mean");
        int64_t rows = 0;
        for (const int l : Ints(sk["labels"])) rows += l == theirs;
        Check(g.rows == rows, "group rows");
    }
    Check(a.compare_values == sk["species"].get<std::vector<std::string>>(), "species values sorted");
    for (const auto& g : a.groups)
        for (size_t v = 0; v < a.compare_values.size(); ++v)
            Check(a.crosstab[static_cast<size_t>(&g - a.groups.data())][v] == sk["crosstab"][map.at(g.label)][v].get<int64_t>(), "crosstab");
    Near(a.adjusted_rand, sk["adjusted_rand"].get<double>(), 1e-12, "adjusted Rand with species");
    Check(a.x.size() == static_cast<size_t>(used) && a.y.size() == a.x.size(), "scatter has every row");
    for (size_t i = 0; i < a.x.size(); ++i) {
        Near(a.x[i], f["pca"]["x"][i].get<double>(), 1e-6, "scatter x is component 1");
        Near(a.y[i], f["pca"]["y"][i].get<double>(), 1e-6, "scatter y is component 2");
    }
    Near(a.x_ratio, f["pca"]["ratio"][0].get<double>(), 1e-9, "component 1 share");

    // The same seed, the same groups; scaling off reads the raw units.
    Check(groups::RunGroups("penguins", km, run, {}).labels == a.labels, "a run repeats");
    groups::GroupsRequest raw = km;
    raw.standardise = false;
    const auto r = groups::RunGroups("penguins", raw, run, {});
    Check(r.ok() && r.inertia > 1e6, "unscaled K-Means is dominated by body_mass_g (inertia " + std::to_string(r.inertia) + ")");

    // Words.
    Check(groups_view::Summary(a) == "3 groups of " + std::to_string(a.groups[0].rows) + ", " + std::to_string(a.groups[1].rows) +
                                         " and " + std::to_string(a.groups[2].rows) + " rows; silhouette " +
                                         groups_view::Score(sk["silhouette"].get<double>()) + ": weak structure.",
          "summary: " + groups_view::Summary(a));
    Check(groups_view::LeftOut(a.rows, a.used) == "2 rows have a missing value and are left out.", "left out");
    Contains(groups_view::MethodLine(a), "inertia 379.4 after", "method line");
    const size_t gentoo = static_cast<size_t>(std::find(a.compare_values.begin(), a.compare_values.end(), "Gentoo") - a.compare_values.begin());
    for (size_t g = 0; g < a.groups.size(); ++g)
        if (a.crosstab[g][gentoo] > 100) {
            Contains(groups_view::Profile(a, g), "high flipper_length_mm", "the Gentoo group's profile");
            Contains(groups_view::Profile(a, g), "low bill_depth_mm", "the Gentoo group's profile");
            Contains(groups_view::CompareSentence(a), groups_view::GroupName(a.groups[g].label) + " is all Gentoo (123).", "compare");
        }
    Contains(groups_view::CompareSentence(a), "Agreement with species (adjusted Rand) 0.793: close.", "agreement");
    const std::string csv = groups_view::GroupsCsv(a);
    Contains(csv, "group,rows,silhouette,mean bill_length_mm", "CSV header");
    Check(std::count(csv.begin(), csv.end(), '\n') >= used + 5, "CSV has a row per row used");

    // ---- DBSCAN, eps 0.6
    groups::GroupsRequest dbs = km;
    dbs.method = groups::Method::DBSCAN;
    dbs.eps = 0.6;
    dbs.min_samples = 5;
    const auto d = groups::RunGroups("penguins", dbs, run, {});
    Check(d.ok(), "DBSCAN: " + d.error);
    Check(d.labels == Ints(f["dbscan"]["labels"]), "DBSCAN labels are scikit-learn's");
    int noise = 0;
    for (const int l : d.labels) noise += l == -1;
    Check(d.noise == noise && d.groups.size() == 2, "DBSCAN: 2 groups and its noise");
    Contains(groups_view::Summary(d), "2 groups and " + std::to_string(noise) + " rows of noise", "DBSCAN summary");

    for (const std::string metric : {"euclidean", "manhattan", "cosine"}) {
        groups::HowManyRequest h;
        h.base = dbs;
        h.base.metric = metric;
        const auto k = groups::RunHowMany("penguins", h, run, {});
        Check(k.ok(), "k-distances " + metric + ": " + k.error);
        const auto truth = Doubles(f["dbscan"]["kdistances"][metric]["curve"]);
        Check(k.kdistances.size() == truth.size(), "k-distance curve length " + metric);
        for (size_t i = 0; i < truth.size(); ++i) Near(k.kdistances[i], truth[i], 1e-9, "k-distance " + metric);
        Near(k.knee, f["dbscan"]["kdistances"][metric]["knee"].get<double>(), 1e-9, "knee " + metric);
        if (metric == "euclidean") Contains(groups_view::HowManySentence(k), "The knee of the curve (Kneedle) is at", "knee words");
    }

    // ---- Hierarchical, ward
    groups::GroupsRequest ward = km;
    ward.method = groups::Method::Hierarchical;
    const auto w = groups::RunGroups("penguins", ward, run, {});
    Check(w.ok(), "ward: " + w.error);
    Check(SamePartition(w.labels, Ints(f["ward"]["labels"])).size() == 3, "ward groups are scipy's");
    Near(w.cophenetic, f["ward"]["cophenetic"].get<double>(), 1e-9, "cophenetic");
    Contains(groups_view::MethodLine(w), "the tree keeps the rows' distances well", "cophenetic words");
    ward.metric = "manhattan";
    Contains(groups::RunGroups("penguins", ward, run, {}).error, "Ward linkage measures euclidean", "ward refuses manhattan");
    ward.metric = "euclidean";

    groups::HowManyRequest hw;
    hw.base = ward;
    const auto wh = groups::RunHowMany("penguins", hw, run, {});
    Check(wh.ok(), "ward how many: " + wh.error);
    const auto Z = f["ward"]["linkage"].get<std::vector<std::vector<double>>>();
    Check(wh.linkage.size() == Z.size(), "linkage size");
    for (size_t i = 0; i < Z.size(); ++i)
        for (size_t c = 0; c < 4; ++c) Near(wh.linkage[i][c], Z[i][c], 1e-9, "linkage");
    const auto cut = Doubles(f["ward"]["silhouette"]);
    for (size_t i = 0; i < cut.size(); ++i) Near(wh.silhouette[i], cut[i], 1e-6, "silhouette of the cut into " + std::to_string(wh.ks[i]));
    const auto top = groups::TopOfTree(Z, used, 12);
    const auto ic = f["ward"]["icoord"], dc = f["ward"]["dcoord"];
    Check(top.xs.size() == ic.size() && top.xs.size() == 11, "dendrogram: 11 links");
    for (size_t l = 0; l < ic.size(); ++l)
        for (size_t c = 0; c < 4; ++c) {
            Near(top.xs[l][c], ic[l][c].get<double>(), 1e-12, "dendrogram x");
            Near(top.ys[l][c], dc[l][c].get<double>(), 1e-12, "dendrogram y");
        }
    Check(top.leaf_rows == f["ward"]["leaf_rows"].get<std::vector<int64_t>>(), "dendrogram leaf sizes");

    // ---- How many: K-Means and Gaussian mixture
    groups::HowManyRequest hk;
    hk.base = km;
    const auto e = groups::RunHowMany("penguins", hk, run, {});
    Check(e.ok() && e.ks == Ints(f["elbow"]["ks"]), "K-Means how many: " + e.error);
    for (size_t i = 0; i < e.ks.size(); ++i) {
        Near(e.inertia[i], f["elbow"]["inertia"][i].get<double>(), 2e-2, "inertia at " + std::to_string(e.ks[i]));
        Near(e.silhouette[i], f["elbow"]["silhouette"][i].get<double>(), 5e-2, "silhouette at " + std::to_string(e.ks[i]));
    }
    Check(e.suggested == f["elbow"]["elbow"].get<int>(), "the elbow is scikit-learn's");
    Contains(groups_view::HowManySentence(e), "The elbow bends at 3 (inertia falls 33% from 2 to 3)", "elbow words");
    Contains(groups_view::HowManySentence(e), "the silhouette is highest at 2", "elbow words");

    groups::HowManyRequest hg;
    hg.base = km;
    hg.base.method = groups::Method::Mixture;
    hg.to = 6;
    const auto g = groups::RunHowMany("penguins", hg, run, {});
    Check(g.ok() && g.ks == Ints(f["gmm"]["ks"]), "mixture how many: " + g.error);
    for (size_t i = 0; i < g.ks.size(); ++i) Near(g.bic[i], f["gmm"]["bic"][i].get<double>(), 2e-2, "BIC at " + std::to_string(g.ks[i]));
    Check(g.suggested == 3, "BIC is lowest at 3 components");
    Contains(groups_view::HowManySentence(g), "BIC is lowest at 3 components", "BIC words");

    groups::GroupsRequest mix = km;
    mix.method = groups::Method::Mixture;
    const auto m = groups::RunGroups("penguins", mix, run, {});
    Check(m.ok() && m.probabilities.size() == static_cast<size_t>(used), "mixture probabilities per row: " + m.error);
    for (const auto& p : m.probabilities) {
        double sum = 0.0;
        for (const double v : p) sum += v;
        Near(sum, 1.0, 1e-9, "probabilities sum to 1");
    }

    // ---- Projection
    groups::ProjectionRequest pr;
    pr.columns = columns;
    pr.colour = "species";
    const auto p = groups::RunProjection("penguins", pr, run, {});
    Check(p.ok(), "PCA: " + p.error);
    const auto ratio = Doubles(f["pca"]["ratio"]);
    Check(p.ratio.size() == ratio.size(), "every component's share");
    for (size_t i = 0; i < ratio.size(); ++i) Near(p.ratio[i], ratio[i], 1e-9, "explained variance ratio");
    for (size_t c = 0; c < 2; ++c)
        for (size_t j = 0; j < columns.size(); ++j) Near(p.loadings[c][j], f["pca"]["components"][c][j].get<double>(), 1e-9, "loading");
    Check(p.colour.size() == p.x.size() && p.colour[0] == "Adelie" && !p.colour_numeric, "coloured by species");
    Check(groups_view::AxisTitle(p, 0) == "component 1 (68.8%)", "axis title " + groups_view::AxisTitle(p, 0));
    Contains(groups_view::ProjectionSentence(p), "Two components keep 88% of the variation.", "PCA words");
    Contains(groups_view::ProjectionSentence(p),
             "Component 1: flipper_length_mm, body_mass_g and bill_length_mm together, against bill_depth_mm.", "component 1 words");
    Contains(groups_view::ProjectionSentence(p), "Component 2: bill_depth_mm and bill_length_mm together.", "component 2 words");

    pr.method = groups::ProjectionMethod::TSNE;
    pr.max_rows = 120;
    pr.iterations = 300;
    const auto t = groups::RunProjection("penguins", pr, run, {});
    Check(t.ok() && t.sampled && t.x.size() == 120 && t.colour.size() == 120, "t-SNE on a sample of 120: " + t.error);
    const auto t2 = groups::RunProjection("penguins", pr, run, {});
    Check(t2.x == t.x && t2.colour == t.colour, "the sample and t-SNE repeat");
    Contains(groups_view::ProjectionSentence(t), "a seeded sample of 120 of 342 rows", "sample words");
    Check(t.used_rows.size() == 120 && std::is_sorted(t.used_rows.begin(), t.used_rows.end()) && t.used_rows.back() < used,
          "each point keeps its row, in table order");
    // Cancel: t-SNE stops at its next report.
    int reports = 0;
    groups::ProjectionHooks hooks;
    hooks.progress = [&](float, const std::string&) { ++reports; };
    hooks.should_stop = [&] { return reports >= 1; };
    const auto stopped = groups::RunProjection("penguins", pr, run, {}, hooks);
    Check(stopped.error == "Cancelled." && reports == 1, "t-SNE stops when asked: " + stopped.error + " after " + std::to_string(reports));
    pr.perplexity = 500;
    Contains(groups::RunProjection("penguins", pr, run, {}).error, "Perplexity must be", "perplexity refused");

    pr.method = groups::ProjectionMethod::UMAP;
    pr.max_rows = 5000;
    const auto u = groups::RunProjection("penguins", pr, run, {});
    Check(u.ok() && !u.sampled && u.x.size() == static_cast<size_t>(used), "UMAP on every row: " + u.error);
    for (size_t i = 0; i < u.x.size(); ++i) Check(std::isfinite(u.x[i]) && std::isfinite(u.y[i]), "UMAP finite");

    // ---- point colours
    {
        const auto c = groups_view::ColourPoints({"b", "a", "", "b"}, false);
        Check(c.index == std::vector<int>({1, 0, -1, 1}) && c.legend.size() == 2 && c.legend[1].first == "b" && c.legend[1].second == 2,
              "text colours: sorted values, missing -1");
        std::vector<std::string> many;
        for (int i = 0; i < 20; ++i) many.push_back("v" + std::to_string(100 + i));
        many.push_back("v100");
        const auto o = groups_view::ColourPoints(many, false);
        Check(o.legend.size() == 18 && o.legend.back().first == "other" && o.legend.back().second == 3 && o.index[0] == 0,
              "beyond 18 values: 17 most frequent and other");
        const auto n = groups_view::ColourPoints({"1", "2", "3", "4", "5", "6", "7", "8"}, true);
        Check(n.legend.size() == 4 && n.index.front() == 0 && n.index.back() == 3, "numbers: four quartile bins");
        const auto gc = groups_view::ColourGroups({1, 0, -1, 1});
        Check(gc.legend.size() == 3 && gc.legend[0].first == "Group 1" && gc.legend[2].first == "noise", "group colours");
    }

    // ---- refusals
    groups::GroupsRequest none;
    Contains(groups::RunGroups("penguins", none, run, {}).error, "Add the number columns", "no columns");
    groups::GroupsRequest many = km;
    many.k = 1;
    Contains(groups::RunGroups("penguins", many, run, {}).error, "2 to 50", "k below 2");

    fs::remove_all(root, ec);
    std::cout << "test_groups_explore: all checks passed\n";
    return 0;
}
