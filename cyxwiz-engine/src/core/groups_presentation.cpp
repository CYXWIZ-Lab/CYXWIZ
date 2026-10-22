#include "groups_presentation.h"

#include "explore_presentation.h"

#include <algorithm>
#include <cmath>
#include <cstdio>

namespace cyxwiz::groups_view {

using explore_view::Figure;
using explore_view::Number;

namespace {

// "a", "a and b", "a, b and c".
std::string Listed(const std::vector<std::string>& items) {
    std::string out;
    for (size_t i = 0; i < items.size(); ++i) {
        if (i > 0) out += i + 1 == items.size() ? " and " : ", ";
        out += items[i];
    }
    return out;
}

std::string Percent(double share) {
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%.0f%%", 100.0 * share);
    return buf;
}

// A value in full for the CSV ("" for NaN).
std::string Full(double v) {
    if (!std::isfinite(v)) return "";
    char buf[40];
    std::snprintf(buf, sizeof(buf), "%.10g", v);
    return buf;
}

}  // namespace

const char* MethodLabel(groups::Method m) {
    switch (m) {
        case groups::Method::KMeans: return "K-Means";
        case groups::Method::DBSCAN: return "DBSCAN";
        case groups::Method::Hierarchical: return "Hierarchical";
        case groups::Method::Mixture: return "Gaussian mixture";
    }
    return "";
}

const char* ProjectionLabel(groups::ProjectionMethod m) {
    switch (m) {
        case groups::ProjectionMethod::PCA: return "PCA";
        case groups::ProjectionMethod::TSNE: return "t-SNE";
        case groups::ProjectionMethod::UMAP: return "UMAP";
    }
    return "";
}

std::string Share(double share) {
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%.1f%%", 100.0 * share);
    return buf;
}

std::string GroupName(int label) { return label < 0 ? "Noise" : "Group " + std::to_string(label + 1); }

std::string Score(double v) {
    if (!std::isfinite(v)) return "";
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%.3f", v);
    return buf;
}

const char* SilhouetteWords(double s) {
    if (!std::isfinite(s)) return "unknown";
    if (s > 0.70) return "strong structure";
    if (s > 0.50) return "reasonable structure";
    if (s > 0.25) return "weak structure";
    return "no real structure";
}

const char* AgreementWords(double ari) {
    if (!std::isfinite(ari)) return "unknown";
    if (ari >= 0.9) return "almost the same";
    if (ari >= 0.65) return "close";
    if (ari >= 0.3) return "partly alike";
    return "hardly alike";
}

std::string Summary(const groups::GroupsAnswer& a) {
    std::vector<std::string> sizes;
    for (const auto& g : a.groups) sizes.push_back(Number(static_cast<size_t>(g.rows)));
    const size_t n = a.groups.size();
    std::string out = explore_view::Count(n, "group", "groups");
    if (a.request.method == groups::Method::DBSCAN) {
        out += " and " + explore_view::Count(static_cast<size_t>(a.noise), "row", "rows") + " of noise";
        if (a.request.method == groups::Method::DBSCAN && n == 0) return out + ": no row has enough neighbours within eps; try a larger eps.";
    } else {
        out += " of " + Listed(sizes) + " rows";
    }
    if (a.scored) out += "; silhouette " + Score(a.silhouette) + ": " + SilhouetteWords(a.silhouette) + ".";
    else if (n < 2) out += ": one group cannot be scored.";
    else out += ".";
    return out;
}

std::string MethodLine(const groups::GroupsAnswer& a) {
    switch (a.request.method) {
        case groups::Method::KMeans:
            return "inertia " + Figure(a.inertia) + " after " + std::to_string(a.iterations) +
                   (a.iterations == 1 ? " iteration" : " iterations") + (a.converged ? "" : " (stopped at the limit)");
        case groups::Method::DBSCAN: return "";
        case groups::Method::Hierarchical: {
            if (!std::isfinite(a.cophenetic)) return "";
            const char* how = a.cophenetic >= 0.75 ? "well" : a.cophenetic >= 0.5 ? "fairly" : "poorly";
            return "cophenetic correlation " + Score(a.cophenetic) + ": the tree keeps the rows' distances " + how;
        }
        case groups::Method::Mixture:
            return "BIC " + Figure(a.bic) + ", AIC " + Figure(a.aic) + (a.converged ? "" : " (stopped at the limit)");
    }
    return "";
}

std::string Profile(const groups::GroupsAnswer& a, size_t group) {
    if (group >= a.groups.size()) return "";
    const auto& g = a.groups[group];
    std::vector<std::pair<double, size_t>> z;
    for (size_t j = 0; j < g.means.size() && j < a.overall_means.size() && j < a.request.columns.size(); ++j) {
        const double sd = j < a.overall_sds.size() ? a.overall_sds[j] : 0.0;
        if (!(sd > 0.0)) continue;
        const double v = (g.means[j] - a.overall_means[j]) / sd;
        if (std::fabs(v) >= 0.5) z.emplace_back(v, j);
    }
    if (z.empty()) return "close to the average";
    std::stable_sort(z.begin(), z.end(), [](const auto& l, const auto& r) { return std::fabs(l.first) > std::fabs(r.first); });
    if (z.size() > 3) z.resize(3);
    std::vector<std::string> parts;
    for (const auto& [v, j] : z) parts.push_back(std::string(v > 0 ? "high " : "low ") + a.request.columns[j]);
    return Listed(parts);
}

std::string CompareSentence(const groups::GroupsAnswer& a) {
    if (a.compare_values.empty() || a.crosstab.empty()) return "";
    std::string out;
    for (size_t g = 0; g < a.groups.size() && g < a.crosstab.size(); ++g) {
        const auto& row = a.crosstab[g];
        int64_t total = 0;
        for (const auto c : row) total += c;
        if (total == 0) continue;
        const size_t top = static_cast<size_t>(std::max_element(row.begin(), row.end()) - row.begin());
        const std::string name = GroupName(a.groups[g].label);
        const std::string& value = a.compare_values[top];
        if (row[top] == total) out += name + " is all " + value + " (" + Number(static_cast<size_t>(total)) + "). ";
        else if (2 * row[top] > total)
            out += name + " is mostly " + value + " (" + Number(static_cast<size_t>(row[top])) + " of " +
                   Number(static_cast<size_t>(total)) + "). ";
        else
            out += name + " is mixed (" + value + " the most, " + Number(static_cast<size_t>(row[top])) + " of " +
                   Number(static_cast<size_t>(total)) + "). ";
    }
    out += "Agreement with " + a.request.compare + " (adjusted Rand) " + Score(a.adjusted_rand) + ": " +
           AgreementWords(a.adjusted_rand) + ".";
    return out;
}

std::string LeftOut(int64_t rows, int64_t used) {
    const int64_t out = rows - used;
    if (out <= 0) return "";
    return explore_view::Count(static_cast<size_t>(out), "row has", "rows have") + " a missing value and " +
           (out == 1 ? "is" : "are") + " left out.";
}

std::string HowManySentence(const groups::HowManyAnswer& a) {
    if (!a.ok()) return a.error;
    switch (a.method) {
        case groups::Method::KMeans: {
            if (a.ks.empty()) return "";
            std::string out;
            if (a.suggested > 0) {
                out = "The elbow bends at " + std::to_string(a.suggested);
                const auto at = std::find(a.ks.begin(), a.ks.end(), a.suggested) - a.ks.begin();
                if (at > 0 && a.inertia[static_cast<size_t>(at - 1)] > 0.0)
                    out += " (inertia falls " + Percent(1.0 - a.inertia[static_cast<size_t>(at)] / a.inertia[static_cast<size_t>(at - 1)]) +
                           " from " + std::to_string(a.ks[static_cast<size_t>(at - 1)]) + " to " + std::to_string(a.suggested) + ")";
                out += "; ";
            }
            size_t best = 0;
            for (size_t i = 1; i < a.silhouette.size(); ++i)
                if (a.silhouette[i] > a.silhouette[best]) best = i;
            if (!a.silhouette.empty() && std::isfinite(a.silhouette[best])) {
                out += std::string(out.empty() ? "T" : "t") + "he silhouette is highest at " + std::to_string(a.ks[best]) +
                       " (" + Score(a.silhouette[best]) + ")";
                const auto at = std::find(a.ks.begin(), a.ks.end(), a.suggested) - a.ks.begin();
                if (a.suggested > 0 && a.ks[best] != a.suggested && static_cast<size_t>(at) < a.silhouette.size())
                    out += " and " + Score(a.silhouette[static_cast<size_t>(at)]) + " at " + std::to_string(a.suggested);
            }
            return out + ".";
        }
        case groups::Method::Mixture: {
            if (a.suggested == 0) return "";
            const auto at = static_cast<size_t>(std::find(a.ks.begin(), a.ks.end(), a.suggested) - a.ks.begin());
            std::string out = "BIC is lowest at " + std::to_string(a.suggested) + " components (" + Figure(a.bic[at]);
            if (at > 0) out += "; " + std::to_string(a.ks[at - 1]) + ": " + Figure(a.bic[at - 1]);
            if (at + 1 < a.bic.size()) out += "; " + std::to_string(a.ks[at + 1]) + ": " + Figure(a.bic[at + 1]);
            return out + ").";
        }
        case groups::Method::Hierarchical: {
            if (a.suggested == 0) return "";
            const auto at = static_cast<size_t>(std::find(a.ks.begin(), a.ks.end(), a.suggested) - a.ks.begin());
            return "Cutting the tree into " + std::to_string(a.suggested) + " groups gives the highest silhouette (" +
                   Score(a.silhouette[at]) + ").";
        }
        case groups::Method::DBSCAN:
            if (!std::isfinite(a.knee)) return "The curve has no knee: the distances are about even.";
            return "The knee of the curve (Kneedle) is at " + Figure(a.knee) + ": a good eps to start from.";
    }
    return "";
}

std::string ComponentWords(const std::vector<double>& loadings, const std::vector<std::string>& columns) {
    std::vector<std::pair<double, size_t>> big;
    for (size_t j = 0; j < loadings.size() && j < columns.size(); ++j)
        if (std::fabs(loadings[j]) >= 0.4) big.emplace_back(loadings[j], j);
    if (big.empty()) return "no column stands out";
    std::stable_sort(big.begin(), big.end(), [](const auto& l, const auto& r) { return std::fabs(l.first) > std::fabs(r.first); });
    const bool lead = big.front().first > 0;
    std::vector<std::string> with, against;
    for (const auto& [v, j] : big) ((v > 0) == lead ? with : against).push_back(columns[j]);
    std::string out = with.size() == 1 ? "mostly " + with.front() : Listed(with) + " together";
    if (!against.empty()) out += ", against " + Listed(against);
    return out;
}

std::string ProjectionSentence(const groups::ProjectionAnswer& a) {
    if (!a.ok()) return a.error;
    const std::string sample = a.sampled ? "a seeded sample of " + Number(a.x.size()) + " of " + Number(static_cast<size_t>(a.used)) + " rows"
                                         : Number(a.x.size()) + " rows";
    switch (a.request.method) {
        case groups::ProjectionMethod::PCA: {
            if (a.ratio.size() < 2) return "";
            std::string out = "Two components keep " + Percent(a.ratio[0] + a.ratio[1]) + " of the variation.";
            for (size_t c = 0; c < a.loadings.size() && c < 2; ++c)
                out += " Component " + std::to_string(c + 1) + ": " + ComponentWords(a.loadings[c], a.request.columns) + ".";
            return out;
        }
        case groups::ProjectionMethod::TSNE: {
            std::string out = "t-SNE of " + sample + " (perplexity " + std::to_string(a.request.perplexity) + ")";
            if (std::isfinite(a.kl_divergence)) out += ", KL divergence " + Figure(a.kl_divergence);
            return out + ". Near points are alike; distances between far clusters mean little.";
        }
        case groups::ProjectionMethod::UMAP:
            return "UMAP of " + sample + " (" + std::to_string(a.request.neighbors) + " neighbours, min distance " +
                   Figure(a.request.min_dist) + "). Near points are alike; distances between far clusters mean little.";
    }
    return "";
}

std::string AxisTitle(const groups::ProjectionAnswer& a, int axis) {
    if (a.request.method == groups::ProjectionMethod::PCA) {
        std::string out = "component " + std::to_string(axis + 1);
        if (static_cast<size_t>(axis) < a.ratio.size()) out += " (" + Share(a.ratio[static_cast<size_t>(axis)]) + ")";
        return out;
    }
    return std::string(ProjectionLabel(a.request.method)) + " " + std::to_string(axis + 1);
}

std::string GroupsCsv(const groups::GroupsAnswer& a) {
    std::vector<std::vector<std::string>> rows;
    std::vector<std::string> head = {"group", "rows", "silhouette"};
    for (const auto& c : a.request.columns) head.push_back("mean " + c);
    rows.push_back(head);
    for (const auto& g : a.groups) {
        std::vector<std::string> r = {GroupName(g.label), std::to_string(g.rows), Score(g.silhouette)};
        for (const double m : g.means) r.push_back(Full(m));
        rows.push_back(r);
    }
    rows.push_back({});
    std::vector<std::string> per = {"row", "group", "component 1", "component 2"};
    for (const auto& g : a.groups)
        if (!a.probabilities.empty()) per.push_back("p " + GroupName(g.label));
    rows.push_back(per);
    for (size_t i = 0; i < a.labels.size(); ++i) {
        std::vector<std::string> r = {std::to_string(i + 1), GroupName(a.labels[i]), i < a.x.size() ? Full(a.x[i]) : "",
                                      i < a.y.size() ? Full(a.y[i]) : ""};
        if (i < a.probabilities.size())
            for (const double p : a.probabilities[i]) r.push_back(Full(p));
        rows.push_back(r);
    }
    return explore_view::Csv(rows);
}

std::string ProjectionCsv(const groups::ProjectionAnswer& a) {
    std::vector<std::vector<std::string>> rows;
    std::vector<std::string> head = {AxisTitle(a, 0), AxisTitle(a, 1)};
    if (!a.request.colour.empty()) head.push_back(a.request.colour);
    rows.push_back(head);
    for (size_t i = 0; i < a.x.size(); ++i) {
        std::vector<std::string> r = {Full(a.x[i]), i < a.y.size() ? Full(a.y[i]) : ""};
        if (!a.request.colour.empty()) r.push_back(i < a.colour.size() ? a.colour[i] : "");
        rows.push_back(r);
    }
    return explore_view::Csv(rows);
}

}  // namespace cyxwiz::groups_view
