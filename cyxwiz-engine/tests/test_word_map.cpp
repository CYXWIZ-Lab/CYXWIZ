// The word map (TOFIX129 D8, M-1 .. M-8): the words placed (the commonest
// without the common English ones, then the word looked up and its
// neighbours), the layouts (t-SNE, UMAP and PCA of the backend) keeping
// three clusters of vectors apart, k-means naming them by their commonest
// words, "leans to a group", and the view around a word.

#include "../src/core/word_map.h"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <map>
#include <random>
#include <set>
#include <string>
#include <vector>

using namespace cyxwiz;

namespace {

void Check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        std::exit(1);
    }
}

// Three clusters of 20 words in 10 dimensions, interleaved (commonest first):
// "pxaa", "qxaa", "rxaa", "pxab", ...; and "the" (a common English word) first.
WordVectors Clusters() {
    WordVectors v;
    v.dim = 10;
    std::mt19937 rng(7);
    std::normal_distribution<float> noise(0.0f, 0.15f);
    const auto add = [&](const std::string& w, int cluster) {
        v.words.push_back(w);
        for (int k = 0; k < 10; ++k) v.values.push_back((k == cluster * 3 ? 1.0f : 0.0f) + noise(rng));
    };
    add("the", 0);
    for (int i = 0; i < 20; ++i)
        for (int c = 0; c < 3; ++c) add(std::string(1, static_cast<char>('p' + c)) + "xa" + std::string(1, static_cast<char>('a' + i)), c);
    v.success = true;
    return v;
}

int Cluster(const std::string& w) { return w[0] - 'p'; }

// The share of words whose k-means cluster holds mostly their own kind.
double Purity(const word_map::MapAnswer& a) {
    std::map<int, std::map<int, int>> count;
    for (size_t i = 0; i < a.words.size(); ++i) ++count[a.cluster[i]][Cluster(a.words[i])];
    int right = 0;
    for (const auto& [c, kinds] : count) {
        int best = 0;
        for (const auto& [k, n] : kinds) best = std::max(best, n);
        right += best;
    }
    return static_cast<double>(right) / static_cast<double>(a.words.size());
}

// The mean distance between words of one kind over the mean between kinds.
double Spread(const word_map::MapAnswer& a) {
    double within = 0, between = 0;
    int nw = 0, nb = 0;
    for (size_t i = 0; i < a.words.size(); ++i)
        for (size_t j = i + 1; j < a.words.size(); ++j) {
            const double d = std::hypot(a.xyz[i][0] - a.xyz[j][0], a.xyz[i][1] - a.xyz[j][1], a.xyz[i][2] - a.xyz[j][2]);
            if (Cluster(a.words[i]) == Cluster(a.words[j])) within += d, ++nw;
            else between += d, ++nb;
        }
    return (within / nw) / (between / nb);
}

}  // namespace

int main() {
    const WordVectors v = Clusters();
    // The words placed.
    const auto words = word_map::MapWords(v, 30, {"pxaf", "pxat", "nope", "the"});
    Check(words.size() == 32, "map words: 30 commonest, then the extra words not among them (" + std::to_string(words.size()) + ")");
    Check(words.front() == "pxaa" && words[30] == "pxat" && words.back() == "the",
          "map words: \"the\" left out of the commonest, but a word looked up is always placed");
    Check(std::set<std::string>(words.begin(), words.end()).size() == words.size(), "map words: each once");

    word_map::MapRequest request;
    request.words = word_map::MapWords(v, 100, {});
    request.clusters = 3;
    for (const word_map::Layout layout : {word_map::Layout::PCA, word_map::Layout::TSNE, word_map::Layout::UMAP}) {
        request.layout = layout;
        const word_map::MapAnswer a = word_map::Place(v, request);
        const std::string name = word_map::LayoutName(layout);
        Check(a.ok(), name + ": " + a.error);
        Check(a.words.size() == 60 && a.xyz.size() == 60 && a.cluster.size() == 60, name + ": every word placed");
        // Scaled so that 98% of the words lie within -1 .. 1 (outliers beyond).
        size_t within = 0;
        double extent = 0;
        for (const auto& p : a.xyz) {
            double r = 0;
            for (const double x : p) r = std::max(r, std::fabs(x));
            extent = std::max(extent, r);
            within += r <= 1.0 + 1e-9 ? 1 : 0;
        }
        Check(within >= 58 && extent >= 1.0 - 1e-9, name + ": 98% within -1 .. 1 (" + std::to_string(within) + " of 60)");
        Check(Purity(a) > 0.95, name + ": k-means finds the three kinds (" + std::to_string(Purity(a)) + ")");
        // 3D t-SNE (its kernel has 2 degrees of freedom) separates less on 60 points: scikit-learn's
        // TSNE(n_components=3, perplexity=19, learning_rate="auto") gives 0.57 on this data.
        const double bound = layout == word_map::Layout::TSNE ? 0.7 : 0.6;
        Check(Spread(a) < bound, name + ": the kinds stay apart (" + std::to_string(Spread(a)) + ")");
        Check(a.cluster_names.size() == 3, name + ": three names");
        for (const std::string& n : a.cluster_names) Check(n.rfind("xaa") == 1, name + ": a cluster named by its commonest word first: " + n);
        if (layout == word_map::Layout::PCA) Check(a.share[0] >= a.share[1] && a.share[1] >= a.share[2] && a.share[0] + a.share[1] + a.share[2] > 0.5, "PCA: the shares");
    }
    Check(!word_map::Place(v, {{"pxaa", "pxab"}, word_map::Layout::PCA, 2, 1}).ok(), "too few words");
    bool stopped = false;
    Check(word_map::Place(v, request, [&] { return stopped = true; }).error == "Cancelled.", "stop");
    Check(stopped, "stop asked");

    // Leans to a group: twice as common in it, in 30 of its rows or more.
    const std::vector<int64_t> group_rows = {1000, 100};
    const std::map<std::string, std::vector<int64_t>> word_rows = {
        {"w1", {10, 40}},   // 40 / 100 vs 50 / 1100 overall: leans to group 1
        {"w2", {500, 60}},  // group 1: 0.6 / (560 / 1100) = 1.18: not twice
        {"w3", {2, 20}},    // twice, but in 20 rows only
    };
    const auto leans = word_map::LeansTo({"w1", "w2", "w3", "w4"}, word_rows, group_rows);
    Check(leans == std::vector<int>({1, -1, -1, -1}), "leans to a group");

    // The view around a word and its neighbours.
    request.layout = word_map::Layout::PCA;
    const word_map::MapAnswer pca = word_map::Place(v, request);
    const word_map::Focus f = word_map::FocusOn(pca, "pxaa", {"pxab", "pxac"});
    Check(f.half >= 0.35 && f.half <= 1.1, "focus: the half width within 0.35 .. 1.1");
    Check(std::fabs(f.centre[0] - pca.xyz[0][0]) < 1e-12, "focus: centred on the word");
    Check(word_map::FocusOn(pca, "nope", {}).half == 1.1, "focus: an unknown word shows the whole map");

    std::cout << "word map: words placed, PCA / t-SNE / UMAP keep the kinds apart, k-means names, leans to a group, focus. OK\n";
    return 0;
}
