#include "word_map.h"

#include "node_executors/word_embeddings_operator.h"
#include "tsne_learning_rate.h"

#include <cyxwiz/clustering.h>
#include <cyxwiz/dimensionality_reduction.h>
#include <cyxwiz/text_processing.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <set>
#include <unordered_map>
#include <unordered_set>

namespace cyxwiz::word_map {
namespace {

bool Letters(const std::string& w) {
    size_t letters = 0;
    for (const unsigned char c : w) {
        if (std::isdigit(c) || c == '_' || c == '\'') return false;
        if (c >= 0x80 || std::isalpha(c)) ++letters;
    }
    return letters >= 3;
}

}  // namespace

const char* LayoutName(Layout layout) {
    switch (layout) {
        case Layout::TSNE: return "t-SNE";
        case Layout::UMAP: return "UMAP";
        case Layout::PCA: return "PCA";
    }
    return "";
}

std::string QueryWord(const std::string& typed) {
    std::string word;
    for (const std::string& w : WordEmbeddingsOperator::SplitWords(typed)) word += (word.empty() ? "" : " ") + w;
    return word.empty() ? TextProcessing::ToLowercase(typed) : word;
}

std::vector<std::string> MapWords(const WordVectors& vectors, size_t n, const std::vector<std::string>& extra) {
    const std::set<std::string>& stop = TextProcessing::GetStopwords();
    std::vector<std::string> words;
    std::unordered_set<std::string> taken;
    for (const std::string& w : vectors.words) {
        if (words.size() >= n) break;
        if (!Letters(w) || stop.count(w)) continue;
        words.push_back(w);
        taken.insert(w);
    }
    std::unordered_set<std::string> known(vectors.words.begin(), vectors.words.end());
    for (const std::string& w : extra)
        if (known.count(w) && taken.insert(w).second) words.push_back(w);
    return words;
}

MapAnswer Place(const WordVectors& vectors, const MapRequest& request, const std::function<bool()>& stop,
                const std::function<void(float, const std::string&)>& progress) {
    MapAnswer a;
    a.request = request;
    if (vectors.dim <= 0) return a.error = "There are no vectors to map.", a;
    std::unordered_map<std::string, size_t> row;
    row.reserve(vectors.words.size());
    for (size_t i = 0; i < vectors.words.size(); ++i) row.emplace(vectors.words[i], i);
    std::vector<std::vector<double>> data;
    for (const std::string& w : request.words) {
        const auto it = row.find(w);
        if (it == row.end()) continue;
        const float* v = vectors.values.data() + it->second * static_cast<size_t>(vectors.dim);
        double norm = 0.0;
        for (int k = 0; k < vectors.dim; ++k) norm += static_cast<double>(v[k]) * v[k];
        norm = std::sqrt(norm);
        if (!(norm > 0.0)) continue;
        std::vector<double> unit(static_cast<size_t>(vectors.dim));
        for (int k = 0; k < vectors.dim; ++k) unit[static_cast<size_t>(k)] = v[k] / norm;
        data.push_back(std::move(unit));
        a.words.push_back(w);
    }
    if (a.words.size() < 10) return a.error = "Too few words to map (" + std::to_string(a.words.size()) + "; 10 or more).", a;
    if (stop && stop()) return a.error = "Cancelled.", a;

    std::vector<std::vector<double>> placed;
    if (request.layout == Layout::PCA) {
        const PCAResult pca = DimensionalityReduction::ComputePCA(data, 3);
        if (!pca.success) return a.error = "PCA: " + pca.error_message, a;
        placed = pca.transformed;
        for (size_t k = 0; k < 3 && k < pca.explained_variance_ratio.size(); ++k) a.share[k] = pca.explained_variance_ratio[k];
    } else if (request.layout == Layout::UMAP) {
        const UMAPResult umap = DimensionalityReduction::ComputeUMAP(data, 3, std::min<int>(15, static_cast<int>(data.size()) - 1));
        if (!umap.success) return a.error = "UMAP: " + umap.error_message, a;
        placed = umap.embeddings;
    } else {
        const int perplexity = std::max(2, std::min(30, static_cast<int>(data.size() - 1) / 3));
        // The progress callback is the only way back into the loop: it throws to stop it.
        struct Stopped {};
        tSNEResult tsne;
        try {
            tsne = DimensionalityReduction::ComputetSNE(data, 3, perplexity, TsneLearningRate(0.0, data.size()), 1000, [&](int iteration, double) {
                if (progress) progress(static_cast<float>(iteration) / 1000.0f, "t-SNE iteration " + std::to_string(iteration) + " of 1,000");
                if (stop && stop()) throw Stopped{};
            });
        } catch (const Stopped&) {
            return a.error = "Cancelled.", a;
        }
        if (!tsne.success) return a.error = "t-SNE: " + tsne.error_message, a;
        placed = tsne.embeddings;
    }
    if (stop && stop()) return a.error = "Cancelled.", a;
    if (placed.size() != a.words.size()) return a.error = "The layout returned " + std::to_string(placed.size()) + " points for " + std::to_string(a.words.size()) + " words.", a;
    // Centred, and scaled so that 98% of the words lie within -1 .. 1 (a few
    // outlying words would otherwise squeeze the rest into the middle); the
    // outliers lie beyond and show when the view zooms out.
    std::array<double, 3> mean{};
    for (const auto& p : placed)
        for (size_t k = 0; k < 3 && k < p.size(); ++k) mean[k] += p[k] / static_cast<double>(placed.size());
    std::vector<double> reach;
    reach.reserve(placed.size());
    for (const auto& p : placed) {
        double r = 0.0;
        for (size_t k = 0; k < 3 && k < p.size(); ++k) r = std::max(r, std::fabs(p[k] - mean[k]));
        reach.push_back(r);
    }
    std::sort(reach.begin(), reach.end());
    double extent = reach[std::min(reach.size() - 1, static_cast<size_t>(0.98 * static_cast<double>(reach.size())))];
    if (!(extent > 0.0)) extent = reach.back() > 0.0 ? reach.back() : 1.0;
    for (const auto& p : placed) {
        std::array<double, 3> q{};
        for (size_t k = 0; k < 3 && k < p.size(); ++k) q[k] = (p[k] - mean[k]) / extent;
        a.xyz.push_back(q);
    }

    // Clusters of the vectors themselves, named by their commonest words (the words come commonest first).
    const int k = std::clamp(request.clusters, 2, std::min(20, static_cast<int>(a.words.size()) / 2));
    const KMeansResult km = Clustering::KMeans(data, k, 300, "kmeans++", 4, 1e-4, request.seed);
    if (!km.success || km.labels.size() != a.words.size()) return a.error = "k-means: " + km.error_message, a;
    a.cluster = km.labels;
    a.cluster_names.assign(static_cast<size_t>(k), "");
    std::vector<int> named(static_cast<size_t>(k), 0);
    for (size_t i = 0; i < a.words.size(); ++i) {
        const size_t c = static_cast<size_t>(std::clamp(a.cluster[i], 0, k - 1));
        if (named[c] >= 3) continue;
        a.cluster_names[c] += (named[c] ? ", " : "") + a.words[i];
        ++named[c];
    }
    return a;
}

std::vector<int> LeansTo(const std::vector<std::string>& words, const std::map<std::string, std::vector<int64_t>>& word_rows,
                         const std::vector<int64_t>& group_rows, double lift, int64_t min_rows) {
    std::vector<int> out(words.size(), -1);
    int64_t total = 0;
    for (const int64_t g : group_rows) total += g;
    if (total <= 0) return out;
    for (size_t i = 0; i < words.size(); ++i) {
        const auto it = word_rows.find(words[i]);
        if (it == word_rows.end()) continue;
        int64_t n = 0;
        for (const int64_t r : it->second) n += r;
        if (n <= 0) continue;
        double best = 0.0;
        int at = -1;
        for (size_t g = 0; g < group_rows.size() && g < it->second.size(); ++g) {
            if (group_rows[g] <= 0) continue;
            const double l = (static_cast<double>(it->second[g]) / static_cast<double>(group_rows[g])) /
                             (static_cast<double>(n) / static_cast<double>(total));
            if (l > best) {
                best = l;
                at = static_cast<int>(g);
            }
        }
        if (at >= 0 && best >= lift && it->second[static_cast<size_t>(at)] >= min_rows) out[i] = at;
    }
    return out;
}

Focus FocusOn(const MapAnswer& map, const std::string& word, const std::vector<std::string>& neighbours) {
    Focus f;
    std::unordered_map<std::string, size_t> at;
    for (size_t i = 0; i < map.words.size(); ++i) at.emplace(map.words[i], i);
    const auto q = at.find(word);
    if (q == at.end()) return f;
    f.centre = map.xyz[q->second];
    double half = 0.0;
    for (const std::string& n : neighbours) {
        const auto it = at.find(n);
        if (it == at.end()) continue;
        for (size_t k = 0; k < 3; ++k) half = std::max(half, std::fabs(map.xyz[it->second][k] - f.centre[k]));
    }
    f.half = std::clamp(half * 1.3, 0.35, 1.1);
    return f;
}

}  // namespace cyxwiz::word_map
