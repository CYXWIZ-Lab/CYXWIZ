#pragma once

// The word map (TOFIX129 D8, approved mockup d8_word_map_mockup.html, M-1 ..
// M-8): word vectors placed in 3D so that words used in the same places sit
// together. Shared by Data Studio's Explore > Text > Embeddings and the Word
// Embeddings node dialog (gui/word_map_view). Pure data in and out; the
// layouts are the backend's (t-SNE, UMAP, PCA on the device) and the clusters
// its k-means. Checked in test_word_map.

#include <cyxwiz/word_embeddings.h>

#include <array>
#include <cstdint>
#include <functional>
#include <map>
#include <string>
#include <vector>

namespace cyxwiz::word_map {

enum class Layout { TSNE, UMAP, PCA };
const char* LayoutName(Layout layout);  // "t-SNE", "UMAP", "PCA"

// The words to place: the `n` commonest of the vectors (they come most
// frequent first) that are words of three letters or more and not common
// English words, then `extra` (the word looked up and its neighbours) when
// they have a vector and are not there yet.
std::vector<std::string> MapWords(const WordVectors& vectors, size_t n, const std::vector<std::string>& extra);

struct MapRequest {
    std::vector<std::string> words;  // MapWords
    Layout layout = Layout::TSNE;
    int clusters = 8;                // k-means k (2 .. 20)
    unsigned seed = 1;
};

struct MapAnswer {
    std::string error;
    bool ok() const { return error.empty(); }
    MapRequest request;
    std::vector<std::string> words;                 // as placed (those without a vector left out)
    std::vector<std::array<double, 3>> xyz;         // centred; 98% within -1 .. 1 (a few outliers beyond)
    std::vector<int> cluster;                       // per word, 0 .. k-1
    std::vector<std::string> cluster_names;         // per cluster: its three commonest words
    std::array<double, 3> share{};                  // PCA: the share of the spread each axis holds
};
// The layout of the request's words (unit vectors, so t-SNE and UMAP measure
// as the cosine does) and their clusters; `stop` ends it early ("Cancelled.").
MapAnswer Place(const WordVectors& vectors, const MapRequest& request, const std::function<bool()>& stop = {},
                const std::function<void(float, const std::string&)>& progress = {});

// The group each word leans to: the group where it is in the most rows for
// the group's size, when that is at least `lift` times its share overall and
// in `min_rows` of the group's rows; -1 otherwise. `word_rows` holds, per
// word, its rows in each group (as `group_rows`).
std::vector<int> LeansTo(const std::vector<std::string>& words, const std::map<std::string, std::vector<int64_t>>& word_rows,
                         const std::vector<int64_t>& group_rows, double lift = 2.0, int64_t min_rows = 30);

// The view around a word and its neighbours: the centre and half the width
// that holds them, with room around (at least 0.35).
struct Focus {
    std::array<double, 3> centre{};
    double half = 1.1;
};
Focus FocusOn(const MapAnswer& map, const std::string& word, const std::vector<std::string>& neighbours);

}  // namespace cyxwiz::word_map
