#pragma once

// The word map (TOFIX129 D8, approved mockup d8_word_map_mockup.html, M-1 ..
// M-8): word vectors in 3D, the word looked up and its nearest words marked
// and joined, labels that never overlap, clusters or groups in colour. One
// widget for both hosts: Data Studio's Explore > Text > Embeddings (a turning
// preview and "Open the word map") and the Word Embeddings node dialog's Map
// tab. The window opens over its host and may be docked anywhere, float as a
// window of its own (A8) or fill the screen. Layouts in the background (core/word_map).

#include "../core/word_map.h"
#include "data_studio/explore_tab.h"

#include <imgui.h>

#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace cyxwiz {

class WordMapView {
public:
    explicit WordMapView(std::string id);

    struct Source {
        std::shared_ptr<const WordVectors> vectors;
        std::string title;  // "statement: word2vec skip-gram, 16,321 words"
        std::string key;    // changes when the vectors change
        // "Leans to a group" (empty: not offered): the groups, each word's rows per group, the rows per group.
        std::vector<std::string> groups;
        std::map<std::string, std::vector<int64_t>> word_rows;
        std::vector<int64_t> group_rows;
    };
    void SetSource(Source source);
    bool HasSource() const { return source_.vectors != nullptr; }
    const std::string& SourceKey() const { return source_.key; }

    // A small turning view of the word and its neighbours (the host's preview).
    void DrawPreview(ImVec2 size);
    // The window, when open; every frame, also when the host is hidden.
    void DrawWindow();
    // Opens the window and brings it to the front (the first time over its
    // host in the main window; its tab docks it or moves it out).
    void Open();
    bool IsOpen() const { return open_; }
    bool IsRunning() const { return job_.Running(); }
    void LookUp(const std::string& word);

private:
    void DrawControls();
    void DrawMap(ImVec2 size, bool full);
    void Ask();
    std::string Title() const;
    void FindNeighbours();
    ImVec4 ColourOf(size_t i) const;

    std::string id_;
    Source source_;
    bool open_ = false;
    // What is shown.
    char query_[128] = "";
    std::string word_;
    int neighbours_count_ = 10;
    std::vector<std::pair<std::string, double>> neighbours_;
    std::string neighbours_for_;
    std::string neighbours_error_;
    word_map::Layout layout_ = word_map::Layout::TSNE;
    int words_ = 1500;
    int clusters_ = 8;
    bool flat_ = false;
    int colour_ = 0;      // 0 clusters, 1 leans to a group, 2 one colour
    bool more_labels_ = true;
    bool spin_ = false;
    float spin_azimuth_ = -45.0f;
    bool refit_ = true;   // set the view limits on the next frame (the whole map, or the word's neighbourhood)
    bool focus_word_ = false;
    std::string export_pending_;
    // The map: the last placed (shown while a new one is placed).
    std::shared_ptr<word_map::MapAnswer> map_;
    std::unordered_map<std::string, size_t> map_index_;
    std::vector<int> leans_;
    std::string asked_key_;
    std::string shown_key_;      // the key of map_
    std::string words_for_;      // what words_list_ was made for (the source, the count, the extra words)
    std::vector<std::string> words_list_;
    std::string extra_key_;      // the words past the commonest, as they enter the key
    ExploreJob<word_map::MapAnswer> job_;
    std::string error_;
};

}  // namespace cyxwiz
