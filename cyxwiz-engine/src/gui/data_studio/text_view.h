#pragma once

// Data Studio > Explore > Text (TOFIX129 D8, approved mockup boards 1-5;
// replaces the Tokenization, Word Frequency, TF-IDF Analysis, Word
// Embeddings and Sentiment Analysis windows). One text column, optionally
// split by a group column, in five looks: Words (the commonest words and
// pairs, per group, words per row), Terms (TF-IDF: what sets each group
// apart, a row's terms and the rows that read most like it), Sentiment
// (VADER on every row, by group), Tokens (what the Tokenizer node makes of
// the rows and what a length limit cuts) and Embeddings (word vectors
// learned from the column, the nearest words, a map). Worked out in the
// background (core/text_explore, core/text_presentation).

#include "explore_tab.h"
#include "../word_map_view.h"
#include "../../core/text_explore.h"

#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace cyxwiz {

class TextView {
public:
    enum class Look { Words, Terms, Sentiment, Tokens, Embeddings };

    void Render(const ExploreContext& context);
    // Opens on this look (Tools > Text, Data > Tokenization).
    void SelectLook(Look look) { look_ = look; }
    bool IsRunning() const {
        return words_job_.Running() || terms_job_.Running() || sentiment_job_.Running() || tokens_job_.Running() || embeddings_job_.Running() ||
               map_view_.IsRunning();
    }
    // The word map's window (when open); every frame, also when the tab is hidden.
    void RenderWindows() { map_view_.DrawWindow(); }

    // Into the graph: a node of `type` (its import name) named `name` with
    // `params`, after the data in use (`step`) or as a branch reading it;
    // returns what happened.
    std::function<std::string(const std::string& type, const std::string& name, std::map<std::string, std::string> params, bool step)> add_to_graph;

private:
    void RenderControls(const DatasetProfile& profile);
    void RenderLookSettings();
    void RenderWords(const ExploreContext& context);
    void RenderTerms(const ExploreContext& context);
    void RenderSentiment(const ExploreContext& context);
    void RenderTokens(const ExploreContext& context);
    void RenderEmbeddings(const ExploreContext& context);
    void RenderRowsLine(const text::TextData& d);
    void Message(const std::string& text, bool error);
    // What shows while a look runs (the text, and Stop); false when stopped.
    template <typename T> bool Waiting(ExploreJob<T>& job, const std::string& key, const char* text);
    void Export(const std::string& name, const std::string& csv);
    std::string BaseKey(const std::string& dataset) const;
    int64_t TableRow() const { return row_ > 0 ? row_ - 1 : -1; }

    std::shared_ptr<const DatasetProfile> seen_;
    Look look_ = Look::Words;
    text::TextRequest base_;
    text::WordsRequest words_;
    text::TermsRequest terms_;
    text::SentimentRequest sentiment_;
    text::TokensRequest tokens_;
    text::EmbeddingsRequest embeddings_;
    int row_ = 0;                  // the row looked at (1-based as shown; 0: the first with text)
    std::string learn_key_;        // Embeddings learn on request (Learn), not on every change
    bool all_rows_ = false;        // Embeddings: every row instead of a sample
    char query_[128] = "";
    int nearest_count_ = 8;
    std::string nearest_for_;      // the query and answer the nearest words were found for
    std::vector<std::pair<std::string, double>> nearest_;
    std::string nearest_error_;
    std::string map_source_;       // the answer the word map shows
    WordMapView map_view_{"text"};
    std::string message_;
    bool message_error_ = false;
    std::string stopped_key_;
    ExploreJob<text::WordsAnswer> words_job_;
    ExploreJob<text::TermsAnswer> terms_job_;
    ExploreJob<text::SentimentAnswer> sentiment_job_;
    ExploreJob<text::TokensAnswer> tokens_job_;
    ExploreJob<text::EmbeddingsAnswer> embeddings_job_;
};

}  // namespace cyxwiz
