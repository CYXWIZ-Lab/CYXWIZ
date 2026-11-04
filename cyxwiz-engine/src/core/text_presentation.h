#pragma once

// Words of Data Studio's Explore tab Text (TOFIX129 D8, approved mockup
// d8_text_mockup.html): the rows read, what each look found as a sentence,
// and the CSV exports. Answers in, text out, so the tab and its test agree.
// No ImGui, no queries.

#include "text_explore.h"

#include <string>
#include <vector>

namespace cyxwiz::text_view {

// "52,681 rows with text (362 empty or missing left out)" / "1,440 rows, all with text".
std::string RowsWords(const text::TextData& d);
// "7 groups in status" / "12 groups in status (the 11 largest, 9 more as other)".
std::string GroupsWords(const text::TextData& d, const std::string& group_column);

std::string WordsSentence(const text::WordsAnswer& a);
std::string TermsSentence(const text::TermsAnswer& a);
std::string SentimentSentence(const text::SentimentAnswer& a);
std::string TokensSentence(const text::TokensAnswer& a);
// `word` and its nearest words (NearestWords).
std::string EmbeddingsSentence(const text::EmbeddingsAnswer& a, const std::string& word,
                               const std::vector<std::pair<std::string, double>>& nearest);

// "negative" / "neutral" / "positive" by the request's thresholds.
const char* SentimentLabel(double compound, const text::SentimentRequest& request);
// "median 56 words per row (95% within 360, longest 5,040)".
std::string LengthWords(const text::LengthFigures& f, const char* unit);

// CSV exports.
std::string WordsCsv(const text::WordsAnswer& a);       // rank, term, count, share
std::string TermsCsv(const text::TermsAnswer& a);       // group, term, mean_tfidf, above_table
std::string SentimentCsv(const text::SentimentAnswer& a);  // row, group, compound, sentiment

}  // namespace cyxwiz::text_view
