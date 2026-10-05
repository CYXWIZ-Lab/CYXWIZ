#pragma once

// Common English words left out of a text column's Top words and phrases
// (TOFIX134 P3 text, board 19; a switch on the widget keeps them).

#include <string>
#include <vector>

namespace cyxwiz::dashboard {

const std::vector<std::string>& EnglishStopWords();  // sorted
bool IsEnglishStopWord(const std::string& word);

}  // namespace cyxwiz::dashboard
