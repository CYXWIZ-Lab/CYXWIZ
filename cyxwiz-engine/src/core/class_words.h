#pragma once

// Words for class names (TOFIX129 D11): Results > Test and the Model window
// say "a 7, read as 2" and "an Anxiety" the same way.

#include <algorithm>
#include <cctype>
#include <string>

namespace cyxwiz {

// A class name with its article: "a 7", "an 8", "an 11", "an Anxiety", "a one".
inline std::string WithArticle(const std::string& name) {
    if (name.empty()) return name;
    const char first = static_cast<char>(std::tolower(static_cast<unsigned char>(name[0])));
    bool an = std::string("aeiou").find(first) != std::string::npos || first == '8';
    // Vowels said with a consonant: one, unit, user, euro.
    std::string lower;
    for (size_t i = 0; i < name.size() && i < 3; ++i) lower += static_cast<char>(std::tolower(static_cast<unsigned char>(name[i])));
    if (lower.rfind("one", 0) == 0 || lower.rfind("uni", 0) == 0 || lower.rfind("use", 0) == 0 || lower.rfind("eu", 0) == 0)
        an = false;
    // Numbers said with a vowel: 11, 18, 11xx, 18xx (eleven, eighteen, eleven hundred).
    const bool digits = std::all_of(name.begin(), name.end(), [](char c) { return std::isdigit(static_cast<unsigned char>(c)) != 0; });
    if (digits && (name.rfind("11", 0) == 0 || name.rfind("18", 0) == 0) && (name.size() == 2 || name.size() == 4))
        an = true;
    return (an ? "an " : "a ") + name;
}

}  // namespace cyxwiz
