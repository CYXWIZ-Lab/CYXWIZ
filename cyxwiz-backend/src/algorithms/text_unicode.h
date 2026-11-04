#pragma once

// Python's reading of characters, for the text code that is checked against
// Python (the word rule, Porter, VADER): str.isalnum, isupper, islower /
// istitle, isspace and lower of one code point, from the tables
// generate_text_fixtures.py writes from Python's Unicode database. The
// tokenizers keep their own classes in text_utf8.h.

#include <string>

namespace cyxwiz::text_unicode {

bool IsAlnum(char32_t cp);
bool IsUpper(char32_t cp);
bool IsLowerOrTitle(char32_t cp);
bool IsSpace(char32_t cp);
// The lower case of one code point; kept when Python lowers it to several.
char32_t Lower(char32_t cp);

// UTF-8 to code points (a byte that starts no valid sequence reads as U+FFFD)
// and back.
std::u32string Decode(const std::string& text);
std::string Encode(const std::u32string& text);

// str.lower and str.isupper of a whole string.
std::u32string LowerString(const std::u32string& text);
bool IsUpperString(const std::u32string& text);

}  // namespace cyxwiz::text_unicode
