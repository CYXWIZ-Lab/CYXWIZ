#include "cyxwiz/text_processing.h"
#include "text_utf8.h"

#include <regex>

namespace cyxwiz {

// ============================================================================
// Text Utilities
// ============================================================================

std::string TextProcessing::CleanText(
    const std::string& text,
    bool remove_urls,
    bool remove_emails,
    bool remove_numbers,
    bool remove_special
) {
    std::string result = text;

    // Remove URLs
    if (remove_urls) {
        std::regex url_pattern(R"(https?://\S+|www\.\S+)");
        result = std::regex_replace(result, url_pattern, " ");
    }

    // Remove emails
    if (remove_emails) {
        std::regex email_pattern(R"(\S+@\S+\.\S+)");
        result = std::regex_replace(result, email_pattern, " ");
    }

    // Remove numbers
    if (remove_numbers) {
        std::regex num_pattern(R"(\d+)");
        result = std::regex_replace(result, num_pattern, " ");
    }

    // Remove special characters (keep letters and digits of any script,
    // spaces and basic punctuation), by UTF-8 character.
    if (remove_special) {
        std::string cleaned;
        for (size_t i = 0; i < result.size();) {
            char32_t cp = 0;
            size_t length = 1;
            const bool valid = text_utf8::Next(result, i, cp, length);
            if (valid && (text_utf8::IsWordCharacter(cp) || text_utf8::IsSpace(cp) || cp == '.' || cp == ',' ||
                          cp == '!' || cp == '?' || cp == '\'' || cp == '-')) {
                cleaned.append(result, i, length);
            } else {
                cleaned += ' ';
            }
            i += length;
        }
        result = cleaned;
    }

    return NormalizeWhitespace(result);
}

std::vector<std::string> TextProcessing::SplitSentences(const std::string& text) {
    return TokenizeSentence(text);
}

std::string TextProcessing::ToLowercase(const std::string& text) {
    // By UTF-8 code point and without the C runtime's locale (std::tolower on
    // each byte turned the lead byte of "\xC3\xA9" into another character once
    // Python set the user's code page). Bytes that are not UTF-8 are kept.
    std::string result;
    result.reserve(text.size());
    for (size_t i = 0; i < text.size();) {
        char32_t cp = 0;
        size_t length = 1;
        if (text_utf8::Next(text, i, cp, length)) text_utf8::Append(result, text_utf8::Lower(cp));
        else result += text[i];
        i += length;
    }
    return result;
}

std::string TextProcessing::RemovePunctuation(const std::string& text) {
    // Punctuation and symbols of any script (TokenizeWord's punctuation), by
    // UTF-8 character; letters, digits, spaces and control characters stay.
    std::string result;
    for (size_t i = 0; i < text.size();) {
        char32_t cp = 0;
        size_t length = 1;
        const bool valid = text_utf8::Next(text, i, cp, length);
        if (valid && (text_utf8::IsWordCharacter(cp) || text_utf8::IsSpace(cp) || cp < 0x20 || cp == 0x7F))
            result.append(text, i, length);
        i += length;
    }
    return result;
}

std::string TextProcessing::NormalizeWhitespace(const std::string& text) {
    std::string result;
    bool prev_space = true;  // Start true to trim leading

    // By UTF-8 character: a byte of a multi-byte letter is never a space.
    for (size_t i = 0; i < text.size();) {
        char32_t cp = 0;
        size_t length = 1;
        if (text_utf8::Next(text, i, cp, length) && text_utf8::IsSpace(cp)) {
            if (!prev_space) {
                result += ' ';
                prev_space = true;
            }
        } else {
            result.append(text, i, length);
            prev_space = false;
        }
        i += length;
    }

    // Trim trailing space
    if (!result.empty() && result.back() == ' ') {
        result.pop_back();
    }

    return result;
}

} // namespace cyxwiz
