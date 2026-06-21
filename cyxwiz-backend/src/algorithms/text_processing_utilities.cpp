#include "cyxwiz/text_processing.h"

#include <cctype>
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

    // Remove special characters (keep alphanumeric and basic punctuation)
    if (remove_special) {
        std::string cleaned;
        for (char c : result) {
            if (std::isalnum(static_cast<unsigned char>(c)) ||
                std::isspace(static_cast<unsigned char>(c)) ||
                c == '.' || c == ',' || c == '!' || c == '?' || c == '\'' || c == '-') {
                cleaned += c;
            } else {
                cleaned += ' ';
            }
        }
        result = cleaned;
    }

    return NormalizeWhitespace(result);
}

std::vector<std::string> TextProcessing::SplitSentences(const std::string& text) {
    return TokenizeSentence(text);
}

std::string TextProcessing::ToLowercase(const std::string& text) {
    std::string result;
    result.reserve(text.size());
    for (char c : text) {
        result += std::tolower(static_cast<unsigned char>(c));
    }
    return result;
}

std::string TextProcessing::RemovePunctuation(const std::string& text) {
    std::string result;
    for (char c : text) {
        if (!std::ispunct(static_cast<unsigned char>(c))) {
            result += c;
        }
    }
    return result;
}

std::string TextProcessing::NormalizeWhitespace(const std::string& text) {
    std::string result;
    bool prev_space = true;  // Start true to trim leading

    for (char c : text) {
        if (std::isspace(static_cast<unsigned char>(c))) {
            if (!prev_space) {
                result += ' ';
                prev_space = true;
            }
        } else {
            result += c;
            prev_space = false;
        }
    }

    // Trim trailing space
    if (!result.empty() && result.back() == ' ') {
        result.pop_back();
    }

    return result;
}

} // namespace cyxwiz