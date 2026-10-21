#include "cyxwiz/text_processing.h"

#include <algorithm>
#include <iomanip>
#include <sstream>

namespace cyxwiz {

static std::map<std::string, double> g_simple_lexicon;
static std::map<std::string, double> g_afinn_lexicon;
static bool g_lexicons_initialized = false;

void TextProcessing::InitSentimentLexicons() {
    if (g_lexicons_initialized) return;

    // Simple sentiment lexicon (common words)
    g_simple_lexicon = {
        // Positive words
        {"good", 0.7}, {"great", 0.9}, {"excellent", 1.0}, {"amazing", 0.95},
        {"wonderful", 0.9}, {"fantastic", 0.95}, {"awesome", 0.9}, {"best", 0.85},
        {"love", 0.8}, {"like", 0.5}, {"happy", 0.8}, {"joy", 0.85}, {"beautiful", 0.8},
        {"perfect", 0.95}, {"nice", 0.6}, {"pleasant", 0.65}, {"positive", 0.7},
        {"brilliant", 0.9}, {"superb", 0.9}, {"outstanding", 0.9}, {"exceptional", 0.9},
        {"impressive", 0.75}, {"remarkable", 0.8}, {"incredible", 0.85}, {"delightful", 0.8},
        {"enjoyable", 0.7}, {"satisfying", 0.7}, {"recommend", 0.6}, {"fun", 0.65},
        {"exciting", 0.75}, {"pleased", 0.7}, {"glad", 0.65}, {"thankful", 0.7},
        {"grateful", 0.75}, {"appreciate", 0.6}, {"friendly", 0.6}, {"helpful", 0.65},

        // Negative words
        {"bad", -0.7}, {"terrible", -0.95}, {"awful", -0.9}, {"horrible", -0.95},
        {"worst", -1.0}, {"hate", -0.9}, {"dislike", -0.6}, {"sad", -0.7},
        {"angry", -0.8}, {"disappointed", -0.75}, {"poor", -0.6}, {"negative", -0.7},
        {"ugly", -0.7}, {"boring", -0.6}, {"annoying", -0.7}, {"frustrating", -0.75},
        {"useless", -0.8}, {"waste", -0.7}, {"fail", -0.75}, {"failed", -0.75},
        {"failure", -0.8}, {"problem", -0.5}, {"issue", -0.4}, {"broken", -0.7},
        {"wrong", -0.6}, {"error", -0.5}, {"mistake", -0.55}, {"difficult", -0.4},
        {"hard", -0.3}, {"pain", -0.6}, {"painful", -0.7}, {"unfortunately", -0.5},
        {"regret", -0.65}, {"sorry", -0.4}, {"disappoint", -0.7}, {"unhappy", -0.75},
        {"upset", -0.65}, {"worried", -0.5}, {"concern", -0.4}, {"fear", -0.6}
    };

    // AFINN-like lexicon (more comprehensive)
    g_afinn_lexicon = g_simple_lexicon;  // Start with simple lexicon
    // Add more words with nuanced scores
    g_afinn_lexicon["abandon"] = -0.4;
    g_afinn_lexicon["ability"] = 0.2;
    g_afinn_lexicon["able"] = 0.2;
    g_afinn_lexicon["abuse"] = -0.6;
    g_afinn_lexicon["accept"] = 0.3;
    g_afinn_lexicon["accident"] = -0.4;
    g_afinn_lexicon["accomplish"] = 0.5;
    g_afinn_lexicon["achieve"] = 0.5;
    g_afinn_lexicon["advantage"] = 0.4;
    g_afinn_lexicon["adventure"] = 0.4;
    g_afinn_lexicon["afraid"] = -0.4;
    g_afinn_lexicon["agree"] = 0.3;
    g_afinn_lexicon["alert"] = 0.1;
    g_afinn_lexicon["alone"] = -0.2;
    g_afinn_lexicon["amaze"] = 0.6;
    g_afinn_lexicon["anger"] = -0.6;
    g_afinn_lexicon["annoy"] = -0.5;
    g_afinn_lexicon["anxiety"] = -0.5;
    g_afinn_lexicon["anxious"] = -0.4;
    g_afinn_lexicon["apologize"] = -0.2;
    g_afinn_lexicon["approve"] = 0.4;
    g_afinn_lexicon["attack"] = -0.5;
    g_afinn_lexicon["attractive"] = 0.5;
    g_afinn_lexicon["avoid"] = -0.2;
    g_afinn_lexicon["award"] = 0.5;

    g_lexicons_initialized = true;
}

// ============================================================================
// Sentiment Analysis
// ============================================================================

SentimentResult TextProcessing::AnalyzeSentiment(
    const std::string& text,
    const std::string& method
) {
    SentimentResult result;

    if (text.empty()) {
        result.label = "neutral";
        result.success = true;
        return result;
    }

    try {
        InitSentimentLexicons();

        // Get lexicon
        const auto& lexicon = (method == "afinn") ? g_afinn_lexicon : g_simple_lexicon;

        // Tokenize
        auto tokens = TokenizeWord(text, true, true);

        // Compute sentiment
        double total_score = 0.0;
        int scored_words = 0;

        for (const auto& token : tokens) {
            auto it = lexicon.find(token);
            if (it != lexicon.end()) {
                double score = it->second;
                result.word_scores.emplace_back(token, score);
                total_score += score;
                scored_words++;

                if (score > 0.1) result.positive_count++;
                else if (score < -0.1) result.negative_count++;
                else result.neutral_count++;
            }
        }

        // Compute polarity (-1 to 1)
        if (scored_words > 0) {
            result.polarity = total_score / scored_words;
            // Clamp to [-1, 1]
            result.polarity = std::max(-1.0, std::min(1.0, result.polarity));
        }

        // Compute subjectivity (0 to 1) - based on proportion of sentiment words
        result.subjectivity = tokens.empty() ? 0.0 :
            static_cast<double>(scored_words) / tokens.size();

        // Determine label
        if (result.polarity > 0.1) {
            result.label = "positive";
        } else if (result.polarity < -0.1) {
            result.label = "negative";
        } else {
            result.label = "neutral";
        }

        // Compute confidence based on consistency
        if (result.positive_count + result.negative_count > 0) {
            int dominant = std::max(result.positive_count, result.negative_count);
            int total = result.positive_count + result.negative_count;
            result.confidence = static_cast<double>(dominant) / total;
        } else {
            result.confidence = 0.5;  // Neutral confidence
        }

        // Generate analysis text
        std::ostringstream analysis;
        analysis << "Analyzed " << tokens.size() << " tokens. ";
        analysis << "Found " << scored_words << " sentiment words: ";
        analysis << result.positive_count << " positive, ";
        analysis << result.negative_count << " negative, ";
        analysis << result.neutral_count << " neutral. ";
        analysis << "Overall sentiment: " << result.label << " ";
        analysis << "(polarity: " << std::fixed << std::setprecision(2) << result.polarity << ", ";
        analysis << "confidence: " << std::fixed << std::setprecision(0) << (result.confidence * 100) << "%).";
        result.analysis = analysis.str();

        result.success = true;
    } catch (const std::exception& e) {
        result.error_message = std::string("Sentiment error: ") + e.what();
    }

    return result;
}

const std::map<std::string, double>& TextProcessing::GetSentimentLexicon(
    const std::string& name
) {
    InitSentimentLexicons();
    return (name == "afinn") ? g_afinn_lexicon : g_simple_lexicon;
}

double TextProcessing::ComputePolarityScore(
    const std::vector<std::string>& tokens,
    const std::map<std::string, double>& lexicon
) {
    double total = 0.0;
    int count = 0;

    for (const auto& token : tokens) {
        std::string lower = ToLowercase(token);
        auto it = lexicon.find(lower);
        if (it != lexicon.end()) {
            total += it->second;
            count++;
        }
    }

    return count > 0 ? total / count : 0.0;
}

} // namespace cyxwiz