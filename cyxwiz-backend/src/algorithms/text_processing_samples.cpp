#include "cyxwiz/text_processing.h"

namespace cyxwiz {

// ============================================================================
// Sample Text Generation
// ============================================================================

std::string TextProcessing::GenerateSampleText(const std::string& type) {
    if (type == "lorem") {
        return "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor "
               "incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud "
               "exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute "
               "irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla "
               "pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia "
               "deserunt mollit anim id est laborum.";
    } else if (type == "news") {
        return "Scientists at the research institute announced a breakthrough discovery today "
               "that could revolutionize our understanding of the universe. The team, led by "
               "Dr. Smith, has been working on this project for over five years. The findings "
               "were published in a leading scientific journal and have already attracted "
               "attention from researchers worldwide. This discovery builds on previous work "
               "and opens new possibilities for future research in the field.";
    } else if (type == "review_positive") {
        return "This product is absolutely amazing! I've never been happier with a purchase. "
               "The quality is outstanding and it exceeded all my expectations. The customer "
               "service was excellent and shipping was incredibly fast. I would highly recommend "
               "this to anyone looking for a great product. Five stars without hesitation! "
               "Best purchase I've made this year. Will definitely buy again.";
    } else if (type == "review_negative") {
        return "Terrible experience with this product. Complete waste of money. It broke within "
               "a week of normal use and customer support was unhelpful. The quality is extremely "
               "poor and nothing like what was advertised. I deeply regret this purchase and "
               "would never recommend it to anyone. Avoid at all costs. Worst product I have "
               "ever bought. Very disappointed and frustrated.";
    } else if (type == "technical") {
        return "Machine learning algorithms process data to identify patterns and make predictions. "
               "Neural networks are a subset of machine learning inspired by the human brain. "
               "Deep learning uses multiple layers of neural networks to learn representations "
               "of data with multiple levels of abstraction. Convolutional neural networks are "
               "particularly effective for image recognition tasks while recurrent neural networks "
               "excel at processing sequential data like text and time series.";
    }

    return "Sample text for analysis. This is a default placeholder text that can be "
           "used for testing various text processing features and algorithms.";
}

std::vector<std::string> TextProcessing::GenerateSampleDocuments(
    int num_docs,
    const std::string& type
) {
    std::vector<std::string> documents;

    if (type == "news") {
        documents = {
            "The stock market experienced significant volatility today as investors reacted to "
            "new economic data. Technology stocks led the decline while energy companies showed gains.",

            "Scientists have discovered a new species of deep-sea fish in the Pacific Ocean. "
            "The creature lives at depths previously thought too extreme for complex life forms.",

            "The government announced new climate change policies aimed at reducing carbon emissions "
            "by fifty percent over the next decade through investments in renewable energy.",

            "A major cybersecurity breach has affected millions of users worldwide. Experts advise "
            "changing passwords immediately and enabling two-factor authentication.",

            "The annual technology conference unveiled several innovative products including "
            "advanced artificial intelligence systems and quantum computing developments."
        };
    } else if (type == "reviews") {
        documents = {
            "Great product, excellent quality. Fast shipping and good customer service. "
            "Would definitely recommend to friends and family.",

            "Disappointing purchase. The item arrived damaged and did not match the description. "
            "Very poor experience overall. Will not buy again.",

            "Average product for the price. Does what it's supposed to do but nothing special. "
            "Might consider other options next time.",

            "Absolutely love this! Best thing I've bought all year. Works perfectly and "
            "the design is beautiful. Five stars!",

            "Terrible quality, broke after one week. Customer service was unhelpful. "
            "Complete waste of money. Avoid this seller."
        };
    } else {
        documents = {
            "Machine learning is transforming industries across the globe.",
            "Data science combines statistics and programming skills.",
            "Deep learning enables computers to learn from large datasets.",
            "Neural networks are inspired by biological brain structure.",
            "Artificial intelligence is revolutionizing healthcare and medicine."
        };
    }

    // Return requested number of documents (with cycling if needed)
    std::vector<std::string> result;
    for (int i = 0; i < num_docs; ++i) {
        result.push_back(documents[i % documents.size()]);
    }

    return result;
}

std::vector<std::string> TextProcessing::GenerateSampleVocabulary(
    int size,
    const std::string& domain
) {
    std::vector<std::string> words;

    if (domain == "tech") {
        words = {"algorithm", "computer", "data", "network", "software", "hardware",
                 "database", "server", "cloud", "security", "encryption", "protocol",
                 "interface", "programming", "code", "debug", "compile", "deploy",
                 "system", "architecture", "framework", "library", "function", "class",
                 "object", "variable", "loop", "condition", "array", "string"};
    } else if (domain == "science") {
        words = {"experiment", "hypothesis", "theory", "research", "analysis", "data",
                 "observation", "measurement", "result", "conclusion", "method", "study",
                 "sample", "control", "variable", "statistics", "significance", "model",
                 "predict", "test", "validate", "replicate", "review", "publish",
                 "discover", "evidence", "proof", "equation", "formula", "constant"};
    } else {
        words = {"the", "be", "to", "of", "and", "a", "in", "that", "have", "it",
                 "for", "not", "on", "with", "he", "as", "you", "do", "at", "this",
                 "but", "his", "by", "from", "they", "we", "say", "her", "she", "or",
                 "an", "will", "my", "one", "all", "would", "there", "their", "what",
                 "so", "up", "out", "if", "about", "who", "get", "which", "go", "me"};
    }

    // Extend or trim to requested size
    std::vector<std::string> result;
    for (int i = 0; i < size; ++i) {
        if (i < static_cast<int>(words.size())) {
            result.push_back(words[i]);
        } else {
            result.push_back("word" + std::to_string(i));
        }
    }

    return result;
}

} // namespace cyxwiz