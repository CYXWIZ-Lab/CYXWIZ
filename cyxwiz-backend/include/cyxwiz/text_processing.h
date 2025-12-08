#pragma once

#include "api_export.h"
#include <vector>
#include <string>
#include <map>
#include <set>
#include <utility>
#include <random>

namespace cyxwiz {

// ============================================================================
// Result Structures
// ============================================================================

struct CYXWIZ_API TokenizationResult {
    std::vector<std::string> tokens;              // Resulting tokens
    std::vector<std::pair<int, int>> spans;       // Character offsets (start, end)
    int token_count = 0;                          // Total number of tokens
    int unique_count = 0;                         // Number of unique tokens
    double avg_token_length = 0.0;                // Average token length
    std::string method;                           // Tokenization method used
    bool success = false;
    std::string error_message;
};

struct CYXWIZ_API WordFrequencyResult {
    std::vector<std::pair<std::string, int>> frequencies;  // (word, count) sorted desc
    int total_words = 0;                          // Total word count
    int unique_words = 0;                         // Unique word count
    double type_token_ratio = 0.0;                // Vocabulary diversity (unique/total)
    std::map<int, int> length_distribution;       // word_length -> count
    double avg_word_length = 0.0;                 // Average word length
    int max_frequency = 0;                        // Highest frequency
    std::string most_common_word;                 // Most frequent word
    bool success = false;
    std::string error_message;
};

struct CYXWIZ_API TFIDFResult {
    std::vector<std::vector<double>> tfidf_matrix;  // docs x terms matrix
    std::vector<std::string> vocabulary;            // Term list (columns)
    std::vector<double> idf_scores;                 // IDF score for each term
    std::vector<std::vector<std::pair<std::string, double>>> doc_top_terms;  // Top terms per doc
    std::vector<std::vector<double>> similarity_matrix;  // Doc-doc cosine similarity
    int num_documents = 0;                          // Number of documents
    int vocab_size = 0;                             // Vocabulary size
    std::string normalization;                      // Normalization used
    bool success = false;
    std::string error_message;
};

struct CYXWIZ_API EmbeddingResult {
    std::vector<std::vector<double>> embeddings;    // words x dimensions matrix
    std::vector<std::string> words;                 // Vocabulary (rows)
    int embedding_dim = 0;                          // Embedding dimension
    std::string method;                             // Method used (onehot, random)
    std::vector<std::pair<std::string, double>> similar_words;  // For similarity query
    double similarity_score = 0.0;                  // Similarity result
    bool success = false;
    std::string error_message;
};

struct CYXWIZ_API SentimentResult {
    double polarity = 0.0;                          // -1.0 (negative) to 1.0 (positive)
    double subjectivity = 0.0;                      // 0.0 (objective) to 1.0 (subjective)
    std::string label;                              // "positive", "negative", "neutral"
    double confidence = 0.0;                        // 0.0 to 1.0
    std::vector<std::pair<std::string, double>> word_scores;  // Word-level sentiment
    int positive_count = 0;                         // Number of positive words
    int negative_count = 0;                         // Number of negative words
    int neutral_count = 0;                          // Number of neutral words
    std::string analysis;                           // Detailed text analysis
    bool success = false;
    std::string error_message;
};

// ============================================================================
// Text Processing Class
// ============================================================================

class CYXWIZ_API TextProcessing {
public:
    // ==================== Tokenization ====================

    /**
     * Tokenize text into tokens
     * @param text Input text
     * @param method "whitespace", "word", "sentence", "ngram"
     * @param ngram_n N for n-gram tokenization
     * @param lowercase Convert to lowercase
     * @param remove_punctuation Remove punctuation marks
     * @return TokenizationResult
     */
    static TokenizationResult Tokenize(
        const std::string& text,
        const std::string& method = "word",
        int ngram_n = 2,
        bool lowercase = true,
        bool remove_punctuation = true
    );

    /**
     * Remove stopwords from tokens
     * @param tokens Input tokens
     * @param language Language for stopwords ("english")
     * @return Filtered tokens
     */
    static std::vector<std::string> RemoveStopwords(
        const std::vector<std::string>& tokens,
        const std::string& language = "english"
    );

    /**
     * Apply Porter stemming to a word
     * @param word Input word
     * @return Stemmed word
     */
    static std::string Stem(const std::string& word);

    /**
     * Apply stemming to multiple words
     * @param words Input words
     * @return Stemmed words
     */
    static std::vector<std::string> StemWords(const std::vector<std::string>& words);

    /**
     * Get English stopwords set
     * @return Set of stopwords
     */
    static const std::set<std::string>& GetStopwords();

    // ==================== Word Frequency ====================

    /**
     * Compute word frequency from text
     * @param text Input text
     * @param top_n Number of top words to return (-1 for all)
     * @param remove_stopwords Remove common words
     * @param min_word_length Minimum word length
     * @return WordFrequencyResult
     */
    static WordFrequencyResult ComputeWordFrequency(
        const std::string& text,
        int top_n = 50,
        bool remove_stopwords = true,
        int min_word_length = 2
    );

    /**
     * Compute word frequency from tokens
     * @param tokens Input tokens
     * @param top_n Number of top words to return
     * @return WordFrequencyResult
     */
    static WordFrequencyResult ComputeWordFrequencyFromTokens(
        const std::vector<std::string>& tokens,
        int top_n = 50
    );

    /**
     * Build vocabulary from documents
     * @param documents List of documents
     * @param min_freq Minimum frequency to include
     * @param max_vocab_size Maximum vocabulary size (-1 for unlimited)
     * @return Word to index mapping
     */
    static std::map<std::string, int> BuildVocabulary(
        const std::vector<std::string>& documents,
        int min_freq = 1,
        int max_vocab_size = -1
    );

    // ==================== TF-IDF ====================

    /**
     * Compute TF-IDF matrix for documents
     * @param documents List of documents
     * @param use_idf Use inverse document frequency
     * @param smooth_idf Add 1 to document frequencies for smoothing
     * @param norm Normalization ("l1", "l2", "none")
     * @return TFIDFResult
     */
    static TFIDFResult ComputeTFIDF(
        const std::vector<std::string>& documents,
        bool use_idf = true,
        bool smooth_idf = true,
        const std::string& norm = "l2"
    );

    /**
     * Compute term frequency for a document
     * @param document Input document
     * @param vocabulary Vocabulary list
     * @return TF vector
     */
    static std::vector<double> ComputeTF(
        const std::string& document,
        const std::vector<std::string>& vocabulary
    );

    /**
     * Compute inverse document frequency
     * @param documents List of documents
     * @param vocabulary Vocabulary list
     * @param smooth Add 1 to document counts
     * @return IDF vector
     */
    static std::vector<double> ComputeIDF(
        const std::vector<std::string>& documents,
        const std::vector<std::string>& vocabulary,
        bool smooth = true
    );

    /**
     * Compute cosine similarity between two vectors
     * @param vec1 First vector
     * @param vec2 Second vector
     * @return Cosine similarity (-1 to 1)
     */
    static double CosineSimilarity(
        const std::vector<double>& vec1,
        const std::vector<double>& vec2
    );

    /**
     * Compute document similarity matrix
     * @param tfidf_matrix TF-IDF matrix (docs x terms)
     * @return Similarity matrix (docs x docs)
     */
    static std::vector<std::vector<double>> ComputeSimilarityMatrix(
        const std::vector<std::vector<double>>& tfidf_matrix
    );

    // ==================== Word Embeddings ====================

    /**
     * Create one-hot embeddings for vocabulary
     * @param vocabulary List of words
     * @return EmbeddingResult
     */
    static EmbeddingResult CreateOneHotEmbeddings(
        const std::vector<std::string>& vocabulary
    );

    /**
     * Create random embeddings for vocabulary
     * @param vocabulary List of words
     * @param embedding_dim Dimension of embeddings
     * @param seed Random seed (-1 for random)
     * @return EmbeddingResult
     */
    static EmbeddingResult CreateRandomEmbeddings(
        const std::vector<std::string>& vocabulary,
        int embedding_dim = 50,
        int seed = -1
    );

    /**
     * Find similar words using cosine similarity
     * @param word Query word
     * @param embeddings Embedding result
     * @param top_n Number of similar words to return
     * @return EmbeddingResult with similar_words populated
     */
    static EmbeddingResult FindSimilarWords(
        const std::string& word,
        const EmbeddingResult& embeddings,
        int top_n = 10
    );

    /**
     * Compute similarity between two words
     * @param word1 First word
     * @param word2 Second word
     * @param embeddings Embedding result
     * @return Cosine similarity
     */
    static double WordSimilarity(
        const std::string& word1,
        const std::string& word2,
        const EmbeddingResult& embeddings
    );

    // ==================== Sentiment Analysis ====================

    /**
     * Analyze sentiment of text using lexicon-based approach
     * @param text Input text
     * @param method "simple", "vader", "afinn"
     * @return SentimentResult
     */
    static SentimentResult AnalyzeSentiment(
        const std::string& text,
        const std::string& method = "simple"
    );

    /**
     * Get sentiment lexicon
     * @param name Lexicon name ("simple", "afinn")
     * @return Word to score mapping
     */
    static const std::map<std::string, double>& GetSentimentLexicon(
        const std::string& name = "simple"
    );

    /**
     * Compute polarity score from tokens and lexicon
     * @param tokens Input tokens
     * @param lexicon Sentiment lexicon
     * @return Polarity score
     */
    static double ComputePolarityScore(
        const std::vector<std::string>& tokens,
        const std::map<std::string, double>& lexicon
    );

    // ==================== Text Utilities ====================

    /**
     * Clean text by removing URLs, emails, special characters
     * @param text Input text
     * @param remove_urls Remove URLs
     * @param remove_emails Remove email addresses
     * @param remove_numbers Remove numbers
     * @param remove_special Remove special characters
     * @return Cleaned text
     */
    static std::string CleanText(
        const std::string& text,
        bool remove_urls = true,
        bool remove_emails = true,
        bool remove_numbers = false,
        bool remove_special = true
    );

    /**
     * Split text into sentences
     * @param text Input text
     * @return List of sentences
     */
    static std::vector<std::string> SplitSentences(const std::string& text);

    /**
     * Convert text to lowercase
     * @param text Input text
     * @return Lowercase text
     */
    static std::string ToLowercase(const std::string& text);

    /**
     * Remove punctuation from text
     * @param text Input text
     * @return Text without punctuation
     */
    static std::string RemovePunctuation(const std::string& text);

    /**
     * Normalize whitespace (collapse multiple spaces, trim)
     * @param text Input text
     * @return Normalized text
     */
    static std::string NormalizeWhitespace(const std::string& text);

    // ==================== Sample Text Generation ====================

    /**
     * Generate sample text
     * @param type "lorem", "news", "review_positive", "review_negative", "technical"
     * @return Sample text
     */
    static std::string GenerateSampleText(const std::string& type = "lorem");

    /**
     * Generate sample documents
     * @param num_docs Number of documents
     * @param type Document type ("news", "reviews", "technical")
     * @return List of documents
     */
    static std::vector<std::string> GenerateSampleDocuments(
        int num_docs,
        const std::string& type = "news"
    );

    /**
     * Generate sample vocabulary
     * @param size Vocabulary size
     * @param domain Domain ("general", "tech", "science")
     * @return List of words
     */
    static std::vector<std::string> GenerateSampleVocabulary(
        int size = 100,
        const std::string& domain = "general"
    );

private:
    // Helper functions
    static std::vector<std::string> TokenizeWhitespace(const std::string& text);
    static std::vector<std::string> TokenizeWord(const std::string& text, bool lowercase, bool remove_punct);
    static std::vector<std::string> TokenizeSentence(const std::string& text);
    static std::vector<std::string> TokenizeNgram(const std::string& text, int n, bool lowercase);

    static void InitStopwords();
    static void InitSentimentLexicons();

    static double L2Norm(const std::vector<double>& vec);
    static double L1Norm(const std::vector<double>& vec);
    static void NormalizeVector(std::vector<double>& vec, const std::string& norm);
};

} // namespace cyxwiz
