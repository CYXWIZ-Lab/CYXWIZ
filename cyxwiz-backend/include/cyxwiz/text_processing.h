// SPDX-License-Identifier: Apache-2.0
// Part of the CyxWiz plugin SDK (see cmake/sdk/NOTICE).
#pragma once

#include "api_export.h"
#include <vector>
#include <string>
#include <map>
#include <set>
#include <utility>

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
    std::string method;                             // How the vectors were made (the caller's name)
    std::vector<std::pair<std::string, double>> similar_words;  // For similarity query
    double similarity_score = 0.0;                  // Similarity result
    bool success = false;
    std::string error_message;
};

// The VADER sentiment of a text (TextProcessing::Vader).
struct CYXWIZ_API VaderScores {
    double negative = 0.0;  // the shares of the text's weight that read negative,
    double neutral = 0.0;   // neutral and positive (they sum to 1; 0 for a text
    double positive = 0.0;  // with no words)
    double compound = 0.0;  // the valences summed and normalised to -1 .. 1
    std::vector<std::pair<std::string, double>> words;  // each word or emoticon as VADER splits
                                                        // the text, with its valence after the rules
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
     * The words of a text: lower-cased runs of letters and digits of any
     * script (Python's str.isalnum), two or more long, an apostrophe between
     * two of them kept ("don't", "i've"; a typographic apostrophe reads as
     * one). The rule Data Studio's Text tab and the TF-IDF / Count Vectorizer
     * nodes share; Python: re.finditer(r"[^\W_]+(?:'[^\W_]+)*", text).
     */
    static std::vector<std::string> Words(const std::string& text);

    /**
     * The word n-grams of tokens, ngram_min to ngram_max words long: the
     * shortest first, each in text order, joined by one space (scikit-learn's
     * word n-grams); empty for an empty or reversed range
     */
    static std::vector<std::string> NGrams(const std::vector<std::string>& tokens, int ngram_min, int ngram_max);

    /**
     * The terms of a text as the TF-IDF and Count Vectorizer nodes and Data
     * Studio's Text tab count them: Words, the English stop words left out
     * first when asked, then NGrams
     */
    static std::vector<std::string> Terms(const std::string& text, bool english_stop_words, int ngram_min, int ngram_max);

    /**
     * Remove the stopwords (GetStopwords) from tokens, compared lower-cased
     * @param tokens Input tokens
     * @param language Language for stopwords ("english")
     * @return Filtered tokens
     */
    static std::vector<std::string> RemoveStopwords(
        const std::vector<std::string>& tokens,
        const std::string& language = "english"
    );

    /**
     * The Porter (1980) stem of a word, lower-cased first: NLTK's
     * PorterStemmer(mode=ORIGINAL_ALGORITHM)
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
     * The English stop words: scikit-learn's ENGLISH_STOP_WORDS (318) plus
     * their contractions ("don't", "i'm", ...), lower case
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

    // ==================== Sentiment ====================

    /**
     * The VADER sentiment of a text (Hutto and Gilbert, 2014): its lexicon,
     * negation, boosters and dampeners, capitals, "but", "least", the special
     * idioms, emoticons, emojis and ! / ?, as vaderSentiment 3.3.2 gives them
     * (unrounded).
     */
    static VaderScores Vader(const std::string& text);

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

private:
    // Helper functions
    static std::vector<std::string> TokenizeWhitespace(const std::string& text);
    static std::vector<std::string> TokenizeWord(const std::string& text, bool lowercase, bool remove_punct);
    static std::vector<std::string> TokenizeSentence(const std::string& text);
    static std::vector<std::string> TokenizeNgram(const std::string& text, int n, bool lowercase);


    static double L2Norm(const std::vector<double>& vec);
    static double L1Norm(const std::vector<double>& vec);
    static void NormalizeVector(std::vector<double>& vec, const std::string& norm);
};

} // namespace cyxwiz
