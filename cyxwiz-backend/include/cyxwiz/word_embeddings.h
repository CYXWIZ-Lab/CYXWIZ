#pragma once

// Word vectors (TOFIX152): pretrained files read as gensim reads them
// (KeyedVectors.load_word2vec_format), and a table lined up with a token
// vocabulary for the Embedding layer. Files are read on the host; the matrix
// work runs on the selected ArrayFire device.

#include "cyxwiz/api_export.h"

#include <cstddef>
#include <string>
#include <vector>

namespace cyxwiz {

// Words and their vectors, row-major [words.size(), dim], in file order.
struct CYXWIZ_API WordVectors {
    std::vector<std::string> words;
    std::vector<float> values;
    int dim = 0;

    bool success = false;
    std::string error_message;
};

// How AlignWordVectors fills the rows of a vocabulary.
struct CYXWIZ_API WordAlignmentOptions {
    bool lowercase_fallback = true;   // a word not found as written is looked up in lowercase
    std::string missing = "zeros";    // rows without a vector: "zeros" or "random" (normal, the vectors' std)
    std::string unknown = "mean";     // the unknown-token row: "mean" of the vectors, or as a missing word
    int pad_id = 0;                   // row of zeros (-1: none)
    int unknown_id = 1;               // -1: none
    unsigned seed = 0;                // for "random"
};

// A vocabulary's rows, row-major [vocab.size(), dim], and how they were found.
struct CYXWIZ_API WordAlignment {
    std::vector<float> values;
    int dim = 0;
    size_t exact = 0;      // found as written
    size_t lowercase = 0;  // found in lowercase
    size_t missing = 0;    // no vector (PAD and UNK are not counted)
    std::vector<std::string> missing_words;  // in vocabulary order

    bool success = false;
    std::string error_message;
};

class CYXWIZ_API WordEmbeddings {
public:
    // Text format: one word per line followed by its values, separated by
    // spaces. A first line of two integers ("count dim", word2vec / fastText
    // .vec) is a header; without it (GloVe .txt) the width comes from the first
    // row. A repeated word keeps its first vector. `max_words` > 0 reads at most
    // that many words.
    static WordVectors LoadText(const std::string& path, size_t max_words = 0);
    // word2vec binary: "count dim\n", then per word the word, a space and dim
    // little-endian float32 values.
    static WordVectors LoadBinary(const std::string& path, size_t max_words = 0);
    // .bin is binary; anything else is text.
    static WordVectors Load(const std::string& path, size_t max_words = 0);

    // The vectors in the order of `vocabulary` (token id = position).
    static WordAlignment Align(const WordVectors& vectors, const std::vector<std::string>& vocabulary,
                               const WordAlignmentOptions& options = {});
};

}  // namespace cyxwiz
