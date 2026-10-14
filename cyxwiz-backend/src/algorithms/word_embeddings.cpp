// Word vectors (TOFIX152): word2vec / GloVe / fastText files read as gensim's
// KeyedVectors.load_word2vec_format reads them, and AlignWordVectors, which
// lines them up with a token vocabulary on the device.

#ifdef _WIN32
#define NOMINMAX
#endif

#include "cyxwiz/word_embeddings.h"

#include "arrayfire_host_materialization.h"
#include "arrayfire_indexed.h"

#include <arrayfire.h>

#include <algorithm>
#include <limits>
#include <charconv>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <fstream>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace cyxwiz {

namespace {

WordVectors Failed(const std::string& message) {
    WordVectors out;
    out.error_message = message;
    return out;
}

std::vector<std::string_view> SplitSpaces(std::string_view line) {
    std::vector<std::string_view> parts;
    size_t i = 0;
    while (i < line.size()) {
        while (i < line.size() && (line[i] == ' ' || line[i] == '\t')) ++i;
        const size_t start = i;
        while (i < line.size() && line[i] != ' ' && line[i] != '\t') ++i;
        if (i > start) parts.push_back(line.substr(start, i - start));
    }
    return parts;
}

bool ParseFloat(std::string_view text, float& value) {
    const auto result = std::from_chars(text.data(), text.data() + text.size(), value);
    return result.ec == std::errc() && result.ptr == text.data() + text.size();
}

bool ParseCount(std::string_view text, long long& value) {
    const auto result = std::from_chars(text.data(), text.data() + text.size(), value);
    return result.ec == std::errc() && result.ptr == text.data() + text.size() && value >= 0;
}

// Adds a word unless it is already there (gensim keeps the first vector).
void AddWord(WordVectors& out, std::unordered_set<std::string>& seen, std::string word, const float* values) {
    if (!seen.insert(word).second) return;
    out.words.push_back(std::move(word));
    out.values.insert(out.values.end(), values, values + out.dim);
}

std::string AsciiLower(const std::string& word) {
    std::string lower = word;
    for (char& c : lower)
        if (c >= 'A' && c <= 'Z') c = static_cast<char>(c - 'A' + 'a');
    return lower;
}

}  // namespace

WordVectors WordEmbeddings::LoadText(const std::string& path, size_t max_words) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return Failed("Cannot open the vector file " + path);
    WordVectors out;
    std::unordered_set<std::string> seen;
    std::vector<float> row;
    std::string line;
    size_t line_number = 0, read = 0;
    long long header_count = -1;
    while (std::getline(in, line)) {
        ++line_number;
        if (!line.empty() && line.back() == '\r') line.pop_back();
        std::string_view view(line);
        if (line_number == 1 && view.size() >= 3 && static_cast<unsigned char>(view[0]) == 0xEF &&
            static_cast<unsigned char>(view[1]) == 0xBB && static_cast<unsigned char>(view[2]) == 0xBF)
            view.remove_prefix(3);  // UTF-8 byte order mark
        const auto parts = SplitSpaces(view);
        if (parts.empty()) continue;
        long long count = 0, dim = 0;
        if (line_number == 1 && parts.size() == 2 && ParseCount(parts[0], count) && ParseCount(parts[1], dim)) {
            if (dim <= 0) return Failed("The header gives a vector size of 0");
            header_count = count;
            out.dim = static_cast<int>(dim);
            continue;
        }
        if (out.dim == 0) out.dim = static_cast<int>(parts.size()) - 1;  // GloVe: the first row sets the width
        if (out.dim <= 0) return Failed("Line " + std::to_string(line_number) + " has a word but no values");
        if (parts.size() < static_cast<size_t>(out.dim) + 1)
            return Failed("Line " + std::to_string(line_number) + " has " + std::to_string(parts.size() - 1) +
                          " values; the vectors have " + std::to_string(out.dim));
        // The last dim parts are the values; the rest is the word (a word may
        // contain spaces, as a few GloVe entries do).
        const size_t word_parts = parts.size() - static_cast<size_t>(out.dim);
        std::string word(parts[0]);
        for (size_t p = 1; p < word_parts; ++p) {
            word += ' ';
            word += parts[p];
        }
        row.resize(static_cast<size_t>(out.dim));
        for (int d = 0; d < out.dim; ++d) {
            if (!ParseFloat(parts[word_parts + static_cast<size_t>(d)], row[static_cast<size_t>(d)]))
                return Failed("Line " + std::to_string(line_number) + ": '" +
                              std::string(parts[word_parts + static_cast<size_t>(d)]) + "' is not a number");
        }
        AddWord(out, seen, std::move(word), row.data());
        ++read;
        if (max_words > 0 && read >= max_words) break;
        if (header_count >= 0 && static_cast<long long>(read) >= header_count) break;
    }
    if (out.words.empty()) return Failed("The vector file has no words: " + path);
    out.success = true;
    return out;
}

WordVectors WordEmbeddings::LoadBinary(const std::string& path, size_t max_words) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return Failed("Cannot open the vector file " + path);
    std::string header;
    if (!std::getline(in, header)) return Failed("The binary vector file has no header");
    if (!header.empty() && header.back() == '\r') header.pop_back();
    const auto parts = SplitSpaces(header);
    long long count = 0, dim = 0;
    if (parts.size() != 2 || !ParseCount(parts[0], count) || !ParseCount(parts[1], dim) || dim <= 0)
        return Failed("The binary header must be 'count dim', got '" + header + "'");
    WordVectors out;
    out.dim = static_cast<int>(dim);
    std::unordered_set<std::string> seen;
    std::vector<float> row(static_cast<size_t>(dim));
    const long long limit = max_words > 0 ? std::min<long long>(count, static_cast<long long>(max_words)) : count;
    for (long long i = 0; i < limit; ++i) {
        std::string word;
        int c = 0;
        while ((c = in.get()) != EOF && (c == '\n' || c == '\r')) {
        }
        while (c != EOF && c != ' ') {
            word.push_back(static_cast<char>(c));
            c = in.get();
        }
        if (c == EOF) return Failed("The binary file ends after " + std::to_string(i) + " of " + std::to_string(count) + " words");
        // Little-endian float32, as gensim writes and reads them on x86 / ARM.
        in.read(reinterpret_cast<char*>(row.data()), static_cast<std::streamsize>(row.size() * sizeof(float)));
        if (!in) return Failed("The binary file ends inside the vector of '" + word + "'");
        AddWord(out, seen, std::move(word), row.data());
    }
    if (out.words.empty()) return Failed("The vector file has no words: " + path);
    out.success = true;
    return out;
}

WordVectors WordEmbeddings::Load(const std::string& path, size_t max_words) {
    const auto dot = path.find_last_of('.');
    const std::string extension = dot == std::string::npos ? "" : AsciiLower(path.substr(dot));
    return extension == ".bin" ? LoadBinary(path, max_words) : LoadText(path, max_words);
}

bool WordEmbeddings::SaveText(const WordVectors& vectors, const std::string& path, std::string* error) {
    if (!vectors.success || vectors.dim <= 0 ||
        vectors.values.size() != vectors.words.size() * static_cast<size_t>(vectors.dim)) {
        if (error) *error = "The word vectors are not loaded";
        return false;
    }
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        if (error) *error = "Cannot write " + path;
        return false;
    }
    out << vectors.words.size() << ' ' << vectors.dim << '\n';
    char buffer[32];
    for (size_t r = 0; r < vectors.words.size(); ++r) {
        out << vectors.words[r];
        for (int d = 0; d < vectors.dim; ++d) {
            // Shortest text that reads back as the same float32.
            const auto result = std::to_chars(buffer, buffer + sizeof(buffer),
                                              vectors.values[r * static_cast<size_t>(vectors.dim) + static_cast<size_t>(d)]);
            out << ' ' << std::string_view(buffer, static_cast<size_t>(result.ptr - buffer));
        }
        out << '\n';
    }
    if (!out) {
        if (error) *error = "Cannot write " + path;
        return false;
    }
    return true;
}

std::vector<std::pair<std::string, double>> WordEmbeddings::Nearest(const WordVectors& vectors,
                                                                     const std::string& word, int k,
                                                                     std::string* error) {
    std::vector<std::pair<std::string, double>> nearest;
    if (!vectors.success || vectors.dim <= 0 || k <= 0 ||
        vectors.values.size() != vectors.words.size() * static_cast<size_t>(vectors.dim))
        return nearest;
    const auto found = std::find(vectors.words.begin(), vectors.words.end(), word);
    if (found == vectors.words.end()) return nearest;
    const unsigned self = static_cast<unsigned>(found - vectors.words.begin());
    try {
        const dim_t n = static_cast<dim_t>(vectors.words.size()), dim = vectors.dim;
        const af::array table = af::array(dim, n, vectors.values.data());  // [dim, n]
        const af::array norms = af::max(af::sqrt(af::sum(table * table, 0)), 1e-12f);
        const af::array unit = table / af::tile(norms, static_cast<unsigned>(dim), 1);
        af::array similarity = af::flat(af::matmul(af::lookup(unit, af::array(1, &self), 1), unit, AF_MAT_TRANS, AF_MAT_NONE));
        similarity(self) = -2.0f;  // leave the word itself out
        af::array sorted, order;
        SortWithIndex(sorted, order, similarity, 0, false);
        const dim_t take = std::min<dim_t>(static_cast<dim_t>(k), n - 1);
        if (take <= 0) return nearest;
        std::vector<float> values(static_cast<size_t>(take));
        std::vector<unsigned> rows(static_cast<size_t>(take));
        af::array top_values = sorted(af::seq(0, static_cast<double>(take) - 1));
        af::array top_rows = order(af::seq(0, static_cast<double>(take) - 1)).as(u32);
        top_values.eval();
        top_rows.eval();
        MaterializeArrayFireToHost(top_values, values.data(), ArrayFireHostSyncCategory::OutputMaterialization,
                                   "WordEmbeddings::Nearest", "top_k_similarity");
        MaterializeArrayFireToHost(top_rows, rows.data(), ArrayFireHostSyncCategory::OutputMaterialization,
                                   "WordEmbeddings::Nearest", "top_k_rows");
        for (dim_t i = 0; i < take; ++i)
            nearest.emplace_back(vectors.words[rows[static_cast<size_t>(i)]], static_cast<double>(values[static_cast<size_t>(i)]));
    } catch (const af::exception& e) {
        nearest.clear();
        if (error) *error = std::string("WordEmbeddings::Nearest failed on the ArrayFire device: ") + e.what();
    }
    return nearest;
}

std::vector<float> WordEmbeddings::RowMeans(const std::vector<float>& table, int dim, const std::vector<int>& tokens,
                                            int tokens_per_row, std::string* error) {
    const auto fail = [&](const std::string& message) {
        if (error) *error = message;
        return std::vector<float>{};
    };
    if (dim <= 0 || table.empty() || table.size() % static_cast<size_t>(dim) != 0)
        return fail("The vector table must be [rows, dim]");
    if (tokens_per_row <= 0 || tokens.empty() || tokens.size() % static_cast<size_t>(tokens_per_row) != 0)
        return fail("The tokens must be [rows, tokens_per_row]");
    const int table_rows = static_cast<int>(table.size() / static_cast<size_t>(dim));
    for (const int id : tokens)
        if (id < -1 || id >= table_rows) return fail("A token id is outside the vector table");
    // The gathered [rows, tokens_per_row, dim] block is indexed with 32-bit
    // offsets on the device: a larger one reads out of bounds.
    if (tokens.size() * static_cast<size_t>(dim) > static_cast<size_t>(std::numeric_limits<int>::max()))
        return fail("The block of rows x tokens x dim is too large for one device step; average fewer rows at a time");
    try {
        const dim_t rows = static_cast<dim_t>(tokens.size() / static_cast<size_t>(tokens_per_row));
        const dim_t width = tokens_per_row, d = dim;
        const af::array vectors = af::array(d, static_cast<dim_t>(table_rows), table.data()).T();      // [table_rows, d]
        const af::array ids = af::array(width, rows, tokens.data()).T();                            // [rows, width]
        const af::array valid = (ids >= 0).as(f32);
        const af::array count = af::max(af::sum(valid, 1), 1.0f);
        const af::array gathered = af::lookup(vectors, af::flat(af::select(ids >= 0, ids, 0.0)), 0);  // [rows*width, d]
        const af::array masked = af::moddims(gathered, rows, width, d) *
                                 af::tile(af::moddims(valid, rows, width, 1), 1, 1, static_cast<unsigned>(d));
        af::array mean = af::moddims(af::sum(masked, 1), rows, d) / af::tile(count, 1, static_cast<unsigned>(d));
        af::array row_major = mean.T();
        row_major.eval();
        std::vector<float> out(static_cast<size_t>(rows * d));
        MaterializeArrayFireToHost(row_major, out.data(), ArrayFireHostSyncCategory::OutputMaterialization,
                                   "WordEmbeddings::RowMeans", "row_major_rows_by_dim");
        return out;
    } catch (const af::exception& e) {
        return fail(std::string("WordEmbeddings::RowMeans failed on the ArrayFire device: ") + e.what());
    }
}

WordAlignment WordEmbeddings::Align(const WordVectors& vectors, const std::vector<std::string>& vocabulary,
                                    const WordAlignmentOptions& options) {
    WordAlignment out;
    const auto fail = [&](const std::string& message) {
        out = WordAlignment{};
        out.error_message = message;
        return out;
    };
    if (!vectors.success || vectors.dim <= 0 || vectors.words.empty() ||
        vectors.values.size() != vectors.words.size() * static_cast<size_t>(vectors.dim))
        return fail("The word vectors are not loaded");
    if (vocabulary.empty()) return fail("The vocabulary is empty");
    if (options.missing != "zeros" && options.missing != "random")
        return fail("Missing words take 'zeros' or 'random'");
    if (options.unknown != "mean" && options.unknown != "missing")
        return fail("The unknown token takes 'mean' or 'missing'");

    // Which file row each token takes: as written first, then in lowercase.
    std::unordered_map<std::string, int> exact, lower;
    for (size_t r = 0; r < vectors.words.size(); ++r) {
        exact.emplace(vectors.words[r], static_cast<int>(r));
        lower.emplace(AsciiLower(vectors.words[r]), static_cast<int>(r));
    }
    const int m = static_cast<int>(vocabulary.size());
    std::vector<unsigned> found_ids, found_rows, missing_ids;
    bool unknown_is_mean = false;
    for (int id = 0; id < m; ++id) {
        if (id == options.pad_id) continue;
        if (id == options.unknown_id && options.unknown == "mean") {
            unknown_is_mean = true;
            continue;
        }
        const std::string& token = vocabulary[static_cast<size_t>(id)];
        if (const auto it = exact.find(token); it != exact.end()) {
            found_ids.push_back(static_cast<unsigned>(id));
            found_rows.push_back(static_cast<unsigned>(it->second));
            ++out.exact;
        } else if (const auto low = options.lowercase_fallback ? lower.find(AsciiLower(token)) : lower.end();
                   low != lower.end()) {
            found_ids.push_back(static_cast<unsigned>(id));
            found_rows.push_back(static_cast<unsigned>(low->second));
            ++out.lowercase;
        } else {
            missing_ids.push_back(static_cast<unsigned>(id));
            if (id != options.unknown_id) {
                ++out.missing;
                out.missing_words.push_back(token);
            }
        }
    }

    try {
        const dim_t n = static_cast<dim_t>(vectors.words.size()), dim = vectors.dim;
        // Host rows are row-major [n, dim]: column-major that is [dim, n].
        const af::array table = af::array(dim, n, vectors.values.data());
        af::array aligned = af::constant(0.0f, dim, static_cast<dim_t>(m), f32);
        if (!found_ids.empty()) {
            const af::array rows(static_cast<dim_t>(found_rows.size()), found_rows.data());
            const af::array ids(static_cast<dim_t>(found_ids.size()), found_ids.data());
            aligned(af::span, ids) = af::lookup(table, rows, 1);
        }
        if (!missing_ids.empty() && options.missing == "random") {
            const float scale = af::stdev<float>(af::flat(table), AF_VARIANCE_POPULATION);
            af::randomEngine engine(AF_RANDOM_ENGINE_DEFAULT, options.seed);
            const af::array ids(static_cast<dim_t>(missing_ids.size()), missing_ids.data());
            aligned(af::span, ids) = af::randn(af::dim4(dim, static_cast<dim_t>(missing_ids.size())), f32, engine) * scale;
        }
        if (unknown_is_mean && options.unknown_id >= 0 && options.unknown_id < m)
            aligned(af::span, options.unknown_id) = af::mean(table, 1);
        out.values.resize(static_cast<size_t>(m) * static_cast<size_t>(dim));
        aligned.eval();
        MaterializeArrayFireToHost(aligned, out.values.data(), ArrayFireHostSyncCategory::OutputMaterialization,
                                   "WordEmbeddings::Align", "row_major_vocab_by_dim");
        out.dim = vectors.dim;
        out.success = true;
    } catch (const af::exception& e) {
        return fail(std::string("WordEmbeddings::Align failed on the ArrayFire device: ") + e.what());
    }
    return out;
}

}  // namespace cyxwiz
