// fastText .bin models (TOFIX152 P6): fastText's own format, as fastText and
// gensim's save_facebook_model write it (version 12):
//   int32 magic 793712314, int32 version; args: int32 dim, ws, epoch,
//   min_count, neg, word_ngrams, loss, model, bucket, minn, maxn,
//   lr_update_rate, double t; dictionary: int32 size, nwords, nlabels, int64
//   ntokens, int64 pruneidx_size, per entry the word and a NUL, int64 count,
//   int8 type, then pruneidx_size int32 pairs; input matrix: bool quantized,
//   int64 rows (nwords + bucket), int64 dim, rows * dim float32; output matrix.
// Only the rows the wanted words need are read (a word of the model: its row
// and its n-grams' rows; another word: its n-grams' rows), in file order and
// in chunks of words, so a 7 GB model never sits whole in memory; the means
// are taken on the device. fastText gives "</s>" no n-grams.

#ifdef _WIN32
#define NOMINMAX
#endif

#include "cyxwiz/word_embeddings.h"

#include "word_vector_device.h"

#include <arrayfire.h>

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace cyxwiz {

namespace {

constexpr int32_t kFastTextMagic = 793712314;
constexpr size_t kWordsPerChunk = 65536;
const char* const kEndOfSentence = "</s>";

template <typename T>
bool ReadValue(std::ifstream& in, T& value) {
    return static_cast<bool>(in.read(reinterpret_cast<char*>(&value), sizeof(T)));
}

// The header and dictionary, and where the input matrix's values start.
struct ParsedFile {
    FastTextFileInfo info;
    int nwords = 0;
    std::streamoff matrix_values = 0;
};

ParsedFile Parse(const std::string& path) {
    ParsedFile out;
    FastTextFileInfo& info = out.info;
    const auto fail = [&](const std::string& message) {
        info = FastTextFileInfo{};
        info.error_message = message;
        return out;
    };
    std::ifstream in(path, std::ios::binary);
    if (!in) return fail("Cannot open " + path);
    std::error_code size_error;
    info.file_bytes = static_cast<unsigned long long>(std::filesystem::file_size(path, size_error));
    int32_t magic = 0, version = 0;
    if (!ReadValue(in, magic) || !ReadValue(in, version)) return fail("The file is too short for a fastText model");
    if (magic != kFastTextMagic) return fail("Not a fastText model (its first bytes are not fastText's magic number)");
    int32_t args[12] = {};
    double sampling = 0.0;
    for (int32_t& arg : args)
        if (!ReadValue(in, arg)) return fail("The fastText header is cut short");
    if (!ReadValue(in, sampling)) return fail("The fastText header is cut short");
    info.version = version;
    info.dim = args[0];
    const int32_t model_kind = args[7];
    info.bucket = args[8];
    info.min_n = args[9];
    info.max_n = args[10];
    info.model = model_kind == 1 ? "cbow" : model_kind == 2 ? "skip-gram" : "supervised";
    if (model_kind == 3) return fail("A supervised (classifier) fastText model has no word vectors to read");
    if (info.dim <= 0 || info.bucket < 0 || info.min_n < 0 || info.max_n < 0)
        return fail("The fastText header has an impossible size");

    int32_t size = 0, nwords = 0, nlabels = 0;
    int64_t ntokens = 0, pruneidx_size = 0;
    if (!ReadValue(in, size) || !ReadValue(in, nwords) || !ReadValue(in, nlabels) || !ReadValue(in, ntokens) ||
        !ReadValue(in, pruneidx_size))
        return fail("The fastText dictionary is cut short");
    if (nlabels > 0) return fail("A supervised (classifier) fastText model has no word vectors to read");
    if (size < 0 || nwords < 0 || nwords > size) return fail("The fastText dictionary has an impossible size");
    info.tokens = ntokens;
    info.words.reserve(static_cast<size_t>(nwords));
    std::string word;
    for (int32_t i = 0; i < size; ++i) {
        if (!std::getline(in, word, '\0')) return fail("The fastText dictionary is cut short");
        int64_t count = 0;
        int8_t type = 0;
        if (!ReadValue(in, count) || !ReadValue(in, type)) return fail("The fastText dictionary is cut short");
        if (i < nwords) info.words.push_back(word);
    }
    if (pruneidx_size > 0) return fail("A pruned (quantized) fastText model is not supported; use its .vec file");
    uint8_t quantized = 0;
    int64_t rows = 0, dim = 0;
    if (!ReadValue(in, quantized) || !ReadValue(in, rows) || !ReadValue(in, dim))
        return fail("The fastText input matrix is cut short");
    if (quantized) return fail("A quantized fastText model (.ftz) is not supported; use its .vec file");
    if (dim != info.dim || rows != static_cast<int64_t>(nwords) + info.bucket)
        return fail("The fastText input matrix is not [words + buckets, dim]");
    out.nwords = nwords;
    out.matrix_values = in.tellg();
    const unsigned long long needed =
        static_cast<unsigned long long>(out.matrix_values) +
        static_cast<unsigned long long>(rows) * static_cast<unsigned long long>(dim) * sizeof(float);
    if (info.file_bytes < needed) return fail("The fastText file is shorter than its input matrix");
    info.success = true;
    return out;
}

// Reads `rows` (sorted, distinct) of the input matrix, row-major.
bool ReadRows(std::ifstream& in, const ParsedFile& file, const std::vector<long long>& rows, std::vector<float>& out) {
    const size_t dim = static_cast<size_t>(file.info.dim);
    out.resize(rows.size() * dim);
    size_t i = 0;
    while (i < rows.size()) {
        size_t j = i + 1;
        while (j < rows.size() && rows[j] == rows[j - 1] + 1) ++j;  // a run of consecutive rows
        const auto row_bytes = static_cast<std::streamoff>(dim * sizeof(float));
        in.seekg(file.matrix_values + static_cast<std::streamoff>(rows[i]) * row_bytes);
        if (!in.read(reinterpret_cast<char*>(out.data() + i * dim), static_cast<std::streamsize>(j - i) * row_bytes))
            return false;
        i = j;
    }
    return true;
}

}  // namespace

bool WordEmbeddings::IsFastTextBin(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    int32_t magic = 0;
    return in && ReadValue(in, magic) && magic == kFastTextMagic;
}

FastTextFileInfo WordEmbeddings::ReadFastTextInfo(const std::string& path) {
    return Parse(path).info;
}

namespace {

WordVectors LoadParsed(const std::string& path, const ParsedFile& file, const std::vector<std::string>* words,
                       std::vector<FastTextSource>* sources, size_t max_words) {
    WordVectors out;
    std::vector<std::string> first_words;
    if (!words) {
        const size_t n = max_words > 0 ? std::min(max_words, file.info.words.size()) : file.info.words.size();
        first_words.assign(file.info.words.begin(), file.info.words.begin() + static_cast<std::ptrdiff_t>(n));
    }
    const std::vector<std::string>& wanted = words ? *words : first_words;
    std::unordered_map<std::string, int> id_of;
    id_of.reserve(file.info.words.size());
    for (size_t w = 0; w < file.info.words.size(); ++w) id_of.emplace(file.info.words[w], static_cast<int>(w));
    if (sources) sources->assign(wanted.size(), FastTextSource::None);
    std::ifstream in(path, std::ios::binary);
    const dim_t dim = file.info.dim;
    out.values.reserve(wanted.size() * static_cast<size_t>(dim));
    try {
        for (size_t first = 0; first < wanted.size(); first += kWordsPerChunk) {
            const size_t last = std::min(wanted.size(), first + kWordsPerChunk);
            // The chunk's file rows, each once, and each word's rows.
            std::vector<std::pair<long long, int>> wanted_rows;  // (file row, word in chunk)
            for (size_t g = first; g < last; ++g) {
                const int local = static_cast<int>(g - first);
                const auto known = id_of.find(wanted[g]);
                const bool eos = wanted[g] == kEndOfSentence;
                std::vector<int> buckets;
                if (!eos)
                    buckets = WordEmbeddings::FastTextBuckets(wanted[g], file.info.min_n, file.info.max_n,
                                                              file.info.bucket);
                if (known != id_of.end()) {
                    wanted_rows.emplace_back(known->second, local);
                    if (sources) (*sources)[g] = FastTextSource::Word;
                } else if (!buckets.empty() && sources) {
                    (*sources)[g] = FastTextSource::Pieces;
                }
                for (const int b : buckets) wanted_rows.emplace_back(static_cast<long long>(file.nwords) + b, local);
            }
            std::vector<long long> file_rows;
            for (const auto& [row, local] : wanted_rows) file_rows.push_back(row);
            std::sort(file_rows.begin(), file_rows.end());
            file_rows.erase(std::unique(file_rows.begin(), file_rows.end()), file_rows.end());
            std::vector<float> table;
            if (!ReadRows(in, file, file_rows, table)) {
                out.error_message = "Cannot read the rows of " + path;
                return out;
            }
            std::vector<int> rows, group;
            rows.reserve(wanted_rows.size());
            group.reserve(wanted_rows.size());
            for (const auto& [row, local] : wanted_rows) {
                const auto at = std::lower_bound(file_rows.begin(), file_rows.end(), row);
                rows.push_back(static_cast<int>(at - file_rows.begin()));
                group.push_back(local);
            }
            const auto means = GroupMeans(table, static_cast<dim_t>(file_rows.size()), dim, rows, group,
                                          static_cast<dim_t>(last - first), "WordEmbeddings::LoadFastTextBin");
            out.values.insert(out.values.end(), means.begin(), means.end());
        }
    } catch (const af::exception& e) {
        out.error_message = std::string("WordEmbeddings::LoadFastTextBin failed on the ArrayFire device: ") + e.what();
        return out;
    }
    out.words = wanted;
    out.dim = static_cast<int>(dim);
    out.success = true;
    return out;
}

}  // namespace

WordVectors WordEmbeddings::LoadFastTextBin(const std::string& path, const std::vector<std::string>* words,
                                            std::vector<FastTextSource>* sources, FastTextFileInfo* info,
                                            size_t max_words) {
    const ParsedFile file = Parse(path);
    if (info) *info = file.info;
    if (!file.info.success) {
        WordVectors out;
        out.error_message = file.info.error_message;
        return out;
    }
    return LoadParsed(path, file, words, sources, max_words);
}

WordVectors WordEmbeddings::LoadFastTextBinFor(const std::string& path, const std::vector<std::string>& vocabulary,
                                               bool lowercase_fallback, FastTextFileInfo* info) {
    const ParsedFile parsed = Parse(path);
    if (info) *info = parsed.info;
    const FastTextFileInfo& file = parsed.info;
    if (!file.success) {
        WordVectors out;
        out.error_message = file.error_message;
        return out;
    }
    std::unordered_set<std::string> model(file.words.begin(), file.words.end());
    std::unordered_set<std::string> taken;
    std::vector<std::string> wanted;
    for (const auto& token : vocabulary) {
        std::string word = token;
        if (!model.count(word)) {
            if (!lowercase_fallback) continue;
            word = AsciiLowercase(token);
            if (!model.count(word)) continue;
        }
        if (taken.insert(word).second) wanted.push_back(word);
    }
    if (wanted.empty()) {
        WordVectors out;
        out.error_message = "No word of the vocabulary is in " + path;
        return out;
    }
    return LoadParsed(path, parsed, &wanted, nullptr, 0);
}

}  // namespace cyxwiz
