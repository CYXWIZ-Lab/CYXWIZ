// word2vec, GloVe and fastText training speed and quality on a real corpus
// (TOFIX152), not a ctest. Input: one sentence per line, words split by spaces
// (already lowercased and split as the Word Embeddings node splits them).
//   benchmark_word_vectors <word2vec|glove|fasttext> <tokens.txt> [epochs] [batch_size] [out.vec] [buckets]
// fastText also prints the nearest words of misspellings it never saw.
// Prints the time and loss (cost) of each epoch and the nearest words of a few
// probes; with out.vec, saves the vectors (word2vec text) for a comparison
// with gensim or scripts/glove_reference.py (scripts/word2vec_neighbour_overlap.py).

#include <cyxwiz/word_embeddings.h>

#include "computation_truth/test_device_selection.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace {

// Reports each finished stage: GloVe's counting (epoch 0) and every epoch.
std::function<bool(double, int)> EpochReporter(std::chrono::steady_clock::time_point start) {
    auto last = std::make_shared<std::chrono::steady_clock::time_point>(start);
    auto reported = std::make_shared<int>(-1);
    return [=](double done, int epoch) {
        const int finished = done >= 1.0 ? epoch : epoch - 1;
        if (finished > *reported) {
            const auto now = std::chrono::steady_clock::now();
            if (finished == 0)
                std::printf("  counted at %.1f s\n", std::chrono::duration<double>(now - start).count());
            else
                std::printf("  epoch %d done at %.1f s (%.1f s)\n", finished,
                            std::chrono::duration<double>(now - start).count(),
                            std::chrono::duration<double>(now - *last).count());
            *last = now;
            *reported = finished;
        }
        return true;
    };
}

}  // namespace

int main(int argc, char** argv) {
    const std::string mode = argc > 1 ? argv[1] : "";
    if (argc < 3 || (mode != "word2vec" && mode != "glove" && mode != "fasttext")) {
        std::fprintf(stderr, "usage: benchmark_word_vectors <word2vec|glove|fasttext> <tokens.txt> [epochs] "
                             "[batch_size] [out.vec]\n");
        return 2;
    }
    std::setvbuf(stdout, nullptr, _IONBF, 0);  // progress shows while it runs
    if (!cyxwiz::test::SelectTestDeviceFromEnvironment()) return 1;
    std::vector<std::vector<std::string>> sentences;
    {
        std::ifstream in(argv[2]);
        std::string line;
        while (std::getline(in, line)) {
            std::istringstream words(line);
            std::vector<std::string> sentence;
            std::string word;
            while (words >> word) sentence.push_back(word);
            sentences.push_back(std::move(sentence));
        }
    }
    const auto start = std::chrono::steady_clock::now();
    cyxwiz::WordVectors vectors;
    if (mode == "word2vec") {
        cyxwiz::Word2VecOptions options;
        options.epochs = argc > 3 ? std::atoi(argv[3]) : 5;
        if (argc > 4) options.batch_size = std::atoi(argv[4]);
        options.on_progress = EpochReporter(start);
        const auto model = cyxwiz::WordEmbeddings::TrainWord2Vec(sentences, options);
        const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        if (!model.success) {
            std::fprintf(stderr, "FAIL: %s\n", model.error_message.c_str());
            return 1;
        }
        std::printf("trained %zu words x %d, %d epochs, batch %d, %lld examples in %.1f s\n",
                    model.vectors.words.size(), model.vectors.dim, options.epochs, options.batch_size, model.examples,
                    seconds);
        std::printf("loss per epoch:");
        for (const double loss : model.epoch_loss) std::printf(" %.4f", loss);
        vectors = model.vectors;
    } else if (mode == "fasttext") {
        cyxwiz::FastTextOptions options;
        options.epochs = argc > 3 ? std::atoi(argv[3]) : 5;
        if (argc > 4) options.batch_size = std::atoi(argv[4]);
        if (argc > 6) options.bucket = std::atoi(argv[6]);
        options.on_progress = EpochReporter(start);
        const auto model = cyxwiz::WordEmbeddings::TrainFastText(sentences, options);
        const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        if (!model.success) {
            std::fprintf(stderr, "FAIL: %s\n", model.error_message.c_str());
            return 1;
        }
        std::printf("trained %zu words x %d, %d epochs, batch %d, %d buckets, %lld examples in %.1f s\n",
                    model.words.size(), model.dim, options.epochs, options.batch_size, model.bucket, model.examples,
                    seconds);
        std::printf("loss per epoch:");
        for (const double loss : model.epoch_loss) std::printf(" %.4f", loss);
        std::printf("\n");
        vectors = cyxwiz::WordEmbeddings::FastTextVectors(model);
        // Misspellings it never saw: their vectors from pieces, next to the words'.
        const std::vector<std::string> unseen = {"anxietyy", "depressionn", "overthinkingg", "psychiatristt",
                                                 "counsellingg", "theraphy"};
        const auto built = cyxwiz::WordEmbeddings::FastTextVectors(model, &unseen);
        cyxwiz::WordVectors with_unseen = vectors;
        with_unseen.words.insert(with_unseen.words.end(), unseen.begin(), unseen.end());
        with_unseen.values.insert(with_unseen.values.end(), built.values.begin(), built.values.end());
        for (const auto& word : unseen) {
            std::printf("%s (pieces):", word.c_str());
            int shown = 0;
            for (const auto& [other, cosine] : cyxwiz::WordEmbeddings::Nearest(with_unseen, word, 8)) {
                if (std::find(unseen.begin(), unseen.end(), other) != unseen.end() || ++shown > 5) continue;
                std::printf(" %s %.3f", other.c_str(), cosine);
            }
            std::printf("\n");
        }
    } else {
        cyxwiz::GloVeOptions options;
        options.epochs = argc > 3 ? std::atoi(argv[3]) : 25;
        if (argc > 4) options.batch_size = std::atoi(argv[4]);
        options.on_progress = EpochReporter(start);
        const auto model = cyxwiz::WordEmbeddings::TrainGloVe(sentences, options);
        const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        if (!model.success) {
            std::fprintf(stderr, "FAIL: %s\n", model.error_message.c_str());
            return 1;
        }
        std::printf("trained %zu words x %d, %d epochs, batch %d, %lld pairs in %.1f s\n", model.vectors.words.size(),
                    model.vectors.dim, options.epochs, options.batch_size, model.pairs, seconds);
        std::printf("cost per epoch:");
        for (const double cost : model.epoch_cost) std::printf(" %.5f", cost);
        vectors = model.vectors;
    }
    std::printf("\n");
    for (const char* probe : {"therapy", "anxiety", "depressed", "happy"}) {
        std::printf("%s:", probe);
        for (const auto& [word, cosine] : cyxwiz::WordEmbeddings::Nearest(vectors, probe, 5))
            std::printf(" %s %.3f", word.c_str(), cosine);
        std::printf("\n");
    }
    std::string error;
    if (argc > 5 && !cyxwiz::WordEmbeddings::SaveText(vectors, argv[5], &error)) {
        std::fprintf(stderr, "FAIL: %s\n", error.c_str());
        return 1;
    }
    return 0;
}
