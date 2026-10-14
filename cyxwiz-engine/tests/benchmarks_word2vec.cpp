// word2vec training speed and quality on a real corpus (TOFIX152), not a
// ctest. Input: one sentence per line, words split by spaces (already
// lowercased and split as the Word Embeddings node splits them).
//   benchmark_word2vec <tokens.txt> [epochs] [batch_size] [out.vec]
// Prints the time and loss of each epoch and the nearest words of a few
// probes; with out.vec, saves the vectors (word2vec text) for a gensim
// comparison (scripts/word2vec_neighbour_overlap.py).

#include <cyxwiz/word_embeddings.h>

#include "computation_truth/test_device_selection.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: benchmark_word2vec <tokens.txt> [epochs] [batch_size] [out.vec]\n");
        return 2;
    }
    if (!cyxwiz::test::SelectTestDeviceFromEnvironment()) return 1;
    std::vector<std::vector<std::string>> sentences;
    {
        std::ifstream in(argv[1]);
        std::string line;
        while (std::getline(in, line)) {
            std::istringstream words(line);
            std::vector<std::string> sentence;
            std::string word;
            while (words >> word) sentence.push_back(word);
            sentences.push_back(std::move(sentence));
        }
    }
    cyxwiz::Word2VecOptions options;
    options.epochs = argc > 2 ? std::atoi(argv[2]) : 5;
    if (argc > 3) options.batch_size = std::atoi(argv[3]);
    const auto start = std::chrono::steady_clock::now();
    auto last = start;
    options.on_progress = [&](double done, int epoch) {
        static int reported = 0;
        if (epoch > reported + 1 || done >= 1.0) {
            const auto now = std::chrono::steady_clock::now();
            std::printf("  epoch %d done at %.1f s (%.1f s)\n", epoch - (done >= 1.0 ? 0 : 1),
                        std::chrono::duration<double>(now - start).count(),
                        std::chrono::duration<double>(now - last).count());
            last = now;
            reported = epoch - (done >= 1.0 ? 0 : 1);
        }
        return true;
    };
    const auto model = cyxwiz::WordEmbeddings::TrainWord2Vec(sentences, options);
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    if (!model.success) {
        std::fprintf(stderr, "FAIL: %s\n", model.error_message.c_str());
        return 1;
    }
    std::printf("trained %zu words x %d, %d epochs, batch %d, %lld examples in %.1f s\n", model.vectors.words.size(),
                model.vectors.dim, options.epochs, options.batch_size, model.examples, seconds);
    std::printf("loss per epoch:");
    for (const double loss : model.epoch_loss) std::printf(" %.4f", loss);
    std::printf("\n");
    for (const char* probe : {"therapy", "anxiety", "depressed", "happy"}) {
        std::printf("%s:", probe);
        for (const auto& [word, cosine] : cyxwiz::WordEmbeddings::Nearest(model.vectors, probe, 5))
            std::printf(" %s %.3f", word.c_str(), cosine);
        std::printf("\n");
    }
    if (argc > 4) {
        std::string error;
        if (!cyxwiz::WordEmbeddings::SaveText(model.vectors, argv[4], &error)) {
            std::fprintf(stderr, "FAIL: %s\n", error.c_str());
            return 1;
        }
    }
    return 0;
}
