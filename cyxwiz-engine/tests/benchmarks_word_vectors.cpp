// word2vec and GloVe training speed and quality on a real corpus (TOFIX152),
// not a ctest. Input: one sentence per line, words split by spaces (already
// lowercased and split as the Word Embeddings node splits them).
//   benchmark_word_vectors <word2vec|glove> <tokens.txt> [epochs] [batch_size] [out.vec]
// Prints the time and loss (cost) of each epoch and the nearest words of a few
// probes; with out.vec, saves the vectors (word2vec text) for a comparison
// with gensim or scripts/glove_reference.py (scripts/word2vec_neighbour_overlap.py).

#include <cyxwiz/word_embeddings.h>

#include "computation_truth/test_device_selection.h"

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
    if (argc < 3 || (mode != "word2vec" && mode != "glove")) {
        std::fprintf(stderr,
                     "usage: benchmark_word_vectors <word2vec|glove> <tokens.txt> [epochs] [batch_size] [out.vec]\n");
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
