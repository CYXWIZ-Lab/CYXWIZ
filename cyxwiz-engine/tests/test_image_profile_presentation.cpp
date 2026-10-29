// Explore > Profile for an image dataset (TOFIX178 1.3e): tiles, classes,
// sizes, the Resize, prepared sets. Main scenario: the first six rows of the
// cats vs dogs index on this PC (2026-10-29), plus a grey and a broken file.

#include "../src/core/image_profile_presentation.h"

#include <cstdlib>
#include <iostream>
#include <string>

using namespace cyxwiz::image_profile;

namespace {

void Check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAILED: " << message << "\n";
        std::exit(1);
    }
}

}  // namespace

int main() {
    Input in;
    in.dataset = "cat_dog";
    in.labels_source = "cat_dog.csv";
    in.class_names = {"0", "1"};
    // dog.11289, cat.10836, cat.7970, cat.2212, dog.2653, dog.3051
    in.labels = {1, 0, 0, 0, 1, 1};
    in.widths = {267, 500, 371, 417, 500, 359};
    in.heights = {399, 370, 457, 500, 373, 359};
    in.channels = {3, 3, 3, 3, 3, 3};
    in.bytes = {20800, 14736, 18061, 18633, 31637, 22695};
    in.resize_width = 64;
    in.resize_height = 64;
    in.resize_mode = "exact";
    in.sets_listed = true;
    in.sets = {{64, 64, 3, "exact", {}, 307200064 + 1452955, true},
               {64, 64, 3, "exact", {"Center Crop 64 \xC3\x97 64", "Gaussian Blur 5 \xC3\x97 5, sigma 1"}, 308653202, false}};

    View v = Build(in);
    Check(v.tiles.size() == 5, "five tiles");
    Check(v.tiles[0].value == "6" && v.tiles[0].label == "images", "images tile");
    Check(v.tiles[1].value == "2" && v.tiles[1].label == "classes", "classes tile");
    Check(v.tiles[2].value == "127 KB" && v.tiles[2].label == "on disk", "on disk: " + v.tiles[2].value);
    Check(v.tiles[3].value == "0" && v.tiles[3].label == "unreadable", "unreadable tile");
    Check(v.tiles[4].value == "RGB" && v.tiles[4].label == "all 6", "colour tile: " + v.tiles[4].label);
    Check(v.classes_title == "Classes \xC2\xB7 from cat_dog.csv", "classes title: " + v.classes_title);
    Check(v.classes.size() == 2 && v.classes[0].name == "0" && v.classes[0].count == 3 && v.classes[1].count == 3 &&
              v.classes[0].text == "3 \xC2\xB7 50%",
          "class rows: " + v.classes[0].text);
    Check(v.balance == "Balanced.", "balanced: " + v.balance);
    Check(v.sizes.size() == 4, "width, height, square, resize");
    Check(v.sizes[0].second == "267 to 500 px, median 417", "width: " + v.sizes[0].second);
    Check(v.sizes[1].second == "359 to 500 px, median 399", "height: " + v.sizes[1].second);
    Check(v.sizes[2].second == "1 image", "one square image: " + v.sizes[2].second);
    Check(v.sizes[3].first == "Resize 64 \xC3\x97 64" &&
              v.sizes[3].second == "exact: stretched to 64 \xC3\x97 64, aspect is not kept.",
          "exact resize, nothing enlarged: " + v.sizes[3].second);
    Check(v.widths.edges.front() == 0 && v.widths.edges.back() == 550 && v.widths.counts.size() == 11, "width bins of 50");
    size_t sum = 0;
    for (size_t c : v.widths.counts) sum += c;
    Check(sum == 6 && v.widths.counts[10] == 2 && v.widths.max_count == 2, "every readable image in one bin; 500 in the last");
    Check(v.sets.size() == 2 && v.sets[0].size == "64 \xC3\x97 64 \xC3\x97 3" && v.sets[0].in_use &&
              v.sets[0].bytes == "309 MB" && v.sets[0].fixed.empty(),
          "the set in use: " + v.sets[0].bytes);
    Check(v.sets[1].fixed == "Center Crop 64 \xC3\x97 64, Gaussian Blur 5 \xC3\x97 5, sigma 1" && !v.sets[1].in_use,
          "a set with fixed transforms");

    // A grey image and a broken file; an enlarged one; another resize mode.
    in.labels.push_back(0);
    in.widths.push_back(42);
    in.heights.push_back(32);
    in.channels.push_back(1);
    in.bytes.push_back(1500);
    in.labels.push_back(1);
    in.widths.push_back(-1);
    in.heights.push_back(-1);
    in.channels.push_back(-1);
    in.bytes.push_back(12);
    in.resize_mode = "fit";
    v = Build(in);
    Check(v.tiles[3].value == "1", "one unreadable");
    Check(v.tiles[4].value == "Mixed" && v.tiles[4].label == "6 colour, 1 grey", "mixed colour: " + v.tiles[4].label);
    Check(v.sizes[3].second == "fit: the whole image is kept and padded with black to 64 \xC3\x97 64. 1 image is smaller "
                               "than 64 \xC3\x97 64 on a side and is enlarged.",
          "fit, one enlarged: " + v.sizes[3].second);

    // Imbalance, a class with no images, no Resize, sets still being read.
    Input lopsided;
    lopsided.class_names = {"a", "b", "c"};
    lopsided.labels = {0, 0, 0, 0, 1};
    lopsided.widths = {10, 10, 10, 10, 10};
    lopsided.heights = {10, 10, 10, 10, 10};
    lopsided.channels = {3, 3, 3, 3, 3};
    lopsided.bytes = {1, 1, 1, 1, 1};
    v = Build(lopsided);
    Check(v.balance == "A class has no images.", "an empty class: " + v.balance);
    Check(v.classes_title == "Classes \xC2\xB7 from the class folders", "class folders");
    Check(v.sizes.back().second == "This graph has no Resize node: training needs one to decode the images.", "no Resize");
    Check(v.sets_note == "Reading cache/images...", "sets being read");
    lopsided.class_names.pop_back();
    Check(Build(lopsided).balance == "Imbalanced: the largest class is 4.0\xC3\x97 the smallest.", "imbalance");
    lopsided.sets_listed = true;
    Check(Build(lopsided).sets_note.rfind("None yet", 0) == 0, "no sets yet");

    // Channel statistics: what Measure does, then the values and Normalize's view of them.
    Input stats = lopsided;
    Check(Build(stats).stats_note.rfind("Needs the graph's Resize", 0) == 0 && !Build(stats).can_measure, "stats need a Resize");
    stats.resize_width = stats.resize_height = 64;
    View s = Build(stats);
    Check(s.can_measure && s.stats.empty() && s.stats_note.rfind("Mean and spread of each channel", 0) == 0, "not measured yet");
    stats.stats_step = Input::StatsStep::Preparing;
    stats.stats_done = 1024;
    stats.stats_total = 25000;
    s = Build(stats);
    Check(!s.can_measure && s.stats_note == "Preparing images: 1,024 of 25,000", "preparing: " + s.stats_note);
    stats.stats_step = Input::StatsStep::Measuring;
    stats.stats_device = "CUDA - GeForce GTX 1050 Ti";
    Check(Build(stats).stats_note == "Measuring on CUDA - GeForce GTX 1050 Ti...", "measuring");
    stats.stats_step = Input::StatsStep::Idle;
    // The cats vs dogs set at 64 x 64 (2026-10-29), with the graph's Normalize 0.5 / 0.5.
    stats.channel_mean = {0.4883, 0.4551, 0.4170};
    stats.channel_std = {0.2548, 0.2481, 0.2508};
    stats.stats_images = 25000;
    stats.stats_device = "CUDA - GeForce GTX 1050 Ti";
    stats.normalize = true;
    stats.norm_mean = 0.5f;
    stats.norm_std = 0.5f;
    s = Build(stats);
    Check(s.stats_title == "Channel statistics \xC2\xB7 64 \xC3\x97 64, 25,000 images, on CUDA - GeForce GTX 1050 Ti",
          "stats title: " + s.stats_title);
    Check(s.channels == std::vector<std::string>({"R", "G", "B"}), "RGB channels");
    Check(s.stats.size() == 2 && s.stats[0].values == std::vector<std::string>({"0.4883", "0.4551", "0.4170"}) &&
              s.stats[1].values == std::vector<std::string>({"0.2548", "0.2481", "0.2508"}),
          "mean and std rows");
    Check(s.stats_note == "Normalize uses mean 0.5, std 0.5: the model's inputs centre at -0.02 (R), -0.09 (G), -0.17 (B) "
                          "with spread 0.51, 0.50, 0.50, where 0 and 1 are the aim.",
          "Normalize's view: " + s.stats_note);
    stats.normalize = false;
    Check(Build(stats).stats_note.rfind("The graph has no Normalize node", 0) == 0, "no Normalize");
    stats.stats_error = "The images could not be prepared: disk full";
    Check(Build(stats).stats_note == "The images could not be prepared: disk full", "an error shows");

    // Histogram bins stay round and few.
    const Histogram big = MakeHistogram({42, 447, 1050});
    Check(big.edges.size() == 12 && big.edges[1] == 100 && big.counts[10] == 1, "bins of 100 up to 1100");
    Check(MakeHistogram({}).counts.empty() && MakeHistogram({-1}).counts.empty(), "no values, no bins");

    std::cout << "test_image_profile_presentation: all checks passed\n";
    return 0;
}
