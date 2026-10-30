// Explore > Images (TOFIX178 1.3e): chips, filter, sort, pages, details.
// Main scenario: the first six rows of the cats vs dogs index on this PC
// (2026-10-29), named cat / dog, plus a broken file.

#include "../src/core/image_gallery_presentation.h"

#include <cstdlib>
#include <iostream>
#include <string>

using namespace cyxwiz::image_gallery;

namespace {

void Check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAILED: " << message << "\n";
        std::exit(1);
    }
}

const char* kDir = "D:/demo/mrcj/datasets/Cats_Dogs/cat_dog/";

Row Image(const std::string& file, int label, int w, int h, int64_t bytes) {
    return {kDir + file, file, label, w, h, 3, bytes, true};
}

}  // namespace

int main() {
    Input in;
    in.class_names = {"0", "1"};
    in.rows = {Image("dog.11289.jpg", 1, 267, 399, 20800), Image("cat.10836.jpg", 0, 500, 370, 14736),
               Image("cat.7970.jpg", 0, 371, 457, 18061),  Image("cat.2212.jpg", 0, 417, 500, 18633),
               Image("dog.2653.jpg", 1, 500, 373, 31637),  Image("dog.3051.jpg", 1, 359, 359, 22695)};
    in.per_page = 4;

    View v = Build(in);
    Check(v.chips.size() == 3, "All, Class 0, Class 1 (no unreadable chip)");
    Check(v.chips[0].label == "All" && v.chips[0].count == 6 && v.chips[0].on, "All is on");
    Check(v.chips[1].label == "Class 0" && v.chips[1].count == 3 && v.chips[2].label == "Class 1", "numbered classes");
    Check(v.shown == std::vector<size_t>({0, 1, 2, 3}) && v.tags[0] == "1", "the first page in dataset order");
    Check(v.showing == "1 to 4 of 6" && v.pages == 2 && !v.has_previous && v.has_next, "page 1 of 2: " + v.showing);
    Check(v.selected == -1 && v.detail.empty(), "nothing selected");

    in.page = 1;
    v = Build(in);
    Check(v.shown == std::vector<size_t>({4, 5}) && v.showing == "5 to 6 of 6" && v.has_previous && !v.has_next, "page 2");
    in.page = 9;
    Check(Build(in).page == 1, "a page past the end shows the last");

    // Names, a class filter, sorting, a selection.
    in.class_names = {"cat", "dog"};
    in.page = 0;
    in.filter = "class:1";
    in.sort = Sort::Largest;
    in.selected = 4;
    v = Build(in);
    Check(v.chips[2].label == "dog" && v.chips[2].on, "named class chip");
    Check(v.matching == 3 && v.shown == std::vector<size_t>({4, 5, 0}), "dogs, largest first (500x373, 359x359, 267x399)");
    Check(v.tags == std::vector<std::string>({"dog", "dog", "dog"}), "tags are class names");
    Check(v.selected == 4 && v.detail_title == "dog.2653.jpg", "the selected dog");
    Check(v.detail.size() == 5 && v.detail[0].second == "5 of 6" && v.detail[1].second == "1 \xC2\xB7 dog" &&
              v.detail[2].second == "500 \xC3\x97 373 px, colour" && v.detail[3].second == "31.6 KB" &&
              v.detail[4].second == std::string(kDir) + "dog.2653.jpg",
          "details: " + v.detail[2].second);
    in.filter = "class:0";
    v = Build(in);
    Check(v.selected == -1 && v.detail.empty(), "a selection the filter hides is dropped");
    in.sort = Sort::Smallest;
    Check(Build(in).shown == std::vector<size_t>({2, 1, 3}), "cats, smallest first (371x457, 500x370, 417x500)");

    // An unreadable file: its own chip, sorted last, its details say so.
    in.rows.push_back({std::string(kDir) + "broken.jpg", "broken.jpg", 0, -1, -1, -1, 12, false});
    in.filter = "all";
    in.per_page = 60;
    v = Build(in);
    Check(v.chips.back().key == "unreadable" && v.chips.back().count == 1, "an unreadable chip");
    Check(v.shown.back() == 6, "unreadable sorts last when smallest first");
    in.filter = "unreadable";
    in.selected = 6;
    v = Build(in);
    Check(v.shown == std::vector<size_t>({6}) && v.detail[2].second == "cannot be read as an image", "unreadable details");
    in.filter = "class:7";
    Check(Build(in).chips[0].on, "an unknown filter falls back to All");

    // Quality measured: chips per reason that occurs, flags, Blurriest, details.
    in.filter = "all";
    in.selected = -1;
    in.sort = Sort::DatasetOrder;
    in.rows.pop_back();  // the unreadable one
    const float blur[] = {812.5f, 300.0f, 950.0f, 650.0f, 1200.0f, 720.0f};
    const char* verdicts[] = {"", "blurry, low contrast, duplicate", "", "blurry", "", "dark"};
    for (size_t i = 0; i < in.rows.size(); ++i) {
        in.rows[i].measured = true;
        in.rows[i].blur = blur[i];
        in.rows[i].brightness = 120.0f;
        in.rows[i].contrast = 0.2f;
        in.rows[i].verdict = verdicts[i];
    }
    in.rows[1].duplicate_of = "dog.11289.jpg";
    v = Build(in);
    Check(v.quality, "quality is measured");
    std::vector<std::string> keys;
    for (const auto& chip : v.chips) keys.push_back(chip.key + "=" + chip.count_text);
    Check(keys == std::vector<std::string>({"all=6", "class:0=3", "class:1=3", "fails=3", "reason:blurry=2",
                                            "reason:low contrast=1", "reason:dark=1", "reason:duplicate=1"}),
          "quality chips");
    Check(v.flags == std::vector<std::string>({"", "blurry +2", "", "blurry", "", "dark"}), "flags");
    in.filter = "reason:blurry";
    Check(Build(in).shown == std::vector<size_t>({1, 3}), "the blurry ones");
    in.filter = "fails";
    in.sort = Sort::Blurriest;
    Check(Build(in).shown == std::vector<size_t>({1, 3, 5}), "failing ones, blurriest first");
    in.selected = 1;
    v = Build(in);
    Check(v.detail.size() == 10 && v.detail[4].first == "Blur" && v.detail[4].second == "300" &&
              v.detail[6].second == "0.20" && v.detail[7].second == "blurry, low contrast, duplicate" &&
              v.detail[8].first == "Same as" && v.detail[8].second == "dog.11289.jpg" &&
              v.detail[9].first == "Path",
          "quality details");
    in.filter = "all";
    in.selected = 0;
    Check(Build(in).detail[7].second == "passes the checks", "a passing image");

    {
        // The list's columns and the viewer (2026-10-30): name, class and
        // file-size sorts, each image's facts, its place and neighbours.
        Input list;
        list.class_names = {"cat", "dog"};
        list.rows = {Image("dog.11289.jpg", 1, 267, 399, 20800), Image("cat.10836.jpg", 0, 500, 370, 14736),
                     Image("cat.7970.jpg", 0, 371, 457, 18061),  Image("cat.2212.jpg", 0, 417, 500, 18633),
                     Image("dog.2653.jpg", 1, 500, 373, 31637),  Image("dog.3051.jpg", 1, 359, 359, 22695)};
        list.per_page = 6;
        list.sort = Sort::Name;
        View lv = Build(list);
        Check(lv.shown == std::vector<size_t>({1, 3, 2, 0, 4, 5}), "by name");
        Check(lv.sizes[0] == "500 \xC3\x97 370" && lv.channels[0] == "colour" && lv.on_disk[0] == "14.7 KB" &&
                  lv.verdicts[0] == "not measured",
              "a row's facts");
        list.sort = Sort::Class;
        Check(Build(list).shown == std::vector<size_t>({1, 2, 3, 0, 4, 5}), "by class, dataset order within");
        list.sort = Sort::LargestFile;
        Check(Build(list).shown == std::vector<size_t>({4, 5, 0, 3, 2, 1}), "largest file first");

        list.sort = Sort::Name;
        list.selected = 2;  // cat.7970.jpg, third by name
        list.strip_radius = 1;
        lv = Build(list);
        Check(lv.position == "3 of 6" && lv.previous == 3 && lv.next == 0, "place and neighbours: " + lv.position);
        Check(lv.strip == std::vector<size_t>({3, 2, 0}), "the filmstrip around it");
        Check(lv.order == std::vector<size_t>({1, 3, 2, 0, 4, 5}), "every match in order");
        list.selected = 1;  // first by name
        lv = Build(list);
        Check(lv.previous == -1 && lv.next == 3 && lv.strip == std::vector<size_t>({1, 3}), "the first has no previous");
    }

    in.rows.clear();
    v = Build(in);
    Check(v.shown.empty() && v.showing == "No image matches." && v.pages == 0, "no images");
    in.rows.assign(25000, Image("cat.0.jpg", 0, 10, 10, 1));
    Check(Build(in).chips[0].count_text == "25,000", "counts with thousands separators");

    std::cout << "test_image_gallery_presentation: all checks passed\n";
    return 0;
}
