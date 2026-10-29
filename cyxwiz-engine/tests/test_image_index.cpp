// The image index table (TOFIX178 1.3e): header reads agree with a decode,
// rows follow the dataset, display names, cancel.

#include "../src/core/image_index.h"
#include "../src/core/prepared_image_set.h"

#include <arrow/api.h>
#include <opencv2/opencv.hpp>

#include <atomic>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

using namespace cyxwiz;
namespace fs = std::filesystem;

namespace {

void Check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAILED: " << message << "\n";
        std::exit(1);
    }
}

cv::Mat Pattern(int width, int height, int type) {
    cv::Mat image(height, width, type);
    cv::randu(image, cv::Scalar::all(0), cv::Scalar::all(CV_MAT_DEPTH(type) == CV_16U ? 65535 : 255));
    return image;
}

void Write(const fs::path& path, const cv::Mat& image) {
    fs::create_directories(path.parent_path());
    Check(cv::imwrite(path.string(), image), "wrote " + path.string());
}

// The header says what a decode gives.
void CheckHeaders(const fs::path& dir) {
    struct Case {
        const char* file;
        int width, height, type;
    };
    const Case cases[] = {
        {"grey.jpg", 41, 23, CV_8UC1},   {"colour.jpg", 64, 48, CV_8UC3}, {"grey.png", 17, 9, CV_8UC1},
        {"colour.png", 30, 20, CV_8UC3}, {"alpha.png", 12, 34, CV_8UC4}, {"deep.png", 25, 15, CV_16UC3},
        {"colour.bmp", 33, 21, CV_8UC3}, {"grey.bmp", 19, 11, CV_8UC1},  {"alpha.bmp", 13, 7, CV_8UC4},
        {"colour.tiff", 22, 14, CV_8UC3},
    };
    for (const auto& c : cases) {
        const fs::path file = dir / c.file;
        Write(file, Pattern(c.width, c.height, c.type));
        const cv::Mat decoded = cv::imread(file.string(), cv::IMREAD_UNCHANGED);
        const ImageHeader h = ReadImageHeader(file.string());
        Check(h.readable, std::string(c.file) + " is readable");
        Check(h.width == decoded.cols && h.height == decoded.rows,
              std::string(c.file) + " size " + std::to_string(h.width) + "x" + std::to_string(h.height) + " vs decode " +
                  std::to_string(decoded.cols) + "x" + std::to_string(decoded.rows));
        Check(h.channels == decoded.channels(), std::string(c.file) + " channels " + std::to_string(h.channels) +
                                                    " vs decode " + std::to_string(decoded.channels()));
    }
    {
        std::ofstream broken(dir / "broken.jpg", std::ios::binary);
        broken << "\xFF\xD8\xFF not really";
    }
    Check(!ReadImageHeader((dir / "broken.jpg").string()).readable, "a broken file is not readable");
    Check(!ReadImageHeader((dir / "missing.png").string()).readable, "a missing file is not readable");
}

std::vector<std::string> Strings(const std::shared_ptr<arrow::Table>& t, const char* name) {
    const auto a = std::static_pointer_cast<arrow::StringArray>(t->CombineChunks().ValueOrDie()->GetColumnByName(name)->chunk(0));
    std::vector<std::string> out;
    for (int64_t i = 0; i < a->length(); ++i) out.push_back(a->IsNull(i) ? "<null>" : a->GetString(i));
    return out;
}

std::vector<int> Ints(const std::shared_ptr<arrow::Table>& t, const char* name) {
    const auto a = std::static_pointer_cast<arrow::Int32Array>(t->CombineChunks().ValueOrDie()->GetColumnByName(name)->chunk(0));
    std::vector<int> out;
    for (int64_t i = 0; i < a->length(); ++i) out.push_back(a->IsNull(i) ? -1 : a->Value(i));
    return out;
}

}  // namespace

int main() {
    const fs::path root = fs::temp_directory_path() / "cyxwiz_test_image_index";
    std::error_code ec;
    fs::remove_all(root, ec);

    CheckHeaders(root / "headers");

    // Class folders cat (2 images + a broken one) and dog (1 image).
    const fs::path folder = root / "pets";
    Write(folder / "cat" / "c1.png", Pattern(40, 30, CV_8UC3));
    Write(folder / "cat" / "c0.jpg", Pattern(20, 10, CV_8UC1));
    {
        std::ofstream broken(folder / "cat" / "c2.png", std::ios::binary);
        broken << "not an image";
    }
    Write(folder / "dog" / "d0.png", Pattern(16, 16, CV_8UC4));
    DataRegistry::ImageDatasetEntry entry;
    entry.folder_path = folder.string();
    entry.layout = 0;
    const ImageSetListing listing = ListImageSet(entry);
    Check(listing.files.size() == 4, "four files listed");

    const auto index = BuildImageIndex(listing, {"Cat", ""}, {}, nullptr, 2);
    Check(index && index->num_rows() == 4 && index->num_columns() == 9, "4 rows x 9 columns");
    const std::vector<std::string> expected_columns = {"path", "file", "label", "label_name", "width",
                                                       "height", "channels", "bytes", "readable"};
    Check(index->schema()->field_names() == expected_columns, "the index columns, in order");
    Check(Strings(index, "file") == std::vector<std::string>({"c0.jpg", "c1.png", "c2.png", "d0.png"}),
          "rows in dataset order");
    Check(Strings(index, "path") == listing.files, "paths are the dataset's");
    Check(Ints(index, "label") == std::vector<int>({0, 0, 0, 1}), "labels are class indices");
    Check(Strings(index, "label_name") == std::vector<std::string>({"Cat", "Cat", "Cat", "dog"}),
          "a display name, else the class name");
    Check(Ints(index, "width") == std::vector<int>({20, 40, -1, 16}) && Ints(index, "height") == std::vector<int>({10, 30, -1, 16}),
          "original sizes; null for the broken file");
    Check(Ints(index, "channels") == std::vector<int>({1, 3, -1, 4}), "channels as decoded");
    const auto combined = index->CombineChunks().ValueOrDie();
    const auto readable = std::static_pointer_cast<arrow::BooleanArray>(combined->GetColumnByName("readable")->chunk(0));
    Check(readable->Value(0) && readable->Value(1) && !readable->Value(2) && readable->Value(3), "the broken file is not readable");
    const auto bytes = std::static_pointer_cast<arrow::Int64Array>(combined->GetColumnByName("bytes")->chunk(0));
    for (int64_t i = 0; i < 4; ++i)
        Check(bytes->Value(i) == static_cast<int64_t>(fs::file_size(listing.files[i])), "bytes are the file size");

    size_t last = 0, calls = 0;
    BuildImageIndex(listing, {}, [&](size_t done, size_t total) {
        Check(total == 4 && done >= last && done <= total, "progress counts up to the total");
        last = done;
        ++calls;
    }, nullptr, 1);
    Check(calls > 0 && last == 4, "progress ends at the total");
    std::atomic<bool> cancel{true};
    Check(BuildImageIndex(listing, {}, {}, &cancel, 2) == nullptr, "a cancelled build gives no table");

    // Class names suggested from file names (cats vs dogs: labels 0 / 1, files cat.N.jpg / dog.N.jpg).
    const std::vector<std::string> files = {"x/dog.11289.jpg", "x/cat.10836.jpg", "x/cat.7970.jpg", "x/dog.2653.jpg"};
    const std::vector<int> labels = {1, 0, 0, 1};
    Check(SuggestClassNames(files, labels, {"0", "1"}) == std::vector<std::string>({"cat", "dog"}), "cat, dog from 0, 1");
    Check(SuggestClassNames(files, labels, {"cat", "1"}) == std::vector<std::string>({"", "dog"}),
          "nothing to suggest where the name already says it");
    Check(SuggestClassNames({"a/cat_1.png", "a/kitten2.png", "b/dog-3.png"}, {0, 0, 1}, {"0", "1"}) ==
              std::vector<std::string>({"", "dog"}),
          "no suggestion when a class's files disagree");
    Check(SuggestClassNames({"img_1.png", "img_2.png"}, {0, 1}, {"0", "1"}) == std::vector<std::string>({"", ""}),
          "no suggestion when two classes share the prefix");
    Check(SuggestClassNames({"001.png"}, {0}, {"0", "1"}) == std::vector<std::string>({"", ""}),
          "no prefix, no images: no suggestion");

    fs::remove_all(root, ec);
    std::cout << "test_image_index: all checks passed\n";
    return 0;
}
