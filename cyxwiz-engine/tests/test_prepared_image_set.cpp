// The prepared image set (TOFIX178 1.3a): every image of an image Data Input
// decoded once to uint8 [N, H, W, 3] at the Resize size and mode, kept in
// cache/images. Generated image sets of both layouts (class folders, a
// labels CSV) with grey, RGBA, 16-bit, JPEG and unreadable files; no
// example dataset.

#include "../src/core/image_dataset_batcher.h"
#include "../src/core/image_utils.h"
#include "../src/core/prepared_image_set.h"

#include <arrayfire.h>
#ifdef LoadImage
#undef LoadImage  // windows.h (via ArrayFire) would rename ImageUtils::LoadImage
#endif
#include <opencv2/opencv.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <set>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using namespace cyxwiz;

namespace {

void Check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << "\n";
        std::exit(1);
    }
}

// A BGR(A) image with a gradient so every pixel differs.
cv::Mat Gradient(int width, int height, int type, int seed) {
    cv::Mat image(height, width, type);
    const int channels = image.channels();
    const double top = image.depth() == CV_16U ? 65535.0 : 255.0;
    for (int y = 0; y < height; ++y)
        for (int x = 0; x < width; ++x)
            for (int c = 0; c < channels; ++c) {
                const double v = std::fmod((x * 7 + y * 13 + c * 29 + seed * 31) / 97.0, 1.0) * top;
                if (image.depth() == CV_16U) image.ptr<uint16_t>(y)[x * channels + c] = static_cast<uint16_t>(v);
                else image.ptr<uint8_t>(y)[x * channels + c] = static_cast<uint8_t>(v);
            }
    return image;
}

void Write(const fs::path& path, const cv::Mat& image) {
    fs::create_directories(path.parent_path());
    Check(cv::imwrite(path.string(), image), "wrote " + path.string());
}

// The pixel at (x, y) of a decoded [H, W, 3] row.
const uint8_t* Pixel(const std::vector<uint8_t>& row, int width, int x, int y) {
    return row.data() + (static_cast<size_t>(y) * width + x) * 3;
}

void CheckExactMatchesImageUtils(const fs::path& file, int width, int height) {
    std::vector<float> reference;
    int w = 0, h = 0, c = 0;
    Check(ImageUtils::LoadImage(file.string(), reference, w, h, c), "ImageUtils reads " + file.string());
    if (w != width || h != height) {
        const auto method = (width < w || height < h) ? ImageUtils::ResizeMethod::Area : ImageUtils::ResizeMethod::Lanczos;
        Check(ImageUtils::ResizeImage(reference, w, h, c, width, height, method), "ImageUtils resizes");
    }
    std::vector<uint8_t> row(static_cast<size_t>(width) * height * 3);
    Check(DecodePreparedImage(file.string(), {width, height, ResizeMode::Exact}, row.data()), "decodes " + file.string());
    double worst = 0.0;
    for (size_t i = 0; i < row.size(); ++i) {
        const double expected = std::clamp(reference[i], 0.0f, 1.0f) * 255.0;
        worst = std::max(worst, std::abs(row[i] - expected));
    }
    // The training path's float pixels, rounded once: at most half a step.
    Check(worst <= 0.5 + 1e-3, file.filename().string() + " matches the float decode (worst " + std::to_string(worst) + ")");
}

void CheckModes(const fs::path& dir) {
    // 40 x 20: left half red, right half blue (BGR on disk).
    cv::Mat image(20, 40, CV_8UC3, cv::Scalar(255, 0, 0));
    image(cv::Rect(0, 0, 20, 20)).setTo(cv::Scalar(0, 0, 255));
    const fs::path file = dir / "halves.png";
    Write(file, image);
    std::vector<uint8_t> row(32 * 32 * 3);
    const auto is = [&](int x, int y, int r, int g, int b) {
        const uint8_t* p = Pixel(row, 32, x, y);
        return std::abs(p[0] - r) <= 2 && std::abs(p[1] - g) <= 2 && std::abs(p[2] - b) <= 2;
    };

    // fit: scaled by 0.8 to 32 x 16, centred: black bands above and below.
    Check(DecodePreparedImage(file.string(), {32, 32, ResizeMode::AspectFit}, row.data()), "fit decodes");
    Check(is(4, 2, 0, 0, 0) && is(4, 29, 0, 0, 0), "fit pads with black above and below");
    Check(is(4, 16, 255, 0, 0) && is(28, 16, 0, 0, 255), "fit keeps the whole image (red left, blue right)");

    // fill: scaled by 1.6 to 64 x 32, the centre 32 columns kept.
    Check(DecodePreparedImage(file.string(), {32, 32, ResizeMode::AspectFill}, row.data()), "fill decodes");
    Check(is(4, 2, 255, 0, 0) && is(4, 29, 255, 0, 0) && is(28, 16, 0, 0, 255), "fill covers the size and crops the sides");

    // center: no scaling: the centre 32 columns, 6 black rows above and below.
    Check(DecodePreparedImage(file.string(), {32, 32, ResizeMode::Center}, row.data()), "center decodes");
    Check(is(10, 2, 0, 0, 0) && is(10, 29, 0, 0, 0), "center pads the short side");
    Check(is(10, 16, 255, 0, 0) && is(22, 16, 0, 0, 255), "center keeps the middle unscaled");

    // exact: stretched to 32 x 32.
    Check(DecodePreparedImage(file.string(), {32, 32, ResizeMode::Exact}, row.data()), "exact decodes");
    Check(is(4, 2, 255, 0, 0) && is(28, 29, 0, 0, 255), "exact stretches");
}

void CheckSet(const DataRegistry::ImageDatasetEntry& entry, const std::string& name) {
    const PreparedImageSpec spec{24, 16, ResizeMode::Exact};
    const auto listing = ListImageSet(entry);
    Check(listing.files.size() == 6, name + ": six files listed");

    auto first = PrepareImageSet(entry, spec, 3);
    Check(first.set != nullptr, name + ": prepared (" + first.error + ")");
    Check(!first.from_cache, name + ": decoded the first time");
    const auto& set = *first.set;
    Check(set.Count() == 6 && set.Height() == 16 && set.Width() == 24 && set.Channels() == 3, name + ": [6, 16, 24, 3]");
    Check(set.Labels() == listing.labels && set.Files() == listing.files, name + ": labels and files in dataset order");
    Check(set.ClassNames() == listing.class_names && set.ClassNames().size() == 2, name + ": two class names");
    Check(set.Unreadable().size() == 1, name + ": one unreadable file");

    std::vector<uint8_t> row(set.RowBytes());
    for (size_t i = 0; i < set.Count(); ++i) {
        const bool readable = DecodePreparedImage(set.Files()[i], spec, row.data());
        if (!readable) std::fill(row.begin(), row.end(), 0);
        Check(std::equal(row.begin(), row.end(), set.Row(i)), name + ": row " + std::to_string(i) + " is the decoded file");
    }

    // Opened from the cache next time; the same bytes whatever the thread count.
    auto again = PrepareImageSet(entry, spec, 1);
    Check(again.set && again.from_cache && again.set->Path() == set.Path(), name + ": opened from the cache");
    const fs::path path = set.Path();
    std::vector<uint8_t> threaded(set.Row(0), set.Row(0) + set.Count() * set.RowBytes());
    first = {};
    again = {};
    std::error_code ec;
    fs::remove(path, ec);
    Check(!ec, name + ": the closed set can be removed");
    auto single = PrepareImageSet(entry, spec, 1);
    Check(single.set && !single.from_cache, name + ": decoded again on one thread");
    Check(std::equal(threaded.begin(), threaded.end(), single.set->Row(0)), name + ": one thread and three agree");

    // Another size or mode is another set.
    auto other = PrepareImageSet(entry, {24, 16, ResizeMode::AspectFit});
    Check(other.set && !other.from_cache && other.set->Path() != single.set->Path(), name + ": fit is its own set");
}

// The batcher trains from the set (1.3b): every readable image once per
// epoch as its row / 255 with its label, the unreadable one and excluded files
// left out - on every ArrayFire backend this machine has.
void CheckBatcher(const DataRegistry::ImageDatasetEntry& entry) {
    const auto prepared = PrepareImageSet(entry, {24, 16, ResizeMode::Exact});
    Check(prepared.set != nullptr, "prepared for the batcher");
    const auto& set = *prepared.set;
    const int available = af::getAvailableBackends();
    for (const af::Backend backend : {AF_BACKEND_CPU, AF_BACKEND_CUDA, AF_BACKEND_OPENCL}) {
        if (!(available & backend)) continue;
        af::setBackend(backend);
        const std::string name = backend == AF_BACKEND_CPU ? "cpu" : backend == AF_BACKEND_CUDA ? "cuda" : "opencl";
        ImageDatasetBatcher batcher(prepared.set, 2, 1.0f, true, 5);
        Check(batcher.GetNumSamples() == 5, name + ": the unreadable image is left out");
        std::set<size_t> seen;
        while (!batcher.IsEpochComplete()) {
            const auto batch = batcher.GetNextBatch();
            Check(batch.data.Shape() == std::vector<size_t>({batch.size, 16, 24, 3}), name + ": [n, 16, 24, 3]");
            const float* pixels = batch.data.ReadData<float>();
            const float* labels = batch.labels.ReadData<float>();
            for (size_t b = 0; b < batch.size; ++b) {
                bool found = false;
                for (size_t i = 0; i < set.Count() && !found; ++i) {
                    const uint8_t* row = set.Row(i);
                    bool same = true;
                    for (size_t k = 0; k < set.RowBytes() && same; ++k)
                        // Within a float step: OpenCL may divide as a multiply.
                        same = std::abs(pixels[b * set.RowBytes() + k] - row[k] / 255.0f) <= 1e-6f;
                    if (same && labels[b] == static_cast<float>(set.Labels()[i])) {
                        found = seen.insert(i).second;
                    }
                }
                Check(found, name + ": each sample is a set row / 255 with its label, once");
            }
        }
        Check(seen.size() == 5, name + ": the epoch covers every readable image");
        batcher.ExcludeFiles({set.Files()[0]});
        Check(batcher.GetNumSamples() == 4, name + ": an excluded file is left out too");
        std::cout << name << ": batches match the set (" << (batcher.ImagesOnDevice() ? "on the device" : "gathered per batch")
                  << ")\n";
    }
}

void CheckChangesAndStops(const fs::path& folder, const DataRegistry::ImageDatasetEntry& entry) {
    const PreparedImageSpec spec{8, 8, ResizeMode::Exact};
    const std::string before = PreparedImageSetKey(entry, spec);
    Check(PrepareImageSet(entry, spec).set != nullptr, "prepared before the edit");
    // An image edited in place (same name, folder time unchanged) is a new set.
    const fs::path edited = folder / "a" / "a0.png";
    Write(edited, Gradient(31, 17, CV_8UC3, 99));
    fs::last_write_time(edited, fs::last_write_time(edited) + std::chrono::seconds(5));
    const std::string after = PreparedImageSetKey(entry, spec);
    Check(after != before, "an image edited in place changes the key");
    auto rebuilt = PrepareImageSet(entry, spec);
    Check(rebuilt.set && !rebuilt.from_cache, "the edited set is decoded again");

    // Stopped through the progress callback: no set, no file left behind.
    const PreparedImageSpec stopped_spec{10, 10, ResizeMode::Exact};
    const auto stopped = PrepareImageSet(entry, stopped_spec, 1, [](size_t, size_t) { return false; });
    Check(!stopped.set && stopped.error == "stopped", "a stopped run reports it");
    const fs::path stopped_path = PreparedImageSetPath(PreparedImageSetKey(entry, stopped_spec));
    Check(!fs::exists(stopped_path), "a stopped run leaves no set");
    for (const auto& file : fs::directory_iterator(stopped_path.parent_path()))
        Check(file.path().string().find(".tmp.") == std::string::npos, "a stopped run leaves no temp file");

    // A truncated file is not a set.
    const fs::path truncated = stopped_path.parent_path() / "truncated.cyximg";
    {
        std::ofstream out(truncated, std::ios::binary);
        out << "CYXIMG1";
        out.write(std::string(100, '\0').data(), 100);
    }
    std::string why;
    Check(PreparedImageSet::Open(truncated, &why) == nullptr && !why.empty(), "a truncated file is refused: " + why);
}

}  // namespace

int main() {
    const fs::path root = fs::temp_directory_path() / "cyxwiz_test_prepared_image_set";
    std::error_code ec;
    fs::remove_all(root, ec);
    SetImageCacheProjectRoot((root / "project").string());

    // Class folders: a = colour PNG, grey PNG, unreadable; b = RGBA PNG, 16-bit PNG, JPEG.
    const fs::path folder = root / "folders";
    Write(folder / "a" / "a0.png", Gradient(40, 30, CV_8UC3, 1));
    Write(folder / "a" / "a1.png", Gradient(17, 9, CV_8UC1, 2));
    {
        std::ofstream broken(folder / "a" / "a2.png");
        broken << "not an image";
    }
    Write(folder / "b" / "b0.png", Gradient(30, 40, CV_8UC4, 3));
    Write(folder / "b" / "b1.png", Gradient(48, 32, CV_16UC3, 4));
    Write(folder / "b" / "b2.jpg", Gradient(64, 64, CV_8UC3, 5));

    for (const char* file : {"a/a0.png", "a/a1.png", "b/b0.png", "b/b1.png", "b/b2.jpg"}) {
        CheckExactMatchesImageUtils(folder / file, 24, 16);  // down
        CheckExactMatchesImageUtils(folder / file, 80, 50);  // up
    }
    CheckModes(root / "modes");

    DataRegistry::ImageDatasetEntry folders;
    folders.folder_path = folder.string();
    folders.layout = 0;
    CheckSet(folders, "class folders");

    // A flat folder and a labels CSV with class names.
    const fs::path flat = root / "flat";
    fs::create_directories(flat);
    {
        std::ofstream csv(root / "labels.csv");
        csv << "image,label\n";
        const char* rows[][2] = {{"p0.png", "cat"}, {"p1.png", "dog"}, {"p2.png", "cat"},
                                 {"p3.png", "dog"}, {"p4.jpg", "cat"}, {"p5.png", "dog"}};
        for (int i = 0; i < 6; ++i) {
            csv << rows[i][0] << ',' << rows[i][1] << '\n';
            if (i == 5) {
                std::ofstream broken(flat / rows[i][0]);
                broken << "not an image";
            } else {
                Write(flat / rows[i][0], Gradient(20 + i * 7, 30 - i * 3, i == 2 ? CV_8UC1 : CV_8UC3, 10 + i));
            }
        }
    }
    DataRegistry::ImageDatasetEntry csv;
    csv.folder_path = flat.string();
    csv.labels_csv = (root / "labels.csv").string();
    csv.layout = 1;
    CheckSet(csv, "labels CSV");

    CheckBatcher(folders);
    CheckChangesAndStops(folder, folders);

    SetImageCacheProjectRoot("");
    fs::remove_all(root, ec);
    std::cout << "test_prepared_image_set: all checks passed\n";
    return 0;
}
