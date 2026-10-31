// The prepared image set (TOFIX178 1.3a): every image of an image Data Input
// decoded once to uint8 [N, H, W, 3] at the Resize size and mode, kept in
// cache/images. Generated image sets of both layouts (class folders, a
// labels CSV) with grey, RGBA, 16-bit, JPEG and unreadable files; no
// example dataset.

#include "../src/core/image_channel_stats.h"
#include "../src/core/image_dataset_batcher.h"
#include "../src/core/image_utils.h"
#include "../src/core/prepared_image_set.h"

#include <arrayfire.h>
#ifdef LoadImage
#undef LoadImage  // windows.h (via ArrayFire) would rename ImageUtils::LoadImage
#endif
#include <opencv2/opencv.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <set>
#include <stdexcept>
#include <variant>
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

// Channel statistics on each device agree with the host over the readable rows.
void CheckChannelStats(const DataRegistry::ImageDatasetEntry& entry) {
    const auto prepared = PrepareImageSet(entry, {24, 16, ResizeMode::Exact});
    Check(prepared.set != nullptr, "prepared for the statistics");
    const auto& set = *prepared.set;
    std::vector<double> sum(3, 0.0), squares(3, 0.0);
    size_t images = 0;
    for (size_t i = 0; i < set.Count(); ++i) {
        if (std::find(set.Unreadable().begin(), set.Unreadable().end(), set.Files()[i]) != set.Unreadable().end()) continue;
        ++images;
        for (size_t k = 0; k < set.RowBytes(); ++k) {
            const double v = set.Row(i)[k] / 255.0;
            sum[k % 3] += v;
            squares[k % 3] += v * v;
        }
    }
    const double n = static_cast<double>(images) * 24 * 16;
    const int available = af::getAvailableBackends();
    for (const af::Backend backend : {AF_BACKEND_CPU, AF_BACKEND_CUDA, AF_BACKEND_OPENCL}) {
        if (!(available & backend)) continue;
        af::setBackend(backend);
        size_t calls = 0;
        const ImageChannelStats stats = MeasureImageChannelStats(set, [&](size_t, size_t) { ++calls; });
        Check(stats.images == 5 && images == 5 && stats.mean.size() == 3 && calls > 0, "five readable images measured");
        for (size_t c = 0; c < 3; ++c) {
            const double mean = sum[c] / n;
            const double sd = std::sqrt(squares[c] / n - mean * mean);
            Check(std::abs(stats.mean[c] - mean) < 1e-5 && std::abs(stats.std_dev[c] - sd) < 1e-5,
                  "channel " + std::to_string(c) + " mean and std as on the host");
        }
    }
    std::atomic<bool> cancel{true};
    Check(MeasureImageChannelStats(set, {}, &cancel).mean.empty(), "a cancelled measure gives nothing");
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
        ImageDatasetBatcher batcher(prepared.set, 2, {1.0f, 0.0f, 0.0f, false, 5}, true, 5);
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

// Fixed transforms done once (1.3c): the leading ones that draw nothing,
// applied on the device to the decoded set; each row within half a step of
// the per-batch float path, on every backend; the batcher reads C = 1.
void CheckFixedTransforms(const DataRegistry::ImageDatasetEntry& entry) {
    using image::ImageOp;
    using image::ImageOpKind;
    ImageOp crop;
    crop.kind = ImageOpKind::CenterCrop;
    crop.height = 8;
    crop.width = 12;
    ImageOp grey;
    grey.kind = ImageOpKind::Grayscale;
    ImageOp erode;
    erode.kind = ImageOpKind::Morphology;
    erode.morphology = image::MorphologyOp::Erode;
    erode.kernel_size = 3;
    ImageOp blur;
    blur.kind = ImageOpKind::GaussianBlur;
    blur.kernel_size = 3;
    blur.sigma = 0.8f;
    ImageOp flip;
    flip.kind = ImageOpKind::HorizontalFlip;

    image::ImageAugmentation compiled;
    compiled.ops = {crop, grey, erode, flip, blur};
    compiled.normalize = true;
    const auto split = SplitFixedImageTransforms(compiled, {});
    Check(split.fixed.size() == 3 && split.per_batch.ops.size() == 2 && split.per_batch.normalize,
          "fixed: the transforms before the first random one; the rest and Normalize per batch");

    const PreparedImageSpec spec{24, 16, ResizeMode::Exact};
    const std::vector<FixedImageStep> fixed = {crop, grey, erode, blur};
    auto base = PrepareImageSet(entry, spec);
    Check(base.set != nullptr, "the decoded set for the fixed transforms");
    auto derived = PrepareImageSet(entry, spec, fixed);
    Check(derived.set && !derived.from_cache, "the fixed transforms are done once: " + derived.error);
    const auto& set = *derived.set;
    Check(set.Count() == base.set->Count() && set.Height() == 8 && set.Width() == 12 && set.Channels() == 1,
          "the done-once set is [N, 8, 12, 1]");
    Check(set.Labels() == base.set->Labels() && set.Unreadable() == base.set->Unreadable(),
          "labels and unreadable files carried over");

    image::ImageAugmentation steps;
    steps.ops = {crop, grey, erode, blur};
    const image::ImageShape input{16, 24, 3};
    const int available = af::getAvailableBackends();
    for (const af::Backend backend : {AF_BACKEND_CPU, AF_BACKEND_CUDA, AF_BACKEND_OPENCL}) {
        if (!(available & backend)) continue;
        af::setBackend(backend);
        std::vector<float> rows(base.set->Count() * input.Size());
        for (size_t k = 0; k < rows.size(); ++k) rows[k] = base.set->Row(0)[k] / 255.0f;
        std::mt19937 rng(0);
        const Tensor expected = steps.Apply(Tensor({base.set->Count(), input.Size()}, rows.data()), input, false, rng);
        const float* values = expected.ReadData<float>();
        double worst = 0.0;
        for (size_t k = 0; k < set.Count() * set.RowBytes(); ++k)
            worst = std::max(worst, std::abs(set.Row(0)[k] / 255.0 - std::clamp(values[k], 0.0f, 1.0f)));
        Check(worst <= 0.5 / 255.0 + 1e-5, "within half a step of the per-batch path (worst " +
                                               std::to_string(worst * 255.0) + " steps)");
    }

    // Opened from the cache without decoding again (the decoded set is gone).
    const fs::path base_path = base.set->Path();
    const fs::path derived_path = set.Path();
    base = {};
    derived = {};
    std::error_code ec;
    fs::remove(base_path, ec);
    auto again = PrepareImageSet(entry, spec, fixed);
    Check(again.set && again.from_cache && again.set->Path() == derived_path, "the done-once set is reused");
    Check(!fs::exists(base_path), "reusing it does not decode the images again");

    ImageDatasetBatcher batcher(again.set, 3, {1.0f, 0.0f, 0.0f, false, 1}, false, 1);
    batcher.SetImageTransforms(split.per_batch);
    const auto batch = batcher.GetNextBatch();
    Check(batch.data.Shape() == std::vector<size_t>({batch.size, 8, 12, 1}), "the batcher reads one channel");

    // The cache lists the sets made from this folder, with the key the graph would use.
    Check(PreparedImageSetKeyFor(entry, spec, fixed) == again.set->Key(), "the key for spec + fixed is the set's");
    Check(PreparedImageSetKeyFor(entry, spec, {}) == PreparedImageSetKey(entry, spec), "no fixed: the decoded key");
    const auto listed = ListPreparedImageSets(entry);
    const auto it = std::find_if(listed.begin(), listed.end(), [&](const auto& s) { return s.key == again.set->Key(); });
    Check(it != listed.end(), "the done-once set is listed for its folder");
    Check(it->width == 12 && it->height == 8 && it->channels == 1 && it->mode == "exact" &&
              it->bytes == fs::file_size(derived_path),
          "listed with its size, mode and bytes");
    Check(it->fixed == std::vector<std::string>({"Center Crop 12 \xC3\x97 8", "Grayscale", "Erode 3 \xC3\x97 3",
                                                 "Gaussian Blur 3 \xC3\x97 3, sigma 0.8"}),
          "listed with its transforms: " + (it->fixed.empty() ? std::string() : it->fixed.back()));
    DataRegistry::ImageDatasetEntry other = entry;
    other.folder_path += "_elsewhere";
    Check(ListPreparedImageSets(other).empty(), "another folder lists none of them");
}

// The OpenCV steps (1.3f) between device transforms: Center Crop on the
// device, CLAHE on the CPU, Grayscale on the device, in that order, each
// rounded to uint8; CLAHE is OpenCV's own on the cropped images.
void CheckOpenCvSteps(const DataRegistry::ImageDatasetEntry& entry) {
    image::ImageOp crop;
    crop.kind = image::ImageOpKind::CenterCrop;
    crop.height = 8;
    crop.width = 12;
    image::ImageOp grey;
    grey.kind = image::ImageOpKind::Grayscale;
    image::ImageOp flip;
    flip.kind = image::ImageOpKind::HorizontalFlip;
    CvImageStep clahe;
    clahe.kind = CvImageStepKind::Contrast;
    clahe.tile_grid = 4;
    clahe.position = 1;  // after the crop

    image::ImageAugmentation compiled;
    compiled.ops = {crop, grey, flip};
    const auto split = SplitFixedImageTransforms(compiled, {clahe});
    Check(split.fixed.size() == 3 && std::holds_alternative<CvImageStep>(split.fixed[1]) &&
              split.per_batch.ops.size() == 1,
          "crop, CLAHE and Grayscale done once; the flip per batch");
    CvImageStep late = clahe;
    late.position = 3;
    bool refused = false;
    try {
        SplitFixedImageTransforms(compiled, {late});
    } catch (const std::invalid_argument&) {
        refused = true;
    }
    Check(refused, "an OpenCV step after a random transform is refused");

    const PreparedImageSpec spec{24, 16, ResizeMode::Exact};
    const auto base = PrepareImageSet(entry, spec);
    const auto derived = PrepareImageSet(entry, spec, split.fixed);
    Check(base.set && derived.set, "the set with an OpenCV step: " + derived.error);
    const auto& set = *derived.set;
    Check(set.Height() == 8 && set.Width() == 12 && set.Channels() == 1, "[N, 8, 12, 1]");

    // Expected: the crop (exact), OpenCV's CLAHE on Lab L, then Grayscale on the device.
    const size_t n = base.set->Count();
    std::vector<float> after_clahe;
    for (size_t i = 0; i < n; ++i) {
        const cv::Mat row(16, 24, CV_8UC3, const_cast<uint8_t*>(base.set->Row(i)));
        cv::Mat cropped = row(cv::Rect(6, 4, 12, 8)).clone(), lab, rgb;
        cv::cvtColor(cropped, lab, cv::COLOR_RGB2Lab);
        std::vector<cv::Mat> planes;
        cv::split(lab, planes);
        cv::createCLAHE(2.0, cv::Size(4, 4))->apply(planes[0], planes[0]);
        cv::merge(planes, lab);
        cv::cvtColor(lab, rgb, cv::COLOR_Lab2RGB);
        for (size_t k = 0; k < rgb.total() * 3; ++k) after_clahe.push_back(rgb.data[k] / 255.0f);
    }
    image::ImageAugmentation grey_only;
    grey_only.ops = {grey};
    std::mt19937 rng(0);
    const Tensor expected =
        grey_only.Apply(Tensor({n, 8 * 12 * 3}, after_clahe.data()), image::ImageShape{8, 12, 3}, false, rng);
    const float* values = expected.ReadData<float>();
    double worst = 0.0;
    for (size_t k = 0; k < n * set.RowBytes(); ++k)
        worst = std::max(worst, std::abs(set.Row(0)[k] / 255.0 - std::clamp(values[k], 0.0f, 1.0f)));
    Check(worst <= 0.5 / 255.0 + 1e-5, "crop, CLAHE, Grayscale in order (worst " + std::to_string(worst * 255.0) + " steps)");

    // Keyed and listed with the step and its settings.
    CvImageStep other = clahe;
    other.clip_limit = 3.0f;
    std::vector<FixedImageStep> changed = split.fixed;
    changed[1] = other;
    Check(PreparedImageSetKeyFor(entry, spec, split.fixed) == set.Key() &&
              PreparedImageSetKeyFor(entry, spec, changed) != set.Key(),
          "another clip limit is another set");
    const auto listed = ListPreparedImageSets(entry);
    const auto it = std::find_if(listed.begin(), listed.end(), [&](const auto& s) { return s.key == set.Key(); });
    Check(it != listed.end() &&
              it->fixed == std::vector<std::string>({"Center Crop 12 \xC3\x97 8", "CLAHE 2.0 / 4", "Grayscale"}),
          "listed with the OpenCV step in words");
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

// Class index = position in the sorted class names (numbers as numbers);
// files in name order. Neither CSV row order nor directory order decides.
void CheckClassOrder(const fs::path& root) {
    const fs::path flat = root / "order_flat";
    {
        std::ofstream csv(root / "order.csv");
        csv << "image,label\n";
        const char* rows[][2] = {{"q0.png", "10"}, {"q1.png", "2"}, {"q2.png", "1"}, {"q3.png", "2"}};
        for (const auto& row : rows) {
            csv << row[0] << ',' << row[1] << '\n';
            Write(flat / row[0], Gradient(12, 10, CV_8UC3, 20));
        }
    }
    DataRegistry::ImageDatasetEntry csv;
    csv.folder_path = flat.string();
    csv.labels_csv = (root / "order.csv").string();
    csv.layout = 1;
    const auto listed = ListImageSet(csv);
    Check(listed.class_names == std::vector<std::string>({"1", "2", "10"}), "CSV classes sort as numbers");
    std::vector<std::string> named;
    for (size_t i = 0; i < listed.files.size(); ++i)
        named.push_back(listed.class_names[listed.labels[i]] + "@" + fs::path(listed.files[i]).filename().string());
    std::sort(named.begin(), named.end());
    Check(named == std::vector<std::string>({"10@q0.png", "1@q2.png", "2@q1.png", "2@q3.png"}),
          "each CSV file keeps its own label");

    for (const char* name : {"dog", "cat"}) {
        const fs::path dir = root / "order_folders" / name;
        Write(dir / "z.png", Gradient(12, 10, CV_8UC3, 21));
        Write(dir / "a.png", Gradient(12, 10, CV_8UC3, 22));
    }
    DataRegistry::ImageDatasetEntry folders;
    folders.folder_path = (root / "order_folders").string();
    folders.layout = 0;
    const auto in_folders = ListImageSet(folders);
    Check(in_folders.class_names == std::vector<std::string>({"cat", "dog"}), "class folders sort by name");
    Check(in_folders.labels == std::vector<int>({0, 0, 1, 1}), "cat is 0, dog is 1");
    Check(fs::path(in_folders.files[0]).filename() == "a.png" && fs::path(in_folders.files[1]).filename() == "z.png",
          "files within a class in name order");

    for (const char* name : {"10", "9"}) Write(root / "order_numbers" / name / "x.png", Gradient(12, 10, CV_8UC3, 23));
    DataRegistry::ImageDatasetEntry numbers;
    numbers.folder_path = (root / "order_numbers").string();
    numbers.layout = 0;
    Check(ListImageSet(numbers).class_names == std::vector<std::string>({"9", "10"}),
          "numbered class folders sort as numbers");
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

    CheckClassOrder(root);
    CheckChannelStats(folders);
    CheckBatcher(folders);
    CheckFixedTransforms(folders);
    CheckOpenCvSteps(folders);
    CheckChangesAndStops(folder, folders);

    SetImageCacheProjectRoot("");
    fs::remove_all(root, ec);
    std::cout << "test_prepared_image_set: all checks passed\n";
    return 0;
}
