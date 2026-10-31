#include "prepared_image_set.h"

#include "cache_prune.h"
#include "datasets/image_csv_dataset.h"
#include "datasets/image_folder_dataset.h"

#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <mutex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <utility>
#include <variant>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace cyxwiz {

namespace {

namespace fs = std::filesystem;
using json = nlohmann::json;

// File: a 64-byte preamble, the pixels from byte 64, then the JSON header.
constexpr char kMagic[8] = {'C', 'Y', 'X', 'I', 'M', 'G', '1', '\0'};
constexpr uint64_t kPixelOffset = 64;
constexpr const char* kFormat = "image_u8_v1";
constexpr const char* kExtension = ".cyximg";
constexpr size_t kChunk = 32;  // images a worker decodes before writing

struct Preamble {
    char magic[8];
    uint64_t pixel_offset;
    uint64_t pixel_bytes;
    uint64_t header_offset;
    uint64_t header_bytes;
    uint64_t reserved[3];
};
static_assert(sizeof(Preamble) == kPixelOffset, "the preamble is 64 bytes");

std::mutex project_root_mutex;
std::string project_root;

uint64_t Fnv(uint64_t hash, const std::string& text) {
    for (unsigned char c : text) {
        hash ^= c;
        hash *= 1099511628211ull;
    }
    return hash;
}

std::string Hex(uint64_t value) {
    char text[17];
    std::snprintf(text, sizeof(text), "%016llx", static_cast<unsigned long long>(value));
    return text;
}

std::string DirectorySignature(const std::string& folder) {
    std::error_code ec;
    std::ostringstream out;
    out << static_cast<long long>(fs::last_write_time(folder, ec).time_since_epoch().count());
    for (const auto& entry : fs::directory_iterator(folder, ec)) {
        if (entry.is_directory(ec)) {
            out << '|' << entry.path().filename().string() << ':'
                << static_cast<long long>(fs::last_write_time(entry.path(), ec).time_since_epoch().count());
        }
    }
    return out.str();
}

const char* ModeName(ResizeMode mode) {
    switch (mode) {
        case ResizeMode::AspectFit: return "fit";
        case ResizeMode::AspectFill: return "fill";
        case ResizeMode::Center: return "center";
        default: return "exact";
    }
}

// Float RGB in [0, 1], as ImageUtils::LoadImage reads it.
bool ReadRgbFloat(const std::string& path, cv::Mat& rgb) {
    cv::Mat img = cv::imread(path, cv::IMREAD_UNCHANGED);
    if (img.empty()) return false;
    switch (img.channels()) {
        case 1: cv::cvtColor(img, img, cv::COLOR_GRAY2RGB); break;
        case 3: cv::cvtColor(img, img, cv::COLOR_BGR2RGB); break;
        case 4: cv::cvtColor(img, img, cv::COLOR_BGRA2RGB); break;
        default: return false;
    }
    const double scale = img.depth() == CV_16U ? 1.0 / 65535.0 : img.depth() == CV_8U ? 1.0 / 255.0 : 1.0;
    img.convertTo(rgb, CV_32FC3, scale);
    return true;
}

cv::Mat ResizeFloat(const cv::Mat& src, int width, int height) {
    if (src.cols == width && src.rows == height) return src;
    const int method = (width < src.cols || height < src.rows) ? cv::INTER_AREA : cv::INTER_LANCZOS4;
    cv::Mat dst;
    cv::resize(src, dst, cv::Size(width, height), 0, 0, method);
    return dst;
}

// `src` centred on a black width x height canvas, cropped where it is larger.
cv::Mat CentreOnCanvas(const cv::Mat& src, int width, int height) {
    cv::Mat canvas(height, width, CV_32FC3, cv::Scalar(0, 0, 0));
    const int w = std::min(width, src.cols), h = std::min(height, src.rows);
    const cv::Rect from((src.cols - w) / 2, (src.rows - h) / 2, w, h);
    const cv::Rect to((width - w) / 2, (height - h) / 2, w, h);
    src(from).copyTo(canvas(to));
    return canvas;
}

// Open sets of this process: the cache rule never removes them.
std::mutex open_mutex;
std::multiset<std::string>& OpenPaths() {
    static auto* paths = new std::multiset<std::string>();  // never destroyed: sets may outlive statics
    return *paths;
}

void PruneImageCache() {
    CacheFolder folder;
    folder.path = ImageCacheDir("images");
    folder.extension = kExtension;
    folder.in_use = [](const fs::path& path) {
        std::lock_guard<std::mutex> lock(open_mutex);
        return OpenPaths().count(path.lexically_normal().string()) > 0;
    };
    const auto pruned = PruneCacheFolder(folder);
    if (pruned.removed > 0)
        spdlog::info("Prepared images: the cache rule removed {} older set{} ({} bytes)", pruned.removed,
                     pruned.removed == 1 ? "" : "s", pruned.freed_bytes);
}

}  // namespace

void SetImageCacheProjectRoot(const std::string& root) {
    std::lock_guard<std::mutex> lock(project_root_mutex);
    project_root = root;
}

fs::path ImageCacheDir(const std::string& kind) {
    std::lock_guard<std::mutex> lock(project_root_mutex);
    if (!project_root.empty()) return fs::path(project_root) / "cache" / kind;
    return fs::temp_directory_path() / "cyxwiz" / kind;
}

ImageSetListing ListImageSet(const DataRegistry::ImageDatasetEntry& entry) {
    // The size does not matter for listing; the datasets read no pixels for it.
    std::shared_ptr<Dataset> dataset;
    if (entry.layout == 1 && !entry.labels_csv.empty())
        dataset = std::make_shared<ImageCSVDataset>(entry.folder_path, entry.labels_csv, 1, 1, 1);
    else
        dataset = std::make_shared<ImageFolderDataset>(entry.folder_path, 1, 1, 1);
    ImageSetListing listing;
    listing.files.reserve(dataset->Size());
    listing.labels.reserve(dataset->Size());
    for (size_t i = 0; i < dataset->Size(); ++i) {
        listing.files.push_back(dataset->GetItemSource(i));
        listing.labels.push_back(dataset->GetItemLabel(i));
    }
    listing.class_names = dataset->GetInfo().class_names;
    return listing;
}

std::vector<std::string> ImageFileIdentityRows(const ImageSetListing& listing) {
    std::vector<std::string> rows;
    rows.reserve(listing.files.size());
    std::error_code ec;
    for (const auto& file : listing.files) {
        const auto size = fs::file_size(file, ec);
        const auto time = static_cast<long long>(fs::last_write_time(file, ec).time_since_epoch().count());
        rows.push_back(file + '|' + std::to_string(ec ? 0 : size) + '|' + std::to_string(time));
    }
    std::sort(rows.begin(), rows.end());
    return rows;
}

std::string ImageSetSignature(const DataRegistry::ImageDatasetEntry& entry) {
    std::string signature = DirectorySignature(entry.folder_path);
    if (!entry.labels_csv.empty()) {
        std::error_code ec;
        signature += '|' + std::to_string(static_cast<long long>(
                               fs::last_write_time(entry.labels_csv, ec).time_since_epoch().count()));
    }
    return signature;
}

bool DecodePreparedImage(const std::string& path, const PreparedImageSpec& spec, uint8_t* out) {
    const int width = spec.width, height = spec.height;
    if (width <= 0 || height <= 0) return false;
    cv::Mat rgb;
    if (!ReadRgbFloat(path, rgb)) return false;
    cv::Mat image;
    switch (spec.mode) {
        case ResizeMode::AspectFit:
        case ResizeMode::AspectFill: {
            const double sx = static_cast<double>(width) / rgb.cols, sy = static_cast<double>(height) / rgb.rows;
            const double s = spec.mode == ResizeMode::AspectFit ? std::min(sx, sy) : std::max(sx, sy);
            const int w = std::max(1, static_cast<int>(std::lround(rgb.cols * s)));
            const int h = std::max(1, static_cast<int>(std::lround(rgb.rows * s)));
            image = CentreOnCanvas(ResizeFloat(rgb, w, h), width, height);
            break;
        }
        case ResizeMode::Center:
            image = CentreOnCanvas(rgb, width, height);
            break;
        default:
            image = ResizeFloat(rgb, width, height);
            break;
    }
    cv::Mat target(height, width, CV_8UC3, out);
    image.convertTo(target, CV_8UC3, 255.0);  // rounds and clamps (Lanczos can overshoot)
    return target.data == out;
}

PreparedImageSet::~PreparedImageSet() {
    if (!mapping_) return;
#ifdef _WIN32
    UnmapViewOfFile(mapping_);
#else
    munmap(mapping_, mapped_bytes_);
#endif
    std::lock_guard<std::mutex> lock(open_mutex);
    auto& paths = OpenPaths();
    const auto it = paths.find(path_.lexically_normal().string());
    if (it != paths.end()) paths.erase(it);
}

std::shared_ptr<const PreparedImageSet> PreparedImageSet::Open(const fs::path& path, std::string* error) {
    const auto fail = [&](const std::string& why) -> std::shared_ptr<const PreparedImageSet> {
        if (error) *error = why;
        return nullptr;
    };
    std::error_code ec;
    const uint64_t size = fs::file_size(path, ec);
    if (ec || size < kPixelOffset) return fail("no prepared image set at " + path.string());

    std::shared_ptr<PreparedImageSet> set(new PreparedImageSet());
    set->path_ = path;
#ifdef _WIN32
    HANDLE file = CreateFileW(path.wstring().c_str(), GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_DELETE, nullptr,
                              OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (file == INVALID_HANDLE_VALUE) return fail("cannot open " + path.string());
    HANDLE mapping = CreateFileMappingW(file, nullptr, PAGE_READONLY, 0, 0, nullptr);
    CloseHandle(file);
    if (!mapping) return fail("cannot map " + path.string());
    void* view = MapViewOfFile(mapping, FILE_MAP_READ, 0, 0, 0);
    CloseHandle(mapping);
    if (!view) return fail("cannot map " + path.string());
#else
    const int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) return fail("cannot open " + path.string());
    void* view = mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
    ::close(fd);
    if (view == MAP_FAILED) return fail("cannot map " + path.string());
#endif
    set->mapping_ = view;
    set->mapped_bytes_ = static_cast<size_t>(size);
    {
        std::lock_guard<std::mutex> lock(open_mutex);
        OpenPaths().insert(path.lexically_normal().string());
    }

    const auto* bytes = static_cast<const uint8_t*>(view);
    Preamble preamble;
    std::memcpy(&preamble, bytes, sizeof(preamble));
    if (std::memcmp(preamble.magic, kMagic, sizeof(kMagic)) != 0 || preamble.pixel_offset != kPixelOffset ||
        preamble.header_bytes == 0 || preamble.header_offset + preamble.header_bytes > size ||
        preamble.pixel_offset + preamble.pixel_bytes > preamble.header_offset)
        return fail("not a complete prepared image set: " + path.string());
    try {
        const auto header = json::parse(bytes + preamble.header_offset,
                                        bytes + preamble.header_offset + preamble.header_bytes);
        if (header.value("format", "") != kFormat) return fail("unknown prepared image format in " + path.string());
        set->key_ = header.value("key", "");
        set->height_ = header.value("height", size_t{0});
        set->width_ = header.value("width", size_t{0});
        set->channels_ = header.value("channels", size_t{3});
        set->labels_ = header.value("labels", std::vector<int>{});
        set->files_ = header.value("files", std::vector<std::string>{});
        set->class_names_ = header.value("class_names", std::vector<std::string>{});
        set->unreadable_ = header.value("unreadable", std::vector<std::string>{});
    } catch (const std::exception& e) {
        return fail(std::string("unreadable prepared image header: ") + e.what());
    }
    if (set->files_.size() != set->labels_.size() ||
        set->Count() * set->RowBytes() != preamble.pixel_bytes)
        return fail("prepared image set sizes do not match: " + path.string());
    set->pixels_ = bytes + preamble.pixel_offset;
    return set;
}

std::string PreparedImageSetKey(const DataRegistry::ImageDatasetEntry& entry, const PreparedImageSpec& spec) {
    // Every file is looked at each time (no memo on the folder signature): a
    // file edited in place changes its own time, not its folder's.
    const std::string id = std::string(kFormat) + '|' + entry.folder_path + '|' + entry.labels_csv + '|' +
                           std::to_string(entry.layout) + '|' + std::to_string(spec.width) + 'x' +
                           std::to_string(spec.height) + '|' + ModeName(spec.mode);
    uint64_t hash = Fnv(1469598103934665603ull, id);
    if (!entry.labels_csv.empty()) {
        std::error_code ec;
        hash = Fnv(hash, '|' + std::to_string(fs::file_size(entry.labels_csv, ec)) + '|' +
                             std::to_string(static_cast<long long>(
                                 fs::last_write_time(entry.labels_csv, ec).time_since_epoch().count())));
    }
    const auto listing = ListImageSet(entry);
    // The class names in index order: a new class order (or a renamed class)
    // makes a new set, since the set stores label indices.
    for (const auto& name : listing.class_names) hash = Fnv(hash, "class|" + name + '\n');
    for (const auto& row : ImageFileIdentityRows(listing)) hash = Fnv(hash, row + '\n');
    return Hex(hash);
}

fs::path PreparedImageSetPath(const std::string& key) {
    return ImageCacheDir("images") / (key + kExtension);
}

namespace {

// What a set file says about its images besides the pixels.
struct SetHeader {
    std::string key;
    std::string folder, labels_csv;  // the image Data Input it was made from
    size_t height = 0, width = 0, channels = 3;
    std::string mode;
    std::vector<int> labels;
    std::vector<std::string> files;
    std::vector<std::string> class_names;
    std::vector<std::string> unreadable;
    json fixed = json::array();  // transforms done at prepare time
};

// Fills rows [begin, end) at `out`; false to give up with `error`.
using ProduceRows = std::function<bool(size_t begin, size_t end, uint8_t* out, std::string& error)>;

// The opened set at `path`, if it is there and has `key`; it counts as
// recently used.
std::shared_ptr<const PreparedImageSet> OpenCached(const fs::path& path, const std::string& key) {
    std::error_code ec;
    if (!fs::exists(path, ec)) return nullptr;
    std::string why;
    auto set = PreparedImageSet::Open(path, &why);
    if (set && set->Key() == key) {
        fs::last_write_time(path, fs::file_time_type::clock::now(), ec);
        return set;
    }
    spdlog::warn("Prepared images: {}; preparing again", why.empty() ? "the set's key differs" : why);
    return nullptr;
}

// Writes the set to a temp file beside `path` (rows by `produce` in chunks on
// `workers` threads, then the header from `finish`), renames it and opens it.
PreparedImageResult WriteImageSet(const fs::path& path, size_t count, size_t row_bytes, size_t chunk,
                                  size_t workers, const ProduceRows& produce,
                                  const std::function<SetHeader()>& finish, const PreparedImageProgress& progress) {
    PreparedImageResult result;
    const auto started = std::chrono::steady_clock::now();
    std::error_code ec;
    fs::create_directories(path.parent_path(), ec);
    const fs::path temp = path.string() + ".tmp." + std::to_string(
        std::chrono::steady_clock::now().time_since_epoch().count());
    std::fstream out(temp, std::ios::binary | std::ios::in | std::ios::out | std::ios::trunc);
    if (!out) {
        result.error = "cannot write " + temp.string();
        return result;
    }
    Preamble preamble{};
    std::memcpy(preamble.magic, kMagic, sizeof(kMagic));
    preamble.pixel_offset = kPixelOffset;
    preamble.pixel_bytes = static_cast<uint64_t>(count) * row_bytes;
    out.write(reinterpret_cast<const char*>(&preamble), sizeof(preamble));  // complete only at the end

    std::atomic<size_t> next_chunk{0}, done{0};
    std::atomic<bool> stop{false}, write_failed{false};
    std::mutex write_mutex;
    std::string produce_error;
    const auto work = [&] {
        std::vector<uint8_t> buffer(chunk * row_bytes);
        while (!stop) {
            const size_t begin = next_chunk.fetch_add(1) * chunk;
            if (begin >= count) return;
            const size_t end = std::min(count, begin + chunk);
            std::string why;
            if (!produce(begin, end, buffer.data(), why)) {
                std::lock_guard<std::mutex> lock(write_mutex);
                produce_error = why;
                stop = true;
                return;
            }
            std::lock_guard<std::mutex> lock(write_mutex);
            out.seekp(static_cast<std::streamoff>(kPixelOffset + begin * row_bytes));
            out.write(reinterpret_cast<const char*>(buffer.data()),
                      static_cast<std::streamsize>((end - begin) * row_bytes));
            if (!out) {
                write_failed = true;
                stop = true;
            }
            const size_t now = done.fetch_add(end - begin) + (end - begin);
            if (progress && !progress(now, count)) stop = true;
        }
    };
    std::vector<std::thread> pool;
    for (size_t w = 1; w < workers; ++w) pool.emplace_back(work);
    work();
    for (auto& t : pool) t.join();

    const auto give_up = [&](const std::string& why) {
        out.close();
        fs::remove(temp, ec);
        result.error = why;
        return result;
    };
    if (write_failed) return give_up("cannot write " + temp.string() + " (disk full?)");
    if (!produce_error.empty()) return give_up(produce_error);
    if (stop || done < count) return give_up("stopped");

    const SetHeader meta = finish();
    json header;
    header["format"] = kFormat;
    header["key"] = meta.key;
    header["folder"] = meta.folder;
    header["labels_csv"] = meta.labels_csv;
    header["count"] = count;
    header["height"] = meta.height;
    header["width"] = meta.width;
    header["channels"] = meta.channels;
    header["mode"] = meta.mode;
    header["fixed_transforms"] = meta.fixed;
    header["labels"] = meta.labels;
    header["files"] = meta.files;
    header["class_names"] = meta.class_names;
    header["unreadable"] = meta.unreadable;
    const std::string text = header.dump();
    preamble.header_offset = kPixelOffset + preamble.pixel_bytes;
    preamble.header_bytes = text.size();
    out.seekp(static_cast<std::streamoff>(preamble.header_offset));
    out.write(text.data(), static_cast<std::streamsize>(text.size()));
    out.seekp(0);
    out.write(reinterpret_cast<const char*>(&preamble), sizeof(preamble));
    out.close();
    if (!out) return give_up("cannot write " + temp.string() + " (disk full?)");

    fs::rename(temp, path, ec);
    if (ec) fs::remove(temp, ec);  // another run wrote the same set meanwhile; theirs is used
    std::string why;
    result.set = PreparedImageSet::Open(path, &why);
    if (!result.set) {
        result.error = why;
        return result;
    }
    result.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    PruneImageCache();
    return result;
}

// Every field the deterministic steps read, so another setting is another set.
json TransformIdentity(const std::vector<FixedImageStep>& steps) {
    json list = json::array();
    for (const auto& step : steps) {
        if (const auto* cv_step = std::get_if<CvImageStep>(&step)) {
            list.push_back(CvImageStepIdentity(*cv_step));
            continue;
        }
        const auto& op = std::get<image::ImageOp>(step);
        char sigma[32];
        std::snprintf(sigma, sizeof(sigma), "%.9g", op.sigma);
        list.push_back({{"kind", static_cast<int>(op.kind)}, {"height", op.height}, {"width", op.width},
                        {"kernel_size", op.kernel_size}, {"sigma", sigma},
                        {"morphology", static_cast<int>(op.morphology)}});
    }
    return list;
}

}  // namespace

PreparedImageResult PrepareImageSet(const DataRegistry::ImageDatasetEntry& entry,
                                    const PreparedImageSpec& spec,
                                    int threads,
                                    const PreparedImageProgress& progress) {
    PreparedImageResult result;
    if (spec.width <= 0 || spec.height <= 0) {
        result.error = "the image size must be positive";
        return result;
    }
    const std::string key = PreparedImageSetKey(entry, spec);
    const fs::path path = PreparedImageSetPath(key);
    if (auto set = OpenCached(path, key)) {
        result.set = std::move(set);
        result.from_cache = true;
        return result;
    }

    const ImageSetListing listing = ListImageSet(entry);
    const size_t count = listing.files.size();
    if (count == 0) {
        result.error = "the image Data Input has no images";
        return result;
    }
    const size_t row_bytes = static_cast<size_t>(spec.width) * static_cast<size_t>(spec.height) * 3;
    const size_t workers = std::max<size_t>(1, std::min<size_t>(
        count / kChunk + 1, threads > 0 ? static_cast<size_t>(threads)
                                        : std::max(1u, std::thread::hardware_concurrency())));
    std::mutex unreadable_mutex;
    std::vector<std::pair<size_t, std::string>> unreadable;
    const ProduceRows decode = [&](size_t begin, size_t end, uint8_t* out, std::string&) {
        for (size_t i = begin; i < end; ++i) {
            uint8_t* row = out + (i - begin) * row_bytes;
            if (!DecodePreparedImage(listing.files[i], spec, row)) {
                std::memset(row, 0, row_bytes);
                std::lock_guard<std::mutex> lock(unreadable_mutex);
                unreadable.emplace_back(i, listing.files[i]);
            }
        }
        return true;
    };
    const auto finish = [&] {
        SetHeader meta;
        meta.key = key;
        meta.folder = entry.folder_path;
        meta.labels_csv = entry.labels_csv;
        meta.height = static_cast<size_t>(spec.height);
        meta.width = static_cast<size_t>(spec.width);
        meta.mode = ModeName(spec.mode);
        meta.labels = listing.labels;
        meta.files = listing.files;
        meta.class_names = listing.class_names;
        std::sort(unreadable.begin(), unreadable.end());
        for (const auto& item : unreadable) meta.unreadable.push_back(item.second);
        return meta;
    };
    result = WriteImageSet(path, count, row_bytes, kChunk, workers, decode, finish, progress);
    if (result.set)
        spdlog::info("Prepared images: {} images at {}x{} ({}) in {:.1f} s on {} threads, {} unreadable -> {}",
                     count, spec.width, spec.height, ModeName(spec.mode), result.seconds, workers,
                     result.set->Unreadable().size(), path.string());
    return result;
}

ImageTransformSplit SplitFixedImageTransforms(const image::ImageAugmentation& compiled,
                                              const std::vector<CvImageStep>& cv_steps) {
    ImageTransformSplit split;
    split.per_batch = compiled;
    split.per_batch.ops.clear();
    bool fixed = true;
    for (auto& step : MergeImageSteps(compiled, cv_steps)) {
        if (const auto* cv_step = std::get_if<CvImageStep>(&step)) {
            if (!fixed)
                throw std::invalid_argument(CvImageStepText(*cv_step) +
                                            " is done once with the images: it must come before the first "
                                            "random transform");
            split.fixed.push_back(std::move(step));
            continue;
        }
        const auto& op = std::get<image::ImageOp>(step);
        fixed = fixed && !image::IsRandomImageOp(op.kind);
        if (fixed) split.fixed.push_back(op);
        else split.per_batch.ops.push_back(op);
    }
    return split;
}

PreparedImageResult PrepareImageSet(const DataRegistry::ImageDatasetEntry& entry,
                                    const PreparedImageSpec& spec,
                                    const std::vector<FixedImageStep>& fixed,
                                    int threads,
                                    const PreparedImageProgress& progress) {
    if (fixed.empty()) return PrepareImageSet(entry, spec, threads, progress);
    // Keyed on the decoded set's key: opened from the cache without decoding.
    const json identity = TransformIdentity(fixed);
    const std::string key = PreparedImageSetKeyFor(entry, spec, fixed);
    const fs::path path = PreparedImageSetPath(key);
    if (auto set = OpenCached(path, key)) {
        PreparedImageResult cached;
        cached.set = std::move(set);
        cached.from_cache = true;
        return cached;
    }
    PreparedImageResult decoded = PrepareImageSet(entry, spec, threads, progress);
    if (!decoded.set) return decoded;
    const auto& base = *decoded.set;

    // The steps as segments: a run of device transforms, or one OpenCV step,
    // each with the shape it receives (checked against that shape).
    struct Segment {
        image::ImageAugmentation device;
        const CvImageStep* cv = nullptr;
        image::ImageShape in, out;
    };
    std::vector<Segment> segments;
    image::ImageShape shape{base.Height(), base.Width(), base.Channels()};
    const image::ImageShape input = shape;
    size_t on_device = 0, on_cpu = 0;
    for (const auto& step : fixed) {
        const auto* cv_step = std::get_if<CvImageStep>(&step);
        const std::string why = cv_step ? ValidateCvImageStep(*cv_step, shape)
                                        : image::ValidateImageOp(std::get<image::ImageOp>(step), shape);
        if (!why.empty()) {
            decoded.set = nullptr;
            decoded.error = FixedImageStepText(step) + ": " + why;
            return decoded;
        }
        if (cv_step || segments.empty() || segments.back().cv) segments.push_back({{}, cv_step, shape, shape});
        Segment& segment = segments.back();
        if (cv_step) {
            ++on_cpu;
            shape = CvImageShapeAfter(*cv_step, shape);
        } else {
            ++on_device;
            segment.device.ops.push_back(std::get<image::ImageOp>(step));
            shape = image::ImageShapeAfter(segment.device.ops.back(), shape);
        }
        segment.out = shape;
    }
    const image::ImageShape output = shape;
    const size_t out_bytes = output.Size();
    // Chunks one at a time: device runs on the device, each OpenCV step on
    // every core across the chunk's images.
    const ProduceRows apply = [&](size_t begin, size_t end, uint8_t* out, std::string& error) {
        const size_t n = end - begin;
        std::vector<uint8_t> current(base.Row(begin), base.Row(begin) + n * input.Size()), next;
        for (const auto& segment : segments) {
            const size_t in_bytes = segment.in.Size(), seg_bytes = segment.out.Size();
            next.resize(n * seg_bytes);
            if (segment.cv) {
                std::mutex error_mutex;
                std::string failed;
                cv::parallel_for_(cv::Range(0, static_cast<int>(n)), [&](const cv::Range& range) {
                    for (int i = range.start; i < range.end; ++i) {
                        std::string why;
                        if (!ApplyCvImageStep(*segment.cv, segment.in, current.data() + i * in_bytes,
                                              next.data() + i * seg_bytes, &why)) {
                            std::lock_guard<std::mutex> lock(error_mutex);
                            if (failed.empty()) failed = base.Files()[begin + i] + ": " + why;
                        }
                    }
                });
                if (!failed.empty()) {
                    error = failed;
                    return false;
                }
            } else {
                std::vector<float> rows(n * in_bytes);
                for (size_t k = 0; k < rows.size(); ++k) rows[k] = current[k] / 255.0f;
                try {
                    std::mt19937 unused(0);  // fixed transforms draw nothing
                    const Tensor result =
                        segment.device.Apply(Tensor({n, in_bytes}, rows.data()), segment.in, false, unused);
                    const float* values = result.ReadData<float>();
                    for (size_t k = 0; k < n * seg_bytes; ++k)
                        next[k] = static_cast<uint8_t>(std::lround(std::clamp(values[k], 0.0f, 1.0f) * 255.0f));
                } catch (const std::exception& e) {
                    error = std::string("the fixed image transforms failed: ") + e.what();
                    return false;
                }
            }
            current.swap(next);
        }
        std::memcpy(out, current.data(), n * out_bytes);
        return true;
    };
    const auto finish = [&] {
        SetHeader meta;
        meta.key = key;
        meta.folder = entry.folder_path;
        meta.labels_csv = entry.labels_csv;
        meta.height = output.height;
        meta.width = output.width;
        meta.channels = output.channels;
        meta.mode = ModeName(spec.mode);
        meta.labels = base.Labels();
        meta.files = base.Files();
        meta.class_names = base.ClassNames();
        meta.unreadable = base.Unreadable();
        meta.fixed = identity;
        return meta;
    };
    const auto started = std::chrono::steady_clock::now();
    PreparedImageResult derived = WriteImageSet(path, base.Count(), out_bytes, 256, 1, apply, finish, progress);
    if (derived.set) {
        derived.seconds += decoded.seconds;
        spdlog::info("Prepared images: {} fixed step{} done once ({} on the device, {} OpenCV on the CPU; "
                     "{}x{}x{} -> {}x{}x{}) in {:.1f} s -> {}", fixed.size(), fixed.size() == 1 ? "" : "s",
                     on_device, on_cpu, input.height, input.width,
                     input.channels, output.height, output.width, output.channels,
                     std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count(),
                     path.string());
    }
    return derived;
}

std::string PreparedImageSetKeyFor(const DataRegistry::ImageDatasetEntry& entry, const PreparedImageSpec& spec,
                                   const std::vector<FixedImageStep>& fixed) {
    const std::string decoded = PreparedImageSetKey(entry, spec);
    if (fixed.empty()) return decoded;
    return Hex(Fnv(Fnv(1469598103934665603ull, decoded), TransformIdentity(fixed).dump()));
}

namespace {

// A header's fixed_transforms entry (TransformIdentity) in words.
std::string FixedTransformText(const json& entry) {
    if (entry.contains("cv")) return entry.value("text", std::string("OpenCV step"));
    image::ImageOp op;
    op.kind = static_cast<image::ImageOpKind>(entry.value("kind", 0));
    op.height = entry.value("height", 0);
    op.width = entry.value("width", 0);
    op.kernel_size = entry.value("kernel_size", 0);
    op.sigma = static_cast<float>(std::atof(entry.value("sigma", std::string("0")).c_str()));
    op.morphology = static_cast<image::MorphologyOp>(entry.value("morphology", 0));
    return ImageOpText(op);
}

bool SameFile(const std::string& a, const std::string& b) {
    return fs::path(a).lexically_normal() == fs::path(b).lexically_normal();
}

}  // namespace

std::vector<PreparedImageSetInfo> ListPreparedImageSets(const DataRegistry::ImageDatasetEntry& entry) {
    std::vector<PreparedImageSetInfo> sets;
    std::error_code ec;
    const fs::path dir = ImageCacheDir("images");
    if (!fs::is_directory(dir, ec)) return sets;
    for (const auto& item : fs::directory_iterator(dir, ec)) {
        if (!item.is_regular_file(ec) || item.path().extension() != kExtension) continue;
        std::ifstream in(item.path(), std::ios::binary);
        Preamble preamble{};
        if (!in.read(reinterpret_cast<char*>(&preamble), sizeof(preamble))) continue;
        if (std::memcmp(preamble.magic, kMagic, sizeof(kMagic)) != 0 || preamble.header_bytes == 0 ||
            preamble.header_bytes > (64ull << 20))
            continue;
        std::string text(static_cast<size_t>(preamble.header_bytes), '\0');
        in.seekg(static_cast<std::streamoff>(preamble.header_offset));
        if (!in.read(text.data(), static_cast<std::streamsize>(text.size()))) continue;
        try {
            const json header = json::parse(text);
            if (header.value("format", "") != kFormat) continue;
            if (!SameFile(header.value("folder", ""), entry.folder_path) ||
                header.value("labels_csv", "") != entry.labels_csv)
                continue;
            PreparedImageSetInfo info;
            info.key = header.value("key", "");
            info.path = item.path().string();
            info.width = header.value("width", 0);
            info.height = header.value("height", 0);
            info.channels = header.value("channels", 0);
            info.mode = header.value("mode", "");
            for (const auto& op : header.value("fixed_transforms", json::array())) info.fixed.push_back(FixedTransformText(op));
            info.bytes = static_cast<uint64_t>(fs::file_size(item.path(), ec));
            sets.push_back(std::move(info));
        } catch (const json::exception&) {
        }
    }
    std::sort(sets.begin(), sets.end(), [](const auto& a, const auto& b) {
        return a.bytes != b.bytes ? a.bytes > b.bytes : a.key < b.key;
    });
    return sets;
}

}  // namespace cyxwiz
