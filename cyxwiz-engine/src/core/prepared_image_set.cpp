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
#include <cstring>
#include <fstream>
#include <map>
#include <mutex>
#include <set>
#include <sstream>
#include <thread>

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
    for (const auto& row : ImageFileIdentityRows(ListImageSet(entry))) hash = Fnv(hash, row + '\n');
    return Hex(hash);
}

fs::path PreparedImageSetPath(const std::string& key) {
    return ImageCacheDir("images") / (key + kExtension);
}

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
    std::error_code ec;
    if (fs::exists(path, ec)) {
        std::string why;
        if (auto set = PreparedImageSet::Open(path, &why); set && set->Key() == key) {
            fs::last_write_time(path, fs::file_time_type::clock::now(), ec);  // recently used
            result.set = std::move(set);
            result.from_cache = true;
            return result;
        }
        spdlog::warn("Prepared images: {}; decoding again", why.empty() ? "the set's key differs" : why);
    }

    const auto started = std::chrono::steady_clock::now();
    const ImageSetListing listing = ListImageSet(entry);
    const size_t count = listing.files.size();
    if (count == 0) {
        result.error = "the image Data Input has no images";
        return result;
    }
    const size_t row_bytes = static_cast<size_t>(spec.width) * static_cast<size_t>(spec.height) * 3;
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

    const size_t workers = std::max<size_t>(1, std::min<size_t>(
        count / kChunk + 1, threads > 0 ? static_cast<size_t>(threads)
                                        : std::max(1u, std::thread::hardware_concurrency())));
    std::atomic<size_t> next_chunk{0}, done{0};
    std::atomic<bool> stop{false}, write_failed{false};
    std::mutex write_mutex, unreadable_mutex;
    std::vector<std::pair<size_t, std::string>> unreadable;
    const auto work = [&] {
        std::vector<uint8_t> buffer(kChunk * row_bytes);
        while (!stop) {
            const size_t begin = next_chunk.fetch_add(1) * kChunk;
            if (begin >= count) return;
            const size_t end = std::min(count, begin + kChunk);
            for (size_t i = begin; i < end; ++i) {
                uint8_t* row = buffer.data() + (i - begin) * row_bytes;
                if (!DecodePreparedImage(listing.files[i], spec, row)) {
                    std::memset(row, 0, row_bytes);
                    std::lock_guard<std::mutex> lock(unreadable_mutex);
                    unreadable.emplace_back(i, listing.files[i]);
                }
            }
            {
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
    if (stop || done < count) return give_up("stopped");

    std::sort(unreadable.begin(), unreadable.end());
    json header;
    header["format"] = kFormat;
    header["key"] = key;
    header["count"] = count;
    header["height"] = spec.height;
    header["width"] = spec.width;
    header["channels"] = 3;
    header["mode"] = ModeName(spec.mode);
    header["labels"] = listing.labels;
    header["files"] = listing.files;
    header["class_names"] = listing.class_names;
    std::vector<std::string> unreadable_files;
    for (const auto& item : unreadable) unreadable_files.push_back(item.second);
    header["unreadable"] = unreadable_files;
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
    if (ec) {
        fs::remove(temp, ec);  // another run wrote the same set meanwhile; theirs is used
    }
    std::string why;
    result.set = PreparedImageSet::Open(path, &why);
    if (!result.set) {
        result.error = why;
        return result;
    }
    result.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    spdlog::info("Prepared images: {} images at {}x{} ({}) in {:.1f} s on {} threads, {} unreadable -> {}", count,
                 spec.width, spec.height, ModeName(spec.mode), result.seconds, workers, unreadable_files.size(),
                 path.string());
    PruneImageCache();
    return result;
}

}  // namespace cyxwiz
