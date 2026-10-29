#include "image_index.h"

#include "image_quality_analysis.h"

#include <arrow/api.h>
#include <opencv2/imgcodecs.hpp>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <array>
#include <filesystem>
#include <fstream>
#include <map>
#include <thread>

namespace fs = std::filesystem;

namespace cyxwiz {

namespace {

uint32_t Be16(const unsigned char* p) { return (uint32_t(p[0]) << 8) | p[1]; }
uint32_t Be32(const unsigned char* p) { return (uint32_t(p[0]) << 24) | (uint32_t(p[1]) << 16) | (uint32_t(p[2]) << 8) | p[3]; }
uint32_t Le16(const unsigned char* p) { return uint32_t(p[0]) | (uint32_t(p[1]) << 8); }
int32_t Le32(const unsigned char* p) {
    return static_cast<int32_t>(uint32_t(p[0]) | (uint32_t(p[1]) << 8) | (uint32_t(p[2]) << 16) | (uint32_t(p[3]) << 24));
}

// JPEG: walk the markers to the first start-of-frame.
bool JpegHeader(std::ifstream& in, ImageHeader& h) {
    unsigned char b[8];
    in.seekg(2);
    while (in.read(reinterpret_cast<char*>(b), 2)) {
        if (b[0] != 0xFF) return false;
        unsigned char marker = b[1];
        while (marker == 0xFF) {  // fill bytes
            if (!in.read(reinterpret_cast<char*>(&marker), 1)) return false;
        }
        if (marker == 0xD8 || marker == 0x01 || (marker >= 0xD0 && marker <= 0xD7)) continue;  // no length
        if (!in.read(reinterpret_cast<char*>(b), 2)) return false;
        const uint32_t length = Be16(b);
        if (length < 2) return false;
        const bool frame = marker >= 0xC0 && marker <= 0xCF && marker != 0xC4 && marker != 0xC8 && marker != 0xCC;
        if (frame) {
            if (!in.read(reinterpret_cast<char*>(b), 6)) return false;
            h.height = static_cast<int>(Be16(b + 1));
            h.width = static_cast<int>(Be16(b + 3));
            const int components = b[5];
            h.channels = components == 1 ? 1 : 3;  // CMYK and YCCK decode to colour
            return h.width > 0 && h.height > 0;
        }
        in.seekg(length - 2, std::ios::cur);
    }
    return false;
}

// PNG: the IHDR chunk follows the signature.
bool PngHeader(const unsigned char* b, size_t n, ImageHeader& h) {
    if (n < 26 || std::string(reinterpret_cast<const char*>(b + 12), 4) != "IHDR") return false;
    h.width = static_cast<int>(Be32(b + 16));
    h.height = static_cast<int>(Be32(b + 20));
    switch (b[25]) {  // colour type
        case 0: h.channels = 1; break;  // grey
        case 2: h.channels = 3; break;  // colour
        case 3: h.channels = 3; break;  // palette
        case 4: h.channels = 4; break;  // grey + alpha decodes as colour + alpha
        case 6: h.channels = 4; break;  // colour + alpha
        default: return false;
    }
    return h.width > 0 && h.height > 0;
}

// BMP: the DIB header after the 14-byte file header.
bool BmpHeader(const unsigned char* b, size_t n, ImageHeader& h) {
    if (n < 30) return false;
    const int32_t dib = Le32(b + 14);
    int bits = 0;
    if (dib == 12) {
        h.width = static_cast<int>(Le16(b + 18));
        h.height = static_cast<int>(Le16(b + 20));
        bits = static_cast<int>(Le16(b + 24));
    } else {
        h.width = Le32(b + 18);
        h.height = std::abs(Le32(b + 22));
        bits = static_cast<int>(Le16(b + 28));
    }
    // A palette (8 bits or fewer) decodes as grey or colour by its entries,
    // 32 bits with or without alpha by its masks: left to the decode.
    if (bits <= 8 || bits == 32) return false;
    h.channels = 3;
    return h.width > 0 && h.height > 0;
}

ImageHeader HeaderByDecoding(const std::string& path) {
    ImageHeader h;
    const cv::Mat image = cv::imread(path, cv::IMREAD_UNCHANGED);
    if (image.empty()) return h;
    h.readable = true;
    h.width = image.cols;
    h.height = image.rows;
    h.channels = image.channels();
    return h;
}

const char* ReasonText(uint32_t bit) {
    switch (bit) {
        case image::kQualityBlurry: return "blurry";
        case image::kQualityDark: return "dark";
        case image::kQualityBright: return "bright";
        case image::kQualityLowContrast: return "low contrast";
        case image::kQualityDuplicate: return "duplicate";
        default: return "";
    }
}

std::string VerdictText(uint32_t reasons) {
    std::string out;
    for (uint32_t bit : {image::kQualityBlurry, image::kQualityLowContrast, image::kQualityDark, image::kQualityBright,
                         image::kQualityDuplicate}) {
        if (!(reasons & bit)) continue;
        if (!out.empty()) out += ", ";
        out += ReasonText(bit);
    }
    return out;
}

template <typename Builder, typename Value>
void AppendOrNull(Builder& builder, bool present, const Value& value) {
    if (present) (void)builder.Append(value);
    else (void)builder.AppendNull();
}

}  // namespace

std::vector<std::string> SuggestClassNames(const std::vector<std::string>& files, const std::vector<int>& labels,
                                           const std::vector<std::string>& class_names) {
    const size_t classes = class_names.size();
    std::vector<std::string> stem(classes);
    std::vector<bool> shared(classes, true), seen(classes, false);
    for (size_t i = 0; i < files.size() && i < labels.size(); ++i) {
        const int label = labels[i];
        if (label < 0 || static_cast<size_t>(label) >= classes || !shared[label]) continue;
        const std::string name = fs::path(files[i]).filename().string();
        const size_t end = name.find_first_of("._-0123456789");
        const std::string prefix = name.substr(0, end == std::string::npos ? name.size() : end);
        if (!seen[label]) {
            stem[label] = prefix;
            seen[label] = true;
        } else if (prefix != stem[label]) {
            shared[label] = false;
        }
    }
    std::vector<std::string> out(classes);
    for (size_t c = 0; c < classes; ++c) {
        if (!seen[c] || !shared[c] || stem[c].empty() || stem[c] == class_names[c]) continue;
        const bool unique = std::count(stem.begin(), stem.end(), stem[c]) == 1;
        if (unique) out[c] = stem[c];
    }
    return out;
}

ImageHeader ReadImageHeader(const std::string& path) {
    ImageHeader h;
    std::ifstream in(path, std::ios::binary);
    if (!in) return h;
    std::array<unsigned char, 32> b{};
    in.read(reinterpret_cast<char*>(b.data()), b.size());
    const size_t n = static_cast<size_t>(in.gcount());
    in.clear();
    bool known = false;
    bool ok = false;
    if (n >= 3 && b[0] == 0xFF && b[1] == 0xD8 && b[2] == 0xFF) {
        known = true;
        ok = JpegHeader(in, h);
    } else if (n >= 8 && b[0] == 0x89 && b[1] == 'P' && b[2] == 'N' && b[3] == 'G') {
        known = true;
        ok = PngHeader(b.data(), n, h);
    } else if (n >= 2 && b[0] == 'B' && b[1] == 'M') {
        known = true;
        ok = BmpHeader(b.data(), n, h);
    }
    if (known && ok) {
        h.readable = true;
        return h;
    }
    in.close();
    return HeaderByDecoding(path);  // another format, or a header this reader cannot follow
}

std::shared_ptr<arrow::Table> BuildImageIndex(const ImageSetListing& listing,
                                              const std::vector<std::string>& label_names,
                                              const std::function<void(size_t, size_t)>& progress,
                                              const std::atomic<bool>* cancel, int threads) {
    const size_t total = listing.files.size();
    std::vector<ImageHeader> headers(total);
    std::vector<int64_t> bytes(total, 0);

    const int workers = std::max(1, threads > 0 ? threads : static_cast<int>(std::thread::hardware_concurrency()));
    std::atomic<size_t> next{0}, done{0};
    std::atomic<bool> stopped{false};
    constexpr size_t kChunk = 64;
    // Reads the next chunk of files; false when none are left or on cancel.
    const auto read_chunk = [&]() -> bool {
        if (cancel && cancel->load()) {
            stopped = true;
            return false;
        }
        const size_t begin = next.fetch_add(kChunk);
        if (begin >= total) return false;
        const size_t end = std::min(total, begin + kChunk);
        for (size_t i = begin; i < end; ++i) {
            headers[i] = ReadImageHeader(listing.files[i]);
            std::error_code ec;
            const auto size = fs::file_size(listing.files[i], ec);
            bytes[i] = ec ? 0 : static_cast<int64_t>(size);
        }
        done += end - begin;
        return true;
    };
    std::vector<std::thread> pool;
    for (int w = 1; w < workers; ++w) pool.emplace_back([&] { while (read_chunk()) {} });
    // The calling thread reads too and reports progress between its chunks.
    while (read_chunk()) {
        if (progress) progress(done.load(), total);
    }
    for (auto& t : pool) t.join();
    if (stopped.load()) return nullptr;
    if (progress) progress(total, total);

    arrow::StringBuilder path_b, file_b, name_b;
    arrow::Int32Builder label_b, width_b, height_b, channels_b;
    arrow::Int64Builder bytes_b;
    arrow::BooleanBuilder readable_b;
    for (size_t i = 0; i < total; ++i) {
        const int label = i < listing.labels.size() ? listing.labels[i] : -1;
        std::string name;
        if (label >= 0 && static_cast<size_t>(label) < label_names.size() && !label_names[label].empty())
            name = label_names[label];
        else if (label >= 0 && static_cast<size_t>(label) < listing.class_names.size())
            name = listing.class_names[label];
        (void)path_b.Append(listing.files[i]);
        (void)file_b.Append(fs::path(listing.files[i]).filename().string());
        (void)label_b.Append(label);
        (void)name_b.Append(name);
        const ImageHeader& h = headers[i];
        AppendOrNull(width_b, h.readable, h.width);
        AppendOrNull(height_b, h.readable, h.height);
        AppendOrNull(channels_b, h.readable, h.channels);
        (void)bytes_b.Append(bytes[i]);
        (void)readable_b.Append(h.readable);
    }
    std::vector<std::shared_ptr<arrow::Array>> columns(9);
    (void)path_b.Finish(&columns[0]);
    (void)file_b.Finish(&columns[1]);
    (void)label_b.Finish(&columns[2]);
    (void)name_b.Finish(&columns[3]);
    (void)width_b.Finish(&columns[4]);
    (void)height_b.Finish(&columns[5]);
    (void)channels_b.Finish(&columns[6]);
    (void)bytes_b.Finish(&columns[7]);
    (void)readable_b.Finish(&columns[8]);
    auto schema = arrow::schema({arrow::field("path", arrow::utf8()), arrow::field("file", arrow::utf8()),
                                 arrow::field("label", arrow::int32()), arrow::field("label_name", arrow::utf8()),
                                 arrow::field("width", arrow::int32()), arrow::field("height", arrow::int32()),
                                 arrow::field("channels", arrow::int32()), arrow::field("bytes", arrow::int64()),
                                 arrow::field("readable", arrow::boolean())});
    return arrow::Table::Make(schema, columns);
}

std::shared_ptr<arrow::Table> AddImageQualityColumns(const std::shared_ptr<arrow::Table>& index,
                                                     const ImageQualityAnalysis& analysis,
                                                     const ImageQualityVerdict& verdict) {
    if (!index) return nullptr;
    std::map<std::string, size_t> measured;
    for (size_t i = 0; i < analysis.files.size(); ++i) measured.emplace(analysis.files[i], i);

    const auto combined = index->CombineChunks().ValueOrDie();
    const auto path_column = combined->GetColumnByName("path");
    if (!path_column) return nullptr;
    const auto paths = path_column->num_chunks() > 0
                           ? std::static_pointer_cast<arrow::StringArray>(path_column->chunk(0))
                           : nullptr;
    arrow::FloatBuilder blur_b, bright_b, contrast_b;
    arrow::StringBuilder verdict_b, dup_b;
    for (int64_t r = 0; paths && r < paths->length(); ++r) {
        const auto it = measured.find(paths->GetString(r));
        const bool has = it != measured.end() && it->second < analysis.metrics.size();
        const size_t i = has ? it->second : 0;
        AppendOrNull(blur_b, has, has ? analysis.metrics[i].blur : 0.0f);
        AppendOrNull(bright_b, has, has ? analysis.metrics[i].brightness : 0.0f);
        AppendOrNull(contrast_b, has, has ? analysis.metrics[i].contrast : 0.0f);
        const bool judged = has && i < verdict.reasons.size();
        AppendOrNull(verdict_b, judged, judged ? VerdictText(verdict.reasons[i]) : std::string());
        const bool dup = judged && i < verdict.duplicate_of.size() && verdict.duplicate_of[i] >= 0 &&
                         static_cast<size_t>(verdict.duplicate_of[i]) < analysis.files.size();
        AppendOrNull(dup_b, dup, dup ? fs::path(analysis.files[verdict.duplicate_of[i]]).filename().string() : std::string());
    }
    std::shared_ptr<arrow::Array> blur, bright, contrast, verdict_a, dup_a;
    (void)blur_b.Finish(&blur);
    (void)bright_b.Finish(&bright);
    (void)contrast_b.Finish(&contrast);
    (void)verdict_b.Finish(&verdict_a);
    (void)dup_b.Finish(&dup_a);

    auto table = index;
    for (const char* name : {"blur", "brightness", "contrast", "verdict", "duplicate_of"}) {
        const int at = table->schema()->GetFieldIndex(name);
        if (at >= 0) table = table->RemoveColumn(at).ValueOrDie();
    }
    const auto add = [&](const char* name, const std::shared_ptr<arrow::DataType>& type, const std::shared_ptr<arrow::Array>& a) {
        table = table->AddColumn(table->num_columns(), arrow::field(name, type), std::make_shared<arrow::ChunkedArray>(a))
                    .ValueOrDie();
    };
    add("blur", arrow::float32(), blur);
    add("brightness", arrow::float32(), bright);
    add("contrast", arrow::float32(), contrast);
    add("verdict", arrow::utf8(), verdict_a);
    add("duplicate_of", arrow::utf8(), dup_a);
    return table;
}

}  // namespace cyxwiz
