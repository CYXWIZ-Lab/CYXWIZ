#include "mj_gif_writer.h"

#include <algorithm>
#include <climits>
#include <cstddef>
#include <cstring>

namespace cyxwiz::plugin::mujoco {
namespace {

constexpr int kMinCodeSize = 8;
constexpr int kClearCode = 1 << kMinCodeSize;  // 256
constexpr int kMaxCode = 4095;
constexpr int kColours15 = 1 << 15;

// The 4x4 Bayer matrix: offsets of about -4 .. +4 levels added before a
// colour is cut to 5 bits a channel (one 5-bit step is 8 levels)
constexpr int kBayer[4][4] = {{0, 8, 2, 10}, {12, 4, 14, 6}, {3, 11, 1, 9}, {15, 7, 13, 5}};

int Dither(int x, int y) { return (kBayer[y & 3][x & 3] * 2 - 15) / 4; }

int To15(int r, int g, int b) {
    r = std::clamp(r, 0, 255) >> 3;
    g = std::clamp(g, 0, 255) >> 3;
    b = std::clamp(b, 0, 255) >> 3;
    return (r << 10) | (g << 5) | b;
}

uint8_t Expand5(int v) { return static_cast<uint8_t>((v << 3) | (v >> 2)); }

int Channel(int colour15, int which) { return (colour15 >> (10 - 5 * which)) & 31; }

void Put16(std::FILE* f, int value) {
    std::fputc(value & 0xFF, f);
    std::fputc((value >> 8) & 0xFF, f);
}

// Variable-width codes, low bits first, in sub-blocks of up to 255 bytes
struct BitWriter {
    std::FILE* file;
    uint32_t bits = 0;
    int count = 0;
    uint8_t block[255];
    int used = 0;

    void Byte(uint8_t b) {
        block[used++] = b;
        if (used == 255) Flush();
    }
    void Flush() {
        if (used == 0) return;
        std::fputc(used, file);
        std::fwrite(block, 1, static_cast<size_t>(used), file);
        used = 0;
    }
    void Code(int code, int size) {
        bits |= static_cast<uint32_t>(code) << count;
        count += size;
        while (count >= 8) {
            Byte(static_cast<uint8_t>(bits & 0xFF));
            bits >>= 8;
            count -= 8;
        }
    }
    void Finish() {
        if (count > 0) Byte(static_cast<uint8_t>(bits & 0xFF));
        bits = 0;
        count = 0;
        Flush();
    }
};

// One box of the median cut: 15-bit colours with their pixel counts
struct ColourBox {
    std::vector<std::pair<int, uint32_t>> colours;
    uint64_t pixels = 0;

    int WidestChannel() const {
        int best = 0, best_range = -1;
        for (int ch = 0; ch < 3; ++ch) {
            int lo = 31, hi = 0;
            for (const auto& entry : colours) {
                lo = std::min(lo, Channel(entry.first, ch));
                hi = std::max(hi, Channel(entry.first, ch));
            }
            if (hi - lo > best_range) {
                best_range = hi - lo;
                best = ch;
            }
        }
        return best;
    }
};

}  // namespace

MjGifWriter::~MjGifWriter() {
    std::string ignored;
    Close(ignored);
}

// Median cut: the box with the most pixels is split at the weighted median of
// its widest channel, until there are 256 boxes or none can be split; each
// box gives its pixels' mean colour
void MjGifWriter::BuildPalette(const uint8_t* rgb) {
    std::vector<uint32_t> histogram(kColours15, 0);
    const size_t pixels = static_cast<size_t>(width_) * height_;
    for (size_t i = 0; i < pixels; ++i) ++histogram[To15(rgb[i * 3], rgb[i * 3 + 1], rgb[i * 3 + 2])];
    std::vector<ColourBox> boxes(1);
    for (int c = 0; c < kColours15; ++c) {
        if (!histogram[c]) continue;
        boxes[0].colours.push_back({c, histogram[c]});
        boxes[0].pixels += histogram[c];
    }
    while (boxes.size() < 256) {
        int pick = -1;
        uint64_t most = 0;
        for (size_t b = 0; b < boxes.size(); ++b) {
            if (boxes[b].colours.size() > 1 && boxes[b].pixels > most) {
                most = boxes[b].pixels;
                pick = static_cast<int>(b);
            }
        }
        if (pick < 0) break;
        ColourBox& box = boxes[static_cast<size_t>(pick)];
        const int ch = box.WidestChannel();
        std::sort(box.colours.begin(), box.colours.end(),
                  [ch](const auto& a, const auto& b) { return Channel(a.first, ch) < Channel(b.first, ch); });
        uint64_t below = 0;
        size_t cut = 1;
        for (size_t i = 0; i + 1 < box.colours.size(); ++i) {
            below += box.colours[i].second;
            cut = i + 1;
            if (below * 2 >= box.pixels) break;
        }
        ColourBox upper;
        upper.colours.assign(box.colours.begin() + static_cast<std::ptrdiff_t>(cut), box.colours.end());
        box.colours.resize(cut);
        box.pixels = 0;
        for (const auto& entry : box.colours) box.pixels += entry.second;
        for (const auto& entry : upper.colours) upper.pixels += entry.second;
        boxes.push_back(std::move(upper));
    }
    palette_.clear();
    for (const auto& box : boxes) {
        uint64_t sum[3] = {0, 0, 0};
        for (const auto& entry : box.colours)
            for (int ch = 0; ch < 3; ++ch) sum[ch] += static_cast<uint64_t>(Expand5(Channel(entry.first, ch))) * entry.second;
        std::array<uint8_t, 3> colour{0, 0, 0};
        if (box.pixels > 0)
            for (int ch = 0; ch < 3; ++ch) colour[ch] = static_cast<uint8_t>((sum[ch] + box.pixels / 2) / box.pixels);
        palette_.push_back(colour);
    }
    if (palette_.empty()) palette_.push_back({0, 0, 0});
    // Every 15-bit colour to its nearest palette colour (green counts most to the eye)
    lookup_.assign(kColours15, 0);
    for (int c = 0; c < kColours15; ++c) {
        const int r = Expand5(Channel(c, 0)), g = Expand5(Channel(c, 1)), b = Expand5(Channel(c, 2));
        int best = 0, best_d = INT_MAX;
        for (size_t p = 0; p < palette_.size(); ++p) {
            const int dr = r - palette_[p][0], dg = g - palette_[p][1], db = b - palette_[p][2];
            const int d = 3 * dr * dr + 4 * dg * dg + 2 * db * db;
            if (d < best_d) {
                best_d = d;
                best = static_cast<int>(p);
            }
        }
        lookup_[static_cast<size_t>(c)] = static_cast<uint8_t>(best);
    }
}

uint8_t MjGifWriter::IndexOf(uint8_t r, uint8_t g, uint8_t b, int x, int y) const {
    if (lookup_.empty()) return 0;
    const int d = Dither(x, y);
    return lookup_[static_cast<size_t>(To15(r + d, g + d, b + d))];
}

bool MjGifWriter::Open(const std::filesystem::path& file, int width, int height, std::string& error) {
    if (file_) {
        error = "A GIF is already being written.";
        return false;
    }
    if (width < 1 || height < 1 || width > 65535 || height > 65535) {
        error = "A GIF frame must be 1 to 65535 pixels wide and high.";
        return false;
    }
#ifdef _WIN32
    file_ = _wfopen(file.wstring().c_str(), L"wb");
#else
    file_ = std::fopen(file.string().c_str(), "wb");
#endif
    if (!file_) {
        error = "Could not create " + file.string() + ".";
        return false;
    }
    path_ = file;
    width_ = width;
    height_ = height;
    frames_ = 0;
    palette_.clear();
    lookup_.clear();
    indices_.assign(static_cast<size_t>(width) * height, 0);
    tree_.assign(static_cast<size_t>(kMaxCode + 1) * 256, 0);
    return true;
}

// The header, the palette (padded to 256 entries) and the loop extension
bool MjGifWriter::WriteHeader(std::string& error) {
    std::fwrite("GIF89a", 1, 6, file_);
    Put16(file_, width_);
    Put16(file_, height_);
    std::fputc(0xF7, file_);  // a global colour table of 256 entries (2^(7+1)), 8 bits a colour
    std::fputc(0, file_);     // background colour
    std::fputc(0, file_);     // square pixels
    for (size_t i = 0; i < 256; ++i) {
        const std::array<uint8_t, 3> colour = i < palette_.size() ? palette_[i] : std::array<uint8_t, 3>{0, 0, 0};
        std::fwrite(colour.data(), 1, 3, file_);
    }
    // Loop forever (the NETSCAPE2.0 application extension)
    const uint8_t loop[] = {0x21, 0xFF, 0x0B, 'N', 'E', 'T', 'S', 'C', 'A', 'P', 'E', '2', '.', '0',
                            0x03, 0x01, 0x00, 0x00, 0x00};
    std::fwrite(loop, 1, sizeof(loop), file_);
    if (std::ferror(file_)) {
        error = "Could not write " + path_.string() + ".";
        return false;
    }
    return true;
}

bool MjGifWriter::AddFrame(const uint8_t* rgb, int delay_cs, std::string& error) {
    if (!file_) {
        error = "The GIF is not open.";
        return false;
    }
    if (frames_ == 0) {  // the first frame gives the palette
        BuildPalette(rgb);
        if (!WriteHeader(error)) return false;
    }
    for (int y = 0; y < height_; ++y) {
        const uint8_t* row = rgb + static_cast<size_t>(y) * width_ * 3;
        uint8_t* out = indices_.data() + static_cast<size_t>(y) * width_;
        for (int x = 0; x < width_; ++x) out[x] = IndexOf(row[x * 3], row[x * 3 + 1], row[x * 3 + 2], x, y);
    }
    // Graphic control: the frame's delay; the frame replaces the one before
    const uint8_t control[] = {0x21, 0xF9, 0x04, 0x04};
    std::fwrite(control, 1, sizeof(control), file_);
    Put16(file_, std::max(2, std::min(delay_cs, 65535)));
    std::fputc(0, file_);  // no transparent colour
    std::fputc(0, file_);
    // Image descriptor: the whole canvas, the global colour table
    std::fputc(0x2C, file_);
    Put16(file_, 0);
    Put16(file_, 0);
    Put16(file_, width_);
    Put16(file_, height_);
    std::fputc(0, file_);
    WriteLzw(indices_);
    ++frames_;
    if (std::ferror(file_)) {
        error = "Writing the GIF failed (is the disk full?).";
        return false;
    }
    return true;
}

// LZW as GIF uses it: 8-bit symbols, codes of 9 to 12 bits, a clear code when
// the dictionary is full
void MjGifWriter::WriteLzw(const std::vector<uint8_t>& indices) {
    std::fputc(kMinCodeSize, file_);
    BitWriter out{file_};
    std::fill(tree_.begin(), tree_.end(), uint16_t{0});
    int code_size = kMinCodeSize + 1;
    int max_code = kClearCode + 1;
    out.Code(kClearCode, code_size);
    int current = -1;
    for (const uint8_t next : indices) {
        if (current < 0) {
            current = next;
            continue;
        }
        uint16_t& child = tree_[static_cast<size_t>(current) * 256 + next];
        if (child != 0) {
            current = child;
            continue;
        }
        out.Code(current, code_size);
        child = static_cast<uint16_t>(++max_code);
        if (max_code >= (1 << code_size)) ++code_size;
        if (max_code == kMaxCode) {
            out.Code(kClearCode, code_size);
            std::fill(tree_.begin(), tree_.end(), uint16_t{0});
            code_size = kMinCodeSize + 1;
            max_code = kClearCode + 1;
        }
        current = next;
    }
    if (current >= 0) {
        out.Code(current, code_size);
        // The decoder adds one more entry on reading this code: when that
        // fills the width, it reads what follows one bit wider (as giflib's
        // encoder does)
        if (max_code + 1 >= (1 << code_size) && code_size < 12) ++code_size;
    }
    out.Code(kClearCode, code_size);
    out.Code(kClearCode + 1, kMinCodeSize + 1);  // end of information
    out.Finish();
    std::fputc(0, file_);  // the image data ends
}

bool MjGifWriter::Close(std::string& error) {
    if (!file_) return true;
    if (frames_ == 0) {  // no frame: still a GIF (header and a black palette)
        palette_.assign(1, {0, 0, 0});
        std::string ignored;
        WriteHeader(ignored);
    }
    std::fputc(0x3B, file_);  // trailer
    const bool failed = std::ferror(file_) != 0;
    const bool closed = std::fclose(file_) == 0;
    file_ = nullptr;
    tree_.clear();
    tree_.shrink_to_fit();
    if (failed || !closed) {
        error = "Finishing the GIF failed (is the disk full?).";
        return false;
    }
    return true;
}

}  // namespace cyxwiz::plugin::mujoco
