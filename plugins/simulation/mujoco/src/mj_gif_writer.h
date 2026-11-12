#pragma once

// MjGifWriter: an animated GIF, written frame by frame (the MuJoCo Viewport's
// Record > GIF). The palette (up to 256 colours) is cut from the first frame's
// colours by median cut and serves every frame: a recorded scene keeps its
// colours (sky, floor, robot), so one table fits it and frames do not flicker
// between palettes. Each pixel takes the nearest palette colour through a
// 15-bit lookup table, after a light 4x4 ordered dither that keeps gradients
// smooth. A frame is written as soon as it is given; the animation loops.

#include <array>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

namespace cyxwiz::plugin::mujoco {

class MjGifWriter {
public:
    MjGifWriter() = default;
    ~MjGifWriter();
    MjGifWriter(const MjGifWriter&) = delete;
    MjGifWriter& operator=(const MjGifWriter&) = delete;

    // Creates the file; every frame has this size. The header and palette are
    // written with the first frame. False with the reason.
    bool Open(const std::filesystem::path& file, int width, int height, std::string& error);
    // One frame: RGB, top row first, width x height x 3 bytes; shown for
    // delay_cs hundredths of a second (at least 2: browsers slow faster
    // frames down to 10).
    bool AddFrame(const uint8_t* rgb, int delay_cs, std::string& error);
    // Ends the file. False with the reason (the file is then incomplete).
    bool Close(std::string& error);
    bool IsOpen() const { return file_ != nullptr; }
    int Frames() const { return frames_; }

    // The palette index AddFrame writes for a pixel at (x, y), once the first
    // frame set the palette. For tests.
    uint8_t IndexOf(uint8_t r, uint8_t g, uint8_t b, int x, int y) const;
    const std::vector<std::array<uint8_t, 3>>& Palette() const { return palette_; }

private:
    void BuildPalette(const uint8_t* rgb);
    bool WriteHeader(std::string& error);
    void WriteLzw(const std::vector<uint8_t>& indices);

    std::FILE* file_ = nullptr;
    std::filesystem::path path_;
    int width_ = 0;
    int height_ = 0;
    int frames_ = 0;
    std::vector<std::array<uint8_t, 3>> palette_;
    std::vector<uint8_t> lookup_;  // 15-bit colour (5 bits a channel) -> palette index
    std::vector<uint8_t> indices_;
    std::vector<uint16_t> tree_;  // 4096 codes x 256 next values: the LZW dictionary
};

}  // namespace cyxwiz::plugin::mujoco
