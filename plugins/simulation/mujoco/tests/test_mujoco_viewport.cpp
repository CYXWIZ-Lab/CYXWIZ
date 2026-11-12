// The MuJoCo Viewport's pieces without a window: the GIF writer (decoded back
// here, frame by frame), the recorder (GIF timing from the simulation, PNG
// frames, screenshots read back) and the saved choices (Preferences >
// Simulation) through a PluginContext that keeps settings in memory.

#include "mj_gif_writer.h"
#include "mj_recorder.h"
#include "mj_viewport_settings.h"

#include "plugin/plugin_context.h"

#include <mujoco/mujoco.h>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <string>
#include <thread>
#include <vector>

using namespace cyxwiz::plugin;
using namespace cyxwiz::plugin::mujoco;

namespace {

int failures = 0;

void Check(bool ok, const std::string& what) {
    if (ok) return;
    ++failures;
    std::printf("FAIL: %s\n", what.c_str());
}

std::vector<uint8_t> ReadFile(const std::filesystem::path& file) {
    std::ifstream in(file, std::ios::binary);
    return std::vector<uint8_t>(std::istreambuf_iterator<char>(in), {});
}

// A small GIF reader: the canvas size, the loop extension, and each frame's
// delay and palette indices
struct Gif {
    int width = 0, height = 0;
    bool loops = false;
    std::vector<std::array<uint8_t, 3>> palette;
    std::vector<int> delays;
    std::vector<std::vector<uint8_t>> frames;
    std::string error;
};

std::vector<uint8_t> DecodeLzw(const std::vector<uint8_t>& data, int min_code_size, size_t pixels,
                               std::string& error) {
    const int clear = 1 << min_code_size, end = clear + 1;
    std::vector<std::vector<uint8_t>> dict;
    const auto reset = [&] {
        dict.clear();
        for (int i = 0; i < clear; ++i) dict.push_back({static_cast<uint8_t>(i)});
        dict.push_back({});
        dict.push_back({});
    };
    reset();
    int code_size = min_code_size + 1;
    size_t bit = 0;
    std::vector<uint8_t> out;
    int previous = -1;
    while (true) {
        if (bit + code_size > data.size() * 8) {
            error = "the image data ended without the end code";
            return out;
        }
        int code = 0;
        for (int i = 0; i < code_size; ++i, ++bit)
            if (data[bit / 8] & (1 << (bit % 8))) code |= 1 << i;
        if (code == clear) {
            reset();
            code_size = min_code_size + 1;
            previous = -1;
            continue;
        }
        if (code == end) break;
        std::vector<uint8_t> entry;
        if (code < static_cast<int>(dict.size())) {
            entry = dict[code];
            if (previous >= 0) {
                std::vector<uint8_t> added = dict[previous];
                added.push_back(entry[0]);
                dict.push_back(added);
            }
        } else if (code == static_cast<int>(dict.size()) && previous >= 0) {
            entry = dict[previous];
            entry.push_back(entry[0]);
            dict.push_back(entry);
        } else {
            error = "a code past the dictionary";
            return out;
        }
        out.insert(out.end(), entry.begin(), entry.end());
        previous = code;
        if (static_cast<int>(dict.size()) == (1 << code_size) && code_size < 12) ++code_size;
    }
    if (out.size() != pixels) error = "the frame has " + std::to_string(out.size()) + " pixels";
    return out;
}

Gif ReadGif(const std::filesystem::path& file) {
    Gif gif;
    const std::vector<uint8_t> b = ReadFile(file);
    if (b.size() < 13 || std::string(b.begin(), b.begin() + 6) != "GIF89a") {
        gif.error = "not a GIF89a";
        return gif;
    }
    gif.width = b[6] | (b[7] << 8);
    gif.height = b[8] | (b[9] << 8);
    size_t p = 13;
    if (b[10] & 0x80) {
        const size_t entries = size_t{1} << ((b[10] & 7) + 1);
        for (size_t i = 0; i < entries; ++i) gif.palette.push_back({b[p + i * 3], b[p + i * 3 + 1], b[p + i * 3 + 2]});
        p += 3 * entries;
    }
    int delay = 0;
    while (p < b.size()) {
        const uint8_t kind = b[p++];
        if (kind == 0x3B) return gif;  // trailer
        if (kind == 0x21) {
            const uint8_t label = b[p++];
            if (label == 0xF9) delay = b[p + 2] | (b[p + 3] << 8);
            if (label == 0xFF && std::string(b.begin() + p + 1, b.begin() + p + 12) == "NETSCAPE2.0") gif.loops = true;
            while (b[p] != 0) p += b[p] + 1u;
            ++p;
            continue;
        }
        if (kind != 0x2C) {
            gif.error = "an unknown block";
            return gif;
        }
        p += 9;  // the frame's position, size and flags
        const int min_code_size = b[p++];
        std::vector<uint8_t> data;
        while (b[p] != 0) {
            data.insert(data.end(), b.begin() + p + 1, b.begin() + p + 1 + b[p]);
            p += b[p] + 1u;
        }
        ++p;
        gif.frames.push_back(
            DecodeLzw(data, min_code_size, static_cast<size_t>(gif.width) * gif.height, gif.error));
        gif.delays.push_back(delay);
        if (!gif.error.empty()) return gif;
    }
    gif.error = "no trailer";
    return gif;
}

// A frame with a gradient, noise-like detail and a block that moves with `t`,
// so the dictionary fills and clears many times
std::vector<uint8_t> Picture(int width, int height, int t) {
    std::vector<uint8_t> rgb(static_cast<size_t>(width) * height * 3);
    for (int y = 0; y < height; ++y)
        for (int x = 0; x < width; ++x) {
            uint8_t* px = &rgb[(static_cast<size_t>(y) * width + x) * 3];
            px[0] = static_cast<uint8_t>(x * 255 / width);
            px[1] = static_cast<uint8_t>(((x * 7 + y * 13 + t * 5) * 2654435761u) >> 24);
            px[2] = (x / 16 + t) % 4 == 0 && y > height / 3 ? 230 : static_cast<uint8_t>(y * 255 / height);
        }
    return rgb;
}

void CheckGifWriter(const std::filesystem::path& dir) {
    const int w = 97, h = 61;  // odd sizes: rows do not line up with anything
    const std::filesystem::path file = dir / "frames.gif";
    std::vector<std::vector<uint8_t>> pictures;
    MjGifWriter gif;
    {
        std::string error;
        Check(gif.Open(file, w, h, error), "the GIF opens: " + error);
        for (int t = 0; t < 4; ++t) {
            pictures.push_back(Picture(w, h, t));
            Check(gif.AddFrame(pictures.back().data(), t == 3 ? 1 : 5 + t, error), "frame " + std::to_string(t) + ": " + error);
        }
        Check(gif.Frames() == 4, "four frames written");
        Check(gif.Close(error), "the GIF closes: " + error);
    }
    const Gif read = ReadGif(file);
    Check(read.error.empty(), "the GIF reads back: " + read.error);
    Check(read.width == w && read.height == h && read.loops, "its size, and it loops");
    Check(read.frames.size() == 4, "four frames read back");
    Check(read.delays == std::vector<int>({5, 6, 7, 2}), "each frame's delay (a delay under 2 becomes 2)");
    Check(read.palette.size() == 256 && !gif.Palette().empty() &&
              std::equal(gif.Palette().begin(), gif.Palette().end(), read.palette.begin()),
          "the file's palette is the one cut from the first frame");
    for (size_t f = 0; f < read.frames.size() && f < pictures.size(); ++f) {
        bool same = true;
        for (int y = 0; y < h && same; ++y)
            for (int x = 0; x < w && same; ++x) {
                const uint8_t* px = &pictures[f][(static_cast<size_t>(y) * w + x) * 3];
                same = read.frames[f][static_cast<size_t>(y) * w + x] == gif.IndexOf(px[0], px[1], px[2], x, y);
            }
        Check(same, "frame " + std::to_string(f) + " decodes to the palette indices written");
    }
}

// The walk's bug: a grey sky came back purple and green. A grey gradient
// with a few coloured shapes (a rendered scene's make-up) keeps its greys
void CheckGifColours(const std::filesystem::path& dir) {
    const int w = 320, h = 180;
    std::vector<uint8_t> rgb(static_cast<size_t>(w) * h * 3);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            uint8_t* px = &rgb[(static_cast<size_t>(y) * w + x) * 3];
            const uint8_t grey = static_cast<uint8_t>(90 + y * 80 / h);  // sky: 90 .. 170
            px[0] = px[1] = px[2] = grey;
            if (y > h / 2 && ((x / 20 + y / 20) & 1)) px[0] = px[1] = px[2] = 20;                 // floor squares
            if (y > h / 2 && !((x / 20 + y / 20) & 1)) { px[0] = 200; px[1] = 225; px[2] = 205; }  // pale green squares
            if (x > 140 && x < 200 && y > 60 && y < 70) { px[0] = 190; px[1] = 140; px[2] = 90; }  // the body
        }
    MjGifWriter gif;
    std::string error;
    Check(gif.Open(dir / "colours.gif", w, h, error) && gif.AddFrame(rgb.data(), 5, error) && gif.Close(error),
          "a scene-like frame is written: " + error);
    const Gif read = ReadGif(dir / "colours.gif");
    Check(read.error.empty() && read.frames.size() == 1, "and reads back: " + read.error);
    if (read.frames.size() != 1) return;
    double total = 0.0;
    int worst_tint = 0;
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            const uint8_t* px = &rgb[(static_cast<size_t>(y) * w + x) * 3];
            const auto& c = read.palette[read.frames[0][static_cast<size_t>(y) * w + x]];
            for (int ch = 0; ch < 3; ++ch) total += std::abs(int(c[ch]) - int(px[ch]));
            if (px[0] == px[1] && px[1] == px[2])
                worst_tint = std::max({worst_tint, std::abs(int(c[0]) - int(c[1])), std::abs(int(c[1]) - int(c[2]))});
        }
    const double mean = total / (3.0 * w * h);
    Check(mean < 4.0, "colours come back within 4 levels on average (got " + std::to_string(mean) + ")");
    Check(worst_tint <= 8, "greys stay grey: no channel more than 8 levels off the others (got " +
                               std::to_string(worst_tint) + ")");
}

bool WaitForResult(MjRecorder& recorder, MjRecorder::Result& result) {
    for (int i = 0; i < 500; ++i) {
        if (recorder.TakeResult(result)) return true;
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    return false;
}

// Bottom row first, as the renderer reads it
std::vector<unsigned char> RendererFrame(int width, int height, int t) {
    const std::vector<uint8_t> top = Picture(width, height, t);
    std::vector<unsigned char> bottom(top.size());
    const size_t row = static_cast<size_t>(width) * 3;
    for (int y = 0; y < height; ++y) std::copy_n(top.data() + row * y, row, bottom.data() + row * (height - 1 - y));
    return bottom;
}

void CheckRecorder(const std::filesystem::path& dir) {
    const int w = 800, h = 300;  // wider than a GIF's 640: scaled down
    // A GIF timed by the simulation: 0.05 s steps play as 5 hundredths
    {
        MjRecorder recorder;
        std::string error;
        Check(recorder.Start(dir / "run.gif", MjRecorder::Format::Gif, error), "a GIF recording starts: " + error);
        Check(!recorder.Start(dir / "again.gif", MjRecorder::Format::Gif, error), "a second one is refused");
        const auto frame = RendererFrame(w, h, 0);
        double t = 0.0;
        for (int i = 0; i < 10; ++i, t += 0.05) {
            recorder.AddFrame(RendererFrame(w, h, i), w, h, t);
            recorder.AddFrame(frame, w, h, t);  // the scene has not moved: not taken
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
        }
        recorder.AddFrame(frame, w, h, 0.0);  // an episode's reset sends the clock back
        Check(recorder.Frames() == 11, "a frame each time the simulation time moves (11, got " +
                                           std::to_string(recorder.Frames()) + ")");
        recorder.Stop();
        MjRecorder::Result result;
        Check(WaitForResult(recorder, result) && result.ok, "the GIF is written: " + result.error);
        Check(result.frames == 11 && result.path == dir / "run.gif", "the result names the file and the frames");
        const Gif read = ReadGif(dir / "run.gif");
        Check(read.error.empty() && read.width == 640 && read.height == 240, "scaled to 640 wide, the same shape");
        Check(read.frames.size() == 11, "11 frames in the GIF");
        int total = 0;
        for (const int d : read.delays) total += d;
        Check(!read.delays.empty() && read.delays.front() == 5 && total == 55,
              "each 0.05 s step plays 5 hundredths; after the reset the last step's length (55 in all, got " +
                  std::to_string(total) + ")");
    }
    // PNG frames into a folder
    {
        MjRecorder recorder;
        std::string error;
        Check(recorder.Start(dir / "frames", MjRecorder::Format::PngFrames, error), "PNG frames start: " + error);
        for (int i = 0; i < 3; ++i) recorder.AddFrame(RendererFrame(64, 32, i), 64, 32, i * 0.02);
        recorder.Stop();
        MjRecorder::Result result;
        Check(WaitForResult(recorder, result) && result.ok && result.frames == 3, "three PNG frames: " + result.error);
        int x = 0, y = 0, n = 0;
        unsigned char* png = stbi_load((dir / "frames" / "frame_000003.png").string().c_str(), &x, &y, &n, 3);
        Check(png && x == 64 && y == 32, "frame_000003.png is 64 x 32");
        if (png) {
            const auto top = Picture(64, 32, 2);
            Check(std::equal(top.begin(), top.end(), png), "the PNG holds the frame with its top row first");
            stbi_image_free(png);
        }
    }
    // A recording where nothing moved says so and leaves no empty folder
    {
        MjRecorder recorder;
        std::string error;
        Check(recorder.Start(dir / "still", MjRecorder::Format::PngFrames, error), "a still recording starts");
        recorder.Stop();
        MjRecorder::Result result;
        Check(WaitForResult(recorder, result) && !result.ok && result.error.find("did not move") != std::string::npos,
              "no frame: the result says the simulation did not move");
        Check(!std::filesystem::exists(dir / "still"), "and its empty folder is gone");
    }
    // A screenshot reads back the right way up
    {
        std::string error;
        Check(MjRecorder::SavePng(dir / "shot.png", RendererFrame(40, 20, 1), 40, 20, error), "screenshot: " + error);
        int x = 0, y = 0, n = 0;
        unsigned char* png = stbi_load((dir / "shot.png").string().c_str(), &x, &y, &n, 3);
        const auto top = Picture(40, 20, 1);
        Check(png && x == 40 && y == 20 && std::equal(top.begin(), top.end(), png), "the screenshot's pixels");
        if (png) stbi_image_free(png);
        Check(!MjRecorder::SavePng(dir / "none.png", {}, 40, 20, error) && !error.empty(), "no frame yet: refused");
    }
}

// Settings in memory, as a host without a settings store keeps them
class SettingsContext final : public PluginContext {
public:
    const std::string& GetPluginId() const override { return id_; }
    const std::filesystem::path& GetPluginDir() const override { return dir_; }
    AssistantContextSnapshot GetAssistantContextSnapshot() const override { return {}; }
    void LogInfo(const std::string&) override {}
    void LogWarn(const std::string&) override {}
    void LogError(const std::string&) override {}
    void LogDebug(const std::string&) override {}
    bool RegisterNodeProvider(INodeProvider*) override { return false; }
    bool RegisterPanelProvider(IPanelProvider*) override { return false; }
    bool RegisterDataProvider(IDataProvider*) override { return false; }
    bool RegisterTrainingHook(ITrainingHook*) override { return false; }
    bool RegisterAnalyticsProvider(IAnalyticsProvider*) override { return false; }
    PluginUi* GetUi() override { return nullptr; }
    bool ShowPanel(const std::string&) override { return false; }
    void RequestFullFrameRate() override {}
    std::filesystem::path GetProjectDir() const override { return {}; }
    std::string GetSetting(const std::string& key, const std::string& fallback) const override {
        const auto it = values.find(key);
        return it == values.end() ? fallback : it->second;
    }
    void SetSetting(const std::string& key, const std::string& value) override { values[key] = value; }
    std::map<std::string, std::string> values;

private:
    std::string id_ = "test";
    std::filesystem::path dir_;
};

void CheckSettings() {
    SettingsContext ctx;
    const ViewportDefaults fresh = LoadViewportDefaults(ctx);
    Check(fresh.playback_camera == PlaybackCamera::ModelTrack && fresh.live_numbers &&
              fresh.record_format == MjRecorder::Format::Gif && !fresh.record_until_stop && fresh.scene == SceneOptions{},
          "nothing saved: the track camera, live numbers on, GIF until the episode ends, MuJoCo's scene");

    ViewportDefaults d;
    d.playback_camera = PlaybackCamera::FollowRoot;
    d.live_numbers = false;
    d.record_format = MjRecorder::Format::PngFrames;
    d.record_until_stop = true;
    d.scene.contact_forces = true;
    d.scene.shadows = false;
    d.scene.endless_floor = false;
    d.scene.frames = mjFRAME_BODY;
    d.scene.labels = mjLABEL_JOINT;
    d.scene.geom_groups[3] = true;
    d.scene.site_groups[0] = false;
    SaveViewportDefaults(ctx, d);
    const ViewportDefaults back = LoadViewportDefaults(ctx);
    Check(back.playback_camera == PlaybackCamera::FollowRoot && !back.live_numbers &&
              back.record_format == MjRecorder::Format::PngFrames && back.record_until_stop,
          "the saved choices read back");
    Check(back.scene == d.scene, "the saved scene options read back");

    Check(DecodeScene("not json") == SceneOptions{} && DecodeScene("") == SceneOptions{},
          "broken text gives MuJoCo's scene");
    SceneOptions odd = DecodeScene(R"({"frames": 99, "labels": -1, "geom_groups": "1x", "sky": "yes"})");
    Check(odd == SceneOptions{}, "values out of range or of the wrong kind keep their defaults");
    Check(PlaybackCameraFromKey("follow") == PlaybackCamera::FollowRoot &&
              PlaybackCameraFromKey("free") == PlaybackCamera::Free &&
              PlaybackCameraFromKey("anything") == PlaybackCamera::ModelTrack,
          "camera keys read back, unknown ones give the track camera");
}

}  // namespace

int main() {
    const std::filesystem::path dir = std::filesystem::temp_directory_path() / "cyxwiz_test_mujoco_viewport";
    std::error_code ec;
    std::filesystem::remove_all(dir, ec);
    std::filesystem::create_directories(dir);
    CheckGifWriter(dir);
    CheckGifColours(dir);
    CheckRecorder(dir);
    CheckSettings();
    std::filesystem::remove_all(dir, ec);
    if (failures == 0) std::printf("test_mujoco_viewport: all checks passed\n");
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
