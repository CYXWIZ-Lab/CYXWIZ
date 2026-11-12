#pragma once

// MjRecorder: what the MuJoCo Viewport saves of its scene (Record menu).
// A screenshot is one PNG. A recording takes the rendered frames whenever
// the simulation time moves and writes them on a worker thread, so the
// Viewport keeps its frame rate: an animated GIF (at most 640 pixels wide,
// timed by the simulation so it plays at real speed) or a folder of PNG
// frames at the Viewport's size, for a video editor or ffmpeg.

#include <atomic>
#include <condition_variable>
#include <deque>
#include <filesystem>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace cyxwiz::plugin::mujoco {

class MjRecorder {
public:
    enum class Format { Gif, PngFrames };

    MjRecorder() = default;
    ~MjRecorder();  // finishes a recording still being written
    MjRecorder(const MjRecorder&) = delete;
    MjRecorder& operator=(const MjRecorder&) = delete;

    // One PNG of a frame: RGB, bottom row first (as the renderer reads it).
    static bool SavePng(const std::filesystem::path& file, const std::vector<unsigned char>& rgb_bottom_up,
                        int width, int height, std::string& error);

    // Starts a recording into `target`: the .gif file, or the folder for PNG
    // frames (created). False with the reason (one is already running).
    bool Start(const std::filesystem::path& target, Format format, std::string& error);
    // A rendered frame (bottom row first) at simulation time `sim_time`.
    // Ignored when not recording or when the time has not moved.
    void AddFrame(const std::vector<unsigned char>& rgb_bottom_up, int width, int height, double sim_time);
    // Stops taking frames; the worker writes what it has, then the result
    // is ready (TakeResult).
    void Stop();

    bool IsRecording() const { return recording_.load(); }
    bool IsFinishing() const { return finishing_.load(); }
    int Frames() const { return frames_.load(); }       // taken so far
    double Seconds() const;                              // simulated time recorded
    Format GetFormat() const { return format_; }

    struct Result {
        bool ok = false;
        std::filesystem::path path;
        int frames = 0;
        int dropped = 0;  // frames skipped because the writer fell behind
        std::string error;
    };
    // The finished recording, once; false while none is ready.
    bool TakeResult(Result& result);

    // Recordings stop by themselves at this many frames
    static constexpr int kMaxGifFrames = 6000;
    static constexpr int kMaxPngFrames = 20000;
    static constexpr int kGifMaxWidth = 640;

private:
    struct Frame {
        std::vector<unsigned char> rgb;  // top row first
        int width = 0;
        int height = 0;
        double sim_time = 0.0;
    };
    void Run();
    void Finish(bool ok, std::string error);

    Format format_ = Format::Gif;
    std::filesystem::path target_;
    std::thread worker_;
    mutable std::mutex mutex_;
    std::condition_variable wake_;
    std::deque<Frame> queue_;
    bool stop_asked_ = false;
    std::atomic<bool> recording_{false};
    std::atomic<bool> finishing_{false};
    std::atomic<int> frames_{0};
    int dropped_ = 0;
    double first_time_ = 0.0;
    double last_time_ = -1.0;
    double recorded_seconds_ = 0.0;
    bool result_ready_ = false;
    Result result_;
};

}  // namespace cyxwiz::plugin::mujoco
