#include "mj_recorder.h"
#include "mj_gif_writer.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include <stb_image_resize2.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <system_error>

namespace cyxwiz::plugin::mujoco {
namespace {

constexpr size_t kMaxQueuedFrames = 60;  // about 100 MB at a full-screen Viewport

std::vector<unsigned char> TopRowFirst(const std::vector<unsigned char>& rgb_bottom_up, int width, int height) {
    const size_t row = static_cast<size_t>(width) * 3;
    std::vector<unsigned char> out(row * height);
    for (int y = 0; y < height; ++y)
        std::memcpy(out.data() + row * y, rgb_bottom_up.data() + row * (height - 1 - y), row);
    return out;
}

void WritePngBytes(void* context, void* data, int size) {
    auto* file = static_cast<std::FILE*>(context);
    std::fwrite(data, 1, static_cast<size_t>(size), file);
}

// stbi_write_png_to_func through our own file handle: the path may hold
// characters outside the ANSI code page on Windows
bool WritePng(const std::filesystem::path& file, const unsigned char* rgb, int width, int height,
              std::string& error) {
#ifdef _WIN32
    std::FILE* f = _wfopen(file.wstring().c_str(), L"wb");
#else
    std::FILE* f = std::fopen(file.string().c_str(), "wb");
#endif
    if (!f) {
        error = "Could not create " + file.string() + ".";
        return false;
    }
    const int written = stbi_write_png_to_func(WritePngBytes, f, width, height, 3, rgb, width * 3);
    const bool failed = std::ferror(f) != 0;
    const bool closed = std::fclose(f) == 0;
    if (!written || failed || !closed) {
        error = "Writing " + file.string() + " failed (is the disk full?).";
        return false;
    }
    return true;
}

}  // namespace

MjRecorder::~MjRecorder() {
    Stop();
    if (worker_.joinable()) worker_.join();
}

bool MjRecorder::SavePng(const std::filesystem::path& file, const std::vector<unsigned char>& rgb_bottom_up,
                         int width, int height, std::string& error) {
    if (width < 1 || height < 1 || rgb_bottom_up.size() < static_cast<size_t>(width) * height * 3) {
        error = "The Viewport has not drawn a frame yet.";
        return false;
    }
    const std::vector<unsigned char> rgb = TopRowFirst(rgb_bottom_up, width, height);
    return WritePng(file, rgb.data(), width, height, error);
}

bool MjRecorder::Start(const std::filesystem::path& target, Format format, std::string& error) {
    if (recording_.load() || finishing_.load()) {
        error = "A recording is already running.";
        return false;
    }
    if (worker_.joinable()) worker_.join();  // the last one is finished
    if (format == Format::PngFrames) {
        std::error_code ec;
        std::filesystem::create_directories(target, ec);
        if (ec) {
            error = "Could not create the folder " + target.string() + ": " + ec.message();
            return false;
        }
    }
    {
        std::lock_guard lock(mutex_);
        format_ = format;
        target_ = target;
        queue_.clear();
        stop_asked_ = false;
        worker_finished_ = false;
        dropped_ = 0;
        first_time_ = 0.0;
        last_time_ = -1.0;
        recorded_seconds_ = 0.0;
        result_ready_ = false;
        result_ = Result{};
    }
    frames_ = 0;
    recording_ = true;
    worker_ = std::thread(&MjRecorder::Run, this);
    return true;
}

void MjRecorder::AddFrame(const std::vector<unsigned char>& rgb_bottom_up, int width, int height, double sim_time) {
    if (!recording_.load() || width < 1 || height < 1 ||
        rgb_bottom_up.size() < static_cast<size_t>(width) * height * 3)
        return;
    {
        std::lock_guard lock(mutex_);
        if (sim_time == last_time_) return;  // the scene has not moved
        if (last_time_ < 0.0) first_time_ = sim_time;
        else if (sim_time > last_time_) recorded_seconds_ += sim_time - last_time_;
        last_time_ = sim_time;
        if (queue_.size() >= kMaxQueuedFrames) {
            ++dropped_;
            return;
        }
        queue_.push_back({TopRowFirst(rgb_bottom_up, width, height), width, height, sim_time});
    }
    const int taken = ++frames_;
    wake_.notify_one();
    const int cap = format_ == Format::Gif ? kMaxGifFrames : kMaxPngFrames;
    if (taken >= cap) Stop();
}

void MjRecorder::Stop() {
    if (!recording_.exchange(false)) return;
    {
        std::lock_guard lock(mutex_);
        if (!worker_finished_) finishing_ = true;  // the worker may have finished on a write error
        stop_asked_ = true;
    }
    wake_.notify_one();
}

double MjRecorder::Seconds() const {
    std::lock_guard lock(mutex_);
    return recorded_seconds_;
}

bool MjRecorder::TakeResult(Result& result) {
    std::lock_guard lock(mutex_);
    if (!result_ready_) return false;
    result = result_;
    result_ready_ = false;
    return true;
}

void MjRecorder::Finish(bool ok, std::string error) {
    {
        std::lock_guard lock(mutex_);
        result_.ok = ok;
        result_.path = target_;
        result_.frames = frames_.load();
        result_.dropped = dropped_;
        result_.error = std::move(error);
        result_ready_ = true;
        queue_.clear();
        worker_finished_ = true;
        finishing_ = false;
    }
    recording_ = false;
}

void MjRecorder::Run() {
    MjGifWriter gif;
    int gif_width = 0, gif_height = 0;
    std::vector<unsigned char> scaled;
    // A GIF frame's delay is known only when the next frame comes: hold one
    Frame held;
    bool holding = false;
    double last_delta = 0.05;
    double carry_cs = 0.0;  // rounding left over, so the total time stays right
    int png_index = 0;
    std::string error;

    const auto write_gif = [&](const Frame& frame, double seconds) {
        if (!gif.IsOpen()) {
            gif_width = std::min(frame.width, kGifMaxWidth);
            gif_height = std::max(1, static_cast<int>(std::lround(static_cast<double>(frame.height) * gif_width /
                                                                  frame.width)));
            if (!gif.Open(target_, gif_width, gif_height, error)) return false;
            scaled.resize(static_cast<size_t>(gif_width) * gif_height * 3);
        }
        const unsigned char* pixels = frame.rgb.data();
        if (frame.width != gif_width || frame.height != gif_height) {
            stbir_resize_uint8_linear(frame.rgb.data(), frame.width, frame.height, frame.width * 3, scaled.data(),
                                      gif_width, gif_height, gif_width * 3, STBIR_RGB);
            pixels = scaled.data();
        }
        const double exact = seconds * 100.0 + carry_cs;
        const int delay = std::max(2, static_cast<int>(std::lround(exact)));
        carry_cs = exact - delay;
        return gif.AddFrame(pixels, delay, error);
    };

    bool ok = true;
    while (ok) {
        Frame frame;
        {
            std::unique_lock lock(mutex_);
            wake_.wait(lock, [this] { return stop_asked_ || !queue_.empty(); });
            if (queue_.empty()) break;  // stopped and everything written
            frame = std::move(queue_.front());
            queue_.pop_front();
        }
        if (format_ == Format::PngFrames) {
            char name[32];
            std::snprintf(name, sizeof(name), "frame_%06d.png", ++png_index);
            ok = WritePng(target_ / name, frame.rgb.data(), frame.width, frame.height, error);
            continue;
        }
        if (holding) {
            // An episode's reset sends the clock back: keep the last step's length
            const double delta = frame.sim_time > held.sim_time ? frame.sim_time - held.sim_time : last_delta;
            last_delta = delta;
            ok = write_gif(held, delta);
        }
        held = std::move(frame);
        holding = true;
    }
    if (ok && holding) ok = write_gif(held, last_delta);
    if (gif.IsOpen()) {
        std::string close_error;
        if (!gif.Close(close_error) && ok) {
            ok = false;
            error = close_error;
        }
    }
    if (ok && frames_.load() == 0) {
        ok = false;
        error = "No frame was recorded: the simulation did not move.";
        std::error_code ec;
        if (format_ == Format::PngFrames) std::filesystem::remove(target_, ec);  // the empty folder it made
    }
    if (!ok && error.empty()) error = "The recording could not be written.";
    // Stop taking frames even when writing failed before Stop was asked
    recording_ = false;
    Finish(ok, error);
}

}  // namespace cyxwiz::plugin::mujoco
