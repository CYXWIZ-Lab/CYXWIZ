#include "image_thumbnails.h"

#include "../core/texture_manager.h"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>

namespace gui {

namespace {
constexpr size_t kMaxQueued = 400;  // older requests (scrolled away) are dropped
}

ImageThumbnails::ImageThumbnails(int size, size_t capacity)
    : size_(std::max(8, size)), capacity_(std::max<size_t>(16, capacity)) {
    for (int i = 0; i < 2; ++i) workers_.emplace_back([this] { Work(); });
}

ImageThumbnails::~ImageThumbnails() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stop_ = true;
        requests_.clear();
    }
    wake_.notify_all();
    for (auto& w : workers_) w.join();
    auto& textures = cyxwiz::TextureManager::Instance();
    for (const auto& [path, entry] : entries_)
        if (entry.thumb.texture) textures.DeleteTexture(static_cast<uint32_t>(entry.thumb.texture));
}

ImageThumbnails::Thumb ImageThumbnails::Get(const std::string& path) {
    Entry& entry = entries_[path];
    entry.used = frame_;
    if (entry.thumb.texture || entry.thumb.failed) return entry.thumb;
    if (!entry.queued) {
        entry.queued = true;
        std::lock_guard<std::mutex> lock(mutex_);
        requests_.push_back(path);
        while (requests_.size() > kMaxQueued) {
            // The oldest request was for a page no longer shown: ask again when drawn.
            const auto old = entries_.find(requests_.front());
            if (old != entries_.end()) old->second.queued = false;
            requests_.pop_front();
        }
        wake_.notify_one();
    }
    return entry.thumb;
}

void ImageThumbnails::Pump(int max_uploads) {
    ++frame_;
    std::vector<Decoded> ready;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        const size_t n = std::min(done_.size(), static_cast<size_t>(std::max(1, max_uploads)));
        ready.assign(std::make_move_iterator(done_.begin()), std::make_move_iterator(done_.begin() + n));
        done_.erase(done_.begin(), done_.begin() + n);
    }
    auto& textures = cyxwiz::TextureManager::Instance();
    for (auto& d : ready) {
        Entry& entry = entries_[d.path];
        entry.queued = false;
        if (d.failed) {
            entry.thumb.failed = true;
            continue;
        }
        const uint32_t id = textures.CreateTextureFromUint8Data(d.rgba.data(), d.width, d.height, 4);
        entry.thumb.texture = static_cast<ImTextureID>(id);
        entry.thumb.width = d.width;
        entry.thumb.height = d.height;
    }
    // Over capacity: drop what was drawn longest ago (never this frame's).
    if (entries_.size() > capacity_) {
        std::vector<std::pair<uint64_t, std::string>> by_age;
        for (const auto& [path, entry] : entries_)
            if (entry.used + 1 < frame_ && !entry.queued) by_age.push_back({entry.used, path});
        std::sort(by_age.begin(), by_age.end());
        for (size_t i = 0; i < by_age.size() && entries_.size() > capacity_; ++i) {
            const auto it = entries_.find(by_age[i].second);
            if (it->second.thumb.texture) textures.DeleteTexture(static_cast<uint32_t>(it->second.thumb.texture));
            entries_.erase(it);
        }
    }
}

void ImageThumbnails::Work() {
    for (;;) {
        std::string path;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            wake_.wait(lock, [this] { return stop_ || !requests_.empty(); });
            if (stop_) return;
            path = std::move(requests_.back());  // the newest: what is on screen now
            requests_.pop_back();
        }
        Decoded d = Decode(path);
        std::lock_guard<std::mutex> lock(mutex_);
        if (stop_) return;
        done_.push_back(std::move(d));
    }
}

ImageThumbnails::Decoded ImageThumbnails::Decode(const std::string& path) const {
    Decoded d;
    d.path = path;
    // 8-bit BGR, as the Engine shows colour. EXIF orientation is ignored, as
    // training, the index and the viewer read the pixels as stored: a phone
    // photo shows here the way the model gets it.
    cv::Mat image = cv::imread(path, cv::IMREAD_COLOR | cv::IMREAD_IGNORE_ORIENTATION);
    if (image.empty()) {
        d.failed = true;
        return d;
    }
    const double scale = std::min(1.0, static_cast<double>(size_) / std::max(image.cols, image.rows));
    d.width = std::max(1, static_cast<int>(image.cols * scale + 0.5));
    d.height = std::max(1, static_cast<int>(image.rows * scale + 0.5));
    cv::Mat small;
    cv::resize(image, small, cv::Size(d.width, d.height), 0, 0, cv::INTER_AREA);
    cv::Mat rgba;
    cv::cvtColor(small, rgba, cv::COLOR_BGR2RGBA);
    d.rgba.assign(rgba.data, rgba.data + rgba.total() * rgba.elemSize());
    return d;
}

}  // namespace gui
