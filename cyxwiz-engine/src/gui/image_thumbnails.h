#pragma once

// Thumbnails of image files for the image gallery and the Data Input
// dialog's preview (TOFIX178 1.3e): the whole image inside size x size,
// aspect kept (a tile crops it to its square when drawn). Files are decoded
// (OpenCV) and shrunk on two worker threads, newest request first; the
// textures are made on the UI thread in Pump(). At most `capacity` textures
// are kept, the least recently drawn go first.

#include <imgui.h>

#include <condition_variable>
#include <cstdint>
#include <deque>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace gui {

class ImageThumbnails {
public:
    ImageThumbnails(int size, size_t capacity);
    ~ImageThumbnails();
    ImageThumbnails(const ImageThumbnails&) = delete;
    ImageThumbnails& operator=(const ImageThumbnails&) = delete;

    struct Thumb {
        ImTextureID texture = 0;   // 0 while loading or when unreadable
        int width = 0, height = 0;
        bool failed = false;       // the file cannot be read as an image
    };
    // The thumbnail of `path`, asking for it when it is not there yet.
    Thumb Get(const std::string& path);
    // UI thread, once a frame: makes textures for decoded files and drops the
    // ones over capacity.
    void Pump(int max_uploads = 12);

private:
    struct Decoded {
        std::string path;
        std::vector<unsigned char> rgba;
        int width = 0, height = 0;
        bool failed = false;
    };
    struct Entry {
        Thumb thumb;
        uint64_t used = 0;
        bool queued = false;
    };
    void Work();
    Decoded Decode(const std::string& path) const;

    const int size_;
    const size_t capacity_;
    uint64_t frame_ = 0;
    std::map<std::string, Entry> entries_;   // UI thread only

    std::mutex mutex_;
    std::condition_variable wake_;
    std::deque<std::string> requests_;       // newest at the back
    std::vector<Decoded> done_;
    bool stop_ = false;
    std::vector<std::thread> workers_;
};

}  // namespace gui
