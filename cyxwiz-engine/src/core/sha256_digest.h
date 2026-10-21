#pragma once

// SHA-256 as lowercase hex, for files and streamed bytes (checkpoint payloads,
// dataset files shipped to Server Nodes).

#include <filesystem>
#include <memory>
#include <string>
#include <string_view>

namespace cyxwiz {

class Sha256Hasher {
public:
    Sha256Hasher();
    ~Sha256Hasher();
    Sha256Hasher(const Sha256Hasher&) = delete;
    Sha256Hasher& operator=(const Sha256Hasher&) = delete;

    bool Update(std::string_view bytes, std::string& error);
    // Lowercase hex digest; the hasher cannot be updated afterwards.
    bool Finish(std::string& digest, std::string& error);

private:
    struct State;
    std::unique_ptr<State> state_;
};

bool Sha256File(const std::filesystem::path& path, std::string& digest, std::string& error);

// A source file's SHA-256 by path, size and write time, remembered for the
// process and in `store_dir` (the ingestion cache, a "digests" folder), so a
// file is hashed at most once, and a large one while it is read anyway (the
// CSV conversion; TOFIX157).
//
// The stamp of a file: its path, size and write time ("" when it cannot be
// read); `bytes` gets its size.
std::string FileStamp(const std::filesystem::path& path, uint64_t* bytes = nullptr);
// The digest recorded for this stamp, or "".
std::string KnownFileDigest(const std::string& stamp, const std::filesystem::path& store_dir);
void RememberFileDigest(const std::string& stamp, const std::filesystem::path& store_dir, const std::string& digest);

}  // namespace cyxwiz
