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

}  // namespace cyxwiz
