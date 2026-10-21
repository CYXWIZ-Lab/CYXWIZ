#pragma once

#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <string>
#include <string_view>

namespace cyxwiz::runtime {

class Sha256Stream {
public:
    Sha256Stream();
    ~Sha256Stream();
    Sha256Stream(Sha256Stream&&) noexcept;
    Sha256Stream& operator=(Sha256Stream&&) noexcept;
    Sha256Stream(const Sha256Stream&) = delete;
    Sha256Stream& operator=(const Sha256Stream&) = delete;

    bool Update(std::string_view bytes, std::string& error);
    bool Finish(std::string& digest, std::string& error);

private:
    struct State;
    std::unique_ptr<State> state_;
};

bool IsLowercaseSha256(std::string_view value);
bool Sha256Bytes(
    std::string_view bytes,
    std::string& digest,
    std::string& error);
using Sha256FileProgress =
    std::function<bool(std::uint64_t completed_bytes)>;

bool Sha256File(
    const std::filesystem::path& path,
    std::string& digest,
    std::string& error,
    const Sha256FileProgress& progress = {});

}  // namespace cyxwiz::runtime
