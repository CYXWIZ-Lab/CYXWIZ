#include "backend_pack_hash.h"

#include <openssl/evp.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <fstream>

namespace cyxwiz::runtime {
namespace {

std::string HexDigest(
    const unsigned char* bytes,
    std::size_t size) {
    static constexpr char kHex[] = "0123456789abcdef";
    std::string output(size * 2, '0');
    for (std::size_t index = 0; index < size; ++index) {
        output[2 * index] = kHex[bytes[index] >> 4];
        output[2 * index + 1] = kHex[bytes[index] & 0x0f];
    }
    return output;
}

}  // namespace

struct Sha256Stream::State {
    State() : context(EVP_MD_CTX_new()) {
        if (!context) {
            initialization_error = "Cannot allocate SHA-256 context";
        } else if (EVP_DigestInit_ex(context, EVP_sha256(), nullptr) != 1) {
            initialization_error = "Cannot initialize SHA-256";
        }
    }

    ~State() { EVP_MD_CTX_free(context); }

    EVP_MD_CTX* context = nullptr;
    std::string initialization_error;
    bool finished = false;
};

Sha256Stream::Sha256Stream() : state_(std::make_unique<State>()) {}
Sha256Stream::~Sha256Stream() = default;
Sha256Stream::Sha256Stream(Sha256Stream&&) noexcept = default;
Sha256Stream& Sha256Stream::operator=(Sha256Stream&&) noexcept = default;

bool Sha256Stream::Update(
    std::string_view bytes,
    std::string& error) {
    if (!state_ || !state_->initialization_error.empty() ||
        state_->finished) {
        error = state_ && !state_->initialization_error.empty()
            ? state_->initialization_error
            : "SHA-256 stream is unavailable";
        return false;
    }
    if (EVP_DigestUpdate(
            state_->context, bytes.data(), bytes.size()) != 1) {
        error = "Cannot update SHA-256";
        return false;
    }
    return true;
}

bool Sha256Stream::Finish(
    std::string& digest,
    std::string& error) {
    if (!state_ || !state_->initialization_error.empty() ||
        state_->finished) {
        error = state_ && !state_->initialization_error.empty()
            ? state_->initialization_error
            : "SHA-256 stream is unavailable";
        return false;
    }
    std::array<unsigned char, EVP_MAX_MD_SIZE> bytes{};
    unsigned int size = 0;
    if (EVP_DigestFinal_ex(state_->context, bytes.data(), &size) != 1 ||
        size != 32) {
        error = "Cannot finalize SHA-256";
        return false;
    }
    state_->finished = true;
    digest = HexDigest(bytes.data(), size);
    return true;
}

bool IsLowercaseSha256(std::string_view value) {
    return value.size() == 64 &&
           std::all_of(value.begin(), value.end(), [](unsigned char c) {
               return std::isdigit(c) || (c >= 'a' && c <= 'f');
           });
}

bool Sha256Bytes(
    std::string_view bytes,
    std::string& digest,
    std::string& error) {
    std::array<unsigned char, 32> output{};
    unsigned int size = 0;
    if (EVP_Digest(
            bytes.data(), bytes.size(), output.data(), &size,
            EVP_sha256(), nullptr) != 1 || size != output.size()) {
        error = "Cannot calculate SHA-256";
        return false;
    }
    digest = HexDigest(output.data(), output.size());
    return true;
}

bool Sha256File(
    const std::filesystem::path& path,
    std::string& digest,
    std::string& error,
    const Sha256FileProgress& progress) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) {
        error = "Cannot open file for hashing: " + path.string();
        return false;
    }
    Sha256Stream hash;
    std::array<char, 64 * 1024> buffer{};
    std::uint64_t completed_bytes = 0;
    while (stream) {
        stream.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
        const auto count = stream.gcount();
        if (count > 0 && !hash.Update(
                std::string_view(
                    buffer.data(), static_cast<std::size_t>(count)),
                error)) {
            return false;
        }
        if (count > 0) {
            completed_bytes += static_cast<std::uint64_t>(count);
            if (progress && !progress(completed_bytes)) {
                error = "File hashing cancelled";
                return false;
            }
        }
    }
    if (!stream.eof()) {
        error = "Cannot read file while hashing: " + path.string();
        return false;
    }
    return hash.Finish(digest, error);
}

}  // namespace cyxwiz::runtime
