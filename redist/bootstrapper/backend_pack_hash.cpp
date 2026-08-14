#include "backend_pack_hash.h"

#include <openssl/evp.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <fstream>
#include <memory>

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
    std::string& error) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) {
        error = "Cannot open file for hashing: " + path.string();
        return false;
    }
    EVP_MD_CTX* raw_context = EVP_MD_CTX_new();
    if (!raw_context) {
        error = "Cannot allocate SHA-256 context";
        return false;
    }
    const auto free_context = [](EVP_MD_CTX* context) {
        EVP_MD_CTX_free(context);
    };
    std::unique_ptr<EVP_MD_CTX, decltype(free_context)> context(
        raw_context, free_context);
    if (EVP_DigestInit_ex(context.get(), EVP_sha256(), nullptr) != 1) {
        error = "Cannot initialize SHA-256";
        return false;
    }
    std::array<char, 64 * 1024> buffer{};
    while (stream) {
        stream.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
        const auto count = stream.gcount();
        if (count > 0 &&
            EVP_DigestUpdate(
                context.get(), buffer.data(), static_cast<std::size_t>(count)) !=
                1) {
            error = "Cannot update SHA-256";
            return false;
        }
    }
    if (!stream.eof()) {
        error = "Cannot read file while hashing: " + path.string();
        return false;
    }
    std::array<unsigned char, EVP_MAX_MD_SIZE> bytes{};
    unsigned int size = 0;
    if (EVP_DigestFinal_ex(context.get(), bytes.data(), &size) != 1 ||
        size != 32) {
        error = "Cannot finalize SHA-256";
        return false;
    }
    digest = HexDigest(bytes.data(), size);
    return true;
}

}  // namespace cyxwiz::runtime
