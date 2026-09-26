#include "sha256_digest.h"

#include <openssl/evp.h>

#include <array>
#include <fstream>
#include <iomanip>
#include <sstream>

namespace cyxwiz {

struct Sha256Hasher::State {
    EVP_MD_CTX* context = nullptr;
    bool ok = false;
    ~State() {
        if (context) EVP_MD_CTX_free(context);
    }
};

Sha256Hasher::Sha256Hasher() : state_(std::make_unique<State>()) {
    state_->context = EVP_MD_CTX_new();
    state_->ok = state_->context && EVP_DigestInit_ex(state_->context, EVP_sha256(), nullptr) == 1;
}

Sha256Hasher::~Sha256Hasher() = default;

bool Sha256Hasher::Update(std::string_view bytes, std::string& error) {
    if (!state_->ok) {
        error = "SHA-256 is not available";
        return false;
    }
    if (!bytes.empty() && EVP_DigestUpdate(state_->context, bytes.data(), bytes.size()) != 1) {
        error = "could not update SHA-256";
        state_->ok = false;
        return false;
    }
    return true;
}

bool Sha256Hasher::Finish(std::string& digest, std::string& error) {
    if (!state_->ok) {
        error = "SHA-256 is not available";
        return false;
    }
    std::array<unsigned char, EVP_MAX_MD_SIZE> bytes{};
    unsigned int length = 0;
    state_->ok = false;
    if (EVP_DigestFinal_ex(state_->context, bytes.data(), &length) != 1 || length != 32) {
        error = "could not finalize SHA-256";
        return false;
    }
    std::ostringstream output;
    output << std::hex << std::setfill('0');
    for (unsigned int index = 0; index < length; ++index) {
        output << std::setw(2) << static_cast<unsigned int>(bytes[index]);
    }
    digest = output.str();
    return true;
}

bool Sha256File(const std::filesystem::path& path, std::string& digest, std::string& error) {
    std::ifstream input(path, std::ios::binary);
    if (!input.is_open()) {
        error = "file is unreadable: " + path.string();
        return false;
    }
    Sha256Hasher hasher;
    std::array<char, 64 * 1024> buffer{};
    while (input.good()) {
        input.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
        const auto count = input.gcount();
        if (count > 0 && !hasher.Update(std::string_view(buffer.data(), static_cast<size_t>(count)), error)) {
            return false;
        }
    }
    if (!input.eof()) {
        error = "could not read file for SHA-256: " + path.string();
        return false;
    }
    return hasher.Finish(digest, error);
}

}  // namespace cyxwiz
