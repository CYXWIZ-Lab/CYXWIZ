#include "sha256_digest.h"

#include <openssl/evp.h>

#include <array>
#include <fstream>
#include <iomanip>
#include <map>
#include <mutex>
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

namespace {

std::mutex& DigestMutex() {
    static std::mutex mutex;
    return mutex;
}

std::map<std::string, std::string>& KnownDigests() {
    static std::map<std::string, std::string> digests;
    return digests;
}

// One small file per stamp: <store>/digests/<sha256 of the stamp>.txt holding
// the stamp and the digest (the stamp is checked, not trusted from the name).
std::filesystem::path DigestFile(const std::string& stamp, const std::filesystem::path& store_dir) {
    Sha256Hasher hasher;
    std::string name, error;
    if (!hasher.Update(stamp, error) || !hasher.Finish(name, error)) return {};
    return store_dir / "digests" / (name.substr(0, 32) + ".txt");
}

}  // namespace

std::string FileStamp(const std::filesystem::path& path, uint64_t* bytes) {
    std::error_code ec;
    const auto size = std::filesystem::file_size(path, ec);
    if (ec) return {};
    const auto time = std::filesystem::last_write_time(path, ec);
    if (ec) return {};
    if (bytes) *bytes = size;
    const auto absolute = std::filesystem::absolute(path, ec);
    return (ec ? path : absolute).lexically_normal().string() + "|" + std::to_string(size) + "|" +
           // long long: libc++'s file clock counts in __int128.
           std::to_string(static_cast<long long>(time.time_since_epoch().count()));
}

std::string KnownFileDigest(const std::string& stamp, const std::filesystem::path& store_dir) {
    if (stamp.empty()) return {};
    {
        std::lock_guard<std::mutex> lock(DigestMutex());
        if (const auto it = KnownDigests().find(stamp); it != KnownDigests().end()) return it->second;
    }
    if (store_dir.empty()) return {};
    std::ifstream input(DigestFile(stamp, store_dir));
    std::string saved_stamp, digest;
    if (!std::getline(input, saved_stamp) || !std::getline(input, digest) || saved_stamp != stamp || digest.size() != 64)
        return {};
    std::lock_guard<std::mutex> lock(DigestMutex());
    KnownDigests()[stamp] = digest;
    return digest;
}

void RememberFileDigest(const std::string& stamp, const std::filesystem::path& store_dir, const std::string& digest) {
    if (stamp.empty() || digest.empty()) return;
    {
        std::lock_guard<std::mutex> lock(DigestMutex());
        KnownDigests()[stamp] = digest;
    }
    if (store_dir.empty()) return;
    const auto file = DigestFile(stamp, store_dir);
    std::error_code ec;
    std::filesystem::create_directories(file.parent_path(), ec);
    const auto temp = std::filesystem::path(file.string() + ".tmp");
    {
        std::ofstream output(temp, std::ios::trunc);
        output << stamp << '\n' << digest << '\n';
        if (!output) return;
    }
    std::filesystem::rename(temp, file, ec);
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
