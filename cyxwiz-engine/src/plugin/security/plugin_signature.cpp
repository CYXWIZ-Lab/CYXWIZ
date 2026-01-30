#include "plugin_signature.h"

// TweetNaCl is plain C
extern "C" {
#include "tweetnacl.h"
}

// TweetNaCl requires a randombytes() implementation.
// We only use verification (crypto_sign_open), never key generation,
// but the symbol must exist to satisfy the linker.
#ifdef _WIN32
#include <windows.h>
#include <bcrypt.h>
#pragma comment(lib, "bcrypt.lib")
extern "C" void randombytes(unsigned char* buf, unsigned long long len) {
    BCryptGenRandom(nullptr, buf, static_cast<ULONG>(len), BCRYPT_USE_SYSTEM_PREFERRED_RNG);
}
#else
#include <fcntl.h>
#include <unistd.h>
extern "C" void randombytes(unsigned char* buf, unsigned long long len) {
    int fd = open("/dev/urandom", O_RDONLY);
    if (fd >= 0) { read(fd, buf, (size_t)len); close(fd); }
}
#endif

#include <spdlog/spdlog.h>
#include <fstream>
#include <cstring>

namespace cyxwiz::plugin::security {

// Placeholder public key — replace with your real Ed25519 public key.
// Generate a keypair with: crypto_sign_keypair(pk, sk)
const uint8_t PluginSignature::ENGINE_PUBLIC_KEY[32] = {
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
};

std::vector<uint8_t> PluginSignature::HashFile(const std::filesystem::path& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) return {};

    // Read entire file
    f.seekg(0, std::ios::end);
    auto size = f.tellg();
    f.seekg(0, std::ios::beg);

    std::vector<uint8_t> data(static_cast<size_t>(size));
    f.read(reinterpret_cast<char*>(data.data()), size);

    // SHA-512
    std::vector<uint8_t> hash(64);
    crypto_hash(hash.data(), data.data(), data.size());
    return hash;
}

std::string PluginSignature::ToHex(const uint8_t* data, size_t len) {
    static const char hex_chars[] = "0123456789abcdef";
    std::string result;
    result.reserve(len * 2);
    for (size_t i = 0; i < len; ++i) {
        result.push_back(hex_chars[(data[i] >> 4) & 0x0F]);
        result.push_back(hex_chars[data[i] & 0x0F]);
    }
    return result;
}

std::vector<uint8_t> PluginSignature::FromHex(const std::string& hex) {
    if (hex.size() % 2 != 0) return {};
    std::vector<uint8_t> result(hex.size() / 2);

    for (size_t i = 0; i < result.size(); ++i) {
        auto hi = hex[i * 2];
        auto lo = hex[i * 2 + 1];

        auto hex_val = [](char c) -> int {
            if (c >= '0' && c <= '9') return c - '0';
            if (c >= 'a' && c <= 'f') return c - 'a' + 10;
            if (c >= 'A' && c <= 'F') return c - 'A' + 10;
            return -1;
        };

        int h = hex_val(hi);
        int l = hex_val(lo);
        if (h < 0 || l < 0) return {};
        result[i] = static_cast<uint8_t>((h << 4) | l);
    }
    return result;
}

SignatureResult PluginSignature::Verify(
    const std::filesystem::path& dll_path,
    const std::string& signature_hex)
{
    if (signature_hex.empty()) {
        return {SignatureStatus::Missing, "No signature in manifest"};
    }

    // Decode the signed message (Ed25519 signature + original message)
    auto signed_msg = FromHex(signature_hex);
    if (signed_msg.empty()) {
        return {SignatureStatus::Invalid, "Invalid hex encoding in signature"};
    }

    // Ed25519 signed messages are at least 64 bytes (signature) + message
    if (signed_msg.size() < 64) {
        return {SignatureStatus::Invalid, "Signature too short"};
    }

    // Hash the DLL file
    auto file_hash = HashFile(dll_path);
    if (file_hash.empty()) {
        return {SignatureStatus::FileError, "Could not read DLL: " + dll_path.string()};
    }

    // crypto_sign_open: verify and extract the original message
    // signed_msg = 64-byte signature + original message
    // We expect the original message to be the SHA-512 hash (64 bytes)
    std::vector<uint8_t> opened_msg(signed_msg.size());
    unsigned long long opened_len = 0;

    int rc = crypto_sign_open(
        opened_msg.data(), &opened_len,
        signed_msg.data(), signed_msg.size(),
        ENGINE_PUBLIC_KEY);

    if (rc != 0) {
        return {SignatureStatus::Invalid, "Ed25519 signature verification failed"};
    }

    // Check that the signed content matches the file hash
    if (opened_len != 64 || std::memcmp(opened_msg.data(), file_hash.data(), 64) != 0) {
        return {SignatureStatus::Invalid, "Signed hash does not match DLL hash"};
    }

    spdlog::info("PluginSignature: Valid signature for {}", dll_path.filename().string());
    return {SignatureStatus::Valid, "OK"};
}

} // namespace cyxwiz::plugin::security
