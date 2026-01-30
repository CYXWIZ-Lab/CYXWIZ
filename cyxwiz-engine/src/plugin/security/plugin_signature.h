#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <filesystem>

namespace cyxwiz::plugin::security {

enum class SignatureStatus {
    Valid,      // Signature matches DLL hash
    Invalid,    // Signature does not match
    Missing,    // No signature provided in manifest
    FileError,  // Could not read file
};

inline const char* SignatureStatusToString(SignatureStatus s) {
    switch (s) {
        case SignatureStatus::Valid:     return "Valid";
        case SignatureStatus::Invalid:   return "Invalid";
        case SignatureStatus::Missing:   return "Missing";
        case SignatureStatus::FileError: return "FileError";
        default: return "Unknown";
    }
}

struct SignatureResult {
    SignatureStatus status = SignatureStatus::Missing;
    std::string message;
};

class PluginSignature {
public:
    // Verify a DLL's Ed25519 signature.
    // signature_hex: hex-encoded Ed25519 signed message (signature + hash).
    // Returns Valid if crypto_sign_open succeeds against ENGINE_PUBLIC_KEY.
    static SignatureResult Verify(const std::filesystem::path& dll_path,
                                  const std::string& signature_hex);

    // SHA-512 hash of a file's contents.
    static std::vector<uint8_t> HashFile(const std::filesystem::path& path);

    // Hex encoding/decoding utilities
    static std::string ToHex(const uint8_t* data, size_t len);
    static std::vector<uint8_t> FromHex(const std::string& hex);

    // Compiled-in public key for verifying official plugins.
    // Replace with your actual Ed25519 public key.
    static const uint8_t ENGINE_PUBLIC_KEY[32];
};

} // namespace cyxwiz::plugin::security
