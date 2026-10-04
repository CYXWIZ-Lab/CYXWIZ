#pragma once

// When a sign-in token (JWT) expires, read from its "exp" claim without
// verifying it: clients use this to refresh before the central server starts
// refusing the token (TOFIX136 E2). Shared by the Engine and the server node.

#include <nlohmann/json.hpp>

#include <string>

namespace cyxwiz {

// Unix seconds of the token's "exp" claim; 0 when it cannot be read.
inline long long JwtExpiry(const std::string& token) {
    // header.payload.signature; the payload is base64url JSON without padding.
    const auto first = token.find('.');
    const auto second = first == std::string::npos ? std::string::npos : token.find('.', first + 1);
    if (second == std::string::npos) return 0;
    static const std::string kAlphabet =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";
    std::string decoded;
    int value = 0;
    int bits = -8;
    for (std::size_t i = first + 1; i < second; ++i) {
        const auto index = kAlphabet.find(token[i]);
        if (index == std::string::npos) return 0;
        value = (value << 6) + static_cast<int>(index);
        bits += 6;
        if (bits >= 0) {
            decoded.push_back(static_cast<char>((value >> bits) & 0xFF));
            bits -= 8;
        }
    }
    try {
        const auto payload = nlohmann::json::parse(decoded);
        return payload.value("exp", 0LL);
    } catch (...) {
        return 0;
    }
}

// Clients refresh their token when less than this remains.
inline constexpr long long kJwtRefreshBeforeSeconds = 2 * 60 * 60;

}  // namespace cyxwiz
