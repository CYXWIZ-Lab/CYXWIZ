// secret_file.cpp - read a secret from a file instead of the command line
#include "core/secret_file.h"

#include <fstream>
#include <iterator>
#include <system_error>

namespace cyxwiz::servernode::core {

SecretFileResult ReadSecretFile(const std::filesystem::path& path) {
    SecretFileResult result;
    std::error_code ec;
    if (!std::filesystem::is_regular_file(path, ec)) {
        result.error = path.string() + " does not exist or is not a file";
        return result;
    }
#ifndef _WIN32
    const auto perms = std::filesystem::status(path, ec).permissions();
    constexpr auto kOthers = std::filesystem::perms::group_all | std::filesystem::perms::others_all;
    if (!ec && (perms & kOthers) != std::filesystem::perms::none) {
        result.error = path.string() + " can be read by other users; run chmod 600 on it";
        return result;
    }
#endif
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        result.error = path.string() + " cannot be read";
        return result;
    }
    std::string text((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    const auto first = text.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) {
        result.error = path.string() + " is empty";
        return result;
    }
    const auto last = text.find_last_not_of(" \t\r\n");
    result.secret = text.substr(first, last - first + 1);
    return result;
}

}  // namespace cyxwiz::servernode::core
