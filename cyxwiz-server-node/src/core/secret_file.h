// secret_file.h - read a secret from a file instead of the command line
#pragma once

#include <filesystem>
#include <string>

namespace cyxwiz::servernode::core {

struct SecretFileResult {
    std::string secret;  // empty when error is set
    std::string error;   // why the file was refused
};

// Reads the secret in `path`, without surrounding whitespace or line ends.
// A secret on the command line is readable by every user of the machine
// (the process list); a file can be kept to its owner. Refuses a file that is
// missing or empty and, outside Windows, one that its group or other users
// can read (chmod 600), as ssh does for private keys.
SecretFileResult ReadSecretFile(const std::filesystem::path& path);

}  // namespace cyxwiz::servernode::core
