#pragma once

#include "product_removal_authorization.h"

#include <cstdint>
#include <filesystem>
#include <string>

namespace cyxwiz::runtime {

// The token is an inherited read end of a pipe whose write end belongs only
// to the launching bootstrapper process. EOF therefore proves that process
// has released its executable and runtime handles.
bool AwaitAuthorizedProductRemoval(
    const std::filesystem::path& install_root,
    std::uintptr_t parent_lifetime_token,
    ProductRemovalAuthorization& authorization,
    std::string& error);

}  // namespace cyxwiz::runtime
