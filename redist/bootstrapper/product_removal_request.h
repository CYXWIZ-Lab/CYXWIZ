#pragma once

#include "product_removal_authorization.h"

#include <filesystem>
#include <string>

namespace cyxwiz::runtime {

std::filesystem::path ProductRemovalRequestPath(
    const std::filesystem::path& install_root);

bool QueueProductRemovalRequest(
    const std::filesystem::path& install_root,
    ProductInstallScope scope,
    ProductRemovalAuthorization& authorization,
    std::string& error);

bool LoadProductRemovalRequest(
    const std::filesystem::path& install_root,
    ProductRemovalAuthorization& authorization,
    std::string& error);

}  // namespace cyxwiz::runtime
