#pragma once

#include "product_installation_receipt.h"
#include "runtime_layout.h"

#include <filesystem>
#include <string>

namespace cyxwiz::runtime {

struct ProductRemovalAuthorization {
    std::filesystem::path install_root;
    ProductInstallScope scope = ProductInstallScope::CurrentUser;
    std::string install_id;
    ActiveRuntimeState runtime;
};

bool CaptureProductRemovalAuthorization(
    const std::filesystem::path& install_root,
    ProductInstallScope scope,
    ProductRemovalAuthorization& authorization,
    std::string& error);

bool ValidateProductRemovalAuthorization(
    const ProductRemovalAuthorization& authorization,
    std::string& error);

}  // namespace cyxwiz::runtime
