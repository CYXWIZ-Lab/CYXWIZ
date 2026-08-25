#pragma once

#include "product_installation_receipt.h"

#include <filesystem>
#include <string>

namespace cyxwiz::runtime {

struct ProductRegistrationRequest {
    std::filesystem::path install_root;
    std::filesystem::path runtime_root;
    ProductInstallScope scope = ProductInstallScope::CurrentUser;
    std::string product_version;
};

struct ProductRegistrationResult {
    bool registered = false;
    std::string message;
};

struct ProductUnregistrationResult {
    bool unregistered = false;
    std::string message;
};

ProductRegistrationResult RegisterInstalledProduct(
    const ProductRegistrationRequest& request);

ProductUnregistrationResult UnregisterInstalledProduct(
    const ProductRegistrationRequest& request);

}  // namespace cyxwiz::runtime
