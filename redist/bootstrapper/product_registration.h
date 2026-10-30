#pragma once

#include "product_installation_receipt.h"

#include <filesystem>
#include <optional>
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

// Where CyxWiz is installed for this scope, as its registration records it
// (Windows: Apps & Features InstallLocation; Linux: the launcher of the
// CyxWiz desktop entry). Nothing when it is not registered, and on macOS,
// whose app bundles live inside the install root.
std::optional<std::filesystem::path> RegisteredInstallRoot(ProductInstallScope scope);

}  // namespace cyxwiz::runtime
