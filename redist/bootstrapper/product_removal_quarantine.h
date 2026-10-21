#pragma once

#include "product_removal_authorization.h"

#include <filesystem>
#include <string>

namespace cyxwiz::runtime {

struct QuarantinedProductInstallation {
    std::filesystem::path original_root;
    std::filesystem::path quarantine_root;
    std::string install_id;
    ProductInstallScope scope = ProductInstallScope::CurrentUser;
};

std::filesystem::path ProductRemovalQuarantinePath(
    const ProductRemovalAuthorization& authorization);

bool QuarantineProductInstallation(
    const ProductRemovalAuthorization& authorization,
    QuarantinedProductInstallation& quarantined,
    std::string& error);

bool ValidateQuarantinedProductInstallation(
    const QuarantinedProductInstallation& quarantined,
    std::string& error);

}  // namespace cyxwiz::runtime
