#pragma once

#include "product_registration.h"

namespace cyxwiz::runtime::detail {

ProductRegistrationResult RegisterPlatformProduct(
    const ProductRegistrationRequest& request);

ProductUnregistrationResult UnregisterPlatformProduct(
    const ProductRegistrationRequest& request);

std::optional<std::filesystem::path> RegisteredPlatformInstallRoot(
    ProductInstallScope scope);

}  // namespace cyxwiz::runtime::detail
