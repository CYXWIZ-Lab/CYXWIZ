#pragma once

#include "product_registration.h"

namespace cyxwiz::runtime::detail {

ProductRegistrationResult RegisterPlatformProduct(
    const ProductRegistrationRequest& request);

ProductUnregistrationResult UnregisterPlatformProduct(
    const ProductRegistrationRequest& request);

}  // namespace cyxwiz::runtime::detail
