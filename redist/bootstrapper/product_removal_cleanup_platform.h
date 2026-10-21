#pragma once

#include "product_removal_cleanup.h"

namespace cyxwiz::runtime::detail {

bool CleanupQuarantineNoFollow(
    const QuarantinedProductInstallation& quarantined,
    ProductRemovalCleanupResult& result,
    std::string& error);

}  // namespace cyxwiz::runtime::detail
