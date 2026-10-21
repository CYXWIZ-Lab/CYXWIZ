#pragma once

#include "product_removal_quarantine.h"

#include <cstdint>
#include <string>

namespace cyxwiz::runtime {

struct ProductRemovalCleanupResult {
    bool complete = false;
    std::uint64_t removed_entries = 0;
};

bool CleanupQuarantinedProductInstallation(
    const QuarantinedProductInstallation& quarantined,
    ProductRemovalCleanupResult& result,
    std::string& error);

}  // namespace cyxwiz::runtime
